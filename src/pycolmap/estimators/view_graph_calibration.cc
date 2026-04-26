#include "colmap/estimators/view_graph_calibration.h"

#include "colmap/scene/camera.h"
#include "colmap/scene/correspondence_graph.h"
#include "colmap/scene/image.h"
#include "colmap/scene/two_view_geometry.h"
#include "colmap/util/logging.h"

#include "pycolmap/helpers.h"

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

using namespace colmap;
using namespace pybind11::literals;
namespace py = pybind11;

namespace {

// Replicate glomap's ViewGraphCalibrator::Solve() flow on top of colmap's pure
// CalibrateFocalLengths function. Bypasses colmap4's higher-level
// CalibrateViewGraph wrapper to preserve byte-for-byte parity with
// pyglomap.run_view_graph_calibration during the pyglomap → pycolmap port.
// Adopting the wrapper's richer behavior (cross_validate_prior_focal_lengths,
// reestimate_relative_pose, F/E recomputation, config flips) is tracked in
// videosfm-private issue #40.
//
// Mutates view_graph, cameras, images in place via opaque-bound references
// (PYBIND11_MAKE_OPAQUE — see pycolmap/scene/types.h). No return value: the
// caller's Python objects already hold the mutated state.
void RunViewGraphCalibration(
    CorrespondenceGraph& view_graph,
    std::unordered_map<camera_t, Camera>& cameras,
    std::unordered_map<image_t, Image>& images,
    const ViewGraphCalibrationOptions& options) {
  py::gil_scoped_release release;

  // Build inputs: one per CALIBRATED/UNCALIBRATED valid pair with an F matrix.
  std::vector<FocalLengthCalibInput> inputs;
  inputs.reserve(view_graph.NumImagePairs());
  std::unordered_map<image_pair_t, CorrespondenceGraph::ImagePair*> pair_lookup;
  pair_lookup.reserve(view_graph.NumImagePairs());
  for (auto& [pair_id, image_pair] : view_graph.MutableImagePairs()) {
    const auto& tvg = image_pair.two_view_geometry;
    if (tvg.config != TwoViewGeometry::CALIBRATED &&
        tvg.config != TwoViewGeometry::UNCALIBRATED)
      continue;
    if (!image_pair.is_valid) continue;
    THROW_CHECK(tvg.F.has_value())
        << "Two-view geometry must have F matrix for VGC";
    inputs.push_back({pair_id,
                      images.at(image_pair.image_id1).CameraId(),
                      images.at(image_pair.image_id2).CameraId(),
                      tvg.F.value()});
    pair_lookup[pair_id] = &image_pair;
  }

  const FocalLengthCalibResult result =
      CalibrateFocalLengths(options, inputs, cameras);
  if (!result.success) {
    py::gil_scoped_acquire acquire;
    throw std::runtime_error("Failed to solve view graph calibration.");
  }

  // CopyBackResults: write focal back to camera.params. Cameras locked via
  // has_prior_focal_length are skipped (they were locked in the optimizer
  // and never moved). Cameras whose ratio was rejected by the optimizer have
  // result.focal_lengths[id] reset to the initial focal, so writing back is a
  // no-op for them — equivalent to glomap's "skip rejected".
  for (auto& [camera_id, camera] : cameras) {
    auto it = result.focal_lengths.find(camera_id);
    if (it == result.focal_lengths.end()) continue;
    if (camera.has_prior_focal_length) continue;
    for (const size_t idx : camera.FocalLengthIdxs()) {
      camera.params[idx] = it->second;
    }
  }

  // FilterImagePairs: invalidate pairs whose squared calibration error exceeds
  // threshold. Mirrors glomap::ViewGraphCalibrator::FilterImagePairs.
  const double max_err_sq =
      options.max_calibration_error * options.max_calibration_error;
  size_t invalid_counter = 0;
  for (const auto& input : inputs) {
    auto it = result.calibration_errors_sq.find(input.pair_id);
    if (it == result.calibration_errors_sq.end()) continue;
    if (it->second > max_err_sq) {
      pair_lookup.at(input.pair_id)->is_valid = false;
      invalid_counter++;
    }
  }
  LOG(INFO) << "VGC: invalidated " << invalid_counter << " / " << inputs.size()
            << " pairs (residual^2 > "
            << options.max_calibration_error * options.max_calibration_error
            << ")";
}

}  // namespace

// ViewGraphCalibrationOptions is already bound by
// src/pycolmap/pipeline/sfm.cc (for the higher-level `calibrate_view_graph`
// wrapper). We reuse the same class — re-binding would error at module init.
void BindViewGraphCalibration(py::module& m) {
  // `options` has no default here: ViewGraphCalibrationOptions is registered
  // by BindPipeline (sfm.cc), which runs after BindEstimators. Defaulting to
  // ViewGraphCalibrationOptions() at this binding site fires before the type
  // is registered and aborts module load. Caller always passes options.
  m.def("run_view_graph_calibration",
        &RunViewGraphCalibration,
        "view_graph"_a,
        "cameras"_a,
        "images"_a,
        "options"_a,
        "Run view graph focal-length calibration on a CorrespondenceGraph + "
        "cameras + images, bypassing colmap4's full CalibrateViewGraph "
        "wrapper. Used during the pyglomap → pycolmap migration to preserve "
        "byte-for-byte parity with pyglomap.run_view_graph_calibration; see "
        "videosfm-private issue #40 for adopting the wrapper later.");
}
