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

// Run focal-length view-graph calibration on top of colmap's pure
// CalibrateFocalLengths function. Bypasses CalibrateViewGraph's richer
// flow (cross_validate_prior_focal_lengths, reestimate_relative_pose,
// F/E recomputation, config flips) — those are tracked separately.
//
// Returns a dict {"view_graph", "cameras", "images"} with the mutated
// state so the caller can rebind locally (pybind11 auto-converts the input
// dicts to C++ copies, so mutations propagate via the returned dict, not
// the inputs).
py::dict RunViewGraphCalibration(CorrespondenceGraph& view_graph,
                                 py::dict cameras_py,
                                 py::dict images_py,
                                 const ViewGraphCalibrationOptions& options) {
  // Convert Python dicts → C++ maps. Holding GIL — iterating Python.
  std::unordered_map<camera_t, Camera> cameras;
  cameras.reserve(cameras_py.size());
  for (auto item : cameras_py) {
    cameras.emplace(py::cast<camera_t>(item.first),
                    py::cast<Camera>(item.second));
  }
  std::unordered_map<image_t, Image> images;
  images.reserve(images_py.size());
  for (auto item : images_py) {
    images.emplace(py::cast<image_t>(item.first),
                   py::cast<Image>(item.second));
  }

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

  FocalLengthCalibResult result;
  {
    py::gil_scoped_release release;
    result = CalibrateFocalLengths(options, inputs, cameras);
  }
  if (!result.success) {
    throw std::runtime_error("Failed to solve view graph calibration.");
  }

  // CopyBackResults: write focal back to camera.params. Cameras locked via
  // has_prior_focal_length are skipped (they were locked in the optimizer
  // and never moved). Cameras whose ratio was rejected by the optimizer have
  // result.focal_lengths[id] reset to the initial focal, so writing back is a
  // no-op for them.
  for (auto& [camera_id, camera] : cameras) {
    auto it = result.focal_lengths.find(camera_id);
    if (it == result.focal_lengths.end()) continue;
    if (camera.has_prior_focal_length) continue;
    for (const size_t idx : camera.FocalLengthIdxs()) {
      camera.params[idx] = it->second;
    }
  }

  // FilterImagePairs: invalidate pairs whose squared calibration error exceeds
  // threshold.
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

  // Build a fresh Python dict (one Camera/Image per entry, deep copy via
  // py::cast) so mutations to the C++ maps propagate back. Returning the
  // C++ map directly through pybind11's STL caster + classh shared_ptr
  // holder can move-from the values, leaving fields empty in the result.
  py::dict cameras_out;
  for (auto& [cid, cam] : cameras) {
    cameras_out[py::cast(cid)] = py::cast(cam);
  }
  py::dict images_out;
  for (auto& [iid, img] : images) {
    images_out[py::cast(iid)] = py::cast(img);
  }
  py::dict output;
  output["view_graph"] = view_graph;
  output["cameras"] = cameras_out;
  output["images"] = images_out;
  return output;
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
        "cameras + images, bypassing CalibrateViewGraph's higher-level "
        "wrapper (cross-validation, F/E recomputation, etc.).");
}
