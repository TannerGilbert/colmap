#include "colmap/sfm/view_graph_manipulation.h"

#include "colmap/scene/camera.h"
#include "colmap/scene/correspondence_graph.h"
#include "colmap/scene/image.h"
#include "colmap/util/types.h"

#include "pycolmap/helpers.h"

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

using namespace colmap;
using namespace pybind11::literals;
namespace py = pybind11;

namespace {

// Wrap colmap::UpdateImagePairsConfig with a dict-in / dict-out pattern.
// cameras and images are explicitly deep-copied via py::cast on entry and
// exit because pybind11's STL caster + classh<Image> shared_ptr holder can
// move-from values during round-trip and silently lose data members.
py::dict RunUpdateImagePairsConfig(CorrespondenceGraph& view_graph,
                                   py::dict cameras_py,
                                   py::dict images_py) {
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

  {
    py::gil_scoped_release release;
    UpdateImagePairsConfig(view_graph, cameras, images);
  }

  // Build fresh Python dicts for the result; cameras/images are unchanged
  // here but round-tripped so the decompose_rel_pose chain receives all
  // three keys uniformly.
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

py::dict RunDecomposeRelPose(CorrespondenceGraph& view_graph,
                             py::dict cameras_py,
                             py::dict images_py) {
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

  {
    py::gil_scoped_release release;
    DecomposeRelPose(view_graph, cameras, images);
  }

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

void BindViewGraphManipulation(py::module& m) {
  m.def("update_image_pairs_config",
        &RunUpdateImagePairsConfig,
        "view_graph"_a,
        "cameras"_a,
        "images"_a,
        "Reclassify UNCALIBRATED pairs as CALIBRATED if both cameras have "
        "valid focal-length priors and the majority of pairs each camera is "
        "involved in are already CALIBRATED. Recomputes F for newly-upgraded "
        "pairs from the existing cam2_from_cam1. Returns dict with mutated "
        "view_graph + cameras + images.");

  m.def("decompose_rel_pose",
        &RunDecomposeRelPose,
        "view_graph"_a,
        "cameras"_a,
        "images"_a,
        "For every valid pair whose both cameras have prior focal lengths, "
        "re-decompose the relative pose from the pair's E/F/H using "
        "EstimateTwoViewGeometryPose. PLANAR pairs upgrade to CALIBRATED; "
        "translations are normalized to unit norm.");

  m.def("filter_pairs_by_inlier_num",
        &FilterPairsByInlierNum,
        "view_graph"_a,
        "min_inlier_num"_a,
        "Mark pairs invalid when their inlier count is below "
        "min_inlier_num. Mutates view_graph.image_pairs[*].is_valid.");

  m.def("filter_pairs_by_inlier_ratio",
        &FilterPairsByInlierRatio,
        "view_graph"_a,
        "min_inlier_ratio"_a,
        "Mark pairs invalid when their inlier ratio (inliers / total "
        "matches) is below min_inlier_ratio.");
}
