#include "colmap/sfm/view_graph_manipulation.h"

#include "colmap/scene/correspondence_graph.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/util/types.h"

#include "pycolmap/helpers.h"

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

using namespace colmap;
using namespace pybind11::literals;
namespace py = pybind11;

void BindViewGraphManipulation(py::module& m) {
  m.def(
      "update_image_pairs_config",
      [](CorrespondenceGraph& view_graph, Reconstruction& rec) {
        py::gil_scoped_release release;
        UpdateImagePairsConfig(view_graph, rec);
      },
      "view_graph"_a,
      "rec"_a,
      "Reclassify UNCALIBRATED pairs as CALIBRATED if both cameras have "
      "valid focal-length priors and the majority of pairs each camera is "
      "involved in are already CALIBRATED. Recomputes F for newly-upgraded "
      "pairs from the existing cam2_from_cam1. Mutates view_graph in place.");

  m.def(
      "decompose_rel_pose",
      [](CorrespondenceGraph& view_graph, Reconstruction& rec) {
        py::gil_scoped_release release;
        DecomposeRelPose(view_graph, rec);
      },
      "view_graph"_a,
      "rec"_a,
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
