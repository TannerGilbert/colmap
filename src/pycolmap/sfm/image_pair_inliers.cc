#include "colmap/sfm/image_pair_inliers.h"

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

// FORK-REMOVAL TODO — entire `BindImagePairInliers` (and the C++
// `ImagePairsInlierCount` it wraps) are fork-only. See
// `.claude/notes/glomap_audit/fork_removal_todo.md` for the removal plan.
void BindImagePairInliers(py::module& m) {
  auto PyOpts =
      py::classh<InlierThresholdOptions>(m, "InlierThresholdOptions")
          .def(py::init<>())
          .def_readwrite("max_epipolar_error_E",
                         &InlierThresholdOptions::max_epipolar_error_E)
          .def_readwrite("max_epipolar_error_F",
                         &InlierThresholdOptions::max_epipolar_error_F)
          .def_readwrite("max_epipolar_error_H",
                         &InlierThresholdOptions::max_epipolar_error_H)
          .def_readwrite("min_angle_from_epipole",
                         &InlierThresholdOptions::min_angle_from_epipole)
          .def_readwrite("max_angle_error",
                         &InlierThresholdOptions::max_angle_error)
          .def_readwrite("max_reprojection_error",
                         &InlierThresholdOptions::max_reprojection_error)
          .def_readwrite("min_triangulation_angle",
                         &InlierThresholdOptions::min_triangulation_angle)
          .def_readwrite("min_inlier_num",
                         &InlierThresholdOptions::min_inlier_num)
          .def_readwrite("min_inlier_ratio",
                         &InlierThresholdOptions::min_inlier_ratio)
          .def_readwrite("max_rotation_error",
                         &InlierThresholdOptions::max_rotation_error);
  MakeDataclass(PyOpts);

  m.def(
      "image_pairs_inlier_count",
      [](CorrespondenceGraph& correspondence_graph,
         Reconstruction& rec,
         const InlierThresholdOptions& options,
         bool clean_inliers) {
        py::gil_scoped_release release;
        ImagePairsInlierCount(
            correspondence_graph, rec, options, clean_inliers);
      },
      "correspondence_graph"_a,
      "rec"_a,
      "options"_a,
      "clean_inliers"_a,
      "Per-pair inlier scoring (Sampson + cheirality + epipole checks). "
      "Updates ImagePair.inliers and is_valid in place.");
}
