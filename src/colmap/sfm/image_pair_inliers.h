#pragma once

#include "colmap/geometry/rigid3.h"
#include "colmap/math/math.h"
#include "colmap/scene/correspondence_graph.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/util/logging.h"
#include "colmap/util/types.h"

namespace colmap {

// FORK-REMOVAL TODO: entire file is fork-only; vanilla colmap scores
// inliers during geometric verification. See fork_removal_todo.md.

// Thresholds used by inlier scoring.
struct InlierThresholdOptions {
  // Maximum epipolar error per match (pixel-space, converted to bearing
  // space at the call site for ESSENTIAL pairs).
  double max_epipolar_error_E = 1.;
  double max_epipolar_error_F = 4.;
  double max_epipolar_error_H = 4.;
  // Bearing-vs-epipole minimum angle (degrees) per inlier match.
  double min_angle_from_epipole = 3.;

  // Compatibility thresholds used by the Python global-mapper pipeline.
  double max_angle_error = 1.;           // in degree, for global positioning
  double max_reprojection_error = 1e-2;  // for bundle adjustment
  double min_triangulation_angle = 1.;   // in degree, for triangulation
  double min_inlier_num = 30;
  double min_inlier_ratio = 0.25;
  double max_rotation_error = 10.;  // in degree, for rotation averaging
};

void ImagePairsInlierCount(CorrespondenceGraph& correspondence_graph,
                           const Reconstruction& rec,
                           const InlierThresholdOptions& options,
                           bool clean_inliers);

}  // namespace colmap
