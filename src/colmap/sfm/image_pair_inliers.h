#pragma once

#include "colmap/geometry/rigid3.h"
#include "colmap/math/math.h"
#include "colmap/scene/correspondence_graph.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/util/logging.h"
#include "colmap/util/types.h"

namespace colmap {

// FORK-REMOVAL TODO — this entire file (InlierThresholdOptions struct,
// ImagePairsInlierCount function, and the implementation in
// image_pair_inliers.cc) is fork-only. Vanilla colmap rescores pair
// inliers inside the geometric-verification RANSAC loop, not as a
// separate "score every pair after relative-pose estimation" pass. The
// fork-side `pycolmap.image_pairs_inlier_count` Python caller in
// `mpsfm/mapper/glomap/phases/relative_pose.py` is the only consumer.
// Slated for removal once reproducibility window closes; see
// `.claude/notes/glomap_audit/fork_removal_todo.md`.

// Thresholds used by inlier scoring.
struct InlierThresholdOptions {
  // Maximum epipolar error per match (pixel-space, converted to bearing
  // space at the call site for ESSENTIAL pairs).
  double max_epipolar_error_E = 1.;
  double max_epipolar_error_F = 4.;
  double max_epipolar_error_H = 4.;
  // Bearing-vs-epipole minimum angle (degrees) per inlier match.
  double min_angle_from_epipole = 3.;
};

void ImagePairsInlierCount(CorrespondenceGraph& correspondence_graph,
                           const Reconstruction& rec,
                           const InlierThresholdOptions& options,
                           bool clean_inliers);

}  // namespace colmap
