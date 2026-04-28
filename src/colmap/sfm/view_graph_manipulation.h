#pragma once

#include "colmap/scene/correspondence_graph.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/util/types.h"

namespace colmap {

// Reclassify UNCALIBRATED pairs as CALIBRATED if both cameras have valid
// focal-length priors and the majority of pairs each camera is involved in
// are already CALIBRATED. For newly-upgraded pairs, recomputes F from the
// existing cam2_from_cam1 pose using F = K2^-T * [t]x R * K1^-1. Mutates
// view_graph in place; cameras/images are read-only here.
void UpdateImagePairsConfig(CorrespondenceGraph& view_graph,
                            const Reconstruction& rec);

// For every valid pair whose both cameras have prior focal lengths,
// re-decompose the relative pose from the pair's E/F/H using
// EstimateTwoViewGeometryPose. PLANAR pairs are upgraded to CALIBRATED.
// Translations are normalized to unit norm (when non-zero). Mutates
// view_graph.image_pairs[*].two_view_geometry in place.
void DecomposeRelPose(CorrespondenceGraph& view_graph, const Reconstruction& rec);

// Mark pairs invalid (is_valid=false) when their inlier count is below
// ``min_inlier_num``.
void FilterPairsByInlierNum(CorrespondenceGraph& view_graph,
                            int min_inlier_num);

// Mark pairs invalid when their inlier ratio (inliers / total matches) is
// below ``min_inlier_ratio``.
void FilterPairsByInlierRatio(CorrespondenceGraph& view_graph,
                              double min_inlier_ratio);

}  // namespace colmap
