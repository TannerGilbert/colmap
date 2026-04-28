// Copyright (c), ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

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
