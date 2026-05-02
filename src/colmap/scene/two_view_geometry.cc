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

#include "colmap/scene/two_view_geometry.h"

#include "colmap/util/logging.h"

#include <algorithm>
#include <unordered_map>

namespace colmap {
namespace {

struct FeatureMatchKey {
  point2D_t point2D_idx1 = kInvalidPoint2DIdx;
  point2D_t point2D_idx2 = kInvalidPoint2DIdx;

  bool operator==(const FeatureMatchKey& other) const {
    return point2D_idx1 == other.point2D_idx1 &&
           point2D_idx2 == other.point2D_idx2;
  }
};

struct FeatureMatchKeyHash {
  size_t operator()(const FeatureMatchKey& key) const {
    return std::hash<point2D_t>()(key.point2D_idx1) ^
           (std::hash<point2D_t>()(key.point2D_idx2) << 1);
  }
};

}  // namespace

void TwoViewGeometry::Invert() {
  if (F) {
    F->transposeInPlace();
  }
  if (E) {
    E->transposeInPlace();
  }
  if (H) {
    *H = H->inverse().eval();
  }
  if (cam2_from_cam1) {
    cam2_from_cam1 = Inverse(*cam2_from_cam1);
  }
  for (auto& match : inlier_matches) {
    std::swap(match.point2D_idx1, match.point2D_idx2);
  }
}

void PreserveInlierLoopClosureProvenance(const TwoViewGeometry& prior,
                                         TwoViewGeometry* updated) {
  THROW_CHECK_NOTNULL(updated);
  if (prior.inlier_matches_are_lc.empty()) {
    updated->inlier_matches_are_lc.clear();
    updated->is_loop_closure = prior.is_loop_closure;
    if (prior.is_loop_closure && !updated->inlier_matches.empty()) {
      updated->inlier_matches_are_lc.assign(updated->inlier_matches.size(),
                                            true);
    }
    return;
  }

  THROW_CHECK_EQ(prior.inlier_matches_are_lc.size(),
                 prior.inlier_matches.size());

  std::unordered_map<FeatureMatchKey, bool, FeatureMatchKeyHash>
      prior_inlier_is_lc;
  prior_inlier_is_lc.reserve(prior.inlier_matches.size());
  for (size_t i = 0; i < prior.inlier_matches.size(); ++i) {
    const FeatureMatch& match = prior.inlier_matches[i];
    prior_inlier_is_lc[{match.point2D_idx1, match.point2D_idx2}] =
        prior.inlier_matches_are_lc[i];
  }

  updated->inlier_matches_are_lc.assign(updated->inlier_matches.size(), false);
  for (size_t i = 0; i < updated->inlier_matches.size(); ++i) {
    const FeatureMatch& match = updated->inlier_matches[i];
    const auto it =
        prior_inlier_is_lc.find({match.point2D_idx1, match.point2D_idx2});
    if (it != prior_inlier_is_lc.end()) {
      updated->inlier_matches_are_lc[i] = it->second;
    }
  }
  updated->is_loop_closure =
      std::any_of(updated->inlier_matches_are_lc.begin(),
                  updated->inlier_matches_are_lc.end(),
                  [](const bool value) { return value; });
}

}  // namespace colmap
