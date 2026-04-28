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
#include "colmap/scene/point3d.h"
#include "colmap/util/types.h"

#include <functional>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <Eigen/Core>

namespace colmap {

// Options for ``EstablishTracksFromCorrGraph``.
struct TrackEstablishmentOptions {
  // Maximum L2 distance (in pixels) between intra-image observations within
  // the same track. Tracks exceeding this on any image are discarded.
  double intra_image_consistency_threshold = 10.0;

  // Minimum distinct images per track (post intra-image consistency).
  int min_num_views_per_track = 3;

  // Greedy length-sorted subsample target: keep tracks until each registered
  // image has at least this many track elements. INT_MAX (default) makes the
  // subsample a no-op (every consistent + length-filtered track is kept).
  int required_tracks_per_view = std::numeric_limits<int>::max();
};

// Predicate: returns true to exclude a match from the union-find phase.
// Arguments are the two endpoints of the match. Optional hook for callers
// that need to skip specific matches (e.g. flagged with downstream
// metadata) without modifying the underlying CorrespondenceGraph.
using MatchPredicate =
    std::function<bool(image_t, point2D_t, image_t, point2D_t)>;

// Build tracks from a correspondence graph via union-find over inlier
// matches, intra-image consistency check, length filter, and an optional
// greedy length-sorted subsample. Returns ``{point3D_id: Point3D}`` with
// each ``Point3D::track`` populated; xyz / color / is_initialized are
// default-constructed and left for downstream triangulation to fill.
//
// Inputs:
//   * ``valid_pair_ids``: image pairs to iterate. Native callers pass
//     ``pose_graph.ValidEdges() | keys``; callers pass the subset of edges to
//     consider (e.g. ``correspondence_graph.image_pairs`` with
//     ``is_valid==true``).
//   * ``corr_graph``: ``ExtractMatchesBetweenImages`` reads the inlier
//     matches stored as ``flat_corrs`` (already filtered to RANSAC inliers
//     by the geom-verify step).
//   * ``image_id_to_keypoints``: per-image 2D keypoints used for the
//     intra-image consistency check.
//   * ``options``: the three thresholds above.
//   * ``ignore_match``: optional. When non-null and returns true for a
//     match, that match is skipped in the union-find phase.
std::unordered_map<point3D_t, Point3D> EstablishTracksFromCorrGraph(
    const std::vector<image_pair_t>& valid_pair_ids,
    const CorrespondenceGraph& corr_graph,
    const std::unordered_map<image_t, std::vector<Eigen::Vector2d>>&
        image_id_to_keypoints,
    const TrackEstablishmentOptions& options,
    const MatchPredicate& ignore_match = MatchPredicate());

// Options for ``SubsampleTracks``.
struct TrackSubsampleOptions {
  // Lower bound on track length (``Track::Length()``). Tracks below this
  // are dropped.
  int min_num_views_per_track = 3;
  // Upper bound on track length (``Track::Length()``). Tracks above this
  // are dropped.
  int max_num_views_per_track = std::numeric_limits<int>::max();
  // Greedy quota target: per-image counter that throttles selection
  // once an image has accumulated this many tracks. ``INT_MAX`` (the
  // default) makes the subsample a near-no-op.
  int required_tracks_per_view = std::numeric_limits<int>::max();
  // Hard cap on selected track count. When the count exceeds this the
  // greedy walk stops.
  int max_num_tracks = std::numeric_limits<int>::max();
};

// Greedy length-sorted subsample. Walks tracks in descending
// ``Track::Length()`` order; for each, increments per-image counters
// for every observation regardless of whether the track is kept, and
// inserts the track if any observation's pre-increment count was within
// the ``required_tracks_per_view`` quota. Returns the selected subset.
std::unordered_map<point3D_t, Point3D> SubsampleTracks(
    const TrackSubsampleOptions& options,
    const std::unordered_set<image_t>& registered_image_ids,
    const std::unordered_map<point3D_t, Point3D>& tracks_full);

}  // namespace colmap
