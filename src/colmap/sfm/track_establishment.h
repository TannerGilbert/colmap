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

// Predicate called for each match (image_id1, point2D_idx1, image_id2,
// point2D_idx2). Returns true to skip (ignore) the match.
using MatchPredicate =
    std::function<bool(image_t, point2D_t, image_t, point2D_t)>;

// Returns a MatchPredicate that ignores exact LC-flagged inlier matches
// (are_lc==true) in the correspondence graph. This excludes LC pairwise
// constraints from the first-pass union-find while still allowing the same
// endpoints to participate in regular tracks through non-LC matches.
MatchPredicate MakeLoopClosureMatchPredicate(
    const std::vector<image_pair_t>& valid_pair_ids,
    const CorrespondenceGraph& corr_graph);

// Build tracks via union-find over matches extracted from the correspondence
// graph with consistency check, length filter, and optional greedy subsample.
// Returns {point3D_id: Point3D} with Track populated; xyz/color left for
// triangulation.
std::unordered_map<point3D_t, Point3D> EstablishTracksFromCorrGraph(
    const std::vector<image_pair_t>& valid_pair_ids,
    const CorrespondenceGraph& corr_graph,
    const std::unordered_map<image_t, std::vector<Eigen::Vector2d>>&
        image_id_to_keypoints,
    const TrackEstablishmentOptions& options,
    const MatchPredicate& ignore_match = {});

// Append LC observations to existing tracks as Track::lc_elements.
// 4 cases: neither/both-distinct/both-same/one-side has a track.
// New track ids minted from max(track_id)+1.
void AppendLoopClosureObservations(
    const std::vector<image_pair_t>& valid_pair_ids,
    const CorrespondenceGraph& corr_graph,
    std::unordered_map<point3D_t, Point3D>& tracks);

struct TrackProblemFilterOptions {
  int min_num_views_per_track = 3;
  int max_num_views_per_track = std::numeric_limits<int>::max();
};

// Filter tracks for the optimization problem. LC observations may augment an
// admitted regular track, but do not satisfy the regular-view admission gate.
std::unordered_map<point3D_t, Point3D> FilterTracksForProblem(
    const TrackProblemFilterOptions& options,
    const std::unordered_set<image_t>& registered_image_ids,
    const std::unordered_map<point3D_t, Point3D>& tracks_full);

}  // namespace colmap
