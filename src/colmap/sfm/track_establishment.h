#pragma once

#include "colmap/scene/correspondence_graph.h"
#include "colmap/scene/point3d.h"
#include "colmap/util/types.h"

#include <limits>
#include <unordered_map>
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

// Build tracks via union-find over inlier matches with consistency check,
// length filter, and optional greedy subsample. Returns {point3D_id: Point3D}
// with Track populated; xyz/color left for triangulation.
std::unordered_map<point3D_t, Point3D> EstablishTracksFromCorrGraph(
    const std::vector<image_pair_t>& valid_pair_ids,
    const CorrespondenceGraph& corr_graph,
    const std::unordered_map<image_t, std::vector<Eigen::Vector2d>>&
        image_id_to_keypoints,
    const TrackEstablishmentOptions& options);

}  // namespace colmap
