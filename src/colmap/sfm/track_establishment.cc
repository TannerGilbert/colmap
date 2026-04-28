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

#include "colmap/sfm/track_establishment.h"

#include "colmap/math/union_find.h"
#include "colmap/util/logging.h"
#include "colmap/util/types.h"

#include <algorithm>
#include <utility>

namespace colmap {

std::unordered_map<point3D_t, Point3D> EstablishTracksFromCorrGraph(
    const std::vector<image_pair_t>& valid_pair_ids,
    const CorrespondenceGraph& corr_graph,
    const std::unordered_map<image_t, std::vector<Eigen::Vector2d>>&
        image_id_to_keypoints,
    const TrackEstablishmentOptions& options,
    const MatchPredicate& ignore_match) {
  using Observation = std::pair<image_t, point2D_t>;

  // Union all matching observations. Iterate ``image_pair.matches`` indexed
  // by ``image_pair.inliers`` directly — works for both the native colmap
  // pipeline (geom-verify writes inliers) and downstream pipelines that
  // populate the same fields without going through the flat_corrs
  // ``FinalizeAfterMatchingComplete`` path.
  UnionFind<Observation> uf;
  for (const image_pair_t pair_id : valid_pair_ids) {
    const auto [image_id1, image_id2] = PairIdToImagePair(pair_id);
    THROW_CHECK(image_id_to_keypoints.count(image_id1))
        << "Missing keypoints for image " << image_id1;
    THROW_CHECK(image_id_to_keypoints.count(image_id2))
        << "Missing keypoints for image " << image_id2;
    const auto& image_pair = corr_graph.ImagePairsMap().at(pair_id);
    const Eigen::MatrixXi& matches = image_pair.matches;
    for (const int idx : image_pair.inliers) {
      const point2D_t p2d1 = static_cast<point2D_t>(matches(idx, 0));
      const point2D_t p2d2 = static_cast<point2D_t>(matches(idx, 1));
      if (ignore_match && ignore_match(image_id1, p2d1, image_id2, p2d2)) {
        continue;
      }
      const Observation obs1(image_id1, p2d1);
      const Observation obs2(image_id2, p2d2);
      if (obs2 < obs1) {
        uf.Union(obs1, obs2);
      } else {
        uf.Union(obs2, obs1);
      }
    }
  }

  // Group observations by their root.
  uf.Compress();
  std::unordered_map<Observation, std::vector<Observation>> track_map;
  for (const auto& [obs, root] : uf.Parents()) {
    track_map[root].push_back(obs);
  }
  LOG(INFO) << "Established " << track_map.size() << " tracks from "
            << uf.Parents().size() << " observations";

  // Validate tracks, check consistency, collect valid ones with lengths.
  std::unordered_map<point3D_t, Point3D> candidate_points3D;
  std::vector<std::pair<size_t, point3D_t>> track_lengths;
  size_t discarded_counter = 0;
  point3D_t next_point3D_id = 0;

  for (const auto& [track_id, observations] : track_map) {
    std::unordered_map<image_t, std::vector<Eigen::Vector2d>> image_id_set;
    Point3D point3D;
    bool is_consistent = true;

    for (const auto& [image_id, feature_id] : observations) {
      const Eigen::Vector2d& xy =
          image_id_to_keypoints.at(image_id).at(feature_id);

      auto it = image_id_set.find(image_id);
      if (it != image_id_set.end()) {
        for (const auto& existing_xy : it->second) {
          const double sq_threshold =
              options.intra_image_consistency_threshold *
              options.intra_image_consistency_threshold;
          if ((existing_xy - xy).squaredNorm() > sq_threshold) {
            is_consistent = false;
            break;
          }
        }
        if (!is_consistent) {
          ++discarded_counter;
          break;
        }
        it->second.push_back(xy);
      } else {
        image_id_set[image_id].push_back(xy);
      }
      point3D.track.AddElement(image_id, feature_id);
    }

    if (!is_consistent) continue;

    const size_t num_images = image_id_set.size();
    if (num_images < static_cast<size_t>(options.min_num_views_per_track))
      continue;

    const point3D_t point3D_id = next_point3D_id++;
    track_lengths.emplace_back(point3D.track.Length(), point3D_id);
    candidate_points3D.emplace(point3D_id, std::move(point3D));
  }

  LOG(INFO) << "Kept " << candidate_points3D.size() << " tracks, discarded "
            << discarded_counter << " due to inconsistency";

  // Sort tracks by length (descending) and apply greedy subsample.
  std::sort(track_lengths.begin(), track_lengths.end(), std::greater<>());

  std::unordered_map<image_t, size_t> tracks_per_image;
  size_t images_left = image_id_to_keypoints.size();
  std::unordered_map<point3D_t, Point3D> selected;
  for (const auto& [track_length, point3D_id] : track_lengths) {
    auto& point3D = candidate_points3D.at(point3D_id);

    // Check if any image in this track still needs more observations.
    const bool should_add = std::any_of(
        point3D.track.Elements().begin(),
        point3D.track.Elements().end(),
        [&](const auto& obs) {
          return tracks_per_image[obs.image_id] <=
                 static_cast<size_t>(options.required_tracks_per_view);
        });
    if (!should_add) continue;

    // Update image counts.
    for (const auto& obs : point3D.track.Elements()) {
      auto& count = tracks_per_image[obs.image_id];
      if (count == static_cast<size_t>(options.required_tracks_per_view))
        --images_left;
      ++count;
    }

    selected.emplace(point3D_id, std::move(point3D));

    if (images_left == 0) break;
  }

  LOG(INFO) << "Before greedy subsample: " << candidate_points3D.size()
            << ", after: " << selected.size();
  return selected;
}

std::unordered_map<point3D_t, Point3D> SubsampleTracks(
    const TrackSubsampleOptions& options,
    const std::unordered_set<image_t>& registered_image_ids,
    const std::unordered_map<point3D_t, Point3D>& tracks_full) {
  std::vector<std::pair<size_t, point3D_t>> track_lengths;
  size_t dropped_by_length = 0;
  for (const auto& [track_id, point3D] : tracks_full) {
    const size_t length = point3D.track.Length();
    if (length < static_cast<size_t>(options.min_num_views_per_track) ||
        length > static_cast<size_t>(options.max_num_views_per_track)) {
      ++dropped_by_length;
      continue;
    }
    track_lengths.emplace_back(length, track_id);
  }
  std::sort(track_lengths.begin(), track_lengths.end(), std::greater<>());

  // Selection domain = registered images.
  std::unordered_map<image_t, int> tracks_per_camera;
  for (const image_t image_id : registered_image_ids) {
    tracks_per_camera[image_id] = 0;
  }

  std::unordered_map<point3D_t, Point3D> selected;
  int cameras_left = static_cast<int>(tracks_per_camera.size());
  for (const auto& [track_length, track_id] : track_lengths) {
    const Point3D& src = tracks_full.at(track_id);

    // Restrict to selection domain.
    Point3D candidate;
    for (const auto& el : src.track.Elements()) {
      if (tracks_per_camera.count(el.image_id) == 0) continue;
      candidate.track.AddElement(el);
    }
    if (candidate.track.Length() <
        static_cast<size_t>(options.min_num_views_per_track)) {
      continue;
    }

    // Greedy quota: a track is added if any element's PRE-increment
    // count is within the target. Counters increment for every kept
    // element regardless of whether the track was added.
    bool added = false;
    for (const auto& el : candidate.track.Elements()) {
      auto& count = tracks_per_camera[el.image_id];
      if (count > options.required_tracks_per_view) continue;
      ++count;
      if (count > options.required_tracks_per_view) --cameras_left;
      if (!added) {
        selected.emplace(track_id, candidate);
        added = true;
      }
    }
    if (cameras_left == 0) break;
    if (static_cast<int>(selected.size()) > options.max_num_tracks) break;
  }
  LOG(INFO) << "Subsampled to " << selected.size() << " tracks (dropped "
            << (tracks_full.size() - selected.size()) << ", "
            << dropped_by_length << " by length)";
  return selected;
}

}  // namespace colmap
