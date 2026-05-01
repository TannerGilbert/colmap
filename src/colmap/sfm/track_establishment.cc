#include "colmap/sfm/track_establishment.h"

#include "colmap/feature/types.h"
#include "colmap/math/union_find.h"
#include "colmap/util/logging.h"

#include <algorithm>
#include <memory>
#include <set>
#include <utility>

namespace colmap {

namespace {

// Encodes (image_id, point2D_idx) into a single 64-bit key for fast LC-pass
// lookups.
inline uint64_t EncodeObservationKey(image_t image_id, point2D_t feature_id) {
  return (static_cast<uint64_t>(image_id) << 32) |
         static_cast<uint64_t>(feature_id);
}

void ValidateLoopClosureImagePairMetadata(
    image_pair_t pair_id,
    const CorrespondenceGraph::ImagePair& image_pair) {
  const int num_matches = image_pair.matches.rows();
  THROW_CHECK_EQ(image_pair.are_lc.size(), static_cast<size_t>(num_matches))
      << "Malformed LC metadata for image pair " << pair_id
      << ": are_lc.size() must match matches.rows()";
  for (const int idx : image_pair.inliers) {
    THROW_CHECK_GE(idx, 0)
        << "Malformed LC metadata for image pair " << pair_id
        << ": negative inlier index";
    THROW_CHECK_LT(idx, num_matches)
        << "Malformed LC metadata for image pair " << pair_id
        << ": inlier index outside matches.rows()";
  }
}

}  // namespace

MatchPredicate MakeLoopClosureMatchPredicate(
    const std::vector<image_pair_t>& valid_pair_ids,
    const CorrespondenceGraph& corr_graph) {
  // Build a set of exact LC-flagged match pairs. Only those pairwise
  // constraints are suppressed from the first union-find pass; the same
  // endpoint may still participate in regular tracks through non-LC matches.
  using LCKey = std::pair<uint64_t, uint64_t>;
  auto lc_matches = std::make_shared<std::set<LCKey>>();
  for (const image_pair_t pair_id : valid_pair_ids) {
    const auto it = corr_graph.ImagePairsMap().find(pair_id);
    if (it == corr_graph.ImagePairsMap().end()) continue;
    const auto [image_id1, image_id2] = PairIdToImagePair(pair_id);
    const auto& image_pair = it->second;
    ValidateLoopClosureImagePairMetadata(pair_id, image_pair);
    const Eigen::MatrixXi& matches = image_pair.matches;
    const std::vector<int>& inliers = image_pair.inliers;
    const std::vector<bool>& are_lc = image_pair.are_lc;
    for (const int idx : inliers) {
      if (are_lc[idx]) {
        const uint64_t key1 = EncodeObservationKey(
            image_id1, static_cast<point2D_t>(matches(idx, 0)));
        const uint64_t key2 = EncodeObservationKey(
            image_id2, static_cast<point2D_t>(matches(idx, 1)));
        lc_matches->insert({key1, key2});
        lc_matches->insert({key2, key1});
      }
    }
  }
  return [lc_matches = std::move(lc_matches)](image_t image_id1,
                                              point2D_t p1,
                                              image_t image_id2,
                                              point2D_t p2) -> bool {
    return lc_matches->count({EncodeObservationKey(image_id1, p1),
                              EncodeObservationKey(image_id2, p2)}) > 0;
  };
}

std::unordered_map<point3D_t, Point3D> EstablishTracksFromCorrGraph(
    const std::vector<image_pair_t>& valid_pair_ids,
    const CorrespondenceGraph& corr_graph,
    const std::unordered_map<image_t, std::vector<Eigen::Vector2d>>&
        image_id_to_keypoints,
    const TrackEstablishmentOptions& options,
    const MatchPredicate& ignore_match) {
  using Observation = std::pair<image_t, point2D_t>;

  // Union all matching observations. Iterate ImagePair metadata directly:
  // VideoSfM populates matches/inliers on ImagePair without always using the
  // flat correspondence graph storage behind ExtractMatchesBetweenImages.
  // Fall back to the native correspondence storage for vanilla COLMAP pairs.
  UnionFind<Observation> uf;
  FeatureMatches extracted_matches;
  for (const image_pair_t pair_id : valid_pair_ids) {
    const auto [image_id1, image_id2] = PairIdToImagePair(pair_id);
    THROW_CHECK(image_id_to_keypoints.count(image_id1))
        << "Missing keypoints for image " << image_id1;
    THROW_CHECK(image_id_to_keypoints.count(image_id2))
        << "Missing keypoints for image " << image_id2;
    const auto& image_pair = corr_graph.ImagePairsMap().at(pair_id);
    const Eigen::MatrixXi& matches = image_pair.matches;
    const auto union_match = [&](const point2D_t p2d1, const point2D_t p2d2) {
      if (ignore_match && ignore_match(image_id1, p2d1, image_id2, p2d2)) {
        return;
      }
      const Observation obs1(image_id1, p2d1);
      const Observation obs2(image_id2, p2d2);
      if (obs2 < obs1) {
        uf.Union(obs1, obs2);
      } else {
        uf.Union(obs2, obs1);
      }
    };
    if (matches.rows() > 0 || !image_pair.inliers.empty()) {
      for (const int idx : image_pair.inliers) {
        THROW_CHECK_GE(idx, 0) << "Negative inlier index for image pair "
                               << pair_id;
        THROW_CHECK_LT(idx, matches.rows())
            << "Inlier index outside matches.rows() for image pair " << pair_id;
        union_match(static_cast<point2D_t>(matches(idx, 0)),
                    static_cast<point2D_t>(matches(idx, 1)));
      }
    } else {
      corr_graph.ExtractMatchesBetweenImages(
          image_id1, image_id2, extracted_matches);
      for (const auto& match : extracted_matches) {
        union_match(match.point2D_idx1, match.point2D_idx2);
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


void AppendLoopClosureObservations(
    const std::vector<image_pair_t>& valid_pair_ids,
    const CorrespondenceGraph& corr_graph,
    std::unordered_map<point3D_t, Point3D>& tracks) {
  // Build the lookup from observation -> track_id, and find the next
  // free track id (max + 1) so newly-minted LC-only tracks never
  // collide with the dense [0, N) ids written by
  // EstablishTracksFromCorrGraph.
  std::unordered_map<uint64_t, point3D_t> obs_to_track;
  point3D_t next_id = 0;
  for (const auto& [track_id, point3D] : tracks) {
    next_id = std::max(next_id, static_cast<point3D_t>(track_id + 1));
    for (const auto& el : point3D.track.Elements()) {
      obs_to_track.emplace(EncodeObservationKey(el.image_id, el.point2D_idx),
                           track_id);
    }
  }

  for (const image_pair_t pair_id : valid_pair_ids) {
    const auto& image_pair = corr_graph.ImagePairsMap().at(pair_id);
    ValidateLoopClosureImagePairMetadata(pair_id, image_pair);
    const Eigen::MatrixXi& matches = image_pair.matches;
    const std::vector<int>& inliers = image_pair.inliers;
    const std::vector<bool>& are_lc = image_pair.are_lc;

    // Skip pairs without any LC inliers (cheap pre-check).
    bool has_lc = false;
    for (const int idx : inliers) {
      if (are_lc[idx]) {
        has_lc = true;
        break;
      }
    }
    if (!has_lc) continue;

    const auto [image_id1, image_id2] = PairIdToImagePair(pair_id);

    for (const int idx : inliers) {
      if (!are_lc[idx]) {
        continue;
      }
      const point2D_t p1 = static_cast<point2D_t>(matches(idx, 0));
      const point2D_t p2 = static_cast<point2D_t>(matches(idx, 1));
      const uint64_t key1 = EncodeObservationKey(image_id1, p1);
      const uint64_t key2 = EncodeObservationKey(image_id2, p2);

      auto it1 = obs_to_track.find(key1);
      auto it2 = obs_to_track.find(key2);
      const bool has_track1 = (it1 != obs_to_track.end());
      const bool has_track2 = (it2 != obs_to_track.end());

      if (!has_track1 && !has_track2) {
        // Mint two reciprocal LC-only tracks. Each gets the regular
        // observation as a Track::Element and the other side as
        // lc_elements.
        const point3D_t tid_a = next_id++;
        const point3D_t tid_b = next_id++;
        Point3D track_a;
        track_a.track.AddElement(image_id1, p1);
        track_a.track.lc_elements.emplace_back(image_id2, p2);
        Point3D track_b;
        track_b.track.AddElement(image_id2, p2);
        track_b.track.lc_elements.emplace_back(image_id1, p1);
        const auto inserted_a = tracks.emplace(tid_a, std::move(track_a));
        THROW_CHECK(inserted_a.second)
            << "Track id collision on " << tid_a
            << " — sequential id minting violated unexpectedly";
        const auto inserted_b = tracks.emplace(tid_b, std::move(track_b));
        THROW_CHECK(inserted_b.second)
            << "Track id collision on " << tid_b
            << " — sequential id minting violated unexpectedly";
        obs_to_track[key1] = tid_a;
        obs_to_track[key2] = tid_b;
        continue;
      }
      if (has_track1 && has_track2) {
        const point3D_t t1 = it1->second;
        const point3D_t t2 = it2->second;
        if (t1 != t2) {
          tracks.at(t1).track.lc_elements.emplace_back(image_id2, p2);
          tracks.at(t2).track.lc_elements.emplace_back(image_id1, p1);
        }
        continue;
      }
      if (has_track1) {
        tracks.at(it1->second).track.lc_elements.emplace_back(image_id2, p2);
      } else {
        tracks.at(it2->second).track.lc_elements.emplace_back(image_id1, p1);
      }
    }
  }
}

std::unordered_map<point3D_t, Point3D> SubsampleTracks(
    const TrackSubsampleOptions& options,
    const std::unordered_set<image_t>& registered_image_ids,
    const std::unordered_map<point3D_t, Point3D>& tracks_full) {
  // Length filter: lower bound counts regular + LC observations; upper
  // bound counts regular only. The asymmetry is intentional.
  std::vector<std::pair<size_t, point3D_t>> track_lengths;
  size_t dropped_by_length = 0;
  for (const auto& [track_id, point3D] : tracks_full) {
    const size_t total =
        point3D.track.Length() + point3D.track.lc_elements.size();
    if (total < static_cast<size_t>(options.min_num_views_per_track) ||
        point3D.track.Length() >
            static_cast<size_t>(options.max_num_views_per_track)) {
      ++dropped_by_length;
      continue;
    }
    track_lengths.emplace_back(point3D.track.Length(), track_id);
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

    // Restrict to selection domain + lc-elements that fall in the
    // selection domain.
    Point3D candidate;
    for (const auto& el : src.track.Elements()) {
      if (tracks_per_camera.count(el.image_id) == 0) continue;
      candidate.track.AddElement(el);
    }
    for (const auto& lc_el : src.track.lc_elements) {
      if (tracks_per_camera.count(lc_el.image_id) == 0) continue;
      candidate.track.lc_elements.emplace_back(lc_el);
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
