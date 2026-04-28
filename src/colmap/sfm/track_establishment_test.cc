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

#include "colmap/scene/correspondence_graph.h"
#include "colmap/scene/image.h"
#include "colmap/scene/point3d.h"
#include "colmap/util/types.h"

#include <algorithm>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <Eigen/Core>
#include <gtest/gtest.h>

namespace colmap {
namespace {

// Build an ImagePair entry directly into the corr_graph's image_pairs map
// with the ``matches`` and ``inliers`` fields populated.
// ``EstablishTracksFromCorrGraph`` only reads those two fields plus the pair
// keys, so we bypass ``AddTwoViewGeometry`` and the colmap flat_corrs path.
void AddImagePair(CorrespondenceGraph& corr_graph,
                  image_t image_id1,
                  image_t image_id2,
                  const std::vector<std::pair<int, int>>& matches,
                  const std::vector<int>& inliers) {
  const image_pair_t pair_id = ImagePairToPairId(image_id1, image_id2);
  CorrespondenceGraph::ImagePair image_pair(image_id1, image_id2);
  Eigen::MatrixXi matches_mat(static_cast<int>(matches.size()), 2);
  for (size_t i = 0; i < matches.size(); ++i) {
    matches_mat(static_cast<int>(i), 0) = matches[i].first;
    matches_mat(static_cast<int>(i), 1) = matches[i].second;
  }
  image_pair.matches = std::move(matches_mat);
  image_pair.inliers = inliers;
  image_pair.num_matches = static_cast<point2D_t>(inliers.size());
  corr_graph.MutableImagePairs().emplace(pair_id, std::move(image_pair));
}

// Build a map ``image_id -> [Vector2d(0,0), Vector2d(1,1), ...]`` so all
// keypoints on a given image are well-separated (intra-image consistency
// trivially holds when all features are at distinct integer pixels).
std::unordered_map<image_t, std::vector<Eigen::Vector2d>>
MakeWellSeparatedKeypoints(const std::vector<image_t>& image_ids,
                           int num_points_per_image) {
  std::unordered_map<image_t, std::vector<Eigen::Vector2d>> result;
  for (const image_t image_id : image_ids) {
    auto& kps = result[image_id];
    kps.reserve(num_points_per_image);
    for (int i = 0; i < num_points_per_image; ++i) {
      // Spread keypoints 100px apart on both axes so every intra-image
      // pair is far above the default 10px threshold.
      kps.emplace_back(100.0 * i, 100.0 * i);
    }
  }
  return result;
}

// Helper: collect keys of corr_graph.image_pairs into a vector.
std::vector<image_pair_t> CollectPairIds(const CorrespondenceGraph& g) {
  std::vector<image_pair_t> ids;
  ids.reserve(g.ImagePairsMap().size());
  for (const auto& [pair_id, _] : g.ImagePairsMap()) ids.push_back(pair_id);
  return ids;
}

// 3 images, 3 valid pairs (1-2, 1-3, 2-3), 5 inlier matches per pair, all
// pointing to the same 5 underlying 3D points (feature index i corresponds
// to point i on every image). Expect 5 tracks of length 3.
TEST(TrackEstablishment, Basic3PairTriangle) {
  CorrespondenceGraph corr_graph;
  corr_graph.AddImage(1, 5);
  corr_graph.AddImage(2, 5);
  corr_graph.AddImage(3, 5);

  std::vector<std::pair<int, int>> matches;
  std::vector<int> inliers;
  for (int i = 0; i < 5; ++i) {
    matches.emplace_back(i, i);
    inliers.push_back(i);
  }
  AddImagePair(corr_graph, 1, 2, matches, inliers);
  AddImagePair(corr_graph, 1, 3, matches, inliers);
  AddImagePair(corr_graph, 2, 3, matches, inliers);

  const auto keypoints = MakeWellSeparatedKeypoints({1, 2, 3}, 5);
  const TrackEstablishmentOptions options;  // defaults: min_views=3
  const auto tracks = EstablishTracksFromCorrGraph(
      CollectPairIds(corr_graph), corr_graph, keypoints, options);

  EXPECT_EQ(tracks.size(), 5);
  for (const auto& [pid, point3D] : tracks) {
    EXPECT_EQ(point3D.track.Length(), 3);
  }
}

// Same geometry but ``min_num_views_per_track = 4`` rejects every length-3
// track.
TEST(TrackEstablishment, LengthFilterDropsShortTracks) {
  CorrespondenceGraph corr_graph;
  corr_graph.AddImage(1, 5);
  corr_graph.AddImage(2, 5);
  corr_graph.AddImage(3, 5);

  std::vector<std::pair<int, int>> matches;
  std::vector<int> inliers;
  for (int i = 0; i < 5; ++i) {
    matches.emplace_back(i, i);
    inliers.push_back(i);
  }
  AddImagePair(corr_graph, 1, 2, matches, inliers);
  AddImagePair(corr_graph, 1, 3, matches, inliers);
  AddImagePair(corr_graph, 2, 3, matches, inliers);

  const auto keypoints = MakeWellSeparatedKeypoints({1, 2, 3}, 5);
  TrackEstablishmentOptions options;
  options.min_num_views_per_track = 4;
  const auto tracks = EstablishTracksFromCorrGraph(
      CollectPairIds(corr_graph), corr_graph, keypoints, options);

  EXPECT_TRUE(tracks.empty());
}

// Drive two distinct features on image 1 (idx 0 and idx 1) into the same
// union-find root via image 2's feature 0:
//   pair (1,2): match (img1:0 <-> img2:0)
//   pair (2,3): match (img2:0 <-> img3:0) -- chain via image 3
//   pair (1,3): match (img1:1 <-> img3:0) -- this fuses img1:0 ~ img1:1
// keypoints for img1:0 = (0,0) and img1:1 = (1000,1000) -> intra-image
// consistency violated (>10px), track dropped.
// 4 images so the surviving tracks (built from idx 1..4) still have length 3.
TEST(TrackEstablishment, IntraImageConsistencyDropsInconsistentTrack) {
  CorrespondenceGraph corr_graph;
  corr_graph.AddImage(1, 5);
  corr_graph.AddImage(2, 5);
  corr_graph.AddImage(3, 5);

  // Inconsistent fusion path for feature 0 chain.
  AddImagePair(corr_graph, 1, 2, {{0, 0}}, {0});
  AddImagePair(corr_graph, 2, 3, {{0, 0}}, {0});
  AddImagePair(corr_graph, 1, 3, {{1, 0}}, {0});

  // Keypoints: image 1 has feature 0 and feature 1 placed FAR apart so the
  // intra-image consistency check rejects the merged track.
  std::unordered_map<image_t, std::vector<Eigen::Vector2d>> keypoints;
  keypoints[1] = {Eigen::Vector2d(0, 0), Eigen::Vector2d(1000, 1000),
                  Eigen::Vector2d(200, 200), Eigen::Vector2d(300, 300),
                  Eigen::Vector2d(400, 400)};
  keypoints[2] = {Eigen::Vector2d(0, 0), Eigen::Vector2d(100, 100),
                  Eigen::Vector2d(200, 200), Eigen::Vector2d(300, 300),
                  Eigen::Vector2d(400, 400)};
  keypoints[3] = {Eigen::Vector2d(0, 0), Eigen::Vector2d(100, 100),
                  Eigen::Vector2d(200, 200), Eigen::Vector2d(300, 300),
                  Eigen::Vector2d(400, 400)};

  TrackEstablishmentOptions options;
  options.min_num_views_per_track = 2;  // accept length-2 if any survived
  options.intra_image_consistency_threshold = 10.0;
  const auto tracks = EstablishTracksFromCorrGraph(
      CollectPairIds(corr_graph), corr_graph, keypoints, options);

  // The single track we constructed is inconsistent; nothing else exists.
  EXPECT_TRUE(tracks.empty());
}

// ``ignore_match`` returning true for any match touching image 1 strips
// image 1's contribution from union-find. The remaining 5 tracks then have
// length 2 (only images 2 and 3) so default min_views=3 drops them all;
// loosening to min_views=2 surfaces them and asserts none of the kept tracks
// reference image 1.
TEST(TrackEstablishment, IgnoreMatchPredicateDropsImage) {
  CorrespondenceGraph corr_graph;
  corr_graph.AddImage(1, 5);
  corr_graph.AddImage(2, 5);
  corr_graph.AddImage(3, 5);

  std::vector<std::pair<int, int>> matches;
  std::vector<int> inliers;
  for (int i = 0; i < 5; ++i) {
    matches.emplace_back(i, i);
    inliers.push_back(i);
  }
  AddImagePair(corr_graph, 1, 2, matches, inliers);
  AddImagePair(corr_graph, 1, 3, matches, inliers);
  AddImagePair(corr_graph, 2, 3, matches, inliers);

  const auto keypoints = MakeWellSeparatedKeypoints({1, 2, 3}, 5);
  const auto pair_ids = CollectPairIds(corr_graph);

  const MatchPredicate ignore_image1 =
      [](image_t i1, point2D_t /*p1*/, image_t i2, point2D_t /*p2*/) {
        return i1 == 1 || i2 == 1;
      };

  // With min_views=3, dropping image 1 leaves only length-2 tracks across
  // images 2-3, all filtered out.
  {
    TrackEstablishmentOptions options;  // default min_views=3
    const auto tracks = EstablishTracksFromCorrGraph(
        pair_ids, corr_graph, keypoints, options, ignore_image1);
    EXPECT_TRUE(tracks.empty());
  }

  // With min_views=2 the surviving tracks become visible; verify image 1 is
  // entirely absent.
  {
    TrackEstablishmentOptions options;
    options.min_num_views_per_track = 2;
    const auto tracks = EstablishTracksFromCorrGraph(
        pair_ids, corr_graph, keypoints, options, ignore_image1);
    EXPECT_EQ(tracks.size(), 5);
    for (const auto& [pid, point3D] : tracks) {
      EXPECT_EQ(point3D.track.Length(), 2);
      for (const auto& el : point3D.track.Elements()) {
        EXPECT_NE(el.image_id, 1u);
      }
    }
  }
}

// Empty ``valid_pair_ids`` yields no tracks and no crash.
TEST(TrackEstablishment, EmptyInputReturnsEmpty) {
  CorrespondenceGraph corr_graph;
  std::unordered_map<image_t, std::vector<Eigen::Vector2d>> keypoints;
  TrackEstablishmentOptions options;
  const auto tracks =
      EstablishTracksFromCorrGraph({}, corr_graph, keypoints, options);
  EXPECT_TRUE(tracks.empty());
}

// ============================================================================
// FindTracksForProblem (greedy subsample + 2-view depth gate)
// ============================================================================

// Build a registered Image with N features.
Image MakeRegisteredImage(image_t image_id, int num_features) {
  Image image;
  image.SetImageId(image_id);
  image.is_registered = true;
  image.features.assign(num_features, Eigen::Vector2d::Zero());
  return image;
}

// Build a Point3D with regular elements (image_id, feature_idx) pairs; no
// LC elements.
Point3D MakePoint3DFromElements(
    const std::vector<std::pair<image_t, point2D_t>>& elements) {
  Point3D p;
  for (const auto& [image_id, point2D_idx] : elements) {
    p.track.AddElement(image_id, point2D_idx);
  }
  return p;
}

// Collect registered image ids from a test fixture.
std::unordered_set<image_t> MakeRegisteredImageIds(
    const std::unordered_map<image_t, Image>& images) {
  std::unordered_set<image_t> ids;
  for (const auto& [image_id, image] : images) {
    if (image.is_registered) ids.insert(image_id);
  }
  return ids;
}

// LengthFilter: with ``min_num_views_per_track=10`` every track here is too
// short, so SubsampleTracks returns empty. Loosening to 2 surfaces every input
// track (assuming the per-view greedy quota is satisfied).
//
// Drives SubsampleTracks.
TEST(FindTracksForProblem, LengthFilter) {
  std::unordered_map<image_t, Image> images;
  images.emplace(1, MakeRegisteredImage(1, 5));
  images.emplace(2, MakeRegisteredImage(2, 5));
  images.emplace(3, MakeRegisteredImage(3, 5));

  std::unordered_map<point3D_t, Point3D> tracks_full;
  for (point2D_t f = 0; f < 5; ++f) {
    tracks_full.emplace(f,
                        MakePoint3DFromElements({{1, f}, {2, f}, {3, f}}));
  }

  const auto reg_ids = MakeRegisteredImageIds(images);
  const std::unordered_map<image_t, std::vector<double>> empty_depths;
  const std::unordered_map<image_t, std::vector<bool>> empty_validity;

  // High-min variant: tracks have length 3, demand 10.
  {
    TrackSubsampleOptions options;
    options.min_num_views_per_track = 10;
    options.required_tracks_per_view = 1000;  // never saturate
    const auto selected = SubsampleTracks(
        options, reg_ids, empty_depths, empty_validity, tracks_full);
    EXPECT_EQ(selected.size(), 0u);
    EXPECT_TRUE(selected.empty());
  }

  // Low-min variant: every length-3 track survives.
  {
    TrackSubsampleOptions options;
    options.min_num_views_per_track = 2;
    options.required_tracks_per_view = 1000;
    const auto selected = SubsampleTracks(
        options, reg_ids, empty_depths, empty_validity, tracks_full);
    EXPECT_EQ(selected.size(), 5u);
  }
}

// MaxLengthFilter: tracks of length 5 dropped when max=4.
//
// Drives SubsampleTracks.
TEST(FindTracksForProblem, MaxLengthFilter) {
  std::unordered_map<image_t, Image> images;
  for (image_t i = 1; i <= 5; ++i) {
    images.emplace(i, MakeRegisteredImage(i, 3));
  }

  std::unordered_map<point3D_t, Point3D> tracks_full;
  for (point2D_t f = 0; f < 3; ++f) {
    tracks_full.emplace(
        f, MakePoint3DFromElements(
               {{1, f}, {2, f}, {3, f}, {4, f}, {5, f}}));
  }

  const auto reg_ids = MakeRegisteredImageIds(images);
  const std::unordered_map<image_t, std::vector<double>> empty_depths;
  const std::unordered_map<image_t, std::vector<bool>> empty_validity;

  TrackSubsampleOptions options;
  options.min_num_views_per_track = 2;
  options.max_num_views_per_track = 4;
  options.required_tracks_per_view = 1000;
  const auto selected = SubsampleTracks(
      options, reg_ids, empty_depths, empty_validity, tracks_full);
  EXPECT_EQ(selected.size(), 0u);
  EXPECT_TRUE(selected.empty());
}

// GreedyQuota: 5 length-3 tracks across 3 images, ``required_tracks_per_view=2``
// per-view quota. Greedy keeps as soon as every image is satisfied -> 2
// tracks suffice (each contributes to all 3 images at once).
//
// Drives SubsampleTracks.
TEST(FindTracksForProblem, GreedyQuota) {
  std::unordered_map<image_t, Image> images;
  for (image_t i = 1; i <= 3; ++i) {
    images.emplace(i, MakeRegisteredImage(i, 5));
  }

  std::unordered_map<point3D_t, Point3D> tracks_full;
  for (point2D_t f = 0; f < 5; ++f) {
    tracks_full.emplace(f,
                        MakePoint3DFromElements({{1, f}, {2, f}, {3, f}}));
  }

  const auto reg_ids = MakeRegisteredImageIds(images);
  const std::unordered_map<image_t, std::vector<double>> empty_depths;
  const std::unordered_map<image_t, std::vector<bool>> empty_validity;

  TrackSubsampleOptions options;
  options.min_num_views_per_track = 2;
  options.required_tracks_per_view = 2;
  const auto selected = SubsampleTracks(
      options, reg_ids, empty_depths, empty_validity, tracks_full);
  const size_t n_selected = selected.size();

  // Each track touches all 3 images so 2 tracks fully satisfy the per-view
  // quota of 2. Bound: at most all 5 input tracks; at least 2 (the quota).
  EXPECT_GE(n_selected, 2u);
  EXPECT_LE(n_selected, 5u);
}

// MinTracksPerViewBugDocumentation: enshrines the actual behaviour of the
// default ``required_tracks_per_view = INT_MAX``.
TEST(FindTracksForProblem, MinTracksPerViewBugDocumentation) {
  std::unordered_map<image_t, Image> images;
  for (image_t i = 1; i <= 3; ++i) {
    images.emplace(i, MakeRegisteredImage(i, 3));
  }

  std::unordered_map<point3D_t, Point3D> tracks_full;
  for (point2D_t f = 0; f < 3; ++f) {
    tracks_full.emplace(f,
                        MakePoint3DFromElements({{1, f}, {2, f}, {3, f}}));
  }

  const auto reg_ids = MakeRegisteredImageIds(images);
  const std::unordered_map<image_t, std::vector<double>> empty_depths;
  const std::unordered_map<image_t, std::vector<bool>> empty_validity;

  TrackSubsampleOptions options;
  options.min_num_views_per_track = 2;
  // options.required_tracks_per_view stays at default = INT_MAX.
  const auto selected = SubsampleTracks(
      options, reg_ids, empty_depths, empty_validity, tracks_full);
  // All 3 tracks are kept — the quota gate is disabled by INT_MAX bar.
  EXPECT_EQ(selected.size(), 3u);
}

// ============================================================================
// ProcessLoopClosurePairs (second pass over LC-marked inlier matches)
// ============================================================================
//
// These tests now drive the two-step ``EstablishTracksFromCorrGraph`` (with
// ``MakeLoopClosureMatchPredicate`` filtering LC matches out of the union-
// find pass) followed by ``AppendLoopClosureObservations`` (which adds LC
// observations as parallel ``Track::lc_elements``). Setup:
//   * Regular matches (``are_lc[idx]=false``) drive native track construction.
//   * LC matches (``are_lc[idx]=true``) are skipped by
//     ``MakeLoopClosureMatchPredicate`` and consumed by
//     ``AppendLoopClosureObservations``.
// The post-condition tracks dict is what we assert on.

// Variant of ``AddImagePair`` that also populates ``are_lc``. Each
// ``lc_match_indices`` entry is a row index into ``matches`` (NOT an index
// into ``inliers``); matching the indexing convention used inside
// ``AppendLoopClosureObservations``.
void AddImagePairWithLC(CorrespondenceGraph& corr_graph,
                        image_t image_id1,
                        image_t image_id2,
                        const std::vector<std::pair<int, int>>& matches,
                        const std::vector<int>& inliers,
                        const std::vector<size_t>& lc_match_indices) {
  const image_pair_t pair_id = ImagePairToPairId(image_id1, image_id2);
  CorrespondenceGraph::ImagePair image_pair(image_id1, image_id2);
  Eigen::MatrixXi matches_mat(static_cast<int>(matches.size()), 2);
  for (size_t i = 0; i < matches.size(); ++i) {
    matches_mat(static_cast<int>(i), 0) = matches[i].first;
    matches_mat(static_cast<int>(i), 1) = matches[i].second;
  }
  image_pair.matches = std::move(matches_mat);
  image_pair.inliers = inliers;
  image_pair.num_matches = static_cast<point2D_t>(inliers.size());
  std::vector<bool> are_lc(matches.size(), false);
  for (const size_t row : lc_match_indices) are_lc[row] = true;
  image_pair.are_lc = std::move(are_lc);
  corr_graph.MutableImagePairs().emplace(pair_id, std::move(image_pair));
}

// Helper: track contains (image_id, p2d_idx) as a regular element.
bool TrackHasElement(const Track& track,
                     image_t image_id,
                     point2D_t p2d_idx) {
  for (const auto& el : track.Elements()) {
    if (el.image_id == image_id && el.point2D_idx == p2d_idx) return true;
  }
  return false;
}

// Helper: track contains (image_id, p2d_idx) as an LC element.
bool TrackHasLCElement(const Track& track,
                       image_t image_id,
                       point2D_t p2d_idx) {
  for (const auto& el : track.lc_elements) {
    if (el.image_id == image_id && el.point2D_idx == p2d_idx) return true;
  }
  return false;
}

// Run the native + LC two-step end-to-end and return the populated tracks
// dict.
std::unordered_map<point3D_t, Point3D> EstablishFullTracks(
    const CorrespondenceGraph& corr_graph,
    const std::unordered_map<image_t, std::vector<Eigen::Vector2d>>& keypoints,
    const TrackEstablishmentOptions& options) {
  const auto pair_ids = CollectPairIds(corr_graph);
  const auto ignore_lc =
      MakeLoopClosureMatchPredicate(pair_ids, corr_graph);
  auto tracks = EstablishTracksFromCorrGraph(
      pair_ids, corr_graph, keypoints, options, ignore_lc);
  AppendLoopClosureObservations(pair_ids, corr_graph, tracks);
  return tracks;
}

// Both endpoints of the LC match already lie on existing native tracks, but
// on DIFFERENT tracks. Expect reciprocal lc_elements added to both; tracks
// stay separate (no merge).
//
// Setup: regular triangle (1,2,3) with feature i <-> feature i, plus a
// regular pair (3,4) extending each feature-i track to image 4 -> 5 native
// tracks of length 4. LC pair (1,4) with one match (img1:feat=0 <->
// img4:feat=1): feat-0 chain holds img1:0; feat-1 chain holds img4:1.
//
// Drives EstablishTracksFromCorrGraph + AppendLoopClosureObservations.
TEST(ProcessLoopClosurePairs, BothExistingTracks) {
  CorrespondenceGraph corr_graph;
  for (image_t i = 1; i <= 4; ++i) corr_graph.AddImage(i, 5);

  std::vector<std::pair<int, int>> reg_matches;
  std::vector<int> reg_inliers;
  for (int i = 0; i < 5; ++i) {
    reg_matches.emplace_back(i, i);
    reg_inliers.push_back(i);
  }
  AddImagePairWithLC(corr_graph, 1, 2, reg_matches, reg_inliers, {});
  AddImagePairWithLC(corr_graph, 1, 3, reg_matches, reg_inliers, {});
  AddImagePairWithLC(corr_graph, 2, 3, reg_matches, reg_inliers, {});
  AddImagePairWithLC(corr_graph, 3, 4, reg_matches, reg_inliers, {});

  // LC-only pair (1,4) with the cross-track match.
  AddImagePairWithLC(corr_graph, 1, 4, {{0, 1}}, {0}, {0});

  const auto kps = MakeWellSeparatedKeypoints({1, 2, 3, 4}, 5);

  TrackEstablishmentOptions opts;
  opts.min_num_views_per_track = 3;
  opts.intra_image_consistency_threshold = 10.0;
  const auto tracks = EstablishFullTracks(corr_graph, kps, opts);

  // Native pass yields 5 tracks; LC pass adds reciprocal lc_elements
  // without minting any new track.
  EXPECT_EQ(tracks.size(), 5u);

  // Locate the two native tracks the LC match's endpoints fall on.
  const auto kInvalid = std::numeric_limits<point3D_t>::max();
  point3D_t tid_a = kInvalid;
  point3D_t tid_b = kInvalid;
  for (const auto& [tid, p3d] : tracks) {
    if (TrackHasElement(p3d.track, 1, 0)) tid_a = tid;
    if (TrackHasElement(p3d.track, 4, 1)) tid_b = tid;
  }
  ASSERT_NE(tid_a, kInvalid);
  ASSERT_NE(tid_b, kInvalid);
  EXPECT_NE(tid_a, tid_b) << "Tracks must remain separate (no merge).";

  // Reciprocal lc_elements present.
  EXPECT_TRUE(TrackHasLCElement(tracks.at(tid_a).track, 4, 1));
  EXPECT_TRUE(TrackHasLCElement(tracks.at(tid_b).track, 1, 0));

  // Sanity: the LC observation is NOT a regular element of either track.
  EXPECT_FALSE(TrackHasElement(tracks.at(tid_a).track, 4, 1));
  EXPECT_FALSE(TrackHasElement(tracks.at(tid_b).track, 1, 0));
}

// Exactly one LC endpoint lies on an existing native track; the other side
// is orphan. Expect lc_element added to the existing track; no new track is
// minted.
//
// Drives EstablishTracksFromCorrGraph + AppendLoopClosureObservations.
TEST(ProcessLoopClosurePairs, OneExistingTrack) {
  CorrespondenceGraph corr_graph;
  for (image_t i = 1; i <= 3; ++i) corr_graph.AddImage(i, 5);
  corr_graph.AddImage(4, 5);  // image 4 has no regular pair -> orphan.

  std::vector<std::pair<int, int>> reg_matches;
  std::vector<int> reg_inliers;
  for (int i = 0; i < 5; ++i) {
    reg_matches.emplace_back(i, i);
    reg_inliers.push_back(i);
  }
  AddImagePairWithLC(corr_graph, 1, 2, reg_matches, reg_inliers, {});
  AddImagePairWithLC(corr_graph, 1, 3, reg_matches, reg_inliers, {});
  AddImagePairWithLC(corr_graph, 2, 3, reg_matches, reg_inliers, {});

  // LC-only pair (1, 4): img1:feat=0 (on native track) <-> img4:feat=0
  // (orphan).
  AddImagePairWithLC(corr_graph, 1, 4, {{0, 0}}, {0}, {0});

  const auto kps = MakeWellSeparatedKeypoints({1, 2, 3, 4}, 5);

  TrackEstablishmentOptions opts;
  opts.min_num_views_per_track = 3;
  opts.intra_image_consistency_threshold = 10.0;
  const auto tracks = EstablishFullTracks(corr_graph, kps, opts);

  // 5 native tracks; no new track minted (one side already on a track).
  EXPECT_EQ(tracks.size(), 5u);

  bool found_track_a = false;
  for (const auto& [tid, p3d] : tracks) {
    if (TrackHasElement(p3d.track, 1, 0)) {
      EXPECT_TRUE(TrackHasLCElement(p3d.track, 4, 0));
      found_track_a = true;
    }
    // (img4, feat=0) should never be a regular element.
    EXPECT_FALSE(TrackHasElement(p3d.track, 4, 0));
  }
  EXPECT_TRUE(found_track_a);
}

// Both LC endpoints are orphan (neither lives on a native track). Expect 2
// new tracks minted, each with 1 regular element + the other side as
// lc_element. Ids land at sequential positions past the native max.
//
// Drives EstablishTracksFromCorrGraph + AppendLoopClosureObservations.
TEST(ProcessLoopClosurePairs, NeitherExistingTrack) {
  CorrespondenceGraph corr_graph;
  for (image_t i = 1; i <= 3; ++i) corr_graph.AddImage(i, 5);
  // Images 4, 5 are orphan.
  corr_graph.AddImage(4, 5);
  corr_graph.AddImage(5, 5);

  std::vector<std::pair<int, int>> reg_matches;
  std::vector<int> reg_inliers;
  for (int i = 0; i < 5; ++i) {
    reg_matches.emplace_back(i, i);
    reg_inliers.push_back(i);
  }
  AddImagePairWithLC(corr_graph, 1, 2, reg_matches, reg_inliers, {});
  AddImagePairWithLC(corr_graph, 1, 3, reg_matches, reg_inliers, {});
  AddImagePairWithLC(corr_graph, 2, 3, reg_matches, reg_inliers, {});

  // LC-only pair (4, 5) carrying one match img4:feat=2 <-> img5:feat=3.
  AddImagePairWithLC(corr_graph, 4, 5, {{2, 3}}, {0}, {0});

  const auto kps = MakeWellSeparatedKeypoints({1, 2, 3, 4, 5}, 5);

  TrackEstablishmentOptions opts;
  opts.min_num_views_per_track = 3;
  opts.intra_image_consistency_threshold = 10.0;
  const auto tracks = EstablishFullTracks(corr_graph, kps, opts);

  // 5 native tracks + 2 minted = 7.
  EXPECT_EQ(tracks.size(), 7u);

  const auto kInvalid = std::numeric_limits<point3D_t>::max();
  point3D_t tid_for_4_2 = kInvalid;
  point3D_t tid_for_5_3 = kInvalid;
  for (const auto& [tid, p3d] : tracks) {
    if (TrackHasElement(p3d.track, 4, 2)) tid_for_4_2 = tid;
    if (TrackHasElement(p3d.track, 5, 3)) tid_for_5_3 = tid;
  }
  ASSERT_NE(tid_for_4_2, kInvalid);
  ASSERT_NE(tid_for_5_3, kInvalid);
  // Native ids are dense [0, 5); minted ids must be >= 5.
  EXPECT_GE(tid_for_4_2, 5u);
  EXPECT_GE(tid_for_5_3, 5u);
  // Sequential — they differ by exactly 1.
  const point3D_t lo = std::min(tid_for_4_2, tid_for_5_3);
  const point3D_t hi = std::max(tid_for_4_2, tid_for_5_3);
  EXPECT_EQ(hi, lo + 1);

  // Each minted track has exactly 1 regular element + 1 lc_element.
  const auto& track_a = tracks.at(tid_for_4_2).track;
  EXPECT_EQ(track_a.Length(), 1u);
  EXPECT_EQ(track_a.lc_elements.size(), 1u);
  EXPECT_TRUE(TrackHasLCElement(track_a, 5, 3));

  const auto& track_b = tracks.at(tid_for_5_3).track;
  EXPECT_EQ(track_b.Length(), 1u);
  EXPECT_EQ(track_b.lc_elements.size(), 1u);
  EXPECT_TRUE(TrackHasLCElement(track_b, 4, 2));
}

// Multiple LC matches across multiple pairs. The internal obs_to_track
// lookup must be updated as each new track is minted so subsequent LC pairs
// find them — otherwise we'd double-mint and end up with 4 minted tracks
// instead of 2.
//
// Setup: regular triangle on (1,2,3) -> 5 native tracks. Then LC pairs
// (4,5) and (5,6), each with one LC match. The two pairs share the
// observation (img5:feat=0). Whichever pair runs first mints a track for
// (img5:feat=0); the second pair must hit "OneExistingTrack" via that
// observation rather than re-mint.
//
// Drives EstablishTracksFromCorrGraph + AppendLoopClosureObservations.
TEST(ProcessLoopClosurePairs, MultipleLCMatchesAcrossPairs) {
  CorrespondenceGraph corr_graph;
  for (image_t i = 1; i <= 6; ++i) corr_graph.AddImage(i, 5);

  std::vector<std::pair<int, int>> reg_matches;
  std::vector<int> reg_inliers;
  for (int i = 0; i < 5; ++i) {
    reg_matches.emplace_back(i, i);
    reg_inliers.push_back(i);
  }
  AddImagePairWithLC(corr_graph, 1, 2, reg_matches, reg_inliers, {});
  AddImagePairWithLC(corr_graph, 1, 3, reg_matches, reg_inliers, {});
  AddImagePairWithLC(corr_graph, 2, 3, reg_matches, reg_inliers, {});

  AddImagePairWithLC(corr_graph, 4, 5, {{0, 0}}, {0}, {0});
  AddImagePairWithLC(corr_graph, 5, 6, {{0, 0}}, {0}, {0});

  const auto kps = MakeWellSeparatedKeypoints({1, 2, 3, 4, 5, 6}, 5);

  TrackEstablishmentOptions opts;
  opts.min_num_views_per_track = 3;
  opts.intra_image_consistency_threshold = 10.0;
  const auto tracks = EstablishFullTracks(corr_graph, kps, opts);

  // 5 native + (2 minted from first LC pair processed) + (0 minted from
  // second LC pair, "OneExistingTrack" branch on img5:feat=0) = 7.
  // Without the obs_to_track update, the second pair would mint two more
  // tracks (or one, depending on order) and we'd see 8 or 9.
  EXPECT_EQ(tracks.size(), 7u);

  // (img5, feat=0) appears as a regular element exactly once across the
  // minted tracks.
  int count_5_0_regular = 0;
  for (const auto& [tid, p3d] : tracks) {
    if (TrackHasElement(p3d.track, 5, 0)) ++count_5_0_regular;
  }
  EXPECT_EQ(count_5_0_regular, 1);
}

// Smoke test: native produces 10 dense tracks (ids in [0, 10)); minted ids
// must start >= 10 and never collide with a native id.
//
// Drives EstablishTracksFromCorrGraph + AppendLoopClosureObservations.
TEST(ProcessLoopClosurePairs, SequentialIdsNoCollision) {
  CorrespondenceGraph corr_graph;
  for (image_t i = 1; i <= 3; ++i) corr_graph.AddImage(i, 10);
  corr_graph.AddImage(4, 10);
  corr_graph.AddImage(5, 10);

  std::vector<std::pair<int, int>> reg_matches;
  std::vector<int> reg_inliers;
  for (int i = 0; i < 10; ++i) {
    reg_matches.emplace_back(i, i);
    reg_inliers.push_back(i);
  }
  AddImagePairWithLC(corr_graph, 1, 2, reg_matches, reg_inliers, {});
  AddImagePairWithLC(corr_graph, 1, 3, reg_matches, reg_inliers, {});
  AddImagePairWithLC(corr_graph, 2, 3, reg_matches, reg_inliers, {});
  AddImagePairWithLC(corr_graph, 4, 5, {{0, 0}}, {0}, {0});

  const auto kps = MakeWellSeparatedKeypoints({1, 2, 3, 4, 5}, 10);

  TrackEstablishmentOptions opts;
  opts.min_num_views_per_track = 3;
  opts.intra_image_consistency_threshold = 10.0;
  const auto tracks = EstablishFullTracks(corr_graph, kps, opts);

  EXPECT_EQ(tracks.size(), 12u);  // 10 native + 2 minted

  // Native ids: dense [0, 10). Verify no collision: every track that holds
  // (img4, feat=0) or (img5, feat=0) as a regular element has id >= 10.
  for (const auto& [tid, p3d] : tracks) {
    const bool minted_endpoint = TrackHasElement(p3d.track, 4, 0) ||
                                 TrackHasElement(p3d.track, 5, 0);
    if (minted_endpoint) {
      EXPECT_GE(tid, 10u) << "minted id " << tid << " collided with native";
    }
  }
}

}  // namespace
}  // namespace colmap
