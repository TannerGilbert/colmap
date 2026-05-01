#include "colmap/sfm/track_establishment.h"

#include "colmap/feature/types.h"
#include "colmap/scene/correspondence_graph.h"
#include "colmap/scene/two_view_geometry.h"
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

// Populate a correspondence graph via the proper AddTwoViewGeometry path so
// that ExtractMatchesBetweenImages can find the correspondences.
void AddImagePair(CorrespondenceGraph& corr_graph,
                  image_t image_id1,
                  image_t image_id2,
                  const std::vector<std::pair<int, int>>& matches,
                  const std::vector<int>& inliers) {
  struct TwoViewGeometry tvg;
  tvg.inlier_matches.reserve(inliers.size());
  for (const int idx : inliers) {
    tvg.inlier_matches.emplace_back(
        static_cast<point2D_t>(matches[idx].first),
        static_cast<point2D_t>(matches[idx].second));
  }
  corr_graph.AddTwoViewGeometry(image_id1, image_id2, std::move(tvg));
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

// Build an Image with N features.
Image MakeImage(image_t image_id, int num_features) {
  Image image;
  image.SetImageId(image_id);
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

// Collect image ids from a filtered-image test fixture.
std::unordered_set<image_t> MakeImageIds(
    const std::unordered_map<image_t, Image>& images) {
  std::unordered_set<image_t> ids;
  for (const auto& [image_id, image] : images) {
    ids.insert(image_id);
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
  images.emplace(1, MakeImage(1, 5));
  images.emplace(2, MakeImage(2, 5));
  images.emplace(3, MakeImage(3, 5));

  std::unordered_map<point3D_t, Point3D> tracks_full;
  for (point2D_t f = 0; f < 5; ++f) {
    tracks_full.emplace(f,
                        MakePoint3DFromElements({{1, f}, {2, f}, {3, f}}));
  }

  const auto reg_ids = MakeImageIds(images);

  // High-min variant: tracks have length 3, demand 10.
  {
    TrackSubsampleOptions options;
    options.min_num_views_per_track = 10;
    options.required_tracks_per_view = 1000;  // never saturate
    const auto selected = SubsampleTracks(
        options, reg_ids, tracks_full);
    EXPECT_EQ(selected.size(), 0u);
    EXPECT_TRUE(selected.empty());
  }

  // Low-min variant: every length-3 track survives.
  {
    TrackSubsampleOptions options;
    options.min_num_views_per_track = 2;
    options.required_tracks_per_view = 1000;
    const auto selected = SubsampleTracks(
        options, reg_ids, tracks_full);
    EXPECT_EQ(selected.size(), 5u);
  }
}

// MaxLengthFilter: tracks of length 5 dropped when max=4.
//
// Drives SubsampleTracks.
TEST(FindTracksForProblem, MaxLengthFilter) {
  std::unordered_map<image_t, Image> images;
  for (image_t i = 1; i <= 5; ++i) {
    images.emplace(i, MakeImage(i, 3));
  }

  std::unordered_map<point3D_t, Point3D> tracks_full;
  for (point2D_t f = 0; f < 3; ++f) {
    tracks_full.emplace(
        f, MakePoint3DFromElements(
               {{1, f}, {2, f}, {3, f}, {4, f}, {5, f}}));
  }

  const auto reg_ids = MakeImageIds(images);

  TrackSubsampleOptions options;
  options.min_num_views_per_track = 2;
  options.max_num_views_per_track = 4;
  options.required_tracks_per_view = 1000;
  const auto selected = SubsampleTracks(
      options, reg_ids, tracks_full);
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
    images.emplace(i, MakeImage(i, 5));
  }

  std::unordered_map<point3D_t, Point3D> tracks_full;
  for (point2D_t f = 0; f < 5; ++f) {
    tracks_full.emplace(f,
                        MakePoint3DFromElements({{1, f}, {2, f}, {3, f}}));
  }

  const auto reg_ids = MakeImageIds(images);

  TrackSubsampleOptions options;
  options.min_num_views_per_track = 2;
  options.required_tracks_per_view = 2;
  const auto selected = SubsampleTracks(
      options, reg_ids, tracks_full);
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
    images.emplace(i, MakeImage(i, 3));
  }

  std::unordered_map<point3D_t, Point3D> tracks_full;
  for (point2D_t f = 0; f < 3; ++f) {
    tracks_full.emplace(f,
                        MakePoint3DFromElements({{1, f}, {2, f}, {3, f}}));
  }

  const auto reg_ids = MakeImageIds(images);

  TrackSubsampleOptions options;
  options.min_num_views_per_track = 2;
  // options.required_tracks_per_view stays at default = INT_MAX.
  const auto selected = SubsampleTracks(
      options, reg_ids, tracks_full);
  // All 3 tracks are kept — the quota gate is disabled by INT_MAX bar.
  EXPECT_EQ(selected.size(), 3u);
}

// ============================================================================
// ProcessLoopClosurePairs (second pass over LC-marked inlier matches)
// ============================================================================
//
// Two-step: ``EstablishTracksFromCorrGraph`` builds native tracks from all
// correspondences in the CG; ``AppendLoopClosureObservations`` then reads
// ``ImagePair.are_lc`` to attach LC observations as ``Track::lc_elements``.
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
  // Register via AddTwoViewGeometry so ExtractMatchesBetweenImages works.
  AddImagePair(corr_graph, image_id1, image_id2, matches, inliers);

  // Populate extended fields on the ImagePair that AddTwoViewGeometry
  // created.
  const image_pair_t pair_id = ImagePairToPairId(image_id1, image_id2);
  auto& image_pair = corr_graph.MutableImagePairs().at(pair_id);
  Eigen::MatrixXi matches_mat(static_cast<int>(matches.size()), 2);
  for (size_t i = 0; i < matches.size(); ++i) {
    matches_mat(static_cast<int>(i), 0) = matches[i].first;
    matches_mat(static_cast<int>(i), 1) = matches[i].second;
  }
  image_pair.matches = std::move(matches_mat);
  image_pair.inliers = inliers;
  std::vector<bool> are_lc(matches.size(), false);
  for (const size_t row : lc_match_indices) are_lc[row] = true;
  image_pair.are_lc = std::move(are_lc);
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

// Populate an LC-only ImagePair directly (no AddTwoViewGeometry). Use for
// tests that call AppendLoopClosureObservations in isolation.
void AddLCOnlyPair(CorrespondenceGraph& corr_graph,
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
  image_pair.are_lc.assign(matches.size(), true);
  corr_graph.MutableImagePairs().emplace(pair_id, std::move(image_pair));
}

// Run the native + LC two-step end-to-end matching production code path:
// MakeLoopClosureMatchPredicate suppresses LC-flagged observations from
// union-find, then AppendLoopClosureObservations attaches them as
// lc_elements.
std::unordered_map<point3D_t, Point3D> EstablishFullTracks(
    const CorrespondenceGraph& corr_graph,
    const std::unordered_map<image_t, std::vector<Eigen::Vector2d>>& keypoints,
    const TrackEstablishmentOptions& options) {
  const auto pair_ids = CollectPairIds(corr_graph);
  auto ignore_match = MakeLoopClosureMatchPredicate(pair_ids, corr_graph);
  TrackEstablishmentOptions opts = options;
  opts.required_tracks_per_view = std::numeric_limits<int>::max();
  auto tracks = EstablishTracksFromCorrGraph(
      pair_ids, corr_graph, keypoints, opts, ignore_match);
  AppendLoopClosureObservations(pair_ids, corr_graph, tracks);
  return tracks;
}

// Both endpoints of the LC match already lie on existing tracks (different
// ones). Expect reciprocal lc_elements added; tracks stay separate.
//
// Calls AppendLoopClosureObservations directly with pre-built tracks.
TEST(ProcessLoopClosurePairs, BothExistingTracks) {
  CorrespondenceGraph corr_graph;
  corr_graph.AddImage(1, 5);
  corr_graph.AddImage(2, 5);

  // LC pair (1,2) with cross-track match: img1:0 <-> img2:1.
  AddLCOnlyPair(corr_graph, 1, 2, {{0, 1}}, {0});

  // Pre-build two tracks with the LC endpoints on different tracks.
  std::unordered_map<point3D_t, Point3D> tracks;
  {
    Point3D p;
    p.track.AddElement(1, 0);
    p.track.AddElement(3, 0);
    tracks.emplace(0, std::move(p));
  }
  {
    Point3D p;
    p.track.AddElement(2, 1);
    p.track.AddElement(3, 1);
    tracks.emplace(1, std::move(p));
  }

  std::vector<image_pair_t> pair_ids = {ImagePairToPairId(1, 2)};
  AppendLoopClosureObservations(pair_ids, corr_graph, tracks);

  // No new tracks minted.
  EXPECT_EQ(tracks.size(), 2u);
  // Reciprocal lc_elements.
  EXPECT_TRUE(TrackHasLCElement(tracks.at(0).track, 2, 1));
  EXPECT_TRUE(TrackHasLCElement(tracks.at(1).track, 1, 0));
  // Not regular elements.
  EXPECT_FALSE(TrackHasElement(tracks.at(0).track, 2, 1));
  EXPECT_FALSE(TrackHasElement(tracks.at(1).track, 1, 0));
}

// Exactly one LC endpoint lies on an existing track; the other is orphan.
// Expect lc_element added to the existing track; no new track minted.
//
// Calls AppendLoopClosureObservations directly with pre-built tracks.
TEST(ProcessLoopClosurePairs, OneExistingTrack) {
  CorrespondenceGraph corr_graph;
  corr_graph.AddImage(1, 5);
  corr_graph.AddImage(4, 5);

  // LC pair (1,4): img1:0 (on track) <-> img4:0 (orphan).
  AddLCOnlyPair(corr_graph, 1, 4, {{0, 0}}, {0});

  // Pre-build one track containing img1:0.
  std::unordered_map<point3D_t, Point3D> tracks;
  {
    Point3D p;
    p.track.AddElement(1, 0);
    p.track.AddElement(2, 0);
    p.track.AddElement(3, 0);
    tracks.emplace(0, std::move(p));
  }

  std::vector<image_pair_t> pair_ids = {ImagePairToPairId(1, 4)};
  AppendLoopClosureObservations(pair_ids, corr_graph, tracks);

  // No new track minted.
  EXPECT_EQ(tracks.size(), 1u);
  EXPECT_TRUE(TrackHasLCElement(tracks.at(0).track, 4, 0));
  EXPECT_FALSE(TrackHasElement(tracks.at(0).track, 4, 0));
}

// Both LC endpoints are orphan (neither on any track). Expect 2 new tracks
// minted, each with 1 regular element + the other side as lc_element.
//
// Calls AppendLoopClosureObservations directly with pre-built tracks.
TEST(ProcessLoopClosurePairs, NeitherExistingTrack) {
  CorrespondenceGraph corr_graph;
  corr_graph.AddImage(4, 5);
  corr_graph.AddImage(5, 5);

  // LC pair (4,5): img4:2 <-> img5:3. Neither endpoint on any track.
  AddLCOnlyPair(corr_graph, 4, 5, {{2, 3}}, {0});

  // Pre-build 5 native tracks on images 1-3 (not touching 4 or 5).
  std::unordered_map<point3D_t, Point3D> tracks;
  for (point3D_t tid = 0; tid < 5; ++tid) {
    Point3D p;
    p.track.AddElement(1, static_cast<point2D_t>(tid));
    p.track.AddElement(2, static_cast<point2D_t>(tid));
    p.track.AddElement(3, static_cast<point2D_t>(tid));
    tracks.emplace(tid, std::move(p));
  }

  std::vector<image_pair_t> pair_ids = {ImagePairToPairId(4, 5)};
  AppendLoopClosureObservations(pair_ids, corr_graph, tracks);

  // 5 native + 2 minted = 7.
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
  EXPECT_GE(tid_for_4_2, 5u);
  EXPECT_GE(tid_for_5_3, 5u);
  const point3D_t lo = std::min(tid_for_4_2, tid_for_5_3);
  const point3D_t hi = std::max(tid_for_4_2, tid_for_5_3);
  EXPECT_EQ(hi, lo + 1);

  EXPECT_EQ(tracks.at(tid_for_4_2).track.Length(), 1u);
  EXPECT_EQ(tracks.at(tid_for_4_2).track.lc_elements.size(), 1u);
  EXPECT_TRUE(TrackHasLCElement(tracks.at(tid_for_4_2).track, 5, 3));

  EXPECT_EQ(tracks.at(tid_for_5_3).track.Length(), 1u);
  EXPECT_EQ(tracks.at(tid_for_5_3).track.lc_elements.size(), 1u);
  EXPECT_TRUE(TrackHasLCElement(tracks.at(tid_for_5_3).track, 4, 2));
}

// Multiple LC matches across multiple pairs sharing an observation.
// obs_to_track must be updated as tracks are minted so the second pair
// hits OneExistingTrack rather than re-minting.
//
// Calls AppendLoopClosureObservations directly with pre-built tracks.
TEST(ProcessLoopClosurePairs, MultipleLCMatchesAcrossPairs) {
  CorrespondenceGraph corr_graph;
  for (image_t i = 4; i <= 6; ++i) corr_graph.AddImage(i, 5);

  // Two LC pairs sharing img5:0.
  AddLCOnlyPair(corr_graph, 4, 5, {{0, 0}}, {0});
  AddLCOnlyPair(corr_graph, 5, 6, {{0, 0}}, {0});

  // Pre-build 5 native tracks on images 1-3 (not touching 4-6).
  std::unordered_map<point3D_t, Point3D> tracks;
  for (point3D_t tid = 0; tid < 5; ++tid) {
    Point3D p;
    p.track.AddElement(1, static_cast<point2D_t>(tid));
    p.track.AddElement(2, static_cast<point2D_t>(tid));
    p.track.AddElement(3, static_cast<point2D_t>(tid));
    tracks.emplace(tid, std::move(p));
  }

  std::vector<image_pair_t> pair_ids = {
      ImagePairToPairId(4, 5), ImagePairToPairId(5, 6)};
  AppendLoopClosureObservations(pair_ids, corr_graph, tracks);

  // First pair: neither on track → mint 2 (img4:0, img5:0).
  // Second pair: img5:0 now on minted track → OneExistingTrack, no mint.
  // Total: 5 + 2 = 7.
  EXPECT_EQ(tracks.size(), 7u);

  int count_5_0 = 0;
  for (const auto& [tid, p3d] : tracks) {
    if (TrackHasElement(p3d.track, 5, 0)) ++count_5_0;
  }
  EXPECT_EQ(count_5_0, 1);
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

// Both LC endpoints fall on the SAME native track. This is a degenerate
// case (self-loop) — AppendLoopClosureObservations should skip it silently
// rather than adding the observation as an lc_element of itself.
//
// Setup: regular triangle (1,2,3) → 5 native tracks. LC pair (1,2) with
// match (img1:feat=0 <-> img2:feat=0) — both endpoints already on the
// same track (the feat-0 chain includes both img1 and img2).
//
// Drives EstablishTracksFromCorrGraph + AppendLoopClosureObservations.
// Both LC endpoints already live on the SAME native track. The code skips
// this case (``t1 == t2``) — no lc_elements should be added. Tested by
// calling AppendLoopClosureObservations directly with a pre-built tracks
// dict, bypassing MakeLoopClosureMatchPredicate (which would suppress the
// observations from union-find).
TEST(ProcessLoopClosurePairs, BothSameTrack) {
  CorrespondenceGraph corr_graph;
  corr_graph.AddImage(1, 5);
  corr_graph.AddImage(2, 5);

  // Populate an LC-only pair directly (no AddTwoViewGeometry needed since
  // we call AppendLoopClosureObservations directly, not EstablishFullTracks).
  const image_pair_t pair_id = ImagePairToPairId(1, 2);
  CorrespondenceGraph::ImagePair image_pair(1, 2);
  Eigen::MatrixXi matches_mat(1, 2);
  matches_mat(0, 0) = 0;
  matches_mat(0, 1) = 0;
  image_pair.matches = std::move(matches_mat);
  image_pair.inliers = {0};
  image_pair.are_lc = {true};
  corr_graph.MutableImagePairs().emplace(pair_id, std::move(image_pair));

  // Pre-build a single track containing both LC endpoints.
  std::unordered_map<point3D_t, Point3D> tracks;
  Point3D p;
  p.track.AddElement(1, 0);
  p.track.AddElement(2, 0);
  tracks.emplace(0, std::move(p));

  std::vector<image_pair_t> pair_ids = {pair_id};
  AppendLoopClosureObservations(pair_ids, corr_graph, tracks);

  // No new tracks minted, no lc_elements added (same-track skip).
  EXPECT_EQ(tracks.size(), 1u);
  EXPECT_EQ(tracks.at(0).track.lc_elements.size(), 0u);
}

// ============================================================================
// SubsampleTracks: max_num_tracks limit
// ============================================================================

// Verify that ``max_num_tracks`` stops the greedy subsample early.
TEST(FindTracksForProblem, MaxNumTracksLimit) {
  std::unordered_map<image_t, Image> images;
  for (image_t i = 1; i <= 3; ++i) {
    images.emplace(i, MakeImage(i, 10));
  }

  std::unordered_map<point3D_t, Point3D> tracks_full;
  for (point2D_t f = 0; f < 10; ++f) {
    tracks_full.emplace(f,
                        MakePoint3DFromElements({{1, f}, {2, f}, {3, f}}));
  }

  const auto reg_ids = MakeImageIds(images);

  TrackSubsampleOptions options;
  options.min_num_views_per_track = 2;
  options.required_tracks_per_view = 1000;
  options.max_num_tracks = 3;
  const auto selected = SubsampleTracks(options, reg_ids, tracks_full);
  // The break fires AFTER inserting when size > max_num_tracks, so we
  // may get max_num_tracks + 1. The key assertion: not all 10 are kept.
  EXPECT_LE(static_cast<int>(selected.size()), options.max_num_tracks + 1);
  EXPECT_LT(selected.size(), tracks_full.size());
}

}  // namespace
}  // namespace colmap
