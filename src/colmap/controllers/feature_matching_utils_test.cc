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

#include "colmap/controllers/feature_matching_utils.h"

#include "colmap/controllers/matcher_cache.h"
#include "colmap/scene/synthetic.h"
#include "colmap/util/testing.h"

#include <algorithm>

#include <gtest/gtest.h>

namespace colmap {
namespace {

struct TestData {
  std::filesystem::path test_dir;
  std::shared_ptr<Database> database;
  std::shared_ptr<FeatureMatcherCache> cache;
  std::vector<image_t> image_ids;
};

TestData CreateTestData(int num_images) {
  TestData data;
  data.test_dir = CreateTestDir();
  const auto database_path = data.test_dir / "database.db";
  data.database = Database::Open(database_path);

  Reconstruction reconstruction;
  SyntheticDatasetOptions options;
  options.num_rigs = num_images;
  options.num_cameras_per_rig = 1;
  options.num_frames_per_rig = 1;
  options.num_points3D = 20;
  options.num_points2D_without_point3D = 3;
  SynthesizeDataset(options, &reconstruction, data.database.get());

  data.cache = std::make_shared<FeatureMatcherCache>(100, data.database);
  data.image_ids = data.cache->GetImageIds();
  return data;
}

FeatureMatchingOptions DefaultMatchingOptions() {
  FeatureMatchingOptions options;
  options.use_gpu = false;
  options.num_threads = 1;
  return options;
}

std::vector<std::pair<image_t, image_t>> AllPairs(
    const std::vector<image_t>& image_ids) {
  std::vector<std::pair<image_t, image_t>> pairs;
  for (size_t i = 0; i < image_ids.size(); ++i) {
    for (size_t j = i + 1; j < image_ids.size(); ++j) {
      pairs.emplace_back(image_ids[i], image_ids[j]);
    }
  }
  return pairs;
}

std::pair<image_t, image_t> NonConsecutiveImagePair(Database& database) {
  std::vector<Image> images = database.ReadAllImages();
  std::sort(images.begin(), images.end(), [](const Image& a, const Image& b) {
    return a.Name() < b.Name();
  });
  THROW_CHECK_GE(images.size(), 3);
  return {images[0].ImageId(), images[2].ImageId()};
}

std::pair<image_t, image_t> ConsecutiveImagePair(Database& database) {
  std::vector<Image> images = database.ReadAllImages();
  std::sort(images.begin(), images.end(), [](const Image& a, const Image& b) {
    return a.Name() < b.Name();
  });
  THROW_CHECK_GE(images.size(), 2);
  return {images[0].ImageId(), images[1].ImageId()};
}

// Match pairs without geometric verification, then clear TVGs.
// Leaves matches in the database ready for a GeometricVerifierController.
void MatchPairsWithoutVerification(
    TestData& data, const std::vector<std::pair<image_t, image_t>>& pairs) {
  data.database->ClearMatches();
  data.database->ClearTwoViewGeometries();
  FeatureMatchingOptions matching_options = DefaultMatchingOptions();
  matching_options.skip_geometric_verification = true;
  TwoViewGeometryOptions geometry_options;
  FeatureMatcherController matcher(
      matching_options, geometry_options, data.cache);
  ASSERT_TRUE(matcher.Setup());
  matcher.Match(pairs);
  data.database->ClearTwoViewGeometries();
}

TEST(MergeLoopClosureInlierMatches,
     MatchesVideoSfMEndpointUniquenessAndOrdering) {
  const FeatureMatches transitive_matches = {
      {1, 10}, {2, 20}, {2, 21}, {3, 20}};
  const FeatureMatches candidate_matches = {
      {1, 99}, {4, 20}, {5, 50}, {6, 60}, {5, 70}};

  FeatureMatches merged_matches;
  std::vector<bool> merged_matches_are_lc;
  MergeLoopClosureInlierMatches(transitive_matches,
                                candidate_matches,
                                &merged_matches,
                                &merged_matches_are_lc);

  ASSERT_EQ(merged_matches.size(), 7);
  EXPECT_EQ(merged_matches[0], FeatureMatch(1, 10));
  EXPECT_EQ(merged_matches[1], FeatureMatch(2, 20));
  EXPECT_EQ(merged_matches[2], FeatureMatch(2, 21));
  EXPECT_EQ(merged_matches[3], FeatureMatch(3, 20));
  EXPECT_EQ(merged_matches[4], FeatureMatch(5, 50));
  EXPECT_EQ(merged_matches[5], FeatureMatch(6, 60));
  EXPECT_EQ(merged_matches[6], FeatureMatch(5, 70));
  EXPECT_EQ(merged_matches_are_lc,
            std::vector<bool>({false, false, false, false, true, true, true}));
}

TEST(MergeLoopClosureInlierMatches, CollidingCandidatesDoNotBecomeLcRows) {
  const FeatureMatches transitive_matches = {{1, 10}, {2, 20}};
  const FeatureMatches candidate_matches = {{1, 30}, {3, 20}};

  FeatureMatches merged_matches;
  std::vector<bool> merged_matches_are_lc;
  MergeLoopClosureInlierMatches(transitive_matches,
                                candidate_matches,
                                &merged_matches,
                                &merged_matches_are_lc);

  ASSERT_EQ(merged_matches.size(), 2);
  EXPECT_EQ(merged_matches[0], FeatureMatch(1, 10));
  EXPECT_EQ(merged_matches[1], FeatureMatch(2, 20));
  EXPECT_EQ(merged_matches_are_lc, std::vector<bool>({false, false}));
}

TEST(FeatureMatcherController, MatchEmptyPairs) {
  auto data = CreateTestData(3);
  data.database->ClearMatches();
  data.database->ClearTwoViewGeometries();

  FeatureMatchingOptions matching_options = DefaultMatchingOptions();
  TwoViewGeometryOptions geometry_options;

  FeatureMatcherController controller(
      matching_options, geometry_options, data.cache);
  ASSERT_TRUE(controller.Setup());

  // Matching empty pairs should return without error
  controller.Match({});

  EXPECT_EQ(data.database->ReadAllMatches().size(), 0);
}

TEST(FeatureMatcherController, MatchSkipsSelfMatches) {
  auto data = CreateTestData(3);
  data.database->ClearMatches();
  data.database->ClearTwoViewGeometries();

  FeatureMatchingOptions matching_options = DefaultMatchingOptions();
  TwoViewGeometryOptions geometry_options;

  FeatureMatcherController controller(
      matching_options, geometry_options, data.cache);
  ASSERT_TRUE(controller.Setup());

  // Self-match pairs should be skipped
  std::vector<std::pair<image_t, image_t>> pairs;
  pairs.reserve(data.image_ids.size());
  for (const auto id : data.image_ids) {
    pairs.emplace_back(id, id);
  }
  controller.Match(pairs);

  EXPECT_EQ(data.database->ReadAllMatches().size(), 0);
}

TEST(FeatureMatcherController, MatchSkipsDuplicatePairs) {
  auto data = CreateTestData(3);
  data.database->ClearMatches();
  data.database->ClearTwoViewGeometries();

  FeatureMatchingOptions matching_options = DefaultMatchingOptions();
  TwoViewGeometryOptions geometry_options;

  FeatureMatcherController controller(
      matching_options, geometry_options, data.cache);
  ASSERT_TRUE(controller.Setup());

  ASSERT_GE(data.image_ids.size(), 3);
  const image_t id1 = data.image_ids[0];
  const image_t id2 = data.image_ids[1];

  // Submit same pair multiple times — should only process once
  controller.Match({{id1, id2}, {id1, id2}, {id1, id2}});

  const auto matches = data.database->ReadAllMatches();
  EXPECT_EQ(matches.size(), 1);
}

TEST(FeatureMatcherController, MatchSkipsExistingResults) {
  auto data = CreateTestData(3);

  FeatureMatchingOptions matching_options = DefaultMatchingOptions();
  TwoViewGeometryOptions geometry_options;

  FeatureMatcherController controller(
      matching_options, geometry_options, data.cache);
  ASSERT_TRUE(controller.Setup());

  ASSERT_GE(data.image_ids.size(), 3);
  const image_t id1 = data.image_ids[0];
  const image_t id2 = data.image_ids[1];

  // Clear and match once
  data.database->ClearMatches();
  data.database->ClearTwoViewGeometries();
  controller.Match({{id1, id2}});

  const auto matches_before = data.database->ReadAllMatches();
  const auto tvg_before = data.database->ReadTwoViewGeometries();
  EXPECT_EQ(matches_before.size(), 1);
  EXPECT_EQ(tvg_before.size(), 1);

  // Match same pair again — should skip since both matches and TVG exist
  controller.Match({{id1, id2}});

  const auto matches_after = data.database->ReadAllMatches();
  const auto tvg_after = data.database->ReadTwoViewGeometries();
  EXPECT_EQ(matches_after.size(), matches_before.size());
  EXPECT_EQ(tvg_after.size(), tvg_before.size());

  // Match with reversed pair — should also be skipped
  controller.Match({{id2, id1}});

  const auto matches_reversed = data.database->ReadAllMatches();
  const auto tvg_reversed = data.database->ReadTwoViewGeometries();
  EXPECT_EQ(matches_reversed.size(), matches_before.size());
  EXPECT_EQ(tvg_reversed.size(), tvg_before.size());
}

TEST(FeatureMatcherController, MatchMultiplePairs) {
  auto data = CreateTestData(4);
  data.database->ClearMatches();
  data.database->ClearTwoViewGeometries();

  FeatureMatchingOptions matching_options = DefaultMatchingOptions();
  TwoViewGeometryOptions geometry_options;

  FeatureMatcherController controller(
      matching_options, geometry_options, data.cache);
  ASSERT_TRUE(controller.Setup());

  // Match all pairs
  const auto pairs = AllPairs(data.image_ids);
  controller.Match(pairs);

  // 4 choose 2 = 6 pairs
  EXPECT_EQ(data.database->ReadAllMatches().size(), 6);
  EXPECT_EQ(data.database->ReadTwoViewGeometries().size(), 6);
}

TEST(FeatureMatcherController, MatchSkipGeometricVerification) {
  auto data = CreateTestData(3);
  data.database->ClearMatches();
  data.database->ClearTwoViewGeometries();

  FeatureMatchingOptions matching_options = DefaultMatchingOptions();
  matching_options.skip_geometric_verification = true;
  TwoViewGeometryOptions geometry_options;

  FeatureMatcherController controller(
      matching_options, geometry_options, data.cache);
  ASSERT_TRUE(controller.Setup());

  ASSERT_GE(data.image_ids.size(), 2);
  controller.Match({{data.image_ids[0], data.image_ids[1]}});

  // Matches should be written even without geometric verification
  EXPECT_EQ(data.database->ReadAllMatches().size(), 1);

  // Verify geometric verification was skipped: TVG should have UNDEFINED config
  const auto tvg =
      data.database->ReadTwoViewGeometry(data.image_ids[0], data.image_ids[1]);
  EXPECT_EQ(tvg.config, TwoViewGeometry::UNDEFINED);
  EXPECT_TRUE(tvg.inlier_matches.empty());
}

TEST(FeatureMatcherController, MatchMarksLoopClosure) {
  auto data = CreateTestData(3);
  const auto [image_id1, image_id2] = NonConsecutiveImagePair(*data.database);
  data.database->ClearMatches();
  data.database->ClearTwoViewGeometries();

  FeatureMatchingOptions matching_options = DefaultMatchingOptions();
  TwoViewGeometryOptions geometry_options;

  FeatureMatcherController controller(
      matching_options, geometry_options, data.cache);
  ASSERT_TRUE(controller.Setup());

  controller.Match({{image_id1, image_id2}}, /*mark_as_loop_closure=*/true);

  const auto tvg = data.database->ReadTwoViewGeometry(image_id1, image_id2);
  EXPECT_TRUE(tvg.is_loop_closure);
}

TEST(FeatureMatcherController, MatchDoesNotMarkDirectPairAsLoopClosure) {
  auto data = CreateTestData(3);
  const auto [image_id1, image_id2] = ConsecutiveImagePair(*data.database);
  data.database->ClearMatches();
  data.database->ClearTwoViewGeometries();

  FeatureMatchingOptions matching_options = DefaultMatchingOptions();
  TwoViewGeometryOptions geometry_options;

  FeatureMatcherController controller(
      matching_options, geometry_options, data.cache);
  ASSERT_TRUE(controller.Setup());

  controller.Match({{image_id1, image_id2}}, /*mark_as_loop_closure=*/true);

  const auto tvg = data.database->ReadTwoViewGeometry(image_id1, image_id2);
  EXPECT_FALSE(tvg.is_loop_closure);
  EXPECT_TRUE(tvg.inlier_matches_are_lc.empty());
}

TEST(FeatureMatcherController, MatchDefaultDoesNotMarkLoopClosure) {
  auto data = CreateTestData(3);
  data.database->ClearMatches();
  data.database->ClearTwoViewGeometries();

  FeatureMatchingOptions matching_options = DefaultMatchingOptions();
  TwoViewGeometryOptions geometry_options;

  FeatureMatcherController controller(
      matching_options, geometry_options, data.cache);
  ASSERT_TRUE(controller.Setup());

  ASSERT_GE(data.image_ids.size(), 2);
  controller.Match({{data.image_ids[0], data.image_ids[1]}});

  const auto tvg =
      data.database->ReadTwoViewGeometry(data.image_ids[0], data.image_ids[1]);
  EXPECT_FALSE(tvg.is_loop_closure);
  EXPECT_TRUE(tvg.inlier_matches_are_lc.empty());
}

TEST(FeatureMatcherController,
     MatchLoopClosureBelowMinInliersDoesNotPersistLc) {
  auto data = CreateTestData(3);
  data.database->ClearMatches();
  data.database->ClearTwoViewGeometries();

  FeatureMatchingOptions matching_options = DefaultMatchingOptions();
  TwoViewGeometryOptions geometry_options;
  geometry_options.min_num_inliers = 1000000;

  FeatureMatcherController controller(
      matching_options, geometry_options, data.cache);
  ASSERT_TRUE(controller.Setup());

  ASSERT_GE(data.image_ids.size(), 2);
  controller.Match({{data.image_ids[0], data.image_ids[1]}},
                   /*mark_as_loop_closure=*/true);

  const auto tvg =
      data.database->ReadTwoViewGeometry(data.image_ids[0], data.image_ids[1]);
  EXPECT_TRUE(tvg.inlier_matches.empty());
  EXPECT_TRUE(tvg.inlier_matches_are_lc.empty());
  EXPECT_FALSE(tvg.is_loop_closure);
}

TEST(FeatureMatcherController, MatchWithProvenanceUpgradesExistingNonLcPair) {
  auto data = CreateTestData(3);
  data.database->ClearMatches();
  data.database->ClearTwoViewGeometries();

  FeatureMatchingOptions matching_options = DefaultMatchingOptions();
  TwoViewGeometryOptions geometry_options;

  FeatureMatcherController controller(
      matching_options, geometry_options, data.cache);
  ASSERT_TRUE(controller.Setup());

  const auto [image_id1, image_id2] = NonConsecutiveImagePair(*data.database);
  controller.Match({{image_id1, image_id2}});
  ASSERT_FALSE(
      data.database->ReadTwoViewGeometry(image_id1, image_id2).is_loop_closure);

  controller.MatchWithProvenance({{image_id1, image_id2, true}});

  const auto tvg = data.database->ReadTwoViewGeometry(image_id1, image_id2);
  EXPECT_TRUE(tvg.is_loop_closure);
  EXPECT_EQ(tvg.inlier_matches.size(), tvg.inlier_matches_are_lc.size());
  EXPECT_TRUE(std::all_of(tvg.inlier_matches_are_lc.begin(),
                          tvg.inlier_matches_are_lc.end(),
                          [](const bool value) { return value; }));
}

TEST(FeatureMatcherController, MatchWithProvenanceDowngradesStaleLcPair) {
  auto data = CreateTestData(3);
  data.database->ClearMatches();
  data.database->ClearTwoViewGeometries();

  FeatureMatchingOptions matching_options = DefaultMatchingOptions();
  TwoViewGeometryOptions geometry_options;

  FeatureMatcherController controller(
      matching_options, geometry_options, data.cache);
  ASSERT_TRUE(controller.Setup());

  const auto [image_id1, image_id2] = NonConsecutiveImagePair(*data.database);
  controller.MatchWithProvenance({{image_id1, image_id2, true}});
  ASSERT_TRUE(
      data.database->ReadTwoViewGeometry(image_id1, image_id2).is_loop_closure);

  controller.MatchWithProvenance({{image_id1, image_id2, false}});

  const auto tvg = data.database->ReadTwoViewGeometry(image_id1, image_id2);
  EXPECT_FALSE(tvg.is_loop_closure);
  EXPECT_TRUE(tvg.inlier_matches_are_lc.empty());
}

TEST(FeatureMatcherController,
     MatchWithProvenanceDoesNotUpgradeDirectTrackingPair) {
  auto data = CreateTestData(3);
  data.database->ClearMatches();
  data.database->ClearTwoViewGeometries();

  FeatureMatchingOptions matching_options = DefaultMatchingOptions();
  TwoViewGeometryOptions geometry_options;

  FeatureMatcherController controller(
      matching_options, geometry_options, data.cache);
  ASSERT_TRUE(controller.Setup());

  ASSERT_GE(data.image_ids.size(), 2);
  const image_t image_id1 = data.image_ids[0];
  const image_t image_id2 = data.image_ids[1];
  controller.Match({{image_id1, image_id2}});
  const TwoViewGeometry before =
      data.database->ReadTwoViewGeometry(image_id1, image_id2);
  ASSERT_FALSE(before.inlier_matches.empty());
  ASSERT_FALSE(before.is_loop_closure);
  ASSERT_TRUE(before.inlier_matches_are_lc.empty());

  controller.MatchWithProvenance({{image_id1, image_id2, true}});

  const TwoViewGeometry after =
      data.database->ReadTwoViewGeometry(image_id1, image_id2);
  EXPECT_EQ(after.inlier_matches, before.inlier_matches);
  EXPECT_TRUE(after.inlier_matches_are_lc.empty());
  EXPECT_FALSE(after.is_loop_closure);
}

TEST(FeatureMatcherController, MatchWithProvenanceMarksMixedLoopClosures) {
  auto data = CreateTestData(3);
  data.database->ClearMatches();
  data.database->ClearTwoViewGeometries();

  FeatureMatchingOptions matching_options = DefaultMatchingOptions();
  TwoViewGeometryOptions geometry_options;

  FeatureMatcherController controller(
      matching_options, geometry_options, data.cache);
  ASSERT_TRUE(controller.Setup());

  ASSERT_GE(data.image_ids.size(), 3);
  controller.MatchWithProvenance({
      {data.image_ids[0], data.image_ids[1], false},
      {data.image_ids[0], data.image_ids[2], true},
  });

  EXPECT_FALSE(
      data.database->ReadTwoViewGeometry(data.image_ids[0], data.image_ids[1])
          .is_loop_closure);
  EXPECT_TRUE(
      data.database->ReadTwoViewGeometry(data.image_ids[0], data.image_ids[2])
          .is_loop_closure);
}

TEST(FeatureMatcherController,
     MatchWithProvenanceMergesTransitiveAndLcInliers) {
  auto data = CreateTestData(3);
  data.database->ClearMatches();
  data.database->ClearTwoViewGeometries();

  FeatureMatchingOptions matching_options = DefaultMatchingOptions();
  TwoViewGeometryOptions geometry_options;
  geometry_options.min_num_inliers = 1;

  FeatureMatcherController controller(
      matching_options, geometry_options, data.cache);
  ASSERT_TRUE(controller.Setup());

  ASSERT_GE(data.image_ids.size(), 3);
  const image_t image_id1 = data.image_ids[0];
  const image_t image_id2 = data.image_ids[1];
  const image_t image_id3 = data.image_ids[2];

  controller.Match(
      {{image_id1, image_id2}, {image_id2, image_id3}, {image_id1, image_id3}});

  TwoViewGeometry tvg12 =
      data.database->ReadTwoViewGeometry(image_id1, image_id2);
  TwoViewGeometry tvg23 =
      data.database->ReadTwoViewGeometry(image_id2, image_id3);
  ASSERT_FALSE(tvg12.inlier_matches.empty());
  ASSERT_FALSE(tvg23.inlier_matches.empty());

  FeatureMatch chain_match12;
  FeatureMatch chain_match23;
  bool found_chain = false;
  for (const FeatureMatch& match12 : tvg12.inlier_matches) {
    for (const FeatureMatch& match23 : tvg23.inlier_matches) {
      if (match12.point2D_idx2 == match23.point2D_idx1) {
        chain_match12 = match12;
        chain_match23 = match23;
        found_chain = true;
        break;
      }
    }
    if (found_chain) {
      break;
    }
  }
  ASSERT_TRUE(found_chain);

  data.database->ClearTwoViewGeometries();
  tvg12.inlier_matches = {chain_match12};
  tvg12.inlier_matches_are_lc.clear();
  tvg12.is_loop_closure = false;
  data.cache->WriteTwoViewGeometry(image_id1, image_id2, tvg12);

  tvg23.inlier_matches = {chain_match23};
  tvg23.inlier_matches_are_lc.clear();
  tvg23.is_loop_closure = false;
  data.cache->WriteTwoViewGeometry(image_id2, image_id3, tvg23);

  controller.MatchWithProvenance({{image_id1, image_id3, true}});

  const auto tvg13 = data.database->ReadTwoViewGeometry(image_id1, image_id3);
  ASSERT_EQ(tvg13.inlier_matches.size(), tvg13.inlier_matches_are_lc.size());
  ASSERT_FALSE(tvg13.inlier_matches_are_lc.empty());
  EXPECT_EQ(
      tvg13.inlier_matches[0],
      FeatureMatch(chain_match12.point2D_idx1, chain_match23.point2D_idx2));
  EXPECT_FALSE(tvg13.inlier_matches_are_lc[0]);
  bool seen_lc = false;
  for (const bool is_lc : tvg13.inlier_matches_are_lc) {
    if (is_lc) {
      seen_lc = true;
    } else {
      EXPECT_FALSE(seen_lc);
    }
  }
  EXPECT_TRUE(std::find(tvg13.inlier_matches_are_lc.begin(),
                        tvg13.inlier_matches_are_lc.end(),
                        true) != tvg13.inlier_matches_are_lc.end());
  EXPECT_TRUE(tvg13.is_loop_closure);
}

TEST(GeometricVerifierController, OptionsAccessor) {
  auto data = CreateTestData(3);

  GeometricVerifierOptions verifier_options;
  verifier_options.num_threads = 1;
  TwoViewGeometryOptions geometry_options;

  GeometricVerifierController controller(
      verifier_options, geometry_options, data.cache);

  EXPECT_EQ(controller.Options().num_threads, 1);
  controller.Options().num_threads = 2;
  EXPECT_EQ(controller.Options().num_threads, 2);
}

TEST(GeometricVerifierController, VerifyEmptyPairs) {
  auto data = CreateTestData(3);
  data.database->ClearTwoViewGeometries();

  GeometricVerifierOptions verifier_options;
  verifier_options.num_threads = 1;
  TwoViewGeometryOptions geometry_options;

  GeometricVerifierController controller(
      verifier_options, geometry_options, data.cache);
  ASSERT_TRUE(controller.Setup());

  // Verifying empty pairs should return without error
  controller.Verify({});

  EXPECT_EQ(data.database->ReadTwoViewGeometries().size(), 0);
}

TEST(GeometricVerifierController, VerifySkipsSelfMatches) {
  auto data = CreateTestData(3);
  data.database->ClearTwoViewGeometries();

  GeometricVerifierOptions verifier_options;
  verifier_options.num_threads = 1;
  TwoViewGeometryOptions geometry_options;

  GeometricVerifierController controller(
      verifier_options, geometry_options, data.cache);
  ASSERT_TRUE(controller.Setup());

  std::vector<std::pair<image_t, image_t>> pairs;
  pairs.reserve(data.image_ids.size());
  for (const auto id : data.image_ids) {
    pairs.emplace_back(id, id);
  }
  controller.Verify(pairs);

  EXPECT_EQ(data.database->ReadTwoViewGeometries().size(), 0);
}

TEST(GeometricVerifierController, VerifySkipsDuplicatePairs) {
  auto data = CreateTestData(3);
  ASSERT_GE(data.image_ids.size(), 2);
  MatchPairsWithoutVerification(data, {{data.image_ids[0], data.image_ids[1]}});

  GeometricVerifierOptions verifier_options;
  verifier_options.num_threads = 1;
  TwoViewGeometryOptions geometry_options;

  GeometricVerifierController controller(
      verifier_options, geometry_options, data.cache);
  ASSERT_TRUE(controller.Setup());

  const image_t id1 = data.image_ids[0];
  const image_t id2 = data.image_ids[1];

  // Submit same pair multiple times — should only process once
  controller.Verify({{id1, id2}, {id1, id2}, {id1, id2}});

  const auto tvgs = data.database->ReadTwoViewGeometries();
  EXPECT_EQ(tvgs.size(), 1);
}

TEST(GeometricVerifierController, VerifyWithExistingMatches) {
  auto data = CreateTestData(4);
  const auto pairs = AllPairs(data.image_ids);
  MatchPairsWithoutVerification(data, pairs);

  GeometricVerifierOptions verifier_options;
  verifier_options.num_threads = 1;
  TwoViewGeometryOptions geometry_options;

  GeometricVerifierController controller(
      verifier_options, geometry_options, data.cache);
  ASSERT_TRUE(controller.Setup());

  controller.Verify(pairs);

  // All 6 pairs should now have TVGs
  EXPECT_EQ(data.database->ReadTwoViewGeometries().size(), 6);
}

TEST(GeometricVerifierController, VerifyMarksLoopClosure) {
  auto data = CreateTestData(3);
  const auto [image_id1, image_id2] = NonConsecutiveImagePair(*data.database);
  MatchPairsWithoutVerification(data, {{image_id1, image_id2}});

  GeometricVerifierOptions verifier_options;
  verifier_options.num_threads = 1;
  TwoViewGeometryOptions geometry_options;

  GeometricVerifierController controller(
      verifier_options, geometry_options, data.cache);
  ASSERT_TRUE(controller.Setup());

  controller.Verify({{image_id1, image_id2}}, /*mark_as_loop_closure=*/true);

  const auto tvg = data.database->ReadTwoViewGeometry(image_id1, image_id2);
  EXPECT_TRUE(tvg.is_loop_closure);
}

TEST(GeometricVerifierController, VerifyDoesNotMarkDirectPairAsLoopClosure) {
  auto data = CreateTestData(3);
  const auto [image_id1, image_id2] = ConsecutiveImagePair(*data.database);
  MatchPairsWithoutVerification(data, {{image_id1, image_id2}});

  GeometricVerifierOptions verifier_options;
  verifier_options.num_threads = 1;
  TwoViewGeometryOptions geometry_options;

  GeometricVerifierController controller(
      verifier_options, geometry_options, data.cache);
  ASSERT_TRUE(controller.Setup());

  controller.Verify({{image_id1, image_id2}}, /*mark_as_loop_closure=*/true);

  const auto tvg = data.database->ReadTwoViewGeometry(image_id1, image_id2);
  EXPECT_FALSE(tvg.is_loop_closure);
  EXPECT_TRUE(tvg.inlier_matches_are_lc.empty());
}

TEST(GeometricVerifierController, VerifyDefaultDoesNotMarkLoopClosure) {
  auto data = CreateTestData(3);
  ASSERT_GE(data.image_ids.size(), 2);
  MatchPairsWithoutVerification(data, {{data.image_ids[0], data.image_ids[1]}});

  GeometricVerifierOptions verifier_options;
  verifier_options.num_threads = 1;
  TwoViewGeometryOptions geometry_options;

  GeometricVerifierController controller(
      verifier_options, geometry_options, data.cache);
  ASSERT_TRUE(controller.Setup());

  controller.Verify({{data.image_ids[0], data.image_ids[1]}});

  const auto tvg =
      data.database->ReadTwoViewGeometry(data.image_ids[0], data.image_ids[1]);
  EXPECT_FALSE(tvg.is_loop_closure);
  EXPECT_TRUE(tvg.inlier_matches_are_lc.empty());
}

TEST(GeometricVerifierController,
     VerifyLoopClosureBelowMinInliersDoesNotPersistLc) {
  auto data = CreateTestData(3);
  ASSERT_GE(data.image_ids.size(), 2);
  MatchPairsWithoutVerification(data, {{data.image_ids[0], data.image_ids[1]}});

  GeometricVerifierOptions verifier_options;
  verifier_options.num_threads = 1;
  TwoViewGeometryOptions geometry_options;
  geometry_options.min_num_inliers = 1000000;

  GeometricVerifierController controller(
      verifier_options, geometry_options, data.cache);
  ASSERT_TRUE(controller.Setup());

  controller.Verify({{data.image_ids[0], data.image_ids[1]}},
                    /*mark_as_loop_closure=*/true);

  const auto tvg =
      data.database->ReadTwoViewGeometry(data.image_ids[0], data.image_ids[1]);
  EXPECT_TRUE(tvg.inlier_matches.empty());
  EXPECT_TRUE(tvg.inlier_matches_are_lc.empty());
  EXPECT_FALSE(tvg.is_loop_closure);
}

TEST(GeometricVerifierController,
     VerifyWithProvenanceUpgradesExistingNonLcPair) {
  auto data = CreateTestData(3);
  const auto [image_id1, image_id2] = NonConsecutiveImagePair(*data.database);
  MatchPairsWithoutVerification(data, {{image_id1, image_id2}});

  GeometricVerifierOptions verifier_options;
  verifier_options.num_threads = 1;
  TwoViewGeometryOptions geometry_options;

  GeometricVerifierController controller(
      verifier_options, geometry_options, data.cache);
  ASSERT_TRUE(controller.Setup());

  controller.Verify({{image_id1, image_id2}});
  ASSERT_FALSE(
      data.database->ReadTwoViewGeometry(image_id1, image_id2).is_loop_closure);

  controller.VerifyWithProvenance({{image_id1, image_id2, true}});

  const auto tvg = data.database->ReadTwoViewGeometry(image_id1, image_id2);
  EXPECT_TRUE(tvg.is_loop_closure);
  EXPECT_EQ(tvg.inlier_matches.size(), tvg.inlier_matches_are_lc.size());
  EXPECT_TRUE(std::all_of(tvg.inlier_matches_are_lc.begin(),
                          tvg.inlier_matches_are_lc.end(),
                          [](const bool value) { return value; }));
}

TEST(GeometricVerifierController,
     VerifyPreservesLegacyPairLevelLoopClosureProvenance) {
  auto data = CreateTestData(3);
  const auto [image_id1, image_id2] = NonConsecutiveImagePair(*data.database);
  MatchPairsWithoutVerification(data, {{image_id1, image_id2}});

  TwoViewGeometry prior =
      data.database->ReadTwoViewGeometry(image_id1, image_id2);
  prior.is_loop_closure = true;
  prior.inlier_matches_are_lc.clear();
  data.database->UpdateTwoViewGeometry(image_id1, image_id2, prior);

  GeometricVerifierOptions verifier_options;
  verifier_options.num_threads = 1;
  TwoViewGeometryOptions geometry_options;

  GeometricVerifierController controller(
      verifier_options, geometry_options, data.cache);
  ASSERT_TRUE(controller.Setup());

  controller.VerifyWithProvenance({{image_id1, image_id2, true}});

  const auto tvg = data.database->ReadTwoViewGeometry(image_id1, image_id2);
  ASSERT_EQ(tvg.inlier_matches.size(), tvg.inlier_matches_are_lc.size());
  EXPECT_TRUE(tvg.is_loop_closure);
  EXPECT_TRUE(std::all_of(tvg.inlier_matches_are_lc.begin(),
                          tvg.inlier_matches_are_lc.end(),
                          [](const bool value) { return value; }));
}

TEST(GeometricVerifierController, VerifyWithProvenanceMarksMixedLoopClosures) {
  auto data = CreateTestData(3);
  ASSERT_GE(data.image_ids.size(), 3);
  MatchPairsWithoutVerification(data,
                                {{data.image_ids[0], data.image_ids[1]},
                                 {data.image_ids[0], data.image_ids[2]}});

  GeometricVerifierOptions verifier_options;
  verifier_options.num_threads = 1;
  TwoViewGeometryOptions geometry_options;

  GeometricVerifierController controller(
      verifier_options, geometry_options, data.cache);
  ASSERT_TRUE(controller.Setup());

  controller.VerifyWithProvenance({
      {data.image_ids[0], data.image_ids[1], false},
      {data.image_ids[0], data.image_ids[2], true},
  });

  EXPECT_FALSE(
      data.database->ReadTwoViewGeometry(data.image_ids[0], data.image_ids[1])
          .is_loop_closure);
  EXPECT_TRUE(
      data.database->ReadTwoViewGeometry(data.image_ids[0], data.image_ids[2])
          .is_loop_closure);
}

}  // namespace
}  // namespace colmap
