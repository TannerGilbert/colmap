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

#include "colmap/controllers/feature_matching.h"

#include "colmap/feature/types.h"
#include "colmap/retrieval/visual_index.h"
#include "colmap/scene/synthetic.h"
#include "colmap/util/testing.h"

#include <algorithm>
#include <fstream>
#include <map>
#include <unordered_map>

#include <gtest/gtest.h>

namespace colmap {
namespace {

void CreateTestDatabase(int num_images, Database& database) {
  Reconstruction unused_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = num_images;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 1;
  synthetic_dataset_options.num_points3D = 20;
  synthetic_dataset_options.num_points2D_without_point3D = 3;
  synthetic_dataset_options.prior_position = true;
  SynthesizeDataset(
      synthetic_dataset_options, &unused_reconstruction, &database);
}

std::unique_ptr<retrieval::VisualIndex> CreateSyntheticVisualIndex() {
  auto visual_index = retrieval::VisualIndex::Create();
  retrieval::VisualIndex::BuildOptions build_options;
  build_options.num_visual_words = 5;
  visual_index->Build(
      build_options,
      FeatureDescriptorsFloat(FeatureExtractorType::SIFT,
                              FeatureDescriptorsFloatData::Random(50, 128)));
  return visual_index;
}

bool IsAdjacentByName(Database& database, image_t image_id1, image_t image_id2) {
  std::vector<Image> images = database.ReadAllImages();
  std::sort(images.begin(), images.end(), [](const Image& a, const Image& b) {
    return a.Name() < b.Name();
  });
  std::unordered_map<image_t, size_t> image_id_to_idx;
  image_id_to_idx.reserve(images.size());
  for (size_t idx = 0; idx < images.size(); ++idx) {
    image_id_to_idx.emplace(images[idx].ImageId(), idx);
  }
  const size_t idx1 = image_id_to_idx.at(image_id1);
  const size_t idx2 = image_id_to_idx.at(image_id2);
  return std::max(idx1, idx2) - std::min(idx1, idx2) == 1;
}

TEST(CreateExhaustiveFeatureMatcher, Nominal) {
  const auto test_dir = CreateTestDir();
  const auto database_path = test_dir / "database.db";
  auto database = Database::Open(database_path);
  CreateTestDatabase(/*num_images=*/4, *database);
  database->ClearMatches();
  database->ClearTwoViewGeometries();

  ExhaustivePairingOptions pairing_options;
  FeatureMatchingOptions matching_options;
  matching_options.use_gpu = false;
  matching_options.num_threads = 1;
  TwoViewGeometryOptions geometry_options;

  auto matcher = CreateExhaustiveFeatureMatcher(
      pairing_options, matching_options, geometry_options, database_path);
  ASSERT_NE(matcher, nullptr);
  matcher->Start();
  matcher->Wait();

  EXPECT_EQ(database->ReadAllMatches().size(), 6);
  EXPECT_EQ(database->ReadTwoViewGeometries().size(), 6);
}

TEST(CreateVocabTreeFeatureMatcher, Nominal) {
  const auto test_dir = CreateTestDir();
  const auto database_path = test_dir / "database.db";
  const auto vocab_tree_path = test_dir / "vocab_tree.bin";

  auto database = Database::Open(database_path);
  CreateTestDatabase(/*num_images=*/4, *database);
  database->ClearMatches();
  database->ClearTwoViewGeometries();

  // Create vocab tree
  CreateSyntheticVisualIndex()->Write(vocab_tree_path);

  VocabTreePairingOptions pairing_options;
  pairing_options.vocab_tree_path = vocab_tree_path;
  pairing_options.num_images = 2;

  FeatureMatchingOptions matching_options;
  matching_options.use_gpu = false;
  matching_options.num_threads = 1;

  TwoViewGeometryOptions geometry_options;

  auto matcher = CreateVocabTreeFeatureMatcher(
      pairing_options, matching_options, geometry_options, database_path);
  ASSERT_NE(matcher, nullptr);
  matcher->Start();
  matcher->Wait();

  // Each image should match with num_images others,
  // while some of the pairs may be redundant.
  EXPECT_GE(database->ReadAllMatches().size(), 4);
  EXPECT_GE(database->ReadTwoViewGeometries().size(), 4);
}

TEST(CreateVocabTreeFeatureMatcher, MarksMatchesAsLoopClosure) {
  const auto test_dir = CreateTestDir();
  const auto database_path = test_dir / "database.db";
  const auto vocab_tree_path = test_dir / "vocab_tree.bin";

  auto database = Database::Open(database_path);
  CreateTestDatabase(/*num_images=*/4, *database);
  database->ClearMatches();
  database->ClearTwoViewGeometries();

  // Create vocab tree
  CreateSyntheticVisualIndex()->Write(vocab_tree_path);

  VocabTreePairingOptions pairing_options;
  pairing_options.vocab_tree_path = vocab_tree_path;
  pairing_options.num_images = 2;
  pairing_options.mark_matches_as_lc = true;

  FeatureMatchingOptions matching_options;
  matching_options.use_gpu = false;
  matching_options.num_threads = 1;

  TwoViewGeometryOptions geometry_options;

  auto matcher = CreateVocabTreeFeatureMatcher(
      pairing_options, matching_options, geometry_options, database_path);
  ASSERT_NE(matcher, nullptr);
  matcher->Start();
  matcher->Wait();

  const auto two_view_geometries = database->ReadTwoViewGeometries();
  ASSERT_GE(two_view_geometries.size(), 4);
  bool saw_lc_pair = false;
  bool saw_direct_pair = false;
  for (const auto& [pair_id, two_view_geometry] : two_view_geometries) {
    const auto [image_id1, image_id2] = PairIdToImagePair(pair_id);
    if (IsAdjacentByName(*database, image_id1, image_id2)) {
      saw_direct_pair = true;
      EXPECT_FALSE(two_view_geometry.is_loop_closure);
      EXPECT_TRUE(two_view_geometry.inlier_matches_are_lc.empty());
    } else {
      saw_lc_pair = true;
      EXPECT_TRUE(two_view_geometry.is_loop_closure);
    }
  }
  EXPECT_TRUE(saw_direct_pair);
  EXPECT_TRUE(saw_lc_pair);
}

TEST(CreateSequentialFeatureMatcher, Nominal) {
  const auto test_dir = CreateTestDir();
  const auto database_path = test_dir / "database.db";
  auto database = Database::Open(database_path);
  CreateTestDatabase(/*num_images=*/5, *database);
  database->ClearMatches();
  database->ClearTwoViewGeometries();

  SequentialPairingOptions pairing_options;
  pairing_options.overlap = 2;
  pairing_options.quadratic_overlap = false;

  FeatureMatchingOptions matching_options;
  matching_options.use_gpu = false;
  matching_options.num_threads = 1;

  TwoViewGeometryOptions geometry_options;

  auto matcher = CreateSequentialFeatureMatcher(
      pairing_options, matching_options, geometry_options, database_path);
  ASSERT_NE(matcher, nullptr);
  matcher->Start();
  matcher->Wait();

  // With 5 images and overlap=2:
  // (0,1), (0,2), (1,2), (1,3), (2,3), (2,4), (3,4)
  EXPECT_EQ(database->ReadAllMatches().size(), 7);
  EXPECT_EQ(database->ReadTwoViewGeometries().size(), 7);
  const std::vector<Image> images = database->ReadAllImages();
  ASSERT_GE(images.size(), 3);
  EXPECT_FALSE(
      database->ReadTwoViewGeometry(images[0].ImageId(), images[1].ImageId())
          .is_loop_closure);
  const auto transitive_pair =
      database->ReadTwoViewGeometry(images[0].ImageId(), images[2].ImageId());
  EXPECT_FALSE(transitive_pair.is_loop_closure);
  EXPECT_TRUE(transitive_pair.inlier_matches_are_lc.empty());
}

TEST(CreateSequentialFeatureMatcher,
     MarksNonConsecutivePairsAsLoopClosureCandidatesWhenEnabled) {
  const auto test_dir = CreateTestDir();
  const auto database_path = test_dir / "database.db";
  auto database = Database::Open(database_path);
  CreateTestDatabase(/*num_images=*/5, *database);
  database->ClearMatches();
  database->ClearTwoViewGeometries();

  SequentialPairingOptions pairing_options;
  pairing_options.overlap = 2;
  pairing_options.quadratic_overlap = false;
  pairing_options.mark_non_consecutive_as_lc = true;

  FeatureMatchingOptions matching_options;
  matching_options.use_gpu = false;
  matching_options.num_threads = 1;

  TwoViewGeometryOptions geometry_options;

  auto matcher = CreateSequentialFeatureMatcher(
      pairing_options, matching_options, geometry_options, database_path);
  ASSERT_NE(matcher, nullptr);
  matcher->Start();
  matcher->Wait();

  const std::vector<Image> images = database->ReadAllImages();
  ASSERT_GE(images.size(), 3);
  const auto direct_pair =
      database->ReadTwoViewGeometry(images[0].ImageId(), images[1].ImageId());
  EXPECT_FALSE(direct_pair.is_loop_closure);
  EXPECT_TRUE(direct_pair.inlier_matches_are_lc.empty());

  const auto extended_pair =
      database->ReadTwoViewGeometry(images[0].ImageId(), images[2].ImageId());
  ASSERT_FALSE(extended_pair.inlier_matches.empty());
  EXPECT_EQ(extended_pair.inlier_matches_are_lc.size(),
            extended_pair.inlier_matches.size());
}

TEST(CreateSpatialFeatureMatcher, Nominal) {
  const auto test_dir = CreateTestDir();
  const auto database_path = test_dir / "database.db";
  auto database = Database::Open(database_path);
  CreateTestDatabase(/*num_images=*/4, *database);
  database->ClearMatches();
  database->ClearTwoViewGeometries();

  SpatialPairingOptions pairing_options;
  pairing_options.max_num_neighbors = 2;
  pairing_options.max_distance = 1e6;

  FeatureMatchingOptions matching_options;
  matching_options.use_gpu = false;
  matching_options.num_threads = 1;

  TwoViewGeometryOptions geometry_options;

  auto matcher = CreateSpatialFeatureMatcher(
      pairing_options, matching_options, geometry_options, database_path);
  ASSERT_NE(matcher, nullptr);
  matcher->Start();
  matcher->Wait();

  EXPECT_GT(database->ReadAllMatches().size(), 0);
  EXPECT_GT(database->ReadTwoViewGeometries().size(), 0);
}

TEST(CreateTransitiveFeatureMatcher, Nominal) {
  const auto test_dir = CreateTestDir();
  const auto database_path = test_dir / "database.db";
  auto database = Database::Open(database_path);
  CreateTestDatabase(/*num_images=*/4, *database);
  database->ClearMatches();
  database->ClearTwoViewGeometries();

  const std::vector<Image> images = database->ReadAllImages();
  ASSERT_GE(images.size(), 3);

  // Create initial matches: 1-2 and 2-3
  TwoViewGeometry two_view_geometry;
  two_view_geometry.config = TwoViewGeometry::CALIBRATED;
  two_view_geometry.inlier_matches = FeatureMatches(10);

  database->WriteTwoViewGeometry(
      images[0].ImageId(), images[1].ImageId(), two_view_geometry);
  database->WriteTwoViewGeometry(
      images[1].ImageId(), images[2].ImageId(), two_view_geometry);

  TransitivePairingOptions pairing_options;
  pairing_options.batch_size = 100;
  pairing_options.num_iterations = 1;

  FeatureMatchingOptions matching_options;
  matching_options.use_gpu = false;
  matching_options.num_threads = 1;

  TwoViewGeometryOptions geometry_options;

  auto matcher = CreateTransitiveFeatureMatcher(
      pairing_options, matching_options, geometry_options, database_path);
  ASSERT_NE(matcher, nullptr);
  matcher->Start();
  matcher->Wait();

  // Should create transitive match 1-3
  const size_t final_matches = database->ReadTwoViewGeometries().size();
  EXPECT_GE(final_matches, 2);  // At least the original 2 matches
}

TEST(CreateImagePairsFeatureMatcher, Nominal) {
  const auto test_dir = CreateTestDir();
  const auto database_path = test_dir / "database.db";
  const auto match_list_path = test_dir / "match_list.txt";

  auto database = Database::Open(database_path);
  CreateTestDatabase(/*num_images=*/4, *database);
  database->ClearMatches();
  database->ClearTwoViewGeometries();

  const std::vector<Image> images = database->ReadAllImages();
  ASSERT_GE(images.size(), 3);

  // Create match list file with specific image pairs
  std::ofstream file(match_list_path);
  file << images[0].Name() << " " << images[1].Name() << "\n";
  file << images[1].Name() << " " << images[2].Name() << "\n";
  file << images[2].Name() << " " << images[3].Name() << "\n";
  file.close();

  ImportedPairingOptions pairing_options;
  pairing_options.match_list_path = match_list_path;

  FeatureMatchingOptions matching_options;
  matching_options.use_gpu = false;
  matching_options.num_threads = 1;

  TwoViewGeometryOptions geometry_options;

  auto matcher = CreateImagePairsFeatureMatcher(
      pairing_options, matching_options, geometry_options, database_path);
  ASSERT_NE(matcher, nullptr);
  matcher->Start();
  matcher->Wait();

  EXPECT_EQ(database->ReadAllMatches().size(), 3);
  EXPECT_EQ(database->ReadTwoViewGeometries().size(), 3);
}

TEST(CreateFeaturePairsFeatureMatcher, Nominal) {
  const auto test_dir = CreateTestDir();
  const auto database_path = test_dir / "database.db";
  const auto match_list_path = test_dir / "feature_match_list.txt";

  auto database = Database::Open(database_path);
  CreateTestDatabase(/*num_images=*/3, *database);
  database->ClearMatches();
  database->ClearTwoViewGeometries();

  const std::vector<Image> images = database->ReadAllImages();
  ASSERT_GE(images.size(), 2);

  // Create feature match list file with many matches for better verification
  std::ofstream file(match_list_path);
  file << images[0].Name() << " " << images[1].Name() << "\n";
  for (int i = 0; i < 15; ++i) {
    file << i << " " << i << "\n";
  }
  file << "\n";  // Empty line separates pairs
  file << images[1].Name() << " " << images[2].Name() << "\n";
  for (int i = 0; i < 15; ++i) {
    file << i << " " << i << "\n";
  }
  file << "\n";
  file.close();

  FeaturePairsMatchingOptions pairing_options;
  pairing_options.match_list_path = match_list_path;
  pairing_options.verify_matches = true;

  FeatureMatchingOptions matching_options;
  matching_options.use_gpu = false;
  matching_options.num_threads = 1;

  TwoViewGeometryOptions geometry_options;
  geometry_options.min_num_inliers = 5;  // Lower threshold for testing

  auto matcher = CreateFeaturePairsFeatureMatcher(
      pairing_options, matching_options, geometry_options, database_path);
  ASSERT_NE(matcher, nullptr);
  matcher->Start();
  matcher->Wait();

  // Should have imported and verified the matches
  EXPECT_GE(database->ReadTwoViewGeometries().size(), 2);
}

TEST(CreateGeometricVerifier, Nominal) {
  const auto test_dir = CreateTestDir();
  const auto database_path = test_dir / "database.db";
  auto database = Database::Open(database_path);
  CreateTestDatabase(/*num_images=*/4, *database);
  database->ClearTwoViewGeometries();

  ExistingMatchedPairingOptions pairing_options;

  GeometricVerifierOptions verifier_options;
  verifier_options.num_threads = 1;

  TwoViewGeometryOptions geometry_options;

  auto verifier = CreateGeometricVerifier(
      verifier_options, pairing_options, geometry_options, database_path);
  ASSERT_NE(verifier, nullptr);
  verifier->Start();
  verifier->Wait();

  EXPECT_GE(database->ReadAllMatches().size(), 3);
  EXPECT_GE(database->ReadTwoViewGeometries().size(), 3);
}

void ExpectRigVerificationResults(const Database& database,
                                  int num_expected_matches,
                                  int num_expected_calibrated,
                                  int num_expected_calibrated_rig) {
  // Verify that two-view geometries were created.
  int num_calibrated = 0;
  int num_calibrated_rig = 0;
  int num_others = 0;
  for (const auto& [pair_id, two_view_geometry] :
       database.ReadTwoViewGeometries()) {
    EXPECT_EQ(two_view_geometry.inlier_matches.size(), num_expected_matches);
    switch (two_view_geometry.config) {
      case TwoViewGeometry::CALIBRATED:
        ++num_calibrated;
        break;
      case TwoViewGeometry::CALIBRATED_RIG:
        ++num_calibrated_rig;
        break;
      default:
        ++num_others;
    }
  }
  // Two calibrated pairs between images in the same frames.
  EXPECT_EQ(num_calibrated, num_expected_calibrated);
  // Four calibrated pairs between images in different frames.
  EXPECT_EQ(num_calibrated_rig, num_expected_calibrated_rig);
  EXPECT_EQ(num_others, 0);
}

TEST(CreateGeometricVerifier, RigVerificationWithNonTrivialFrames) {
  const auto test_dir = CreateTestDir();
  const auto database_path = test_dir / "database.db";
  auto database = Database::Open(database_path);

  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 3;
  synthetic_dataset_options.num_frames_per_rig = 2;
  synthetic_dataset_options.num_points3D = 25;
  synthetic_dataset_options.match_config =
      SyntheticDatasetOptions::MatchConfig::EXHAUSTIVE;
  synthetic_dataset_options.camera_has_prior_focal_length = true;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction, database.get());

  std::map<image_pair_t, std::map<std::pair<point2D_t, point2D_t>, bool>>
      expected_lc_masks;
  for (const auto& [pair_id, tvg] : database->ReadTwoViewGeometries()) {
    const auto [image_id1, image_id2] = PairIdToImagePair(pair_id);
    TwoViewGeometry mixed_lc_tvg = tvg;
    mixed_lc_tvg.inlier_matches_are_lc.resize(
        mixed_lc_tvg.inlier_matches.size());
    auto& expected_mask = expected_lc_masks[pair_id];
    for (size_t i = 0; i < mixed_lc_tvg.inlier_matches.size(); ++i) {
      const bool is_lc = i % 3 == 0;
      const FeatureMatch& match = mixed_lc_tvg.inlier_matches[i];
      mixed_lc_tvg.inlier_matches_are_lc[i] = is_lc;
      expected_mask[{match.point2D_idx1, match.point2D_idx2}] = is_lc;
    }
    mixed_lc_tvg.is_loop_closure =
        std::any_of(mixed_lc_tvg.inlier_matches_are_lc.begin(),
                    mixed_lc_tvg.inlier_matches_are_lc.end(),
                    [](const bool value) { return value; });
    database->UpdateTwoViewGeometry(image_id1, image_id2, mixed_lc_tvg);
  }

  ExistingMatchedPairingOptions pairing_options;

  GeometricVerifierOptions verifier_options;
  verifier_options.num_threads = -1;
  verifier_options.rig_verification = true;

  TwoViewGeometryOptions geometry_options;
  geometry_options.min_num_inliers = 5;

  auto verifier = CreateGeometricVerifier(
      verifier_options, pairing_options, geometry_options, database_path);
  ASSERT_NE(verifier, nullptr);
  verifier->Start();
  verifier->Wait();

  // All pairs should be overwritten with calibrated rig pairs.
  ExpectRigVerificationResults(*database,
                               synthetic_dataset_options.num_points3D,
                               /*num_expected_calibrated=*/0,
                               /*num_expected_calibrated_rig=*/15);

  int num_preserved_pairs = 0;
  int num_mask_preserved_pairs = 0;
  for (const auto& [pair_id, tvg] : database->ReadTwoViewGeometries()) {
    const auto expected_mask_it = expected_lc_masks.find(pair_id);
    if (expected_mask_it == expected_lc_masks.end()) {
      continue;
    }
    ++num_preserved_pairs;
    if (tvg.inlier_matches_are_lc.empty()) {
      EXPECT_FALSE(tvg.is_loop_closure);
      continue;
    }
    ++num_mask_preserved_pairs;
    ASSERT_EQ(tvg.inlier_matches_are_lc.size(), tvg.inlier_matches.size());
    std::vector<bool> expected_mask(tvg.inlier_matches.size(), false);
    for (size_t i = 0; i < tvg.inlier_matches.size(); ++i) {
      const FeatureMatch& match = tvg.inlier_matches[i];
      const auto row_it = expected_mask_it->second.find(
          {match.point2D_idx1, match.point2D_idx2});
      if (row_it != expected_mask_it->second.end()) {
        expected_mask[i] = row_it->second;
      }
    }
    EXPECT_EQ(tvg.inlier_matches_are_lc, expected_mask);
    EXPECT_EQ(tvg.is_loop_closure,
              std::any_of(expected_mask.begin(),
                          expected_mask.end(),
                          [](const bool value) { return value; }));
    EXPECT_NE(std::find(tvg.inlier_matches_are_lc.begin(),
                        tvg.inlier_matches_are_lc.end(),
                        true),
              tvg.inlier_matches_are_lc.end());
    EXPECT_NE(std::find(tvg.inlier_matches_are_lc.begin(),
                        tvg.inlier_matches_are_lc.end(),
                        false),
              tvg.inlier_matches_are_lc.end());
  }
  EXPECT_GT(num_preserved_pairs, 0);
  EXPECT_GT(num_mask_preserved_pairs, 0);
}

TEST(CreateGeometricVerifier, RigVerificationWithTrivialFrames) {
  const auto test_dir = CreateTestDir();
  const auto database_path = test_dir / "database.db";
  auto database = Database::Open(database_path);

  Reconstruction reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 2;
  synthetic_dataset_options.num_points3D = 25;
  synthetic_dataset_options.match_config =
      SyntheticDatasetOptions::MatchConfig::EXHAUSTIVE;
  synthetic_dataset_options.camera_has_prior_focal_length = true;
  SynthesizeDataset(synthetic_dataset_options, &reconstruction, database.get());

  ExistingMatchedPairingOptions pairing_options;

  GeometricVerifierOptions verifier_options;
  verifier_options.num_threads = 1;
  verifier_options.rig_verification = true;

  TwoViewGeometryOptions geometry_options;
  geometry_options.min_num_inliers = 5;

  auto verifier = CreateGeometricVerifier(
      verifier_options, pairing_options, geometry_options, database_path);
  ASSERT_NE(verifier, nullptr);
  verifier->Start();
  verifier->Wait();

  // Trivial frames should be skipped and unmodified.
  ExpectRigVerificationResults(*database,
                               synthetic_dataset_options.num_points3D,
                               /*num_expected_calibrated=*/1,
                               /*num_expected_calibrated_rig=*/0);
}

TEST(CreateGeometricVerifier, Guided) {
  const auto test_dir = CreateTestDir();
  const auto database_path = test_dir / "database.db";
  auto database = Database::Open(database_path);
  Reconstruction gt_reconstruction;

  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 5;
  synthetic_dataset_options.num_points3D = 50;
  synthetic_dataset_options.inlier_match_ratio = 0.6;
  synthetic_dataset_options.two_view_geometry_has_relative_pose = true;
  SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database.get());

  // Clear all inlier matches. cam2_from_cam1 is already gt from the synthesized
  // database.
  std::vector<std::pair<image_pair_t, TwoViewGeometry>> gt_two_view_geometries =
      database->ReadTwoViewGeometries();
  for (const auto& [pair_id, _] : gt_two_view_geometries) {
    const auto [image_id1, image_id2] = PairIdToImagePair(pair_id);
    database->DeleteInlierMatches(image_id1, image_id2);
  }

  ExistingMatchedPairingOptions pairing_options;
  GeometricVerifierOptions verifier_options;
  verifier_options.num_threads = 1;
  verifier_options.use_existing_relative_pose = true;

  TwoViewGeometryOptions geometry_options;

  auto verifier = CreateGeometricVerifier(
      verifier_options, pairing_options, geometry_options, database_path);
  ASSERT_NE(verifier, nullptr);
  verifier->Start();
  verifier->Wait();

  // Check validity after guided geometric verification.
  std::vector<std::pair<image_pair_t, TwoViewGeometry>> two_view_geometries =
      database->ReadTwoViewGeometries();
  EXPECT_GE(two_view_geometries.size(), gt_two_view_geometries.size());
  for (size_t i = 0; i < two_view_geometries.size(); ++i) {
    EXPECT_EQ(two_view_geometries[i].first, gt_two_view_geometries[i].first);
    EXPECT_EQ(two_view_geometries[i].second.cam2_from_cam1,
              gt_two_view_geometries[i].second.cam2_from_cam1);
    EXPECT_TRUE(gt_two_view_geometries[i].second.E.value().isApprox(
        two_view_geometries[i].second.E.value()));
    // Should at least have all the original inliers. Some generated outliers
    // can be accidentally inliers as well.
    EXPECT_GE(two_view_geometries[i].second.inlier_matches.size(),
              gt_two_view_geometries[i].second.inlier_matches.size());
  }
}

}  // namespace
}  // namespace colmap
