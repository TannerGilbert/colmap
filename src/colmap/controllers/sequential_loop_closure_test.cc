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

#include "colmap/controllers/sequential_loop_closure.h"

#include "colmap/controllers/feature_matching.h"
#include "colmap/scene/synthetic.h"
#include "colmap/util/testing.h"

#include <algorithm>

#include <gtest/gtest.h>

namespace colmap {
namespace {

struct TestData {
  std::filesystem::path database_path;
  std::shared_ptr<Database> database;
  std::shared_ptr<FeatureMatcherCache> cache;
};

TestData CreateTestData(const int num_images) {
  TestData data;
  data.database_path = CreateTestDir() / "database.db";
  data.database = Database::Open(data.database_path);

  Reconstruction reconstruction;
  SyntheticDatasetOptions options;
  options.num_rigs = num_images;
  options.num_cameras_per_rig = 1;
  options.num_frames_per_rig = 1;
  options.num_points3D = 20;
  options.num_points2D_without_point3D = 3;
  SynthesizeDataset(options, &reconstruction, data.database.get());

  data.cache = std::make_shared<FeatureMatcherCache>(100, data.database);
  return data;
}

std::vector<Image> OrderedImages(Database& database) {
  std::vector<Image> images = database.ReadAllImages();
  std::sort(images.begin(), images.end(), [](const Image& a, const Image& b) {
    return a.Name() < b.Name();
  });
  return images;
}

TEST(SequentialLoopClosurePostprocess, MergesTransitiveAndCandidateRows) {
  auto data = CreateTestData(3);
  const std::vector<Image> images = OrderedImages(*data.database);
  ASSERT_GE(images.size(), 3);
  const image_t image_id1 = images[0].ImageId();
  const image_t image_id2 = images[1].ImageId();
  const image_t image_id3 = images[2].ImageId();

  data.database->ClearTwoViewGeometries();

  TwoViewGeometry tvg12;
  tvg12.inlier_matches = {{0, 0}};
  data.cache->WriteTwoViewGeometry(image_id1, image_id2, tvg12);

  TwoViewGeometry tvg23;
  tvg23.inlier_matches = {{0, 0}};
  data.cache->WriteTwoViewGeometry(image_id2, image_id3, tvg23);

  TwoViewGeometry tvg13;
  tvg13.inlier_matches = {{0, 0}, {1, 1}};
  data.cache->WriteTwoViewGeometry(image_id1, image_id3, tvg13);

  SequentialPairingOptions options;
  options.overlap = 2;
  options.quadratic_overlap = false;
  options.use_lc_provenance = true;
  DeriveSequentialLoopClosureProvenance(data.cache, options);

  const TwoViewGeometry direct12 =
      data.database->ReadTwoViewGeometry(image_id1, image_id2);
  EXPECT_FALSE(direct12.is_loop_closure);
  EXPECT_TRUE(direct12.inlier_matches_are_lc.empty());

  const TwoViewGeometry derived13 =
      data.database->ReadTwoViewGeometry(image_id1, image_id3);
  ASSERT_EQ(derived13.inlier_matches.size(), 2);
  ASSERT_EQ(derived13.inlier_matches_are_lc.size(), 2);
  EXPECT_EQ(derived13.inlier_matches[0], FeatureMatch(0, 0));
  EXPECT_EQ(derived13.inlier_matches[1], FeatureMatch(1, 1));
  EXPECT_FALSE(derived13.inlier_matches_are_lc[0]);
  EXPECT_TRUE(derived13.inlier_matches_are_lc[1]);
  EXPECT_TRUE(derived13.is_loop_closure);
}

TEST(SequentialLoopClosurePostprocess, UsesDirectPairsAsTrackingSeeds) {
  auto data = CreateTestData(3);
  const std::vector<Image> images = OrderedImages(*data.database);
  ASSERT_GE(images.size(), 3);
  const image_t image_id1 = images[0].ImageId();
  const image_t image_id2 = images[1].ImageId();
  const image_t image_id3 = images[2].ImageId();

  data.database->ClearTwoViewGeometries();

  TwoViewGeometry stale_lc12;
  stale_lc12.inlier_matches = {{0, 0}};
  stale_lc12.inlier_matches_are_lc = {true};
  stale_lc12.is_loop_closure = true;
  data.cache->WriteTwoViewGeometry(image_id1, image_id2, stale_lc12);

  TwoViewGeometry direct23;
  direct23.inlier_matches = {{0, 0}};
  data.cache->WriteTwoViewGeometry(image_id2, image_id3, direct23);

  TwoViewGeometry candidate13;
  candidate13.inlier_matches = {{0, 0}, {1, 1}};
  data.cache->WriteTwoViewGeometry(image_id1, image_id3, candidate13);

  SequentialPairingOptions options;
  options.overlap = 2;
  options.quadratic_overlap = false;
  options.use_lc_provenance = true;
  DeriveSequentialLoopClosureProvenance(data.cache, options);

  const TwoViewGeometry cleaned12 =
      data.database->ReadTwoViewGeometry(image_id1, image_id2);
  EXPECT_FALSE(cleaned12.is_loop_closure);
  EXPECT_TRUE(cleaned12.inlier_matches_are_lc.empty());

  const TwoViewGeometry derived13 =
      data.database->ReadTwoViewGeometry(image_id1, image_id3);
  ASSERT_EQ(derived13.inlier_matches.size(), 2);
  ASSERT_EQ(derived13.inlier_matches_are_lc.size(), 2);
  EXPECT_EQ(derived13.inlier_matches[0], FeatureMatch(0, 0));
  EXPECT_EQ(derived13.inlier_matches[1], FeatureMatch(1, 1));
  EXPECT_FALSE(derived13.inlier_matches_are_lc[0]);
  EXPECT_TRUE(derived13.inlier_matches_are_lc[1]);
  EXPECT_TRUE(derived13.is_loop_closure);
}

TEST(SequentialLoopClosurePostprocess, IgnoresPairsOutsideGeneratedSet) {
  auto data = CreateTestData(4);
  const std::vector<Image> images = OrderedImages(*data.database);
  ASSERT_GE(images.size(), 4);
  const image_t image_id1 = images[0].ImageId();
  const image_t image_id2 = images[1].ImageId();
  const image_t image_id3 = images[2].ImageId();
  const image_t image_id4 = images[3].ImageId();

  data.database->ClearTwoViewGeometries();

  TwoViewGeometry direct12;
  direct12.inlier_matches = {{0, 0}};
  data.cache->WriteTwoViewGeometry(image_id1, image_id2, direct12);

  TwoViewGeometry direct23;
  direct23.inlier_matches = {{0, 0}};
  data.cache->WriteTwoViewGeometry(image_id2, image_id3, direct23);

  TwoViewGeometry sequential13;
  sequential13.inlier_matches = {{0, 0}, {1, 1}};
  data.cache->WriteTwoViewGeometry(image_id1, image_id3, sequential13);

  TwoViewGeometry loop14;
  loop14.inlier_matches = {{0, 0}, {1, 1}};
  data.cache->WriteTwoViewGeometry(image_id1, image_id4, loop14);

  SequentialPairingOptions options;
  options.overlap = 2;
  options.quadratic_overlap = false;
  options.use_lc_provenance = true;
  DeriveSequentialLoopClosureProvenance(data.cache, options);

  const TwoViewGeometry ignored =
      data.database->ReadTwoViewGeometry(image_id1, image_id4);
  EXPECT_FALSE(ignored.is_loop_closure);
  EXPECT_TRUE(ignored.inlier_matches_are_lc.empty());
}

TEST(SequentialLoopClosurePostprocess, KeepsRigFramePairsNonLc) {
  const auto database_path = CreateTestDir() / "database.db";
  auto database = Database::Open(database_path);

  Reconstruction reconstruction;
  SyntheticDatasetOptions options;
  options.num_rigs = 1;
  options.num_cameras_per_rig = 2;
  options.num_frames_per_rig = 2;
  options.num_points3D = 5;
  SynthesizeDataset(options, &reconstruction, database.get());

  auto cache = std::make_shared<FeatureMatcherCache>(100, database);
  const std::vector<Image> images = OrderedImages(*database);
  ASSERT_EQ(images.size(), 4);

  const auto write_lc_tvg = [&](const image_t image_id1,
                                const image_t image_id2) {
    TwoViewGeometry tvg;
    tvg.inlier_matches = {{0, 0}};
    tvg.inlier_matches_are_lc = {true};
    tvg.is_loop_closure = true;
    cache->WriteTwoViewGeometry(image_id1, image_id2, tvg);
  };

  database->ClearTwoViewGeometries();
  const image_t same_frame_image1 = images[0].ImageId();
  const image_t same_frame_image2 = images[1].ImageId();
  const image_t adjacent_frame_image1 = images[0].ImageId();
  const image_t adjacent_frame_image2 = images[2].ImageId();
  write_lc_tvg(same_frame_image1, same_frame_image2);
  write_lc_tvg(adjacent_frame_image1, adjacent_frame_image2);

  SequentialPairingOptions pairing_options;
  pairing_options.overlap = 1;
  pairing_options.quadratic_overlap = false;
  pairing_options.expand_rig_images = true;
  pairing_options.use_lc_provenance = true;
  DeriveSequentialLoopClosureProvenance(cache, pairing_options);

  const TwoViewGeometry same_frame_tvg =
      database->ReadTwoViewGeometry(same_frame_image1, same_frame_image2);
  EXPECT_FALSE(same_frame_tvg.is_loop_closure);
  EXPECT_TRUE(same_frame_tvg.inlier_matches_are_lc.empty());

  const TwoViewGeometry adjacent_frame_tvg =
      database->ReadTwoViewGeometry(adjacent_frame_image1,
                                    adjacent_frame_image2);
  EXPECT_FALSE(adjacent_frame_tvg.is_loop_closure);
  EXPECT_TRUE(adjacent_frame_tvg.inlier_matches_are_lc.empty());
}

TEST(SequentialLoopClosurePostprocess, UsesRigFrameDistanceForExpandedPairs) {
  const auto database_path = CreateTestDir() / "database.db";
  auto database = Database::Open(database_path);

  Reconstruction reconstruction;
  SyntheticDatasetOptions options;
  options.num_rigs = 1;
  options.num_cameras_per_rig = 2;
  options.num_frames_per_rig = 3;
  options.num_points3D = 5;
  SynthesizeDataset(options, &reconstruction, database.get());

  auto cache = std::make_shared<FeatureMatcherCache>(100, database);
  const std::vector<Image> images = OrderedImages(*database);
  ASSERT_EQ(images.size(), 6);
  ASSERT_EQ(images[0].Name(), "camera000001_frame000000.png");
  ASSERT_EQ(images[1].Name(), "camera000001_frame000001.png");
  ASSERT_EQ(images[2].Name(), "camera000001_frame000002.png");
  ASSERT_EQ(images[5].Name(), "camera000002_frame000002.png");

  database->ClearTwoViewGeometries();

  TwoViewGeometry frame01;
  frame01.inlier_matches = {{0, 0}};
  cache->WriteTwoViewGeometry(images[0].ImageId(), images[1].ImageId(), frame01);

  TwoViewGeometry frame12;
  frame12.inlier_matches = {{0, 0}};
  cache->WriteTwoViewGeometry(images[1].ImageId(), images[5].ImageId(), frame12);

  TwoViewGeometry frame02_candidate;
  frame02_candidate.inlier_matches = {{0, 0}, {1, 1}};
  cache->WriteTwoViewGeometry(
      images[0].ImageId(), images[5].ImageId(), frame02_candidate);

  SequentialPairingOptions pairing_options;
  pairing_options.overlap = 2;
  pairing_options.quadratic_overlap = false;
  pairing_options.expand_rig_images = true;
  pairing_options.use_lc_provenance = true;
  DeriveSequentialLoopClosureProvenance(cache, pairing_options);

  const TwoViewGeometry derived = database->ReadTwoViewGeometry(
      images[0].ImageId(), images[5].ImageId());
  ASSERT_EQ(derived.inlier_matches_are_lc.size(), derived.inlier_matches.size());
  ASSERT_EQ(derived.inlier_matches.size(), 2);
  EXPECT_FALSE(derived.inlier_matches_are_lc[0]);
  EXPECT_TRUE(derived.inlier_matches_are_lc[1]);
  EXPECT_TRUE(derived.is_loop_closure);
}

TEST(SequentialLoopClosurePostprocess, StopsBeforeWritingRemainingPairs) {
  auto data = CreateTestData(3);
  const std::vector<Image> images = OrderedImages(*data.database);
  ASSERT_GE(images.size(), 3);
  const image_t image_id1 = images[0].ImageId();
  const image_t image_id2 = images[1].ImageId();
  const image_t image_id3 = images[2].ImageId();

  data.database->ClearTwoViewGeometries();

  TwoViewGeometry direct12;
  direct12.inlier_matches = {{0, 0}};
  data.cache->WriteTwoViewGeometry(image_id1, image_id2, direct12);

  TwoViewGeometry candidate13;
  candidate13.inlier_matches = {{0, 0}, {1, 1}};
  data.cache->WriteTwoViewGeometry(image_id1, image_id3, candidate13);

  SequentialPairingOptions options;
  options.overlap = 2;
  options.quadratic_overlap = false;
  options.use_lc_provenance = true;

  DeriveSequentialLoopClosureProvenance(data.cache, options, []() {
    return true;
  });

  const TwoViewGeometry unchanged =
      data.database->ReadTwoViewGeometry(image_id1, image_id3);
  EXPECT_FALSE(unchanged.is_loop_closure);
  EXPECT_TRUE(unchanged.inlier_matches_are_lc.empty());
}

TEST(SequentialLoopClosurePostprocess, RunsAfterSequentialMatcher) {
  auto data = CreateTestData(5);
  data.database->ClearMatches();
  data.database->ClearTwoViewGeometries();

  SequentialPairingOptions pairing_options;
  pairing_options.overlap = 2;
  pairing_options.quadratic_overlap = false;
  pairing_options.use_lc_provenance = true;

  FeatureMatchingOptions matching_options;
  matching_options.use_gpu = false;
  matching_options.num_threads = 1;

  TwoViewGeometryOptions geometry_options;

  auto matcher = CreateSequentialFeatureMatcher(
      pairing_options, matching_options, geometry_options, data.database_path);
  ASSERT_NE(matcher, nullptr);
  matcher->Start();
  matcher->Wait();

  const std::vector<Image> images = OrderedImages(*data.database);
  const TwoViewGeometry before =
      data.database->ReadTwoViewGeometry(images[0].ImageId(),
                                         images[2].ImageId());
  EXPECT_FALSE(before.is_loop_closure);
  EXPECT_TRUE(before.inlier_matches_are_lc.empty());

  DeriveSequentialLoopClosureProvenance(data.database_path, pairing_options);

  const TwoViewGeometry after =
      data.database->ReadTwoViewGeometry(images[0].ImageId(),
                                         images[2].ImageId());
  ASSERT_EQ(after.inlier_matches_are_lc.size(), after.inlier_matches.size());
  EXPECT_FALSE(after.inlier_matches_are_lc.empty());
}

}  // namespace
}  // namespace colmap
