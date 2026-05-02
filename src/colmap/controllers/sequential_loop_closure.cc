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

#include "colmap/scene/correspondence_graph.h"
#include "colmap/scene/database.h"
#include "colmap/util/logging.h"
#include "colmap/util/misc.h"
#include "colmap/util/timer.h"

#include <algorithm>
#include <functional>
#include <limits>
#include <unordered_map>
#include <unordered_set>

namespace colmap {
namespace {

bool HasLoopClosureInliers(const TwoViewGeometry& two_view_geometry) {
  if (!two_view_geometry.inlier_matches_are_lc.empty()) {
    THROW_CHECK_EQ(two_view_geometry.inlier_matches_are_lc.size(),
                   two_view_geometry.inlier_matches.size());
    return std::any_of(two_view_geometry.inlier_matches_are_lc.begin(),
                       two_view_geometry.inlier_matches_are_lc.end(),
                       [](const bool value) { return value; });
  }
  return two_view_geometry.is_loop_closure &&
         !two_view_geometry.inlier_matches.empty();
}

void MergeLoopClosureInlierMatches(
    const FeatureMatches& transitive_matches,
    const FeatureMatches& candidate_matches,
    FeatureMatches* merged_matches,
    std::vector<bool>* merged_matches_are_lc) {
  THROW_CHECK_NOTNULL(merged_matches);
  THROW_CHECK_NOTNULL(merged_matches_are_lc);

  std::unordered_set<point2D_t> transitive_points1;
  std::unordered_set<point2D_t> transitive_points2;
  merged_matches->clear();
  merged_matches_are_lc->clear();
  merged_matches->reserve(transitive_matches.size() + candidate_matches.size());
  merged_matches_are_lc->reserve(merged_matches->capacity());

  for (const FeatureMatch& match : transitive_matches) {
    transitive_points1.insert(match.point2D_idx1);
    transitive_points2.insert(match.point2D_idx2);
    merged_matches->push_back(match);
    merged_matches_are_lc->push_back(false);
  }

  for (const FeatureMatch& match : candidate_matches) {
    if (transitive_points1.count(match.point2D_idx1) > 0 ||
        transitive_points2.count(match.point2D_idx2) > 0) {
      continue;
    }
    merged_matches->push_back(match);
    merged_matches_are_lc->push_back(true);
  }
}

class AdjacentTransitiveMatchExtractor {
 public:
  explicit AdjacentTransitiveMatchExtractor(
      std::shared_ptr<FeatureMatcherCache> cache)
      : cache_(std::move(cache)) {
    BuildSequenceIndex();
  }

  bool IsDirectSequentialPair(const image_t image_id1,
                              const image_t image_id2) const {
    return IsAdjacentPair(image_id1, image_id2);
  }

  FeatureMatches ExtractBetweenImages(const image_t image_id1,
                                      const image_t image_id2) {
    EnsureGraphBuilt();

    FeatureMatches transitive_matches;
    if (!correspondence_graph_.ExistsImage(image_id1) ||
        !correspondence_graph_.ExistsImage(image_id2) ||
        !cache_->ExistsKeypoints(image_id1)) {
      return transitive_matches;
    }

    std::unordered_set<point2D_t> used_points1;
    std::unordered_set<point2D_t> used_points2;
    std::vector<CorrespondenceGraph::Correspondence> correspondences;
    const size_t num_points2D1 = cache_->GetKeypoints(image_id1)->size();
    for (point2D_t point2D_idx1 = 0; point2D_idx1 < num_points2D1;
         ++point2D_idx1) {
      correspondence_graph_.ExtractTransitiveCorrespondences(
          image_id1,
          point2D_idx1,
          /*transitivity=*/std::numeric_limits<size_t>::max(),
          &correspondences);
      for (const auto& correspondence : correspondences) {
        if (correspondence.image_id != image_id2) {
          continue;
        }
        if (used_points1.insert(point2D_idx1).second &&
            used_points2.insert(correspondence.point2D_idx).second) {
          transitive_matches.emplace_back(point2D_idx1,
                                          correspondence.point2D_idx);
        }
        break;
      }
    }
    return transitive_matches;
  }

 private:
  static constexpr size_t kInvalidDistance =
      std::numeric_limits<size_t>::max();

  void EnsureGraphBuilt() {
    if (graph_built_) {
      return;
    }
    BuildGraph();
    graph_built_ = true;
  }

  void BuildSequenceIndex() {
    std::vector<Image> ordered_images;
    ordered_images.reserve(cache_->GetImageIds().size());
    for (const image_t image_id : cache_->GetImageIds()) {
      ordered_images.push_back(cache_->GetImage(image_id));
    }
    std::sort(ordered_images.begin(),
              ordered_images.end(),
              [](const Image& image1, const Image& image2) {
                return image1.Name() < image2.Name();
              });

    image_id_to_sequence_idx_.reserve(ordered_images.size());
    for (size_t idx = 0; idx < ordered_images.size(); ++idx) {
      const Image& image = ordered_images[idx];
      image_id_to_sequence_idx_.emplace(image.ImageId(), idx);
      if (image.HasFrameId()) {
        image_id_to_frame_id_.emplace(image.ImageId(), image.FrameId());
        if (frame_id_to_sequence_idx_.count(image.FrameId()) == 0) {
          frame_id_to_sequence_idx_.emplace(image.FrameId(),
                                            frame_id_to_sequence_idx_.size());
        }
      }
    }
  }

  void BuildGraph() {
    for (const image_t image_id : cache_->GetImageIds()) {
      if (cache_->ExistsKeypoints(image_id)) {
        correspondence_graph_.AddImage(image_id,
                                       cache_->GetKeypoints(image_id)->size());
      }
    }

    cache_->AccessDatabase([&](Database& database) {
      for (auto& [pair_id, two_view_geometry] :
           database.ReadTwoViewGeometries()) {
        const auto [image_id1, image_id2] = PairIdToImagePair(pair_id);
        AddNonLcAdjacentGeometry(image_id1, image_id2, two_view_geometry);
      }
    });
  }

  bool IsAdjacentPair(const image_t image_id1, const image_t image_id2) const {
    const size_t distance = SequenceDistance(image_id1, image_id2);
    return distance == 0 || distance == 1;
  }

  size_t SequenceDistance(const image_t image_id1,
                          const image_t image_id2) const {
    const auto frame_id1_it = image_id_to_frame_id_.find(image_id1);
    const auto frame_id2_it = image_id_to_frame_id_.find(image_id2);
    if (frame_id1_it != image_id_to_frame_id_.end() &&
        frame_id2_it != image_id_to_frame_id_.end()) {
      if (frame_id1_it->second == frame_id2_it->second) {
        return 0;
      }
      const auto frame_idx1_it =
          frame_id_to_sequence_idx_.find(frame_id1_it->second);
      const auto frame_idx2_it =
          frame_id_to_sequence_idx_.find(frame_id2_it->second);
      if (frame_idx1_it != frame_id_to_sequence_idx_.end() &&
          frame_idx2_it != frame_id_to_sequence_idx_.end()) {
        return std::max(frame_idx1_it->second, frame_idx2_it->second) -
               std::min(frame_idx1_it->second, frame_idx2_it->second);
      }
    }

    const auto seq_idx1_it = image_id_to_sequence_idx_.find(image_id1);
    const auto seq_idx2_it = image_id_to_sequence_idx_.find(image_id2);
    if (seq_idx1_it == image_id_to_sequence_idx_.end() ||
        seq_idx2_it == image_id_to_sequence_idx_.end()) {
      return kInvalidDistance;
    }
    return std::max(seq_idx1_it->second, seq_idx2_it->second) -
           std::min(seq_idx1_it->second, seq_idx2_it->second);
  }

  void AddNonLcAdjacentGeometry(const image_t image_id1,
                                const image_t image_id2,
                                const TwoViewGeometry& two_view_geometry) {
    if (!IsAdjacentPair(image_id1, image_id2) ||
        !correspondence_graph_.ExistsImage(image_id1) ||
        !correspondence_graph_.ExistsImage(image_id2)) {
      return;
    }

    if (two_view_geometry.inlier_matches.empty()) {
      return;
    }

    TwoViewGeometry non_lc_geometry = two_view_geometry;
    non_lc_geometry.inlier_matches_are_lc.clear();
    non_lc_geometry.is_loop_closure = false;
    correspondence_graph_.AddTwoViewGeometry(
        image_id1, image_id2, std::move(non_lc_geometry));
  }

  std::shared_ptr<FeatureMatcherCache> cache_;
  std::unordered_map<image_t, size_t> image_id_to_sequence_idx_;
  std::unordered_map<image_t, frame_t> image_id_to_frame_id_;
  std::unordered_map<frame_t, size_t> frame_id_to_sequence_idx_;
  CorrespondenceGraph correspondence_graph_;
  bool graph_built_ = false;
};

std::unordered_set<image_pair_t> GenerateSequentialPairIds(
    const std::shared_ptr<FeatureMatcherCache>& cache,
    const SequentialPairingOptions& options) {
  SequentialPairGenerator pair_generator(options, cache);
  std::unordered_set<image_pair_t> pair_ids;
  for (const auto& [image_id1, image_id2] : pair_generator.AllPairs()) {
    if (image_id1 != image_id2) {
      pair_ids.insert(ImagePairToPairId(image_id1, image_id2));
    }
  }
  return pair_ids;
}

}  // namespace

bool SequentialLoopClosurePostprocessEnabled(
    const SequentialPairingOptions& options) {
  return options.use_lc_provenance;
}

void DeriveSequentialLoopClosureProvenance(
    const std::shared_ptr<FeatureMatcherCache>& cache,
    const SequentialPairingOptions& options,
    const std::function<bool()>& is_stopped) {
  THROW_CHECK_NOTNULL(cache);
  if (!SequentialLoopClosurePostprocessEnabled(options)) {
    return;
  }

  Timer timer;
  timer.Start();
  LOG_HEADING1("Deriving sequential loop-closure provenance");

  AdjacentTransitiveMatchExtractor transitive_match_extractor(cache);
  const std::unordered_set<image_pair_t> generated_pair_ids =
      GenerateSequentialPairIds(cache, options);
  const auto finalize_geometry = [&](const image_t image_id1,
                                     const image_t image_id2,
                                     TwoViewGeometry* two_view_geometry) {
    THROW_CHECK_NOTNULL(two_view_geometry);
    if (two_view_geometry->inlier_matches.empty()) {
      two_view_geometry->inlier_matches_are_lc.clear();
      two_view_geometry->is_loop_closure = false;
      return;
    }

    if (transitive_match_extractor.IsDirectSequentialPair(image_id1,
                                                          image_id2)) {
      two_view_geometry->inlier_matches_are_lc.clear();
      two_view_geometry->is_loop_closure = false;
      return;
    }

    const FeatureMatches transitive_matches =
        transitive_match_extractor.ExtractBetweenImages(image_id1, image_id2);

    FeatureMatches merged_matches;
    std::vector<bool> inlier_matches_are_lc;
    MergeLoopClosureInlierMatches(transitive_matches,
                                  two_view_geometry->inlier_matches,
                                  &merged_matches,
                                  &inlier_matches_are_lc);
    two_view_geometry->inlier_matches = std::move(merged_matches);
    two_view_geometry->inlier_matches_are_lc = std::move(inlier_matches_are_lc);
    two_view_geometry->is_loop_closure =
        HasLoopClosureInliers(*two_view_geometry);
  };

  std::vector<std::pair<image_pair_t, TwoViewGeometry>> two_view_geometries;
  cache->AccessDatabase([&](Database& database) {
    two_view_geometries = database.ReadTwoViewGeometries();
  });

  for (auto& [pair_id, two_view_geometry] : two_view_geometries) {
    if (is_stopped && is_stopped()) {
      break;
    }
    if (generated_pair_ids.count(pair_id) == 0) {
      continue;
    }
    const auto [image_id1, image_id2] = PairIdToImagePair(pair_id);
    finalize_geometry(image_id1, image_id2, &two_view_geometry);
    cache->DeleteTwoViewGeometry(image_id1, image_id2);
    cache->WriteTwoViewGeometry(image_id1, image_id2, two_view_geometry);
  }

  timer.PrintMinutes();
}

void DeriveSequentialLoopClosureProvenance(
    const std::filesystem::path& database_path,
    const SequentialPairingOptions& options,
    const std::function<bool()>& is_stopped) {
  if (!SequentialLoopClosurePostprocessEnabled(options)) {
    return;
  }
  auto database = Database::Open(database_path);
  auto cache = std::make_shared<FeatureMatcherCache>(
      options.CacheSize(), std::move(database));
  DeriveSequentialLoopClosureProvenance(cache, options, is_stopped);
}

}  // namespace colmap
