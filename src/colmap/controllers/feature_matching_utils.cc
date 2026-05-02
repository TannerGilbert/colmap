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

#include "colmap/estimators/two_view_geometry.h"
#include "colmap/feature/sift.h"
#include "colmap/feature/utils.h"
#include "colmap/scene/correspondence_graph.h"
#include "colmap/util/cuda.h"
#include "colmap/util/misc.h"

#if defined(COLMAP_CUDA_ENABLED)
#include <cuda_runtime.h>
#endif

#include <algorithm>
#include <limits>
#include <unordered_map>
#include <unordered_set>

namespace colmap {

void MergeLoopClosureInlierMatches(const FeatureMatches& transitive_matches,
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

namespace {

bool IsLcInlier(const TwoViewGeometry& two_view_geometry, const size_t idx) {
  if (!two_view_geometry.inlier_matches_are_lc.empty()) {
    THROW_CHECK_EQ(two_view_geometry.inlier_matches_are_lc.size(),
                   two_view_geometry.inlier_matches.size());
    return two_view_geometry.inlier_matches_are_lc[idx];
  }
  return two_view_geometry.is_loop_closure;
}

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

bool IsDirectSequentialPair(FeatureMatcherCache& cache,
                            const image_t image_id1,
                            const image_t image_id2) {
  std::vector<Image> ordered_images;
  ordered_images.reserve(cache.GetImageIds().size());
  for (const image_t image_id : cache.GetImageIds()) {
    ordered_images.push_back(cache.GetImage(image_id));
  }
  std::sort(ordered_images.begin(),
            ordered_images.end(),
            [](const Image& image1, const Image& image2) {
              return image1.Name() < image2.Name();
            });

  size_t sequence_idx1 = std::numeric_limits<size_t>::max();
  size_t sequence_idx2 = std::numeric_limits<size_t>::max();
  for (size_t idx = 0; idx < ordered_images.size(); ++idx) {
    const image_t image_id = ordered_images[idx].ImageId();
    if (image_id == image_id1) {
      sequence_idx1 = idx;
    } else if (image_id == image_id2) {
      sequence_idx2 = idx;
    }
  }
  if (sequence_idx1 == std::numeric_limits<size_t>::max() ||
      sequence_idx2 == std::numeric_limits<size_t>::max()) {
    return false;
  }
  return std::max(sequence_idx1, sequence_idx2) -
             std::min(sequence_idx1, sequence_idx2) ==
         1;
}

struct FeatureMatchKey {
  point2D_t point2D_idx1 = kInvalidPoint2DIdx;
  point2D_t point2D_idx2 = kInvalidPoint2DIdx;

  bool operator==(const FeatureMatchKey& other) const {
    return point2D_idx1 == other.point2D_idx1 &&
           point2D_idx2 == other.point2D_idx2;
  }
};

struct FeatureMatchKeyHash {
  size_t operator()(const FeatureMatchKey& key) const {
    return std::hash<point2D_t>()(key.point2D_idx1) ^
           (std::hash<point2D_t>()(key.point2D_idx2) << 1);
  }
};

void PreserveInlierLoopClosureProvenance(const TwoViewGeometry& prior,
                                         TwoViewGeometry* updated) {
  THROW_CHECK_NOTNULL(updated);
  if (prior.inlier_matches_are_lc.empty()) {
    if (prior.is_loop_closure && !updated->inlier_matches.empty()) {
      updated->inlier_matches_are_lc.assign(updated->inlier_matches.size(),
                                            true);
      updated->is_loop_closure = true;
    }
    return;
  }

  THROW_CHECK_EQ(prior.inlier_matches_are_lc.size(),
                 prior.inlier_matches.size());

  std::unordered_map<FeatureMatchKey, bool, FeatureMatchKeyHash>
      prior_inlier_is_lc;
  prior_inlier_is_lc.reserve(prior.inlier_matches.size());
  for (size_t i = 0; i < prior.inlier_matches.size(); ++i) {
    const FeatureMatch& match = prior.inlier_matches[i];
    prior_inlier_is_lc[{match.point2D_idx1, match.point2D_idx2}] =
        prior.inlier_matches_are_lc[i];
  }

  updated->inlier_matches_are_lc.assign(updated->inlier_matches.size(), false);
  for (size_t i = 0; i < updated->inlier_matches.size(); ++i) {
    const FeatureMatch& match = updated->inlier_matches[i];
    const auto it =
        prior_inlier_is_lc.find({match.point2D_idx1, match.point2D_idx2});
    if (it != prior_inlier_is_lc.end()) {
      updated->inlier_matches_are_lc[i] = it->second;
    }
  }
  updated->is_loop_closure =
      std::any_of(updated->inlier_matches_are_lc.begin(),
                  updated->inlier_matches_are_lc.end(),
                  [](const bool value) { return value; });
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

  void AddFinalizedOutput(const FeatureMatcherData& output) {
    if (!graph_built_) {
      return;
    }
    AddNonLcAdjacentGeometry(
        output.image_id1, output.image_id2, output.two_view_geometry);
  }

 private:
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
      image_id_to_sequence_idx_.emplace(ordered_images[idx].ImageId(), idx);
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
    const auto seq_idx1_it = image_id_to_sequence_idx_.find(image_id1);
    const auto seq_idx2_it = image_id_to_sequence_idx_.find(image_id2);
    if (seq_idx1_it == image_id_to_sequence_idx_.end() ||
        seq_idx2_it == image_id_to_sequence_idx_.end()) {
      return false;
    }
    return std::max(seq_idx1_it->second, seq_idx2_it->second) -
               std::min(seq_idx1_it->second, seq_idx2_it->second) ==
           1;
  }

  void AddNonLcAdjacentGeometry(const image_t image_id1,
                                const image_t image_id2,
                                const TwoViewGeometry& two_view_geometry) {
    if (!IsAdjacentPair(image_id1, image_id2) ||
        !correspondence_graph_.ExistsImage(image_id1) ||
        !correspondence_graph_.ExistsImage(image_id2)) {
      return;
    }

    FeatureMatches non_lc_matches;
    non_lc_matches.reserve(two_view_geometry.inlier_matches.size());
    for (size_t i = 0; i < two_view_geometry.inlier_matches.size(); ++i) {
      if (!IsLcInlier(two_view_geometry, i)) {
        non_lc_matches.push_back(two_view_geometry.inlier_matches[i]);
      }
    }
    if (non_lc_matches.empty()) {
      return;
    }

    TwoViewGeometry non_lc_geometry = two_view_geometry;
    non_lc_geometry.inlier_matches = std::move(non_lc_matches);
    non_lc_geometry.inlier_matches_are_lc.clear();
    non_lc_geometry.is_loop_closure = false;
    correspondence_graph_.AddTwoViewGeometry(
        image_id1, image_id2, std::move(non_lc_geometry));
  }

  std::shared_ptr<FeatureMatcherCache> cache_;
  std::unordered_map<image_t, size_t> image_id_to_sequence_idx_;
  CorrespondenceGraph correspondence_graph_;
  bool graph_built_ = false;
};

void MergeLcCandidateWithTransitiveMatches(
    AdjacentTransitiveMatchExtractor& transitive_match_extractor,
    FeatureMatcherData* output) {
  THROW_CHECK_NOTNULL(output);
  if (!output->is_loop_closure ||
      output->two_view_geometry.inlier_matches.empty()) {
    return;
  }

  FeatureMatches transitive_matches =
      transitive_match_extractor.ExtractBetweenImages(output->image_id1,
                                                      output->image_id2);

  FeatureMatches merged_matches;
  std::vector<bool> inlier_matches_are_lc;
  MergeLoopClosureInlierMatches(transitive_matches,
                                output->two_view_geometry.inlier_matches,
                                &merged_matches,
                                &inlier_matches_are_lc);

  output->two_view_geometry.inlier_matches = std::move(merged_matches);
  output->two_view_geometry.inlier_matches_are_lc =
      std::move(inlier_matches_are_lc);
  output->two_view_geometry.is_loop_closure =
      std::any_of(output->two_view_geometry.inlier_matches_are_lc.begin(),
                  output->two_view_geometry.inlier_matches_are_lc.end(),
                  [](const bool value) { return value; });
}

void FinalizeLoopClosureProvenance(
    AdjacentTransitiveMatchExtractor& transitive_match_extractor,
    FeatureMatcherData* output) {
  THROW_CHECK_NOTNULL(output);
  if (output->two_view_geometry.inlier_matches.empty()) {
    output->two_view_geometry.inlier_matches_are_lc.clear();
    output->two_view_geometry.is_loop_closure = false;
    return;
  }

  if (transitive_match_extractor.IsDirectSequentialPair(output->image_id1,
                                                        output->image_id2)) {
    output->is_loop_closure = false;
    output->two_view_geometry.inlier_matches_are_lc.clear();
    output->two_view_geometry.is_loop_closure = false;
    return;
  }

  MergeLcCandidateWithTransitiveMatches(transitive_match_extractor, output);
  if (!output->two_view_geometry.inlier_matches_are_lc.empty()) {
    output->two_view_geometry.is_loop_closure =
        HasLoopClosureInliers(output->two_view_geometry);
    return;
  }
  output->two_view_geometry.is_loop_closure = output->is_loop_closure;
}

}  // namespace

FeatureMatcherWorker::FeatureMatcherWorker(
    const FeatureMatchingOptions& matching_options,
    const TwoViewGeometryOptions& geometry_options,
    const std::shared_ptr<FeatureMatcherCache>& cache,
    JobQueue<Input>* input_queue,
    JobQueue<Output>* output_queue)
    : matching_options_(matching_options),
      geometry_options_(geometry_options),
      cache_(cache),
      input_queue_(input_queue),
      output_queue_(output_queue) {
  THROW_CHECK(matching_options_.Check());

  if (matching_options_.RequiresOpenGL()) {
    opengl_context_ = std::make_unique<OpenGLContextManager>();
  }
}

void FeatureMatcherWorker::Run() {
  if (opengl_context_ != nullptr) {
    THROW_CHECK(opengl_context_->MakeCurrent());
  }

#if defined(COLMAP_CUDA_ENABLED)
  if (matching_options_.use_gpu) {
    // Initialize CUDA device for this worker thread
    const std::vector<int> gpu_indices =
        CSVToVector<int>(matching_options_.gpu_index);
    THROW_CHECK_EQ(gpu_indices.size(), 1)
        << "Each matching worker can only use one GPU";
    const int gpu_index = gpu_indices[0];

    if (gpu_index >= 0) {
      SetBestCudaDevice(gpu_index);
      LOG(INFO) << "Bind FeatureMatcherWorker to GPU device " << gpu_index;
    }
  }
#endif

  if (matching_options_.type == FeatureMatcherType::SIFT_BRUTEFORCE) {
    // TODO(jsch): This is a bit ugly, but currently cannot think of a better
    // way to inject the shared descriptor index cache.
    THROW_CHECK_NOTNULL(matching_options_.sift)->cpu_descriptor_index_cache =
        &cache_->GetFeatureDescriptorIndexCache();
    THROW_CHECK_NOTNULL(matching_options_.sift->cpu_descriptor_index_cache);
  }

  // Minimize the amount of allocated GPU memory by computing the maximum number
  // of descriptors for any image over the whole database.
  matching_options_.max_num_matches = std::min<int>(
      matching_options_.max_num_matches, cache_->MaxNumKeypoints());

  std::unique_ptr<FeatureMatcher> matcher =
      FeatureMatcher::Create(matching_options_);
  if (matcher == nullptr) {
    LOG(ERROR) << "Failed to create feature matcher.";
    SignalInvalidSetup();
    return;
  }

  SignalValidSetup();

  while (true) {
    if (IsStopped()) {
      break;
    }

    auto input_job = input_queue_->Pop();
    if (input_job.IsValid()) {
      auto& data = input_job.Data();

      if (!cache_->ExistsDescriptors(data.image_id1) ||
          !cache_->ExistsDescriptors(data.image_id2)) {
        THROW_CHECK(output_queue_->Push(std::move(data)));
        continue;
      }

      const auto& camera1 =
          cache_->GetCamera(cache_->GetImage(data.image_id1).CameraId());
      const auto& camera2 =
          cache_->GetCamera(cache_->GetImage(data.image_id2).CameraId());

      if (matching_options_.guided_matching) {
        matcher->MatchGuided(
            geometry_options_.ransac_options.max_error,
            {
                data.image_id1,
                &camera1,
                cache_->GetKeypoints(data.image_id1),
                cache_->GetDescriptors(data.image_id1),
                cache_->FindImagePosePriorOrNull(data.image_id1),
            },
            {
                data.image_id2,
                &camera2,
                cache_->GetKeypoints(data.image_id2),
                cache_->GetDescriptors(data.image_id2),
                cache_->FindImagePosePriorOrNull(data.image_id2),
            },
            &data.two_view_geometry);
      } else {
        matcher->Match(
            {
                data.image_id1,
                &camera1,
                cache_->GetKeypoints(data.image_id1),
                cache_->GetDescriptors(data.image_id1),
                cache_->FindImagePosePriorOrNull(data.image_id1),
            },
            {
                data.image_id2,
                &camera2,
                cache_->GetKeypoints(data.image_id2),
                cache_->GetDescriptors(data.image_id2),
                cache_->FindImagePosePriorOrNull(data.image_id2),
            },
            &data.matches);
      }

      THROW_CHECK(output_queue_->Push(std::move(data)));
    }
  }
}

namespace {

class VerifierWorker : public Thread {
 public:
  using Input = FeatureMatcherData;
  using Output = FeatureMatcherData;

  VerifierWorker(const TwoViewGeometryOptions& options,
                 std::shared_ptr<FeatureMatcherCache> cache,
                 JobQueue<Input>* input_queue,
                 JobQueue<Output>* output_queue,
                 const bool use_existing_relative_pose = false)
      : options_(options),
        cache_(std::move(cache)),
        use_existing_relative_pose_(use_existing_relative_pose),
        input_queue_(input_queue),
        output_queue_(output_queue) {
    THROW_CHECK(options_.Check());
  }

 protected:
  void Run() override {
    while (true) {
      if (IsStopped()) {
        break;
      }

      auto input_job = input_queue_->Pop();
      if (input_job.IsValid()) {
        auto& data = input_job.Data();

        if (data.matches.size() <
            static_cast<size_t>(options_.min_num_inliers)) {
          THROW_CHECK(output_queue_->Push(std::move(data)));
          continue;
        }

        const auto& camera1 =
            cache_->GetCamera(cache_->GetImage(data.image_id1).CameraId());
        const auto& camera2 =
            cache_->GetCamera(cache_->GetImage(data.image_id2).CameraId());
        const auto keypoints1 = cache_->GetKeypoints(data.image_id1);
        const auto keypoints2 = cache_->GetKeypoints(data.image_id2);
        const std::vector<Eigen::Vector2d> points1 =
            FeatureKeypointsToPointsVector(*keypoints1);
        const std::vector<Eigen::Vector2d> points2 =
            FeatureKeypointsToPointsVector(*keypoints2);
        const TwoViewGeometry prior_two_view_geometry = data.two_view_geometry;

        if (use_existing_relative_pose_ &&
            data.two_view_geometry.cam2_from_cam1.has_value()) {
          data.two_view_geometry = TwoViewGeometryFromKnownRelativePose(
              camera1,
              points1,
              camera2,
              points2,
              *data.two_view_geometry.cam2_from_cam1,
              data.matches,
              options_.min_num_inliers,
              options_.ransac_options.max_error);
        } else {
          data.two_view_geometry = EstimateTwoViewGeometry(
              camera1, points1, camera2, points2, data.matches, options_);
        }
        PreserveInlierLoopClosureProvenance(prior_two_view_geometry,
                                            &data.two_view_geometry);

        THROW_CHECK(output_queue_->Push(std::move(data)));
      }
    }
  }

 private:
  const TwoViewGeometryOptions options_;
  std::shared_ptr<FeatureMatcherCache> cache_;
  const bool use_existing_relative_pose_;
  JobQueue<Input>* input_queue_;
  JobQueue<Output>* output_queue_;
};

}  // namespace

FeatureMatcherController::FeatureMatcherController(
    const FeatureMatchingOptions& matching_options,
    const TwoViewGeometryOptions& geometry_options,
    std::shared_ptr<FeatureMatcherCache> cache)
    : matching_options_(matching_options),
      geometry_options_(geometry_options),
      cache_(std::move(cache)),
      is_setup_(false) {
  THROW_CHECK(matching_options_.Check());
  THROW_CHECK(geometry_options_.Check());
  THROW_CHECK_EQ(geometry_options_.ransac_options.num_threads, 1)
      << "Parallel RANSAC is not supported inside multi-threaded matching";

  const int num_threads = GetEffectiveNumThreads(matching_options_.num_threads);
  THROW_CHECK_GT(num_threads, 0);

  std::vector<int> gpu_indices = CSVToVector<int>(matching_options_.gpu_index);
  THROW_CHECK_GT(gpu_indices.size(), 0);

#if defined(COLMAP_CUDA_ENABLED)
  if (matching_options_.use_gpu && gpu_indices.size() == 1 &&
      gpu_indices[0] == -1) {
    const int num_cuda_devices = GetNumCudaDevices();
    THROW_CHECK_GT(num_cuda_devices, 0);
    gpu_indices.resize(num_cuda_devices);
    std::iota(gpu_indices.begin(), gpu_indices.end(), 0);
  }
#endif  // COLMAP_CUDA_ENABLED

  // If skip_geometric_verification, match directly to output_queue_.
  const bool skip_geometric_verification =
      matching_options_.skip_geometric_verification &&
      !matching_options_.guided_matching;
  JobQueue<FeatureMatcherData>* matcher_output_queue =
      skip_geometric_verification ? &output_queue_ : &verifier_queue_;

  if (matching_options_.use_gpu) {
    auto worker_matching_options = matching_options_;
    // The first matching is always without guided matching.
    worker_matching_options.guided_matching = false;
    matchers_.reserve(gpu_indices.size());
    for (const auto& gpu_index : gpu_indices) {
      worker_matching_options.gpu_index = std::to_string(gpu_index);
      matchers_.emplace_back(
          std::make_unique<FeatureMatcherWorker>(worker_matching_options,
                                                 geometry_options_,
                                                 cache_,
                                                 &matcher_queue_,
                                                 matcher_output_queue));
    }
  } else {
    auto worker_matching_options = matching_options_;
    // Prevent nested threading.
    worker_matching_options.num_threads = 1;
    // The first matching is always without guided matching.
    worker_matching_options.guided_matching = false;
    matchers_.reserve(num_threads);
    for (int i = 0; i < num_threads; ++i) {
      matchers_.emplace_back(
          std::make_unique<FeatureMatcherWorker>(worker_matching_options,
                                                 geometry_options_,
                                                 cache_,
                                                 &matcher_queue_,
                                                 matcher_output_queue));
    }
  }

  verifiers_.reserve(num_threads);
  if (matching_options_.guided_matching) {
    // Redirect the verification output to final round of guided matching.
    for (int i = 0; i < num_threads; ++i) {
      verifiers_.emplace_back(std::make_unique<VerifierWorker>(
          geometry_options_, cache_, &verifier_queue_, &guided_matcher_queue_));
    }

    if (matching_options_.use_gpu) {
      auto worker_matching_options = matching_options_;
      guided_matchers_.reserve(gpu_indices.size());
      for (const auto& gpu_index : gpu_indices) {
        worker_matching_options.gpu_index = std::to_string(gpu_index);
        guided_matchers_.emplace_back(
            std::make_unique<FeatureMatcherWorker>(worker_matching_options,
                                                   geometry_options_,
                                                   cache_,
                                                   &guided_matcher_queue_,
                                                   &output_queue_));
      }
    } else {
      auto worker_matching_options = matching_options_;
      // Prevent nested threading.
      worker_matching_options.num_threads = 1;
      guided_matchers_.reserve(num_threads);
      for (int i = 0; i < num_threads; ++i) {
        guided_matchers_.emplace_back(
            std::make_unique<FeatureMatcherWorker>(worker_matching_options,
                                                   geometry_options_,
                                                   cache_,
                                                   &guided_matcher_queue_,
                                                   &output_queue_));
      }
    }
  } else if (!matching_options.skip_geometric_verification) {
    for (int i = 0; i < num_threads; ++i) {
      verifiers_.emplace_back(std::make_unique<VerifierWorker>(
          geometry_options_, cache_, &verifier_queue_, &output_queue_));
    }
  }
}

FeatureMatcherController::~FeatureMatcherController() {
  matcher_queue_.Wait();
  verifier_queue_.Wait();
  guided_matcher_queue_.Wait();
  output_queue_.Wait();

  for (auto& matcher : matchers_) {
    matcher->Stop();
  }

  for (auto& verifier : verifiers_) {
    verifier->Stop();
  }

  for (auto& guided_matcher : guided_matchers_) {
    guided_matcher->Stop();
  }

  matcher_queue_.Stop();
  verifier_queue_.Stop();
  guided_matcher_queue_.Stop();
  output_queue_.Stop();

  for (auto& matcher : matchers_) {
    matcher->Wait();
  }

  for (auto& verifier : verifiers_) {
    verifier->Wait();
  }

  for (auto& guided_matcher : guided_matchers_) {
    guided_matcher->Wait();
  }
}

bool FeatureMatcherController::Setup() {
  for (auto& matcher : matchers_) {
    matcher->Start();
  }

  for (auto& verifier : verifiers_) {
    verifier->Start();
  }

  for (auto& guided_matcher : guided_matchers_) {
    guided_matcher->Start();
  }

  for (auto& matcher : matchers_) {
    if (!matcher->CheckValidSetup()) {
      return false;
    }
  }

  for (auto& guided_matcher : guided_matchers_) {
    if (!guided_matcher->CheckValidSetup()) {
      return false;
    }
  }

  is_setup_ = true;

  return true;
}

void FeatureMatcherController::Match(
    const std::vector<std::pair<image_t, image_t>>& image_pairs,
    bool mark_as_loop_closure) {
  MatchWithProvenance(
      FeatureMatcherImagePairs(image_pairs, mark_as_loop_closure));
}

void FeatureMatcherController::MatchWithProvenance(
    const std::vector<FeatureMatcherImagePair>& image_pairs) {
  THROW_CHECK_NOTNULL(cache_);
  THROW_CHECK(is_setup_);

  if (image_pairs.empty()) {
    return;
  }

  //////////////////////////////////////////////////////////////////////////////
  // Match the image pairs
  //////////////////////////////////////////////////////////////////////////////

  std::vector<FeatureMatcherImagePair> unique_image_pairs;
  unique_image_pairs.reserve(image_pairs.size());
  std::unordered_map<image_pair_t, size_t> image_pair_id_to_idx;
  image_pair_id_to_idx.reserve(image_pairs.size());

  for (const auto& image_pair : image_pairs) {
    const image_t image_id1 = image_pair.image_id1;
    const image_t image_id2 = image_pair.image_id2;
    // Avoid self-matches.
    if (image_id1 == image_id2) {
      continue;
    }

    // Avoid self-matches within a frame.
    if (matching_options_.skip_image_pairs_in_same_frame) {
      const Image& image1 = cache_->GetImage(image_id1);
      const Image& image2 = cache_->GetImage(image_id2);
      if (image1.HasFrameId() && image2.HasFrameId() &&
          image1.FrameId() == image2.FrameId()) {
        continue;
      }
    }

    // Avoid duplicate image pairs.
    const image_pair_t pair_id = ImagePairToPairId(image_id1, image_id2);
    const auto [it, inserted] =
        image_pair_id_to_idx.emplace(pair_id, unique_image_pairs.size());
    if (!inserted) {
      unique_image_pairs[it->second].is_loop_closure =
          unique_image_pairs[it->second].is_loop_closure ||
          image_pair.is_loop_closure;
      continue;
    }
    unique_image_pairs.push_back(image_pair);
  }

  size_t num_outputs = 0;
  for (const auto& image_pair : unique_image_pairs) {
    const image_t image_id1 = image_pair.image_id1;
    const image_t image_id2 = image_pair.image_id2;
    const bool exists_matches = cache_->ExistsMatches(image_id1, image_id2);
    const bool exists_two_view_geometry =
        cache_->ExistsTwoViewGeometry(image_id1, image_id2);

    const bool existing_geometry_has_lc =
        exists_two_view_geometry &&
        HasLoopClosureInliers(cache_->GetTwoViewGeometry(image_id1, image_id2));
    const bool existing_geometry_is_direct_tracking =
        exists_two_view_geometry && image_pair.is_loop_closure &&
        !existing_geometry_has_lc &&
        IsDirectSequentialPair(*cache_, image_id1, image_id2);
    const bool should_skip_existing_geometry =
        image_pair.is_loop_closure
            ? (existing_geometry_has_lc || existing_geometry_is_direct_tracking)
            : !existing_geometry_has_lc;
    if (exists_matches && exists_two_view_geometry &&
        should_skip_existing_geometry) {
      continue;
    }

    num_outputs += 1;

    // If only one of the matches or inlier matches exist, we recompute them
    // from scratch and delete the existing results. This must be done before
    // pushing the jobs to the queue, otherwise database constraints might fail
    // when writing an existing result into the database.
    if (exists_two_view_geometry) {
      cache_->DeleteTwoViewGeometry(image_id1, image_id2);
    }

    FeatureMatcherData data;
    data.image_id1 = image_id1;
    data.image_id2 = image_id2;
    data.is_loop_closure = image_pair.is_loop_closure;

    if (exists_matches) {
      data.matches = cache_->GetMatches(image_id1, image_id2);
      cache_->DeleteMatches(image_id1, image_id2);
      THROW_CHECK(verifier_queue_.Push(std::move(data)));
    } else {
      THROW_CHECK(matcher_queue_.Push(std::move(data)));
    }
  }

  //////////////////////////////////////////////////////////////////////////////
  // Write results to database
  //////////////////////////////////////////////////////////////////////////////

  if (num_outputs == 0) {
    THROW_CHECK_EQ(output_queue_.Size(), 0);
    return;
  }

  AdjacentTransitiveMatchExtractor transitive_match_extractor(cache_);
  for (size_t i = 0; i < num_outputs; ++i) {
    auto output_job = output_queue_.Pop();
    THROW_CHECK(output_job.IsValid());
    auto& output = output_job.Data();

    if (output.matches.size() <
        static_cast<size_t>(geometry_options_.min_num_inliers)) {
      output.matches = {};
    }

    if (output.two_view_geometry.inlier_matches.size() <
        static_cast<size_t>(geometry_options_.min_num_inliers)) {
      output.two_view_geometry = TwoViewGeometry();
    }

    FinalizeLoopClosureProvenance(transitive_match_extractor, &output);
    cache_->WriteMatches(output.image_id1, output.image_id2, output.matches);
    cache_->WriteTwoViewGeometry(
        output.image_id1, output.image_id2, output.two_view_geometry);
    transitive_match_extractor.AddFinalizedOutput(output);
  }

  THROW_CHECK_EQ(output_queue_.Size(), 0);
}

GeometricVerifierController::GeometricVerifierController(
    const GeometricVerifierOptions& options,
    const TwoViewGeometryOptions& geometry_options,
    std::shared_ptr<FeatureMatcherCache> cache)
    : geometry_options_(geometry_options),
      cache_(std::move(cache)),
      options_(options),
      is_setup_(false) {
  THROW_CHECK(geometry_options_.Check());

  const int num_threads = GetEffectiveNumThreads(options_.num_threads);

  // Run geometric verification
  for (int i = 0; i < num_threads; ++i) {
    verifiers_.emplace_back(
        std::make_unique<VerifierWorker>(geometry_options_,
                                         cache_,
                                         &verifier_queue_,
                                         &output_queue_,
                                         options_.use_existing_relative_pose));
  }
}

GeometricVerifierController::~GeometricVerifierController() {
  verifier_queue_.Wait();
  output_queue_.Wait();

  for (auto& verifier : verifiers_) {
    verifier->Stop();
  }

  verifier_queue_.Stop();
  output_queue_.Stop();

  for (auto& verifier : verifiers_) {
    verifier->Wait();
  }
}

const GeometricVerifierOptions& GeometricVerifierController::Options() const {
  return options_;
}

GeometricVerifierOptions& GeometricVerifierController::Options() {
  return options_;
}

bool GeometricVerifierController::Setup() {
  for (auto& verifier : verifiers_) {
    verifier->Start();
  }

  is_setup_ = true;
  return true;
}

void GeometricVerifierController::Verify(
    const std::vector<std::pair<image_t, image_t>>& image_pairs,
    bool mark_as_loop_closure) {
  VerifyWithProvenance(
      FeatureMatcherImagePairs(image_pairs, mark_as_loop_closure));
}

void GeometricVerifierController::VerifyWithProvenance(
    const std::vector<FeatureMatcherImagePair>& image_pairs) {
  THROW_CHECK_NOTNULL(cache_);
  THROW_CHECK(is_setup_);

  if (image_pairs.empty()) {
    return;
  }

  //////////////////////////////////////////////////////////////////////////////
  // Verify the matches from the image pairs
  //////////////////////////////////////////////////////////////////////////////

  std::vector<FeatureMatcherImagePair> unique_image_pairs;
  unique_image_pairs.reserve(image_pairs.size());
  std::unordered_map<image_pair_t, size_t> image_pair_id_to_idx;
  image_pair_id_to_idx.reserve(image_pairs.size());

  for (const auto& image_pair : image_pairs) {
    const image_t image_id1 = image_pair.image_id1;
    const image_t image_id2 = image_pair.image_id2;
    // Avoid self-matches.
    if (image_id1 == image_id2) {
      continue;
    }

    // Avoid duplicate image pairs.
    const image_pair_t pair_id = ImagePairToPairId(image_id1, image_id2);
    const auto [it, inserted] =
        image_pair_id_to_idx.emplace(pair_id, unique_image_pairs.size());
    if (!inserted) {
      unique_image_pairs[it->second].is_loop_closure =
          unique_image_pairs[it->second].is_loop_closure ||
          image_pair.is_loop_closure;
      continue;
    }
    unique_image_pairs.push_back(image_pair);
  }

  size_t num_outputs = 0;
  for (const auto& image_pair : unique_image_pairs) {
    const image_t image_id1 = image_pair.image_id1;
    const image_t image_id2 = image_pair.image_id2;
    const bool exists_matches = cache_->ExistsMatches(image_id1, image_id2);
    const bool exists_inlier_matches =
        cache_->ExistsInlierMatches(image_id1, image_id2);

    const bool existing_geometry_has_lc =
        exists_inlier_matches &&
        HasLoopClosureInliers(cache_->GetTwoViewGeometry(image_id1, image_id2));
    const bool existing_geometry_is_direct_tracking =
        exists_inlier_matches && image_pair.is_loop_closure &&
        !existing_geometry_has_lc &&
        IsDirectSequentialPair(*cache_, image_id1, image_id2);
    const bool should_skip_existing_geometry =
        image_pair.is_loop_closure
            ? (existing_geometry_has_lc || existing_geometry_is_direct_tracking)
            : !existing_geometry_has_lc;
    if (exists_matches && exists_inlier_matches &&
        should_skip_existing_geometry) {
      continue;
    }
    const bool preserve_prior_geometry =
        exists_inlier_matches &&
        (image_pair.is_loop_closure || existing_geometry_has_lc);
    const TwoViewGeometry prior_two_view_geometry =
        preserve_prior_geometry
            ? cache_->GetTwoViewGeometry(image_id1, image_id2)
            : TwoViewGeometry();

    // If only one of the matches or inlier matches exist, we recompute them
    // from scratch and delete the existing results. This must be done before
    // pushing the jobs to the queue, otherwise database constraints might fail
    // when writing an existing result into the database.
    if (exists_inlier_matches) {
      cache_->DeleteTwoViewGeometry(image_id1, image_id2);
    }

    FeatureMatcherData data;
    data.image_id1 = image_id1;
    data.image_id2 = image_id2;
    data.is_loop_closure = image_pair.is_loop_closure;

    if (exists_matches) {
      num_outputs += 1;
      data.matches = cache_->GetMatches(image_id1, image_id2);
      // There exists a two view geometry without inlier matches.
      if (preserve_prior_geometry) {
        data.two_view_geometry = prior_two_view_geometry;
      } else if (cache_->ExistsTwoViewGeometry(image_id1, image_id2)) {
        data.two_view_geometry =
            cache_->GetTwoViewGeometry(image_id1, image_id2);
        data.is_loop_closure =
            data.is_loop_closure || data.two_view_geometry.is_loop_closure;
      }
      THROW_CHECK(verifier_queue_.Push(std::move(data)));
    }
  }

  //////////////////////////////////////////////////////////////////////////////
  // Write results to database
  //////////////////////////////////////////////////////////////////////////////

  if (num_outputs == 0) {
    THROW_CHECK_EQ(output_queue_.Size(), 0);
    return;
  }

  AdjacentTransitiveMatchExtractor transitive_match_extractor(cache_);
  for (size_t i = 0; i < num_outputs; ++i) {
    auto output_job = output_queue_.Pop();
    THROW_CHECK(output_job.IsValid());
    auto& output = output_job.Data();

    if (output.matches.size() <
        static_cast<size_t>(geometry_options_.min_num_inliers)) {
      output.matches = {};
    }

    if (output.two_view_geometry.inlier_matches.size() <
        static_cast<size_t>(geometry_options_.min_num_inliers)) {
      output.two_view_geometry = TwoViewGeometry();
    }

    if (cache_->ExistsTwoViewGeometry(output.image_id1, output.image_id2)) {
      cache_->DeleteTwoViewGeometry(output.image_id1, output.image_id2);
    }
    FinalizeLoopClosureProvenance(transitive_match_extractor, &output);
    cache_->WriteTwoViewGeometry(
        output.image_id1, output.image_id2, output.two_view_geometry);
    transitive_match_extractor.AddFinalizedOutput(output);
  }

  THROW_CHECK_EQ(output_queue_.Size(), 0);
}

}  // namespace colmap
