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

#include "colmap/sfm/view_graph_manipulation.h"

#include "colmap/estimators/two_view_geometry.h"
#include "colmap/geometry/essential_matrix.h"
#include "colmap/util/logging.h"
#include "colmap/util/threading.h"

namespace colmap {
namespace {

constexpr double kTranslationNormEps = 1e-12;

}  // namespace

void UpdateImagePairsConfig(
    CorrespondenceGraph& view_graph,
    const std::unordered_map<camera_t, Camera>& cameras,
    const std::unordered_map<image_t, Image>& images) {
  // For each camera, count (total_pairs_involved, calibrated_pairs_involved)
  // restricted to pairs where both cameras have prior focal lengths.
  std::unordered_map<camera_t, std::pair<int, int>> camera_counter;
  for (auto& [pair_id, image_pair] : view_graph.MutableImagePairs()) {
    if (!image_pair.is_valid) continue;
    const camera_t cid1 = images.at(image_pair.image_id1).CameraId();
    const camera_t cid2 = images.at(image_pair.image_id2).CameraId();
    const Camera& c1 = cameras.at(cid1);
    const Camera& c2 = cameras.at(cid2);
    if (!c1.has_prior_focal_length || !c2.has_prior_focal_length) continue;

    const int cfg = image_pair.two_view_geometry.config;
    if (cfg == TwoViewGeometry::CALIBRATED) {
      camera_counter[cid1].first++;
      camera_counter[cid2].first++;
      camera_counter[cid1].second++;
      camera_counter[cid2].second++;
    } else if (cfg == TwoViewGeometry::UNCALIBRATED) {
      camera_counter[cid1].first++;
      camera_counter[cid2].first++;
    }
  }

  // A camera is considered valid if more than half of its participating
  // (prior-calibrated) pairs are already CALIBRATED.
  std::unordered_map<camera_t, bool> camera_validity;
  for (const auto& [cid, counter] : camera_counter) {
    camera_validity[cid] =
        counter.first > 0 && counter.second * 1.0 / counter.first > 0.5;
  }

  // Upgrade UNCALIBRATED pairs to CALIBRATED for camera-validity-True cameras
  // and recompute F from the existing cam2_from_cam1.
  for (auto& [pair_id, image_pair] : view_graph.MutableImagePairs()) {
    if (!image_pair.is_valid) continue;
    auto& tvg = image_pair.two_view_geometry;
    if (tvg.config != TwoViewGeometry::UNCALIBRATED) continue;

    const camera_t cid1 = images.at(image_pair.image_id1).CameraId();
    const camera_t cid2 = images.at(image_pair.image_id2).CameraId();
    if (!camera_validity[cid1] || !camera_validity[cid2]) continue;

    tvg.config = TwoViewGeometry::CALIBRATED;
    THROW_CHECK(tvg.cam2_from_cam1.has_value())
        << "UNCALIBRATED pair upgraded to CALIBRATED must have cam2_from_cam1";
    const Camera& c1 = cameras.at(cid1);
    const Camera& c2 = cameras.at(cid2);
    tvg.F = FundamentalFromEssentialMatrix(
        c2.CalibrationMatrix(),
        EssentialMatrixFromPose(*tvg.cam2_from_cam1),
        c1.CalibrationMatrix());
  }
}

void DecomposeRelPose(CorrespondenceGraph& view_graph,
                      std::unordered_map<camera_t, Camera>& cameras,
                      std::unordered_map<image_t, Image>& images) {
  // Collect pairs to decompose: valid + both cameras have prior focal length.
  std::vector<image_pair_t> pair_ids;
  for (auto& [pair_id, image_pair] : view_graph.MutableImagePairs()) {
    if (!image_pair.is_valid) continue;
    const camera_t cid1 = images.at(image_pair.image_id1).CameraId();
    const camera_t cid2 = images.at(image_pair.image_id2).CameraId();
    if (!cameras.at(cid1).has_prior_focal_length ||
        !cameras.at(cid2).has_prior_focal_length)
      continue;
    pair_ids.push_back(pair_id);
  }
  LOG(INFO) << "Decompose relative pose for " << pair_ids.size() << " pairs";

  ThreadPool thread_pool(ThreadPool::kMaxNumThreads);
  for (const image_pair_t pid : pair_ids) {
    thread_pool.AddTask([&, pid]() {
      auto& image_pair = view_graph.MutableImagePairs().at(pid);
      const image_t iid1 = image_pair.image_id1;
      const image_t iid2 = image_pair.image_id2;
      const camera_t cid1 = images.at(iid1).CameraId();
      const camera_t cid2 = images.at(iid2).CameraId();
      const Camera& c1 = cameras.at(cid1);
      const Camera& c2 = cameras.at(cid2);

      // Snapshot original config before EstimateTwoViewGeometryPose mutates it.
      const int original_config = image_pair.two_view_geometry.config;

      // Estimator mutates two_view_geometry in place: re-fits cam2_from_cam1
      // (and may revise config) using the existing E/F/H from the pair.
      EstimateTwoViewGeometryPose(c1,
                                  images.at(iid1).features,
                                  c2,
                                  images.at(iid2).features,
                                  &image_pair.two_view_geometry);

      // PLANAR pairs with prior calibration get force-upgraded to CALIBRATED.
      if (original_config == TwoViewGeometry::PLANAR &&
          c1.has_prior_focal_length && c2.has_prior_focal_length) {
        image_pair.two_view_geometry.config = TwoViewGeometry::CALIBRATED;
        return;
      }
      // (Already filtered to prior-calibrated pairs above, so the
      // early-return-without-prior-focal branch never fires.)

      // Normalize translation to unit norm when non-zero.
      auto& tvg = image_pair.two_view_geometry;
      if (tvg.cam2_from_cam1.has_value() &&
          tvg.cam2_from_cam1->translation().norm() > kTranslationNormEps) {
        const Eigen::Vector3d normalized =
            tvg.cam2_from_cam1->translation().normalized();
        tvg.cam2_from_cam1->translation() = normalized;
      }
    });
  }
  thread_pool.Wait();

  size_t pure_rotation = 0;
  for (const image_pair_t pid : pair_ids) {
    const auto& tvg = view_graph.MutableImagePairs().at(pid).two_view_geometry;
    if (tvg.config != TwoViewGeometry::CALIBRATED &&
        tvg.config != TwoViewGeometry::PLANAR_OR_PANORAMIC)
      pure_rotation++;
  }
  LOG(INFO) << "Decompose relative pose done. " << pure_rotation
            << " pairs are pure rotation";
}

void FilterPairsByInlierNum(CorrespondenceGraph& view_graph,
                            int min_inlier_num) {
  size_t num_invalid = 0;
  for (auto& [pair_id, image_pair] : view_graph.MutableImagePairs()) {
    if (!image_pair.is_valid) continue;
    if (static_cast<int>(image_pair.inliers.size()) < min_inlier_num) {
      image_pair.is_valid = false;
      ++num_invalid;
    }
  }
  LOG(INFO) << "Filtered " << num_invalid
            << " relative poses with inlier number < " << min_inlier_num;
}

void FilterPairsByInlierRatio(CorrespondenceGraph& view_graph,
                              double min_inlier_ratio) {
  size_t num_invalid = 0;
  for (auto& [pair_id, image_pair] : view_graph.MutableImagePairs()) {
    if (!image_pair.is_valid) continue;
    const auto num_matches = image_pair.matches.rows();
    if (num_matches <= 0) continue;
    const double ratio =
        static_cast<double>(image_pair.inliers.size()) / num_matches;
    if (ratio < min_inlier_ratio) {
      image_pair.is_valid = false;
      ++num_invalid;
    }
  }
  LOG(INFO) << "Filtered " << num_invalid
            << " relative poses with inlier ratio < " << min_inlier_ratio;
}

}  // namespace colmap
