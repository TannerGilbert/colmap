#include "colmap/sfm/view_graph_manipulation.h"

#include "colmap/estimators/two_view_geometry.h"
#include "colmap/geometry/essential_matrix.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/util/logging.h"
#include "colmap/util/threading.h"

namespace colmap {
namespace {

constexpr double kTranslationNormEps = 1e-12;

}  // namespace

void UpdateImagePairsConfig(CorrespondenceGraph& view_graph,
                            const Reconstruction& rec) {
  // For each camera, count (total_pairs_involved, calibrated_pairs_involved)
  // restricted to pairs where both cameras have prior focal lengths.
  std::unordered_map<camera_t, std::pair<int, int>> camera_counter;
  for (auto& [pair_id, image_pair] : view_graph.MutableImagePairs()) {
    if (!image_pair.is_valid) continue;
    const camera_t cid1 = rec.Image(image_pair.image_id1).CameraId();
    const camera_t cid2 = rec.Image(image_pair.image_id2).CameraId();
    const Camera& c1 = rec.Camera(cid1);
    const Camera& c2 = rec.Camera(cid2);
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

    const camera_t cid1 = rec.Image(image_pair.image_id1).CameraId();
    const camera_t cid2 = rec.Image(image_pair.image_id2).CameraId();
    if (!camera_validity[cid1] || !camera_validity[cid2]) continue;

    // Skip pairs whose decomposition never yielded a relative pose (e.g.
    // F-only path or RANSAC failure). Without cam2_from_cam1 we cannot
    // recompute F from intrinsics; leaving the pair UNCALIBRATED lets
    // downstream filters drop it instead of crashing here.
    if (!tvg.cam2_from_cam1.has_value()) continue;

    tvg.config = TwoViewGeometry::CALIBRATED;
    const Camera& c1 = rec.Camera(cid1);
    const Camera& c2 = rec.Camera(cid2);
    tvg.F = FundamentalFromEssentialMatrix(
        c2.CalibrationMatrix(),
        EssentialMatrixFromPose(*tvg.cam2_from_cam1),
        c1.CalibrationMatrix());
  }
}

void DecomposeRelPose(CorrespondenceGraph& view_graph,
                      const Reconstruction& rec) {
  // Collect pairs to decompose: valid + both cameras have prior focal length.
  std::vector<image_pair_t> pair_ids;
  for (auto& [pair_id, image_pair] : view_graph.MutableImagePairs()) {
    if (!image_pair.is_valid) continue;
    const camera_t cid1 = rec.Image(image_pair.image_id1).CameraId();
    const camera_t cid2 = rec.Image(image_pair.image_id2).CameraId();
    if (!rec.Camera(cid1).has_prior_focal_length ||
        !rec.Camera(cid2).has_prior_focal_length)
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
      const camera_t cid1 = rec.Image(iid1).CameraId();
      const camera_t cid2 = rec.Image(iid2).CameraId();
      const Camera& c1 = rec.Camera(cid1);
      const Camera& c2 = rec.Camera(cid2);

      // Snapshot original config before EstimateTwoViewGeometryPose mutates it.
      const int original_config = image_pair.two_view_geometry.config;

      // Estimator mutates two_view_geometry in place: re-fits cam2_from_cam1
      // (and may revise config) using the existing E/F/H from the pair.
      EstimateTwoViewGeometryPose(c1,
                                  rec.Image(iid1).features,
                                  c2,
                                  rec.Image(iid2).features,
                                  &image_pair.two_view_geometry);

      // PLANAR pairs with prior calibration get force-upgraded to CALIBRATED.
      if (original_config == TwoViewGeometry::PLANAR &&
          c1.has_prior_focal_length && c2.has_prior_focal_length) {
        image_pair.two_view_geometry.config = TwoViewGeometry::CALIBRATED;
        return;
      }
      // (Already filtered to prior-calibrated pairs above, so the
      // early-return-without-prior-focal path never fires.)

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
