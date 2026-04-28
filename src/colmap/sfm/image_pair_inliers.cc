#include "colmap/sfm/image_pair_inliers.h"

#include "colmap/geometry/essential_matrix.h"
#include "colmap/geometry/homography_matrix.h"
#include "colmap/scene/reconstruction.h"

namespace colmap {
namespace {
constexpr double EPS = 1e-12;

// PoseLib-style cheirality with z-distance bounds (unbatched).
bool CheckCheirality(const Rigid3d& pose,
                     const Eigen::Vector3d& x1,
                     const Eigen::Vector3d& x2,
                     double min_z,
                     double max_z) {
  // This code assumes that x1 and x2 are unit vectors.
  const Eigen::Vector3d Rx1 = pose.rotation() * x1;
  const double a = -Rx1.dot(x2);
  const double b1 = -Rx1.dot(pose.translation());
  const double b2 = x2.dot(pose.translation());
  // Note: we drop the factor 1.0/(1-a*a) since it is always positive.
  const double lambda1 = b1 - a * b2;
  const double lambda2 = -a * b1 + b2;
  min_z = min_z * (1 - a * a);
  max_z = max_z * (1 - a * a);
  bool status = lambda1 > min_z && lambda2 > min_z;
  status = status && (lambda1 < max_z) && (lambda2 < max_z);
  return status;
}

// F-cheirality orientation signum (no native counterpart).
// Code from GC-RANSAC by Daniel Barath.
double GetOrientationSignum(const Eigen::Matrix3d& F,
                            const Eigen::Vector3d& epipole,
                            const Eigen::Vector2d& pt1,
                            const Eigen::Vector2d& pt2) {
  double signum1 = F(0, 0) * pt2[0] + F(1, 0) * pt2[1] + F(2, 0);
  double signum2 = epipole(1) - epipole(2) * pt1[1];
  return signum1 * signum2;
}

// Sampson error on Vec3 bearing rays (divides by z+EPS).
double SampsonError(const Eigen::Matrix3d& E,
                    const Eigen::Vector3d& x1,
                    const Eigen::Vector3d& x2) {
  Eigen::Vector3d Ex1 = E * x1 / (EPS + x1[2]);
  Eigen::Vector3d Etx2 = E.transpose() * x2 / (EPS + x2[2]);
  double C = Ex1.dot(x2);
  double Cx = Ex1.head(2).squaredNorm();
  double Cy = Etx2.head(2).squaredNorm();
  return C * C / (Cx + Cy);
}

class ImagePairInliers {
 public:
  ImagePairInliers(CorrespondenceGraph::ImagePair& image_pair,
                   const Reconstruction& rec,
                   const InlierThresholdOptions& options)
      : image_pair(image_pair), rec(rec), options(options) {}

  // Score the pair via Sampson + cheirality + degeneracy gates. Stores
  // surviving match indices in ``image_pair.inliers``.
  void ScoreError();

 protected:
  void ScoreErrorEssential();
  void ScoreErrorFundamental();
  void ScoreErrorHomography();

  CorrespondenceGraph::ImagePair& image_pair;
  const Reconstruction& rec;
  const InlierThresholdOptions& options;
};

void ImagePairInliers::ScoreError() {
  if (image_pair.two_view_geometry.config == TwoViewGeometry::PLANAR ||
      image_pair.two_view_geometry.config == TwoViewGeometry::PANORAMIC ||
      image_pair.two_view_geometry.config ==
          TwoViewGeometry::PLANAR_OR_PANORAMIC) {
    ScoreErrorHomography();
  } else if (image_pair.two_view_geometry.config ==
             TwoViewGeometry::UNCALIBRATED) {
    ScoreErrorFundamental();
  } else if (image_pair.two_view_geometry.config ==
             TwoViewGeometry::CALIBRATED) {
    ScoreErrorEssential();
  }
}

void ImagePairInliers::ScoreErrorEssential() {
  const Rigid3d& cam2_from_cam1 =
      *image_pair.two_view_geometry.cam2_from_cam1;
  const Eigen::Matrix3d E = EssentialMatrixFromPose(cam2_from_cam1);

  // eij = camera i on image j
  Eigen::Vector3d epipole12 = cam2_from_cam1.translation();
  Eigen::Vector3d epipole21 = Inverse(cam2_from_cam1).translation();

  if (epipole12.norm() > EPS) epipole12.normalize();
  if (epipole21.norm() > EPS) epipole21.normalize();

  if (epipole12[2] < 0) epipole12 = -epipole12;
  if (epipole21[2] < 0) epipole21 = -epipole21;

  if (image_pair.inliers.size() > 0) {
    image_pair.inliers.clear();
  }

  const image_t image_id1 = image_pair.image_id1;
  const image_t image_id2 = image_pair.image_id2;

  // Convert the threshold from pixel space to normalized space.
  const double thres =
      options.max_epipolar_error_E * 0.5 *
      (1. / rec.Camera(rec.Image(image_id1).CameraId()).MeanFocalLength() +
       1. / rec.Camera(rec.Image(image_id2).CameraId()).MeanFocalLength());
  const double sq_threshold = thres * thres;

  double thres_epipole = std::cos(DegToRad(options.min_angle_from_epipole));
  double thres_angle = 1;
  thres_angle += 1e-6;
  thres_epipole += 1e-6;

  const size_t total_matches = image_pair.matches.rows();
  for (size_t k = 0; k < total_matches; ++k) {
    // Use the undistorted features.
    const Eigen::Vector3d pt1 =
        rec.Image(image_id1).features_undist[image_pair.matches(k, 0)];
    const Eigen::Vector3d pt2 =
        rec.Image(image_id2).features_undist[image_pair.matches(k, 1)];
    const double r2 = SampsonError(E, pt1, pt2);

    if (r2 >= sq_threshold) continue;

    // Cheirality.
    if (!CheckCheirality(cam2_from_cam1, pt1, pt2, 1e-2, 100.)) continue;

    // Angle gate (placeholder; currently disabled).
    const double diff_angle = pt1.dot(cam2_from_cam1.rotation().inverse() * pt2);
    if (diff_angle >= thres_angle) continue;

    // Epipole gate.
    const double diff_epipole1 = pt1.dot(epipole21);
    const double diff_epipole2 = pt2.dot(epipole12);
    if (diff_epipole1 >= thres_epipole || diff_epipole2 >= thres_epipole) {
      continue;
    }

    image_pair.inliers.push_back(k);
  }
}

void ImagePairInliers::ScoreErrorFundamental() {
  if (image_pair.inliers.size() > 0) {
    image_pair.inliers.clear();
  }

  Eigen::Vector3d epipole = (*image_pair.two_view_geometry.F)
                                .row(0)
                                .cross((*image_pair.two_view_geometry.F).row(2));
  bool status = false;
  for (auto i = 0; i < 3; i++) {
    if ((epipole(i) > EPS) || (epipole(i) < -EPS)) {
      status = true;
      break;
    }
  }
  if (!status) {
    epipole = (*image_pair.two_view_geometry.F)
                  .row(1)
                  .cross((*image_pair.two_view_geometry.F).row(2));
  }

  std::vector<double> signums;
  int positive_count = 0;
  int negative_count = 0;

  const image_t image_id1 = image_pair.image_id1;
  const image_t image_id2 = image_pair.image_id2;

  const double thres = options.max_epipolar_error_F;
  const double sq_threshold = thres * thres;

  const size_t total_matches = image_pair.matches.rows();
  std::vector<int> inliers_pre;
  for (size_t k = 0; k < total_matches; ++k) {
    const Eigen::Vector2d pt1 =
        rec.Image(image_id1).features[image_pair.matches(k, 0)];
    const Eigen::Vector2d pt2 =
        rec.Image(image_id2).features[image_pair.matches(k, 1)];
    const double r2 =
        ComputeSquaredSampsonError(pt1.homogeneous(),
                                   pt2.homogeneous(),
                                   *image_pair.two_view_geometry.F);

    if (r2 >= sq_threshold) continue;

    signums.push_back(GetOrientationSignum(
        *image_pair.two_view_geometry.F, epipole, pt1, pt2));
    if (signums.back() > 0) {
      positive_count++;
    } else {
      negative_count++;
    }
    inliers_pre.push_back(k);
  }

  // If we cannot distinguish the signum, the pair is invalid.
  if (positive_count == negative_count) return;
  const bool is_positive = (positive_count > negative_count);

  for (size_t k = 0; k < inliers_pre.size(); k++) {
    if ((signums[k] > 0) == is_positive) {
      image_pair.inliers.push_back(inliers_pre[k]);
    }
  }
}

void ImagePairInliers::ScoreErrorHomography() {
  if (image_pair.inliers.size() > 0) {
    image_pair.inliers.clear();
  }

  const image_t image_id1 = image_pair.image_id1;
  const image_t image_id2 = image_pair.image_id2;

  const double thres = options.max_epipolar_error_H;
  const double sq_threshold = thres * thres;
  const size_t total_matches = image_pair.matches.rows();
  for (size_t k = 0; k < total_matches; ++k) {
    const Eigen::Vector2d pt1 =
        rec.Image(image_id1).features[image_pair.matches(k, 0)];
    const Eigen::Vector2d pt2 =
        rec.Image(image_id2).features[image_pair.matches(k, 1)];
    const double r2 = ComputeSquaredHomographyError(
        pt1, pt2, *image_pair.two_view_geometry.H);

    // TODO: cheirality check for homography. Is that a thing?
    if (r2 < sq_threshold) {
      image_pair.inliers.push_back(k);
    }
  }
}

}  // namespace

void ImagePairsInlierCount(CorrespondenceGraph& correspondence_graph,
                           const Reconstruction& rec,
                           const InlierThresholdOptions& options,
                           bool clean_inliers) {
  for (auto& [pair_id, image_pair] :
       correspondence_graph.MutableImagePairs()) {
    if (!clean_inliers && image_pair.inliers.size() > 0) continue;
    image_pair.inliers.clear();

    if (image_pair.is_valid == false) continue;
    ImagePairInliers inlier_finder(image_pair, rec, options);
    inlier_finder.ScoreError();
  }
}

}  // namespace colmap
