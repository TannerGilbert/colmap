// Copyright (c), ETH Zurich and UNC Chapel Hill.
// All rights reserved.
// (BSD-3-Clause license, see LICENSE)

#pragma once

#include "colmap/estimators/cost_functions/utils.h"
#include "colmap/geometry/rigid3.h"
#include "colmap/sensor/models.h"

namespace colmap {

// Weighted reprojection error with Mahalanobis norm.
// Uses pre-computed Cholesky factors L of precision matrix: P = L @ L^T.
// Residual: [L00*rx, L10*rx + L11*ry] where (rx,ry) = projected - observed.
// Params: point3D[3], cam_from_world[7], camera_params[N]
template <typename CameraModel>
class WeightedReprojErrorCostFunctor
    : public AutoDiffCostFunctor<WeightedReprojErrorCostFunctor<CameraModel>,
                                 2,
                                 3,
                                 7,
                                 CameraModel::num_params> {
 public:
  WeightedReprojErrorCostFunctor(const Eigen::Vector2d& point2D,
                                 double L00,
                                 double L10,
                                 double L11)
      : point2D_(point2D), L00_(L00), L10_(L10), L11_(L11) {}

  template <typename T>
  bool operator()(const T* const point3D_in_world,
                  const T* const cam_from_world,
                  const T* const camera_params,
                  T* residuals) const {
    const Eigen::Matrix<T, 3, 1> point3D_in_cam =
        EigenQuaternionMap<T>(cam_from_world) *
            EigenVector3Map<T>(point3D_in_world) +
        EigenVector3Map<T>(cam_from_world + 4);

    T proj_x, proj_y;
    if (CameraModel::ImgFromCam(camera_params,
                                point3D_in_cam[0],
                                point3D_in_cam[1],
                                point3D_in_cam[2],
                                &proj_x,
                                &proj_y)) {
      T rx = proj_x - T(point2D_(0));
      T ry = proj_y - T(point2D_(1));
      residuals[0] = T(L00_) * rx;
      residuals[1] = T(L10_) * rx + T(L11_) * ry;
    } else {
      residuals[0] = T(0);
      residuals[1] = T(0);
    }
    return true;
  }

 private:
  const Eigen::Vector2d point2D_;
  const double L00_, L10_, L11_;
};

// Constant-pose variant: bakes in a fixed camera pose.
// Params: point3D[3], camera_params[N]
template <typename CameraModel>
class WeightedReprojErrorConstantPoseCostFunctor
    : public AutoDiffCostFunctor<
          WeightedReprojErrorConstantPoseCostFunctor<CameraModel>,
          2,
          3,
          CameraModel::num_params> {
 public:
  WeightedReprojErrorConstantPoseCostFunctor(const Eigen::Vector2d& point2D,
                                             const Rigid3d& cam_from_world,
                                             double L00,
                                             double L10,
                                             double L11)
      : cam_from_world_(cam_from_world),
        weighted_reproj_cost_(point2D, L00, L10, L11) {}

  template <typename T>
  bool operator()(const T* const point3D_in_world,
                  const T* const camera_params,
                  T* residuals) const {
    const Eigen::Matrix<T, 7, 1> cam_from_world =
        cam_from_world_.params.cast<T>();
    return weighted_reproj_cost_(
        point3D_in_world, cam_from_world.data(), camera_params, residuals);
  }

 private:
  const Rigid3d cam_from_world_;
  const WeightedReprojErrorCostFunctor<CameraModel> weighted_reproj_cost_;
};

}  // namespace colmap
