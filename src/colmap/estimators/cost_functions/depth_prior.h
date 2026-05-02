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

#pragma once

#include "colmap/estimators/cost_functions/utils.h"
#include "colmap/geometry/rigid3.h"

#include <ceres/ceres.h>

namespace colmap {

// Depth prior: constrains cam-frame z of a 3D point against monocular depth
// with learnable affine correction: depth_pred = shift + mono_depth * exp(scale)
// Residual: (R*p + t).z - shift - mono_depth * exp(scale)
// Params: cam_from_world[7], point3D[3], shift_scale[2]
struct ScaledDepthErrorCostFunctor
    : public AutoDiffCostFunctor<ScaledDepthErrorCostFunctor, 1, 7, 3, 2> {
  explicit ScaledDepthErrorCostFunctor(double depth) : depth_(depth) {}

  template <typename T>
  bool operator()(const T* const cam_from_world,
                  const T* const point3D,
                  const T* const shift_scale,
                  T* residuals) const {
    *residuals = (EigenQuaternionMap<T>(cam_from_world) *
                  EigenVector3Map<T>(point3D))[2] +
                 cam_from_world[6] - shift_scale[0] -
                 T(depth_) * exp(shift_scale[1]);
    return true;
  }

 private:
  const double depth_;
};

// Constant-pose variant: bakes in a fixed camera pose.
// Params: point3D[3], shift_scale[2]
struct ScaledDepthErrorConstantPoseCostFunctor
    : public AutoDiffCostFunctor<ScaledDepthErrorConstantPoseCostFunctor,
                                 1,
                                 3,
                                 2> {
  ScaledDepthErrorConstantPoseCostFunctor(const Rigid3d& cam_from_world,
                                          double depth)
      : cam_from_world_(cam_from_world), depth_cost_(depth) {}

  template <typename T>
  bool operator()(const T* const point3D,
                  const T* const shift_scale,
                  T* residuals) const {
    const Eigen::Matrix<T, 7, 1> cam_from_world =
        cam_from_world_.params.cast<T>();
    return depth_cost_(cam_from_world.data(), point3D, shift_scale, residuals);
  }

 private:
  const Rigid3d cam_from_world_;
  const ScaledDepthErrorCostFunctor depth_cost_;
};

// Log-space depth prior.
// Residual: log(d_predicted) - (log(mono_depth) + scale)
// Params: cam_from_world[7], point3D[3], shift_scale[2]
struct LogScaledDepthErrorCostFunctor
    : public AutoDiffCostFunctor<LogScaledDepthErrorCostFunctor, 1, 7, 3, 2> {
  explicit LogScaledDepthErrorCostFunctor(double depth) : depth_(depth) {}

  template <typename T>
  bool operator()(const T* const cam_from_world,
                  const T* const point3D,
                  const T* const shift_scale,
                  T* residuals) const {
    T d_pred = (EigenQuaternionMap<T>(cam_from_world) *
                EigenVector3Map<T>(point3D))[2] +
               cam_from_world[6];
    if (d_pred <= T(0)) {
      *residuals = T(0);
      return true;
    }
    *residuals =
        ceres::log(d_pred) - (ceres::log(T(depth_)) + shift_scale[1]);
    return true;
  }

 private:
  const double depth_;
};

}  // namespace colmap
