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

#include "colmap/estimators/cost_functions/motion_averaging.h"

#include "colmap/estimators/cost_functions/utils.h"
#include "colmap/util/eigen_matchers.h"

#include <random>

#include <gtest/gtest.h>

namespace colmap {
namespace {

TEST(BATAPairwiseDirectionCostFunctor, ZeroResidual) {
  const Eigen::Vector3d pos1(1, 2, 3);
  const Eigen::Vector3d pos2(2, 3, 4);
  const double scale = 1.0;
  const Eigen::Vector3d direction = pos2 - pos1;

  BATAPairwiseDirectionCostFunctor cost_functor(direction);

  Eigen::Vector3d residuals;
  EXPECT_TRUE(cost_functor(pos1.data(), pos2.data(), &scale, residuals.data()));

  EXPECT_THAT(residuals, EigenMatrixNear(Eigen::Vector3d(0, 0, 0), 1e-10));
}

TEST(BATAPairwiseDirectionCostFunctor, NonZeroResidual) {
  const Eigen::Vector3d pos1(1, 2, 3);
  const Eigen::Vector3d pos2(4, 5, 6);
  const double scale = 2.0;
  const Eigen::Vector3d direction(1, 1, 1);

  BATAPairwiseDirectionCostFunctor cost_functor(direction);

  Eigen::Vector3d residuals;
  EXPECT_TRUE(cost_functor(pos1.data(), pos2.data(), &scale, residuals.data()));

  const Eigen::Vector3d expected_residuals = direction - scale * (pos2 - pos1);
  EXPECT_THAT(residuals, EigenMatrixNear(expected_residuals, 1e-10));
}

TEST(BATAPairwiseDirectionCostFunctor, DifferentScale) {
  const Eigen::Vector3d pos1(1, 2, 3);
  const Eigen::Vector3d pos2(2, 4, 6);
  const double scale = 0.5;
  const Eigen::Vector3d direction = scale * (pos2 - pos1);

  BATAPairwiseDirectionCostFunctor cost_functor(direction);

  Eigen::Vector3d residuals;
  EXPECT_TRUE(cost_functor(pos1.data(), pos2.data(), &scale, residuals.data()));

  EXPECT_THAT(residuals, EigenMatrixNear(Eigen::Vector3d(0, 0, 0), 1e-10));
}

TEST(BATAPairwiseDirectionCostFunctor, Create) {
  const Eigen::Vector3d direction(1, 0, 0);
  std::unique_ptr<ceres::CostFunction> cost_function(
      BATAPairwiseDirectionCostFunctor::Create(direction));
  ASSERT_NE(cost_function, nullptr);
}

TEST(RigBATAPairwiseDirectionConstantRigCostFunctor, ZeroResidual) {
  const Eigen::Vector3d point3D(1, 2, 3);
  const Eigen::Vector3d rig_in_world(3, 2, 1);
  const double scale = 1.5;
  const Eigen::Vector3d cam_from_rig_dir(0.25, 0.5, 0.75);
  const Eigen::Vector3d cam_from_point3D_dir =
      scale * (point3D - rig_in_world + cam_from_rig_dir);

  RigBATAPairwiseDirectionConstantRigCostFunctor cost_functor(
      cam_from_point3D_dir, cam_from_rig_dir);

  Eigen::Vector3d residuals;
  EXPECT_TRUE(cost_functor(
      point3D.data(), rig_in_world.data(), &scale, residuals.data()));

  EXPECT_THAT(residuals, EigenMatrixNear(Eigen::Vector3d(0, 0, 0), 1e-10));
}

TEST(RigBATAPairwiseDirectionConstantRigCostFunctor, NonZeroResidual) {
  const Eigen::Vector3d point3D(3, 4, 5);
  const Eigen::Vector3d rig_in_world(1, 2, 3);
  const double scale = 2.0;
  const Eigen::Vector3d cam_from_rig_dir(0.1, 0.2, 0.3);
  const Eigen::Vector3d cam_from_point3D_dir(1, 1, 1);

  RigBATAPairwiseDirectionConstantRigCostFunctor cost_functor(
      cam_from_point3D_dir, cam_from_rig_dir);

  Eigen::Vector3d residuals;
  EXPECT_TRUE(cost_functor(
      point3D.data(), rig_in_world.data(), &scale, residuals.data()));

  const Eigen::Vector3d expected_residuals =
      cam_from_point3D_dir -
      scale * (point3D - rig_in_world + cam_from_rig_dir);
  EXPECT_THAT(residuals, EigenMatrixNear(expected_residuals, 1e-10));
}

TEST(RigBATAPairwiseDirectionConstantRigCostFunctor, Create) {
  const Eigen::Vector3d cam_from_point3D_dir(1, 0, 0);
  const Eigen::Vector3d cam_from_rig_dir(0, 1, 0);
  std::unique_ptr<ceres::CostFunction> cost_function(
      RigBATAPairwiseDirectionConstantRigCostFunctor::Create(
          cam_from_point3D_dir, cam_from_rig_dir));
  ASSERT_NE(cost_function, nullptr);
}

TEST(RigBATAPairwiseDirectionCostFunctor, ZeroResidual) {
  const Eigen::Vector3d point3D(5, 5, 5);
  const Eigen::Vector3d rig_in_world(1, 1, 1);
  const Eigen::Vector3d cam_in_rig(0.5, 0.5, 0.5);
  const double scale = 1.0;
  const Eigen::Quaterniond rig_from_world_rot = Eigen::Quaterniond::Identity();
  const Eigen::Vector3d cam_from_rig_dir =
      rig_from_world_rot.inverse() * cam_in_rig;
  const Eigen::Vector3d cam_from_point3D_dir =
      scale * (point3D - rig_in_world - cam_from_rig_dir);

  RigBATAPairwiseDirectionCostFunctor cost_functor(cam_from_point3D_dir,
                                                   rig_from_world_rot);

  Eigen::Vector3d residuals;
  EXPECT_TRUE(cost_functor(point3D.data(),
                           rig_in_world.data(),
                           cam_in_rig.data(),
                           &scale,
                           residuals.data()));

  EXPECT_THAT(residuals, EigenMatrixNear(Eigen::Vector3d(0, 0, 0), 1e-10));
}

TEST(RigBATAPairwiseDirectionCostFunctor, NonZeroResidual) {
  const Eigen::Vector3d point3D(3, 4, 5);
  const Eigen::Vector3d rig_in_world(1, 2, 3);
  const Eigen::Vector3d cam_in_rig(0.2, 0.3, 0.4);
  const double scale = 2.0;
  const Eigen::Quaterniond rig_from_world_rot =
      Eigen::Quaterniond(0.707, 0.707, 0, 0).normalized();
  const Eigen::Vector3d cam_from_point3D_dir(1, 1, 1);

  RigBATAPairwiseDirectionCostFunctor cost_functor(cam_from_point3D_dir,
                                                   rig_from_world_rot);

  Eigen::Vector3d residuals;
  EXPECT_TRUE(cost_functor(point3D.data(),
                           rig_in_world.data(),
                           cam_in_rig.data(),
                           &scale,
                           residuals.data()));

  const Eigen::Vector3d cam_from_rig_dir =
      rig_from_world_rot.toRotationMatrix().transpose() * cam_in_rig;
  const Eigen::Vector3d expected_residuals =
      cam_from_point3D_dir -
      scale * (point3D - rig_in_world - cam_from_rig_dir);
  EXPECT_THAT(residuals, EigenMatrixNear(expected_residuals, 1e-10));
}

TEST(RigBATAPairwiseDirectionCostFunctor, Create) {
  const Eigen::Vector3d cam_from_point3D_dir(1, 0, 0);
  const Eigen::Quaterniond rig_from_world_rot = Eigen::Quaterniond::Identity();
  std::unique_ptr<ceres::CostFunction> cost_function(
      RigBATAPairwiseDirectionCostFunctor::Create(cam_from_point3D_dir,
                                                  rig_from_world_rot));
  ASSERT_NE(cost_function, nullptr);
}

// CWCF<BATA>(cov_world, t_obs) should produce the same squared residual norm
// as a per-axis whitened BATA residual (residual.{x,y,z} / sigma_{x,y,z}),
// when cov_world = R^T diag(sigma^2) R. The tests below verify this both
// on randomized inputs and on the closed-form identity / isotropic case.

TEST(CovarianceWeightedBATAPairwiseDirection, MatchesPerAxisWhitened) {
  std::mt19937 rng(42);
  std::uniform_real_distribution<double> uni(-1.0, 1.0);
  std::uniform_real_distribution<double> sigma_uni(0.05, 1.5);
  std::uniform_real_distribution<double> scale_uni(0.1, 3.0);

  for (int trial = 0; trial < 20; ++trial) {
    // Random unit translation observation t_obs.
    Eigen::Vector3d t_obs(uni(rng), uni(rng), uni(rng));
    while (t_obs.norm() < 1e-3) {
      t_obs = Eigen::Vector3d(uni(rng), uni(rng), uni(rng));
    }
    t_obs.normalize();

    // Random rotation R (cam_from_world).
    const Eigen::Quaterniond q(uni(rng), uni(rng), uni(rng), uni(rng));
    const Eigen::Quaterniond q_norm = q.normalized();
    const Eigen::Matrix3d R = q_norm.toRotationMatrix();

    // Random anisotropic sigmas and per-axis variances.
    const double sigma_x = sigma_uni(rng);
    const double sigma_y = sigma_uni(rng);
    const double sigma_z = sigma_uni(rng);
    const Eigen::Vector3d sigma2(
        sigma_x * sigma_x, sigma_y * sigma_y, sigma_z * sigma_z);

    // cov_world = R^T diag(sigma^2) R (the encoding the call site uses).
    const Eigen::Matrix3d cov_world = R.transpose() * sigma2.asDiagonal() * R;

    // Random parameters.
    Eigen::Vector3d c1(uni(rng), uni(rng), uni(rng));
    Eigen::Vector3d c2(uni(rng), uni(rng), uni(rng));
    double scale = scale_uni(rng);

    // Evaluate native CWCF<BATA>(cov_world, t_obs).
    std::unique_ptr<ceres::CostFunction> cost_function(
        CovarianceWeightedCostFunctor<BATAPairwiseDirectionCostFunctor>::Create(
            cov_world, t_obs));
    ASSERT_NE(cost_function, nullptr);

    Eigen::Vector3d residuals(std::numeric_limits<double>::quiet_NaN(),
                              std::numeric_limits<double>::quiet_NaN(),
                              std::numeric_limits<double>::quiet_NaN());
    const double* parameters[3] = {c1.data(), c2.data(), &scale};
    EXPECT_TRUE(cost_function->Evaluate(parameters, residuals.data(), nullptr));
    const double native_sqnorm = residuals.squaredNorm();

    // Per-axis whitened squared norm: r_world^T * R^T * diag(1/sigma^2) * R *
    // r_world, computed directly.
    const Eigen::Vector3d r_world = t_obs - scale * (c2 - c1);
    const Eigen::Vector3d r_cam = R * r_world;
    const Eigen::Vector3d r_cam_whitened(
        r_cam.x() / sigma_x, r_cam.y() / sigma_y, r_cam.z() / sigma_z);
    const double expected_sqnorm = r_cam_whitened.squaredNorm();

    // Identity-required: equal up to floating-point noise.
    EXPECT_NEAR(native_sqnorm, expected_sqnorm, 1e-10)
        << "trial " << trial << ": native=" << native_sqnorm
        << " expected=" << expected_sqnorm;
  }
}

TEST(CovarianceWeightedBATAPairwiseDirection, IdentityRotationIsotropicSigma) {
  // With R = I and sigma_x = sigma_y = sigma_z = sigma, the whitened
  // residual must equal the plain BATA residual divided by sigma.
  const Eigen::Vector3d t_obs(1.0, 2.0, 3.0);
  const Eigen::Vector3d c1(0.5, -0.2, 1.1);
  const Eigen::Vector3d c2(2.0, 0.7, -0.4);
  double scale = 1.3;
  const double sigma = 0.5;
  const Eigen::Matrix3d cov_world =
      (sigma * sigma) * Eigen::Matrix3d::Identity();

  std::unique_ptr<ceres::CostFunction> weighted(
      CovarianceWeightedCostFunctor<BATAPairwiseDirectionCostFunctor>::Create(
          cov_world, t_obs));
  std::unique_ptr<ceres::CostFunction> plain(
      BATAPairwiseDirectionCostFunctor::Create(t_obs));

  Eigen::Vector3d r_weighted, r_plain;
  const double* parameters[3] = {c1.data(), c2.data(), &scale};
  EXPECT_TRUE(weighted->Evaluate(parameters, r_weighted.data(), nullptr));
  EXPECT_TRUE(plain->Evaluate(parameters, r_plain.data(), nullptr));

  const Eigen::Vector3d expected = r_plain / sigma;
  EXPECT_THAT(r_weighted, EigenMatrixNear(expected, 1e-12));
}

TEST(CovarianceWeightedBATAPairwiseDirection, ZeroResidualWhenAligned) {
  // r_world = t_obs - scale * (c2 - c1) = 0 ⇒ residual = 0 regardless of
  // covariance.
  const Eigen::Vector3d c1(1.0, 2.0, 3.0);
  const Eigen::Vector3d c2(4.0, 6.0, 8.0);
  double scale = 0.7;
  const Eigen::Vector3d t_obs = scale * (c2 - c1);

  // Arbitrary non-isotropic, non-axis-aligned covariance.
  const Eigen::Quaterniond q = Eigen::Quaterniond(
      Eigen::AngleAxisd(0.6, Eigen::Vector3d(1, 2, 3).normalized()));
  const Eigen::Matrix3d R = q.toRotationMatrix();
  const Eigen::Vector3d sigma2(0.04, 0.25, 0.09);
  const Eigen::Matrix3d cov_world = R.transpose() * sigma2.asDiagonal() * R;

  std::unique_ptr<ceres::CostFunction> cost_function(
      CovarianceWeightedCostFunctor<BATAPairwiseDirectionCostFunctor>::Create(
          cov_world, t_obs));

  Eigen::Vector3d residuals;
  const double* parameters[3] = {c1.data(), c2.data(), &scale};
  EXPECT_TRUE(cost_function->Evaluate(parameters, residuals.data(), nullptr));
  EXPECT_THAT(residuals, EigenMatrixNear(Eigen::Vector3d(0, 0, 0), 1e-12));
}

}  // namespace
}  // namespace colmap
