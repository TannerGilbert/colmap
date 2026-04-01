// Copyright (c), ETH Zurich and UNC Chapel Hill.
// All rights reserved.
// (BSD-3-Clause license, see LICENSE)

#include "colmap/estimators/cost_functions/depth_prior.h"

#include "colmap/estimators/cost_functions/pose_prior.h"
#include "colmap/geometry/rigid3.h"

#include <ceres/ceres.h>
#include <gtest/gtest.h>

namespace colmap {
namespace {

TEST(ScaledDepthErrorCostFunctor, ZeroResidualAtIdentity) {
  const double mono_depth = 5.0;
  std::unique_ptr<ceres::CostFunction> cost_function(
      ScaledDepthErrorCostFunctor::Create(mono_depth));

  Rigid3d cam_from_world;
  Eigen::Vector3d point3D(0, 0, mono_depth);
  Eigen::Vector2d shift_scale(0, 0);

  double residual = std::numeric_limits<double>::quiet_NaN();
  const double* parameters[3] = {cam_from_world.params.data(),
                                 point3D.data(),
                                 shift_scale.data()};
  EXPECT_TRUE(cost_function->Evaluate(parameters, &residual, nullptr));
  EXPECT_NEAR(residual, 0.0, 1e-10);
}

TEST(ScaledDepthErrorCostFunctor, NonZeroResidualWithShiftScale) {
  const double mono_depth = 5.0;
  std::unique_ptr<ceres::CostFunction> cost_function(
      ScaledDepthErrorCostFunctor::Create(mono_depth));

  Rigid3d cam_from_world;
  Eigen::Vector3d point3D(0, 0, 10.0);
  Eigen::Vector2d shift_scale(1.0, 0.5);

  // Expected: d_cam = 10.0, shift=1.0, depth*exp(scale)=5*exp(0.5)
  // residual = 10.0 - 1.0 - 5*exp(0.5)
  double expected = 10.0 - 1.0 - 5.0 * std::exp(0.5);

  double residual = std::numeric_limits<double>::quiet_NaN();
  const double* parameters[3] = {cam_from_world.params.data(),
                                 point3D.data(),
                                 shift_scale.data()};
  EXPECT_TRUE(cost_function->Evaluate(parameters, &residual, nullptr));
  EXPECT_NEAR(residual, expected, 1e-10);
}

TEST(ScaledDepthErrorConstantPoseCostFunctor, MatchesVariablePose) {
  const double mono_depth = 3.0;
  Rigid3d cam_from_world(
      Eigen::Quaterniond(Eigen::AngleAxisd(0.1, Eigen::Vector3d::UnitY())),
      Eigen::Vector3d(1, 2, 3));

  std::unique_ptr<ceres::CostFunction> variable(
      ScaledDepthErrorCostFunctor::Create(mono_depth));
  std::unique_ptr<ceres::CostFunction> constant(
      ScaledDepthErrorConstantPoseCostFunctor::Create(cam_from_world,
                                                      mono_depth));

  Eigen::Vector3d point3D(0.5, -0.3, 4.0);
  Eigen::Vector2d shift_scale(0.1, 0.2);

  double residual_var = 0, residual_const = 0;
  const double* params_var[3] = {cam_from_world.params.data(),
                                 point3D.data(),
                                 shift_scale.data()};
  const double* params_const[2] = {point3D.data(), shift_scale.data()};

  EXPECT_TRUE(variable->Evaluate(params_var, &residual_var, nullptr));
  EXPECT_TRUE(constant->Evaluate(params_const, &residual_const, nullptr));
  EXPECT_NEAR(residual_var, residual_const, 1e-10);
}

TEST(LogScaledDepthErrorCostFunctor, ZeroResidualAtIdentity) {
  const double mono_depth = 5.0;
  std::unique_ptr<ceres::CostFunction> cost_function(
      LogScaledDepthErrorCostFunctor::Create(mono_depth));

  Rigid3d cam_from_world;
  Eigen::Vector3d point3D(0, 0, mono_depth);
  Eigen::Vector2d shift_scale(0, 0);

  double residual = std::numeric_limits<double>::quiet_NaN();
  const double* parameters[3] = {cam_from_world.params.data(),
                                 point3D.data(),
                                 shift_scale.data()};
  EXPECT_TRUE(cost_function->Evaluate(parameters, &residual, nullptr));
  EXPECT_NEAR(residual, 0.0, 1e-10);
}

TEST(HeightPriorCostFunctor, ZeroResidualAtOrigin) {
  std::unique_ptr<ceres::CostFunction> cost_function(
      HeightPriorCostFunctor::Create(1.0, 0.0, 1));

  Rigid3d cam_from_world;  // identity → world pos = (0,0,0)

  double residual = std::numeric_limits<double>::quiet_NaN();
  const double* parameters[1] = {cam_from_world.params.data()};
  EXPECT_TRUE(cost_function->Evaluate(parameters, &residual, nullptr));
  EXPECT_NEAR(residual, 0.0, 1e-10);
}

TEST(HeightPriorCostFunctor, CorrectAxisSelection) {
  // Pose: translation = (0, 0, -5), identity rotation
  // → world position = -R^T * t = (0, 0, 5)
  Rigid3d cam_from_world(Eigen::Quaterniond::Identity(),
                         Eigen::Vector3d(0, 0, -5));

  // axis=2 (z), target=5 → residual should be 0
  std::unique_ptr<ceres::CostFunction> cf_z(
      HeightPriorCostFunctor::Create(1.0, 5.0, 2));
  double residual = std::numeric_limits<double>::quiet_NaN();
  const double* params[1] = {cam_from_world.params.data()};
  EXPECT_TRUE(cf_z->Evaluate(params, &residual, nullptr));
  EXPECT_NEAR(residual, 0.0, 1e-10);

  // axis=0 (x), target=0 → residual should be 0 (x=0)
  std::unique_ptr<ceres::CostFunction> cf_x(
      HeightPriorCostFunctor::Create(1.0, 0.0, 0));
  EXPECT_TRUE(cf_x->Evaluate(params, &residual, nullptr));
  EXPECT_NEAR(residual, 0.0, 1e-10);

  // axis=2 (z), target=3 → residual = 1*(5-3) = 2
  std::unique_ptr<ceres::CostFunction> cf_z2(
      HeightPriorCostFunctor::Create(1.0, 3.0, 2));
  EXPECT_TRUE(cf_z2->Evaluate(params, &residual, nullptr));
  EXPECT_NEAR(residual, 2.0, 1e-10);
}

TEST(HeightPriorCostFunctor, InvSigmaScalesResidual) {
  Rigid3d cam_from_world(Eigen::Quaterniond::Identity(),
                         Eigen::Vector3d(0, -2, 0));
  // world pos = (0, 2, 0), axis=1, target=0 → base residual = 2

  std::unique_ptr<ceres::CostFunction> cf1(
      HeightPriorCostFunctor::Create(1.0, 0.0, 1));
  std::unique_ptr<ceres::CostFunction> cf3(
      HeightPriorCostFunctor::Create(3.0, 0.0, 1));

  double r1, r3;
  const double* params[1] = {cam_from_world.params.data()};
  cf1->Evaluate(params, &r1, nullptr);
  cf3->Evaluate(params, &r3, nullptr);
  EXPECT_NEAR(r3, 3.0 * r1, 1e-10);
}

}  // namespace
}  // namespace colmap
