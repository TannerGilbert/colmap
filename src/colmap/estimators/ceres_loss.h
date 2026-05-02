#pragma once

#include <memory>

#include <ceres/loss_function.h>

namespace colmap {

// Supported Ceres robust loss kernels.
enum class LossFunctionType {
  TRIVIAL,
  SOFT_L1,
  CAUCHY,
  HUBER,
};

// Build a ceres::LossFunction from a typed config.
std::unique_ptr<ceres::LossFunction> CreateLossFunction(
    LossFunctionType loss_function_type, double loss_function_scale);

// (type, scale, weight) triple. Wraps in ScaledLoss when weight != 1.
struct LossConfig {
  LossFunctionType type = LossFunctionType::TRIVIAL;
  double scale = 1.0;
  double weight = 1.0;

  std::unique_ptr<ceres::LossFunction> CreateLossFunction() const {
    auto loss = colmap::CreateLossFunction(type, scale);
    if (weight != 1.0) {
      loss.reset(new ceres::ScaledLoss(
          loss.release(), weight, ceres::TAKE_OWNERSHIP));
    }
    return loss;
  }
};

}  // namespace colmap
