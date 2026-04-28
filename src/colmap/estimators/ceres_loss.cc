#include "colmap/estimators/ceres_loss.h"

#include "colmap/util/logging.h"

namespace colmap {

std::unique_ptr<ceres::LossFunction> CreateLossFunction(
    LossFunctionType loss_function_type, double loss_function_scale) {
  switch (loss_function_type) {
    case LossFunctionType::TRIVIAL:
      return std::make_unique<ceres::TrivialLoss>();
    case LossFunctionType::SOFT_L1:
      return std::make_unique<ceres::SoftLOneLoss>(loss_function_scale);
    case LossFunctionType::CAUCHY:
      return std::make_unique<ceres::CauchyLoss>(loss_function_scale);
    case LossFunctionType::HUBER:
      return std::make_unique<ceres::HuberLoss>(loss_function_scale);
  }
  LOG(FATAL) << "Unhandled LossFunctionType: "
             << static_cast<int>(loss_function_type);
}

}  // namespace colmap
