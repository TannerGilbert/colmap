#include "colmap/estimators/ceres_loss.h"

#include "colmap/util/logging.h"

namespace colmap {

std::string_view LossFunctionTypeToString(
    const LossFunctionType loss_function_type) {
  switch (loss_function_type) {
    case LossFunctionType::TRIVIAL:
      return "TRIVIAL";
    case LossFunctionType::SOFT_L1:
      return "SOFT_L1";
    case LossFunctionType::CAUCHY:
      return "CAUCHY";
    case LossFunctionType::HUBER:
      return "HUBER";
  }
  LOG(FATAL) << "Unhandled LossFunctionType: "
             << static_cast<int>(loss_function_type);
}

LossFunctionType LossFunctionTypeFromString(const std::string_view name) {
  if (name == "TRIVIAL") {
    return LossFunctionType::TRIVIAL;
  } else if (name == "SOFT_L1") {
    return LossFunctionType::SOFT_L1;
  } else if (name == "CAUCHY") {
    return LossFunctionType::CAUCHY;
  } else if (name == "HUBER") {
    return LossFunctionType::HUBER;
  }
  LOG(FATAL) << "Invalid LossFunctionType: " << name;
}

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
