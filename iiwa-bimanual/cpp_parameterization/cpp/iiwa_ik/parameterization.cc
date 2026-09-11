#include "parameterization.h"

std::unique_ptr<drake::planning::IrisParameterizationFunction>
MakeParameterization(const BimanualConfig& config,
                     const AutoDiffConfig& ad_config) {

  // Capture the config by value in lambdas.
  auto parameterization_double = [=](const Eigen::VectorXd &x) {
    return IiwaBimanualParameterization<double>(
        x, config, ad_config, nullptr);
  };

  auto parameterization_autodiff =
      [=](const Eigen::VectorX<drake::AutoDiffXd> &x) {
        return IiwaBimanualParameterization<drake::AutoDiffXd>(
            x, config, ad_config, nullptr);
      };

  bool is_threadsafe = true;

  return std::make_unique<drake::planning::IrisParameterizationFunction>(
      parameterization_double, parameterization_autodiff, is_threadsafe, 8);
}