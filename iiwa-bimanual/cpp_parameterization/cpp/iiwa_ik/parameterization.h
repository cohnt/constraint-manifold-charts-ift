#pragma once

#include "drake/planning/iris/iris_common.h"
#include "iiwa_analytic_ik.h"
#include <Eigen/Dense>
#include <memory>

/** Factory function that returns a parameterization function object
 *  for the bimanual IIWA, with the given shoulder/elbow/wrist configuration. */
std::unique_ptr<drake::planning::IrisParameterizationFunction>
MakeParameterization(const BimanualConfig& config,
                     const AutoDiffConfig& ad_config = AutoDiffConfig{});