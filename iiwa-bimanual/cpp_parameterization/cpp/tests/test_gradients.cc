#define _USE_MATH_DEFINES
#include "iiwa_ik/iiwa_analytic_ik.h"
#include <cmath>
#include <gtest/gtest.h>
#include "drake/math/autodiff_gradient.h"

using drake::AutoDiffXd;
using drake::math::InitializeAutoDiff;
using drake::math::ExtractGradient;
using drake::math::ExtractValue;

// Helper to compute finite difference gradient of a function f: VectorXd -> VectorXd
Eigen::MatrixXd ComputeFiniteDifference(
    std::function<Eigen::VectorXd(const Eigen::VectorXd&)> f,
    const Eigen::VectorXd& x, double epsilon = 1e-7) {
  int m = f(x).size();
  int n = x.size();
  Eigen::MatrixXd J(m, n);
  for (int i = 0; i < n; ++i) {
    Eigen::VectorXd x_plus = x;
    x_plus(i) += epsilon;
    Eigen::VectorXd x_minus = x;
    x_minus(i) -= epsilon;
    J.col(i) = (f(x_plus) - f(x_minus)) / (2.0 * epsilon);
  }
  return J;
}

class ParameterizationGradientTest : public ::testing::TestWithParam<std::tuple<bool, bool, bool>> {
protected:
  double grasp_distance = 0.6;
  Eigen::VectorXd x_val;

  void SetUp() override {
    x_val.resize(8);
    // Use a known reachable point
    x_val << -0.353, -0.957, -1.232, 1.976, 0.517, 1.387, -2.770, 0.5;
  }
};

TEST_P(ParameterizationGradientTest, AutoDiffMatchesFiniteDifference) {
  auto [shoulder_up, elbow_up, wrist_up] = GetParam();
  BimanualConfig config(shoulder_up, elbow_up, wrist_up, grasp_distance);

  // Non-IFT version
  auto f = [&](const Eigen::VectorXd& x) {
    return IiwaBimanualParameterization<double>(
        x, config, AutoDiffConfig{ .use_ift = false }, nullptr);
  };
  Eigen::MatrixXd J_fd = ComputeFiniteDifference(f, x_val);

  drake::AutoDiffVecXd x_ad = InitializeAutoDiff(x_val);
  drake::AutoDiffVecXd q_full_ad = IiwaBimanualParameterization<AutoDiffXd>(
      x_ad, config, AutoDiffConfig{ .use_ift = false }, nullptr);
  Eigen::MatrixXd J_ad = ExtractGradient(q_full_ad);

  EXPECT_TRUE(J_ad.isApprox(J_fd, 1e-4)) << "Failed for config: " << shoulder_up << elbow_up << wrist_up;
}


TEST_P(ParameterizationGradientTest, IftAutoDiffMatchesFiniteDifference) {
  auto [shoulder_up, elbow_up, wrist_up] = GetParam();
  BimanualConfig config(shoulder_up, elbow_up, wrist_up, grasp_distance);

  // Finite Difference of the analytic version (as reference)
  auto f = [&](const Eigen::VectorXd& x) {
    return IiwaBimanualParameterization<double>(
        x, config, AutoDiffConfig{ .use_ift = false }, nullptr);
  };
  Eigen::MatrixXd J_fd = ComputeFiniteDifference(f, x_val);

  // IFT version
  drake::AutoDiffVecXd x_ad = InitializeAutoDiff(x_val);
  drake::AutoDiffVecXd q_full_ad = IiwaBimanualParameterization<AutoDiffXd>(
      x_ad, config, AutoDiffConfig{ .use_ift = true }, nullptr);
  Eigen::MatrixXd J_ad = ExtractGradient(q_full_ad);


  // Note: We use a slightly looser tolerance for IFT as it's a numerical derivative approach
  double diff = (J_ad - J_fd).norm();
  if (diff > 5e-3) {
    std::cout << "IFT Graduate error for config: " << shoulder_up << elbow_up << wrist_up << " is " << diff << std::endl;
    std::cout << "Max element-wise diff: " << (J_ad - J_fd).cwiseAbs().maxCoeff() << std::endl;
  }
  EXPECT_TRUE(J_ad.isApprox(J_fd, 1e-3)) << "IFT Failed for config: " << shoulder_up << elbow_up << wrist_up 
                                         << "\nNorm diff: " << diff;
}

INSTANTIATE_TEST_SUITE_P(
    AllConfigs, ParameterizationGradientTest,
    ::testing::Combine(::testing::Bool(), ::testing::Bool(), ::testing::Bool()));
