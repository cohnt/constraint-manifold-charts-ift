#define _USE_MATH_DEFINES
#include "iiwa_ik/iiwa_analytic_ik.h"
#include <cmath>
#include <gtest/gtest.h>

using drake::AutoDiffXd;

class ParameterizationValueTest : public ::testing::TestWithParam<std::tuple<bool, bool, bool>> {
protected:
  double grasp_distance = 0.6;
};

TEST_P(ParameterizationValueTest, IsInverseOfCorrectPose) {
  auto [shoulder_up, elbow_up, wrist_up] = GetParam();
  BimanualConfig config(shoulder_up, elbow_up, wrist_up, grasp_distance);

  // Pick a point in the controlled space
  Eigen::VectorXd x_val(8);
  x_val << -0.353, -0.957, -1.232, 1.976, 0.517, 1.387, -2.770, 0.5;

  // 1) Compute goal pose from controlled arm
  const Eigen::Matrix<double, 7, 1> q_c = x_val.head<7>();
  const double psi_target = x_val(7);

  Eigen::Matrix4<double> X_Wc = FK_IiwaDh<double>(q_c);
  Eigen::Matrix4<double> X_Wgoal = LinkageTransformFromControlledEE<double>(X_Wc, grasp_distance);

  // 2) Compute subordinate arm configuration via parameterization (Analytic version)
  Eigen::VectorXd q_full_analytic = IiwaBimanualParameterization<double>(
      x_val, config, AutoDiffConfig{ .use_ift = false }, nullptr);

  // 3) Compute subordinate arm configuration via parameterization (IFT version)
  //    IFT version is only available for AutoDiffXd, so we use AutoDiffXd here.
  drake::AutoDiffVecXd x_ad = drake::math::InitializeAutoDiff(x_val);
  drake::AutoDiffVecXd q_full_ad = IiwaBimanualParameterization<AutoDiffXd>(
      x_ad, config, AutoDiffConfig{ .use_ift = true }, nullptr);
  Eigen::VectorXd q_full_ift = drake::math::ExtractValue(q_full_ad);

  Eigen::Matrix<double, 7, 1> q_s_ift = q_full_ift.tail<7>();

  // 3) Verify subordinate arm reaches the goal (IFT version)
  Eigen::Matrix4<double> X_Ws = FK_IiwaDh<double>(q_s_ift);
  
  EXPECT_TRUE(X_Ws.isApprox(X_Wgoal, 1e-10)) << "Pose mismatch for IFT config: " 
                                             << shoulder_up << elbow_up << wrist_up;

  // 4) Verify psi of the resulting configuration matches the target psi (IFT version)
  double psi_actual = ComputePsi_Iiwa<double>(q_s_ift, config);

  EXPECT_NEAR(psi_actual, psi_target, 1e-10) << "Psi mismatch for IFT config: " 
                                             << shoulder_up << elbow_up << wrist_up;
  
  // Also log the improvement if analytic was off
  double psi_analytic = ComputePsi_Iiwa<double>(q_full_analytic.tail<7>(), config);
  if (std::abs(psi_analytic - psi_target) > 1e-6) {
    // std::cout << "IFT refined psi from " << psi_analytic << " to " << psi_actual << " (target " << psi_target << ")" << std::endl;
  }
}

INSTANTIATE_TEST_SUITE_P(
    AllConfigs, ParameterizationValueTest,
    ::testing::Combine(::testing::Bool(), ::testing::Bool(), ::testing::Bool()));


TEST(ParameterizationTest, IftConsistency) {
  BimanualConfig config(true, true, true, 0.6);

  Eigen::VectorXd x_val(8);
  x_val << -0.353, -0.957, -1.232, 1.976, 0.517, 1.387, -2.770, 0.5;

  drake::AutoDiffVecXd x_ad = drake::math::InitializeAutoDiff(x_val);

  drake::AutoDiffVecXd q_full_ad = IiwaBimanualParameterization<AutoDiffXd>(
      x_ad, config, AutoDiffConfig{ .use_ift = true }, nullptr);

  Eigen::VectorXd q_full_reg = IiwaBimanualParameterization<double>(
      x_val, config, AutoDiffConfig{ .use_ift = false }, nullptr);

  EXPECT_TRUE(drake::math::ExtractValue(q_full_ad).isApprox(q_full_reg, 1e-6));
}
