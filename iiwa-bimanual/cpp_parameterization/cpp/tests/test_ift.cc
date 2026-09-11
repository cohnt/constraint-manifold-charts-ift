#include <gtest/gtest.h>
#include "iiwa_ik/iiwa_analytic_ik.h"
#include <cmath>
#include <drake/math/autodiff.h>
#include <drake/math/autodiff_gradient.h>
#include <iostream>

using drake::AutoDiffXd;

TEST(IiwaIkTest, IftCalculations) {
  BimanualConfig config(true, true, true, 0.6);

  // x = q_c, psi (8D)
  Eigen::VectorXd x_val(8);
  x_val << -0.35389792367445905, -0.9575655830412433, -1.2329584958629203,
      1.9767613358728013, 0.517735077297024, 1.387235911675766,
      -2.7709720627217385, 0.012092731771290294;

  // y = q_sub (7D)
  Eigen::VectorXd q_s_d(7);
  q_s_d << 2.4648314486007132, 1.2121277540631108, 0.007040232106980958,
      1.308584045159517, 2.518718184118714, 2.203800876906688,
      1.130281145942008;

  // 1) Build tildeJ_s(q_s)
  const Eigen::Matrix<AutoDiffXd, 7, 1> q_s_ad =
      drake::math::InitializeAutoDiff(q_s_d);

  const Eigen::Matrix4<AutoDiffXd> X_Ws_ad = FK_IiwaDh<AutoDiffXd>(q_s_ad);
  const Eigen::Matrix<AutoDiffXd, 6, 1> pose6_s_ad = Pose6FromMatrix4(X_Ws_ad);
  const AutoDiffXd psi_s_ad = ComputePsi_Iiwa<AutoDiffXd>(q_s_ad, config);

  Eigen::Matrix<double, 7, 7> tildeJ_s;
  for (int r = 0; r < 6; ++r) {
    tildeJ_s.row(r) = pose6_s_ad(r).derivatives().transpose();
  }
  tildeJ_s.row(6) = psi_s_ad.derivatives().transpose();

  Eigen::JacobiSVD<Eigen::MatrixXd> svd(tildeJ_s);
  double cond = svd.singularValues()(0) /
                 svd.singularValues()(svd.singularValues().size() - 1);
  std::cout << "Condition Number of tildeJ_s: " << cond << "\n";

  // 2) Build J_goal_qc(q_c)
  const Eigen::Matrix<double, 7, 1> q_c_d = x_val.head<7>();
  const Eigen::Matrix<AutoDiffXd, 7, 1> q_c_ad =
      drake::math::InitializeAutoDiff(q_c_d);

  const Eigen::Matrix4<AutoDiffXd> X_Wc_ad = FK_IiwaDh<AutoDiffXd>(q_c_ad);
  const Eigen::Matrix4<AutoDiffXd> X_Wgoal_ad =
      LinkageTransformFromControlledEE<AutoDiffXd>(X_Wc_ad, config.grasp_distance);

  const Eigen::Matrix<AutoDiffXd, 6, 1> pose6_goal_ad =
      Pose6FromMatrix4(X_Wgoal_ad);

  Eigen::Matrix<double, 6, 7> J_goal_qc;
  for (int r = 0; r < 6; ++r) {
    J_goal_qc.row(r) = pose6_goal_ad(r).derivatives().transpose();
  }

  // Basic sanity checks
  EXPECT_GT(cond, 0.0);
  EXPECT_EQ(tildeJ_s.rows(), 7);
  EXPECT_EQ(tildeJ_s.cols(), 7);
}

namespace {

// A configuration that is reachable across the whole psi domain.
Eigen::VectorXd ReachableQTilde() {
  Eigen::VectorXd q(8);
  q << -0.6430910102907225, 1.9156121024586796, -1.7968254667817805,
      1.2945447141185198, -0.023834531305537934, -0.876966810663043,
      -1.7041643160834519, 1.45;
  return q;
}

std::vector<AutoDiffConfig> AllIftStrategies() {
  std::vector<AutoDiffConfig> out;
  auto make = [](IftSingularityHandling h, double lambda, bool aniso) {
    AutoDiffConfig c;
    c.use_ift = true;
    c.ift_handling = h;
    c.lambda = lambda;
    c.use_anisotropic_damping = aniso;
    return c;
  };
  out.push_back(make(IftSingularityHandling::kZero, 0.0, false));
  out.push_back(make(IftSingularityHandling::kPseudoinverse, 0.0, false));
  out.push_back(make(IftSingularityHandling::kLevenbergMarquardt, 1e-5, false));
  out.push_back(make(IftSingularityHandling::kResidualDamping, 5.0, false));
  out.push_back(make(IftSingularityHandling::kResidualDamping, 10.0, true));
  out.push_back(make(IftSingularityHandling::kFullNewton, 1e-6, false));
  AutoDiffConfig svt = make(IftSingularityHandling::kLevenbergMarquardt, 0.0, false);
  svt.svt_epsilon = 0.02;
  svt.svt_lambda_max = 0.01;
  out.push_back(svt);
  return out;
}

}  // namespace

// The IK depends on psi only through sin/cos, so psi and psi + 2*pi denote the
// same configuration and must produce identical gradients. Regression test for a
// bug where the psi residual was not wrapped, which silently damped the
// residual-based strategies over the half of the domain above pi.
TEST(IiwaIkTest, PsiPeriodicity) {
  BimanualConfig config(true, true, false, 0.6);
  const Eigen::VectorXd q = ReachableQTilde();
  Eigen::VectorXd q_shifted = q;
  q_shifted(7) += 2.0 * M_PI;

  for (const auto& ad_config : AllIftStrategies()) {
    drake::AutoDiffVecXd a = drake::math::InitializeAutoDiff(q);
    drake::AutoDiffVecXd b = drake::math::InitializeAutoDiff(q_shifted);
    Eigen::MatrixXd ga = drake::math::ExtractGradient(
        IiwaBimanualParameterizationIFT(a, config, ad_config, nullptr));
    Eigen::MatrixXd gb = drake::math::ExtractGradient(
        IiwaBimanualParameterizationIFT(b, config, ad_config, nullptr));
    EXPECT_LT((ga - gb).cwiseAbs().maxCoeff(), 1e-9)
        << "gradients differ across psi -> psi + 2*pi for handling "
        << static_cast<int>(ad_config.ift_handling)
        << " (anisotropic=" << ad_config.use_anisotropic_damping << ")";
  }
}

// At a reachable configuration the residual is zero, so the residual-driven
// domain-extension approximations must collapse to the exact IFT solution and
// agree with ordinary autodiff. Sweeping psi across the full planning domain
// [0, 2*pi] is what makes this catch residual bugs confined to part of the domain.
//
// Levenberg-Marquardt is deliberately excluded: both the constant and SVT variants
// regularize based on lambda and the conditioning of J_A, not the residual, so they
// damp at reachable configurations by design.
TEST(IiwaIkTest, ReachableConfigsMatchAutodiff) {
  BimanualConfig config(true, true, false, 0.6);
  Eigen::VectorXd q = ReachableQTilde();

  AutoDiffConfig ad_plain;
  ad_plain.use_ift = false;

  std::vector<AutoDiffConfig> residual_based;
  for (const auto& c : AllIftStrategies()) {
    if (c.ift_handling != IftSingularityHandling::kLevenbergMarquardt) {
      residual_based.push_back(c);
    }
  }
  // zero, pseudoinverse, residual damping (std + anisotropic), full Newton.
  ASSERT_EQ(residual_based.size(), 5u);

  const int steps = 16;
  for (int i = 0; i < steps; ++i) {
    q(7) = 0.05 + (2.0 * M_PI - 0.1) * i / (steps - 1);
    drake::AutoDiffVecXd q_ad = drake::math::InitializeAutoDiff(q);
    Eigen::MatrixXd reference = drake::math::ExtractGradient(
        IiwaBimanualParameterization<AutoDiffXd>(q_ad, config, ad_plain, nullptr));

    for (const auto& ad_config : residual_based) {
      drake::AutoDiffVecXd in = drake::math::InitializeAutoDiff(q);
      Eigen::MatrixXd g = drake::math::ExtractGradient(
          IiwaBimanualParameterizationIFT(in, config, ad_config, nullptr));
      EXPECT_LT((g - reference).cwiseAbs().maxCoeff(), 1e-8)
          << "IFT gradient disagrees with autodiff at reachable psi=" << q(7)
          << " for handling " << static_cast<int>(ad_config.ift_handling)
          << " (anisotropic=" << ad_config.use_anisotropic_damping << ")";
    }
  }
}
