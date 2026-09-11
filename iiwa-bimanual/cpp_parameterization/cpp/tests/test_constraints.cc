#define _USE_MATH_DEFINES
#include <cmath>
#include <Eigen/Dense>
#include "iiwa_ik/constraints.h"
#include <gtest/gtest.h>
#include "drake/math/autodiff_gradient.h"

using drake::AutoDiffXd;
using drake::math::InitializeAutoDiff;
using drake::math::ExtractGradient;
using drake::math::ExtractValue;

class ConstraintsGradientTest : public ::testing::TestWithParam<std::tuple<bool, bool, bool, bool, IftSingularityHandling>> {
 protected:
  double grasp_distance = 0.6;
};

TEST_P(ConstraintsGradientTest, ReachableConstraintGradient) {
  auto [shoulder_up, elbow_up, wrist_up, use_ift, ift_handling] = GetParam();
  BimanualConfig config(shoulder_up, elbow_up, wrist_up, grasp_distance);
  IiwaBimanualReachableConstraint constraint(config);

  Eigen::VectorXd x(8);
  x << -0.4, -1.0, -1.1, 1.8, 0.4, 1.2, -2.5, 0.1;

  auto f = [&](const Eigen::VectorXd& x_in) {
    Eigen::VectorXd y_out(constraint.num_constraints());
    constraint.Eval(x_in, &y_out);
    return y_out;
  };

  double eps = 1e-7;
  Eigen::MatrixXd J_fd(constraint.num_constraints(), 8);
  for (int i = 0; i < 8; ++i) {
    Eigen::VectorXd x_p = x; x_p(i) += eps;
    Eigen::VectorXd x_m = x; x_m(i) -= eps;
    J_fd.col(i) = (f(x_p) - f(x_m)) / (2.0 * eps);
  }

  drake::AutoDiffVecXd x_ad = InitializeAutoDiff(x);
  drake::AutoDiffVecXd y_ad(constraint.num_constraints());
  constraint.Eval(x_ad, &y_ad);
  Eigen::MatrixXd J_ad = ExtractGradient(y_ad);

  EXPECT_TRUE(J_ad.isApprox(J_fd, 1e-4)) << "Gradient mismatch for ReachableConstraint in config: " 
                                         << shoulder_up << elbow_up << wrist_up;
}

TEST_P(ConstraintsGradientTest, JointLimitConstraintGradient) {
  auto [shoulder_up, elbow_up, wrist_up, use_ift, ift_handling] = GetParam();
  BimanualConfig config(shoulder_up, elbow_up, wrist_up, grasp_distance);
  AutoDiffConfig ad_config{ .use_ift = use_ift, .ift_handling = ift_handling };
  Eigen::VectorXd lb = Eigen::VectorXd::Constant(7, -M_PI);
  Eigen::VectorXd ub = Eigen::VectorXd::Constant(7, M_PI);
  IiwaBimanualJointLimitConstraint constraint(lb, ub, config, ad_config);

  Eigen::VectorXd x(8);
  x << -0.4, -1.0, -1.1, 1.8, 0.4, 1.2, -2.5, 0.1;

  auto f = [&](const Eigen::VectorXd& x_in) {
    Eigen::VectorXd y_out(constraint.num_constraints());
    constraint.Eval(x_in, &y_out);
    return y_out;
  };

  double eps = 1e-7;
  Eigen::MatrixXd J_fd(constraint.num_constraints(), 8);
  for (int i = 0; i < 8; ++i) {
    Eigen::VectorXd x_p = x; x_p(i) += eps;
    Eigen::VectorXd x_m = x; x_m(i) -= eps;
    J_fd.col(i) = (f(x_p) - f(x_m)) / (2.0 * eps);
  }

  drake::AutoDiffVecXd x_ad = InitializeAutoDiff(x);
  drake::AutoDiffVecXd y_ad(constraint.num_constraints());
  constraint.Eval(x_ad, &y_ad);
  Eigen::MatrixXd J_ad = ExtractGradient(y_ad);

  EXPECT_TRUE(J_ad.isApprox(J_fd, 1e-4)) << "Gradient mismatch for JointLimitConstraint in config: " 
                                         << shoulder_up << elbow_up << wrist_up;
}

TEST_P(ConstraintsGradientTest, PsiSingularityConstraintGradient) {
  auto [shoulder_up, elbow_up, wrist_up, use_ift, ift_handling] = GetParam();
  BimanualConfig config(shoulder_up, elbow_up, wrist_up, grasp_distance);
  AutoDiffConfig ad_config{ .use_ift = use_ift, .ift_handling = ift_handling };
  IiwaBimanualPsiSingularityConstraint constraint(config, ad_config);

  Eigen::VectorXd x(8);
  x << -0.4, -1.0, -1.1, 1.8, 0.4, 1.2, -2.5, 0.1;

  auto f = [&](const Eigen::VectorXd& x_in) {
    Eigen::VectorXd y_out(constraint.num_constraints());
    constraint.Eval(x_in, &y_out);
    return y_out;
  };

  double eps = 1e-7;
  Eigen::MatrixXd J_fd(constraint.num_constraints(), 8);
  for (int i = 0; i < 8; ++i) {
    Eigen::VectorXd x_p = x; x_p(i) += eps;
    Eigen::VectorXd x_m = x; x_m(i) -= eps;
    J_fd.col(i) = (f(x_p) - f(x_m)) / (2.0 * eps);
  }

  drake::AutoDiffVecXd x_ad = InitializeAutoDiff(x);
  drake::AutoDiffVecXd y_ad(constraint.num_constraints());
  constraint.Eval(x_ad, &y_ad);
  Eigen::MatrixXd J_ad = ExtractGradient(y_ad);

  EXPECT_TRUE(J_ad.isApprox(J_fd, 1e-4)) << "Gradient mismatch for PsiSingularityConstraint in config: " 
                                         << shoulder_up << elbow_up << wrist_up;
}

TEST_P(ConstraintsGradientTest, OldStyleReachableConstraintGradient) {
  auto [shoulder_up, elbow_up, wrist_up, use_ift, ift_handling] = GetParam();
  BimanualConfig config(shoulder_up, elbow_up, wrist_up, grasp_distance);
  AutoDiffConfig ad_config{ .use_ift = use_ift, .ift_handling = ift_handling };
  OldStyleReachableConstraint constraint(config, ad_config);

  Eigen::VectorXd x(8);
  x << -0.353, -0.957, -1.232, 1.976, 0.517, 1.387, -2.770, 0.012;

  auto f = [&](const Eigen::VectorXd& x_in) {
    Eigen::VectorXd y_out(constraint.num_constraints());
    constraint.Eval(x_in, &y_out);
    return y_out;
  };

  double eps = 1e-7;
  Eigen::MatrixXd J_fd(constraint.num_constraints(), 8);
  for (int i = 0; i < 8; ++i) {
    Eigen::VectorXd x_p = x; x_p(i) += eps;
    Eigen::VectorXd x_m = x; x_m(i) -= eps;
    J_fd.col(i) = (f(x_p) - f(x_m)) / (2.0 * eps);
  }

  drake::AutoDiffVecXd x_ad = InitializeAutoDiff(x);
  drake::AutoDiffVecXd y_ad(constraint.num_constraints());
  constraint.Eval(x_ad, &y_ad);
  Eigen::MatrixXd J_ad = ExtractGradient(y_ad);

  for (int i = 0; i < J_ad.rows(); ++i) {
    for (int j = 0; j < J_ad.cols(); ++j) {
      EXPECT_NEAR(J_ad(i, j), J_fd(i, j), 1e-3) 
          << "Gradient mismatch for OldStyleReachableConstraint at (" << i << "," << j << ") in config: " 
          << shoulder_up << elbow_up << wrist_up;
    }
  }
}


INSTANTIATE_TEST_SUITE_P(
    AllConfigs, ConstraintsGradientTest,
    ::testing::Combine(::testing::Bool(), ::testing::Bool(), ::testing::Bool(),
                       ::testing::Bool(),
                       ::testing::Values(IftSingularityHandling::kPseudoinverse,
                                         IftSingularityHandling::kZero)));
