#include "iiwa_ik/costs.h"
#include <gtest/gtest.h>
#include "drake/math/autodiff_gradient.h"

using drake::AutoDiffXd;
using drake::math::InitializeAutoDiff;
using drake::math::ExtractGradient;
using drake::math::ExtractValue;

class CostsTest : public ::testing::TestWithParam<std::tuple<bool, bool, bool, bool, IftSingularityHandling>> {
 protected:
  double grasp_distance = 0.6;
};

TEST_P(CostsTest, PathCostEval) {
  auto [shoulder_up, elbow_up, wrist_up, use_ift, ift_handling] = GetParam();
  int num_vars = 8;
  int num_waypoints = 2;
  BimanualConfig config(shoulder_up, elbow_up, wrist_up, grasp_distance);
  AutoDiffConfig ad_config{ .use_ift = use_ift, .ift_handling = ift_handling };
  bool square = true;

  IiwaBimanualPathCost cost(num_vars, num_waypoints, config, ad_config, square);

  // x = [q_c1, psi1, q_c2, psi2] (16D for 2 positions)
  Eigen::VectorXd x(16);
  x << -0.4, -1.0, -1.1, 1.8, 0.4, 1.2, -2.5, 0.1,
       -0.4, -1.0, -1.1, 1.8, 0.4, 1.2, -2.5, 0.1;

  Eigen::VectorXd y(1);
  cost.Eval(x, &y);

  EXPECT_EQ(y.size(), 1);
  EXPECT_TRUE(std::isfinite(y(0)));
  // Cost should be 0 for identical points
  EXPECT_NEAR(y(0), 0.0, 1e-10);
}

TEST_P(CostsTest, PathCostGradient) {
  auto [shoulder_up, elbow_up, wrist_up, use_ift, ift_handling] = GetParam();
  int num_vars = 8;
  int num_waypoints = 2;
  BimanualConfig config(shoulder_up, elbow_up, wrist_up, grasp_distance);
  AutoDiffConfig ad_config{ .use_ift = use_ift, .ift_handling = ift_handling };
  bool square = true;

  IiwaBimanualPathCost cost(num_vars, num_waypoints, config, ad_config, square);


  Eigen::VectorXd x(16);
  // Slightly different points to have non-zero gradient
  x << -0.4, -1.0, -1.1, 1.8, 0.4, 1.2, -2.5, 0.1,
       -0.380, -0.980, -1.080, 1.780, 0.420, 1.180, -2.480, 0.120;

  auto f = [&](const Eigen::VectorXd& x_in) {
    Eigen::VectorXd y_out(1);
    cost.Eval(x_in, &y_out);
    return y_out;
  };

  // Finite difference
  double eps = 1e-7;
  Eigen::MatrixXd J_fd(1, 16);
  for (int i = 0; i < 16; ++i) {
    Eigen::VectorXd x_p = x; x_p(i) += eps;
    Eigen::VectorXd x_m = x; x_m(i) -= eps;
    J_fd(0, i) = (f(x_p)(0) - f(x_m)(0)) / (2.0 * eps);
  }

  // AutoDiff
  drake::AutoDiffVecXd x_ad = InitializeAutoDiff(x);
  drake::AutoDiffVecXd y_ad(1);
  cost.Eval(x_ad, &y_ad);
  Eigen::MatrixXd J_ad = ExtractGradient(y_ad);

  EXPECT_TRUE(J_ad.isApprox(J_fd, 1e-3)) << "Gradient mismatch for config: "
                                         << shoulder_up << elbow_up << wrist_up
                                         << " use_ift=" << use_ift
                                         << " ift_handling=" << (int)ift_handling;
}

// The flat decision vector is read as a (num_positions x num_control_points)
// matrix laid out column-major, so it holds one whole control point at a time.
// Callers assemble that vector themselves -- from Python, via
// `control_points().flatten(order='F')` -- and the row-major alternative is
// silently accepted, evaluating to a finite number that is the energy of a
// scrambled path. Pin the layout against a hand-computed reference so a caller
// that gets it wrong is caught here rather than in a diverging trajectory.
// The waypoint count is deliberately not 8, so a transposed layout is a
// different shape and cannot coincidentally agree.
TEST(CostsLayoutTest, FlatVectorIsControlPointMajor) {
  const int num_positions = 8;
  const int num_waypoints = 3;
  BimanualConfig config(true, true, false, 0.6);
  AutoDiffConfig ad_config{};

  Eigen::Matrix<double, 8, 3> control_points;
  control_points.col(0) << -0.40, -1.00, -1.10, 1.80, 0.40, 1.20, -2.50, 0.10;
  control_points.col(1) << -0.34, -0.93, -1.19, 1.86, 0.37, 1.15, -2.44, 0.16;
  control_points.col(2) << -0.27, -0.88, -1.25, 1.90, 0.31, 1.11, -2.39, 0.23;

  // Reference: the energy of the path the control points actually describe,
  // measured in the full 14-D configuration space.
  double expected = 0.0;
  for (int i = 1; i < num_waypoints; ++i) {
    const Eigen::VectorXd q0 = IiwaBimanualParameterization<double>(
        control_points.col(i - 1), config, ad_config, nullptr);
    const Eigen::VectorXd q1 = IiwaBimanualParameterization<double>(
        control_points.col(i), config, ad_config, nullptr);
    expected += (q1 - q0).squaredNorm();
  }
  ASSERT_GT(expected, 1e-6) << "Degenerate reference path; pick distinct points.";

  IiwaBimanualPathCost cost(num_positions, num_waypoints, config, ad_config,
                            /* square = */ true);

  // Column-major: [cp0(8), cp1(8), cp2(8)].
  Eigen::VectorXd x_col_major(num_positions * num_waypoints);
  for (int i = 0; i < num_waypoints; ++i) {
    x_col_major.segment(i * num_positions, num_positions) = control_points.col(i);
  }
  Eigen::VectorXd y(1);
  cost.Eval(x_col_major, &y);
  EXPECT_NEAR(y(0), expected, 1e-9);

  // Row-major: [joint0 across all control points, joint1 across all, ...].
  // This is what numpy's default flatten produces, and it must not be mistaken
  // for a valid encoding of the same path.
  Eigen::VectorXd x_row_major(num_positions * num_waypoints);
  for (int j = 0; j < num_positions; ++j) {
    x_row_major.segment(j * num_waypoints, num_waypoints) =
        control_points.row(j).transpose();
  }
  cost.Eval(x_row_major, &y);
  EXPECT_GT(std::abs(y(0) - expected), 1.0)
      << "Row-major input agreed with the column-major reference, so this test "
         "can no longer detect a transposed decision vector.";
}

INSTANTIATE_TEST_SUITE_P(
    AllConfigs, CostsTest,
    ::testing::Combine(::testing::Bool(), ::testing::Bool(), ::testing::Bool(),
                       ::testing::Bool(),
                       ::testing::Values(IftSingularityHandling::kPseudoinverse,
                                         IftSingularityHandling::kZero)));
