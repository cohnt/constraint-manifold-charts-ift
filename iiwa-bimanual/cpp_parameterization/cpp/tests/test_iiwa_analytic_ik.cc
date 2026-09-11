#define _USE_MATH_DEFINES
#include "iiwa_ik/iiwa_analytic_ik.h"
#include <cmath>
#include <gtest/gtest.h>

TEST(IiwaAnalyticIkTest, ForwardKinematics) {
  Eigen::Matrix<double, 7, 1> q;
  q.setZero();

  Eigen::Matrix4d X = FK_IiwaDh<double>(q);

  // At q=0, the robot should be in a known configuration.
  // Base at z=0.36, joints add displacements.
  static const auto d = IiwaKinematicParameters::d();
  EXPECT_NEAR(X(2, 3), d(0) + d(2) + d(4) + d(6), 1e-6);
}

TEST(IiwaAnalyticIkTest, ComputePsi) {
  BimanualConfig config(true, true, true, 0.6);

  Eigen::Matrix<double, 7, 1> q;
  q << 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7;

  double psi = ComputePsi_Iiwa<double>(q, config);
  EXPECT_TRUE(std::isfinite(psi));
}

