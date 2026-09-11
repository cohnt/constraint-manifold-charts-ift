#define _USE_MATH_DEFINES
#include "iiwa_ik/iiwa_analytic_ik.h"
#include <cmath>
#include <memory>
#include <random>
#include <vector>
#include <gtest/gtest.h>
#include "drake/math/autodiff_gradient.h"
#include <drake/multibody/parsing/parser.h>
#include <drake/multibody/plant/multibody_plant.h>
#include <drake/multibody/tree/multibody_tree_indexes.h>
#include <drake/systems/framework/diagram_builder.h>

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
    Eigen::VectorXd fp = f(x_plus);
    Eigen::VectorXd fm = f(x_minus);
    if ((fp - fm).norm() < 1e-12) {
        // std::cout << "Warning: FD column " << i << " is zero!" << std::endl;
    }
    J.col(i) = (fp - fm) / (2.0 * epsilon);
  }
  return J;
}

TEST(NewtonHessianTest, JacobianMatchesFiniteDifference) {
  Eigen::Matrix<double, 7, 1> q;
  q << 0.1, -0.5, 0.3, 1.2, -0.8, 0.6, -0.3;

  auto f = [&](const Eigen::VectorXd& x) {
    drake::AutoDiffVecXd q_ad = InitializeAutoDiff(x);
    Eigen::Matrix4<AutoDiffXd> X = FK_IiwaDh<AutoDiffXd>(q_ad);
    Eigen::Matrix<AutoDiffXd, 6, 1> pose = Pose6FromMatrix4<AutoDiffXd>(X);
    return ExtractValue(pose);
  };
  Eigen::MatrixXd J_fd = ComputeFiniteDifference(f, q);

  drake::AutoDiffVecXd q_ad = InitializeAutoDiff(q);
  Eigen::Matrix4<AutoDiffXd> X = FK_IiwaDh<AutoDiffXd>(q_ad);
  Eigen::Matrix<AutoDiffXd, 6, 1> pose = Pose6FromMatrix4<AutoDiffXd>(X);
  Eigen::MatrixXd J_ad = ExtractGradient(pose);

  EXPECT_TRUE(J_ad.isApprox(J_fd, 1e-4));

  // Check analytic Jacobian (all 6 rows)
  Eigen::Matrix<double, 6, 7> J_analytic = ComputePoseJacobianAnalytic<double>(q);
  EXPECT_TRUE(J_ad.isApprox(J_analytic, 1e-6));
}

TEST(NewtonHessianTest, FullHessianMatchesFiniteDifference) {
  Eigen::Matrix<double, 7, 1> q;
  q << 0.1, -0.5, 0.3, 1.2, -0.8, 0.6, -0.3;

  auto get_jacobian_row = [&](const Eigen::VectorXd& q_in, int row_idx) {
    drake::AutoDiffVecXd q_ad = InitializeAutoDiff(q_in);
    Eigen::Matrix4<AutoDiffXd> X = FK_IiwaDh<AutoDiffXd>(q_ad);
    Eigen::Matrix<AutoDiffXd, 6, 1> pose = Pose6FromMatrix4<AutoDiffXd>(X);
    Eigen::MatrixXd J = ExtractGradient(pose);
    Eigen::VectorXd row = J.row(row_idx).transpose();
    return row;
  };

  auto H_analytic = ComputeKinematicHessians(q);

  for (int i = 0; i < 6; ++i) {
    auto f = [&](const Eigen::VectorXd& x) {
      return get_jacobian_row(x, i);
    };
    Eigen::MatrixXd H_fd = ComputeFiniteDifference(f, q);
    
    // Check symmetry
    EXPECT_TRUE(H_analytic[i].isApprox(H_analytic[i].transpose(), 1e-12))
        << "Hessian for component " << i << " is not symmetric.";
    
    // Check against FD
    double diff = (H_analytic[i] - H_fd).norm();
    EXPECT_TRUE(H_analytic[i].isApprox(H_fd, 5e-5)) 
        << "Hessian for component " << i << " mismatch. Norm diff: " << diff;
  }
}

TEST(NewtonHessianTest, FullNewtonMatchesReachable) {
  BimanualConfig config;
  AutoDiffConfig ad_config;
  ad_config.use_ift = true;
  ad_config.ift_handling = IftSingularityHandling::kFullNewton;
  ad_config.lambda = 1e-6;

  // Known reachable point
  Eigen::Matrix<double, 8, 1> x_val;
  x_val << -0.353, -0.957, -1.232, 1.976, 0.517, 1.387, -2.770, 0.5;

  drake::AutoDiffVecXd x_ad = InitializeAutoDiff(x_val);
  
  // Newton
  drake::AutoDiffVecXd q_newton = IiwaBimanualParameterization<AutoDiffXd>(
      x_ad, config, ad_config, nullptr);
  Eigen::MatrixXd J_newton = ExtractGradient(q_newton);

  // Reference (No IFT)
  drake::AutoDiffVecXd q_ref = IiwaBimanualParameterization<AutoDiffXd>(
      x_ad, config, AutoDiffConfig{ .use_ift = false }, nullptr);
  Eigen::MatrixXd J_ref = ExtractGradient(q_ref);

  EXPECT_TRUE(J_newton.isApprox(J_ref, 1e-4));
}



// The IFT damping strategies form J^T J + sum_i r_i H_i, which is only meaningful
// when the residual r, the rows of the analytic Jacobian, and the Hessian slices
// H[i] all index the same pose components in the same frame.
//
// Regression test for a bug where the residual came from PoseDistance, which
// returns body-frame [axis-angle; position], while J and H use world-frame
// [position; rpy] -- so rotation residuals were multiplied by position Hessians
// and vice versa. That error is invisible at reachable configurations (r ~ 0),
// which is why FullNewtonMatchesReachable did not catch it.
//
// Verifying that Pose6Residual is the differential of the chart J differentiates:
//   Pose6Residual(FK(q + dq), FK(q)) ~= ComputePoseJacobianAnalytic(q) * dq
TEST(NewtonHessianTest, ResidualChartMatchesJacobian) {
  Eigen::Matrix<double, 7, 1> q;
  q << 0.3, 0.8, -0.4, 1.1, 0.2, -0.7, 0.5;

  const Eigen::Matrix<double, 6, 7> J = ComputePoseJacobianAnalytic<double>(q);
  const Eigen::Matrix4d X0 = FK_IiwaDh<double>(q);

  // Probe every joint direction so a swapped block cannot cancel out.
  for (int j = 0; j < 7; ++j) {
    Eigen::Matrix<double, 7, 1> dq = Eigen::Matrix<double, 7, 1>::Zero();
    dq(j) = 1e-6;

    const Eigen::Matrix4d X1 = FK_IiwaDh<double>(
        (q + dq).eval());
    const Eigen::Matrix<double, 6, 1> residual = Pose6Residual<double>(X1, X0);
    const Eigen::Matrix<double, 6, 1> predicted = J * dq;

    EXPECT_LT((residual - predicted).cwiseAbs().maxCoeff(), 1e-9)
        << "residual chart disagrees with the analytic Jacobian along joint " << j
        << "\n  residual:  " << residual.transpose()
        << "\n  J * dq:    " << predicted.transpose();
  }
}

// ---------------------------------------------------------------------------
// Cross-check of the analytic kinematic-Hessian path against Drake.
//
// The repo has one analytic path for the derivative of the geometric Jacobian
// (ComputeGeometricJacobianDerivatives) and it is now used by every boundary
// reachability evaluation. Drake's AutoDiffXd plant computes the same quantity
// independently, so it is the natural oracle. Measured: the two agree to ~1e-15,
// and the analytic path is ~34x faster (2.1 us vs 70.8 us per call), which is why
// the analytic one is the path kept.
// ---------------------------------------------------------------------------


namespace {

// Builds a single welded IIWA whose link 7 frame coincides with our DH chain.
// (Verified separately: ComputePoseJacobianGeometric matches Drake's Jacobian for
// iiwa_link_7 to 5.6e-16.)
std::unique_ptr<drake::multibody::MultibodyPlant<double>> BuildIiwaPlant() {
  auto plant = std::make_unique<drake::multibody::MultibodyPlant<double>>(0.0);
  drake::multibody::Parser parser(plant.get());
  parser.package_map().AddPackageXml(std::string(IIWA_IK_REPO_DIR) +
                                     "/package.xml");
  parser.AddModels(std::string(IIWA_IK_REPO_DIR) +
                   "/models/iiwa14_convex_decimated_collision.urdf");
  plant->WeldFrames(plant->world_frame(), plant->GetFrameByName("base"));
  plant->Finalize();
  return plant;
}

}  // namespace

TEST(NewtonHessianTest, BoundaryJacobianDerivativesMatchDrake) {
  auto plant = BuildIiwaPlant();
  auto plant_ad = drake::systems::System<double>::ToAutoDiffXd(*plant);
  auto context_ad = plant_ad->CreateDefaultContext();
  const auto& frame_E = plant_ad->GetFrameByName("iiwa_link_7");
  const auto& frame_W = plant_ad->world_frame();

  std::mt19937 gen(0);
  std::uniform_real_distribution<double> dist(-2.0, 2.0);

  for (int trial = 0; trial < 5; ++trial) {
    Eigen::Matrix<double, 7, 1> q;
    for (int i = 0; i < 7; ++i) q(i) = dist(gen);

    // Ours: analytic.
    const std::vector<Eigen::Matrix<double, 6, 7>> ours =
        ComputeGeometricJacobianDerivatives(q, 0.0);

    // Drake: autodiff through the plant's Jacobian.
    const drake::AutoDiffVecXd q_ad = InitializeAutoDiff(Eigen::VectorXd(q));
    plant_ad->SetPositions(context_ad.get(), q_ad);
    Eigen::Matrix<drake::AutoDiffXd, 6, Eigen::Dynamic> J_ad(6, 7);
    const Eigen::Vector3<drake::AutoDiffXd> p_zero =
        Eigen::Vector3<drake::AutoDiffXd>::Zero();
    plant_ad->CalcJacobianSpatialVelocity(
        *context_ad, drake::multibody::JacobianWrtVariable::kV, frame_E, p_zero,
        frame_W, frame_W, &J_ad);

    for (int k = 0; k < 7; ++k) {
      Eigen::Matrix<double, 6, 7> theirs_k;
      for (int r = 0; r < 6; ++r) {
        for (int c = 0; c < 7; ++c) {
          const Eigen::VectorXd& d = J_ad(r, c).derivatives();
          theirs_k(r, c) = (d.size() > 0) ? d(k) : 0.0;
        }
      }
      EXPECT_LT((ours[k] - theirs_k).cwiseAbs().maxCoeff(), 1e-10)
          << "analytic dJ/dq_" << k << " disagrees with Drake's AutoDiff plant"
          << " on trial " << trial;
    }
  }
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
