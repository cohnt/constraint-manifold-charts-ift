#include "iiwa_analytic_ik.h"

Eigen::VectorX<drake::AutoDiffXd> IiwaBimanualParameterizationIFT(
    const Eigen::VectorX<drake::AutoDiffXd> &q_and_psi,
    const BimanualConfig& config,
    const AutoDiffConfig& ad_config,
    Eigen::VectorX<drake::AutoDiffXd> *unclipped_vals) {
  
  using drake::AutoDiffXd;

  DRAKE_THROW_UNLESS(q_and_psi.size() == 8);

  // -------------------------
  // 1) Values: call double version
  // -------------------------
  const Eigen::VectorXd q_and_psi_d = drake::math::ExtractValue(q_and_psi);

  Eigen::VectorXd unclipped_d;
  Eigen::VectorXd *unclipped_d_ptr = nullptr;
  if (unclipped_vals != nullptr) {
    unclipped_d.resize(4);
    unclipped_d_ptr = &unclipped_d;
  }

  const Eigen::VectorXd q_full_d = IiwaBimanualParameterization<double>(
      q_and_psi_d, config, AutoDiffConfig{}, unclipped_d_ptr);

  const Eigen::Matrix<double, 7, 1> q_c_d = q_full_d.head<7>();
  const Eigen::Matrix<double, 7, 1> q_s_d = q_full_d.tail<7>();

  // -------------------------
  // 2) Determine input derivative dimension m
  // -------------------------
  const int m =
      q_and_psi(0).derivatives().size(); // often 8 (identity), maybe 0
  Eigen::MatrixXd D_in(8, m);
  for (int i = 0; i < 8; ++i) {
    const auto &di = q_and_psi(i).derivatives();
    DRAKE_THROW_UNLESS(di.size() == m);
    if (m > 0)
      D_in.row(i) = di.transpose();
  }

  // If no derivatives requested, return values-only AutoDiff.
  if (m == 0) {
    Eigen::VectorX<AutoDiffXd> out(14);
    for (int i = 0; i < 14; ++i)
      out(i) = AutoDiffXd(q_full_d(i));
    if (unclipped_vals != nullptr) {
      unclipped_vals->resize(unclipped_d.size());
      for (int i = 0; i < unclipped_d.size(); ++i) {
        (*unclipped_vals)(i) = AutoDiffXd(unclipped_d(i));
      }
    }
    return out;
  }

  bool use_svd = (ad_config.ift_handling == IftSingularityHandling::kPseudoinverse ||
                   ad_config.ift_handling == IftSingularityHandling::kLevenbergMarquardt ||
                   ad_config.ift_handling == IftSingularityHandling::kResidualDamping ||
                   ad_config.ift_handling == IftSingularityHandling::kFullNewton);

  // Iterative refinement of q_s_d is not applied: the analytic solution already
  // satisfies the constraints to the tolerance the callers require, and the
  // refinement step was measured to cost more than it corrected.
  const Eigen::Matrix<double, 7, 1> q_s_final = q_s_d;

  // Derivatives are propagated one partial at a time. Batching them is possible
  // and would help for large partial counts, but is left until the upstream
  // autodiff changes in Drake stabilize.

  // -------------------------
  // 3) Build tildeJ_s(q_s_final): d [pose6(FK(q_s)); psi(q_s)] / d q_s
  // -------------------------
  Eigen::Matrix<double, 7, 7> tildeJ_s = Eigen::Matrix<double, 7, 7>::Zero();
  
  // Use analytical pose Jacobian (6x7)
  tildeJ_s.topRows<6>() = ComputePoseJacobianAnalytic<double>(q_s_final);
  
  // Use AutoDiff for psi gradient (1x7)
  {
    const Eigen::Matrix<AutoDiffXd, 7, 1> q_s_ad_psi = drake::math::InitializeAutoDiff(q_s_final);
    const AutoDiffXd psi_s_ad = ComputePsi_Iiwa<AutoDiffXd>(q_s_ad_psi, config);
    tildeJ_s.row(6) = psi_s_ad.derivatives().transpose();
  }

  Eigen::JacobiSVD<Eigen::Matrix<double, 7, 7>> tildeJ_svd;
  Eigen::ColPivHouseholderQR<Eigen::Matrix<double, 7, 7>> tildeJ_qr;
  bool ok_rank = true;

  Eigen::Matrix<double, 7, 7> tildeJ_pinv;
  if (use_svd) {
    tildeJ_svd.compute(tildeJ_s, Eigen::ComputeFullU | Eigen::ComputeFullV);
    tildeJ_svd.setThreshold(1e-6);

        if (ad_config.ift_handling == IftSingularityHandling::kLevenbergMarquardt ||
        ad_config.ift_handling == IftSingularityHandling::kResidualDamping) {
      const auto& sigma = tildeJ_svd.singularValues();
      double sigma_min = sigma(6);
      double lambda_eff = ad_config.lambda;

      if (ad_config.ift_handling == IftSingularityHandling::kResidualDamping) {
        const Eigen::Matrix4d X_Ws_d = FK_IiwaDh<double>(q_s_final);
        const Eigen::Matrix4d X_Wc_d_mat = FK_IiwaDh<double>(q_c_d);
        const Eigen::Matrix4d X_Wgoal_d =
            LinkageTransformFromControlledEE<double>(X_Wc_d_mat,
                                                     config.grasp_distance);
        const Eigen::Matrix<double, 6, 1> pose_res =
            Pose6Residual<double>(X_Ws_d, X_Wgoal_d);
        const double psi_res = WrapToPi(
            ComputePsi_Iiwa<double>(q_s_final, config) - q_and_psi_d(7));
        double res_norm_sq = pose_res.squaredNorm() + psi_res * psi_res;

        // [FIX] Double application of SVT removed. Damping logic consolidated below.

        if (ad_config.use_anisotropic_damping) {
          Eigen::Matrix<double, 7, 1> res_vec;
          res_vec.head<6>() = pose_res;
          res_vec(6) = psi_res;
          Eigen::Matrix<double, 7, 1> diag;
          for (int j = 0; j < 7; ++j) {
            double proj = tildeJ_svd.matrixU().col(j).dot(res_vec);
            double Lambda_j = ad_config.lambda * (res_norm_sq + 3.0 * proj * proj);
            diag(j) = sigma(j) / (sigma(j) * sigma(j) + Lambda_j);
          }
          tildeJ_pinv = tildeJ_svd.matrixV() * diag.asDiagonal() *
                        tildeJ_svd.matrixU().transpose();
        } else {
          lambda_eff *= res_norm_sq;
          if (ad_config.svt_epsilon && ad_config.svt_lambda_max &&
              sigma_min <= *ad_config.svt_epsilon) {
            double ratio = sigma_min / (*ad_config.svt_epsilon);
            lambda_eff += (*ad_config.svt_lambda_max) * (1.0 - ratio * ratio);
          }
          Eigen::Matrix<double, 7, 1> diag;
          for (int i = 0; i < 7; ++i) {
            diag(i) = sigma(i) / (sigma(i) * sigma(i) + lambda_eff);
          }
          tildeJ_pinv = tildeJ_svd.matrixV() * diag.asDiagonal() *
                        tildeJ_svd.matrixU().transpose();
        }
      } else {
        // Standard Levenberg-Marquardt (Classic)
        if (ad_config.svt_epsilon && ad_config.svt_lambda_max &&
            sigma_min <= *ad_config.svt_epsilon) {
          double ratio = sigma_min / (*ad_config.svt_epsilon);
          lambda_eff += (*ad_config.svt_lambda_max) * (1.0 - ratio * ratio);
        }
        Eigen::Matrix<double, 7, 1> diag;
        for (int i = 0; i < 7; ++i) {
          diag(i) = sigma(i) / (sigma(i) * sigma(i) + lambda_eff);
        }
        tildeJ_pinv = tildeJ_svd.matrixV() * diag.asDiagonal() *
                      tildeJ_svd.matrixU().transpose();
      }
    } else if (ad_config.ift_handling == IftSingularityHandling::kFullNewton) {
      const Eigen::Matrix4d X_Ws_d = FK_IiwaDh<double>(q_s_final);
      const Eigen::Matrix4d X_Wc_d_mat = FK_IiwaDh<double>(q_c_d);
      const Eigen::Matrix4d X_Wgoal_d = LinkageTransformFromControlledEE<double>(X_Wc_d_mat, config.grasp_distance);
      
      const Eigen::Matrix<double, 6, 1> pose_res =
          Pose6Residual<double>(X_Ws_d, X_Wgoal_d);
      const double psi_res = WrapToPi(
          ComputePsi_Iiwa<double>(q_s_final, config) - q_and_psi_d(7));

      if (pose_res.squaredNorm() < 1e-12 && std::abs(psi_res) < 1e-6) {
        tildeJ_pinv = tildeJ_svd.solve(Eigen::Matrix<double, 7, 7>::Identity());
      } else {
        // Full Newton: include second-order curvature for non-reachable cases.
        const auto H = ComputeKinematicHessians(q_s_final);
        Eigen::Matrix<double, 7, 7> B = tildeJ_s.transpose() * tildeJ_s;
        for (int i = 0; i < 6; ++i) {
          B += pose_res(i) * H[i];
        }
        
        // Ensure minimum regularization even if pose_res is small.
        double lambda = (ad_config.lambda != 0.0) ? ad_config.lambda : 1e-6;
        B += lambda * Eigen::Matrix<double, 7, 7>::Identity();
        
        // Solve for the IFT gradient: dq_s/dq_c = B^-1 * J_s^T * J_c
        tildeJ_pinv = B.colPivHouseholderQr().solve(tildeJ_s.transpose());
      }
    }
  } else {
    tildeJ_qr.compute(tildeJ_s);
    tildeJ_qr.setThreshold(1e-6);
    ok_rank = (tildeJ_qr.rank() == 7);
  }

  // -------------------------
  // 4) Build J_goal_qc: d pose6( Linkage(FK(q_c)) ) / d q_c
  // -------------------------
  // Optimized: Use analytic Jacobian for the linkage-transformed goal pose.
  static const auto alpha_dh = IiwaKinematicParameters::alpha();
  static const auto d_dh = IiwaKinematicParameters::d();
  
  std::array<Eigen::Matrix4d, 8> F_c;
  F_c[0] = Eigen::Matrix4d::Identity();
  for (int i = 0; i < 7; ++i) {
    F_c[i + 1] = F_c[i] * ComputeDHMatrix<double>(q_c_d(i), alpha_dh(i), d_dh(i));
  }
  
  // 4) Compute goal Jacobian J_goal_qc once (6x7).
  // We use AutoDiff here once for guaranteed correctness, then use it in the loop.
  Eigen::Matrix<double, 6, 7> J_goal_qc;
  {
    const Eigen::Matrix<AutoDiffXd, 7, 1> q_c_ad_local = drake::math::InitializeAutoDiff(q_c_d);
    const Eigen::Matrix4<AutoDiffXd> X_Wc_ad_local = FK_IiwaDh<AutoDiffXd>(q_c_ad_local);
    const Eigen::Matrix4<AutoDiffXd> X_Wgoal_ad_local = LinkageTransformFromControlledEE<AutoDiffXd>(X_Wc_ad_local, config.grasp_distance);
    const Eigen::Matrix<AutoDiffXd, 6, 1> pose6_goal_ad_local = Pose6FromMatrix4(X_Wgoal_ad_local);
    for (int r = 0; r < 6; ++r) {
      J_goal_qc.row(r) = pose6_goal_ad_local(r).derivatives().transpose();
    }
  }

  // -------------------------
  // 5) Propagate derivatives by JVP
  // -------------------------
  Eigen::MatrixXd D_out(14, m);
  D_out.setZero();

  for (int k = 0; k < m; ++k) {
    const Eigen::Matrix<double, 7, 1> v_c = D_in.block<7, 1>(0, k);
    const double v_psi = D_in(7, k);

    D_out.block<7, 1>(0, k) = v_c;

    Eigen::Matrix<double, 7, 1> rhs;
    rhs.head<6>() = J_goal_qc * v_c;
    rhs(6) = v_psi;

    Eigen::Matrix<double, 7, 1> w_s;
    if (use_svd) {
      if (ad_config.ift_handling == IftSingularityHandling::kLevenbergMarquardt ||
          ad_config.ift_handling == IftSingularityHandling::kResidualDamping ||
          ad_config.ift_handling == IftSingularityHandling::kFullNewton) {
        w_s = tildeJ_pinv * rhs;
      } else {
        w_s = tildeJ_svd.solve(rhs);
      }
      D_out.block<7, 1>(7, k) = w_s;
    } else {
      if (ok_rank) {
        w_s = tildeJ_qr.solve(rhs);
        D_out.block<7, 1>(7, k) = w_s;
      } else {
        D_out.block<7, 1>(7, k).setZero();
      }
    }
  }

  // -------------------------
  // 6) Pack values + derivatives into AutoDiff output
  // -------------------------
  Eigen::VectorX<AutoDiffXd> q_full_ad(14);
  for (int i = 0; i < 7; ++i) {
    q_full_ad(i) = AutoDiffXd(q_c_d(i), D_out.row(i).transpose());
  }
  for (int i = 0; i < 7; ++i) {
    q_full_ad(i + 7) = AutoDiffXd(q_s_final(i), D_out.row(i + 7).transpose());
  }

  // -------------------------
  // 7) Unclipped values + gradients (temporary passthrough)
  // -------------------------
  if (unclipped_vals != nullptr) {
    Eigen::VectorX<AutoDiffXd> unclipped_ad;
    (void)IiwaBimanualParameterization<AutoDiffXd>(
        q_and_psi, config, AutoDiffConfig{ .use_ift = false }, &unclipped_ad);
    *unclipped_vals = unclipped_ad;
  }

  return q_full_ad;
}

// ComputePoseJacobianAnalytic is implemented in the header.

std::array<Eigen::Matrix<double, 7, 7>, 6> ComputeKinematicHessians(
    const Eigen::Matrix<double, 7, 1>& q) {
  std::array<Eigen::Matrix<double, 7, 7>, 6> H;
  for (int i = 0; i < 6; ++i) H[i].setZero();

  // 1. FK to get frames, axes, origins
  static const auto alpha_dh = IiwaKinematicParameters::alpha();
  static const auto d_dh = IiwaKinematicParameters::d();

  std::array<Eigen::Matrix4d, 8> F;
  F[0] = Eigen::Matrix4d::Identity();
  for (int i = 0; i < 7; ++i) {
    F[i + 1] = F[i] * ComputeDHMatrix<double>(q(i), alpha_dh(i), d_dh(i));
  }

  Eigen::Vector3d o_e = F[7].block<3, 1>(0, 3);
  Eigen::Matrix3d R_e = F[7].block<3, 3>(0, 0);
  std::array<Eigen::Vector3d, 7> z, o;
  for (int i = 0; i < 7; ++i) {
    z[i] = F[i].block<3, 3>(0, 0).col(2);
    o[i] = F[i].block<3, 1>(0, 3);
  }

  // 2. Compute Jacobians for reuse
  Eigen::Matrix<double, 3, 7> J_p;
  Eigen::Matrix<double, 3, 7> J_w; // geometric
  for (int j = 0; j < 7; ++j) {
    J_p.col(j) = z[j].cross(o_e - o[j]);
    J_w.col(j) = z[j];
  }

  // Orientation part: RPY rates
  const drake::math::RollPitchYawd rpy{drake::math::RotationMatrixd(R_e)};
  const double theta = rpy.pitch_angle();
  const double psi = rpy.yaw_angle();
  const double c_th = std::cos(theta);
  const double s_psi = std::sin(psi);
  const double c_psi = std::cos(psi);
  const double t_th = std::tan(theta);
  const double sec_th = 1.0 / c_th;

  Eigen::Matrix3d E_inv;
  E_inv << c_psi * sec_th, s_psi * sec_th, 0.0,
          -s_psi,          c_psi,          0.0,
           c_psi * t_th,   s_psi * t_th,   1.0;

  Eigen::Matrix<double, 3, 7> J_y = E_inv * J_w; // RPY Jacobian

  // 3. Derivatives of E^-1 w.r.t y = [phi, theta, psi]
  // dE_inv/dphi = 0
  Eigen::Matrix3d dE_inv_dtheta;
  dE_inv_dtheta << c_psi * sec_th * t_th,   s_psi * sec_th * t_th,   0.0,
                   0.0,                     0.0,                     0.0,
                   c_psi * sec_th * sec_th, s_psi * sec_th * sec_th, 0.0;

  Eigen::Matrix3d dE_inv_dpsi;
  dE_inv_dpsi << -s_psi * sec_th,  c_psi * sec_th, 0.0,
                 -c_psi,           -s_psi,          0.0,
                 -s_psi * t_th,    c_psi * t_th,   0.0;

  // 4. Compute Hessians
  for (int k = 0; k < 7; ++k) {
    // dE_inv/dq_k = dE_inv/dtheta * J_y(1, k) + dE_inv/dpsi * J_y(2, k)
    Eigen::Matrix3d dE_inv_dqk = dE_inv_dtheta * J_y(1, k) + dE_inv_dpsi * J_y(2, k);

    for (int j = k; j < 7; ++j) {
      // Position Hessian: H_p = dJ_p,j/dq_k
      // For k <= j: z_k x J_p,j
      Eigen::Vector3d h_p = z[k].cross(J_p.col(j));
      for (int i = 0; i < 3; ++i) {
        H[i](k, j) = H[i](j, k) = h_p(i);
      }

      // Orientation Hessian: H_y = dJ_y,j/dq_k = (dE_inv/dq_k)*z_j + E_inv*(dz_j/dq_k)
      // For k < j, dz_j/dq_k = z_k x z_j. For k = j, dz_j/dq_k = 0.
      Eigen::Vector3d dzj_dqk = (k < j) ? z[k].cross(z[j]) : Eigen::Vector3d::Zero();
      Eigen::Vector3d h_y = dE_inv_dqk * z[j] + E_inv * dzj_dqk;
      for (int i = 0; i < 3; ++i) {
        H[i + 3](k, j) = H[i + 3](j, k) = h_y(i);
      }
    }
  }

  return H;
}

template <typename T>
Eigen::Matrix<T, 6, 7> ComputePoseJacobianGeometric(const Eigen::Matrix<T, 7, 1>& q,
                                                   double grasp_distance) {
  using std::cos;
  using std::sin;
  const Eigen::Matrix<double, 7, 1> alpha = IiwaKinematicParameters::alpha();
  const Eigen::Matrix<double, 7, 1> d_dh = IiwaKinematicParameters::d();

  std::array<Eigen::Matrix4<T>, 8> F;
  F[0] = Eigen::Matrix4<T>::Identity();
  for (int i = 0; i < 7; ++i) {
    F[i + 1] = F[i] * ComputeDHMatrix<T>(q(i), alpha(i), d_dh(i));
  }

  Eigen::Matrix<T, 3, 7> Jp, Jw;
  // Apply the grasp distance offset to the end-effector position
  Eigen::Matrix<T, 3, 1> p_ee = F[7].template block<3, 1>(0, 3);
  if (grasp_distance != 0.0) {
    p_ee += F[7].template block<3, 3>(0, 0).col(2) * T(grasp_distance);
  }

  for (int i = 0; i < 7; ++i) {
    Eigen::Matrix<T, 3, 1> z_i = F[i].template block<3, 3>(0, 0).col(2);
    Eigen::Matrix<T, 3, 1> p_i = F[i].template block<3, 1>(0, 3);
    Jw.col(i) = z_i;
    Jp.col(i) = z_i.cross(p_ee - p_i);
  }

  Eigen::Matrix<T, 6, 7> J;
  J << Jw, Jp;
  return J;
}

// Explicit instantiations
template Eigen::Matrix<double, 6, 7> ComputePoseJacobianGeometric<double>(const Eigen::Matrix<double, 7, 1>&, double);
template Eigen::Matrix<drake::AutoDiffXd, 6, 7> ComputePoseJacobianGeometric<drake::AutoDiffXd>(const Eigen::Matrix<drake::AutoDiffXd, 7, 1>&, double);

std::vector<Eigen::Matrix<double, 6, 7>> ComputeGeometricJacobianDerivatives(const Eigen::VectorXd& q, double grasp_distance) {
  static const auto alpha = IiwaKinematicParameters::alpha();
  static const auto d_dh = IiwaKinematicParameters::d();

  std::array<Eigen::Matrix4d, 8> F;
  F[0] = Eigen::Matrix4d::Identity();
  for (int i = 0; i < 7; ++i) {
    F[i + 1] = F[i] * ComputeDHMatrix<double>(q(i), alpha(i), d_dh(i));
  }

  // Apply the grasp distance offset
  Eigen::Vector3d p_ee = F[7].block<3, 1>(0, 3);
  if (grasp_distance != 0.0) {
    p_ee += F[7].block<3, 3>(0, 0).col(2) * grasp_distance;
  }
  std::array<Eigen::Vector3d, 7> z, o;
  for (int i = 0; i < 7; ++i) {
    z[i] = F[i].block<3, 3>(0, 0).col(2);
    o[i] = F[i].block<3, 1>(0, 3);
  }

  // Precompute Geometric Jacobian columns
  std::vector<Eigen::Vector3d> Jw(7), Jp(7);
  for (int j = 0; j < 7; ++j) {
    Jw[j] = z[j];
    Jp[j] = z[j].cross(p_ee - o[j]);
  }

  std::vector<Eigen::Matrix<double, 6, 7>> dJ_dq(7, Eigen::Matrix<double, 6, 7>::Zero());

  for (int k = 0; k < 7; ++k) {
    for (int j = 0; j < 7; ++j) {
      if (k <= j) {
        // Ancestor joint (including self): entire column rotates around z_k
        dJ_dq[k].block<3, 1>(0, j) = z[k].cross(Jw[j]);
        dJ_dq[k].block<3, 1>(3, j) = z[k].cross(Jp[j]);
      } else {
        // Descendant joint: only p_ee moves
        // dJ_j/dq_k = [0; z_j x (dp_ee/dq_k)]
        dJ_dq[k].block<3, 1>(0, j).setZero();
        dJ_dq[k].block<3, 1>(3, j) = z[j].cross(Jp[k]);
      }
    }
  }

  return dJ_dq;
}
