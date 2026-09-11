#pragma once

#include <algorithm>
#include <cmath>
#include <optional>
#include <type_traits>

#include <Eigen/Dense>

#include "drake/math/autodiff.h"
#include "drake/math/rotation_matrix.h"
#include "drake/math/roll_pitch_yaw.h"

// Using declarations (optional, but clearer)
using std::atan2;
using std::cos;
using std::max;
using std::min;
using std::sin;

/** Configuration for the bimanual IIWA setup. */
struct BimanualConfig {
  bool shoulder_up{true};
  bool elbow_up{true};
  bool wrist_up{false};
  double grasp_distance{0.6};
  double clipping_margin = 1e-4;
  double clipping_margin_psi = 1e-4;
  // Boundary reachability parameters. Both are DERIVED from kinematics by
  // scripts/analysis/calibrate_boundary_threshold.py -- never tuned against
  // downstream success rate, which drives tau upward until the constraint stops
  // constraining (the previous tau = 60.0 rejected 0% of unreachable samples).
  //   tau  = 2.46 from velocity amplification: sigma* = v_task / qdot_max with
  //          v_task = 0.1 m/s and qdot_max = 1.309 rad/s.
  //   eps  = 1e-6, bracketed by the median sigma_min^2 (1.3e-2) above and the
  //          float floor sigma_max^2 * eps_machine (3.1e-15) below.
  double boundary_threshold = 2.46;
  double boundary_epsilon = 1e-6;
  // Length scale L (metres) used to make the boundary constraint unit-coherent.
  // The geometric Jacobian mixes angular rows (rad/s) with linear rows (m/s), so
  // det(J J^T) has mixed units and no single boundary_epsilon is meaningful.
  // Scaling the angular rows by L puts every entry in metres. L is the maximum
  // |p_ee| over the joint-limit box; 1.86 m for the IIWA-14 at grasp_distance 0.6.
  // Appended last on purpose: callers construct BimanualConfig with nine
  // positional arguments, and reordering would silently change their meaning.
  double boundary_length_scale = 1.86;

  int GC2() const { return shoulder_up ? 1 : -1; }
  int GC4() const { return elbow_up ? 1 : -1; }
  int GC6() const { return wrist_up ? 1 : -1; }
};

enum class IftSingularityHandling {
  kPseudoinverse,
  kZero,
  kLevenbergMarquardt,
  kResidualDamping,
  kFullNewton,
};

/// Compute the analytical 6x7 Jacobian [J_pos; J_rpy] using DH parameters.
template <typename T>
Eigen::Matrix<T, 6, 7> ComputePoseJacobianAnalytic(const Eigen::Matrix<T, 7, 1>& q);

/// Compute the 6 kinematic Hessians of pose6(FK_IiwaDh(q)) w.r.t. q.
/// Each H_i is a 7x7 symmetric matrix: H_i(j,k) = ∂²f_i / ∂q_j ∂q_k
/// where f_i is the i-th component of Pose6FromMatrix4(FK_IiwaDh(q)).
/// Uses a mix of closed-form for position and AutoDiff for orientation
/// (without nested types).
std::array<Eigen::Matrix<double, 7, 7>, 6> ComputeKinematicHessians(
    const Eigen::Matrix<double, 7, 1>& q);

enum class ReachabilityType {
  kProbing = 0,
  kDirect = 1,
  kBoundary = 2
};

/**
 * Computes the geometric pose Jacobian J_s (6x7) for the IIWA arm at
 * configuration q. J = [J_angular; J_linear] where the top 3 rows are angular
 * velocity (relative to world) and the bottom 3 rows are linear velocity (of
 * the EE origin relative to world).
 */
template <typename T>
Eigen::Matrix<T, 6, 7> ComputePoseJacobianGeometric(const Eigen::Matrix<T, 7, 1>& q,
                                                   double grasp_distance = 0.0);

/**
 * Computes the partial derivatives of the geometric Jacobian J_s with respect
 * to q_j. Returns a vector of 7 matrices of size 6x7.
 */
std::vector<Eigen::Matrix<double, 6, 7>>
ComputeGeometricJacobianDerivatives(const Eigen::VectorXd &q, double grasp_distance = 0.0);

/** Configuration for AutoDiff and IFT derivatives. */
struct AutoDiffConfig {
  bool use_ift{true};
  IftSingularityHandling ift_handling{IftSingularityHandling::kPseudoinverse};

  double lambda{0.0};
  std::optional<double> svt_epsilon;
  std::optional<double> svt_lambda_max;
  bool use_anisotropic_damping{false};
};

/** Centralized kinematic parameters for the IIWA arm. */
struct IiwaKinematicParameters {
  static Eigen::Matrix<double, 7, 1> alpha() {
    Eigen::Matrix<double, 7, 1> a;
    a << -M_PI_2, M_PI_2, M_PI_2, -M_PI_2, -M_PI_2, M_PI_2, 0.0;
    return a;
  }
  static Eigen::Matrix<double, 7, 1> d() {
    Eigen::Matrix<double, 7, 1> d_vec;
    d_vec << 0.36, 0.0, 0.42, 0.0, 0.4, 0.0, 0.126 - 0.045;
    return d_vec;
  }
  static Eigen::Vector3d base_translation() {
    return Eigen::Vector3d(0, -0.765, 0);
  }
};

template <typename T>
Eigen::Matrix4<T> ComputeDHMatrix(const T &ti, double ai, double di) {
  T ct = cos(ti);
  T st = sin(ti);
  double ca = std::cos(ai);
  double sa = std::sin(ai);

  Eigen::Matrix4<T> mat;
  mat << ct, -st * ca, st * sa, 0, st, ct * ca, -ct * sa, 0, 0, sa, ca, di, 0,
      0, 0, 1;
  return mat;
}

template <typename T>
Eigen::Matrix3<T> CrossProductMatrix(const Eigen::Matrix<T, 3, 1> &a) {
  Eigen::Matrix3<T> A;
  A << 0, -a(2), a(1), a(2), 0, -a(0), -a(1), a(0), 0;
  return A;
}

// Pose chart: [p(3); rpy(3)] in world frame.
template <typename T>
Eigen::Matrix<T, 6, 1> Pose6FromMatrix4(const Eigen::Matrix4<T>& X_W) {
  Eigen::Matrix<T, 6, 1> y;
  y.template head<3>() = X_W.template block<3,1>(0,3);

  const Eigen::Matrix<T, 3, 3> R = X_W.template block<3,3>(0,0);
  const drake::math::RotationMatrix<T> Rm(R);
  const drake::math::RollPitchYaw<T> rpy(Rm);
  y.template tail<3>() = rpy.vector();  // [roll, pitch, yaw]
  return y;
}

// Wrap an angle to the shortest geodesic representative in [-pi, pi].
// The IK depends on psi only through sin/cos, so psi and psi + 2*pi describe the
// same configuration; ComputePsi_Iiwa returns [-pi, pi] while the planning domain
// spans [0, 2*pi], so residuals against a desired psi must be wrapped.
inline double WrapToPi(double a) { return std::atan2(std::sin(a), std::cos(a)); }

// Residual in the SAME chart as ComputePoseJacobianAnalytic and
// ComputeKinematicHessians: world-frame [p(3); rpy(3)], i.e. the difference of two
// Pose6FromMatrix4 values. The rpy components are wrapped to [-pi, pi].
//
// This must stay chart-consistent with the Jacobian it is paired with: the IFT
// damping strategies form J^T J + sum_i r_i H_i, which is only meaningful when
// r(i), the rows of J, and H[i] all index the same pose components.
template <typename T>
Eigen::Matrix<T, 6, 1> Pose6Residual(const Eigen::Matrix4<T>& X_achieved,
                                     const Eigen::Matrix4<T>& X_goal) {
  Eigen::Matrix<T, 6, 1> r =
      Pose6FromMatrix4<T>(X_achieved) - Pose6FromMatrix4<T>(X_goal);
  // Wrapping shifts by a constant multiple of 2*pi, so it leaves derivatives
  // untouched everywhere except the measure-zero antipode.
  for (int i = 3; i < 6; ++i) {
    if constexpr (std::is_same_v<T, double>) {
      r(i) = WrapToPi(r(i));
    } else {
      const double wrapped = WrapToPi(drake::ExtractDoubleOrThrow(r(i)));
      r(i) += T(wrapped - drake::ExtractDoubleOrThrow(r(i)));
    }
  }
  return r;
}

// WARNING: different convention from Pose6FromMatrix4/Pose6Residual. This returns
// [axis-angle(3); position(3)] expressed in the body frame of X1, whereas the pose
// chart used by the analytic Jacobian and the kinematic Hessians is world-frame
// [position(3); rpy(3)]. The two are NOT interchangeable; pairing this residual with
// those Jacobians/Hessians silently mismatches rotation and translation components.
// Currently unused.
template <typename T>
Eigen::Matrix<T, 6, 1> PoseDistance(
    const Eigen::Matrix4<T>& X1,
    const Eigen::Matrix4<T>& X2) {
  // X_diff = X1^-1 * X2
  // We compute the transform from X1 to X2 in the frame of X1.
  const Eigen::Matrix3<T> R1 = X1.template block<3,3>(0,0);
  const Eigen::Matrix<T, 3, 1> p1 = X1.template block<3,1>(0,3);
  const Eigen::Matrix3<T> R2 = X2.template block<3,3>(0,0);
  const Eigen::Matrix<T, 3, 1> p2 = X2.template block<3,1>(0,3);

  const Eigen::Matrix3<T> R_diff = R1.transpose() * R2;
  const Eigen::Matrix<T, 3, 1> p_diff = R1.transpose() * (p2 - p1);

  const drake::math::RotationMatrix<T> Rm(R_diff);
  const Eigen::AngleAxis<T> aa = Rm.ToAngleAxis();

  Eigen::Matrix<T, 6, 1> out;
  out.template head<3>() = aa.angle() * aa.axis();
  out.template tail<3>() = p_diff;
  return out;
}

template <typename T> T ScalarClip(const T &val, double a, double b) {
  if constexpr (std::is_same_v<T, drake::AutoDiffXd>) {
    Eigen::VectorXd zero_deriv =
        Eigen::VectorXd::Zero(val.derivatives().size());
    T a_ad(a, zero_deriv);
    T b_ad(b, zero_deriv);
    return max(a_ad, min(b_ad, val));
  } else {
    return std::max(static_cast<T>(a), std::min(static_cast<T>(b), val));
  }
}

template <typename T> T SafeArccos(const T &val, double a, double b) {
  return acos(ScalarClip(val, a, b));
}

template <typename T>
Eigen::Matrix4<T> FK_IiwaDh(const Eigen::Matrix<T, 7, 1>& q) {
  static const auto alpha = IiwaKinematicParameters::alpha();
  static const auto d = IiwaKinematicParameters::d();
  Eigen::Matrix4<T> X = ComputeDHMatrix(q(0), alpha(0), d(0));
  for (int i = 1; i < 7; ++i) {
    X = X * ComputeDHMatrix(q(i), alpha(i), d(i));
  }
  return X;
}

template <typename T>
T ComputePsi_Iiwa(
    const Eigen::Matrix<T, 7, 1> &q,
    const BimanualConfig& config,
    const AutoDiffConfig& ad_config = AutoDiffConfig{},
    Eigen::VectorX<T> *unclipped_vals = nullptr) {

  const double clip = 1.0 - config.clipping_margin;
  const double clip_psi = 1.0 - config.clipping_margin_psi;

  static const auto d = IiwaKinematicParameters::d();
  const double d_bs = d(0);
  const double d_se = d(2);
  const double d_ew = d(4);
  const double d_wf = d(6);

  // FK to get T_07, p_07, R_07.
  const Eigen::Matrix4<T> T_07 = FK_IiwaDh<T>(q);
  const Eigen::Matrix<T, 3, 1> p_07 = T_07.template block<3,1>(0,3);
  const Eigen::Matrix<T, 3, 3> R_07 = T_07.template block<3,3>(0,0);

  const Eigen::Matrix<T, 3, 1> p_02(T(0), T(0), T(d_bs));
  const Eigen::Matrix<T, 3, 1> p_67(T(0), T(0), T(d_wf));

  // p_26 = p_07 - p_02 - R_07*p_67
  const Eigen::Matrix<T, 3, 1> p_26 = p_07 - p_02 - R_07 * p_67;

  // theta_4v from arccos law.
  const T p26_dot = p_26.dot(p_26);
  T arccos_in_4v =
      (p26_dot - T(d_se*d_se) - T(d_ew*d_ew)) / (T(2.0*d_se*d_ew));
  const T theta_4v = T(config.GC4()) * SafeArccos(arccos_in_4v, -clip, clip);

  // theta_1v
  const T theta_1v = atan2(p_26(1), p_26(0));

  // theta_2v
  const T p26_norm = p_26.norm();
  T arccos_in_phi =
      (T(d_se*d_se) + p26_dot - T(d_ew*d_ew)) / (T(2.0*d_se) * p26_norm);
  const T phi = SafeArccos(arccos_in_phi, -clip, clip);
  const T theta_2v =
      atan2(p_26.template head<2>().norm(), p_26(2)) + T(config.GC4()) * phi;

  const T theta_3v = T(0);

  static const auto alpha = IiwaKinematicParameters::alpha();
  // Build T_vs for theta_v = [theta_1v, theta_2v, theta_3v, theta_4v]
  const Eigen::Matrix4<T> T_01_v = ComputeDHMatrix(theta_1v, alpha(0), d(0));
  const Eigen::Matrix4<T> T_12_v = ComputeDHMatrix(theta_2v, alpha(1), d(1));
  const Eigen::Matrix4<T> T_23_v = ComputeDHMatrix(theta_3v, alpha(2), d(2));
  const Eigen::Matrix4<T> T_34_v = ComputeDHMatrix(theta_4v, alpha(3), d(3));

  const Eigen::Matrix4<T> T_02_v = T_01_v * T_12_v;
  const Eigen::Matrix4<T> T_04_v = T_02_v * T_23_v * T_34_v;

  const Eigen::Matrix<T,3,1> p_02_v = T_02_v.template block<3,1>(0,3);
  const Eigen::Matrix<T,3,1> p_04_v = T_04_v.template block<3,1>(0,3);

  // Now compute actual T_04 and T_06 using q.
  const Eigen::Matrix4<T> T_04 =
      ComputeDHMatrix(q(0), alpha(0), d(0)) *
      ComputeDHMatrix(q(1), alpha(1), d(1)) *
      ComputeDHMatrix(q(2), alpha(2), d(2)) *
      ComputeDHMatrix(q(3), alpha(3), d(3));
  const Eigen::Matrix<T,3,1> p_04 = T_04.template block<3,1>(0,3);

  const Eigen::Matrix4<T> T_06 =
      T_04 *
      ComputeDHMatrix(q(4), alpha(4), d(4)) *
      ComputeDHMatrix(q(5), alpha(5), d(5));
  const Eigen::Matrix<T,3,1> p_06 = T_06.template block<3,1>(0,3);

  const Eigen::Matrix<T,3,1> p_06_v = p_06;

  // v_se_v, v_sw_v, v_sew_v_hat
  const Eigen::Matrix<T,3,1> v_se_v = (p_04_v - p_02_v).normalized();
  const Eigen::Matrix<T,3,1> v_sw_v = (p_06_v - p_02_v).normalized();
  const Eigen::Matrix<T,3,1> v_sew_v = v_se_v.cross(v_sw_v);
  const Eigen::Matrix<T,3,1> v_sew_v_hat = v_sew_v.normalized();

  // v_se, v_sw, v_sew_hat from actual geometry
  const Eigen::Matrix<T,3,1> v_se = (p_04 - p_02).normalized();
  const Eigen::Matrix<T,3,1> v_sw = (p_06 - p_02).normalized();
  const Eigen::Matrix<T,3,1> v_sew = v_se.cross(v_sw);
  const Eigen::Matrix<T,3,1> v_sew_hat = v_sew.normalized();

  // sign = sign( (v_sew_v_hat x v_sew_hat) dot p_26 )
  const T triple = (v_sew_v_hat.cross(v_sew_hat)).dot(p_26);
  const T sg_psi = (triple >= T(0)) ? T(1) : T(-1);

  // psi = sign * acos( v_sew_v_hat dot v_sew_hat )
  const T arccos_in_psi = v_sew_v_hat.dot(v_sew_hat);
  const T psi = sg_psi * SafeArccos(arccos_in_psi, -clip_psi, clip_psi);

  if (unclipped_vals != nullptr) {
    unclipped_vals->resize(3);
    (*unclipped_vals)(0) = arccos_in_4v;
    (*unclipped_vals)(1) = arccos_in_phi;
    (*unclipped_vals)(2) = arccos_in_psi;
  }

  return psi;
}

// Linkage mapping used in your C++ function:
// tf_goal = LinkageTransformFromControlledEE(FK(q_c))
template <typename T>
Eigen::Matrix4<T> LinkageTransformFromControlledEE(
    const Eigen::Matrix4<T>& X_WC,
    double grasp_distance) {

  const double ang = (180.0 - 2.0 * 68.0) * M_PI / 180.0;
  const T c = cos(T(ang));
  const T s = sin(T(ang));

  Eigen::Matrix3<T> R1;
  R1 << T(-1), T(0), T(0),
        T(0),  T(1), T(0),
        T(0),  T(0), T(-1);

  Eigen::Matrix3<T> R2;
  R2 << c, -s, T(0),
        s,  c, T(0),
        T(0), T(0), T(1);

  Eigen::Matrix3<T> R3;
  R3 << T(-1), T(0), T(0),
        T(0),  T(-1), T(0),
        T(0),  T(0),  T(1);

  Eigen::Matrix4<T> X = X_WC;

  // Rotate: R = R * R1 * R2 * R3
  X.template block<3,3>(0,0) = X.template block<3,3>(0,0) * R1 * R2 * R3;

  // Translate along local -z by grasp_distance
  X.template block<3,1>(0,3) +=
      X.template block<3,3>(0,0) * Eigen::Matrix<T,3,1>(T(0), T(0), T(-grasp_distance));

  // Base translation offset
  X.template block<3,1>(0,3) += IiwaKinematicParameters::base_translation().cast<T>();

  return X;
}



Eigen::VectorX<drake::AutoDiffXd> IiwaBimanualParameterizationIFT(
    const Eigen::VectorX<drake::AutoDiffXd>& q_and_psi,
    const BimanualConfig& config,
    const AutoDiffConfig& ad_config,
    Eigen::VectorX<drake::AutoDiffXd>* unclipped_vals);

template <typename T>
Eigen::VectorX<T> IiwaBimanualParameterization(
    const Eigen::VectorX<T> &q_and_psi,
    const BimanualConfig& config,
    const AutoDiffConfig& ad_config = AutoDiffConfig{},
    Eigen::VectorX<T> *unclipped_vals = nullptr) {

  DRAKE_THROW_UNLESS(q_and_psi.size() == 8);

  if constexpr (std::is_same_v<T, drake::AutoDiffXd>) {
    if (ad_config.use_ift) {
      return IiwaBimanualParameterizationIFT(
          q_and_psi, config, ad_config, unclipped_vals);
    }
  }

  const Eigen::VectorX<T> q_controlled = q_and_psi.head(7);
  const T psi = q_and_psi.tail(1)[0];

  Eigen::VectorX<T> q_subordinate(7);
  Eigen::VectorX<T> q_full(14);
  q_full.head(7) = q_controlled;

  static const auto iiwa_alpha = IiwaKinematicParameters::alpha();
  static const auto iiwa_d = IiwaKinematicParameters::d();
  const double d_bs = iiwa_d[0];
  const double d_se = iiwa_d[2];
  const double d_ew = iiwa_d[4];
  const double d_wf = iiwa_d[6];

  const double clip = 1.0 - config.clipping_margin;
  const double clip_psi = 1.0 - config.clipping_margin_psi;

  // Forward kinematics.
  const Eigen::Matrix4<T> X_Wc = FK_IiwaDh<T>(q_controlled);
  const Eigen::Matrix4<T> tf_goal = LinkageTransformFromControlledEE<T>(X_Wc, config.grasp_distance);

  if (unclipped_vals != nullptr) {
    unclipped_vals->resize(4);
  }

  const Eigen::Matrix<T, 3, 1> p_02(0.0, 0.0, d_bs);
  const Eigen::Matrix<T, 3, 1> p_67(0.0, 0.0, d_wf);

  const Eigen::Matrix<T, 3, 1> p_07 = tf_goal.template block<3, 1>(0, 3);
  const Eigen::Matrix3<T> R_07 = tf_goal.template block<3, 3>(0, 0);

  // EQ (3)
  const Eigen::Matrix<T, 3, 1> p_26 = p_07 - p_02 - R_07 * p_67;
  const Eigen::Matrix<T, 3, 1> p_26_hat = p_26.normalized();

  // EQ (5)
  const T theta_1v = atan2(p_26(1), p_26(0));

  // EQ (7)
  const T p_26_norm = p_26.norm();
  const T p_26_dot = p_26.dot(p_26); // = ||p_26||²

  T arccos_in = (d_se * d_se + p_26_dot - d_ew * d_ew) / (2.0 * d_se * p_26_norm);
  if (unclipped_vals != nullptr) {
    (*unclipped_vals)(0) = arccos_in;
  }

  const T phi = SafeArccos(arccos_in, -clip, clip);
  const T theta_2v = atan2(p_26.template head<2>().norm(), p_26(2)) + config.GC4() * phi;
  const T theta_3v = T(0);

  // EQ (4)
  arccos_in = (p_26_dot - d_se * d_se - d_ew * d_ew) / (2.0 * d_se * d_ew);
  if (unclipped_vals != nullptr) {
    (*unclipped_vals)(1) = arccos_in;
  }

  const T theta_4v = config.GC4() * SafeArccos(arccos_in, -clip, clip);
  q_subordinate[3] = theta_4v;

  // Build transforms
  const Eigen::Matrix4<T> T_01_v = ComputeDHMatrix(theta_1v, iiwa_alpha(0), iiwa_d(0));
  const Eigen::Matrix4<T> T_12_v = ComputeDHMatrix(theta_2v, iiwa_alpha(1), iiwa_d(1));
  const Eigen::Matrix4<T> T_23_v = ComputeDHMatrix(theta_3v, iiwa_alpha(2), iiwa_d(2));

  const Eigen::Matrix4<T> T_03_v = T_01_v * T_12_v * T_23_v;
  const Eigen::Matrix3<T> R_03_v = T_03_v.template block<3, 3>(0, 0);

  // EQ (15)
  const Eigen::Matrix3<T> cprod_p_26 = CrossProductMatrix(p_26_hat);
  const Eigen::Matrix3<T> A_s = cprod_p_26 * R_03_v;
  const Eigen::Matrix3<T> B_s = -cprod_p_26 * cprod_p_26 * R_03_v;
  const Eigen::Matrix3<T> C_s = p_26_hat * p_26_hat.transpose() * R_03_v;

  // EQ (17)-(19)
  q_subordinate(0) =
      atan2(config.GC2() * (A_s(1, 1) * sin(psi) + B_s(1, 1) * cos(psi) + C_s(1, 1)),
            config.GC2() * (A_s(0, 1) * sin(psi) + B_s(0, 1) * cos(psi) + C_s(0, 1)));

  arccos_in = A_s(2, 1) * sin(psi) + B_s(2, 1) * cos(psi) + C_s(2, 1);
  if (unclipped_vals != nullptr) {
    (*unclipped_vals)(2) = arccos_in;
  }
  q_subordinate(1) = config.GC2() * SafeArccos(arccos_in, -clip_psi, clip_psi);

  q_subordinate(2) =
      atan2(config.GC2() * (-A_s(2, 2) * sin(psi) - B_s(2, 2) * cos(psi) - C_s(2, 2)),
            config.GC2() * (-A_s(2, 0) * sin(psi) - B_s(2, 0) * cos(psi) - C_s(2, 0)));

  // EQ (20)
  const Eigen::Matrix4<T> T_34 = ComputeDHMatrix(theta_4v, iiwa_alpha(3), iiwa_d(3));
  const Eigen::Matrix3<T> R_34 = T_34.template block<3, 3>(0, 0);

  const Eigen::Matrix3<T> A_w = R_34.transpose() * A_s.transpose() * R_07;
  const Eigen::Matrix3<T> B_w = R_34.transpose() * B_s.transpose() * R_07;
  const Eigen::Matrix3<T> C_w = R_34.transpose() * C_s.transpose() * R_07;

  // EQ (22)-(24)
  q_subordinate(4) =
      atan2(config.GC6() * (A_w(1, 2) * sin(psi) + B_w(1, 2) * cos(psi) + C_w(1, 2)),
            config.GC6() * (A_w(0, 2) * sin(psi) + B_w(0, 2) * cos(psi) + C_w(0, 2)));

  arccos_in = A_w(2, 2) * sin(psi) + B_w(2, 2) * cos(psi) + C_w(2, 2);
  if (unclipped_vals != nullptr) {
    (*unclipped_vals)(3) = arccos_in;
  }
  q_subordinate(5) = config.GC6() * SafeArccos(arccos_in, -clip_psi, clip_psi);

  q_subordinate(6) =
      atan2(config.GC6() * (A_w(2, 1) * sin(psi) + B_w(2, 1) * cos(psi) + C_w(2, 1)),
            config.GC6() * (-A_w(2, 0) * sin(psi) - B_w(2, 0) * cos(psi) - C_w(2, 0)));

  q_full.tail(7) = q_subordinate;
  return q_full;
}


template <typename T>
Eigen::Matrix<T, 6, 7> ComputePoseJacobianAnalytic(const Eigen::Matrix<T, 7, 1>& q) {
  static const auto alpha = IiwaKinematicParameters::alpha();
  static const auto d_dh = IiwaKinematicParameters::d();

  std::array<Eigen::Matrix4<T>, 8> F;
  F[0] = Eigen::Matrix4<T>::Identity();
  for (int i = 0; i < 7; ++i) {
    F[i + 1] = F[i] * ComputeDHMatrix<T>(q(i), alpha(i), d_dh(i));
  }

  Eigen::Matrix<T, 3, 1> o_e = F[7].template block<3, 1>(0, 3);
  Eigen::Matrix<T, 6, 7> J;

  // Position Jacobian (top 3 rows)
  for (int j = 0; j < 7; ++j) {
    Eigen::Matrix<T, 3, 1> z_j = F[j].template block<3, 3>(0, 0).col(2);
    Eigen::Matrix<T, 3, 1> o_j = F[j].template block<3, 1>(0, 3);
    J.template block<3, 1>(0, j) = z_j.cross(o_e - o_j);
  }

  // Orientation Jacobian (bottom 3 rows, RPY rates)
  const Eigen::Matrix3<T> R = F[7].template block<3, 3>(0, 0);
  const drake::math::RotationMatrix<T> Rm(R);
  const drake::math::RollPitchYaw<T> rpy(Rm);

  const T& theta = rpy.pitch_angle();
  const T& psi = rpy.yaw_angle();

  const T c_th = cos(theta);
  const T s_psi = sin(psi);
  const T c_psi = cos(psi);
  const T t_th = tan(theta);
  const T sec_th = T(1.0) / c_th;

  // E^-1 mapping: geometric angular velocity -> RPY rates
  Eigen::Matrix<T, 3, 3> E_inv;
  E_inv << c_psi * sec_th, s_psi * sec_th, T(0.0), 
           -s_psi, c_psi, T(0.0),
           c_psi * t_th, s_psi * t_th, T(1.0);

  for (int j = 0; j < 7; ++j) {
    Eigen::Matrix<T, 3, 1> z_j = F[j].template block<3, 3>(0, 0).col(2);
    J.template block<3, 1>(3, j) = E_inv * z_j;
  }

  return J;
}
