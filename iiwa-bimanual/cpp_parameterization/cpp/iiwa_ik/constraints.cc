#include "constraints.h"
#include <cassert>

// Shared boundary-reachability evaluation.
//
// b(q_s) = -log det( S J(q_s) (S J(q_s))^T + eps I ),  S = diag(L,L,L,1,1,1)
//
// Both BoundaryReachabilityConstraint and FullFeasibilityConstraint's kBoundary
// branch evaluate this, so it lives in one place: the value and the gradient must
// not be able to drift apart, and the analytic Jacobian derivative below is the
// single Hessian path (verified against Drake's AutoDiff plant to ~1e-15 and ~34x
// faster -- see BoundaryJacobianDerivativesMatchDrake in test_newton_hessian.cc).
//
// The row scaling S is what makes det(J J^T) unit-coherent: the geometric Jacobian
// mixes angular rows (rad/s) with linear rows (m/s), so without it the determinant
// has mixed units and no single epsilon is meaningful. See BoundaryReachLengthScale.
namespace {

Eigen::Matrix<double, 6, 7> ScaledGeometricJacobian(
    const Eigen::Matrix<double, 7, 1>& q_s, double grasp_distance,
    double length_scale) {
  Eigen::Matrix<double, 6, 7> J =
      ComputePoseJacobianGeometric<double>(q_s, grasp_distance);
  J.topRows<3>() *= length_scale;  // angular rows: rad -> m
  return J;
}

// Value of b, and its gradient with respect to q_s (analytic, via
// ComputeGeometricJacobianDerivatives). `dy_dqs` may be null when only the value
// is needed.
void EvalBoundaryReach(const Eigen::Matrix<double, 7, 1>& q_s,
                       double grasp_distance, double epsilon,
                       double length_scale, double* value,
                       Eigen::Matrix<double, 1, 7>* dy_dqs) {
  const Eigen::Matrix<double, 6, 7> J =
      ScaledGeometricJacobian(q_s, grasp_distance, length_scale);
  const Eigen::Matrix<double, 6, 6> A =
      J * J.transpose() + epsilon * Eigen::Matrix<double, 6, 6>::Identity();
  if (value != nullptr) {
    *value = -std::log(A.determinant());
  }
  if (dy_dqs == nullptr) {
    return;
  }
  // d/dq_k [-log det A] = -2 trace( J^T A^-1 dJ/dq_k ).
  const Eigen::Matrix<double, 6, 6> A_inv = A.inverse();
  std::vector<Eigen::Matrix<double, 6, 7>> dJ_dq =
      ComputeGeometricJacobianDerivatives(q_s, grasp_distance);
  const Eigen::Matrix<double, 7, 6> JTAinv = J.transpose() * A_inv;
  for (int k = 0; k < 7; ++k) {
    dJ_dq[k].topRows<3>() *= length_scale;  // same scaling as J
    (*dy_dqs)(k) = -2.0 * (JTAinv * dJ_dq[k]).trace();
  }
}

}  // namespace

IiwaBimanualReachableConstraint::IiwaBimanualReachableConstraint(
    const BimanualConfig& config)
    : drake::solvers::Constraint(4, // output dimension
                                 8, // input dimension
                                 Eigen::Vector4d::Constant(-(1.0 - config.clipping_margin)),
                                 Eigen::Vector4d::Constant(1.0 - config.clipping_margin)),
      config_(config) {
  set_is_thread_safe(true);
}

template <typename T>
void IiwaBimanualReachableConstraint::DoEvalGeneric(
    const Eigen::Ref<const Eigen::VectorX<T>> &q, Eigen::VectorX<T> *y) const {
  Eigen::VectorX<T> unclipped;
  IiwaBimanualParameterization<T>(q, config_, AutoDiffConfig{ .use_ift = false }, &unclipped);
  *y = unclipped; // length 4
}


void IiwaBimanualReachableConstraint::DoEval(
    const Eigen::Ref<const Eigen::VectorXd> &q, Eigen::VectorXd *y) const {
  DoEvalGeneric<double>(q, y);
}

void IiwaBimanualReachableConstraint::DoEval(
    const Eigen::Ref<const drake::AutoDiffVecXd> &q,
    drake::AutoDiffVecXd *y) const {
  DoEvalGeneric<drake::AutoDiffXd>(q, y);
}

void IiwaBimanualReachableConstraint::DoEval(
    const Eigen::Ref<const drake::VectorX<drake::symbolic::Variable>> &q,
    drake::VectorX<drake::symbolic::Expression> *y) const {
  DoEvalGeneric<drake::symbolic::Expression>(q, y);
}

// --------------------------------------------------

OldStyleReachableConstraint::OldStyleReachableConstraint(
    const BimanualConfig& config, const AutoDiffConfig& ad_config,
    double tolerance)
    : drake::solvers::Constraint(1, 8, Eigen::VectorXd::Zero(1),
                                 Eigen::VectorXd::Constant(1, tolerance)),
      config_(config), ad_config_(ad_config) {
  set_is_thread_safe(true);
}

template <typename T>
void OldStyleReachableConstraint::DoEvalGeneric(
    const Eigen::Ref<const Eigen::VectorX<T>> &q, Eigen::VectorX<T> *y) const {
  y->resize(1);

  const Eigen::VectorX<T> q_full = IiwaBimanualParameterization<T>(
      q, config_, ad_config_,
      /*unclipped_vals=*/static_cast<Eigen::VectorX<T> *>(nullptr));

  const Eigen::Matrix<T, 7, 1> q_c = q_full.template head<7>();
  const Eigen::Matrix<T, 7, 1> q_s = q_full.template tail<7>();

  // Compute actual subordinate EE pose.
  const Eigen::Matrix4<T> X_Ws = FK_IiwaDh<T>(q_s);

  // Compute expected subordinate EE pose from the controlled arm.
  const Eigen::Matrix4<T> X_Wc = FK_IiwaDh<T>(q_c);
  const Eigen::Matrix4<T> X_Wgoal =
      LinkageTransformFromControlledEE<T>(X_Wc, config_.grasp_distance);

  // Residual = Frobenius norm of literal matrix difference.
  const Eigen::Matrix4<T> E = X_Ws - X_Wgoal;

  T sumsq = T(0);
  for (int r = 0; r < 4; ++r) {
    for (int c = 0; c < 4; ++c) {
      sumsq += E(r, c) * E(r, c);
    }
  }
  (*y)(0) = sumsq;
}


void OldStyleReachableConstraint::DoEval(
    const Eigen::Ref<const Eigen::VectorXd> &q, Eigen::VectorXd *y) const {
  DoEvalGeneric<double>(q, y);
}

void OldStyleReachableConstraint::DoEval(
    const Eigen::Ref<const drake::AutoDiffVecXd> &q,
    drake::AutoDiffVecXd *y) const {
  DoEvalGeneric<drake::AutoDiffXd>(q, y);
}

void OldStyleReachableConstraint::DoEval(
    const Eigen::Ref<const drake::VectorX<drake::symbolic::Variable>> &q,
    drake::VectorX<drake::symbolic::Expression> *y) const {
  DoEvalGeneric<drake::symbolic::Expression>(q, y);
}

// --------------------------------------------------

IiwaBimanualJointLimitConstraint::IiwaBimanualJointLimitConstraint(
    const Eigen::VectorXd &lower_bound, const Eigen::VectorXd &upper_bound,
    const BimanualConfig& config, const AutoDiffConfig& ad_config)
    : drake::solvers::Constraint(lower_bound.size(), 8, lower_bound,
                                 upper_bound),
      config_(config), ad_config_(ad_config) {
  set_is_thread_safe(true);
}

template <typename T>
void IiwaBimanualJointLimitConstraint::DoEvalGeneric(
    const Eigen::Ref<const Eigen::VectorX<T>> &q, Eigen::VectorX<T> *y) const {
  // Only the subordinate arm's joints matter.
  *y = IiwaBimanualParameterization<T>(q, config_, ad_config_, nullptr).tail(7);
}


void IiwaBimanualJointLimitConstraint::DoEval(
    const Eigen::Ref<const Eigen::VectorXd> &q, Eigen::VectorXd *y) const {
  DoEvalGeneric<double>(q, y);
}

void IiwaBimanualJointLimitConstraint::DoEval(
    const Eigen::Ref<const drake::AutoDiffVecXd> &q,
    drake::AutoDiffVecXd *y) const {
  DoEvalGeneric<drake::AutoDiffXd>(q, y);
}

void IiwaBimanualJointLimitConstraint::DoEval(
    const Eigen::Ref<const drake::VectorX<drake::symbolic::Variable>> &q,
    drake::VectorX<drake::symbolic::Expression> *y) const {
  DoEvalGeneric<drake::symbolic::Expression>(q, y);
}

// --------------------------------------------------

IiwaBimanualCollisionFreeConstraint::IiwaBimanualCollisionFreeConstraint(
    const BimanualConfig& config, const AutoDiffConfig& ad_config,
    std::shared_ptr<drake::multibody::MinimumDistanceLowerBoundConstraint>
        minimum_distance_lower_bound_constraint)
    : drake::solvers::Constraint(
          minimum_distance_lower_bound_constraint->num_constraints(), 8,
          minimum_distance_lower_bound_constraint->lower_bound(),
          minimum_distance_lower_bound_constraint->upper_bound()),
      config_(config), ad_config_(ad_config),
      minimum_distance_lower_bound_constraint_(
          minimum_distance_lower_bound_constraint) {
  assert(minimum_distance_lower_bound_constraint_ != nullptr);
  set_is_thread_safe(true);
}

template <typename T>
void IiwaBimanualCollisionFreeConstraint::DoEvalGeneric(
    const Eigen::Ref<const Eigen::VectorX<T>> &q, Eigen::VectorX<T> *y) const {
  Eigen::VectorX<T> q_full =
      IiwaBimanualParameterization<T>(q, config_, ad_config_, nullptr);
  minimum_distance_lower_bound_constraint_->Eval(q_full, y);
}


void IiwaBimanualCollisionFreeConstraint::DoEval(
    const Eigen::Ref<const Eigen::VectorXd> &q, Eigen::VectorXd *y) const {
  DoEvalGeneric<double>(q, y);
}

void IiwaBimanualCollisionFreeConstraint::DoEval(
    const Eigen::Ref<const drake::AutoDiffVecXd> &q,
    drake::AutoDiffVecXd *y) const {
  DoEvalGeneric<drake::AutoDiffXd>(q, y);
}

// --------------------------------------------------

// The output vector is stacked in the order (unclipped values, subordinate arm
// limits, min distance constraint)
FullFeasibilityConstraint::FullFeasibilityConstraint(
    const Eigen::VectorXd &joint_lower, const Eigen::VectorXd &joint_upper,
    const BimanualConfig &config, const AutoDiffConfig &ad_config,
    std::shared_ptr<drake::multibody::MinimumDistanceLowerBoundConstraint>
        md_constraint,
    ReachabilityType reach_type, bool use_psi_singularity_constraint)
    : drake::solvers::Constraint(
          CountOutputs(reach_type, use_psi_singularity_constraint, 7), 8,
          CalculateLowerBound(reach_type, use_psi_singularity_constraint,
                              joint_lower, config),
          CalculateUpperBound(reach_type, use_psi_singularity_constraint,
                              joint_upper, config)),
      config_(config), ad_config_(ad_config),
      minimum_distance_lower_bound_constraint_(std::move(md_constraint)),
      reach_type_(reach_type),
      use_psi_singularity_constraint_(use_psi_singularity_constraint) {
  DRAKE_DEMAND(minimum_distance_lower_bound_constraint_ != nullptr);
  set_is_thread_safe(true);
}

int FullFeasibilityConstraint::CountOutputs(
    ReachabilityType reach_type, bool use_psi_singularity_constraint,
    int joint_size) {
  int n_reach = (reach_type == ReachabilityType::kProbing) ? 4 : 1;
  int n_psi = use_psi_singularity_constraint ? 3 : 0;
  return n_reach + joint_size + 1 + n_psi;
}

Eigen::VectorXd FullFeasibilityConstraint::CalculateLowerBound(
    ReachabilityType reach_type, bool use_psi_singularity_constraint,
    const Eigen::VectorXd &joint_lower, const BimanualConfig &config) {
  int n_reach = (reach_type == ReachabilityType::kProbing) ? 4 : 1;
  int n_psi = use_psi_singularity_constraint ? 3 : 0;
  Eigen::VectorXd lb(n_reach + joint_lower.size() + 1 + n_psi);
  if (reach_type == ReachabilityType::kProbing) {
    lb.head(4).setConstant(-(1.0 - config.clipping_margin));
  } else if (reach_type == ReachabilityType::kDirect) {
    lb(0) = -std::numeric_limits<double>::infinity();
  } else {
    // kBoundary
    lb(0) = -std::numeric_limits<double>::infinity();
  }
  lb.segment(n_reach, joint_lower.size()) = joint_lower;
  lb(n_reach + joint_lower.size()) = -std::numeric_limits<double>::infinity();
  if (use_psi_singularity_constraint) {
    lb.tail(3).setConstant(-(1.0 - config.clipping_margin_psi));
  }
  return lb;
}

Eigen::VectorXd FullFeasibilityConstraint::CalculateUpperBound(
    ReachabilityType reach_type, bool use_psi_singularity_constraint,
    const Eigen::VectorXd &joint_upper, const BimanualConfig &config) {
  int n_reach = (reach_type == ReachabilityType::kProbing) ? 4 : 1;
  int n_psi = use_psi_singularity_constraint ? 3 : 0;
  Eigen::VectorXd ub(n_reach + joint_upper.size() + 1 + n_psi);
  if (reach_type == ReachabilityType::kProbing) {
    ub.head(4).setConstant(1.0 - config.clipping_margin);
  } else if (reach_type == ReachabilityType::kDirect) {
    ub(0) = 1e-4; // threshold for direct reachability
  } else {
    // kBoundary
    ub(0) = config.boundary_threshold;
  }
  ub.segment(n_reach, joint_upper.size()) = joint_upper;
  ub(n_reach + joint_upper.size()) = 1.0;
  if (use_psi_singularity_constraint) {
    ub.tail(3).setConstant(1.0 - config.clipping_margin_psi);
  }
  return ub;
}

template <typename T>
void FullFeasibilityConstraint::DoEvalGeneric(
    const Eigen::Ref<const Eigen::VectorX<T>> &q, Eigen::VectorX<T> *y) const {
  int n_reach = (reach_type_ == ReachabilityType::kProbing) ? 4 : 1;
  int n_psi = use_psi_singularity_constraint_ ? 3 : 0;
  y->resize(n_reach + 7 + 1 + n_psi);

  // Parameterization: unclipped reachability output.
  // Only the probing formulation reads `unclipped`. Requesting it unconditionally
  // is expensive under the IFT, where supplying the pointer forces a second, full
  // autodiff parameterization pass whose result is then discarded.
  const bool need_unclipped = (reach_type_ == ReachabilityType::kProbing);
  Eigen::VectorX<T> unclipped(4);
  Eigen::VectorX<T> q_full = IiwaBimanualParameterization<T>(
      q, config_, ad_config_, need_unclipped ? &unclipped : nullptr);

  if (reach_type_ == ReachabilityType::kProbing) {
    y->segment(0, 4) = unclipped;
  } else if (reach_type_ == ReachabilityType::kDirect) {
    const Eigen::Matrix<T, 7, 1> q_c = q_full.template head<7>();
    const Eigen::Matrix<T, 7, 1> q_s = q_full.template tail<7>();

    const Eigen::Matrix4<T> X_Ws = FK_IiwaDh<T>(q_s);
    const Eigen::Matrix4<T> X_Wc = FK_IiwaDh<T>(q_c);
    const Eigen::Matrix4<T> X_Wgoal =
        LinkageTransformFromControlledEE<T>(X_Wc, config_.grasp_distance);

    const Eigen::Matrix4<T> E = X_Ws - X_Wgoal;
    T sumsq = T(0);
    for (int r = 0; r < 4; ++r) {
      for (int c = 0; c < 4; ++c) {
        sumsq += E(r, c) * E(r, c);
      }
    }
    (*y)(0) = sumsq;
  } else {
    // kBoundary. The AutoDiffXd overload of DoEval recomputes this row through
    // the analytic gradient path and overwrites it, so differentiating the
    // determinant here would be pure waste; compute the value only.
    const Eigen::Matrix<T, 7, 1> q_s = q_full.template tail<7>();
    if constexpr (std::is_same_v<T, drake::AutoDiffXd>) {
      double b = 0.0;
      EvalBoundaryReach(drake::math::ExtractValue(q_s), config_.grasp_distance,
                        config_.boundary_epsilon, config_.boundary_length_scale,
                        &b, nullptr);
      (*y)(0) = T(b);
    } else if constexpr (std::is_same_v<T, double>) {
      double b = 0.0;
      EvalBoundaryReach(q_s, config_.grasp_distance, config_.boundary_epsilon,
                        config_.boundary_length_scale, &b, nullptr);
      (*y)(0) = b;
    } else {
      // Symbolic: fall back to the templated expression.
      Eigen::Matrix<T, 6, 7> J =
          ComputePoseJacobianGeometric<T>(q_s, config_.grasp_distance);
      J.template topRows<3>() *= T(config_.boundary_length_scale);
      Eigen::Matrix<T, 6, 6> A =
          J * J.transpose() + config_.boundary_epsilon *
                                  Eigen::Matrix<T, 6, 6>::Identity();
      (*y)(0) = -log(A.determinant());
    }
  }


  // Evaluate collision distance
  Eigen::VectorX<T> min_distance_output(1);
  minimum_distance_lower_bound_constraint_->Eval(q_full, &min_distance_output);

  // Assign segments
  y->segment(n_reach, 7) = q_full.tail(7);          // joint limits
  y->segment(n_reach + 7, 1) = min_distance_output; // collision

  if (use_psi_singularity_constraint_) {
    const Eigen::Matrix<T, 7, 1> q_s = q_full.template tail<7>();
    Eigen::VectorX<T> unclipped_psi_vals;
    ComputePsi_Iiwa<T>(q_s, config_, ad_config_, &unclipped_psi_vals);
    y->segment(n_reach + 7 + 1, 3) = unclipped_psi_vals;
  }

}

void FullFeasibilityConstraint::DoEval(
    const Eigen::Ref<const Eigen::VectorXd> &q, Eigen::VectorXd *y) const {
  DoEvalGeneric<double>(q, y);
}

void FullFeasibilityConstraint::DoEval(
    const Eigen::Ref<const drake::AutoDiffVecXd> &q,
    drake::AutoDiffVecXd *y) const {
  int num_vars = q(0).derivatives().size();
  if (num_vars == 0) {
    Eigen::VectorXd y_val;
    DoEval(drake::math::ExtractValue(q), &y_val);
    y->resize(y_val.size());
    for (int i = 0; i < y_val.size(); ++i) {
      (*y)(i).value() = y_val(i);
      (*y)(i).derivatives() = Eigen::VectorXd::Zero(0);
    }
    return;
  }

  if (reach_type_ != ReachabilityType::kBoundary) {
    DoEvalGeneric<drake::AutoDiffXd>(q, y);
    return;
  }

  // Boundary reachability needs analytical gradient for stability
  drake::AutoDiffVecXd q_full = IiwaBimanualParameterization<drake::AutoDiffXd>(
      q, config_, ad_config_, nullptr);
  Eigen::Matrix<drake::AutoDiffXd, 7, 1> q_s_ad = q_full.tail<7>();
  Eigen::Matrix<double, 7, 1> q_s = drake::math::ExtractValue(q_s_ad);

  double b = 0.0;
  Eigen::Matrix<double, 1, 7> dy_dqs;
  EvalBoundaryReach(q_s, config_.grasp_distance, config_.boundary_epsilon,
                    config_.boundary_length_scale, &b, &dy_dqs);

  Eigen::MatrixXd dqs_dqtilde(7, num_vars);
  for (int i = 0; i < 7; ++i) {
    dqs_dqtilde.row(i) = q_s_ad(i).derivatives().transpose();
  }
  Eigen::VectorXd grad_reach = dy_dqs * dqs_dqtilde;

  // Evaluate other parts using Generic
  DoEvalGeneric<drake::AutoDiffXd>(q, y);

  // Patch the reachability part
  (*y)(0).value() = b;
  (*y)(0).derivatives() = grad_reach;
}

// --------------------------------------------------

IiwaBimanualPsiSingularityConstraint::IiwaBimanualPsiSingularityConstraint(
    const BimanualConfig& config, const AutoDiffConfig& ad_config)
    : drake::solvers::Constraint(3, // output dimension
                                 8, // input dimension
                                 Eigen::Vector3d::Constant(-(1.0 - config.clipping_margin_psi)),
                                 Eigen::Vector3d::Constant(1.0 - config.clipping_margin_psi)),
      config_(config), ad_config_(ad_config) {
  set_is_thread_safe(true);
}


template <typename T>
void IiwaBimanualPsiSingularityConstraint::DoEvalGeneric(
    const Eigen::Ref<const Eigen::VectorX<T>> &q, Eigen::VectorX<T> *y) const {
  // First get the full configuration using the parameterization.
  Eigen::VectorX<T> q_full =
      IiwaBimanualParameterization<T>(q, config_, ad_config_, nullptr);

  const Eigen::Matrix<T, 7, 1> q_s = q_full.template tail<7>();

  // Now compute psi and capture the unclipped arccos values.
  Eigen::VectorX<T> unclipped_psi_vals;
  ComputePsi_Iiwa<T>(q_s, config_, ad_config_, &unclipped_psi_vals);

  *y = unclipped_psi_vals;
}


void IiwaBimanualPsiSingularityConstraint::DoEval(
    const Eigen::Ref<const Eigen::VectorXd> &q, Eigen::VectorXd *y) const {
  DoEvalGeneric<double>(q, y);
}

void IiwaBimanualPsiSingularityConstraint::DoEval(
    const Eigen::Ref<const drake::AutoDiffVecXd> &q,
    drake::AutoDiffVecXd *y) const {
  DoEvalGeneric<drake::AutoDiffXd>(q, y);
}

void IiwaBimanualPsiSingularityConstraint::DoEval(
    const Eigen::Ref<const drake::VectorX<drake::symbolic::Variable>> &q,
    drake::VectorX<drake::symbolic::Expression> *y) const {
  DoEvalGeneric<drake::symbolic::Expression>(q, y);
}
// --------------------------------------------------

BoundaryReachabilityConstraint::BoundaryReachabilityConstraint(
    const BimanualConfig &config, const AutoDiffConfig &ad_config,
    double threshold, double epsilon)
    : drake::solvers::Constraint(
          1, 8,
          Eigen::VectorXd::Constant(1, -std::numeric_limits<double>::infinity()),
          Eigen::VectorXd::Constant(1, threshold)),
      config_(config), ad_config_(ad_config), threshold_(threshold),
      epsilon_(epsilon) {
  set_is_thread_safe(true);
}

template <typename T>
void BoundaryReachabilityConstraint::DoEvalGeneric(
    const Eigen::Ref<const Eigen::VectorX<T>> &q, Eigen::VectorX<T> *y) const {
  y->resize(1);
  Eigen::VectorX<T> q_full =
      IiwaBimanualParameterization<T>(q, config_, ad_config_, nullptr);
  Eigen::Matrix<T, 7, 1> q_s = q_full.template tail<7>();

  if constexpr (std::is_same_v<T, double>) {
    double b = 0.0;
    EvalBoundaryReach(q_s, config_.grasp_distance, epsilon_,
                      config_.boundary_length_scale, &b, nullptr);
    (*y)(0) = b;
  } else {
    Eigen::Matrix<T, 6, 7> J =
        ComputePoseJacobianGeometric<T>(q_s, config_.grasp_distance);
    J.template topRows<3>() *= T(config_.boundary_length_scale);
    Eigen::Matrix<T, 6, 6> JJT = J * J.transpose();
    Eigen::Matrix<T, 6, 6> A =
        JJT + epsilon_ * Eigen::Matrix<T, 6, 6>::Identity().template cast<T>();
    (*y)(0) = -log(A.determinant());
  }
}

void BoundaryReachabilityConstraint::DoEval(
    const Eigen::Ref<const Eigen::VectorXd> &q, Eigen::VectorXd *y) const {
  DoEvalGeneric<double>(q, y);
}

void BoundaryReachabilityConstraint::DoEval(
    const Eigen::Ref<const drake::AutoDiffVecXd> &q,
    drake::AutoDiffVecXd *y) const {
  int num_vars = q(0).derivatives().size();
  if (num_vars == 0) {
    Eigen::VectorXd y_val;
    DoEval(drake::math::ExtractValue(q), &y_val);
    y->resize(y_val.size());
    for (int i = 0; i < y_val.size(); ++i) {
      (*y)(i).value() = y_val(i);
      (*y)(i).derivatives() = Eigen::VectorXd::Zero(0);
    }
    return;
  }

  drake::AutoDiffVecXd q_full = IiwaBimanualParameterization<drake::AutoDiffXd>(
      q, config_, ad_config_, nullptr);
  Eigen::Matrix<drake::AutoDiffXd, 7, 1> q_s_ad = q_full.tail<7>();
  Eigen::Matrix<double, 7, 1> q_s = drake::math::ExtractValue(q_s_ad);

  double b = 0.0;
  Eigen::Matrix<double, 1, 7> dy_dqs;
  EvalBoundaryReach(q_s, config_.grasp_distance, epsilon_,
                    config_.boundary_length_scale, &b, &dy_dqs);

  Eigen::MatrixXd dqs_dqtilde(7, num_vars);
  for (int i = 0; i < 7; ++i) {
    dqs_dqtilde.row(i) = q_s_ad(i).derivatives().transpose();
  }
  Eigen::VectorXd grad_y = dy_dqs * dqs_dqtilde;

  y->resize(1);
  (*y)(0).value() = b;
  (*y)(0).derivatives() = grad_y;
}
