#pragma once

#include "drake/multibody/inverse_kinematics/minimum_distance_lower_bound_constraint.h"
#include "drake/solvers/mathematical_program.h"
#include "iiwa_analytic_ik.h"
#include "parameterization.h"
#include <Eigen/Dense>
#include <memory>

/** Reachability constraint for the bimanual IIWA arm.
 *  Maps q_and_psi (8-dimensional) to unclipped values (4-dimensional). */
class IiwaBimanualReachableConstraint final
    : public drake::solvers::Constraint {
public:
  IiwaBimanualReachableConstraint(const BimanualConfig& config);

private:
  template <typename T>
  void DoEvalGeneric(const Eigen::Ref<const Eigen::VectorX<T>> &q,
                     Eigen::VectorX<T> *y) const;

  void DoEval(const Eigen::Ref<const Eigen::VectorXd> &q,
              Eigen::VectorXd *y) const override;
  void DoEval(const Eigen::Ref<const drake::AutoDiffVecXd> &q,
              drake::AutoDiffVecXd *y) const override;
  void
  DoEval(const Eigen::Ref<const drake::VectorX<drake::symbolic::Variable>> &q,
         drake::VectorX<drake::symbolic::Expression> *y) const override;

  BimanualConfig config_;
};


/** Joint limit constraint for the subordinate arm of the bimanual IIWA. */
class IiwaBimanualJointLimitConstraint final
    : public drake::solvers::Constraint {
public:
  IiwaBimanualJointLimitConstraint(const Eigen::VectorXd &lower_bound,
                                   const Eigen::VectorXd &upper_bound,
                                   const BimanualConfig& config,
                                   const AutoDiffConfig& ad_config);

private:
  template <typename T>
  void DoEvalGeneric(const Eigen::Ref<const Eigen::VectorX<T>> &q,
                     Eigen::VectorX<T> *y) const;

  void DoEval(const Eigen::Ref<const Eigen::VectorXd> &q,
              Eigen::VectorXd *y) const override;
  void DoEval(const Eigen::Ref<const drake::AutoDiffVecXd> &q,
              drake::AutoDiffVecXd *y) const override;
  void
  DoEval(const Eigen::Ref<const drake::VectorX<drake::symbolic::Variable>> &q,
         drake::VectorX<drake::symbolic::Expression> *y) const override;
  BimanualConfig config_;
  AutoDiffConfig ad_config_;
};

/** Parameterized collision-free constraint. */
class IiwaBimanualCollisionFreeConstraint final
    : public drake::solvers::Constraint {
public:
  IiwaBimanualCollisionFreeConstraint(
      const BimanualConfig& config,
      const AutoDiffConfig& ad_config,
      std::shared_ptr<drake::multibody::MinimumDistanceLowerBoundConstraint>
          minimum_distance_lower_bound_constraint);

private:
  template <typename T>
  void DoEvalGeneric(const Eigen::Ref<const Eigen::VectorX<T>> &q,
                     Eigen::VectorX<T> *y) const;

  void DoEval(const Eigen::Ref<const Eigen::VectorXd> &q,
              Eigen::VectorXd *y) const override;
  void DoEval(const Eigen::Ref<const drake::AutoDiffVecXd> &q,
              drake::AutoDiffVecXd *y) const override;
  void
  DoEval(const Eigen::Ref<const drake::VectorX<drake::symbolic::Variable>> &q,
         drake::VectorX<drake::symbolic::Expression> *y) const override {
    throw std::logic_error(
        "MinimumDistanceLowerBoundConstraint::DoEval() does not work for "
        "symbolic variables.");
  }

  BimanualConfig config_;
  AutoDiffConfig ad_config_;
  std::shared_ptr<drake::multibody::MinimumDistanceLowerBoundConstraint>
      minimum_distance_lower_bound_constraint_{};
};

/** Old-style reachability constraint for the bimanual IIWA arm.
 *  Maps q_and_psi (8-dimensional) to a 1-vector residual:
 *    y(0) = || X_Ws(q_s) - X_Wgoal(q_c) ||_F^2
 *  where q_full = [q_c; q_s] = IiwaBimanualParameterization(q_and_psi, ...),
 *  X_Wgoal is the expected subordinate EE pose derived from the controlled arm.
 */
class OldStyleReachableConstraint final : public drake::solvers::Constraint {
public:
  OldStyleReachableConstraint(const BimanualConfig& config,
                              const AutoDiffConfig& ad_config,
                              double tolerance = 1e-6);


private:
  template <typename T>
  void DoEvalGeneric(const Eigen::Ref<const Eigen::VectorX<T>> &q,
                     Eigen::VectorX<T> *y) const;

  void DoEval(const Eigen::Ref<const Eigen::VectorXd> &q,
              Eigen::VectorXd *y) const override;
  void DoEval(const Eigen::Ref<const drake::AutoDiffVecXd> &q,
              drake::AutoDiffVecXd *y) const override;
  void
  DoEval(const Eigen::Ref<const drake::VectorX<drake::symbolic::Variable>> &q,
         drake::VectorX<drake::symbolic::Expression> *y) const override;

  BimanualConfig config_;
  AutoDiffConfig ad_config_;
};

/** All-in-one feasibility constraint. */
class FullFeasibilityConstraint final : public drake::solvers::Constraint {
public:
  FullFeasibilityConstraint(
      const Eigen::VectorXd &lower_bound, const Eigen::VectorXd &upper_bound,
      const BimanualConfig& config,
      const AutoDiffConfig& ad_config,
      std::shared_ptr<drake::multibody::MinimumDistanceLowerBoundConstraint>
          minimum_distance_lower_bound_constraint,
      ReachabilityType reach_type = ReachabilityType::kProbing,
      bool use_psi_singularity_constraint = false);

private:
  template <typename T>
  void DoEvalGeneric(const Eigen::Ref<const Eigen::VectorX<T>> &q,
                     Eigen::VectorX<T> *y) const;

  void DoEval(const Eigen::Ref<const Eigen::VectorXd> &q,
              Eigen::VectorXd *y) const override;
  void DoEval(const Eigen::Ref<const drake::AutoDiffVecXd> &q,
              drake::AutoDiffVecXd *y) const override;
  void
  DoEval(const Eigen::Ref<const drake::VectorX<drake::symbolic::Variable>> &q,
         drake::VectorX<drake::symbolic::Expression> *y) const override {
    throw std::logic_error(
        "FullFeasibilityConstraint::DoEval() does not work for "
        "symbolic variables.");
  }

  BimanualConfig config_;
  AutoDiffConfig ad_config_;
  std::shared_ptr<drake::multibody::MinimumDistanceLowerBoundConstraint>
      minimum_distance_lower_bound_constraint_{};
  ReachabilityType reach_type_{};
  bool use_psi_singularity_constraint_{};

  static int CountOutputs(ReachabilityType reach_type,
                          bool use_psi_singularity_constraint, int joint_size);
  static Eigen::VectorXd
  CalculateLowerBound(ReachabilityType reach_type,
                      bool use_psi_singularity_constraint,
                      const Eigen::VectorXd &joint_lower,
                      const BimanualConfig& config);
  static Eigen::VectorXd
  CalculateUpperBound(ReachabilityType reach_type,
                      bool use_psi_singularity_constraint,
                      const Eigen::VectorXd &joint_upper,
                      const BimanualConfig& config);
};

/** Arccos clipping constraint for the psi angle calculation of the bimanual
 * IIWA. Maps q_and_psi (8-dimensional) to unclipped arccos inputs
 * (3-dimensional). */
class IiwaBimanualPsiSingularityConstraint final
    : public drake::solvers::Constraint {
public:
  IiwaBimanualPsiSingularityConstraint(const BimanualConfig& config,
                                       const AutoDiffConfig& ad_config);

private:
  template <typename T>
  void DoEvalGeneric(const Eigen::Ref<const Eigen::VectorX<T>> &q,
                     Eigen::VectorX<T> *y) const;

  void DoEval(const Eigen::Ref<const Eigen::VectorXd> &q,
              Eigen::VectorXd *y) const override;
  void DoEval(const Eigen::Ref<const drake::AutoDiffVecXd> &q,
              drake::AutoDiffVecXd *y) const override;
  void
  DoEval(const Eigen::Ref<const drake::VectorX<drake::symbolic::Variable>> &q,
         drake::VectorX<drake::symbolic::Expression> *y) const override;
  BimanualConfig config_;
  AutoDiffConfig ad_config_;
};

/**
 * Singularity-aware reachability constraint based on the log-determinant of the
 * Gram matrix of the geometric Jacobian.
 *
 * Why enforcing non-singularity is enough to enforce reachability for this
 * analytic IK: the IK is correct up to kinematic and representational
 * singularities, so a point outside the reachable set (no solutions) must trigger
 * a domain error somewhere, and the only domain-limited operation here is arccos.
 * The arccos domain boundaries are therefore either workspace boundaries or
 * representational singularities, and both imply a singular kinematic Jacobian.
 */
class BoundaryReachabilityConstraint final : public drake::solvers::Constraint {
public:
  BoundaryReachabilityConstraint(const BimanualConfig &config,
                                 const AutoDiffConfig &ad_config,
                                 double threshold, double epsilon);

private:
  template <typename T>
  void DoEvalGeneric(const Eigen::Ref<const Eigen::VectorX<T>> &q,
                     Eigen::VectorX<T> *y) const;

  void DoEval(const Eigen::Ref<const Eigen::VectorXd> &q,
              Eigen::VectorXd *y) const override;
  void DoEval(const Eigen::Ref<const drake::AutoDiffVecXd> &q,
              drake::AutoDiffVecXd *y) const override;
  void
  DoEval(const Eigen::Ref<const drake::VectorX<drake::symbolic::Variable>> &q,
         drake::VectorX<drake::symbolic::Expression> *y) const override {
    throw std::logic_error(
        "BoundaryReachabilityConstraint::DoEval() does not work for symbolic variables.");
  }

  BimanualConfig config_;
  AutoDiffConfig ad_config_;
  double threshold_;
  double epsilon_;
};
