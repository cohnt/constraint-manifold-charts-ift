#pragma once

#include "drake/solvers/mathematical_program.h"
#include "iiwa_analytic_ik.h"
#include <Eigen/Dense>

/**
 * Path energy cost for a sequence of control points for the subordinate arm.
 * Accumulates squared differences between consecutive configurations.
 * If square is true, returns path energy, otherwise returns path length.
 */
class IiwaBimanualPathCost final : public drake::solvers::Cost {
public:
  IiwaBimanualPathCost(int num_positions, int num_control_points,
                       const BimanualConfig& config,
                       const AutoDiffConfig& ad_config, bool square);

private:
  // Generic evaluation template
  template <typename T>
  T DoEvalGeneric(const Eigen::Ref<const Eigen::Matrix<T, Eigen::Dynamic, 1>>
                      &control_points_flat) const;

  // Overrides for Drake Cost
  void DoEval(const Eigen::Ref<const Eigen::VectorXd> &x,
              Eigen::VectorXd *y) const override;
  void DoEval(const Eigen::Ref<const drake::AutoDiffVecXd> &x,
              drake::AutoDiffVecXd *y) const override;
  void
  DoEval(const Eigen::Ref<const drake::VectorX<drake::symbolic::Variable>> &x,
         drake::VectorX<drake::symbolic::Expression> *y) const override;

  int num_positions_;
  int num_control_points_;
  BimanualConfig config_;
  AutoDiffConfig ad_config_;
  bool square_;
};