#include "costs.h"

IiwaBimanualPathCost::IiwaBimanualPathCost(int num_positions,
                                           int num_control_points,
                                           const BimanualConfig& config,
                                           const AutoDiffConfig& ad_config,
                                           bool square)
    : drake::solvers::Cost(num_positions * num_control_points),
      num_positions_(num_positions), num_control_points_(num_control_points),
      config_(config), ad_config_(ad_config), square_(square) {
  set_is_thread_safe(true);
}

template <typename T>
T IiwaBimanualPathCost::DoEvalGeneric(
    const Eigen::Ref<const Eigen::Matrix<T, Eigen::Dynamic, 1>>
        &control_points_flat) const {

  Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>
      control_points(control_points_flat.data(), num_positions_,
                     num_control_points_);

  T total = T(0);
  for (int i = 1; i < num_control_points_; ++i) {
    Eigen::Matrix<T, Eigen::Dynamic, 1> q0 = IiwaBimanualParameterization<T>(
        control_points.col(i - 1), config_, ad_config_, nullptr);
    Eigen::Matrix<T, Eigen::Dynamic, 1> q1 = IiwaBimanualParameterization<T>(
        control_points.col(i), config_, ad_config_, nullptr);
    Eigen::Matrix<T, Eigen::Dynamic, 1> delta = q1 - q0;
    if (square_) {
      total += delta.squaredNorm();
    } else {
      total += delta.norm();
    }
  }

  return total;
}


void IiwaBimanualPathCost::DoEval(const Eigen::Ref<const Eigen::VectorXd> &x,
                                  Eigen::VectorXd *y) const {
  (*y)(0) = DoEvalGeneric<double>(x);
}

void IiwaBimanualPathCost::DoEval(
    const Eigen::Ref<const drake::AutoDiffVecXd> &x,
    drake::AutoDiffVecXd *y) const {
  (*y)(0) = DoEvalGeneric<drake::AutoDiffXd>(x);
}

void IiwaBimanualPathCost::DoEval(
    const Eigen::Ref<const drake::VectorX<drake::symbolic::Variable>> &x,
    drake::VectorX<drake::symbolic::Expression> *y) const {
  (*y)(0) = DoEvalGeneric<drake::symbolic::Expression>(x);
}