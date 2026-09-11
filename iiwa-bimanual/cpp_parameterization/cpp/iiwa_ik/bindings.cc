#include <memory>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "constraints.h"
#include "costs.h"
#include "parameterization.h"

namespace py = pybind11;

PYBIND11_MODULE(_iiwa_ik, m) {
  py::class_<BimanualConfig>(m, "BimanualConfig")
      .def(py::init<bool, bool, bool, double, double, double, double, double,
                    double>(),
           py::arg("shoulder_up") = true, py::arg("elbow_up") = true,
           py::arg("wrist_up") = true, py::arg("grasp_distance") = 0.5,
           py::arg("clipping_margin") = 1e-4,
           py::arg("clipping_margin_psi") = 1e-4,
           py::arg("boundary_threshold") = 2.46,
           py::arg("boundary_epsilon") = 1e-6,
           py::arg("boundary_length_scale") = 1.86)
      .def_readwrite("shoulder_up", &BimanualConfig::shoulder_up)
      .def_readwrite("elbow_up", &BimanualConfig::elbow_up)
      .def_readwrite("wrist_up", &BimanualConfig::wrist_up)
      .def_readwrite("grasp_distance", &BimanualConfig::grasp_distance)
      .def_readwrite("clipping_margin", &BimanualConfig::clipping_margin)
      .def_readwrite("clipping_margin_psi", &BimanualConfig::clipping_margin_psi)
      .def_readwrite("boundary_threshold", &BimanualConfig::boundary_threshold)
      .def_readwrite("boundary_epsilon", &BimanualConfig::boundary_epsilon)
      .def_readwrite("boundary_length_scale",
                     &BimanualConfig::boundary_length_scale);

  py::enum_<ReachabilityType>(m, "ReachabilityType")
      .value("kProbing", ReachabilityType::kProbing)
      .value("kDirect", ReachabilityType::kDirect)
      .value("kBoundary", ReachabilityType::kBoundary)
      .export_values();

  py::enum_<IftSingularityHandling>(m, "IftSingularityHandling")
      .value("kPseudoinverse", IftSingularityHandling::kPseudoinverse)
      .value("kZero", IftSingularityHandling::kZero)
      .value("kLevenbergMarquardt", IftSingularityHandling::kLevenbergMarquardt)
      .value("kResidualDamping", IftSingularityHandling::kResidualDamping)
      .value("kFullNewton", IftSingularityHandling::kFullNewton)
      .export_values();

  py::class_<AutoDiffConfig>(m, "AutoDiffConfig")
      .def(py::init<bool, IftSingularityHandling, double, std::optional<double>, std::optional<double>, bool>(),
           py::arg("use_ift") = true,
           py::arg("ift_handling") = IftSingularityHandling::kPseudoinverse,
           // Exposed as `lambda_` because `lambda` is a reserved word in
           // Python: named `lambda`, this argument could not be passed by
           // keyword at all, which forced callers into positional construction
           // and made it easy to drop silently.
           py::arg("lambda_") = 0.0,
           py::arg("svt_epsilon") = std::nullopt,
           py::arg("svt_lambda_max") = std::nullopt,
           py::arg("use_anisotropic_damping") = false)
      .def_readwrite("use_ift", &AutoDiffConfig::use_ift)
      .def_readwrite("ift_handling", &AutoDiffConfig::ift_handling)
      .def_readwrite("lambda_", &AutoDiffConfig::lambda)
      .def_readwrite("svt_epsilon", &AutoDiffConfig::svt_epsilon)
      .def_readwrite("svt_lambda_max", &AutoDiffConfig::svt_lambda_max)
      .def_readwrite("use_anisotropic_damping", &AutoDiffConfig::use_anisotropic_damping);

  py::class_<IiwaBimanualReachableConstraint, drake::solvers::Constraint,
             std::shared_ptr<IiwaBimanualReachableConstraint>>(
      m, "IiwaBimanualReachableConstraint")
      .def(py::init<const BimanualConfig&>(), py::arg("config"));

  py::class_<IiwaBimanualPsiSingularityConstraint, drake::solvers::Constraint,
             std::shared_ptr<IiwaBimanualPsiSingularityConstraint>>(
      m, "IiwaBimanualPsiSingularityConstraint")
      .def(py::init<const BimanualConfig&, const AutoDiffConfig&>(),
           py::arg("config"), py::arg("ad_config") = AutoDiffConfig{});

  py::class_<IiwaBimanualJointLimitConstraint, drake::solvers::Constraint,
             std::shared_ptr<IiwaBimanualJointLimitConstraint>>(
      m, "IiwaBimanualJointLimitConstraint")
      .def(py::init<Eigen::VectorXd, Eigen::VectorXd, const BimanualConfig&,
                    const AutoDiffConfig&>(),
           py::arg("lower_bound"), py::arg("upper_bound"), py::arg("config"),
           py::arg("ad_config") = AutoDiffConfig{});

  py::class_<OldStyleReachableConstraint, drake::solvers::Constraint,
             std::shared_ptr<OldStyleReachableConstraint>>(
      m, "OldStyleReachableConstraint")
      .def(py::init<const BimanualConfig&, const AutoDiffConfig&, double>(),
           py::arg("config"), py::arg("ad_config") = AutoDiffConfig{},
           py::arg("tolerance") = 1e-6);

  py::class_<IiwaBimanualCollisionFreeConstraint, drake::solvers::Constraint,
             std::shared_ptr<IiwaBimanualCollisionFreeConstraint>>(
      m, "IiwaBimanualCollisionFreeConstraint")
      .def(py::init<const BimanualConfig&, const AutoDiffConfig&,
                    std::shared_ptr<
                        drake::multibody::MinimumDistanceLowerBoundConstraint>>(),
           py::arg("config"), py::arg("ad_config") = AutoDiffConfig{},
           py::arg("minimum_distance_lower_bound_constraint"));

  py::class_<FullFeasibilityConstraint, drake::solvers::Constraint,
             std::shared_ptr<FullFeasibilityConstraint>>(
      m, "FullFeasibilityConstraint")
      .def(py::init<Eigen::VectorXd, Eigen::VectorXd, const BimanualConfig&,
                    const AutoDiffConfig&,
                    std::shared_ptr<
                        drake::multibody::MinimumDistanceLowerBoundConstraint>,
                    ReachabilityType, bool>(),
           py::arg("lower_bound"), py::arg("upper_bound"), py::arg("config"),
           py::arg("ad_config") = AutoDiffConfig{},
           py::arg("minimum_distance_lower_bound_constraint"),
           py::arg("reach_type") = ReachabilityType::kProbing,
           py::arg("use_psi_singularity_constraint") = false);

  py::class_<BoundaryReachabilityConstraint, drake::solvers::Constraint,
             std::shared_ptr<BoundaryReachabilityConstraint>>(
      m, "BoundaryReachabilityConstraint")
      .def(py::init<const BimanualConfig&, const AutoDiffConfig&, double, double>(),
           py::arg("config"), py::arg("ad_config") = AutoDiffConfig{},
           py::arg("threshold"), py::arg("epsilon"));

  py::class_<IiwaBimanualPathCost, drake::solvers::Cost,
             std::shared_ptr<IiwaBimanualPathCost>>(m, "IiwaBimanualPathCost")
      .def(py::init<int, int, const BimanualConfig&, const AutoDiffConfig&,
                    bool>(),
           py::arg("num_positions"), py::arg("num_control_points"),
           py::arg("config"), py::arg("ad_config") = AutoDiffConfig{},
           py::arg("square"));


  m.def("MakeParameterization", &MakeParameterization, py::arg("config"),
        py::arg("ad_config") = AutoDiffConfig{});

  m.def("ComputePoseJacobianGeometric", &ComputePoseJacobianGeometric<double>,
        py::arg("q"), py::arg("grasp_distance") = 0.0);
  m.def("ComputeGeometricJacobianDerivatives", &ComputeGeometricJacobianDerivatives,
        py::arg("q"), py::arg("grasp_distance") = 0.0);
}
