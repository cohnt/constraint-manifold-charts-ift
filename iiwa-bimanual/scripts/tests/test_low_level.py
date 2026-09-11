import os
import sys
import numpy as np
import unittest
from pydrake.all import (
    InitializeAutoDiff,
    ExtractValue,
    ExtractGradient,
)

# Add local paths
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_dir = os.path.abspath(os.path.join(script_dir, "../.."))
sys.path.append(os.path.join(repo_dir, "cpp_parameterization/python"))
sys.path.append(repo_dir)

from iiwa_ik import (
    MakeParameterization,
    AutoDiffConfig,
    IiwaBimanualReachableConstraint,
    IiwaBimanualJointLimitConstraint,
    IiwaBimanualPsiSingularityConstraint,
    IiwaBimanualPathCost,
    IftSingularityHandling,
    BimanualConfig
)
from src.iiwa_analytic_ik import iiwa_limits_lower, iiwa_limits_upper

class TestLowLevel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.grasp_distance = 0.6
        cls.x = np.array([-0.4, -1.0, -1.1, 1.8, 0.4, 1.2, -2.5, 0.1])
        # x = [q_c (7), psi (1)]

    def check_gradient(self, constraint_or_cost, x, atol=1e-3):
        # 1. Eval double
        y_val = constraint_or_cost.Eval(x)
        
        # 2. Eval AutoDiff and check gradient
        x_ad = InitializeAutoDiff(x)
        y_ad = constraint_or_cost.Eval(x_ad)
        grad_ad = ExtractGradient(y_ad)
        
        # 3. Finite difference
        def f(x_in):
            return constraint_or_cost.Eval(x_in)
        
        eps = 1e-6
        J_fd = np.zeros((len(y_val), len(x)))
        for i in range(len(x)):
            x_p = x.copy(); x_p[i] += eps
            x_m = x.copy(); x_m[i] -= eps
            J_fd[:, i] = (f(x_p) - f(x_m)) / (2.0 * eps)
        
        self.assertTrue(np.allclose(grad_ad, J_fd, rtol=1e-3, atol=atol), 
                        f"Gradient mismatch for {constraint_or_cost.__class__.__name__}")

    def test_reachable_constraint(self):
        config = BimanualConfig(shoulder_up=True, elbow_up=True, wrist_up=True, grasp_distance=self.grasp_distance)
        c = IiwaBimanualReachableConstraint(config)
        self.check_gradient(c, self.x)

    def test_joint_limit_constraint(self):
        config = BimanualConfig(shoulder_up=True, elbow_up=True, wrist_up=True, grasp_distance=self.grasp_distance)
        c = IiwaBimanualJointLimitConstraint(
            iiwa_limits_lower, iiwa_limits_upper, config, AutoDiffConfig(False)
        )
        self.check_gradient(c, self.x)

    def test_psi_singularity_constraint(self):
        config = BimanualConfig(shoulder_up=True, elbow_up=True, wrist_up=True, grasp_distance=self.grasp_distance)
        c = IiwaBimanualPsiSingularityConstraint(config, AutoDiffConfig(False))
        self.check_gradient(c, self.x)

    def test_path_cost(self):
        # x_path = [x1, x2] (16D)
        x_path = np.concatenate([self.x, self.x + 0.01])
        config = BimanualConfig(shoulder_up=True, elbow_up=True, wrist_up=True, grasp_distance=self.grasp_distance)
        c = IiwaBimanualPathCost(8, 2, config, AutoDiffConfig(True), True)
        self.check_gradient(c, x_path)

    def test_path_cost_control_point_ordering(self):
        """
        IiwaBimanualPathCost reads its flat decision vector as a
        (num_positions x num_control_points) matrix laid out column-major, so
        `control_points()` must be flattened with order='F'. numpy's default is
        row-major, which the cost silently accepts: it returns a finite number
        that is the energy of a scrambled path, and trajopt then follows its
        gradient away from any sensible trajectory. That bug shipped in the
        notebook, where nothing compared the cost against a known value, so
        pin the ordering against a hand-computed reference here.
        """
        from pydrake.all import (
            BsplineBasis,
            BsplineTrajectory,
            KinematicTrajectoryOptimization,
        )

        config = BimanualConfig(shoulder_up=True, elbow_up=True, wrist_up=False,
                                grasp_distance=self.grasp_distance)
        ad_config = AutoDiffConfig(use_ift=False)
        parameterization = MakeParameterization(
            config, ad_config).get_parameterization_double()

        # A short, non-degenerate path in the parameterized space. Four control
        # points keeps the count different from the 8 positions, so a transposed
        # layout is a different shape and cannot coincidentally agree.
        control_points = np.array([
            self.x + 0.06 * i * np.array([1, -0.8, -1, 0.7, -0.3, 0.5, 0.9, 0.6])
            for i in range(4)
        ]).T
        self.assertEqual(control_points.shape, (8, 4))

        trajopt = KinematicTrajectoryOptimization(BsplineTrajectory(
            BsplineBasis(4, control_points.shape[1],
                         initial_parameter_value=0.0,
                         final_parameter_value=1.0),
            control_points))
        self.assertEqual(trajopt.control_points().shape, control_points.shape)

        # Reference: energy of the path the control points describe, measured in
        # the full 14-D configuration space.
        q_full = np.array([parameterization(control_points[:, i])
                           for i in range(control_points.shape[1])])
        expected = float(np.sum(np.diff(q_full, axis=0) ** 2))
        self.assertGreater(expected, 1e-6)

        cost = IiwaBimanualPathCost(trajopt.num_positions(),
                                    trajopt.num_control_points(),
                                    config, ad_config, True)

        binding = trajopt.prog().AddCost(
            cost, trajopt.control_points().flatten(order='F'))
        value = float(cost.Eval(
            trajopt.prog().GetInitialGuess(binding.variables()))[0])
        self.assertAlmostEqual(value, expected, places=9)

        # The row-major flatten must not be mistaken for the same path, or this
        # test can no longer detect a transposed decision vector.
        wrong = float(cost.Eval(control_points.flatten())[0])
        self.assertGreater(abs(wrong - expected), 1.0)

    def test_parameterization_consistency(self):
        config_ad = BimanualConfig(shoulder_up=True, elbow_up=True, wrist_up=True, grasp_distance=self.grasp_distance)
        p_ad = MakeParameterization(config_ad, AutoDiffConfig(True))
        config_double = BimanualConfig(shoulder_up=True, elbow_up=True, wrist_up=True, grasp_distance=self.grasp_distance)
        p_double = MakeParameterization(config_double, AutoDiffConfig(False))
        
        val_ad = p_ad.get_parameterization_double()(self.x)
        val_double = p_double.get_parameterization_double()(self.x)
        
        self.assertTrue(np.allclose(val_ad, val_double, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
