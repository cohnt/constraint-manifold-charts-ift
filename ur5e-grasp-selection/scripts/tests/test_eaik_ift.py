import unittest
import numpy as np
import os, sys
from pydrake.all import StartMeshcat, RigidTransform, RollPitchYaw, AutoDiffXd

# Add repo root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.util import BuildEnv
from src.eaik_ik import EaikIK
from src.ift_gradients import IftGradient
from src.ur_experiments import UrProblemOptions, UrIKProblemNewFormulation

class TestEaikIft(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.meshcat = StartMeshcat()
        cls.diagram = BuildEnv(cls.meshcat, visualize=False)
        cls.plant = cls.diagram.GetSubsystemByName("plant")
        cls.arm_instance = cls.plant.GetModelInstanceByName("ur5e")
        
        # IK/IFT Tools
        urdf_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../models/universal_robots/ur_description/urdf/ur5e.urdf"))
        cls.ik_solver = EaikIK(urdf_path)
        cls.ift_tool = IftGradient(cls.plant, cls.arm_instance)

    def test_eaik_roundtrip(self):
        # Forward kinematics
        q_test = np.array([0.1, -1.2, 0.8, -1.5, -1.5, 0.2])
        X_WE = self.ik_solver.fk(q_test)
        
        # Inverse kinematics
        sols = self.ik_solver.solve(X_WE)
        self.assertTrue(len(sols) > 0, "EAIK failed to find any solutions for a valid FK pose")
        
        # Check if one of the solutions matches q_test (approximately)
        found = False
        for s in sols:
            if np.allclose(s, q_test, atol=1e-3):
                found = True
                break
        self.assertTrue(found, f"EAIK solutions {sols} do not include original q {q_test}")

    def test_ift_gradient(self):
        q = np.array([0.1, -1.2, 0.8, -1.5, -1.5, 0.2])
        # The pose below is exactly reachable, so the spatial residual is zero and the
        # residual-damped inverse reduces to the exact J^-1 that finite differences see.
        J_inv = self.ift_tool.compute_dq_dpose(q, residual_6d=np.zeros(6))

        # Finite difference check
        eps = 1e-6
        X_WE = self.ik_solver.fk(q)
        
        # Perturb in x
        X_WE_eps = X_WE.copy()
        X_WE_eps[0, 3] += eps
        sols_eps = self.ik_solver.solve(X_WE_eps)
        # Pick solution closest to q
        q_eps = min(sols_eps, key=lambda s: np.linalg.norm(s - q))
        
        dq_dx_num = (q_eps - q) / eps
        # dq/dx is the 4th column of dq/dpose (indices 3,4,5 are translation)
        # Spatial velocity V = [w; v]. Translation v_x is index 3.
        dq_dx_ift = J_inv[:, 3]
        
        np.testing.assert_allclose(dq_dx_num, dq_dx_ift, atol=1e-4)

    def test_grasp_params_round_trip(self):
        """
        p must describe the grasp frame in the mug frame, not tool0 in the mug frame.
        Seeding p from a joint configuration and mapping it back must be the identity;
        the two differ by a conjugation by X_EG, which does not cancel.
        """
        prob = UrIKProblemNewFormulation(self.diagram)
        rng = np.random.default_rng(0)
        for _ in range(25):
            q_gold = rng.uniform(-np.pi, np.pi, 6)
            prob.plant.SetPositions(prob.plant_context, prob.arm_instance, q_gold)
            X_WM = prob.ee_frame.CalcPoseInWorld(prob.plant_context).multiply(prob.X_EG)

            q_init = rng.uniform(-np.pi, np.pi, 6)
            p = prob.GraspParamsFromQ(q_init, X_WM)

            X_WG_from_p = X_WM.multiply(RigidTransform(RollPitchYaw(p[3:]), p[:3]))
            prob.plant.SetPositions(prob.plant_context, prob.arm_instance, q_init)
            X_WG_actual = prob.ee_frame.CalcPoseInWorld(prob.plant_context).multiply(prob.X_EG)
            np.testing.assert_allclose(
                X_WG_from_p.GetAsMatrix4(), X_WG_actual.GetAsMatrix4(), atol=1e-9
            )

    def test_branch_selection_prefers_exact_solutions(self):
        """
        EAIK returns least-squares rows mixed in with exact ones (about 6% of rows even
        for reachable poses).  Branch selection must never return one of those when an
        exact solution exists, and must still return one when nothing else is available,
        since the least-squares rows are the domain extension EIK.
        """
        prob = UrIKProblemNewFormulation(self.diagram)
        rng = np.random.default_rng(1)

        n_checked = 0
        for _ in range(60):
            q = rng.uniform(-np.pi, np.pi, 6)
            X_WE = self.ik_solver.fk(q)
            sols, is_ls = self.ik_solver.solve_all(X_WE)
            self.assertGreater(len(sols), 0, "EAIK should never return an empty set")
            if is_ls.all():
                continue
            prob.q_branch_ref = q
            q_sel = prob.SelectBranch(sols, is_ls)
            # The selected row must be exact: its FK must reproduce the requested pose.
            np.testing.assert_allclose(self.ik_solver.fk(q_sel), X_WE, atol=1e-6)
            n_checked += 1
        self.assertGreater(n_checked, 0)

        # Far outside the workspace, only least-squares rows exist; selection must still
        # return one rather than failing.
        X_far = self.ik_solver.fk(np.zeros(6))
        X_far[:3, 3] *= 5.0
        sols, is_ls = self.ik_solver.solve_all(X_far)
        self.assertGreater(len(sols), 0)
        prob.q_branch_ref = np.zeros(6)
        self.assertEqual(prob.SelectBranch(sols, is_ls).shape, (6,))

    def test_boundary_constraint_dy_dq(self):
        """
        The constraint's analytic dy/dq must match finite differences of its value.

        This deliberately isolates dy/dq from the chain rule dy/dp = (dy/dq)(dq/dp): the
        dq/dp factor is the damped IFT approximation, so finite-differencing the full
        chain would measure the approximation, not the implementation.  It is dy/dq that
        the Jacobian row scaling touches.
        """
        prob = UrIKProblemNewFormulation(self.diagram, use_boundary_reach=True)
        constraint = prob._boundary_constraint

        rng = np.random.default_rng(3)
        for _ in range(5):
            q = rng.uniform(-1.5, 1.5, 6)
            J, dJ_dq = constraint._compute_J_and_dJdq(q)
            A_inv = np.linalg.inv(J @ J.T + constraint.epsilon * np.eye(6))
            JT_Ainv = J.T @ A_inv
            analytic = np.array([-2.0 * np.trace(JT_Ainv @ dJ_dq[k]) for k in range(6)])

            eps = 1e-6
            numeric = np.empty(6)
            for k in range(6):
                qp, qm = q.copy(), q.copy()
                qp[k] += eps
                qm[k] -= eps
                numeric[k] = (constraint._value(qp) - constraint._value(qm)) / (2 * eps)
            np.testing.assert_allclose(analytic, numeric, rtol=1e-4, atol=1e-5)

    def test_boundary_constraint_is_length_scaled(self):
        """
        The Jacobian must be length-scaled in both the value and gradient paths, or the
        two disagree and epsilon has no coherent units.
        """
        prob = UrIKProblemNewFormulation(self.diagram, use_boundary_reach=True)
        constraint = prob._boundary_constraint
        q = np.array([0.3, -1.0, 0.7, -1.2, -1.4, 0.5])

        J_from_grad_path, _ = constraint._compute_J_and_dJdq(q)
        sigma = np.linalg.svd(J_from_grad_path, compute_uv=False)
        value_from_sigma = -np.sum(np.log(sigma ** 2 + constraint.epsilon))
        self.assertAlmostEqual(constraint._value(q), value_from_sigma, places=9)

    def test_optimization_loop(self):
        prob = UrIKProblemNewFormulation(self.diagram)
        options = UrProblemOptions(
            avoid_collisions=False,
            target_mug=RigidTransform(RollPitchYaw(0, 0, 0), [0.5, 0, 0.1])
        )
        prob.ApplyOptions(options)
        # Set a non-singular initial guess to prevent SQP active-set gimbal lock
        prob.prog.SetInitialGuess(prob.p, np.array([0.0, 0.0, 0.0, 0.1, -0.2, 0.3]))
        res = prob.Solve()
        self.assertTrue(res.is_success(), "NewFormulation failed to solve a simple problem")

if __name__ == "__main__":
    unittest.main()
