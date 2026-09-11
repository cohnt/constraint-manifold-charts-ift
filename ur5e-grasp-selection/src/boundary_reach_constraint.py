"""
boundary_reach_constraint.py

Implements a Python-native analogue of the C++ BoundaryReachabilityConstraint
from the bimanual IIWA codebase.  The constraint is:

    y(0) = -log(det(J(q) · J(q)ᵀ + ε · I))  ≤  threshold

where J(q) is the 6×6 geometric (spatial-velocity) Jacobian of the UR5e EE.

For the AutoDiff (gradient) path the Jacobian derivative dJ/dq is computed
analytically using MultibodyPlant<AutoDiffXd>.  The chain rule then gives:

    dy/dp = (dy/dq) · (dq/dp)

where dq/dp is the IFT Jacobian inverse (already supplied by GetQ).

This mirrors the C++ DoEval specialisation for AutoDiffXd in constraints.cc.
"""

import numpy as np
import pydrake.autodiffutils as adutils
from pydrake.all import (
    JacobianWrtVariable,
    InitializeAutoDiff,
    ExtractGradient,
    ExtractValue,
)


class BoundaryReachConstraint:
    """
    Callable-constraint (not a drake.Constraint subclass; used with
    prog.AddConstraint(fn, lb, ub, vars)) that evaluates:

        y = -log(det(J·Jᵀ + ε·I))

    Inputs (p_vars): 6-DOF minimal grasp parameters.
    Output (y):      scalar ≤ threshold  ⟹  lb=[−∞], ub=[threshold].

    The gradient w.r.t. p_vars is computed analytically via
    MultibodyPlant<AutoDiffXd>.

    Parameters
    ----------
    problem : UrIKProblemNewFormulation
        Parent problem, used to call GetQ() and access plant/ad_plant.
    epsilon : float
        Regularisation for the Gram matrix (analogous to boundary_epsilon).
    length_scale : float
        Characteristic length used to make J dimensionally homogeneous (see below).

    Units
    -----
    The spatial-velocity Jacobian's first three rows are angular (rad/s) and its last
    three are translational (m/s), so det(J Jᵀ) mixes units and no single epsilon is
    dimensionally coherent.  Scaling the angular rows by a characteristic length L puts
    every entry in metres, so the singular values are in metres and epsilon is in m².
    L = 1.12 m is the maximum |p_tool0| over the UR5e's configuration space (measured by
    sampling); with that scaling, sigma_max is ~2.2 and the median sigma_min is ~0.073.

    Choosing epsilon
    ----------------
    epsilon must be far below a typical sigma_min² (0.073² = 5.4e-3) so it does not
    perturb well-conditioned configurations, well above the floating-point floor
    (sigma_max² * eps_machine ~ 1e-15), and it caps the constraint's gradient, which is
    O(1/epsilon) at an exact singularity.  1e-6 satisfies all three.
    """

    DEFAULT_LENGTH_SCALE = 1.12

    def __init__(self, problem, epsilon: float = 1e-6,
                 length_scale: float = DEFAULT_LENGTH_SCALE):
        self.problem = problem
        self.epsilon = epsilon
        self.length_scale = length_scale
        # Left-multiplier that converts the angular rows of J into metres.
        self._row_scale = np.diag([length_scale] * 3 + [1.0] * 3)

    # ------------------------------------------------------------------
    # Helper: compute J and dJ/dq analytically via MultibodyPlant<AD>
    # ------------------------------------------------------------------
    def _compute_J_and_dJdq(self, q_val: np.ndarray):
        """
        Returns
        -------
        J : (6, 6) float ndarray
            Geometric Jacobian at q_val.
        dJ_dq : list of (6, 6) float ndarrays, length 6
            dJ_dq[k] = ∂J/∂q_k.
        """
        prob = self.problem
        nq = len(q_val)

        # ── AutoDiff plant: q seeded with identity gradient ──────────────
        q_ad = InitializeAutoDiff(q_val, np.eye(nq))
        prob.ad_plant.SetPositions(
            prob.ad_plant_context, prob.ad_arm_instance, q_ad
        )
        J_ad = prob.ad_plant.CalcJacobianSpatialVelocity(
            prob.ad_plant_context,
            JacobianWrtVariable.kV,
            prob.ad_plant.GetFrameByName("tool0", prob.ad_arm_instance),
            np.zeros(3),
            prob.ad_plant.world_frame(),
            prob.ad_plant.world_frame(),
        )
        # J_ad is (6, nq) of AutoDiffXd
        # Slice to the arm's DOF indices (same logic as plant_jacobian_for_model)
        J_ad = self._slice_jacobian_ad(prob, J_ad)

        J_val = ExtractValue(J_ad)           # (6, 6) float
        # ExtractGradient gives (6*6, nq).  Drake follows Eigen and flattens the matrix
        # COLUMN-major, so the reshape must use Fortran order; numpy's default C order
        # silently yields the transpose of ∂J/∂q_k.
        dJ_flat = ExtractGradient(J_ad)      # (36, 6)
        # dJ_dq[k] = ∂J/∂q_k is a (6, 6) matrix
        dJ_dq = [dJ_flat[:, k].reshape(6, 6, order="F") for k in range(nq)]

        # Row scaling is a constant left-multiplication, so it passes through d/dq.
        J_val = self._row_scale @ J_val
        dJ_dq = [self._row_scale @ dJ for dJ in dJ_dq]

        return J_val, dJ_dq

    @staticmethod
    def _slice_jacobian_ad(prob, J_full_ad):
        """Extract columns belonging to the arm model instance."""
        indices = []
        for idx in prob.ad_plant.GetJointIndices():
            joint = prob.ad_plant.get_joint(idx)
            if joint.model_instance() == prob.ad_arm_instance:
                for k in range(joint.num_velocities()):
                    indices.append(joint.velocity_start() + k)
        return J_full_ad[:, indices]

    # ------------------------------------------------------------------
    # Constraint value (double path)
    # ------------------------------------------------------------------
    def _value(self, q_val: np.ndarray) -> float:
        prob = self.problem
        prob.plant.SetPositions(prob.plant_context, prob.arm_instance, q_val)
        J = prob.plant.CalcJacobianSpatialVelocity(
            prob.plant_context,
            JacobianWrtVariable.kV,
            prob.ee_frame,
            np.zeros(3),
            prob.plant.world_frame(),
            prob.plant.world_frame(),
        )
        # Slice to arm DOF
        indices = []
        for idx in prob.plant.GetJointIndices():
            joint = prob.plant.get_joint(idx)
            if joint.model_instance() == prob.arm_instance:
                for k in range(joint.num_velocities()):
                    indices.append(joint.velocity_start() + k)
        J = self._row_scale @ J[:, indices]

        A = J @ J.T + self.epsilon * np.eye(6)
        sign, logdet = np.linalg.slogdet(A)
        if sign <= 0:
            return 1e10   # degenerate
        return -logdet

    # ------------------------------------------------------------------
    # Main callable (used with prog.AddConstraint)
    # ------------------------------------------------------------------
    def __call__(self, p_vars):
        is_ad = isinstance(p_vars[0], adutils.AutoDiffXd)

        # ── Get q (with IFT gradient if AD) ──────────────────────────────
        q = self.problem.GetQ(p_vars)   # returns AutoDiffXd array if AD

        if not is_ad:
            q_val = np.asarray(q, dtype=float)
            return np.array([self._value(q_val)])

        # ── AutoDiff path ─────────────────────────────────────────────────
        q_val  = ExtractValue(q)          # (6,) float
        dq_dp  = ExtractGradient(q)       # (6, n_vars)  = dq/dp

        J_val, dJ_dq = self._compute_J_and_dJdq(q_val)

        A     = J_val @ J_val.T + self.epsilon * np.eye(6)
        A_inv = np.linalg.inv(A)

        # dy/dq_k = -2 · tr(Jᵀ · A⁻¹ · dJ/dq_k)   [mirrors constraints.cc]
        JT_Ainv = J_val.T @ A_inv          # (6, 6)
        dy_dq = np.array([-2.0 * np.trace(JT_Ainv @ dJ_dq[k])
                           for k in range(len(q_val))])  # (6,)

        # dy/dp = (dy/dq) · (dq/dp)
        dy_dp = dy_dq @ dq_dp              # (n_vars,)

        sign, logdet = np.linalg.slogdet(A)
        y_val = -logdet if sign > 0 else 1e10

        return adutils.InitializeAutoDiff(
            np.array([y_val]),
            dy_dp.reshape(1, -1),
        )

    # ------------------------------------------------------------------
    # The same measure as an objective, in q-space
    # ------------------------------------------------------------------
    def value_and_grad_q(self, q_val: np.ndarray):
        """
        Return (y, dy/dq) for y = -log det(J·Jᵀ + ε·I) as a function of q alone.

        This is the identical quantity __call__ evaluates; the only difference is that
        the chain rule stops at q instead of continuing through dq/dp.  Factoring it out
        is what lets the *Old* formulation carry a manipulability objective built from
        exactly the same expression as the Boundary formulation's constraint, so an
        objective comparison is not confounded by two different manipulability measures.
        """
        J_val, dJ_dq = self._compute_J_and_dJdq(np.asarray(q_val, dtype=float))
        A = J_val @ J_val.T + self.epsilon * np.eye(6)
        sign, logdet = np.linalg.slogdet(A)
        if sign <= 0:
            return 1e10, np.zeros(len(q_val))
        JT_Ainv = J_val.T @ np.linalg.inv(A)
        dy_dq = np.array([-2.0 * np.trace(JT_Ainv @ dJ_dq[k])
                          for k in range(len(q_val))])
        return -logdet, dy_dq

    def cost_in_q(self, q_vars):
        """
        Drake cost callable over the *joint* variables, for the Old formulation.

        Branches on AutoDiffXd per the repo convention for hand-assembled gradients.
        """
        is_ad = isinstance(q_vars[0], adutils.AutoDiffXd)
        q_val = adutils.ExtractValue(q_vars).flatten() if is_ad \
            else np.asarray(q_vars, dtype=float)
        y, dy_dq = self.value_and_grad_q(q_val)
        if not is_ad:
            return y
        dq = adutils.ExtractGradient(q_vars)          # (6, n_vars)
        return adutils.InitializeAutoDiff(
            np.array([y]), (dy_dq @ dq).reshape(1, -1)
        ).flatten()[0]

    def cost_in_p(self, p_vars):
        """
        Drake cost callable over the grasp parameters, for the minimal-coordinate
        formulations.  Reuses __call__, which already carries the IFT chain rule.

        __call__ is a *constraint* callable, so it returns a length-1 vector — and on the
        AutoDiff path InitializeAutoDiff makes that (1, 1).  A cost must return a true
        scalar, so flatten before indexing; out[0] alone yields a (1,) array and Drake
        rejects it with "Return value must be of .ndim = 0".
        """
        return np.asarray(self(p_vars)).flatten()[0]
