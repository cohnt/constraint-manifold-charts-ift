import numpy as np
from dataclasses import dataclass
from functools import partial

import pydrake.autodiffutils as adutils
from pydrake.all import (
    InverseKinematics,
    SnoptSolver,
    IpoptSolver,
    NloptSolver,
    SolverOptions,
    RollPitchYaw,
    RollPitchYaw_,
    RigidTransform,
    RigidTransform_,
    MathematicalProgram,
    AutoDiffXd,
    MinimumDistanceLowerBoundConstraint,
)

from src.util import RepoDir, BuildEnv
from src.eaik_ik import EaikIK
from src.ift_gradients import IftGradient
from src.boundary_reach_constraint import BoundaryReachConstraint

# UR5e joint limits (from the URDF).  These are the hardware limits: the real UR5e
# permits +/-360 deg on all six joints, and the +/-pi elbow bound is the ROS-Industrial
# self-collision convention that the URDF inherits.  The URDF is left alone on purpose —
# it has to keep describing the robot faithfully.
ur5e_limits_lower = np.array([-2*np.pi, -2*np.pi, -np.pi, -2*np.pi, -2*np.pi, -2*np.pi])
ur5e_limits_upper = np.array([ 2*np.pi,  2*np.pi,  np.pi,  2*np.pi,  2*np.pi,  2*np.pi])

# Limits imposed on the *solver*, in software only, so the three formulations are
# compared over the same feasible set.
#
# EAIK's roots come out of atan2, so the minimal-coordinate formulations (Direct and
# Boundary) are structurally confined to [-pi, pi] on every joint and can never pay for
# a 2*pi wrap.  The Old formulation, given the full +/-2*pi hardware box, could land on a
# wrapped copy of the same physical configuration and be charged ||q - q_nominal||^2
# including that wrap: in the 100x10 benchmark 37 of 595 successful Old runs scored above
# 6*pi^2 ~ 59.2 (max 185.6), which is exactly the region the other two methods cannot
# reach.  That inflated Old's mean cost while leaving its median close to Direct's.
#
# Restricting the optimizer to [-pi, pi] is physically lossless: every joint angle
# outside that range wraps to an angle inside it that gives the identical pose, identical
# collisions and identical task-space error, so no physical solution is removed — only
# its redundant wrapped representatives.  The hardware genuinely allows +/-2*pi, which is
# why this restriction lives here and not in the URDF.
ur5e_solver_limits_lower = np.full(6, -np.pi)
ur5e_solver_limits_upper = np.full(6,  np.pi)

@dataclass
class UrProblemOptions:
    # Collision options
    avoid_collisions: bool = True
    minimum_distance: float = 0.001
    influence_distance_offset: float = 0.01

    # Cost options
    impose_joint_centering_cost: bool = True
    joint_centering_cost_multiplier: float = 1e0
    # What the centering cost is measured against.  "nominal" is ||q - q_nominal||^2, the
    # posture prior all published numbers use.  "initial" is ||q - q_initial||^2, i.e. stay
    # near the seed configuration, which is what re-planning actually asks for and what
    # rewards the branch tracking the minimal-coordinate methods do natively.  Applied
    # identically to all three formulations, so the comparison stays fair.
    cost_reference: str = "nominal"   # "nominal" or "initial"
    # Optional per-joint weights for the centering cost: sum(w_i * (q_i - q_ref_i)^2).
    # None means uniform.  A shoulder/elbow-heavy weighting expresses that moving the big
    # proximal joints is more expensive than the wrist.
    joint_cost_weights: np.ndarray = None
    # Weight on a manipulability objective, w * (-log det(J Jᵀ + εI)).  0 disables it.
    # The expression is the Boundary formulation's constraint function reused as a cost,
    # so all three formulations optimise the identical measure and the comparison is not
    # confounded by two different definitions of manipulability.
    manipulability_cost_weight: float = 0.0
    manipulability_epsilon: float = 1e-6
    ift_damping_lam: float = 1e-4
    ift_strategy: str = "residual"
    square_frobenius_norm: bool = False
    
    # Constraints
    joint_limits: bool = True
    mug_pose_constraint: bool = True
    mug_height: float = 0.08  # Approx height of the mug
    # Bounds on the grasp frame's orientation in the mug frame.  The defaults are
    # deliberately vacuous, so the grasp orientation is unconstrained.  Tightening the
    # roll/pitch bound forces upright grasps that differ only in yaw about the mug axis,
    # which is what the grasp-selection figure wants; the benchmark leaves them open.
    grasp_roll_pitch_bound: float = np.pi
    grasp_yaw_bounds: tuple = (-np.pi, np.pi)
    
    # Solver options.  These belong to the *comparison*, not to any one formulation, so
    # UrIKProblem.Solve applies them identically to all three.
    solver: str = 'SNOPT'          # SNOPT, IPOPT or NLOPT
    max_wall_time: float = 10.0
    # Feasibility is what verify_solution ultimately re-checks, so it stays tight.
    # Optimality is the knob worth sweeping: the task tolerance is 1e-2 m, four orders
    # looser, so a very tight optimality tolerance may buy only extra iterations.
    major_optimality_tol: float = 1e-6
    minor_optimality_tol: float = 1e-6
    major_feasibility_tol: float = 1e-6
    # NLOPT has no wall-clock limit in Drake's wrapper, only an evaluation cap.
    nlopt_max_eval: int = 20000

    # Initial guess
    q_initial: np.ndarray = None
    target_mug: RigidTransform = RigidTransform()

class UrIKProblem:
    def __init__(self, diagram, urdf_path=None):
        self.diagram = diagram
        self.plant = diagram.GetSubsystemByName("plant")
        self.arm_instance = self.plant.GetModelInstanceByName("ur5e")
        
        self.diagram_context = self.diagram.CreateDefaultContext()
        self.plant_context = self.plant.GetMyContextFromRoot(self.diagram_context)
        
        self.ee_frame = self.plant.GetFrameByName("tool0", self.arm_instance)
        
        # For IFT/EAIK
        self.ik_solver = EaikIK(urdf_path)
        self.ift_tool = IftGradient(self.plant, self.arm_instance)
        
        # For AutoDiff (FK, etc.)
        self.ad_plant = self.plant.ToAutoDiffXd()
        self.ad_plant_context = self.ad_plant.CreateDefaultContext()
        self.ad_arm_instance = self.ad_plant.GetModelInstanceByName("ur5e")
        
        # Nominal q: the midpoint of the joint limits, which for the UR5e is the origin.
        # Centering on it makes the cost ||q - q_nominal||^2 = ||q||^2.  The solver box is
        # symmetric too, so tightening it to +/-pi leaves q_nominal (and hence the
        # objective) unchanged; the assertion pins that down.
        self.q_nominal = 0.5 * (ur5e_limits_lower + ur5e_limits_upper)
        assert np.allclose(self.q_nominal, np.zeros(6)), "q_nominal must be the origin"
        assert np.allclose(
            self.q_nominal, 0.5 * (ur5e_solver_limits_lower + ur5e_solver_limits_upper)
        ), "solver limits must be centered on the same q_nominal"
        self.q_target = self.q_nominal.copy()
        self.cost_weights = np.ones(6)

        # The gripper grasp frame G relative to the arm flange E (tool0).
        # G is centered between the fingers, 0.15 m along tool0's +z: its x-axis is the
        # approach direction (tool0 +z) and its y-axis is the finger-closing direction.
        self.X_EG = RigidTransform(RollPitchYaw(0, -np.pi/2, np.pi/2), np.array([0, 0, 0.15]))

    def GraspParamsFromQ(self, q, X_WM):
        """
        p = [x, y, z, r, p, y] of the grasp frame G in the manipuland frame M, for the
        configuration q.  This is the inverse of the map the decision variables define,
        so it is the correct way to seed p from a joint-space initial guess.
        """
        self.plant.SetPositions(self.plant_context, self.arm_instance, np.asarray(q).flatten())
        X_WG = self.ee_frame.CalcPoseInWorld(self.plant_context).multiply(self.X_EG)
        X_MG = X_WM.inverse().multiply(X_WG)
        return np.concatenate([
            X_MG.translation(),
            X_MG.rotation().ToRollPitchYaw().vector(),
        ])

    def EvalCollision(self, q, minimum_distance=0.001):
        # Returns > 0 if in collision (penalty)
        q = adutils.ExtractValue(q.flatten())
        
        self.plant.SetPositions(self.plant_context, self.arm_instance, q)
        query_object = self.diagram.GetSubsystemByName("scene_graph").get_query_output_port().Eval(
            self.diagram.GetSubsystemByName("scene_graph").GetMyContextFromRoot(self.diagram_context)
        )
            
        dist = query_object.ComputeSignedDistancePairwiseClosestPoints(minimum_distance)
        penalty = 0
        for d in dist:
            if d.distance < minimum_distance:
                penalty += (minimum_distance - d.distance)
        return penalty

    def EvalCollisionFreeConstraint(self, p_vars):
        q = self.GetQ(p_vars)
        return self.collision_free_constraint.Eval(q)

    def Solve(self):
        """
        Solve self.prog with the configured solver.

        Shared by every formulation so the three are never accidentally compared under
        different solver settings -- tolerances and the time limit are a property of the
        comparison, not of the formulation.

        NLOPT is the odd one out: Drake's wrapper exposes no wall-clock limit, only an
        evaluation cap, so `max_wall_time` does not bind for it and its runs are not
        directly time-comparable with the other two.
        """
        name = self.options.solver.upper()
        solver_options = SolverOptions()

        if name == 'SNOPT':
            solver = SnoptSolver()
            solver_options.SetOption(SnoptSolver.id(), "Timing level", 3)
            solver_options.SetOption(SnoptSolver.id(), "Time limit", self.options.max_wall_time)
            solver_options.SetOption(SnoptSolver.id(), "Major optimality tolerance",
                                     self.options.major_optimality_tol)
            solver_options.SetOption(SnoptSolver.id(), "Minor optimality tolerance",
                                     self.options.minor_optimality_tol)
            solver_options.SetOption(SnoptSolver.id(), "Major feasibility tolerance",
                                     self.options.major_feasibility_tol)
        elif name == 'IPOPT':
            solver = IpoptSolver()
            solver_options.SetOption(IpoptSolver.id(), "max_wall_time", self.options.max_wall_time)
            solver_options.SetOption(IpoptSolver.id(), "tol", self.options.major_optimality_tol)
            solver_options.SetOption(IpoptSolver.id(), "acceptable_tol",
                                     self.options.major_optimality_tol)
            solver_options.SetOption(IpoptSolver.id(), "constr_viol_tol",
                                     self.options.major_feasibility_tol)
            solver_options.SetOption(IpoptSolver.id(), "acceptable_constr_viol_tol",
                                     self.options.major_feasibility_tol)
        elif name == 'NLOPT':
            solver = NloptSolver()
            solver_options.SetOption(NloptSolver.id(), NloptSolver.ConstraintToleranceName(),
                                     self.options.major_feasibility_tol)
            solver_options.SetOption(NloptSolver.id(), NloptSolver.XRelativeToleranceName(),
                                     self.options.major_optimality_tol)
            solver_options.SetOption(NloptSolver.id(), NloptSolver.XAbsoluteToleranceName(),
                                     self.options.major_optimality_tol)
            solver_options.SetOption(NloptSolver.id(), NloptSolver.MaxEvalName(),
                                     self.options.nlopt_max_eval)
        else:
            raise ValueError(f"Unknown solver: {self.options.solver}")

        return solver.Solve(self.prog, solver_options=solver_options)

    def _ResolveCostReference(self, options: UrProblemOptions):
        """
        Set self.q_target and self.cost_weights from the options.

        Called at the top of every ApplyOptions, so all three formulations minimise the
        same objective by construction rather than by three copies agreeing.  The
        benchmark's eval_cost scores against q_target, so the scored objective is the
        minimised one whichever reference is chosen.
        """
        ref = options.cost_reference
        if ref == "nominal":
            self.q_target = self.q_nominal.copy()
        elif ref == "initial":
            if options.q_initial is None:
                raise ValueError('cost_reference="initial" requires options.q_initial')
            self.q_target = np.asarray(options.q_initial, dtype=float).copy()
        else:
            raise ValueError(f"Unknown cost_reference: {ref!r}")

        if options.joint_cost_weights is None:
            self.cost_weights = np.ones(6)
        else:
            w = np.asarray(options.joint_cost_weights, dtype=float)
            if w.shape != (6,):
                raise ValueError(f"joint_cost_weights must have shape (6,), got {w.shape}")
            if np.any(w < 0):
                raise ValueError("joint_cost_weights must be non-negative")
            self.cost_weights = w

class UrIKProblemOldFormulation(UrIKProblem):
    """
    Standard NLP formulation: optimize q (6 DOF) with EE-on-mug constraint.
    """
    def ApplyOptions(self, options: UrProblemOptions):
        self.options = options
        self._ResolveCostReference(options)
        self.ik = InverseKinematics(self.plant, self.plant_context)
        self.prog = self.ik.prog()
        
        if self.options.q_initial is not None:
            self.prog.SetInitialGuess(self.ik.q(), self.options.q_initial)
        else:
            self.prog.SetInitialGuess(self.ik.q(), self.q_nominal)

        if self.options.joint_limits:
            # Solver box, not the hardware box: see ur5e_solver_limits_* above.
            self.prog.AddBoundingBoxConstraint(
                ur5e_solver_limits_lower, ur5e_solver_limits_upper, self.ik.q()
            )

        if self.options.avoid_collisions:
            self.collision_free_constraint = MinimumDistanceLowerBoundConstraint(
                plant=self.plant,
                bound=self.options.minimum_distance,
                influence_distance_offset=self.options.influence_distance_offset,
                plant_context=self.plant_context
            )
            self.prog.AddConstraint(self.collision_free_constraint, self.ik.q())

        if self.options.mug_pose_constraint:
            # EE tool0 should be at mug frame with some tolerance in z/orientation
            # We'll use a custom constraint for EE relative to mug
            rp = self.options.grasp_roll_pitch_bound
            yaw_lb, yaw_ub = self.options.grasp_yaw_bounds
            self.prog.AddConstraint(
                partial(self.EvalMugRelativePose, mug_pose=self.options.target_mug),
                lb=np.array([0.0, 0.0, -self.options.mug_height/4, -rp, -rp, yaw_lb]),
                ub=np.array([0.0, 0.0,  self.options.mug_height/4,  rp,  rp, yaw_ub]),
                vars=self.ik.q()
            )

        if self.options.impose_joint_centering_cost:
            q_err = self.ik.q() - self.q_target
            self.prog.AddCost(
                self.options.joint_centering_cost_multiplier
                * q_err.dot(self.cost_weights * q_err)
            )

        if self.options.manipulability_cost_weight != 0.0:
            w = self.options.manipulability_cost_weight
            brc = BoundaryReachConstraint(self, self.options.manipulability_epsilon)
            self.prog.AddCost(lambda qv: w * brc.cost_in_q(qv), vars=self.ik.q())

    def EvalMugRelativePose(self, q, mug_pose):
        # returns [x, y, z, r, p, y] of grasp frame in mug frame
        is_ad = isinstance(q[0], AutoDiffXd)
        
        # Set positions
        if is_ad:
            self.ad_plant.SetPositions(self.ad_plant_context, self.ad_arm_instance, q)
            ad_ee_frame = self.ad_plant.GetFrameByName("tool0", self.ad_arm_instance)
            X_WE = ad_ee_frame.CalcPoseInWorld(self.ad_plant_context)
            X_EG_ad = RigidTransform_[AutoDiffXd](self.X_EG.GetAsMatrix4())
            X_WG = X_WE.multiply(X_EG_ad)
            
            X_WM = RigidTransform_[AutoDiffXd](mug_pose.GetAsMatrix4())
            X_MG = X_WM.inverse().multiply(X_WG)
            
            xyz = X_MG.translation()
            rpy = X_MG.rotation().ToRollPitchYaw().vector()
            return np.concatenate([xyz, rpy])
        else:
            self.plant.SetPositions(self.plant_context, self.arm_instance, q)
            X_WE = self.ee_frame.CalcPoseInWorld(self.plant_context)
            X_WG = X_WE.multiply(self.X_EG)
            X_MG = mug_pose.inverse().multiply(X_WG)
            xyz = X_MG.translation()
            rpy = X_MG.rotation().ToRollPitchYaw().vector()
            return np.concatenate([xyz, rpy])


class UrIKProblemNewFormulation(UrIKProblem):
    """
    Minimal coordinate formulation: optimize grasp parameters (6 DOF) relative to mug.
    q is computed via EAIK, and gradients via IFT.
    """
    def __init__(self, diagram, use_boundary_reach=False, epsilon=1e-2, threshold=20.0,
                 q_branch_ref=None):
        super().__init__(diagram)
        self.use_boundary_reach = use_boundary_reach
        self.epsilon = epsilon
        self.threshold = threshold
        # Reference configuration for branch selection; see SelectBranch.  Set this to the
        # solve's initial guess.  Falls back to q_nominal when unset.
        self.q_branch_ref = q_branch_ref
        if use_boundary_reach:
            self._boundary_constraint = BoundaryReachConstraint(self, epsilon)

    def ApplyOptions(self, options: UrProblemOptions):
        self.options = options
        self._ResolveCostReference(options)
        # GetQ depends on target_mug and on the IFT knobs, all of which live in options.
        self._getq_cache = {}
        self.prog = MathematicalProgram()

        # Decision variables: EE pose in mug frame p = [x, y, z, roll, pitch, yaw]
        self.p = self.prog.NewContinuousVariables(6, "grasp_params")
        self.prog.SetInitialGuess(self.p, np.zeros(6))

        h2 = options.mug_height / 4.0
        rp = options.grasp_roll_pitch_bound
        yaw_lb, yaw_ub = options.grasp_yaw_bounds
        self.prog.AddBoundingBoxConstraint(
            np.array([0.0, 0.0, -h2, -rp, -rp, yaw_lb]),
            np.array([0.0, 0.0,  h2,  rp,  rp, yaw_ub]),
            self.p,
        )

        # ── Reachability constraint ───────────────────────────────────────────
        if self.use_boundary_reach:
            # -log(det(J·Jᵀ + ε·I)) ≤ threshold
            self.prog.AddConstraint(
                self._boundary_constraint,
                lb=np.array([-np.inf]),
                ub=np.array([self.threshold]),
                vars=self.p,
            )
        else:
            # FK residual: ||FK(IK(X_WE(p))) - X_WE(p)||_F^2 * 100 <= 0 (no slack)
            self.reachability_binding = self.prog.AddConstraint(
                self.ReachabilityConstraint, lb=[0], ub=[0], vars=self.p
            )

        # ── Joint limits ──────────────────────────────────────────────────────
        if options.joint_limits:
            # Same solver box as the Old formulation, for consistency.  It is non-binding
            # here either way: EAIK's atan2 roots already lie in [-pi, pi].
            self.prog.AddConstraint(
                self.GetQ, lb=ur5e_solver_limits_lower, ub=ur5e_solver_limits_upper,
                vars=self.p,
            )

        # ── Costs ─────────────────────────────────────────────────────────────
        # Joint centering: ||q - q_target||^2 (see _ResolveCostReference)
        if options.impose_joint_centering_cost:
            self.prog.AddCost(self._joint_centering_cost, vars=self.p)

        if options.manipulability_cost_weight != 0.0:
            w = options.manipulability_cost_weight
            # Reuse the boundary instance when there is one, so the Boundary formulation's
            # cost and constraint share a cache-warm evaluator; build one otherwise.
            brc = getattr(self, "_boundary_constraint", None)
            if brc is None or brc.epsilon != options.manipulability_epsilon:
                brc = BoundaryReachConstraint(self, options.manipulability_epsilon)
            self.prog.AddCost(lambda pv: w * brc.cost_in_p(pv), vars=self.p)

        if options.avoid_collisions:
            self.collision_free_constraint = MinimumDistanceLowerBoundConstraint(
                plant=self.plant,
                bound=options.minimum_distance,
                influence_distance_offset=options.influence_distance_offset,
                plant_context=self.plant_context
            )
            self.prog.AddConstraint(
                self.EvalCollisionFreeConstraint,
                lb=[-np.inf],
                ub=[1.0],
                vars=self.p
            )

    def ReachabilityConstraint(self, p_vars):
        q = self.GetQ(p_vars)
        is_ad = isinstance(p_vars[0], AutoDiffXd)
        
        if is_ad:
            X_MG = RigidTransform_[AutoDiffXd](
                RollPitchYaw_[AutoDiffXd](p_vars[3:]),
                p_vars[:3]
            )
            X_WG_desired = RigidTransform_[AutoDiffXd](self.options.target_mug.GetAsMatrix4()).multiply(X_MG)
            X_EG_ad = RigidTransform_[AutoDiffXd](self.X_EG.GetAsMatrix4())
            X_WE_desired = X_WG_desired.multiply(X_EG_ad.inverse())
            
            # Actual pose from q
            self.ad_plant.SetPositions(self.ad_plant_context, self.ad_arm_instance, q)
            X_actual = self.ad_plant.GetFrameByName("tool0", self.ad_arm_instance).CalcPoseInWorld(self.ad_plant_context)
            
            A = X_actual.GetAsMatrix4()
            B = X_WE_desired.GetAsMatrix4()
            diff = A - B
            
            # Frobenius norm
            res = 0
            for i in range(4):
                for j in range(4):
                    res += diff[i, j] * diff[i, j]
            if self.options.square_frobenius_norm:
                return np.array([100.0 * res])
            else:
                return np.array([10.0 * (res ** 0.5)])
        else:
            p_val = p_vars
            X_MG = RigidTransform(RollPitchYaw(p_val[3:]), p_val[:3])
            X_WG_desired = self.options.target_mug.multiply(X_MG)
            X_WE_desired = X_WG_desired.multiply(self.X_EG.inverse())
            
            self.plant.SetPositions(self.plant_context, self.arm_instance, q)
            X_actual = self.plant.GetFrameByName("tool0", self.arm_instance).CalcPoseInWorld(self.plant_context)
            
            A = X_actual.GetAsMatrix4()
            B = X_WE_desired.GetAsMatrix4()
            diff = A - B
            if self.options.square_frobenius_norm:
                return np.array([100.0 * np.sum(diff**2)])
            else:
                return np.array([10.0 * np.sqrt(np.sum(diff**2))])

    def SelectBranch(self, sols, is_ls):
        """
        Pick one IK solution, tracking a branch continuously as the target pose moves.

        EAIK returns its solutions in subproblem-enumeration order and guarantees nothing
        about that order: when a subproblem's root count changes, every downstream index
        shifts, so a fixed list index does not denote a fixed kinematic branch (measured:
        ~11% of poses swap the branch at an index under a 1e-4 m perturbation, and that
        rate is unchanged by 2*pi wrapping, so the swaps are real).  Selecting the
        solution nearest to a reference configuration is order-independent, and because a
        least-squares root is the merge point of the two exact roots that vanish at a
        fold, joint values stay continuous across the workspace boundary.

        Exact solutions are preferred over least-squares ones; the least-squares rows are
        still selectable when nothing else is available, since they are EIK.
        """
        if len(sols) == 0:
            raise RuntimeError("EAIK returned no solutions, which should not happen.")

        q_ref = self.q_branch_ref if self.q_branch_ref is not None else self.q_nominal
        Q = np.asarray(sols, dtype=float).reshape(len(sols), -1)
        exact = ~np.asarray(is_ls, dtype=bool)
        if exact.any():
            rows = np.flatnonzero(exact)
            Q = Q[rows]
        else:
            rows = np.arange(len(sols))

        # Wrapped distance for every candidate at once: the per-candidate Python closure
        # this replaces was one of the three largest lines in the profile.
        d = (Q - q_ref + np.pi) % (2 * np.pi) - np.pi
        return sols[rows[int(np.argmin(np.einsum("ij,ij->i", d, d)))]]

    # Number of distinct p values kept in the GetQ cache.  Every cost and constraint in
    # this formulation calls GetQ independently, and the solver evaluates them all at the
    # same p: measured 4.1 GetQ calls per distinct p.  A handful of entries is enough to
    # collapse that to one, and the cache is cleared whenever anything it depends on
    # changes (see ApplyOptions and the q_branch_ref setter).
    GETQ_CACHE_SIZE = 8

    @property
    def q_branch_ref(self):
        """Reference configuration for SelectBranch; set this to the solve's initial guess."""
        return self._q_branch_ref

    @q_branch_ref.setter
    def q_branch_ref(self, value):
        # Changing the reference changes which branch GetQ returns, so anything cached
        # against the old reference is stale.  Making this a property means callers cannot
        # produce a stale hit by assigning the attribute directly.
        self._q_branch_ref = value
        self._getq_cache = {}

    def GetQ(self, p_vars):
        # p_vars is [x, y, z, r, p, y]
        is_ad = isinstance(p_vars[0], AutoDiffXd)
        p_val = adutils.ExtractValue(p_vars) if is_ad else p_vars

        q_val, dq_dp = self._SolveIkAt(p_val, need_gradient=is_ad)

        if not is_ad:
            return q_val

        # Package into AutoDiff — flatten to (6,) so downstream arithmetic
        # (e.g. q_err * q_err) produces scalar AutoDiffXd, not a 2D array.
        return adutils.InitializeAutoDiff(
            q_val, dq_dp @ adutils.ExtractGradient(p_vars)).flatten()

    def _SolveIkAt(self, p_val, need_gradient):
        """
        (q, dq/dp) at a given parameter value, memoised on p.

        The cached dq/dp is a function of p alone — it is J^-1 dpose/dp evaluated at the
        recovered q — so re-attaching it to a *different* set of AutoDiff seeds later is
        exact, not an approximation.  It does depend on the branch reference and on the
        applied options, which is why both of those invalidate the cache.

        A value-only entry is not reusable when a gradient is wanted, hence the second
        condition on the hit; the reverse is fine.
        """
        key = np.asarray(p_val, dtype=float).tobytes()
        hit = self._getq_cache.get(key)
        if hit is not None and (hit[1] is not None or not need_gradient):
            return hit

        q_val, dq_dp = self._SolveIkUncached(p_val, need_gradient=need_gradient)
        if len(self._getq_cache) >= self.GETQ_CACHE_SIZE:
            self._getq_cache.clear()
        self._getq_cache[key] = (q_val, dq_dp)
        return q_val, dq_dp

    def _SolveIkUncached(self, p_val, need_gradient):
        # Compute target pose of tool0 in world
        X_MG = RigidTransform(RollPitchYaw(p_val[3:]), p_val[:3])
        X_WG = self.options.target_mug.multiply(X_MG)
        X_WE = X_WG.multiply(self.X_EG.inverse())

        # Solve IK.  EAIK always returns a non-empty set: outside the reachable workspace
        # it substitutes least-squares roots, which is exactly the domain extension EIK.
        sols, is_ls = self.ik_solver.solve_all(X_WE.GetAsMatrix4())
        q_val = self.SelectBranch(sols, is_ls)

        q_val = q_val.flatten()

        if not need_gradient:
            return q_val, None

        # --- IFT GRADIENT ---
        # Compute 6D spatial residual error for IFT damping
        # X_WE is the target pose, X_EE_reached is the current pose.
        # r = [r_rot, r_trans]
        X_EE_reached = self.ik_solver.fk(q_val)
        X_EE_curr = RigidTransform(X_EE_reached)
        
        # Rotational error (axis-angle in World frame)
        R_err = X_WE.rotation().multiply(X_EE_curr.rotation().inverse())
        angle_axis = R_err.ToAngleAxis()
        r_rot = angle_axis.axis() * angle_axis.angle()
        
        # Translational error
        r_trans = X_WE.translation() - X_EE_curr.translation()
        
        residual_6d = np.concatenate([r_rot, r_trans])
        
        J_inv = self.ift_tool.compute_dq_dpose(
            q_val, residual_6d=residual_6d, lam=self.options.ift_damping_lam, strategy=self.options.ift_strategy
        )
        dpose_dp = self.ComputeDPoseDP(p_val)
        return q_val, J_inv @ dpose_dp

    def ComputeDPoseDP(self, p_val):
        # Returns 6x6 relating p_dot to spatial velocity V_WE
        # p = [x, y, z, r, p, y] for X_MG
        # V_WE = [w_WE; v_WE]
        
        X_MG = RigidTransform(RollPitchYaw(p_val[3:]), p_val[:3])
        X_WG = self.options.target_mug.multiply(X_MG)
        X_WE = X_WG.multiply(self.X_EG.inverse())
        
        R_WM = self.options.target_mug.rotation().matrix()
        
        # dw_WG / dp_rot
        rpy = p_val[3:]
        E_mat = self.GetRpyToAngularVelocityMatrix(rpy)
        dw_dp_rot = R_WM @ E_mat
        
        # dv_WG / dp_trans
        dv_dp_trans = R_WM
        
        # G and E are rigidly attached: w_WE = w_WG
        # v_WE = v_WG + w_WG x p_GE_W
        p_GE_W = X_WE.translation() - X_WG.translation()
        
        # skew symmetric matrix for p_GE_W
        px, py, pz = p_GE_W
        skew_p_GE = np.array([
            [  0, -pz,  py],
            [ pz,   0, -px],
            [-py,  px,   0]
        ])
        
        dpose_dp = np.zeros((6, 6))
        # Top 3 rows: w_WE
        dpose_dp[:3, 3:] = dw_dp_rot
        
        # Bottom 3 rows: v_WE = v_WG - p_GE_W x w_WG = v_WG - skew(p_GE_W) @ w_WG
        dpose_dp[3:, :3] = dv_dp_trans
        dpose_dp[3:, 3:] = -skew_p_GE @ dw_dp_rot
        
        return dpose_dp

    def GetRpyToAngularVelocityMatrix(self, rpy):
        # E(r,p,y) such that w = E * rpy_dot
        rpy_obj = RollPitchYaw(rpy)
        E = np.zeros((3, 3))
        E[:, 0] = rpy_obj.CalcAngularVelocityInParentFromRpyDt(np.array([1, 0, 0]))
        E[:, 1] = rpy_obj.CalcAngularVelocityInParentFromRpyDt(np.array([0, 1, 0]))
        E[:, 2] = rpy_obj.CalcAngularVelocityInParentFromRpyDt(np.array([0, 0, 1]))
        return E



    def _joint_centering_cost(self, p_vars):
        """
        ||q(p) - q_target||^2, AutoDiff-safe via GetQ.

        q_target and cost_weights are both resolved by _ResolveCostReference from the
        shared options, so this is the identical objective the Old formulation minimises;
        a caller that retargets one formulation retargets all three.
        """
        q = self.GetQ(p_vars)
        q_err = q - self.q_target
        return self.options.joint_centering_cost_multiplier * q_err.dot(
            self.cost_weights * q_err
        )



