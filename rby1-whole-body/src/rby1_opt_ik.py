"""
Optimization-based IK for the RBY1 using the new parameterized formulation.

Adapted from analytic-and-optimization-ik/src/rby1_experiments.py.
The key idea: decision variables are (base_xyt, torso_joints, right_eef_xyzrpy,
psi_right, left_eef_xyzrpy, psi_left); the analytic IK reconstructs the full
joint configuration on every function/constraint evaluation.
"""

import os
from dataclasses import dataclass, field
from functools import partial
from typing import Optional
from warnings import warn

import numpy as np

from pydrake.all import (
    AutoDiffXd,
    CommonSolverOption,
    ExtractGradient,
    ExtractValue,
    InitializeAutoDiff,
    IpoptSolver,
    LoadModelDirectives,
    MathematicalProgram,
    MeshcatVisualizer,
    MeshcatVisualizerParams,
    MinimumDistanceLowerBoundConstraint,
    NloptSolver,
    Parser,
    ProcessModelDirectives,
    RigidTransform,
    RigidTransform_,
    RobotDiagramBuilder,
    Role,
    RollPitchYaw,
    RollPitchYaw_,
    SnoptSolver,
    SolverOptions,
    eq,
    sqrt,
)

from common import RepoDir
from exps.timing_utils import mark, record
from rby1_analytic_ik import DIST_TO_MID_GRIPPER_FROM_EE, Rby1IK


# Support polygon (CCW) of the RBY1 base in the base frame.
support_polygon_xyzs = np.array([
    [ 0.228000, -0.265000, 0.0],  # right wheel
    [ 0.228000,  0.265000, 0.0],  # left wheel
    [-0.248686,  0.066310, 0.0],  # left caster
    [-0.248686, -0.066310, 0.0],  # right caster
])


def _com_support_polygon_residuals(com_xy, base_xyt):
    """Per-edge residuals for the COM-inside-support-polygon constraint.

    residual_i = -inward_normal_i · (com_xy - vertex_i)
    All residuals ≤ 0  ⟺  COM is inside the support polygon.

    Compatible with both plain numpy arrays and Drake AutoDiffXd — the caller
    is responsible for passing the appropriate scalar types.
    """
    tx, ty, theta = base_xyt[0], base_xyt[1], base_xyt[2]
    c, s = np.cos(theta), np.sin(theta)
    T = np.array([[c, -s, tx], [s, c, ty], [0.0, 0.0, 1.0]])
    n = len(support_polygon_xyzs)
    pts_hom = np.hstack([support_polygon_xyzs[:, :2], np.ones((n, 1))])
    verts = (T @ pts_hom.T).T[:, :2]
    residuals = []
    for i in range(n):
        d = verts[(i + 1) % n] - verts[i]
        inward_normal = np.array([-d[1], d[0]])
        residuals.append(-inward_normal @ (com_xy - verts[i]))
    return np.array(residuals)


# Edge lengths of the support polygon, in residual order (edge i runs from
# vertex i to vertex i+1). Frame-invariant (rigid transforms preserve lengths),
# so these match the world-frame residuals at any base pose.
support_polygon_edge_lengths = np.linalg.norm(
    np.roll(support_polygon_xyzs[:, :2], -1, axis=0) - support_polygon_xyzs[:, :2],
    axis=1,
)

# Nominal inward inset (m) for the conservative CoM stability margin. This is the
# rear-edge clearance of the ready pose when leaning back; the polygon inradius is
# ~0.1766 m, so this leaves a non-empty feasible region.
DEFAULT_SUPPORT_POLYGON_INSET = 0.0823


def _rear_support_edge_index():
    """Index (residual/edge order) of the rear support-polygon edge — the one
    between the two back casters, which the CoM approaches when the robot leans
    backward. Identified as the edge whose inward normal points most in +x
    (i.e. the edge sits at the back, facing forward into the polygon)."""
    v = support_polygon_xyzs[:, :2]
    n = len(v)
    best_i, best_nx = 0, -np.inf
    for i in range(n):
        d = v[(i + 1) % n] - v[i]
        inward_normal = np.array([-d[1], d[0]])          # CCW polygon
        nx = inward_normal[0] / np.linalg.norm(inward_normal)
        if nx > best_nx:
            best_nx, best_i = nx, i
    return best_i


# Only the rear edge is inset for the conservative CoM margin: it is the binding
# constraint when the robot pitches/leans backward (the 0.0823 m margin seen in
# visualised during development). The front/side edges keep the nominal bound.
REAR_SUPPORT_EDGE_INDEX = _rear_support_edge_index()


def _stability_constraint_ub(inset):
    """Per-edge upper bound for the CoM residual constraint, insetting only the
    rear support edge inward by ``inset`` metres.

    Because ``residual_i = -(edge_length_i) * (orthogonal CoM-to-edge distance)``,
    requiring the CoM to keep a margin of ``inset`` metres from the rear edge is
    exactly ``residual_rear <= -inset * edge_length_rear``; all other edges keep
    the nominal ``residual_i <= 0``. ``inset=0`` reproduces the nominal bound on
    every edge. This is the algebraic equivalent of moving only the rear edge
    inward — no vertex recomputation, and it cannot produce a degenerate polygon.
    """
    ub = np.zeros(len(support_polygon_xyzs))
    i = REAR_SUPPORT_EDGE_INDEX
    ub[i] = -float(inset) * support_polygon_edge_lengths[i]
    return ub


def inset_support_polygon_xy(inset, edges=(REAR_SUPPORT_EDGE_INDEX,)):
    """Support-polygon vertices (base frame, xy) with selected ``edges`` moved
    inward by ``inset`` metres along their inward normals.

    Defaults to the rear edge only, matching ``_stability_constraint_ub``. Each
    vertex becomes the intersection of its two (possibly offset) adjacent edge
    lines, so insetting one edge only moves that edge's two endpoints. For
    visualization / geometry only.
    """
    v = support_polygon_xyzs[:, :2]
    n = len(v)
    edges = set(edges)
    normals = np.empty((n, 2))
    offsets = np.empty(n)
    for i in range(n):
        d = v[(i + 1) % n] - v[i]
        nrm = np.array([-d[1], d[0]])           # inward normal (CCW polygon)
        nrm /= np.linalg.norm(nrm)
        normals[i] = nrm
        offsets[i] = nrm @ v[i] + (inset if i in edges else 0.0)  # offset edge line
    out = np.empty((n, 2))
    for j in range(n):
        i0 = (j - 1) % n                        # the two edges meeting at vertex j
        out[j] = np.linalg.solve(
            np.array([normals[i0], normals[j]]),
            np.array([offsets[i0], offsets[j]]),
        )
    return out


_COM_INSTANCE_CACHE: dict = {}


def _com_instances(plant):
    """The 7 model instances CoM stability sums over, cached per plant.

    ``GetModelInstanceByName`` is a string lookup and this function sits inside
    the RRT validity check (~20 calls per extend), where the seven lookups were
    a measurable share of per-sample cost. Keyed by id(plant): plants live for
    the whole worker process, and a dead plant's stale entry can never collide
    because its id would have to be reused by another MultibodyPlant that also
    gets queried here -- at which point the names resolve identically anyway.
    """
    key = id(plant)
    got = _COM_INSTANCE_CACHE.get(key)
    if got is None:
        got = tuple(plant.GetModelInstanceByName(n) for n in (
            "base", "torso", "right_arm", "left_arm",
            "right_gripper", "left_gripper", "head"))
        _COM_INSTANCE_CACHE[key] = got
    return got


def check_com_stability(q_23, plant, plant_context, inset=0.0):
    """Return True if the robot COM lies inside the base support polygon.

    q_23: 23-dim config [base(3), torso(6), right_arm(7), left_arm(7)].
    inset: inward safety margin (m) applied to the rear support edge only (the
        back-caster edge). The CoM must keep at least ``inset`` clearance from
        that edge; ``0.0`` (default) is the nominal support polygon.
    """
    (base_inst, torso_inst, right_inst, left_inst,
     rg_inst, lg_inst, head_inst) = _com_instances(plant)

    plant.SetPositions(plant_context, base_inst,  q_23[:3])
    plant.SetPositions(plant_context, torso_inst, q_23[3:9])
    plant.SetPositions(plant_context, right_inst, q_23[9:16])
    plant.SetPositions(plant_context, left_inst,  q_23[16:23])

    com = plant.CalcCenterOfMassPositionInWorld(
        plant_context,
        [base_inst, torso_inst, right_inst, left_inst, rg_inst, lg_inst, head_inst],
    )

    residuals = _com_support_polygon_residuals(com[:2], q_23[:3])
    return bool(np.all(residuals <= _stability_constraint_ub(inset)))


@dataclass
class Rby1ProblemOptionsNew:
    # The manipulability log-barrier (cost and constraint forms) lived here and
    # is gone: it duplicated the direct-reachability constraint's job, its
    # finite-difference gradient was computed from unsliced Jacobians while its
    # value used arm-sliced ones (so IPOPT was handed a gradient of a different
    # function), and evaluating it cost a Jacobian per decision variable per
    # call. Reachability is enforced by impose_direct_reachability_constraint in
    # IK and by the FK-residual constraint in _lift_trajopt.
    impose_direct_reachability_constraint: bool = True
    direct_reachability_threshold: float = 1e-2
    ik_stepback: float = 0.0
    impose_flex_constraints: bool = False
    impose_stability_constraints: bool = True
    support_polygon_inset: float = DEFAULT_SUPPORT_POLYGON_INSET


@dataclass
class Rby1ProblemOptions:
    no_collisions: bool = True
    minimum_distance: float = 0.0
    influence_distance_offset: float = 0.01
    impose_joint_centering_cost: bool = True
    joint_nominal: bool = True
    sqrt_centering_cost: bool = False
    joint_centering_cost_multiplier: float = 1.0
    # Explicit 23-D centering target [base(3), torso(6), right(7), left(7)].
    # When set (and impose_joint_centering_cost is on), the centering cost pulls
    # toward this instead of q_center. Needed because q_nominal is overwritten by
    # q_initial, which callers seed with a *random* draw -- reusing it as the
    # posture target would make the target random too. Rows outside the 23
    # active dofs (head, grippers) carry no cost either way.
    posture_nominal: Optional[np.ndarray] = None
    # Sign-agnostic left/right arm symmetry cost: mult * sum_j (|qR_j| - |qL_j|)^2.
    # |.| makes it invariant to the arms' mirrored sign conventions. 0 disables.
    arm_symmetry_cost_multiplier: float = 0.0

    q_initial: Optional[np.ndarray] = None
    left_target: RigidTransform = field(default_factory=RigidTransform)
    right_target: RigidTransform = field(default_factory=RigidTransform)
    right_pose_constraint: bool = True
    left_pose_constraint: bool = True
    right_gcp: Optional[np.ndarray] = None
    left_gcp: Optional[np.ndarray] = None

    # When set to +1 or -1, constrains the sign of arm joint 5 (0-indexed,
    # the wrist-singularity joint used for the acos ±θ IKFast branch) to stay
    # on the same side of zero.  Set this after the first successful IK solve
    # in a multi-candidate search so subsequent solves do not need to explore
    # the infeasible half-plane.  When impose_flex_constraints is True (the
    # default in _default_ik_options), these tighten the flex-constraint bounds
    # at zero extra cost.  Otherwise, they install a standalone sign constraint.
    joint5_sign_right: Optional[int] = None
    joint5_sign_left: Optional[int] = None

    solver: str = "IPOPT"  # "IPOPT", "SNOPT", or "NLOPT"
    print_file_name: str = "/tmp/optimizer_output.log"
    acceptable_tol: float = 1e-3
    acceptable_dual_inf_tol: float = 1e-3
    acceptable_compl_inf_tol: float = 1e-3
    acceptable_constr_viol_tol: float = 1e-3
    acceptable_iter: int = 1
    max_wall_time: float = 10.0
    print_level: int = 5

    pin_base: bool = False
    pin_torso: bool = False
    # boundary_fallback on by default so the direct-reachability constraint
    # (default-on) gets smooth, damped gradients on out-of-reach requests; the
    # constraint itself still rejects truly-unreachable targets at its threshold.
    boundary_fallback: bool = True
    boundary_damping_lambda: float = 0.1

    new_formulation_options: Rby1ProblemOptionsNew = field(
        default_factory=Rby1ProblemOptionsNew
    )


def _visualization_callback(
    vars, diagram, diagram_context, plant, plant_context, vars_to_q, joint_file
):
    q = vars_to_q(vars)
    if joint_file is not None:
        with open(joint_file, "a") as f:
            f.write(" ".join(map(str, vars)) + "\n")
    if not np.all(np.isfinite(q)):
        warn("nans detected in position.")
        return
    plant.SetPositions(plant_context, q)
    diagram.ForcedPublish(diagram_context)


class Rby1IKProblem:
    def __init__(self, diagram):
        self.diagram = diagram
        self.plant = diagram.GetSubsystemByName("plant")
        self.autodiff_plant = self.plant.ToAutoDiffXd()

        self.diagram_context = self.diagram.CreateDefaultContext()
        self.plant_context = self.plant.GetMyContextFromRoot(self.diagram_context)
        self.autodiff_context = self.autodiff_plant.CreateDefaultContext()

        self.left_gripper = self.plant.GetFrameByName("ee_left")
        self.left_gripper_ad = self.autodiff_plant.GetFrameByName("ee_left")
        self.right_gripper = self.plant.GetFrameByName("ee_right")
        self.right_gripper_ad = self.autodiff_plant.GetFrameByName("ee_right")

    def Solve(self, visualize=False, joint_file=None):
        if os.path.exists(self.options.print_file_name):
            with open(self.options.print_file_name, "r+") as f:
                f.seek(0)
                f.truncate()

        solver_options = SolverOptions()
        solver_options.SetOption(
            CommonSolverOption.kPrintFileName, self.options.print_file_name
        )

        if self.options.solver == "IPOPT":
            solver = IpoptSolver()
            solver_options.SetOption(IpoptSolver().solver_id(), "file_print_level", self.options.print_level)
            solver_options.SetOption(IpoptSolver().solver_id(), "print_user_options", "yes")
            solver_options.SetOption(IpoptSolver().solver_id(), "acceptable_tol", self.options.acceptable_tol)
            solver_options.SetOption(IpoptSolver().solver_id(), "acceptable_dual_inf_tol", self.options.acceptable_dual_inf_tol)
            solver_options.SetOption(IpoptSolver().solver_id(), "acceptable_compl_inf_tol", self.options.acceptable_compl_inf_tol)
            solver_options.SetOption(IpoptSolver().solver_id(), "acceptable_constr_viol_tol", self.options.acceptable_constr_viol_tol)
            solver_options.SetOption(IpoptSolver().solver_id(), "acceptable_iter", self.options.acceptable_iter)
            solver_options.SetOption(IpoptSolver().solver_id(), "max_wall_time", self.options.max_wall_time)

        elif self.options.solver == "SNOPT":
            solver = SnoptSolver()
            solver_options.SetOption(SnoptSolver.id(), "Major print level", self.options.print_level)
            solver_options.SetOption(SnoptSolver.id(), "Timing Level", 3)
            solver_options.SetOption(SnoptSolver.id(), "Time Limit", self.options.max_wall_time)
            solver_options.SetOption(SnoptSolver.id(), "Major optimality tolerance", self.options.acceptable_tol)
            solver_options.SetOption(SnoptSolver.id(), "Minor optimality tolerance", self.options.acceptable_tol)
            solver_options.SetOption(SnoptSolver.id(), "Major feasibility tolerance", self.options.acceptable_constr_viol_tol)
            solver_options.SetOption(SnoptSolver.id(), "Verify level", -1)

        elif self.options.solver == "NLOPT":
            solver = NloptSolver()
            solver_options.SetOption(NloptSolver.id(), "xtol_rel", self.options.acceptable_tol)
            solver_options.SetOption(NloptSolver.id(), NloptSolver.AlgorithmName(), "LD_AUGLAG_EQ")
            solver_options.SetOption(NloptSolver.id(), "constraint_tol", self.options.acceptable_constr_viol_tol)
            solver_options.SetOption(NloptSolver.id(), "max_time", self.options.max_wall_time)

        else:
            raise ValueError(f"Unknown solver: {self.options.solver!r}")

        return self._solve_internal(visualize, solver, solver_options, joint_file)


class Rby1IKProblemNewFormulation(Rby1IKProblem):
    def __init__(self, diagram):
        super().__init__(diagram)

        self.base_instance = self.plant.GetModelInstanceByName("base")
        self.torso_instance = self.plant.GetModelInstanceByName("torso")
        self.right_arm_instance = self.plant.GetModelInstanceByName("right_arm")
        self.left_arm_instance = self.plant.GetModelInstanceByName("left_arm")
        self.head_instance = self.plant.GetModelInstanceByName("head")
        self.left_gripper_instance = self.plant.GetModelInstanceByName("left_gripper")
        self.right_gripper_instance = self.plant.GetModelInstanceByName("right_gripper")

        self.base_instance_ad = self.autodiff_plant.GetModelInstanceByName("base")
        self.torso_instance_ad = self.autodiff_plant.GetModelInstanceByName("torso")
        self.right_arm_instance_ad = self.autodiff_plant.GetModelInstanceByName("right_arm")
        self.left_arm_instance_ad = self.autodiff_plant.GetModelInstanceByName("left_arm")
        self.head_instance_ad = self.autodiff_plant.GetModelInstanceByName("head")
        self.left_gripper_instance_ad = self.autodiff_plant.GetModelInstanceByName("left_gripper")
        self.right_gripper_instance_ad = self.autodiff_plant.GetModelInstanceByName("right_gripper")

        self.base_gen_pos_idxs = self._get_gen_pos_idxs(self.base_instance)
        self.torso_gen_pos_idxs = self._get_gen_pos_idxs(self.torso_instance)
        self.right_arm_gen_pos_idxs = self._get_gen_pos_idxs(self.right_arm_instance)
        self.left_arm_gen_pos_idxs = self._get_gen_pos_idxs(self.left_arm_instance)

    def _get_gen_pos_idxs(self, model_instance):
        pos_idxs = []
        for jidx in self.plant.GetJointIndices(model_instance):
            j = self.plant.get_joint(jidx)
            if j.num_positions() > 0:
                pos_idxs += list(range(j.position_start(), j.position_start() + j.num_positions()))
        return pos_idxs

    def ApplyOptions(self, options: Rby1ProblemOptions):
        self.options = options
        self.analytic_ik = Rby1IK(boundary_fallback=options.boundary_fallback)

        self.prog = MathematicalProgram()
        self.xyt_base = self.prog.NewContinuousVariables(3, "base")
        self.joints_torso = self.prog.NewContinuousVariables(6, "torso")
        self.xyzrpy_right = self.prog.NewContinuousVariables(6, "right_gripper")
        self.psi_right = self.prog.NewContinuousVariables(1, "psi_right")
        self.xyzrpy_left = self.prog.NewContinuousVariables(6, "left_gripper")
        self.psi_left = self.prog.NewContinuousVariables(1, "psi_left")
        self.lumped_vars = np.hstack((
            self.xyt_base, self.joints_torso,
            self.xyzrpy_right, self.psi_right,
            self.xyzrpy_left, self.psi_left,
        ))

        if self.options.q_initial is not None:
            q_init = self.options.q_initial  # 23-dim [base(3), torso(6), right(7), left(7)]
            q_full = self.plant.GetDefaultPositions()
            q_full[self.base_gen_pos_idxs]      = q_init[:3]
            q_full[self.torso_gen_pos_idxs]     = q_init[3:9]
            q_full[self.right_arm_gen_pos_idxs] = q_init[9:16]
            q_full[self.left_arm_gen_pos_idxs]  = q_init[16:23]
            self.plant.SetDefaultPositions(q_full)
        self.q_nominal = self.plant.GetDefaultPositions()

        lower_limits = self.plant.GetPositionLowerLimits()
        upper_limits = self.plant.GetPositionUpperLimits()
        self.q_center = np.where(
            np.isinf(lower_limits) | np.isinf(upper_limits),
            self.q_nominal,
            (lower_limits + upper_limits) / 2.0
        )

        self.prog.SetInitialGuess(self.psi_right, [self.q_nominal[self.right_arm_gen_pos_idxs[2]]])
        self.prog.SetInitialGuess(self.psi_left,  [self.q_nominal[self.left_arm_gen_pos_idxs[2]]])

        if self.options.right_gcp is not None and self.options.left_gcp is not None:
            self.GC_right = self.options.right_gcp
            self.GC_left = self.options.left_gcp
        else:
            q_23_nominal = np.concatenate([
                self.q_nominal[self.base_gen_pos_idxs],
                self.q_nominal[self.torso_gen_pos_idxs],
                self.q_nominal[self.right_arm_gen_pos_idxs],
                self.q_nominal[self.left_arm_gen_pos_idxs],
            ])
            self.GC_right, self.GC_left = self.analytic_ik.gcp(q_23_nominal)

        self.plant.SetPositions(self.plant_context, self.q_nominal)
        left_guess = self.left_gripper.CalcPoseInWorld(self.plant_context)
        right_guess = self.right_gripper.CalcPoseInWorld(self.plant_context)
        self.prog.SetInitialGuess(self.xyzrpy_left, np.hstack((
            left_guess.translation(), RollPitchYaw(left_guess.rotation()).vector()
        )))
        self.prog.SetInitialGuess(self.xyzrpy_right, np.hstack((
            right_guess.translation(), RollPitchYaw(right_guess.rotation()).vector()
        )))
        self.prog.SetInitialGuess(self.xyt_base, self.q_nominal[:3])
        self.prog.SetInitialGuess(self.joints_torso, self.q_nominal[self.torso_gen_pos_idxs])

        all_lb = self.plant.GetPositionLowerLimits()
        all_ub = self.plant.GetPositionUpperLimits()
        if self.options.pin_base:
            base_nom = self.q_nominal[:3]
            self.prog.AddBoundingBoxConstraint(base_nom, base_nom, self.xyt_base)
        if self.options.pin_torso:
            torso_nom = self.q_nominal[self.torso_gen_pos_idxs]
            self.prog.AddBoundingBoxConstraint(torso_nom, torso_nom, self.joints_torso)
        else:
            self.prog.AddBoundingBoxConstraint(
                all_lb[self.torso_gen_pos_idxs], all_ub[self.torso_gen_pos_idxs], self.joints_torso
            )
        self.prog.AddBoundingBoxConstraint(
            all_lb[self.right_arm_gen_pos_idxs[2]], all_ub[self.right_arm_gen_pos_idxs[2]], self.psi_right
        )
        self.prog.AddBoundingBoxConstraint(
            all_lb[self.left_arm_gen_pos_idxs[2]], all_ub[self.left_arm_gen_pos_idxs[2]], self.psi_left
        )

        nfo = self.options.new_formulation_options
        if nfo.impose_direct_reachability_constraint:
            self._add_direct_reachability_constraint()
        if self.options.no_collisions:
            self._add_collision_free_constraint()
        if self.options.left_pose_constraint:
            self._add_left_gripper_constraint()
        if self.options.right_pose_constraint:
            self._add_right_gripper_constraint()
        if self.options.impose_joint_centering_cost:
            self._add_joint_centering_cost()
        if self.options.arm_symmetry_cost_multiplier > 0.0:
            self._add_arm_symmetry_cost()
        if self.options.new_formulation_options.impose_flex_constraints:
            self._add_flex_constraints()
        elif (self.options.joint5_sign_right is not None
              or self.options.joint5_sign_left is not None):
            # flex constraints not requested, but a sign constraint is — add it
            # as a standalone (cheaper than the full flex constraint).
            self._add_joint5_sign_only_constraints()
        if self.options.new_formulation_options.impose_stability_constraints:
            self._add_stability_constraints()

    def VarsToQ(self, lumped_vars):
        q_23 = self.analytic_ik.compute_ik(
            lumped_vars, right_gcp=self.GC_right, left_gcp=self.GC_left, boundary_tol=1e-2,
            boundary_damping_lambda=self.options.boundary_damping_lambda,
        )
        if isinstance(lumped_vars[0], AutoDiffXd):
            # Uniform-length (n_d) zero-gradient padding for every dof, matching
            # q_23's own derivative size -- self.autodiff_plant.GetDefaultPositions()
            # instead returns zero-SIZE-derivative AutoDiffXd, so overwriting the
            # active slots below with q_23's real (n_d-length) gradients produced a
            # ragged array (mismatched derivative-vector lengths per element). That
            # ragged array breaks downstream Eval() calls needing a uniform Jacobian
            # width (e.g. the collision-free constraint), surfacing as a
            # PyFunctionConstraint AutoDiffXd/float TypeError.
            q_default_val = self.plant.GetDefaultPositions()
            if q_23 is None:
                n_d = lumped_vars[0].derivatives().size
                return np.array([AutoDiffXd(np.nan, np.zeros(n_d)) for _ in q_default_val])
            n_d = q_23[0].derivatives().size
            q_full = np.array([AutoDiffXd(v, np.zeros(n_d)) for v in q_default_val])
        else:
            q_full = self.plant.GetDefaultPositions()
            if q_23 is None:
                q_full[:] = np.nan
                return q_full
        q_full[self.base_gen_pos_idxs] = q_23[:3]
        q_full[self.torso_gen_pos_idxs] = q_23[3:9]
        q_full[self.right_arm_gen_pos_idxs] = q_23[9:16]
        q_full[self.left_arm_gen_pos_idxs] = q_23[16:23]
        return q_full

    def _add_left_gripper_constraint(self):
        target = self.options.left_target
        goal = np.hstack((target.translation(), target.rotation().ToRollPitchYaw().vector()))
        c = self.prog.AddConstraint(eq(self.xyzrpy_left, goal))
        c.evaluator().set_description("Left Gripper Pose Constraint")

    def _add_right_gripper_constraint(self):
        target = self.options.right_target
        goal = np.hstack((target.translation(), target.rotation().ToRollPitchYaw().vector()))
        c = self.prog.AddConstraint(eq(self.xyzrpy_right, goal))
        c.evaluator().set_description("Right Gripper Pose Constraint")

    def _add_collision_free_constraint(self):
        self._collision_free_constraint = MinimumDistanceLowerBoundConstraint(
            plant=self.plant,
            bound=self.options.minimum_distance,
            influence_distance_offset=self.options.influence_distance_offset,
            plant_context=self.plant_context,
        )
        def _collision_eval(v):
            q = self.VarsToQ(v)
            is_ad = isinstance(q[0], AutoDiffXd)
            q_val = ExtractValue(q) if is_ad else q
            if not np.all(np.isfinite(q_val)):
                # Must match q's type: a bare float here breaks the
                # PyFunctionConstraint's AutoDiffXd/float invariant whenever
                # the IK for one arm fails (q contains NaN) mid-solve.
                if is_ad:
                    n_d = q[0].derivatives().size
                    return np.array([AutoDiffXd(np.inf, np.zeros(n_d))])
                return np.array([np.inf])
            return self._collision_free_constraint.Eval(q)
        c = self.prog.AddConstraint(
            _collision_eval,
            lb=[-np.inf], ub=[1.0],
            vars=self.lumped_vars,
        )
        c.evaluator().set_description("Collision-Free Constraint")

    def _eval_joint_centering_cost(self, lumped_vars):
        q = self.VarsToQ(lumped_vars)
        q_vals = [qi.value() for qi in q] if isinstance(q[0], AutoDiffXd) else q
        if not np.all(np.isfinite(q_vals)):
            return AutoDiffXd(np.inf) if isinstance(q[0], AutoDiffXd) else np.inf
        if self._centering_target is not None:
            q = q - self._centering_target
        elif self.options.joint_nominal:
            q = q - self.q_center
        cost = q.dot(self._centering_cost_mat @ q)
        return sqrt(cost) if self.options.sqrt_centering_cost else cost

    def _add_joint_centering_cost(self):
        n = self.plant.num_positions()
        self._centering_cost_mat = self.options.joint_centering_cost_multiplier * np.eye(n)
        self._centering_cost_mat[:4, :4] = 0
        self._centering_target = None
        if self.options.posture_nominal is not None:
            # Explicit 23-D target mapped into full-plant coordinates; dofs the
            # lumped IK never touches (head, grippers) get zero cost so the
            # target's value there is irrelevant.
            active = (list(self.base_gen_pos_idxs) + list(self.torso_gen_pos_idxs)
                      + list(self.right_arm_gen_pos_idxs) + list(self.left_arm_gen_pos_idxs))
            inactive = np.setdiff1d(np.arange(n), active)
            self._centering_cost_mat[inactive, inactive] = 0
            target = np.zeros(n)
            pn = np.asarray(self.options.posture_nominal, dtype=float)
            target[self.base_gen_pos_idxs] = pn[:3]
            target[self.torso_gen_pos_idxs] = pn[3:9]
            target[self.right_arm_gen_pos_idxs] = pn[9:16]
            target[self.left_arm_gen_pos_idxs] = pn[16:23]
            self._centering_target = target
        c = self.prog.AddCost(self._eval_joint_centering_cost, vars=self.lumped_vars)
        c.evaluator().set_description("Joint centering cost")

    def _eval_arm_symmetry_cost(self, lumped_vars):
        q = self.VarsToQ(lumped_vars)
        q_vals = [qi.value() for qi in q] if isinstance(q[0], AutoDiffXd) else q
        if not np.all(np.isfinite(q_vals)):
            return AutoDiffXd(np.inf) if isinstance(q[0], AutoDiffXd) else np.inf
        d = np.abs(q[self.right_arm_gen_pos_idxs]) - np.abs(q[self.left_arm_gen_pos_idxs])
        return self.options.arm_symmetry_cost_multiplier * d.dot(d)

    def _add_arm_symmetry_cost(self):
        c = self.prog.AddCost(self._eval_arm_symmetry_cost, vars=self.lumped_vars)
        c.evaluator().set_description("Arm symmetry cost")

    def _eval_com_in_support_polygon(self, q_val):
        autodiffing = isinstance(q_val[0], AutoDiffXd)

        base_pos = q_val[self.base_gen_pos_idxs]
        torso_pos = q_val[self.torso_gen_pos_idxs]
        right_pos = q_val[self.right_arm_gen_pos_idxs]
        left_pos = q_val[self.left_arm_gen_pos_idxs]

        if autodiffing:
            if not np.all(np.isfinite(ExtractValue(q_val))):
                return InitializeAutoDiff(np.full(len(support_polygon_xyzs), np.inf))
            self.autodiff_plant.SetPositions(self.autodiff_context, self.base_instance_ad, base_pos)
            self.autodiff_plant.SetPositions(self.autodiff_context, self.torso_instance_ad, torso_pos)
            self.autodiff_plant.SetPositions(self.autodiff_context, self.right_arm_instance_ad, right_pos)
            self.autodiff_plant.SetPositions(self.autodiff_context, self.left_arm_instance_ad, left_pos)
            com = self.autodiff_plant.CalcCenterOfMassPositionInWorld(
                self.autodiff_context,
                [self.base_instance_ad, self.torso_instance_ad,
                 self.right_arm_instance_ad, self.left_arm_instance_ad,
                 self.right_gripper_instance_ad, self.left_gripper_instance_ad, self.head_instance_ad],
            )
        else:
            if not np.all(np.isfinite(q_val)):
                return np.full(len(support_polygon_xyzs), np.inf)
            self.plant.SetPositions(self.plant_context, self.base_instance, base_pos)
            self.plant.SetPositions(self.plant_context, self.torso_instance, torso_pos)
            self.plant.SetPositions(self.plant_context, self.right_arm_instance, right_pos)
            self.plant.SetPositions(self.plant_context, self.left_arm_instance, left_pos)
            com = self.plant.CalcCenterOfMassPositionInWorld(
                self.plant_context,
                [self.base_instance, self.torso_instance,
                 self.right_arm_instance, self.left_arm_instance,
                 self.right_gripper_instance, self.left_gripper_instance, self.head_instance],
            )

        return _com_support_polygon_residuals(com[:2], q_val[:3])

    def _add_stability_constraints(self):
        inset = self.options.new_formulation_options.support_polygon_inset
        c = self.prog.AddConstraint(
            lambda v: self._eval_com_in_support_polygon(self.VarsToQ(v)),
            lb=np.full(len(support_polygon_xyzs), -np.inf),
            ub=_stability_constraint_ub(inset),
            vars=self.lumped_vars,
        )
        c.evaluator().set_description(
            f"COM inside support polygon (inset {inset:.4f} m)"
        )

    def _add_flex_constraints(self):
        """Arm joint-limit flex constraints, with optional joint-5 sign tightening.

        Evaluates VarsToQ once, then checks all 14 arm joints against their
        plant limits.  If joint5_sign_right / joint5_sign_left are set (+1 or
        -1), the corresponding bound for joint 5 (0-indexed) is further
        tightened so IPOPT cannot explore the opposite (singular) half-plane
        — at zero additional cost because VarsToQ is only called once.
        """
        all_lb = self.plant.GetPositionLowerLimits()
        all_ub = self.plant.GetPositionUpperLimits()
        lb = np.concatenate([all_lb[self.right_arm_gen_pos_idxs], all_lb[self.left_arm_gen_pos_idxs]])
        ub = np.concatenate([all_ub[self.right_arm_gen_pos_idxs], all_ub[self.left_arm_gen_pos_idxs]])

        # Tighten joint-5 bounds based on the sign constraint: if sign is +1,
        # lb[5] becomes max(lb[5], 0); if -1, ub[5] becomes min(ub[5], 0).
        # Joint 5 is index 5 within each 7-DOF arm block (0-indexed).
        if self.options.joint5_sign_right is not None:
            s = int(self.options.joint5_sign_right)
            if s > 0:
                lb[5] = max(lb[5], 0.0)
            else:
                ub[5] = min(ub[5], 0.0)
        if self.options.joint5_sign_left is not None:
            s = int(self.options.joint5_sign_left)
            if s > 0:
                lb[7 + 5] = max(lb[7 + 5], 0.0)  # left arm block starts at index 7
            else:
                ub[7 + 5] = min(ub[7 + 5], 0.0)

        def _flex_eval(v):
            q = self.VarsToQ(v)
            return np.concatenate([q[self.right_arm_gen_pos_idxs], q[self.left_arm_gen_pos_idxs]])

        c = self.prog.AddConstraint(
            _flex_eval,
            lb=lb, ub=ub, vars=self.lumped_vars,
        )
        c.evaluator().set_description("Arm joint limits")

    def _add_joint5_sign_only_constraints(self):
        """Standalone sign constraint for joint 5 when flex constraints are off.

        Evaluates VarsToQ once and returns the 2-element vector
        [sign_r * q_right[5], sign_l * q_left[5]] constrained to be >= 0.
        Only adds the constraint for whichever arm(s) have a sign specified.
        """
        sign_r = self.options.joint5_sign_right
        sign_l = self.options.joint5_sign_left
        r_idx = self.right_arm_gen_pos_idxs[5]
        l_idx = self.left_arm_gen_pos_idxs[5]

        if sign_r is not None and sign_l is not None:
            _sr, _sl, _ri, _li = int(sign_r), int(sign_l), r_idx, l_idx
            def _both(v):
                q = self.VarsToQ(v)
                q_val = ExtractValue(q) if isinstance(q[0], AutoDiffXd) else q
                if not np.all(np.isfinite(q_val)):
                    return np.array([np.inf, np.inf])
                return np.array([_sr * q[_ri], _sl * q[_li]])
            c = self.prog.AddConstraint(_both, lb=[0.0, 0.0], ub=[np.inf, np.inf], vars=self.lumped_vars)
            c.evaluator().set_description("Joint-5 sign constraints (both arms)")
        elif sign_r is not None:
            _sr, _ri = int(sign_r), r_idx
            def _right(v):
                q = self.VarsToQ(v)
                q_val = ExtractValue(q) if isinstance(q[0], AutoDiffXd) else q
                if not np.all(np.isfinite(q_val)):
                    return np.array([np.inf])
                return np.array([_sr * q[_ri]])
            c = self.prog.AddConstraint(_right, lb=[0.0], ub=[np.inf], vars=self.lumped_vars)
            c.evaluator().set_description(f"Right arm joint-5 sign constraint (sign={sign_r:+d})")
        elif sign_l is not None:
            _sl, _li = int(sign_l), l_idx
            def _left(v):
                q = self.VarsToQ(v)
                q_val = ExtractValue(q) if isinstance(q[0], AutoDiffXd) else q
                if not np.all(np.isfinite(q_val)):
                    return np.array([np.inf])
                return np.array([_sl * q[_li]])
            c = self.prog.AddConstraint(_left, lb=[0.0], ub=[np.inf], vars=self.lumped_vars)
            c.evaluator().set_description(f"Left arm joint-5 sign constraint (sign={sign_l:+d})")


    def _eval_direct_reachability_constraint(self, lumped_vars):
        """Per-arm FK-residual reachability: 100·‖FK(IK(p)) − X_des(p)‖²_F for each
        gripper, where X_des is built from the decision-variable EEF poses and FK is
        taken on the analytic-IK reconstruction (``VarsToQ``). When the analytic IK
        must project a request onto its reachable boundary, FK(q) departs from the
        request and the residual grows — so bounding it ≤ threshold rejects poses
        that are not genuinely reachable. Returns ``[res_right, res_left]``."""
        autodiffing = isinstance(lumped_vars[0], AutoDiffXd)
        n_grads = lumped_vars[0].derivatives().size if autodiffing else 0

        q = self.VarsToQ(lumped_vars)
        q_val = ExtractValue(q) if autodiffing else q
        if not np.all(np.isfinite(q_val)):
            if autodiffing:
                return np.array([AutoDiffXd(np.inf, np.zeros(n_grads))] * 2)
            return np.full(2, np.inf)

        if autodiffing:
            self.autodiff_plant.SetPositions(self.autodiff_context, q)
            X_r_fk = self.right_gripper_ad.CalcPoseInWorld(self.autodiff_context).GetAsMatrix4()
            X_l_fk = self.left_gripper_ad.CalcPoseInWorld(self.autodiff_context).GetAsMatrix4()
            X_r_des = RigidTransform_[AutoDiffXd](
                RollPitchYaw_[AutoDiffXd](lumped_vars[12:15]), lumped_vars[9:12]
            ).GetAsMatrix4()
            X_l_des = RigidTransform_[AutoDiffXd](
                RollPitchYaw_[AutoDiffXd](lumped_vars[19:22]), lumped_vars[16:19]
            ).GetAsMatrix4()
        else:
            self.plant.SetPositions(self.plant_context, q)
            X_r_fk = self.right_gripper.CalcPoseInWorld(self.plant_context).GetAsMatrix4()
            X_l_fk = self.left_gripper.CalcPoseInWorld(self.plant_context).GetAsMatrix4()
            X_r_des = RigidTransform(
                RollPitchYaw(lumped_vars[12:15]), lumped_vars[9:12]
            ).GetAsMatrix4()
            X_l_des = RigidTransform(
                RollPitchYaw(lumped_vars[19:22]), lumped_vars[16:19]
            ).GetAsMatrix4()

        d_r = X_r_fk - X_r_des
        d_l = X_l_fk - X_l_des
        return np.array([100.0 * np.sum(d_r * d_r), 100.0 * np.sum(d_l * d_l)])

    def _add_direct_reachability_constraint(self):
        threshold = self.options.new_formulation_options.direct_reachability_threshold
        c = self.prog.AddConstraint(
            self._eval_direct_reachability_constraint,
            lb=[-np.inf, -np.inf],
            ub=[threshold, threshold],
            vars=self.lumped_vars,
        )
        c.evaluator().set_description("Direct reachability constraint")

    def _solve_internal(self, visualize, solver, solver_options, joint_file):
        if visualize:
            self.prog.AddVisualizationCallback(
                partial(
                    _visualization_callback,
                    diagram=self.diagram,
                    diagram_context=self.diagram_context,
                    plant=self.plant,
                    plant_context=self.plant_context,
                    vars_to_q=self.VarsToQ,
                    joint_file=joint_file,
                ),
                self.lumped_vars,
            )
        self.plant.SetPositions(self.plant_context, self.q_nominal)
        self.diagram.ForcedPublish(self.diagram_context)
        # The solver call alone -- the only interval that may be quoted as
        # "optimizer time" for the IK. Everything the program needed to exist
        # (the AutoDiff plant clone, the constraint registration) is timed
        # separately as ik.build in solve_ik.
        with record("ik.solve", self.options.solver):
            result = solver.Solve(self.prog, solver_options=solver_options)
            mark(success=bool(result.is_success()),
                 n_vars=self.prog.num_vars())
        return result


def MakeRby1Diagram(meshcat=None):
    """Build a RobotDiagram with the full RBY1 model (base, torso, arms, head, grippers).

    If meshcat is provided, attaches a MeshcatVisualizer so that ForcedPublish calls
    (e.g. from the IK visualization callback) update the browser view.
    """
    with record("infra.diagram", "ik"):
        return _make_rby1_diagram(meshcat)


def _make_rby1_diagram(meshcat=None):
    builder = RobotDiagramBuilder(time_step=0.0)
    plant = builder.plant()
    parser = Parser(plant)
    parser.package_map().AddPackageXml(os.path.join(RepoDir(), "package.xml"))
    ProcessModelDirectives(
        LoadModelDirectives(os.path.join(
            RepoDir(),
            "models/ruby/rby1_description_drake/add_rby1_sim_with_holonomic_base_actuators.dmd.yaml",
        )),
        parser,
    )
    if meshcat is not None:
        params = MeshcatVisualizerParams()
        params.delete_on_initialization_event = False
        params.role = Role.kIllustration
        params.prefix = "ik_visual"
        MeshcatVisualizer.AddToBuilder(
            builder.builder(), builder.scene_graph(), meshcat, params
        )
    plant.Finalize()
    return builder.Build()


def solve_ik(
    right_target: RigidTransform,
    left_target: RigidTransform,
    q_initial: np.ndarray = None,
    diagram=None,
    options: Rby1ProblemOptions = None,
    visualize: bool = False,
) -> np.ndarray | None:
    """Solve IK for the RBY1 given target poses for both grippers.

    Args:
        right_target: Desired world-frame pose of the right end-effector.
        left_target:  Desired world-frame pose of the left end-effector.
        q_initial:    Optional 23-dim [base(3), torso(6), right(7), left(7)] initial
                      guess for the joint configuration. Uses the plant default if None.
        diagram:      Optional pre-built RobotDiagram. A new one is built if None.
                      Pass a diagram built with MakeRby1Diagram(meshcat) to enable
                      live meshcat visualization during the solve.
        options:      Full solver/problem options. Defaults are used if None.
        visualize:    If True, register a visualization callback that publishes the
                      robot pose to meshcat at each IPOPT iteration.

    Returns:
        23-dim joint configuration [base(3), torso(6), right(7), left(7)] on success,
        or None if the solve fails.
    """
    if diagram is None:
        diagram = MakeRby1Diagram()
    if options is None:
        options = Rby1ProblemOptions()

    options.right_target = right_target
    options.left_target = left_target
    if q_initial is not None:
        options.q_initial = q_initial

    # Program construction, separated from the solve. It is not a rounding
    # error: Rby1IKProblem.__init__ clones the plant to AutoDiffXd and
    # ApplyOptions registers every cost and constraint, once per IK attempt.
    with record("ik.build"):
        problem = Rby1IKProblemNewFormulation(diagram)
        problem.ApplyOptions(options)
    result = problem.Solve(visualize=visualize)

    if not result.is_success():
        return None

    q_full_plant = problem.VarsToQ(result.GetSolution(problem.lumped_vars))
    q_out = np.concatenate([
        q_full_plant[problem.base_gen_pos_idxs],
        q_full_plant[problem.torso_gen_pos_idxs],
        q_full_plant[problem.right_arm_gen_pos_idxs],
        q_full_plant[problem.left_arm_gen_pos_idxs],
    ])
    if not np.all(np.isfinite(q_out)):
        return None
    return q_out
