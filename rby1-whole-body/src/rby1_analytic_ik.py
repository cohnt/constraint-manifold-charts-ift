"""
An interface that wraps around that calls ikfast implementations, and
also respects the joint limits of the free parameter.
"""

from common import RepoDir
from rainbow_left_arm_ik import get_ik as left_get_ik, get_fk as _left_get_fk
from rainbow_right_arm_ik import get_ik as right_get_ik, get_fk as _right_get_fk

from typing import Optional
import math as _math


from pydrake.all import (
    AutoDiffXd,
    InitializeAutoDiff,
    PiecewiseQuaternionSlerp,
    RigidTransform,
    RigidTransform_,
    RotationMatrix,
    RotationMatrix_,
    ExtractValue,
    ExtractGradient,
    JacobianWrtVariable,
    RollPitchYaw,
    RollPitchYaw_,
    MultibodyPlant,
    Parser,
)

import numpy as np
import scipy


def _damped_lstsq(J: np.ndarray, rhs: np.ndarray, lambda_eff: float) -> np.ndarray:
    """Solve J x = rhs via SVD-based damped pseudoinverse with scalar damping lambda_eff.

    When lambda_eff <= 0, falls back to scipy.linalg.lstsq (identical behavior).
    Otherwise applies isotropic Levenberg-Marquardt damping:
        x = V * diag(sigma / (sigma^2 + lambda_eff)) * U^T * rhs
    """
    if lambda_eff <= 0.0:
        return scipy.linalg.lstsq(J, rhs)[0]
    U, sigma, Vt = np.linalg.svd(J, full_matrices=False)
    d = sigma / (sigma ** 2 + lambda_eff)
    return Vt.T @ (d[:, np.newaxis] * (U.T @ rhs))


# ── Global configuration parameters (GCP) ─────────────────────────────────────
#
# A GCP is the (wrist, elbow, shoulder) sign triple naming which of IKFast's 8
# solution branches a configuration lives on. The classification below is an
# exact bijection with the branch actually taken, mirroring the
# `SolveLeftArmBranch<ElbowBranch, ShoulderBranch, WristSign>` template of the
# branchless split solvers. Sign convention: +1 == Ep/Sp == template branch 0.
#
# It replaces a naive sign/threshold reading of the *wrapped output angles*,
# which was not a bijection: measured over 400 random reachable poses per arm,
# the naive labelling collided on 1382 (left) / 1390 (right) of the 8-solution
# lists, while this one collides on zero and returns exactly one solution per
# branch per pose. Two of its three predicates were simply wrong --
#   * elbow: it split at pi/2, but IKFast splits at A + pi/2 = -0.23236 with
#     A = -1.80315340003661. Since the elbow's joint limit is [-2.618, 0.01],
#     the old test was constant +1 across the whole reachable range, so the
#     elbow axis carried no information at all and the GCP equality check in
#     the constrained BiRRT's validity predicate never rejected on it.
#   * shoulder: the branch centre is pose-dependent (it moves with
#     x1158 = atan2(-px, -py)), not fixed at zero.
# The wrist predicate was already exact and is unchanged.

# j16/j25 = A +- x121 with x121 in [-pi/2, pi/2], giving the two branches the
# fixed disjoint raw windows [A-pi/2, A+pi/2] and [A+pi/2, A+3pi/2]. The shared
# boundary is exact and pose-independent.
_GCP_ELBOW_CENTRE = -1.80315340003661
_GCP_EP_MIN = _GCP_ELBOW_CENTRE - _math.pi / 2   # -3.37395...
_GCP_EP_MAX = _GCP_ELBOW_CENTRE + _math.pi / 2   # -0.23236... == Em's raw min

# Bound into locals at import: these are called ~2.3M times per planned point.
_fmod = _math.fmod
_atan2 = _math.atan2
_TWO_PI = 2.0 * _math.pi
_HALF_PI = _math.pi / 2

# Position part of `TransformToSolverFrame`, per arm. Only px and py are needed
# (they are all that x1158 depends on). THE TWO ARMS DIFFER: the right arm flips
# the sign of the 0.342021230003561 and 0.0529448864045513 terms in py. Using
# one arm's constants for the other silently crosses the two arms' shoulder
# branches, which is why the arm is a required argument everywhere below.
_GCP_PY_CONST = -2.72185494444282e-7
_GCP_PY_R22 = -0.145464356471299
_GCP_PY_PZ = -0.93969222526679
_GCP_PY_SIGNED = {          # arm -> (coeff of in_py, coeff of in_r12)
    "left": (0.342021230003561, 0.0529448864045513),
    "right": (-0.342021230003561, -0.0529448864045513),
}


def gcp_shoulder_reference(eetrans, eerot, arm: str) -> float:
    """``x1158 = atan2(-px, -py)`` -- the pose-dependent centre of the shoulder
    branch split, in the solver's internal frame.

    ``eetrans``/``eerot`` are the target pose in the same frame IKFast's
    ``get_ik`` takes (i.e. ``T_tl5_eef``); ``eerot`` is row-major 3x3.

    This depends only on the *pose*, not on which branch was taken, so
    ``compute_ik`` computes it once per arm per call and reuses it to label all
    8 solutions -- there is no per-solution cost on the hot path.
    """
    try:
        py_c, r12_c = _GCP_PY_SIGNED[arm]
    except KeyError:
        raise ValueError(f"arm must be 'left' or 'right', got {arm!r}") from None
    in_r02 = eerot[0][2]
    in_r12 = eerot[1][2]
    in_r22 = eerot[2][2]
    in_px, in_py, in_pz = eetrans[0], eetrans[1], eetrans[2]
    px = in_px + 0.1548 * in_r02
    py = (_GCP_PY_CONST + _GCP_PY_R22 * in_r22 + _GCP_PY_PZ * in_pz
          + py_c * in_py + r12_c * in_r12)
    return _atan2(-px, -py)


def _wrap_to_window(x: float, lo: float) -> float:
    """Wrap ``x`` into ``[lo, lo + 2*pi)``.

    Deliberately ``math`` and not ``numpy``: this runs inside ``gcp_key``, the
    planner's hottest predicate (a single-seed profile counted ~2.3M calls), and
    on Python scalars ``np.fmod`` measures 0.436 us against ``math.fmod``'s
    0.062 us -- a 7x difference that lands directly on every constrained-leg
    validity check and every trajopt constraint evaluation.
    """
    y = _fmod(x - lo, _TWO_PI)
    if y < 0.0:
        y += _TWO_PI
    return y + lo


def gcp_key(sol, x1158: float) -> int:
    """GCP as a 3-bit int: (wrist<<2 | elbow<<1 | shoulder), 1 bits == +1.

    ``x1158`` is :func:`gcp_shoulder_reference` for the pose ``sol`` solves.

    The single source of truth for the labelling; :func:`gcp_of_arm` is this in
    +-1 array form. The integer form exists because this is the hottest predicate
    in the planner -- ``compute_ik`` labels every IKFast solution to pick the one
    matching a requested GCP, and profiling the single-seed pipeline counted
    2,294,748 calls to the array-returning version plus the same number of
    ``np.array_equal`` comparisons, which together were the two most expensive
    lines in the entire profile. Comparing ints allocates nothing.

    Both ``sol[3]`` and ``sol[0]`` come back already wrapped to [-pi, pi] by the
    solver, which can fold a value across its own branch boundary, so each is
    re-folded into that branch's raw window before the comparison.
    """
    return (((1 if sol[5] >= 0 else 0) << 2)
            | ((1 if _wrap_to_window(sol[3], _GCP_EP_MIN) < _GCP_EP_MAX else 0) << 1)
            | (1 if _wrap_to_window(sol[0] + x1158, -_HALF_PI) < _HALF_PI else 0))


def gcp_key_of(gcp) -> int:
    """``gcp_key`` for an existing +-1 GCP array, to compare against."""
    return (((1 if gcp[0] > 0 else 0) << 2)
            | ((1 if gcp[1] > 0 else 0) << 1)
            | (1 if gcp[2] > 0 else 0))


def _first_matching_sol(sols, key: int, x1158: float):
    """The solution in ``sols`` whose label equals ``key``, else None.

    Early-exits on the match rather than building the whole filtered list --
    ``compute_ik`` only ever takes element 0 of it. Under this labelling the
    match is unique, so "first" and "the" coincide.
    """
    for s in sols:
        if gcp_key(s, x1158) == key:
            return s
    return None


def _key_to_gcp(k: int) -> np.ndarray:
    return np.array([1 if k & 4 else -1,
                     1 if k & 2 else -1,
                     1 if k & 1 else -1], dtype=np.int8)


def gcp_of_arm(sol, arm: str, x1158: Optional[float] = None):
    """Global configuration parameters for one arm's IKFast solution.

    sol: 7-element sequence [j22/j13, j23/j14, j24/j15(free), j25/j16,
                              j26/j17, j27/j18, j28/j19]
    arm: "left" or "right" -- required, because the solver-frame transform the
         shoulder predicate needs is sign-mirrored between the two arms.
    x1158: :func:`gcp_shoulder_reference` for the pose this solution solves.
         Optional only because most callers hold a configuration and not the
         pose it came from; when omitted it is recovered by running IKFast's own
         FK on ``sol``, which costs one native call. Pass it when you already
         have the pose (``compute_ik`` does) to skip that.

    Returns: 3-element int8 array [wrist_sign, elbow_branch, shoulder_sign]
      GCP[0]: +1 if sol[5] >= 0                    (j27/j18, acos +-theta)
      GCP[1]: +1 on the elbow branch j16 = A + x121  (Ep)
      GCP[2]: +1 on the shoulder branch j13 = -x1158 - x1159  (Sp)

    See :func:`gcp_key` for the derivation and for what the superseded
    labelling got wrong.
    """
    if x1158 is None:
        get_fk = _left_get_fk if arm == "left" else _right_get_fk
        if arm not in ("left", "right"):
            raise ValueError(f"arm must be 'left' or 'right', got {arm!r}")
        eetrans, eerot = get_fk([float(v) for v in sol])
        x1158 = gcp_shoulder_reference(eetrans, eerot, arm)
    return _key_to_gcp(gcp_key(sol, x1158))

from typing import Literal, TypeVar
from pathlib import Path
from warnings import warn
import os

DIST_TO_MID_GRIPPER_FROM_EE = (
    0.073 + 0.03
)  # 0.073 is dist to fing base, fingers are roughly 6cm long


# stand-in method for QR plant construction
def MakeRby1RobotPlant() -> MultibodyPlant:
    plant = MultibodyPlant(time_step=1.0)
    parser = Parser(plant)
    package_xml_path = os.path.join(RepoDir(), "package.xml")
    parser.package_map().AddPackageXml(package_xml_path)  # type: ignore
    assets_path = Path(__file__).parent.parent / "models" / "ruby"
    directives_file = (
        assets_path
        / "rby1_description_drake"
        / "add_rby1_sim_with_holonomic_base_actuators.dmd.yaml"
    )
    parser.AddModels(directives_file)
    plant.Finalize()
    return plant


# What Tomas does
# - samples between the minimum/maximum joint of rainbow joint angle 2 (zero relative)
# - then calls ik_fast with rotation matrix, position matrices in the form of lists and the free parameter (according to the urdf)
# - rot is 3x3 nested list, pos is 3-long list, + a float parameter for the last joint


# Desiderata
# - load in a ruby plant/urdf, and query the joint limits of joint #2 of both arms as a constructor
# use a plant for this.
# - input: desired transorm, relative to torso frame + redundancy parameter --> raise an exception of redundancy parameter is out of range.
# assume numpy input for now; easy to change later.
# - output: ik solutions


# questions/things to test
# - if there are multiple ik solutions, is the ordering re:c-bundles consistent when we perturb the end-effector goal?
#   - appear to be consistent with some local perturbations.
# - what is the behavior of ikfast when no solution is available?
# returns 'None'


# implementing full IK and gradients for trajopt
#   what we need
#       - a numericafunction that will consume minimum coordinates and output joint coordinates -- it appears that grouping the coordinates into arguments is actually okay.
#           - how are gradients taken through this solve?
#           - all poses are automatically represented using RigidTransform_[AutoDiffXd] (the templated generic with AutoDiff floats) for drake to autodiff. Instead, we need to find how to manually specify gradients to the solver.
#               - It's not that bad -- we just typecheck if we are in autodiff mode and then compute the gradient ourselves: https://github.com/RobotLocomotion/drake/blob/master/tutorials/custom_gradients.ipynb


# FloatType = TypeVar("FloatType", AutoDiffXd, float)
# ArrType = TypeVar("ArrType", np.ndarray[object], np.ndarray[np.float64])


class Rby1IK:
    def __init__(self, boundary_fallback: bool = True,
                 boundary_damping_lambda: float = 0.0):
        """``boundary_damping_lambda`` is the per-instance default for
        ``compute_ik``'s damping term, applied whenever a call does not name one.

        It exists because the damping only ever mattered to callers that set
        boundary_fallback=True, and those callers reach compute_ik through
        closures (trajopt constraint callables, TOPPRA's traj_fn) whose call
        sites are far from where the instance is chosen. Setting it once at
        construction is the difference between regularising every gradient this
        instance produces and regularising none of them: a sweep measured 14%
        IK success at lambda=0 against 100% at anything in [1e-3, 1], and
        rby1_opt_ik already defaults
        to 0.1 -- but _lift_trajopt's own instance was left at the 0.0 signature
        default, so the IK stage was regularised and the trajopt stage was not.

        Zero keeps the old behaviour, which is what the boundary_fallback=False
        (validity) instances want: they return NaN rather than a projection, so
        there is no boundary residual to damp.
        """
        self.boundary_fallback = boundary_fallback
        self.boundary_damping_lambda = boundary_damping_lambda
        self.plant = MakeRby1RobotPlant()
        self.context = self.plant.CreateDefaultContext()

        # named pieces of multibody plant
        self.base_instance = self.plant.GetModelInstanceByName("base")
        self.torso_instance = self.plant.GetModelInstanceByName("torso")
        self.right_arm_instance = self.plant.GetModelInstanceByName("right_arm")
        self.left_arm_instance = self.plant.GetModelInstanceByName("left_arm")

        self.base_gen_pos_idxs = self._get_gen_pos_idxs(self.base_instance)
        self.torso_gen_pos_idxs = self._get_gen_pos_idxs(self.torso_instance)
        self.right_arm_gen_pos_idxs = self._get_gen_pos_idxs(self.right_arm_instance)
        self.left_arm_gen_pos_idxs = self._get_gen_pos_idxs(self.left_arm_instance)

        self.frame_eef_right = self.plant.GetFrameByName("ee_right")
        self.frame_eef_left = self.plant.GetFrameByName("ee_left")
        self.frame_torso_end = self.plant.GetFrameByName("link_torso_5")
        self.world_frame = self.plant.world_frame()

        # autodiffed versions
        self.plant_adiff = self.plant.ToAutoDiffXd()
        self.context_adiff = self.plant_adiff.CreateDefaultContext()

        self.frame_eef_right_adiff = self.plant_adiff.GetFrameByName("ee_right")
        self.frame_eef_left_adiff = self.plant_adiff.GetFrameByName("ee_left")
        self.frame_torso_end_adiff = self.plant_adiff.GetFrameByName("link_torso_5")
        self.world_frame_adiff = self.plant_adiff.world_frame()

        free_joint_left = self.plant.GetJointByName("left_arm_2")
        free_joint_right = self.plant.GetJointByName("right_arm_2")

        self.left_free_lb, self.left_free_ub = (
            free_joint_left.position_lower_limit(),
            free_joint_left.position_upper_limit(),
        )
        self.right_free_lb, self.right_free_ub = (
            free_joint_right.position_lower_limit(),
            free_joint_right.position_upper_limit(),
        )

    def _get_gen_pos_idxs(self, model_inst_idx: int) -> list[int]:
        joint_idxs = self.plant.GetJointIndices(model_inst_idx)
        joints = [self.plant.get_joint(_jidx) for _jidx in joint_idxs]
        pos_idxs = []

        for _j in joints:
            _n_pos = _j.num_positions()
            if _n_pos > 0:
                _start_pos = _j.position_start()
                pos_idxs += list(range(_start_pos, _start_pos + _n_pos))

        return pos_idxs

    def compute_ik(
        self,
        q_base_torso_reef_leef: np.ndarray[np.float64] | np.ndarray[object],
        right_gcp=None,
        left_gcp=None,
        boundary_tol: float = 1e-3,
        boundary_damping_lambda: Optional[float] = None,
        # base_pos: np.ndarray[np.float64] | np.ndarray[object],
        # torso_pos: np.ndarray[np.float64] | np.ndarray[object],
        # T_W_reef: RigidTransform_[float] | RigidTransform_[AutoDiffXd],
        # right_free: float | AutoDiffXd,
        # T_W_leef: RigidTransform_[np.float64] | RigidTransform_[AutoDiffXd],
        # left_free: float | AutoDiffXd,
    ):
        if boundary_damping_lambda is None:
            boundary_damping_lambda = self.boundary_damping_lambda

        base_pos = q_base_torso_reef_leef[:3]
        torso_pos = q_base_torso_reef_leef[3 : 3 + 6]
        T_W_reef_param = q_base_torso_reef_leef[9 : 9 + 6]
        right_free = q_base_torso_reef_leef[15 : 15 + 1]
        T_W_leef_param = q_base_torso_reef_leef[16 : 16 + 6]
        left_free = q_base_torso_reef_leef[22:23]

        autodiffing = isinstance(right_free[0], AutoDiffXd)
        template_type = AutoDiffXd if autodiffing else float

        # T_W_reef = rigid_transform_from_exp_coordinates(T_W_reef_param)
        # T_W_leef = rigid_transform_from_exp_coordinates(T_W_leef_param)

        T_W_reef = RigidTransform_[template_type](
            RollPitchYaw_[template_type](T_W_reef_param[3:]), T_W_reef_param[:3]
        )
        T_W_leef = RigidTransform_[template_type](
            RollPitchYaw_[template_type](T_W_leef_param[3:]), T_W_leef_param[:3]
        )

        # compute fk of base to torso_link_5 (we are not using the fk module)
        # because we need access to fine grained differentiation
        # NOTE: maybe we should merge these modules later.

        if autodiffing:
            # NOTE: these values remain the same before/after the parameterization,

            base_pos_val = ExtractValue(base_pos)
            torso_pos_val = ExtractValue(torso_pos)
            right_free_val = ExtractValue(right_free)
            left_free_val = ExtractValue(left_free)

            base_pos_grad = ExtractGradient(base_pos)
            torso_pos_grad = ExtractGradient(torso_pos)
            right_free_grad = ExtractGradient(right_free)
            left_free_grad = ExtractGradient(left_free)

            T_W_leef_adiff = T_W_leef.GetAsMatrix4()
            # ensure that ordering is in row-major to make it easier to unflatten gradient calculations
            T_W_leef_val = ExtractValue(T_W_leef_adiff)
            T_W_leef_float = RigidTransform_[float](T_W_leef_val)
            T_W_leef_grad = ExtractGradient(T_W_leef_adiff)

            T_W_reef_adiff = T_W_reef.GetAsMatrix4()
            # ensure that ordering is in row-major to make it easier to unflatten gradient calculations
            T_W_reef_val = ExtractValue(T_W_reef_adiff)
            T_W_reef_float = RigidTransform_[float](T_W_reef_val)
            T_W_reef_grad = ExtractGradient(T_W_reef_adiff)

            # we have to reshape matrices since we get partials with respect to every parameter, and all matrices
            # are flatted out in this representation (e.g. mat4x4 diff time results in a 16 X 1 grad vector)
            # we'll have to do some fancy vectorization: https://drake.mit.edu/pydrake/pydrake.autodiffutils.html#pydrake.autodiffutils.ExtractGradient
            # reshape gradients from 16 X n partials to n partials X 4 X 4
            # Fortran-style (column-major) order, which is how drake formats arrays.
            T_W_reef_grad_shaped = T_W_reef_grad.T.reshape((-1, 4, 4), order="F")
            T_W_leef_grad_shaped = T_W_leef_grad.T.reshape((-1, 4, 4), order="F")

        else:
            base_pos_val = base_pos
            torso_pos_val = torso_pos
            right_free_val = right_free
            left_free_val = left_free
            T_W_reef_float = T_W_reef
            T_W_leef_float = T_W_leef

        # assert self.right_free_lb <= right_free_val <= self.right_free_ub, (
        #     f"right_free must be within bounds: ({self.right_free_lb}, {self.right_free_ub})"
        # )
        # assert self.left_free_lb <= left_free_val <= self.left_free_ub, (
        #     f"left_free must be within bounds: ({self.left_free_lb}, {self.left_free_ub})"
        # )

        self.plant.SetPositions(self.context, self.base_instance, base_pos_val)
        self.plant.SetPositions(self.context, self.torso_instance, torso_pos_val)

        # torso link five position relative to world frame
        T_W_tl5 = self.plant.CalcRelativeTransform(
            self.context, self.world_frame, self.frame_torso_end
        )
        T_tl5_reef = T_W_tl5.InvertAndCompose(T_W_reef_float)
        T_tl5_leef = T_W_tl5.InvertAndCompose(T_W_leef_float)

        # compute right analytic ik solutions
        right_target_np = T_tl5_reef.GetAsMatrix4()
        right_target_rot = right_target_np[:3, :3].tolist()
        right_target_pos = right_target_np[:3, 3].tolist()

        right_sols = right_get_ik(
            right_target_rot, right_target_pos, [right_free_val.item()]
        )  # type: ignore

        # compute left analytic ik solutions
        left_target_np = T_tl5_leef.GetAsMatrix4()
        left_target_rot = left_target_np[:3, :3].tolist()
        left_target_pos = left_target_np[:3, 3].tolist()

        left_sols = left_get_ik(
            left_target_rot, left_target_pos, [left_free_val.item()]
        )  # type: ignore

        # Right arm solution selection
        right_failed = False
        right_used_boundary = False
        if right_sols is None:
            warn(
                "IK solution not found for right arm."
                + (" Searching for nearest workspace boundary point." if self.boundary_fallback else "")
            )
            if self.boundary_fallback:
                qright_val = self._find_workspace_boundary_sol(
                    T_W_tl5, T_tl5_reef, right_free_val.item(), arm="right", tol=boundary_tol,
                )
                right_used_boundary = True
            else:
                qright_val = np.zeros(7)  # placeholder; right_failed signals nan on output
                right_failed = True
        elif right_gcp is not None:
            # x1158 depends only on the target pose, so it is computed once here
            # and reused to label all of this pose's solutions.
            right_x1158 = gcp_shoulder_reference(
                right_target_pos, right_target_rot, "right")
            right_matching = _first_matching_sol(
                right_sols, gcp_key_of(right_gcp), right_x1158)
            if right_matching is None:
                qright_val = np.zeros(7)  # placeholder; right_failed signals nan on output
                right_failed = True
            else:
                qright_val = right_matching
        else:
            qright_val = right_sols[0]

        # Left arm solution selection
        left_failed = False
        left_used_boundary = False
        if left_sols is None:
            warn(
                "IK solution not found for left arm."
                + (" Searching for nearest workspace boundary point." if self.boundary_fallback else "")
            )
            if self.boundary_fallback:
                qleft_val = self._find_workspace_boundary_sol(
                    T_W_tl5, T_tl5_leef, left_free_val.item(), arm="left", tol=boundary_tol,
                )
                left_used_boundary = True
            else:
                qleft_val = np.zeros(7)  # placeholder; left_failed signals nan on output
                left_failed = True
        elif left_gcp is not None:
            left_x1158 = gcp_shoulder_reference(
                left_target_pos, left_target_rot, "left")
            left_matching = _first_matching_sol(
                left_sols, gcp_key_of(left_gcp), left_x1158)
            if left_matching is None:
                qleft_val = np.zeros(7)  # placeholder; left_failed signals nan on output
                left_failed = True
            else:
                qleft_val = left_matching
        else:
            qleft_val = left_sols[0]

        # When both arms fail in autodiff mode, skip Drake's Jacobian computation entirely
        if right_failed and left_failed and autodiffing:
            _n_p = base_pos_grad.shape[1]
            return np.concatenate([
                base_pos, torso_pos,
                InitializeAutoDiff(np.full(7, np.nan), np.zeros((7, _n_p))).flatten(),
                InitializeAutoDiff(np.full(7, np.nan), np.zeros((7, _n_p))).flatten(),
            ])

        if not autodiffing:
            return np.concatenate([
                base_pos, torso_pos,
                np.full(7, np.nan) if right_failed else qright_val,
                np.full(7, np.nan) if left_failed else qleft_val,
            ])

        # convert from derivative of RigidBody parameters to a spatial velocity vector (e.g. [translational, rotational]) via https://math.stackexchange.com/questions/4534687/axis-angle-representation-of-angular-velocity

        # compute Lie algebra representation and then read of the spatial angular velocity
        # NOTE: I'm not sure this is the most _numerically_ stable way to extract the spatial vel
        omega_mat_leef_stacked = (
            T_W_leef_grad_shaped[:, :3, :3] @ T_W_leef_val[:3, :3].T
        )  # type: ignore

        # use advanced indexing to extract the right bits of the lie algebra representation matrix
        angvel_rows = np.array([2, 0, 1])
        angvel_cols = np.array([1, 2, 0])
        angvel_leef_stacked = omega_mat_leef_stacked[:, angvel_rows, angvel_cols]
        spatial_ang_vel_leef_vstacked = np.hstack(
            [angvel_leef_stacked, T_W_leef_grad_shaped[:, :3, 3]]
        )  # n_partials X 6

        omega_mat_reef_stacked = (
            T_W_reef_grad_shaped[:, :3, :3] @ T_W_reef_val[:3, :3].T
        )  # type: ignore
        angvel_reef_stacked = omega_mat_reef_stacked[:, angvel_rows, angvel_cols]
        spatial_ang_vel_reef_vstacked = np.hstack(
            [angvel_reef_stacked, T_W_reef_grad_shaped[:, :3, 3]]
        )  # n_partials X 6

        # compute geometric Jacobian (relative to joint positions of robot configuration) for each eef
        # positions have already been set for the base and torso, so we just need to set the right and left arms
        self.plant.SetPositions(self.context, self.right_arm_instance, qright_val)
        self.plant.SetPositions(self.context, self.left_arm_instance, qleft_val)

        # Compute squared residual norms for boundary-fallback damping.
        # The 7D residual = [rot_err(3), trans_err(3), free_err(1)] measures how far
        # the boundary solution is from the original (unreachable) target.
        right_res_norm_sq = 0.0
        if boundary_damping_lambda > 0.0 and right_used_boundary and not np.any(np.isnan(qright_val)):
            T_reef_achieved = self.plant.CalcRelativeTransform(
                self.context, self.world_frame, self.frame_eef_right
            )
            trans_err_r = T_W_reef_float.translation() - T_reef_achieved.translation()
            R_err_r = T_W_reef_float.rotation().multiply(T_reef_achieved.rotation().inverse())
            aa_r = R_err_r.ToAngleAxis()
            rot_err_r = aa_r.axis() * aa_r.angle()
            pose_res_r = np.concatenate([rot_err_r, trans_err_r])
            free_res_r = qright_val[2] - right_free_val.item()
            right_res_norm_sq = float(np.dot(pose_res_r, pose_res_r) + free_res_r ** 2)

        left_res_norm_sq = 0.0
        if boundary_damping_lambda > 0.0 and left_used_boundary and not np.any(np.isnan(qleft_val)):
            T_leef_achieved = self.plant.CalcRelativeTransform(
                self.context, self.world_frame, self.frame_eef_left
            )
            trans_err_l = T_W_leef_float.translation() - T_leef_achieved.translation()
            R_err_l = T_W_leef_float.rotation().multiply(T_leef_achieved.rotation().inverse())
            aa_l = R_err_l.ToAngleAxis()
            rot_err_l = aa_l.axis() * aa_l.angle()
            pose_res_l = np.concatenate([rot_err_l, trans_err_l])
            free_res_l = qleft_val[2] - left_free_val.item()
            left_res_norm_sq = float(np.dot(pose_res_l, pose_res_l) + free_res_l ** 2)

        jac_geom_right_eef = self.plant.CalcJacobianSpatialVelocity(
            self.context,
            JacobianWrtVariable.kQDot,
            frame_B=self.frame_eef_right,
            p_BoBp_B=np.zeros(3),
            frame_A=self.world_frame,
            frame_E=self.world_frame,
        )

        jac_geom_left_eef = self.plant.CalcJacobianSpatialVelocity(
            self.context,
            JacobianWrtVariable.kQDot,
            frame_B=self.frame_eef_left,
            p_BoBp_B=np.zeros(3),
            frame_A=self.world_frame,
            frame_E=self.world_frame,
        )

        # extract the relevant submatrices of the Jacobian for the relevant chains
        del_reef_del_qbase = jac_geom_right_eef[:, self.base_gen_pos_idxs]
        del_reef_del_qtorso = jac_geom_right_eef[:, self.torso_gen_pos_idxs]

        del_reef_del_qright = jac_geom_right_eef[:, self.right_arm_gen_pos_idxs]
        del_reef_del_qright_aug = np.vstack(
            [del_reef_del_qright, np.array([0, 0, 1, 0, 0, 0, 0])]
        )  # NOTE: is there a less magic numbery way to do this?

        del_leef_del_qbase = jac_geom_left_eef[:, self.base_gen_pos_idxs]
        del_leef_del_qtorso = jac_geom_left_eef[:, self.torso_gen_pos_idxs]

        del_leef_del_qleft = jac_geom_left_eef[:, self.left_arm_gen_pos_idxs]
        del_leef_del_qleft_aug = np.vstack(
            [del_leef_del_qleft, np.array([0, 0, 1, 0, 0, 0, 0])]
        )  # NOTE: is there a less magic numbery way to do this?

        # proceed with by solving the linear system
        # J_geom qdot =  diff base diff torso + differentiated eefs;
        # exploiting that J_geom (when considered for the whole system) is lower triangular

        # base + torso diffs + redundancy parameters are already identified
        # solve right arm diffs

        # tranpose shenigans is because we use n_partials as a batching dimension,
        # per numpy broadcasting rules, but need to convert back to n_partials
        # being the second dimension per Eigen AutoDiff rules when we're done.
        right_resid = (
            spatial_ang_vel_reef_vstacked.T  # (6 x n_partials) (cuz transpose)
            - del_reef_del_qbase @ base_pos_grad  # (6 x 3) * (3 x n_partials)
            - del_reef_del_qtorso @ torso_pos_grad  # (6 x 6) * (6 x n_partials)
        )  # 6 X n_partials
        right_resid_aug = np.vstack(
            [right_resid, right_free_grad]
        )  # 7 X n_partials (due to augmentation)
        right_lambda_eff = boundary_damping_lambda * right_res_norm_sq
        qright_grad = _damped_lstsq(del_reef_del_qright_aug, right_resid_aug, right_lambda_eff)

        # solve left arm diffs
        left_resid = (
            spatial_ang_vel_leef_vstacked.T
            - del_leef_del_qbase @ base_pos_grad
            - del_leef_del_qtorso @ torso_pos_grad
        )
        left_resid_aug = np.vstack([left_resid, left_free_grad])
        left_lambda_eff = boundary_damping_lambda * left_res_norm_sq
        qleft_grad = _damped_lstsq(del_leef_del_qleft_aug, left_resid_aug, left_lambda_eff)

        n_partials = qright_grad.shape[1]
        right_out = (
            InitializeAutoDiff(np.full(7, np.nan), np.zeros((7, n_partials))).flatten()
            if right_failed
            else InitializeAutoDiff(qright_val, qright_grad).flatten()
        )
        left_out = (
            InitializeAutoDiff(np.full(7, np.nan), np.zeros((7, n_partials))).flatten()
            if left_failed
            else InitializeAutoDiff(qleft_val, qleft_grad).flatten()
        )
        return np.concatenate([base_pos, torso_pos, right_out, left_out])

    def _find_workspace_boundary_sol(
        self,
        T_W_tl5,
        T_tl5_target,
        free_val: float,
        arm: str,
        tol: float = 1e-3,
        max_iter: int = 100,
    ) -> np.ndarray:
        """Binary search in SE(3) from the ready-pose EEF toward T_tl5_target
        to find the workspace boundary, then return joint angles of the closest
        reachable point.

        Assumes base and torso positions are already set in self.context.
        Terminates when both the translation interval and the angular interval
        between the in-workspace and out-of-workspace endpoints are narrower than tol.
        """
        get_ik_fn = right_get_ik if arm == "right" else left_get_ik
        arm_instance = (
            self.right_arm_instance if arm == "right" else self.left_arm_instance
        )
        eef_frame = (
            self.frame_eef_right if arm == "right" else self.frame_eef_left
        )

        # Compute ready-pose EEF position in torso frame as the in-workspace anchor.
        # Ready pose from RainbowRobotics/rby1-sdk (17_teleoperation_with_joint_mapping.py).
        ready_right = np.deg2rad([0.0, -5.0, 0.0, -120.0, 0.0, 70.0, 0.0])
        ready_left  = np.deg2rad([0.0,  5.0, 0.0, -120.0, 0.0, 70.0, 0.0])
        q_ready = ready_right if arm == "right" else ready_left
        self.plant.SetPositions(self.context, arm_instance, q_ready)
        T_W_eef_ready = self.plant.CalcRelativeTransform(
            self.context, self.world_frame, eef_frame
        )
        T_tl5_eef_zero = T_W_tl5.InvertAndCompose(T_W_eef_ready)

        mat_zero = T_tl5_eef_zero.GetAsMatrix4()
        sols_zero = get_ik_fn(
            mat_zero[:3, :3].tolist(), mat_zero[:3, 3].tolist(), [free_val]
        )
        if sols_zero is None:
            warn(
                f"{arm} arm: ready-pose EEF also has no IK solution; "
                "cannot perform workspace boundary search."
            )
            return np.full(7, np.nan)

        q_boundary = np.array(sols_zero[0], dtype=float)

        R_in = T_tl5_eef_zero.rotation()   # RotationMatrix (in workspace)
        t_in = T_tl5_eef_zero.translation().copy()
        R_out = T_tl5_target.rotation()    # RotationMatrix (out of workspace)
        t_out = T_tl5_target.translation().copy()

        for _ in range(max_iter):
            # Convergence: both translation and angular intervals narrow enough
            trans_dist = np.linalg.norm(t_in - t_out)
            R_rel = R_in.inverse().multiply(R_out)
            cos_angle = np.clip((np.trace(R_rel.matrix()) - 1.0) / 2.0, -1.0, 1.0)
            angle_dist = np.arccos(cos_angle)
            if trans_dist < tol and angle_dist < tol:
                break

            # SE(3) midpoint: lerp in R^3, SLERP in SO(3)
            t_mid = 0.5 * (t_in + t_out)
            slerp = PiecewiseQuaternionSlerp([0.0, 1.0], [R_in, R_out])
            R_mid = RotationMatrix(slerp.orientation(0.5))

            T_mid = RigidTransform(R_mid, t_mid)
            mat = T_mid.GetAsMatrix4()
            sols = get_ik_fn(mat[:3, :3].tolist(), mat[:3, 3].tolist(), [free_val])

            if sols is not None:
                R_in = R_mid
                t_in = t_mid
                q_boundary = np.array(sols[0], dtype=float)
            else:
                R_out = R_mid
                t_out = t_mid

        return q_boundary

    def gcp(self, q_joints):
        """GCPs for both arms from a full 23-joint configuration.

        q_joints: 23-element array [base(3), torso(6), right(7), left(7)]
        Returns: (right_gcp, left_gcp), each a 3-element int8 array
        """
        return (gcp_of_arm(q_joints[9:16], "right"),
                gcp_of_arm(q_joints[16:23], "left"))

    def compute_fk(
        self, q_base_torso_right_left
    ):  # base_pos, torso_pos, right_arm_pos, left_arm_pos):

        # even though this is slower, implement via multibodyplant to reduce
        # number of self-implemented parts that need to work
        # spatial jacobian computation after

        base_pos = q_base_torso_right_left[:3]
        torso_pos = q_base_torso_right_left[3 : 3 + 6]
        right_arm_pos = q_base_torso_right_left[9 : 9 + 7]
        left_arm_pos = q_base_torso_right_left[16 : 16 + 7]

        if base_pos.dtype == float:
            self.plant.SetPositions(self.context, self.base_instance, base_pos)
            self.plant.SetPositions(self.context, self.torso_instance, torso_pos)
            self.plant.SetPositions(
                self.context, self.right_arm_instance, right_arm_pos
            )
            self.plant.SetPositions(self.context, self.left_arm_instance, left_arm_pos)
            reef_frame = self.plant.CalcRelativeTransform(
                self.context, self.world_frame, self.frame_eef_right
            )
            leef_frame = self.plant.CalcRelativeTransform(
                self.context, self.world_frame, self.frame_eef_left
            )

        else:
            self.plant_adiff.SetPositions(
                self.context_adiff, self.base_instance, base_pos
            )
            self.plant_adiff.SetPositions(
                self.context_adiff, self.torso_instance, torso_pos
            )
            self.plant_adiff.SetPositions(
                self.context_adiff, self.right_arm_instance, right_arm_pos
            )
            self.plant_adiff.SetPositions(
                self.context_adiff, self.left_arm_instance, left_arm_pos
            )
            reef_frame = self.plant_adiff.CalcRelativeTransform(
                self.context_adiff, self.world_frame_adiff, self.frame_eef_right_adiff
            )
            leef_frame = self.plant_adiff.CalcRelativeTransform(
                self.context_adiff, self.world_frame_adiff, self.frame_eef_left_adiff
            )

        # reef_param = rigid_transform_to_exp_coordinates(reef_frame)
        # leef_param = rigid_transform_to_exp_coordinates(leef_frame)

        reef_param = np.concatenate(
            [reef_frame.translation(), reef_frame.rotation().ToRollPitchYaw().vector()]
        )
        leef_param = np.concatenate(
            [leef_frame.translation(), leef_frame.rotation().ToRollPitchYaw().vector()]
        )

        return np.concatenate(
            [
                base_pos,
                torso_pos,
                reef_param,
                right_arm_pos[2:3].reshape(
                    1,
                ),
                leef_param,
                left_arm_pos[2:3].reshape(
                    1,
                ),
            ]
        )

    def compute_kinematic_jac_from_joints(
        self,
        q_base_torso_right_left: np.ndarray,
    ):
        """
        Returns Jacobians in order (right_eef, left_eef)
        """

        base_pos = q_base_torso_right_left[:3]
        torso_pos = q_base_torso_right_left[3 : 3 + 6]
        right_arm_pos = q_base_torso_right_left[9 : 9 + 7]
        left_arm_pos = q_base_torso_right_left[16 : 16 + 7]
        
        autodiffing = isinstance(q_base_torso_right_left[0], AutoDiffXd)

        if not autodiffing:

            self.plant.SetPositions(self.context, self.right_arm_instance, right_arm_pos) 
            self.plant.SetPositions(self.context, self.left_arm_instance, left_arm_pos)
            self.plant.SetPositions(self.context, self.base_instance, base_pos)
            self.plant.SetPositions(self.context, self.torso_instance, torso_pos)


            jac_geom_right_eef = self.plant.CalcJacobianSpatialVelocity(
                self.context,
                JacobianWrtVariable.kQDot,
                frame_B=self.frame_eef_right,
                p_BoBp_B=np.zeros(3),
                frame_A=self.frame_torso_end,
                frame_E=self.world_frame,
            )

            jac_geom_left_eef = self.plant.CalcJacobianSpatialVelocity(
                self.context,
                JacobianWrtVariable.kQDot,
                frame_B=self.frame_eef_left,
                p_BoBp_B=np.zeros(3),
                frame_A=self.frame_torso_end,
                frame_E=self.world_frame,
            )

            return jac_geom_right_eef, jac_geom_left_eef
        

        self.plant_adiff.SetPositions(self.context_adiff, self.right_arm_instance, right_arm_pos) 
        self.plant_adiff.SetPositions(self.context_adiff, self.left_arm_instance, left_arm_pos)
        self.plant_adiff.SetPositions(self.context_adiff, self.base_instance, base_pos)
        self.plant_adiff.SetPositions(self.context_adiff, self.torso_instance, torso_pos)


        jac_geom_right_eef_adiff = self.plant_adiff.CalcJacobianSpatialVelocity(
            self.context_adiff,
            JacobianWrtVariable.kV,
            frame_B=self.frame_eef_right_adiff,
            p_BoBp_B=InitializeAutoDiff(np.zeros(3)),
            frame_A=self.frame_torso_end_adiff,
            frame_E=self.world_frame_adiff,
        )

        jac_geom_left_eef_adiff = self.plant_adiff.CalcJacobianSpatialVelocity(
            self.context_adiff,
            JacobianWrtVariable.kV,
            frame_B=self.frame_eef_left_adiff,
            p_BoBp_B=InitializeAutoDiff(np.zeros(3)),
            frame_A=self.frame_torso_end_adiff,
            frame_E=self.world_frame_adiff,
        )

        return jac_geom_right_eef_adiff, jac_geom_left_eef_adiff




    def untransform_gripper(
        self, grip_target: RigidTransform_[float] | RigidTransform_[AutoDiffXd]
    ):
        ad = isinstance(grip_target, RigidTransform_[AutoDiffXd])
        type_used = AutoDiffXd if ad else float
        T = RigidTransform_[type_used](
            np.array(
                [
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, -DIST_TO_MID_GRIPPER_FROM_EE],
                    [0, 0, 0, 1],
                ]
            )
        )
        T_gripper = grip_target.multiply(T.inverse())

        return T_gripper


def hat(v):
    """Skew-symmetric matrix."""
    # fmt: off
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ], dtype=object)
    # fmt: on


def rigid_transform_from_exp_coordinates(q: np.ndarray[float] | np.ndarray[object]):
    """
    q: length-6 numpy array of AutoDiffXd
       (x, y, z, a, b, c)
       where (a,b,c) are exponential coordinates.

    Returns:
        RigidTransform_[AutoDiffXd]
    """

    autodiffing = isinstance(q[0], AutoDiffXd)
    template_type = AutoDiffXd if autodiffing else float

    p = np.array([q[0], q[1], q[2]])
    phi = np.array([q[3], q[4], q[5]])

    theta_sq = phi.dot(phi)
    theta = np.sqrt(theta_sq)

    Phi = hat(phi)

    # sinc(theta) = sin(theta)/theta
    sinc_theta = diffable_sinc(theta / np.pi)

    # (1 - cos(theta)) / theta^2 = 0.5 * sinc(theta/2)^2
    sinc_half = diffable_sinc((theta / 2.0) / np.pi)
    A = sinc_theta
    B = 0.5 * sinc_half * sinc_half

    R = np.eye(3) + A * Phi + B * (Phi @ Phi)

    R = RotationMatrix_[template_type](R)

    return RigidTransform_[template_type](R, p)


def vee(Phi):
    """Inverse of hat(): maps a 3x3 skew matrix to a 3-vector."""
    return np.array([Phi[2, 1], Phi[0, 2], Phi[1, 0]])


def rigid_transform_to_exp_coordinates(
    X: RigidTransform_[float] | RigidTransform_[AutoDiffXd],
):
    """
    Smooth, branch-free(ish) inverse of rigid_transform_from_exp_coordinates.

    Returns q = [p_x, p_y, p_z, phi_x, phi_y, phi_z] where phi is axis-angle
    exponential coordinates such that R = Exp(hat(phi)).

    Notes:
      - Stable near theta=0 (uses atan2 + sinc).
      - Not guaranteed to match the "principal log" perfectly near theta≈pi.
    """
    R = X.rotation().matrix()
    p = X.translation()

    autodiffing = isinstance(R[0, 0], AutoDiffXd)
    dtype = object if autodiffing else float

    # v = vee(R - Rᵀ) = 2 sin(theta) * axis
    v = vee(R - R.T)

    # s = ||v|| / 2 = |sin(theta)|
    v_dot = v[0] * v[0] + v[1] * v[1] + v[2] * v[2]
    s = 0.5 * np.sqrt(v_dot)

    # c = cos(theta)
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    c = (tr - 1.0) / 2.0

    # theta in [0, pi], robust near 0 without acos/clamping
    theta = np.arctan2(s, c)

    # alpha = theta / (2 sin(theta)) = 1 / (2 * (sin(theta)/theta))
    # sin(theta)/theta = sinc(theta/pi) in numpy's normalized convention
    alpha = 1.0 / (2.0 * diffable_sinc(theta / np.pi))

    # Phi = log(R) = alpha * (R - Rᵀ), phi = vee(Phi)
    phi = vee(alpha * (R - R.T))

    return np.array([p[0], p[1], p[2], phi[0], phi[1], phi[2]], dtype=dtype)


def diffable_sinc(x, eps=1e-20):
    # taken from: https://github.com/numpy/numpy/blob/8f8bbcb57a69fc488a91612c362cc08dea959f7f/numpy/lib/_function_base_impl.py#L3811
    x = np.asanyarray(x)
    x = np.pi * x
    y = np.where(x, x, eps)
    return np.sin(y) / y


if __name__ == "__main__":
    rby1_ik = Rby1IK()

    q_base = np.zeros(3)
    q_torso = np.array([0, -0.05, 0.05, 0, 0, 0])
    q_right_arm = np.array([1.5, 0.3, 0, 0.3, 0, 0.1, 0.1])
    q_left_arm = np.array([1.5, 0.3, 0, 0.3, 0, 0.1, 0.1])

    q_full = np.concatenate([q_base, q_torso, q_right_arm, q_left_arm])

    print("testing non-diff fk/ik")
    fk_param = rby1_ik.compute_fk(q_full)
    ik_param = rby1_ik.compute_ik(fk_param)

    fk_param_2 = rby1_ik.compute_fk(ik_param)
    print(fk_param - fk_param_2)

    print("testing diff fk/ik")

    fk_param_adiff = InitializeAutoDiff(fk_param).flatten()
    fk_param_again_adiff = rby1_ik.compute_fk(rby1_ik.compute_ik(fk_param_adiff))

    param_grads = ExtractGradient(fk_param_again_adiff)

    print(np.linalg.norm(param_grads - np.eye(23)))
    print(param_grads)

    # --- GCP tests ---
    # The invariant that matters is that the labelling is a *bijection* with
    # IKFast's solution branches: for any reachable pose, the 8 returned
    # solutions must carry 8 distinct labels, and re-solving a pose must hand
    # back the same configuration under the same label. The superseded
    # sign/threshold labelling failed both (its elbow bit was constant across
    # the whole joint range, and its shoulder bit was pose-independent), which
    # is what these replace.
    print("\ntesting gcp_of_arm is a bijection with the solution branches")
    _rng = np.random.default_rng(0)
    _lo = np.array([-2.8, -1.6, -2.8, -2.618, -2.8, -1.6, -2.8])
    _hi = np.array([ 2.8,  1.6,  2.8,  0.01,   2.8,  1.6,  2.8])
    for _arm, _get_fk, _get_ik in (("left", _left_get_fk, left_get_ik),
                                   ("right", _right_get_fk, right_get_ik)):
        _n_poses = 0
        for _ in range(50):
            _q = _rng.uniform(_lo, _hi)
            _eetrans, _eerot = _get_fk(_q.tolist())
            _sols = _get_ik(_eerot, _eetrans, [_q[2]])
            if not _sols:
                continue
            _n_poses += 1
            _x = gcp_shoulder_reference(_eetrans, _eerot, _arm)
            _keys = [gcp_key(_s, _x) for _s in _sols]
            assert len(set(_keys)) == len(_keys), (
                f"{_arm}: {len(_keys) - len(set(_keys))} solutions share a GCP label"
            )
            # The sampled configuration must be recovered by its own label.
            _k = gcp_key(_q, _x)
            _back = _first_matching_sol(_sols, _k, _x)
            assert _back is not None, f"{_arm}: no solution carries the sampled config's own label"
            assert np.max(np.abs(np.asarray(_back) - _q)) < 1e-6, (
                f"{_arm}: label {_k} selected a different configuration"
            )
        assert _n_poses > 0
        print(f"  {_arm}: {_n_poses} poses, all 8 branches distinctly labelled: OK")

    print("testing the two arms' shoulder references are not interchangeable")
    # Guards against the arms' sign-mirrored solver-frame transforms being
    # crossed: classifying a right-arm solution with the left-arm transform
    # must not silently produce the same answer.
    _q = _rng.uniform(_lo, _hi)
    _eetrans, _eerot = _right_get_fk(_q.tolist())
    assert not np.isclose(gcp_shoulder_reference(_eetrans, _eerot, "right"),
                          gcp_shoulder_reference(_eetrans, _eerot, "left")), \
        "left/right solver-frame transforms coincide -- the per-arm constants are wrong"
    try:
        gcp_shoulder_reference(_eetrans, _eerot, "either")
        raise AssertionError("bad arm name was accepted")
    except ValueError:
        pass
    print("  per-arm frames distinct, bad arm name rejected: OK")

    print("testing Rby1IK.gcp round-trip")
    right_gcp, left_gcp = rby1_ik.gcp(q_full)
    print(f"  right_gcp={right_gcp}  left_gcp={left_gcp}")
    # GCP extracted from q_full should match gcp_of_arm applied to the arm subvectors
    assert np.array_equal(right_gcp, gcp_of_arm(q_right_arm, "right"))
    assert np.array_equal(left_gcp,  gcp_of_arm(q_left_arm, "left"))
    print("  Rby1IK.gcp: OK")

    print("testing compute_ik with GCPs")
    # FK -> IK with GCP locked to initial branch -> FK again; param round-trip should hold
    ik_with_gcp = rby1_ik.compute_ik(fk_param, right_gcp=right_gcp, left_gcp=left_gcp)
    assert ik_with_gcp is not None, "compute_ik returned None for a valid GCP"
    fk_roundtrip = rby1_ik.compute_fk(ik_with_gcp)
    param_err = np.linalg.norm(fk_param - fk_roundtrip)
    print(f"  param round-trip error with GCP: {param_err:.2e}  (should be ~0)")
    assert param_err < 1e-4, f"round-trip error too large: {param_err}"

    # GCP that doesn't exist should return None
    bad_gcp = np.array([-right_gcp[0], -right_gcp[1], -right_gcp[2]], dtype=np.int8)
    result_none = rby1_ik.compute_ik(fk_param, right_gcp=bad_gcp)
    # May or may not be None depending on whether any solution with all-flipped GCP exists;
    # just verify the call doesn't crash and returns either None or a valid array
    print(f"  inverted GCP result: {result_none is None and 'None (branch absent)' or 'solution found on other branch'}")

    print("testing compute_ik with only one GCP set (other arm unconstrained)")
    ik_right_only = rby1_ik.compute_ik(fk_param, right_gcp=right_gcp)
    assert ik_right_only is not None, "compute_ik returned None with only right_gcp set"
    assert np.array_equal(gcp_of_arm(ik_right_only[9:16], "right"), right_gcp), "right arm GCP not respected"
    print("  right_gcp only: OK")

    print("\nAll GCP tests passed.")

    print("\ntesting boundary_damping_lambda with in-workspace target (should not affect gradient)")
    # boundary_damping_lambda > 0 but IK succeeds, so right_used_boundary/left_used_boundary
    # remain False and res_norm_sq stays 0 — the damped path is never entered.
    fk_param_adiff = InitializeAutoDiff(fk_param).flatten()
    ik_nodamp = rby1_ik.compute_ik(fk_param_adiff, boundary_damping_lambda=0.0)
    ik_damp   = rby1_ik.compute_ik(fk_param_adiff, boundary_damping_lambda=1e3)
    grad_nodamp = ExtractGradient(ik_nodamp)
    grad_damp   = ExtractGradient(ik_damp)
    grad_diff = np.linalg.norm(grad_damp - grad_nodamp)
    print(f"  gradient difference (should be 0): {grad_diff:.2e}")
    assert grad_diff == 0.0, f"damping altered in-workspace gradient: diff={grad_diff}"
    print("  boundary_damping_lambda in-workspace: OK")
