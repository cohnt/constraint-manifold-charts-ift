"""Precompute error timelines for all 20 grid points.

Loads plan pickles and execution records, aligns planned/commanded/measured
trajectories, computes constraint violation (SE(3) distance of relative
gripper transform), joint position error, and EE position error.

Saves results to video/error_data.pkl for use by render_segment1.py and
render_segment2.py.

Usage:
    .venv/bin/python scripts/video/compute_errors.py
    .venv/bin/python scripts/video/compute_errors.py --points 0 5  # subset
"""

import argparse
import glob
import os
import pickle
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src"))


import plan_format  # noqa: F401  -- installs the pre-rename module aliases the
                    # committed plan and execution pickles were written with
from pydrake.math import RigidTransform, RotationMatrix, RollPitchYaw

from rby1_analytic_ik import Rby1IK
from rainbow_right_arm_ik import get_fk as right_get_fk
from rainbow_left_arm_ik import get_fk as left_get_fk

BODY = slice(2, 22)
CHAINS = {"torso": slice(0, 6), "right": slice(6, 13), "left": slice(13, 20)}

CONSTRAINED_LEGS = {"lift", "place"}

PLAN_DIR = os.path.join(REPO, "plans", "grid_cache")
RESULTS_DIR = os.path.join(REPO, "results", "2026-08-11")
OUTPUT = os.path.join(REPO, "video", "error_data.pkl")


def load_plan(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_exec(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def find_exec_record(point_idx):
    pattern = os.path.join(RESULTS_DIR, f"point_{point_idx:02d}_exec_*.pkl")
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No execution record for point {point_idx}")
    return matches[-1]


def commanded_from_step(step):
    """Extract (N, 20) commanded body joints from a plan step."""
    cmds = step["cmds"]
    return np.array([
        np.concatenate([
            c["torso"].target_position,
            c["right"].target_position,
            c["left"].target_position,
        ]) for c in cmds
    ])


def align(actual_t, command_t):
    """Index of the waypoint in force at each actual sample."""
    return np.searchsorted(command_t, actual_t, side="right") - 1


def body_to_q23(body_20):
    """Prepend zero base to get 23-DOF config for FK."""
    return np.concatenate([[0.0, 0.0, 0.0], body_20])


def fk_to_transforms(fk_out):
    """Extract right and left EE RigidTransforms from compute_fk output."""
    r_xyz = fk_out[9:12]
    r_rpy = fk_out[12:15]
    l_xyz = fk_out[16:19]
    l_rpy = fk_out[19:22]
    X_right = RigidTransform(RollPitchYaw(r_rpy), r_xyz)
    X_left = RigidTransform(RollPitchYaw(l_rpy), l_xyz)
    return X_right, X_left


def se3_distance(X_desired, X_actual):
    """SE(3) distance with equal position (m) and orientation (rad) weighting."""
    X_err = X_desired.inverse() @ X_actual
    dp = X_err.translation()
    dtheta = X_err.rotation().IsNearlyIdentity(np.pi)
    angle = X_err.rotation().ToAngleAxis().angle()
    return float(np.sqrt(np.dot(dp, dp) + angle * angle))


# ── The constraint metric: IKFast FK, position and rotation kept apart ──────
#
# The gripper-to-gripper transform MUST be evaluated with IKFast's own get_fk,
# not with Drake FK. The IKFast extensions and the Drake URDF describe robots
# that differ by 1.156e-6 rad at the shoulder mount (the generated .cpp rounds
# the +/-20 degree mount to 20.0000663 deg), so a Drake-evaluated transform for a
# configuration IKFast produced carries a spurious ~3e-7 m floor that is not real
# error. Measured through get_fk the constraint holds to 2e-15 m; measured
# through Drake it looks like ~3e-7 m of "violation" that does not exist.
# Do not "fix" the URDF or regenerate
# the codegen -- that would invalidate the committed plan cache and the
# 2026-08-11 hardware run.
#
# Both arms mount to link_torso_5, so X_left_right needs the 14 arm joints only:
# no torso, no base. That also makes it directly applicable to measured state.
#
# Position (m) and rotation (rad) are returned separately. Summing them into one
# scalar, as se3_distance above does, adds metres to radians -- fine as a rough
# planner cost, wrong for a number shown on screen.


def _mat(arm7, get_fk):
    t, R = get_fk([float(v) for v in arm7])
    M = np.eye(4)
    M[:3, :3] = np.asarray(R, dtype=float)
    M[:3, 3] = np.asarray(t, dtype=float)
    return M


def _inv(M):
    Mi = np.eye(4)
    Mi[:3, :3] = M[:3, :3].T
    Mi[:3, 3] = -M[:3, :3].T @ M[:3, 3]
    return Mi


def x_left_right(right7, left7):
    """X_left_right via IKFast FK, from the 14 arm joints alone."""
    return _inv(_mat(left7, left_get_fk)) @ _mat(right7, right_get_fk)


def constraint_error(X_ref, X_now):
    """(position error in m, rotation error in rad) between two transforms.

    The angle is atan2(||skew part||, (trace-1)/2), not arccos of the trace:
    near identity the arccos form loses half its digits to cancellation and
    bottoms out around 1e-8 rad -- for the *planned* legs, whose true residual
    is ~1e-15 rad (IKFast is the parameterization's own model, so the
    constraint is machine-precision by construction), arccos rounds the whole
    trace to exactly 1.0 and returns a bit-exact 0.0. See
    scripts/ee_constraint_report.py:deviation, which uses this same form and
    documents the same cancellation.
    """
    E = _inv(X_ref) @ X_now
    dp = float(np.linalg.norm(E[:3, 3]))
    R = E[:3, :3]
    v = 0.5 * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return dp, float(np.arctan2(np.linalg.norm(v), (np.trace(R) - 1.0) / 2.0))


def compute_point_errors(point_idx, ik):
    """Compute all error timelines for one grid point."""
    plan_path = os.path.join(PLAN_DIR, f"point_{point_idx:02d}.pkl")
    exec_path = find_exec_record(point_idx)

    plan = load_plan(plan_path)
    record = load_exec(exec_path)

    # Reference the constraint to the first sample of the carry itself, NOT to
    # meta["q_grasp"] -- that field is rounded to 6 decimals when the plan is
    # written (scripts/plan_grid.py), which injects its own
    # spurious 3-4e-7 m offset. meta["legs"]["lift"]["q"][0] is unrounded.
    carry_legs = [l for l in plan["meta"]["legs"] if l["name"] in CONSTRAINED_LEGS]
    if not carry_legs:
        raise RuntimeError(f"point {point_idx} has no constrained leg to reference")
    q_carry_start = np.asarray(carry_legs[0]["q"])[0]
    X_LR_fallback = x_left_right(q_carry_start[9:16], q_carry_start[16:23])

    plan_steps = plan["steps"]
    exec_steps = record["steps"]

    t_offset = None
    results = []

    for plan_step, exec_step in zip(plan_steps, exec_steps):
        if plan_step["type"] != "trajectory":
            continue

        name = plan_step["name"]
        cmds_body = commanded_from_step(plan_step)
        n_cmds = len(cmds_body)
        dt = float(plan_step["cmds"][0]["right"].duration)

        logs = exec_step["logs"]
        actual_pos = logs["joint_position"][:, BODY]
        actual_t = logs["timestamp"]
        cmd_t = logs["command_timestamps"]

        if t_offset is None:
            t_offset = cmd_t[0]

        t_actual_rel = actual_t - t_offset
        t_cmd_rel = cmd_t - t_offset

        idx = align(actual_t, cmd_t)
        valid = (idx >= 0) & (idx < n_cmds)

        n_actual = len(actual_pos)
        step_result = {
            "name": name,
            "is_constrained": name in CONSTRAINED_LEGS,
            "t_actual": t_actual_rel,
            "t_cmd": t_cmd_rel,
            # position (m) and rotation (rad), via IKFast FK -- see the note
            # above constraint_error(). The old combined-scalar
            # constraint_violation_* keys were Drake-evaluated and are gone on
            # purpose: at this magnitude they reported model round-off.
            "cv_pos_planned": np.zeros(n_cmds),
            "cv_rot_planned": np.zeros(n_cmds),
            "cv_pos_commanded": np.zeros(n_cmds),
            "cv_rot_commanded": np.zeros(n_cmds),
            "cv_pos_measured": np.zeros(n_actual),
            "cv_rot_measured": np.zeros(n_actual),
            "joint_error": np.zeros(n_actual),
            "ee_error_right": np.zeros(n_actual),
            "ee_error_left": np.zeros(n_actual),
        }

        planned_t = np.arange(n_cmds) * dt
        leg_match = [l for l in plan["meta"]["legs"] if l["name"] == name]
        if leg_match:
            leg_q = leg_match[0]["q"]
            planned_t = np.linspace(0, leg_match[0]["duration"], len(leg_q))

        # ── Each constrained leg is referenced to its OWN first sample ───────
        #
        # Every constrained leg freezes its own gripper-to-gripper transform,
        # and the planner freezes it with Drake's CalcRelativeTransform while
        # generating the trajectory through IKFast (_make_lift_helpers in
        # src/rby1_planning.py). Each freeze therefore crosses the Drake/IKFast
        # model boundary once, which the two models' 1.156e-6 rad shoulder-mount
        # disagreement turns into a ~6e-7 m offset between what "lift" holds and
        # what "place" holds.
        #
        # Referencing every leg to lift's start made that offset appear as a
        # step in the *planned* constraint violation at the lift->place seam:
        # lift sat at 2e-15 m and place at a perfectly flat 5.9e-7 m, with zero
        # variation across the leg. A DC offset with no structure is the
        # signature of a reference-frame mismatch, not of planner error, and on
        # the log axis it read as eight orders of magnitude of "violation" that
        # the planner does not actually commit.
        #
        # Referencing each leg to the transform it actually holds is the honest
        # measurement, and it is what the caption claims: within a leg the
        # constraint is eliminated by construction. The seam step itself is
        # real, but it is a modelling artifact and belongs in the notes, not
        # plotted as trajectory error.
        if leg_match and name in CONSTRAINED_LEGS:
            q0 = np.asarray(leg_match[0]["q"])[0]
            X_LR_desired = x_left_right(q0[9:16], q0[16:23])
        else:
            X_LR_desired = X_LR_fallback

        # commanded: the 14 arm joints of each waypoint actually sent
        for k in range(n_cmds):
            b = cmds_body[k]
            dp, dr = constraint_error(X_LR_desired, x_left_right(b[6:13], b[13:20]))
            step_result["cv_pos_commanded"][k] = dp
            step_result["cv_rot_commanded"][k] = dr

        # planned: the trajectory itself, sampled denser than the commands
        if leg_match:
            leg_q = np.asarray(leg_match[0]["q"])
            pos = np.zeros(len(leg_q))
            rot = np.zeros(len(leg_q))
            for k in range(len(leg_q)):
                dp, dr = constraint_error(
                    X_LR_desired, x_left_right(leg_q[k][9:16], leg_q[k][16:23])
                )
                pos[k], rot[k] = dp, dr
            step_result["cv_pos_planned"] = pos
            step_result["cv_rot_planned"] = rot
            step_result["t_planned"] = planned_t
        else:
            step_result["t_planned"] = np.arange(n_cmds) * dt

        for i in range(n_actual):
            a = actual_pos[i]
            dp, dr = constraint_error(X_LR_desired, x_left_right(a[6:13], a[13:20]))
            step_result["cv_pos_measured"][i] = dp
            step_result["cv_rot_measured"][i] = dr

            if valid[i]:
                q23_meas = body_to_q23(a)
                fk_meas = ik.compute_fk(q23_meas)
                X_r_m, X_l_m = fk_to_transforms(fk_meas)
                cmd_body_k = cmds_body[idx[i]]
                step_result["joint_error"][i] = float(
                    np.max(np.abs(actual_pos[i] - cmd_body_k))
                )

                q23_ck = body_to_q23(cmd_body_k)
                fk_ck = ik.compute_fk(q23_ck)
                X_r_c, X_l_c = fk_to_transforms(fk_ck)

                step_result["ee_error_right"][i] = float(
                    np.linalg.norm(X_r_m.translation() - X_r_c.translation())
                )
                step_result["ee_error_left"][i] = float(
                    np.linalg.norm(X_l_m.translation() - X_l_c.translation())
                )

        results.append(step_result)

    total_duration = t_actual_rel[-1] if len(t_actual_rel) else 0.0

    return {
        "steps": results,
        "total_duration": float(total_duration),
        "wall_s": record.get("wall_s", 0.0),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--points", type=int, nargs="*", default=list(range(20)),
        help="Which points to compute (default: all 20)",
    )
    ap.add_argument("--output", default=OUTPUT)
    args = ap.parse_args()

    print("Initializing Rby1IK (loads Drake plant)...")
    ik = Rby1IK()
    print("Done.")

    all_errors = {}
    for pt in args.points:
        print(f"[{pt:2d}/19] Computing errors for point {pt}...")
        try:
            all_errors[pt] = compute_point_errors(pt, ik)
            n_steps = len(all_errors[pt]["steps"])
            dur = all_errors[pt]["total_duration"]
            print(f"         {n_steps} trajectory steps, {dur:.1f}s total")
        except Exception as e:
            print(f"         ERROR: {e}")
            import traceback
            traceback.print_exc()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(all_errors, f)
    print(f"\nWrote {args.output} ({len(all_errors)} points)")


if __name__ == "__main__":
    main()
