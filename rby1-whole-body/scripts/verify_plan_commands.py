"""Verify the COMMANDS a plan will stream, not the sampled record of it.

verify_cached_plan.py checks ``meta["legs"][i]["q"]`` -- a (n,23) record sampled
at viz_hz for auditing. On hardware the robot client streamed ``steps[i]["cmds"]`` --
JointPositionCommands sampled at hz, built by a different call. Those are two
different artifacts from the same trajectory, and only the second one reaches the
robot. Nothing checks it.

Per plan this checks, on the commanded waypoints themselves:
  1. the three guarantees (collision, CoM, joint limits) in the same per-leg
     collision model plan_grid_point used;
  2. that the commands and the sampled record describe the same motion
     (max deviation of the command stream from the record, both resampled);
  3. seams -- the joint-space jump between the last waypoint of one step and the
     first of the next, which the robot executes as an unplanned move;
  4. implied joint velocity between consecutive waypoints against the plant's
     own velocity limits (a waypoint pair that needs more than the limit means
     the controller will silently stretch the leg);
  5. the base DOFs never move (follow_joint_trajectory_stiff rejects base motion);
  6. the client's OWN guard -- ``joint_limit_violations`` against the vendor's
     ROBOT_JOINT_LIMITS at its 1e-6 rad tolerance, which is what
     the robot client's joint-limit check ran before streaming anything.
     This is stricter than 4 and than verify_trajectory's 1e-3 rad tolerance,
     and deliberately so: since `3ed6df0` the planner's limits EQUAL the robot's
     on the binding joints, with no margin between them, so a B-spline overshoot
     of a few tens of microradians is enough for the client to refuse the plan
     outright. verify_trajectory passing is not sufficient to conclude a plan
     will execute.
"""
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pydrake.all import PiecewisePolynomial

from plan_grid import box_pose_for
from plan_format.commands import JOINT_LIMIT_EPS, joint_limit_violations
from rby1_planning import check_com_stability
from verify_cached_plan import _infra_empty_handed, _infra_with_box, _placed_box_pose


def cmds_to_q23(cmds, base_xyt):
    """(n,23) from the command stream: [base(3), torso(6), right(7), left(7)]."""
    out = np.empty((len(cmds), 23))
    for i, c in enumerate(cmds):
        out[i, 0:3] = base_xyt
        out[i, 3:9] = c["torso"].target_position
        out[i, 9:16] = c["right"].target_position
        out[i, 16:23] = c["left"].target_position
    return out


def _resample(q23, n):
    ts = np.linspace(0.0, 1.0, q23.shape[0])
    return PiecewisePolynomial.FirstOrderHold(ts, q23.T).vector_values(
        np.linspace(0.0, 1.0, n)).T


def check_plan(path):
    blob = pickle.load(open(path, "rb"))
    meta = blob["meta"]
    legs = {lg["name"]: lg for lg in meta["legs"]}
    bx, by = float(meta["bx"]), float(meta["by"])

    wb = [lg for lg in meta["legs"] if lg.get("with_box")]
    q_grasp23 = np.asarray(wb[0]["q"], float)[0]
    q_place_end23 = np.asarray(wb[-1]["q"], float)[-1]
    pb, cb, db, lb_, ee_body, _ = _infra_with_box(q_grasp23)
    pre, cpre, dpre, lpre = _infra_empty_handed(box_pose_for(bx, by), "box")
    X_placed = _placed_box_pose(pb, db, lb_, ee_body, q_place_end23)
    ppost, cpost, dpost, lpost = _infra_empty_handed(X_placed, "placed")

    base_xyt = np.asarray(meta["legs"][0]["q"], float)[0, :3]
    vmax = pre.GetVelocityUpperLimits()          # full-plant, 23 active DOF order
    q_lb, q_ub = pre.GetPositionLowerLimits(), pre.GetPositionUpperLimits()

    problems = []
    seen_box = False
    prev_end, prev_name = None, None
    n_wp_total = 0

    for step in blob["steps"]:
        if step["type"] != "trajectory":
            prev_name = f"{prev_name} -> [{step['name']}]" if prev_name else step["name"]
            continue
        name = step["name"]
        q23 = cmds_to_q23(step["cmds"], base_xyt)
        n_wp_total += q23.shape[0]
        lg = legs.get(name)

        if lg is not None and lg.get("with_box"):
            seen_box, plant, checker, layout = True, pb, cb, lb_
        elif seen_box:
            plant, checker, layout = ppost, cpost, lpost
        else:
            plant, checker, layout = pre, cpre, lpre

        ctx = plant.CreateDefaultContext()
        q_full = plant.GetPositions(ctx)

        worst_clear_ok = True
        worst_com = np.inf
        max_jl = 0.0
        for i, q in enumerate(q23):
            q_full[layout.plant_idxs] = q
            plant.SetPositions(ctx, q_full)
            qf = plant.GetPositions(ctx)
            if not checker.CheckConfigCollisionFree(qf):
                worst_clear_ok = False
                problems.append(f"{name}: commanded waypoint {i} is IN COLLISION")
                break
            if not check_com_stability(q, plant, ctx, inset=0.0):
                problems.append(f"{name}: commanded waypoint {i} CoM outside support polygon")
                break
            max_jl = max(max_jl, float(np.max(q_lb[layout.plant_idxs] - q)),
                         float(np.max(q - q_ub[layout.plant_idxs])))

        if max_jl > 1e-3:
            problems.append(f"{name}: commanded joint-limit excess {max_jl:.2e} rad "
                            f"(vs the planner's own model)")

        # 6. the client's guard, verbatim: this is what decides whether
        # the plan would ever have been streamed at all.
        for i, c in enumerate(step["cmds"]):
            for chain in ("torso", "right", "left"):
                for j, q, lo, hi in joint_limit_violations(chain, c[chain].target_position):
                    problems.append(
                        f"{name}: waypoint {i} {chain} joint {j} = {q:+.6f} outside the "
                        f"ROBOT's [{lo:+.6f}, {hi:+.6f}] by "
                        f"{max(lo - q, q - hi) * 1000:.3f} mrad -- "
                        f"the robot would REFUSE this plan")

        # 2. commands vs the audited record
        if lg is not None:
            rec = np.asarray(lg["q"], float)
            n = max(rec.shape[0], q23.shape[0])
            dev = np.max(np.abs(_resample(q23, n) - _resample(rec, n)))
            if dev > 5e-3:
                problems.append(f"{name}: commands deviate from the verified record "
                                f"by {dev:.2e} rad")

        # 3. seams
        if prev_end is not None:
            seam = float(np.max(np.abs(q23[0] - prev_end)))
            if seam > 1e-2:
                problems.append(f"seam {prev_name} -> {name}: {seam:.3e} rad jump")
        prev_end, prev_name = q23[-1], name

        # 4. implied velocity vs plant limits
        durs = np.array([c["right"].duration for c in step["cmds"]])
        dq = np.abs(np.diff(q23, axis=0))
        dt = np.maximum(durs[1:], 1e-9)[:, None]
        ratio = (dq / dt) / np.maximum(np.abs(vmax[layout.plant_idxs]), 1e-9)
        worst = float(np.max(ratio))
        if worst > 1.0:
            j = int(np.unravel_index(np.argmax(ratio), ratio.shape)[1])
            problems.append(f"{name}: implied velocity {worst:.2f}x the limit on DOF {j}")

        # 5. base motion
        if np.max(np.abs(q23[:, :3] - base_xyt)) > 1e-9:
            problems.append(f"{name}: commanded base moves")

    return problems, n_wp_total


def main():
    paths = sorted(sys.argv[1:])
    total_wp = 0
    bad = {}
    for p in paths:
        try:
            problems, n = check_plan(p)
        except Exception as e:
            bad[p] = [f"ERROR {type(e).__name__}: {e}"]
            print(f"{os.path.basename(p)}: ERROR {type(e).__name__}: {e}", flush=True)
            continue
        total_wp += n
        status = "PASS" if not problems else "FAIL"
        print(f"{os.path.basename(p)}: [{status}] {n} commanded waypoints", flush=True)
        for pr in problems:
            print(f"    - {pr}", flush=True)
        if problems:
            bad[p] = problems
    print(f"\n=== {len(paths) - len(bad)}/{len(paths)} plans pass on their COMMANDED "
          f"waypoints ({total_wp} waypoints checked)")


if __name__ == "__main__":
    main()
