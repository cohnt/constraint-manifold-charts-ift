"""Re-verify a cached plan pickle against this branch's guarantees.

Why this exists
---------------
A cached plan's ``meta["legs"]`` stores each leg as a sampled ``(n, 23)`` array of
active-DOF configurations plus a ``with_box`` flag. That is enough to re-run the
three guarantee checks on a plan *this branch never planned* -- in particular one
produced by a different branch, whose collision model or success gate may have
differed.

That is the whole point. A collaborator's branch reports 12/20 on the same grid,
but its ``robot_model_instances`` omits the grippers and the head, so gripper- and
head-vs-world pairs were never tested, and its success gate only asked whether the
trajopt and TOPPRA stages had *run* and whether the duration was positive. Its
successes and ours are therefore not comparable numbers, and the points where it
succeeds and we fail (14 and 16) are exactly the ones worth re-measuring: either a
valid lift/place path exists there and our search is missing it, or the path is one
our collision model rejects.

The infrastructure per leg mirrors plan_grid_point's, because otherwise the
comparison is unfair in the other direction -- a held box that is not filtered
against the fingers holding it reports penetration everywhere:

  * ``with_box=False`` legs before the place: open box as a static obstacle at the
    pick pose, open grippers.
  * ``with_box=True`` legs: box attached via HELD, table as obstacle, and exactly
    the grasp-pose ee<->box contacts filtered -- the same loop plan_grid_point runs
    at ``q_grasp``, seeded here from the first sample of the first with_box leg.
  * ``with_box=False`` legs after the place: placed box as a static obstacle, FK'd
    from the last configuration of the last with_box leg.

Usage
-----
    verify_cached_plan.py PKL [PKL ...] [--bx BX --by BY] [--dense-factor 4]

``--bx``/``--by`` override the box centre; by default it comes from the pickle's
own meta, which is where the planning harness recorded it.
"""

import argparse
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


import plan_format  # noqa: F401  -- installs the pre-rename module aliases the
                    # committed plan and execution pickles were written with
from pydrake.all import PiecewisePolynomial, RigidTransform, RollPitchYaw

from plan_grid import (
    HAND, HELD, OFFSET, RPY, TABLE, box_pose_for, open_box_walls,
)
from rby1_planning import (
    Rby1ActiveJointLayout, make_default_rby1_infrastructure, verify_trajectory,
)


def _traj_from_samples(q23: np.ndarray, duration: float):
    """First-order hold through the stored samples.

    Linear interpolation is the right reading of a sampled record: it is the
    tightest curve that passes through every configuration the plan actually
    committed to, and it cannot invent a detour between them. Checking it more
    densely than the samples is therefore a strictly stronger test than checking
    the samples alone, which is what --dense-factor buys.
    """
    q23 = np.asarray(q23, dtype=float)
    n = q23.shape[0]
    ts = np.linspace(0.0, max(float(duration), 1e-6), n)
    return PiecewisePolynomial.FirstOrderHold(ts, q23.T)


def _infra_empty_handed(X_W_box, prefix):
    plant, checker, diagram = make_default_rby1_infrastructure(
        obstacles=[TABLE] + open_box_walls(X_W_box, prefix=prefix), open_grippers=True)
    return plant, checker, diagram, Rby1ActiveJointLayout(plant)


def _infra_with_box(q_grasp23):
    """Box-attached infrastructure with the grasp-pose ee<->box contacts filtered.

    Same construction and same filter loop as plan_grid_point. The filter is what
    makes a held box checkable at all: the fingers are in contact with the wall
    they are holding by construction, so without it every sample reports
    penetration and the comparison is meaningless.
    """
    plant, checker, diagram = make_default_rby1_infrastructure(
        held_boxes=HELD, obstacles=TABLE)
    layout = Rby1ActiveJointLayout(plant)
    ctx = diagram.CreateDefaultContext()
    pctx = plant.GetMyContextFromRoot(ctx)
    q_full = plant.GetPositions(pctx)
    q_full[layout.plant_idxs] = q_grasp23
    plant.SetPositions(pctx, q_full)
    ee_body = plant.GetBodyByName(f"ee_{HAND}", plant.GetModelInstanceByName(f"{HAND}_arm"))
    qobj = plant.get_geometry_query_input_port().Eval(pctx)
    insp = qobj.inspector()
    n_filtered = 0
    for pp in qobj.ComputePointPairPenetration():
        bA = plant.GetBodyFromFrameId(insp.GetFrameId(pp.id_A))
        bB = plant.GetBodyFromFrameId(insp.GetFrameId(pp.id_B))
        if ee_body.index() in (bA.index(), bB.index()):
            other = bB if bA.index() == ee_body.index() else bA
            checker.SetCollisionFilteredBetween(ee_body.index(), other.index(), True)
            n_filtered += 1
    return plant, checker, diagram, layout, ee_body, n_filtered


def _placed_box_pose(plant, diagram, layout, ee_body, q_place_end23):
    ctx = diagram.CreateDefaultContext()
    pctx = plant.GetMyContextFromRoot(ctx)
    q_full = plant.GetPositions(pctx)
    q_full[layout.plant_idxs] = q_place_end23
    plant.SetPositions(pctx, q_full)
    return plant.EvalBodyPoseInWorld(pctx, ee_body) @ RigidTransform(RollPitchYaw(RPY), OFFSET)


def verify_plan(path, *, bx=None, by=None, dense_factor=4):
    with open(path, "rb") as f:
        blob = pickle.load(f)
    meta = blob.get("meta", {}) or {}
    legs = meta.get("legs") or []
    if not legs:
        print(f"{path}: no legs in meta -- nothing to verify")
        return None
    bx = float(meta["bx"]) if bx is None else float(bx)
    by = float(meta["by"]) if by is None else float(by)

    # The grasp config for the filter loop is the first sample of the first
    # with_box leg: that is the configuration the gripper closed in.
    with_box_legs = [lg for lg in legs if lg.get("with_box")]
    if not with_box_legs:
        print(f"{path}: no with_box legs -- cannot build the carried-box model")
        return None
    q_grasp23 = np.asarray(with_box_legs[0]["q"], dtype=float)[0]
    q_place_end23 = np.asarray(with_box_legs[-1]["q"], dtype=float)[-1]

    pb, cb, db, lb_, ee_body, n_filt = _infra_with_box(q_grasp23)
    pre, cpre, dpre, _ = _infra_empty_handed(box_pose_for(bx, by), "box")
    X_placed = _placed_box_pose(pb, db, lb_, ee_body, q_place_end23)
    ppost, cpost, dpost, _ = _infra_empty_handed(X_placed, "placed")

    print(f"\n=== {os.path.basename(path)}  (index={meta.get('index')}, "
          f"box=({bx:+.3f}, {by:+.3f}))")
    print(f"    grasp-pose ee<->box pairs filtered: {n_filt}")

    seen_box = False
    rows, all_ok = [], True
    for lg in legs:
        name = lg["name"]
        q23 = np.asarray(lg["q"], dtype=float)
        traj = _traj_from_samples(q23, lg["duration"])
        if lg.get("with_box"):
            seen_box, plant, checker, diagram = True, pb, cb, db
            model = "held box"
        elif seen_box:
            plant, checker, diagram = ppost, cpost, dpost
            model = "placed box obstacle"
        else:
            plant, checker, diagram = pre, cpre, dpre
            model = "pick box obstacle"

        # At the stored samples, then denser. The first answers "is the record
        # itself valid"; the second answers "does the interpolation between the
        # samples stay valid", which is the question a 30 Hz record cannot settle.
        rep = verify_trajectory(traj, plant, checker, diagram,
                                n_samples=q23.shape[0], required_clearance=0.0)
        rep_dense = verify_trajectory(traj, plant, checker, diagram,
                                      n_samples=max(2, dense_factor * q23.shape[0]),
                                      required_clearance=0.0)
        ok = rep.ok and rep_dense.ok
        all_ok = all_ok and ok
        rows.append((name, ok, rep, rep_dense, model))
        flag = "PASS" if ok else "FAIL"
        print(f"    [{flag}] {name:<14} ({model}, {q23.shape[0]} samples)")
        print(f"           clearance {rep.min_clearance*1000:+8.3f} mm "
              f"at t={rep.min_clearance_t:.2f}s  pair={rep.min_clearance_pair}")
        print(f"           CoM margin {rep.worst_stability_margin*1000:+8.3f} mm   "
              f"joint-limit excess {rep.max_joint_limit_violation:.3e} rad")
        if not rep.ok:
            print(f"           at-sample failures: {'; '.join(rep.failures)}")
        if rep.ok and not rep_dense.ok:
            print(f"           passes at its own samples but FAILS {dense_factor}x "
                  f"denser: {'; '.join(rep_dense.failures)} "
                  f"(clearance {rep_dense.min_clearance*1000:+.3f} mm, "
                  f"pair={rep_dense.min_clearance_pair})")

    print(f"    ---> {'ALL LEGS PASS' if all_ok else 'PLAN FAILS'}")
    return all_ok, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pkl", nargs="+", help="cached plan pickle(s) to verify")
    ap.add_argument("--bx", type=float, default=None)
    ap.add_argument("--by", type=float, default=None)
    ap.add_argument("--dense-factor", type=int, default=4,
                    help="also check this many times denser than the stored samples")
    args = ap.parse_args()

    verdicts = {}
    for p in args.pkl:
        try:
            out = verify_plan(p, bx=args.bx, by=args.by, dense_factor=args.dense_factor)
        except Exception as e:
            print(f"\n=== {os.path.basename(p)}: ERROR {type(e).__name__}: {e}")
            verdicts[p] = None
            continue
        verdicts[p] = None if out is None else out[0]

    print("\n=== summary")
    for p, v in verdicts.items():
        print(f"  {os.path.basename(p):<24} "
              f"{'PASS' if v else ('FAIL' if v is False else 'ERROR/SKIP')}")
    return 0 if all(v for v in verdicts.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
