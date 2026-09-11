#!/usr/bin/env python3
"""End-effector constraint violation, at three levels, for the paper's results table.

The constrained legs (``lift``, ``place``) carry the box with the gripper-to-gripper
transform frozen at the grasp configuration. This measures how far

    X_left_right(q) = X_left(q)^-1 @ X_right(q)

drifts from its value at that configuration, for three different things that could be
called "the trajectory":

  planned    meta["legs"][name]["q"], the (N, 23) samples the plan was verified on
  commanded  steps[i]["cmds"], the 20 Hz waypoints actually streamed to the robot
  measured   logs["joint_position"], what the robot's encoders read back

WHICH KINEMATIC MODEL. Reported twice, and the two disagree by ~1e-6 for a reason that
is not the planner's:

  ikfast  the model the parameterization is *defined* in. constrained_plan maps a 14-D
          state to joint angles through IKFast at every trajectory evaluation, so the
          constraint is eliminated by construction and the planned residual here is
          machine precision (~2e-15). A nonzero number in this column is a real defect.
  drake   the URDF used for collision checking, verification and rendering. Its shoulder
          mount is rpy=0.349065850 (20 deg); the IKFast codegen hard-codes the same mount
          as atan2(0.342021230003561, 0.93969222526679) = 20.0000663 deg. That 1.156e-6
          rad difference puts a ~3e-7 m floor under any Drake-evaluated claim about an
          IKFast-produced configuration, and the two arms' errors compose into ~1e-6 m
          here. It is a modelling discrepancy, not motion.

Both arms mount to the same torso link, so X_left_right needs the 14 arm joints alone --
no torso, no base, no odometry. That is what lets the measured stream be evaluated in the
IKFast model too, and why base pose never enters any of this.

Usage:
    venv/bin/python scripts/ee_constraint_report.py
    venv/bin/python scripts/ee_constraint_report.py --indices 0 5
    venv/bin/python scripts/ee_constraint_report.py --json out.json --csv out.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))


import plan_format  # noqa: F401  -- installs the pre-rename module aliases the
                    # committed plan and execution pickles were written with
from rainbow_left_arm_ik import get_fk as _left_get_fk    # noqa: E402
from rainbow_right_arm_ik import get_fk as _right_get_fk  # noqa: E402

DEFAULT_CACHE = REPO / "plans" / "grid_cache"
DEFAULT_RESULTS = REPO / "results" / "2026-08-11"

CONSTRAINED_LEGS = ("lift", "place")

# Slices of the 23-D active configuration. Mirrors Rby1ActiveJointLayout, restated here
# so the primary (IKFast) path does not need Drake at all.
TORSO = slice(3, 9)
RIGHT_ARM = slice(9, 16)
LEFT_ARM = slice(16, 23)

# Slices of the robot's 24-D state vector (utilities.parse_state_vector).
STATE_TORSO = slice(2, 8)
STATE_RIGHT = slice(8, 15)
STATE_LEFT = slice(15, 22)


# ------------------------------------------------------------------ kinematics

def _rt(eetrans, eerot):
    """IKFast's (translation, row-major rotation) as a 4x4."""
    T = np.eye(4)
    T[:3, :3] = np.asarray(eerot, dtype=float).reshape(3, 3)
    T[:3, 3] = np.asarray(eetrans, dtype=float).ravel()
    return T


def _inv(T):
    Ti = np.eye(4)
    R = T[:3, :3]
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ T[:3, 3]
    return Ti


def rel_ikfast(q23):
    """X_left_right in the IKFast model, from arm joints only."""
    Tr = _rt(*_right_get_fk([float(v) for v in q23[RIGHT_ARM]]))
    Tl = _rt(*_left_get_fk([float(v) for v in q23[LEFT_ARM]]))
    return _inv(Tl) @ Tr


class DrakeRel:
    """X_left_right in the Drake model. Imported lazily -- Drake is the cross-check."""

    def __init__(self):
        import rby1_planning as P
        from rby1_planning import _Setup

        plant, cc, diagram = P.make_default_rby1_infrastructure()[:3]
        self.setup = _Setup.build(plant, cc, diagram)
        self.plant = plant
        self.ctx = plant.CreateDefaultContext()
        self.world = plant.world_frame()
        self.fr = plant.GetFrameByName("ee_right")
        self.fl = plant.GetFrameByName("ee_left")

    def __call__(self, q23):
        self.setup.set_q23_into(self.ctx, np.asarray(q23, dtype=float))
        Tr = self.plant.CalcRelativeTransform(self.ctx, self.world, self.fr)
        Tl = self.plant.CalcRelativeTransform(self.ctx, self.world, self.fl)
        return _inv(Tl.GetAsMatrix4()) @ Tr.GetAsMatrix4()


def deviation(T, T0):
    """(translation metres, rotation radians) of T relative to the frozen T0.

    The angle is atan2(||skew part||, (trace-1)/2), not arccos of the trace: near
    identity the arccos form loses half its digits to cancellation and bottoms out
    around 1e-8 rad, which is well above the residuals this script has to resolve.
    """
    dt = float(np.linalg.norm(T[:3, 3] - T0[:3, 3]))
    R = T0[:3, :3].T @ T[:3, :3]
    v = 0.5 * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return dt, float(np.arctan2(np.linalg.norm(v), (np.trace(R) - 1.0) / 2.0))


# ---------------------------------------------------------------- assembly

def q23_from_cmd(cmd):
    q = np.zeros(23)
    q[TORSO] = cmd["torso"].target_position
    q[RIGHT_ARM] = cmd["right"].target_position
    q[LEFT_ARM] = cmd["left"].target_position
    return q


def q23_from_state(row):
    q = np.zeros(23)
    q[TORSO] = row[STATE_TORSO]
    q[RIGHT_ARM] = row[STATE_RIGHT]
    q[LEFT_ARM] = row[STATE_LEFT]
    return q


def _as_logs(entry):
    logs = entry.get("logs")
    if logs is None:
        return []
    return logs if isinstance(logs, list) else [logs]


def measured_q23(record, leg):
    """Measured configurations for a leg, windowed to its command send window.

    Outside that window the robot is settling or idle between RPCs, which is not part of
    the commanded motion and would report a violation nothing asked for.
    """
    for step in record.get("steps", []):
        if step.get("name") != leg:
            continue
        for log in _as_logs(step):
            jp = log.get("joint_position")
            ts = log.get("timestamp")
            ct = log.get("command_timestamps")
            if jp is None or ts is None or ct is None or len(ct) == 0:
                continue
            jp, ts, ct = np.asarray(jp), np.asarray(ts), np.asarray(ct)
            keep = (ts >= ct[0]) & (ts <= ct[-1])
            return [q23_from_state(r) for r in jp[keep]], float(np.median(np.diff(ct)))
    return [], float("nan")


def find_record(results_dir, index):
    hits = sorted(Path(results_dir).glob(f"point_{index:02d}_exec_*.pkl"))
    return hits[-1] if hits else None


# ---------------------------------------------------------------- per point

def analyse_point(index, cache, results_dir, drake_rel):
    plan_path = Path(cache) / f"point_{index:02d}.pkl"
    if not plan_path.exists():
        return None
    with open(plan_path, "rb") as f:
        plan = pickle.load(f)
    meta = plan["meta"]
    legs = {leg["name"]: leg for leg in meta["legs"]}
    steps = {s.get("name"): s for s in plan["steps"]}

    # Two references, because they answer different questions.
    #
    #   leg    each leg against its OWN first planned sample. This is the constraint
    #          constrained_plan actually enforces: it re-freezes the transform at the
    #          configuration each leg starts from.
    #   grasp  everything against the grasp configuration -- the end-to-end quantity,
    #          which additionally carries the step taken at the lift->place seam.
    #
    # The grasp configuration comes from legs["lift"]["q"][0], not meta["q_grasp"]:
    # the latter is rounded to 6 decimals and injects a few 1e-7 m of its own.
    q_grasp = np.asarray(legs["lift"]["q"][0], dtype=float)
    models_fn = [("ikfast", rel_ikfast)] + ([] if drake_rel is None
                                            else [("drake", drake_rel)])
    ref_grasp = {m: fn(q_grasp) for m, fn in models_fn}

    record = None
    rec_path = find_record(results_dir, index)
    if rec_path is not None:
        with open(rec_path, "rb") as f:
            record = pickle.load(f)

    out = {"index": index, "plan": str(plan_path),
           "record": str(rec_path) if rec_path else None, "legs": {}}

    for leg in CONSTRAINED_LEGS:
        if leg not in legs:
            continue
        series = {
            "planned": [np.asarray(q, dtype=float) for q in legs[leg]["q"]],
            "commanded": [q23_from_cmd(c) for c in steps[leg].get("cmds", [])],
        }
        if record is not None:
            series["measured"], _ = measured_q23(record, leg)

        # This leg's own frozen transform, from its first planned sample.
        q_leg0 = np.asarray(legs[leg]["q"][0], dtype=float)
        ref_leg = {m: fn(q_leg0) for m, fn in models_fn}

        entry = {}
        for level, qs in series.items():
            if not len(qs):
                continue
            per_model = {}
            for model, fn in models_fn:
                devs_leg = [deviation(fn(q), ref_leg[model]) for q in qs]
                devs_gr = [deviation(fn(q), ref_grasp[model]) for q in qs]
                cell = {"n": int(len(qs))}
                for tag, devs in (("leg", devs_leg), ("grasp", devs_gr)):
                    t = np.array([d[0] for d in devs])
                    r = np.array([d[1] for d in devs])
                    cell[tag] = {
                        "max_trans_m": float(t.max()), "mean_trans_m": float(t.mean()),
                        "max_rot_rad": float(r.max()), "mean_rot_rad": float(r.mean()),
                    }
                per_model[model] = cell
            entry[level] = per_model
        out["legs"][leg] = entry

    # The lift->place seam. constrained_plan re-freezes the transform for each leg by
    # Drake FK (_make_lift_helpers) and then realises it through IKFast IK, so the
    # transform place holds is offset from the one IKFast reads at the end of lift by
    # exactly the two models' disagreement. A one-time step, not a drift.
    if "place" in legs and "lift" in legs:
        q_lift_end = np.asarray(legs["lift"]["q"][-1], dtype=float)
        q_place_0 = np.asarray(legs["place"]["q"][0], dtype=float)
        seam = {"max_joint_rad": float(np.abs(q_place_0 - q_lift_end).max())}
        for model, fn in models_fn:
            dt, dr = deviation(fn(q_place_0), fn(q_lift_end))
            seam[model] = {"trans_m": dt, "rot_rad": dr}
            # The same two configurations, measured across the model boundary.
            if drake_rel is not None:
                dmt, dmr = deviation(rel_ikfast(q_lift_end), drake_rel(q_lift_end))
                seam["model_disagreement"] = {"trans_m": dmt, "rot_rad": dmr}
        out["seam"] = seam

    # The grasp handoff: reach_descend's last sample against the configuration lift
    # freezes. A step, not a drift, and larger than either -- so it is reported apart.
    if "reach_descend" in legs:
        q_end = np.asarray(legs["reach_descend"]["q"][-1], dtype=float)
        out["handoff"] = {"max_joint_rad": float(np.abs(q_end - q_grasp).max())}
        for model, fn in models_fn:
            dt, dr = deviation(fn(q_end), ref_grasp[model])
            out["handoff"][model] = {"trans_m": dt, "rot_rad": dr}

    return out


# ------------------------------------------------------------------ report

def pool(points, level, model, ref, field):
    vals = []
    for p in points:
        for leg in CONSTRAINED_LEGS:
            e = p["legs"].get(leg, {}).get(level, {}).get(model)
            if e:
                vals.append(e[ref][field])
    return np.array(vals) if vals else np.array([])


def fmt(v, unit):
    if unit == "mm":
        v *= 1e3
    elif unit == "mrad":
        v *= 1e3
    if v == 0:
        return "0"
    if abs(v) < 1e-4 or abs(v) >= 1e5:
        return f"{v:.2e}"
    return f"{v:.4f}"


REF_BLURB = {
    "leg": "each leg against its own first planned sample -- the constraint the planner "
           "enforces",
    "grasp": "everything against the grasp configuration -- end-to-end, includes the "
             "lift->place seam",
}


def report(points, models):
    print(f"\nEE constraint violation -- {len(points)} points, "
          f"legs {'+'.join(CONSTRAINED_LEGS)} pooled")
    print("Drift of X_left_right = X_left^-1 @ X_right from its frozen value.\n")

    for ref in ("leg", "grasp"):
        print(f"  REFERENCE '{ref}': {REF_BLURB[ref]}")
        for model in models:
            print(f"\n    [{model} model]")
            print(f"      {'level':<10} {'max trans':>12} {'mean trans':>12} "
                  f"{'max rot':>12} {'mean rot':>12}   samples")
            print(f"      {'':<10} {'(mm)':>12} {'(mm)':>12} "
                  f"{'(mrad)':>12} {'(mrad)':>12}")
            for level in ("planned", "commanded", "measured"):
                mt = pool(points, level, model, ref, "max_trans_m")
                if not len(mt):
                    continue
                at = pool(points, level, model, ref, "mean_trans_m")
                mr = pool(points, level, model, ref, "max_rot_rad")
                ar = pool(points, level, model, ref, "mean_rot_rad")
                n = sum(int(p["legs"].get(lg, {}).get(level, {}).get(model, {})
                            .get("n", 0))
                        for p in points for lg in CONSTRAINED_LEGS)
                print(f"      {level:<10} {fmt(mt.max(),'mm'):>12} "
                      f"{fmt(at.mean(),'mm'):>12} {fmt(mr.max(),'mrad'):>12} "
                      f"{fmt(ar.mean(),'mrad'):>12}   {n}")
        print()

    # Per leg, since place is materially worse than lift under load.
    print("  [per leg, max over points, reference 'leg']")
    header = f"    {'leg':<8} {'level':<10}"
    for model in models:
        header += f" {model + ' mm':>12} {model + ' mrad':>13}"
    print(header)
    for leg in CONSTRAINED_LEGS:
        for level in ("planned", "commanded", "measured"):
            cells, any_cell = "", False
            for model in models:
                vals = [p["legs"].get(leg, {}).get(level, {}).get(model)
                        for p in points]
                vals = [v["leg"] for v in vals if v]
                if not vals:
                    cells += f" {'-':>12} {'-':>13}"
                    continue
                any_cell = True
                cells += (f" {fmt(max(v['max_trans_m'] for v in vals), 'mm'):>12}"
                          f" {fmt(max(v['max_rot_rad'] for v in vals), 'mrad'):>13}")
            if any_cell:
                print(f"    {leg:<8} {level:<10}{cells}")
    print()

    def _steps(key, title):
        rows = [p[key] for p in points if key in p]
        if not rows:
            return
        print(f"  [{title}]")
        jm = np.array([h["max_joint_rad"] for h in rows])
        print(f"    joint step        max {jm.max():.2e} rad   "
              f"median {np.median(jm):.2e} rad")
        for model in models:
            t = np.array([h[model]["trans_m"] for h in rows if model in h])
            r = np.array([h[model]["rot_rad"] for h in rows if model in h])
            if len(t):
                print(f"    {model:<8} step      max {fmt(t.max(),'mm')} mm / "
                      f"{fmt(r.max(),'mrad')} mrad   "
                      f"median {fmt(float(np.median(t)),'mm')} mm")
        dis = [h["model_disagreement"] for h in rows if "model_disagreement" in h]
        if dis:
            t = np.array([d["trans_m"] for d in dis])
            print(f"    ikfast-vs-drake disagreement at the same configuration: "
                  f"median {fmt(float(np.median(t)),'mm')} mm "
                  f"(this is what the step is)")
        print()

    _steps("handoff", "grasp handoff: reach_descend final sample -> what lift freezes")
    _steps("seam", "lift->place seam: lift final sample -> what place freezes")


def write_csv(points, path, models):
    rows = []
    for p in points:
        for leg, levels in p["legs"].items():
            for level, per_model in levels.items():
                for model, e in per_model.items():
                    for ref in ("leg", "grasp"):
                        rows.append({"index": p["index"], "leg": leg, "level": level,
                                     "model": model, "reference": ref,
                                     "n": e["n"], **e[ref]})
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", default=str(DEFAULT_CACHE))
    ap.add_argument("--results-dir", default=str(DEFAULT_RESULTS))
    ap.add_argument("--indices", type=int, nargs="*", default=None)
    ap.add_argument("--no-drake", action="store_true",
                    help="skip the Drake cross-check column (fast, no pydrake import)")
    ap.add_argument("--json", default=None)
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    indices = args.indices
    if indices is None:
        indices = sorted(int(p.stem.split("_")[1])
                         for p in Path(args.cache).glob("point_[0-9][0-9].pkl"))

    drake_rel = None if args.no_drake else DrakeRel()
    models = ["ikfast"] + ([] if args.no_drake else ["drake"])

    points = []
    for i in indices:
        p = analyse_point(i, args.cache, args.results_dir, drake_rel)
        if p is None:
            print(f"  point {i:02d}: no cached plan, skipped")
            continue
        points.append(p)

    if not points:
        print("no points analysed")
        return 1

    report(points, models)

    n_exec = sum(1 for p in points if p["record"])
    print(f"  {len(points)} plans, {n_exec} with an execution record.")
    if n_exec < len(points):
        print("  Points without a record contribute planned/commanded rows only.")

    if args.json:
        with open(args.json, "w") as f:
            json.dump({"points": points}, f, indent=2)
        print(f"  wrote {args.json}")
    if args.csv:
        write_csv(points, args.csv, models)
    return 0


if __name__ == "__main__":
    sys.exit(main())
