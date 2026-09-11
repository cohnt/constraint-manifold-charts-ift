"""Survey grasp-IK candidate generation schemes against downstream ground truth.

For each (grid point, variant, seed) this draws ONE grasp candidate with that
variant's IK objective, records cheap posture features (`grasp_posture_score`
among them), and -- with ``--probe`` -- runs the same `_carry_is_plannable`
lift+place probe the pipeline uses, capturing plannability plus the probe's
lift path length/duration. Two questions, one dataset:

1. Score validation: does `grasp_posture_score` rank candidates the way the
   probe's measured lift length / duration does? (Run the `none` variant with
   ``--probe``.)
2. Generation-scheme comparison: which IK objective produces feasible,
   low-scoring, *diverse* candidate pools? (Run every variant, probe the
   shortlist.) Decision rule: a variant must not fall below `none`'s per-point
   IK success / probe pass counts (hard floor -- the clearance-ranking
   experiment recorded in `_search_ik`'s docstring is what happens when a
   preference is allowed to override plannability), then lower per-point
   minimum score and higher pool diversity win.

Each unit forks a child (same pattern as the pipeline's grasp screen), so a
native IK crash costs one candidate, not the sweep. Work is capped at --jobs
concurrent children.

Usage:
    .venv/bin/python scripts/grasp_candidate_survey.py --variants none --probe
    .venv/bin/python scripts/grasp_candidate_survey.py \
        --variants none midpoint ready random random_gcp symmetry \
        --out scratch/grasp_survey.json
"""

import argparse
import contextlib
import io
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import plan_grid as E
from rby1_opt_ik import Rby1ProblemOptions
from rby1_planning import (
    Rby1ActiveJointLayout,
    _all_gcps,
    _default_ik_options,
    _random_q_in_gcp,
    _Setup,
    grasp_and_standoff,
    grasp_posture_score,
    make_default_rby1_infrastructure,
)

VARIANTS = ("none", "midpoint", "ready", "random", "random_gcp", "symmetry",
            "ready_symmetry")


def _variant_options(variant, mult, point_index, seed, setup, rng_seed_base):
    """Build grasp_ik_options for one candidate draw, or None for the control.

    The random targets are seeded by (point, seed) so the sweep is reproducible
    and each candidate of a point pulls toward a *different* posture -- the
    diversity mechanism: it perturbs the optimum, not just the IPOPT start.
    """
    if variant == "none":
        return None
    opts = _default_ik_options()
    if variant in ("midpoint", "ready", "random", "random_gcp", "ready_symmetry"):
        opts.impose_joint_centering_cost = True
        opts.sqrt_centering_cost = True
        opts.joint_centering_cost_multiplier = mult
    if variant == "midpoint":
        opts.joint_nominal = True          # existing limits-midpoint target
    elif variant in ("ready", "ready_symmetry"):
        opts.posture_nominal = E.Q_READY.copy()
    elif variant == "random":
        rng = np.random.default_rng(1000 * point_index + seed)
        target = rng.uniform(setup.q_lb, setup.q_ub)
        target[:3] = 0.0
        opts.posture_nominal = target
    elif variant == "random_gcp":
        rng = np.random.default_rng(1000 * point_index + seed)
        target = _random_q_in_gcp(_all_gcps()[1], setup, rng)
        if target is None:
            target = np.zeros(23)
        opts.posture_nominal = target
    if variant in ("symmetry", "ready_symmetry"):
        opts.arm_symmetry_cost_multiplier = mult
    return opts


def _survey_child(conn, unit):
    """Fork target: one candidate draw (+ optional probe). Never raises."""
    point, bx, by, variant, mult, seed, probe, gcp_index = (
        unit["point"], unit["bx"], unit["by"], unit["variant"], unit["mult"],
        unit["seed"], unit["probe"], unit["gcp_index"])
    out = dict(point=point, variant=variant, mult=mult, seed=seed,
               ik_ok=False, ik_s=None, why="", score=None, features=None,
               probe_ok=None, probe_s=None, lift_len23=None, lift_dur_s=None,
               place_len14=None, q_grasp=None)
    t0 = time.perf_counter()
    try:
        with contextlib.redirect_stdout(io.StringIO()), \
             contextlib.redirect_stderr(io.StringIO()):
            right_target, left_target = E.grasp_targets_for_box(bx, by)
            plant, checker, diagram = make_default_rby1_infrastructure(
                obstacles=[E.TABLE] + E.open_box_walls(E.box_pose_for(bx, by)),
                open_grippers=True)
            setup = _Setup.build(plant, checker, diagram)
            opts = _variant_options(variant, mult, point, seed, setup, 0)
            t_ik = time.perf_counter()
            q_standoff, q_grasp = grasp_and_standoff(
                plant, checker, diagram, right_target, left_target,
                standoff=E.STANDOFF_M, rng_seed=seed, gcp_index=gcp_index,
                grasp_ik_options=opts)
            out["ik_s"] = round(time.perf_counter() - t_ik, 2)
            out["ik_ok"] = True
            q = np.asarray(q_grasp)
            m = np.minimum(q - setup.q_lb, setup.q_ub - q)
            out["score"] = round(grasp_posture_score(q), 4)
            out["features"] = dict(
                asym_shoulder=round(float(abs(q[10] - q[17])), 4),
                dist_ready=round(float(np.linalg.norm(q[3:23] - E.Q_READY[3:23])), 4),
                jl_margin=round(float(min(m[9:16].min(), m[16:23].min())), 4),
                torso=[round(float(v), 4) for v in q[3:9]],
            )
            out["q_grasp"] = [round(float(v), 5) for v in q]
            if probe:
                plant_box, checker_box, diagram_box = make_default_rby1_infrastructure(
                    held_boxes=E.HELD, obstacles=E.TABLE)
                layout_box = Rby1ActiveJointLayout(plant_box)
                ee_body = plant_box.GetBodyByName(
                    f"ee_{E.HAND}", plant_box.GetModelInstanceByName(f"{E.HAND}_arm"))
                toppra = dict(toppra_acceleration_scale=0.1, toppra_velocity_scale=0.1)
                t_p = time.perf_counter()
                # 4-tuple since the probe-reuse work (868ccc9): the trailing
                # payload is the RRT-level lift/place the real legs warm-start
                # from. The survey has no legs to warm-start, so it is dropped --
                # but it must still be unpacked. Unpacking 3 here raised
                # "too many values to unpack", which the outer handler recorded
                # in `why` while leaving probe_ok=None, and the progress line
                # only prints `why` when the *IK* fails. So every probe silently
                # produced no ground truth from 868ccc9 until this was fixed.
                # lift_ik_candidates matches the pipeline's own screen
                # (`_probe` in plan_grid_point). Left at the planner default the
                # probe runs at a lower goal-IK fidelity than the thing this
                # dataset is supposed to predict, so a candidate could be
                # recorded unplannable that the real screen would accept.
                ok, why, diag, _payload = E._carry_is_plannable(
                    plant_box, checker_box, diagram_box, layout_box, ee_body,
                    q, toppra,
                    lift_ik_candidates=E.GRASP_PROBE_IK_CANDIDATES)
                out.update(probe_ok=bool(ok), probe_s=round(time.perf_counter() - t_p, 2),
                           why=why, **(diag or {}))
    except Exception as e:
        out["why"] = f"{type(e).__name__}: {str(e)[:120]}"
    out["total_s"] = round(time.perf_counter() - t0, 2)
    try:
        conn.send(out)
    except Exception:
        pass
    conn.close()


def _run_units(units, jobs, budget_s):
    """Run children in a bounded pool; every unit gets a result dict."""
    results, pending, launched = [], [], 0
    def _launch(u):
        parent, child = mp.Pipe(duplex=False)
        p = mp.Process(target=_survey_child, args=(child, u))
        p.start()
        child.close()
        return (u, p, parent, time.time())
    while launched < len(units) or pending:
        while launched < len(units) and len(pending) < jobs:
            pending.append(_launch(units[launched]))
            launched += 1
        u, p, conn, t0 = pending.pop(0)
        remaining = max(1.0, budget_s - (time.time() - t0))
        got = None
        if conn.poll(remaining):
            try:
                got = conn.recv()
            except EOFError:
                got = None
        conn.close()
        p.join(timeout=10)
        if p.is_alive():
            p.kill()
            p.join()
        results.append(got or dict(point=u["point"], variant=u["variant"],
                                   mult=u["mult"], seed=u["seed"], ik_ok=False,
                                   why="survey child produced no result"))
        done = len(results)
        r = results[-1]
        tag = (f"[{done}/{len(units)}] pt{r['point']:2d} {r['variant']}"
               f" m={r.get('mult')} seed={r['seed']}")
        if r.get("ik_ok"):
            if r.get("probe_ok") is None:
                # Either --probe was off, or the probe itself threw. The latter
                # is silent otherwise (`why` is only printed on IK failure) and
                # yields a row with no ground truth at all, so say so loudly.
                probe_txt = ("" if not u.get("probe") else
                             f"  PROBE DID NOT RUN ({r.get('why', '')[:70]})")
            else:
                probe_txt = (f" probe={'PASS' if r['probe_ok'] else 'fail'}"
                             f" lift_len={r.get('lift_len23')}")
            print(f"{tag}: score={r.get('score')}{probe_txt}")
        else:
            print(f"{tag}: IK FAILED ({r.get('why', '')[:60]})")
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--indices", type=int, nargs="*", default=None)
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--variants", nargs="+", default=["none"], choices=VARIANTS)
    ap.add_argument("--multipliers", type=float, nargs="+", default=[1.0])
    ap.add_argument("--gcp-index", type=int, default=1)
    ap.add_argument("--jobs", type=int, default=10)
    ap.add_argument("--probe", action="store_true",
                    help="run the lift+place plannability probe per candidate")
    ap.add_argument("--grid-path", default=E.DEFAULT_GRID_PATH)
    ap.add_argument("--out", default="scratch/grasp_survey.json")
    args = ap.parse_args()

    grid = np.load(args.grid_path)
    pts = grid["grid_points"]
    indices = args.indices if args.indices else list(range(len(pts)))

    units = []
    for point in indices:
        bx, by = float(pts[point][0]), float(pts[point][1])
        for variant in args.variants:
            for mult in (args.multipliers if variant != "none" else [0.0]):
                for seed in range(args.seeds):
                    units.append(dict(point=point, bx=bx, by=by, variant=variant,
                                      mult=mult, seed=seed, probe=args.probe,
                                      gcp_index=args.gcp_index))
    budget = (E.GRASP_PROBE_TIMEOUT_S * 4 + 120.0) if args.probe else 120.0
    print(f"{len(units)} units, --jobs {args.jobs}, per-child budget {budget:.0f}s")
    t0 = time.perf_counter()
    results = _run_units(units, args.jobs, budget)
    print(f"survey wall: {time.perf_counter() - t0:.0f}s")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    existing = []
    if os.path.exists(args.out):
        existing = json.load(open(args.out))
        print(f"appending to {len(existing)} existing rows in {args.out}")
    json.dump(existing + results, open(args.out, "w"))
    print(f"{len(existing) + len(results)} rows -> {args.out}")


if __name__ == "__main__":
    main()
