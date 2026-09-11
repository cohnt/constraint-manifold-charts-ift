#!/usr/bin/env python3
"""Planning cost per leg, and per stage within a leg, from the event log.

``scripts/timing_report.py`` attributes a point's wall time to solver *categories*
(ik.solve, rrt.birrt, trajopt.solve, ...) and reconciles exactly against the measured
wall clock. It does not say which of the six legs that time went to, which is the split
the paper's six-step description asks for: four unconstrained legs planned by IK + BiRRT
+ trajopt, and two constrained legs planned in the reduced 14-D space.

READ THIS BEFORE QUOTING A PER-LEG WALL NUMBER. The run had ``leg_parallel=True``: the
legs of one point were planned concurrently in forked children. Their wall intervals
therefore OVERLAP and do not sum to the point's latency -- a point costs roughly the max
over its legs, not the sum. Two columns are reported for that reason:

  wall   t1 - t0 of the leg event. What that leg took on its own. Comparable between
         legs; NOT summable into a point total.
  cpu    cpu_self + cpu_children. Processor-seconds, which ARE additive, and are what a
         sequential-per-leg implementation would have to pay.

Usage:
    venv/bin/python scripts/planning_time_by_leg.py
    venv/bin/python scripts/planning_time_by_leg.py --csv out.csv
"""

from __future__ import annotations

import argparse
import collections
import csv
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from timing_report import load_events  # noqa: E402

DEFAULT_LOG = REPO / "plans" / "grid_cache" / "timing"

UNCONSTRAINED = ("reach_approach", "reach_descend", "home_retreat", "home")
CONSTRAINED = ("lift", "place")
LEG_ORDER = ("reach_approach", "reach_descend", "lift", "place",
             "home_retreat", "home")

# A leg's DIRECT children are its stages. Attribution must stop there: cpu_self on a
# stage event already includes everything nested inside it, so walking the whole subtree
# counts stage.trajopt once and its trajopt.probe/solve children a second time.
#
# Two categories of work are charged OUTSIDE the legs, which is why the per-leg totals
# here are much smaller than a point's latency, and why two columns read zero:
#
#   IK       all 3394 ik.solve events in this log sit under grasp.probe / grasp.ik /
#            stage.ik / point. Grasp selection and goal IK run before the legs fork, so
#            no IK is charged to a leg -- ik.solve can be the largest single latency
#            category for a point while contributing nothing to any leg.
#   BiRRT    on the constrained legs. lift and place have no stage.rrt child at all;
#            they go straight to trajopt from a path the grasp screening already found.
#            Their BiRRT cost is in the carry_probe_lift / carry_probe_place probes,
#            which is where a candidate grasp is tested for liftability.
#
# So this table answers "what did planning THIS leg cost, given a grasp", not "what did
# the point cost". scripts/timing_report.py is the whole-point accounting.
STAGE_LABEL = {
    "stage.rrt": "BiRRT+shortcut",
    "stage.trajopt": "trajopt",
    "stage.toppra": "TOPPRA",
    "verify.leg": "verify",
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", default=None,
                    help="event log .jsonl (default: newest under plans/grid_cache/timing)")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    log = args.log
    if log is None:
        hits = sorted(Path(DEFAULT_LOG).glob("run_*.jsonl"))
        if not hits:
            print(f"no event log under {DEFAULT_LOG}")
            return 1
        log = str(hits[-1])
    print(f"event log: {log}\n")

    recs, _runs = load_events([log])
    children = collections.defaultdict(list)
    for rid, r in recs.items():
        if r.get("parent"):
            children[r["parent"]].append(rid)

    legs = [r for r in recs.values() if r.get("cat") == "leg"]
    if not legs:
        print("no leg events in this log")
        return 1

    # Only legs that were kept: a discarded/killed child planned work that was thrown
    # away, and charging it to a leg would double-count against the shipped plan.
    per_leg = collections.defaultdict(list)
    per_leg_stage = collections.defaultdict(lambda: collections.defaultdict(float))

    for leg in legs:
        name = leg.get("label")
        if name is None or leg.get("killed"):
            continue
        t1 = leg.get("t1")
        if t1 is None:
            continue
        wall = t1 - leg["t0"]
        cpu = (leg.get("cpu_self") or 0.0) + (leg.get("cpu_children") or 0.0)
        per_leg[name].append((wall, cpu))
        for did in children.get(leg["id"], []):
            d = recs[did]
            stage = STAGE_LABEL.get(d.get("cat"))
            if stage is None or d.get("t1") is None:
                continue
            per_leg_stage[name][stage] += (d.get("cpu_self") or 0.0)

    print("PER LEG -- one row per leg name, aggregated over the points that planned it")
    print("wall intervals OVERLAP between legs of a point (leg_parallel=True); "
          "cpu is additive.\n")
    print(f"  {'leg':<16} {'kind':<13} {'n':>3} {'wall mean':>10} {'wall med':>9} "
          f"{'wall max':>9} {'cpu mean':>9}")
    rows = []
    for name in LEG_ORDER:
        if name not in per_leg:
            continue
        w = np.array([x[0] for x in per_leg[name]])
        c = np.array([x[1] for x in per_leg[name]])
        kind = "constrained" if name in CONSTRAINED else "unconstrained"
        print(f"  {name:<16} {kind:<13} {len(w):>3} {w.mean():>9.2f}s "
              f"{np.median(w):>8.2f}s {w.max():>8.2f}s {c.mean():>8.2f}s")
        rows.append({"leg": name, "kind": kind, "n": len(w),
                     "wall_mean_s": w.mean(), "wall_median_s": float(np.median(w)),
                     "wall_max_s": w.max(), "cpu_mean_s": c.mean()})

    print()
    print("BY KIND -- mean per leg instance")
    for kind, names in (("unconstrained", UNCONSTRAINED), ("constrained", CONSTRAINED)):
        w = np.array([x[0] for n in names for x in per_leg.get(n, [])])
        c = np.array([x[1] for n in names for x in per_leg.get(n, [])])
        if not len(w):
            continue
        print(f"  {kind:<15} n={len(w):<4} wall mean {w.mean():>6.2f}s  "
              f"median {np.median(w):>6.2f}s   cpu mean {c.mean():>6.2f}s")

    print("\nSTAGE BREAKDOWN -- processor-seconds per leg instance, by algorithm stage")
    print("  No IK column: grasp selection and goal IK run before the legs fork, so no "
          "IK\n  time is charged to any leg (see the note in this file's header).")
    stages = ["BiRRT+shortcut", "trajopt", "TOPPRA", "verify"]
    print(f"  {'leg':<16}" + "".join(f"{s:>10}" for s in stages) + f"{'total':>10}")
    for name in LEG_ORDER:
        if name not in per_leg_stage:
            continue
        n = len(per_leg[name])
        vals = [per_leg_stage[name].get(s, 0.0) / n for s in stages]
        print(f"  {name:<16}" + "".join(f"{v:>9.2f}s" for v in vals)
              + f"{sum(vals):>9.2f}s")
        if args.csv:
            for row in rows:
                if row["leg"] == name:
                    row.update({f"cpu_{s}_s": v for s, v in zip(stages, vals)})

    print("\nStage totals should track the leg's cpu mean closely; any gap is leg-level "
          "\n  setup outside a stage event.")

    if args.csv and rows:
        keys = sorted({k for r in rows for k in r})
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
