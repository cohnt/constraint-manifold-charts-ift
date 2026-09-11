"""Report the three guarantees for every cached grid plan.

Reads the per-leg evidence stored in each cached plan's meta (written by
plan_grid's _verify_leg) and prints one row per leg plus a
per-point and overall summary:

  * collision avoidance  -- worst signed distance, and the body pair achieving it
  * static stability     -- worst CoM margin to the support polygon
  * joint limits         -- worst excess beyond a limit

This reports what each plan recorded at planning time. To re-derive the numbers
instead of trusting the cached verdict -- the honest check if the planner or the
models have changed since -- use scripts/verify_cached_plan.py, which replays the
stored trajectories through the checks again.

Usage:
    python scripts/grid_guarantee_report.py
    python scripts/verify_cached_plan.py plans/grid_cache/point_*.pkl
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import plan_grid as E
from plan_format.plan_io import load_plan

LEG_ORDER = ["reach_approach", "reach_descend", "lift", "place",
             "home_retreat", "home"]


def load_all(cache_dir):
    out = {}
    if not os.path.isdir(cache_dir):
        return out
    for f in sorted(os.listdir(cache_dir)):
        if not (f.startswith("point_") and f.endswith(".pkl")) or "debug" in f:
            continue
        idx = int(f[len("point_"):-len(".pkl")])
        try:
            _, meta = load_plan(os.path.join(cache_dir, f))
        except Exception as e:
            print(f"[{idx:2d}] could not load: {type(e).__name__}: {e}")
            continue
        out[idx] = meta
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", default=E.DEFAULT_CACHE_DIR)
    ap.add_argument("--grid-path", default=E.DEFAULT_GRID_PATH)
    args = ap.parse_args()

    metas = load_all(args.cache_dir)
    status_path = os.path.join(args.cache_dir, E.STATUS_FILENAME)
    status = {}
    if os.path.exists(status_path):
        with open(status_path) as f:
            status = json.load(f)

    n_grid = 20
    try:
        grid = np.load(args.grid_path)
        key = "grid_points" if "grid_points" in grid else list(grid.keys())[0]
        n_grid = len(grid[key])
    except Exception:
        pass

    print("=" * 100)
    print("GRID GUARANTEE REPORT")
    print("=" * 100)
    print(f"cache: {args.cache_dir}")
    print(f"grid points: {n_grid}   cached plans: {len(metas)}")

    worst_clear = np.inf
    worst_com = np.inf
    worst_jl = 0.0
    all_ok = True
    per_point = []

    for idx in sorted(metas):
        meta = metas[idx]
        ev = meta.get("evidence") or {}
        bx, by = meta.get("bx"), meta.get("by")
        print(f"\n[{idx:2d}] box=({bx:+.3f}, {by:+.3f})")
        if not ev:
            print("     (no evidence recorded -- cached before verification existed)")
            continue
        names = [n for n in LEG_ORDER if n in ev] + \
                [n for n in ev if n not in LEG_ORDER]
        pt_clear, pt_com, pt_jl, pt_dur, pt_ok = np.inf, np.inf, 0.0, 0.0, True
        print(f"     {'leg':16s} {'dur(s)':>7} {'clearance':>11} {'CoM(mm)':>9} "
              f"{'jl(rad)':>9}  tightest pair")
        for nm in names:
            e = ev[nm]
            pt_clear = min(pt_clear, e["min_clearance_m"])
            pt_com = min(pt_com, e["com_margin_m"])
            pt_jl = max(pt_jl, e["joint_limit_excess_rad"])
            pt_dur += e["duration"]
            pt_ok &= e["ok"]
            print(f"     {nm:16s} {e['duration']:7.2f} "
                  f"{e['min_clearance_m']*1000:8.2f}mm {e['com_margin_m']*1000:9.1f} "
                  f"{e['joint_limit_excess_rad']:9.1e}  {e['min_clearance_pair']}")
        print(f"     {'WORST':16s} {pt_dur:7.2f} {pt_clear*1000:8.2f}mm "
              f"{pt_com*1000:9.1f} {pt_jl:9.1e}  -> {'PASS' if pt_ok else 'FAIL'}")
        worst_clear = min(worst_clear, pt_clear)
        worst_com = min(worst_com, pt_com)
        worst_jl = max(worst_jl, pt_jl)
        all_ok &= pt_ok
        per_point.append((idx, pt_ok, pt_clear, pt_com, pt_jl, pt_dur))

    failed = [k for k, v in status.items()
              if isinstance(v, dict) and v.get("status") == "failed"]

    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"points with a cached, verified plan : {len(per_point)} / {n_grid}")
    if failed:
        print(f"points that failed to plan         : {len(failed)} "
              f"({', '.join(sorted(failed, key=lambda s: int(s) if s.isdigit() else 0))})")
        for k in sorted(failed, key=lambda s: int(s) if s.isdigit() else 0):
            v = status[k]
            print(f"    [{k:>2}] stage={v.get('stage')}  {str(v.get('detail'))[:90]}")
    if per_point:
        print(f"\nworst across all cached plans:")
        print(f"  collision avoidance : {worst_clear*1000:+8.2f} mm "
              f"({'no penetration' if worst_clear > 0 else 'PENETRATION'})")
        print(f"  static stability    : {worst_com*1000:+8.2f} mm CoM margin "
              f"({'inside support polygon' if worst_com > 0 else 'OUTSIDE'})")
        print(f"  joint limits        : {worst_jl:8.2e} rad excess")
        print(f"\nall legs of all cached points pass: {all_ok}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
