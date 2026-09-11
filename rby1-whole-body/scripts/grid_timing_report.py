#!/usr/bin/env python
"""Per-stage and per-leg timing totals for a grid cache, for before/after comparison.

The provenance report answers "did every leg really come from trajopt"; this one
answers "where did the time go", which is the other half of judging an optimisation.
Both are needed: a change that halves trajopt but drops a leg to a BiRRT fallback has
not sped anything up, it has lowered the bar.

Reads ``meta["timings"]`` out of each cached plan -- so it costs nothing and describes
the run that actually produced the cache, rather than a fresh measurement that would
have its own contention.

    .venv/bin/python scripts/grid_timing_report.py --cache-dir plans/grid_cache
    .venv/bin/python scripts/grid_timing_report.py --cache-dir A --compare-to B

**These are wall-clock seconds summed across points that ran concurrently**, so the
total exceeds the run's own wall time and is sensitive to how many workers were used.
Only compare two caches planned at the same ``--jobs``: SNOPT's and the BiRRT's limits
are both wall-clock, so contention inflates per-solve times *and* eats solve budget.

**Superseded for latency claims** by ``scripts/timing_report.py``, which reads the
event log instead. The sums here cannot be latency -- a point's stages include
concurrent forked children -- and they omit every discarded branch (losing grasp
seeds, failed GCP branches, re-seeded legs). Use this for a quick stage-level A/B
of two caches; use timing_report.py for anything quoted as a runtime result. See
the folder README's runtime section.
"""
import argparse
import collections
import glob
import os
import pickle
import plan_format  # noqa: F401  -- installs the pre-rename module aliases the
                    # committed plan and execution pickles were written with


def load_timings(cache_dir):
    """{index: {leg: {stage: seconds}}} for every cached plan in ``cache_dir``."""
    out = {}
    for path in sorted(glob.glob(os.path.join(cache_dir, "point_[0-9][0-9].pkl"))):
        with open(path, "rb") as f:
            plan = pickle.load(f)
        meta = plan.get("meta", {})
        idx = meta.get("index", os.path.basename(path))
        out[idx] = meta.get("timings", {}) or {}
    return out


def load_walls(cache_dir):
    """{index: wall_s} where recorded (per-point wall latency; None entries skipped)."""
    out = {}
    for path in sorted(glob.glob(os.path.join(cache_dir, "point_[0-9][0-9].pkl"))):
        with open(path, "rb") as f:
            meta = pickle.load(f).get("meta", {})
        if meta.get("wall_s") is not None:
            out[meta.get("index", os.path.basename(path))] = float(meta["wall_s"])
    return out


def totals(timings):
    """(per-stage counter, per-leg -> per-stage counter, per-point total).

    ``None`` stage values (a stage a leg deliberately skipped, e.g. rrt_only
    legs record trajopt/toppra as None) are ignored rather than crashing the
    Counter arithmetic.
    """
    stage = collections.Counter()
    leg = collections.defaultdict(collections.Counter)
    point = {}
    for idx, legs in timings.items():
        t = 0.0
        for leg_name, stages in legs.items():
            for stage_name, secs in (stages or {}).items():
                if secs is None:
                    continue
                stage[stage_name] += secs
                leg[leg_name][stage_name] += secs
                t += secs
        point[idx] = t
    return stage, leg, point


STAGES = ("trajopt", "rrt", "ik", "toppra")


def _fmt_delta(new, old):
    if not old:
        return "     --"
    return f"{100.0 * (new - old) / old:+6.1f}%"


def report(cache_dir, compare_to=None):
    stage, leg, point = totals(load_timings(cache_dir))
    grand = sum(stage.values())
    if not grand:
        print(f"no timings found in {cache_dir}")
        return
    base = None
    if compare_to:
        base_stage, base_leg, base_point = totals(load_timings(compare_to))
        base = (base_stage, base_leg, base_point, sum(base_stage.values()))

    print(f"cache      : {cache_dir}   ({len(point)} points)")
    if compare_to:
        print(f"compared to: {compare_to}   ({len(base[2])} points)")
    print()

    hdr = f"{'stage':10s} {'seconds':>9} {'share':>7}"
    if base:
        hdr += f" {'before':>9} {'delta':>8}"
    print(hdr)
    print("-" * len(hdr))
    for name, secs in stage.most_common():
        line = f"{name:10s} {secs:9.1f} {100.0 * secs / grand:6.1f}%"
        if base:
            line += f" {base[0][name]:9.1f} {_fmt_delta(secs, base[0][name]):>8}"
        print(line)
    line = f"{'TOTAL':10s} {grand:9.1f} {100.0:6.1f}%"
    if base:
        line += f" {base[3]:9.1f} {_fmt_delta(grand, base[3]):>8}"
    print(line)

    print()
    hdr = f"{'leg':16s}" + "".join(f"{s:>9}" for s in STAGES) + f"{'total':>9}"
    if base:
        hdr += f" {'delta':>8}"
    print(hdr)
    print("-" * len(hdr))
    for leg_name, counter in sorted(leg.items(), key=lambda kv: -sum(kv[1].values())):
        tot = sum(counter.values())
        line = f"{leg_name:16s}" + "".join(f"{counter[s]:9.1f}" for s in STAGES) + f"{tot:9.1f}"
        if base:
            line += f" {_fmt_delta(tot, sum(base[1][leg_name].values())):>8}"
        print(line)

    print()
    slow = sorted(point.items(), key=lambda kv: -kv[1])
    med = sorted(point.values())[len(point) // 2]
    print(f"per point: mean {grand / len(point):.1f}s  median {med:.1f}s  "
          f"range {min(point.values()):.1f}-{max(point.values()):.1f}s")
    print("slowest   : " + ", ".join(f"{i} {t:.0f}s" for i, t in slow[:5]))
    walls = load_walls(cache_dir)
    if walls:
        # Stage sums above are CPU cost; with parallel legs they exceed the
        # point's wall latency. This is what a robot would actually wait.
        w = sorted(walls.values())
        print(f"wall/point: mean {sum(w) / len(w):.1f}s  median {w[len(w) // 2]:.1f}s  "
              f"range {w[0]:.1f}-{w[-1]:.1f}s   (n={len(w)} with wall_s recorded)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", default="plans/grid_cache")
    ap.add_argument("--compare-to", default=None,
                    help="second cache to diff against; must have been planned at "
                         "the same --jobs for the comparison to mean anything.")
    args = ap.parse_args()
    report(args.cache_dir, args.compare_to)


if __name__ == "__main__":
    main()
