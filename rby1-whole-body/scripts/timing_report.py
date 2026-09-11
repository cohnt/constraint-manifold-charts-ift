#!/usr/bin/env python
"""Runtime accounting for the planner, from the event log written by
``src/exps/timing_utils``.

Two views, because there are two honest answers and they are not the same number:

**Latency** (the headline for a paper). How long one plan took to produce, and
where that time went. Inside a point the planner forks -- grasp candidates are
screened concurrently, the reach legs overlap the winner's carry probe -- so the
seconds spent in concurrent branches cannot simply be added. This view walks each
point's tree and credits every instant of its wall time to exactly one category:
the work that was running alone, or, where several branches ran at once, the
branch that gated the parent's resumption. The categories therefore SUM TO THE
MEASURED WALL TIME, and the residual against it is printed rather than absorbed.

**Work.** Total processor-seconds across every process the point forked, kept and
discarded separately. This is what the point cost the machine, and it exceeds the
latency by exactly the concurrency. Discarded here means work that never reached
the shipped plan: losing grasp candidates, IK attempts rejected after solving,
failed GCP branches, re-seeded legs, TOPPRA rungs that failed.

    scripts/timing_report.py --cache-dir plans/grid_cache
    scripts/timing_report.py --log plans/grid_cache/timing/run_20260810-222505.jsonl
    scripts/timing_report.py --cache-dir A --per-point --csv out.csv

Only compare runs whose recorded configuration matches: the header event carries
--jobs, --leg-parallel, core count and git commit, and ``--check-config`` refuses
to aggregate logs that disagree on the first two.
"""
import argparse
import collections
import glob
import json
import os
import statistics
import sys


# ── Loading ───────────────────────────────────────────────────────────────────

def load_events(paths):
    """Return ``(records_by_id, run_headers)``.

    A record may appear twice: an "open" line written before the work starts and
    a full line written when it finishes. The close line wins. An id with only an
    open line belongs to a process that was killed mid-block (a losing grasp
    child, an OOM'd worker) -- it is kept, flagged ``killed``, and its end is
    inferred from its descendants, because that frame is precisely the discarded
    work this report exists to account for.
    """
    recs, runs = {}, []
    for path in paths:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue                # torn final line of a killed writer
                if r.get("cat") == "run":
                    runs.append(r.get("extra", {}))
                    continue
                prev = recs.get(r["id"])
                if prev is None or (prev.get("open") and not r.get("open")):
                    recs[r["id"]] = r
    for r in recs.values():
        r["killed"] = bool(r.get("open"))
    return recs, runs


def build_tree(recs):
    """``(children, roots)``; also closes killed frames over their descendants."""
    children = collections.defaultdict(list)
    for r in recs.values():
        children[r.get("parent")].append(r)

    # A killed frame has no t1. Its descendants do, so give it the latest end
    # among them (and its own start when it has none) -- an underestimate of what
    # it really cost, never an overestimate.
    def close(r):
        end = r.get("t1")
        kids = [close(k) for k in children[r["id"]]]
        if end is None:
            end = max([r["t0"]] + [e for e in kids if e is not None])
            r["t1"] = end
            r["inferred_end"] = True
        return end

    roots = [r for r in recs.values()
             if r.get("parent") is None or r["parent"] not in recs]
    for r in roots:
        close(r)
    for r in recs.values():                 # any subtree not reached above
        if r.get("t1") is None:
            close(r)
    return children, roots


# ── Interval algebra ──────────────────────────────────────────────────────────

def _union_len(intervals):
    total, cur_a, cur_b = 0.0, None, None
    for a, b in sorted(intervals):
        if cur_b is None or a > cur_b:
            total += 0.0 if cur_b is None else cur_b - cur_a
            cur_a, cur_b = a, b
        else:
            cur_b = max(cur_b, b)
    if cur_b is not None:
        total += cur_b - cur_a
    return total


def _intersect(xs, ys):
    out = []
    for a, b in xs:
        for c, d in ys:
            lo, hi = max(a, c), min(b, d)
            if hi > lo:
                out.append((lo, hi))
    return out


def _length(xs):
    return sum(b - a for a, b in xs)


# ── Latency view ──────────────────────────────────────────────────────────────

def attribute_latency(node, intervals, children, out, discarded=False):
    """Credit ``intervals`` of ``node``'s span to categories, recursively.

    Within a span, any instant covered by no child is this node's own work. An
    instant covered by several concurrent children is credited to the child that
    ENDS LAST among those running -- the one whose completion the parent was
    waiting on. That is the only child whose duration the parent's latency
    actually depends on; the others were free, and show up in the work view.
    """
    kids = sorted(children[node["id"]], key=lambda r: r["t0"])
    if not kids:
        out[(node["cat"], discarded or bool(node.get("discarded")))] += _length(intervals)
        return

    # Elementary segments of the node's span, each owned by one child or none.
    bounds = sorted({node["t0"], node["t1"]}
                    | {k["t0"] for k in kids} | {k["t1"] for k in kids})
    own, owned = [], collections.defaultdict(list)
    for a, b in zip(bounds, bounds[1:]):
        if b <= a:
            continue
        active = [k for k in kids if k["t0"] <= a and k["t1"] >= b]
        if not active:
            own.append((a, b))
        else:
            gate = max(active, key=lambda k: (k["t1"], k["t1"] - k["t0"], k["id"]))
            owned[gate["id"]].append((a, b))

    node_discarded = discarded or bool(node.get("discarded"))
    out[(node["cat"], node_discarded)] += _length(_intersect(own, intervals))
    by_id = {k["id"]: k for k in kids}
    for kid_id, segs in owned.items():
        part = _intersect(segs, intervals)
        if part:
            attribute_latency(by_id[kid_id], part, children, out, node_discarded)


# ── Work view ─────────────────────────────────────────────────────────────────

def work_totals(node, children, out, counts, discarded=False):
    """Self time (span minus the union of children's spans) per category."""
    kids = children[node["id"]]
    self_s = (node["t1"] - node["t0"]) - _union_len(
        [(k["t0"], k["t1"]) for k in kids])
    node_discarded = discarded or bool(node.get("discarded"))
    out[(node["cat"], node_discarded)] += max(self_s, 0.0)
    counts[node["cat"]] += 1
    for k in kids:
        work_totals(k, children, out, counts, node_discarded)


# ── Reporting ─────────────────────────────────────────────────────────────────

SOLVER_CATS = ("ik.solve", "trajopt.solve", "trajopt.solve_retry", "toppra.solve")


def _fmt_table(rows, headers):
    widths = [max(len(str(h)), max((len(str(r[i])) for r in rows), default=0))
              for i, h in enumerate(headers)]
    line = "  ".join(h.ljust(w) if i == 0 else h.rjust(w)
                     for i, (h, w) in enumerate(zip(headers, widths)))
    out = [line, "-" * len(line)]
    for r in rows:
        out.append("  ".join(str(c).ljust(w) if i == 0 else str(c).rjust(w)
                             for i, (c, w) in enumerate(zip(r, widths))))
    return "\n".join(out)


def analyse(recs, children, roots):
    """Per-point latency and work breakdowns."""
    points = sorted((r for r in recs.values() if r["cat"] == "point"),
                    key=lambda r: r["t0"])
    per_point = []
    for p in points:
        lat = collections.Counter()
        attribute_latency(p, [(p["t0"], p["t1"])], children, lat)
        work = collections.Counter()
        counts = collections.Counter()
        work_totals(p, children, work, counts)
        # Prefer the grid index recorded on the enclosing gcp_branch: "point 5"
        # is what the cache, the status manifest and the user all call it.
        parent = recs.get(p.get("parent"))
        idx = (parent or {}).get("extra", {}).get("index")
        label = p.get("label") if idx is None else f"{idx:02d}"
        per_point.append(dict(
            rec=p, label=label, wall=p["t1"] - p["t0"],
            latency=lat, work=work, counts=counts,
            ik_phase=ik_phase_totals(p, children),
            killed=bool(p.get("inferred_end"))))
    return per_point


def ik_phase_totals(node, children):
    """Seconds spent in IK draws, split search vs ranking.

    An `ik.attempt` is a whole draw (build + solve + the feasibility checks that
    may reject it), and draws never overlap within a process, so the record's own
    span is its cost. `phase` is "search" up to the first feasible candidate and
    "rank" after it -- ranking draws, all but one of which are thrown away by
    construction. Summed as work, not latency: draws inside a forked grasp child
    run concurrently with the parent's.
    """
    out = collections.Counter()
    stack = [node]
    while stack:
        r = stack.pop()
        if r["cat"] == "ik.attempt":
            out[r.get("extra", {}).get("phase", "?")] += r["t1"] - r["t0"]
        stack.extend(children[r["id"]])
    return out


def report(per_point, runs, top=18, per_point_detail=False, roll_up=False):
    if not per_point:
        print("no `point` records in the log -- was the run started with the "
              "timing log enabled?")
        return

    def key(cat):
        return cat.split(".")[0] if roll_up else cat

    walls = [p["wall"] for p in per_point]
    lat = collections.Counter()
    work = collections.Counter()
    counts = collections.Counter()
    # Per-category totals for ONE point, kept+discarded together, so the table can
    # report a max alongside the mean. The paper quotes both on this basis: the
    # unit is one plan, and the max is over points, not over solver calls (a
    # single call's max is a different and much smaller number).
    lat_by_point = collections.defaultdict(collections.Counter)
    for p in per_point:
        for (c, d), s in p["latency"].items():
            lat[(key(c), d)] += s
            lat_by_point[p["label"]][key(c)] += s
        for (c, d), s in p["work"].items():
            work[(key(c), d)] += s
        for c, k in p["counts"].items():        # roll up too, or --roll-up shows 0
            counts[key(c)] += k

    def per_point_max(cat):
        """Largest single point's total for ``cat`` -- the table's max column."""
        return max((tot[cat] for tot in lat_by_point.values()), default=0.0)

    if runs:
        r = runs[0]
        print(f"run        : {r.get('git_commit', '')[:10]}"
              f"{' +dirty' if r.get('git_dirty') else ''}  "
              f"jobs={r.get('jobs')} leg_parallel={r.get('leg_parallel')} "
              f"sequential={r.get('sequential')}  "
              f"host={r.get('hostname')} ({r.get('cpu_count')} cores)  "
              f"drake={r.get('drake_version') or '?'} "
              f"({os.path.basename(os.path.dirname(r.get('drake_path') or '')) or '?'})")
    n = len(per_point)
    print(f"points     : {n}"
          + (f"   ({sum(1 for p in per_point if p['killed'])} killed mid-plan)"
             if any(p["killed"] for p in per_point) else ""))
    print(f"latency    : mean {statistics.mean(walls):7.1f} s   "
          f"median {statistics.median(walls):7.1f} s   "
          f"range {min(walls):.1f}-{max(walls):.1f} s")
    lat_total = sum(lat.values())
    work_total = sum(work.values())
    print(f"work/CPU   : {work_total:.1f} s total across all processes "
          f"({work_total / max(lat_total, 1e-9):.2f}x the latency -- the excess "
          f"is concurrency)")
    print()

    # ── Latency ──
    print("LATENCY -- where each plan's wall time went (sums to the wall time)")
    rows = []
    cats = sorted({c for c, _ in lat}, key=lambda c: -(lat[(c, False)] + lat[(c, True)]))
    for c in cats[:top]:
        kept, disc = lat[(c, False)], lat[(c, True)]
        tot = kept + disc
        if tot < 0.05:
            continue
        rows.append((c, f"{tot / n:8.2f}", f"{per_point_max(c):8.2f}",
                     f"{100 * tot / lat_total:5.1f}%",
                     f"{kept / n:8.2f}", f"{disc / n:8.2f}"))
    rows.append(("TOTAL", f"{lat_total / n:8.2f}", f"{max(walls):8.2f}",
                 f"{100.0:5.1f}%",
                 f"{sum(v for (c, d), v in lat.items() if not d) / n:8.2f}",
                 f"{sum(v for (c, d), v in lat.items() if d) / n:8.2f}"))
    print(_fmt_table(rows, ("category", "mean s/pt", "max s/pt", "share",
                            "kept", "discarded")))
    resid = sum(walls) - lat_total
    print(f"reconciliation: attributed {lat_total:.2f} s vs measured wall "
          f"{sum(walls):.2f} s   residual {resid:+.3f} s "
          f"({100 * abs(resid) / max(sum(walls), 1e-9):.3f}%)")
    print()

    # ── Solver share ──
    solver_lat = sum(v for (c, _), v in lat.items() if c in SOLVER_CATS)
    solver_work = sum(v for (c, _), v in work.items() if c in SOLVER_CATS)
    if not roll_up:
        print(f"solver calls only ({', '.join(SOLVER_CATS)}):")
        print(f"    {100 * solver_lat / max(lat_total, 1e-9):.1f}% of latency, "
              f"{100 * solver_work / max(work_total, 1e-9):.1f}% of work -- "
              f"the rest is program construction, search, verification and "
              f"infrastructure.")
        print()

    ik_phase = collections.Counter()
    for p in per_point:
        ik_phase.update(p["ik_phase"])
    if sum(ik_phase.values()) > 0.05:
        srch, rank = ik_phase.get("search", 0.0), ik_phase.get("rank", 0.0)
        if srch + rank < 0.05:
            print(f"IK draws: {sum(ik_phase.values()) / n:.1f} s/point, phase "
                  f"unrecorded (log predates the search/rank split).")
        else:
            print(f"IK draws: {srch / n:.1f} s/point finding the first feasible "
                  f"candidate, {rank / n:.1f} s/point ranking further ones "
                  f"({100 * rank / (srch + rank):.0f}% of IK time; all but one "
                  f"ranking candidate is discarded by construction).")
        print()

    # ── Work ──
    print("WORK -- processor-seconds across every forked process")
    rows = []
    cats = sorted({c for c, _ in work},
                  key=lambda c: -(work[(c, False)] + work[(c, True)]))
    for c in cats[:top]:
        kept, disc = work[(c, False)], work[(c, True)]
        tot = kept + disc
        if tot < 0.05:
            continue
        rows.append((c, f"{tot / n:8.2f}", f"{100 * tot / work_total:5.1f}%",
                     f"{kept / n:8.2f}", f"{disc / n:8.2f}", counts[c]))
    kept_all = sum(v for (c, d), v in work.items() if not d)
    disc_all = sum(v for (c, d), v in work.items() if d)
    rows.append(("TOTAL", f"{work_total / n:8.2f}", f"{100.0:5.1f}%",
                 f"{kept_all / n:8.2f}", f"{disc_all / n:8.2f}",
                 sum(counts.values())))
    print(_fmt_table(rows, ("category", "s/point", "share", "kept",
                            "discarded", "n")))
    print(f"discarded work is {100 * disc_all / max(work_total, 1e-9):.1f}% of "
          f"all processor time spent.")

    if per_point_detail:
        print()
        print("PER POINT")
        rows = []
        for p in per_point:
            top_cat, top_s = "", 0.0
            for (c, _), s in p["latency"].items():
                if s > top_s:
                    top_cat, top_s = c, s
            w = sum(p["work"].values())
            rows.append((p["label"] or "?", f"{p['wall']:8.1f}", f"{w:8.1f}",
                         f"{w / max(p['wall'], 1e-9):5.2f}x",
                         f"{top_cat} ({top_s:.1f}s)"))
        print(_fmt_table(rows, ("point", "latency", "work", "ratio",
                                "largest latency category")))


def write_csv(per_point, path, roll_up=False):
    import csv

    def key(c):
        return c.split(".")[0] if roll_up else c

    cats = sorted({key(c) for p in per_point
                   for (c, _) in list(p["latency"]) + list(p["work"])})
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        # "point" is both the grid-point label and a category; disambiguate the
        # latter, or a duplicate header silently breaks csv.DictReader.
        w.writerow(["point", "view", "discarded", "total"]
                   + [f"cat:{c}" for c in cats])
        for p in per_point:
            for view in ("latency", "work"):
                for disc in (False, True):
                    agg = collections.Counter()
                    for (c, d), s in p[view].items():
                        if d == disc:
                            agg[key(c)] += s
                    w.writerow([p["label"], view, int(disc),
                                round(sum(agg.values()), 4)]
                               + [round(agg[c], 4) for c in cats])
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", default="plans/grid_cache",
                    help="read every log in <cache-dir>/timing/")
    ap.add_argument("--log", nargs="*", default=None,
                    help="explicit log file(s), instead of --cache-dir")
    ap.add_argument("--per-point", action="store_true",
                    help="also print one row per grid point")
    ap.add_argument("--roll-up", action="store_true",
                    help="aggregate categories to their prefix (ik, rrt, "
                         "trajopt, toppra, verify, infra, ...)")
    ap.add_argument("--top", type=int, default=18,
                    help="show this many categories per table (default 18)")
    ap.add_argument("--csv", default=None, metavar="PATH")
    ap.add_argument("--check-config", action="store_true",
                    help="refuse to aggregate logs recorded at different "
                         "--jobs / --leg-parallel settings")
    args = ap.parse_args()

    paths = args.log or sorted(glob.glob(
        os.path.join(args.cache_dir, "timing", "*.jsonl")))
    paths = [p for p in paths if os.path.exists(p)]
    if not paths:
        where = " ".join(args.log) if args.log else os.path.join(
            args.cache_dir, "timing")
        sys.exit(f"no event logs found in {where}. Plan with the timing log "
                 f"enabled (it is on by default; --no-timing-log disables it).")

    recs, runs = load_events(paths)
    if args.check_config and len({(r.get("jobs"), r.get("leg_parallel"))
                                  for r in runs}) > 1:
        sys.exit("logs were recorded at different --jobs/--leg-parallel "
                 "settings; their numbers are not comparable.")
    children, roots = build_tree(recs)
    per_point = analyse(recs, children, roots)
    report(per_point, runs, top=args.top, per_point_detail=args.per_point,
           roll_up=args.roll_up)
    if args.csv:
        write_csv(per_point, args.csv, roll_up=args.roll_up)


if __name__ == "__main__":
    main()
