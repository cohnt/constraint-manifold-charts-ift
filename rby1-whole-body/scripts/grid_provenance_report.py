"""Report where each cached grid leg's trajectory actually came from.

grid_guarantee_report.py answers "is the cached plan valid". This answers the
question that used to be unanswerable: **was the trajectory optimised, or is it a
raw BiRRT+shortcut path that trajopt failed on?**

Both used to look identical in the cache. A leg's captured stages contain a
``trajopt`` entry only when trajopt produced the shipped trajectory, but that key
was set *before* the dense acceptance check and popped only when the solver itself
reported failure -- so a leg whose optimised curve was rejected for penetrating
between its constrained samples kept the key and read as optimised. Auditing the
run before that was fixed: 9 of 15 "successes" had a lift or place built from the
fallback, and that count was a lower bound.

Also reports, per point:

  * ``gcp_index`` -- which grasp GCP branch planned it. constrained_plan locks the
    lift and place to the branch the grasp was solved in, so this is the
    constrained manifold those two legs were searched in. It used to be hardcoded.
  * the worker log's own verdict on each trajopt solve, parsed out of the captured
    stdout: whether the initial guess was feasible, and what each solver rung did.
    That distinction -- started infeasible vs started feasible and lost it -- is the
    fork in the diagnosis, and it only exists on disk because the log is persisted.

Usage:
    .venv/bin/python scripts/grid_provenance_report.py
    .venv/bin/python scripts/grid_provenance_report.py --verbose
"""

import argparse
import json
import os
import pickle
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import plan_grid as E
from plan_format.plan_io import load_plan

LEG_ORDER = ["reach_approach", "reach_descend", "lift", "place",
             "home_retreat", "home"]

# Legs planned with do_trajopt=False, for which BiRRT+shortcut is the intended result
# rather than a downgrade, so they read as "n/a" instead of "FALLBACK".
#
# Read from the harness rather than hardcoded: reach_descend and home_retreat used to
# belong here unconditionally, and once SHORT_LEG_TRAJOPT turned trajopt on for them a
# hardcoded set silently reported two genuinely optimised legs as "n/a" -- which
# under-counts exactly the thing this report exists to count.
NO_TRAJOPT_BY_DESIGN = (
    set() if getattr(E, "SHORT_LEG_TRAJOPT", False)
    else {"reach_descend", "home_retreat"}
)

# The label carries the guess's control-point multiplicity since
# _select_guess_multiplicity started reporting per-multiplicity feasibility, so this
# has to tolerate "reach mult=1" as well as a bare "reach". Both are counted, which is
# what makes "guess_infeasible" readable: an infeasible mult=1 guess followed by a
# feasible mult=3 one is the selector working, not a problem.
RE_GUESS = re.compile(
    r"\[trajopt (\w+)(?: mult=(\d+))?\] initial guess (FEASIBLE|INFEASIBLE)")
RE_SOLVER = re.compile(r"\[trajopt (\w+)\] (\w+) \((\d+)\) (?:also )?failed")
RE_RETRY = re.compile(r"\[trajopt (\w+)\] guess was feasible -- retrying with IPOPT")
RE_DENSE = re.compile(r"\[(\w+)\] trajopt solved but its output fails a dense check "
                      r"\(([^)]*(?:\([^)]*\))?[^)]*)\)")
RE_SPLICE = re.compile(r"\[toppra\] WARNING: the default configuration was spliced into "
                       r"(\d+) of (\d+)")


def summarise_log(log: str) -> dict:
    """Counts of the diagnostics the planner printed, per trajopt label."""
    out = dict(guess_feasible=0, guess_infeasible=0, solver_failures=[],
               ipopt_retries=0, dense_rejections=[], splices=[],
               guess_multiplicities=[])
    for label, mult, verdict in RE_GUESS.findall(log):
        out["guess_feasible" if verdict == "FEASIBLE" else "guess_infeasible"] += 1
        if mult:
            out["guess_multiplicities"].append(f"{label}:{mult}={verdict[0]}")
    for label, solver, code in RE_SOLVER.findall(log):
        out["solver_failures"].append(f"{label}:{solver}({code})")
    out["ipopt_retries"] = len(RE_RETRY.findall(log))
    for label, detail in RE_DENSE.findall(log):
        out["dense_rejections"].append(f"{label}: {detail}")
    for n, tot in RE_SPLICE.findall(log):
        out["splices"].append(f"{n}/{tot}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", default=E.DEFAULT_CACHE_DIR)
    ap.add_argument("--verbose", action="store_true",
                    help="also print each point's parsed solver/dense-check detail")
    args = ap.parse_args()

    status = {}
    sp = E.status_path_for(args.cache_dir)
    if os.path.exists(sp):
        status = json.load(open(sp))

    # grasp seed / screening comes from the cached plan's meta, not status.json
    grasp = {}
    uncertified = {}          # idx -> {leg names the optimiser did not certify}
    fell_back = {}            # idx -> {leg names that shipped the BiRRT fallback}
    for idx in range(20):
        p = E.cache_path_for(args.cache_dir, idx)
        if not os.path.exists(p):
            continue
        try:
            _, meta = load_plan(p)
        except Exception as e:
            # Reported, not swallowed. A bare `except: continue` here silently hid a
            # NameError (load_plan was never imported) and every point rendered its
            # grasp seed as "-", which read as "this run predates grasp screening"
            # rather than "this report is broken".
            print(f"  [warn] point {idx}: could not read cached plan "
                  f"({type(e).__name__}: {e})")
            continue
        grasp[idx] = (meta.get("grasp_seed"), meta.get("grasp_screen") or [])
        tj = meta.get("trajopt") or {}
        uncertified[idx] = {leg for leg, d in tj.items()
                            if (d or {}).get("trajopt_uncertified")}
        # The planner now tags a fallback where it happens, so this no longer has to be
        # inferred from a missing captured stage -- which needed the debug pickle, and
        # therefore could not distinguish "shipped the fallback" from "no debug record".
        # The inference below is kept for plans cached before the tag existed.
        fell_back[idx] = {leg for leg, d in tj.items()
                          if (d or {}).get("trajopt_fell_back")}

    hdr = (f"{'pt':>3} {'status':<8} {'gcp':>3} {'grasp':>5} "
           + " ".join(f"{n[:6]:>6}" for n in LEG_ORDER))
    print(hdr)
    print("-" * len(hdr))

    n_pts = n_ok = 0
    n_full_trajopt = 0
    fallback_pts, missing_logs = [], []
    per_leg_fallback = {n: 0 for n in LEG_ORDER}

    for idx in range(20):
        dbg_path = E.debug_path_for(args.cache_dir, idx)
        st = status.get(str(idx), {})
        if not os.path.exists(dbg_path):
            continue
        n_pts += 1
        blob = pickle.load(open(dbg_path, "rb"))
        debug = blob.get("debug", {}) or {}
        log = blob.get("log", "") or ""
        if not log:
            missing_logs.append(idx)

        gcp = st.get("gcp_index")
        cells, any_fb = [], False
        unc = uncertified.get(idx, set())
        fell = fell_back.get(idx, set())
        for name in LEG_ORDER:
            rec = debug.get(name)
            if rec is None:
                cells.append(f"{'-':>6}")
                continue
            stages = set((rec.get("stages") or {}))
            if name in fell:
                # The tag wins over the stage inference: it is the planner's own record
                # of the event, written at the fallback site.
                cells.append(f"{'FALLBK':>6}")
                per_leg_fallback[name] += 1
                any_fb = True
            elif not stages:
                cells.append(f"{'RAISED':>6}")
            elif name in NO_TRAJOPT_BY_DESIGN:
                cells.append(f"{'n/a':>6}")
            elif "trajopt" in stages:
                # opt* marks a leg the optimiser did not certify: the returned iterate
                # satisfies every constraint and passed the dense check, but the solver
                # stopped on a limit or a numerical breakdown rather than converging.
                # Distinguishing it from a clean 'opt' is the point -- it is a standing
                # flag that trajopt is not converging on that problem.
                cells.append(f"{('opt*' if name in unc else 'opt'):>6}")
            else:
                cells.append(f"{'FALLBK':>6}")
                per_leg_fallback[name] += 1
                any_fb = True
        ok = st.get("status") == "success"
        n_ok += ok
        if ok and not any_fb:
            n_full_trajopt += 1
        if ok and any_fb:
            fallback_pts.append(idx)
        gseed, gscreen = grasp.get(idx, (None, []))
        print(f"{idx:>3} {st.get('status', '?'):<8} "
              f"{(str(gcp) if gcp is not None else '-'):>3} "
              f"{(str(gseed) if gseed is not None else '-'):>5} " + " ".join(cells))
        if gscreen and len(gscreen) > 1:
            # Only worth printing when the screen actually rejected something -- that
            # is the case where the first grasp draw would have lost the point.
            rej = [f"seed+{g['seed']} {g['probe_s']:.0f}s" for g in gscreen
                   if not g["plannable"]]
            if rej:
                print(f"      grasp candidates rejected (could not lift+place): {', '.join(rej)}")
        if args.verbose:
            s = summarise_log(log)
            if s["guess_infeasible"]:
                print(f"      guesses: {s['guess_feasible']} feasible, "
                      f"{s['guess_infeasible']} INFEASIBLE")
            if s["solver_failures"]:
                print(f"      solver failures: {', '.join(s['solver_failures'])}"
                      f"  (IPOPT rungs run: {s['ipopt_retries']})")
            for d in s["dense_rejections"]:
                print(f"      dense-rejected: {d}")
            for d in s["splices"]:
                print(f"      DEFAULT-POSE SPLICE: {d} samples")

    print()
    print(f"points with a debug record : {n_pts}")
    print(f"successes                  : {n_ok}")
    print(f"  of which fully optimised : {n_full_trajopt}  "
          f"(every trajopt-bearing leg came from trajopt)")
    print(f"  of which ship a fallback : {len(fallback_pts)}  {fallback_pts}")
    unc_pts = sorted(i for i, legs in uncertified.items() if legs)
    print(f"  with uncertified solves  : {len(unc_pts)}  {unc_pts}"
          + ("   <-- trajopt did not converge on these; the iterate was feasible "
             "and passed the dense check, but this is a flag, not a pass"
             if unc_pts else ""))
    for i in unc_pts:
        print(f"      point {i}: " + ", ".join(sorted(uncertified[i])))
    fb = {k: v for k, v in per_leg_fallback.items() if v}
    print(f"fallback legs by name      : {fb if fb else 'none'}")
    if missing_logs:
        print(f"points with no captured log: {missing_logs} "
              f"(planned before the log was persisted -- replan to diagnose)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
