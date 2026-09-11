"""
Statistics for the three-way IK benchmark.

Two lenses on the same per-run records, and they answer different questions:

  * **Single-start** (primary) — one solve from one initial guess.  This is the metric the
    paper reports and the one every tuning decision is scored on.
  * **Multi-start** (secondary) — the ten guesses belonging to one target treated as a
    restart sequence, since a practitioner who cares about the answer restarts on failure.
    A method that fails more often but is cheap per attempt can still win here.

Everything below is post-processing over the records the benchmark already writes, so old
logs can be re-scored without re-solving.

A note on which numbers survive machine load.  `--max-wall-time` is wall clock, so under
CPU contention a solve that would have finished trips the cap and is recorded as a
*failure*.  That makes success rate, every timing column, and the composition of the mutual
subsets load-dependent.  Only cost-conditional-on-success is immune.  `cap_hit_stats` exists
so a run can state whether the cap bound at all.
"""

from math import comb

import numpy as np


# ── Basic descriptive statistics ──────────────────────────────────────────────

def compute_stats(values):
    """Descriptive statistics for a list of floats, ignoring NaN/inf.

    Quantiles are reported alongside the mean because these distributions are heavily
    right-skewed: a mean solve time can be 4x the median purely because of a tail against
    the wall-clock cap.
    """
    finite = [v for v in values if np.isfinite(v)]
    if not finite:
        return {"n": 0, "mean": None, "std": None, "median": None,
                "min": None, "max": None, "p90": None, "p99": None}
    a = np.asarray(finite, dtype=float)
    return {
        "n":      int(a.size),
        "mean":   float(a.mean()),
        "std":    float(a.std()),
        "median": float(np.median(a)),
        "min":    float(a.min()),
        "max":    float(a.max()),
        "p90":    float(np.percentile(a, 90)),
        "p99":    float(np.percentile(a, 99)),
    }


# ── Paired significance tests ─────────────────────────────────────────────────

def mcnemar_exact(a, b):
    """
    Two-sided exact McNemar test (a binomial sign test on the discordant pairs).

    `a` is the count of pairs where the first method succeeded and the second did not, `b`
    the reverse.  Concordant pairs carry no information about which method is better and
    are correctly excluded.  This is the right test here because every formulation is run
    on the identical (target, initial guess) pair, so the success indicators are paired,
    not two independent samples.
    """
    n = a + b
    if n == 0:
        return 1.0
    k = min(a, b)
    p = sum(comb(n, i) for i in range(0, k + 1)) / 2 ** n * 2
    return float(min(1.0, p))


def wilcoxon_signed_rank(x, y):
    """
    Two-sided Wilcoxon signed-rank p-value for paired samples, or None if scipy is
    unavailable or every pair is tied.  Used on paired cost and time differences over a
    mutual-success subset, where the differences are decidedly non-normal.
    """
    d = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    d = d[np.isfinite(d)]
    if d.size == 0 or np.allclose(d, 0.0):
        return None
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        return None
    try:
        return float(wilcoxon(d, alternative="two-sided").pvalue)
    except ValueError:
        return None


# ── Bootstrap confidence intervals ────────────────────────────────────────────

def bootstrap_ci(records, key, stat="mean", group="target", n_boot=2000, seed=0,
                 alpha=0.05):
    """
    Percentile bootstrap CI, resampling whole *targets* rather than individual runs.

    The ten guesses inside one target share a target pose, so their outcomes are
    correlated; resampling runs independently would understate the interval.  Resampling
    the cluster is the standard fix and is what makes "Direct's cost is slightly worse"
    either a real effect or noise.

    `stat` is "mean" (over finite values, i.e. successful runs for cost) or "rate" (mean of
    a boolean field over all runs).
    """
    groups = {}
    for r in records:
        groups.setdefault(r[group], []).append(r)
    keys = list(groups)
    if not keys:
        return {"point": None, "lo": None, "hi": None, "n_boot": 0}

    def evaluate(chosen):
        vals = []
        for k in chosen:
            for r in groups[k]:
                vals.append(r[key])
        if stat == "rate":
            return float(np.mean([bool(v) for v in vals])) if vals else float("nan")
        finite = [v for v in vals if np.isfinite(v)]
        return float(np.mean(finite)) if finite else float("nan")

    point = evaluate(keys)
    rng = np.random.default_rng(seed)
    idx = np.arange(len(keys))
    draws = []
    for _ in range(n_boot):
        sample = [keys[i] for i in rng.choice(idx, size=len(keys), replace=True)]
        v = evaluate(sample)
        if np.isfinite(v):
            draws.append(v)
    if not draws:
        return {"point": point, "lo": None, "hi": None, "n_boot": 0}
    return {
        "point":  point,
        "lo":     float(np.percentile(draws, 100 * alpha / 2)),
        "hi":     float(np.percentile(draws, 100 * (1 - alpha / 2))),
        "n_boot": len(draws),
    }


# ── Single-start aggregates ───────────────────────────────────────────────────

def compute_all_stats(records, key):
    """Success rate plus cost/time statistics, over all runs and over successes only."""
    all_costs = [r[f"{key}_cost"] for r in records]
    all_times = [r[f"{key}_time"] for r in records]
    successes = [r[f"{key}_ok"]   for r in records]

    succ_costs = [c for c, s in zip(all_costs, successes) if s]
    succ_times = [t for t, s in zip(all_times, successes) if s]

    return {
        "success_rate": float(np.mean(successes)) if successes else 0.0,
        "cost_all":     compute_stats(all_costs),
        "time_all":     compute_stats(all_times),
        "cost_success": compute_stats(succ_costs),
        "time_success": compute_stats(succ_times),
    }


def mutual_stats(records, keys):
    """
    Cost/time statistics on the subset where *every* named formulation succeeded, plus the
    paired tests on that subset.  This is the fairest head-to-head: all methods solved the
    same instances.
    """
    keys = list(keys)
    mutual = [r for r in records if all(r[f"{k}_ok"] for k in keys)]
    out = {"n_mutual": len(mutual)}
    if not mutual:
        return out
    for k in keys:
        out[k] = {
            "cost": compute_stats([r[f"{k}_cost"] for r in mutual]),
            "time": compute_stats([r[f"{k}_time"] for r in mutual]),
        }
    if len(keys) == 2:
        a, b = keys
        out["paired"] = {
            "cost_wilcoxon_p": wilcoxon_signed_rank(
                [r[f"{a}_cost"] for r in mutual], [r[f"{b}_cost"] for r in mutual]),
            "time_wilcoxon_p": wilcoxon_signed_rank(
                [r[f"{a}_time"] for r in mutual], [r[f"{b}_time"] for r in mutual]),
            f"{a}_cheaper_frac": float(np.mean(
                [r[f"{a}_cost"] < r[f"{b}_cost"] for r in mutual])),
            f"{a}_faster_frac": float(np.mean(
                [r[f"{a}_time"] < r[f"{b}_time"] for r in mutual])),
            f"{a}_dominates_frac": float(np.mean(
                [r[f"{a}_cost"] < r[f"{b}_cost"] and r[f"{a}_time"] < r[f"{b}_time"]
                 for r in mutual])),
        }
    return out


def paired_success_stats(records, key_a, key_b):
    """McNemar's exact test on the paired success indicators of two formulations."""
    a_only = sum(1 for r in records if r[f"{key_a}_ok"] and not r[f"{key_b}_ok"])
    b_only = sum(1 for r in records if r[f"{key_b}_ok"] and not r[f"{key_a}_ok"])
    return {
        f"{key_a}_only": a_only,
        f"{key_b}_only": b_only,
        "both":    sum(1 for r in records if r[f"{key_a}_ok"] and r[f"{key_b}_ok"]),
        "neither": sum(1 for r in records
                       if not r[f"{key_a}_ok"] and not r[f"{key_b}_ok"]),
        "mcnemar_exact_p": mcnemar_exact(a_only, b_only),
    }


# ── Diagnostics ───────────────────────────────────────────────────────────────

def cap_hit_stats(records, keys, max_wall_time, frac=0.95):
    """
    How often each formulation ran out of time.

    Two independent signals, because neither alone is reliable: the wall-clock fraction
    catches solves that ran long, and the solver status catches SNOPT's own resource-limit
    exit (its time limit is soft, checked only between major iterations, so observed times
    overshoot the cap and a pure wall-clock test both over- and under-counts).  A non-zero
    cap-hit rate means the run's success rates are not comparable against a run made under
    different machine load.

    The two SNOPT codes are distinguished because they mean different things: INFO 32 is
    "major iteration limit reached" — the solve was cut short by iteration budget, not by
    the clock, and would not be fixed by a faster machine — while INFO 34 is the actual
    time limit.
    """
    n = max(len(records), 1)
    out = {"max_wall_time": max_wall_time, "wall_frac": frac}
    for k in keys:
        by_time = sum(1 for r in records
                      if np.isfinite(r.get(f"{k}_time", np.nan))
                      and r[f"{k}_time"] >= frac * max_wall_time)
        n_iter_limit = sum(1 for r in records if r.get(f"{k}_solver_status") == 32)
        n_time_limit = sum(1 for r in records if r.get(f"{k}_solver_status") == 34)
        out[k] = {
            "n_over_wall_frac":  by_time,
            "frac_over_wall":    by_time / n,
            "n_solver_timeout":  n_time_limit,
            "n_iteration_limit": n_iter_limit,
        }
    return out


def failure_reason_histogram(records, keys):
    """Counts of each `verify_solution` rejection reason, per formulation."""
    out = {}
    for k in keys:
        hist = {}
        for r in records:
            if r.get(f"{k}_ok"):
                continue
            reason = r.get(f"{k}_fail_reason") or "unknown"
            hist[reason] = hist.get(reason, 0) + 1
        out[k] = hist
    return out


# ── Multi-start (secondary lens) ──────────────────────────────────────────────

def multistart_stats(records, keys, budgets=(1, 2, 3, 5)):
    """
    Treat each target's guesses as a restart sequence and report what a multi-start user
    would actually experience.

    Per target and per formulation:
      best_cost                — lowest cost over that target's successful guesses
      solved                   — did any guess succeed
      guesses_to_first_success — 1-based index of the first success
      time_to_first_success    — cumulative solve time through that first success
      total_time               — time spent on all of the target's guesses

    Aggregated over targets, plus a restart-budget curve (fraction of targets solved within
    the first k guesses) and pairwise best-cost win rates.
    """
    keys = list(keys)
    by_target = {}
    for r in records:
        by_target.setdefault(r["target"], []).append(r)
    for runs in by_target.values():
        runs.sort(key=lambda r: r["guess_idx"])

    per_target = {k: [] for k in keys}
    for tgt, runs in sorted(by_target.items()):
        for k in keys:
            costs = [r[f"{k}_cost"] for r in runs
                     if r[f"{k}_ok"] and np.isfinite(r[f"{k}_cost"])]
            first = None
            cum = 0.0
            t_first = float("nan")
            for i, r in enumerate(runs):
                t = r[f"{k}_time"]
                cum += t if np.isfinite(t) else 0.0
                if first is None and r[f"{k}_ok"]:
                    first = i + 1
                    t_first = cum
            per_target[k].append({
                "target": tgt,
                "solved": first is not None,
                "best_cost": float(min(costs)) if costs else float("nan"),
                "guesses_to_first_success": first,
                "time_to_first_success": t_first,
                "total_time": cum,
                "n_success": len(costs),
            })

    out = {"n_targets": len(by_target), "budgets": list(budgets)}
    for k in keys:
        rows = per_target[k]
        n = max(len(rows), 1)
        out[k] = {
            "any_success_rate": float(np.mean([r["solved"] for r in rows])),
            "best_cost":        compute_stats([r["best_cost"] for r in rows]),
            "time_to_first_success": compute_stats(
                [r["time_to_first_success"] for r in rows if r["solved"]]),
            "total_time":       compute_stats([r["total_time"] for r in rows]),
            "median_guesses_to_first_success": (
                float(np.median([r["guesses_to_first_success"] for r in rows
                                 if r["solved"]]))
                if any(r["solved"] for r in rows) else None),
            "solved_within": {
                str(b): float(sum(1 for r in rows
                                  if r["solved"] and r["guesses_to_first_success"] <= b) / n)
                for b in budgets
            },
        }

    # Pairwise best-cost comparison on targets both methods solved.
    out["best_cost_wins"] = {}
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            rows_a = {r["target"]: r for r in per_target[a]}
            rows_b = {r["target"]: r for r in per_target[b]}
            common = [t for t in rows_a
                      if rows_a[t]["solved"] and rows_b[t]["solved"]]
            if not common:
                out["best_cost_wins"][f"{a}_vs_{b}"] = {"n": 0}
                continue
            ca = np.array([rows_a[t]["best_cost"] for t in common])
            cb = np.array([rows_b[t]["best_cost"] for t in common])
            tie = np.abs(ca - cb) <= 1e-6
            out["best_cost_wins"][f"{a}_vs_{b}"] = {
                "n": len(common),
                f"{a}_wins": int(np.sum((ca < cb) & ~tie)),
                f"{b}_wins": int(np.sum((cb < ca) & ~tie)),
                "ties": int(np.sum(tie)),
                "wilcoxon_p": wilcoxon_signed_rank(ca, cb),
            }

    out["_per_target"] = per_target
    return out
