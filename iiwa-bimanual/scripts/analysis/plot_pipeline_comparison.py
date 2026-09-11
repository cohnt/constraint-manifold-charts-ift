"""
plot_pipeline_comparison.py

Reads per-config JSON files produced by run_full_comparison.py (named
out/timing_<short_name>.json) as well as the aggregate
out/timing_results_full_comparison.json and produces:

  1. A printed timing table (same format as run_full_comparison.py)
  2. A grouped bar chart of per-stage runtimes
  3. A grouped bar chart of TOPPRA-retimed trajectory durations
  4. A summary of which stages succeeded / fell back to backup

Usage
-----
  python scripts/analysis/plot_pipeline_comparison.py          # use out/ dir
  python scripts/analysis/plot_pipeline_comparison.py --out my_out_dir
"""

import os
import sys
import glob
import json
import argparse
import math
import matplotlib
matplotlib.use("Agg")          # headless backend; remove if you want a window
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── Column layout ─────────────────────────────────────────────────────────────
# Each tuple: (json_key_or_None, display_label, unit)
TIMING_COLS = [
    ("iris_total_time",   "IRIS/seed", "s"),
    ("gcs_solve_time",    "GCS",    "s"),
    (("toppra","GCS"),    "G-TOP",  "s"),
    ("trajopt_solve_time","TrjOpt", "s"),
    (("toppra","Trajopt"),"T-TOP",  "s"),
    ("rrt_plan_time",     "RRT",    "s"),
    (("toppra","RRT"),    "R-TOP",  "s"),
]

STAGE_COLORS = {
    "IRIS/seed": "#4e79a7",
    "GCS":    "#f28e2b",
    "G-TOP":  "#e15759",
    "TrjOpt": "#76b7b2",
    "T-TOP":  "#59a14f",
    "RRT":    "#edc948",
    "R-TOP":  "#b07aa1",
}

TOPPRA_DURATION_KEYS = [
    (("toppra_durations","GCS"),    "GCS trajectory (s)"),
    (("toppra_durations","RRT"),    "RRT trajectory (s)"),
    (("toppra_durations","Trajopt"),"Trajopt trajectory (s)"),
]


def _get(record, key):
    """Retrieve a value from a record using a flat or nested key."""
    if isinstance(key, tuple):
        d = record
        for k in key:
            if not isinstance(d, dict) or k not in d:
                return None
            d = d[k]
        return d
    if key == "iris_total_time":
        return iris_time_per_seed(record)
    return record.get(key, None)


def iris_time_per_seed(record):
    """
    IRIS time normalized per seed point.

    `iris_total_time` is the total across all seeds, but the per-seed figure is
    what gets reported. The seed count is read from the record (written by
    run_full_comparison.py) rather than hardcoded, so this stays correct if the
    seed list changes. Older records predate that field; fall back to the total
    and say so rather than silently dividing by the wrong number.
    """
    total = record.get("iris_total_time", None)
    if total is None:
        return None
    n_seeds = record.get("iris_num_seeds", None)
    if not n_seeds:
        raise KeyError(
            "Record is missing `iris_num_seeds`, so the IRIS column cannot be "
            "normalized per seed. This record predates the current schema; "
            "re-run run_full_comparison.py rather than mixing it with new results.")
    return total / n_seeds


# Number of configurations in run_full_comparison.py's CONFIGS list, and so the
# number of rows the paper's table must have. A partial out/ directory produces
# a correct-looking table with too few rows, which has happened more than once;
# warn loudly rather than letting it pass silently.
EXPECTED_NUM_CONFIGS = 17


def load_records(out_dir):
    """
    Load all per-config JSONs from out_dir.  Falls back to the aggregate file
    if no per-config files are found.  Returns a list of record dicts.
    """
    per_config = sorted(glob.glob(os.path.join(out_dir, "timing_*.json")))
    # Exclude the aggregate file itself
    per_config = [p for p in per_config
                  if not os.path.basename(p).startswith("timing_results")]

    if per_config:
        records = []
        for path in per_config:
            with open(path) as f:
                records.append(json.load(f))
        return records

    # Fall back to aggregate
    agg = os.path.join(out_dir, "timing_results_full_comparison.json")
    if os.path.exists(agg):
        with open(agg) as f:
            return json.load(f)

    return []


def print_table(records):
    """Print the same timing table as run_full_comparison.py."""
    col_labels = [c[1] for c in TIMING_COLS] + ["GCS ok", "SNOPT", "TOPPRA ok"]
    # Widths: 8 for IRIS/GCS/G-TOP, 15 for stochastic stages (for std display),
    # then the three outcome columns.
    col_widths = [9, 8, 8, 15, 15, 15, 15, 6, 9, 9]

    max_config_len = max(len(r['config']) for r in records) if records else 43
    header = f"{'CONFIGURATION':<{max_config_len}} | " + " | ".join(f"{l:<{w}}" for l, w in zip(col_labels, col_widths))
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))

    for rec in records:
        vals = []
        for i, (key, label, _) in enumerate(TIMING_COLS):
            v = _get(rec, key)
            if v is None or v <= 0:
                vals.append("—")
                continue
            
            std_key = None
            if isinstance(key, str):
                # e.g. trajopt_solve_time -> trajopt_solve_time_std
                std_key = f"{key}_std"
            elif isinstance(key, tuple) and key[0] == "toppra":
                # e.g. ("toppra", "Trajopt") -> toppra_Trajopt_std
                std_key = f"toppra_{key[1]}_std"
            
            std_v = rec.get(std_key, 0.0) if std_key else 0.0
            
            s = f"{v:.2f}"
            if std_v > 1e-3:
                s += f" (±{std_v:.2f})"
            vals.append(s)

        # A stage runtime is not interpretable without that stage's outcome.
        gcs_ok = rec.get("gcs_success", None)
        vals.append("—" if gcs_ok is None else ("yes" if gcs_ok else "NO"))

        # SNOPT success rate is a first-class column: the Trajopt runtimes are
        # means over all trials, and a failing trial can burn the full solver
        # time limit, so a runtime is not interpretable without its success rate.
        n_trials = len(rec.get("rrt_trials", []))
        s_rate = rec.get('trajopt_success_rate', None)
        if s_rate is None:
            vals.append("—")
        elif n_trials:
            vals.append(f"{round(s_rate * n_trials):.0f}/{n_trials}")
        else:
            vals.append(f"{s_rate*100:.0f}%")

        # Same reasoning for TOPPRA: the runtime columns above are means over
        # *successful* solves only, so the count of those solves belongs beside
        # them. Summed over the GCS/RRT/Trajopt retimings.
        succ = rec.get("toppra_num_success", {})
        att = rec.get("toppra_num_attempts", {})
        vals.append(f"{sum(succ.values())}/{sum(att.values())}" if att else "—")

        line = f"{rec['config']:<{max_config_len}} | " + " | ".join(f"{v:<{w}}" for v, w in zip(vals, col_widths))
        print(line)
    print("=" * len(header) + "\n")


def print_stats(records):
    """Print per-stage statistics across all trials of all configs."""
    print("=== Stage Statistics (across all completed configs) ===\n")
    for key, label, unit in TIMING_COLS:
        values = [_get(r, key) for r in records]
        values = [v for v in values if v is not None and v > 0]
        if not values:
            print(f"  {label:<8}: no data")
            continue
        print(f"  {label:<8}: n={len(values)}  "
              f"min={min(values):.2f}{unit}  "
              f"max={max(values):.2f}{unit}  "
              f"mean={np.mean(values):.2f}{unit}  "
              f"median={np.median(values):.2f}{unit}")

    # TOPPRA success/backup counts (Trial-level)
    print()
    print("=== TOPPRA outcome counts (Total Trials) ===\n")
    
    total_trials = sum(len(r.get("rrt_trials", [r])) for r in records)
    
    for traj_type in ("GCS", "RRT", "Trajopt"):
        successes = 0
        backups = 0
        n_expected = 0
        for r in records:
            trials = r.get("rrt_trials", [r])
            for t in trials:
                # For GCS, it's deterministic and usually only in trial 0 or the summary.
                if traj_type == "GCS":
                    is_backup = _get(r, ("toppra_backup", "GCS")) is True
                    has_time  = _get(r, ("toppra", "GCS")) is not None
                    if has_time and not is_backup:
                        successes += 1
                    if is_backup:
                        backups += 1
                    n_expected += 1
                    break # Only count once per config
                elif traj_type == "RRT":
                    is_backup = _get(t, ("toppra_backup", traj_type)) is True
                    has_time  = _get(t, ("toppra", traj_type)) is not None
                    if has_time and not is_backup:
                        successes += 1
                    if is_backup:
                        backups += 1
                    n_expected += 1
                elif traj_type == "Trajopt":
                    # Only count if TrajOpt solver actually succeeded
                    if t.get("trajopt_solve_success") is True:
                        is_backup = _get(t, ("toppra_backup", traj_type)) is True
                        has_time  = _get(t, ("toppra", traj_type)) is not None
                        if has_time and not is_backup:
                            successes += 1
                        if is_backup:
                            backups += 1
                        n_expected += 1
        
        print(f"  {traj_type:<8}: {successes}/{n_expected} successes"
              + (f", {backups} backups" if backups else ""))

    # TrajOpt solver successes (Trial-level)
    to_success = 0
    for r in records:
        trials = r.get("rrt_trials", [r])
        for t in trials:
            if t.get("trajopt_solve_success") is True:
                to_success += 1
    
    print(f"\n  TrajOpt solver: {to_success}/{total_trials} SNOPT successes")
    print()


def plot_runtimes(records, out_dir):
    """Grouped bar chart: per-stage runtime for each config."""
    configs    = [r["config"] for r in records]
    stages     = [(key, label) for key, label, _ in TIMING_COLS]
    n_configs  = len(configs)
    n_stages   = len(stages)

    x = np.arange(n_configs)
    width = 0.8 / n_stages

    fig, ax = plt.subplots(figsize=(max(10, n_configs * 1.5), 6))
    for i, (key, label) in enumerate(stages):
        values = [_get(r, key) or 0.0 for r in records]
        
        # Extract standard deviation for error bars
        std_key = None
        if isinstance(key, str):
            std_key = f"{key}_std"
        elif isinstance(key, tuple) and key[0] == "toppra":
            std_key = f"toppra_{key[1]}_std"
        
        stds = [r.get(std_key, 0.0) if std_key else 0.0 for r in records]
        
        offset = (i - n_stages / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width,
                      yerr=stds, capsize=3,
                      label=label, color=STAGE_COLORS.get(label, None),
                      alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Time (s)")
    ax.set_title("Per-stage runtimes by configuration")
    ax.legend(ncol=4, fontsize=8)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(axis="y", which="both", alpha=0.3)
    plt.tight_layout()

    path = os.path.join(out_dir, "runtimes.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_toppra_durations(records, out_dir):
    """
    Grouped bar chart: TOPPRA-retimed trajectory durations.
    Only configs / trajectory types where TOPPRA succeeded are shown.
    """
    traj_types = [("GCS", "#e15759"), ("RRT", "#edc948"), ("Trajopt", "#59a14f")]
    configs    = [r["config"] for r in records]
    n_configs  = len(configs)
    n_types    = len(traj_types)

    x     = np.arange(n_configs)
    width = 0.8 / n_types

    fig, ax = plt.subplots(figsize=(max(10, n_configs * 1.5), 5))
    has_data = False
    for i, (ttype, color) in enumerate(traj_types):
        dur_key = ("toppra_durations", ttype)
        values  = [_get(r, dur_key) or 0.0 for r in records]
        if any(v > 0 for v in values):
            has_data = True
        offset = (i - n_types / 2 + 0.5) * width
        ax.bar(x + offset, values, width, label=f"{ttype} traj", color=color, alpha=0.85)

    if not has_data:
        # Fall back: read toppra timing as proxy if duration not stored
        plt.close(fig)
        return

    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Trajectory duration (s)")
    ax.set_title("TOPPRA-retimed trajectory durations by configuration")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    path = os.path.join(out_dir, "traj_durations.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_stage_breakdown(records, out_dir):
    """Stacked bar chart showing how total time is split across stages."""
    # Only include timing stages (not TOPPRA, as they overlap with the stages)
    pipeline_stages = [
        ("iris_total_time",    "IRIS/seed", STAGE_COLORS["IRIS/seed"]),
        ("gcs_solve_time",     "GCS",    STAGE_COLORS["GCS"]),
        ("rrt_plan_time",      "RRT",    STAGE_COLORS["RRT"]),
        ("rrt_shortcut_time",  "RRT shortcut", "#a9cce3"),
        ("trajopt_solve_time", "TrajOpt",STAGE_COLORS["TrjOpt"]),
    ]
    configs = [r["config"] for r in records]
    x = np.arange(len(configs))

    fig, ax = plt.subplots(figsize=(max(10, len(configs) * 1.5), 6))
    bottom = np.zeros(len(configs))
    for key, label, color in pipeline_stages:
        vals = np.array([r.get(key) or 0.0 for r in records])
        # Note: stack plots don't easily show error bars for middle segments,
        # so we'll skip yerr here but the total height will be accurate.
        ax.bar(x, vals, label=label, bottom=bottom, color=color, alpha=0.85)
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Cumulative time (s)")
    ax.set_title("Planning time breakdown by stage")
    ax.legend(ncol=3, fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    path = os.path.join(out_dir, "time_breakdown.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Plot results from run_full_comparison.py")
    parser.add_argument("--out", default=None,
                        help="Directory containing the timing JSON files (default: <repo>/out)")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_dir   = os.path.abspath(os.path.join(script_dir, "../.."))
    out_dir    = args.out if args.out else os.path.join(repo_dir, "out")
    plot_dir   = os.path.join(out_dir, "plots", "pipeline_comparison")
    os.makedirs(plot_dir, exist_ok=True)

    records = load_records(out_dir)
    if not records:
        print(f"No timing JSON files found in {out_dir}. Run run_full_comparison.py first.")
        sys.exit(1)

    print(f"\nLoaded {len(records)} configuration(s) from {out_dir}\n")

    if len(records) != EXPECTED_NUM_CONFIGS:
        print(f"  *** WARNING: expected {EXPECTED_NUM_CONFIGS} configurations, found "
              f"{len(records)}. The table below is NOT the paper's table. ***")
        print("  out/ is gitignored and --skip-existing reuses whatever it finds, so a")
        print("  partial or stale out/ silently yields a table with the wrong rows.")
        print("  Move the old results aside and re-run the full sweep.\n")

    # ── Printed output ──────────────────────────────────────────────────────
    print_table(records)
    print_stats(records)

    # ── Plots ───────────────────────────────────────────────────────────────
    print("Generating plots...")
    plot_runtimes(records, plot_dir)
    plot_toppra_durations(records, plot_dir)
    plot_stage_breakdown(records, plot_dir)

    print(f"\nAll plots saved to {plot_dir}\n")


if __name__ == "__main__":
    main()
