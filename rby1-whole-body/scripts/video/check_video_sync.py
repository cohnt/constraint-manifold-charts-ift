"""Cross-check the hardware videos against the execution records' wall clock.

Everything needed to align the plots with the footage is logged in absolute Unix
time, so the alignment is derivable rather than detectable:

  * the execution record has t_start / t_end and, per step, absolute
    logs["timestamp"] and logs["command_timestamps"];
  * each phone video carries a container creation_time, written when the file is
    closed, so recording_start = creation_time - duration. The filename's
    timestamp corroborates it to about a second.

From those two clocks this reports, per seed, where the robot's motion actually
sits inside its video and how that compares with videos/trim_points.json.

Three windows matter and they are NOT the same:
  robot   -- t_start .. t_end, every motion the camera saw, premove included
  plotted -- first trajectory command .. last state sample, what the error plots
             actually draw (compute_errors.py skips the premove)
  trim    -- what trim_points.json currently cuts

Usage:
    .venv/bin/python scripts/video/check_video_sync.py
    .venv/bin/python scripts/video/check_video_sync.py --json out.json
"""

import argparse
import datetime
import glob
import json
import os
import pickle
import re
import subprocess
import sys

import numpy as np
import plan_format  # noqa: F401  -- installs the pre-rename module aliases the
                    # committed plan and execution pickles were written with

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESULTS_DIR = os.path.join(REPO, "results", "2026-08-11")
HW_DIR = os.path.join(REPO, "videos")
TRIM_JSON = os.path.join(HW_DIR, "trim_points.json")

FNAME_RE = re.compile(r"(\d{8})_(\d{6})")

# Tolerance for "the filename agrees with creation_time - duration".
FNAME_TOL_S = 3.0


def probe(path):
    out = subprocess.check_output([
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_format", "-show_streams", path,
    ]).decode()
    d = json.loads(out)
    fmt = d["format"]
    tags = fmt.get("tags", {})
    dur = float(fmt["duration"])

    created = tags.get("creation_time")
    created_epoch = None
    if created:
        # Container creation_time is UTC ("...Z").
        dt = datetime.datetime.strptime(created, "%Y-%m-%dT%H:%M:%S.%f%z")
        created_epoch = dt.timestamp()

    # The phone also records its own UTC offset; keep it for reporting only.
    utc_offset = tags.get("com.samsung.android.utc_offset")

    return {"duration": dur, "created_epoch": created_epoch,
            "utc_offset": utc_offset}


def fname_epoch(fname):
    """Local wall-clock epoch encoded in the filename (recording start)."""
    m = FNAME_RE.search(fname)
    if not m:
        return None
    dt = datetime.datetime.strptime(m.group(0), "%Y%m%d_%H%M%S")
    return dt.timestamp()


def exec_windows(path):
    """Absolute-time windows for one execution record."""
    with open(path, "rb") as f:
        r = pickle.load(f)

    traj = [s for s in r["steps"] if s["type"] == "trajectory" and s.get("logs")]
    if not traj:
        raise RuntimeError(f"no trajectory steps with logs in {path}")

    first_cmd = float(traj[0]["logs"]["command_timestamps"][0])
    last_state = float(traj[-1]["logs"]["timestamp"][-1])

    premove = r.get("premove") or {}
    pre_cmd = premove.get("command_timestamps")
    premove_first = float(pre_cmd[0]) if pre_cmd is not None and len(pre_cmd) else None

    # How far the joints travel during the premove, to judge whether the camera
    # would see it as motion.
    premove_travel = None
    if premove.get("joint_position") is not None and len(premove["joint_position"]):
        jp = np.asarray(premove["joint_position"])[:, 2:22]
        premove_travel = float(np.max(np.abs(jp[-1] - jp[0])))

    return {
        "t_start": float(r["t_start"]),
        "t_end": float(r["t_end"]),
        "wall_s": float(r.get("wall_s", 0.0)),
        "premove_first_cmd": premove_first,
        "premove_wall_s": float(r.get("premove_wall_s", 0.0)),
        "premove_travel_rad": premove_travel,
        "first_traj_cmd": first_cmd,
        "last_state": last_state,
        "n_traj_steps": len(traj),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", default=RESULTS_DIR)
    ap.add_argument("--hw-dir", default=HW_DIR)
    ap.add_argument("--trim-json", default=TRIM_JSON)
    ap.add_argument("--error-data", default=os.path.join(REPO, "video", "error_data.pkl"))
    ap.add_argument("--json", default=None, help="write the full table as JSON")
    ap.add_argument("--write-trim", action="store_true",
                    help="rewrite trim_points.json from the logged clock "
                         "(keeps a .bak and prints the change per seed)")
    args = ap.parse_args()

    videos = sorted(f for f in os.listdir(args.hw_dir) if f.endswith(".mp4"))
    execs = sorted(glob.glob(os.path.join(args.results_dir, "point_*_exec_*.pkl")))

    with open(args.trim_json) as f:
        trim = json.load(f)

    all_errors = {}
    if os.path.exists(args.error_data):
        with open(args.error_data, "rb") as f:
            all_errors = pickle.load(f)

    print(f"{len(videos)} videos, {len(execs)} execution records\n")

    # ---- 1. is the filename really the recording start? ----------------
    print("=" * 108)
    print("1. VIDEO TIME BASE   (recording_start = creation_time - duration, "
          "cross-checked against the filename)")
    print("=" * 108)
    print(f"{'video':<24} {'dur':>7} {'fname->start':>13} {'created-dur':>13} "
          f"{'delta':>8}  {'tz':>6}")
    vinfo = {}
    for v in videos:
        p = probe(os.path.join(args.hw_dir, v))
        fe = fname_epoch(v)
        start = (p["created_epoch"] - p["duration"]
                 if p["created_epoch"] is not None else fe)
        delta = start - fe if (fe is not None and start is not None) else float("nan")
        vinfo[v] = {"duration": p["duration"], "start_epoch": start,
                    "fname_epoch": fe, "delta_fname_vs_created": delta}
        flag = "" if abs(delta) <= FNAME_TOL_S else "   <-- disagree"
        print(f"{v:<24} {p['duration']:>7.2f} "
              f"{datetime.datetime.fromtimestamp(fe).strftime('%H:%M:%S'):>13} "
              f"{datetime.datetime.fromtimestamp(start).strftime('%H:%M:%S.%f')[:-4]:>13} "
              f"{delta:>+8.2f}  {str(p['utc_offset']):>6}{flag}")

    # ---- 2. map seeds to videos by wall clock ---------------------------
    print()
    print("=" * 108)
    print("2. SEED -> VIDEO MAPPING   (the robot's motion window must fall inside "
          "the recording window)")
    print("=" * 108)
    einfo, mapping = {}, {}
    for ep in execs:
        pt = int(re.search(r"point_(\d+)_exec", os.path.basename(ep)).group(1))
        einfo[pt] = exec_windows(ep)

    # Match each seed to the recording whose start is nearest its motion, then
    # report the overhang at each end. Strict containment is the wrong test: in
    # several seeds the premove begins a fraction of a second before the
    # recording's first frame, which is a lead-in problem, not a mapping error.
    print(f"{'pt':>3} {'robot start':>12} {'robot end':>10} {'video':<24} {'idx':>4} "
          f"{'lead-in':>8} {'lead-out':>9}")
    for pt in sorted(einfo):
        e = einfo[pt]
        best = min(videos, key=lambda v: abs(vinfo[v]["start_epoch"] - e["t_start"]))
        vs = vinfo[best]["start_epoch"]
        ve = vs + vinfo[best]["duration"]
        mapping[pt] = best
        sorted_idx = videos.index(best)

        lead_in = e["t_start"] - vs      # <0: robot moved before recording began
        lead_out = ve - e["t_end"]       # <0: recording stopped before robot did
        flags = []
        if sorted_idx != pt:
            flags.append(f"sorted index {sorted_idx} != pt {pt}")
        if lead_in < 0:
            flags.append(f"robot started {-lead_in:.2f}s BEFORE recording")
        if lead_out < 0:
            flags.append(f"recording ended {-lead_out:.2f}s BEFORE robot")
        flag = ("   <-- " + "; ".join(flags)) if flags else ""
        print(f"{pt:>3} "
              f"{datetime.datetime.fromtimestamp(e['t_start']).strftime('%H:%M:%S'):>12} "
              f"{datetime.datetime.fromtimestamp(e['t_end']).strftime('%H:%M:%S'):>10} "
              f"{best:<24} {sorted_idx:>4} {lead_in:>+8.2f} {lead_out:>+9.2f}{flag}")

    # ---- 3. the three windows, in video seconds -------------------------
    print()
    print("=" * 108)
    print("3. WINDOWS IN VIDEO SECONDS   (robot = incl. premove, plotted = what "
          "the plots draw, trim = current)")
    print("=" * 108)
    print(f"{'pt':>3} | {'robot in':>8} {'robot out':>9} {'rdur':>6} | "
          f"{'plot in':>8} {'plot out':>8} {'pdur':>6} | "
          f"{'trim in':>8} {'trim out':>8} {'tdur':>6} | "
          f"{'in err':>7} {'dur err':>7} {'pre':>6}")
    rows = []
    for pt in sorted(einfo):
        e, v = einfo[pt], mapping[pt]
        if v is None:
            print(f"{pt:>3} |  -- unmapped --")
            continue
        vs = vinfo[v]["start_epoch"]
        t = trim.get(v, {})

        robot_in = e["t_start"] - vs
        robot_out = e["t_end"] - vs
        plot_in = e["first_traj_cmd"] - vs
        plot_out = e["last_state"] - vs
        trim_in = t.get("start_s", float("nan"))
        trim_out = t.get("end_s", float("nan"))

        pdur = plot_out - plot_in
        tdur = trim_out - trim_in
        ed = all_errors.get(pt, {})
        # what the plots actually span, straight from error_data
        plotted_span = ed.get("total_duration", float("nan"))

        in_err = trim_in - plot_in
        dur_err = tdur - plotted_span

        row = {
            "point": pt, "video": v,
            "video_start_epoch": vs, "video_duration": vinfo[v]["duration"],
            "robot_in": robot_in, "robot_out": robot_out,
            "plot_in": plot_in, "plot_out": plot_out,
            "plotted_span_from_error_data": plotted_span,
            "trim_in": trim_in, "trim_out": trim_out,
            "trim_start_err_vs_plot": in_err,
            "trim_dur_err_vs_plot": dur_err,
            "premove_s": e["first_traj_cmd"] - e["t_start"],
            "premove_travel_rad": e["premove_travel_rad"],
        }
        rows.append(row)

        print(f"{pt:>3} | {robot_in:>8.2f} {robot_out:>9.2f} "
              f"{robot_out - robot_in:>6.2f} | "
              f"{plot_in:>8.2f} {plot_out:>8.2f} {pdur:>6.2f} | "
              f"{trim_in:>8.2f} {trim_out:>8.2f} {tdur:>6.2f} | "
              f"{in_err:>+7.2f} {dur_err:>+7.2f} {row['premove_s']:>6.2f}")

    # ---- 4. verdict -----------------------------------------------------
    print()
    print("=" * 108)
    print("4. SUMMARY")
    print("=" * 108)
    if rows:
        in_errs = np.array([r["trim_start_err_vs_plot"] for r in rows])
        dur_errs = np.array([r["trim_dur_err_vs_plot"] for r in rows])
        pres = np.array([r["premove_s"] for r in rows])
        travel = np.array([r["premove_travel_rad"] or 0.0 for r in rows])
        overruns = [r for r in rows
                    if r["robot_out"] > r["video_duration"] + 0.01]

        print(f"trim start vs plot start : mean {in_errs.mean():+.2f}s  "
              f"min {in_errs.min():+.2f}s  max {in_errs.max():+.2f}s  "
              f"|max| {np.abs(in_errs).max():.2f}s")
        print(f"trim duration vs plotted : mean {dur_errs.mean():+.2f}s  "
              f"min {dur_errs.min():+.2f}s  max {dur_errs.max():+.2f}s  "
              f"|max| {np.abs(dur_errs).max():.2f}s")
        print(f"premove before 1st traj  : mean {pres.mean():.2f}s  "
              f"min {pres.min():.2f}s  max {pres.max():.2f}s   "
              f"(max joint travel {travel.max():.3f} rad)")
        if overruns:
            print(f"\nrobot motion runs past the end of the recording for "
                  f"{len(overruns)} seed(s):")
            for r in overruns:
                print(f"   pt {r['point']:>2}: robot ends {r['robot_out']:.2f}s, "
                      f"video is {r['video_duration']:.2f}s long "
                      f"(short by {r['robot_out'] - r['video_duration']:.2f}s)")

    if args.json:
        with open(args.json, "w") as f:
            json.dump({"videos": vinfo, "rows": rows}, f, indent=2, default=float)
        print(f"\nWrote {args.json}")

    if args.write_trim:
        write_trim(args.trim_json, rows, trim)

    return 0


def write_trim(path, rows, old):
    """Rewrite trim_points.json so each cut starts where the plotted data starts.

    start_s is the first trajectory command and end_s the last state sample, both
    in video seconds. The premove is deliberately excluded: it is ~9.4s during
    which the joints travel ~0.0002 rad, so the robot is visibly stationary and
    the first trajectory command is the moment it starts to move.
    """
    print()
    print("=" * 108)
    print("REWRITING TRIM POINTS FROM THE LOGGED CLOCK")
    print("=" * 108)

    new = {}
    problems = []
    print(f"{'pt':>3} {'video':<24} {'old start':>9} -> {'new start':>9} "
          f"{'shift':>7} | {'old end':>8} -> {'new end':>8} {'dur':>7}")
    for r in sorted(rows, key=lambda r: r["point"]):
        v = r["video"]
        start_s = r["plot_in"]
        end_s = r["plot_out"]

        if start_s < 0:
            problems.append(f"pt {r['point']}: motion starts {start_s:.2f}s "
                            f"before the recording; clamped to 0")
            start_s = 0.0
        if end_s > r["video_duration"]:
            problems.append(f"pt {r['point']}: motion ends {end_s:.2f}s but the "
                            f"recording is only {r['video_duration']:.2f}s; clamped")
            end_s = r["video_duration"]

        o = old.get(v, {})
        new[v] = {"start_s": round(start_s, 2), "end_s": round(end_s, 2),
                  "duration_s": round(end_s - start_s, 2)}
        shift = start_s - o.get("start_s", float("nan"))
        print(f"{r['point']:>3} {v:<24} {o.get('start_s', float('nan')):>9.2f} -> "
              f"{start_s:>9.2f} {shift:>+7.2f} | "
              f"{o.get('end_s', float('nan')):>8.2f} -> {end_s:>8.2f} "
              f"{end_s - start_s:>7.2f}")

    if os.path.exists(path):
        bak = path + ".bak"
        with open(path) as f:
            prev = f.read()
        with open(bak, "w") as f:
            f.write(prev)
        print(f"\nBacked up previous trim points to {bak}")

    with open(path, "w") as f:
        json.dump(new, f, indent=2)
    print(f"Wrote {path} ({len(new)} entries)")

    if problems:
        print("\nClamped:")
        for p in problems:
            print("   " + p)


if __name__ == "__main__":
    sys.exit(main())
