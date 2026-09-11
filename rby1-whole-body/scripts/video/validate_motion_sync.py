"""Validate the video time base by comparing detected motion against the logs.

check_video_sync.py derives where each seed's motion sits inside its recording
from two clocks: the execution record's absolute timestamps, and
recording_start = creation_time - duration. That second clock is an assumption
about what a Samsung phone writes into the container, and the trim depends on it
to sub-second accuracy, so it needs checking against the pixels.

This measures motion onset/offset directly from each video and reports the
residual against the logged window. What matters is not that the residual is
zero -- the robot accelerates from rest, so a detector always fires late -- but
that it is *consistent* across seeds. A tight spread means the time base is
sound and the offset is physical; a wide spread means the time base is wrong.

Detection differs from detect_motion.py in three ways that matter for measuring
rather than merely finding motion:
  * the ROI is the crop the video actually shows, not a 70% centre box;
  * the threshold is per-video, from the profile's own noise floor, instead of a
    fixed 2.5 that one seed failed to cross at all;
  * onset requires MIN_RUN consecutive samples above it, so a single frame of
    camera shake or someone walking past cannot define the start.

Usage:
    .venv/bin/python scripts/video/validate_motion_sync.py
    .venv/bin/python scripts/video/validate_motion_sync.py --sample-fps 10
"""

import argparse
import json
import os
import subprocess
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# The region the composited video actually shows, as the docstring says -- so it
# follows hardware_framing.HW_CROP rather than repeating a literal that silently
# stops matching when the crop is retuned. Measured both ways when the crop
# changed on 2026-08-17: the residual spreads are the same to within 1% (onset sd
# 8.11 s vs 8.18 s, offset sd 3.49 s vs 3.51 s), so this coupling costs nothing.
from hardware_framing import HW_CROP  # noqa: E402
ROI = f"crop={HW_CROP}"

SAMPLE_FPS = 10.0
# Decode straight to small grayscale frames -- ffmpeg's scaler both downsamples
# and low-passes, which replaces the Gaussian blur and is far faster than
# pulling full 1080p frames into Python.
PROF_W, PROF_H = 240, 160

# The robot is moving through most of each recording, so the profile's median is
# a *moving* frame, not the noise floor. Anchor the floor at a low percentile and
# set the threshold as a fraction of the profile's dynamic range.
FLOOR_PCTL = 10.0
PEAK_PCTL = 95.0
RANGE_FRAC = 0.12
MIN_FLOOR_MARGIN = 0.25
MIN_RUN = 3


def motion_profile(path, sample_fps):
    """Mean abs frame difference over the shown ROI, sampled at sample_fps."""
    cmd = [
        "ffmpeg", "-v", "error", "-i", path,
        "-vf", f"{ROI},fps={sample_fps},scale={PROF_W}:{PROF_H},format=gray",
        "-f", "rawvideo", "-",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed on {path}: {proc.stderr.decode()[-300:]}")

    frame_bytes = PROF_W * PROF_H
    n = len(proc.stdout) // frame_bytes
    frames = np.frombuffer(proc.stdout[:n * frame_bytes], dtype=np.uint8)
    frames = frames.reshape(n, PROF_H, PROF_W).astype(np.int16)

    diffs = np.abs(np.diff(frames, axis=0)).mean(axis=(1, 2))
    times = (np.arange(1, n) / sample_fps)
    return times, diffs


def onset_offset(times, diffs):
    """First and last sustained excursion above a dynamic-range threshold."""
    if len(diffs) == 0:
        return None, None, None

    floor = np.percentile(diffs, FLOOR_PCTL)
    peak = np.percentile(diffs, PEAK_PCTL)
    thresh = max(floor + RANGE_FRAC * (peak - floor), floor + MIN_FLOOR_MARGIN)

    moving = diffs > thresh
    # Require MIN_RUN consecutive samples so a single spike cannot set the edge.
    runs = np.convolve(moving.astype(int), np.ones(MIN_RUN, dtype=int), mode="valid")
    sustained = runs >= MIN_RUN
    if not sustained.any():
        return None, None, thresh

    first = int(np.argmax(sustained))
    last = len(sustained) - 1 - int(np.argmax(sustained[::-1]))
    return times[first], times[last + MIN_RUN - 1], thresh


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hw-dir", default=os.path.join(REPO, "videos"))
    ap.add_argument("--sync-json", default=None,
                    help="output of check_video_sync.py --json (recomputed if omitted)")
    ap.add_argument("--sample-fps", type=float, default=SAMPLE_FPS)
    ap.add_argument("--json", default=None)
    ap.add_argument("--figure", default=os.path.join(REPO, "video", "motion_sync.png"),
                    help="20-panel motion profile figure with the logged window overlaid")
    args = ap.parse_args()

    if args.sync_json:
        with open(args.sync_json) as f:
            sync = json.load(f)
    else:
        tmp = os.path.join(REPO, "video", "_sync_tmp.json")
        subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__),
                                                     "check_video_sync.py"),
                        "--json", tmp], check=True, stdout=subprocess.DEVNULL)
        with open(tmp) as f:
            sync = json.load(f)
        os.remove(tmp)

    rows = {r["point"]: r for r in sync["rows"]}

    print(f"{'pt':>3} {'video':<24} {'plot in':>8} {'onset':>7} {'resid':>7} | "
          f"{'plot out':>8} {'offset':>7} {'resid':>7} | {'thr':>5}")
    out = []
    profiles = {}
    for pt in sorted(rows):
        r = rows[pt]
        path = os.path.join(args.hw_dir, r["video"])
        times, diffs = motion_profile(path, args.sample_fps)
        onset, offset, thresh = onset_offset(times, diffs)
        profiles[pt] = (times, diffs, thresh, onset, offset, r)

        res_in = (onset - r["plot_in"]) if onset is not None else float("nan")
        res_out = (offset - r["plot_out"]) if offset is not None else float("nan")
        out.append({"point": pt, "video": r["video"], "onset": onset,
                    "offset": offset, "plot_in": r["plot_in"],
                    "plot_out": r["plot_out"], "resid_in": res_in,
                    "resid_out": res_out, "threshold": thresh})

        o_s = f"{onset:7.2f}" if onset is not None else "   none"
        f_s = f"{offset:7.2f}" if offset is not None else "   none"
        print(f"{pt:>3} {r['video']:<24} {r['plot_in']:>8.2f} {o_s} {res_in:>+7.2f} | "
              f"{r['plot_out']:>8.2f} {f_s} {res_out:>+7.2f} | {thresh:>5.2f}",
              flush=True)

    ri = np.array([o["resid_in"] for o in out], dtype=float)
    ro = np.array([o["resid_out"] for o in out], dtype=float)
    ri, ro = ri[~np.isnan(ri)], ro[~np.isnan(ro)]

    print("\n" + "=" * 92)
    print("RESIDUALS  (detected motion minus the logged window; consistency is "
          "what validates the time base)")
    print("=" * 92)
    for name, a in (("onset - plot_in", ri), ("offset - plot_out", ro)):
        if len(a):
            print(f"{name:<20} mean {a.mean():+6.2f}s  median {np.median(a):+6.2f}s  "
                  f"sd {a.std():5.2f}s  min {a.min():+6.2f}s  max {a.max():+6.2f}s")

    if args.figure:
        plot_profiles(profiles, args.figure)
        print(f"\nWrote {args.figure}")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2, default=float)
        print(f"Wrote {args.json}")
    return 0


def plot_profiles(profiles, out_path):
    """One panel per seed: the motion profile with the logged window overlaid.

    The point of the figure is that the logged window (green) should bracket the
    raised part of the profile. That is the check that cannot be faked by a
    threshold choice.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(profiles)
    ncol = 4
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 2.4 * nrow))
    axes = np.atleast_1d(axes).ravel()

    for ax, pt in zip(axes, sorted(profiles)):
        times, diffs, thresh, onset, offset, r = profiles[pt]
        ax.plot(times, diffs, lw=0.8, color="#333333")
        ax.axhline(thresh, color="#999999", lw=0.7, ls=":")
        ax.axvspan(r["plot_in"], r["plot_out"], color="#66bb6a", alpha=0.22,
                   label="logged window")
        if onset is not None:
            ax.axvline(onset, color="#ef5350", lw=1.0, ls="--", label="detected")
        if offset is not None:
            ax.axvline(offset, color="#ef5350", lw=1.0, ls="--")
        ax.set_title(f"pt {pt}  {r['video'][9:15]}", fontsize=9)
        ax.tick_params(labelsize=7)
        ax.set_xlim(0, times[-1] if len(times) else 1)
    for ax in axes[n:]:
        ax.axis("off")
    axes[0].legend(fontsize=7, loc="upper right")
    fig.suptitle("Frame-difference motion profile vs the window derived from the "
                 "execution logs", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=110, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
