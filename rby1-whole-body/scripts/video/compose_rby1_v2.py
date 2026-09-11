"""Compose the RB-Y1 experiment segment for Video 2.

Shows seed 0 side-by-side (Drake sim + hardware) at 2x, followed by the
existing 20-point montage.

Usage:
    .venv/bin/python scripts/video/compose_rby1_v2.py
"""

import argparse
import json
import os
import subprocess
import sys

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hardware_framing import HW_CROP, PANE_W, PANE_H  # noqa: E402
from ral_explainers import results_panel  # noqa: E402
# The shared "what is a grid point" card for the side-by-side beat's inset
# -- see montage_callouts' module docstring for why it lives there rather
# than in ral_explainers.py or compose_montage.py. This cut used to also
# import a notes panel and a tile-outline/arrow overlay from that module for
# the montage tail; both are gone now that the montage tail is just
# [montage][results panel] (see normalize_montage below), and the "what is a
# grid point" explanation they used to carry moved to this inset instead.
from montage_callouts import render_grid_inset, OVERVIEW_INSET_X, OVERVIEW_INSET_Y  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
VIDEO_DIR = os.path.join(REPO, "video")

BG_HEX = "0x1a1a2e"
FPS = 30
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

SEED = 0
SPEED = 2.0

# How much longer the panel-filled montage plays than the raw montage clip,
# so the panels are actually readable (6 s of montage -> 9 s here).
MONTAGE_HOLD_S = 3.0

# When the "what is a grid point" inset appears in the side-by-side beat.
# See the comment on its overlay filter below for the measurement behind it.
INSET_FADE_IN_S = 2.6


def build_seed_map(hw_dir):
    files = sorted(f for f in os.listdir(hw_dir) if f.endswith(".mp4"))
    return {i: f for i, f in enumerate(files)}


def _panel_width(montage_path):
    """1920 minus the montage's own width, read from the file with ffprobe
    rather than hard-coded -- duplicated (not imported) from
    render_segment2.py's own ``_panel_width``, the same shape/not-imported
    trade-off ``normalize_montage`` below already makes against that
    module's version of the same fill."""
    w = int(subprocess.check_output([
        "ffprobe", "-v", "quiet", "-show_entries", "stream=width",
        "-of", "csv=p=0", montage_path,
    ]).decode().strip())
    return 1920 - w


def normalize_montage(src, dst, right_png, hold_s=MONTAGE_HOLD_S):
    """Fill the montage's one dead letterbox bar (right only) with the
    results panel, then hold on the last frame for `hold_s` extra seconds so
    the panel is actually readable.

    Copied in shape (not imported) from
    scripts/video/render_segment2.py's normalize_montage(), the RA-L pass's
    version of the same fill -- importing it would drag its 500-line
    matplotlib-driven renderer onto the overview path for what is otherwise
    ~15 lines of ffmpeg. `tpad` runs after `fps` so the cloned hold frames
    land at 30 fps, and the fade-out start is computed from the *padded*
    duration, not the original 6 s clip -- the same ordering trap
    normalize_segment() below has to avoid.

    Used to also take a `left_png` (a notes panel) and an `annotation_png`
    (tile outlines/arrows, composited full-frame before the fps/tpad/fade
    chain so it froze into the held tail along with everything else). Both
    are gone along with the left bar they filled -- see the module docstring
    and montage_callouts.py's.
    """
    dur = float(subprocess.check_output([
        "ffprobe", "-v", "quiet", "-show_entries", "format=duration",
        "-of", "csv=p=0", src,
    ]).decode().strip())
    padded_dur = dur + hold_s
    fade_out = max(0, padded_dur - 0.3)
    filter_complex = (
        f"[0:v][1:v]hstack=inputs=2,"
        f"fps={FPS},"
        f"tpad=stop_mode=clone:stop_duration={hold_s},"
        f"fade=in:0:9,fade=out:st={fade_out:.2f}:d=0.3[out]"
    )
    cmd = [
        "ffmpeg", "-y",
        "-i", src,
        "-loop", "1", "-t", f"{dur:.2f}", "-i", right_png,
        "-filter_complex", filter_complex,
        "-map", "[out]",
        "-c:v", "libx264", "-crf", "20", "-preset", "medium",
        "-pix_fmt", "yuv420p", "-an",
        dst,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"normalize_montage failed:\n{result.stderr[-800:]}")
        sys.exit(1)
    return dst


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hw-dir", default=os.path.join(REPO, "videos"))
    ap.add_argument("--trim-json", default=os.path.join(REPO, "videos", "trim_points.json"))
    ap.add_argument("--drake-video", default=os.path.join(VIDEO_DIR, f"drake_blender_{SEED:02d}.mp4"))
    ap.add_argument("--montage", default=os.path.join(VIDEO_DIR, "montage_20.mp4"))
    ap.add_argument("--output", default=os.path.join(VIDEO_DIR, "v2_rby1_hardware.mp4"))
    ap.add_argument("--target-duration", type=float, default=15.0,
                    help="Target duration for the side-by-side portion")
    args = ap.parse_args()

    with open(args.trim_json) as f:
        trim = json.load(f)

    seed_map = build_seed_map(args.hw_dir)
    hw_file = seed_map[SEED]
    hw_path = os.path.join(args.hw_dir, hw_file)
    t = trim.get(hw_file, {})
    start_s = t.get("start_s", 0)
    end_s = t.get("end_s", None)
    hw_dur = t.get("duration_s", 40.0)

    video_dur = min(hw_dur / SPEED, args.target_duration)

    scratch = os.path.join(VIDEO_DIR, "scratch_rby1_v2")
    os.makedirs(scratch, exist_ok=True)

    # Part 1: side-by-side Drake + hardware
    sbs_path = os.path.join(scratch, "side_by_side.mp4")

    # "What is a grid point" inset for the sim pane's empty upper-left --
    # see montage_callouts.render_grid_inset for the placement measurement
    # and its caveat about the robot's arm sweeping into that corner for
    # part of the beat.
    grid_inset_png = os.path.join(VIDEO_DIR, "v2_grid_inset.png")
    render_grid_inset(grid_inset_png)

    inputs = []
    if os.path.exists(args.drake_video):
        inputs.extend(["-i", args.drake_video])
    else:
        print(f"Warning: Drake video not found at {args.drake_video}, using hardware only")
        inputs.extend(["-i", hw_path])

    if end_s:
        inputs.extend(["-ss", str(start_s), "-to", str(end_s)])
    inputs.extend(["-i", hw_path])
    inputs.extend(["-loop", "1", "-t", f"{video_dur:.2f}", "-i", grid_inset_png])

    drake_speed = 1.0
    if os.path.exists(args.drake_video):
        drake_dur = float(subprocess.check_output([
            "ffprobe", "-v", "quiet", "-show_entries", "format=duration",
            "-of", "csv=p=0", args.drake_video,
        ]).decode().strip())
        drake_speed = drake_dur / hw_dur * SPEED

    # Both halves are composed at exactly the same pane size and only then
    # stacked and padded ONCE to 1920x1080. Previously each pane was
    # letterboxed into 960x1080 independently, so the 16:9 sim render settled
    # at 960x540 while the hardware crop settled at 960x640: two content bands
    # of different heights, each centred in its own pane, visibly misaligned.
    # The pane and the sim render size both come from hardware_framing, so they
    # cannot drift apart.
    filter_str = (
        f"[0:v]scale={PANE_W}:{PANE_H}:force_original_aspect_ratio=decrease,"
        f"pad={PANE_W}:{PANE_H}:(ow-iw)/2:(oh-ih)/2:color={BG_HEX},"
        f"setpts=PTS/{drake_speed:.3f},"
        f"drawtext=text='Simulation (Drake)':"
        f"fontfile={FONT_PATH}:fontsize=30:fontcolor=#4fc3f7:"
        f"borderw=1:bordercolor=black:x=10:y=10[drake];"
        f"[1:v]crop={HW_CROP},"
        f"scale={PANE_W}:{PANE_H}:force_original_aspect_ratio=decrease,"
        f"pad={PANE_W}:{PANE_H}:(ow-iw)/2:(oh-ih)/2:color={BG_HEX},"
        f"setpts=PTS/{SPEED:.1f},"
        f"drawtext=text='Hardware':"
        f"fontfile={FONT_PATH}:fontsize=30:fontcolor=#66bb6a:"
        f"borderw=1:bordercolor=black:x=10:y=10[hw];"
        f"[drake][hw]hstack=inputs=2,"
        f"drawtext=text='{SPEED:.0f}×':"
        f"fontfile={FONT_PATH}:fontsize=28:fontcolor=white:"
        f"borderw=2:bordercolor=black:x=w-tw-16:y=12,"
        f"pad=1920:1080:(ow-iw)/2:(oh-ih)/2:color={BG_HEX}[padded];"
        # The inset overlay goes here, after the 1920x1080 pad, so
        # OVERVIEW_INSET_X/OVERVIEW_INSET_Y are frame coordinates rather
        # than pane-local ones -- it has to be its own filter statement (not
        # chained with a comma) because overlay is the first filter in this
        # graph that takes a second, explicit input pad. (Named
        # OVERVIEW_*, not OVERLAY_X/Y, because montage_callouts.
        # render_grid_inset now has a second caller -- the RA-L cut's
        # ral_annotations.py -- with its own, different placement; this
        # name says which one these constants belong to.)
        # ...and it fades in at INSET_FADE_IN_S rather than being up from
        # frame 0. Measured on the Drake render, not assumed: sampling the
        # inset's rectangle every 0.5 s of the beat and counting pixels that
        # differ from the pane's flat background puts the robot's raised arm
        # inside that corner for the first ~2.5 s (3.5% of the rect), clear
        # from ~3 s, and back to ~4-6% after t~9.5 s. The late overlap is
        # background and the outer edge of the carried box, which the panel
        # can sit over; the early one clips the raised arm itself, which it
        # should not -- so the panel simply arrives after the arm has come
        # down and then stays for the rest of the beat.
        f"[2:v]format=rgba,fade=in:st={INSET_FADE_IN_S}:d=0.4:alpha=1[inset];"
        f"[padded][inset]overlay={OVERVIEW_INSET_X}:{OVERVIEW_INSET_Y}[withinset];"
        # The bottom 85px bar already carries the "20/20" line below; this is
        # the empty top 85px bar, one centred line naming which grid point
        # this is -- SEED, not a hardcoded 0, so it can't drift if SEED changes.
        f"[withinset]drawtext=text='Grid point {SEED} of 20 — the same plan, "
        f"in simulation and on the robot':"
        f"fontfile={FONT_PATH}:fontsize=28:fontcolor=#e0e0e0:"
        f"borderw=1:bordercolor=black:x=(w-tw)/2:y=26,"
        f"drawtext=text='RB-Y1 Humanoid\\: 20/20 Pick-and-Place':"
        f"fontfile={FONT_PATH}:fontsize=32:fontcolor=white:"
        f"borderw=1:bordercolor=black:x=(w-tw)/2:y=h-80"
        f"[out]"
    )

    cmd = [
        "ffmpeg", "-y",
        *inputs,
        "-filter_complex", filter_str,
        "-map", "[out]",
        "-c:v", "libx264", "-crf", "22", "-preset", "medium",
        "-pix_fmt", "yuv420p",
        "-t", f"{video_dur:.2f}",
        "-an",
        sbs_path,
    ]
    print("Rendering side-by-side...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ffmpeg error:\n{result.stderr[-800:]}")
        sys.exit(1)

    # Part 2: normalize and concatenate with montage
    parts = [sbs_path]
    if os.path.exists(args.montage):
        parts.append(args.montage)

    # Name distinct from the RA-L pass's montage_results_panel.png so the two
    # video builds can't race over one file if they ever run concurrently --
    # both cuts now fill the montage's one remaining bar with the same panel
    # content (ral_explainers.results_panel), just rendered at each cut's own
    # derived width.
    results_panel_png = os.path.join(VIDEO_DIR, "v2_montage_results_panel.png")

    norm_parts = []
    for i, p in enumerate(parts):
        norm_p = os.path.join(scratch, f"norm_{i}.mp4")
        if p == args.montage:
            panel_w = _panel_width(p)
            Image.fromarray(results_panel(size=(panel_w, 1080))).save(results_panel_png)
            normalize_montage(p, norm_p, results_panel_png)
        else:
            dur = float(subprocess.check_output([
                "ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                "-of", "csv=p=0", p,
            ]).decode().strip())
            fade_out = max(0, dur - 0.3)
            cmd = [
                "ffmpeg", "-y", "-i", p,
                "-vf", (
                    f"scale=1920:1080:force_original_aspect_ratio=decrease,"
                    f"pad=1920:1080:(ow-iw)/2:(oh-ih)/2:color={BG_HEX},"
                    f"fps={FPS},"
                    f"fade=in:0:9,fade=out:st={fade_out:.2f}:d=0.3"
                ),
                "-c:v", "libx264", "-crf", "20", "-preset", "medium",
                "-pix_fmt", "yuv420p", "-an",
                norm_p,
            ]
            subprocess.run(cmd, capture_output=True, text=True, check=True)
        norm_parts.append(norm_p)

    concat_file = os.path.join(scratch, "concat.txt")
    with open(concat_file, "w") as f:
        for p in norm_parts:
            f.write(f"file '{os.path.abspath(p)}'\n")

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", concat_file,
        "-c:v", "libx264", "-crf", "20", "-preset", "medium",
        "-pix_fmt", "yuv420p",
        args.output,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Concat failed:\n{result.stderr[-500:]}")
        sys.exit(1)

    dur = float(subprocess.check_output([
        "ffprobe", "-v", "quiet", "-show_entries", "format=duration",
        "-of", "csv=p=0", args.output,
    ]).decode().strip())
    print(f"Wrote {args.output} ({dur:.1f}s)")


if __name__ == "__main__":
    main()
