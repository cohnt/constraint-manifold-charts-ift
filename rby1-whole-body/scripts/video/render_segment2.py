"""Render Segment 2: all 20 trajectories in groups of 4, then the montage.

Part A: 5 groups × 4 points at 2x speed, each with compact error plots.
Part B: existing montage_20.mp4 appended as a closing summary, its one dead
        letterbox bar (right only -- see ``normalize_montage``) filled with
        the results panel from ``ral_explainers``.

Layout per group (1920x1080):
  Top row (4×480x540): hardware videos labeled "Point N"
  Band (1920x72): two centred lines defining the constraint-violation metric
    and its units, stated once for the whole group (see BAND_LINE1/BAND_LINE2)
  Bottom row (4×480x468): compact constraint violation plot per point

Usage:
    .venv/bin/python scripts/video/render_segment2.py
"""

import argparse
import concurrent.futures as cf
import json
import multiprocessing as mp
import os
import pickle
import subprocess
import sys
import time

import numpy as np
from PIL import Image

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

BG_HEX = "0x1a1a2e"
SPEED = 2.0
CELL_W, CELL_H = 480, 540
# The metric definition (what "constraint violation" means, its units, which
# colour is which) is stated once, on a band between the video row and the
# plot row, instead of repeated in four cell titles. Its height comes out of
# the plot row rather than the video row: the hardware frames are the thing the
# viewer is actually watching, and the plots reflow to whatever height they get.
# 72px carries two lines (24pt + 20pt) instead of one; see BAND_LINE1/2 below.
BAND_H = 72
VIDEO_H = CELL_H
PLOT_H = CELL_H - BAND_H
GRID_NX = 4
FPS = 30
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import plan_format  # noqa: F401  -- installs the pre-rename module aliases the
                    # committed plan and execution pickles were written with
from frame_parallel import frame_workers  # noqa: E402
from hardware_framing import HW_CROP  # noqa: E402  (see that module)
from ral_explainers import results_panel  # noqa: E402
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

# Sized for 720p viewing: each 480px cell becomes 320px wide, a 0.667 scale, so
# anything under ~13pt here drops below 9pt effective and stops being legible.
FS_TITLE = 15
FS_LABEL = 12
FS_TICK = 11
FS_LEGEND = 10
FS_NOTE = 10

COL_POS = "#ff7043"
COL_ROT = "#4fc3f7"

# The metric is defined once here instead of in each cell title (see the cell
# title comment below). Both lines were measured with PIL's font metrics at
# 1920px before being accepted: line 1 is 1133px, line 2 is 1239px ("coral" ->
# "red" per Tommy's review), both with room to spare against the 1920px band
# width.
BAND_LINE1 = ("Constraint violation: drift of the gripper-to-gripper pose "
              "from its value at the grasp")
BAND_LINE2 = ("position (mm, red) and orientation (mrad, blue) are "
              "reported separately — planned ≡ 0 — all videos "
              "{speed:.0f}× speed").format(speed=SPEED)


def _escape(text):
    """Escape a literal string for ffmpeg's drawtext= option.

    Matches the pattern in render_with_blender.py: backslash first, then the
    characters drawtext and the filter-graph parser each treat specially. The
    band text below has a colon and several commas that need this.
    """
    for ch in ("\\", ":", "'", "%", ",", "[", "]", ";"):
        text = text.replace(ch, "\\" + ch)
    return text


# How many 1080p ffmpeg jobs to run at once. Each is already multi-threaded, so
# this is deliberately far below the core count.
GROUP_JOBS = 5


# Read in the pool workers through fork inheritance; the error data is a few
# hundred MB of arrays and there is no reason to pickle it 20 times.
_ALL_ERRORS = None


def _plot_task(args):
    """One point's plot video.  Module level so a pool can dispatch it."""
    pt, out_path, duration_s = args
    render_error_plot_video(pt, _ALL_ERRORS[pt], out_path, duration_s, SPEED)
    return pt, out_path


def build_seed_map(hw_dir):
    files = sorted(f for f in os.listdir(hw_dir) if f.endswith(".mp4"))
    return {i: f for i, f in enumerate(files)}


def render_error_plot_video(point_idx, pt_data, out_path, duration_s, speed):
    """Render a compact CV plot video for one point (constrained legs only)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    from PIL import Image

    BG_COLOR = "#1a1a2e"
    TEXT_COLOR = "#e0e0e0"
    ACCENT_GREEN = "#66bb6a"
    ACCENT_YELLOW = "#ffd54f"
    ACCENT_CORAL = "#ff7043"

    steps = pt_data["steps"]
    pos_t, pos_v, rot_t, rot_v = [], [], [], []
    plan_t, plan_v = [], []

    for s in steps:
        if not s.get("is_constrained", False):
            continue
        # State logging starts a fixed 1.50s before the leg's first command;
        # restrict the measured trace to the commanded window (see
        # render_segment1.py).
        t_actual, t_cmd = s["t_actual"], s["t_cmd"]
        m = (t_actual >= t_cmd[0]) & (t_actual <= t_cmd[-1])
        pos_t.append(t_actual[m])
        pos_v.append(s["cv_pos_measured"][m])
        rot_t.append(t_actual[m])
        rot_v.append(s["cv_rot_measured"][m])
        # cv_*_planned is on t_planned (the trajectory, sampled denser than the
        # commands), NOT on t_cmd -- pairing it with t_cmd raises an IndexError
        # on the visibility mask because the two lengths differ.
        t_p = s.get("t_planned")
        if t_p is not None and len(t_p) == len(s["cv_pos_planned"]):
            plan_t.append(t_p + (t_cmd[0] if len(t_cmd) else 0.0))
            plan_v.append(s["cv_pos_planned"])

    cat = lambda xs: np.concatenate(xs) if xs else np.array([])
    MM, MRAD = 1e3, 1e3
    pos_t, pos_v = cat(pos_t), cat(pos_v) * MM
    rot_t, rot_v = cat(rot_t), cat(rot_v) * MRAD
    plan_t, plan_v = cat(plan_t), cat(plan_v) * MM

    # Measured position and orientation on twin axes, plus the planned trace,
    # which at ~1e-12 mm draws as a flat line on zero. Carrying it as a real
    # trace with a legend entry beats a free-floating caption: an earlier version
    # placed that caption bottom-centre, where the measured traces cross straight
    # through it every time tracking is briefly good. The legend cannot collide.
    y_pos = max(float(pos_v.max()) if len(pos_v) else 1.0, 1e-3) * 1.50
    y_rot = max(float(rot_v.max()) if len(rot_v) else 1.0, 1e-3) * 1.50

    n_frames = int(duration_s * FPS) + 1

    fig, ax = plt.subplots(figsize=(CELL_W / 100, PLOT_H / 100), dpi=100)
    twin = ax.twinx()
    fig.patch.set_facecolor(BG_COLOR)
    fig.subplots_adjust(left=0.19, right=0.81, top=0.86, bottom=0.14)

    # Everything that does not depend on the frame is built once. The loop used
    # to clear both axes and re-create every artist -- title, labels, spines,
    # grid, three lines and the legend -- per frame, which is most of the cost
    # of this pass; only the line data and the x limits actually change.
    ax.set_facecolor(BG_COLOR)
    # Kept short on purpose. The cells are CELL_W wide and the title is
    # centred on the whole cell, so anything wider than about 400 px at
    # FS_TITLE bleeds into the neighbouring panel's title -- which is
    # exactly what "Point N — constraint violation, measured" (479 px)
    # did. The metric itself ("constraint violation", its units, which
    # colour is which) is now defined once in the group's speed band
    # (BAND_LINE1/2), so the title only has to say what this cell is:
    # "Point 19 — measured" measures 180px at FS_TITLE, comfortably under
    # the 480px cell.
    ax.set_title(f"Point {point_idx} — measured",
                 color=TEXT_COLOR, fontsize=FS_TITLE, fontweight="bold", pad=6)
    ax.set_ylabel("position (mm)", color=COL_POS, fontsize=FS_LABEL)
    ax.set_xlabel("time (s)", color=TEXT_COLOR, fontsize=FS_LABEL)
    twin.yaxis.set_label_position("right")
    twin.yaxis.tick_right()
    twin.set_ylabel("orientation (mrad)", color=COL_ROT, fontsize=FS_LABEL)
    ax.tick_params(axis="y", colors=COL_POS, labelsize=FS_TICK)
    ax.tick_params(axis="x", colors=TEXT_COLOR, labelsize=FS_TICK)
    twin.tick_params(axis="y", colors=COL_ROT, labelsize=FS_TICK)
    ax.spines["bottom"].set_color(TEXT_COLOR)
    ax.spines["left"].set_color(COL_POS)
    ax.spines["top"].set_visible(False)
    twin.spines["right"].set_color(COL_ROT)
    twin.spines["top"].set_visible(False)
    twin.spines["left"].set_visible(False)
    ax.grid(True, alpha=0.15, color=TEXT_COLOR)
    # The x window grows every frame, so the tick locator runs every frame too.
    # AutoLocator sizes its tick budget from a measurement taken during the
    # draw, and that measurement does not come out the same on a re-used axes
    # as on one that was just cleared -- reusing the artists silently moved a
    # stretch of frames from 1 s to 2 s ticks. Stating the budget (7, what the
    # cleared axes resolved to) makes the tick spacing a property of the script
    # instead of a side effect of how often it is redrawn.
    ax.xaxis.set_major_locator(MaxNLocator(nbins=7, steps=[1, 2, 2.5, 5, 10]))

    # Two artists per trace, as before: an empty, fully opaque line that exists
    # only to give the legend its swatch, and the alpha-0.9 line that carries
    # the data. Keeping them separate is what makes the legend independent of
    # how much of the trace has been revealed.
    handles, traces = [], []
    for target, x, y, color, label, lw in (
            (ax, pos_t, pos_v, COL_POS, "measured position", 1.4),
            (twin, rot_t, rot_v, COL_ROT, "measured orientation", 1.4),
            (ax, plan_t, plan_v, ACCENT_GREEN, "planned ≡ 0", 1.6)):
        h, = target.plot([], [], color=color, linewidth=lw, label=label)
        handles.append(h)
        line, = target.plot([], [], color=color, linewidth=lw, alpha=0.9)
        traces.append((line, x, y))

    ax.set_ylim(0.0, y_pos)
    twin.set_ylim(0.0, y_rot)

    # 1.5x headroom above the peaks is what keeps the three-entry legend from
    # sitting on the data.
    ax.legend(handles=handles, fontsize=FS_LEGEND, loc="upper left",
              facecolor=BG_COLOR, edgecolor=TEXT_COLOR,
              labelcolor=TEXT_COLOR, framealpha=0.85)

    x0 = pos_t[0] - 0.5 if len(pos_t) else 0.0

    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{CELL_W}x{PLOT_H}", "-pix_fmt", "rgb24",
        "-r", str(FPS),
        "-i", "-",
        "-c:v", "libx264", "-crf", "22", "-preset", "fast",
        "-pix_fmt", "yuv420p", "-an",
        out_path,
    ]
    pipe = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    for frame_idx in range(n_frames):
        t_video = frame_idx / FPS
        t_traj = t_video * speed

        for line, x, y in traces:
            mask = x <= t_traj
            if mask.any():
                line.set_data(x[mask], y[mask])
            else:
                line.set_data([], [])

        x1 = max(t_traj + 1, x0 + 5)
        ax.set_xlim(x0, x1)
        twin.set_xlim(x0, x1)

        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        arr = np.asarray(buf)[:, :, :3].copy()

        img = Image.fromarray(arr).resize((CELL_W, PLOT_H), Image.LANCZOS)
        pipe.stdin.write(np.array(img).tobytes())

    pipe.stdin.close()
    pipe.wait()
    plt.close(fig)


def group_duration(point_indices, trim, seed_map, target_duration=None):
    """Seconds of finished video for one group: its longest clip, sped up.

    Split out of ``render_group`` because the plot videos have to be this exact
    length and are now rendered before any group runs.
    """
    hw_durs = [trim.get(seed_map[pt], {}).get("duration_s", 40.0)
               for pt in point_indices]
    video_dur = max(hw_durs) / SPEED
    if target_duration:
        video_dur = min(video_dur, target_duration)
    return video_dur


def render_group(group_idx, point_indices, trim, seed_map, hw_dir, out_path,
                 plot_paths, video_dur):
    """Render one group of 4 points using ffmpeg filter_complex."""
    inputs = []
    for pt in point_indices:
        fname = seed_map[pt]
        t = trim.get(fname, {})
        start = t.get("start_s", 0)
        end = t.get("end_s", None)
        path = os.path.join(hw_dir, fname)
        if end:
            inputs.extend(["-ss", str(start), "-to", str(end)])
        inputs.extend(["-i", path])

    for pp in plot_paths:
        inputs.extend(["-i", pp])

    filters = []
    n = len(point_indices)
    for i in range(n):
        filters.append(
            f"[{i}:v]crop={HW_CROP},"
            f"scale={CELL_W}:{VIDEO_H}:force_original_aspect_ratio=decrease,"
            f"pad={CELL_W}:{VIDEO_H}:(ow-iw)/2:(oh-ih)/2:color={BG_HEX},"
            f"setpts=PTS/{SPEED:.1f},"
            f"tpad=stop_mode=clone:stop_duration=30,"
            # Sized for 720p: these 480px cells become 320px, so 16px lettering
            # was ~11px on screen.
            f"drawtext=text='Point {point_indices[i]}':"
            f"fontfile={FONT_PATH}:"
            f"fontsize=26:fontcolor=white:borderw=2:bordercolor=black:"
            f"x=10:y=8[v{i}]"
        )

    for i in range(n):
        filters.append(f"[{n + i}:v]scale={CELL_W}:{PLOT_H}[p{i}]")

    top_inputs = "".join(f"[v{i}]" for i in range(n))
    filters.append(f"{top_inputs}hstack=inputs={n}[top]")

    bot_inputs = "".join(f"[p{i}]" for i in range(n))
    filters.append(f"{bot_inputs}hstack=inputs={n}[bot]")

    # The metric definition for the whole group, on two centred lines. The
    # colour source is given an explicit duration so the band is not the one
    # input that never reaches EOF. y offsets are fixed pixel values rather
    # than an expression, sized from the two fonts' measured heights (23px at
    # 24pt, 19px at 20pt) so the pair sits centred in the 72px band with an
    # even gap between them and the top/bottom edges.
    filters.append(
        f"color=c={BG_HEX}:s=1920x{BAND_H}:r={FPS}:d={video_dur:.2f},"
        f"drawtext=text='{_escape(BAND_LINE1)}':"
        f"fontfile={FONT_PATH}:"
        f"fontsize=24:fontcolor=white:"
        f"x=(w-tw)/2:y=10,"
        f"drawtext=text='{_escape(BAND_LINE2)}':"
        f"fontfile={FONT_PATH}:"
        f"fontsize=20:fontcolor=0xa0a0a0:"
        f"x=(w-tw)/2:y=43[band]"
    )

    filters.append("[top][band][bot]vstack=inputs=3[out]")

    filter_str = ";\n".join(filters)

    cmd = [
        "ffmpeg", "-y",
        *inputs,
        "-filter_complex", filter_str,
        "-map", "[out]",
        "-c:v", "libx264", "-crf", "22", "-preset", "medium",
        "-pix_fmt", "yuv420p",
        "-t", f"{video_dur:.2f}",
        "-an",
        out_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return False, f"  Group {group_idx} ffmpeg error:\n{result.stderr[-800:]}"

    dur = float(subprocess.check_output([
        "ffprobe", "-v", "quiet", "-show_entries", "format=duration",
        "-of", "csv=p=0", out_path,
    ]).decode().strip())
    # Returned rather than printed: the groups run concurrently, so printing
    # from inside would interleave five progress streams.
    return True, f"  Group {group_idx}: {dur:.1f}s"


def normalize_part(src, dst):
    """Letterbox one part to 1920x1080 at FPS, with the in/out fades."""
    dur = float(subprocess.check_output([
        "ffprobe", "-v", "quiet", "-show_entries", "format=duration",
        "-of", "csv=p=0", src,
    ]).decode().strip())
    fade_out = max(0, dur - 0.3)
    subprocess.run([
        "ffmpeg", "-y", "-i", src,
        "-vf", (
            f"scale=1920:1080:force_original_aspect_ratio=decrease,"
            f"pad=1920:1080:(ow-iw)/2:(oh-ih)/2:color={BG_HEX},"
            f"fps={FPS},"
            f"fade=in:0:9,fade=out:st={fade_out:.2f}:d=0.3"
        ),
        "-c:v", "libx264", "-crf", "20", "-preset", "medium",
        "-pix_fmt", "yuv420p", "-an",
        dst,
    ], capture_output=True, text=True, check=True)
    return dst


# The montage used to leave two dead letterbox bars (912px montage in a
# 1920px frame) and PANEL_W was a module-level constant splitting that gap
# two ways. It is now one bar only, on the right, and its width depends on
# the montage's own width (1420px at compose_montage.py's 5x4 default, but
# this module must not hard-code that) -- see _panel_width below.


def _panel_width(montage_path):
    """1920 minus the montage's own width, read from the file with ffprobe
    rather than hard-coded: the montage's width is compose_montage.py's to
    own (1420px at its current 5x4 default -- see that module's docstring
    for why 5x4 replaced 4x5), and this module must not silently disagree
    with it if that layout, or its default, ever changes again."""
    w = int(subprocess.check_output([
        "ffprobe", "-v", "quiet", "-show_entries", "stream=width",
        "-of", "csv=p=0", montage_path,
    ]).decode().strip())
    return 1920 - w


def normalize_montage(src, dst, right_png):
    """Fill the montage's one dead letterbox bar (now on the right only) with
    the results panel.

    Special-cased rather than folded into ``normalize_part``: the montage is
    already exactly its own width x 1080 (no scale/pad needed, unlike every
    other part), and its flank comes from a static panel PNG, not a flat
    colour pad. Same fps/fade treatment as ``normalize_part`` so the two
    paths look alike at their seams.

    Used to also take a `left_png` for a "What Is a Point?" panel
    (ral_explainers.grid_panel) filling a second bar on the left. That panel,
    and the left bar itself, are gone: the montage widened from 912px to
    1420px when compose_montage.py's default layout changed from the
    physical 4x5 grid to the camera-matched 5x4 one, which left only one
    flank to fill -- and the "what is a grid point" explanation this panel
    used to carry now appears far earlier, on segment 1, via
    ral_annotations.py's grid inset, so it did not need a new home here.
    """
    dur = float(subprocess.check_output([
        "ffprobe", "-v", "quiet", "-show_entries", "format=duration",
        "-of", "csv=p=0", src,
    ]).decode().strip())
    fade_out = max(0, dur - 0.3)
    # [0]=montage video, [1]=right panel (looped still).
    filter_complex = (
        f"[0:v][1:v]hstack=inputs=2,"
        f"fps={FPS},fade=in:0:9,fade=out:st={fade_out:.2f}:d=0.3[out]"
    )
    subprocess.run([
        "ffmpeg", "-y",
        "-i", src,
        "-loop", "1", "-t", f"{dur:.2f}", "-i", right_png,
        "-filter_complex", filter_complex,
        "-map", "[out]",
        "-c:v", "libx264", "-crf", "20", "-preset", "medium",
        "-pix_fmt", "yuv420p", "-an",
        dst,
    ], capture_output=True, text=True, check=True)
    return dst


def run_concurrently(calls, workers):
    """Run ``calls`` (zero-argument callables) at once, results in order.

    These are ffmpeg jobs, so the work happens in child processes and threads
    are enough.  A handful at a time rather than all of them: each encoder is
    already multi-threaded, and oversubscribing turns a win into a wash.
    """
    if workers <= 1:
        return [c() for c in calls]
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        return [f.result() for f in [ex.submit(c) for c in calls]]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--error-data", default=os.path.join(REPO, "video", "error_data.pkl"))
    ap.add_argument("--hw-dir", default=os.path.join(REPO, "videos"))
    ap.add_argument("--trim-json", default=os.path.join(REPO, "videos", "trim_points.json"))
    ap.add_argument("--montage", default=os.path.join(REPO, "video", "montage_20.mp4"))
    ap.add_argument("--output", default=os.path.join(REPO, "video", "segment2_all20.mp4"))
    args = ap.parse_args()

    with open(args.error_data, "rb") as f:
        all_errors = pickle.load(f)

    with open(args.trim_json) as f:
        trim = json.load(f)

    seed_map = build_seed_map(args.hw_dir)

    groups = [list(range(i, min(i + 4, 20))) for i in range(0, 20, 4)]
    scratch_dir = os.path.join(REPO, "video", "scratch_seg2")
    os.makedirs(scratch_dir, exist_ok=True)

    durations = [group_duration(pts, trim, seed_map) for pts in groups]

    # All 20 plot videos first, in parallel. They are independent matplotlib
    # renders and used to run one after another inside each group, which left
    # this stage single-threaded for most of its wall time.
    global _ALL_ERRORS
    _ALL_ERRORS = all_errors
    tasks = [(pt, os.path.join(scratch_dir, f"plot_{pt:02d}.mp4"), durations[g])
             for g, pts in enumerate(groups) for pt in pts]
    workers = min(frame_workers(), len(tasks))
    print(f"Rendering {len(tasks)} error plots across {workers} processes...")
    t0 = time.time()
    if workers > 1:
        with mp.get_context("fork").Pool(workers) as pool:
            for pt, path in pool.imap_unordered(_plot_task, tasks):
                print(f"  point {pt}: {os.path.basename(path)}")
    else:
        for t in tasks:
            _plot_task(t)
    plot_of = {pt: path for pt, path, _ in tasks}
    print(f"  plots: {time.time() - t0:.1f}s")

    group_paths = [os.path.join(scratch_dir, f"group_{g}.mp4")
                   for g in range(len(groups))]
    for g_idx, pts in enumerate(groups):
        print(f"Group {g_idx}: points {pts}")
    t0 = time.time()
    results = run_concurrently([
        (lambda g=g_idx, pts=pts, out=out, dur=dur: render_group(
            g, pts, trim, seed_map, args.hw_dir, out,
            [plot_of[pt] for pt in pts], dur))
        for g_idx, (pts, out, dur) in enumerate(
            zip(groups, group_paths, durations))
    ], GROUP_JOBS)
    for ok, message in results:
        print(message)
    print(f"  groups: {time.time() - t0:.1f}s")
    # A failed group used to be ignored, which left the previous take of that
    # group on disk and concatenated it into the finished segment.
    if not all(ok for ok, _ in results):
        sys.exit(1)

    concat_parts = list(group_paths)
    montage_present = os.path.exists(args.montage)
    results_png = None
    if montage_present:
        concat_parts.append(args.montage)
        print(f"\nAppending montage: {args.montage}")

        # The montage's one dead letterbox bar (right only -- see
        # normalize_montage's docstring) becomes the paper's headline
        # numbers. Written under video/ as a build intermediate, alongside
        # this stage's other artifacts.
        panel_w = _panel_width(args.montage)
        results_png = os.path.join(REPO, "video", "montage_results_panel.png")
        Image.fromarray(results_panel(size=(panel_w, 1080))).save(results_png)

    concat_file = os.path.join(scratch_dir, "concat.txt")
    with open(concat_file, "w") as f:
        for p in concat_parts:
            f.write(f"file '{os.path.abspath(p)}'\n")

    norm_dir = os.path.join(scratch_dir, "norm")
    os.makedirs(norm_dir, exist_ok=True)
    t0 = time.time()

    def _norm_call(i, p):
        dst = os.path.join(norm_dir, f"{i:02d}.mp4")
        if montage_present and p == args.montage:
            return lambda: normalize_montage(p, dst, results_png)
        return lambda src=p, dst=dst: normalize_part(src, dst)

    norm_paths = run_concurrently(
        [_norm_call(i, p) for i, p in enumerate(concat_parts)], GROUP_JOBS)

    print(f"  normalize: {time.time() - t0:.1f}s")

    norm_concat = os.path.join(scratch_dir, "norm_concat.txt")
    with open(norm_concat, "w") as f:
        for p in norm_paths:
            f.write(f"file '{os.path.abspath(p)}'\n")

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", norm_concat,
        "-c:v", "libx264", "-crf", "20", "-preset", "medium",
        "-pix_fmt", "yuv420p",
        args.output,
    ]
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(f"  concat: {time.time() - t0:.1f}s")
    if result.returncode != 0:
        print(f"Final concat failed:\n{result.stderr[-500:]}")
        sys.exit(1)

    dur = float(subprocess.check_output([
        "ffprobe", "-v", "quiet", "-show_entries", "format=duration",
        "-of", "csv=p=0", args.output,
    ]).decode().strip())
    print(f"\nWrote {args.output} ({dur:.1f}s)")


if __name__ == "__main__":
    main()
