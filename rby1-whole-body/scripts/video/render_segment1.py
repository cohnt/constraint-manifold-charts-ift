"""Render Segment 1: single point at 1x with synchronized error plots.

Two-pass approach for speed:
  Pass 1: pre-render error plots as a standalone video (640x1080)
  Pass 2: ffmpeg composites hardware video + plot video side by side

Layout (1920x1080):
  Left 2/3 (1280x1080): hardware video
  Right 1/3 (640x1080): 2 stacked log/linear-scale plots

Usage:
    .venv/bin/python scripts/video/render_segment1.py
    .venv/bin/python scripts/video/render_segment1.py --point 5
"""

import argparse
import json
import os
import pickle
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

BG_COLOR = "#1a1a2e"
BG_HEX = "0x1a1a2e"
TEXT_COLOR = "#e0e0e0"
SUBTITLE_COLOR = "#a0a0a0"
ACCENT_BLUE = "#4fc3f7"
ACCENT_CORAL = "#ff7043"
ACCENT_GREEN = "#66bb6a"
ACCENT_YELLOW = "#ffd54f"

PLOT_W, PLOT_H = 640, 1080
FPS = 30
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import plan_format  # noqa: F401  -- installs the pre-rename module aliases the
                    # committed plan and execution pickles were written with
from frame_parallel import parallel_frames  # noqa: E402
from hardware_framing import HW_CROP  # noqa: E402  (see that module)
from ral_explainers import explainer_screen_a, explainer_screen_b  # noqa: E402
SPEED_LABEL = "1×"

# Font sizes are chosen so the plots stay readable when the finished 1920x1080
# video is viewed at 720p: the 640px-wide plot column becomes 427px, a 0.667
# scale, so anything below ~14pt here lands under 10pt effective and stops being
# legible. Sizes that looked fine at 1080p were roughly 2/3 of these.
FS_TITLE = 17
FS_SUBTITLE = 11
FS_LABEL = 14
FS_TICK = 12
FS_LEGEND = 12
FS_LEG_NAME = 9
FS_CAPTION = 9

# One colour per physical quantity, held constant across both panels, so the
# reader learns "coral = position, blue = orientation" once.
COL_POS = "#ff7043"
COL_ROT = "#4fc3f7"

# Short on-plot labels for the six legs, in the order the grid pipeline always
# produces them (plan_grid.py: reach_approach -> reach_descend
# -> lift -> place -> home_retreat -> home). Anything not in this map falls
# back to its raw step name so an unexpected leg still gets *a* label.
LEG_LABELS = {
    "reach_approach": "reach",
    "reach_descend": "descend",
    "lift": "lift",
    "place": "place",
    "home_retreat": "retreat",
    "home": "home",
}

FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def _escape(text):
    """Escape a literal string for ffmpeg's drawtext= option.

    Copied from render_with_blender.py's `_escape` (backslash first, then the
    characters drawtext and the filter-graph parser each treat specially) --
    the ribbon text now carries em dashes and could carry colons.
    """
    for ch in ("\\", ":", "'", "%", ",", "[", "]", ";"):
        text = text.replace(ch, "\\" + ch)
    return text


def build_seed_map(hw_dir):
    files = sorted(f for f in os.listdir(hw_dir) if f.endswith(".mp4"))
    return {i: f for i, f in enumerate(files)}


def collect_series(steps, key):
    """Gather planned / commanded / measured series for one metric key.

    key is "pos" or "rot". Only constrained legs (lift/place) carry a
    meaningful constraint, so only those contribute to the planned/commanded/
    measured series.

    Returns (planned_t, planned_v, cmd_t, cmd_v, meas_t, meas_v, legs):
      - legs: (t0, t1, name, is_constrained) for EVERY leg, constrained or not
        -- the superset callers need for leg-boundary vlines, leg-name labels
        and the grey unconstrained-span shading, so there is one source of
        truth for "what happened when" instead of a boundary list and a
        separate unconstrained list that could disagree.
    """
    planned_t, planned_v = [], []
    cmd_t, cmd_v = [], []
    meas_t, meas_v = [], []
    legs = []

    for s in steps:
        t_cmd, t_actual = s["t_cmd"], s["t_actual"]
        name = s.get("name", "")
        is_constrained = s.get("is_constrained", False)

        if len(t_cmd) > 0:
            interval_end = t_actual[-1] if len(t_actual) > 0 else t_cmd[-1]
            legs.append((float(t_cmd[0]), float(interval_end), name, is_constrained))

        if not is_constrained:
            continue

        # State logging starts a fixed 1.50s before the leg's first command, so
        # the measured trace would otherwise extend to the left of the commanded
        # and planned ones. Restrict it to the commanded window so all three
        # curves cover exactly the same interval.
        m = (t_actual >= t_cmd[0]) & (t_actual <= t_cmd[-1])
        meas_t.append(t_actual[m])
        meas_v.append(s[f"cv_{key}_measured"][m])
        cmd_t.append(t_cmd)
        cmd_v.append(s[f"cv_{key}_commanded"])

        if "t_planned" in s:
            t_p = s["t_planned"] + (t_cmd[0] if len(t_cmd) else 0.0)
            planned_t.append(t_p)
            planned_v.append(s[f"cv_{key}_planned"])

    cat = lambda xs: np.concatenate(xs) if xs else np.array([])
    meas_t_arr, meas_v_arr = cat(meas_t), cat(meas_v)

    return (cat(planned_t), cat(planned_v), cat(cmd_t), cat(cmd_v),
            meas_t_arr, meas_v_arr, legs)


def explainer_alpha(t, t_fade_end):
    """Alpha for explainer screens A and B (screen A, screen B) at plot time t.

        t < 0.3            -> (0, 0)
        0.3 -> 0.8          -> A fades in
        0.8 -> t_mid        -> A at full
        t_mid -> t_mid+0.5  -> crossfade A -> B
        t_mid+0.5 -> t_out  -> B at full
        t_out -> t_out+0.7  -> B fades out
        t_out+0.7 ->        -> (0, 0)

    t_out is t_fade_end (0.7s before the first constrained leg's first
    command); t_mid is the midpoint of [1.0, t_fade_end]. Both screens are
    full-frame (640x1080, the whole plot column), so mutually-exclusive alphas
    are exact opacity, not a blend guess.
    """
    t_out = t_fade_end
    t_mid = (1.0 + t_out) / 2.0
    if t < 0.3:
        return 0.0, 0.0
    if t < 0.8:
        return (t - 0.3) / 0.5, 0.0
    if t < t_mid:
        return 1.0, 0.0
    if t < t_mid + 0.5:
        f = (t - t_mid) / 0.5
        return 1.0 - f, f
    if t < t_out:
        return 0.0, 1.0
    if t < t_out + 0.7:
        return 0.0, 1.0 - (t - t_out) / 0.7
    return 0.0, 0.0


def _composite_explainer(arr, screen_a_f, screen_b_f, alpha_a, alpha_b):
    """Alpha-composite the (float32) explainer screens over an RGB frame."""
    if alpha_a <= 0.0 and alpha_b <= 0.0:
        return arr
    out = arr.astype(np.float32) * (1.0 - alpha_a - alpha_b)
    if alpha_a > 0.0:
        out += screen_a_f * alpha_a
    if alpha_b > 0.0:
        out += screen_b_f * alpha_b
    return np.clip(out, 0, 255).astype(np.uint8)


def render_plot_video(pt_data, out_path, total_duration):
    """Pre-render the constraint-violation plot video.

    One panel per source of error -- planned trajectory, then measured -- each
    carrying position (mm, left axis) and orientation (mrad, right axis) of the
    gripper-to-gripper transform. Splitting by source rather than by quantity is
    what makes the comparison legible: the two panels can then use independent y
    scales, which they must, because the planned error is ~1e-12 mm and the
    measured is ~1 mm.

    Commanded is not drawn. The commanded waypoints are samples of the very same
    trajectory, so through IKFast's FK they agree with planned to the sample rate
    (max identical to 5 significant figures across the grid); a third trace would
    lie exactly under the planned one.

    Data is only plotted during the constrained legs (lift/place); unconstrained
    spans are greyed.

    Linear axes, deliberately. Measured against IKFast's own FK the planned
    constraint violation is ~2e-15 m -- genuinely zero at this scale. A log axis
    cannot draw that against a 1e-3 measured trace without inventing a floor,
    which is how a fabricated 1e-16 line once reached the screen.

    Every artist -- title, subtitle, axis furniture, lines, leg labels, the
    grey-span shading -- is built ONCE per worker and only its data/visibility
    changes per frame. Going back to `ax.clear()` and
    rebuilding everything every frame is a measured 46s -> 27s regression
    elsewhere in this pipeline (render_segment2.render_error_plot_video); this
    loop is now built the same way.
    """
    steps = pt_data["steps"]

    n_frames = int(total_duration * FPS) + 1

    pos = collect_series(steps, "pos")
    rot = collect_series(steps, "rot")
    legs = pos[6]  # (t0, t1, name, is_constrained) -- same for both keys

    boundaries = [l[0] for l in legs]
    unconstrained_intervals = [(l[0], l[1]) for l in legs if not l[3]]
    constrained_legs = [l for l in legs if l[3]]

    MM, MRAD = 1e3, 1e3

    # (title, subtitle, position-t, position-v(mm), orientation-t, orientation-v(mrad))
    panels = [
        ("Constraint violation — planned",
         "eliminated by construction; 10⁻¹² mm and mrad are floating-point round-off",
         pos[0], pos[1] * MM, rot[0], rot[1] * MRAD),
        ("Constraint violation — measured on the robot",
         "controller tracking error — the planner's own violation is the panel above",
         pos[4], pos[5] * MM, rot[4], rot[5] * MRAD),
    ]

    # When the explainer screens fade out (0.7s before the first constrained
    # leg's first command) and the phase ribbon windows key off this same
    # boundary -- derived, never hardcoded, so a different point (a different
    # --point) still lines up.
    if constrained_legs:
        first_constrained_t0 = constrained_legs[0][0]
    else:
        first_constrained_t0 = total_duration  # degenerate fallback
    t_fade_end = first_constrained_t0 - 0.7

    def top_of(v, floor):
        # 1.45x, so the legend in the upper-left never lands on a peak.
        return max(float(v.max()) if len(v) else floor, floor) * 1.45

    def decade_span(*arrays):
        """Decade-aligned (lo, hi) covering the positive values in arrays.

        The low end comes from the 1st percentile of the positive values, not the
        minimum: near-exact cancellations reach 1e-20 mm on some points and would
        otherwise stretch the axis over twenty pointless decades.
        """
        pos = np.concatenate([a[a > 0] for a in arrays if len(a)]) \
            if any(len(a) for a in arrays) else np.array([])
        if not len(pos):
            return 1e-15, 1e-3
        lo = 10.0 ** np.floor(np.log10(np.percentile(pos, 1)))
        hi = 10.0 ** np.ceil(np.log10(pos.max()))
        return float(lo), float(max(hi, lo * 10))

    # The planned panel is log: on a linear axis its trace is a flat line on zero,
    # which says "negligible" but hides that it is dancing around 1e-13 mm inside
    # each leg and steps once, at the lift->place re-freeze. The measured panel
    # stays linear -- its 0-2 mm range needs no log.
    planned_log = True
    pl_pos = decade_span(panels[0][3])
    pl_rot = decade_span(panels[0][5])
    # Historical guard, not expected to fire any more: this used to trigger
    # every time, because compute_errors.py computed the planned rotation
    # error as arccos((trace-1)/2), which rounds to bit-exact 0.0 this close
    # to identity and left decade_span() nothing positive to span (it fell
    # through to its own (1e-15, 1e-3) default, decades away from the
    # position panel's). compute_errors.constraint_error now uses
    # atan2(||skew||, (trace-1)/2) instead -- see the comment there, and
    # scripts/ee_constraint_report.py:deviation which sets the precedent --
    # so the rotation trace is a genuine ~1e-12 mrad and gets a real,
    # non-degenerate decade_span() of its own. Kept as a fallback in case a
    # future dataset (e.g. a single-sample leg) again has no positive
    # rotation value to span.
    if pl_rot[0] >= pl_rot[1] / 10:
        pl_rot = (pl_pos[0], pl_rot[1])

    ylims = [(pl_pos, pl_rot),
             ((0.0, top_of(panels[1][3], 1.0)), (0.0, top_of(panels[1][5], 1.0)))]

    # First unconstrained span (grey = no box held) caption anchor: its
    # midpoint, revealed once the span has some visible width so the caption
    # never points at nothing.
    grey_caption_xy = None
    grey_caption_reveal_t = None
    if unconstrained_intervals:
        g0, g1 = unconstrained_intervals[0]
        grey_caption_xy = (g0 + g1) / 2.0
        grey_caption_reveal_t = g0 + 0.3

    # Every frame is an independent function of its index, so the plot pass
    # fans out across processes; see frame_parallel. The Figure, its twin
    # axes, and the explainer screens are built once per worker.
    def make_renderer():
        from matplotlib.ticker import MaxNLocator

        dpi = 100
        fig, axes = plt.subplots(2, 1, figsize=(PLOT_W / dpi, PLOT_H / dpi),
                                 dpi=dpi)
        fig.patch.set_facecolor(BG_COLOR)
        # top/hspace make room for the subtitle line each panel carries under
        # its title. The subtitle sits at axes-fraction y=1.06 (see below) so
        # it clears the tallest tick label on either y-axis of that panel --
        # top shrank and hspace grew again (0.87/0.46 -> 0.85/0.55) after that
        # move landed the text on top of the topmost tick labels at 1.02.
        fig.subplots_adjust(left=0.19, right=0.81, top=0.85, bottom=0.09,
                            hspace=0.55)
        twins = [ax.twinx() for ax in axes]

        screen_a_f = explainer_screen_a(size=(PLOT_W, PLOT_H)).astype(np.float32)
        screen_b_f = explainer_screen_b(size=(PLOT_W, PLOT_H)).astype(np.float32)

        panel_state = []
        for i, (ax, twin, p) in enumerate(zip(axes, twins, panels)):
            title, subtitle, px, py, rx, ry = p
            lim_pos, lim_rot = ylims[i]
            is_log = planned_log and i == 0

            ax.set_facecolor(BG_COLOR)
            ax.set_title(title, color=TEXT_COLOR, fontsize=FS_TITLE,
                        fontweight="bold", pad=30)
            ax.text(0.5, 1.06, subtitle, transform=ax.transAxes, ha="center",
                    va="bottom", fontsize=FS_SUBTITLE, color=SUBTITLE_COLOR)
            ax.set_ylabel("position (mm)", color=COL_POS, fontsize=FS_LABEL)
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
            # Reusing axes across frames (rather than clearing) changes what
            # AutoLocator resolves to (handoff: render_segment2 lost this the
            # same way); pin the budget the cleared axes used to land on.
            ax.xaxis.set_major_locator(MaxNLocator(nbins=7, steps=[1, 2, 2.5, 5, 10]))

            if is_log:
                ax.set_yscale("log")
                twin.set_yscale("log")
            ax.set_ylim(*lim_pos)
            twin.set_ylim(*lim_rot)

            handles = []
            traces = []
            # Each series carries its OWN axis's floor: position lives on ax
            # (lim_pos), orientation on twin (lim_rot), and the two floors
            # differ whenever the rotation trace is bit-exactly zero and gets
            # re-based onto the position floor above.
            for target, x, y, color, label, floor in (
                (ax, px, py, COL_POS, "position", lim_pos[0]),
                (twin, rx, ry, COL_ROT, "orientation", lim_rot[0]),
            ):
                h, = target.plot([], [], color=color, lw=1.8, label=label)
                handles.append(h)
                line, = target.plot([], [], color=color, lw=1.8, alpha=0.9)
                traces.append((line, x, y, floor))
            # Anchored below the leg-name-label strip (y~0.93-0.99) rather
            # than flush with the top of the axes, so the two never collide --
            # a legend placed at the default "upper left" corner sits directly
            # under "reach"'s label.
            ax.legend(handles=handles, fontsize=FS_LEGEND, loc="upper left",
                      bbox_to_anchor=(0.0, 0.88), facecolor=BG_COLOR,
                      edgecolor=TEXT_COLOR, labelcolor=TEXT_COLOR)

            # Leg-boundary vlines, one Line2D per leg start, created invisible
            # and revealed once the playhead passes it -- never rebuilt.
            vlines = [(ax.axvline(b, color=TEXT_COLOR, alpha=0.2, lw=0.5,
                                  ls=":", visible=False), b)
                      for b in boundaries if b > 0]

            # Grey "no box held" spans, one Rectangle per unconstrained
            # interval in the blended (data-x, axes-fraction-y) transform that
            # axvspan itself uses internally -- so it can be widened per frame
            # via set_x/set_width instead of being re-created.
            span_trans = ax.get_xaxis_transform()
            spans = []
            for t_start, t_end in unconstrained_intervals:
                patch = mpatches.Rectangle((t_start, 0), 0, 1, transform=span_trans,
                                           facecolor="#444444", alpha=0.25,
                                           edgecolor="none", zorder=0)
                ax.add_patch(patch)
                spans.append((patch, t_start, t_end))

            panel_state.append(dict(ax=ax, twin=twin, traces=traces,
                                    vlines=vlines, spans=spans, is_log=is_log))

        axes[-1].set_xlabel("time (s)", color=TEXT_COLOR, fontsize=FS_LABEL)

        # Leg-name labels along the top of the TOP panel only, in data-x /
        # axes-fraction-y so they track the x window growth without moving in
        # y as the log axis' data range would otherwise imply.
        top_trans = axes[0].get_xaxis_transform()
        leg_labels = []
        for (t0, t1, name, _is_c) in legs:
            label = LEG_LABELS.get(name, name)
            mid = (t0 + t1) / 2.0
            txt = axes[0].text(mid, 0.93, label, transform=top_trans,
                               ha="center", va="top", fontsize=FS_LEG_NAME,
                               color=TEXT_COLOR, alpha=0.85, visible=False)
            leg_labels.append([txt, t0, mid, None])  # halfwidth_px filled below

        grey_caption = None
        if grey_caption_xy is not None:
            grey_caption = axes[0].text(
                grey_caption_xy, 0.06, "grey = no box held", transform=top_trans,
                ha="center", va="bottom", fontsize=FS_CAPTION, color="#999999",
                visible=False)

        # A leg's label sits at its span's midpoint, but the x-window only
        # grows to "now" -- the last leg or two (whose span extends beyond
        # the current time) can have a midpoint past the visible right edge,
        # spilling the label into the tick-label gutter outside the axes.
        # Clamping needs the label's rendered pixel width, which depends only
        # on the font and string (not on the x data scale), so measure it
        # once against a throwaway draw rather than every frame.
        # Text.get_window_extent() reports a degenerate (near-zero) bbox for
        # an invisible artist -- these labels are created invisible (revealed
        # progressively per frame) -- so flip them visible for this one
        # measurement draw, then back off; the per-frame loop below is what
        # actually controls when each becomes visible.
        for entry in leg_labels:
            entry[0].set_visible(True)
        fig.canvas.draw()
        _setup_renderer = fig.canvas.get_renderer()
        AX_PX_WIDTH = axes[0].get_window_extent(_setup_renderer).width
        LABEL_MARGIN_PX = 4.0  # keep the label's edge a few px inside the spine
        for entry in leg_labels:
            entry[3] = (entry[0].get_window_extent(_setup_renderer).width / 2.0
                        + LABEL_MARGIN_PX)
            entry[0].set_visible(False)

        def one(frame_idx):
            t = frame_idx / FPS
            xmax = max(t + 1, 5)

            for state in panel_state:
                ax, twin = state["ax"], state["twin"]
                for line, x, y, floor in state["traces"]:
                    mask = x <= t
                    if mask.any():
                        yv = y[mask]
                        if state["is_log"]:
                            # Exact zeros are real (bit-identical transforms)
                            # but log(0) is -inf, which matplotlib masks --
                            # leaving holes in the line. Draw them on this
                            # series's own axis floor instead, where they read
                            # as "at or below this", which is true.
                            yv = np.maximum(yv, floor)
                        line.set_data(x[mask], yv)
                    else:
                        line.set_data([], [])
                ax.set_xlim(0, xmax)
                twin.set_xlim(0, xmax)

                for vline, b in state["vlines"]:
                    if b <= t:
                        vline.set_visible(True)
                for patch, t_start, t_end in state["spans"]:
                    vis_start = max(0.0, t_start)
                    vis_end = min(t, t_end)
                    width = max(0.0, vis_end - vis_start)
                    patch.set_x(vis_start)
                    patch.set_width(width)

            # Pass 1: reveal + clamp each label independently inside the
            # visible window (data 0..xmax). A leg whose span runs past "now"
            # (notably the last leg or two) would otherwise place its label's
            # midpoint past the right edge, into the tick-label gutter.
            px_per_data = AX_PX_WIDTH / xmax
            LABEL_GAP_PX = 6.0
            gap_data = LABEL_GAP_PX / px_per_data
            visible_entries = []  # [txt, margin_data, own_clamp] in leg order
            for txt, t0, mid, halfwidth_px in leg_labels:
                if t >= t0:
                    txt.set_visible(True)
                    margin_data = halfwidth_px / px_per_data
                    own_clamp = min(mid, xmax - margin_data)
                    own_clamp = max(own_clamp, margin_data)
                    visible_entries.append([txt, margin_data, own_clamp])

            # Pass 2: a leg label clamped against the window edge can now
            # land on top of the PREVIOUS leg's label, which sits at its own
            # unclamped, naturally-spaced position (the window has grown
            # past it already). Resolve that by nudging earlier labels
            # further left, never later ones further right -- an earlier
            # label reliably has slack to its own left (its neighbour is
            # further back in time, hence further left still), whereas
            # pushing a later label right would walk it straight back past
            # the window edge pass 1 just enforced. Back-to-front so each
            # label is pushed clear of the (already finalised) one after it.
            final_x = [None] * len(visible_entries)
            for i in range(len(visible_entries) - 1, -1, -1):
                _txt, margin_data, own_clamp = visible_entries[i]
                x = own_clamp
                if i + 1 < len(visible_entries):
                    _, next_margin, _ = visible_entries[i + 1]
                    max_x = final_x[i + 1] - next_margin - gap_data - margin_data
                    x = min(x, max_x)
                x = max(x, margin_data)  # never push past the window's own left edge
                final_x[i] = x

            for (txt, _margin_data, _own_clamp), x in zip(visible_entries, final_x):
                txt.set_position((x, 0.93))

            if grey_caption is not None and t >= grey_caption_reveal_t:
                grey_caption.set_visible(True)

            fig.canvas.draw()
            buf = fig.canvas.buffer_rgba()
            arr = np.asarray(buf)[:, :, :3]

            h, w = arr.shape[:2]
            if (w, h) != (PLOT_W, PLOT_H):
                from PIL import Image
                arr = np.array(Image.fromarray(arr).resize((PLOT_W, PLOT_H), Image.LANCZOS))

            alpha_a, alpha_b = explainer_alpha(t, t_fade_end)
            arr = _composite_explainer(arr, screen_a_f, screen_b_f, alpha_a, alpha_b)

            return arr

        return one

    print(f"  Rendering {n_frames} plot frames...")
    # The pool is forked before ffmpeg starts: a worker that inherited the write
    # end of this pipe would keep ffmpeg from ever seeing EOF.
    frames = parallel_frames(make_renderer, n_frames, progress_every=200,
                             label="plot")
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{PLOT_W}x{PLOT_H}", "-pix_fmt", "rgb24",
        "-r", str(FPS),
        "-i", "-",
        "-c:v", "libx264", "-crf", "22", "-preset", "fast",
        "-pix_fmt", "yuv420p", "-an",
        out_path,
    ]
    pipe = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    for arr in frames:
        pipe.stdin.write(arr.tobytes())

    pipe.stdin.close()
    pipe.wait()
    if pipe.returncode != 0:
        stderr = pipe.stderr.read().decode() if pipe.stderr else ""
        print(f"  ffmpeg error:\n{stderr[-500:]}")
        return False

    print(f"  Plot video: {out_path}")
    return True


def build_phase_segments(steps):
    """Merge the six legs into the five human-readable phase-ribbon windows.

    The two gripper actions (grasp, release) are not trajectory legs in
    `steps` -- they are the *gaps* between the last unconstrained leg before
    the constrained block and the first constrained leg (grasp), and between
    the last constrained leg and the first unconstrained leg after it
    (release). Deriving the phases this way needs nothing beyond
    `is_constrained` and `t_cmd`, so it keeps working even if leg names change.

    Returns a list of (t0, t1, label, ffmpeg_color) windows, already merged --
    "reach_approach" and "reach_descend" collapse into one "Reaching for the
    box" window rather than two identical-text windows back to back.
    """
    idx_constrained = [i for i, s in enumerate(steps) if s.get("is_constrained", False)]
    if not idx_constrained:
        t0 = steps[0]["t_cmd"][0]
        t1 = steps[-1]["t_cmd"][-1]
        return [(t0, t1, "Reaching for the box", "white")]

    i0, i1 = idx_constrained[0], idx_constrained[-1]
    segs = []
    if i0 > 0:
        segs.append((steps[0]["t_cmd"][0], steps[i0 - 1]["t_cmd"][-1],
                     "Reaching for the box", "white"))
        segs.append((steps[i0 - 1]["t_cmd"][-1], steps[i0]["t_cmd"][0],
                     "Grasping", "white"))
    segs.append((steps[i0]["t_cmd"][0], steps[i1]["t_cmd"][-1],
                 "Carrying — constraint active", "0x66bb6a"))
    if i1 < len(steps) - 1:
        segs.append((steps[i1]["t_cmd"][-1], steps[i1 + 1]["t_cmd"][0],
                     "Releasing", "white"))
        segs.append((steps[i1 + 1]["t_cmd"][0], steps[-1]["t_cmd"][-1],
                     "Returning home", "white"))
    return segs


def composite_videos(hw_path, plot_path, out_path, start_s, duration, point_idx, steps):
    """Use ffmpeg to composite hardware video + plot video side by side."""
    ribbon = "".join(
        f",drawtext=text='{_escape(label)}':fontfile={FONT_PATH}:fontsize=26:"
        f"fontcolor={color}:borderw=2:bordercolor=black:x=24:y=112:"
        f"enable='between(t,{t0:.3f},{t1:.3f})'"
        for t0, t1, label, color in build_phase_segments(steps)
    )

    filter_str = (
        f"[0:v]trim=start={start_s:.2f},setpts=PTS-STARTPTS,"
        f"crop={HW_CROP},"
        f"scale=1280:1080:force_original_aspect_ratio=decrease,"
        f"pad=1280:1080:(ow-iw)/2:(oh-ih)/2:color={BG_HEX},"
        # Sized for 720p viewing (see the FS_* note): the hardware pane is 1280px
        # wide here and 853px at 720p, so these land at ~24px and ~19px.
        f"drawtext=text='{_escape(f'Grid point {point_idx} of 20')}':"
        f"fontfile={FONT_PATH}:fontsize=36:fontcolor=white:"
        f"borderw=2:bordercolor=black:x=24:y=24,"
        f"drawtext=text='{_escape('one of 20 box pick-up positions')}':"
        f"fontfile={FONT_PATH}:fontsize=22:fontcolor=white:"
        f"borderw=2:bordercolor=black:x=24:y=72"
        f"{ribbon},"
        f"drawtext=text='{_escape(SPEED_LABEL + ' real time')}':"
        f"fontfile={FONT_PATH}:fontsize=28:fontcolor=white:"
        f"borderw=2:bordercolor=black:x=24:y=h-48"
        f"[vid];"
        f"[1:v]scale=640:1080[plot];"
        f"[vid][plot]hstack=inputs=2[out]"
    )

    cmd = [
        "ffmpeg", "-y",
        "-i", hw_path,
        "-i", plot_path,
        "-filter_complex", filter_str,
        "-map", "[out]",
        "-c:v", "libx264", "-crf", "20", "-preset", "medium",
        "-pix_fmt", "yuv420p",
        "-t", f"{duration:.2f}",
        "-an",
        out_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Composite failed:\n{result.stderr[-800:]}")
        return False
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--point", type=int, default=0)
    ap.add_argument("--error-data", default=os.path.join(REPO, "video", "error_data.pkl"))
    ap.add_argument("--hw-dir", default=os.path.join(REPO, "videos"))
    ap.add_argument("--trim-json", default=os.path.join(REPO, "videos", "trim_points.json"))
    ap.add_argument("--video-offset", type=float, default=0.0)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    if args.output is None:
        args.output = os.path.join(REPO, "video", f"segment1_point{args.point:02d}.mp4")

    with open(args.error_data, "rb") as f:
        all_errors = pickle.load(f)

    if args.point not in all_errors:
        print(f"Point {args.point} not in error data. Available: {sorted(all_errors.keys())}")
        sys.exit(1)

    pt_data = all_errors[args.point]
    total_duration = pt_data["total_duration"]

    with open(args.trim_json) as f:
        trim = json.load(f)

    seed_map = build_seed_map(args.hw_dir)
    hw_file = seed_map[args.point]
    hw_path = os.path.join(args.hw_dir, hw_file)
    start_s = trim.get(hw_file, {}).get("start_s", 0.0) + args.video_offset

    scratch = os.path.join(REPO, "video", "scratch_seg1")
    os.makedirs(scratch, exist_ok=True)
    plot_path = os.path.join(scratch, f"plots_{args.point:02d}.mp4")

    print(f"Point {args.point}: {total_duration:.1f}s at 1x")
    print(f"Hardware video: {hw_file} (start={start_s:.1f}s)")

    print("\nPass 1: Rendering error plots...")
    if not render_plot_video(pt_data, plot_path, total_duration):
        sys.exit(1)

    print("\nPass 2: Compositing hardware + plots...")
    if not composite_videos(hw_path, plot_path, args.output,
                           start_s, total_duration, args.point, pt_data["steps"]):
        sys.exit(1)

    dur = float(subprocess.check_output([
        "ffprobe", "-v", "quiet", "-show_entries", "format=duration",
        "-of", "csv=p=0", args.output,
    ]).decode().strip())
    print(f"\nWrote {args.output} ({dur:.1f}s)")


if __name__ == "__main__":
    main()
