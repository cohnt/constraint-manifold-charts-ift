"""Static explanatory images for the RA-L supplementary video.

Four panels that answer the questions the advisor's review raised (see
the video pipeline's stage table):
what the constraint is, what the plots show, what a "grid point" is, and what
the headline numbers are. Every panel is drawn with matplotlib -- not PIL --
because the gripper-to-gripper transform equation needs mathtext, and
mathtext needs no LaTeX install (unlike manim's ``MathTex``, which
``build_video.py`` has to check for separately).

Public API, imported by the segment renderers and the montage step:

    explainer_screen_a(size=(640, 1080)) -> np.ndarray   # uint8 RGB (H, W, 3)
    explainer_screen_b(size=(640, 1080)) -> np.ndarray
    grid_panel(size=(504, 1080)) -> np.ndarray
    results_panel(size=(504, 1080)) -> np.ndarray

Each panel's content is built once at a top anchor to measure its own
height, then rebuilt shifted so it sits vertically centred in the column
(equal empty margin above and below) rather than pinned to the top -- see
``_render_centered``.

Run directly to dump all four as PNGs for eyeballing:

    venv/bin/python scripts/video/ral_explainers.py --out-dir scratch/ral_explainers
"""

import argparse
import json
import os
import pickle
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import plan_format  # noqa: F401  -- installs the pre-rename module aliases the
                    # committed plan and execution pickles were written with

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# House palette (make_cards.py's palette, as hex -- this module draws with
# matplotlib, which wants hex/named strings rather than the 0-255 tuples
# make_cards.py uses for PIL).
BG = "#1a1a2e"
TEXT = "#e0e0e0"
ACCENT = "#4fc3f7"
GREEN = "#66bb6a"
CORAL = "#ff7043"
MUTED = "#a0a0a0"

# The frozen paper numbers (see the folder README). These are the
# fallback -- and the values ee_constraint_report.py's pooling method must
# still reproduce from video/error_data.pkl, or the on-screen numbers would
# silently drift from the paper.
_FALLBACK = {
    "n_success": 20,
    "n_total": 20,
    "pos_mean_mm": 0.55,
    "pos_max_mm": 2.23,
    "rot_mean_mrad": 1.60,
    "rot_max_mrad": 6.81,
}
_PLANNED_POS_MAX = "6.1 × 10⁻¹² mm"
_PLANNED_ROT_MAX = "4.9 × 10⁻¹² mrad"
_MEDIAN_PLANNING_TIME = "43.6 s"

_MARGIN_PX = 40
_TOP_MARGIN_PX = 40    # top-anchored pass used only to measure content height
_BOTTOM_MARGIN_PX = 40  # minimum bottom clearance if content is tall


# --------------------------------------------------------------------- core

def _new_figure(size):
    w, h = size
    fig = plt.figure(figsize=(w / 100.0, h / 100.0), dpi=100)
    fig.patch.set_facecolor(BG)
    return fig


def _finish(fig):
    """Render a figure to a (H, W, 3) uint8 RGB array and close it."""
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    rgb = buf[..., :3].copy()
    plt.close(fig)
    return rgb


def _initial_renderer(fig):
    fig.canvas.draw()
    return fig.canvas.get_renderer()


def _text_width_px(fig, renderer, text, fontsize, weight="normal"):
    t = fig.text(0, 0, text, fontsize=fontsize, fontweight=weight)
    width = t.get_window_extent(renderer=renderer).width
    t.remove()
    return width


def _wrap_words(fig, renderer, text, fontsize, max_width_px, weight="normal"):
    """Greedy word-wrap using actual rendered widths, not a guessed char count."""
    words = text.split()
    lines, cur = [], []
    for word in words:
        trial = " ".join(cur + [word])
        if cur and _text_width_px(fig, renderer, trial, fontsize, weight) > max_width_px:
            lines.append(" ".join(cur))
            cur = [word]
        else:
            cur.append(word)
    if cur:
        lines.append(" ".join(cur))
    return lines


def _place_paragraph(fig, renderer, x_frac, y_top_frac, text, fontsize, color, W, H,
                      weight="normal", ha="left", linespacing=1.45, gap_after_px=22,
                      wrap=True):
    """Draw a (possibly multi-line) text block anchored at its top, return the
    artist and the figure-fraction y just below it (minus a gap), measured
    from the real rendered bbox rather than an assumed line height."""
    if wrap:
        max_w = W - x_frac * W - _MARGIN_PX
        lines = _wrap_words(fig, renderer, text, fontsize, max_w, weight)
        body = "\n".join(lines)
    else:
        body = text
    t = fig.text(x_frac, y_top_frac, body, fontsize=fontsize, color=color, ha=ha,
                 va="top", fontweight=weight, linespacing=linespacing)
    bb = t.get_window_extent(renderer=renderer)
    return t, bb.y0 / H - gap_after_px / H


def _heading(fig, renderer, text, W, H, y_top_frac, fontsize=22, gap_after_px=14):
    """Draw the heading + rule. Returns (next_y_frac, heading_top_px) -- the
    top_px is the true visual top of the content block, used by
    ``_render_centered`` to measure content height."""
    margin_frac = _MARGIN_PX / W
    t = fig.text(margin_frac, y_top_frac, text, fontsize=fontsize, fontweight="bold",
                 color=ACCENT, ha="left", va="top")
    bb = t.get_window_extent(renderer=renderer)
    top_px = bb.y1
    rule_y = bb.y0 / H - 10 / H
    fig.add_artist(Line2D([margin_frac, 1 - margin_frac], [rule_y, rule_y],
                           transform=fig.transFigure, color=ACCENT, linewidth=1.2,
                           alpha=0.55))
    return rule_y - gap_after_px / H, top_px


def _build_centered_figure(build_fn, size):
    """Call build_fn(fig, renderer, W, H, y_start_frac) -> bottom_px twice:
    once on a throwaway figure anchored at the top, purely to measure the
    content's total height, then again on the real figure with the block
    re-based so it sits vertically centred in the column. build_fn's
    internal gaps and wrap decisions only depend on W/x, not on y_start, so
    shifting the starting y shifts the whole flow uniformly.

    Returns the finished-but-not-closed figure -- see _finish. Callers that
    only want the rendered array should go through _render_centered instead;
    this is exposed separately so build_panel_figures() can hand out the
    figure itself without rasterising it."""
    W, H = size
    y0 = 1 - _TOP_MARGIN_PX / H

    probe = _new_figure(size)
    probe_renderer = _initial_renderer(probe)
    top_px, bottom_px = build_fn(probe, probe_renderer, W, H, y0)
    plt.close(probe)

    content_h = top_px - bottom_px
    available_h = H - _TOP_MARGIN_PX - _BOTTOM_MARGIN_PX
    if content_h >= available_h:
        y_start = y0  # no slack to centre with -- keep the original top anchor
    else:
        desired_top_margin = (H - content_h) / 2
        shift_px = (H - desired_top_margin) - top_px
        y_start = y0 + shift_px / H

    fig = _new_figure(size)
    renderer = _initial_renderer(fig)
    build_fn(fig, renderer, W, H, y_start)
    return fig


def _render_centered(build_fn, size):
    return _finish(_build_centered_figure(build_fn, size))


# ------------------------------------------------------------- explainer A

# The two 504 px panels (grid_panel, results_panel) sit either side of
# montage_20.mp4 in the final montage frame, whose tile grid starts at the
# very top of the 1080 px column. Vertically centring them (as the two 640
# px segment-1 overlays are, below) opens an empty band above their heading
# that the montage tiles do not have, so the three columns read as
# misaligned. These two are top-aligned instead, both with the SAME modest
# top margin, so their headings sit on one line across the frame.
_MONTAGE_TOP_MARGIN_PX = 35


def _build_top_aligned_figure(build_fn, size, top_margin_px=_MONTAGE_TOP_MARGIN_PX):
    """Same split as _build_centered_figure: returns the finished-but-not-
    closed figure so build_panel_figures() can hand it out directly."""
    W, H = size
    fig = _new_figure(size)
    renderer = _initial_renderer(fig)
    y_start = 1 - top_margin_px / H
    build_fn(fig, renderer, W, H, y_start)
    return fig


def _render_top_aligned(build_fn, size, top_margin_px=_MONTAGE_TOP_MARGIN_PX):
    return _finish(_build_top_aligned_figure(build_fn, size, top_margin_px))


def _build_screen_a(fig, renderer, W, H, y_start):
    margin_frac = _MARGIN_PX / W

    y, top_px = _heading(fig, renderer, "What the Robot Is Doing", W, H, y_start)

    body1 = ("Both grippers grasp the same box and carry it to the table. "
             "While the box is held, the pose of the right gripper relative "
             "to the left may not change:")
    _, y = _place_paragraph(fig, renderer, margin_frac, y, body1, fontsize=16,
                             color=TEXT, W=W, H=H, gap_after_px=26)

    # Monogram notation, matching the paper (preamble/preamble.tex:238,
    # sections/methodology.tex:45): {}^{A}X^{B} is the pose of frame B
    # relative to (and expressed in) frame A. L, R are the two gripper
    # frames, W is world. Composed rather than inverted -- (^AX^B)^-1 is
    # just ^BX^A, so ^LX^R = ^LX^W * ^WX^R (the inner W's cancel), matching
    # how the paper itself writes compositions (methodology.tex:45). The
    # leading "{}" on each factor is required -- a bare "^{L}X" has no
    # nucleus for the superscript and mathtext refuses to parse it.
    eq_single = (r"${}^{L}X^{R}(q) = {}^{L}X^{W}(q)\, "
                 r"{}^{W}X^{R}(q) = \mathrm{constant}$")
    content_w = W - 2 * _MARGIN_PX
    single_w = _text_width_px(fig, renderer, eq_single, 18)
    if single_w <= content_w:
        eq_h = fig.text(0.5, y, eq_single, fontsize=18, color=TEXT, ha="center",
                         va="top")
        bb = eq_h.get_window_extent(renderer=renderer)
        y = bb.y0 / H - 26 / H
    else:
        # Break at the first "=" rather than shrinking the font.
        eq_line1 = r"${}^{L}X^{R}(q)$"
        eq_line2 = (r"$= {}^{L}X^{W}(q)\, "
                     r"{}^{W}X^{R}(q) = \mathrm{constant}$")
        for line in (eq_line1, eq_line2):
            t = fig.text(0.5, y, line, fontsize=18, color=TEXT, ha="center", va="top")
            bb = t.get_window_extent(renderer=renderer)
            y = bb.y0 / H - 6 / H
        y -= 20 / H

    body2 = ("This equality is not a cost or a penalty. It is eliminated by "
             "construction — the planner searches a space where it "
             "always holds.")
    last, _ = _place_paragraph(fig, renderer, margin_frac, y, body2, fontsize=16,
                                color=TEXT, W=W, H=H)
    bottom_px = last.get_window_extent(renderer=renderer).y0

    return top_px, bottom_px


def explainer_screen_a(size=(640, 1080)) -> np.ndarray:
    return _render_centered(_build_screen_a, size)


# ------------------------------------------------------------- explainer B

def _build_screen_b(fig, renderer, W, H, y_start):
    margin_frac = _MARGIN_PX / W

    y, top_px = _heading(fig, renderer, "What the Plots Show", W, H, y_start)

    body1 = (r"How far ${}^{L}X^{R}$ drifts from its value at the grasp. A pose "
             "error has two parts, and metres cannot be added to radians, "
             "so both are plotted separately:")
    _, y = _place_paragraph(fig, renderer, margin_frac, y, body1, fontsize=16,
                             color=TEXT, W=W, H=H, gap_after_px=26)

    swatch_w_frac = 0.07
    for color, label in (
            (CORAL, r"position   $\Vert\Delta p\Vert$   mm"),
            (ACCENT, r"orientation   $\Delta\theta$   mrad")):
        line_y = y - 8 / H
        fig.add_artist(Line2D([margin_frac, margin_frac + swatch_w_frac],
                               [line_y, line_y], transform=fig.transFigure,
                               color=color, linewidth=4))
        t = fig.text(margin_frac + swatch_w_frac + 0.02, y, label, fontsize=15,
                     color=color, ha="left", va="top")
        bb = t.get_window_extent(renderer=renderer)
        y = bb.y0 / H - 18 / H
    y -= 10 / H

    body2 = ("Top panel: the planned trajectory — what the planner "
             "produced.")
    _, y = _place_paragraph(fig, renderer, margin_frac, y, body2, fontsize=16,
                             color=TEXT, W=W, H=H, gap_after_px=16)

    body3 = ("Bottom panel: measured on the robot — what the encoders "
             "read back.")
    _, y = _place_paragraph(fig, renderer, margin_frac, y, body3, fontsize=16,
                             color=TEXT, W=W, H=H, gap_after_px=26)

    body4 = ("The gap between them is controller tracking, not planner "
             "error.")
    last, _ = _place_paragraph(fig, renderer, margin_frac, y, body4, fontsize=16,
                                color=TEXT, W=W, H=H)
    bottom_px = last.get_window_extent(renderer=renderer).y0

    return top_px, bottom_px


def explainer_screen_b(size=(640, 1080)) -> np.ndarray:
    # An earlier pass tried pinning the "Top panel:" / "Bottom panel:"
    # paragraphs to the actual on-screen y-ranges of the two plot panels
    # they describe (y 76-464 / y 595-983 in this same 640x1080 column, per
    # render_segment1.py). That target range overlaps where the heading,
    # intro paragraph and colour legend already have to sit -- there is no
    # slack to move the "Top panel" caption down to the target's vertical
    # centre without colliding with the legend above it. Per the brief
    # ("if that fights the vertical centring, just centre the block and
    # don't force it"), this falls back to simple centring.
    return _render_centered(_build_screen_b, size)


# ------------------------------------------------------------------- grid

def _build_grid_panel(fig, renderer, W, H, y_start):
    margin_frac = _MARGIN_PX / W

    y, top_px = _heading(fig, renderer, "What Is a “Point”?", W, H, y_start,
                          fontsize=20)

    npz_path = os.path.join(REPO, "data", "box_placement_grid.npz")
    grid = np.load(npz_path)
    pts = grid["grid_points"]  # base/world frame, metres
    nx = int(grid["grid_nx"])
    ny = int(grid["grid_ny"])
    pitch_mm = float(grid["grid_pitch"]) * 1e3

    body = (f"20 pick-up positions for the box, on a {nx} × {ny} floor "
            f"grid, {pitch_mm:.0f} mm pitch, in front of the robot.")
    _, y = _place_paragraph(fig, renderer, margin_frac, y, body, fontsize=16,
                             color=TEXT, W=W, H=H, gap_after_px=18)

    # ------------------------------------------------------ indexed grid --
    # Plain top-down Cartesian view: world x plotted horizontally
    # (increasing right), world y plotted vertically (increasing up). This
    # matches how compose_montage.py lays the 4x5 hardware tiles out on
    # screen -- index = ix*grid_ny + iy (compose_montage.py:66), column
    # position is ix (x), row position is iy (y) with the highest iy drawn
    # on top -- so point 0 (x=0, y=0) lands bottom-left, point 4 (x=0,
    # y=120) top-left, point 15 (x=90, y=0) bottom-right, point 19 (x=90,
    # y=120) top-right.
    #
    # The left margin is wider than the shared column margin to leave room
    # for the tick labels (up to three digits, "120") and the rotated
    # y-axis label; the right margin is narrower than the shared margin
    # because the axes draws its own border and nothing bleeds past it the
    # way wrapped paragraph text does. That asymmetry is what lets this
    # single axes grow now that the locator view is gone and no longer
    # shares the panel's width with anything else.
    axes_left_frac = 64 / W
    axes_right_pad_frac = 8 / W
    axes_w_frac = 1 - axes_left_frac - axes_right_pad_frac
    axes_h_frac = (axes_w_frac * W) / H
    axes_top = y
    axes_bottom = axes_top - axes_h_frac
    ax = fig.add_axes((axes_left_frac, axes_bottom, axes_w_frac, axes_h_frac))

    x0, y0 = pts[:, 0].min(), pts[:, 1].min()
    xs_mm = (pts[:, 0] - x0) * 1e3
    ys_mm = (pts[:, 1] - y0) * 1e3
    extent_x_mm = xs_mm.max()
    extent_y_mm = ys_mm.max()

    ax.set_facecolor(BG)
    # Markers sized (and the index label sized down a point) so a two-digit
    # index like "19" sits fully inside its own circle -- checked against
    # the marker's actual rendered radius, not assumed.
    ax.scatter(xs_mm, ys_mm, s=230, color=ACCENT, zorder=3, edgecolors="none")
    for idx, (px, py) in enumerate(zip(xs_mm, ys_mm)):
        ax.annotate(str(idx), (px, py), color=BG, fontsize=8, fontweight="bold",
                    ha="center", va="center", zorder=4)

    # set_aspect("equal") + set_box_aspect(1) together mean: equal data
    # scaling in a square box. The data extent itself (90 x 120 mm) is not
    # square, so the view window has to be -- centred on the data, sized to
    # the larger of the two extents plus padding, same span on both axes.
    # Handing set_xlim/set_ylim two *different* spans here would make
    # matplotlib silently override one of them to enforce the square data
    # aspect, which pushed an extra out-of-range tick (120 mm on the y
    # axis) past the figure edge.
    pad = pitch_mm * 0.65
    half_span = max(extent_x_mm, extent_y_mm) / 2 + pad
    ch, cv = extent_x_mm / 2, extent_y_mm / 2  # horizontal=x, vertical=y
    ax.set_xlim(ch - half_span, ch + half_span)
    ax.set_ylim(cv - half_span, cv + half_span)
    ax.set_aspect("equal")
    ax.set_box_aspect(1)

    # Matplotlib's default locator places a "nice" tick (0, 20, 40, ...)
    # wherever one falls, including right at the view edge -- whose label
    # then spills past the axes box. Drop any tick within 8% of the span of
    # either edge instead of trusting the default locator to avoid it.
    tick_step = 20
    edge_pad = 0.08 * (2 * half_span)
    lo_h, hi_h = ch - half_span, ch + half_span
    lo_v, hi_v = cv - half_span, cv + half_span
    h_ticks = [v for v in range(0, int(hi_h) + tick_step, tick_step)
               if lo_h + edge_pad <= v <= hi_h - edge_pad]
    v_ticks = [v for v in range(0, int(hi_v) + tick_step, tick_step)
               if lo_v + edge_pad <= v <= hi_v - edge_pad]
    ax.set_xticks(h_ticks)
    ax.set_yticks(v_ticks)
    ax.set_xlabel("x (mm)", color=TEXT, fontsize=11)
    ax.set_ylabel("y (mm)", color=TEXT, fontsize=11)
    ax.tick_params(colors=TEXT, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(TEXT)

    tight = ax.get_tightbbox(renderer)
    y = tight.y0 / H - 20 / H

    extent_line = f"{extent_x_mm:.0f} × {extent_y_mm:.0f} mm"
    t = fig.text(0.5, y, extent_line, fontsize=13, color=MUTED, ha="center", va="top")
    bb = t.get_window_extent(renderer=renderer)
    y = bb.y0 / H - 22 / H

    closing = "Every point is carried to the same place pose on the table."
    last, _ = _place_paragraph(fig, renderer, margin_frac, y, closing, fontsize=16,
                                color=TEXT, W=W, H=H)
    bottom_px = last.get_window_extent(renderer=renderer).y0

    return top_px, bottom_px



def grid_panel(size=(504, 1080)) -> np.ndarray:
    return _render_top_aligned(_build_grid_panel, size)


# ---------------------------------------------------------------- results

def _load_grid_counts():
    path = os.path.join(REPO, "plans", "grid_cache", "status.json")
    try:
        with open(path) as f:
            status = json.load(f)
        points = {k: v for k, v in status.items() if k.isdigit()}
        n_total = len(points)
        if n_total == 0:
            raise ValueError("no grid points in status.json")
        n_success = sum(1 for p in points.values() if p.get("status") == "success")
        return n_success, n_total
    except Exception as exc:  # noqa: BLE001 - never let a data hiccup crash a video build
        print(f"ral_explainers: could not read grid status from {path} ({exc}); "
              "using the frozen literal.", file=sys.stderr)
        return _FALLBACK["n_success"], _FALLBACK["n_total"]


def _load_measured_violation():
    """Pool cv_pos_measured / cv_rot_measured over the two constrained legs,
    all 20 points, windowed to each step's command-send window (matching
    scripts/ee_constraint_report.py's measured_q23, which is what produced
    the frozen published numbers) -- otherwise idle/settle
    samples outside that window pull the mean down and the panel would show a
    number the paper does not."""
    path = os.path.join(REPO, "video", "error_data.pkl")
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
        pos_means, pos_maxes, rot_means, rot_maxes = [], [], [], []
        for point in data.values():
            for step in point.get("steps", []):
                if not step.get("is_constrained"):
                    continue
                ta = np.asarray(step["t_actual"])
                tc = np.asarray(step["t_cmd"])
                if len(ta) == 0 or len(tc) == 0:
                    continue
                keep = (ta >= tc.min()) & (ta <= tc.max())
                p = np.asarray(step["cv_pos_measured"])[keep]
                r = np.asarray(step["cv_rot_measured"])[keep]
                if len(p) == 0:
                    continue
                pos_means.append(p.mean())
                pos_maxes.append(p.max())
                rot_means.append(r.mean())
                rot_maxes.append(r.max())
        if not pos_means:
            raise ValueError("no constrained-leg samples found")
        computed = {
            "pos_mean_mm": float(np.mean(pos_means)) * 1e3,
            "pos_max_mm": float(np.max(pos_maxes)) * 1e3,
            "rot_mean_mrad": float(np.mean(rot_means)) * 1e3,
            "rot_max_mrad": float(np.max(rot_maxes)) * 1e3,
        }
        for key, fallback_val in _FALLBACK.items():
            if key not in computed:
                continue
            if abs(computed[key] - fallback_val) > 0.02:
                print(f"ral_explainers: computed {key}={computed[key]:.4f} disagrees "
                      f"with the frozen value {fallback_val} (see the folder README) "
                      "by more than rounding; using the frozen literal instead.",
                      file=sys.stderr)
                return {k: _FALLBACK[k] for k in
                        ("pos_mean_mm", "pos_max_mm", "rot_mean_mrad", "rot_max_mrad")}
        return computed
    except Exception as exc:  # noqa: BLE001
        print(f"ral_explainers: could not compute measured violation from {path} "
              f"({exc}); using the frozen literal.", file=sys.stderr)
        return {k: _FALLBACK[k] for k in
                ("pos_mean_mm", "pos_max_mm", "rot_mean_mrad", "rot_max_mrad")}


def _stat_block(fig, renderer, x_frac, y_top_frac, value, label, W, H,
                 value_fontsize=20, label_fontsize=13, value_color=GREEN,
                 gap_after_value_px=6, gap_after_block_px=22, allow_wrap=True,
                 min_fontsize=14):
    max_w = W - x_frac * W - _MARGIN_PX
    fs = value_fontsize
    if allow_wrap:
        if _text_width_px(fig, renderer, value, fs, "bold") > max_w:
            value_text = "\n".join(_wrap_words(fig, renderer, value, fs, max_w, "bold"))
        else:
            value_text = value
    else:
        # Keep the value on one line -- shrink the font instead of wrapping,
        # down to a floor, rather than breaking a short numeric line like
        # "1.60 / 6.81 mrad" across two lines.
        while fs > min_fontsize and _text_width_px(fig, renderer, value, fs, "bold") > max_w:
            fs -= 1
        value_text = value
    v = fig.text(x_frac, y_top_frac, value_text, fontsize=fs,
                 fontweight="bold", color=value_color, ha="left", va="top",
                 linespacing=1.25)
    bb = v.get_window_extent(renderer=renderer)
    label_y = bb.y0 / H - gap_after_value_px / H
    lines = _wrap_words(fig, renderer, label, label_fontsize, max_w)
    l = fig.text(x_frac, label_y, "\n".join(lines), fontsize=label_fontsize,
                 color=MUTED, ha="left", va="top", linespacing=1.3)
    bb2 = l.get_window_extent(renderer=renderer)
    return bb2.y0 / H - gap_after_block_px / H, bb2.y0


def _build_results_panel(fig, renderer, W, H, y_start):
    margin_frac = _MARGIN_PX / W

    y, top_px = _heading(fig, renderer, "Results", W, H, y_start, fontsize=22)

    n_success, n_total = _load_grid_counts()
    y, _ = _stat_block(fig, renderer, margin_frac, y, f"{n_success} / {n_total}",
                        "grid points planned and executed", W, H, value_fontsize=26,
                        gap_after_block_px=30)

    section_t = fig.text(margin_frac, y, "Constraint Violation", fontsize=16,
                          fontweight="bold", color=TEXT, ha="left", va="top")
    bb = section_t.get_window_extent(renderer=renderer)
    y = bb.y0 / H - 16 / H

    y, _ = _stat_block(fig, renderer, margin_frac, y, _PLANNED_POS_MAX,
                        "planned — max, machine precision", W, H)

    y, _ = _stat_block(fig, renderer, margin_frac, y, _PLANNED_ROT_MAX,
                        "planned — max, within each leg", W, H)

    numbers = _load_measured_violation()
    # One unwrapped "mean / max" line rather than a comma-joined sentence,
    # which used to break with "max" orphaned on its own line at this width.
    pos_value = f"{numbers['pos_mean_mm']:.2f} / {numbers['pos_max_mm']:.2f} mm"
    y, _ = _stat_block(fig, renderer, margin_frac, y, pos_value,
                        "measured on the robot — mean / max", W, H, allow_wrap=False)

    rot_value = f"{numbers['rot_mean_mrad']:.2f} / {numbers['rot_max_mrad']:.2f} mrad"
    y, _ = _stat_block(fig, renderer, margin_frac, y, rot_value,
                        "measured on the robot — mean / max", W, H, allow_wrap=False,
                        gap_after_block_px=32)

    _, bottom_px = _stat_block(fig, renderer, margin_frac, y, _MEDIAN_PLANNING_TIME,
                                "planning time per grid point", W, H, value_fontsize=24)

    return top_px, bottom_px


def results_panel(size=(504, 1080)) -> np.ndarray:
    return _render_top_aligned(_build_results_panel, size)


# ------------------------------------------------------------- introspection

# (panel name, build_fn, default size, anchoring mode) for every public panel,
# in the order build_panel_figures() yields them. Anchoring mode must match
# the *_panel()/explainer_screen_*() function above exactly -- this is the
# single place that decides it, so a panel's anchoring can't drift out of
# sync between what ships and what the layout checker validates.
_PANEL_SPECS = (
    ("explainer_screen_a", _build_screen_a, (640, 1080), "centered"),
    ("explainer_screen_b", _build_screen_b, (640, 1080), "centered"),
    ("grid_panel", _build_grid_panel, (504, 1080), "top_aligned"),
    ("results_panel", _build_results_panel, (504, 1080), "top_aligned"),
)


def build_panel_figures(sizes=None):
    """Yield (name, fig) for each panel, anchored exactly as the shipped
    renders are. The caller owns the figures and must plt.close() them.

    ``sizes``, if given, is a {panel_name: (w, h)} override of the default
    sizes above -- e.g. to check a panel at the size it is actually placed
    at in a montage frame."""
    sizes = sizes or {}
    for name, build_fn, default_size, mode in _PANEL_SPECS:
        size = sizes.get(name, default_size)
        if mode == "centered":
            fig = _build_centered_figure(build_fn, size)
        else:
            fig = _build_top_aligned_figure(build_fn, size)
        yield name, fig


# --------------------------------------------------------------------- CLI

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=os.path.join(REPO, "scratch", "ral_explainers"))
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    panels = {
        "explainer_screen_a.png": explainer_screen_a(),
        "explainer_screen_b.png": explainer_screen_b(),
        "grid_panel.png": grid_panel(),
        "results_panel.png": results_panel(),
    }
    for name, arr in panels.items():
        path = os.path.join(args.out_dir, name)
        plt.imsave(path, arr)
        print(f"wrote {path}  {arr.shape[1]}x{arr.shape[0]}")


if __name__ == "__main__":
    main()
