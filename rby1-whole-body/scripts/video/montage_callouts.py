"""The shared "what is a grid point" card, ``render_grid_inset`` -- used by
the overview cut's side-by-side beat (``compose_rby1_v2.py``) and, through an
import of this one function, by the RA-L cut's segment 1
(``ral_annotations.py``).

This module used to also own a set of overview-only montage-tail artwork
tied to the physical 4x5 layout: a notes panel with three "how the planner
used the torso" observations, rounded outlines around the tiles those
observations pointed at, and per-tile leader arrows. That artwork is
deleted here -- the montage tail on both cuts is now just
``[montage][results panel]`` (see ``compose_rby1_v2.normalize_montage`` and
``render_segment2.normalize_montage``), so there is no longer a left column
for a notes panel to fill, and the "what is a grid point" explanation this
module still carries moved earlier: this card, on the side-by-side beat and,
via ``ral_annotations.py``, on segment 1. The deleted code stays recoverable
in git history (commits 480519c and bf337dc) rather than being left
commented out.

    render_grid_inset(path) -> None
        A small RGBA card answering "what is a grid point", shared by two
        callers at two different placements: the overview cut's side-by-side
        beat (over the Drake sim pane's empty upper-left corner) and the
        RA-L cut's segment 1 (over a static patch of the hardware pane --
        segment 1 *is* point 0, so the same "point 0 is ringed" card is
        correct there too). Each caller owns its own placement constants
        (see the comment above OVERVIEW_INSET_X/OVERVIEW_INSET_Y below for
        this module's, and ``ral_annotations.SEG1_INSET_X/Y`` for the RA-L
        cut's) -- this function only draws the card, never where it goes.

Run directly to sanity-check the card's own geometry or dump review stills:

    venv/bin/python scripts/video/montage_callouts.py --check
    venv/bin/python scripts/video/montage_callouts.py --review-stills
"""

import argparse
import os
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Palette and the two heading/renderer helpers shared with the RA-L explainer
# panels, so this card is built the same way grid_panel/results_panel are --
# deliberately not the wider private surface (_wrap_words, _text_width_px,
# ...), which is reimplemented locally below in a dozen lines rather than
# widening the dependency on ral_explainers' internals.
from ral_explainers import BG, TEXT, ACCENT, MUTED, _initial_renderer, _heading  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
VIDEO_DIR = os.path.join(REPO, "video")


# ---------------------------------------------------------------- helpers

def _text_width_px(fig, renderer, text, fontsize, weight="normal"):
    """Local copy of ral_explainers._text_width_px. Not imported: the import
    list above is deliberately limited to the handful of helpers
    render_grid_inset needs, and this (plus _wrap_words/_text_height_px
    below) is small enough that duplicating it here is cheaper than
    widening that surface."""
    t = fig.text(0, 0, text, fontsize=fontsize, fontweight=weight)
    width = t.get_window_extent(renderer=renderer).width
    t.remove()
    return width


def _text_height_px(fig, renderer, text, fontsize, weight="normal"):
    t = fig.text(0, 0, text, fontsize=fontsize, fontweight=weight)
    height = t.get_window_extent(renderer=renderer).height
    t.remove()
    return height


def _wrap_words(fig, renderer, text, fontsize, max_width_px, weight="normal"):
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


# -------------------------------------------------------------- grid inset

# INSET_W/INSET_H are the card's own rendered size (render_grid_inset takes
# no size argument, so every caller gets this size) and are shared by both
# consumers of this card. OVERVIEW_INSET_X/OVERVIEW_INSET_Y below are NOT
# shared -- they are the overview cut's own placement decision, named with
# that prefix (rather than the generic OVERLAY_X/OVERLAY_Y this pair used to
# be called) once this card gained a second caller with a different
# placement: scripts/video/ral_annotations.py, whose SEG1_INSET_X/
# SEG1_INSET_Y put the same card over a different static patch, on the RA-L
# cut's segment 1 (see that module for its own measurement and safe-zone
# reasoning -- it does not import anything from this section, on purpose,
# so a change to the overview's placement here cannot silently move the
# RA-L card).
#
# Placement, and the reasoning behind it, are compose_rby1_v2.py's to own
# (it's the one that calls `overlay=`), but the safe-zone numbers are
# measured against THIS artwork's size, so they live together with it here;
# compose_rby1_v2.py imports OVERVIEW_INSET_X/OVERVIEW_INSET_Y/INSET_W/
# INSET_H rather than repeating the numbers.
#
# Measured off real frames of video/scratch_rby1_v2/side_by_side.mp4 at
# t=7s and t=12s (2026-09-07): the sim pane's upper-left is empty flat
# background there; the sim table occupies roughly x 60..335, y 555..945,
# and the robot occupies x 300..570. The burned-in "Simulation (Drake)"
# label sits at (10,10)-(310,40) (compose_rby1_v2.py's drawtext at
# x=10:y=10, fontsize=30). An earlier placement (overlay=30:545) put the
# inset directly on top of the table -- corrected to the upper-left corner
# instead.
#
# CAVEAT (found sampling every ~1 s of the beat, not just two frames, while
# sizing this inset -- 2026-09-07): the upper-left corner is only unbroken
# background for the *middle* of the 15 s beat. Diffing each sampled frame
# against the pane's flat background colour shows the raised arm sweeping
# up into this same region during the opening reach (t~0.5-3 s, content as
# high as y~226 at x~150-320) and again during the lift/place near the end
# (t~11-14 s, up to y~409). Only t~4-10 s is genuinely clear there. The
# rectangle below is still correct against the two frames it was measured
# from and is what compose_rby1_v2.py places the inset in, but a static
# inset held for the whole 15 s beat will sit in front of the arm during
# those two windows rather than beside it. Flagged for a rendering owner to
# decide (e.g. only overlay it during t=4..10) rather than fixed here,
# since that is a timing decision this module has no say in.
OVERVIEW_INSET_X, OVERVIEW_INSET_Y = 20, 140
INSET_W, INSET_H = 400, 395

# The safe zone the inset must stay inside: clear of the pane label, clear
# of the table (y >= 555), clear of the robot (x >= 430), with a few px of
# margin on each side. ~420x430 was the first ask; 400x395 is the largest
# square-ish card that still fits this zone with 10 px of margin on its
# growing edges.
INSET_SAFE_X0, INSET_SAFE_X1 = 20, 430
INSET_SAFE_Y0, INSET_SAFE_Y1 = 140, 545
_SIM_LABEL_RECT = (10, 10, 310, 40)

assert INSET_SAFE_X0 <= OVERVIEW_INSET_X and OVERVIEW_INSET_X + INSET_W <= INSET_SAFE_X1
assert INSET_SAFE_Y0 <= OVERVIEW_INSET_Y and OVERVIEW_INSET_Y + INSET_H <= INSET_SAFE_Y1

# A two-digit index must actually fit inside its marker -- measured against
# the glyphs' own rendered extent (ral_explainers._build_grid_panel's own
# marker size was picked "checked by eye", which this reuses the idea of
# but makes explicit): if it doesn't fit at a legible font size, drop the
# index labels rather than ship specks nobody can read.
_INDEX_FONTSIZE = 9                 # floor requested is 8pt; one point of headroom
_INDEX_LABEL_SAFETY = 1.4           # required marker diameter = glyph extent * this
_INDEX_LABEL_MAX_FILL = 0.85        # marker must not exceed this fraction of point spacing


def render_grid_inset(path):
    """Render the ~400x395 RGBA inset card answering "what is a grid point",
    used by the overview cut's side-by-side beat (see the placement comment
    above) and the RA-L cut's segment 1 (see ``ral_annotations.py``'s own
    placement comment). The card itself does not vary by caller -- only
    where each one overlays it does."""
    W, H = INSET_W, INSET_H
    fig = plt.figure(figsize=(W / 100.0, H / 100.0), dpi=100)
    fig.patch.set_alpha(0)  # transparent everywhere the card patch doesn't cover
    renderer = _initial_renderer(fig)

    # Rounded semi-opaque backing + 1px border, in the house palette. Drawn
    # as a figure-level patch rather than an axes facecolor so the rounded
    # corners stay genuinely transparent (nothing painted there at all)
    # instead of being clipped out of an opaque rectangle after the fact.
    card_margin = 0.015
    fig.add_artist(FancyBboxPatch(
        (card_margin, card_margin), 1 - 2 * card_margin, 1 - 2 * card_margin,
        transform=fig.transFigure, boxstyle="round,pad=0,rounding_size=0.05",
        linewidth=1.2, edgecolor=mcolors.to_rgba(MUTED, 1.0),
        facecolor=mcolors.to_rgba(BG, 0.88), zorder=0))

    # Tighter than the 504-wide montage panels' own margins/gaps -- this
    # card is smaller and the diagram, not the heading, needs the room.
    pad_px = 14
    margin_frac = pad_px / W
    y = 1 - pad_px / H
    y, _ = _heading(fig, renderer, "What Is a “Point”?", W, H, y,
                     fontsize=17, gap_after_px=8)

    body = ("A “point” is one of 20 box positions — a 4 × 5 "
            "grid, 30 mm apart. Every one is carried to the same place pose "
            "on the table.")
    max_w = W - 2 * pad_px
    lines = _wrap_words(fig, renderer, body, 13, max_w)
    body_t = fig.text(margin_frac, y, "\n".join(lines), fontsize=13, color=TEXT,
                       ha="left", va="top", linespacing=1.3)
    bb = body_t.get_window_extent(renderer=renderer)
    y = bb.y0 / H - 10 / H

    # ---- scatter: the CAMERA's view of the grid, not the plan view --------
    # This card is overlaid beside the real footage (the overview cut's sim
    # pane, the RA-L cut's segment 1), and in that footage the twenty floor
    # markers are visible. It is therefore the *image* the diagram has to
    # agree with, which is not the orientation ral_explainers.grid_panel
    # uses -- that one sits beside compose_montage's tiles and has to agree
    # with them instead. The two are a 90 degree rotation apart, and both
    # are correct where they sit.
    #
    # Measured, not assumed (2026-09-07): detecting the cardboard box in the
    # first frame of all 20 hardware clips and least-squares fitting image
    # position against the known box position gives
    #     u (image right) = -125*world_x + 181*world_y + 567
    #     v (image down)  =  +58*world_x -  58*world_y + 800
    # i.e. the robot's left (+y) runs to the right of frame and away from
    # the robot (+x) runs down it. Confirmed independently by locating the
    # yellow floor markers themselves in one frame: they resolve into five
    # columns 12.5 px apart in u and rows 8 px apart in v -- five across,
    # four deep, which only matches y-across/x-down.
    #
    # So: horizontal = world y increasing right, vertical = world x
    # increasing DOWN. Point 0 lands top-left, 4 top-right, 15 bottom-left,
    # 19 bottom-right.
    npz_path = os.path.join(REPO, "data", "box_placement_grid.npz")
    grid = np.load(npz_path)
    pts = grid["grid_points"]
    x_far = pts[:, 0].max()
    y0 = pts[:, 1].min()
    xs_mm = (pts[:, 1] - y0) * 1e3          # horizontal: world y, right
    ys_mm = (x_far - pts[:, 0]) * 1e3       # vertical: world x, downward
    extent = max(xs_mm.max(), ys_mm.max())
    pad_mm = float(grid["grid_pitch"]) * 1e3 * 0.7

    # Reserve one line at the bottom for the point-0 caption before sizing
    # the (square) axes into whatever vertical room is left.
    caption_h_px = _text_height_px(fig, renderer, "Ag", 11) + 8
    axes_top = y
    axes_bottom_floor = (pad_px + caption_h_px) / H
    max_h_px = (axes_top - axes_bottom_floor) * H
    max_w_px = W - 2 * pad_px
    side_px = min(max_h_px, max_w_px)
    axes_w_frac, axes_h_frac = side_px / W, side_px / H
    axes_left_frac = (W - side_px) / 2 / W
    axes_bottom = axes_top - axes_h_frac
    ax = fig.add_axes((axes_left_frac, axes_bottom, axes_w_frac, axes_h_frac))
    ax.patch.set_alpha(0)

    # Square axes: equal data scaling AND a square axes box (standing
    # requirement in this repo for plots of physical scenes).
    # Centre each axis on its OWN data, not on the larger extent: the two
    # are 120 mm and 90 mm, so a shared centre would push the shorter axis
    # 15 mm off-centre inside the square box.
    half_span = extent / 2 + pad_mm
    cx, cy = xs_mm.max() / 2, ys_mm.max() / 2
    ax.set_xlim(cx - half_span, cx + half_span)
    ax.set_ylim(cy - half_span, cy + half_span)
    ax.set_aspect("equal")
    ax.set_box_aspect(1)
    ax.axis("off")  # no ticks/labels needed at this size (per spec)

    # Marker size is NOT scaled proportionally from ral_explainers'
    # grid_panel (s=230 in a ~432 px wide axes) -- at this card's much
    # smaller axes that scaling gave markers ~9 px across, which measured
    # illegible for a two-digit index at any font size. Instead: measure the
    # actual rendered extent of the widest label ("19") at the fontsize
    # we're about to use, require the marker to be at least that extent
    # times a safety factor across, and only keep the per-point labels if
    # that requirement still leaves a legible gap to the *next* marker
    # (point spacing, in px, comes straight from this axes' data-to-pixel
    # scale). If it doesn't fit, ship dots and the point-0 ring with no
    # labels rather than specks nobody can read.
    dx_per_mm = side_px / (2 * half_span)
    spacing_px = float(grid["grid_pitch"]) * 1e3 * dx_per_mm
    label_w = _text_width_px(fig, renderer, "19", _INDEX_FONTSIZE, weight="bold")
    label_h = _text_height_px(fig, renderer, "19", _INDEX_FONTSIZE, weight="bold")
    need_diam_px = max(label_w, label_h) * _INDEX_LABEL_SAFETY
    show_labels = need_diam_px <= spacing_px * _INDEX_LABEL_MAX_FILL
    marker_diam_px = need_diam_px if show_labels else min(18.0, spacing_px * 0.5)
    # scatter's `s` is marker area in points^2; dpi converts the px diameter
    # measured above into points before squaring.
    diam_pt = marker_diam_px * 72.0 / fig.dpi
    marker_s = np.pi * (diam_pt / 2) ** 2
    print(f"render_grid_inset: spacing={spacing_px:.1f}px, need_diam="
          f"{need_diam_px:.1f}px -> {'labels' if show_labels else 'dots only'} "
          f"(marker diam {marker_diam_px:.1f}px)")

    ax.scatter(xs_mm, ys_mm, s=marker_s, color=ACCENT, zorder=3, edgecolors="none")
    if show_labels:
        for idx, (px, py) in enumerate(zip(xs_mm, ys_mm)):
            ax.annotate(str(idx), (px, py), color=BG, fontsize=_INDEX_FONTSIZE,
                        fontweight="bold", ha="center", va="center", zorder=4)
    # Ring point 0 -- the run shown in the side-by-side beat.
    ax.scatter([xs_mm[0]], [ys_mm[0]], s=marker_s * 2.4, facecolors="none",
               edgecolors=ACCENT, linewidths=1.6, zorder=5)

    fig.text(0.5, axes_bottom - 4 / H, "point 0 — the run on screen",
              fontsize=11, color=MUTED, ha="center", va="top")

    fig.canvas.draw()
    arr = np.asarray(fig.canvas.buffer_rgba()).copy()
    plt.close(fig)
    plt.imsave(path, arr)


# --------------------------------------------------------------------- CLI

def _rects_overlap(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return min(ax1, bx1) > max(ax0, bx0) and min(ay1, by1) > max(ay0, by0)


def _check_inset():
    ok = True
    rect = (OVERVIEW_INSET_X, OVERVIEW_INSET_Y, OVERVIEW_INSET_X + INSET_W, OVERVIEW_INSET_Y + INSET_H)
    safe = (INSET_SAFE_X0, INSET_SAFE_Y0, INSET_SAFE_X1, INSET_SAFE_Y1)
    inside_safe = (safe[0] <= rect[0] and rect[2] <= safe[2] and
                   safe[1] <= rect[1] and rect[3] <= safe[3])
    print(f"  inset rect {rect} inside safe zone {safe} -> "
          f"{'OK' if inside_safe else 'FAIL'}")
    ok &= inside_safe
    clear_of_label = not _rects_overlap(rect, _SIM_LABEL_RECT)
    print(f"  inset clear of 'Simulation (Drake)' label {_SIM_LABEL_RECT} -> "
          f"{'OK' if clear_of_label else 'FAIL'}")
    ok &= clear_of_label
    return ok


def _run_check():
    print("Checking grid inset placement...")
    ok = _check_inset()
    print("[ OK ]" if ok else "[FAIL]", "montage_callouts geometry check")
    return ok


def _extract_still(src, t, out_path, vf=None):
    # -copyts: a documented trap in this pipeline (overview_captions.py,
    # ffmpeg seek ordering) -- `-ss` before `-i` otherwise resets
    # the decoded frames' own pts to ~0 at the seek point, which matters the
    # moment any filter reads `t` (a fade, say). Nothing extracted here uses
    # a live time-based filter, but paying the one flag up front is cheaper
    # than re-debugging this trap a third time.
    cmd = ["ffmpeg", "-y", "-ss", f"{t:.3f}", "-copyts", "-i", src]
    if vf:
        cmd += ["-vf", vf]
    cmd += ["-frames:v", "1", out_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"_extract_still({src}, t={t}) failed:\n{result.stderr[-500:]}")
        sys.exit(1)


def _cli_review_stills(out_dir):
    os.makedirs(out_dir, exist_ok=True)

    inset_png = os.path.join(out_dir, "00_grid_inset.png")
    render_grid_inset(inset_png)
    print(f"wrote {inset_png}")

    # ---- side-by-side + inset composite still ----
    sbs_src = os.path.join(VIDEO_DIR, "scratch_rby1_v2", "side_by_side.mp4")
    sbs_t = 7.0
    if not os.path.exists(sbs_src):
        sbs_src = os.path.join(VIDEO_DIR, "v2_rby1_hardware.mp4")
    sbs_raw = os.path.join(out_dir, "_sbs_raw.png")
    _extract_still(sbs_src, sbs_t, sbs_raw)
    base = Image.open(sbs_raw).convert("RGBA")
    inset = Image.open(inset_png).convert("RGBA")
    base.alpha_composite(inset, (OVERVIEW_INSET_X, OVERVIEW_INSET_Y))
    sbs_out = os.path.join(out_dir, "01_side_by_side_inset.png")
    base.convert("RGB").save(sbs_out)
    os.remove(sbs_raw)
    print(f"wrote {sbs_out}  (from {sbs_src} @ t={sbs_t}s)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--review-stills", action="store_true")
    ap.add_argument("--out-dir", default=os.path.join(REPO, "scratch", "overview_montage_review"))
    args = ap.parse_args()

    if not args.check and not args.review_stills:
        ap.error("nothing to do -- pass --check and/or --review-stills")

    ok = True
    if args.check:
        ok = _run_check()
    if args.review_stills:
        _cli_review_stills(args.out_dir)
    if args.check:
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
