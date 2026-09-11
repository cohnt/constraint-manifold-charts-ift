"""Assembly-time annotation pass for the RA-L supplementary video
(``assemble_v1.py``).

This is the RA-L cut's analogue of ``overview_captions.py`` on the overview
(Video 2) path: new on-screen artwork for an already-rendered segment,
applied by the *assembler* rather than baked into the segment itself. The
difference from ``overview_captions.py`` is the kind of artwork -- these are
not drawtext captions but small composited images (a card with a diagram, and
text boxes with arrows), because an arrow needs a fill and an outline that
``drawtext`` alone cannot draw, so each one is rendered once to a PNG (via
PIL, same approach as ``montage_callouts.py``'s tile outlines and leader
lines) and then looped and alpha-faded into the assembly's filter graph as an
extra input, the same trick ``compose_rby1_v2.py`` uses for its own
"what is a grid point" inset.

Why assembly time at all, rather than in the segment renderers: editing
``render_segment1.py`` or ``render_segment2.py`` marks their outputs stale,
and each is a multi-minute-to-25-minute rebuild (segment2 alone forks five
ffmpeg jobs plus a montage pass; see their own module docstrings). Every
number this module draws from -- burned-in text positions, cell geometry,
group timing windows -- is read off the *finished* renders (this module's own
comments record where each measurement came from) rather than recomputed by
importing those scripts, so a caption wording change here only re-runs
``assemble_v1.py``'s ~51 s concat pass, never a segment re-render.

Two pieces of artwork:

    SEG1_INSET -- the "what is a grid point" card, composited onto segment 1
        (grid point 0) at a different position than the overview cut's own
        placement of the same card. Built by *importing*
        ``montage_callouts.render_grid_inset`` -- see that function's
        docstring for why the same card is correct for both cuts.

    CALLOUTS -- six text-box-and-arrow callouts on segment 2, tying five
        "the torso folds forward here" observations and one "widest lean"
        observation to specific point cells in specific groups.

Public entry points, both keyed by the assembly segment name used in
``assemble_v1.SEGMENTS`` ("title" | "segment1" | "segment2"):

    overlay_inputs(name) -> [png_path, ...]
        Extra still-image files ``assemble_v1.normalize_segment`` must add as
        looped ``-i`` inputs, in the order ``overlay_filter`` below assumes
        they land at ffmpeg input indices 1..N (input 0 is always the
        segment's own base video). Renders the PNGs fresh on every call --
        cheap (a handful of small PIL/matplotlib images), and it means a
        wording or placement edit here needs no separate "regenerate assets"
        step.

    overlay_filter(name, base_label) -> str
        The filter_complex fragment that composites those inputs onto
        `base_label` (e.g. ``"[scaled]"``), fading each in and out at its own
        window, ending at the fixed output label ``"[annotated]"``. Empty
        string for a segment with nothing to draw (title, and anything with
        no ``overlay_inputs``) -- the caller then skips filter_complex
        entirely, per the hard requirement that segments with no annotations
        render byte-for-byte as before.

Run directly to check the geometry or dump review stills:

    venv/bin/python scripts/video/ral_annotations.py --check
    venv/bin/python scripts/video/ral_annotations.py --review-stills
"""

import argparse
import os
import subprocess
import sys

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# The shared "what is a point" card. render_grid_inset draws exactly the
# artwork wanted here (heading, sentence, 4x5 diagram with point 0 ringed) --
# and segment 1 *is* point 0, so the ring is correct for this cut too. See
# its docstring in montage_callouts.py for the size it always renders at
# (INSET_W x INSET_H) and why that module now frames it as shared rather than
# overview-only.
from montage_callouts import render_grid_inset, INSET_W, INSET_H  # noqa: E402
from ral_explainers import BG, TEXT, ACCENT, CORAL, MUTED  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
VIDEO_DIR = os.path.join(REPO, "video")
ASSET_DIR = os.path.join(REPO, "scratch", "ral_annotations")

FRAME_W, FRAME_H = 1920, 1080

# ===================================================================
# Segment 1: the grid-inset card
# ===================================================================

# render_segment1.py's own layout (not re-derived here, just recorded: the
# hardware pane is x 0..1280, the plot column -- where the explainer screens
# live -- is x 1280..1920). The card must stay inside the hardware pane; nothing
# here can safely composite over the plot column, which is real matplotlib
# content for the whole 46.4 s segment.
HW_PANE_X0, HW_PANE_X1 = 0, 1280

# Measured (2026-09-07) directly on video/segment1_point00.mp4: a
# max-minus-min activity map over frames sampled every 2 s, restricted to the
# rectangle (820,150)-(820+INSET_W, 150+INSET_H). Fraction of pixels with more
# than 25 levels of variation inside that rectangle:
#   t in [14, 34] s -> 0.0004   (effectively frozen)
#   t in [14, 40] s -> 0.0009   (still effectively frozen)
#   t in [14, 44] s -> 0.049    (the arms are sweeping back into frame by here)
# against 0.34 for the same-sized rectangle at the pane's centre over the same
# window. The card only needs to be readable inside [14, 40) s, so its window
# (below) sits with margin inside the first two measurements and clears well
# before the 44 s mark where the background starts moving again.
SEG1_INSET_X, SEG1_INSET_Y = 820, 150

# The explainer screens (ral_explainers.explainer_screen_a/b) occupy the plot
# column from segment-local t=0.3 to t=14.21 (A fades in 0.3-0.8, holds to
# 7.26, crossfades to B by 7.76, B holds to 13.51, fades out by 14.21) --
# render_segment1.py, not re-derived here. This card's fade-in starts at
# 14.4 s, just after B has fully cleared, so the two explanatory surfaces
# (plot-column text, hardware-pane card) are never both mid-fade at once.
SEG1_INSET_FADE_IN_START_S = 14.4
SEG1_INSET_FADE_IN_DUR_S = 0.5
SEG1_INSET_FADE_OUT_END_S = 39.5
SEG1_INSET_FADE_OUT_DUR_S = 0.5

# The four burned-in text blocks on segment 1's hardware pane
# (render_segment1.py's composite_videos), as (text, fontsize, x, y) with y
# meaning drawtext's own y (top of the glyph box), in the same
# DejaVuSans-Bold face every one of them uses. "Carrying — constraint
# active" stands in for the phase ribbon (x=24, y=112): it is the longest of
# the five phase labels build_phase_segments can emit, so measuring its rect
# is the conservative choice for a slot whose actual text changes over time.
_SEG1_FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
_SEG1_BURNED_TEXT = [
    ("Grid point 0 of 20", 36, 24, 24),
    ("one of 20 box pick-up positions", 22, 24, 72),
    ("Carrying — constraint active", 26, 24, 112),
    ("1× real time", 28, 24, FRAME_H - 48),
]


def _text_rect(text, fontsize, x, y, font_path=_SEG1_FONT_PATH):
    """(x0, y0, x1, y1) of `text` at `fontsize`/`font_path`, anchored the way
    ffmpeg's drawtext anchors it (x, y = top-left of the glyph box), measured
    from the font's own metrics rather than guessed -- borderw=2 in the
    original drawtext calls adds a couple of px on every side, folded in
    here as a flat margin rather than modelled exactly."""
    font = ImageFont.truetype(font_path, fontsize)
    width = font.getlength(text)
    ascent, descent = font.getmetrics()
    margin = 3  # borderw=2 stroke, rounded up
    return (x - margin, y - margin, x + width + margin, y + ascent + descent + margin)


def seg1_inset_rect():
    return (SEG1_INSET_X, SEG1_INSET_Y, SEG1_INSET_X + INSET_W, SEG1_INSET_Y + INSET_H)


# ===================================================================
# Segment 2: the fold/lean callouts
# ===================================================================

# Cell geometry, matching render_segment2.py's CELL_W/CELL_H/GRID_NX exactly
# (duplicated as plain numbers, not imported: render_segment2.py is on the
# do-not-edit list for this change, and importing it would still mark it
# "read" by a script that watches its own mtime for staleness in
# build_video.py's dependency graph -- these four numbers are stable video
# geometry, not something this module should need render_segment2.py's
# 500-line renderer in `sys.path` to state).
CELL_W, CELL_H = 480, 540
GRID_NX = 4

# Five groups' start/end times in segment2-LOCAL time, ffprobe'd from the
# real build intermediates (2026-09-07) rather than recomputed from
# group_duration() -- each group's length is the longest of its own four
# clips sped up 2x, which is a per-clip fact this module has no business
# re-deriving. The montage tail (122.333-128.300) plays no callout.
GROUP_WINDOWS = {
    0: (0.000, 23.200),    # points 0-3
    1: (23.200, 48.533),   # points 4-7
    2: (48.533, 75.600),   # points 8-11
    3: (75.600, 99.600),   # points 12-15
    4: (99.600, 122.333),  # points 16-19
}

# Every group's grasp (start of the first constrained leg, from
# video/error_data.pkl) falls 5.7-7.3 s into the group at 2x speed
# (verified fact, not re-derived here). A window of group_start+3.0 to
# group_start+15.0 therefore covers the approach, the grasp itself and the
# lift for every group with margin on both ends -- shortest group (group 3,
# 24.0 s long) still has 9 s of margin after the window closes before the
# next group's clips start.
CALLOUT_WINDOW_START_OFFSET_S = 3.0
CALLOUT_WINDOW_END_OFFSET_S = 15.0
CALLOUT_FADE_S = 0.4

# All numbers below come from meta["q_grasp"][3:9] -- the torso block of the
# 23-D active vector (src/rby1_planning.py's ActiveLayout: torso =
# slice(3, 9)) -- read from the committed plans/grid_cache/point_NN.pkl
# through plan_format.plan_io.load_plan, at the grasp waypoint, exactly
# like montage_callouts.OBSERVATIONS (see its comment for the pitch/roll/
# twist definitions this reuses). Verified across all 20 cached points
# (2026-09-07): total torso pitch (torso_1+torso_2+torso_3) is 100-111 deg
# for points {2, 8, 11, 13, 16} against 50-71 deg for the other fifteen, and
# those five span all four grid columns -- box position does not predict the
# branch. Point 19's roll (torso_0+torso_4) is -24.2 deg, the largest
# magnitude of the twenty (rounds to the "24°" below).
#
# A third observation (mean |twist| rising left-to-right across the grid
# columns) was in the overview cut's drafted notes and cut there for
# legibility at tile size (montage_callouts.py's own comment on
# OBSERVATIONS) -- it is not repeated here either. The RA-L cut's cells are
# four times a montage tile's area, but the reason for cutting it was that
# 16.6 deg of twist is not a *visibly different posture* the way a 100+ deg
# fold or a 24 deg lean is, not that it wouldn't fit; that reason applies
# here just as much.
CALLOUTS = [
    dict(group=0, point=2, color=ACCENT, text=(
        "The whole body adapts. Point 2 folds the torso forward instead of "
        "standing tall; the base never moves, so every difference between "
        "these runs is torso and arms.")),
    dict(group=2, point=8, color=ACCENT, text="Folded forward here too"),
    dict(group=2, point=11, color=ACCENT, text="Folded forward here too"),
    dict(group=3, point=13, color=ACCENT, text=(
        "Point 13 folds forward as well. Five of the twenty do, and the box "
        "position does not predict which — it is a different branch of the "
        "analytic IK.")),
    dict(group=4, point=16, color=ACCENT, text="Folded forward here too"),
    dict(group=4, point=19, color=CORAL, text=(
        "Point 19’s box is the farthest away and the farthest left. The "
        "torso leans sideways by 24° — the widest posture of the twenty.")),
]

_CALLOUT_FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
_CALLOUT_FONT_SIZE = 19
_CALLOUT_LINE_H_PX = 26          # measured leading a touch over 19px text
_CALLOUT_PAD_X_PX = 16
_CALLOUT_PAD_TOP_PX = 14
_CALLOUT_PAD_BOTTOM_PX = 14
_CALLOUT_BOX_LEFT_PX = 12        # cell-local; left/right margins both 12px
_CALLOUT_BOX_W_PX = 456          # so the box, and the arrow under it, are
                                  # centred in the 480px cell (12+456+12=480)
_CALLOUT_BOX_TOP_PX = 56         # clears the "Point N" label (y 8..~40)
_CALLOUT_ARROW_TIP_Y_PX = 250    # the robot's head/torso at the grasp
_CALLOUT_MIN_ARROW_LEN_PX = 30
_CALLOUT_BORDER_RGB = (0xa0, 0xa0, 0xa0)   # MUTED
_CALLOUT_BG_RGBA = (0x1a, 0x1a, 0x2e, round(0.85 * 255))  # BG @ 0.85 alpha
_CALLOUT_TEXT_RGB = (0xe0, 0xe0, 0xe0)     # TEXT


def _cell_x(point_idx):
    """Left edge, in the assembled 1920px frame, of the video cell a point
    lands in -- render_segment2.py's own indexing: cell = idx % 4 (idx
    counts 0-3 within whichever 4-point group it is in), video row spans the
    full CELL_W per cell starting at x=0."""
    return (point_idx % GRID_NX) * CELL_W


def _wrap_glyphs(font, text, max_width_px):
    """Greedy word-wrap using the font's own rendered width (font.getlength),
    not a guessed character count -- same approach as
    ral_explainers._wrap_words/montage_callouts._wrap_words, reimplemented
    for PIL's ImageFont rather than matplotlib's text-extent API since this
    module draws with PIL throughout (it needs filled arrowheads and rounded
    boxes, which PIL's ImageDraw does directly)."""
    words = text.split()
    lines, cur = [], []
    for word in words:
        trial = " ".join(cur + [word])
        if cur and font.getlength(trial) > max_width_px:
            lines.append(" ".join(cur))
            cur = [word]
        else:
            cur.append(word)
    if cur:
        lines.append(" ".join(cur))
    return lines


def _callout_box_rect(lines):
    """(left, top, right, bottom) of a callout's text box, cell-local."""
    h = _CALLOUT_PAD_TOP_PX + len(lines) * _CALLOUT_LINE_H_PX + _CALLOUT_PAD_BOTTOM_PX
    return (_CALLOUT_BOX_LEFT_PX, _CALLOUT_BOX_TOP_PX,
            _CALLOUT_BOX_LEFT_PX + _CALLOUT_BOX_W_PX, _CALLOUT_BOX_TOP_PX + h)


def _callout_arrow(lines):
    """(start_xy, tip_xy), cell-local: straight down from the box's bottom
    edge, centred on the box, to the fixed tip height. Both the box's left
    margin and its width are symmetric (12px both sides of a 456px box in a
    480px cell), so the box's centre x is always cell-local 240 -- matching
    the fixed tip x the spec measured against the robot's torso, with no
    per-callout centring math needed.

    If a callout's box ever grows tall enough to leave less than the
    minimum arrow length, the fallback is to shrink the box's padding rather
    than move the fixed tip (which is pinned to where the robot actually is
    in every clip, not to whatever box happens to be above it) -- see the
    assertion below, which is what would catch that case; at the six
    callouts actually shipped here (1-4 lines) it is never exercised.
    """
    _, _, right, bottom = _callout_box_rect(lines)
    cx = right - _CALLOUT_BOX_W_PX / 2
    tip = (cx, _CALLOUT_ARROW_TIP_Y_PX)
    start = (cx, bottom)
    length = tip[1] - start[1]
    if length < _CALLOUT_MIN_ARROW_LEN_PX:
        raise RuntimeError(
            f"callout arrow only {length:.0f}px (need >= "
            f"{_CALLOUT_MIN_ARROW_LEN_PX}px) -- shrink _CALLOUT_LINE_H_PX/"
            "_CALLOUT_PAD_*_PX for this box, don't move the tip")
    return start, tip


def _draw_arrowhead(draw, tip, rgb, ss, size_px=9):
    """Small filled triangle at `tip`, pointing straight down (every callout
    arrow in this module points down), outlined in dark so it reads over
    footage of any brightness. Same construction as
    montage_callouts._draw_arrowhead's "down" case."""
    s = size_px * ss
    tx, ty = tip[0] * ss, tip[1] * ss
    pts = [(tx, ty), (tx - s * 0.6, ty - s), (tx + s * 0.6, ty - s)]
    draw.polygon(pts, fill=rgb + (255,), outline=(0, 0, 0, 255))


def _render_callout(callout, path):
    """Render one callout (rounded text box + downward arrow) as a
    CELL_W x CELL_H transparent RGBA PNG, cell-local -- assemble_v1.py
    overlays it at (cell_x, 0)."""
    font = ImageFont.truetype(_CALLOUT_FONT_PATH, _CALLOUT_FONT_SIZE)
    max_text_w = _CALLOUT_BOX_W_PX - 2 * _CALLOUT_PAD_X_PX
    lines = _wrap_glyphs(font, callout["text"], max_text_w)
    box = _callout_box_rect(lines)
    start, tip = _callout_arrow(lines)
    rgb = callout["color"]
    if isinstance(rgb, str):
        # ACCENT/CORAL come in as "#4fc3f7"-style hex from ral_explainers.
        rgb = tuple(int(rgb[i:i + 2], 16) for i in (1, 3, 5))

    # Supersample 4x for anti-aliased rounded corners and arrow edges, same
    # trick as montage_callouts.render_annotation_overlay -- plain PIL
    # ImageDraw has no anti-aliasing of its own.
    SS = 4
    img = Image.new("RGBA", (CELL_W * SS, CELL_H * SS), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    box_ss = [c * SS for c in box]
    draw.rounded_rectangle(box_ss, radius=10 * SS, fill=_CALLOUT_BG_RGBA,
                            outline=_CALLOUT_BORDER_RGB + (255,), width=SS)

    text_x = (box[0] + _CALLOUT_PAD_X_PX) * SS
    text_y = (box[1] + _CALLOUT_PAD_TOP_PX) * SS
    for i, line in enumerate(lines):
        draw.text((text_x, text_y + i * _CALLOUT_LINE_H_PX * SS), line,
                   font=ImageFont.truetype(_CALLOUT_FONT_PATH, _CALLOUT_FONT_SIZE * SS),
                   fill=_CALLOUT_TEXT_RGB + (255,))

    poly = [(start[0] * SS, start[1] * SS), (tip[0] * SS, tip[1] * SS)]
    draw.line(poly, fill=(0, 0, 0, 255), width=round(4.5 * SS), joint="curve")
    draw.line(poly, fill=rgb + (255,), width=round(2.5 * SS), joint="curve")
    _draw_arrowhead(draw, tip, rgb, SS)

    img = img.resize((CELL_W, CELL_H), Image.LANCZOS)
    img.save(path)


def callout_window(callout):
    """(start, end) in segment2-local time for `callout`'s fade window."""
    g0, _g1 = GROUP_WINDOWS[callout["group"]]
    return g0 + CALLOUT_WINDOW_START_OFFSET_S, g0 + CALLOUT_WINDOW_END_OFFSET_S


# ===================================================================
# Public API
# ===================================================================

def overlay_inputs(name):
    """PNG paths for `name`'s overlays, in the order overlay_filter assumes
    they are ffmpeg inputs 1..N (input 0 is the segment's own base video).
    Empty list for a segment with nothing to draw."""
    os.makedirs(ASSET_DIR, exist_ok=True)
    if name == "segment1":
        path = os.path.join(ASSET_DIR, "seg1_inset.png")
        render_grid_inset(path)
        return [path]
    if name == "segment2":
        paths = []
        for i, c in enumerate(CALLOUTS):
            p = os.path.join(ASSET_DIR, f"seg2_callout_{i:02d}_g{c['group']}_pt{c['point']:02d}.png")
            _render_callout(c, p)
            paths.append(p)
        return paths
    return []


def overlay_filter(name, base_label):
    """The filter_complex fragment compositing `name`'s overlays onto
    `base_label` (e.g. "[scaled]"). Always ends at the fixed output label
    "[annotated]" so the caller does not need to know how many overlays a
    segment has. Empty string if `name` has none."""
    if name == "segment1":
        fi0, fi1 = SEG1_INSET_FADE_IN_START_S, SEG1_INSET_FADE_IN_START_S + SEG1_INSET_FADE_IN_DUR_S
        fo0 = SEG1_INSET_FADE_OUT_END_S - SEG1_INSET_FADE_OUT_DUR_S
        return (
            f"[1:v]format=rgba,"
            f"fade=in:st={SEG1_INSET_FADE_IN_START_S:g}:d={SEG1_INSET_FADE_IN_DUR_S:g}:alpha=1,"
            f"fade=out:st={fo0:g}:d={SEG1_INSET_FADE_OUT_DUR_S:g}:alpha=1[seg1inset];"
            f"{base_label}[seg1inset]overlay={SEG1_INSET_X}:{SEG1_INSET_Y}[annotated]"
        )

    if name == "segment2":
        parts = []
        prev = base_label
        n = len(CALLOUTS)
        for i, c in enumerate(CALLOUTS, start=1):
            start, end = callout_window(c)
            fade_start_label = f"cofade{i}"
            out_label = "annotated" if i == n else f"cov{i}"
            parts.append(
                f"[{i}:v]format=rgba,"
                f"fade=in:st={start:g}:d={CALLOUT_FADE_S:g}:alpha=1,"
                f"fade=out:st={end - CALLOUT_FADE_S:g}:d={CALLOUT_FADE_S:g}:alpha=1"
                f"[{fade_start_label}]"
            )
            x = _cell_x(c["point"])
            parts.append(f"{prev}[{fade_start_label}]overlay={x}:0[{out_label}]")
            prev = f"[{out_label}]"
        return ";".join(parts)

    return ""


# --------------------------------------------------------------------- CLI

def _rects_overlap(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return min(ax1, bx1) > max(ax0, bx0) and min(ay1, by1) > max(ay0, by0)


def _check_seg1():
    ok = True
    rect = seg1_inset_rect()
    inside_pane = HW_PANE_X0 <= rect[0] and rect[2] <= HW_PANE_X1
    print(f"  seg1 inset rect {tuple(round(v) for v in rect)} inside hardware "
          f"pane x[{HW_PANE_X0},{HW_PANE_X1}] -> {'OK' if inside_pane else 'FAIL'}")
    ok &= inside_pane

    for text, fontsize, x, y in _SEG1_BURNED_TEXT:
        tr = _text_rect(text, fontsize, x, y)
        clear = not _rects_overlap(rect, tr)
        print(f"  seg1 inset clear of {text!r}@{fontsize}px rect "
              f"{tuple(round(v) for v in tr)} -> {'OK' if clear else 'FAIL'}")
        ok &= clear

    win_ok = (SEG1_INSET_FADE_IN_START_S < SEG1_INSET_FADE_OUT_END_S and
              SEG1_INSET_FADE_IN_DUR_S > 0 and SEG1_INSET_FADE_OUT_DUR_S > 0)
    print(f"  seg1 inset window [{SEG1_INSET_FADE_IN_START_S}, "
          f"{SEG1_INSET_FADE_OUT_END_S}] well-formed -> {'OK' if win_ok else 'FAIL'}")
    ok &= win_ok
    return ok


def _check_seg2():
    ok = True
    for i, c in enumerate(CALLOUTS):
        cell_x = _cell_x(c["point"])
        font = ImageFont.truetype(_CALLOUT_FONT_PATH, _CALLOUT_FONT_SIZE)
        lines = _wrap_glyphs(font, c["text"], _CALLOUT_BOX_W_PX - 2 * _CALLOUT_PAD_X_PX)
        box = _callout_box_rect(lines)
        label = f"g{c['group']}/pt{c['point']}"

        in_cell = 0 <= box[0] and box[2] <= CELL_W
        below_label = box[1] >= 48   # "Point N" label occupies roughly y 8..40
        print(f"  [{label}] box (cell-local) {tuple(round(v) for v in box)}, "
              f"{len(lines)} line(s) -> in-cell {'OK' if in_cell else 'FAIL'}, "
              f"below label {'OK' if below_label else 'FAIL'}")
        ok &= in_cell and below_label

        try:
            start, tip = _callout_arrow(lines)
        except RuntimeError as exc:
            print(f"  [{label}] arrow: FAIL ({exc})")
            ok = False
            continue
        length = tip[1] - start[1]
        len_ok = length >= _CALLOUT_MIN_ARROW_LEN_PX
        starts_at_box = abs(start[1] - box[3]) < 1e-6
        tip_in_cell = 0 <= tip[0] <= CELL_W and 0 <= tip[1] <= CELL_H
        print(f"  [{label}] arrow length {length:.0f}px -> "
              f"{'OK' if len_ok else 'FAIL (< 30px)'}; starts at box bottom -> "
              f"{'OK' if starts_at_box else 'FAIL'}; tip inside cell -> "
              f"{'OK' if tip_in_cell else 'FAIL'}")
        ok &= len_ok and starts_at_box and tip_in_cell

        g0, g1 = GROUP_WINDOWS[c["group"]]
        start_t, end_t = callout_window(c)
        win_ok = g0 <= start_t and end_t <= g1
        print(f"  [{label}] window [{start_t:.2f}, {end_t:.2f}] inside group "
              f"{c['group']}'s window [{g0:.2f}, {g1:.2f}] -> "
              f"{'OK' if win_ok else 'FAIL'}")
        ok &= win_ok

        abs_cell_x = cell_x
        print(f"  [{label}] video cell x=[{abs_cell_x}, {abs_cell_x + CELL_W}]")
    return ok


def _run_check():
    print("Checking segment 1 inset placement...")
    seg1_ok = _check_seg1()
    print("Checking segment 2 callout geometry...")
    seg2_ok = _check_seg2()
    ok = seg1_ok and seg2_ok
    print("[ OK ]" if ok else "[FAIL]", "ral_annotations geometry check")
    return ok


def _extract_still(src, t, out_path):
    # -copyts before -ss+-i: a documented trap in this pipeline
    # (montage_callouts.py, overview_captions.py, and this module
    # handoff.md) -- without it, a filter reading the frame's own `t` sees
    # ~0 at the seek point rather than the real timestamp. Nothing in this
    # extraction itself uses a time-based filter (it is a plain single-frame
    # grab from an already-rendered segment; the composite happens afterward
    # in PIL, not in ffmpeg), but paying the flag up front matches the rest
    # of this pipeline and costs nothing.
    cmd = ["ffmpeg", "-y", "-ss", f"{t:.3f}", "-copyts", "-i", src,
           "-frames:v", "1", out_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"_extract_still({src}, t={t}) failed:\n{result.stderr[-500:]}")
        sys.exit(1)


def _cli_review_stills(out_dir):
    os.makedirs(out_dir, exist_ok=True)

    seg1_src = os.path.join(VIDEO_DIR, "segment1_point00.mp4")
    seg2_src = os.path.join(VIDEO_DIR, "segment2_all20.mp4")

    inset_png = os.path.join(out_dir, "_seg1_inset.png")
    render_grid_inset(inset_png)
    inset = Image.open(inset_png).convert("RGBA")

    idx = 0
    for t in (20.0, 38.0):
        raw = os.path.join(out_dir, "_raw.png")
        _extract_still(seg1_src, t, raw)
        base = Image.open(raw).convert("RGBA")
        base.alpha_composite(inset, (SEG1_INSET_X, SEG1_INSET_Y))
        out_path = os.path.join(out_dir, f"{idx:02d}_seg1_inset_t{t:.0f}.png")
        base.convert("RGB").save(out_path)
        os.remove(raw)
        print(f"wrote {out_path}  (from {seg1_src} @ t={t:.0f}s)")
        idx += 1
    os.remove(inset_png)

    for c in CALLOUTS:
        g0, _g1 = GROUP_WINDOWS[c["group"]]
        t = g0 + 7.0   # the grasp: 5.7-7.3s into every group at 2x speed
        callout_png = os.path.join(out_dir, "_callout.png")
        _render_callout(c, callout_png)
        raw = os.path.join(out_dir, "_raw.png")
        _extract_still(seg2_src, t, raw)
        base = Image.open(raw).convert("RGBA")
        callout = Image.open(callout_png).convert("RGBA")
        base.alpha_composite(callout, (_cell_x(c["point"]), 0))
        out_path = os.path.join(
            out_dir, f"{idx:02d}_seg2_g{c['group']}_pt{c['point']:02d}.png")
        base.convert("RGB").save(out_path)
        os.remove(raw)
        os.remove(callout_png)
        print(f"wrote {out_path}  (from {seg2_src} @ t={t:.2f}s)")
        idx += 1


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--review-stills", action="store_true")
    ap.add_argument("--out-dir", default=os.path.join(REPO, "scratch", "ral_review"))
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
