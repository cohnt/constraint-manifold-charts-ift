"""On-screen captions for the overview / promotional video (Video 2).

Every new caption here is drawn by ``assemble_v2.normalize_segment``, not
baked into the segment it labels. Two of these segments (`iiwa_bimanual`,
`iiwa_iris`) are a full Cycles render and the rest go through a Blender
segment driver with a frame cache that only sometimes hits, so putting new
text in the render script would mean editing wording could cost a render.
Drawing it at assembly instead means editing a caption marks only the
`assemble` stage stale (see `build_video.stage_sources`'s mtime-based
staleness, and its trap noted there) -- an ffmpeg pass over already-rendered
frames, not a re-render.

``CAPTIONS`` is keyed by the segment name used in ``assemble_v2.SEGMENTS``.
Segments with no entry pass through unchanged. Boxes are given as
``(x, block_bottom_y, max_width)`` where ``block_bottom_y`` is the y of the
*bottom* of the text block, not its top -- ``block_top()`` derives the top
from the line count so ``caption_filter`` and the checker
(`check_overview_captions.py`) always agree on where the block actually sits.

``OCCUPIED`` records where each segment's *existing*, already-burned-in text
sits, duplicated by hand from the render scripts that draw it (see the
comment above ``OCCUPIED`` below) -- see this module's caution there before
trusting it against a script that has since been edited.
"""

import argparse
import os
import subprocess

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
VIDEO_DIR = os.path.join(REPO, "video")

FRAME_W, FRAME_H = 1920, 1080

# DejaVuSans *regular*, not the Bold face render_with_blender.py's burned-in
# titles use -- these captions are prose, not headings, and sit underneath
# that existing bold text rather than competing with it.
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_SIZE = 26
COLOR = "0xe0e0e0"
LINE_SPACING = 10
FADE_IN_S = 0.4
HOLD_S = 3.0


def _escape(text):
    """Escape a literal string for ffmpeg's drawtext= option.

    Based on render_with_blender.py:545-552 (render_segment1.py already keeps
    its own copy the same way). render_with_blender's titles/subtitles have
    never contained an apostrophe, so its loop just backslash-prefixes one
    the same as the other special characters -- but that does not survive
    ffmpeg's filtergraph parser here, and it fails in a way that does not
    point at the apostrophe: two backslash-escaped quotes in the *same*
    -vf chain (this caption text needs "RB-Y1's" and "robot's" in one
    caption's four-line drawtext chain) silently corrupted a *different*
    drawtext's arguments, which then failed with "No such filter" pointing
    at a fragment of that other filter's alpha expression. The standard
    ffmpeg-documented fix for one literal quote -- close the quote, insert
    an escaped quote, reopen: '\\'' -- rendered fine in isolation but broke
    again the moment a second one appeared in the same chain.
    Sidestepped rather than chased further: swap the apostrophe for the
    Unicode right single quotation mark (U+2019), which is not one of
    drawtext's special characters at all, so nothing needs escaping. DejaVu
    Sans renders it identically to the ASCII apostrophe at this size, and
    typesetting a contraction with a curly quote rather than a straight one
    is a wording-preserving substitution, not a reword.
    """
    for ch in ("\\", ":", "'", "%", ",", "[", "]", ";"):
        text = text.replace(ch, "\\" + ch)
    return text


# Segment name -> caption. `align` is "left", "right" or "center"; `x` is the
# left edge for "left", the right edge for "right", and unused for "center".
# `hold_s` is the tpad hold applied to the *whole segment* before the caption
# is drawn (see hold_filter) -- 0 means no hold from this module.
CAPTIONS = {
    "domain_extension": {
        "lines": [
            "IKFast returns no solution outside the",
            "reachable workspace. A least-squares",
            "projection extends the chart, so the",
            "optimizer never sees a NaN outside the",
            "feasible set.",
        ],
        # Left-aligned with a fixed left edge, not right-aligned against
        # x=1880: the block sits bottom-RIGHT only because the burned-in
        # q* equation owns the bottom-left, and a ragged right edge reads
        # more easily than a ragged left one.
        "x": 1360, "block_bottom_y": 1030, "max_width": 520,
        "align": "left", "hold_s": HOLD_S,
    },
    "boundary_reach": {
        "lines": [
            "The plot is the value of the boundary reachability constraint,",
            "evaluated as the arm is driven out of the feasible set. The",
            "workspace boundary is where the Jacobian loses rank, so holding",
            "this below τ keeps every pose reachable.",
        ],
        "x": 40, "block_bottom_y": 1030, "max_width": 1160,
        "align": "left", "hold_s": HOLD_S,
    },
    "iiwa_bimanual": {
        "lines": [
            # Not "two arms hold one plank": the IIWA experiment's
            # old_shelves.dmd.yaml loads two IIWAs, two WSG grippers, a table
            # and shelves and NO manipuland, and this segment comes from that
            # notebook's trajectory.html. The plank people remember is added
            # by this repo's generate_iiwa_meshcat.py, which is not in this
            # pipeline. Nothing is held on screen.
            "The two grippers must hold a constant relative transform throughout",
            "the motion. The planner searches the smaller space in which that",
            "already holds, so the constraint cannot be violated.",
        ],
        "x": 42, "block_bottom_y": 1030, "max_width": 1360,
        "align": "left", "hold_s": HOLD_S,
    },
    "iiwa_iris": {
        "lines": [
            "Every configuration in this region is collision-free and holds the",
            "grasp, so a planner can move anywhere inside it without checking again.",
            "The region is 8-dimensional; the two arms together have 14 joints.",
        ],
        "x": 42, "block_bottom_y": 1030, "max_width": 1360,
        "align": "left", "hold_s": HOLD_S,
    },
    "rby1_stability": {
        # Wrapped at ~700px rather than the usual wide line, and one line
        # longer for it (4 instead of 3) -- reviewed against the real frame:
        # the scene's support-polygon overlay is a *rendered* triangle at
        # x 830-1030, y 876-981 (measured, not from any script's drawtext
        # call, so it is not in OCCUPIED below and the checker cannot see
        # it), and the robot's base -- and so that triangle -- sit still for
        # the whole segment. The plan's original 3-line wrap ran every line
        # out to x ~= 830-920, straight through it. This wrap keeps every
        # line under x = 40 + 700 = 740, clearing the triangle by ~90px. The
        # sentence itself is unchanged, only where it breaks.
        "lines": [
            "The RB-Y1’s torso can bend and extend far enough to",
            "carry the robot’s center of mass outside its support",
            "polygon. Static stability is an inequality constraint on",
            "the same minimal coordinates.",
        ],
        # No hold: this segment is already 18.9 s, long enough to read without
        # padding (see the video pipeline's stage table).
        "x": 40, "block_bottom_y": 990, "max_width": 780,
        "align": "left", "hold_s": 0.0,
    },
    "rby1_hardware": {
        "lines": [
            "Grid point 0 of 20 — the same plan, in simulation and on the robot",
        ],
        # Single line at y=26 inside the 85 px top bar compose_rby1_v2.py
        # leaves empty; derived backward through the usual block_bottom_y/
        # block_top formula (26 + 1*(28+10) = 64) rather than adding a second
        # code path just for a one-line block.
        "x": None, "block_bottom_y": 64, "max_width": 1840,
        "align": "center", "hold_s": 0.0, "font_size": 28,
        # The hold for this segment's *tail* (the montage) lives in
        # compose_rby1_v2.py, not here -- it is baked in before assembly ever
        # sees this segment, unlike the tpad holds the other five get here.
        #
        # burned_upstream=True: caption_filter() does not draw this one.
        # compose_rby1_v2.py already draws this exact line (same text, same
        # x=(w-tw)/2:y=26, fontsize=28, #e0e0e0) into v2_rby1_hardware.mp4
        # itself -- verified by extracting a still with both this entry's
        # caption_filter *and* the segment's own burned-in text active, which
        # produced a visibly double-struck ghost of the same string. This
        # entry stays in CAPTIONS anyway so check_overview_captions.py still
        # validates the position (overflow/collision/contrast) that
        # compose_rby1_v2.py commits to -- it is the only cross-check that
        # exists between the two files, since they are owned and edited by
        # different agents in this pass.
        "burned_upstream": True,
    },
}

# These rectangles are NOT computed from anything -- they are hand-copied from
# the render scripts that draw the existing burned-in text, so the checker
# below can tell a new caption from colliding with it. If one of those scripts
# moves its text, this dict goes stale silently; nothing re-derives it.
OCCUPIED = {
    "domain_extension": [
        (30, 30, 30 + 400, 30 + 40, "render_domain_extension_blender.py:139 title"),
        (30, 70, 30 + 400, 70 + 30, "render_domain_extension_blender.py title/subtitle band"),
        (30, 1080 - 60, 30 + 600, 1080 - 60 + 40,
         "render_domain_extension_blender.py:151 (30, HEIGHT-60) equation"),
    ],
    "boundary_reach": [
        (20, 20, 20 + 400, 20 + 40, "render_boundary_reach_blender.py:168 (20,20) title"),
        (20, 55, 20 + 400, 55 + 30, "render_boundary_reach_blender.py:171/174 (20,55) subtitle"),
        # The whole right pane is the plot -- render_boundary_reach_blender.py:52
        # DRAKE_W = 1280 puts the arm render at x 0..1280, so the plot owns
        # x 1280..1920 for the segment's full height.
        (1280, 0, 1920, 1080, "render_boundary_reach_blender.py:52 DRAKE_W plot pane"),
    ],
    "iiwa_bimanual": [
        (42, 42, 42 + 600, 42 + 40, "render_with_blender.py:563 (42,42) fontsize=40 title"),
        (42, 98, 42 + 600, 98 + 30, "render_with_blender.py:567 (42,98) fontsize=28 subtitle"),
        (42, 146, 42 + 400, 146 + 28, "render_with_blender.py:575 (42,146) fontsize=26 speed"),
    ],
    "iiwa_iris": [
        (42, 42, 42 + 600, 42 + 40, "render_with_blender.py:563 (42,42) fontsize=40 title"),
        (42, 98, 42 + 600, 98 + 30, "render_with_blender.py:567 (42,98) fontsize=28 subtitle"),
        (42, 146, 42 + 400, 146 + 28, "render_with_blender.py:575 (42,146) fontsize=26 speed"),
    ],
    "rby1_stability": [
        (30, 30, 30 + 500, 30 + 40, "render_static_stability_blender.py:194 (30,30) title"),
        (30, 65, 30 + 400, 65 + 30, "render_static_stability_blender.py:196-202 (30,65) status"),
        (30, 90, 30 + 400, 90 + 30, "render_static_stability_blender.py:204 (30,90) rear margin"),
        (30, 115, 30 + 300, 115 + 28, "render_static_stability_blender.py:212 (30,115) speed"),
        (30, 1080 - 50, 30 + 500, 1080 - 50 + 30,
         "render_static_stability_blender.py:206 (30, HEIGHT-50) inset"),
        # A single label anchor, not a text box -- widened by a few px in each
        # direction since drawtext centres text on an anchor, not left-aligns it.
        (1258 - 60, 812 - 20, 1258 + 60, 812 + 20,
         "render_static_stability_blender.py:82 COM_LABEL_ANCHOR"),
    ],
    "rby1_hardware": [
        # NOT included: a rectangle for the 85 px top bar itself. That bar is
        # empty padding around the 960x910 side-by-side panes, not burned-in
        # text -- it is exactly where this segment's own caption goes (see
        # CAPTIONS["rby1_hardware"], burned_upstream). Only its bottom
        # counterpart carries text (below), plus the pane labels inside it.
        (10, 10, 10 + 300, 10 + 30, 'compose_rby1_v2.py "Simulation (Drake)"/"Hardware" pane labels'),
        (0, 1080 - 80, 1920, 1080, 'compose_rby1_v2.py y=h-80 "RB-Y1 Humanoid: 20/20 Pick-and-Place"'),
        # Side-by-side part only: the "what is a grid point" inset. Numbers
        # hand-copied from montage_callouts.OVERLAY_X/OVERLAY_Y/INSET_W/
        # INSET_H as of this writing -- like every other rectangle in this
        # dict, not re-derived, so it goes stale silently if those move.
        (20, 140, 20 + 400, 140 + 395,
         'compose_rby1_v2.py overlay=20:140 grid inset '
         '(montage_callouts.render_grid_inset, OVERLAY_X/Y/INSET_W/INSET_H)'),
        # NOT included: the results panel that fills the montage's one
        # remaining letterbox bar during this segment's tail
        # (ral_explainers.results_panel, via compose_rby1_v2.normalize_montage
        # -- the montage tail used to also carry a notes panel and a tile
        # outline/arrow overlay from montage_callouts.py; both are gone, see
        # that module's docstring). This dict has no time axis: a rectangle
        # in it is treated as occupied for the segment's whole duration. The
        # results panel and this segment's caption never share a frame -- the
        # caption is burned into the side-by-side part, the panel exists only
        # in the montage part -- so recording the panel here reports a
        # collision that cannot happen on screen.
    ],
}


def block_top(name):
    """Where the block of text starts (its top y), derived from block_bottom_y
    and the line count -- the one thing caption_filter and the checker must
    never compute independently, on pain of silently drifting apart."""
    cap = CAPTIONS[name]
    n = len(cap["lines"])
    line_h = cap.get("font_size", FONT_SIZE) + LINE_SPACING
    return cap["block_bottom_y"] - n * line_h


def hold_seconds(name):
    """The tpad hold, in seconds, this module wants applied to `name` -- 0.0
    if none. assemble_v2.py reads this directly (not by parsing hold_filter's
    output) to recompute the fade-out start against the padded duration."""
    cap = CAPTIONS.get(name)
    return cap["hold_s"] if cap else 0.0


def hold_filter(name):
    """The tpad filter fragment for `name`, or "" if it has no hold."""
    hold = hold_seconds(name)
    if hold <= 0:
        return ""
    return f"tpad=stop_mode=clone:stop_duration={hold:g}"


def caption_filter(name):
    """The drawtext filter chain for `name`, or "" if it has no caption.

    One drawtext per line rather than embedding newlines in a single
    drawtext -- simpler to reason about and to test line-by-line, and it is
    what render_with_blender.overlay_filter already does for its own
    multi-line title/subtitle/speed stack.
    """
    cap = CAPTIONS.get(name)
    if not cap:
        return ""
    if cap.get("burned_upstream"):
        # Already drawn into the segment itself (see the comment on this
        # entry) -- drawing it again here would double-strike the same text.
        return ""
    font_size = cap.get("font_size", FONT_SIZE)
    align = cap["align"]
    top = block_top(name)
    alpha = f"if(lt(t,{FADE_IN_S:g}),t/{FADE_IN_S:g},1)"
    parts = []
    for k, line in enumerate(cap["lines"]):
        y = top + k * (font_size + LINE_SPACING)
        if align == "right":
            x = f"{cap['x']}-tw"
        elif align == "center":
            x = "(w-tw)/2"
        else:
            x = str(cap["x"])
        parts.append(
            f"drawtext=fontfile={FONT_PATH}:text='{_escape(line)}'"
            f":fontsize={font_size}:fontcolor={COLOR}:borderw=2:bordercolor=black"
            f":alpha='{alpha}':x={x}:y={y}"
        )
    return ",".join(parts)


# Segment name -> its rendered source, for --review-stills. Kept separate from
# assemble_v2.SEGMENTS so this module has no import on assemble_v2 (it is
# assemble_v2 that imports this module, not the other way around).
_REVIEW_SOURCES = {
    "domain_extension": "v2_domain_extension.mp4",
    "boundary_reach": "v2_boundary_reach.mp4",
    "iiwa_bimanual": "v2_iiwa_bimanual.mp4",
    "iiwa_iris": "v2_iiwa_iris.mp4",
    "rby1_stability": "v2_rby1_stability.mp4",
    "rby1_hardware": "v2_rby1_hardware.mp4",
}


def _duration(path):
    return float(subprocess.check_output([
        "ffprobe", "-v", "quiet", "-show_entries", "format=duration",
        "-of", "csv=p=0", path,
    ]).decode().strip())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--review-stills", action="store_true",
                     help="extract one frame per caption with its filter "
                          "applied, for a fast visual review")
    ap.add_argument("--out-dir", default=os.path.join(REPO, "scratch", "overview_review"))
    args = ap.parse_args()

    if not args.review_stills:
        ap.error("nothing to do without --review-stills")

    os.makedirs(args.out_dir, exist_ok=True)
    for name in CAPTIONS:
        src = os.path.join(VIDEO_DIR, _REVIEW_SOURCES[name])
        if not os.path.exists(src):
            print(f"[skip] {name}: {src} does not exist")
            continue
        # A frame late in the segment so the caption sits over representative
        # content rather than the first frame of a fade-in. rby1_hardware is
        # fixed at t=5, inside the side-by-side part -- 80% of its duration
        # would land in the montage tail, which this caption is not about.
        if name == "rby1_hardware":
            t = 5.0
        else:
            t = 0.8 * _duration(src)
        out_path = os.path.join(args.out_dir, f"{name}.png")
        # -copyts: caption_filter's alpha fade-in reads the drawtext `t`
        # variable, which is the frame's own pts. Without -copyts, input
        # seeking (-ss before -i) resets pts to ~0 at the seek point, so every
        # extracted still landed inside the 0.4 s fade-in and the caption came
        # out invisible or half-drawn regardless of how far into the segment
        # `t` (the still's timestamp) actually was. In the real assembly
        # there is no seek -- the segment plays from its own t=0 -- so
        # -copyts is what makes this preview agree with that.
        cmd = ["ffmpeg", "-y", "-ss", f"{t:.2f}", "-copyts", "-i", src]
        vf = caption_filter(name)
        # rby1_hardware's entry is burned_upstream (see CAPTIONS): its
        # caption_filter is deliberately "" and the still is just the segment
        # as compose_rby1_v2.py already produced it, not a no-op passthrough.
        if vf:
            cmd += ["-vf", vf]
        cmd += ["-frames:v", "1", out_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"[FAIL] {name}:\n{result.stderr[-500:]}")
            continue
        print(out_path)


if __name__ == "__main__":
    main()
