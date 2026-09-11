"""Sanity check for overview_captions.CAPTIONS: does any caption box overflow
the frame or its own declared max_width, collide with a segment's existing
burned-in text, or sit on a backdrop too close to its own text colour to
read?

Modelled on check_explainer_layout.py's contract: three independent finding
classes, printed and counted, `sys.exit(1)` if any fired.

  overflow  -- a line's rendered width exceeds its caption's max_width, or
               the caption's block leaves the 1920x1080 frame.
  collision -- the caption's block rectangle intersects one of the rectangles
               named in OCCUPIED for that segment. OCCUPIED is hand-copied
               from the render scripts (see the comment above it in
               overview_captions.py) and only covers *burned-in* text -- a
               collision with something else in the frame (a rendered 3D
               object, say) will not be caught here; that class of problem
               was found by eye during --review-stills, not by this checker.
  contrast  -- the real backdrop under the block, sampled from the segment's
               own mp4 with ffmpeg's signalstats, is within 60 luma of the
               caption text colour (#e0e0e0, luma ~224). Skipped (not
               failed) if ffmpeg or the segment mp4 is unavailable.

Run:

    venv/bin/python scripts/video/check_overview_captions.py
"""

import os
import re
import subprocess
import sys

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import overview_captions as oc

# Luma of #e0e0e0 (COLOR), computed the same way ffmpeg's signalstats reports
# YAVG (BT.601 luma): 0.299*R + 0.587*G + 0.114*B, R=G=B=0xe0=224 -> 224.
_TEXT_LUMA = 224.0
_CONTRAST_MIN_DELTA = 60.0


def _line_geometry(name, k, line, draw):
    """(x0, y0, x1, y1) of one caption line's rendered ink, plus its advance
    width -- the same quantity ffmpeg's drawtext calls `tw` and uses for
    x=<edge>-tw / x=(w-tw)/2, so right- and centre-aligned boxes land where
    caption_filter will actually put them."""
    cap = oc.CAPTIONS[name]
    font_size = cap.get("font_size", oc.FONT_SIZE)
    font = ImageFont.truetype(oc.FONT_PATH, font_size)
    width = draw.textlength(line, font=font)
    top = oc.block_top(name)
    y = top + k * (font_size + oc.LINE_SPACING)
    align = cap["align"]
    if align == "right":
        x0 = cap["x"] - width
    elif align == "center":
        x0 = (oc.FRAME_W - width) / 2
    else:
        x0 = cap["x"]
    bbox = draw.textbbox((x0, y), line, font=font)
    return bbox, width


def _boxes_overlap(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ox = min(ax1, bx1) - max(ax0, bx0)
    oy = min(ay1, by1) - max(ay0, by0)
    return ox > 0 and oy > 0


def _sample_backdrop_luma(name, box):
    """Mean luma under `box` in the segment's own mp4, or None if the mp4 or
    ffmpeg is unavailable. Sampled at the same point in the segment
    --review-stills uses, so a "clean" run and a "clean" still agree."""
    src_name = oc._REVIEW_SOURCES.get(name)
    if src_name is None:
        return None
    src = os.path.join(oc.VIDEO_DIR, src_name)
    if not os.path.exists(src):
        return None

    x0, y0, x1, y1 = box
    x0 = max(0, int(x0))
    y0 = max(0, int(y0))
    x1 = min(oc.FRAME_W, int(round(x1)))
    y1 = min(oc.FRAME_H, int(round(y1)))
    w, h = x1 - x0, y1 - y0
    if w <= 0 or h <= 0:
        return None

    t = 5.0 if name == "rby1_hardware" else 0.8 * oc._duration(src)
    crop = f"crop={w}:{h}:{x0}:{y0}"
    cmd = [
        "ffmpeg", "-ss", f"{t:.2f}", "-i", src,
        "-vf", f"{crop},signalstats,metadata=print:key=lavfi.signalstats.YAVG:file=-",
        "-frames:v", "1", "-f", "null", "/dev/null",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    m = re.search(r"lavfi\.signalstats\.YAVG=([\d.]+)", result.stdout + result.stderr)
    if not m:
        return None
    return float(m.group(1))


def check_caption(name):
    cap = oc.CAPTIONS[name]
    img = Image.new("RGB", (1, 1))
    draw = ImageDraw.Draw(img)

    overflows = []
    line_boxes = []
    for k, line in enumerate(cap["lines"]):
        bbox, width = _line_geometry(name, k, line, draw)
        line_boxes.append(bbox)
        if width > cap["max_width"]:
            overflows.append((line, f"width {width:.0f}px > max_width {cap['max_width']}px"))
        x0, y0, x1, y1 = bbox
        if x0 < 0 or y0 < 0 or x1 > oc.FRAME_W or y1 > oc.FRAME_H:
            overflows.append((line, f"box=({x0:.0f}, {y0:.0f}, {x1:.0f}, {y1:.0f}) "
                                     f"outside [0, {oc.FRAME_W}] x [0, {oc.FRAME_H}]"))

    block_box = (
        min(b[0] for b in line_boxes), min(b[1] for b in line_boxes),
        max(b[2] for b in line_boxes), max(b[3] for b in line_boxes),
    )

    collisions = []
    for rect in oc.OCCUPIED.get(name, []):
        rx0, ry0, rx1, ry1, provenance = rect
        if _boxes_overlap(block_box, (rx0, ry0, rx1, ry1)):
            collisions.append(provenance)

    contrast = []
    luma = _sample_backdrop_luma(name, block_box)
    if luma is None:
        print(f"  [skip] {name}: contrast not checked (ffmpeg or segment mp4 unavailable)")
    elif abs(luma - _TEXT_LUMA) < _CONTRAST_MIN_DELTA:
        contrast.append(f"backdrop luma {luma:.0f} within {_CONTRAST_MIN_DELTA:.0f} "
                         f"of text luma {_TEXT_LUMA:.0f}")

    n_bad = len(overflows) + len(collisions) + len(contrast)
    if n_bad:
        print(f"[FAIL] {name}: {len(overflows)} overflow, {len(collisions)} collision, "
              f"{len(contrast)} contrast finding(s)")
        for line, msg in overflows:
            print(f"    overflow   {line!r}  {msg}")
        for provenance in collisions:
            print(f"    collision  caption block overlaps  {provenance!r}")
        for msg in contrast:
            print(f"    contrast   {msg}")
    else:
        print(f"[ OK ] {name}: no overflow, collision or contrast findings")
    return n_bad == 0


def main():
    ok = True
    for name in oc.CAPTIONS:
        ok &= check_caption(name)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
