"""Compose all 20 hardware videos into a grid at 10x speed.

Three layouts, ``--layout``:

* ``5x4`` (default) -- matches the CAMERA's view of the grid, not the plan
  view: 5 columns (iy=0..4, world y, the robot's left runs right across
  frame) by 4 rows (ix=0..3, world x, away-from-the-robot runs down frame).
  Measured, not assumed (2026-09-07) by detecting the box in all 20 hardware
  clips and fitting image position against known box position:
      u (image right) = -125*world_x + 181*world_y + 567
      v (image down)  =  +58*world_x -  58*world_y + 800
  confirmed independently by locating the yellow floor markers, which
  resolve into five columns and four rows. So the tile at (column c, row r)
  is grid point ``r * GRID_NY + c`` -- top row is points 0..4, bottom row
  15..19, exactly the ordering ``montage_callouts.render_grid_inset`` draws
  its card in. This replaced ``4x5`` as the default once the montage
  stopped sitting next to a plan-view diagram (the notes panel that used to
  explain the physical layout is gone): a viewer now only ever compares the
  montage against the hardware footage beside/around it, so agreeing with
  the camera is the property worth having, not agreeing with the table.
* ``4x5`` -- the previous default, matching the physical grid instead: 4
  columns (ix=0..3) by 5 rows (iy=4..0, top to bottom), so a cell's position
  on screen was where the box actually sat on the table as seen from above.
  Kept selectable for anything that still wants the plan-view correspondence.
* ``2x10`` -- 10 columns x 2 rows, landscape. Fills the 1920 px width
  of the canvas the assembly pads to, so the montage is a band across the frame
  instead of a portrait block with bars down both sides. Cells are ordered by
  point index (0..9 on top, 10..19 below), because two rows cannot carry
  either 4x5 or 5x4 physical/camera layout; each cell is labelled with its
  point number, which is what identifies it in the rest of the video. Tried
  and rejected as the default: filling the width shrinks each tile from
  228x216 to 190x180, which is too cramped to read the arms.

Either way the cells carry the hardware crop (see hardware_framing.py) and the
tile size is whatever fits 1920x1080.

Usage:
    .venv/bin/python scripts/video/compose_montage.py
    .venv/bin/python scripts/video/compose_montage.py [--layout 2x10]
"""

import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hardware_framing import HW_CROP, HW_CROP_W, HW_CROP_H  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
VIDEO_DIR = os.path.join(REPO, "video")
HW_DIR = os.path.join(REPO, "videos")

GRID_NX, GRID_NY = 4, 5          # the physical grid: 4 across, 5 deep
CANVAS_W, CANVAS_H = 1920, 1080  # what the assembly pads every segment to
SPEED = 10.0


def cell_size(cols, rows):
    """The largest cell of the crop's aspect that tiles inside the canvas.

    Cells carry the hardware crop, so their aspect is the crop's (~1.06), not
    the canvas's. This montage used to scale the *whole* 1920x1080 frame into a
    480x216 cell: every tile was mostly wall, desks and floor, with the robot
    about 60 px tall.

    Whichever of width and height binds is what sets the tile: at 4x5 or 5x4
    it is the height (four or five rows fill 1080 exactly, leaving bars at
    the sides), at 2x10 the width (ten columns of 192 px fill 1920, leaving
    bars above and below). Both are even, because libx264 with yuv420p needs
    even dimensions and an odd cell multiplies into an odd montage.
    """
    h = min(CANVAS_H // rows, int((CANVAS_W // cols) * HW_CROP_H / HW_CROP_W))
    w = int(round(h * HW_CROP_W / HW_CROP_H))
    return w - (w % 2), h - (h % 2)


def cell_grid(layout):
    """(cols, rows, seed_at[row][col]) for a layout name."""
    if layout == "5x4":
        # Column = world y index (iy, 0..4), row = world x index (ix, 0..3)
        # -- matches the camera (see the module docstring's u/v fit), and is
        # the same top-left-is-0/bottom-right-is-19 ordering
        # montage_callouts.render_grid_inset draws its card in.
        return GRID_NY, GRID_NX, [
            [r * GRID_NY + c for c in range(GRID_NY)]
            for r in range(GRID_NX)]
    if layout == "4x5":
        # Visual row 0 is iy=4 (the far row), so the screen matches the table.
        return GRID_NX, GRID_NY, [
            [ix * GRID_NY + (GRID_NY - 1 - r) for ix in range(GRID_NX)]
            for r in range(GRID_NY)]
    if layout == "2x10":
        return 10, 2, [[r * 10 + c for c in range(10)] for r in range(2)]
    raise ValueError(f"unknown layout {layout!r}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trim-json", default=os.path.join(HW_DIR, "trim_points.json"))
    ap.add_argument("--target-duration", type=float, default=6.0)
    ap.add_argument("--layout", choices=["5x4", "4x5", "2x10"], default="5x4")
    ap.add_argument("--output", default=os.path.join(VIDEO_DIR, "montage_20.mp4"))
    args = ap.parse_args()

    cols, rows, seed_at = cell_grid(args.layout)
    cell_w, cell_h = cell_size(cols, rows)
    print(f"Layout {args.layout}: {cols}x{rows} cells of {cell_w}x{cell_h} "
          f"-> {cols * cell_w}x{rows * cell_h}")

    hw_files = sorted(f for f in os.listdir(HW_DIR) if f.endswith(".mp4"))
    assert len(hw_files) == 20, f"Expected 20 videos, got {len(hw_files)}"

    if os.path.exists(args.trim_json):
        with open(args.trim_json) as f:
            trim = json.load(f)
    else:
        trim = {}

    # Build inputs
    inputs = []
    for seed in range(20):
        fname = hw_files[seed]
        t = trim.get(fname, {})
        start = t.get("start_s", 0)
        end = t.get("end_s", None)
        path = os.path.join(HW_DIR, fname)
        if end:
            inputs.extend(["-ss", str(start), "-to", str(end)])
        inputs.extend(["-i", path])

    filters = []
    # Point labels scale with the tile: 18 px was chosen against a 228 px cell,
    # and the 2x10 tiles are smaller.
    label_size = max(12, int(round(cell_w * 18 / 228)))

    for seed in range(20):
        # tpad freezes each cell on its own last frame once its clip ends.
        # Without it hstack stops as soon as the shortest cell hits EOF, so the
        # whole montage was truncated to the shortest run. Freezing also keeps a
        # cell from ever being asked for footage past its trim -- which is how a
        # person walking in at ~61s reached point 15 in the old build.
        filters.append(
            f"[{seed}:v]crop={HW_CROP},"
            f"scale={cell_w}:{cell_h}:force_original_aspect_ratio=decrease,"
            f"pad={cell_w}:{cell_h}:(ow-iw)/2:(oh-ih)/2:color=0x1a1a2e,"
            f"setpts=PTS/{SPEED:.1f},"
            f"tpad=stop_mode=clone:stop_duration=30,"
            f"drawtext=text='Point {seed}':"
            f"fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:"
            f"fontsize={label_size}:fontcolor=white:borderw=2:bordercolor=black:"
            f"x=6:y=4"
            f"[c{seed}]"
        )

    for r in range(rows):
        row_inputs = "".join(f"[c{seed}]" for seed in seed_at[r])
        filters.append(f"{row_inputs}hstack=inputs={cols}[row{r}]")

    # Stack all rows, then annotate the whole montage with its playback speed.
    row_refs = "".join(f"[row{r}]" for r in range(rows))
    if rows > 1:
        filters.append(f"{row_refs}vstack=inputs={rows}[stacked]")
    else:
        filters.append(f"{row_refs}null[stacked]")
    # Inside the montage at 5x4/4x5 (both fill the canvas height exactly, so
    # there is no room below), below it at 2x10 -- a two-row band leaves the
    # canvas empty under it, and the caption is easier to read there than on
    # top of the footage.
    caption_pad = 0 if args.layout in ("5x4", "4x5") else 54
    if caption_pad:
        filters.append(
            f"[stacked]pad={cols * cell_w}:{rows * cell_h + caption_pad}:0:0:"
            f"color=0x1a1a2e[padded]")
    caption_in = "[padded]" if caption_pad else "[stacked]"
    filters.append(
        f"{caption_in}drawtext=text='All 20 points — {SPEED:.0f}×':"
        f"fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:"
        f"fontsize=34:fontcolor=white:borderw=2:bordercolor=black:"
        f"x=(w-tw)/2:y=h-46[out]"
    )

    filter_str = ";\n".join(filters)

    cmd = [
        "ffmpeg", "-y",
        *inputs,
        "-filter_complex", filter_str,
        "-map", "[out]",
        "-c:v", "libx264", "-crf", "22", "-preset", "medium",
        "-pix_fmt", "yuv420p",
        "-t", str(args.target_duration),
        "-an",
        args.output,
    ]

    print(f"Composing 20-point montage at {SPEED:.0f}x speed...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ffmpeg stderr:\n{result.stderr[-1500:]}")
        sys.exit(1)

    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
