"""Sanity check: does any artist in a rendered ral_explainers panel overflow
the panel's own bounds, or overlap another artist?

This renders each panel through ``ral_explainers.build_panel_figures()`` --
the exact function the shipped PNGs are built from -- rather than
reimplementing the two-pass centering/top-alignment logic here. If
ral_explainers.py's anchoring changes, this checker changes with it; it
cannot silently validate a layout that is not the one being rendered.

Two independent classes of finding are reported per panel:

  overflow  -- any Text bbox, or any axes tight bbox, extends outside the
               panel's own [0, W] x [0, H] pixel bounds. This is what caught
               a stray out-of-range tick label pushed off the right edge of
               the grid panel, and a rotated y-axis label overflowing the
               left margin, earlier in this work.
  collision -- any two of those boxes intersect with positive area.

Per-axes text (tick labels, axis labels) is treated as one combined tight
bbox per axes rather than one box per tick label, so a paragraph landing on
top of a plot is caught without flagging every tick label against every
other one.

Run after `venv/bin/python scripts/video/ral_explainers.py --out-dir
scratch/ral_explainers`:

    venv/bin/python scripts/video/check_explainer_layout.py
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ral_explainers as re_mod

# How much of an offending artist's text to show in a finding line.
_LABEL_MAX_CHARS = 60


def _short(text):
    text = " ".join(str(text).split())
    if len(text) > _LABEL_MAX_CHARS:
        text = text[:_LABEL_MAX_CHARS - 3] + "..."
    return text


def _boxes_overlap(a, b):
    """True if two (x0, y0, x1, y1) pixel boxes overlap with positive area."""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ox = min(ax1, bx1) - max(ax0, bx0)
    oy = min(ay1, by1) - max(ay0, by0)
    return ox > 0 and oy > 0


def _collect_boxes(fig, renderer):
    """(label, (x0, y0, x1, y1)) for every Text artist and every axes'
    combined tight bbox, in figure pixel coordinates."""
    boxes = []
    for t in fig.texts:
        if not t.get_text().strip():
            continue
        bb = t.get_window_extent(renderer=renderer)
        boxes.append((t.get_text(), (bb.x0, bb.y0, bb.x1, bb.y1)))
    for ax in fig.axes:
        bb = ax.get_tightbbox(renderer)
        boxes.append(("<axes>", (bb.x0, bb.y0, bb.x1, bb.y1)))
    return boxes


def check_panel(name, fig):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    W, H = fig.canvas.get_width_height()

    boxes = _collect_boxes(fig, renderer)

    overflows = []
    for label, (x0, y0, x1, y1) in boxes:
        if x0 < 0 or y0 < 0 or x1 > W or y1 > H:
            overflows.append((label, (x0, y0, x1, y1)))

    collisions = []
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            label_i, box_i = boxes[i]
            label_j, box_j = boxes[j]
            if _boxes_overlap(box_i, box_j):
                collisions.append((label_i, label_j))

    plt.close(fig)

    n_bad = len(overflows) + len(collisions)
    if n_bad:
        print(f"[FAIL] {name}: {len(overflows)} overflow, "
              f"{len(collisions)} collision finding(s)")
        for label, (x0, y0, x1, y1) in overflows:
            print(f"    overflow   {_short(label)!r}  "
                  f"box=({x0:.1f}, {y0:.1f}, {x1:.1f}, {y1:.1f})  "
                  f"outside [0, {W}] x [0, {H}]")
        for a, b in collisions:
            print(f"    collision  {_short(a)!r}  overlaps  {_short(b)!r}")
    else:
        print(f"[ OK ] {name}: no overflow or collisions")
    return n_bad == 0


def main():
    ok = True
    for name, fig in re_mod.build_panel_figures():
        ok &= check_panel(name, fig)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
