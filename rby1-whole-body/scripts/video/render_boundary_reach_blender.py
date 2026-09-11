"""Boundary-reachability segment: Blender render (left) + constraint plot (right).

Replaces ``render_boundary_reach.py``'s VTK offscreen render.  Split screen, same
as before: the arm on the left at 1280x1080, the regularized boundary constraint
``-log det(J J^T + eps I)`` on the right at 640x1080 with its threshold line,
feasible/infeasible labels and the "kinematic singularity" callout.

The plot is its own panel and the two labels sit in the corner, so most of this
segment is screen-anchored; the one exception is the residual dimension, which
is projected with the camera Blender reported for *this* panel (``camera.json``
in this segment's frame directory, 1280 px wide), never with the wider
domain-extension camera.  The constraint values come from ``v2_domain_ext_annotations.npz`` so they
are computed once, by the same script that recorded the motion, instead of being
re-derived from a second copy of the kinematics.

The 3D content is the *same* meshcat recording the domain-extension segment uses
(that was already true of the VTK versions), rendered again at this segment's
resolution.

Usage:
    .venv/bin/python scripts/video/render_boundary_reach_blender.py
    .venv/bin/python scripts/video/render_boundary_reach_blender.py --force
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from blender_segment_driver import encode, probe_duration, render_segment  # noqa: E402
from frame_parallel import parallel_frames  # noqa: E402
# The residual dimension is drawn exactly as the domain-extension segment draws
# it -- same helper, same style, same threshold -- because it is the same
# measurement of the same motion, and two hand-matched copies would drift.
from render_domain_extension_blender import (  # noqa: E402
    RESIDUAL_VISIBLE_M, draw_dimension)
from segment_camera import SegmentCamera  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
HTML = os.path.join(REPO, "video", "v2_domain_extension.html")
ANNOT = os.path.join(REPO, "video", "v2_domain_ext_annotations.npz")
FRAME_DIR = os.path.join(REPO, "video", "scratch_blender_boundary_seg")
OUT = os.path.join(REPO, "video", "v2_boundary_reach.mp4")

DRAKE_W, DRAKE_H = 1280, 1080
PLOT_W, PLOT_H = 640, 1080
OUT_W, OUT_H = 1920, 1080
FPS = 30
BG_COLOR = "#1a1a2e"
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

TAU = 9.0

# Same camera as the domain-extension segment (and as the VTK render it
# replaces); only the horizontal extent differs because this panel is narrower.
CAM_EYE = (1.2, -1.4, 1.0)
CAM_TARGET = (0.4, 0.0, 0.55)
CAM_VFOV = 0.8


def draw_plot(ax, times, constraint_values, i, duration):
    """The right-hand panel for frame ``i`` -- constraint so far, plus threshold."""
    ax.clear()
    ax.set_facecolor(BG_COLOR)
    ax.set_title("Boundary Reachability Constraint",
                 color="#e0e0e0", fontsize=12, fontweight="bold", pad=8)
    ax.set_ylabel(r"$-\log\det(JJ^T + \epsilon I)$", color="#e0e0e0", fontsize=10)
    ax.set_xlabel("time (s)", color="#e0e0e0", fontsize=10)
    ax.tick_params(colors="#e0e0e0", labelsize=8)
    for sp in ("bottom", "left"):
        ax.spines[sp].set_color("#e0e0e0")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(True, alpha=0.15, color="#e0e0e0")

    t_arr = times[:i + 1]
    c_arr = constraint_values[:i + 1]
    if len(t_arr) > 1:
        for j in range(len(t_arr) - 1):
            c = "#66bb6a" if c_arr[j] < TAU else "#ff7043"
            ax.plot(t_arr[j:j + 2], c_arr[j:j + 2], color=c, lw=2)

    ax.axhline(TAU, color="#ffd54f", ls="--", lw=1.5, alpha=0.8)
    ax.text(duration * 0.02, TAU + 1.0, f"τ = {TAU:.0f}",
            color="#ffd54f", fontsize=10, va="bottom")
    ax.text(duration * 0.85, TAU - 1.5, "feasible", color="#66bb6a",
            fontsize=11, fontweight="bold", ha="center", va="top")
    ax.text(duration * 0.85, TAU + 1.5, "infeasible", color="#ff7043",
            fontsize=11, fontweight="bold", ha="center", va="bottom")

    ax.set_xlim(0, duration)
    ax.set_ylim(bottom=constraint_values.min() * 0.9,
                top=max(constraint_values.max() * 1.15, TAU * 1.5))
    ax.axvline(times[i], color="#ff7043", alpha=0.4, lw=1, ls="--")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true",
                    help="re-run Blender even if frames are cached")
    args = ap.parse_args()

    ann = np.load(ANNOT)
    constraint = np.asarray(ann["constraint"], dtype=float)
    residual = np.asarray(ann["residual"], dtype=float)
    target_p = np.asarray(ann["target"], dtype=float)
    actual_p = np.asarray(ann["p_actual"], dtype=float)
    n_ann = len(constraint)
    times = np.arange(n_ann) / FPS
    duration = n_ann / FPS

    cam_request = SegmentCamera.from_look_at(CAM_EYE, CAM_TARGET, DRAKE_W, DRAKE_H,
                                            vfov_rad=CAM_VFOV)
    frames, cam = render_segment(HTML, FRAME_DIR, cam_request, fps=FPS,
                                 force=args.force)

    if len(frames) != n_ann:
        raise RuntimeError(
            f"{len(frames)} Blender frames but {n_ann} annotation rows -- "
            "regenerate the HTML with generate_domain_ext_meshcat.py")

    crossing = np.argmax(constraint > TAU) if np.any(constraint > TAU) else None
    print(f"  constraint {constraint.min():.2f}..{constraint.max():.2f}, "
          f"tau={TAU}, first crossing at frame {crossing}")

    try:
        font = ImageFont.truetype(FONT_PATH, 24)
        font_small = ImageFont.truetype(FONT_PATH, 18)
        font_sing = ImageFont.truetype(FONT_PATH, 22)
        font_dim = ImageFont.truetype(FONT_PATH, 20)
    except OSError:
        font = font_small = font_sing = font_dim = ImageFont.load_default()

    # Every frame is an independent function of its index, so the overlay pass
    # fans out across processes; see frame_parallel. The Figure is built once
    # per worker rather than once per frame -- rebuilding it per frame costs
    # more than the parallelism saves, and sharing one across processes is not
    # something matplotlib supports.
    def make_renderer():
        fig, ax = plt.subplots(figsize=(PLOT_W / 100, PLOT_H / 100), dpi=100)
        fig.patch.set_facecolor(BG_COLOR)

        def one(i):
            left = Image.open(frames[i]).convert("RGB")
            if left.size != (DRAKE_W, DRAKE_H):
                left = left.resize((DRAKE_W, DRAKE_H), Image.LANCZOS)

            draw_plot(ax, times, constraint, i, duration)
            fig.subplots_adjust(left=0.18, right=0.92, top=0.88, bottom=0.12)
            fig.canvas.draw()
            plot_arr = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
            plot_img = Image.fromarray(plot_arr).resize((PLOT_W, PLOT_H),
                                                        Image.LANCZOS)

            composite = np.zeros((OUT_H, OUT_W, 3), dtype=np.uint8)
            composite[:, :DRAKE_W] = np.asarray(left, dtype=np.uint8)
            composite[:, DRAKE_W:] = np.asarray(plot_img, dtype=np.uint8)

            img = Image.fromarray(composite)
            d = ImageDraw.Draw(img)
            d.text((20, 20), "Boundary Reachability",
                   fill=(0x4f, 0xc3, 0xf7), font=font)
            if constraint[i] < TAU:
                d.text((20, 55), "Interior of workspace",
                       fill=(0x66, 0xbb, 0x6a), font=font_small)
            else:
                d.text((20, 55), "Kinematic singularity",
                       fill=(0xff, 0x70, 0x43), font=font_sing)

            # The residual, on the left panel, in the same CAD dimension the
            # domain-extension segment uses. The plot answers "is the arm at
            # the boundary"; without this the viewer cannot see *how far* the
            # target has run away from the gripper while it sits there, which
            # is the thing the two segments share. The projection is this
            # panel's camera (1280 wide), not the other segment's.
            if residual[i] >= RESIDUAL_VISIBLE_M:
                p_a = cam.project_int(actual_p[i])
                p_t = cam.project_int(target_p[i])
                if p_a and p_t:
                    draw_dimension(d, p_a, p_t,
                                   f"{residual[i]*1000:.1f} mm",
                                   font_dim, (0xff, 0x70, 0x43))

            return np.asarray(img, dtype=np.uint8)

        return one

    n = encode(parallel_frames(make_renderer, len(frames)),
               OUT, OUT_W, OUT_H, fps=FPS)
    print(f"Wrote {OUT} ({n} frames, {probe_duration(OUT):.2f}s)")


if __name__ == "__main__":
    main()
