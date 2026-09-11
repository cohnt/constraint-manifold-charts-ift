"""Domain-extension segment: Blender render + 3D-anchored annotation overlays.

Replaces ``render_domain_extension.py``'s VTK offscreen render so every 3D shot in
the overview video comes out of the same meshcat -> Blender pipeline.  The 3D
content and the annotations come from two files written by
``generate_domain_ext_meshcat.py``:

* ``video/v2_domain_extension.html`` -> Blender frames (robot only)
* ``video/v2_domain_ext_annotations.npz`` -> per-frame target / actual / residual

The target pose is a second, free-floating Schunk gripper in the scene itself,
visible throughout and coincident with the bolted one until the target leaves
the reachable workspace; only the residual *dimension* is drawn here, and it is
projected with the camera Blender reported
after setting up the scene (``camera.json``), never with the VTK camera.  The
framing (eye, target, vertical field of view) is the same as the VTK render's,
so the segment looks like the one it replaces apart from the lighting and
background.

Usage:
    .venv/bin/python scripts/video/render_domain_extension_blender.py
    .venv/bin/python scripts/video/render_domain_extension_blender.py --force
    .venv/bin/python scripts/video/render_domain_extension_blender.py --probe 270
"""

import argparse
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from blender_segment_driver import encode, probe_duration, render_segment  # noqa: E402
from frame_parallel import parallel_frames  # noqa: E402
from segment_camera import SegmentCamera  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
HTML = os.path.join(REPO, "video", "v2_domain_extension.html")
ANNOT = os.path.join(REPO, "video", "v2_domain_ext_annotations.npz")
FRAME_DIR = os.path.join(REPO, "video", "scratch_blender_domext_seg")
OUT = os.path.join(REPO, "video", "v2_domain_extension.mp4")

WIDTH, HEIGHT = 1920, 1080
FPS = 30
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

# Identical to the VTK render's camera: same eye, same look-at point, same
# vertical field of view (Drake's CameraInfo(W, H, 0.8)).
CAM_EYE = (1.2, -1.4, 1.0)
CAM_TARGET = (0.4, 0.0, 0.55)
CAM_VFOV = 0.8

RESIDUAL_VISIBLE_M = 0.01     # below this the tracking is exact, so no line

# The dimension line's height in the frame, in pixels, for the whole segment.
#
# Nothing the segment renders reaches this far down under the measured span:
# across every annotated frame the lowest non-background pixel between the two
# measured points is y = 618 (the target gripper at the end of the sweep), and
# the arm and its base -- which do reach y = 881 -- are entirely to the left of
# the span. An offset measured *from the grippers* instead put the dimension
# line across the forearm, because the two points sit high in the frame and the
# arm hangs below them.
DIM_BASELINE_Y = 760


def load_fonts():
    try:
        return (ImageFont.truetype(FONT_PATH, 28),
                ImageFont.truetype(FONT_PATH, 20),
                ImageFont.truetype(FONT_PATH, 22))
    except OSError:
        d = ImageFont.load_default()
        return d, d, d


def draw_dimension(draw, p1, p2, text, font, color,
                   baseline_y=DIM_BASELINE_Y, arrow=13, ext_gap=6, ext_over=9):
    """Draw a CAD-style linear dimension between two pixel points.

    Extension lines drop from the two measured points to a horizontal dimension
    line well below the robot, arrowheads point outward into them, and the
    value sits below the middle. Drawn this way rather than as a bare line with
    dots because it reads unambiguously as *a measurement of the gap*, which is
    the quantity the segment is about.

    The dimension line is horizontal and at a fixed height, the way a drawing
    dimensions a horizontal distance, while the number is the full 3-D
    residual. Those agree here because the target walks along world +x: the
    out-of-plane part of the gap is a rounding error next to the part this line
    spans.
    """
    x1, x2 = float(p1[0]), float(p2[0])
    if abs(x2 - x1) < 2:
        return
    sgn = 1.0 if x2 > x1 else -1.0

    a = (x1, float(baseline_y))
    b = (x2, float(baseline_y))

    # Extension lines start just below the measured point, as in a drawing, and
    # overshoot the dimension line slightly.
    for p, q in ((p1, a), (p2, b)):
        draw.line([(p[0], p[1] + ext_gap), (q[0], baseline_y + ext_over)],
                  fill=color, width=2)

    draw.line([a, b], fill=color, width=2)

    half = arrow * 0.34
    for tip, s in ((a, sgn), (b, -sgn)):
        bx = tip[0] + s * arrow
        draw.polygon([tip, (bx, tip[1] + half), (bx, tip[1] - half)], fill=color)

    mx = (a[0] + b[0]) / 2.0
    tb = draw.textbbox((0, 0), text, font=font)
    tw, th = tb[2] - tb[0], tb[3] - tb[1]
    # Below the dimension line, clear of it by more than half the text height:
    # at +12 the glyph tails sat on the line.
    draw.text((mx - tw / 2.0, baseline_y + th / 2.0 + 12 - tb[1]),
              text, fill=color, font=font)


def annotate(img, cam, ann, i, fonts):
    """Draw one frame's overlays."""
    font, font_small, font_eq = fonts
    draw = ImageDraw.Draw(img)

    target = ann["target"][i]
    p_actual = ann["p_actual"][i]
    residual = float(ann["residual"][i])

    # The target is a real, free-floating gripper in the scene now (see
    # generate_domain_ext_meshcat.py), not a 2D rectangle drawn here, and it is
    # on screen for the whole segment: while the target is reachable it sits on
    # top of the bolted gripper, and it separates from it only past the
    # boundary. Only the dimension and the caption are gated on the threshold.
    draw.text((30, 30), "IK Domain Extension", fill=(0x4f, 0xc3, 0xf7), font=font)

    if residual < RESIDUAL_VISIBLE_M:
        draw.text((30, 70), "Tracking within reachable workspace",
                  fill=(0x66, 0xbb, 0x6a), font=font_small)
    else:
        draw.text((30, 70), "Target beyond workspace boundary",
                  fill=(0xff, 0x70, 0x43), font=font_small)
        # "approximately equals": q* is the least-squares projection of an
        # unreachable target, so it does not satisfy FK(q) = X_target. Writing
        # "=" claims the constraint is met exactly, which is the opposite of
        # what this segment exists to show.
        draw.text((30, HEIGHT - 60), "q* ≈ argmin ||FK(q) - X_target||²",
                  fill=(0xff, 0xd5, 0x4f), font=font_eq)

        p_a = cam.project_int(p_actual)
        p_t = cam.project_int(target)
        if p_a and p_t:
            draw_dimension(draw, p_a, p_t, f"{residual*1000:.1f} mm",
                           font_small, (0xff, 0x70, 0x43))
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true",
                    help="re-run Blender even if frames are cached")
    ap.add_argument("--probe", type=int, default=None,
                    help="write one annotated frame to scratch and exit "
                         "(for eyeballing the annotation anchoring)")
    args = ap.parse_args()

    ann = dict(np.load(ANNOT))
    cam_request = SegmentCamera.from_look_at(CAM_EYE, CAM_TARGET, WIDTH, HEIGHT,
                                            vfov_rad=CAM_VFOV)
    frames, cam = render_segment(HTML, FRAME_DIR, cam_request, fps=FPS,
                                 force=args.force)

    n_ann = len(ann["residual"])
    if len(frames) != n_ann:
        raise RuntimeError(
            f"{len(frames)} Blender frames but {n_ann} annotation rows -- the "
            "recording and the annotation data are out of step; regenerate the "
            "HTML with generate_domain_ext_meshcat.py")

    fonts = load_fonts()

    if args.probe is not None:
        i = args.probe
        img = Image.open(frames[i]).convert("RGB")
        annotate(img, cam, ann, i, fonts)
        out = os.path.join(FRAME_DIR, f"probe_{i:04d}.png")
        img.save(out)
        print(f"Wrote {out}")
        return

    # Every frame is an independent function of its index, so the overlay pass
    # fans out across processes; see frame_parallel.
    def make_renderer():
        def one(i):
            img = Image.open(frames[i]).convert("RGB")
            if img.size != (WIDTH, HEIGHT):
                img = img.resize((WIDTH, HEIGHT), Image.LANCZOS)
            annotate(img, cam, ann, i, fonts)
            return np.asarray(img, dtype=np.uint8)
        return one

    n = encode(parallel_frames(make_renderer, len(frames)),
               OUT, WIDTH, HEIGHT, fps=FPS)
    print(f"Wrote {OUT} ({n} frames, {probe_duration(OUT):.2f}s)")


if __name__ == "__main__":
    main()
