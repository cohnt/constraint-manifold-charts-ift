"""Static-stability segment: Blender render + support polygon / CoM overlays.

Replaces ``render_static_stability.py``'s VTK offscreen render.  The RBY1 replays
grid point 00 while the nominal and conservative support polygons are drawn on the
floor and the CoM's ground projection is marked, coloured by which polygon it is
inside.

Those three annotations are anchored in the scene, so they are projected with the
camera Blender reported after building the scene (``camera.json``) rather than the
VTK camera the original used -- that is the whole point of the conversion.  The
polygon vertices arrive in the *base* frame from
``generate_stability_meshcat.py``'s npz and are rotated/translated by the
per-frame base pose here, exactly as the VTK version did.

Framing differs deliberately from the VTK render in one way: the vertical field of
view is wider, because the old camera cropped the robot's arms off the top of the
frame.  Eye and look-at point are otherwise the same view direction.

Usage:
    .venv/bin/python scripts/video/render_static_stability_blender.py
    .venv/bin/python scripts/video/render_static_stability_blender.py --force
    .venv/bin/python scripts/video/render_static_stability_blender.py --probe 60
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
HTML = os.path.join(REPO, "video", "v2_rby1_stability.html")
ANNOT = os.path.join(REPO, "video", "v2_stability_annotations.npz")
FRAME_DIR = os.path.join(REPO, "video", "scratch_blender_stability_seg")
OUT = os.path.join(REPO, "video", "v2_rby1_stability.mp4")

WIDTH, HEIGHT = 1920, 1080
FPS = 30
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

ACCENT_GREEN = (0x66, 0xbb, 0x6a)
ACCENT_YELLOW = (0xff, 0xd5, 0x4f)
ACCENT_CORAL = (0xff, 0x70, 0x43)
ACCENT_BLUE = (0x4f, 0xc3, 0xf7)
NOMINAL_EDGE = (255, 255, 100)
INSET_EDGE = (255, 165, 0)

# Same eye as the VTK render, which cropped the torso and both arms off the top
# of the frame; this looks from the same direction with a wider vertical field
# of view and a higher look-at point.
CAM_EYE = (1.5, 1.5, 1.0)
CAM_TARGET = (0.0, 0.0, 0.98)
# Fitted by rendering, not by projecting body origins -- an origin is a point,
# and the arms' *geometry* reaches well past the origins of the links carrying
# it, so a fit to origins does not bound what appears on screen. 1.35 rad
# (77 deg) put the subject inside a 3.4 m tall frame at this 2.13 m standoff and
# left the robot about a quarter of the frame wide; 0.85 and 1.05 both clipped
# the raised arms. 1.18 rad holds the fully raised arms and the box on the floor
# with headroom, at roughly twice the previous subject size.
CAM_VFOV = 1.18

# The polygons are drawn just above the floor plane, as in the VTK version.
FLOOR_Z = 0.01

# Where the "CoM" label sits, in pixels, for the whole segment. Fixed, not
# offset from the marker: a label that rides a moving marker is itself a moving
# thing to read, and the eye follows the text instead of the dot. Only the
# leader line moves now.
#
# This anchor is on the empty background to the right of the robot. At this
# camera the robot's silhouette ends at about x = 1100 and the table at about
# x = 900, while the CoM marker stays inside x = 910..966, y = 896..929 across
# the whole replay -- so the label is clear of both the robot and the polygon,
# and the leader never crosses the machine.
COM_LABEL_ANCHOR = (1258, 812)

# The CoM's ground track, drawn behind the marker so the path it takes through
# the support polygon is visible rather than having to be remembered frame by
# frame. Deliberately one colour: the marker already carries the
# inside/outside verdict, and a trace that changed colour would read as a second
# signal.
#
# Bright magenta, because the trace has to be read against a mid-grey floor and
# through two polygon outlines. The muted lavender it started as was the same
# luminance as the floor and simply disappeared; magenta is the one strong hue
# not already spoken for -- green, yellow and coral are the CoM verdict, yellow
# and orange the two polygons, blue the caption. Drawn 4 px wide with a dark
# outline pass under it so it stays legible where it crosses a polygon edge.
TRACE_COLOR = (0xff, 0x3d, 0xd7)
TRACE_SHADOW = (0x2a, 0x00, 0x22)
TRACE_WIDTH = 4


def polygon_pixels(cam, verts_xy, base_xyt):
    """Project a base-frame polygon onto the floor.

    Returns None unless *every* vertex projects -- dropping the ones that fall
    outside the frame (which the VTK helper did) silently deforms the polygon.
    """
    c, s = np.cos(base_xyt[2]), np.sin(base_xyt[2])
    R2 = np.array([[c, -s], [s, c]])
    world_xy = (R2 @ np.asarray(verts_xy).T).T + base_xyt[:2]
    pts = []
    for xy in world_xy:
        p = cam.project_int(np.array([xy[0], xy[1], FLOOR_Z]), clip_to_frame=False)
        if p is None:
            return None
        pts.append(p)
    return pts


def load_fonts():
    try:
        return (ImageFont.truetype(FONT_PATH, 26),
                ImageFont.truetype(FONT_PATH, 18))
    except OSError:
        d = ImageFont.load_default()
        return d, d


def com_pixels(cam, ann):
    """Project every frame's CoM onto the floor plane, once, up front.

    The trace needs the whole path, so projecting per frame inside the
    compositing loop would redo the same work O(n^2) times.
    """
    return [cam.project_int(np.array([c[0], c[1], FLOOR_Z]), clip_to_frame=False)
            for c in ann["com"]]


def annotate(img, cam, ann, i, fonts, com_px):
    font, font_small = fonts
    draw = ImageDraw.Draw(img)

    base_xyt = ann["base_xyt"][i]
    com = ann["com"][i]
    is_conservative = bool(ann["is_conservative"][i])
    is_nominal = bool(ann["is_nominal"][i])
    rear_margin = float(ann["rear_margin"][i])
    inset_mm = float(ann["inset"]) * 1000.0

    nominal_pts = polygon_pixels(cam, ann["nominal_verts"], base_xyt)
    if nominal_pts and len(nominal_pts) >= 3:
        for j in range(len(nominal_pts)):
            draw.line([nominal_pts[j], nominal_pts[(j + 1) % len(nominal_pts)]],
                      fill=NOMINAL_EDGE, width=2)

    inset_pts = polygon_pixels(cam, ann["inset_verts"], base_xyt)
    if inset_pts and len(inset_pts) >= 3:
        for j in range(len(inset_pts)):
            draw.line([inset_pts[j], inset_pts[(j + 1) % len(inset_pts)]],
                      fill=INSET_EDGE, width=2)

    # The track so far, under the marker.
    track = [p for p in com_px[:i + 1] if p is not None]
    if len(track) >= 2:
        draw.line(track, fill=TRACE_SHADOW, width=TRACE_WIDTH + 4,
                  joint="curve")
        draw.line(track, fill=TRACE_COLOR, width=TRACE_WIDTH, joint="curve")

    com_2d = cam.project_int(np.array([com[0], com[1], FLOOR_Z]))
    if com_2d:
        r = 8
        color = (ACCENT_GREEN if is_conservative
                 else ACCENT_YELLOW if is_nominal else ACCENT_CORAL)
        draw.ellipse([com_2d[0]-r, com_2d[1]-r, com_2d[0]+r, com_2d[1]+r],
                     fill=color, outline=(255, 255, 255))

        # Leader line from the marker to the fixed label, so the marker is
        # identified without the viewer having to infer it from the caption.
        lx, ly = COM_LABEL_ANCHOR
        # Start the line at the marker's edge, not its centre, so the dot reads
        # as a filled disc rather than a lollipop.
        dx, dy = lx - com_2d[0], ly - com_2d[1]
        d = max(np.hypot(dx, dy), 1e-6)
        sx = com_2d[0] + dx * (r + 2) / d
        sy = com_2d[1] + dy * (r + 2) / d
        draw.line([(sx, sy), (lx, ly)], fill=color, width=2)

        text = "CoM"
        tb = draw.textbbox((0, 0), text, font=font_small)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        # A short horizontal foot under the text, the way a callout leader ends.
        draw.line([(lx, ly), (lx + tw + 8, ly)], fill=color, width=2)
        draw.text((lx + 4, ly - th - 6), text, fill=color, font=font_small)

    draw.text((30, 30), "Static Stability Constraint", fill=ACCENT_BLUE, font=font)
    if is_conservative:
        draw.text((30, 65), "CoM inside conservative polygon",
                  fill=ACCENT_GREEN, font=font_small)
    elif is_nominal:
        draw.text((30, 65), "CoM inside nominal polygon",
                  fill=ACCENT_YELLOW, font=font_small)
    else:
        draw.text((30, 65), "CoM outside support polygon",
                  fill=ACCENT_CORAL, font=font_small)
    draw.text((30, 90), f"Rear margin: {rear_margin*1000:.1f} mm",
              fill=(200, 200, 200), font=font_small)
    draw.text((30, HEIGHT - 50), f"Inset: {inset_mm:.0f} mm (rear edge only)",
              fill=(160, 160, 160), font=font_small)
    # The playback rate comes from the npz, written by the generator that chose
    # the sampling, so the label cannot drift from what the frames actually show.
    speed = float(ann["speed"]) if "speed" in ann else None
    if speed:
        draw.text((30, 115), f"{speed:.0f}× real time",
                  fill=(200, 200, 200), font=font_small)
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true",
                    help="re-run Blender even if frames are cached")
    ap.add_argument("--probe", type=int, default=None,
                    help="write one annotated frame to scratch and exit")
    ap.add_argument("--frame-end", type=int, default=None,
                    help="render only up to this frame (framing checks)")
    ap.add_argument("--frame-dir", default=FRAME_DIR)
    args = ap.parse_args()

    ann = dict(np.load(ANNOT))
    cam_request = SegmentCamera.from_look_at(CAM_EYE, CAM_TARGET, WIDTH, HEIGHT,
                                            vfov_rad=CAM_VFOV)
    frames, cam = render_segment(HTML, args.frame_dir, cam_request, fps=FPS,
                                 frame_end=args.frame_end, force=args.force)

    n_ann = len(ann["com"])
    fonts = load_fonts()
    com_px = com_pixels(cam, ann)

    if args.probe is not None or args.frame_end is not None:
        # Clamp both lookups to the same index so a probe never annotates one
        # frame with another frame's data.
        i = min(args.probe if args.probe is not None else 0,
                len(frames) - 1, n_ann - 1)
        img = Image.open(frames[i]).convert("RGB")
        annotate(img, cam, ann, i, fonts, com_px)
        out = os.path.join(args.frame_dir, f"probe_{i:04d}.png")
        img.save(out)
        print(f"Wrote {out}")
        return

    if len(frames) != n_ann:
        raise RuntimeError(
            f"{len(frames)} Blender frames but {n_ann} annotation rows -- "
            "regenerate the HTML with generate_stability_meshcat.py")

    # Every frame is an independent function of its index, so the overlay pass
    # fans out across processes; see frame_parallel.
    def make_renderer():
        def one(i):
            img = Image.open(frames[i]).convert("RGB")
            if img.size != (WIDTH, HEIGHT):
                img = img.resize((WIDTH, HEIGHT), Image.LANCZOS)
            annotate(img, cam, ann, i, fonts, com_px)
            return np.asarray(img, dtype=np.uint8)
        return one

    n = encode(parallel_frames(make_renderer, len(frames)),
               OUT, WIDTH, HEIGHT, fps=FPS)
    print(f"Wrote {OUT} ({n} frames, {probe_duration(OUT):.2f}s)")


if __name__ == "__main__":
    main()
