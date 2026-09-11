"""Blender-render all 20 grid points from existing meshcat HTML files.

Imports each meshcat HTML into Blender, renders frames, encodes to MP4.

Usage:
    .venv/bin/python scripts/video/blender_render_grid.py
    .venv/bin/python scripts/video/blender_render_grid.py --indices 1 2 3
"""

import argparse
import glob
import os
import subprocess
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from blender_paths import BLENDER  # noqa: E402
from hardware_framing import SIM_RENDER_W, SIM_RENDER_H  # noqa: E402
HTML_DIR = os.path.join(REPO, "scratch", "grid_html_30fps_final")
OUT_DIR = os.path.join(REPO, "video")
SCRATCH_BASE = os.path.join(REPO, "video", "scratch_blender_grid")

RENDER_SCRIPT = '''
import sys, os, math, glob
sys.path.insert(0, os.path.expanduser('~/.config/blender/5.0/extensions/user_default'))
import bpy
from meshcat_html_importer.blender_impl.scene_builder import build_scene_from_file

args = sys.argv[sys.argv.index("--") + 1:]
html_path = args[0]
out_dir = args[1]

build_scene_from_file(html_path, target_fps=30, clear_scene=True)

# Delete collision geometry entirely —
# hiding is unreliable across Blender/EEVEE versions.
# Keep grid_viz_ objects (the pick/place box walls).
to_remove = []
for obj in bpy.data.objects:
    if obj.type == 'MESH' and obj.name.startswith('collision_'):
        to_remove.append(obj)
for obj in to_remove:
    bpy.data.objects.remove(obj, do_unlink=True)
print(f'Removed {len(to_remove)} collision/grid_viz objects')

# Smooth shading + materials tuned to the meshcat-imported per-face colors.
# The importer preserves RGB values as material names (e.g. '69,69,69') on
# the Principled BSDF Base Color input.  Dark parts (robot panels) should
# stay matte; lighter parts get a subtle metallic sheen.
for obj in bpy.data.objects:
    if obj.type != 'MESH':
        continue
    for poly in obj.data.polygons:
        poly.use_smooth = True
    if not obj.data.materials:
        continue
    for mat in obj.data.materials:
        if not mat or not mat.use_nodes:
            continue
        bsdf = mat.node_tree.nodes.get('Principled BSDF')
        if not bsdf:
            continue
        color = bsdf.inputs['Base Color'].default_value
        brightness = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
        if brightness > 0.5:
            bsdf.inputs['Metallic'].default_value = 0.3
            bsdf.inputs['Roughness'].default_value = 0.35
        else:
            bsdf.inputs['Metallic'].default_value = 0.05
            bsdf.inputs['Roughness'].default_value = 0.6

n_obj = len([o for o in bpy.data.objects if o.type == 'MESH'])
f_start = bpy.context.scene.frame_start
f_end = bpy.context.scene.frame_end
print(f'Visual objects: {n_obj}, Frames: {f_start}-{f_end}')

# Framing fitted to the subject rather than guessed: the swept bounding box of
# the robot, the carried box and the pick/place boxes over every frame of point
# 0, projected onto this camera's axes, is 2.21 m wide by 2.11 m tall, centred
# at (0.161, 0.057, 0.944), 3.8 m along the original 3/4 view direction.
#
# The lens is derived from the frame height rather than hardcoded. Blender's
# AUTO sensor fit maps the 36 mm sensor width onto the LONGER axis, so the
# vertical field of view is set by sensor_width * H / W: changing the render
# aspect changes the vertical framing unless the lens moves with it. The
# reference fit was 31 mm at 960x640; scaling by H/640 keeps the subject the
# same height on screen at any frame height. (The pane is now near-square
# rather than 3:2 -- see hardware_framing.py -- which also happens to suit
# content that is itself nearly square, 2.21 x 2.11 m.)
#
# The table is deliberately NOT part of the fit, so it bleeds off the left edge
# instead of dominating the frame as a featureless grey slab.
#
# Safe for all 20 points: the grid spans only 90 x 120 mm
# (data/box_placement_grid.npz), far inside the margin.
cam = bpy.data.cameras.new('C')
cam.lens = 31.0 * SIM_H / 640.0
cam.clip_end = 100
co = bpy.data.objects.new('C', cam)
bpy.context.scene.collection.objects.link(co)
bpy.context.scene.camera = co
co.location = (3.811, -0.199, 1.969)
t = bpy.data.objects.new('T', None)
t.location = (0.161, 0.057, 0.944)
bpy.context.scene.collection.objects.link(t)
tc = co.constraints.new(type='TRACK_TO')
tc.target = t
tc.track_axis = 'TRACK_NEGATIVE_Z'
tc.up_axis = 'UP_Y'

# Studio lighting
key = bpy.data.lights.new('Key', 'SUN'); key.energy = 4; key.color = (1, 0.98, 0.95)
ko = bpy.data.objects.new('Key', key)
ko.rotation_euler = (math.radians(50), math.radians(10), math.radians(-30))
bpy.context.scene.collection.objects.link(ko)
fill = bpy.data.lights.new('Fill', 'AREA'); fill.energy = 150; fill.size = 4
fo = bpy.data.objects.new('Fill', fill); fo.location = (-2, 2, 2.5)
bpy.context.scene.collection.objects.link(fo)
rim = bpy.data.lights.new('Rim', 'SPOT'); rim.energy = 200; rim.spot_size = math.radians(60)
ro = bpy.data.objects.new('Rim', rim); ro.location = (-1.5, -2, 3)
ro.rotation_euler = (math.radians(35), 0, math.radians(-150))
bpy.context.scene.collection.objects.link(ro)

w = bpy.data.worlds.new('W')
bpy.context.scene.world = w
w.use_nodes = True
w.node_tree.nodes['Background'].inputs['Color'].default_value = (0.102, 0.102, 0.18, 1)

s = bpy.context.scene
# Matches the hardware crop's aspect exactly, so the two halves of the
# side-by-side have identical content height and line up (see
# compose_rby1_v2.py and hardware_framing.py -- SIM_RENDER_W/H come from there).
# At 16:9 the sim half letterboxed against the hardware pane and the two
# visibly disagreed.
s.render.resolution_x = SIM_W
s.render.resolution_y = SIM_H
s.render.fps = 30
# Cycles on the GPU when one is present, on the CPU otherwise; prints which.
sys.path.insert(0, SCRIPTS_DIR)
from blender_render_common import configure_render_quality
configure_render_quality(s, samples=128)
s.render.image_settings.file_format = 'PNG'

for f in glob.glob(os.path.join(out_dir, 'frame_*.png')):
    os.remove(f)
s.render.filepath = os.path.join(out_dir, 'frame_')

# PREVIEW_FRAMES > 0 renders only the first few frames, for checking the camera
# without paying for the whole segment. The camera above is fixed, so a preview
# frames exactly the way the full render will.
if PREVIEW_FRAMES > 0:
    f_end = min(f_end, f_start + PREVIEW_FRAMES - 1)
    s.frame_end = f_end

n = f_end - f_start + 1
print(f'Rendering {n} frames at {SIM_W}x{SIM_H}, lens {cam.lens:.1f}mm...')
bpy.ops.render.render(animation=True)
print('DONE')
'''


def render_point(idx, preview_frames=0, force=False):
    html_path = os.path.join(HTML_DIR, f"point_{idx:02d}.html")
    if not os.path.exists(html_path):
        print(f"  [{idx:2d}] no HTML file")
        return None

    # A preview gets its own frames directory and its own output file: the
    # render script clears frame_*.png in whatever directory it is given, so
    # sharing one would destroy a finished frame set, and a 3-frame mp4 must
    # never be mistaken for the segment.
    suffix = "_preview" if preview_frames else ""
    frames_dir = os.path.join(SCRATCH_BASE, f"point_{idx:02d}{suffix}")
    os.makedirs(frames_dir, exist_ok=True)

    mp4_path = os.path.join(OUT_DIR, f"drake_blender_{idx:02d}{suffix}.mp4")

    # Check if already rendered (never for a preview, which is deliberately
    # short and must not be mistaken for, or overwrite, a finished segment).
    # This is an *existence* check, not a freshness check: it cannot tell that
    # the HTML it was rendered from has since been regenerated, so re-export a
    # point and this stage will happily keep the stale video. Pass --force after
    # changing anything upstream of the export.
    if preview_frames == 0 and not force and os.path.exists(mp4_path):
        dur = float(subprocess.check_output([
            "ffprobe", "-v", "quiet", "-show_entries", "format=duration",
            "-of", "csv=p=0", mp4_path,
        ]).decode().strip())
        if dur > 5:
            print(f"  [{idx:2d}] already rendered ({dur:.1f}s)")
            return mp4_path

    import tempfile
    script_file = os.path.join(frames_dir, "render.py")
    with open(script_file, "w") as f:
        # Bind the directory holding blender_render_common.py; the template
        # runs inside Blender, where this repo is not on sys.path.
        f.write("SCRIPTS_DIR = %r\n" % os.path.dirname(os.path.abspath(__file__)))
        f.write("SIM_W, SIM_H = %d, %d\n" % (SIM_RENDER_W, SIM_RENDER_H))
        f.write("PREVIEW_FRAMES = %d\n" % preview_frames)
        f.write(RENDER_SCRIPT)

    print(f"  [{idx:2d}] Blender rendering...")
    result = subprocess.run(
        [BLENDER, "--background", "--python", script_file,
         "--", os.path.abspath(html_path), os.path.abspath(frames_dir)],
        # Cycles is slower per frame than the EEVEE path this replaced, and a
        # timeout that fires after the render has done all its work is the
        # worst failure mode available here.
        capture_output=True, text=True, timeout=7200,
    )

    if result.returncode != 0:
        print(f"  [{idx:2d}] Blender failed")
        for line in result.stdout.split("\n")[-5:]:
            if line.strip():
                print(f"         {line.strip()}")
        return None

    frames = sorted(glob.glob(os.path.join(frames_dir, "frame_*.png")))
    if not frames:
        print(f"  [{idx:2d}] no frames rendered")
        return None

    subprocess.run(
        ["ffmpeg", "-y", "-framerate", "30",
         "-i", os.path.join(frames_dir, "frame_%04d.png"),
         "-c:v", "libx264", "-crf", "22", "-preset", "fast",
         "-pix_fmt", "yuv420p", "-an", mp4_path],
        capture_output=True, text=True, check=True,
    )

    dur = float(subprocess.check_output([
        "ffprobe", "-v", "quiet", "-show_entries", "format=duration",
        "-of", "csv=p=0", mp4_path,
    ]).decode().strip())
    print(f"  [{idx:2d}] {len(frames)} frames -> {dur:.1f}s")
    return mp4_path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--indices", type=int, nargs="*", default=list(range(20)))
    ap.add_argument("--force", action="store_true",
                    help="re-render even when the output mp4 is already there; "
                         "needed whenever the point's HTML was re-exported")
    ap.add_argument("--preview-frames", type=int, default=0, metavar="N",
                    help="render only the first N frames, to a separate "
                         "*_preview.mp4, for checking the camera cheaply")
    args = ap.parse_args()

    os.makedirs(SCRATCH_BASE, exist_ok=True)

    # Copy existing point_00 blender render if available
    existing = os.path.join(OUT_DIR, "drake_point_00_blender.mp4")
    target = os.path.join(OUT_DIR, "drake_blender_00.mp4")
    if (not args.force and os.path.exists(existing)
            and not os.path.exists(target)):
        import shutil
        shutil.copy(existing, target)
        print("  [ 0] copied existing Blender render")

    results = {}
    for idx in args.indices:
        results[idx] = render_point(idx, preview_frames=args.preview_frames,
                                    force=args.force)

    ok = sum(1 for v in results.values() if v)
    print(f"\n{ok}/{len(args.indices)} points rendered")


if __name__ == "__main__":
    main()
