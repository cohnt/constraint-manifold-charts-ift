"""Blender-render the IIWA bimanual segment from its meshcat HTML.

Same meshcat-HTML -> Blender path the RBY1 grid renders use
(``blender_render_grid.py`` is the reference), but framed for this scene: two
IIWAs facing each other across a table, a two-plate shelf unit, and the carried
plank.  ``generate_iiwa_meshcat.py`` writes the HTML; this writes the MP4.

    .venv/bin/python scripts/video/generate_iiwa_meshcat.py            # HTML
    .venv/bin/python scripts/video/blender_render_iiwa.py              # MP4

Useful while framing:
    .venv/bin/python scripts/video/blender_render_iiwa.py --dump
        list every imported object with its world bounding box, render nothing
    .venv/bin/python scripts/video/blender_render_iiwa.py --still 0.85
        render one frame at 85% of the way through and stop

This supersedes the Drake-VTK renderer in ``render_iiwa_experiment.py``.  That
script still works and is still the place the trajectory is *verified*; it is
just not what produces video/v2_iiwa_bimanual.mp4 any more.

Traps this script already accounts for (do not re-discover them):
  * meshcat strips OBJ material files, so meshes arrive with flat per-face
    colours; materials are set from those colours with a brightness-aware
    metallic/roughness rule, as in blender_render_grid.py;
  * collision geometry is *deleted*, not hidden -- hide_render is unreliable
    across engine versions;
  * the plank is the whole point of the shot, so its presence in the Blender
    scene is asserted, not assumed: --dump prints it and the render aborts if no
    object named after it survived the import.
"""

import argparse
import glob
import json
import os
import subprocess
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from blender_paths import BLENDER, EXTENSIONS  # noqa: E402

DEFAULT_HTML = os.path.join(REPO, "video", "v2_iiwa_bimanual.html")
DEFAULT_OUT = os.path.join(REPO, "video", "v2_iiwa_bimanual.mp4")
SCRATCH = os.path.join(REPO, "video", "scratch_blender_iiwa_place")

# Camera.  The scene occupies x in [-0.3, 1.05], y in [-0.3, 1.05], z in
# [0, 1.3]: iiwa_left is based at the origin, iiwa_right at y = 0.765, and the
# shelf unit at x = 0.8 with its back wall at x = 1.0.
#
# This is nearly a front view, looking along +x into the shelf's open face, with
# a small -y offset for depth.  It was chosen against rendered stills of the
# first and last frames, not by eye over the scene: from a 3/4 view (azimuth 40
# deg or more off the front, which is where blender_render_grid.py's camera
# lives) the *pick* pose is unusable, because the plank sits between two arms
# that reach along y and the near arm's forearm covers it almost completely.
# Near the front, the plank is broadside at both ends of the motion.
CAM_LOC = (-1.85, -0.55, 1.08)
CAM_TARGET = (0.58, 0.38, 0.48)
CAM_LENS = 35.0

# Padding, in seconds, of cloned frames at the head of the clip.  The tail hold
# is baked into the meshcat recording instead (see generate_iiwa_meshcat --hold)
# so that the held pose is a real simulated pose.
HEAD_HOLD_S = 0.8

RENDER_SCRIPT = r'''
import sys, os, math, glob, json
sys.path.insert(0, EXTENSIONS_PATH)
import bpy
from mathutils import Vector
from meshcat_html_importer.blender_impl.scene_builder import build_scene_from_file

args = sys.argv[sys.argv.index("--") + 1:]
html_path, out_dir, cfg_json = args[0], args[1], args[2]
cfg = json.loads(cfg_json)

build_scene_from_file(html_path, target_fps=cfg["fps"], clear_scene=True)

# --- collision geometry: delete, never hide (hide_render is unreliable) ------
removed = [o for o in bpy.data.objects
           if o.type == 'MESH' and ('collision' in o.name.lower()
                                    or 'proximity' in o.name.lower())]
for o in removed:
    bpy.data.objects.remove(o, do_unlink=True)
print('BR: removed %d collision objects' % len(removed))


def world_bounds(obj):
    pts = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    return ([min(p[i] for p in pts) for i in range(3)],
            [max(p[i] for p in pts) for i in range(3)])


meshes = [o for o in bpy.data.objects if o.type == 'MESH']
print('BR: %d mesh objects' % len(meshes))
for o in sorted(meshes, key=lambda o: o.name):
    lo, hi = world_bounds(o)
    mats = ','.join(m.name for m in o.data.materials if m) if o.data.materials else '-'
    print('BR:   %-58s x[%6.3f %6.3f] y[%6.3f %6.3f] z[%6.3f %6.3f]  %s'
          % (o.name, lo[0], hi[0], lo[1], hi[1], lo[2], hi[2], mats))

plank = [o for o in meshes if cfg["plank_token"] in o.name.lower()]
print('BR: plank objects: %s' % [o.name for o in plank])
if not plank:
    print('BR: FATAL no plank object survived the import')
    sys.exit(3)
for o in plank:
    lo, hi = world_bounds(o)
    print('BR: plank bounds x[%.3f %.3f] y[%.3f %.3f] z[%.3f %.3f] hide_render=%s'
          % (lo[0], hi[0], lo[1], hi[1], lo[2], hi[2], o.hide_render))
    o.hide_render = False
    o.hide_viewport = False

if cfg["dump_only"]:
    print('BR: DUMP DONE')
    sys.exit(0)

# --- materials: the importer leaves flat per-face colours on the BSDF --------
for obj in meshes:
    for poly in obj.data.polygons:
        poly.use_smooth = True
    for mat in (obj.data.materials or []):
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

# --- camera -----------------------------------------------------------------
cam = bpy.data.cameras.new('C')
cam.lens = cfg["lens"]
cam.clip_end = 100
co = bpy.data.objects.new('C', cam)
bpy.context.scene.collection.objects.link(co)
bpy.context.scene.camera = co
co.location = tuple(cfg["cam_loc"])
t = bpy.data.objects.new('T', None)
t.location = tuple(cfg["cam_target"])
bpy.context.scene.collection.objects.link(t)
tc = co.constraints.new(type='TRACK_TO')
tc.target = t
tc.track_axis = 'TRACK_NEGATIVE_Z'
tc.up_axis = 'UP_Y'

# --- lighting ---------------------------------------------------------------
# Key from over the camera's left shoulder so the shelf's open front and the
# plank's top face both catch light; fill from the far side to keep the far arm
# off black; rim from behind the shelf to separate the arms from the backdrop.
key = bpy.data.lights.new('Key', 'SUN')
key.energy = 4.0
key.color = (1.0, 0.98, 0.95)
ko = bpy.data.objects.new('Key', key)
ko.rotation_euler = (math.radians(52), math.radians(-14), math.radians(40))
bpy.context.scene.collection.objects.link(ko)

fill = bpy.data.lights.new('Fill', 'AREA')
fill.energy = 320
fill.size = 5
fo = bpy.data.objects.new('Fill', fill)
fo.location = (-1.0, 2.6, 2.2)
fo.rotation_euler = (math.radians(40), 0, math.radians(150))
bpy.context.scene.collection.objects.link(fo)

rim = bpy.data.lights.new('Rim', 'SPOT')
rim.energy = 400
rim.spot_size = math.radians(80)
ro = bpy.data.objects.new('Rim', rim)
ro.location = (2.2, 0.4, 2.4)
ro.rotation_euler = (math.radians(48), 0, math.radians(90))
bpy.context.scene.collection.objects.link(ro)

# Soft top light so the plank's upper face reads against the shelf plate.
top = bpy.data.lights.new('Top', 'AREA')
top.energy = 260
top.size = 3
to_ = bpy.data.objects.new('Top', top)
to_.location = (0.3, 0.4, 2.6)
bpy.context.scene.collection.objects.link(to_)

w = bpy.data.worlds.new('W')
bpy.context.scene.world = w
w.use_nodes = True
w.node_tree.nodes['Background'].inputs['Color'].default_value = (0.102, 0.102, 0.18, 1)

# --- render -----------------------------------------------------------------
s = bpy.context.scene
s.render.resolution_x = cfg["width"]
s.render.resolution_y = cfg["height"]
s.render.resolution_percentage = 100
s.render.fps = cfg["fps"]
try:
    s.render.engine = 'BLENDER_EEVEE_NEXT'
except Exception:
    s.render.engine = 'BLENDER_EEVEE'
try:
    s.eevee.taa_render_samples = cfg["samples"]
except AttributeError:
    pass
s.render.image_settings.file_format = 'PNG'

f_start, f_end = s.frame_start, s.frame_end
print('BR: frames %d-%d' % (f_start, f_end))

for f in glob.glob(os.path.join(out_dir, 'frame_*.png')):
    os.remove(f)
s.render.filepath = os.path.join(out_dir, 'frame_')

if cfg["still"] is not None:
    f = int(round(f_start + cfg["still"] * (f_end - f_start)))
    s.frame_start = s.frame_end = f
    s.frame_set(f)
    print('BR: still at frame %d' % f)
    bpy.ops.render.render(animation=True)
else:
    bpy.ops.render.render(animation=True)
print('BR: DONE')
'''


def run_blender(html, frames_dir, cfg, timeout=3600):
    os.makedirs(frames_dir, exist_ok=True)
    script = RENDER_SCRIPT.replace("EXTENSIONS_PATH", repr(EXTENSIONS))
    script_file = os.path.join(frames_dir, "render.py")
    with open(script_file, "w") as f:
        f.write(script)
    cmd = [BLENDER, "--background", "--python", script_file, "--",
           os.path.abspath(html), os.path.abspath(frames_dir), json.dumps(cfg)]
    print("$ " + " ".join(cmd[:4]) + " ...")
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    for line in proc.stdout.splitlines():
        if line.startswith("BR:") or "Error" in line or "Traceback" in line:
            print(line)
    if proc.returncode != 0:
        print(proc.stdout[-4000:])
        print(proc.stderr[-4000:])
        raise SystemExit(f"blender failed with code {proc.returncode}")
    return proc.stdout


FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def encode(frames_dir, out_path, fps, head_hold_s, captions=True):
    """Overlay the captions and pipe the frames to ffmpeg.

    The tail hold is already in the meshcat recording (real simulated poses);
    the head hold is made here by repeating the first frame.
    """
    from PIL import Image, ImageDraw, ImageFont

    frames = sorted(glob.glob(os.path.join(frames_dir, "frame_*.png")))
    if not frames:
        raise SystemExit("no frames rendered")

    lines = []
    if captions:
        import render_iiwa_experiment as vtk_renderer
        try:
            font = ImageFont.truetype(FONT_PATH, 28)
            font_small = ImageFont.truetype(FONT_PATH, 20)
        except OSError:
            font = font_small = ImageFont.load_default()
        lines = vtk_renderer.CAPTIONS(font, font_small, dark_bg=True)

    first = Image.open(frames[0])
    w, h = first.size
    cmd = ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{w}x{h}", "-r", str(fps), "-i", "pipe:0",
           "-c:v", "libx264", "-crf", "20", "-preset", "slow",
           "-pix_fmt", "yuv420p", "-an", out_path]
    pipe = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    order = [frames[0]] * int(round(head_hold_s * fps)) + frames
    for path in order:
        img = Image.open(path).convert("RGB")
        if lines:
            draw = ImageDraw.Draw(img)
            for y, text, fill, fnt in lines:
                draw.text((30, y), text, fill=fill, font=fnt)
        pipe.stdin.write(img.tobytes())
    pipe.stdin.close()
    pipe.wait()
    if pipe.returncode != 0:
        print(pipe.stderr.read().decode()[-3000:])
        raise SystemExit("ffmpeg failed")
    dur = float(subprocess.check_output([
        "ffprobe", "-v", "quiet", "-show_entries", "format=duration",
        "-of", "csv=p=0", out_path]).decode().strip())
    size = subprocess.check_output([
        "ffprobe", "-v", "quiet", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,nb_frames",
        "-of", "csv=p=0", out_path]).decode().strip()
    print(f"Wrote {out_path}: {len(frames)} rendered frames, {dur:.2f}s, {size}")
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--html", default=DEFAULT_HTML)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--frames-dir", default=SCRATCH)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--samples", type=int, default=48)
    ap.add_argument("--dump", action="store_true",
                    help="list imported objects and exit without rendering")
    ap.add_argument("--still", type=float, default=None,
                    help="render a single frame at this fraction of the clip")
    ap.add_argument("--head-hold", type=float, default=HEAD_HOLD_S)
    ap.add_argument("--no-captions", action="store_true",
                    help="render without the on-frame caption lines")
    ap.add_argument("--cam-loc", type=float, nargs=3, default=list(CAM_LOC))
    ap.add_argument("--cam-target", type=float, nargs=3, default=list(CAM_TARGET))
    ap.add_argument("--lens", type=float, default=CAM_LENS)
    args = ap.parse_args()

    if not os.path.exists(args.html):
        raise SystemExit(f"{args.html} does not exist -- run generate_iiwa_meshcat.py first")

    cfg = dict(
        fps=args.fps, width=args.width, height=args.height, samples=args.samples,
        cam_loc=list(args.cam_loc), cam_target=list(args.cam_target), lens=args.lens,
        dump_only=bool(args.dump), still=args.still, plank_token="plank",
    )
    frames_dir = args.frames_dir + ("_still" if args.still is not None else "")
    run_blender(args.html, frames_dir, cfg)
    if args.dump:
        return
    if args.still is not None:
        frames = sorted(glob.glob(os.path.join(frames_dir, "frame_*.png")))
        print("still: " + (frames[-1] if frames else "NONE"))
        return
    encode(frames_dir, args.out, args.fps, args.head_hold,
           captions=not args.no_captions)


if __name__ == "__main__":
    main()
