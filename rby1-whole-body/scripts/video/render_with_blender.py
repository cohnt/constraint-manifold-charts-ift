"""Render a meshcat HTML recording through Blender with drake-blender-tools.

Imports the meshcat HTML into Blender, sets up camera and lighting,
and renders to a video file.

Usage:
    .venv/bin/python scripts/video/render_with_blender.py video/v2_iiwa_bimanual.html --output video/v2_iiwa_bimanual.mp4
    .venv/bin/python scripts/video/render_with_blender.py video/v2_eaik_grasp.html --output video/v2_eaik_grasp.mp4 --orbit
"""

import argparse
import os
import subprocess
import sys
import tempfile

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from blender_paths import BLENDER  # noqa: E402

RENDER_SCRIPT_TEMPLATE = '''
import bpy
import math
import os
import sys
from mathutils import Vector

# The add-on lives in Blender's user extensions directory, outside this repo.
# Import its API directly rather than calling bpy.ops.import_scene.meshcat_recording:
# that operator only exists once the extension is *enabled* in user preferences, and
# in --background it generally is not, so the operator call raised and this script
# used to carry on and render an empty scene.  build_scene_from_file is the entry
# point the rest of the pipeline uses.
sys.path.insert(0, os.path.expanduser('~/.config/blender/5.0/extensions/user_default'))
from meshcat_html_importer.blender_impl.scene_builder import build_scene_from_file
from meshcat_html_importer.parser.html_extractor import parse_html_recording

# SCRIPTS_DIR is prepended to this template by the driver, so the shared
# Blender helpers are importable from inside Blender's own interpreter.
sys.path.insert(0, SCRIPTS_DIR)
from blender_render_common import configure_render_quality

args = sys.argv[sys.argv.index("--") + 1:]
html_path = args[0]
output_dir = args[1]
fps = int(args[2])
orbit = args[3] == "true"
width = int(args[4])
height = int(args[5])
yaw_deg = float(args[6])
samples = int(args[7])
preview_frames = int(args[8])   # 0 = render the whole animation

# --speed asks for a render rate above the output rate, and the importer can
# only interpolate *down* to it -- above the recording's own rate it hands its
# keyframes back unchanged and the extra frames are duplicates, which is the
# stutter the retiming filter used to produce.  Refuse instead: re-record the
# motion at a higher rate.
recording_fps = float(parse_html_recording(html_path)["animation_fps"])
if fps > recording_fps:
    raise SystemExit(
        "[render] asked for %d fps but %s was recorded at %g fps; the extra "
        "frames would be duplicates" % (fps, os.path.basename(html_path),
                                        recording_fps))

# No try/except: a failed import must stop the run.  Rendering an empty scene
# looks like success all the way through to the assembly, which then ships the
# previous segment.
objs = build_scene_from_file(html_path, target_fps=fps, clear_scene=True)

# Collision geometry is deleted, never hidden -- hide_render is unreliable across
# engine versions.  Classify by meshcat path, not by object name: the importer
# emits names like `visual.001` and `collision_Convex.012`, which are neither
# unique nor stable.  Generators publish only the illustration role, so this is
# defence in depth.
by_path = {}
for path, obj in objs.items():
    if path.startswith('/drake/collision/') or path.startswith('/drake/proximity/'):
        bpy.data.objects.remove(obj, do_unlink=True)
    elif path.startswith('/drake/visual/'):
        by_path[path] = obj

# Meshcat strips OBJ material files, so textures arrive as flat per-face colours.
# The rule below, and the table/shelf materials after it, are the ones the sibling
# repo's swept-volume figure uses, so this segment matches that figure.
#
# No metallic. The scene's bright materials are the white shelf panels and the
# arms' light-grey shells -- painted steel and plastic, neither of them mirrors.
# At Metallic 0.3 and Roughness 0.4 they threw a blown highlight across the top
# arm and clipped that whole side of the shelf to pure white.
for obj in bpy.data.objects:
    if obj.type != 'MESH':
        continue
    for poly in obj.data.polygons:
        poly.use_smooth = True
    try:
        if hasattr(obj.data, 'use_auto_smooth'):
            obj.data.use_auto_smooth = True
            obj.data.auto_smooth_angle = math.radians(30)
    except Exception:
        pass
    for mat in (obj.data.materials or []):
        if not mat or not mat.use_nodes:
            continue
        bsdf = mat.node_tree.nodes.get('Principled BSDF')
        if not bsdf:
            continue
        c = bsdf.inputs['Base Color'].default_value
        brightness = 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]
        bsdf.inputs['Metallic'].default_value = 0.0
        bsdf.inputs['Roughness'].default_value = 0.65 if brightness > 0.6 else 0.6


# --- table and shelf materials ----------------------------------------------
# The table and shelves arrive from Drake as flat untextured colour, and a white
# shelf panel is the brightest thing in frame -- the first surface to clip and the
# one the arms are hardest to read against.  Worn steel and plywood are darker and
# carry their own detail, so the arms sit on them rather than in front of them.
#
# Must run after the pass above, which walks obj.data.materials and would
# otherwise put its roughness pass on top of these.

def _principled(name):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    return mat, nt, nt.nodes['Principled BSDF']


def _object_coords(nt, scale):
    """Object-space texture coordinates, so the grain does not swim with the mesh.

    Object rather than Generated: Generated normalises to each object's own
    bounding box, which would give the shelf's thin side panels a wildly
    different grain scale from its back panel.
    """
    coords = nt.nodes.new('ShaderNodeTexCoord')
    mapping = nt.nodes.new('ShaderNodeMapping')
    mapping.inputs['Scale'].default_value = scale
    nt.links.new(coords.outputs['Object'], mapping.inputs['Vector'])
    return mapping


def _mix(nt, data_type, fac, a, b):
    """A Mix node wired to `fac`, blending `a` -> `b`; either may be a socket.

    `ShaderNodeMix` carries one set of sockets per data type and only the ones
    matching `data_type` are enabled, so A and B are found by filtering on
    `enabled` -- looking them up by name alone returns the float pair whatever
    the node is set to.
    """
    node = nt.nodes.new('ShaderNodeMix')
    node.data_type = data_type
    slots = {sock.name: sock for sock in node.inputs if sock.enabled}
    for name, value in (('A', a), ('B', b)):
        if hasattr(value, 'is_output'):
            nt.links.new(value, slots[name])
        else:
            slots[name].default_value = value
    nt.links.new(fac, node.inputs['Factor'])
    return [sock for sock in node.outputs if sock.enabled][0]


def steel_material():
    """Old machine-shop steel: near-black, scratched back to bare metal, rusting.

    Scratches are noise sampled through a 90:1 stretched coordinate space, so it
    is fine across the surface and smeared along it, clipped to a narrow band so
    only the peaks survive as thin bright lines.  Rust is a coarser noise
    thresholded into patches; it is not metal, so it drives Metallic to zero as
    well as colouring -- leaving it metallic is what makes procedural rust read
    as orange chrome.
    """
    mat, nt, bsdf = _principled('BlackScratchedMetal')

    scratch_map = _object_coords(nt, (1.5, 30.0, 1.5))
    scratch = nt.nodes.new('ShaderNodeTexNoise')
    scratch.inputs['Scale'].default_value = 6.0
    scratch.inputs['Detail'].default_value = 8.0
    scratch.inputs['Roughness'].default_value = 0.75
    nt.links.new(scratch_map.outputs['Vector'], scratch.inputs['Vector'])

    scratch_ramp = nt.nodes.new('ShaderNodeValToRGB')
    scratch_ramp.color_ramp.elements[0].position = 0.48
    scratch_ramp.color_ramp.elements[1].position = 0.64
    nt.links.new(scratch.outputs['Fac'], scratch_ramp.inputs['Fac'])

    rust_map = _object_coords(nt, (1.0, 1.0, 1.0))
    rust = nt.nodes.new('ShaderNodeTexNoise')
    rust.inputs['Scale'].default_value = 2.2
    rust.inputs['Detail'].default_value = 9.0
    rust.inputs['Roughness'].default_value = 0.65
    nt.links.new(rust_map.outputs['Vector'], rust.inputs['Vector'])

    rust_ramp = nt.nodes.new('ShaderNodeValToRGB')
    rust_ramp.color_ramp.elements[0].position = 0.50
    rust_ramp.color_ramp.elements[1].position = 0.72
    nt.links.new(rust.outputs['Fac'], rust_ramp.inputs['Fac'])

    scratched = _mix(nt, 'RGBA', scratch_ramp.outputs['Color'],
                     (0.030, 0.030, 0.034, 1.0),    # near-black worn finish
                     (0.46, 0.46, 0.48, 1.0))       # bare steel in the scratch
    base = _mix(nt, 'RGBA', rust_ramp.outputs['Color'],
                scratched, (0.20, 0.072, 0.026, 1.0))
    nt.links.new(base, bsdf.inputs['Base Color'])

    invert = nt.nodes.new('ShaderNodeMath')
    invert.operation = 'SUBTRACT'
    invert.inputs[0].default_value = 1.0
    nt.links.new(rust_ramp.outputs['Color'], invert.inputs[1])
    nt.links.new(invert.outputs['Value'], bsdf.inputs['Metallic'])

    scratch_rough = _mix(nt, 'FLOAT', scratch_ramp.outputs['Color'], 0.62, 0.35)
    rough = _mix(nt, 'FLOAT', rust_ramp.outputs['Color'], scratch_rough, 0.90)
    nt.links.new(rough, bsdf.inputs['Roughness'])

    height = nt.nodes.new('ShaderNodeMath')
    height.operation = 'ADD'
    nt.links.new(scratch_ramp.outputs['Color'], height.inputs[0])
    nt.links.new(rust_ramp.outputs['Color'], height.inputs[1])
    bump = nt.nodes.new('ShaderNodeBump')
    bump.inputs['Strength'].default_value = 0.15
    nt.links.new(height.outputs['Value'], bump.inputs['Height'])
    nt.links.new(bump.outputs['Normal'], bsdf.inputs['Normal'])
    return mat


def wood_material():
    """Plank wood: warm grain from a distorted wave texture."""
    mat, nt, bsdf = _principled('ShelfWood')
    bsdf.inputs['Metallic'].default_value = 0.0
    bsdf.inputs['Roughness'].default_value = 0.45

    mapping = _object_coords(nt, (1.0, 1.0, 1.0))
    wave = nt.nodes.new('ShaderNodeTexWave')
    wave.wave_type = 'BANDS'
    wave.bands_direction = 'Z'
    wave.inputs['Scale'].default_value = 5.0
    wave.inputs['Distortion'].default_value = 3.0
    wave.inputs['Detail'].default_value = 3.0
    wave.inputs['Detail Scale'].default_value = 1.4
    nt.links.new(mapping.outputs['Vector'], wave.inputs['Vector'])

    ramp = nt.nodes.new('ShaderNodeValToRGB')
    # Low contrast on purpose: a single object-space direction cannot run along
    # every panel of the unit at once, so the bands are kept subtle enough that
    # the ones crossing a shelf the "wrong" way do not draw the eye.
    ramp.color_ramp.elements[0].position = 0.25
    ramp.color_ramp.elements[0].color = (0.150, 0.076, 0.036, 1.0)
    ramp.color_ramp.elements[1].position = 0.75
    ramp.color_ramp.elements[1].color = (0.275, 0.150, 0.072, 1.0)
    nt.links.new(wave.outputs['Fac'], ramp.inputs['Fac'])
    nt.links.new(ramp.outputs['Color'], bsdf.inputs['Base Color'])

    bump = nt.nodes.new('ShaderNodeBump')
    bump.inputs['Strength'].default_value = 0.06
    nt.links.new(wave.outputs['Fac'], bump.inputs['Height'])
    nt.links.new(bump.outputs['Normal'], bsdf.inputs['Normal'])
    return mat


def load_blenderkit_material(slug, root):
    """Append a cached BlenderKit material by asset slug, or return None.

    BlenderKit stores each downloaded asset under `<root>/<slug>_<uuid>/
    <name>_<uuid>.blend`, one material per file.  Append rather than link so the
    render does not depend on the cache staying put, and match on the slug prefix
    because the uuids differ per machine and per download.  BlenderKit truncates
    the slug in the directory name to 16 characters, so try the truncation too.

    The cache is the user's own BlenderKit library and is not part of this repo,
    so every caller has to cope with this returning None.
    """
    import glob
    hits = sorted(glob.glob(os.path.join(root, slug + '*', '*.blend')))
    if not hits and len(slug) > 16:
        hits = sorted(glob.glob(os.path.join(root, slug[:16] + '*', '*.blend')))
    if not hits:
        return None
    with bpy.data.libraries.load(hits[0], link=False) as (src, dst):
        if not src.materials:
            return None
        dst.materials = [src.materials[0]]
    return dst.materials[0]


def box_project_uvs(obj, texels_per_metre=1.0):
    """Give `obj` a world-space box projection in its own UV layer.

    Drake's shelf panels arrive as six-polygon boxes with no UV layer at all, so
    an image-textured material has nothing to sample and renders as flat colour.
    Projecting each face along its dominant world axis, in metres, also fixes
    texel density: every surface gets the same texture scale regardless of how
    large the panel is, which is what stops the thin shelf edges from showing a
    wildly magnified crop of the wood.

    Done by hand rather than with `bpy.ops.uv.cube_project`, which needs an
    edit-mode context that does not exist under --background.
    """
    mesh = obj.data
    uv = mesh.uv_layers.get('BoxProject') or mesh.uv_layers.new(name='BoxProject')
    uv.active = uv.active_render = True
    basis = obj.matrix_world.to_3x3()
    for poly in mesh.polygons:
        normal = basis @ poly.normal
        axis = max(range(3), key=lambda i: abs(normal[i]))
        u_axis, v_axis = ((1, 2), (0, 2), (0, 1))[axis]
        for loop_index in poly.loop_indices:
            point = obj.matrix_world @ mesh.vertices[mesh.loops[loop_index].vertex_index].co
            uv.data[loop_index].uv = (point[u_axis] * texels_per_metre,
                                      point[v_axis] * texels_per_metre)
    return uv


def darken_base_color(mat, k):
    """Return a copy of `mat` with its Base Color multiplied by `k`.

    The table's brightness is otherwise decided by luck. Its UVs are box-projected
    in world space, so which part of the 2048x2048 source photograph lands under
    the camera depends on where the table happens to sit -- and that photo holds
    both near-black oxidised areas and pale worn-bare ones. The same asset renders
    dark in the bimanual scene and pale here purely because the two tables are
    about 0.4 m apart in world space.
    
    Scaling the lamps cannot fix that: it is global, so it darkens the arms in
    the same breath, and the arms are flat untextured colour that already matches
    the bimanual segment exactly. Darkening this one material instead leaves the
    lighting -- and therefore the arms -- identical to that segment.
    """
    if k == 1.0:
        return mat
    mat = mat.copy()
    nt = mat.node_tree
    for node in list(nt.nodes):
        if node.type != 'BSDF_PRINCIPLED':
            continue
        bc = node.inputs['Base Color']
        mul = nt.nodes.new('ShaderNodeMix')
        mul.data_type = 'RGBA'
        mul.blend_type = 'MULTIPLY'
        mul.inputs['Factor'].default_value = 1.0
        slots = {sock.name: sock for sock in mul.inputs if sock.enabled}
        if bc.is_linked:
            nt.links.new(bc.links[0].from_socket, slots['A'])
        else:
            slots['A'].default_value = tuple(bc.default_value)
        slots['B'].default_value = (k, k, k, 1.0)
        nt.links.new([o for o in mul.outputs if o.enabled][0], bc)
    return mat


def resolve_material(spec, fallback, root, label):
    """`spec` is a BlenderKit slug, or "procedural" for the built-in."""
    if spec == 'procedural':
        return fallback(), 'procedural'
    mat = load_blenderkit_material(spec, root)
    if mat is not None:
        return mat, 'blenderkit:' + spec
    print("[material] WARNING: no BlenderKit asset matching '%s' under %s; falling "
          "back to the procedural %s. Download it in Blender first, or pass "
          "--%s-material procedural to silence this." % (spec, root, label, label))
    return fallback(), 'procedural (fallback)'


_bk_root = os.path.expanduser(blenderkit_dir)
_table_mat, _table_src = resolve_material(table_material, steel_material, _bk_root, 'table')
if table_brightness != 1.0:
    _table_mat = darken_base_color(_table_mat, table_brightness)
    _table_src += ' x%.2f' % table_brightness
_shelf_mat, _shelf_src = resolve_material(shelf_material, wood_material, _bk_root, 'shelf')
_styled = {'table': (_table_mat, table_texels), 'old_shelves': (_shelf_mat, shelf_texels)}
_counts = dict.fromkeys(_styled, 0)
for path, obj in by_path.items():
    model = path[len('/drake/visual/'):].split('/', 1)[0]
    entry = _styled.get(model)
    if entry is None or obj.type != 'MESH':
        continue
    mat, texels = entry
    box_project_uvs(obj, texels)
    obj.data.materials.clear()
    obj.data.materials.append(mat)
    _counts[model] += 1
print('[material] table: %d objects from %s, old_shelves: %d objects from %s'
      % (_counts['table'], _table_src, _counts['old_shelves'], _shelf_src))

scene = bpy.context.scene
meshes = [o for o in bpy.data.objects if o.type == 'MESH']
if not meshes:
    raise RuntimeError("no meshes imported from " + html_path)

# Frame on the robots and what they interact with, not on the furniture. The
# IIWA scene's table is 2.18 m across and drops a metre below the floor for its
# legs, so including it framed a 1.9 m tall region when the arms and shelves
# occupy 0.93 m -- the arms came out about half the size they should be. These
# objects still render; they are just allowed to bleed off-frame.
subject = [o for o in meshes
           if not any(w in o.name.lower() for w in ('table', 'ground', 'floor'))]
if not subject:
    subject = meshes

# Fit the camera to the subject's swept bounding box instead of guessing a
# location: sample the animation, project every bound-box corner onto the camera
# axes, then solve for the focal length that contains it with a margin.
#
# yaw_deg rotates the camera's standpoint about the world Z axis, measured
# counter-clockwise from above. The IIWA segment uses -90, which puts both arms
# in the foreground with the shelves behind them; at the default 0 the camera
# looks along the shelves and the near arm occludes the far one.
_az = math.radians(-45.0 + yaw_deg)               # base 3/4 view is at -45 deg
u = Vector((math.cos(_az), math.sin(_az), 0.45)); u.normalize()
fwd = -u
right = fwd.cross(Vector((0, 0, 1))); right.normalize()
up = right.cross(fwd); up.normalize()

step = max(1, (scene.frame_end - scene.frame_start) // 60)
amin = bmin = cmin = 1e9
amax = bmax = cmax = -1e9
for fr in range(scene.frame_start, scene.frame_end + 1, step):
    scene.frame_set(fr)
    bpy.context.view_layer.update()
    for o in subject:
        M = o.matrix_world
        for corner in o.bound_box:
            p = M @ Vector(corner)
            a = p.dot(right); b = p.dot(up); d = p.dot(fwd)
            amin = min(amin, a); amax = max(amax, a)
            bmin = min(bmin, b); bmax = max(bmax, b)
            cmin = min(cmin, d); cmax = max(cmax, d)
centre = ((amin + amax) / 2) * right + ((bmin + bmax) / 2) * up + ((cmin + cmax) / 2) * fwd
half_w = (amax - amin) / 2
half_h = (bmax - bmin) / 2
radius = max(half_w, half_h, 0.1)
dist = 3.0 * radius
cam_loc = centre + dist * u

MARGIN = 1.12
sensor_w = 36.0
sensor_h = sensor_w * height / width
near = dist - (cmax - (cmin + cmax) / 2)
lens = min((sensor_w / 2) * near / (half_w * MARGIN),
           (sensor_h / 2) * near / (half_h * MARGIN))

cam_data = bpy.data.cameras.new("Camera")
cam_data.lens = lens
cam_data.clip_end = 200
cam_obj = bpy.data.objects.new("Camera", cam_data)
scene.collection.objects.link(cam_obj)
scene.camera = cam_obj
cam_obj.location = cam_loc

target = bpy.data.objects.new("CameraTarget", None)
target.location = centre
scene.collection.objects.link(target)
constraint = cam_obj.constraints.new(type="TRACK_TO")
constraint.target = target
constraint.track_axis = "TRACK_NEGATIVE_Z"
constraint.up_axis = "UP_Y"

if orbit:
    total = scene.frame_end - scene.frame_start + 1
    base = math.atan2(u.y, u.x)
    horiz = math.hypot(cam_loc.x - centre.x, cam_loc.y - centre.y)
    for i in range(total):
        angle = base + 2 * math.pi * i / total
        cam_obj.location = (centre.x + horiz * math.cos(angle),
                            centre.y + horiz * math.sin(angle),
                            cam_loc.z)
        cam_obj.keyframe_insert(data_path="location",
                                frame=scene.frame_start + i)

key = bpy.data.lights.new("KeyLight", type="SUN")
key.energy = 4.0
key.color = (1.0, 0.98, 0.95)
key_obj = bpy.data.objects.new("KeyLight", key)
key_obj.rotation_euler = (math.radians(50), math.radians(10), math.radians(-30))
scene.collection.objects.link(key_obj)

fill = bpy.data.lights.new("FillLight", type="AREA")
fill.energy = 150.0
fill.size = 4.0
fill_obj = bpy.data.objects.new("FillLight", fill)
fill_obj.location = centre + Vector((-2, 2, 2.5))
scene.collection.objects.link(fill_obj)

rim = bpy.data.lights.new("RimLight", type="SPOT")
rim.energy = 200.0
rim.spot_size = math.radians(60)
rim_obj = bpy.data.objects.new("RimLight", rim)
rim_obj.location = centre + Vector((-1.5, -2, 3))
rim_obj.rotation_euler = (math.radians(35), 0, math.radians(-150))
scene.collection.objects.link(rim_obj)

# The lights sit at fixed offsets from the subject centre while the camera
# distance is solved from the subject's size, so a scene with a smaller subject
# -- the IRIS one has no shelves -- is lit from proportionally closer and renders
# brighter, until the tabletop is the brightest thing in frame.
#
# Scale the lamps, not the view transform: an exposure adjustment also darkens
# the world background, and that backdrop colour is shared with every other
# segment in the cut, so moving it breaks continuity across a cut that nothing
# else in the video justifies.
if light_scale != 1.0:
    for lamp in (key, fill, rim):
        lamp.energy *= light_scale
    print("[render] light energies scaled by %.2f" % light_scale)

world = bpy.data.worlds.new("World")
scene.world = world
world.use_nodes = True
world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.102, 0.102, 0.18, 1.0)

scene.render.resolution_x = width
scene.render.resolution_y = height
scene.render.fps = fps
scene.render.image_settings.file_format = "PNG"
scene.render.filepath = os.path.join(output_dir, "frame_")
# Cycles on the GPU when one is present, on the CPU otherwise. This prints
# the engine and device it settled on -- a silent CPU or EEVEE fallback looks
# identical to success apart from that line.
engine = configure_render_quality(scene, samples=samples)

# The camera fit above still samples the whole animation, so a preview frames
# the same way the full render will -- it just stops rendering early.
if preview_frames > 0:
    scene.frame_end = min(scene.frame_end,
                          scene.frame_start + preview_frames - 1)

n = scene.frame_end - scene.frame_start + 1
print("Camera lens %.1fmm at %.2fm, yaw %+.0f deg; %s; rendering %d frames to %s"
      % (lens, dist, yaw_deg, engine, n, output_dir))
bpy.ops.render.render(animation=True)
print("Render complete")
'''


# Overlay text, matching the annotated segments' style: the same corner, the
# same two sizes, the same accent blue for the title. This renderer produces
# segments that have no data overlay of their own (the IIWA ones), and an
# unlabelled shot of two arms moving does not tell the viewer what it is.
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
TITLE_COLOR = "0x4fc3f7"
SUBTITLE_COLOR = "0xd0d0d0"


def _escape(text):
    """Escape a literal string for ffmpeg's drawtext= option."""
    # Backslash first, then the characters drawtext and the filter-graph parser
    # each treat specially.
    for ch in ("\\", ":", "'", "%", ",", "[", "]", ";"):
        text = text.replace(ch, "\\" + ch)
    return text


SPEED_COLOR = "0xc8c8c8"


def overlay_filter(title, subtitle, speed=1.0):
    """Build the drawtext filter chain, or "" when there is nothing to draw."""
    parts = []
    if title:
        parts.append(
            f"drawtext=fontfile={FONT_PATH}:text='{_escape(title)}'"
            f":fontsize=40:fontcolor={TITLE_COLOR}:x=42:y=42")
    if subtitle:
        parts.append(
            f"drawtext=fontfile={FONT_PATH}:text='{_escape(subtitle)}'"
            f":fontsize=28:fontcolor={SUBTITLE_COLOR}:x=42:y=98")
    # A retimed segment says so on screen, the way the stability and montage
    # segments do. Silently slowing a trajectory down misrepresents how fast
    # the plan actually runs, which is a number this project quotes elsewhere.
    if abs(speed - 1.0) > 1e-9:
        label = f"{speed:g}× real time"
        parts.append(
            f"drawtext=fontfile={FONT_PATH}:text='{_escape(label)}'"
            f":fontsize=26:fontcolor={SPEED_COLOR}:x=42:y=146")
    return ",".join(parts)


def render_fps_for(speed, fps):
    """The rate to sample the recording at so ``speed`` costs no smoothness.

    A segment played at 0.5x has to show twice as many frames per second of
    plan. Retiming in the encode (``setpts``, then ``fps``) got that count by
    repeating each rendered frame, which halves the *effective* frame rate: the
    segment then stutters against every other segment in the cut. Sampling the
    recording at ``fps / speed`` instead gives real, distinct in-between poses,
    and encoding those at ``fps`` plays them out at the requested rate.

    This works because the recording is denser than the render. Drake writes
    meshcat animations at 64 fps, so a 30 fps render at 0.5x asks for 60 -- the
    importer interpolates down to it, as it already does for 30. Above the
    recording rate there is nothing left to interpolate from and the importer
    hands back its own keyframes, so the check below refuses rather than
    quietly reintroducing duplicated frames.
    """
    exact = fps / speed
    rounded = int(round(exact))
    if abs(exact - rounded) > 1e-6:
        raise SystemExit(
            f"--speed {speed:g} needs a render rate of {exact:g} fps, which is "
            f"not a whole number of frames per second; pick a speed that "
            f"divides {fps}")
    return rounded


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("html_path", help="Path to meshcat HTML file")
    ap.add_argument("--output", required=True, help="Output MP4 path")
    ap.add_argument("--orbit", action="store_true", help="Camera orbits around scene")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--yaw", type=float, default=0.0,
                    help="degrees to rotate the camera about the world Z axis "
                         "from the default 3/4 view; positive swings it to the "
                         "viewer's left")
    ap.add_argument("--samples", type=int, default=128,
                    help="Cycles sample ceiling (adaptive sampling stops "
                         "earlier where the image has converged)")
    ap.add_argument("--preview-frames", type=int, default=0, metavar="N",
                    help="render only the first N frames, for checking the "
                         "camera and the look cheaply (0 = the whole segment)")
    ap.add_argument("--title", default=None,
                    help="segment title drawn in the top-left corner")
    ap.add_argument("--subtitle", default=None,
                    help="one line of explanatory text under the title")
    ap.add_argument("--speed", type=float, default=1.0,
                    help="playback rate relative to the planned trajectory; "
                         "0.5 plays it at half speed by rendering at twice the "
                         "output frame rate, so the segment keeps --fps. "
                         "Anything other than 1 is labelled on screen")
    # The table and shelves are restyled to match the IIWA experiment's
    # swept-volume figure, which uses two BlenderKit assets read from the user's
    # own cache. They are absent from a clone, so each falls back to a
    # procedural material of the same character; pass "procedural" to ask for
    # that deliberately.
    ap.add_argument("--table-material", default="old-metal",
                    help="BlenderKit asset slug for the table, or 'procedural'")
    ap.add_argument("--shelf-material", default="old-plywood",
                    help="BlenderKit asset slug for the shelves, or 'procedural'")
    ap.add_argument("--blenderkit-dir", default="~/blenderkit_data/materials",
                    help="BlenderKit material cache to append assets from")
    ap.add_argument("--light-scale", type=float, default=1.0, metavar="K",
                    help="multiply every lamp's energy by K; below 1 darkens the "
                         "lit surfaces. Per-scene correction for the lights "
                         "sitting at fixed offsets while the camera distance "
                         "scales with the subject. Leaves the world background "
                         "alone, which an exposure adjustment would not")
    ap.add_argument("--table-brightness", type=float, default=1.0, metavar="K",
                    help="multiply the table material's base colour by K. The "
                         "asset is a photograph with both dark and pale regions "
                         "and its UVs are world-space, so which tone a scene "
                         "gets depends on where its table sits; this pins it "
                         "down without touching the lights, and so without "
                         "touching the arms")
    ap.add_argument("--table-texel-scale", type=float, default=1.0)
    ap.add_argument("--shelf-texel-scale", type=float, default=1.0)
    ap.add_argument("--timeout", type=int, default=7200,
                    help="seconds to allow the Blender subprocess")
    args = ap.parse_args()

    if not os.path.exists(args.html_path):
        print(f"Error: {args.html_path} not found")
        sys.exit(1)

    if not os.path.exists(BLENDER):
        print(f"Error: Blender not found at {BLENDER}")
        sys.exit(1)

    # Sample the recording finely enough that --speed costs frames, not
    # smoothness; the encode below still writes --fps.
    render_fps = render_fps_for(args.speed, args.fps)

    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = os.path.join(tmpdir, "render_script.py")
        with open(script_path, "w") as f:
            # The template runs inside Blender, where this repo is not on
            # sys.path; bind the directory holding blender_render_common.py.
            f.write("SCRIPTS_DIR = %r\n" % os.path.dirname(os.path.abspath(__file__)))
            # Material choices, bound as constants rather than appended to the
            # positional argv the template already parses by index.
            f.write("table_material = %r\n" % args.table_material)
            f.write("shelf_material = %r\n" % args.shelf_material)
            f.write("blenderkit_dir = %r\n" % args.blenderkit_dir)
            f.write("table_texels = %r\n" % args.table_texel_scale)
            f.write("shelf_texels = %r\n" % args.shelf_texel_scale)
            f.write("light_scale = %r\n" % args.light_scale)
            f.write("table_brightness = %r\n" % args.table_brightness)
            f.write(RENDER_SCRIPT_TEMPLATE)

        frames_dir = os.path.join(tmpdir, "frames")
        os.makedirs(frames_dir)

        html_abs = os.path.abspath(args.html_path)

        cmd = [
            BLENDER, "--background", "--python", script_path,
            "--", html_abs, frames_dir, str(render_fps),
            "true" if args.orbit else "false",
            str(args.width), str(args.height),
            str(args.yaw), str(args.samples), str(args.preview_frames),
        ]

        print(f"Running Blender render...")
        # 600 s was less than a 600-frame EEVEE render takes, so a good run could
        # die on the timeout after doing all the work.
        result = subprocess.run(cmd, capture_output=True, text=True,
                                timeout=args.timeout)

        if result.returncode != 0:
            print(f"Blender failed (rc={result.returncode}):")
            print(result.stderr[-1000:] if result.stderr else "no stderr")
            print(result.stdout[-1000:] if result.stdout else "no stdout")
            sys.exit(1)

        # Echo the engine/device and camera lines. Which engine and device the
        # render actually used is the difference between a clean GPU render and
        # a silent CPU or EEVEE fallback, and is invisible from the exit status.
        for line in (result.stdout or "").splitlines():
            if (line.startswith("[render]") or line.startswith("[material]")
                    or line.startswith("Camera lens")):
                print("  " + line)

        # Check if frames were produced
        frames = sorted(f for f in os.listdir(frames_dir) if f.endswith(".png"))
        if not frames:
            print("No frames rendered. Blender output:")
            print(result.stdout[-2000:])
            sys.exit(1)

        print(f"  {len(frames)} frames rendered")

        # Encode to MP4, drawing the title/subtitle overlay in the same pass.
        # The frames were rendered at render_fps; playing them out at args.fps
        # is what makes the segment run at args.speed. No setpts, so every
        # output frame is a distinct render.
        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-framerate", str(args.fps),
            "-i", os.path.join(frames_dir, "frame_%04d.png"),
        ]
        vf = overlay_filter(args.title, args.subtitle, args.speed)
        if vf:
            ffmpeg_cmd += ["-vf", vf]
        ffmpeg_cmd += [
            "-c:v", "libx264", "-crf", "20", "-preset", "medium",
            "-pix_fmt", "yuv420p", "-an",
            args.output,
        ]
        subprocess.run(ffmpeg_cmd, capture_output=True, text=True, check=True)

        dur = float(subprocess.check_output([
            "ffprobe", "-v", "quiet", "-show_entries", "format=duration",
            "-of", "csv=p=0", args.output,
        ]).decode().strip())
        print(f"Wrote {args.output} ({dur:.1f}s)")


if __name__ == "__main__":
    main()
