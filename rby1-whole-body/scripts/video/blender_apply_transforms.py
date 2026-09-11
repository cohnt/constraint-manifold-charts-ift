"""Blender script: import RB-Y1 OBJ meshes and apply pre-computed X_WG transforms.

Run inside Blender:
    BLENDER --background --python scripts/video/blender_apply_transforms.py -- \\
        --transforms video/transforms_point00.pkl --output-dir video/scratch_blender_direct/point_00
"""

import sys
import os
import math
import pickle
import glob

import bpy
import mathutils

argv = sys.argv
if "--" in argv:
    argv = argv[argv.index("--") + 1:]
else:
    argv = []

import argparse
import plan_format  # noqa: F401  -- installs the pre-rename module aliases the
                    # committed plan and execution pickles were written with
ap = argparse.ArgumentParser()
ap.add_argument("--transforms", required=True)
ap.add_argument("--output-dir", required=True)
ap.add_argument("--width", type=int, default=960)
ap.add_argument("--height", type=int, default=540)
ap.add_argument("--test-frame", type=int, default=None,
                help="Render only this frame (for quality testing)")
args = ap.parse_args(argv)

print(f"Loading transforms from {args.transforms}...")
with open(args.transforms, "rb") as f:
    data = pickle.load(f)

vis_geos = data["visual_geometries"]
frames = data["frames"]
fps = data["fps"]

print(f"  {len(vis_geos)} visual geometries, {len(frames)} frames")

# Clear scene
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete()
for m in bpy.data.meshes:
    bpy.data.meshes.remove(m)
for m in bpy.data.materials:
    bpy.data.materials.remove(m)

# Import each visual geometry's OBJ mesh
blender_objects = []
mesh_cache = {}

for i, vg in enumerate(vis_geos):
    mesh_path = vg.get("mesh_path")
    if not mesh_path or not os.path.exists(mesh_path):
        blender_objects.append(None)
        continue

    if mesh_path not in mesh_cache:
        bpy.ops.wm.obj_import(filepath=mesh_path)
        obj = bpy.context.selected_objects[-1]
        # Bake the import transform into mesh data so the object is at identity
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        for poly in obj.data.polygons:
            poly.use_smooth = True
        mesh_cache[mesh_path] = obj.data
        obj.name = f"{vg['body_name']}_{i}"
        blender_objects.append(obj)
    else:
        obj = bpy.data.objects.new(f"{vg['body_name']}_{i}", mesh_cache[mesh_path])
        bpy.context.scene.collection.objects.link(obj)
        blender_objects.append(obj)

n_imported = sum(1 for o in blender_objects if o is not None)
print(f"Imported {n_imported} mesh objects")

# Improve materials
for obj in blender_objects:
    if obj is None:
        continue
    if obj.data.materials:
        for mat in obj.data.materials:
            if mat and mat.use_nodes:
                bsdf = mat.node_tree.nodes.get("Principled BSDF")
                if bsdf:
                    bsdf.inputs['Metallic'].default_value = 0.2
                    bsdf.inputs['Roughness'].default_value = 0.4

# Apply X_WG transforms as keyframes
bpy.context.scene.frame_start = 0
bpy.context.scene.frame_end = len(frames) - 1

if args.test_frame is not None:
    frame_range = [args.test_frame]
    print(f"Test mode: rendering only frame {args.test_frame}")
else:
    frame_range = range(len(frames))
    print(f"Applying {len(frames)} frames of keyframes...")

for frame_idx in frame_range:
    frame_data = frames[frame_idx]
    for geo_idx, obj in enumerate(blender_objects):
        if obj is None:
            continue
        T = frame_data[geo_idx]["X_WB"]
        mat4 = mathutils.Matrix([T[0], T[1], T[2], T[3]]).transposed()
        loc, rot, scale = mat4.decompose()
        obj.rotation_mode = 'QUATERNION'
        obj.location = loc
        obj.rotation_quaternion = rot
        obj.scale = scale
        obj.keyframe_insert(data_path="location", frame=frame_idx)
        obj.keyframe_insert(data_path="rotation_quaternion", frame=frame_idx)
        obj.keyframe_insert(data_path="scale", frame=frame_idx)

    if frame_idx % 200 == 0:
        print(f"  keyframe {frame_idx}/{len(frames)}")

print("Keyframes applied")

# Camera
cam = bpy.data.cameras.new('Camera')
cam.lens = 30
cam.clip_end = 100
co = bpy.data.objects.new('Camera', cam)
bpy.context.scene.collection.objects.link(co)
bpy.context.scene.camera = co
co.location = (2.0, -1.5, 1.0)

target = bpy.data.objects.new('CamTarget', None)
target.location = (0.0, 0.0, 0.8)
bpy.context.scene.collection.objects.link(target)
tc = co.constraints.new(type='TRACK_TO')
tc.target = target
tc.track_axis = 'TRACK_NEGATIVE_Z'
tc.up_axis = 'UP_Y'

# Studio lighting
key = bpy.data.lights.new('Key', 'SUN')
key.energy = 4.0
key.color = (1.0, 0.98, 0.95)
ko = bpy.data.objects.new('Key', key)
ko.rotation_euler = (math.radians(50), math.radians(10), math.radians(-30))
bpy.context.scene.collection.objects.link(ko)

fill = bpy.data.lights.new('Fill', 'AREA')
fill.energy = 150
fill.size = 4.0
fo = bpy.data.objects.new('Fill', fill)
fo.location = (-2, 2, 2.5)
bpy.context.scene.collection.objects.link(fo)

# Ground plane
bpy.ops.mesh.primitive_plane_add(size=10, location=(0, 0, 0))
ground = bpy.context.active_object
ground.name = "Ground"
gmat = bpy.data.materials.new("GroundMat")
gmat.use_nodes = True
gmat.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.15, 0.15, 0.18, 1)
gmat.node_tree.nodes["Principled BSDF"].inputs["Roughness"].default_value = 0.8
ground.data.materials.append(gmat)

# World
world = bpy.data.worlds.new('World')
bpy.context.scene.world = world
world.use_nodes = True
world.node_tree.nodes['Background'].inputs['Color'].default_value = (0.102, 0.102, 0.18, 1)

# Render settings
s = bpy.context.scene
s.render.resolution_x = args.width
s.render.resolution_y = args.height
s.render.fps = fps
s.render.engine = 'BLENDER_EEVEE'
s.render.image_settings.file_format = 'PNG'

out_dir = os.path.abspath(args.output_dir)
os.makedirs(out_dir, exist_ok=True)

if args.test_frame is not None:
    s.frame_set(args.test_frame)
    s.render.filepath = os.path.join(out_dir, f"test_frame_{args.test_frame:04d}.png")
    bpy.ops.render.render(write_still=True)
    print(f"Test frame saved to {s.render.filepath}")
else:
    for f in glob.glob(os.path.join(out_dir, "frame_*.png")):
        os.remove(f)
    s.render.filepath = os.path.join(out_dir, "frame_")
    print(f"Rendering {len(frames)} frames to {out_dir}...")
    bpy.ops.render.render(animation=True)
    print("RENDER_DONE")
