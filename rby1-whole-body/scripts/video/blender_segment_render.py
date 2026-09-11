"""Render one meshcat HTML recording to PNG frames with Blender.  Runs *inside* Blender.

The generic version of ``blender_render_grid.py``'s embedded render script: same
importer, same collision-geometry deletion, same material tuning and studio
lighting, but the camera/resolution come from a JSON config so one file can serve
segments that need different framing.

It is invoked by the driver in ``blender_segment_driver.py``:

    blender --background --python blender_segment_render.py -- <config.json>

and the config is::

    {
      "html": "...", "out_dir": "...",
      "width": 1920, "height": 1080, "fps": 30, "samples": 32,
      "camera_matrix_world": [[...4x4...]], "lens_mm": 23.95,
      "sensor_width_mm": 36.0,
      "keep_prefixes": ["grid_viz_"]        # optional, informational
    }

On success it writes ``frame_%04d.png`` for every animation frame plus
``camera.json`` -- the camera it actually rendered with, read back from Blender
after the scene is set up, so the overlay pass projects with exactly this camera
instead of a recomputed guess.  It prints ``SEGMENT_RENDER_DONE`` last; the driver
checks for that rather than trusting the exit status.
"""

import glob
import json
import math
import os
import sys

sys.path.insert(0, os.path.expanduser('~/.config/blender/5.0/extensions/user_default'))
# This file runs inside Blender, which does not put its own directory on
# sys.path, so the shared helpers next to it need an explicit entry.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bpy  # noqa: E402  (must follow the sys.path insert)
import mathutils  # noqa: E402
from blender_render_common import configure_render_quality  # noqa: E402
from meshcat_html_importer.blender_impl.scene_builder import (  # noqa: E402
    build_scene_from_file,
)


COLLISION_PATH_PREFIXES = ("/drake/collision/", "/drake/proximity/",
                           "/drake/inertia/", "/drake/contact_forces/")


def delete_collision_geometry(created_objects):
    """Remove collision meshes outright, selected by meshcat *path*.

    Two reasons this is path-based rather than name-based:

    * ``hide_render`` is unreliable across Blender/EEVEE versions, so the geometry
      has to go rather than be hidden -- the same conclusion
      ``blender_render_grid.py`` reached.
    * the importer derives object *names* from paths and collapses the leaf when it
      is called ``visual``, so a proximity mesh can land in the scene named plain
      ``collision`` and slip past a ``collision_*`` name filter.  The path the
      importer hands back is unambiguous.

    A scene built by ``make_default_rby1_infrastructure`` carries a second
    MeshcatVisualizer at prefix ``collision``; even when only the ``visual`` one is
    published, its geometry has been observed in the export, so this always runs.
    """
    doomed, kept = [], 0
    for path, obj in created_objects.items():
        if path.startswith(COLLISION_PATH_PREFIXES):
            doomed.append(obj)
        else:
            kept += 1
    for obj in doomed:
        try:
            bpy.data.objects.remove(obj, do_unlink=True)
        except ReferenceError:
            pass

    # Backstop for anything that reached the scene outside created_objects.
    stragglers = [o for o in bpy.data.objects
                  if o.type == "MESH" and o.name.startswith("collision")]
    for obj in stragglers:
        bpy.data.objects.remove(obj, do_unlink=True)
    return len(doomed), len(stragglers), kept


def tune_materials():
    """Smooth shading plus the brightness-aware metallic/roughness split.

    Meshcat strips OBJ material files, so meshes arrive carrying flat per-face
    colours; this is the same compensation the grid renders use, kept identical so
    converted segments match the RBY1 sim renders already in the video.
    """
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        for poly in obj.data.polygons:
            poly.use_smooth = True
        if not obj.data.materials:
            continue
        for mat in obj.data.materials:
            if not mat or not mat.use_nodes:
                continue
            bsdf = mat.node_tree.nodes.get("Principled BSDF")
            if not bsdf:
                continue
            color = bsdf.inputs["Base Color"].default_value
            brightness = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
            if brightness > 0.5:
                bsdf.inputs["Metallic"].default_value = 0.3
                bsdf.inputs["Roughness"].default_value = 0.35
            else:
                bsdf.inputs["Metallic"].default_value = 0.05
                bsdf.inputs["Roughness"].default_value = 0.6


def setup_camera(matrix_world, lens_mm, sensor_width_mm):
    """Place the camera from an explicit matrix -- no TRACK_TO constraint.

    A constraint would mean the matrix the overlay pass projects with depends on
    when the dependency graph was evaluated; setting ``matrix_world`` directly
    makes the render camera and the projection camera the same object.
    """
    cam = bpy.data.cameras.new("SegmentCam")
    cam.lens = lens_mm
    cam.sensor_fit = "HORIZONTAL"
    cam.sensor_width = sensor_width_mm
    cam.shift_x = 0.0
    cam.shift_y = 0.0
    cam.clip_start = 0.05
    cam.clip_end = 100.0
    obj = bpy.data.objects.new("SegmentCam", cam)
    bpy.context.scene.collection.objects.link(obj)
    bpy.context.scene.camera = obj
    obj.matrix_world = mathutils.Matrix([list(r) for r in matrix_world])
    return obj


def setup_lighting():
    """Three-point studio lighting, matching the grid renders."""
    key = bpy.data.lights.new("Key", "SUN")
    key.energy = 4
    key.color = (1.0, 0.98, 0.95)
    ko = bpy.data.objects.new("Key", key)
    ko.rotation_euler = (math.radians(50), math.radians(10), math.radians(-30))
    bpy.context.scene.collection.objects.link(ko)

    fill = bpy.data.lights.new("Fill", "AREA")
    fill.energy = 150
    fill.size = 4
    fo = bpy.data.objects.new("Fill", fill)
    fo.location = (-2, 2, 2.5)
    bpy.context.scene.collection.objects.link(fo)

    rim = bpy.data.lights.new("Rim", "SPOT")
    rim.energy = 200
    rim.spot_size = math.radians(60)
    ro = bpy.data.objects.new("Rim", rim)
    ro.location = (-1.5, -2, 3)
    ro.rotation_euler = (math.radians(35), 0, math.radians(-150))
    bpy.context.scene.collection.objects.link(ro)


def setup_world():
    w = bpy.data.worlds.new("W")
    bpy.context.scene.world = w
    w.use_nodes = True
    w.node_tree.nodes["Background"].inputs["Color"].default_value = (
        0.102, 0.102, 0.18, 1.0)


def setup_render(width, height, fps, samples):
    s = bpy.context.scene
    s.render.resolution_x = width
    s.render.resolution_y = height
    s.render.resolution_percentage = 100
    s.render.pixel_aspect_x = 1.0
    s.render.pixel_aspect_y = 1.0
    s.render.fps = fps
    # Cycles on the GPU when one is present, EEVEE otherwise; prints which.
    configure_render_quality(s, samples=samples)
    s.render.image_settings.file_format = "PNG"


def main():
    cfg_path = sys.argv[sys.argv.index("--") + 1:][0]
    with open(cfg_path) as f:
        cfg = json.load(f)

    out_dir = cfg["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    created = build_scene_from_file(cfg["html"], target_fps=cfg.get("fps", 30),
                                    clear_scene=True)

    n_by_path, n_stragglers, n_kept = delete_collision_geometry(created)
    tune_materials()

    n_mesh = len([o for o in bpy.data.objects if o.type == "MESH"])
    print(f"SEGMENT_RENDER removed {n_by_path} collision objects by path "
          f"(+{n_stragglers} by name), kept {n_kept} visual paths, "
          f"{n_mesh} meshes remain")

    cam_obj = setup_camera(cfg["camera_matrix_world"], cfg["lens_mm"],
                           cfg.get("sensor_width_mm", 36.0))
    setup_lighting()
    setup_world()
    setup_render(cfg["width"], cfg["height"], cfg.get("fps", 30),
                 cfg.get("samples", 32))

    s = bpy.context.scene
    if cfg.get("frame_end") is not None:
        s.frame_end = min(s.frame_end, int(cfg["frame_end"]))
    f_start, f_end = s.frame_start, s.frame_end
    print(f"SEGMENT_RENDER frames {f_start}-{f_end} at "
          f"{cfg['width']}x{cfg['height']}")

    # Read the camera back *after* setup so the overlay pass projects with the
    # camera that rendered, not with a recomputed one.
    bpy.context.view_layer.update()
    M = cam_obj.matrix_world
    with open(os.path.join(out_dir, "camera.json"), "w") as f:
        json.dump(dict(
            matrix_world=[list(M[r]) for r in range(4)],
            lens_mm=cam_obj.data.lens,
            sensor_width_mm=cam_obj.data.sensor_width,
            sensor_fit=cam_obj.data.sensor_fit,
            width=s.render.resolution_x,
            height=s.render.resolution_y,
            frame_start=f_start,
            frame_end=f_end,
        ), f, indent=2)

    for f in glob.glob(os.path.join(out_dir, "frame_*.png")):
        os.remove(f)
    s.render.filepath = os.path.join(out_dir, "frame_")

    bpy.ops.render.render(animation=True)
    print("SEGMENT_RENDER_DONE")


if __name__ == "__main__":
    main()
