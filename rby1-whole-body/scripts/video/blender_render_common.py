"""Shared Blender rendering setup for all video segments.

Called from within Blender's Python. Adds smooth shading, improves materials,
sets up studio lighting and camera.
"""

import addon_utils
import bpy
import math
import os


BG_COLOR = (0.102, 0.102, 0.18, 1.0)  # #1a1a2e


def _allow_cpu_render():
    """Escape hatch for rendering on the CPU on a machine that has a GPU."""
    return os.environ.get("BLENDER_ALLOW_CPU_RENDER", "").strip() not in ("", "0")


def improve_scene_quality():
    """Apply smooth shading and improve materials on all imported meshcat objects."""
    for obj in bpy.data.objects:
        if obj.type != 'MESH':
            continue

        # Smooth shading
        for poly in obj.data.polygons:
            poly.use_smooth = True

        # Auto-smooth (API varies by Blender version)
        try:
            if hasattr(obj.data, 'use_auto_smooth'):
                obj.data.use_auto_smooth = True
                obj.data.auto_smooth_angle = math.radians(30)
        except Exception:
            pass

        # Improve materials — add slight metallic/roughness
        if obj.data.materials:
            for mat in obj.data.materials:
                if mat and mat.use_nodes:
                    bsdf = mat.node_tree.nodes.get("Principled BSDF")
                    if bsdf:
                        # Slightly metallic for robot parts
                        color = bsdf.inputs['Base Color'].default_value
                        brightness = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
                        if brightness > 0.6:
                            bsdf.inputs['Metallic'].default_value = 0.3
                            bsdf.inputs['Roughness'].default_value = 0.4
                        else:
                            bsdf.inputs['Metallic'].default_value = 0.1
                            bsdf.inputs['Roughness'].default_value = 0.5


def setup_studio_lighting():
    """Three-point lighting setup for clean robot renders."""
    # Key light — warm sun
    key = bpy.data.lights.new('Key', 'SUN')
    key.energy = 4.0
    key.color = (1.0, 0.98, 0.95)
    ko = bpy.data.objects.new('Key', key)
    ko.rotation_euler = (math.radians(50), math.radians(10), math.radians(-30))
    bpy.context.scene.collection.objects.link(ko)

    # Fill light — soft area
    fill = bpy.data.lights.new('Fill', 'AREA')
    fill.energy = 150
    fill.size = 4.0
    fill.color = (0.9, 0.95, 1.0)
    fo = bpy.data.objects.new('Fill', fill)
    fo.location = (-2, 2, 2.5)
    fo.rotation_euler = (math.radians(45), 0, math.radians(120))
    bpy.context.scene.collection.objects.link(fo)

    # Rim/back light
    rim = bpy.data.lights.new('Rim', 'SPOT')
    rim.energy = 300
    rim.spot_size = math.radians(60)
    rim.color = (0.85, 0.9, 1.0)
    ro = bpy.data.objects.new('Rim', rim)
    ro.location = (-1, -2, 3)
    ro.rotation_euler = (math.radians(35), 0, math.radians(-150))
    bpy.context.scene.collection.objects.link(ro)


def setup_world():
    """Dark background matching the video style."""
    world = bpy.data.worlds.new('World')
    bpy.context.scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes['Background']
    bg.inputs['Color'].default_value = BG_COLOR
    bg.inputs['Strength'].default_value = 0.3


def setup_camera(location, target, lens=30):
    """Create camera with track-to constraint."""
    cam = bpy.data.cameras.new('Camera')
    cam.lens = lens
    cam.clip_end = 100
    co = bpy.data.objects.new('Camera', cam)
    bpy.context.scene.collection.objects.link(co)
    bpy.context.scene.camera = co
    co.location = location

    t = bpy.data.objects.new('CamTarget', None)
    t.location = target
    bpy.context.scene.collection.objects.link(t)

    tc = co.constraints.new(type='TRACK_TO')
    tc.target = t
    tc.track_axis = 'TRACK_NEGATIVE_Z'
    tc.up_axis = 'UP_Y'
    return co


def enable_cycles_gpu(prefer=("OPTIX", "CUDA")):
    """Enable the Cycles add-on and select an NVIDIA compute backend.

    Returns the backend name that was selected, or None if no GPU is usable.

    Cycles is not enabled under `--factory-startup`, so `addon_utils.enable`
    has to run before `scene.render.engine = 'CYCLES'` is even a valid
    assignment.  `get_devices()` must be called after setting
    `compute_device_type` or the device list stays empty and every device
    silently ends up with `use = False`, which is a full CPU fallback that
    otherwise looks exactly like a successful GPU render.
    """
    try:
        addon_utils.enable("cycles", default_set=True, persistent=True)
    except Exception as e:
        print(f"[render] could not enable the Cycles add-on: {e}")
        return None

    addon = bpy.context.preferences.addons.get("cycles")
    if addon is None:
        return None
    cprefs = addon.preferences

    for backend in prefer:
        try:
            cprefs.compute_device_type = backend
        except TypeError:
            continue  # this build has no such backend
        try:
            cprefs.get_devices()
        except Exception:
            pass
        if not any(d.type == backend for d in cprefs.devices):
            continue
        # Enable only the accelerator. Leaving the CPU on as well makes Cycles
        # split tiles between devices, which is slower here than the GPU alone.
        for d in cprefs.devices:
            d.use = (d.type == backend)
        return backend
    return None


def _accelerator_present(cprefs=None):
    """True when this machine has a GPU Cycles ought to be able to use.

    Two independent signals, because either one alone has a blind spot: the
    kernel's NVIDIA node proves a card is installed even when Cycles has failed
    to enumerate it (a driver/library mismatch, which is exactly the case that
    used to degrade quietly), and the device list catches a non-NVIDIA
    accelerator on a machine with no such node.
    """
    if os.path.exists("/proc/driver/nvidia/version"):
        return True
    if cprefs is None:
        addon = bpy.context.preferences.addons.get("cycles")
        cprefs = addon.preferences if addon else None
    if cprefs is None:
        return False
    try:
        return any(d.type != "CPU" for d in cprefs.devices)
    except Exception:
        return False


def configure_render_quality(scene=None, samples=128, prefer_gpu=True,
                             denoise=True):
    """Put the scene on Cycles/GPU if possible, else Cycles/CPU, else EEVEE.

    Cycles with OptiX denoising is what removes the graininess the EEVEE
    renders had; on this workstation's RTX 3080 Ti it is also faster than the
    EEVEE path it replaces.

    A machine without an NVIDIA card still renders, on the CPU -- slower, but
    with the same renderer and so the same look. EEVEE is only the last resort,
    for a Blender build where Cycles cannot be enabled at all.

    Always prints the engine and device actually selected. A silent fall back
    to CPU or to EEVEE changes both how long the render takes and what it
    looks like, and is otherwise indistinguishable from success -- so on a
    machine that *has* an accelerator, failing to select it raises instead of
    falling back (``BLENDER_ALLOW_CPU_RENDER=1`` overrides).
    """
    s = scene or bpy.context.scene
    backend = enable_cycles_gpu() if prefer_gpu else None

    if backend is not None:
        s.render.engine = 'CYCLES'
        s.cycles.device = 'GPU'
        s.cycles.samples = samples
        # Adaptive sampling stops early wherever the estimate has converged, so
        # the sample count above is a ceiling rather than a flat cost.
        try:
            s.cycles.use_adaptive_sampling = True
            s.cycles.adaptive_threshold = 0.01
        except AttributeError:
            pass
        if denoise:
            try:
                s.cycles.use_denoising = True
                s.cycles.denoiser = ('OPTIX' if backend == 'OPTIX'
                                     else 'OPENIMAGEDENOISE')
            except (AttributeError, TypeError):
                pass
        print(f"[render] CYCLES on GPU via {backend}, {samples} samples, "
              f"denoise={denoise}")
        return f"CYCLES/{backend}"

    # A machine with a GPU that nonetheless failed device selection is a broken
    # setup, not a slow one: a segment that takes minutes on the RTX 3080 Ti
    # takes an hour on the CPU, and the printed "[render]" line is one line in a
    # build log thousands long, so the degradation has gone unnoticed for a
    # whole build before. Fail loudly instead, and let a deliberate CPU run say
    # so with BLENDER_ALLOW_CPU_RENDER=1.
    if prefer_gpu and _accelerator_present() and not _allow_cpu_render():
        raise RuntimeError(
            "Cycles found no usable GPU backend, but this machine has one "
            "(OPTIX/CUDA not selectable -- check the NVIDIA driver and that "
            "Blender's Cycles add-on can enumerate devices). Refusing to fall "
            "back to a CPU render; set BLENDER_ALLOW_CPU_RENDER=1 to override.")

    # No usable GPU. Cycles on the CPU is the first fallback, not EEVEE: it is
    # slower, but it is the *same* renderer, so a machine without an NVIDIA card
    # still produces frames that match the ones rendered here. Falling straight
    # to EEVEE would change the look of the segment as well as the speed.
    try:
        addon_utils.enable("cycles", default_set=True, persistent=True)
        s.render.engine = 'CYCLES'
        s.cycles.device = 'CPU'
        s.cycles.samples = samples
        try:
            s.cycles.use_adaptive_sampling = True
            s.cycles.adaptive_threshold = 0.01
        except AttributeError:
            pass
        if denoise:
            try:
                s.cycles.use_denoising = True
                s.cycles.denoiser = 'OPENIMAGEDENOISE'
            except (AttributeError, TypeError):
                pass
        why = "prefer_gpu=False" if not prefer_gpu else "no NVIDIA GPU backend"
        print(f"[render] CYCLES on the CPU ({why}), {samples} samples, "
              f"denoise={denoise}")
        return "CYCLES/CPU"
    except Exception as e:
        print(f"[render] Cycles unavailable on the CPU too ({e})")

    # Last resort: the rasteriser. Give it more TAA samples than the old default
    # so the result is at least less noisy than before.
    try:
        s.render.engine = 'BLENDER_EEVEE_NEXT'
    except Exception:
        s.render.engine = 'BLENDER_EEVEE'
    try:
        s.eevee.use_gtao = True
        s.eevee.gtao_distance = 1.0
        s.eevee.use_ssr = True
    except AttributeError:
        pass
    try:
        s.eevee.taa_render_samples = max(samples, 64)
    except AttributeError:
        pass
    print(f"[render] neither GPU nor CPU Cycles is available - falling back to "
          f"{s.render.engine}")
    return s.render.engine


def setup_render(width=1920, height=1080, fps=30, samples=128,
                 prefer_gpu=True):
    """Configure resolution, frame rate and the render engine."""
    s = bpy.context.scene
    s.render.resolution_x = width
    s.render.resolution_y = height
    s.render.fps = fps
    configure_render_quality(s, samples=samples, prefer_gpu=prefer_gpu)
    s.render.image_settings.file_format = 'PNG'
