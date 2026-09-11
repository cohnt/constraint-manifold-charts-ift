"""Chronophotography-style swept-volume still, rendered inside Blender.

Runs in Blender's own Python, never the repo's venv:

    blender --background --python scripts/figures/blender_swept_volume.py -- \
        --html notebooks/trajectory.html --out out/figures/swept_volume.png

Use scripts/figures/render_swept_volume.py, which finds Blender, checks the
add-on, and validates what came back.

The picture: N poses along one bimanual motion drawn in a single image, all at
the same opacity, spaced evenly along the path the arms travel.  Building it in
3-D rather than compositing 2-D frames means occlusion between the ghosts, the
shelves and the table is physically correct.
"""

import math
import os
import sys

import addon_utils
import bpy
from mathutils import Vector

# The add-on is a plain package once its extension directory is on sys.path;
# `bpy.ops.import_scene.meshcat_html` only exists once the extension has been
# *enabled* in preferences, which it generally has not been under --background.
# Importing the module works either way, so import it.  `sys.path` does not
# expand `~`.
EXTENSIONS = os.path.expanduser("~/.config/blender/5.0/extensions/user_default")

MOVING_MODELS = ("iiwa_left", "iiwa_right", "wsg_left", "wsg_right")
# Rendered, but not framed to.  The table slab is much wider than the motion and
# its legs run to the floor, so fitting the camera to it puts the subject in the
# middle of a lot of tabletop.
UNFRAMED_MODELS = ("table",)
COLLISION_PREFIX = "/drake/collision/"
VISUAL_PREFIX = "/drake/visual/"

BG_COLOR = (0.102, 0.102, 0.18, 1.0)  # #1a1a2e


# ── argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    import argparse
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--html", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-poses", type=int, default=6)
    ap.add_argument("--t-start", type=float, default=0.0)
    ap.add_argument("--t-end", type=float, default=1.0)
    ap.add_argument("--spacing", default="arclength", choices=["arclength", "time"])
    ap.add_argument("--alpha", type=float, default=0.45)
    ap.add_argument("--endpoint-alpha", type=float, default=None)
    ap.add_argument("--static-tol", type=float, default=1e-4)
    ap.add_argument("--mid-bias", type=float, default=0.35)
    ap.add_argument("--pose-nudge", type=float, nargs="*", default=None)
    ap.add_argument("--drop-poses", type=int, nargs="*", default=None)
    ap.add_argument("--table-material", default="old-metal")
    ap.add_argument("--shelf-material", default="old-plywood")
    ap.add_argument("--table-texel-scale", type=float, default=1.0)
    ap.add_argument("--shelf-texel-scale", type=float, default=1.0)
    ap.add_argument("--blenderkit-dir", default=BLENDERKIT_DIR)
    ap.add_argument("--resolution", type=int, nargs=2, default=[2800, 2400])
    ap.add_argument("--samples", type=int, default=128)
    ap.add_argument("--device", default="auto",
                    choices=["auto", "OPTIX", "CUDA", "HIP", "ONEAPI", "CPU"])
    ap.add_argument("--engine", default="CYCLES", choices=["CYCLES", "EEVEE"])
    ap.add_argument("--camera-azimuth", type=float, default=None)
    ap.add_argument("--camera-elevation", type=float, default=None)
    ap.add_argument("--camera-distance", type=float, default=None)
    ap.add_argument("--lens", type=float, default=50.0)
    ap.add_argument("--frame-margin", type=float, default=1.10)
    ap.add_argument("--opaque-bg", action="store_true")
    ap.add_argument("--world-strength", type=float, default=0.35)
    ap.add_argument("--ambient", type=float, default=0.45)
    ap.add_argument("--light-strength", type=float, default=1.0)
    ap.add_argument("--ghost-shadows", default="last", choices=["none", "last", "all"])
    return ap.parse_args(argv)


# ── import and classify ───────────────────────────────────────────────────────

def import_scene(html_path):
    """Import the Meshcat page and return {meshcat_path: object}."""
    if EXTENSIONS not in sys.path:
        sys.path.insert(0, EXTENSIONS)
    try:
        from meshcat_html_importer.blender_impl.scene_builder import build_scene_from_file
    except ImportError as e:
        raise SystemExit(
            f"Could not import the meshcat_html_importer add-on from {EXTENSIONS}: {e}\n"
            "Install it with: python3 scripts/figures/install_meshcat_importer.py") from e

    objs = build_scene_from_file(html_path, target_fps=30, clear_scene=True,
                                 hierarchical_collections=True)
    print(f"[import] {len(objs)} objects from {html_path}")
    return objs


def classify(objs):
    """Split the imported objects into (moving, static), deleting collision geometry.

    Classification is by meshcat path, never by object name: the importer emits
    names like `visual`, `visual.001`, `collision_Convex.012`, which are neither
    unique nor stable.

    The collision geometry is here at all because the notebook attaches a second
    MeshcatVisualizer with `role=Role.kProximity` and `prefix="collision"`.  It is
    invisible in the browser (`visible_by_default=False`) but StaticHtml serialises
    it anyway, and the add-on only auto-skips the *default* Drake proximity prefix
    (`/drake/proximity/`), which the notebook renamed out from under it.  Delete
    rather than hide: `hide_render` has proven unreliable across engine versions.
    """
    moving, static, framed_static, by_path, removed = [], [], [], {}, 0
    for path, obj in objs.items():
        if path.startswith(COLLISION_PREFIX):
            bpy.data.objects.remove(obj, do_unlink=True)
            removed += 1
        elif path.startswith(VISUAL_PREFIX):
            model = path[len(VISUAL_PREFIX):].split("/", 1)[0]
            if model in MOVING_MODELS:
                moving.append(obj)
            else:
                static.append(obj)
                by_path[path] = obj
                if model not in UNFRAMED_MODELS:
                    framed_static.append(obj)
        else:
            static.append(obj)
            framed_static.append(obj)

    print(f"[classify] {len(moving)} moving, {len(static)} static "
          f"({len(framed_static)} of them framed to), "
          f"{removed} collision objects deleted")
    if not moving:
        raise SystemExit(
            "No moving geometry found. Expected /drake/visual/{" +
            ",".join(MOVING_MODELS) + "}/... -- this HTML is a different scene.")
    if not static:
        raise SystemExit(
            "No static geometry found (shelves, table). This HTML is a different scene.")
    return moving, static, framed_static, by_path


# ── ghosting ──────────────────────────────────────────────────────────────────

def frame_window(t_start, t_end):
    scene = bpy.context.scene
    f0, f1 = scene.frame_start, scene.frame_end
    span = f1 - f0
    lo = int(round(f0 + t_start * span))
    hi = int(round(f0 + t_end * span))
    return f0, f1, list(range(lo, hi + 1))


def scan_motion(moving, window):
    """World-space bounding-box corners of every moving object at every frame.

    Corners rather than object origins: a link that spins about its own origin --
    `iiwa_link_1` on the shoulder, for one -- never translates, so a test on
    `matrix_world.translation` alone would call it static and draw it once.
    """
    scene = bpy.context.scene
    track = {}
    for frame in window:
        scene.frame_set(frame)
        # Without this the depsgraph has not re-evaluated and every frame reads
        # back the same matrices.
        bpy.context.view_layer.update()
        for obj in moving:
            m = obj.matrix_world
            track.setdefault(obj, []).append([m @ Vector(c) for c in obj.bound_box])
    return track


def split_static(moving, track, tol):
    """Separate the links that actually move from the ones welded in place.

    `iiwa_link_0` is bolted to the table, so ghosting it stacks `n_poses`
    coincident copies of the same mesh.  Identical surfaces at identical depths
    z-fight, and each layer of alpha multiplies into the next, so the bases came
    out both speckled and far more opaque than the rest of the arm.  Drawing
    them once, opaque, is what they are.
    """
    truly_moving, welded = [], []
    for obj in moving:
        corners = track[obj]
        first = corners[0]
        travel = max((c - f).length for frame in corners for c, f in zip(frame, first))
        (welded if travel <= tol else truly_moving).append(obj)
    print(f"[static] {len(truly_moving)} links move, {len(welded)} are welded in "
          f"place and drawn once")
    if not truly_moving:
        raise SystemExit("Nothing moves over the sampled window -- check --t-start/--t-end.")
    return truly_moving, welded


def bias_fractions(n, mid_bias):
    """`n` fractions of [0, 1], pushed together in the middle of the interval.

    Even spacing along the path still leaves the sweep looking crowded at both
    ends, because the two arms sit nearly on top of each other where the motion
    starts and where it finishes; it is only through the middle that the poses
    are visually distinct enough to be worth spending on.  So the fractions get
    warped by

        s(u) = u + (b / 2*pi) * sin(2*pi*u),   ds/du = 1 + b*cos(2*pi*u)

    which keeps the endpoints (s(0)=0, s(1)=1), stays monotone for b < 1, and
    stretches the steps near u=0 and u=1 by (1+b) while squeezing those near
    u=0.5 by (1-b).  `--mid-bias 0` is the plain even spacing.
    """
    if n == 1:
        return [0.0]
    out = []
    for i in range(n):
        u = i / (n - 1)
        out.append(u + (mid_bias / (2 * math.pi)) * math.sin(2 * math.pi * u))
    return out


# Tuned by eye against the azimuth-90 view, where the poses stack vertically.
# Pose 4 sat almost on top of the final pose -- their grippers are 0.703 m and
# 0.730 m up, 27 mm apart, while the gap below it to pose 3 is 0.21 m -- so it
# is pulled back down the path to even out the top of the stack: the three top
# grippers go 0.491 / 0.619 / 0.730 instead of 0.491 / 0.703 / 0.730.  Note the sign:
# the gripper rises monotonically along this trajectory, so *earlier* is *lower*.
#
# Settled, do not re-tune: at -0.10 pose 4 lands on frame 36, where the gripper
# spans z 0.578..0.659 and so straddles the middle shelf board at 0.593..0.607.
# It is genuinely collision-free -- the planner had it clear, and the overlap is
# only in height, not in space -- but it does mean there is no daylight over the
# board from the azimuth-90 camera.  We looked at the alternatives and kept
# this one.  For the record: frame 35 still straddles the board by 19 mm, and
# only frame 34 (-0.14) clears it, which moves the pose visibly too far.
DEFAULT_POSE_NUDGE = (0.0, 0.0, 0.0, 0.0, -0.10, 0.0)

# Pose 1's gripper sits 49 mm above pose 0's (0.149 m against 0.100 m), close
# enough that the two read as one clump at the bottom of the stack rather than as
# two configurations.  Dropped rather than moved: the poses either side of it are
# where they should be, and re-spacing five poses over the whole path would shift
# all of them.  Indices count from the start of the trajectory, before the drop.
DEFAULT_DROP_POSES = (1,)


def apply_nudge(fractions, args):
    """Per-pose adjustments to where along the path each pose is taken.

    In fractions of total arc length, added to the evenly-spaced-then-biased
    values.  This is how an individual pose gets moved without hand-placing it:
    it still comes off the trajectory, just from a different point on it.
    """
    n = len(fractions)
    nudge = args.pose_nudge
    if nudge is None:
        # The default is tuned for the default pose count and means nothing at
        # any other; silently reusing it would misplace a pose.
        nudge = DEFAULT_POSE_NUDGE if n == len(DEFAULT_POSE_NUDGE) else (0.0,) * n
    if len(nudge) != n:
        raise SystemExit(f"--pose-nudge takes {n} values, one per pose, got {len(nudge)}")
    if not any(nudge):
        return list(fractions)

    out = [min(1.0, max(0.0, f + d)) for f, d in zip(fractions, nudge)]
    for i in range(1, n):                     # keep the poses in trajectory order
        out[i] = max(out[i], out[i - 1])
    print(f"[poses] nudged by {list(nudge)} -> {[round(f, 3) for f in out]}")
    return out


def apply_drops(fractions, args):
    """Remove whole poses by index, leaving the rest exactly where they were."""
    n = len(fractions)
    drops = args.drop_poses
    if drops is None:
        drops = DEFAULT_DROP_POSES if n == len(DEFAULT_POSE_NUDGE) else ()
    drops = sorted(set(drops))
    if not drops:
        return list(fractions)
    for i in drops:
        if not 0 <= i < n:
            raise SystemExit(f"--drop-poses index {i} is outside 0..{n - 1}")
    if len(drops) >= n:
        raise SystemExit("--drop-poses would remove every pose")
    kept = [f for i, f in enumerate(fractions) if i not in drops]
    print(f"[poses] dropped {drops}, {len(kept)} of {n} poses kept")
    return kept


def sample_frames(window, track, truly_moving, args):
    """Pick `n_poses` frames spaced along the motion, weighted toward the middle.

    The recording is TOPPRA-retimed and then held for a ~1 s settling tail, so it
    creeps away from the start, races through the middle and sits still at the
    end.  Sampling that uniformly in time piles ghosts up at both ends and leaves
    a gap through the interesting part.  Walking the cumulative path length of the
    moving geometry instead spaces the poses the way the eye reads them, and the
    settling tail contributes no length, so it costs nothing to leave the clip
    untrimmed.  `--mid-bias` then biases where along that path the poses fall; see
    `bias_fractions`.
    """
    n = args.n_poses
    if n == 1:
        return [window[0]]
    fractions = apply_drops(apply_nudge(bias_fractions(n, args.mid_bias), args), args)

    if args.spacing == "time":
        frames = [window[int(round((len(window) - 1) * f))] for f in fractions]
        print(f"[poses] uniform in time, mid-bias {args.mid_bias}, frames {frames}")
        return frames

    cumulative = [0.0]
    for k in range(1, len(window)):
        step = sum((track[obj][k][j] - track[obj][k - 1][j]).length
                   for obj in truly_moving for j in range(8))
        cumulative.append(cumulative[-1] + step)
    total = cumulative[-1]
    if total <= 0.0:
        raise SystemExit("The sampled window has zero path length.")

    frames, k = [], 0
    for f in fractions:
        target = total * f
        while k + 1 < len(cumulative) and cumulative[k + 1] < target:
            k += 1
        # Nearest of the two frames bracketing the target arc length.
        j = k + 1 if (k + 1 < len(cumulative)
                      and abs(cumulative[k + 1] - target) < abs(cumulative[k] - target)) else k
        frames.append(window[j])
    print(f"[poses] arc length over frames {window[0]}-{window[-1]} (total {total:.2f}), "
          f"mid-bias {args.mid_bias}, frames {frames}")
    return frames


def ghost_alpha(index, n_poses, args):
    """Uniform across the sweep, so no pose reads as heavier than another.

    `--endpoint-alpha` opts back into solid first/last poses; it is off by
    default because a mixed ramp made the opacity look like an artifact rather
    than a cue.
    """
    if args.endpoint_alpha is not None and index in (0, n_poses - 1):
        return args.endpoint_alpha
    return args.alpha


def alpha_material(cache, source, alpha):
    """A copy of `source` with Principled Alpha set, one per (material, alpha)."""
    key = (source.name if source else None, round(alpha, 4))
    if key in cache:
        return cache[key]
    mat = source.copy() if source else bpy.data.materials.new("ghost")
    mat.name = f"{mat.name}_a{round(alpha * 100):03d}"
    if not mat.use_nodes:
        mat.use_nodes = True
    for node in mat.node_tree.nodes:
        if node.type == "BSDF_PRINCIPLED":
            node.inputs["Alpha"].default_value = alpha
            break
    cache[key] = mat
    return mat


def bake_ghosts(moving, frames, args):
    """One copy of every moving object at every sampled frame.

    The mesh datablock is shared -- the geometry is rigid, only the transform
    differs -- and the faded material is attached through an OBJECT-linked slot
    so the shared mesh data is untouched.
    """
    scene = bpy.context.scene
    cache = {}
    n = len(frames)
    ghosts = []
    for i, frame in enumerate(frames):
        scene.frame_set(frame)
        # Without this the depsgraph has not re-evaluated and every ghost is
        # baked at the same pose.
        bpy.context.view_layer.update()

        alpha = ghost_alpha(i, n, args)
        coll = bpy.data.collections.new(f"Sweep_{i:02d}")
        scene.collection.children.link(coll)

        # Six stacked semi-transparent poses each casting a shadow means every pose
        # is dimmed by the ones in front of it, which reads as uneven opacity rather
        # than as depth.  Only the final pose casts, so the figure stays anchored to
        # the table without the sweep shadowing itself.
        casts = (args.ghost_shadows == "all"
                 or (args.ghost_shadows == "last" and i == n - 1))

        for obj in moving:
            ghost = obj.copy()          # shares obj.data
            ghost.animation_data_clear()
            ghost.matrix_world = obj.matrix_world.copy()
            ghost.visible_shadow = casts
            coll.objects.link(ghost)

            for slot_index, slot in enumerate(ghost.material_slots):
                source = obj.data.materials[slot_index] if slot_index < len(obj.data.materials) else None
                slot.link = 'OBJECT'
                slot.material = alpha_material(cache, source, alpha)
            ghosts.append(ghost)
        top = max((obj.matrix_world @ Vector(c)).z for obj in moving for c in obj.bound_box)
        print(f"[ghost] pose {i} at frame {frame}, alpha {alpha:.2f}, top z {top:.3f}")

    # The originals are still animated; leaving them in would draw an extra,
    # unfaded pose on top of the sweep.
    for obj in list(moving):
        bpy.data.objects.remove(obj, do_unlink=True)
    scene.frame_start = scene.frame_end = scene.frame_current = 0
    print(f"[ghost] {len(ghosts)} ghost objects in {n} poses, {len(cache)} faded materials")
    return ghosts


# ── look ──────────────────────────────────────────────────────────────────────

def improve_scene_quality():
    """Smooth shading and a mild metallic/roughness pass on all mesh objects."""
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
        if obj.data.materials:
            for mat in obj.data.materials:
                if mat and mat.use_nodes:
                    bsdf = mat.node_tree.nodes.get("Principled BSDF")
                    if bsdf:
                        color = bsdf.inputs['Base Color'].default_value
                        brightness = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
                        # No metallic. The scene's bright materials are the white
                        # shelf panels and the arms' light-grey shells -- painted
                        # steel and plastic, neither of them mirrors. At Metallic
                        # 0.3 and Roughness 0.4 they threw a blown highlight across
                        # the top arm and clipped that whole side of the shelf to
                        # pure white.
                        if brightness > 0.6:
                            bsdf.inputs['Metallic'].default_value = 0.0
                            bsdf.inputs['Roughness'].default_value = 0.65
                        else:
                            bsdf.inputs['Metallic'].default_value = 0.0
                            bsdf.inputs['Roughness'].default_value = 0.6


def aimed_area_light(name, center, radius, azimuth, elevation, distance_scale,
                     power, size, color, focus):
    """A soft area light on an orbit around the subject, aimed at its centre.

    Placed relative to the subject rather than at fixed world coordinates: the
    old lights were pinned near the origin, which for this scene is under the
    table and off to one side, so the arms were lit obliquely and the shelf threw
    most of the reach poses into shadow.  Power scales with distance squared so
    the exposure does not change when the subject's bounding radius does.
    """
    data = bpy.data.lights.new(name, 'AREA')
    distance = distance_scale * radius
    data.energy = power * distance * distance
    data.size = size * radius
    data.color = color
    obj = bpy.data.objects.new(name, data)
    az, el = math.radians(azimuth), math.radians(elevation)
    obj.location = center + Vector((
        distance * math.cos(el) * math.cos(az),
        distance * math.cos(el) * math.sin(az),
        distance * math.sin(el),
    ))
    con = obj.constraints.new(type='TRACK_TO')
    con.target = focus
    con.track_axis = 'TRACK_NEGATIVE_Z'
    con.up_axis = 'UP_Y'
    bpy.context.scene.collection.objects.link(obj)
    return obj


def _principled(name):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    return mat, nt, nt.nodes["Principled BSDF"]


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
    for name, value in (("A", a), ("B", b)):
        if hasattr(value, "is_output"):
            nt.links.new(value, slots[name])
        else:
            slots[name].default_value = value
    nt.links.new(fac, node.inputs['Factor'])
    return [sock for sock in node.outputs if sock.enabled][0]


def steel_material():
    """Old machine-shop steel: near-black, scratched back to bare metal, rusting.

    Three procedural layers, each doing one thing:

    * **Scratches** -- noise sampled through a 90:1 stretched coordinate space,
      so it is fine across the surface and smeared along it, then clipped to a
      narrow band of the ramp so only the peaks survive as thin bright lines.
      Where the finish is scratched off, bare steel shows through: lighter and
      *less* rough than the surrounding dark paint.
    * **Rust** -- a second, coarser noise thresholded into patches. Rust is not
      metal, so it drives `Metallic` to zero and `Roughness` up as well as
      colouring; leaving it metallic is what makes procedural rust read as
      orange chrome.
    * **Bump** -- the two heights summed, weakly.

    Deliberately not shiny: roughness runs 0.35 in the scratches to 0.9 in the
    rust, so the table reads as a worn shop surface rather than throwing the
    kind of highlight the white shelf used to.
    """
    mat, nt, bsdf = _principled("BlackScratchedMetal")

    scratch_map = _object_coords(nt, (1.5, 30.0, 1.5))
    scratch = nt.nodes.new('ShaderNodeTexNoise')
    scratch.inputs['Scale'].default_value = 6.0
    scratch.inputs['Detail'].default_value = 8.0
    scratch.inputs['Roughness'].default_value = 0.75
    nt.links.new(scratch_map.outputs['Vector'], scratch.inputs['Vector'])

    # Clipped hard: only the top of the noise becomes a scratch, so the surface
    # is mostly intact dark finish with sparse bright lines through it.
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

    # A high threshold keeps the rust to a few patches rather than a coating.
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

    # Metal everywhere except the rust.
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
    mat, nt, bsdf = _principled("ShelfWood")
    bsdf.inputs['Metallic'].default_value = 0.0
    bsdf.inputs['Roughness'].default_value = 0.45

    mapping = _object_coords(nt, (1.0, 1.0, 1.0))
    # Bands along one axis, distorted by noise so they wander like real grain
    # instead of reading as printed stripes.
    wave = nt.nodes.new('ShaderNodeTexWave')
    wave.wave_type = 'BANDS'
    wave.bands_direction = 'Z'
    wave.inputs['Scale'].default_value = 5.0
    wave.inputs['Distortion'].default_value = 3.0
    wave.inputs['Detail'].default_value = 3.0
    wave.inputs['Detail Scale'].default_value = 1.4
    nt.links.new(mapping.outputs['Vector'], wave.inputs['Vector'])

    ramp = nt.nodes.new('ShaderNodeValToRGB')
    # Low contrast on purpose: the grain should read as wood tone, not as
    # printed stripes.  A single object-space direction cannot run along every
    # panel of the unit at once, so the bands are kept subtle enough that the
    # ones crossing a shelf the "wrong" way do not draw the eye.
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


BLENDERKIT_DIR = os.path.expanduser("~/blenderkit_data/materials")


def load_blenderkit_material(slug, root):
    """Append a cached BlenderKit material by asset slug, or return None.

    BlenderKit stores each downloaded asset under
    `<root>/<slug>_<uuid>/<name>_<uuid>.blend`, one material per file.  We append
    rather than link so the render does not depend on the cache staying put, and
    match on the slug prefix because the uuids differ per machine and per
    download.

    The cache is the user's own BlenderKit library and is not part of this repo,
    so every caller has to cope with this returning None.
    """
    import glob
    # BlenderKit truncates the slug in the directory name to 16 characters, so
    # "rusted-steel-plate" is stored under "rusted-steel-pla_<uuid>". Try the
    # full slug first, then the truncation.
    hits = sorted(glob.glob(os.path.join(root, f"{slug}*", "*.blend")))
    if not hits and len(slug) > 16:
        hits = sorted(glob.glob(os.path.join(root, f"{slug[:16]}*", "*.blend")))
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
    texel density: every surface in the scene gets the same texture scale
    regardless of how large the panel is, which is what stops the thin shelf
    edges from showing a wildly magnified crop of the wood.

    Done by hand rather than with `bpy.ops.uv.cube_project`, which needs an edit
    -mode context that does not exist under --background.
    """
    mesh = obj.data
    uv = mesh.uv_layers.get("BoxProject") or mesh.uv_layers.new(name="BoxProject")
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


def resolve_material(spec, fallback, root, label):
    """`spec` is a BlenderKit slug, or "procedural" for the built-in."""
    if spec == "procedural":
        return fallback(), "procedural"
    mat = load_blenderkit_material(spec, root)
    if mat is not None:
        return mat, f"blenderkit:{spec}"
    print(f"[material] WARNING: no BlenderKit asset matching '{spec}' under {root}; "
          f"falling back to the procedural {label}. Download it in Blender first, or "
          f"pass --{label}-material procedural to silence this.")
    return fallback(), "procedural (fallback)"


def restyle_static(by_path, args):
    """Give the table and the shelves real materials.

    Both arrive from Drake as flat untextured colour, and a white shelf panel is
    the brightest thing in frame -- the first surface to clip and the one the
    ghosts are hardest to read against.  Steel and wood are darker and carry
    their own detail, so the arms sit on them rather than in front of them.

    Must run after improve_scene_quality, which walks obj.data.materials and
    would otherwise put its metallic/roughness pass on top of these.
    """
    root = os.path.expanduser(args.blenderkit_dir)
    table_mat, table_src = resolve_material(args.table_material, steel_material, root, "table")
    shelf_mat, shelf_src = resolve_material(args.shelf_material, wood_material, root, "shelf")

    styled = {"table": (table_mat, args.table_texel_scale),
              "old_shelves": (shelf_mat, args.shelf_texel_scale)}
    counts = dict.fromkeys(styled, 0)
    for path, obj in by_path.items():
        model = path[len(VISUAL_PREFIX):].split("/", 1)[0]
        entry = styled.get(model)
        if entry is None or obj.type != 'MESH':
            continue
        mat, texels = entry
        box_project_uvs(obj, texels)
        obj.data.materials.clear()
        obj.data.materials.append(mat)
        counts[model] += 1
    print(f"[material] table: {counts['table']} objects from {table_src}, "
          f"old_shelves: {counts['old_shelves']} objects from {shelf_src}")
    if not all(counts.values()):
        print("[material] WARNING: expected both a table and shelves; check the model names")


def setup_studio_lighting(args, center, radius, camera_azimuth, focus):
    """Four large soft sources ringed around the subject, plus a sun.

    Wrapping the light around the camera axis is what clears the shelf: a single
    key from one side puts every pose that reaches into the shelving into its
    shadow, and no amount of extra key power fixes that -- it needs a source on
    the far side too.  The sources are broad (several times the subject radius)
    so their shadows stay soft, and the whole rig scales with `--light-strength`.
    """
    k = args.light_strength
    az = camera_azimuth
    # The key is deliberately weak relative to the fills.  Rendering the four
    # sources one at a time, the key accounted for every clipped pixel in frame
    # and the other three for none: it lands near-normal on the upper surfaces of
    # the topmost arm and on that side of the shelf, and at a conventional
    # key-to-fill ratio it burns both out.  A near-flat ratio costs some
    # modelling, which is the right trade for a figure whose subject is six
    # overlapping transparent poses -- contrast between them matters more than
    # contrast across any one of them.  Note `specular_factor` is not the lever
    # here; Cycles all but ignores it, and the blowout is diffuse anyway.
    aimed_area_light('Key', center, radius, az + 40, 45, 2.6, 9 * k, 2.5,
                     (1.0, 0.98, 0.95), focus)
    aimed_area_light('Fill', center, radius, az - 55, 20, 2.8, 9 * k, 3.0,
                     (0.90, 0.95, 1.0), focus)
    # Behind the shelf, so the poses buried in it are not lit only from the front.
    aimed_area_light('Back', center, radius, az + 165, 35, 3.0, 6 * k, 3.0,
                     (0.92, 0.94, 1.0), focus)
    aimed_area_light('Top', center, radius, az + 90, 80, 2.6, 6 * k, 3.0,
                     (1.0, 1.0, 1.0), focus)

    sun = bpy.data.lights.new('Sun', 'SUN')
    sun.energy = 0.35 * k
    sun.angle = math.radians(20)     # a wide disc, for soft sun shadows
    sun.color = (1.0, 0.98, 0.95)
    so = bpy.data.objects.new('Sun', sun)
    so.location = center + Vector((0, 0, 4 * radius))
    con = so.constraints.new(type='TRACK_TO')
    con.target = focus
    con.track_axis = 'TRACK_NEGATIVE_Z'
    con.up_axis = 'UP_Y'
    bpy.context.scene.collection.objects.link(so)
    print(f"[light] 4 area sources + sun, strength x{k:.2f}, "
          f"ring radius {2.6 * radius:.2f} m")


def setup_world(args):
    """Dark backdrop to the camera, bright ambient to everything else.

    One Background node cannot do both: raising its strength to lift the shadows
    also raises the visible backdrop until the figure floats on a grey field.
    Splitting on `Is Camera Ray` keeps the navy the camera sees at
    `--world-strength` while the light the geometry receives comes from a much
    brighter neutral dome at `--ambient` -- which is what actually fills the
    shadows the shelf and the ghosts cast on each other.
    """
    world = bpy.data.worlds.new('World')
    bpy.context.scene.world = world
    world.use_nodes = True
    nt = world.node_tree
    nt.nodes.clear()

    visible = nt.nodes.new('ShaderNodeBackground')
    visible.inputs['Color'].default_value = BG_COLOR
    visible.inputs['Strength'].default_value = args.world_strength

    ambient = nt.nodes.new('ShaderNodeBackground')
    ambient.inputs['Color'].default_value = (0.55, 0.58, 0.65, 1.0)
    ambient.inputs['Strength'].default_value = args.ambient

    path = nt.nodes.new('ShaderNodeLightPath')
    mix = nt.nodes.new('ShaderNodeMixShader')
    out = nt.nodes.new('ShaderNodeOutputWorld')
    nt.links.new(path.outputs['Is Camera Ray'], mix.inputs['Fac'])
    nt.links.new(ambient.outputs['Background'], mix.inputs[1])
    nt.links.new(visible.outputs['Background'], mix.inputs[2])
    nt.links.new(mix.outputs['Shader'], out.inputs['Surface'])
    print(f"[light] world backdrop {args.world_strength:.2f}, ambient {args.ambient:.2f}")


# ── camera ────────────────────────────────────────────────────────────────────

def corners_of(objects):
    """World-space bounding-box corners of `objects` as a flat list."""
    pts = []
    for obj in objects:
        if obj.type != 'MESH':
            continue
        pts.extend(obj.matrix_world @ Vector(c) for c in obj.bound_box)
    if not pts:
        raise SystemExit("Nothing to frame the camera on.")
    return pts


def sphere_of(pts):
    lo = Vector((min(p[i] for p in pts) for i in range(3)))
    hi = Vector((max(p[i] for p in pts) for i in range(3)))
    return (lo + hi) / 2.0, max((hi - lo).length / 2.0, 1e-6)


def fit_frustum(pts, direction, tan_w, tan_h, margin, iterations=24):
    """Smallest camera distance, and the aim point, that just contains `pts`.

    The old fit used the subject's bounding *sphere*, which is the loosest
    possible bound on a subject as elongated as two arms reaching into a shelf:
    it reserves room for a ball around the whole thing and leaves most of the
    frame empty.  This solves the frustum directly.

    With the aim point C, camera at C - D*direction, and per-point camera-space
    coordinates u (right), v (up), w (along the view), a point is inside the
    frustum when |u| <= (D + w)*tan_w and |v| <= (D + w)*tan_h, so

        D = max over points of max(|u|/tan_w - w, |v|/tan_h - w).

    C is then slid within the view plane to balance the two sides and the
    distance re-solved, which converges in a few passes; that recentring is what
    stops one extreme corner from pushing the whole subject off to the side.
    """
    up_hint = Vector((0.0, 0.0, 1.0))
    up = (up_hint - direction * up_hint.dot(direction)).normalized()
    right = direction.cross(up).normalized()

    center, _ = sphere_of(pts)
    distance = 0.0
    for _ in range(iterations):
        local = [((p - center).dot(right), (p - center).dot(up), (p - center).dot(direction))
                 for p in pts]
        distance = max(max(abs(u) / tan_w - w, abs(v) / tan_h - w) for u, v, w in local)
        distance = max(distance, 1e-3)
        # Screen-space extents at this distance, in [-1, 1]; recentre on their midpoint.
        sx = [u / ((distance + w) * tan_w) for u, v, w in local]
        sy = [v / ((distance + w) * tan_h) for u, v, w in local]
        du = 0.5 * (max(sx) + min(sx)) * tan_w * distance
        dv = 0.5 * (max(sy) + min(sy)) * tan_h * distance
        if abs(du) < 1e-6 and abs(dv) < 1e-6:
            break
        center = center + right * du + up * dv

    # How much of each axis the subject actually fills, so the resolution can be
    # chosen to match rather than padded with empty frame.
    local = [((p - center).dot(right), (p - center).dot(up), (p - center).dot(direction))
             for p in pts]
    fill_x = max(abs(u) / ((distance + w) * tan_w) for u, v, w in local)
    fill_y = max(abs(v) / ((distance + w) * tan_h) for u, v, w in local)
    return center, distance * margin, fill_x, fill_y


def setup_camera(args, subject, light_subject):
    """Orbit the subject and solve the distance so it just fills the frame.

    Resolution must already be set: Blender's AUTO sensor fit maps the 36 mm
    sensor width onto the longer axis, so the field of view depends on the
    aspect ratio.
    """
    scene = bpy.context.scene
    pts = corners_of(subject)

    # Defaults picked by contact sheet.  The shelf unit sits at +x from the arms, so
    # any azimuth looking from that side (the -90..+90 half) puts its side panel
    # straight through the motion; 180 is clear but flat and symmetric.  225 is the
    # three-quarter view: both arms stay distinct, the arch of the bimanual motion
    # reads, and the shelf falls behind and to the side.
    azimuth = math.radians(args.camera_azimuth if args.camera_azimuth is not None else 225.0)
    elevation = math.radians(args.camera_elevation if args.camera_elevation is not None else 22.0)

    cam_data = bpy.data.cameras.new('Camera')
    cam_data.lens = args.lens
    cam_data.clip_end = 1000.0

    aspect = scene.render.resolution_x / scene.render.resolution_y
    half_sensor = math.atan(cam_data.sensor_width / (2 * args.lens))
    if aspect >= 1.0:                      # AUTO fit puts the sensor on the long axis
        tan_w, tan_h = math.tan(half_sensor), math.tan(half_sensor) / aspect
    else:
        tan_h, tan_w = math.tan(half_sensor), math.tan(half_sensor) * aspect

    orbit = Vector((
        math.cos(elevation) * math.cos(azimuth),
        math.cos(elevation) * math.sin(azimuth),
        math.sin(elevation),
    ))
    center, distance, fill_x, fill_y = fit_frustum(pts, -orbit, tan_w, tan_h,
                                                   args.frame_margin)
    if args.camera_distance is not None:
        distance = args.camera_distance

    cam = bpy.data.objects.new('Camera', cam_data)
    scene.collection.objects.link(cam)
    scene.camera = cam
    cam.location = center + orbit * distance

    target = bpy.data.objects.new('CamTarget', None)
    target.location = center
    scene.collection.objects.link(target)
    con = cam.constraints.new(type='TRACK_TO')
    con.target = target
    con.track_axis = 'TRACK_NEGATIVE_Z'
    con.up_axis = 'UP_Y'

    # Printed so a good framing can be pinned as the driver's defaults.  A fill
    # well under 1.0 on one axis means that much of the frame is empty, and the
    # resolution's aspect ratio wants changing rather than the camera.
    print(f"[camera] center=({center.x:.3f}, {center.y:.3f}, {center.z:.3f}) "
          f"azimuth={math.degrees(azimuth):+.1f} elevation={math.degrees(elevation):+.1f} "
          f"distance={distance:.3f} lens={args.lens:.1f} "
          f"fill=({fill_x:.2f}, {fill_y:.2f})")

    # The lighting rig hangs off the subject it lights, not the framing bounds.
    light_center, light_radius = sphere_of(corners_of(light_subject))
    return light_center, light_radius, math.degrees(azimuth), target


# ── render configuration ──────────────────────────────────────────────────────

def enable_cycles_gpu(prefer=("OPTIX", "CUDA", "HIP", "ONEAPI")):
    """Enable the Cycles add-on and select a compute backend.

    Returns the backend name selected, or None if no GPU is usable.

    Cycles is not enabled under `--factory-startup`, so `addon_utils.enable`
    has to run before `scene.render.engine = 'CYCLES'` is even a valid
    assignment. `get_devices()` must be called after setting
    `compute_device_type` or the device list stays empty and every device
    silently ends up with `use = False` -- a full CPU fallback that otherwise
    looks exactly like a successful GPU render.  Filter on `d.type`, never on
    `d.name`: the same GPU is enumerated once per backend it supports.
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


def configure_render(args):
    """Put the scene on Cycles, on a GPU where there is one and on the CPU where
    there is not.

    Deliberately *not* the EEVEE fallback the sibling repos use.  This figure is
    stacked semi-transparent geometry, and what makes that composite correctly is
    `cycles.transparent_max_bounces`, which EEVEE has no equivalent for -- so an
    automatic drop to EEVEE would not be a slower render of the same picture, it
    would be a different picture.  Cycles on CPU is slow and right.  EEVEE is
    still reachable with `--engine EEVEE`, explicitly.

    Always prints the engine and device actually selected: a silent fall back to
    the CPU changes only how long the render takes and is otherwise
    indistinguishable from success.
    """
    scene = bpy.context.scene

    if args.engine == "EEVEE":
        try:
            scene.render.engine = 'BLENDER_EEVEE_NEXT'
        except Exception:
            scene.render.engine = 'BLENDER_EEVEE'
        try:
            scene.eevee.taa_render_samples = max(args.samples, 64)
        except AttributeError:
            pass
        print(f"[render] {scene.render.engine} by request "
              f"-- ghost transparency will not match Cycles")
        return scene.render.engine

    if args.device == "CPU":
        backend = None
    elif args.device == "auto":
        backend = enable_cycles_gpu()
    else:
        backend = enable_cycles_gpu(prefer=(args.device,))
        if backend is None:
            raise SystemExit(
                f"--device {args.device} was requested but no {args.device} device is "
                "usable in this Blender build. Use --device auto or --device CPU.")

    if args.device == "CPU":
        # Still needs the add-on enabled before the engine can be assigned.
        try:
            addon_utils.enable("cycles", default_set=True, persistent=True)
        except Exception as e:
            raise SystemExit(f"could not enable the Cycles add-on: {e}")

    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'GPU' if backend else 'CPU'
    scene.cycles.samples = args.samples
    try:
        scene.cycles.use_adaptive_sampling = True
        # Tighter than the 0.01 default: six layers of stacked transparency are
        # the noisiest thing in frame and adaptive sampling will otherwise call
        # them converged while they are still visibly grainy.
        scene.cycles.adaptive_threshold = 0.004
    except AttributeError:
        pass
    try:
        scene.cycles.use_denoising = True
        scene.cycles.denoiser = 'OPTIX' if backend == 'OPTIX' else 'OPENIMAGEDENOISE'
    except (AttributeError, TypeError):
        pass

    # Overlapping ghosts go black in places without a generous transparent
    # bounce budget -- every layer of alpha costs one.
    scene.cycles.transparent_max_bounces = 32
    scene.cycles.max_bounces = 16

    if backend:
        print(f"[render] CYCLES on GPU via {backend}, {args.samples} samples")
    else:
        print(f"[render] CYCLES on CPU, {args.samples} samples")
    return f"CYCLES/{backend or 'CPU'}"


def configure_output(args):
    scene = bpy.context.scene
    scene.render.resolution_x, scene.render.resolution_y = args.resolution
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'
    # Transparent by default: the figure goes into the paper over whatever the
    # page background is, and a baked-in navy field would sit in a box on it.
    # The world's visible colour still matters under --opaque-bg, and the ambient
    # dome lights the scene either way -- film_transparent only affects camera
    # rays that miss all geometry.
    scene.render.film_transparent = not args.opaque_bg
    # AgX rolls off highlights and desaturates; a technical figure wants the
    # colours it was given.
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    if not os.path.isfile(args.html):
        raise SystemExit(f"No such Meshcat page: {args.html}")

    objs = import_scene(args.html)
    moving, static, framed_static, static_by_path = classify(objs)

    # Before baking, deliberately.  The ghosts render through OBJECT-linked material
    # slots, so their materials are copies taken here; improve_scene_quality walks
    # obj.data.materials, which for a ghost is the shared mesh-level list it no longer
    # renders from.  Run it after baking and the metallic/roughness pass silently
    # applies to nothing visible.
    improve_scene_quality()
    restyle_static(static_by_path, args)

    f0, f1, window = frame_window(args.t_start, args.t_end)
    print(f"[poses] animation frames {f0}-{f1}")
    track = scan_motion(moving, window)
    moving, welded = split_static(moving, track, args.static_tol)
    frames = sample_frames(window, track, moving, args)
    ghosts = bake_ghosts(moving, frames, args)

    configure_output(args)          # resolution first: the camera fit reads the aspect
    configure_render(args)
    # The lights are placed relative to the fitted subject and the camera axis, so
    # the camera has to be solved first.
    # Framed to the arms, their bases and the shelf; the table is rendered but
    # deliberately not framed to.
    center, radius, azimuth, focus = setup_camera(args, ghosts + welded + framed_static,
                                                  ghosts + welded)
    setup_studio_lighting(args, center, radius, azimuth, focus)
    setup_world(args)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    bpy.context.scene.render.filepath = os.path.abspath(args.out)
    print(f"[render] {args.resolution[0]}x{args.resolution[1]} -> {args.out}")
    bpy.ops.render.render(write_still=True)
    print(f"[render] wrote {args.out}")


if __name__ == "__main__":
    main()
