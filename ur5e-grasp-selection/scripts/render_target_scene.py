"""
render_target_scene.py

Render the target-scene figure in Blender from a Meshcat scene exported by
scripts/visualize_target_scene.py.

This runs inside Blender's own Python, not the project venv:

    blender --background --python scripts/render_target_scene.py -- \
        --html out/target_scene.html --out out/target_scene.png

Use scripts/render_target_scene.sh, which fills in the Blender path and defaults.

This is a deliberate standalone copy of scripts/render_grasp_figure.py rather than an
import of it, so that nothing done for this figure can move the published one.  The cost
is that four functions here each encode a bug that returns silently if simplified away --
assign_ur_link_colors, restore_ur_flat_shading, separate_coplanar_tables and
enable_gpu_devices -- and they now exist in two places.  Fix them in both.

What differs from the grasp figure: the scene is the benchmark's cluttered one (four
shelf units, a bin, seven decorative mugs) rather than a bare table, there is exactly one
arm and it is opaque, the framing is wide enough to hold the arm and the shelf units that
flank it rather than cropping to the grasp, and the lighting scales with that framing
instead of staying tuned for a 0.3 m subject.
"""

import argparse
import math
import os
import sys
from xml.etree import ElementTree

import bpy
from mathutils import Vector

# The scene comes in with Drake's world axes, which are already Z-up like Blender's, so
# camera placement below is in the same coordinates the visualization script uses.
#
# Unlike the grasp figure this is an establishing shot, not a close-up: the target sits in
# the middle of the benchmark's clutter and the point is to see the clutter.  The default
# target is the middle of the work surface rather than the mug, and the default frame
# radius (see --frame-radius) is metres rather than centimetres.  Only the *direction* of
# this position is used when auto-framing is on; the distance is solved for.  It
# corresponds to azimuth +169.94 deg, elevation +17.98 deg about the target, 3.402 m out.
#
# That is the near-opposite side from where this figure was first framed (azimuth -20.06
# deg, the same elevation and distance).  From the -20 side the arm reads against the far
# shelves but the two near units are behind the camera, so the scene looks emptier than
# the benchmark's actually is; from +169.94 the arm sits in the gap between all four
# shelf units with the bin behind it, and the red target mug stays clear of the gripper
# body.  A straight 180 deg orbit (+159.94) is worse: a near shelf upright crosses the
# gripper and hides the mug.
DEFAULT_CAMERA_POS = (-3.0366, 0.5653, 1.47)
DEFAULT_CAMERA_TARGET = (0.15, 0.0, 0.42)

# Crop rectangle, as fractions of the frame: left, top, right, bottom, with the origin at
# the top-left.  Fractions rather than pixels so the crop survives a change of
# --resolution; at the default 3000x1688 this is pixels (360, 262) to (2692, 1504).
#
# The full frame at the default camera carries dead space on all four sides -- empty sky
# above the shelf tops, bare table to the left, right and front -- and this is where a
# hand-crop of the default view put the edges.  Cycles renders the border only, so a
# cropped figure is also a cheaper one.  Pass --crop to move it, --no-crop for the full
# frame; either way the camera does not move, so the framing of what remains is unchanged.
DEFAULT_CROP = (0.12, 262.0 / 1688.0, 2692.0 / 3000.0, 1504.0 / 1688.0)

ADDON_MODULE = "bl_ext.user_default.meshcat_html_importer"


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--html", required=True, help="Meshcat StaticHtml scene to import.")
    ap.add_argument("--out", required=True, help="Output PNG path.")
    ap.add_argument("--samples", type=int, default=256, help="Cycles samples.")
    ap.add_argument("--resolution", type=int, nargs=2, default=[3000, 1688],
                    help="Render resolution, width height.")
    ap.add_argument("--crop", type=float, nargs=4, default=list(DEFAULT_CROP),
                    metavar=("LEFT", "TOP", "RIGHT", "BOTTOM"),
                    help="Crop the render to this rectangle, as fractions of the frame "
                         "from the top-left corner. Cycles renders the border only, so "
                         "this costs nothing. Default: the hand-picked crop of the "
                         "default view.")
    ap.add_argument("--no-crop", action="store_true",
                    help="Render the full frame instead of cropping to --crop.")
    ap.add_argument("--camera-pos", type=float, nargs=3, default=list(DEFAULT_CAMERA_POS))
    ap.add_argument("--camera-target", type=float, nargs=3, default=list(DEFAULT_CAMERA_TARGET))
    ap.add_argument("--camera-azimuth", type=float, default=None,
                    help="Orbit the camera around the target, in degrees (0 = +x, 90 = +y). "
                         "Increasing it swings the camera right and turns the view left, "
                         "which is how you shift parallax to see between the arms. "
                         "Overrides --camera-pos.")
    ap.add_argument("--camera-elevation", type=float, default=None,
                    help="Camera elevation above the horizontal, in degrees. 90 is straight "
                         "down. Used with --camera-azimuth.")
    ap.add_argument("--camera-distance", type=float, default=None,
                    help="Camera distance from the target in metres. Only relevant when "
                         "auto-framing is off; otherwise framing sets it.")
    ap.add_argument("--lens", type=float, default=50.0, help="Camera focal length in mm.")
    ap.add_argument("--opaque-background", action="store_true",
                    help="Render a solid backdrop instead of a transparent one.")
    ap.add_argument("--device", choices=["CPU", "GPU"], default="GPU",
                    help="GPU tries OptiX, then CUDA/HIP/oneAPI, and falls back to CPU "
                         "with a warning if none has an enabled device.")
    ap.add_argument("--no-auto-frame", action="store_true",
                    help="Use --camera-pos verbatim instead of fitting the subject in frame.")
    ap.add_argument("--arm-alpha", type=float, default=None,
                    help="Override the arm transparency used when restyling (0-1). This "
                         "figure has one arm and defaults to opaque; the flag is kept for "
                         "ghosting it against the clutter.")
    ap.add_argument("--no-restyle", action="store_true",
                    help="Keep the flat materials from the Meshcat export.")
    ap.add_argument("--no-wood-table", action="store_true",
                    help="Leave the table with its imported material instead of wood.")
    ap.add_argument("--view-transform", type=str, default="Standard",
                    choices=["Standard", "AgX", "Filmic"],
                    help="Colour management. Standard keeps flat colours saturated; AgX "
                         "rolls off highlights but desaturates.")
    ap.add_argument("--exposure", type=float, default=0.0, help="Exposure stops.")
    ap.add_argument("--frame-radius", type=float, default=0.72,
                    help="Radius in metres around the target to fit in frame. The grasp "
                         "figure uses 0.30 because it is a close-up; this is an "
                         "establishing shot, so it needs enough to hold the shelves and "
                         "the bin. Lighting scales with this.")
    return ap.parse_args(argv)


def enable_importer():
    if ADDON_MODULE not in bpy.context.preferences.addons:
        bpy.ops.preferences.addon_enable(module=ADDON_MODULE)


def import_scene(html_path):
    bpy.ops.import_scene.meshcat_html(
        filepath=html_path,
        clear_scene=True,
        hierarchical_collections=True,
    )


def report_transparency():
    """Report what is transparent.

    The grasp figure fails if its arms arrive opaque; this one is the other way round --
    there is a single arm and it is meant to be solid -- so this is a report, not a check.
    Anything transparent here is either the finray fingers or something that slipped
    through restyle_scene, and is worth seeing in the log.
    """
    transparent, opaque = [], []
    for material in bpy.data.materials:
        if not material.use_nodes or material.users == 0:
            # Wholesale replacement leaves the imported materials in the file with no
            # users.  Counting them reports transparency that is not in the render.
            continue
        for node in material.node_tree.nodes:
            if node.type == "BSDF_PRINCIPLED":
                alpha = node.inputs["Alpha"].default_value
                (transparent if alpha < 0.999 else opaque).append((material.name, alpha))
                break
    print(f"[materials] {len(transparent)} transparent, {len(opaque)} opaque")
    if transparent:
        alphas = sorted({round(a, 3) for _, a in transparent})
        print(f"[materials] transparent alphas present: {alphas} "
              f"(expected none outside the finray fingers)")
    return len(transparent)


# Materials, keyed by the Drake model name that owns the geometry.
#
# The UR arms are the exception: their real per-link colours now survive the whole
# Drake -> Meshcat -> Blender chain (the visualization script splits each `.dae` into one
# OBJ per colour and writes the colour into the generated URDF), so the ur5e entry's RGB
# is only a fallback for slots that somehow arrive without a base colour.  Everything
# else is authored here, because the Meshcat export carries only flat colours for it.
#
# There is one arm here and nothing to see through it, so ur5e and wsg are opaque.  Note
# that --arm-alpha defaults to None and falls through to the ur5e entry's alpha, so the
# opacity has to be set *here*; changing only the flag's default would do nothing.
#
# The seven decorative mugs all load the same models/mug/mug_simple_red.urdf as the target
# does, so they are muted to a plain ceramic and only target_mug keeps the saturated red.
# Without that the goal object is one of eight identical mugs.  restyle_scene keys on the
# Drake model instance, so they are individually addressable despite sharing a mesh.
#
# (base colour RGB, alpha, metallic, roughness)
# Base colours are LINEAR, not sRGB: a value that looks mid-grey as a hex swatch renders
# roughly one stop brighter than expected (linear 0.34 displays at about sRGB 0.61, which
# is why the shelves first came out pale beige rather than plywood).  These are picked in
# linear and checked against the rendered pixels.
_SHELF_MATERIAL = ((0.115, 0.088, 0.065), 1.00, 0.00, 0.68)  # dark stained plywood
_MUG_MATERIAL = ((0.62, 0.60, 0.55), 1.00, 0.00, 0.35)       # unglazed off-white ceramic
MATERIALS = {
    "ur5e":       ((0.55, 0.56, 0.58), 1.00, 0.05, 0.50),   # fallback only; see above
    "wsg":        ((0.16, 0.17, 0.19), 1.00, 0.85, 0.30),   # Schunk dark anodised body
    "target_mug": ((0.72, 0.10, 0.09), 1.00, 0.00, 0.28),   # glazed red ceramic
    "binF":       ((0.13, 0.145, 0.17), 1.00, 0.00, 0.75),  # matte moulded plastic
    "shelves":    _SHELF_MATERIAL,
    "shelves2":   _SHELF_MATERIAL,
    "shelves3":   _SHELF_MATERIAL,
    "shelves4":   _SHELF_MATERIAL,
    "mug":        _MUG_MATERIAL,
    "mug2":       _MUG_MATERIAL,
    "mug3":       _MUG_MATERIAL,
    "mug4":       _MUG_MATERIAL,
    "rmug":       _MUG_MATERIAL,
    "rmug2":      _MUG_MATERIAL,
    "rmug3":      _MUG_MATERIAL,
}
# The finray fingers already read well, so their imported blue is left alone.
FINGER_MATERIAL_HINT = "finray"
TABLE_MODELS = ("table", "table2")

# Every Drake model owning_model is allowed to name.  Deriving it keeps the lookup and the
# styling from drifting apart: a model in MATERIALS but not here is invisible to
# restyle_scene and quietly keeps its imported placeholder.
STYLED_MODELS = frozenset(MATERIALS) | frozenset(TABLE_MODELS)


def owning_model(obj):
    """
    Drake model instance that owns an object, from its collection ancestry.

    The importer flattens object names to `visualizer_Mesh.NNN`, but with
    hierarchical_collections it nests them as
    MeshcatObjects/drake/visualizer/<model>/<link>/<object>, so the model name is
    recoverable from the collection tree.

    The set of names recognised here is derived from MATERIALS and TABLE_MODELS rather
    than hardcoded.  Adding a model to MATERIALS and not here is silent: restyle_scene
    would never see the name, the object would keep Meshcat's flat placeholder, and the
    only symptom is a line in the unmatched warning.

    Matching is exact on the collection's base name.  The grasp figure additionally
    accepts a `<model>_N` prefix because it draws three numbered copies of the arm; this
    scene has one of each, and the prefix rule is actively unsafe here -- the target mug's
    link collection is `mug_body_link`, which would match the decorative model `mug`.
    """
    for collection in obj.users_collection:
        node = collection
        chain = []
        while node is not None:
            chain.append(node.name)
            parents = [c for c in bpy.data.collections
                       if node.name in {ch.name for ch in c.children}]
            node = parents[0] if parents else None
        for name in chain:
            base = name.split(".")[0]
            if base in STYLED_MODELS:
                return base
    return None


def _meshcat_chain(obj):
    """
    Collection ancestry of an imported object, innermost first, or [] if it is not part of
    the Meshcat hierarchy.

    An object belongs to more than one collection (the scene's own "Collection" as well as
    the imported tree), so the branch that actually reaches MeshcatObjects has to be picked
    out rather than assumed to be the first.
    """
    for collection in obj.users_collection:
        node = collection
        chain = []
        while node is not None:
            chain.append(node.name)
            parents = [c for c in bpy.data.collections
                       if node.name in {ch.name for ch in c.children}]
            node = parents[0] if parents else None
        if any(name.split(".")[0] == "MeshcatObjects" for name in chain):
            return chain
    return []


def owning_link(obj):
    """
    Drake link that owns an object, from its collection ancestry.

    The tree is MeshcatObjects/drake/visualizer/<model>/<link>/<object>, so the link is one
    level above the object's own collection.  Blender uniquifies repeated collection names
    across the three arms (`forearm_link`, `forearm_link.001`, ...), so the `.NNN` suffix is
    stripped.
    """
    chain = _meshcat_chain(obj)
    return chain[1].split(".")[0] if len(chain) > 1 else None


# Path to the generated per-colour URDF, relative to the repo root.  This script runs
# inside Blender, so the repo root comes from the script's own location rather than cwd.
UR_COLOUR_URDF = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models/universal_robots/ur_description/urdf/ur5e_drake_collision_obj.urdf",
)


# How far a merged-mesh polygon centroid may sit from the nearest per-colour OBJ face
# centroid, in metres, and still count as that face.  The match is exact in practice --
# measured max distance 0.0 over all 21 link meshes -- because Meshcat carries the OBJ
# vertices through verbatim; this is a guard against a silently reprocessed mesh, not a
# tolerance anything relies on.  1e-6 m is far below the ~1e-4 m spacing between distinct
# faces on these meshes.
CENTROID_MATCH_TOL = 1e-6

# Fraction of a link's polygons allowed to miss that tolerance before the link is left
# grey rather than part-coloured.
UNMATCHED_FRACTION_SLACK = 0.001


def _obj_face_centroids(path):
    """
    Face centroids of an OBJ, in the order the file lists them.

    Only `v` and `f` are read: the per-colour OBJs are pure geometry (the colour lives in
    the generated URDF's `<material>`), and face indices may carry `v/vt/vn` triples.
    """
    verts = []
    centroids = []
    with open(path, "r") as f:
        for line in f:
            if line.startswith("v "):
                verts.append(tuple(float(x) for x in line.split()[1:4]))
            elif line.startswith("f "):
                idx = [int(tok.split("/")[0]) - 1 for tok in line.split()[1:]]
                n = len(idx)
                centroids.append(tuple(
                    sum(verts[i][axis] for i in idx) / n for axis in range(3)))
    return centroids


def _parse_ur_colour_urdf(urdf_path=UR_COLOUR_URDF):
    """
    Per-UR-link colour groups, read off the generated URDF: a list of
    (rgb, [face centroid, ...]) in `<visual>` order.

    `expand_visuals_by_color` writes one `<visual>` per colour group per link, each naming
    its own mesh OBJ and its own `<material>`.  The centroids come from the OBJ on disk and
    are what identifies a face downstream -- see assign_ur_link_colors for why the group
    *order* recorded here cannot be used.
    """
    if not os.path.exists(urdf_path):
        print(f"[ur-colours] WARNING: {urdf_path} not found; arms will stay grey")
        return {}

    tree = ElementTree.parse(urdf_path)
    urdf_dir = os.path.dirname(urdf_path)
    groups = {}
    for link in tree.getroot().iterfind("link"):
        link_name = link.get("name")
        entries = []
        for visual in link.iterfind("visual"):
            mesh = visual.find("geometry/mesh")
            colour = visual.find("material/color")
            if mesh is None or colour is None:
                continue
            filename = mesh.get("filename", "")
            # package://eaik_ift_experiment/models/... -> a path under the repo
            rel = filename.split("ur_description/", 1)[-1]
            obj_path = os.path.normpath(os.path.join(urdf_dir, "..", rel))
            if not os.path.exists(obj_path):
                print(f"[ur-colours] WARNING: missing mesh {obj_path}")
                continue
            rgba = [float(v) for v in colour.get("rgba").split()]
            entries.append((tuple(rgba[:3]), _obj_face_centroids(obj_path)))
        if entries:
            groups[link_name] = entries
    return groups


def _colour_lookup(entries):
    """
    KD-tree over a link's per-colour face centroids, plus the group index of each.

    Built once per link and shared by the three arms, since they instance the same meshes.
    """
    from mathutils.kdtree import KDTree

    labels = []
    for group, (_, centroids) in enumerate(entries):
        labels.extend([group] * len(centroids))
    tree = KDTree(len(labels))
    i = 0
    for _, centroids in entries:
        for centroid in centroids:
            tree.insert(centroid, i)
            i += 1
    tree.balance()
    return tree, labels


def restore_ur_flat_shading():
    """
    Undo the normal-smoothing the Meshcat export applies to the arm meshes.

    The UR `.dae` meshes are authored **faceted**: they are finely tessellated and every
    corner normal equals its face normal.  Measured over all 20 per-colour OBJs, a corner
    normal departs from its face normal by a median of 0.00 deg and a 95th percentile of
    0.06 deg.  (A handful of corners read 3-27 deg, all on near-degenerate slivers where the
    cross-product face normal is itself meaningless.)  Sharp features -- the engraved UR
    logo, the screw bosses, the joint-cap rims -- are held by that, not by edge tags.

    The export does not preserve it.  Meshcat welds coincident vertices, which throws away
    the per-corner split at every crease and leaves Blender averaging one smooth normal per
    vertex: `wrist3_d1d1d1.obj` has 2210 vertices for 911 faces on disk and arrives here with
    485.  The imported corner normals then depart from their face normals by a median of
    24.7 deg and up to 82.7 deg, which rounds off every crease -- the engraved logo groove
    renders as an inflated ridge and the screw bosses as blisters.

    Restoring flat shading reproduces the authored normals to within that 0.06 deg, so it is
    the faithful reconstruction rather than a stylistic choice.  The custom normal attribute
    has to be removed as well as the faces set flat, because custom split normals override
    the face normals and would otherwise keep the smoothing.
    """
    flattened = 0
    for obj in bpy.data.objects:
        if obj.type != "MESH" or owning_model(obj) != "ur5e":
            continue
        mesh = obj.data
        if "custom_normal" in mesh.attributes:
            mesh.attributes.remove(mesh.attributes["custom_normal"])
        for polygon in mesh.polygons:
            polygon.use_smooth = False
        flattened += 1
    print(f"[ur-shading] restored flat (as-authored) shading on {flattened} link mesh(es); "
          f"the Meshcat export welds vertices and averages the creases away")
    return flattened


def assign_ur_link_colors(groups=None):
    """
    Reassign the four real UR link colours to the imported arm meshes, in Blender.

    Why this is needed at all.  The colours are carried correctly as far as Drake -- reading
    the illustration properties off the model inspector for `ur5e_drake_collision_obj.urdf`
    gives the four distinct colours -- but **Drake's Meshcat visualizer merges all of a
    link's visual geometries into a single Meshcat object**, so the one-OBJ-per-colour split
    collapses on export.  Each arm arrives here as 7 meshes, one per link, carrying one flat
    grey material.  The earlier fix, which plumbed the colours through Drake by splitting
    each `.dae` by colour and writing a `<material>` per `<visual>`, cannot survive that
    merge; its Blender half (`tune_imported_material`, which keeps imported colours) is
    still correct and still runs, it just has nothing left to keep.

    The recovery: the merge preserves the *geometry* exactly -- every polygon of the merged
    mesh is one of the faces of one of the per-colour OBJs, vertex for vertex -- so a face
    is identified by its centroid, and its colour is the colour of the OBJ that centroid
    came from.  Each link mesh gets one material slot per colour group and every polygon is
    assigned to the slot of its nearest OBJ face centroid (measured max distance 0.0 across
    all 21 link meshes, so this is a lookup rather than a nearest-neighbour approximation).

    Do not go back to assigning polygons by cumulative face range in `<visual>` order.  That
    was the previous implementation and it is wrong: the merged mesh is *not* the
    concatenation of the colour groups in URDF order.  Measured against the geometric truth,
    the ranges put the right colour on 47% of upper_arm_link's faces, 49% of wrist_2_link's
    and 80% of wrist_1_link's -- the wrist end caps came out dark grey instead of UR blue.
    The merged order is neither the URDF's nor even contiguous per colour on every link, so
    no reordering of the groups fixes it; only the geometry identifies a face.
    """
    if groups is None:
        groups = _parse_ur_colour_urdf()
    if not groups:
        return 0

    palette = {}
    lookups = {}
    recoloured, skipped, partial = 0, [], []
    for obj in bpy.data.objects:
        if obj.type != "MESH" or owning_model(obj) != "ur5e":
            continue
        link = owning_link(obj)
        entries = groups.get(link)
        if not entries:
            skipped.append((obj.name, link, "no colour groups in the URDF"))
            continue

        if link not in lookups:
            lookups[link] = _colour_lookup(entries)
        tree, labels = lookups[link]

        # Match first, and only commit the materials if the match is clean, so a link that
        # cannot be identified is left with its imported grey rather than part-coloured.
        slots, unmatched, worst = [], 0, 0.0
        for polygon in obj.data.polygons:
            _, index, distance = tree.find(polygon.center)
            worst = max(worst, distance)
            if distance > CENTROID_MATCH_TOL:
                unmatched += 1
            slots.append(labels[index])
        n_polys = len(slots)
        if unmatched > max(1, int(UNMATCHED_FRACTION_SLACK * n_polys)):
            skipped.append((obj.name, link,
                            f"{unmatched}/{n_polys} polygons have no face within "
                            f"{CENTROID_MATCH_TOL:g} m of an OBJ centroid "
                            f"(worst {worst:.2e} m)"))
            continue
        if unmatched:
            partial.append((obj.name, link, unmatched, n_polys, worst))

        # Inherit alpha and the rest of the shading from whatever the import produced, so
        # the transparency that came through Drake is not thrown away here.
        alpha = 1.0
        source = obj.data.materials[0] if obj.data.materials else None
        if source is not None and source.use_nodes:
            bsdf = next((n for n in source.node_tree.nodes
                         if n.type == "BSDF_PRINCIPLED"), None)
            if bsdf is not None:
                alpha = bsdf.inputs["Alpha"].default_value

        obj.data.materials.clear()
        for rgb, _ in entries:
            key = (rgb, round(alpha, 4))
            material = palette.get(key)
            if material is None:
                hexname = "%02x%02x%02x" % tuple(int(round(255 * c)) for c in rgb)
                material = build_principled(f"URColor_{hexname}", rgb, alpha,
                                            MATERIALS["ur5e"][2], MATERIALS["ur5e"][3])
                palette[key] = material
            obj.data.materials.append(material)

        for polygon, slot in zip(obj.data.polygons, slots):
            polygon.material_index = slot
        recoloured += 1

    print(f"[ur-colours] recoloured {recoloured} link mesh(es) from {len(palette)} "
          f"materials: {sorted(set(k[0] for k in palette))}")
    for name, link, unmatched, total, worst in partial:
        print(f"[ur-colours] {name} (link={link}): {unmatched}/{total} polygons matched "
              f"only loosely (worst {worst:.2e} m), coloured by nearest face")
    for name, link, why in skipped:
        print(f"[ur-colours] WARNING: left {name} (link={link}) grey -- {why}")
    return recoloured


def build_principled(name, rgb, alpha, metallic, roughness):
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    bsdf = material.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*rgb, 1.0)
    bsdf.inputs["Alpha"].default_value = alpha
    bsdf.inputs["Metallic"].default_value = metallic
    bsdf.inputs["Roughness"].default_value = roughness
    return material


def build_wood_material(name="figure_wood"):
    """Procedural wood for the table: noise-stretched rings through a warm colour ramp."""
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    tree = material.node_tree
    bsdf = tree.nodes["Principled BSDF"]
    bsdf.inputs["Roughness"].default_value = 0.35
    bsdf.inputs["Metallic"].default_value = 0.0

    coord = tree.nodes.new("ShaderNodeTexCoord")
    mapping = tree.nodes.new("ShaderNodeMapping")
    # Stretch along one axis so the grain runs the length of the table.
    mapping.inputs["Scale"].default_value = (1.0, 9.0, 1.0)

    noise = tree.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 3.2
    noise.inputs["Detail"].default_value = 4.0
    noise.inputs["Roughness"].default_value = 0.6

    wave = tree.nodes.new("ShaderNodeTexWave")
    wave.wave_type = "BANDS"
    wave.bands_direction = "Y"
    wave.inputs["Scale"].default_value = 2.2
    wave.inputs["Distortion"].default_value = 9.0
    wave.inputs["Detail"].default_value = 1.5

    ramp = tree.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0.18
    ramp.color_ramp.elements[0].color = (0.13, 0.06, 0.025, 1.0)
    ramp.color_ramp.elements[1].position = 0.78
    ramp.color_ramp.elements[1].color = (0.33, 0.20, 0.11, 1.0)

    tree.links.new(coord.outputs["Object"], mapping.inputs["Vector"])
    tree.links.new(mapping.outputs["Vector"], noise.inputs["Vector"])
    tree.links.new(noise.outputs["Fac"], wave.inputs["Vector"])
    tree.links.new(wave.outputs["Fac"], ramp.inputs["Fac"])
    tree.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    return material


# Vertical separation applied to table2 in Blender, in metres.  Purely a rendering
# artifact fix -- see separate_coplanar_tables().
TABLE2_Z_OFFSET = 1e-4


def separate_coplanar_tables(offset=TABLE2_Z_OFFSET):
    """
    Nudge table2 down so it is not exactly coplanar with table.

    The scene deliberately contains two overlapping `table_wide` slabs (the robot's
    standing surface and the work surface), welded at the same height, so their tops are
    *exactly* coincident over the overlap.  Cycles then has each surface shadow the other
    there and the whole overlap region renders pure black -- which is what the figure used
    to show.  Diagnosing this is easy to get wrong: the region has alpha 255, so it is
    geometry rather than background, and a `scene.ray_cast` through it lands on the table
    with its correct wood material.

    Magnitude, measured rather than guessed (sweep at 0, 1e-7, 1e-6, 3e-6, 1e-5, 3e-5,
    1e-4, 3e-4, 1e-3 m):

      * The failure is an exact tie, not an epsilon threshold -- only offset == 0 goes
        black (37% of the solid pixels); every nonzero offset down to 1e-7 m is clean.
      * It is also renderer-dependent: it reproduces on the CPU/Embree path but not on
        OptiX, so it would come back the moment someone rendered with --device CPU.
      * 1e-4 m is 1000x above the smallest offset that works, yet only ~0.15 px of
        vertical displacement at the 3000 px default, so it cannot read as a seam.  The
        very small offsets are rejected on margin: at 1e-7 m the separation is comparable
        to float32 vertex precision across a 1.5 m slab.

    This is applied in Blender and nowhere else on purpose.  Drake's scene -- the IK
    problem, the collision checks and the exported Meshcat HTML -- keeps both tables at
    z = 0, so no solver result depends on a cosmetic offset.
    """
    moved = 0
    for obj in bpy.data.objects:
        if obj.type == "MESH" and owning_model(obj) == "table2":
            obj.location.z -= offset
            moved += 1
    print(f"[tables] separated {moved} table2 object(s) by {offset:g} m to break the "
          f"coincident-face self-shadow tie")
    return moved


def tune_imported_material(material, alpha=None, metallic=0.05, roughness=0.50,
                           fallback_rgb=(0.55, 0.56, 0.58)):
    """
    Keep an imported material's base colour, adjust only its shading parameters.

    Used for the UR links, whose four real colours arrive with the geometry.  Replacing
    the material outright -- as every other model here does -- would flatten them back to
    one grey, which is exactly the bug this pipeline used to have.
    """
    if not material.use_nodes:
        material.use_nodes = True
    bsdf = next((n for n in material.node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
    if bsdf is None:
        return False
    base = bsdf.inputs["Base Color"].default_value
    if max(base[0], base[1], base[2]) <= 0.0:
        bsdf.inputs["Base Color"].default_value = (*fallback_rgb, 1.0)
    if alpha is not None:
        bsdf.inputs["Alpha"].default_value = alpha
    bsdf.inputs["Metallic"].default_value = metallic
    bsdf.inputs["Roughness"].default_value = roughness
    material.blend_method = "BLEND"
    return True


def restyle_scene(arm_alpha=None, wood_table=True):
    """
    Restyle the imported materials per Drake model.

    Two different treatments:

      * ur5e -- the imported per-slot base colours are the real UR link colours and are
        kept; only alpha, metallic and roughness are set.
      * everything else -- the import carries a flat placeholder colour, so the material
        is replaced wholesale.  Alpha is re-applied here rather than inherited, because
        the arms and gripper bodies must stay transparent or the three overlapping
        configurations become unreadable.

    Anything that reaches the end unmatched is reported, not silently skipped: a silently
    skipped table was how half the table came out pure black.
    """
    cache = {}
    counts = {}
    tuned_materials = set()
    unmatched = {}
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue

        model = owning_model(obj)

        if not obj.data.materials:
            # No material slot at all.  Tables still need the wood material, so append
            # one instead of dropping the object on the floor.
            if model in TABLE_MODELS and wood_table:
                material = cache.get("wood") or build_wood_material()
                cache["wood"] = material
                obj.data.materials.append(material)
                counts["table"] = counts.get("table", 0) + 1
            else:
                print(f"[materials] WARNING: {obj.name} (model={model}) has no material "
                      f"slots and was left unstyled")
            continue

        existing = obj.data.materials[0].name.lower()
        if FINGER_MATERIAL_HINT in existing:
            # Keep the imported blue, but not the imported alpha.  The fingers arrive at
            # 0.5 from the Drake scene, which is right when every arm is ghosted and wrong
            # here, where the one arm is solid and see-through fingertips read as a bug.
            for slot_material in obj.data.materials:
                if slot_material is None or not slot_material.use_nodes:
                    continue
                for node in slot_material.node_tree.nodes:
                    if node.type == "BSDF_PRINCIPLED":
                        node.inputs["Alpha"].default_value = 1.0
                        break
                slot_material.blend_method = "OPAQUE"
            counts["finray (recoloured opaque)"] = \
                counts.get("finray (recoloured opaque)", 0) + 1
            continue

        if model in TABLE_MODELS:
            if not wood_table:
                continue
            material = cache.get("wood") or build_wood_material()
            cache["wood"] = material
        elif model == "ur5e":
            # Keep each slot's own colour; only restyle it.
            alpha = arm_alpha if arm_alpha is not None else MATERIALS["ur5e"][1]
            _, _, metallic, roughness = MATERIALS["ur5e"]
            for slot_material in obj.data.materials:
                if slot_material is None or slot_material.name in tuned_materials:
                    continue
                tune_imported_material(slot_material, alpha=alpha, metallic=metallic,
                                       roughness=roughness,
                                       fallback_rgb=MATERIALS["ur5e"][0])
                tuned_materials.add(slot_material.name)
            counts["ur5e (recoloured in place)"] = \
                counts.get("ur5e (recoloured in place)", 0) + 1
            continue
        elif model in MATERIALS:
            rgb, alpha, metallic, roughness = MATERIALS[model]
            material = cache.get(model) or build_principled(
                f"figure_{model}", rgb, alpha, metallic, roughness)
            cache[model] = material
        else:
            unmatched[str(model)] = unmatched.get(str(model), 0) + 1
            continue

        for slot in range(len(obj.data.materials)):
            obj.data.materials[slot] = material
        counts[model] = counts.get(model, 0) + 1

    print("[materials] restyled: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    if unmatched:
        print("[materials] WARNING: unmatched objects kept their imported material: "
              + ", ".join(f"model={k}: {v}" for k, v in sorted(unmatched.items())))
    print(f"[materials] ur5e distinct colours kept: "
          f"{sorted(_material_colours(tuned_materials))}")
    return counts


def _material_colours(names):
    """Rounded base colours of the named materials, for the log line above."""
    colours = set()
    for name in names:
        material = bpy.data.materials.get(name)
        if material is None or not material.use_nodes:
            continue
        bsdf = next((n for n in material.node_tree.nodes
                     if n.type == "BSDF_PRINCIPLED"), None)
        if bsdf is not None:
            c = bsdf.inputs["Base Color"].default_value
            colours.add((round(c[0], 3), round(c[1], 3), round(c[2], 3)))
    return colours


def scene_points(name_filter=None):
    """World-space bounding-box corners of the imported meshes."""
    points = []
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        if name_filter is not None and not name_filter(obj.name):
            continue
        points.extend(obj.matrix_world @ Vector(c) for c in obj.bound_box)
    return points


def setup_camera(position, target, lens, auto_frame=True, frame_radius=0.30):
    camera_data = bpy.data.cameras.new("figure_camera")
    camera_data.lens = lens
    camera = bpy.data.objects.new("figure_camera", camera_data)
    bpy.context.scene.collection.objects.link(camera)

    position = Vector(position)
    target = Vector(target)
    direction = (target - position).normalized()
    camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()

    if auto_frame:
        # Frame a sphere of radius `margin` around the target rather than the whole
        # scene: the figure is a close-up of the grasp, with the arms deliberately
        # running out of frame.  Fitting every object would pull the camera back until
        # the mug is a speck.  The importer flattens object names, so there is nothing
        # to select on anyway -- but the mug's world position is known, and it is the
        # subject.
        scene = bpy.context.scene
        aspect = scene.render.resolution_x / scene.render.resolution_y
        half_w = math.atan(camera_data.sensor_width / (2 * lens))
        half_h = math.atan(math.tan(half_w) / max(aspect, 1e-9))
        distance = frame_radius / math.sin(min(half_w, half_h))
        camera.location = target - direction * distance
    else:
        camera.location = position

    bpy.context.scene.camera = camera
    return camera


# Framing radius the light rig below was authored against, in metres.  The offsets and
# wattages are scaled off this: offsets linearly, so the rig keeps its shape relative to
# the subject, and power quadratically, because an area light's illuminance falls off with
# the square of the distance.  Without the scaling a rig tuned for a 0.3 m close-up leaves
# the far side of a 1 m scene black.
LIGHTING_REFERENCE_RADIUS = 0.30


def setup_lighting(target, frame_radius=LIGHTING_REFERENCE_RADIUS):
    """Three-point lighting aimed at the subject, plus a soft ambient world."""
    world = bpy.data.worlds.new("figure_world") if not bpy.data.worlds else bpy.data.worlds[0]
    bpy.context.scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg is not None:
        bg.inputs["Color"].default_value = (0.16, 0.17, 0.19, 1.0)
        bg.inputs["Strength"].default_value = 0.7

    focus = Vector(target)
    scale = max(frame_radius, 1e-6) / LIGHTING_REFERENCE_RADIUS
    # (name, offset from focus, power W, radius m) -- all authored at frame radius 0.30
    lights = [
        ("key",  Vector(( 0.9, -0.9, 1.3)), 45.0, 1.2),
        ("fill", Vector((-1.0, -0.7, 0.7)), 18.0, 1.5),
        ("rim",  Vector((-0.4,  1.1, 1.0)), 30.0, 1.0),
    ]
    print(f"[lighting] frame radius {frame_radius:.2f} m -> offsets x{scale:.2f}, "
          f"power x{scale ** 2:.2f}")
    for name, offset, power, radius in lights:
        offset = offset * scale
        data = bpy.data.lights.new(name, type="AREA")
        data.energy = power * scale ** 2
        data.size = radius * scale
        obj = bpy.data.objects.new(name, data)
        bpy.context.scene.collection.objects.link(obj)
        obj.location = focus + offset
        obj.rotation_euler = (-offset).to_track_quat("-Z", "Y").to_euler()


def enable_gpu_devices(preferred=("OPTIX", "CUDA", "HIP", "ONEAPI")):
    """
    Actually turn the GPU on, and say so.

    Setting `scene.cycles.device = "GPU"` alone does nothing: Cycles renders on whatever
    devices are flagged `use` under the addon preferences, and by default none are, so it
    silently falls back to CPU while the log claims GPU.  Both halves are needed --
    `compute_device_type` picks the backend, and each device must be individually enabled.

    Note the same physical GPU is enumerated once per backend it supports (this box lists
    the 3080 Ti under both CUDA and OPTIX), so devices must be filtered by `d.type`, never
    by name.

    Returns the backend name that ended up with at least one enabled device, or None.
    """
    prefs = bpy.context.preferences.addons["cycles"].preferences
    for backend in preferred:
        try:
            prefs.compute_device_type = backend
        except TypeError:
            continue  # backend not compiled into this Blender
        prefs.get_devices()
        enabled = [d.name for d in prefs.devices if d.type == backend]
        for device in prefs.devices:
            device.use = (device.type == backend)
        if enabled:
            print(f"[device] {backend}: {len(enabled)} device(s) enabled "
                  f"({', '.join(sorted(set(enabled)))})")
            return backend
        print(f"[device] {backend}: no devices found")
    return None


def setup_render(args):
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = args.samples
    scene.cycles.use_denoising = True
    if args.device == "GPU":
        backend = enable_gpu_devices()
        if backend is None:
            print("[device] WARNING: no usable GPU backend; rendering on CPU instead")
            scene.cycles.device = "CPU"
        else:
            scene.cycles.device = "GPU"
            if backend == "OPTIX":
                # OptiX's denoiser is far faster than OpenImageDenoise and, checked
                # against a CPU/OIDN render of this scene, does not shift the flat
                # figure colours.
                scene.cycles.denoiser = "OPTIX"
    else:
        scene.cycles.device = "CPU"
    print(f"[device] cycles.device={scene.cycles.device}, "
          f"denoiser={scene.cycles.denoiser}")

    # Blender defaults to the AgX view transform, which desaturates and washes out the
    # flat colours a technical figure wants.  Make it explicit and controllable.
    scene.view_settings.view_transform = args.view_transform
    scene.view_settings.exposure = args.exposure
    scene.view_settings.look = "None"

    scene.render.resolution_x, scene.render.resolution_y = args.resolution
    scene.render.resolution_percentage = 100

    # Blender's border is measured from the bottom-left and --crop from the top-left, so
    # the vertical pair swaps and flips.  use_crop_to_border makes the saved PNG the
    # border itself rather than the full frame with the rest left transparent.
    if args.no_crop:
        scene.render.use_border = False
    else:
        left, top, right, bottom = args.crop
        scene.render.use_border = True
        scene.render.use_crop_to_border = True
        scene.render.border_min_x, scene.render.border_max_x = left, right
        scene.render.border_min_y, scene.render.border_max_y = 1.0 - bottom, 1.0 - top
        width, height = args.resolution
        print(f"[crop] ({left:.4f}, {top:.4f}) to ({right:.4f}, {bottom:.4f}) -> "
              f"{round((right - left) * width)}x{round((bottom - top) * height)} px "
              f"of {width}x{height}")
    scene.render.film_transparent = not args.opaque_background
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.filepath = args.out

    # Alpha-blended materials need the transparent shadow/light paths that Cycles gives
    # by default, but bump transmission bounces so stacked transparent arms stay readable.
    scene.cycles.transparent_max_bounces = 32
    scene.cycles.max_bounces = 16


def main():
    args = parse_args()

    print(f"[import] {args.html}")
    enable_importer()
    import_scene(args.html)
    n_objects = len([o for o in bpy.data.objects if o.type == "MESH"])
    print(f"[import] {n_objects} mesh objects")
    if n_objects == 0:
        raise SystemExit("Import produced no geometry; nothing to render.")

    separate_coplanar_tables()
    if not args.no_restyle:
        # Rebuild the per-colour material split first; restyle_scene then tunes those
        # materials in place rather than flattening them.
        assign_ur_link_colors()
        restore_ur_flat_shading()
        restyle_scene(arm_alpha=args.arm_alpha, wood_table=not args.no_wood_table)
    report_transparency()

    setup_render(args)   # resolution must be set before the camera fits to the aspect

    camera_pos = args.camera_pos
    if args.camera_azimuth is not None or args.camera_elevation is not None:
        target = Vector(args.camera_target)
        offset = Vector(args.camera_pos) - target
        flat = math.hypot(offset.x, offset.y)
        azimuth = math.radians(args.camera_azimuth) if args.camera_azimuth is not None \
            else math.atan2(offset.y, offset.x)
        elevation = math.radians(args.camera_elevation) if args.camera_elevation is not None \
            else math.atan2(offset.z, flat)
        distance = args.camera_distance if args.camera_distance is not None else offset.length
        camera_pos = target + Vector((
            distance * math.cos(elevation) * math.cos(azimuth),
            distance * math.cos(elevation) * math.sin(azimuth),
            distance * math.sin(elevation),
        ))
        print(f"[camera] azimuth {math.degrees(azimuth):+.1f} deg, "
              f"elevation {math.degrees(elevation):+.1f} deg -> "
              f"({camera_pos.x:.3f}, {camera_pos.y:.3f}, {camera_pos.z:.3f})")

    setup_camera(camera_pos, args.camera_target, args.lens,
                 auto_frame=not args.no_auto_frame, frame_radius=args.frame_radius)
    setup_lighting(args.camera_target, frame_radius=args.frame_radius)

    print(f"[render] {args.resolution[0]}x{args.resolution[1]}, "
          f"{args.samples} samples, device {args.device}")
    bpy.ops.render.render(write_still=True)
    print(f"[render] wrote {args.out}")


if __name__ == "__main__":
    main()
