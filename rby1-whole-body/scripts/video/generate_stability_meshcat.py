"""Generate the meshcat HTML + annotation data for the static-stability segment.

The RBY1 replaying grid point 00 as a complete pick-and-place: it approaches the
box with its grippers open, closes them on the grasp, carries the box, sets it
down and opens again. Each leg is resampled to its own duration at
``PLAYBACK_SPEED``, so the replay runs at a stated multiple of real time.

That motion comes from ``plan_grid.build_viz``'s pair of
diagrams -- empty-handed with the grippers open, box-attached with them closed --
selected per leg by the leg's ``with_box`` tag, exactly as ``grid_visualizer.py``
replays a point interactively. The box is meshcat scenery in all three of its
states -- waiting at the pick pose, carried in the gripper, resting where it was
placed -- with one root per state and only one of them visible at a time.

Two artefacts:

* ``video/v2_rby1_stability.html`` -- static meshcat recording at 30 fps,
  containing only the illustration geometry.  Only the ``visual``
  MeshcatVisualizer is published, so the proximity/collision tree never reaches
  meshcat and cannot end up in the export.
* ``video/v2_stability_annotations.npz`` -- per-frame CoM, base pose and the
  stability verdicts, plus the support-polygon vertices in the base frame.  The
  support polygon and the CoM marker are drawn as overlays on the Blender frames
  (projected with the Blender camera), so this is the data that anchors them;
  frame i of the render corresponds to row i here.

Usage:
    .venv/bin/python scripts/video/generate_stability_meshcat.py
    .venv/bin/python scripts/video/generate_stability_meshcat.py --index 5
"""

import argparse
import os
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "scripts"))

from pydrake.all import StartMeshcat  # noqa: E402
from pydrake.geometry import Rgba  # noqa: E402
from pydrake.math import RigidTransform, RollPitchYaw  # noqa: E402

from rby1_opt_ik import (  # noqa: E402
    DEFAULT_SUPPORT_POLYGON_INSET, _com_instances,
    _com_support_polygon_residuals, _stability_constraint_ub,
    inset_support_polygon_xy, support_polygon_xyzs,
)
from plan_format.plan_io import load_plan  # noqa: E402
import plan_grid as E  # noqa: E402
import grid_visualizer as G  # noqa: E402

FPS = 30

# How much faster than real time the replay runs. Grid point 00 takes 37.8 s on
# the robot, so at 2x the segment is about 19 s -- long enough to watch the CoM
# move across the support polygon, which is the whole point of the shot. The
# previous fixed 30-frames-per-leg sampling ran it at roughly 5x, fast enough
# that the CoM looked like it teleported. The rendered speed is written into the
# annotation npz so the overlay can label it rather than repeating the number.
PLAYBACK_SPEED = 2.0

# The box while it is being carried, drawn as scenery beside grid_visualizer's
# pick and placed boxes rather than using the plant's welded held box -- see the
# comment where the scenery is built.
CARRIED_ROOT = "grid_viz/carried_box"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index", type=int, default=0, help="grid point to replay")
    args = ap.parse_args()

    out_html = os.path.join(REPO, "video", "v2_rby1_stability.html")
    out_npz = os.path.join(REPO, "video", "v2_stability_annotations.npz")

    plan_path = os.path.join(REPO, "plans", "grid_cache", f"point_{args.index:02d}.pkl")
    _, meta = load_plan(plan_path)
    legs = meta["legs"]

    print("Starting meshcat...")
    meshcat = StartMeshcat()

    # ── The scene: two diagrams, and the box as scenery ─────────────────────
    #
    # build_viz gives the same pair grid_visualizer.py replays with: an
    # empty-handed diagram whose grippers are OPEN, and a box-attached diagram
    # whose grippers are CLOSED. Both write the same meshcat paths, so
    # selecting one per leg is what makes the grippers visibly close on the
    # grasp and open again on release. The segment used to publish a single
    # open-gripper diagram with the box welded to the table as an obstacle, so
    # the robot reached into a box it never picked up.
    #
    # The pick and placed boxes are meshcat scenery rather than plant
    # obstacles, for the same reason: an obstacle is in the scene for the whole
    # recording, which would leave the box sitting at the pick pose while the
    # robot carries a copy of it away.
    print("Building Drake infrastructure...")
    viz = E.build_viz(meshcat)
    plant_b, pctx_b, _ctx_b, diagram_b, idxs_b = viz[True]
    ee_body = plant_b.GetBodyByName(
        f"ee_{E.HAND}", plant_b.GetModelInstanceByName(f"{E.HAND}_arm"))

    # ── Why the carried box is scenery too ──────────────────────────────────
    #
    # The box-attached diagram already draws a held box welded to ``ee_right``,
    # and the interactive viewer simply toggles that geometry's visibility per
    # leg. That does not survive the export to Blender: setting ``visible`` on a
    # *leaf geometry* path detaches it from its frame's transform track, so the
    # importer places it at a fixed world pose for the entire animation -- the
    # carried box hung motionless beside the torso while the robot picked up
    # nothing. Visibility set on a *group* path (``grid_viz/pick_box``) comes
    # through correctly, transform and all.
    #
    # So the plant's held box is hidden for the whole recording and the carried
    # box is drawn as a third scenery box under its own root, posed per frame by
    # the same FK composition the placed box uses. Group path, group transform:
    # both channels survive.
    rgba = Rgba(*E.BOX_COLOR)
    for root in (G.PICK_ROOT, G.PLACED_ROOT, CARRIED_ROOT):
        G.add_open_box(meshcat, root, rgba)
    held_paths = [p for p in G._held_box_meshcat_paths(diagram_b)
                  if meshcat.HasPath(p)]
    if not held_paths:
        print("WARNING: could not locate the held box's meshcat paths; the "
              "plant's held box will stay visible during the empty-handed legs.")

    kinds = G.leg_scene_kinds(legs)

    # Static boxes are posed once, outside the recording: they do not move, and
    # only their visibility belongs in the animation.
    meshcat.SetTransform(G.PICK_ROOT, E.box_pose_for(meta["bx"], meta["by"]))
    carried = [lg for lg in legs if lg.get("with_box")]
    if carried:
        q_place_end = np.asarray(carried[-1]["q"])[-1]
        q_full = plant_b.GetPositions(pctx_b)
        q_full[idxs_b] = q_place_end
        plant_b.SetPositions(pctx_b, q_full)
        meshcat.SetTransform(
            G.PLACED_ROOT,
            plant_b.EvalBodyPoseInWorld(pctx_b, ee_body)
            @ RigidTransform(RollPitchYaw(E.RPY), E.OFFSET))

    # Sample each leg, keeping which leg (and so which box state and which
    # diagram) every frame belongs to.
    #
    # The frame count comes from the leg's *duration*, not from its sample
    # count: the legs are time-parameterized by TOPPRA and sampled at whatever
    # rate they were cached at, so taking a fixed number of samples per leg made
    # a 1.5 s leg and an 11.9 s leg take the same time on screen. Resampling by
    # duration keeps the relative pace of the six legs, and makes PLAYBACK_SPEED
    # mean what it says.
    frames = []   # (q23, with_box, kind)
    for leg, kind in zip(legs, kinds):
        q = np.asarray(leg["q"], dtype=float)
        n = max(2, int(round(float(leg["duration"]) * FPS / PLAYBACK_SPEED)))
        for j in np.linspace(0, len(q) - 1, n):
            frames.append((q[int(round(j))], bool(leg.get("with_box")), kind))
    n_frames = len(frames)
    motion_s = sum(float(lg["duration"]) for lg in legs)
    print(f"{len(legs)} legs, {motion_s:.1f}s of motion -> {n_frames} frames "
          f"({n_frames / FPS:.2f}s at {FPS} fps, {PLAYBACK_SPEED:.1f}x)")
    for leg, kind in zip(legs, kinds):
        print(f"  {leg['name']:<14} box: {kind}")

    com_instances = list(_com_instances(plant_b))
    ub_conservative = _stability_constraint_ub(DEFAULT_SUPPORT_POLYGON_INSET)

    meshcat.DeleteRecording()
    meshcat.StartRecording(frames_per_second=FPS,
                           set_visualizations_while_recording=False)

    coms, base_xyts, cons_flags, nom_flags, rear_margins = [], [], [], [], []

    for i, (q23, with_box, kind) in enumerate(frames):
        plant, plant_ctx, ctx, diagram, idxs = viz[with_box]
        q_full = plant.GetPositions(plant_ctx)
        q_full[idxs] = q23
        plant.SetPositions(plant_ctx, q_full)

        # The CoM is evaluated on whichever plant posed this frame, so the
        # carried box's mass is included exactly on the legs that carry it --
        # which is the point of a stability segment.
        com = plant.CalcCenterOfMassPositionInWorld(
            plant_ctx, list(_com_instances(plant)))
        base_xyt = np.asarray(q23[:3], dtype=float)
        residuals = _com_support_polygon_residuals(com[:2], base_xyt)

        coms.append(np.asarray(com, dtype=float).copy())
        base_xyts.append(base_xyt.copy())
        cons_flags.append(bool(np.all(residuals <= ub_conservative)))
        nom_flags.append(bool(np.all(residuals <= 0)))
        rear_margins.append(float(-residuals[-1]) if len(residuals) else 0.0)

        t = i / FPS
        # Re-asserted every frame, not only at leg boundaries: a meshcat
        # animation interpolates keyframe tracks, and a boolean with keyframes
        # only at transitions is at the mercy of that interpolation.
        pick_v, placed_v, held_v = G.SCENE_VISIBILITY[kind]
        meshcat.SetProperty(G.PICK_ROOT, "visible", pick_v, time_in_recording=t)
        meshcat.SetProperty(G.PLACED_ROOT, "visible", placed_v,
                            time_in_recording=t)
        if held_v:
            # Posed from *this* frame's configuration, so the box tracks the
            # gripper through the carry instead of sitting at one pose. A leg
            # that carries the box is by definition posed in the box-attached
            # diagram, so `plant` here is the plant `ee_body` belongs to.
            assert plant is plant_b
            meshcat.SetTransform(
                CARRIED_ROOT,
                plant.EvalBodyPoseInWorld(plant_ctx, ee_body)
                @ RigidTransform(RollPitchYaw(E.RPY), E.OFFSET),
                time_in_recording=t)
        meshcat.SetProperty(CARRIED_ROOT, "visible", held_v,
                            time_in_recording=t)
        # Constant False, never animated: the plant's own held box is replaced
        # by the scenery box above, and a leaf path that is always invisible can
        # safely lose its transform track.
        for path in held_paths:
            meshcat.SetProperty(path, "visible", False, time_in_recording=t)

        # Publish the visual visualizer alone -- NOT the whole diagram -- so the
        # collision visualizer's geometry never reaches meshcat (the same reason
        # grid_visualizer.py does it this way).
        vis = diagram.GetSubsystemByName("meshcat_visualizer(visual)")
        ctx.SetTime(t)
        vis.ForcedPublish(vis.GetMyContextFromRoot(ctx))

        if i % 30 == 0:
            print(f"  frame {i}/{n_frames}")

    meshcat.StopRecording()
    meshcat.PublishRecording()

    print("Exporting StaticHtml...")
    with open(out_html, "w") as f:
        f.write(meshcat.StaticHtml())
    print(f"Wrote {out_html} ({os.path.getsize(out_html)/1e6:.1f} MB)")

    np.savez(
        out_npz,
        fps=np.array(FPS), n_frames=np.array(n_frames),
        index=np.array(args.index),
        # What the overlay labels the segment with, so the number on screen
        # cannot drift away from the number the frames were sampled at.
        speed=np.array(PLAYBACK_SPEED),
        motion_s=np.array(motion_s),
        com=np.array(coms), base_xyt=np.array(base_xyts),
        is_conservative=np.array(cons_flags), is_nominal=np.array(nom_flags),
        rear_margin=np.array(rear_margins),
        # Support polygon in the *base* frame; the overlay rotates/translates it
        # by base_xyt per frame, so it never needs rby1_opt_ik.
        nominal_verts=np.asarray(support_polygon_xyzs[:, :2], dtype=float),
        inset_verts=np.asarray(
            inset_support_polygon_xy(DEFAULT_SUPPORT_POLYGON_INSET), dtype=float),
        inset=np.array(DEFAULT_SUPPORT_POLYGON_INSET),
    )
    print(f"Wrote {out_npz}")
    n_cons = int(np.sum(cons_flags))
    print(f"CoM inside the conservative polygon on {n_cons}/{n_frames} frames; "
          f"rear margin {min(rear_margins)*1000:.1f}..{max(rear_margins)*1000:.1f} mm")


if __name__ == "__main__":
    main()
