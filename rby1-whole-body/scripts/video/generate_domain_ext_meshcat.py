"""Generate the meshcat HTML + annotation data for the domain-extension and
boundary-reachability segments of the overview video.

IIWA + Schunk WSG tracking a straight-line target that walks out past the edge of
the reachable workspace.  Both segments use *this one* trajectory -- that was
already true of the VTK renders (``render_boundary_reach.py``'s docstring calls it
out) and staying that way means the viewer sees the same motion twice, from the
same camera, once annotated with the least-squares residual and once with the
regularized boundary constraint.

Two artefacts, so the Blender render never has to re-derive kinematics:

* ``video/v2_domain_extension.html`` -- static meshcat recording at 30 fps.  It
  contains *only* the robot: the ghost gripper, the residual line and every label
  are 2D overlays composited onto the Blender frames afterwards, projected with
  the Blender camera (see ``segment_camera.py``).
* ``video/v2_domain_ext_annotations.npz`` -- per-frame ``target``, ``p_actual``,
  ``residual``, ``manipulability`` and ``constraint`` arrays, indexed the same way
  the animation frames are (frame i <-> row i, because the recording and the
  Blender import both run at 30 fps).

Usage:
    .venv/bin/python scripts/video/generate_domain_ext_meshcat.py
    .venv/bin/python scripts/video/generate_domain_ext_meshcat.py --end-x 1.2
"""

import argparse
import os
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
IIWA_REPO = os.path.join(REPO, os.pardir, "iiwa-bimanual")
sys.path.insert(0, os.path.join(IIWA_REPO, "src"))

from iiwa_analytic_ik import Analytic_IK_7DoF  # noqa: E402

from pydrake.all import (  # noqa: E402
    AddMultibodyPlantSceneGraph, DiagramBuilder, MeshcatVisualizer,
    MeshcatVisualizerParams, Parser, RigidTransform, Role, RollPitchYaw,
    StartMeshcat,
)
from pydrake.multibody.tree import FixedOffsetFrame, JacobianWrtVariable  # noqa: E402

FPS = 30
DURATION = 12.0

# Same target sweep the VTK segments used, so the converted segments show the
# same motion: the arm tracks a straight line until the line leaves the
# reachable workspace, then holds its best least-squares configuration.
START_POS = np.array([0.4, 0.0, 0.7])
# How far the target walks. The reachable set ends at x = 0.745 m along this
# line (measured by sweeping the analytic IK and evaluating FK: the residual is
# 0.0 mm up to 0.725 and 9.6 mm at 0.750), and past it the residual grows very
# nearly linearly at 0.94 mm per mm of target travel. x = 1.07 therefore ends
# the segment about 305 mm short of the target -- a divergence you can read off
# the screen, where the old 0.95 endpoint left only 192 mm.
DEFAULT_END_X = 1.07
EE_RPY = RollPitchYaw(np.pi / 2, 0, 0)
GC, PSI = (1, 1, 1), 0.0

# Regularization for the boundary constraint -log det(J J^T + eps I).
EPSILON = 1e-6

# Above this residual the target has left the reachable workspace. Nothing is
# shown or hidden on it any more -- both grippers are visible throughout -- but
# it must match RESIDUAL_VISIBLE_M in render_domain_extension_blender.py, which
# gates the residual dimension and the caption on the same threshold.
TARGET_VISIBLE_M = 0.01

WSG_ATTACH_OFFSET = RigidTransform(
    RollPitchYaw(np.radians([90, 0, 68])), [0, 0, 0.09]
)

# The target gripper is drawn a few millimetres to one side of the pose it
# represents. While the target is reachable the two grippers are the same mesh
# on the same transform, and two coincident surfaces make Cycles speckle where
# it cannot decide which one it hit. 4 mm is about two pixels at this framing --
# invisible as a displacement, and well under the 10 mm at which the segment
# starts calling the target unreachable -- but far more than any ray-tracing
# tolerance, so the ambiguity is gone. +y is directly away from the camera.
TARGET_RENDER_OFFSET = RigidTransform([0.0, 0.004, 0.0])


def build_scene(meshcat):
    builder = DiagramBuilder()
    plant, sg = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
    parser = Parser(plant)
    parser.package_map().AddPackageXml(os.path.join(IIWA_REPO, "package.xml"))

    iiwa_urdf = os.path.join(
        IIWA_REPO, "models", "iiwa14_convex_decimated_collision.urdf"
    )
    iiwa = parser.AddModels(iiwa_urdf)[0]
    plant.WeldFrames(
        plant.world_frame(), plant.GetFrameByName("base", iiwa), RigidTransform()
    )

    wsg_sdf = os.path.join(
        IIWA_REPO, "models", "wsg_finray",
        "wsg50_110_finray_fingers_box_collision.sdf",
    )
    wsg = parser.AddModels(wsg_sdf)[0]
    wsg_attach = plant.AddFrame(
        FixedOffsetFrame(
            "wsg_attach", plant.GetBodyByName("iiwa_link_7", iiwa).body_frame(),
            WSG_ATTACH_OFFSET,
        )
    )
    plant.WeldFrames(wsg_attach, plant.GetFrameByName("body", wsg))

    # A second gripper, deliberately NOT welded to anything, so it is a floating
    # body whose pose can be set per frame. This is the *target*: the pose the
    # arm is being asked to reach. Showing the target as a gripper rather than
    # as an abstract marker is what makes the divergence at the workspace
    # boundary legible -- the two grippers coincide exactly while the target is
    # reachable, and visibly separate once it is not.
    target_parser = Parser(plant, model_name_prefix="target")
    target_parser.package_map().AddPackageXml(
        os.path.join(IIWA_REPO, "package.xml"))
    wsg_target = target_parser.AddModels(wsg_sdf)[0]

    plant.Finalize()

    # Illustration role only, and we publish this subsystem alone, so the
    # collision geometry never reaches meshcat and cannot leak into the export.
    params = MeshcatVisualizerParams()
    params.role = Role.kIllustration
    params.prefix = "visual"
    params.publish_period = 1.0 / FPS
    vis = MeshcatVisualizer.AddToBuilder(builder, sg, meshcat, params)

    diagram = builder.Build()
    return diagram, plant, iiwa, wsg_target, vis


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--end-x", type=float, default=DEFAULT_END_X,
                    help="where along +x the target stops; the residual it "
                         "ends at is printed at the end of the run "
                         f"(default {DEFAULT_END_X} m, about 305 mm)")
    args = ap.parse_args()
    end_pos = np.array([args.end_x, START_POS[1], START_POS[2]])

    out_html = os.path.join(REPO, "video", "v2_domain_extension.html")
    out_npz = os.path.join(REPO, "video", "v2_domain_ext_annotations.npz")
    manip_path = os.path.join(REPO, "video", "v2_manipulability.npy")

    print("Starting meshcat...")
    meshcat = StartMeshcat()

    print("Building IIWA + WSG scene...")
    diagram, plant, iiwa, wsg_target, vis = build_scene(meshcat)
    ctx = diagram.CreateDefaultContext()
    plant_ctx = plant.GetMyMutableContextFromRoot(ctx)
    vis_ctx = vis.GetMyContextFromRoot(ctx)

    target_body = plant.GetBodyByName("body", wsg_target)

    ik = Analytic_IK_7DoF()
    ee_frame = plant.GetFrameByName("iiwa_link_7", iiwa)
    n_frames = int(DURATION * FPS)

    print(f"Recording {n_frames} frames at {FPS} fps...")
    meshcat.DeleteRecording()
    meshcat.StartRecording(frames_per_second=FPS,
                           set_visualizations_while_recording=False)

    targets, actuals, residuals, manips, constraints = [], [], [], [], []
    last_valid_q = None

    for i in range(n_frames):
        t = i / FPS
        target = START_POS + (i / n_frames) * (end_pos - START_POS)
        X_target = RigidTransform(EE_RPY, target)

        try:
            q = ik.IK(X_target, GC, PSI)
            if q is not None and not np.any(np.isnan(q)):
                last_valid_q = np.asarray(q, dtype=float).copy()
        except Exception:
            pass

        plant.SetPositions(plant_ctx, iiwa,
                           last_valid_q if last_valid_q is not None else np.zeros(7))

        # ── The target gripper ──────────────────────────────────────────────
        #
        # On screen for the whole segment, riding the commanded pose with the
        # same offset the bolted gripper has. It used to be parked out of frame
        # until the residual crossed 10 mm, which meant a second gripper simply
        # appeared out of nothing partway through -- the pop was the most
        # eye-catching event in the shot, and it hid the very thing the segment
        # is about: the two grippers are *together* while the target is
        # reachable and separate only once it is not.
        #
        # It is posed, never hidden, and that is load-bearing. Setting `visible`
        # on a *leaf geometry* path detaches it from its frame's transform
        # track, so all three target meshes import stuck at the world origin,
        # buried inside the arm's base -- and these paths are leaves, because
        # the target's geometry hangs directly off the model's frame.
        # (Visibility on a *group* path does survive; that is how
        # generate_stability_meshcat.py hides its scenery boxes.) Verified by
        # importing the export into Blender and reading object positions; do not
        # verify this by grepping the HTML, which is base64 msgpack.
        X_actual = plant.CalcRelativeTransform(
            plant_ctx, plant.world_frame(), ee_frame)
        p_actual = X_actual.translation()
        residual = float(np.linalg.norm(p_actual - target))

        plant.SetFreeBodyPose(
            plant_ctx, target_body,
            TARGET_RENDER_OFFSET @ X_target @ WSG_ATTACH_OFFSET)

        J = plant.CalcJacobianSpatialVelocity(
            plant_ctx, JacobianWrtVariable.kQDot, ee_frame, [0, 0, 0],
            plant.world_frame(), plant.world_frame(),
        )
        JJT = J @ J.T
        det = np.linalg.det(JJT)
        det_reg = np.linalg.det(JJT + EPSILON * np.eye(JJT.shape[0]))

        targets.append(target.copy())
        actuals.append(p_actual.copy())
        residuals.append(residual)
        manips.append(float(max(det, 1e-20)))
        constraints.append(float(-np.log(max(det_reg, 1e-30))))

        # The visualizer records transforms into the animation at the context
        # time; publishing only this subsystem keeps the proximity tree out.
        ctx.SetTime(t)
        vis.ForcedPublish(vis_ctx)

    meshcat.StopRecording()
    meshcat.PublishRecording()

    residuals = np.array(residuals)
    constraints = np.array(constraints)
    print(f"Residual range: {residuals.min()*1000:.1f} to "
          f"{residuals.max()*1000:.1f} mm (target ends at x = {args.end_x:.3f} m)")
    print(f"Constraint range: {constraints.min():.2f} to {constraints.max():.2f}")

    print("Exporting StaticHtml...")
    os.makedirs(os.path.dirname(out_html), exist_ok=True)
    with open(out_html, "w") as f:
        f.write(meshcat.StaticHtml())
    print(f"Wrote {out_html} ({os.path.getsize(out_html)/1e6:.1f} MB)")

    np.savez(out_npz, fps=np.array(FPS), n_frames=np.array(n_frames),
             target=np.array(targets), p_actual=np.array(actuals),
             residual=residuals, manipulability=np.array(manips),
             constraint=constraints, epsilon=np.array(EPSILON))
    print(f"Wrote {out_npz}")

    # Kept for compatibility with the VTK scripts, which read this file.
    np.save(manip_path, np.array(manips))
    print(f"Wrote {manip_path}")


if __name__ == "__main__":
    main()
