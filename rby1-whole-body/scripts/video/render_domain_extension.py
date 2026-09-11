"""Render the IK domain extension visualization for Video 2.  SUPERSEDED.

SUPERSEDED by ``generate_domain_ext_meshcat.py`` + ``render_domain_extension_blender.py``,
which produce ``video/v2_domain_extension.mp4`` through the meshcat -> Blender
pipeline so that every 3D shot in the overview video is Blender-rendered and
visually consistent.  This script still works and still writes the same file; it
is kept as the fallback if the Blender path breaks.  If you run it, be aware it
overwrites the Blender-rendered segment (and ``video/v2_manipulability.npy``).

IIWA + Schunk WSG tracking a straight-line target.  A ghost gripper shows
the desired EE pose; a measurement line visualises the least-squares
residual when the target exceeds the reachable workspace.

Usage:
    .venv/bin/python scripts/video/render_domain_extension.py
"""

import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pydrake.all import (
    AddMultibodyPlantSceneGraph, ClippingRange, ColorRenderCamera,
    CameraInfo, DiagramBuilder, MakeRenderEngineVtk, Parser,
    RenderCameraCore, RenderEngineVtkParams, RigidTransform,
    RollPitchYaw, RotationMatrix,
)
from pydrake.multibody.tree import JacobianWrtVariable

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
IIWA_REPO = os.path.join(REPO, os.pardir, "iiwa-bimanual")
sys.path.insert(0, os.path.join(IIWA_REPO, "src"))

from iiwa_analytic_ik import Analytic_IK_7DoF

RENDERER = "vtk"
WIDTH, HEIGHT = 1920, 1080
FPS = 30
DURATION = 12.0
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

WSG_ATTACH_OFFSET = RigidTransform(
    RollPitchYaw(np.radians([90, 0, 68])), [0, 0, 0.09]
)


def _look_at(eye, target, up=np.array([0.0, 0.0, 1.0])):
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(-up, forward)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    R = np.column_stack([right, down, forward])
    return RigidTransform(RotationMatrix(R), eye)


def build_scene():
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
    from pydrake.multibody.tree import FixedOffsetFrame
    wsg_attach = plant.AddFrame(
        FixedOffsetFrame(
            "wsg_attach", plant.GetBodyByName("iiwa_link_7", iiwa).body_frame(),
            WSG_ATTACH_OFFSET,
        )
    )
    plant.WeldFrames(wsg_attach, plant.GetFrameByName("body", wsg))

    plant.Finalize()
    diagram = builder.Build()
    return diagram, plant, sg, iiwa


def compute_trajectory(plant, iiwa, plant_ctx):
    ik_solver = Analytic_IK_7DoF()
    ee_frame = plant.GetFrameByName("iiwa_link_7", iiwa)
    n_frames = int(DURATION * FPS)

    start_pos = np.array([0.4, 0.0, 0.7])
    end_pos = np.array([0.95, 0.0, 0.7])
    ee_rpy = RollPitchYaw(np.pi / 2, 0, 0)
    GC, psi = (1, 1, 1), 0.0

    configs, targets, residuals, manip_values = [], [], [], []
    last_valid_q = None

    for i in range(n_frames):
        t_frac = i / n_frames
        target = start_pos + t_frac * (end_pos - start_pos)
        targets.append(target.copy())
        X_target = RigidTransform(ee_rpy, target)

        try:
            q = ik_solver.IK(X_target, GC, psi)
            if q is not None and not np.any(np.isnan(q)):
                last_valid_q = q.copy()
        except Exception:
            pass

        if last_valid_q is not None:
            configs.append(last_valid_q.copy())
            plant.SetPositions(plant_ctx, iiwa, last_valid_q)
            X_actual = plant.CalcRelativeTransform(
                plant_ctx, plant.world_frame(), ee_frame,
            )
            residuals.append(float(np.linalg.norm(X_actual.translation() - target)))
        else:
            configs.append(np.zeros(7))
            residuals.append(float(np.linalg.norm(target - start_pos)))

        J = plant.CalcJacobianSpatialVelocity(
            plant_ctx, JacobianWrtVariable.kQDot,
            ee_frame, [0, 0, 0],
            plant.world_frame(), plant.world_frame(),
        )
        manip_values.append(max(np.linalg.det(J @ J.T), 1e-20))

    return configs, targets, residuals, manip_values


def project_3d_to_2d(point_3d, X_WC, cam_info):
    p_camera = X_WC.inverse() @ point_3d
    if p_camera[2] <= 0:
        return None
    fx = cam_info.focal_x()
    fy = cam_info.focal_y()
    cx = cam_info.center_x()
    cy = cam_info.center_y()
    u = fx * p_camera[0] / p_camera[2] + cx
    v = fy * p_camera[1] / p_camera[2] + cy
    if 0 <= u < WIDTH and 0 <= v < HEIGHT:
        return int(u), int(v)
    return None


def main():
    out_path = os.path.join(REPO, "video", "v2_domain_extension.mp4")

    print("Building IIWA + WSG scene...")
    diagram, plant, sg, iiwa = build_scene()

    if not sg.HasRenderer(RENDERER):
        sg.AddRenderer(RENDERER, MakeRenderEngineVtk(RenderEngineVtkParams()))

    ctx = diagram.CreateDefaultContext()
    plant_ctx = plant.GetMyMutableContextFromRoot(ctx)
    sg_ctx = sg.GetMyMutableContextFromRoot(ctx)

    ee_frame = plant.GetFrameByName("iiwa_link_7", iiwa)
    cam_info = CameraInfo(WIDTH, HEIGHT, 0.8)
    X_WC = _look_at(np.array([1.2, -1.4, 1.0]), np.array([0.4, 0.0, 0.55]))
    core = RenderCameraCore(RENDERER, cam_info, ClippingRange(0.05, 20.0),
                            RigidTransform())
    camera = ColorRenderCamera(core, show_window=False)

    print("Computing IK trajectory...")
    configs, targets, residuals, manip_values = compute_trajectory(
        plant, iiwa, plant_ctx
    )

    np.save(os.path.join(REPO, "video", "v2_manipulability.npy"),
            np.array(manip_values))

    try:
        font = ImageFont.truetype(FONT_PATH, 28)
        font_small = ImageFont.truetype(FONT_PATH, 20)
        font_eq = ImageFont.truetype(FONT_PATH, 22)
    except OSError:
        font = font_small = font_eq = ImageFont.load_default()

    n_frames = len(configs)
    pipe = subprocess.Popen(
        ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{WIDTH}x{HEIGHT}", "-r", str(FPS), "-i", "pipe:0",
         "-c:v", "libx264", "-crf", "20", "-preset", "fast",
         "-pix_fmt", "yuv420p", "-an", out_path],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )

    print(f"Rendering {n_frames} frames...")
    for i in range(n_frames):
        plant.SetPositions(plant_ctx, iiwa, configs[i])

        qobj = sg.get_query_output_port().Eval(sg_ctx)
        color_image = qobj.RenderColorImage(camera, sg.world_frame_id(), X_WC)
        arr = np.asarray(color_image.data).reshape(HEIGHT, WIDTH, 4)[:, :, :3].copy()

        img = Image.fromarray(arr)
        draw = ImageDraw.Draw(img)

        # Ghost gripper at target position
        p_ghost = project_3d_to_2d(targets[i], X_WC, cam_info)
        if p_ghost:
            gw, gh = 30, 20
            draw.rectangle([p_ghost[0]-gw, p_ghost[1]-gh, p_ghost[0]+gw, p_ghost[1]+gh],
                           outline=(100, 180, 255), width=2)
            draw.rectangle([p_ghost[0]-gw+2, p_ghost[1]-gh+2, p_ghost[0]+gw-2, p_ghost[1]+gh-2],
                           outline=(100, 180, 255, 80), width=1)

        draw.text((30, 30), "IK Domain Extension", fill=(0x4f, 0xc3, 0xf7), font=font)

        if residuals[i] < 0.01:
            draw.text((30, 70), "Tracking within reachable workspace",
                      fill=(0x66, 0xbb, 0x6a), font=font_small)
        else:
            draw.text((30, 70), "Target beyond workspace boundary",
                      fill=(0xff, 0x70, 0x43), font=font_small)
            draw.text((30, 100), f"Residual: {residuals[i]*1000:.1f} mm",
                      fill=(0xff, 0x70, 0x43), font=font_small)
            draw.text((30, HEIGHT - 60),
                      "q* = argmin ||FK(q) - X_target||²",
                      fill=(0xff, 0xd5, 0x4f), font=font_eq)

            X_actual = plant.CalcRelativeTransform(
                plant_ctx, plant.world_frame(), ee_frame,
            )
            p_a = project_3d_to_2d(X_actual.translation(), X_WC, cam_info)
            p_t = project_3d_to_2d(targets[i], X_WC, cam_info)
            if p_a and p_t:
                draw.line([p_a, p_t], fill=(0xff, 0x70, 0x43), width=3)
                for p in (p_a, p_t):
                    draw.ellipse([p[0]-5, p[1]-5, p[0]+5, p[1]+5],
                                 fill=(0xff, 0x70, 0x43))

        pipe.stdin.write(np.array(img).tobytes())
        if i % 60 == 0:
            print(f"  frame {i}/{n_frames}")

    pipe.stdin.close()
    pipe.wait()
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
