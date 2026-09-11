"""Render the static stability constraint visualization for the RBY1.  SUPERSEDED.

SUPERSEDED by ``generate_stability_meshcat.py`` + ``render_static_stability_blender.py``,
which produce ``video/v2_rby1_stability.mp4`` through the meshcat -> Blender
pipeline so that every 3D shot in the overview video is Blender-rendered and
visually consistent.  This script still works and still writes the same file; it
is kept as the fallback if the Blender path breaks.  If you run it, be aware it
overwrites the Blender-rendered segment.

Shows the robot executing a trajectory with the support polygon and CoM
projection overlaid on the floor.  The CoM marker changes colour based
on its position relative to the conservative support polygon.

Usage:
    .venv/bin/python scripts/video/render_static_stability.py
"""

import os
import pickle
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

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "scripts"))

from rby1_planning import Rby1ActiveJointLayout, make_default_rby1_infrastructure
from rby1_opt_ik import (
    check_com_stability, support_polygon_xyzs, inset_support_polygon_xy,
    DEFAULT_SUPPORT_POLYGON_INSET, _com_support_polygon_residuals,
    _stability_constraint_ub, _com_instances,
)
from plan_format.plan_io import load_plan
import plan_grid as E

RENDERER = "vtk"
WIDTH, HEIGHT = 1920, 1080
FPS = 30
BG_HEX = "0x1a1a2e"
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

ACCENT_GREEN = (0x66, 0xbb, 0x6a)
ACCENT_YELLOW = (0xff, 0xd5, 0x4f)
ACCENT_CORAL = (0xff, 0x70, 0x43)
ACCENT_BLUE = (0x4f, 0xc3, 0xf7)


def _look_at(eye, target, up=np.array([0.0, 0.0, 1.0])):
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(-up, forward)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    R = np.column_stack([right, down, forward])
    return RigidTransform(RotationMatrix(R), eye)


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


def get_support_polygon_2d(base_xyt, X_WC, cam_info, inset=0.0):
    if inset > 0:
        verts = inset_support_polygon_xy(inset)
    else:
        verts = support_polygon_xyzs[:, :2]

    cos_t, sin_t = np.cos(base_xyt[2]), np.sin(base_xyt[2])
    R2 = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
    translated = (R2 @ verts.T).T + base_xyt[:2]

    pts_2d = []
    for xy in translated:
        p = project_3d_to_2d(np.array([xy[0], xy[1], 0.01]), X_WC, cam_info)
        if p:
            pts_2d.append(p)
    return pts_2d


def main():
    out_path = os.path.join(REPO, "video", "v2_rby1_stability.mp4")

    plan_path = os.path.join(REPO, "plans", "grid_cache", "point_00.pkl")
    _, meta = load_plan(plan_path)
    legs = meta["legs"]

    print("Building Drake infrastructure...")
    obstacles = [E.TABLE] + E.open_box_walls(E.box_pose_for(meta["bx"], meta["by"]))
    plant, checker, diagram = make_default_rby1_infrastructure(
        obstacles=obstacles, open_grippers=True
    )
    layout = Rby1ActiveJointLayout(plant)

    sg = diagram.GetSubsystemByName("scene_graph")
    if not sg.HasRenderer(RENDERER):
        sg.AddRenderer(RENDERER, MakeRenderEngineVtk(RenderEngineVtkParams()))

    ctx = diagram.CreateDefaultContext()
    plant_ctx = plant.GetMyContextFromRoot(ctx)
    sg_ctx = sg.GetMyContextFromRoot(ctx)

    cam_info = CameraInfo(WIDTH, HEIGHT, 0.9)
    X_WC = _look_at(np.array([1.5, 1.5, 1.0]),
                     np.array([0.0, 0.0, 0.4]))
    core = RenderCameraCore(RENDERER, cam_info, ClippingRange(0.05, 20.0),
                            RigidTransform())
    camera = ColorRenderCamera(core, show_window=False)

    try:
        font = ImageFont.truetype(FONT_PATH, 26)
        font_small = ImageFont.truetype(FONT_PATH, 18)
    except OSError:
        font = font_small = ImageFont.load_default()

    all_configs = []
    for leg in legs:
        stride = max(1, len(leg["q"]) // 30)
        for q23 in leg["q"][::stride]:
            all_configs.append(q23)

    n_frames = len(all_configs)
    print(f"Rendering {n_frames} frames...")

    pipe = subprocess.Popen(
        ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{WIDTH}x{HEIGHT}", "-r", str(FPS), "-i", "pipe:0",
         "-c:v", "libx264", "-crf", "20", "-preset", "fast",
         "-pix_fmt", "yuv420p", "-an", out_path],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )

    (base_inst, torso_inst, right_inst, left_inst,
     rg_inst, lg_inst, head_inst) = _com_instances(plant)

    for i, q23 in enumerate(all_configs):
        q_full = plant.GetPositions(plant_ctx)
        q_full[layout.plant_idxs] = q23
        plant.SetPositions(plant_ctx, q_full)

        com = plant.CalcCenterOfMassPositionInWorld(
            plant_ctx,
            [base_inst, torso_inst, right_inst, left_inst, rg_inst, lg_inst, head_inst],
        )

        base_xyt = q23[:3]
        residuals = _com_support_polygon_residuals(com[:2], base_xyt)
        ub_conservative = _stability_constraint_ub(DEFAULT_SUPPORT_POLYGON_INSET)
        is_conservative = bool(np.all(residuals <= ub_conservative))
        is_nominal = bool(np.all(residuals <= 0))

        rear_margin = -residuals[-1] if len(residuals) > 0 else 0.0

        qobj = sg.get_query_output_port().Eval(sg_ctx)
        color_image = qobj.RenderColorImage(camera, sg.world_frame_id(), X_WC)
        arr = np.asarray(color_image.data).reshape(HEIGHT, WIDTH, 4)[:, :, :3].copy()

        img = Image.fromarray(arr)
        draw = ImageDraw.Draw(img)

        nominal_pts = get_support_polygon_2d(base_xyt, X_WC, cam_info, inset=0.0)
        if len(nominal_pts) >= 3:
            draw.polygon(nominal_pts, outline=(255, 255, 100, 180))
            for j in range(len(nominal_pts)):
                p1 = nominal_pts[j]
                p2 = nominal_pts[(j + 1) % len(nominal_pts)]
                draw.line([p1, p2], fill=(255, 255, 100), width=2)

        cons_pts = get_support_polygon_2d(
            base_xyt, X_WC, cam_info, inset=DEFAULT_SUPPORT_POLYGON_INSET
        )
        if len(cons_pts) >= 3:
            for j in range(len(cons_pts)):
                p1 = cons_pts[j]
                p2 = cons_pts[(j + 1) % len(cons_pts)]
                draw.line([p1, p2], fill=(255, 165, 0), width=2)

        com_2d = project_3d_to_2d(np.array([com[0], com[1], 0.01]), X_WC, cam_info)
        if com_2d:
            r = 8
            if is_conservative:
                color = ACCENT_GREEN
            elif is_nominal:
                color = ACCENT_YELLOW
            else:
                color = ACCENT_CORAL
            draw.ellipse([com_2d[0]-r, com_2d[1]-r, com_2d[0]+r, com_2d[1]+r],
                         fill=color, outline=(255, 255, 255))

        draw.text((30, 30), "Static Stability Constraint",
                  fill=ACCENT_BLUE, font=font)

        if is_conservative:
            draw.text((30, 65), "CoM inside conservative polygon",
                      fill=ACCENT_GREEN, font=font_small)
        elif is_nominal:
            draw.text((30, 65), "CoM inside nominal polygon",
                      fill=ACCENT_YELLOW, font=font_small)
        else:
            draw.text((30, 65), "CoM OUTSIDE support polygon",
                      fill=ACCENT_CORAL, font=font_small)

        draw.text((30, 90),
                  f"Rear margin: {rear_margin*1000:.1f} mm",
                  fill=(200, 200, 200), font=font_small)

        draw.text((30, HEIGHT - 50),
                  f"Inset: {DEFAULT_SUPPORT_POLYGON_INSET*1000:.0f} mm (rear edge only)",
                  fill=(160, 160, 160), font=font_small)

        pipe.stdin.write(np.array(img).tobytes())
        if i % 30 == 0:
            print(f"  frame {i}/{n_frames}")

    pipe.stdin.close()
    pipe.wait()
    dur = n_frames / FPS
    print(f"Wrote {out_path} ({dur:.1f}s, {n_frames} frames)")


if __name__ == "__main__":
    main()
