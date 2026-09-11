"""Render boundary reachability constraint visualization for Video 2.  SUPERSEDED.

SUPERSEDED by ``generate_domain_ext_meshcat.py`` + ``render_boundary_reach_blender.py``,
which produce ``video/v2_boundary_reach.mp4`` through the meshcat -> Blender
pipeline so that every 3D shot in the overview video is Blender-rendered and
visually consistent.  This script still works and still writes the same file; it
is kept as the fallback if the Blender path breaks.  If you run it, be aware it
overwrites the Blender-rendered segment.

Split-screen: IIWA + WSG arm (left), regularized boundary constraint plot
(right).  Uses the same trajectory as the domain extension segment for
viewer consistency.  Plots the true constraint -log det(J*J^T + eps*I)
with a threshold line, and annotates "kinematic singularity" when the
constraint is violated.

Usage:
    .venv/bin/python scripts/video/render_boundary_reach.py
"""

import os
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pydrake.all import (
    AddMultibodyPlantSceneGraph, ClippingRange, ColorRenderCamera,
    CameraInfo, DiagramBuilder, MakeRenderEngineVtk, Parser,
    RenderCameraCore, RenderEngineVtkParams, RigidTransform,
    RollPitchYaw, RotationMatrix,
)
from pydrake.multibody.tree import JacobianWrtVariable, FixedOffsetFrame

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
IIWA_REPO = os.path.join(REPO, os.pardir, "iiwa-bimanual")
sys.path.insert(0, os.path.join(IIWA_REPO, "src"))

from iiwa_analytic_ik import Analytic_IK_7DoF

RENDERER = "vtk"
DRAKE_W, DRAKE_H = 1280, 1080
PLOT_W, PLOT_H = 640, 1080
OUT_W, OUT_H = 1920, 1080
FPS = 30
DURATION = 12.0
BG_COLOR = "#1a1a2e"
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

EPSILON = 1e-6
TAU = 9.0

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


def main():
    out_path = os.path.join(REPO, "video", "v2_boundary_reach.mp4")

    ik = Analytic_IK_7DoF()
    n_frames = int(DURATION * FPS)

    start_pos = np.array([0.4, 0.0, 0.7])
    end_pos = np.array([0.95, 0.0, 0.7])
    ee_rpy = RollPitchYaw(np.pi / 2, 0, 0)
    GC, psi = (1, 1, 1), 0.0

    print("Building IIWA + WSG scene...")
    builder = DiagramBuilder()
    plant, sg = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
    parser = Parser(plant)
    parser.package_map().AddPackageXml(os.path.join(IIWA_REPO, "package.xml"))

    iiwa_urdf = os.path.join(
        IIWA_REPO, "models", "iiwa14_convex_decimated_collision.urdf"
    )
    model = parser.AddModels(iiwa_urdf)[0]
    plant.WeldFrames(
        plant.world_frame(), plant.GetFrameByName("base", model), RigidTransform()
    )

    wsg_sdf = os.path.join(
        IIWA_REPO, "models", "wsg_finray",
        "wsg50_110_finray_fingers_box_collision.sdf",
    )
    wsg = parser.AddModels(wsg_sdf)[0]
    wsg_attach = plant.AddFrame(
        FixedOffsetFrame(
            "wsg_attach", plant.GetBodyByName("iiwa_link_7", model).body_frame(),
            WSG_ATTACH_OFFSET,
        )
    )
    plant.WeldFrames(wsg_attach, plant.GetFrameByName("body", wsg))

    plant.Finalize()
    diagram = builder.Build()

    if not sg.HasRenderer(RENDERER):
        sg.AddRenderer(RENDERER, MakeRenderEngineVtk(RenderEngineVtkParams()))

    ctx = diagram.CreateDefaultContext()
    plant_ctx = plant.GetMyMutableContextFromRoot(ctx)
    sg_ctx = sg.GetMyMutableContextFromRoot(ctx)

    ee_frame = plant.GetFrameByName("iiwa_link_7", model)

    configs, times, constraint_values = [], [], []
    last_valid_q = None

    print("Computing trajectory...")
    for i in range(n_frames):
        t_frac = i / n_frames
        target = start_pos + t_frac * (end_pos - start_pos)
        times.append(i / FPS)

        X_target = RigidTransform(ee_rpy, target)
        try:
            q = ik.IK(X_target, GC, psi)
            if q is not None and not np.any(np.isnan(q)):
                last_valid_q = q
        except Exception:
            pass

        configs.append(last_valid_q.copy() if last_valid_q is not None else np.zeros(7))
        plant.SetPositions(plant_ctx, model, configs[-1])

        J = plant.CalcJacobianSpatialVelocity(
            plant_ctx, JacobianWrtVariable.kQDot,
            ee_frame, [0, 0, 0],
            plant.world_frame(), plant.world_frame(),
        )
        JJT = J @ J.T
        det_reg = np.linalg.det(JJT + EPSILON * np.eye(JJT.shape[0]))
        constraint_values.append(-np.log(max(det_reg, 1e-30)))

    crossing_idx = None
    for j, v in enumerate(constraint_values):
        if v > TAU:
            crossing_idx = j
            break

    print(f"Constraint range: {min(constraint_values):.1f} to {max(constraint_values):.1f}")
    print(f"Threshold tau={TAU}, first crossing at frame {crossing_idx}")

    X_WC = _look_at(np.array([1.2, -1.4, 1.0]), np.array([0.4, 0.0, 0.55]))
    core = RenderCameraCore(RENDERER, CameraInfo(DRAKE_W, DRAKE_H, 0.8),
                            ClippingRange(0.05, 20.0), RigidTransform())
    camera = ColorRenderCamera(core, show_window=False)

    fig, ax = plt.subplots(figsize=(PLOT_W / 100, PLOT_H / 100), dpi=100)
    fig.patch.set_facecolor(BG_COLOR)

    try:
        font = ImageFont.truetype(FONT_PATH, 24)
        font_small = ImageFont.truetype(FONT_PATH, 18)
        font_sing = ImageFont.truetype(FONT_PATH, 22)
    except OSError:
        font = font_small = font_sing = ImageFont.load_default()

    pipe = subprocess.Popen(
        ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{OUT_W}x{OUT_H}", "-r", str(FPS), "-i", "pipe:0",
         "-c:v", "libx264", "-crf", "20", "-preset", "fast",
         "-pix_fmt", "yuv420p", "-an", out_path],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )

    print(f"Rendering {n_frames} frames...")
    for i in range(n_frames):
        plant.SetPositions(plant_ctx, model, configs[i])

        qobj = sg.get_query_output_port().Eval(sg_ctx)
        color_image = qobj.RenderColorImage(camera, sg.world_frame_id(), X_WC)
        drake_arr = np.asarray(color_image.data).reshape(
            DRAKE_H, DRAKE_W, 4
        )[:, :, :3].copy()

        ax.clear()
        ax.set_facecolor(BG_COLOR)
        ax.set_title("Boundary Reachability Constraint",
                      color="#e0e0e0", fontsize=12, fontweight="bold", pad=8)
        ax.set_ylabel(r"$-\log\det(JJ^T + \epsilon I)$",
                       color="#e0e0e0", fontsize=10)
        ax.set_xlabel("time (s)", color="#e0e0e0", fontsize=10)
        ax.tick_params(colors="#e0e0e0", labelsize=8)
        for sp in ("bottom", "left"):
            ax.spines[sp].set_color("#e0e0e0")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.grid(True, alpha=0.15, color="#e0e0e0")

        t_arr = np.array(times[:i+1])
        c_arr = np.array(constraint_values[:i+1])

        if len(t_arr) > 1:
            for j in range(len(t_arr) - 1):
                c = "#66bb6a" if c_arr[j] < TAU else "#ff7043"
                ax.plot(t_arr[j:j+2], c_arr[j:j+2], color=c, lw=2)

        ax.axhline(TAU, color="#ffd54f", ls="--", lw=1.5, alpha=0.8)
        ax.text(DURATION * 0.02, TAU + 1.0, f"τ = {TAU:.0f}",
                color="#ffd54f", fontsize=10, va="bottom")
        ax.text(DURATION * 0.85, TAU - 1.5, "feasible",
                color="#66bb6a", fontsize=11, fontweight="bold",
                ha="center", va="top")
        ax.text(DURATION * 0.85, TAU + 1.5, "infeasible",
                color="#ff7043", fontsize=11, fontweight="bold",
                ha="center", va="bottom")

        ax.set_xlim(0, DURATION)
        y_max = max(max(constraint_values) * 1.15, TAU * 1.5)
        ax.set_ylim(bottom=min(constraint_values) * 0.9, top=y_max)
        ax.axvline(times[i], color="#ff7043", alpha=0.4, lw=1, ls="--")

        fig.subplots_adjust(left=0.18, right=0.92, top=0.88, bottom=0.12)
        fig.canvas.draw()
        plot_arr = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        plot_img = Image.fromarray(plot_arr).resize((PLOT_W, PLOT_H), Image.LANCZOS)

        composite = np.zeros((OUT_H, OUT_W, 3), dtype=np.uint8)
        composite[:, :DRAKE_W] = drake_arr
        composite[:, DRAKE_W:] = np.array(plot_img)

        img = Image.fromarray(composite)
        draw = ImageDraw.Draw(img)
        draw.text((20, 20), "Boundary Reachability",
                  fill=(0x4f, 0xc3, 0xf7), font=font)

        if constraint_values[i] < TAU:
            draw.text((20, 55), "Interior of workspace",
                      fill=(0x66, 0xbb, 0x6a), font=font_small)
        else:
            draw.text((20, 55), "Kinematic singularity",
                      fill=(0xff, 0x70, 0x43), font=font_sing)

        pipe.stdin.write(np.array(img).tobytes())
        if i % 60 == 0:
            print(f"  frame {i}/{n_frames}")

    pipe.stdin.close()
    pipe.wait()
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
