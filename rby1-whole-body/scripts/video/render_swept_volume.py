"""Render a swept-volume visualization of one full pick-and-place cycle.

Renders the robot at evenly-spaced configurations, extracts the robot pixels
from the background, tints each pose by leg phase, and composites them all
onto a clean background with consistent opacity. First and last poses are
shown at full opacity.

Usage:
    .venv/bin/python scripts/video/render_swept_volume.py
    .venv/bin/python scripts/video/render_swept_volume.py --index 3 --n-poses 25
"""

import argparse
import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pydrake.all import (ClippingRange, ColorRenderCamera, CameraInfo,
                         MakeRenderEngineVtk, RenderCameraCore,
                         RenderEngineVtkParams, RigidTransform,
                         RollPitchYaw, RotationMatrix)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import common
import plan_grid as E
from rby1_planning import Rby1ActiveJointLayout, make_default_rby1_infrastructure

RENDERER = "vtk"
WIDTH, HEIGHT = 1920, 1080
FPS = 30

# Side view — shows the robot bending over toward the floor and reaching the table
CAM_POS = np.array([0.55, 1.80, 1.10])
CAM_LOOK_AT = np.array([0.40, -0.20, 0.40])
CAM_FOV_Y = 0.95

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT_DIR = os.path.join(REPO, "video")

LEG_SCENE = {
    "reach_approach": "open_box",
    "reach_descend": "open_box",
    "lift": "held",
    "place": "held",
    "home_retreat": "placed",
    "home": "placed",
}

LEG_TINT = {
    "reach_approach": np.array([79, 195, 247]),    # blue
    "reach_descend":  np.array([79, 195, 247]),
    "lift":           np.array([255, 112, 67]),     # coral
    "place":          np.array([255, 112, 67]),
    "home_retreat":   np.array([102, 187, 106]),    # green
    "home":           np.array([102, 187, 106]),
}

# Background color for the VTK renderer (Drake default is ~204,204,204)
BG_DETECT_THRESHOLD = 15


def _look_at(eye, target, up=np.array([0.0, 0.0, 1.0])):
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    right = np.cross(-up, forward)
    right = right / np.linalg.norm(right)
    down = np.cross(forward, right)
    R = np.column_stack([right, down, forward])
    return RigidTransform(RotationMatrix(R), eye)


def make_camera():
    core = RenderCameraCore(
        RENDERER,
        CameraInfo(WIDTH, HEIGHT, CAM_FOV_Y),
        ClippingRange(0.05, 20.0),
        RigidTransform(),
    )
    return ColorRenderCamera(core, show_window=False)


def build_scene(kind, bx, by, placed_box_pose):
    if kind == "open_box":
        obstacles = [E.TABLE] + E.open_box_walls(E.box_pose_for(bx, by))
        plant, checker, diagram = make_default_rby1_infrastructure(
            obstacles=obstacles, open_grippers=True)
    elif kind == "held":
        from rby1_planning import HeldBox
        held = HeldBox(hand=E.HAND, size=E.SIZE, offset=E.OFFSET, rpy=E.RPY,
                       visual=True, open_top=True, wall_thickness=E.WALL_T,
                       wall_height=E.WALL_HEIGHT)
        plant, checker, diagram = make_default_rby1_infrastructure(
            held_boxes=held, obstacles=E.TABLE)
    elif kind == "placed":
        obstacles = [E.TABLE] + E.open_box_walls(placed_box_pose, prefix="placed")
        plant, checker, diagram = make_default_rby1_infrastructure(
            obstacles=obstacles, open_grippers=True)
    else:
        raise ValueError(kind)
    return plant, checker, diagram, Rby1ActiveJointLayout(plant)


def add_renderer(diagram):
    sg = diagram.GetSubsystemByName("scene_graph")
    if not sg.HasRenderer(RENDERER):
        sg.AddRenderer(RENDERER, MakeRenderEngineVtk(RenderEngineVtkParams()))
    return sg


def get_font(size):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",):
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def render_swept_volume(index, cache_dir, n_poses=25, duration_s=4.0):
    from plan_format.plan_io import load_plan

    path = E.cache_path_for(cache_dir, index)
    _, meta = load_plan(path)
    legs, bx, by = meta["legs"], meta["bx"], meta["by"]

    # Compute placed box pose
    placed_pose = E.box_pose_for(bx, by)
    place_leg = next((l for l in legs if l["name"] == "place"), None)
    if place_leg is not None:
        plant_h, _, diagram_h, layout_h = build_scene("held", bx, by, placed_pose)
        ctx = diagram_h.CreateDefaultContext()
        pctx = plant_h.GetMyContextFromRoot(ctx)
        q_full = plant_h.GetPositions(pctx)
        q_full[layout_h.plant_idxs] = place_leg["q"][-1]
        plant_h.SetPositions(pctx, q_full)
        ee = plant_h.GetBodyByName(f"ee_{E.HAND}",
                                   plant_h.GetModelInstanceByName(f"{E.HAND}_arm"))
        placed_pose = (plant_h.EvalBodyPoseInWorld(pctx, ee)
                       @ RigidTransform(RollPitchYaw(E.RPY), E.OFFSET))

    # Collect all configs with metadata
    all_configs = []
    for leg in legs:
        kind = LEG_SCENE.get(leg["name"], "open_box")
        for q in leg["q"]:
            all_configs.append((q, kind, leg["name"]))

    # Sample poses with heavier weighting on the interesting phases
    # reach_descend, lift, place are the bending/carrying phases
    leg_boundaries = []
    offset = 0
    for leg in legs:
        n = len(leg["q"])
        leg_boundaries.append((offset, offset + n, leg["name"]))
        offset += n

    # Allocate poses per leg proportional to duration but with minimum 2 per leg
    total_configs = len(all_configs)
    sample_indices = set()
    # Always include first and last
    sample_indices.add(0)
    sample_indices.add(total_configs - 1)

    # Key poses: start/end of each leg, plus midpoints of interesting legs
    for start, end, name in leg_boundaries:
        sample_indices.add(start)
        sample_indices.add(end - 1)
        mid = (start + end) // 2
        sample_indices.add(mid)
        # Extra samples for interesting legs
        if name in ("reach_descend", "lift", "place", "reach_approach"):
            q1 = start + (end - start) // 3
            q2 = start + 2 * (end - start) // 3
            sample_indices.add(q1)
            sample_indices.add(q2)

    # Fill remaining budget with evenly spaced samples
    remaining = n_poses - len(sample_indices)
    if remaining > 0:
        even = np.linspace(0, total_configs - 1, remaining + 2, dtype=int)[1:-1]
        sample_indices.update(even.tolist())

    sample_indices = sorted(sample_indices)[:n_poses]
    print(f"  Sampling {len(sample_indices)} poses from {total_configs} configs")

    camera = make_camera()
    X_WC = _look_at(CAM_POS, CAM_LOOK_AT)

    # --- Phase 1: Render a clean background (no robot visible) ---
    # Use the first scene config but at a "zero" pose — actually just render
    # the first frame and use it as the reference background
    bg_q, bg_kind, _ = all_configs[0]
    plant, checker, diagram, layout = build_scene(bg_kind, bx, by, placed_pose)
    sg = add_renderer(diagram)
    ctx = diagram.CreateDefaultContext()
    pctx = plant.GetMyContextFromRoot(ctx)
    sg_ctx = sg.GetMyContextFromRoot(ctx)
    qobj = sg.get_query_output_port().Eval(sg_ctx)
    bg_img = qobj.RenderColorImage(camera, sg.world_frame_id(), X_WC)
    bg_arr = np.asarray(bg_img.data).reshape(HEIGHT, WIDTH, 4)[:, :, :3].copy()

    # --- Phase 2: Render all sampled poses ---
    rendered_poses = []
    scene_cache = {}

    for pose_idx in sample_indices:
        q23, kind, leg_name = all_configs[pose_idx]

        if kind not in scene_cache:
            p, c, d, l = build_scene(kind, bx, by, placed_pose)
            s = add_renderer(d)
            scene_cache[kind] = (p, d, l, s)
        plant, diagram, layout, sg = scene_cache[kind]

        ctx = diagram.CreateDefaultContext()
        pctx = plant.GetMyContextFromRoot(ctx)
        sg_ctx = sg.GetMyContextFromRoot(ctx)

        q_full = plant.GetPositions(pctx)
        q_full[layout.plant_idxs] = q23
        plant.SetPositions(pctx, q_full)

        qobj = sg.get_query_output_port().Eval(sg_ctx)
        img = qobj.RenderColorImage(camera, sg.world_frame_id(), X_WC)
        arr = np.asarray(img.data).reshape(HEIGHT, WIDTH, 4)[:, :, :3].copy()

        rendered_poses.append((arr, leg_name, pose_idx))

    # --- Phase 3: Composite with robot extraction and tinting ---
    # Start with a darkened version of the background
    canvas = bg_arr.astype(np.float64) * 0.6

    n_total = len(rendered_poses)
    for i, (arr, leg_name, pose_idx) in enumerate(rendered_poses):
        # Extract robot pixels: where this frame differs from the background
        diff = np.abs(arr.astype(np.float64) - bg_arr.astype(np.float64))
        mask = diff.max(axis=2) > BG_DETECT_THRESHOLD
        # Soften mask edges slightly
        from scipy.ndimage import binary_dilation
        mask = binary_dilation(mask, iterations=1)

        # Tint the robot pixels
        tint = LEG_TINT.get(leg_name, np.array([200, 200, 200]))
        tinted = arr.astype(np.float64)
        tinted[mask] = tinted[mask] * 0.5 + tint.astype(np.float64) * 0.5

        # Alpha: first and last at full, others at 0.5
        is_endpoint = (i == 0 or i == n_total - 1)
        alpha = 0.85 if is_endpoint else 0.45

        # Composite only robot pixels onto canvas
        robot_mask = mask.astype(np.float64)[:, :, np.newaxis]
        canvas = canvas * (1 - robot_mask * alpha) + tinted * robot_mask * alpha

    result = Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8))

    # Add legend and title
    draw = ImageDraw.Draw(result)
    font = get_font(28)
    small_font = get_font(20)

    draw.text((20, 20), f"Swept Volume — Point {index}", font=font, fill=(79, 195, 247))

    legend_y = 60
    for label, color in [("Reach", (79, 195, 247)), ("Carry", (255, 112, 67)),
                          ("Home", (102, 187, 106))]:
        draw.rectangle([20, legend_y, 40, legend_y + 16], fill=color)
        draw.text((48, legend_y - 2), label, font=small_font, fill=(200, 200, 200))
        legend_y += 24

    # Encode as static video
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "swept_volume.mp4")
    n_frames = int(duration_s * FPS)
    raw = np.array(result)
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{WIDTH}x{HEIGHT}", "-r", str(FPS),
        "-i", "pipe:0",
        "-c:v", "libx264", "-crf", "18", "-preset", "medium",
        "-pix_fmt", "yuv420p",
        "-t", str(duration_s),
        out_path,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    frame_bytes = raw.tobytes()
    for _ in range(n_frames):
        proc.stdin.write(frame_bytes)
    proc.stdin.close()
    proc.wait()
    print(f"Wrote {out_path} ({len(sample_indices)} poses, {duration_s}s)")
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", default=E.DEFAULT_CACHE_DIR)
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--n-poses", type=int, default=25)
    ap.add_argument("--duration", type=float, default=4.0)
    args = ap.parse_args()
    render_swept_volume(args.index, args.cache_dir, args.n_poses, args.duration)


if __name__ == "__main__":
    main()
