"""Render Drake simulation videos for the supplementary video grid.

Produces one MP4 per seed at the target resolution, with clean annotation
(leg name + point label). Reuses the camera and scene infrastructure from
render_grid_videos.py.

Usage:
    .venv/bin/python scripts/video/render_drake_cells.py
    .venv/bin/python scripts/video/render_drake_cells.py --indices 0 3 11 16
"""

import argparse
import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pydrake.all import (ClippingRange, ColorRenderCamera, CameraInfo,
                         DepthRange, DepthRenderCamera, MakeRenderEngineVtk,
                         RenderCameraCore, RenderEngineVtkParams, RigidTransform,
                         RollPitchYaw, RotationMatrix)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import common
import plan_grid as E
from rby1_planning import Rby1ActiveJointLayout, make_default_rby1_infrastructure

RENDERER = "vtk"
WIDTH, HEIGHT = 960, 540
FPS = 30

CAM_POS = np.array([2.40, 0.20, 1.10])
CAM_LOOK_AT = np.array([0.30, -0.20, 0.55])
CAM_FOV_Y = 0.90

LEG_SCENE = {
    "reach_approach": "open_box",
    "reach_descend": "open_box",
    "lift": "held",
    "place": "held",
    "home_retreat": "placed",
    "home": "placed",
}

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT_DIR = os.path.join(REPO, "video")


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
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def annotate(img, index, leg_name, t, dur):
    draw = ImageDraw.Draw(img)
    font = get_font(18)
    small_font = get_font(14)

    # Top-left: point label
    label = f"Point {index}"
    draw.text((8, 6), label, font=font, fill=(79, 195, 247))

    # Top-right: leg name + time
    leg_text = f"{leg_name}  {t:.1f}/{dur:.1f}s"
    bbox = draw.textbbox((0, 0), leg_text, font=small_font)
    tw = bbox[2] - bbox[0]
    draw.text((WIDTH - tw - 8, 8), leg_text, font=small_font, fill=(200, 200, 200))

    return img


def encode_mp4(frames_iter, out_path, fps=FPS):
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{WIDTH}x{HEIGHT}", "-r", str(fps),
        "-i", "pipe:0",
        "-c:v", "libx264", "-crf", "20", "-preset", "medium",
        "-pix_fmt", "yuv420p",
        out_path,
    ]
    proc = subprocess.Popen(
        cmd, stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )
    n = 0
    for frame in frames_iter:
        proc.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
        n += 1
    proc.stdin.close()
    err = proc.stderr.read()
    proc.stderr.close()
    proc.wait(timeout=900)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {err.decode()[:400]}")
    return n


def render_point(index, cache_dir, out_dir):
    from plan_format.plan_io import load_plan

    path = E.cache_path_for(cache_dir, index)
    if not os.path.exists(path):
        print(f"  [{index:2d}] no plan pickle")
        return None
    _, meta = load_plan(path)
    legs, bx, by = meta["legs"], meta["bx"], meta["by"]

    # Compute placed box pose from the last config of the place leg
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

    camera = make_camera()
    X_WC = _look_at(CAM_POS, CAM_LOOK_AT)

    def frame_gen():
        scene_cache = {}
        for leg in legs:
            kind = LEG_SCENE.get(leg["name"], "open_box")
            if kind not in scene_cache:
                plant, checker, diagram, layout = build_scene(kind, bx, by, placed_pose)
                sg = add_renderer(diagram)
                scene_cache[kind] = (plant, diagram, layout, sg)
            plant, diagram, layout, sg = scene_cache[kind]
            ctx = diagram.CreateDefaultContext()
            pctx = plant.GetMyContextFromRoot(ctx)
            sg_ctx = sg.GetMyContextFromRoot(ctx)

            qs = leg["q"]
            dur = leg["duration"]
            for i, q23 in enumerate(qs):
                q_full = plant.GetPositions(pctx)
                q_full[layout.plant_idxs] = q23
                plant.SetPositions(pctx, q_full)
                qobj = sg.get_query_output_port().Eval(sg_ctx)
                img = qobj.RenderColorImage(camera, sg.world_frame_id(), X_WC)
                arr = np.asarray(img.data).reshape(HEIGHT, WIDTH, 4)[:, :, :3]
                pil = Image.fromarray(arr.copy())
                t = dur * i / max(len(qs) - 1, 1)
                annotate(pil, index, leg["name"], t, dur)
                yield np.asarray(pil)

    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"drake_point_{index:02d}.mp4")
    n = encode_mp4(frame_gen(), out)
    total_dur = sum(l["duration"] for l in legs)
    print(f"  [{index:2d}] {n} frames -> {out}  ({total_dur:.1f}s plan)")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", default=E.DEFAULT_CACHE_DIR)
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--indices", type=int, nargs="*", default=[0, 3, 11, 16])
    args = ap.parse_args()

    print(f"Rendering {len(args.indices)} Drake cell(s): {args.indices}")
    for idx in args.indices:
        try:
            render_point(idx, args.cache_dir, args.out_dir)
        except Exception as e:
            print(f"  [{idx:2d}] FAILED: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
