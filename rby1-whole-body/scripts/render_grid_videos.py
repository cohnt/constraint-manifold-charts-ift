"""Render a video per cached grid-point plan, from a viewpoint that reveals collisions.

For each point in ``plans/grid_cache``, replays the plan leg by leg through the same
infrastructure the planner used -- the open box as a static obstacle for the reach
legs, the box attached to the gripper for the carry legs, the placed box as an
obstacle for the home legs -- and renders frames from a camera placed in front of
the robot looking back at it and the box, which is the view in which a gripper
entering the box or the table is visible rather than hidden behind the arm.

Each frame is additionally annotated with the leg name and the *measured* minimum
signed distance at that configuration, and framed in red when that distance is
negative. A collision is therefore both geometrically visible and called out
explicitly, so the video does not rely on the viewer spotting a few millimetres of
overlap.

Encoding uses gst-launch-1.0 with VP9 into WebM: this environment has no ffmpeg and
no pip, but GStreamer's vp9enc/webmmux are present.

Usage:
    .venv/bin/python scripts/render_grid_videos.py                 # all cached points
    .venv/bin/python scripts/render_grid_videos.py --indices 0 5   # just these
"""

import argparse
import os
import pickle
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw
from pydrake.all import (ClippingRange, ColorRenderCamera, CameraInfo,
                         DepthRange, DepthRenderCamera, MakeRenderEngineVtk,
                         RenderCameraCore, RenderEngineVtkParams, RigidTransform,
                         RollPitchYaw, RotationMatrix)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common
import plan_grid as E
from rby1_planning import Rby1ActiveJointLayout, make_default_rby1_infrastructure

RENDERER = "vtk"
WIDTH, HEIGHT = 720, 540
FPS = 30  # matches sample_leg's viz_hz, so playback is real-time at stride 1

# Camera in front of the robot (+x, robot base at the origin facing +x), raised to
# roughly torso height and offset slightly to +y.
#
# Chosen by rendering the worst-case frames (end of reach_descend, end of place) from
# several poses: the table is a 1.2 x 0.6 x 0.72 m slab at (0.32, -0.79) and a camera
# placed too low, too close, or on the -y side puts it between the lens and the robot,
# which would hide exactly the contact this video exists to reveal. From here the table
# sits to the left of frame, the whole robot fits, and both the gripper-vs-box interface
# at the grasp and the box-vs-table interface at the place are unobstructed.
CAM_POS = np.array([2.40, 0.20, 1.10])
CAM_LOOK_AT = np.array([0.30, -0.20, 0.55])
CAM_FOV_Y = 0.90  # rad; wide enough to keep the head in frame at full arm extension

LEG_SCENE = {
    "reach_approach": "open_box",
    "reach_descend": "open_box",
    "lift": "held",
    "place": "held",
    "home_retreat": "placed",
    "home": "placed",
}


def _look_at(eye, target, up=np.array([0.0, 0.0, 1.0])):
    """Camera pose with +z into the scene and +y down, as Drake's renderer wants."""
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
    """(plant, checker, diagram, layout) for one of the three scene variants."""
    if kind == "open_box":
        obstacles = [E.TABLE] + E.open_box_walls(E.box_pose_for(bx, by))
        plant, checker, diagram = make_default_rby1_infrastructure(
            obstacles=obstacles, open_grippers=True)
    elif kind == "held":
        held = E.HeldBox(hand=E.HAND, size=E.SIZE, offset=E.OFFSET, rpy=E.RPY,
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


def clearance_at(plant, pctx, ignore_pairs):
    """(min signed distance, pair label). Negative means penetration."""
    qobj = plant.get_geometry_query_input_port().Eval(pctx)
    insp = qobj.inspector()

    def bn(gid):
        return plant.GetBodyFromFrameId(insp.GetFrameId(gid)).name()

    worst, label = np.inf, ""
    for pp in qobj.ComputePointPairPenetration():
        a, b = bn(pp.id_A), bn(pp.id_B)
        if frozenset((a, b)) in ignore_pairs:
            continue
        if -pp.depth < worst:
            worst, label = -float(pp.depth), f"{a}/{b}"
    if worst < np.inf:
        return worst, label
    for p in qobj.ComputeSignedDistancePairwiseClosestPoints(0.06):
        a, b = bn(p.id_A), bn(p.id_B)
        if frozenset((a, b)) in ignore_pairs:
            continue
        if p.distance < worst:
            worst, label = float(p.distance), f"{a}/{b}"
    return (worst if worst < np.inf else 0.06), label


def annotate(img, lines, collision):
    d = ImageDraw.Draw(img)
    pad = 6
    box_h = 15 * len(lines) + 2 * pad
    d.rectangle([0, 0, WIDTH, box_h], fill=(0, 0, 0))
    for i, line in enumerate(lines):
        d.text((pad, pad + 15 * i), line,
               fill=(255, 90, 90) if collision else (235, 235, 235))
    if collision:
        for w in range(6):
            d.rectangle([w, w, WIDTH - 1 - w, HEIGHT - 1 - w], outline=(255, 0, 0))
    return img


def encode_webm(frames, out_path, fps=FPS):
    """Pipe raw RGB frames into gst-launch (VP9/WebM). No ffmpeg in this env."""
    cmd = [
        "gst-launch-1.0", "-q", "fdsrc", "!",
        "rawvideoparse", f"width={WIDTH}", f"height={HEIGHT}", "format=rgb",
        f"framerate={fps}/1", "!",
        "videoconvert", "!", "vp9enc", "deadline=1", "cpu-used=4", "!",
        "webmmux", "!", "filesink", f"location={out_path}",
    ]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                         stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        for f in frames:
            p.stdin.write(np.ascontiguousarray(f, dtype=np.uint8).tobytes())
        p.stdin.close()
    except BrokenPipeError:
        pass
    # p.wait() + read, not p.communicate(): communicate re-flushes stdin, which we
    # have already closed to signal end-of-stream, and raises ValueError.
    err = p.stderr.read()
    p.stderr.close()
    p.wait(timeout=900)
    if p.returncode != 0:
        raise RuntimeError(f"gst-launch failed (rc={p.returncode}): "
                           f"{err.decode(errors='replace')[:400]}")


def render_point(index, cache_dir, out_dir, stride=1):
    from plan_format.plan_io import load_plan

    path = E.cache_path_for(cache_dir, index)
    if not os.path.exists(path):
        return None
    _, meta = load_plan(path)
    legs, bx, by = meta["legs"], meta["bx"], meta["by"]
    evidence = meta.get("evidence", {})

    # The placed-box pose is FK of the held box at the last config of the place leg.
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
    frames = []
    worst_overall = np.inf

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

        # The held box legitimately touches the fingers that hold it; those pairs
        # are filtered during planning too, and flagging them would mark every
        # carry frame as a collision.
        ignore = set()
        if kind == "held":
            for hand in ("right", "left"):
                for f in ("ee_body", "ee_finger_1", "ee_finger_2"):
                    ignore.add(frozenset((f"ee_{E.HAND}", f)))
            for nm in (f"link_{E.HAND}_arm_3", f"link_{E.HAND}_arm_4",
                       f"link_{E.HAND}_arm_5", f"link_{E.HAND}_arm_6",
                       "FT_sensor_R", "FT_sensor_L"):
                ignore.add(frozenset((f"ee_{E.HAND}", nm)))

        qs = leg["q"][::stride]
        dur = leg["duration"]
        for i, q23 in enumerate(qs):
            q_full = plant.GetPositions(pctx)
            q_full[layout.plant_idxs] = q23
            plant.SetPositions(pctx, q_full)
            d, pair = clearance_at(plant, pctx, ignore)
            worst_overall = min(worst_overall, d)
            qobj = sg.get_query_output_port().Eval(sg_ctx)
            img = qobj.RenderColorImage(camera, sg.world_frame_id(), X_WC)
            arr = np.asarray(img.data).reshape(HEIGHT, WIDTH, 4)[:, :, :3]
            pil = Image.fromarray(arr.copy())
            t = dur * i / max(len(qs) - 1, 1)
            annotate(pil, [
                f"grid point {index}   box=({bx:+.3f}, {by:+.3f})",
                f"leg {leg['name']}   t={t:5.2f}/{dur:.2f}s",
                f"min signed distance {d*1000:+7.2f} mm   {pair}",
            ], collision=d < 0)
            frames.append(np.asarray(pil))

    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"point_{index:02d}.webm")
    encode_webm(frames, out)
    total = sum(e["duration"] for e in evidence.values()) if evidence else 0.0
    print(f"[{index:2d}] {len(frames):4d} frames -> {out}  "
          f"(worst clearance {worst_overall*1000:+.2f} mm, plan {total:.1f}s)")
    return dict(index=index, path=out, worst_clearance_m=float(worst_overall),
                n_frames=len(frames))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", default=E.DEFAULT_CACHE_DIR)
    ap.add_argument("--out-dir", default=os.path.join(common.RepoDir(), "scratch", "videos"))
    ap.add_argument("--indices", type=int, nargs="*", default=None)
    ap.add_argument("--stride", type=int, default=1,
                    help="keep every Nth sampled configuration (default 1 = every one; ''>1 makes playback faster than real time and choppier)")
    args = ap.parse_args()

    idxs = args.indices
    if idxs is None:
        idxs = sorted(
            int(f[len("point_"):-len(".pkl")])
            for f in os.listdir(args.cache_dir)
            if f.startswith("point_") and f.endswith(".pkl") and "debug" not in f
        )
    print(f"rendering {len(idxs)} point(s): {idxs}")
    out = []
    for i in idxs:
        try:
            r = render_point(i, args.cache_dir, args.out_dir, stride=args.stride)
            if r:
                out.append(r)
        except Exception as e:
            print(f"[{i:2d}] FAILED: {type(e).__name__}: {e}")
    bad = [r for r in out if r["worst_clearance_m"] < 0]
    print(f"\n{len(out)} video(s) written to {args.out_dir}")
    if bad:
        print(f"WARNING: {len(bad)} point(s) show penetration: "
              f"{[r['index'] for r in bad]}")
    else:
        print("No penetration in any rendered frame.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
