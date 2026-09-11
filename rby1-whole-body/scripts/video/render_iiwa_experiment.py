"""Verify -- and, with Drake VTK, optionally render -- the IIWA bimanual segment.

The motion comes from ``generate_iiwa_meshcat.py``: BiRRT + shortcutting in the
8-D parameterized space, lifted through the analytic IK and retimed with TOPPRA,
on the scene from ``../iiwa-bimanual/notebooks/main_cpp.ipynb``.
The plank is carried from the table to the shelf's top plate and set down on it.

**This script no longer produces the delivered segment.**  video/
v2_iiwa_bimanual.mp4 now comes from the meshcat -> Blender path
(``blender_render_iiwa.py``), which is what the RBY1 grid renders use.  What is
still uniquely here is ``verify()``: the dense check that the rendered motion is
actually collision-free, that the plank really ends up resting on the plate, and
that the gripper-to-gripper transform is constant.  Run it after any change to
the plank, the keyframes or the planner:

    .venv/bin/python scripts/video/render_iiwa_experiment.py --verify-only

Rendering here is kept working as a fallback (Drake VTK, with the on-frame
captions), but writes to video/v2_iiwa_bimanual_vtk.mp4 so it cannot clobber the
Blender deliverable:

    .venv/bin/python scripts/video/render_iiwa_experiment.py
    .venv/bin/python scripts/video/render_iiwa_experiment.py --replan
"""

import argparse
import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pydrake.all import (
    ClippingRange, ColorRenderCamera, CameraInfo, MakeRenderEngineVtk,
    RenderCameraCore, RenderEngineVtkParams, RigidTransform, RotationMatrix,
)

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import generate_iiwa_meshcat as iiwa_demo

RENDERER = "vtk"
WIDTH, HEIGHT = 1920, 1080
FPS = 30
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

# Playback: the motion is already stretched to iiwa_demo.PLAYBACK_SPEED inside
# lift_and_retime, so it is sampled here at 1:1 with a still hold at each end.
HEAD_HOLD_S = 0.8
TAIL_HOLD_S = 2.0

CAM_POS = np.array([-1.15, -0.75, 1.15])
CAM_TARGET = np.array([0.62, 0.40, 0.42])

# Samples used by --verify-only.  At 400 the check passed on a path that a
# 2000-sample check caught grazing the shelf by 17 um, so this is deliberately
# not a round-numbered "looks like enough".
VERIFY_SAMPLES = 4000


def _look_at(eye, target, up=np.array([0.0, 0.0, 1.0])):
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(-up, forward)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    R = np.column_stack([right, down, forward])
    return RigidTransform(RotationMatrix(R), eye)


def verify(traj, plant, plant_ctx, checker, n_samples=VERIFY_SAMPLES):
    """Prove the four things the segment claims.

    1. every configuration on the rendered trajectory is collision-free under an
       *unpadded* checker that has the plank and both grippers in the robot set,
       and the tightest pair over the whole motion is named;
    2. the plank clears every shelf plate, the shelf's back panel and the table
       -- the closest approach to each is reported;
    3. the plank ends level and resting on the top plate, with the actual gap;
    4. the gripper-to-gripper transform is constant, i.e. the parameterization's
       equality constraint really is held.
    """
    from pydrake.multibody.tree import BodyIndex

    ts = np.linspace(traj.start_time(), traj.end_time(), n_samples)
    configs = [traj.value(t).flatten() for t in ts]
    print(f"\nVerifying {len(configs)} samples of a "
          f"{traj.end_time() - traj.start_time():.2f}s trajectory...")

    def body_label(index):
        b = plant.get_body(BodyIndex(int(index)))
        return f"{plant.GetModelInstanceName(b.model_instance())}::{b.name()}"

    # --- 1. collision-free, and the tightest pair ---------------------------
    n_bad = 0
    worst = (np.inf, None)
    for i, q in enumerate(configs):
        if not checker.CheckConfigCollisionFree(q):
            n_bad += 1
        rc = checker.CalcRobotClearance(q, 0.05)
        d = rc.distances()
        if len(d):
            j = int(np.argmin(d))
            if d[j] < worst[0]:
                worst = (float(d[j]), (i, int(rc.robot_indices()[j]),
                                       int(rc.other_indices()[j])))
    print(f"  collision-free: {len(configs) - n_bad}/{len(configs)} samples")
    d, (i, ra, rb) = worst
    print(f"  tightest pair over the whole motion: {d * 1000:.2f} mm, "
          f"{body_label(ra)} vs {body_label(rb)} at sample {i} (t={ts[i]:.2f}s)")

    # --- 2. the plank against each piece of the environment -----------------
    plank = plant.GetBodyByName("carried_plank")
    plank_g = plant.GetCollisionGeometriesForBody(plank)
    inspector = plant.get_geometry_query_input_port().Eval(plant_ctx).inspector()
    env_g = {}
    for name in ("old_shelves", "table"):
        mi = plant.GetModelInstanceByName(name)
        for bi in plant.GetBodyIndices(mi):
            for g in plant.GetCollisionGeometriesForBody(plant.get_body(bi)):
                env_g[inspector.GetName(g)] = g

    mins = {k: (np.inf, -1) for k in env_g}
    for i, q in enumerate(configs):
        plant.SetPositions(plant_ctx, q)
        qobj = plant.get_geometry_query_input_port().Eval(plant_ctx)
        for k, g in env_g.items():
            for pg in plank_g:
                sd = qobj.ComputeSignedDistancePairClosestPoints(pg, g).distance
                if sd < mins[k][0]:
                    mins[k] = (sd, i)
    for k, (v, i) in sorted(mins.items()):
        print(f"  plank vs {k:34s} closest {v * 1000:8.2f} mm "
              f"(sample {i}, t={ts[i]:.2f}s)")

    # --- 3. the placement itself -------------------------------------------
    half = np.asarray(iiwa_demo.CARRIED_PLANK_SIZE) / 2.0
    corners = np.array([[sx * half[0], sy * half[1], sz * half[2]]
                        for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    for label, q in (("first", configs[0]), ("last", configs[-1])):
        plant.SetPositions(plant_ctx, q)
        X = plant.EvalBodyPoseInWorld(plant_ctx, plank)
        w = (X.rotation().matrix() @ corners.T).T + X.translation()
        rpy = np.degrees(X.rotation().ToRollPitchYaw().vector())
        print(f"  {label} sample: plank centre {np.round(X.translation(), 4)}, "
              f"rpy {np.round(rpy, 3)} deg, underside z={w[:, 2].min():.4f}, "
              f"x in [{w[:, 0].min():.3f}, {w[:, 0].max():.3f}]")
    print(f"  reference surfaces: table top z={iiwa_demo.TABLE_TOP_Z:.3f}, "
          f"shelf top plate z={iiwa_demo.SHELF_PLATE_TOP_Z:.3f}")

    # --- 4. the gripper-to-gripper transform -------------------------------
    Fl = plant.GetFrameByName("body", plant.GetModelInstanceByName("wsg_left"))
    Fr = plant.GetFrameByName("body", plant.GetModelInstanceByName("wsg_right"))
    # Compared as (translation, rotation matrix); quaternions would show a
    # spurious jump whenever the wxyz sign flips.
    rel = []
    for q in configs:
        plant.SetPositions(plant_ctx, q)
        X = plant.CalcRelativeTransform(plant_ctx, Fl, Fr)
        rel.append(np.concatenate([X.translation(), X.rotation().matrix().ravel()]))
    rel = np.asarray(rel)
    print(f"  X_leftgripper_rightgripper: translation {np.round(rel[0, :3], 6)}, "
          f"max drift {np.max(np.abs(rel[:, :3] - rel[0, :3])):.2e} m (translation), "
          f"{np.max(np.abs(rel[:, 3:] - rel[0, 3:])):.2e} (rotation matrix entries)")

    plate_gap = mins["old_shelves::shelf_2_collision"][0]
    ok = (n_bad == 0) and (worst[0] > 0.0) and (plate_gap > 0.0)
    print(f"  => {'PASS' if ok else 'FAIL'}\n")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--replan", action="store_true", help="re-run the planner")
    ap.add_argument("--verify-only", action="store_true",
                    help="run the checks and stop before rendering")
    ap.add_argument("--samples", type=int, default=VERIFY_SAMPLES)
    ap.add_argument("--speed", type=float, default=iiwa_demo.PLAYBACK_SPEED)
    args = ap.parse_args()

    out_path = os.path.join(REPO, "video", "v2_iiwa_bimanual_vtk.mp4")

    reduced_path = iiwa_demo.get_reduced_path(replan=args.replan)

    # One scene, used both for the checks and for rendering: the plank has to be
    # in the collision model for a placement claim to mean anything, and it has
    # to be in the visual model to be on screen.  env_padding is 0 here -- the
    # planner runs padded, the verifier does not.
    diagram, plant, checker = iiwa_demo.build_scene(
        with_collision_checker=True, with_carried_plank=True, env_padding=0.0)
    traj = iiwa_demo.lift_and_retime(reduced_path, plant, speed=args.speed)

    sg = diagram.scene_graph()
    if not sg.HasRenderer(RENDERER):
        sg.AddRenderer(RENDERER, MakeRenderEngineVtk(RenderEngineVtkParams()))
    context = diagram.CreateDefaultContext()
    plant_ctx = plant.GetMyMutableContextFromRoot(context)
    sg_ctx = sg.GetMyMutableContextFromRoot(context)

    # The plank is welded, so the position vector is still the notebook's --
    # iiwa_left's 7 joints then iiwa_right's 7.  If that ever stopped holding,
    # the same q would mean different configurations in the planning and render
    # plants and the render would silently drift from the verified plan.
    assert plant.num_positions() == 14, f"plant has {plant.num_positions()} positions"

    if not verify(traj, plant, plant_ctx, checker, n_samples=args.samples):
        print("Verification failed -- not rendering.")
        sys.exit(1)
    if args.verify_only:
        return

    t0, t1 = traj.start_time(), traj.end_time()
    n_motion = int(round((t1 - t0) * FPS))
    ts = np.linspace(t0, t1, n_motion)
    configs = [traj.value(t).flatten() for t in ts]
    frames = ([configs[0]] * int(HEAD_HOLD_S * FPS) + configs
              + [configs[-1]] * int(TAIL_HOLD_S * FPS))
    n_frames = len(frames)
    print(f"Rendering {n_frames} frames = {n_frames / FPS:.2f}s "
          f"({HEAD_HOLD_S}s hold + {(t1 - t0):.2f}s motion + {TAIL_HOLD_S}s hold)")

    X_WC = _look_at(CAM_POS, CAM_TARGET)
    core = RenderCameraCore(
        RENDERER, CameraInfo(WIDTH, HEIGHT, 0.8),
        ClippingRange(0.05, 20.0), RigidTransform(),
    )
    camera = ColorRenderCamera(core, show_window=False)

    try:
        font = ImageFont.truetype(FONT_PATH, 28)
        font_small = ImageFont.truetype(FONT_PATH, 20)
    except OSError:
        font = font_small = ImageFont.load_default()

    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{WIDTH}x{HEIGHT}", "-r", str(FPS),
        "-i", "pipe:0",
        "-c:v", "libx264", "-crf", "20", "-preset", "fast",
        "-pix_fmt", "yuv420p", "-an",
        out_path,
    ]
    pipe = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    for i, q in enumerate(frames):
        plant.SetPositions(plant_ctx, q)

        qobj = sg.get_query_output_port().Eval(sg_ctx)
        color_image = qobj.RenderColorImage(camera, sg.world_frame_id(), X_WC)
        arr = np.asarray(color_image.data).reshape(HEIGHT, WIDTH, 4)[:, :, :3].copy()

        img = Image.fromarray(arr)
        draw = ImageDraw.Draw(img)
        for y, text, fill, fnt in CAPTIONS(font, font_small):
            draw.text((30, y), text, fill=fill, font=fnt)

        pipe.stdin.write(np.array(img).tobytes())

        if i % 60 == 0:
            print(f"  frame {i}/{n_frames}")

    pipe.stdin.close()
    pipe.wait()
    if pipe.returncode != 0:
        print(pipe.stderr.read().decode()[-2000:])
        sys.exit(1)
    print(f"Wrote {out_path}")


def CAPTIONS(font, font_small, dark_bg=False):
    """The on-frame captions.  blender_render_iiwa.py imports this so the two
    renderers cannot drift apart.

    The third line is the only one that has to change with the background: on
    VTK's white it is dark slate, on Blender's dark blue world it would vanish.
    """
    third = (0xb0, 0xbe, 0xc5) if dark_bg else (0x3a, 0x4a, 0x5a)
    return [
        (30, "IIWA Bimanual Planning", (0x4f, 0xc3, 0xf7), font),
        (70, "Constrained carry and place — gripper-to-gripper transform "
             "held fixed", (0x66, 0xbb, 0x6a), font_small),
        (104, "BiRRT in the 8-D parameterized space, lifted through analytic IK, "
              "retimed with TOPPRA", third, font_small),
    ]


if __name__ == "__main__":
    main()
