"""Drive ``blender_segment_render.py`` and read its frames back.

The half of the meshcat -> Blender pipeline that runs in the normal venv: writes a
render config, shells out to Blender, verifies the run produced the frames it
claimed to, and hands the caller the frame list plus the camera the render
actually used.  Overlay compositing happens in the caller, which is the only part
that differs between segments.
"""

from __future__ import annotations

import glob
import json
import os
import subprocess
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from blender_paths import BLENDER  # noqa: E402
RENDER_SCRIPT = os.path.join(os.path.dirname(__file__), "blender_segment_render.py")

sys.path.insert(0, os.path.dirname(__file__))
from segment_camera import SegmentCamera  # noqa: E402


def render_segment(html, out_dir, camera: SegmentCamera, *, fps=30, samples=128,
                   frame_end=None, force=False, timeout=7200):
    """Blender-render ``html`` into ``out_dir``; return (frames, camera_used).

    ``camera`` fixes the framing; the camera returned is the one read back out of
    Blender (``camera.json``), which is what any 3D-anchored overlay must project
    with.  Skips the Blender run when the frames on disk were rendered from this
    same scene file with this same camera (see the fingerprint below), so
    re-running the overlay pass is cheap; ``force`` re-renders regardless.
    """
    if not os.path.exists(html):
        raise FileNotFoundError(f"no meshcat HTML at {html} -- run its generator first")

    os.makedirs(out_dir, exist_ok=True)
    cam_json = os.path.join(out_dir, "camera.json")
    cfg_path = os.path.join(out_dir, "render_config.json")
    frames = sorted(glob.glob(os.path.join(out_dir, "frame_*.png")))

    # Everything that decides what the frames look like, in one dict: the scene
    # (by path, size and mtime) and the camera. It is written beside the frames
    # and compared on the next run.
    #
    # The scene fingerprint is the load-bearing half. Comparing cameras alone
    # meant that regenerating the meshcat HTML -- a different animation, a
    # different box, a different gripper -- left the old frames in place and the
    # segment silently kept showing the previous take.
    #
    # This is compared for equality rather than with a tolerance because both
    # sides are the *requested* values this function wrote, not values read back
    # out of Blender in single precision. (The camera Blender reports is read
    # from camera.json separately, and is what overlays project with.)
    st = os.stat(html)
    cfg = dict(
        html=os.path.abspath(html),
        html_mtime=st.st_mtime, html_size=st.st_size,
        out_dir=os.path.abspath(out_dir),
        width=camera.width, height=camera.height, fps=fps, samples=samples,
        camera_matrix_world=camera.matrix_world.tolist(),
        lens_mm=camera.lens_mm,
        sensor_width_mm=camera.sensor_width_mm,
        frame_end=frame_end,
    )

    if frames and os.path.exists(cam_json) and os.path.exists(cfg_path) and not force:
        try:
            with open(cfg_path) as f:
                cached_cfg = json.load(f)
        except (OSError, ValueError):
            cached_cfg = None
        if cached_cfg == json.loads(json.dumps(cfg)):
            print(f"  reusing {len(frames)} cached frames in {out_dir}")
            return frames, SegmentCamera.load(cam_json)
        print("  cached frames were rendered from a different scene or camera; "
              "re-rendering")

    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)

    print(f"  Blender rendering {camera.width}x{camera.height} from "
          f"{os.path.basename(html)} ...")
    result = subprocess.run(
        [BLENDER, "--background", "--python", RENDER_SCRIPT, "--", cfg_path],
        capture_output=True, text=True, timeout=timeout,
    )
    # "[render]" carries the engine and compute device actually used. A silent
    # fall back from Cycles/GPU to CPU or to EEVEE changes both the runtime and
    # the look, and is invisible in the exit status.
    for line in result.stdout.split("\n"):
        if line.startswith("SEGMENT_RENDER") or line.startswith("[render]"):
            print(f"    {line}")

    # Never trust the exit status alone: Blender has exited 0 on scenes that
    # rendered nothing.
    if "SEGMENT_RENDER_DONE" not in result.stdout:
        tail = "\n".join(
            ln for ln in (result.stdout + result.stderr).split("\n")[-25:] if ln.strip())
        raise RuntimeError(f"Blender render did not finish:\n{tail}")

    frames = sorted(glob.glob(os.path.join(out_dir, "frame_*.png")))
    if not frames:
        raise RuntimeError(f"Blender reported success but wrote no frames to {out_dir}")

    cam_used = SegmentCamera.load(cam_json)
    print(f"  {len(frames)} frames rendered")
    return frames, cam_used


def encode(frames_iter, out_path, width, height, fps=30, crf=20):
    """Pipe RGB frames straight into x264.  ``frames_iter`` yields HxWx3 uint8."""
    pipe = subprocess.Popen(
        ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{width}x{height}", "-r", str(fps), "-i", "pipe:0",
         "-c:v", "libx264", "-crf", str(crf), "-preset", "fast",
         "-pix_fmt", "yuv420p", "-an", out_path],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )
    n = 0
    for arr in frames_iter:
        pipe.stdin.write(arr.tobytes())
        n += 1
    pipe.stdin.close()
    rc = pipe.wait()
    if rc != 0:
        raise RuntimeError(f"ffmpeg failed ({rc}): {pipe.stderr.read().decode()[-500:]}")
    return n


def probe_duration(path):
    return float(subprocess.check_output([
        "ffprobe", "-v", "quiet", "-show_entries", "format=duration",
        "-of", "csv=p=0", path,
    ]).decode().strip())
