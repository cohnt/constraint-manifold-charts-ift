#!/usr/bin/env python3
"""Render the swept-volume figure: several poses of one bimanual motion in one image.

    python3 scripts/figures/render_swept_volume.py            # -> out/figures/swept_volume.png

Run from the repo root. This is the driver; the picture is actually built by
scripts/figures/blender_swept_volume.py running inside Blender. The split exists
because that half can only import `bpy`, and this half has to run with nothing
but stdlib -- in particular it must not import pydrake, so the figure can be
re-rendered on a machine that has no Drake.

What it does beyond invoking Blender is check the result. Blender exits 0 on a
great many partial failures, and the two that matter here are silent: an import
that produced no geometry, and a render that quietly fell back off the GPU. Both
look exactly like success unless someone reads the log, so this reads the log.

Input is a Meshcat static page. The default is `data/swept_volume_trajectory.html`,
which the last cell of notebooks/main_cpp.ipynb writes:

    with open("trajectory.html", "w") as f:
        f.write(meshcat.StaticHtml())

The default input is committed, so a fresh clone reproduces the figure with one
command and without a Drake install -- this half of the pipeline is stdlib only.
"""

import argparse
import os
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, SCRIPT_DIR)

from blender_paths import find_blender  # noqa: E402
import install_meshcat_importer  # noqa: E402

BLENDER_SCRIPT = os.path.join(SCRIPT_DIR, "blender_swept_volume.py")
# Mirrors the fallback in blender_swept_volume.setup_camera; it lives here too so
# the output filename can be derived before Blender is launched.
DEFAULT_AZIMUTH = 225.0
# Committed on purpose. The notebook's own export is gitignored scratch that
# changes with every stochastic planning run, so the published figure is pinned
# to one trajectory and a clone needs no Drake to reproduce it.
PINNED_HTML = "data/swept_volume_trajectory.html"
GPU_BACKENDS = ("OPTIX", "CUDA", "HIP", "ONEAPI")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--html", default=os.path.join(REPO_ROOT, PINNED_HTML),
                    help=f"input Meshcat static HTML (default: {PINNED_HTML}, which is "
                         "committed so a fresh clone can render the figure without Drake). "
                         "Pass a fresh export from the notebook to re-render against a "
                         "newly planned trajectory.")
    ap.add_argument("--out", default=None,
                    help="output PNG (default: out/figures/swept_volume_<azimuth>.png, so "
                         "rendering several orbits does not overwrite one file)")
    ap.add_argument("--n-poses", type=int, default=6,
                    help="ghost poses sampled along the motion (default: 6). Two arms "
                         "overlap heavily and 8 read as mush here; if you need more "
                         "poses, spread them rather than fading them further.")
    ap.add_argument("--t-start", type=float, default=0.0,
                    help="fraction of the clip to start the sweep at (default: 0.0)")
    ap.add_argument("--t-end", type=float, default=1.0,
                    help="fraction to end at (default: 1.0). The recording includes a ~1 s "
                         "settling tail where the robot holds the final pose, but arc-length "
                         "spacing gives it no weight, so the clip needs no trimming.")
    ap.add_argument("--spacing", default="arclength", choices=["arclength", "time"],
                    help="space the poses evenly along the path the geometry travels "
                         "(default) or evenly in time. The trajectory is TOPPRA-retimed and "
                         "then held, so uniform-in-time bunches the poses at both ends.")
    ap.add_argument("--alpha", type=float, default=0.45,
                    help="opacity of every ghost pose (default: 0.45), uniform by design")
    ap.add_argument("--endpoint-alpha", type=float, default=None,
                    help="opt back into solid first/last poses; off by default, since a "
                         "mixed ramp reads as an artifact rather than a cue")
    ap.add_argument("--mid-bias", type=float, default=0.35,
                    help="0 spaces the poses evenly along the path; up to (but not "
                         "including) 1 pushes them together through the middle of the "
                         "motion and spreads them out at the ends, where the two arms "
                         "nearly coincide and extra poses only add clutter (default: 0.35; 0.6 merges the two middle poses here)")
    ap.add_argument("--pose-nudge", type=float, nargs="*", default=None,
                    help="per-pose adjustment to where along the path each pose is taken, "
                         "in fractions of total arc length, one value per pose. This is how "
                         "a single pose is moved without hand-placing it -- it still comes "
                         "off the trajectory, just from a different point on it. The default "
                         "pulls pose 4 back to even out the top of the stack; note the "
                         "gripper rises monotonically here, so earlier is lower.")
    ap.add_argument("--table-material", default="old-metal",
                    help="BlenderKit asset slug for the table, or 'procedural' for the "
                         "built-in scratched black metal (default: old-metal). Read from "
                         "your own BlenderKit cache, which is not part of this repo -- "
                         "download the asset in Blender first, or the script warns and "
                         "falls back to procedural.")
    ap.add_argument("--shelf-material", default="old-plywood",
                    help="BlenderKit asset slug for the shelves, or 'procedural' "
                         "(default: old-plywood)")
    ap.add_argument("--table-texel-scale", type=float, default=1.0,
                    help="texture repeats per metre on the table (default: 1.0)")
    ap.add_argument("--shelf-texel-scale", type=float, default=1.0,
                    help="texture repeats per metre on the shelves (default: 1.0)")
    ap.add_argument("--blenderkit-dir", default="~/blenderkit_data/materials",
                    help="where the BlenderKit material cache lives")
    ap.add_argument("--drop-poses", type=int, nargs="*", default=None,
                    help="remove whole poses by index, counting from the start of the "
                         "trajectory and before the drop. Leaves the remaining poses "
                         "exactly where they were, unlike lowering --n-poses, which "
                         "re-spaces all of them. Default drops pose 1, which clumps with "
                         "pose 0 at the bottom of the stack; pass no values to keep all.")
    ap.add_argument("--static-tol", type=float, default=1e-4,
                    help="metres of travel below which a link counts as welded in place "
                         "and is drawn once instead of ghosted (default: 1e-4)")
    ap.add_argument("--resolution", type=int, nargs=2, default=[2800, 2400],
                    metavar=("W", "H"))
    ap.add_argument("--samples", type=int, default=128,
                    help="Cycles samples (default: 128). This is a cap, not a target -- "
                         "what cleared the graininess was the adaptive threshold, and "
                         "against a 1024-sample reference 128 differs by a mean 0.5/255.")
    ap.add_argument("--device", default="auto",
                    choices=["auto", *GPU_BACKENDS, "CPU"],
                    help="auto picks a GPU backend if there is one and Cycles-on-CPU if "
                         "not; naming a backend makes its absence a hard error")
    ap.add_argument("--engine", default="CYCLES", choices=["CYCLES", "EEVEE"],
                    help="EEVEE is much faster and composites the stacked ghosts "
                         "differently; it is not the same picture")
    ap.add_argument("--camera-azimuth", type=float, default=DEFAULT_AZIMUTH,
                    help=f"orbit azimuth in degrees (default: {DEFAULT_AZIMUTH:.0f}, chosen "
                         "by contact sheet). 225 and 135 are the three-quarter views with "
                         "the arms clear of the shelf; 90/180/270 are the side and back "
                         "views; 315 and 45 put the shelf's back panel across the frame.")
    ap.add_argument("--camera-elevation", type=float, default=None,
                    help="orbit elevation in degrees (default: 22)")
    ap.add_argument("--camera-distance", type=float, default=None)
    ap.add_argument("--lens", type=float, default=50.0)
    ap.add_argument("--frame-margin", type=float, default=1.1,
                    help="slack around the fitted bounding sphere (default: 1.1)")
    ap.add_argument("--opaque-bg", action="store_true",
                    help="render the navy backdrop instead of leaving it transparent. "
                         "The default is a transparent film, so the figure drops into the "
                         "paper over the page rather than in a box of its own colour.")
    ap.add_argument("--world-strength", type=float, default=0.35,
                    help="brightness of the backdrop the camera sees (default: 0.35)")
    ap.add_argument("--ambient", type=float, default=0.45,
                    help="brightness of the ambient dome the geometry is lit by "
                         "(default: 0.45). Separate from --world-strength so the shadows "
                         "can be filled without the backdrop going grey.")
    ap.add_argument("--light-strength", type=float, default=1.0,
                    help="multiplier on the whole light rig (default: 1.0)")
    ap.add_argument("--ghost-shadows", default="last", choices=["none", "last", "all"],
                    help="which poses cast shadows (default: last). With `all`, every "
                         "pose is dimmed by the ones in front of it, which reads as "
                         "uneven opacity rather than as depth.")
    ap.add_argument("--timeout", type=float, default=7200.0)
    ap.add_argument("--verbose", action="store_true",
                    help="forward Blender's stdout (it is noisy)")
    args = ap.parse_args()
    if args.out is None:
        args.out = os.path.join(REPO_ROOT,
                                f"out/figures/swept_volume_{args.camera_azimuth % 360:03.0f}.png")
    return args


def child_args(args):
    out = ["--html", args.html, "--out", args.out,
           "--n-poses", str(args.n_poses),
           "--t-start", str(args.t_start), "--t-end", str(args.t_end),
           "--spacing", args.spacing, "--alpha", str(args.alpha),
           "--static-tol", str(args.static_tol), "--mid-bias", str(args.mid_bias),
           "--table-material", args.table_material,
           "--shelf-material", args.shelf_material,
           "--table-texel-scale", str(args.table_texel_scale),
           "--shelf-texel-scale", str(args.shelf_texel_scale),
           "--blenderkit-dir", args.blenderkit_dir,
           "--resolution", str(args.resolution[0]), str(args.resolution[1]),
           "--samples", str(args.samples),
           "--device", args.device, "--engine", args.engine,
           "--lens", str(args.lens), "--frame-margin", str(args.frame_margin),
           "--world-strength", str(args.world_strength),
           "--ambient", str(args.ambient),
           "--light-strength", str(args.light_strength),
           "--ghost-shadows", args.ghost_shadows]
    if args.drop_poses is not None:
        out += ["--drop-poses", *[str(v) for v in args.drop_poses]]
    if args.pose_nudge is not None:
        out += ["--pose-nudge", *[str(v) for v in args.pose_nudge]]
    for flag, value in (("--endpoint-alpha", args.endpoint_alpha),
                        ("--camera-azimuth", args.camera_azimuth),
                        ("--camera-elevation", args.camera_elevation),
                        ("--camera-distance", args.camera_distance)):
        if value is not None:
            out += [flag, str(value)]
    if args.opaque_bg:
        out.append("--opaque-bg")
    return out


def last_line(text):
    """The last non-empty line, which for our child is its SystemExit message."""
    for line in reversed((text or "").splitlines()):
        if line.strip():
            return line.strip()
    return ""


def fail(message, stdout=None, stderr=None):
    """Report a failure with the diagnosis first and the noise afterwards.

    Blender writes on the order of a hundred lines of glTF chatter to stdout before
    anything interesting, so the child's own message -- which lands on stderr -- goes
    first and gets repeated in the headline. Burying it under the transcript is how
    "fails loudly" turns back into "looks like it hung".
    """
    print(f"error: {message}", file=sys.stderr)
    if stderr:
        print("--- blender stderr ---", file=sys.stderr)
        print(stderr, file=sys.stderr)
    if stdout:
        print("--- blender stdout ---", file=sys.stderr)
        print(stdout, file=sys.stderr)
    sys.exit(1)


def main():
    args = parse_args()

    blender = find_blender()
    print(f"[blender] {blender}")

    if not install_meshcat_importer.check(verbose=False):
        print("[addon] vendored add-on not installed or stale; installing")
        install_meshcat_importer.install()
        if not install_meshcat_importer.check(verbose=True):
            fail("could not install the meshcat_html_importer add-on. Run: "
                 "python3 scripts/figures/install_meshcat_importer.py")

    if not os.path.isfile(args.html):
        fail(f"no Meshcat page at {args.html}\n"
             f"  {PINNED_HTML} is committed, so this should not happen in a clean clone --\n"
             "  check the file was not deleted, or `git checkout` it.\n"
             "  To render a newly planned trajectory instead, run notebooks/main_cpp.ipynb\n"
             "  end to end (its last cell writes trajectory.html) and pass it with --html.\n"
             "  Refusing to pick a different scene.")

    cmd = [blender, "--background", "--python", BLENDER_SCRIPT, "--", *child_args(args)]
    print(f"[blender] rendering {args.html} -> {args.out}")
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout)
    except subprocess.TimeoutExpired:
        fail(f"Blender did not finish within {args.timeout:.0f}s")

    if args.verbose:
        print(proc.stdout)

    if proc.returncode != 0:
        detail = last_line(proc.stderr) or last_line(proc.stdout)
        fail(f"Blender exited {proc.returncode}" + (f": {detail}" if detail else ""),
             proc.stdout, proc.stderr)

    if not os.path.isfile(args.out):
        fail(f"Blender reported success but wrote no {args.out}", proc.stdout, proc.stderr)
    if os.path.getsize(args.out) == 0:
        fail(f"{args.out} is zero-length", proc.stdout, proc.stderr)

    # A GPU named explicitly must actually have been used. Dropping to the CPU is
    # ~20x slower and otherwise indistinguishable from success.
    if args.device in GPU_BACKENDS and args.engine == "CYCLES":
        if f"[render] CYCLES on GPU via {args.device}" not in proc.stdout:
            fail(f"--device {args.device} was requested but the render did not report "
                 f"running on it", proc.stdout, proc.stderr)

    for line in proc.stdout.splitlines():
        if line.startswith(("[import]", "[classify]", "[poses]", "[ghost] ", "[camera]",
                            "[render]")):
            print(line)
    print(f"[done] {args.out} ({os.path.getsize(args.out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
