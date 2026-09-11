"""Build the promo video (and/or the two hardware-supplementary cuts) end to end.

One command for the whole chain. Every stage declares the file it produces, so
the build is **resumable**: a stage whose output already exists *and is newer
than the script that produced it* is skipped, and `--force` re-runs everything
(or `--force-from <stage>` from a point onward).

    .venv/bin/python scripts/video/build_video.py overview
    .venv/bin/python scripts/video/build_video.py ral
    .venv/bin/python scripts/video/build_video.py all --dry-run
    .venv/bin/python scripts/video/build_video.py overview --force-from iiwa_bimanual_html

Stages fail loudly. Nothing here pipes into `tail`: a crashed renderer that
still prints its usual closing lines is the exact failure this pipeline has
shipped before, so each stage's exit status is checked and the stage's own
output file is confirmed to exist and be non-trivial afterwards.

## From a fresh clone

The intent is that the command above is the *only* one you run. Given the
environment (`pip install -e '.[video]'`, Blender 5.0.x, ffmpeg, the DejaVu
fonts), this script builds the IIWA experiment's C++ extension itself,
then renders all three deliverables: the promo video and both hardware-
supplementary cuts (anonymous and named -- they share every segment and
differ only in the title card, so both are always built together).
Everything else either ships in the repo -- the 20 plan pickles, the
execution records, the models -- or is derived here.

## What this cannot do for you

- **The raw hardware capture is not reproducible.** `videos/*.mp4` is the
  2026-08-11 phone capture; both hardware-supplementary cuts and the RB-Y1
  side-by-side need it. It is not in the repo and cannot be re-derived: ask
  the authors for it. If it is missing this script stops and says so, rather than
  substituting a different clip -- a positional seed map silently pairing the
  wrong footage with the wrong plan is a mistake this project has already
  made once.
- **The IIWA experiment folder.** Three overview stages read `../iiwa-bimanual`
  (its models, its `src/`, and its notebook), so `--skip-iiwa` -- which only skips
  the *segment* -- does not remove the requirement. In this release it is a
  sibling folder of this one, so nothing needs fetching.

Drake is *not* on that list. That repo's C++ extension has to be built against
an install carrying `drake-config.cmake`, which the pip wheel is not, and the two
stages that import the built extension have to run against that same install --
so absent `$DRAKE_INSTALL_DIR`, a Drake binary for this machine's Ubuntu codename
is downloaded into `drake-binary/` and used for both. Only Ubuntu is packaged
that way; anywhere else, point `$DRAKE_INSTALL_DIR` at an install by hand.
"""

import argparse
import ast
import glob
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PY = sys.executable
VIDEO = os.path.join(REPO, "video")
MEDIA = os.path.join(REPO, "media", "videos", "manim_scenes_v2", "1080p30")
SCRATCH_HTML = os.path.join(REPO, "scratch", "grid_html_30fps_final")
SIBLING = os.path.join(REPO, os.pardir, "iiwa-bimanual")
HW_DIR = os.path.join(REPO, "videos")
TRIM_JSON = os.path.join(HW_DIR, "trim_points.json")

# Where a downloaded Drake binary is unpacked (gitignored), and where it comes
# from. Only used when $DRAKE_INSTALL_DIR does not already name an install.
DRAKE_CACHE = os.path.join(REPO, "drake-binary")
DRAKE_NIGHTLY_URL = ("https://drake-packages.csail.mit.edu/drake/nightly/"
                     "drake-latest-{codename}.tar.gz")

# Resolved once by check_prereqs and read by iiwa_env(); None if there is none.
_DRAKE_PREFIX = None

# A stage is (name, output path, argv, minimum plausible output size in bytes).
# The size floor is what catches a renderer that produced a file but no frames.
def _py(*args):
    return [PY, *args]


def overview_stages(args):
    v = lambda n: os.path.join(VIDEO, n)
    stages = [
        ("cards", v("v2_title.mp4"),
         _py("scripts/video/make_cards_v2.py"), 10_000),
        # A stage tracks exactly one output, so the results card needs its own entry
        # or it would never be rebuilt on a tree where v2_title.mp4 is already fresh.
        # Running make_cards_v2.py twice costs about two seconds.
        ("cards_results", v("v2_results.mp4"),
         _py("scripts/video/make_cards_v2.py"), 10_000),
        ("manim", os.path.join(MEDIA, "RBY1PipelineScene.mp4"),
         _py("-m", "manim", "-qh", "--fps", "30",
             "scripts/video/manim_scenes_v2.py",
             "IKParameterizationScene", "RBY1PipelineScene"), 100_000),
        ("domain_ext_html", v("v2_domain_extension.html"),
         _py("scripts/video/generate_domain_ext_meshcat.py"), 1_000_000),
        # The renderers below read their meshcat HTML from a hardcoded path,
        # so it has to be declared: a regenerated scene must re-render, and
        # blender_segment_driver's frame cache now notices the same change.
        ("domain_ext", v("v2_domain_extension.mp4"),
         _py("scripts/video/render_domain_extension_blender.py"), 100_000,
         {"deps": [v("v2_domain_extension.html")]}),
        ("boundary_reach", v("v2_boundary_reach.mp4"),
         _py("scripts/video/render_boundary_reach_blender.py"), 100_000,
         {"deps": [v("v2_domain_extension.html")]}),
        # This stage imports the sibling's iiwa_ik extension (it puts
        # cpp_parameterization/python on sys.path itself), so it needs the Drake
        # that extension was built against -- see iiwa_env().
        ("iris_html", v("v2_iiwa_iris.html"),
         _py("scripts/video/generate_iiwa_iris_meshcat.py"), 1_000_000,
         {"env": iiwa_env}),
        # --table-brightness 0.18, where the bimanual segment uses the default 1.
        #
        # The table material is a 2048x2048 photograph holding both near-black
        # oxidised areas and pale worn-bare ones, and its UVs are box-projected in
        # *world space*. So which tone a scene gets is decided by where its table
        # happens to sit: this scene's table is offset about 0.4 m in x and y from
        # the bimanual one, which slid the projection nearly half a tile onto a
        # pale region. Same asset, same lights -- tabletop at median luminance 120
        # against the bimanual segment's 52. 0.18 puts it at 52.6.
        #
        # Darkening this one material is what keeps the arms right. They are flat
        # untextured colour and already match the bimanual segment exactly, so any
        # global lever moves them for no reason. Two were tried and rejected:
        # --light-scale 0.22 hit the table (60) but dragged the arm highlights from
        # 160 down to 90, and an exposure adjustment also darkened the shared world
        # backdrop (92 -> 44), which would put a visible brightness step at the cut.
        # --table-texel-scale 3 moved the table the wrong way (120 -> 124): world-
        # space box projection changes which *frequency* of the photo is visible,
        # not which part of its tonal range.
        #
        # Stills for all of these are in scratch/iris_look/ (gitignored).
        ("iris", v("v2_iiwa_iris.mp4"),
         _py("scripts/video/render_with_blender.py", v("v2_iiwa_iris.html"),
             "--output", v("v2_iiwa_iris.mp4"),
             "--table-brightness", "0.18",
             "--title", "Constrained IRIS Regions",
             "--subtitle", "A random walk inside one convex, collision-free "
                           "region grown in the minimal coordinates"), 100_000),
        ("stability_html", v("v2_rby1_stability.html"),
         _py("scripts/video/generate_stability_meshcat.py"), 1_000_000),
        ("stability", v("v2_rby1_stability.mp4"),
         _py("scripts/video/render_static_stability_blender.py"), 100_000,
         {"deps": [v("v2_rby1_stability.html")]}),
        # The RB-Y1 side-by-side. grid_html re-export is the fix that keeps
        # proximity geometry out of the Blender renders.
        #
        # Only point 0 appears in either video, and each export is ~90 MB and
        # about a minute, so this asks for the one point rather than all 20 --
        # which is 15 minutes to produce 19 files nothing downstream reads. Drop
        # the --show to refresh the whole set (compose_grid.py uses it).
        ("grid_html", os.path.join(SCRATCH_HTML, "point_00.html"),
         _py("scripts/grid_visualizer.py", "--fps", "30", "--no-serve",
             "--show", "0", "--export-html", SCRATCH_HTML), 1_000_000),
        # --force because this stage only runs when something upstream changed,
        # and blender_render_grid's own check is existence, not freshness.
        ("grid_render", v("drake_blender_00.mp4"),
         _py("scripts/video/blender_render_grid.py", "--indices", "0",
             "--force"), 100_000,
         {"deps": [os.path.join(SCRATCH_HTML, "point_00.html")]}),
        ("montage", v("montage_20.mp4"),
         _py("scripts/video/compose_montage.py"), 100_000),
        ("rby1_hardware", v("v2_rby1_hardware.mp4"),
         _py("scripts/video/compose_rby1_v2.py"), 100_000,
         {"deps": [v("drake_blender_00.mp4"), v("montage_20.mp4"),
                   os.path.join(REPO, "scripts", "video",
                                "hardware_framing.py")]}),
    ]
    if not args.skip_iiwa:
        stages += iiwa_stages()
    # The assembly depends on every segment ahead of it. Without that, a
    # rebuilt segment sat on disk while the previous cut of the whole video was
    # skipped as "already built".
    stages.append(("assemble", v("promo_video.mp4"),
                   _py("scripts/video/assemble_v2.py"), 1_000_000,
                   {"deps": [st[1] for st in stages]}))
    return stages


def iiwa_stages():
    """The IIWA bimanual segment, planned in the IIWA experiment folder.

    main_cpp.ipynb runs KinematicTrajectoryOptimization, lifts through the
    parameterization and retimes with TOPPRA, then writes trajectory.html.
    Retiming a BiRRT+shortcut polyline instead -- which this repo's
    generate_iiwa_meshcat.py used to do -- is what made TOPPRA fail.
    """
    nb_out = os.path.join(SIBLING, "notebooks", "trajectory.html")
    html = os.path.join(VIDEO, "v2_iiwa_bimanual.html")
    return [
        # --output goes to a scratch notebook, NOT /dev/null: nbconvert appends
        # ".ipynb" to whatever it is given, so "--output /dev/null" tries to
        # create "/dev/null.ipynb" and dies with PermissionError *after* every
        # cell has run and trajectory.html has been written. That is a nonzero
        # exit on a run that actually succeeded.
        ("iiwa_bimanual_html", nb_out,
         _py("-m", "nbconvert", "--to", "notebook", "--execute",
             "--ExecutePreprocessor.timeout=9000",
             "--output", os.path.join(VIDEO, "main_cpp_executed.ipynb"),
             "main_cpp.ipynb"), 1_000_000,
         {"cwd": os.path.join(SIBLING, "notebooks"),
          # A callable: iiwa_env() reads the Drake prefix that check_prereqs
          # resolves, which happens after the stage list is built.
          "env": lambda: {**iiwa_env(), "MPLBACKEND": "Agg"}}),
        ("iiwa_bimanual_copy", html, ["__copy__", nb_out, html], 1_000_000),
        # --yaw -90 puts both arms in the foreground with the shelves behind
        # them, rather than looking down the length of the shelves.
        ("iiwa_bimanual", os.path.join(VIDEO, "v2_iiwa_bimanual.mp4"),
         _py("scripts/video/render_with_blender.py", html,
             "--output", os.path.join(VIDEO, "v2_iiwa_bimanual.mp4"),
             "--yaw", "-90",
             # Half speed: the retimed plan is only 1.97 s of motion, which is
             # too quick to read. The rate is drawn on screen by the renderer,
             # so the segment does not quietly misstate how fast the plan runs.
             "--speed", "0.5",
             "--title", "Constrained Bimanual Planning",
             "--subtitle", "Trajectory optimization in minimal coordinates — "
                           "the grasp constraint holds by construction"),
         100_000),
    ]


def ral_stages(args):
    v = lambda n: os.path.join(VIDEO, n)
    # Both hardware cuts are built every time. They share every segment and
    # differ only in the title card, so the anonymous submission copy and
    # the public named copy cannot drift apart -- which is what a single
    # fixed title.mp4 selected by a --named flag used to risk.
    cards = [
        ("cards_ral", v("title_anonymous.mp4"),
         _py("scripts/video/make_cards.py"), 10_000),
        ("cards_ral_named", v("title_named.mp4"),
         _py("scripts/video/make_cards.py", "--named"), 10_000),
    ]
    segments = [
        # compute_errors is the only stage that reads plans/ and results/;
        # re-run it (--force-from errors) whenever either changes.
        ("errors", v("error_data.pkl"),
         _py("scripts/video/compute_errors.py"), 1_000),
        # --video-offset stays at 0. Trim points are derived from the execution
        # records' wall clock by check_video_sync.py, never hand-tuned.
        ("segment1", v("segment1_point00.mp4"),
         _py("scripts/video/render_segment1.py", "--point", "0"), 100_000),
        ("montage_ral", v("montage_20.mp4"),
         _py("scripts/video/compose_montage.py"), 100_000),
        # Both of these append montage_20.mp4, which they read from a
        # hardcoded default rather than from argv.
        ("segment2", v("segment2_all20.mp4"),
         _py("scripts/video/render_segment2.py"), 100_000,
         {"deps": [v("montage_20.mp4")]}),
    ]
    shared = [st[1] for st in segments]
    # Each assembly deps on its own card only, so regenerating one cut's
    # title card does not rebuild the other cut's video.
    return cards + segments + [
        ("assemble_ral", v("hardware_supplementary_anonymous.mp4"),
         _py("scripts/video/assemble_v1.py"), 1_000_000,
         {"deps": shared + [v("title_anonymous.mp4")]}),
        ("assemble_ral_named", v("hardware_supplementary_non_anonymous.mp4"),
         _py("scripts/video/assemble_v1.py", "--named"), 1_000_000,
         {"deps": shared + [v("title_named.mp4")]}),
    ]



# Packages the *video* pipeline needs which the planner does not, so they are
# not in [project.dependencies] and an editable install alone does not bring
# them in. Named here so a missing one is reported up front rather than by a
# stage crashing a quarter of an hour in.
VIDEO_IMPORTS = [
    ("manim", "the two explainer scenes"),
    ("nbconvert", "executing the IIWA experiment's notebook"),
]

# Several scripts pass this file to PIL, and compose_montage/compose_grid pass
# it straight to ffmpeg's drawtext, which has no fallback: no font, no montage.
FONT_DIR = "/usr/share/fonts/truetype/dejavu"
FONTS = ["DejaVuSans.ttf", "DejaVuSans-Bold.ttf"]


def _is_drake_prefix(path):
    """Does ``path`` name a Drake install the sibling's CMake can build against?"""
    return bool(path) and os.path.isfile(
        os.path.join(path, "lib", "cmake", "drake", "drake-config.cmake"))


def drake_version(prefix):
    """The install's own version string (a nightly date), or ``"unknown"``."""
    try:
        with open(os.path.join(prefix, "share", "doc", "drake",
                               "VERSION.TXT")) as f:
            return f.read().split()[0]
    except (OSError, IndexError):
        return "unknown"


def ubuntu_codename():
    """``noble``, etc. -- from VERSION_CODENAME, which is what Drake packages by.

    Not ``ID``: on a derivative distribution that is the derivative's own name
    (this workstation reports ``tuxedo``) while VERSION_CODENAME still names the
    Ubuntu release the binaries are built for.
    """
    try:
        with open("/etc/os-release") as f:
            for line in f:
                if line.startswith("VERSION_CODENAME="):
                    return line.split("=", 1)[1].strip().strip('"')
    except OSError:
        pass
    return None


def ensure_drake(problems, may_fetch):
    """A Drake install carrying ``drake-config.cmake``, downloading one if needed.

    Two things need it and they must be the *same* install: cmake links the
    sibling's ``_iiwa_ik`` against it, and the stages that import that extension
    run against its bindings (see ``iiwa_env``). The pip wheel can serve neither
    -- it ships no CMake package config at all.

    ``$DRAKE_INSTALL_DIR`` wins, so a machine with a source build keeps using it
    and nothing is downloaded. Otherwise the binary nightly for this machine's
    Ubuntu codename is fetched once into ``drake-binary/`` (gitignored). It is
    "latest" rather than the pinned wheel version because nightlies are retained
    only ~56 days: pinning here would rot into a 404 on exactly the fresh machine
    this is meant to serve. The two builds can therefore differ -- which is safe,
    since both users of this prefix take it together -- and the version actually
    used is printed.
    """
    env_prefix = os.environ.get("DRAKE_INSTALL_DIR")
    if _is_drake_prefix(env_prefix):
        return os.path.abspath(env_prefix)
    if env_prefix:
        problems.append(
            f"$DRAKE_INSTALL_DIR={env_prefix} has no "
            "lib/cmake/drake/drake-config.cmake, so nothing can be built against\n"
            "    it. Unset it to let this build download a Drake binary, or point "
            "it at a\n    source or binary install.")
        return None

    cached = os.path.join(DRAKE_CACHE, "drake")
    if _is_drake_prefix(cached):
        return cached

    codename = ubuntu_codename()
    if not codename:
        problems.append(
            "no Drake install, and this machine's /etc/os-release names no Ubuntu\n"
            "    VERSION_CODENAME, so there is no binary package to download. Point\n"
            "    $DRAKE_INSTALL_DIR at a source or binary Drake install.")
        return None

    url = DRAKE_NIGHTLY_URL.format(codename=codename)
    if not may_fetch:
        problems.append(
            f"no Drake install; the real build downloads {url}\n"
            f"    into {os.path.relpath(DRAKE_CACHE, REPO)}/ (--dry-run does not)")
        return None

    print(f"[fetch] {url} (~54 MB)", flush=True)
    tarball = os.path.join(DRAKE_CACHE, "drake.tar.gz")
    os.makedirs(DRAKE_CACHE, exist_ok=True)
    try:
        urllib.request.urlretrieve(url, tarball)
        with tarfile.open(tarball) as tf:
            tf.extractall(DRAKE_CACHE, filter="tar")
    except Exception as e:  # network, disk, a codename with no package
        problems.append(
            f"could not install Drake from {url}: {e}\n"
            "    Point $DRAKE_INSTALL_DIR at a source or binary Drake install "
            "instead.")
        return None
    finally:
        if os.path.exists(tarball):
            os.remove(tarball)

    if not _is_drake_prefix(cached):
        problems.append(
            f"downloaded {url} but {cached} has no "
            "lib/cmake/drake/drake-config.cmake.")
        return None
    return cached


def drake_python_site_packages(prefix):
    """``<prefix>/lib/python3.X/site-packages``, if it carries ``pydrake``."""
    if not prefix:
        return None
    for d in sorted(glob.glob(os.path.join(prefix, "lib", "python3.*",
                                           "site-packages"))):
        if os.path.isdir(os.path.join(d, "pydrake")):
            return d
    return None


def iiwa_env():
    """Environment for the two stages that import the sibling's ``iiwa_ik``.

    ``_iiwa_ik`` is compiled against one Drake. Import it under a *different*
    pydrake -- the venv's pip wheel, say -- and pybind11 has two type registries,
    so the extension's base types do not resolve::

        ImportError: generic_type: type "IiwaBimanualReachableConstraint"
                     referenced unknown base type "drake::solvers::Constraint"

    ...ten minutes into the build, at iris_html. PYTHONPATH takes precedence over
    the venv's site-packages, so naming the build-against install first makes the
    two agree. This was invisible on the original workstation, whose PYTHONPATH
    already pointed at the source build the extension had been compiled against;
    on a clean machine the two never match by accident.
    """
    parts = [drake_python_site_packages(_DRAKE_PREFIX),
             os.path.join(SIBLING, "cpp_parameterization", "python"),
             os.environ.get("PYTHONPATH", "")]
    seen, out = set(), []
    for part in parts:
        for entry in (part or "").split(os.pathsep):
            if entry and entry not in seen:
                seen.add(entry)
                out.append(entry)
    return {"PYTHONPATH": os.pathsep.join(out)}


EXT_DIR = os.path.join(SIBLING, "cpp_parameterization", "python", "iiwa_ik")
EXT_STAMP = os.path.join(EXT_DIR, "built_against.txt")


def _stamp_text(prefix):
    # Prefix and version, because "latest" changes underneath the same path. A
    # source build reports "unknown" (it ships no VERSION.TXT), so switching
    # between two source builds is the one change this does not catch.
    return f"{os.path.abspath(prefix)} {drake_version(prefix)}\n"


def sibling_extension_built(prefix=None):
    """Is the sibling's ``iiwa_ik`` importable, and built against ``prefix``?

    Size, because the LTO link leaves a 0-byte .so behind for a while. Stamp,
    because the Drake this builds against moves: an .so left over from a previous
    one does not look missing, it imports as an ``unknown base type`` ImportError
    (see ``iiwa_env``). A stamp mismatch counts as *not built*, so it is rebuilt
    -- half a minute -- rather than crashing a stage a quarter of an hour in.
    """
    ok = os.path.isdir(EXT_DIR) and any(
        f.endswith(".so") and os.path.getsize(os.path.join(EXT_DIR, f)) > 1e6
        for f in os.listdir(EXT_DIR))
    if not ok or prefix is None:
        return ok
    try:
        with open(EXT_STAMP) as f:
            return f.read() == _stamp_text(prefix)
    except OSError:
        return False



def ensure_sibling(problems, prefix, need_extension, may_fetch):
    """Clone (and if asked, build) the IIWA repo three overview stages read.

    It is a separate, private repo, so the failure that actually happens to a
    new person is a permission denied on the clone -- not a missing directory
    they forgot to create. Say so, and say who to ask, instead of printing a
    clone command that will fail the same way when they paste it.
    """
    if not os.path.isdir(SIBLING):
        problems.append(
            f"the IIWA experiment folder is missing at {os.path.relpath(SIBLING, REPO)}.\n"
            "    It ships alongside this one in the release; three overview stages need\n"
            "    it: domain_ext_html and iris_html load its models and src/, and\n"
            "    iiwa_bimanual_html executes its notebook.")
        return

    if not need_extension or sibling_extension_built(prefix):
        return

    if prefix is None:
        # ensure_drake already said why, in this same problem list.
        return
    if not may_fetch or shutil.which("cmake") is None:
        problems.append(
            "the IIWA experiment's C++ extension (iiwa_ik) is not built against "
            f"{prefix}, and\n    this build cannot build it for you: "
            + ("cmake is not on PATH" if shutil.which("cmake") is None else
               "--dry-run does not build") + ".\n"
            "    See the folder README.")
        return

    print(f"[build] sibling C++ extension against Drake at {prefix} "
          "(several minutes)", flush=True)
    build_dir = os.path.join("cpp_parameterization", "build")
    for argv in (["cmake", "-S", os.path.join("cpp_parameterization", "cpp"),
                  "-B", build_dir, f"-DCMAKE_PREFIX_PATH={prefix}",
                  "-DOPTIMIZED_BUILD=ON"],
                 ["cmake", "--build", build_dir, "--target", "_iiwa_ik",
                  "-j", str(max(1, (os.cpu_count() or 2) - 1))]):
        r = subprocess.run(argv, cwd=SIBLING)
        if r.returncode != 0:
            problems.append(
                f"building the sibling's C++ extension failed ({' '.join(argv[:3])} "
                f"... exited {r.returncode}). See ../docs/INSTALL.md.")
            return
    # The link is LTO and the .so is 0 bytes until it finishes; importing it in
    # that window fails with "file too short", which reads like a broken build
    # rather than an unfinished one.
    if not sibling_extension_built():
        problems.append(
            "the sibling's C++ extension built but its .so is still short -- the "
            "LTO link has not finished. Wait for it to reach ~64 MB and re-run.")
        return
    with open(EXT_STAMP, "w") as f:
        f.write(_stamp_text(prefix))


def check_extension_imports(problems):
    """Import iiwa_ik the way the stages will, before spending an hour rendering."""
    env = dict(os.environ)
    env.update(iiwa_env())
    r = subprocess.run([PY, "-c", "import iiwa_ik"], env=env,
                       capture_output=True, text=True, cwd=REPO)
    if r.returncode != 0:
        tail = (r.stderr.strip().splitlines() or ["(no output)"])[-1]
        problems.append(
            f"the sibling's iiwa_ik extension does not import against "
            f"{_DRAKE_PREFIX}:\n"
            f"    {tail}\n"
            "    An \"unknown base type\" here is the pybind11 two-registry "
            "problem: the .so was\n    built against a different Drake than the "
            "one on PYTHONPATH. Deleting the .so in\n"
            f"    {os.path.relpath(EXT_DIR, REPO)}/ and re-running rebuilds it "
            "against the right one.")


def print_problems(problems, header):
    print(header + "\n")
    for p in problems:
        print("  * " + p + "\n")


def check_prereqs(need_hardware, need_sibling, need_sibling_ext,
                  may_fetch=True):
    """Fail before spending an hour of rendering, not during.

    ``may_fetch`` is off for --dry-run, which reports what is missing without
    downloading, cloning or compiling anything, and warns instead of exiting.

    Anything the RA-L half also needs is fatal here. Problems that belong to the
    *overview* half alone -- Drake, the IIWA experiment folder, its extension --
    are returned instead, so that ``all`` can still build the RA-L video on a
    machine that cannot build the overview.
    """
    problems = []

    blender = None
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from blender_paths import find_blender
        blender = find_blender()
    except Exception as e:
        problems.append(f"Blender: {e}")

    if blender:
        installer = os.path.join(REPO, "scripts", "video",
                                 "install_meshcat_importer.py")
        r = subprocess.run([PY, installer, "--check"],
                           capture_output=True, text=True)
        if r.returncode != 0 and may_fetch:
            # The add-on lives outside the repo, so a new machine always fails
            # this check -- and the remedy is exactly "install the vendored
            # copy", which is not worth making someone run by hand. It backs up
            # any existing tree rather than overwriting it, so a *drifted*
            # install is preserved as .bak instead of lost.
            print("[install] pinned meshcat importer add-on", flush=True)
            subprocess.run([PY, installer])
            r = subprocess.run([PY, installer, "--check"],
                               capture_output=True, text=True)
        if r.returncode != 0:
            problems.append(
                "meshcat importer add-on does not match the pinned tree"
                + ("" if may_fetch else " (the real build installs it; "
                                        "--dry-run does not)") + "; run\n"
                "    .venv/bin/python scripts/video/install_meshcat_importer.py\n"
                f"  ({r.stdout.strip() or r.stderr.strip()})")

    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            problems.append(f"{tool} not on PATH")

    missing_fonts = [f for f in FONTS
                     if not os.path.isfile(os.path.join(FONT_DIR, f))]
    if missing_fonts:
        problems.append(
            f"missing {', '.join(missing_fonts)} under {FONT_DIR}. Every card and "
            "overlay names these by absolute path, and the ffmpeg drawtext ones "
            "have no fallback (Debian/Ubuntu: fonts-dejavu-core).")

    # The pipeline runs its stages as `sys.executable -m ...`, so the packages
    # have to be importable *here*, not merely installed somewhere.
    for module, why in VIDEO_IMPORTS:
        r = subprocess.run([PY, "-c", f"import {module}"],
                           capture_output=True, text=True)
        if r.returncode != 0:
            problems.append(
                f"{module} is not importable by {PY} (needed for {why}); install "
                f"the video extra:\n    pip install -e '.[video]'")
        elif module == "manim":
            # Every equation in the explainer scenes is a MathTex, which manim
            # renders by shelling out to latex and dvisvgm. Missing, it fails
            # per-scene with manim's own error, well into the build.
            missing_tex = [t for t in ("latex", "dvisvgm")
                           if shutil.which(t) is None]
            if missing_tex:
                problems.append(
                    f"{', '.join(missing_tex)} not on PATH; manim renders the "
                    "MathTex equations in manim_scenes_v2.py through LaTeX "
                    "(Debian/Ubuntu: texlive, texlive-latex-extra, dvisvgm).")

    if need_hardware:
        clips = sorted(f for f in os.listdir(HW_DIR)
                       if f.endswith(".mp4")) if os.path.isdir(HW_DIR) else []
        if len(clips) != 20:
            problems.append(
                f"expected 20 hardware clips in {HW_DIR}, found {len(clips)}. "
                "This capture is NOT reproducible and is not distributable, so "
                "it is not part of the release. Do not substitute other clips: "
                "compose_rby1_v2.py maps seed->file positionally via "
                "sorted(os.listdir()).")
        else:
            # The clips are addressed *by name*: render_segment1 and
            # compose_rby1_v2 look each one up in trim_points.json with
            # .get(name, {}), so a renamed or re-encoded set does not fail --
            # it silently trims from 0.0 s and the plots run against footage
            # that is tens of seconds out of step. Twenty files of the right
            # count is not the same as the right twenty files.
            with open(TRIM_JSON) as f:
                expected = sorted(json.load(f))
            if clips != expected:
                unknown = [c for c in clips if c not in expected]
                absent = [e for e in expected if e not in clips]
                problems.append(
                    f"the clips in {HW_DIR} do not match videos/trim_points.json.\n"
                    + (f"    not in trim_points.json: {', '.join(unknown)}\n"
                       if unknown else "")
                    + (f"    missing: {', '.join(absent)}\n" if absent else "")
                    + "    Those names are the sync keys, and an unmatched one "
                      "silently trims from 0.0 s.\n"
                      "    Restore the original files under their original names; "
                      "do not re-derive the trims.")

    if problems:
        print_problems(problems, "Cannot build:" if may_fetch else
                       "Would not be able to build:")
        if may_fetch:
            sys.exit(1)

    sibling_problems = []
    if need_sibling:
        global _DRAKE_PREFIX
        _DRAKE_PREFIX = (ensure_drake(sibling_problems, may_fetch)
                         if need_sibling_ext else None)
        ensure_sibling(sibling_problems, _DRAKE_PREFIX, need_sibling_ext,
                       may_fetch)
        if need_sibling_ext and _DRAKE_PREFIX and not sibling_problems:
            try:
                import pydrake
                other = os.path.dirname(pydrake.__file__)
            except ImportError:
                other = "no importable pydrake"
            print(f"[drake] iiwa_ik built and run against {_DRAKE_PREFIX} "
                  f"({drake_version(_DRAKE_PREFIX)});\n"
                  f"        every other stage uses {other}")
            check_extension_imports(sibling_problems)
    return sibling_problems


# The video scripts import a handful of shared helper modules -- the crop and
# pane geometry, the camera projection overlays use, the render settings, the
# credits, the frame-compositing pool -- which no stage names in argv but which
# decide what its output looks like. They are found by reading the imports
# rather than by keeping a list per stage: a list would be one more thing to
# forget, and forgetting it ships the previous version of a segment.
VIDEO_DIR = os.path.join(REPO, "scripts", "video")


def local_imports(path, seen=None):
    """Absolute paths of the scripts/video modules ``path`` imports, transitively."""
    seen = seen if seen is not None else set()
    path = os.path.abspath(path)
    if path in seen or not os.path.isfile(path):
        return seen
    seen.add(path)
    try:
        tree = ast.parse(open(path).read())
    except (OSError, SyntaxError):
        return seen
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
    for n in names:
        mod = os.path.join(VIDEO_DIR, n + ".py")
        if os.path.isfile(mod):
            local_imports(mod, seen)
    return seen


def stage_sources(argv, opts):
    """Everything a stage's output depends on, as absolute paths.

    Three sources, none of which has to be maintained by hand for the common
    case: any existing file named in argv (the stage's own script, and the
    meshcat HTML for the renderers that take one as an argument), whatever
    scripts/video modules those scripts import, plus an explicit ``deps`` list for inputs the script
    hardcodes instead of taking as an argument -- and for the assembly, whose
    inputs are every segment.
    """
    # Only source-shaped extensions: argv also carries *output* paths, and
    # treating "--output .../main_cpp_executed.ipynb" as an input made a
    # notebook stage look permanently stale and re-run a 2.5-hour execution.
    named = [os.path.abspath(a) for a in argv if isinstance(a, str)
             and a.endswith((".py", ".html")) and os.path.isfile(a)]
    imported = set()
    for p in named:
        if p.endswith(".py"):
            local_imports(p, imported)
    return (named + sorted(imported - set(named))
            + [os.path.abspath(d) for d in opts.get("deps", ())])


def run_stage(stage, args):
    name, out, argv, min_size = stage[:4]
    opts = stage[4] if len(stage) > 4 else {}

    # Resumability is by output existence, but an output that predates the
    # script that produced it is not "already built" -- it is the *previous*
    # version of the segment. Editing manim_scenes_v2.py and re-running the
    # build once skipped the manim stage on exactly this basis and shipped the
    # old gradients diagram inside an otherwise-updated video.
    #
    # Staleness is decided by source-file *mtime*, not by argv content: only
    # files named in argv (plus their local imports and any explicit `deps`)
    # are checked. Editing a literal string embedded in build_video.py itself
    # -- one of the --title/--subtitle values above, say -- does not mark a
    # stage stale, because build_video.py is not in any stage's argv. Such a
    # change needs --force-from.
    #
    # A dep that does not exist yet is an upstream stage that has not run; when
    # it does, its output is newer and this fires normally. Calling getmtime on
    # it unguarded turned --dry-run on a half-built tree into a traceback.
    stale = [p for p in stage_sources(argv, opts)
             if os.path.exists(out) and os.path.exists(p)
             and os.path.getmtime(p) > os.path.getmtime(out)]

    if (os.path.exists(out) and os.path.getsize(out) >= min_size
            and not stale and not args._forcing(name)):
        print(f"[skip] {name}: {os.path.relpath(out, REPO)} already built")
        return
    if stale and not args._forcing(name):
        print(f"[stale] {name}: {os.path.relpath(stale[0], REPO)} is newer than "
              f"{os.path.relpath(out, REPO)}")

    if args.dry_run:
        print(f"[would run] {name}: {' '.join(argv[:4])} ...")
        return

    print(f"[run ] {name}")
    t0 = time.time()

    if argv[0] == "__copy__":
        shutil.copy(argv[1], argv[2])
    else:
        env = dict(os.environ)
        # Callable, for stages whose environment depends on something resolved
        # after the stage list was built (the Drake prefix).
        env_opt = opts.get("env", {})
        env.update(env_opt() if callable(env_opt) else env_opt)
        r = subprocess.run(argv, cwd=opts.get("cwd", REPO), env=env)
        if r.returncode != 0:
            print(f"\n[FAIL] {name} exited {r.returncode}")
            sys.exit(r.returncode)

    # A zero exit is not proof the stage produced anything.
    if not os.path.exists(out):
        print(f"\n[FAIL] {name} exited 0 but did not write {out}")
        sys.exit(1)
    size = os.path.getsize(out)
    if size < min_size:
        print(f"\n[FAIL] {name} wrote only {size} bytes to {out} "
              f"(expected >= {min_size}) -- treating as a failed render")
        sys.exit(1)
    print(f"[ok  ] {name}  {time.time() - t0:.0f}s  "
          f"-> {os.path.relpath(out, REPO)} ({size / 1e6:.1f} MB)")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target", choices=["overview", "ral", "all"])
    ap.add_argument("--force", action="store_true",
                    help="rebuild every stage")
    ap.add_argument("--force-from", metavar="STAGE", default=None,
                    help="rebuild from this stage onward")
    ap.add_argument("--skip-iiwa", action="store_true",
                    help="leave any existing v2_iiwa_bimanual.mp4 alone and do "
                         "not run the IIWA experiment's notebook")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan without running anything")
    ap.add_argument("--list", action="store_true",
                    help="print the stage names and exit")
    args = ap.parse_args()

    # RA-L first under `all`: it needs neither Drake nor the IIWA experiment
    # repo, so a machine that cannot build the overview still gets one video
    # rather than nothing. (The dedup below then keeps montage_ral as the
    # producer of montage_20.mp4 -- same script, same output.)
    ral = ral_stages(args) if args.target in ("ral", "all") else []
    overview_only = overview_stages(args) if args.target in ("overview", "all") else []
    stages = ral + overview_only
    ral_names = {st[0] for st in ral}

    # `all` asks for montage_20.mp4 from both halves; keep the first.
    seen, deduped = set(), []
    for s in stages:
        if s[1] in seen:
            continue
        seen.add(s[1])
        deduped.append(s)
    stages = deduped

    if args.list:
        for s in stages:
            print(s[0])
        return

    names = [s[0] for s in stages]
    if args.force_from and args.force_from not in names:
        ap.error(f"unknown stage {args.force_from!r}; --list shows them")
    start = names.index(args.force_from) if args.force_from else None

    def _forcing(name):
        if args.force:
            return True
        return start is not None and names.index(name) >= start
    args._forcing = _forcing

    os.makedirs(VIDEO, exist_ok=True)
    # --skip-iiwa only skips the *segment*. domain_ext_html and iris_html read
    # the IIWA experiment too, and iris_html imports its built extension, so the
    # flag cannot switch the whole check off -- doing that let a build get 20
    # minutes in before failing on an import.
    overview = args.target in ("overview", "all")
    sibling_problems = check_prereqs(
        need_hardware=True,
        need_sibling=overview,
        need_sibling_ext=overview,
        may_fetch=not args.dry_run)

    if sibling_problems:
        print_problems(sibling_problems,
                       "Cannot build the overview video:" if not args.dry_run
                       else "Would not be able to build the overview video:")
        if not args.dry_run:
            if args.target == "overview":
                sys.exit(1)
            # `all`: build what does not depend on any of the above, then fail.
            print("The RA-L video needs none of that, so it is built anyway.\n")
            stages = [st for st in stages if st[0] in ral_names]

    for s in stages:
        run_stage(s, args)

    print("\nDone.")
    for name in ("promo_video.mp4", "hardware_supplementary_anonymous.mp4",
                 "hardware_supplementary_non_anonymous.mp4"):
        p = os.path.join(VIDEO, name)
        if os.path.exists(p):
            dur = subprocess.check_output([
                "ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                "-of", "csv=p=0", p]).decode().strip()
            print(f"  {p}  ({float(dur):.1f}s)")

    if sibling_problems and not args.dry_run:
        print()
        print_problems(sibling_problems, "The overview video was NOT built:")
        sys.exit(1)


if __name__ == "__main__":
    main()
