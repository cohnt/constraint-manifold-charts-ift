"""Interactive visualiser for the 20-point grid runs: click a point, watch its
cached plan in meshcat.

Serves a small local page with the grid laid out as it is on the table (4 columns
of box x, 5 rows of box y, index = ix*grid_ny + iy), coloured by status and
annotated with each point's key numbers, next to the live meshcat view. Clicking a
point replays that point's cached plan -- reach, grasp, lift, place, home -- in
meshcat. Nothing is replanned: every configuration comes from
``plans/grid_cache/point_NN.pkl``.

Why a local web page rather than a matplotlib window
----------------------------------------------------
The obvious implementation is a matplotlib figure with an ``mpl_connect``
``button_press_event`` handler. It does not work here: this machine has no
graphical session for the user's shell (``DISPLAY`` is unset, matplotlib resolves
to the ``Agg`` backend), so ``plt.show()`` would open nothing. Meshcat is already a
browser page, so the browser is the one surface guaranteed to be reachable -- over
an SSH tunnel as much as locally. Putting the grid in the same page as the meshcat
iframe also means the render appears next to the point you clicked instead of in
another window.

The interaction is a thin shell over one plain function, ``show_point(index)``, so
the mechanism can be swapped (or driven from a script, or from ``--show``) without
touching the replay logic.

The canonical status diagram is still ``plan_grid.plot_grid_diagram``:
it is rendered to a PNG (into a temp dir -- this script never writes to the cache)
and shown beside the clickable grid, so the legend, the corner rings and the
iteration-order arrows come from the one place that already draws them, and the
clickable HTML grid stays crisp and can carry per-point numbers the PNG cannot fit.

Why one meshcat and two persistent diagrams
-------------------------------------------
``render_grid_videos.py`` builds a fresh plant per *scene* per point (open box as a
static obstacle for the reach legs, box attached to the gripper for the carry legs,
placed box as an obstacle for the home legs), which is right for offline rendering
but wrong for an interactive viewer: every new meshcat-attached diagram re-uploads
the robot's whole mesh set to the browser, and every plant leaks pydrake memory
that a long-lived session never gets back.

So the robot comes from ``plan_grid.build_viz``'s two persistent
diagrams -- empty-handed and box-attached -- built once and selected per leg by the
leg's own cached ``with_box`` tag. That keeps the *model* per leg exactly what the
planner used for the carried box (including that the empty-handed diagram's fingers
are open and the box-attached one's are closed, which is what the robot actually
does). The two *static* box states, on the other hand, are per point and are only
scenery in a replay -- nothing is collision-checked here -- so they are drawn
directly into meshcat from ``WALL_SPECS`` (the same wall geometry
``open_box_walls`` hands the planner) and shown or hidden per leg. Visibility, not
reconstruction, is also the only way to get this right at all: meshcat keeps
whatever geometry was last sent, so with three swapped plants the pick box would
still be sitting on the table while the robot carries it away.

Which box is shown for which leg follows ``verify_cached_plan.py``'s rule rather
than a leg-name table: ``with_box`` legs show the attached box, legs before the
first ``with_box`` leg show the pick box, legs after it show the placed box (FK of
the held box at the last carried configuration). That is name-independent, so it
still holds for a plan whose legs were named differently.

Why a recorded animation instead of real-time playback
------------------------------------------------------
A point's plan runs 60-100 s. Streaming it to meshcat in real time (what
``preview_legs`` does) would block the click for that long and offer no way to
rewind the one second that matters. Instead each click builds a meshcat *animation*
and publishes it, which returns in about a second and gives the browser a
play/pause/scrub slider over the whole plan. ``--realtime`` restores the streaming
behaviour (via ``preview_legs``) for anyone who wants it.

Usage:
    .venv/bin/python scripts/grid_visualizer.py
    .venv/bin/python scripts/grid_visualizer.py --cache-dir plans/grid_cache_run3_14of20
    .venv/bin/python scripts/grid_visualizer.py --show 0 --no-serve   # scripted check
Then open the page it prints (grid + embedded meshcat) and click a point.
"""

import argparse
import atexit
import html
import json
import os
import queue
import shutil
import sys
import tempfile
import threading
import time
import traceback
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit, urlunsplit

import matplotlib
import numpy as np

# Force the non-interactive backend before anything imports pyplot: plot_grid_diagram
# is reused here to draw the status PNG, and this script never opens a window (there is
# no display for the user's shell, and the page is the UI). Setting it explicitly also
# keeps a stray DISPLAY from pulling in a GUI backend that would want the main thread
# it cannot have.
matplotlib.use("Agg")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pydrake.all import Box, Rgba, RigidTransform, Role, RollPitchYaw, StartMeshcat

import plan_grid as E
from plan_format.plan_io import load_plan
from plan_format.trajectory_conversion import preview_legs

# Meshcat paths for the two static box states this script draws itself. Under
# "grid_viz/" so they are visibly not part of the plant's own "visual/" tree, and
# with the five walls as children of one parent, so a whole box moves (or hides)
# with a single call on the parent.
PICK_ROOT = "grid_viz/pick_box"
PLACED_ROOT = "grid_viz/placed_box"
# The box while it is carried. Scenery, like the other two, rather than the
# box-attached plant's own welded held box: setting `visible` on a *leaf*
# geometry path detaches it from its frame's transform track when the recording
# is exported and imported into Blender, so the carried box arrived in the video
# hanging motionless beside the torso. A group root keeps both channels. See
# On visibility across the meshcat -> Blender
# boundary".
CARRIED_ROOT = "grid_viz/carried_box"

# Animation frame rate. The stored legs are sampled at the planner's viz_hz (30 Hz),
# which is finer than needed for inspection: meshcat interpolates between keyframes,
# so 15 fps still plays smoothly while halving the animation the browser has to
# swallow for a 90 s plan.
DEFAULT_FPS = 15.0

# Colours mirror plot_grid_diagram's, so the clickable grid and the PNG beside it
# agree on what green and red mean.
COLOR_SUCCESS = "#008300"
COLOR_FAILED = "#e34948"
COLOR_UNKNOWN = "#b0b0b0"
COLOR_CORNER = "#2a78d6"

# Box visibility per leg scene: (pick box, placed box, attached box).
SCENE_VISIBILITY = {
    "pick": (True, False, False),
    "held": (False, False, True),
    "placed": (False, True, False),
}


def leg_scene_kinds(legs):
    """Which box state each leg is replayed against: pick / held / placed.

    verify_cached_plan.py's rule, not render_grid_videos.py's leg-name table: a
    ``with_box`` leg carries the box; anything before the first such leg is still
    approaching the box on the table; anything after it has released the box where
    it was placed. Derived from the plan's own tags, so it holds for a plan with
    differently named legs (or a reach-only corner plan, which is all "pick").
    """
    kinds, seen_box = [], False
    for leg in legs:
        if leg.get("with_box"):
            seen_box = True
            kinds.append("held")
        else:
            kinds.append("placed" if seen_box else "pick")
    return kinds


def add_open_box(meshcat, root, rgba):
    """Draw one open-top box (base + 4 walls) as children of ``root``.

    Straight from WALL_SPECS -- the same (name, size, local offset) list
    open_box_walls turns into the planner's obstacles -- so the box the viewer
    shows is the box the planner planned around. The walls are placed in the
    box frame once; the box is positioned afterwards by transforming ``root``.

    Module level so the offline segment generators can build the same scenery
    the interactive viewer does (see scripts/video/generate_stability_meshcat.py).
    """
    for name, size, local in E.WALL_SPECS:
        path = f"{root}/{name}"
        meshcat.SetObject(path, Box(*size), rgba)
        meshcat.SetTransform(path, RigidTransform(np.asarray(local, float)))
    meshcat.SetProperty(root, "visible", False)


def _held_box_meshcat_paths(diagram, prefix="visual"):
    """Meshcat paths of the held-box visual geometry in a box-attached diagram.

    MeshcatVisualizer publishes each geometry at
    ``<prefix>/<scoped frame name>/<scoped geometry name>`` with ``::`` replaced by
    ``/``, so the paths can be *derived* from the scene graph instead of hardcoded
    as strings. The caller checks each one with ``Meshcat.HasPath`` before relying
    on it, which turns a future change in that convention into a printed warning
    and a slightly wrong-looking replay rather than a silent no-op.

    The held box has to be hidden per leg because meshcat geometry is keyed by path:
    the box is registered on the ``ee_right`` frame, both diagrams drive that frame,
    so once created it follows the gripper during the empty-handed legs too.
    """
    insp = diagram.GetSubsystemByName("scene_graph").model_inspector()
    paths = []
    for gid in insp.GetAllGeometryIds(Role.kIllustration):
        gname = insp.GetName(gid)
        if "held_box" not in gname:
            continue
        fid = insp.GetFrameId(gid)
        fname = "" if fid == insp.world_frame_id() else insp.GetName(fid)
        parts = [prefix] + [s.replace("::", "/") for s in (fname, gname) if s]
        paths.append("/".join(parts))
    return paths


def _log_tail(cache_dir, index, n_lines=6):
    """Last few meaningful lines of a point's planner log, or "".

    The fallback for *why* a point has no plan when status.json does not say (it is
    written at the end of a --precompute run, so a run still in flight has none):
    ``point_NN.log`` is on disk from the first GCP branch onwards.
    """
    path = E.log_path_for(cache_dir, index)
    if not os.path.exists(path):
        return ""
    try:
        with open(path, errors="replace") as f:
            lines = [ln.rstrip() for ln in f.readlines() if ln.strip()]
    except OSError:
        return ""
    return "\n".join(lines[-n_lines:])


def _fmt(value, spec="", dash="-"):
    """Format a number that may legitimately be missing (older plans, no status)."""
    if value is None:
        return dash
    try:
        return format(value, spec)
    except (TypeError, ValueError):
        return str(value)


class _Job:
    """One piece of work handed to the main thread, plus somewhere to put its result."""

    def __init__(self, fn):
        self.fn = fn
        self.done = threading.Event()
        self.result = None

    def run(self):
        try:
            self.result = self.fn()
        except Exception as e:               # never let one bad job stop the pump
            traceback.print_exc()
            self.result = dict(ok=False, message=f"{type(e).__name__}: {e}")
        finally:
            self.done.set()


class GridPlanViewer:
    """Everything the page needs: the cache summary, the PNG, and the meshcat replay.

    Deliberately usable without the web server: ``show_point(index)`` is a plain
    function and ``summaries`` is plain data, so the same object drives ``--show``,
    a scripted check, or any other front end.

    **Everything that touches meshcat runs on the thread that built it.** Drake's
    Meshcat pins itself to its creating thread: every method asserts
    ``IsThread(main_thread_id_)`` (meshcat.cc:933) and, when that fails, raises
    ``SystemExit`` -- which ``threading`` discards without printing anything. An HTTP
    worker calling ``show_point`` directly therefore died mid-request and dropped the
    connection with no error in the log and no traceback anywhere, which is exactly
    what it did before ``submit``/``run_main_loop`` existed. So the workers only ever
    *submit* work, and the main thread runs all of it.
    """

    def __init__(self, cache_dir, grid_path, *, fps=DEFAULT_FPS, realtime=False,
                 html_dir=None):
        self.cache_dir = cache_dir
        self.grid_path = grid_path
        self.fps = float(fps)
        self.realtime = bool(realtime)
        # When set, the page serves pre-exported self-contained renders instead of
        # driving the live meshcat. That removes the websocket from the picture
        # entirely, which matters because the meshcat port is a second port the viewer
        # has to be able to reach and in practice often cannot.
        self.html_dir = html_dir

        grid = np.load(grid_path)
        self.grid_points = np.asarray(grid["grid_points"], dtype=float)
        self.grid_nx = int(grid["grid_nx"])
        self.grid_ny = int(grid["grid_ny"])
        self.corner_idxs = list(E.corner_indices(self.grid_nx, self.grid_ny))
        self.n_points = len(self.grid_points)

        # One meshcat for the whole session, one diagram pair for the whole session.
        self.meshcat = StartMeshcat()
        # Cached because web_url() is itself a Meshcat call, and the page that wants it
        # is rendered on an HTTP worker thread where any Meshcat call is fatal.
        self.meshcat_url = self.meshcat.web_url()
        self.viz = E.build_viz(self.meshcat)

        # make_default_rby1_infrastructure attaches a second MeshcatVisualizer at
        # Role.kProximity under the "collision" prefix, and that visualizer registers
        # its geometry with meshcat when the diagram is *built* -- before anything is
        # published. So merely declining to publish it is not enough to keep it out:
        # the tree is already there, `visible_by_default=False` only hides it, and
        # StaticHtml serialises hidden geometry just the same. Delete the subtree.
        # Nothing republishes it, because only the visual recorders are ever
        # published below and in _pose_robot.
        #
        # This matters beyond page weight: the Blender renders import these exports,
        # and a proximity mesh that arrives there is a stray grey blob in the video.
        # The Blender-side filter matches on object names and cannot be relied on --
        # the importer can collapse a leaf so a proximity mesh arrives named plain
        # "collision".
        self.meshcat.Delete("collision")

        # Per-diagram handles for the *visual* MeshcatVisualizer only. Publishing that
        # subsystem instead of the whole diagram keeps the collision-geometry tree out
        # of the recording, which halves what the browser has to load.
        self.recorders = {}
        for with_box, (plant, pctx, ctx, diagram, idxs) in self.viz.items():
            rec = diagram.GetSubsystemByName("meshcat_visualizer(visual)")
            self.recorders[with_box] = (rec, rec.GetMyContextFromRoot(ctx))
            rec.ForcedPublish(rec.GetMyContextFromRoot(ctx))

        plant_b, _, _, diagram_b, _ = self.viz[True]
        self.ee_body = plant_b.GetBodyByName(
            f"ee_{E.HAND}", plant_b.GetModelInstanceByName(f"{E.HAND}_arm"))

        # Static pick/placed boxes, drawn once and thereafter only moved and hidden.
        rgba = Rgba(*E.BOX_COLOR)
        for root in (PICK_ROOT, PLACED_ROOT, CARRIED_ROOT):
            self._add_open_box(root, rgba)
        self.held_paths = [p for p in _held_box_meshcat_paths(diagram_b)
                           if self.meshcat.HasPath(p)]
        if not self.held_paths:
            print("WARNING: could not locate the held box's meshcat paths; the "
                  "plant's own held box will stay visible alongside the "
                  "scenery one during the carried legs.")

        self.jobs = queue.Queue()          # HTTP workers -> the main thread
        self.selected = None
        # The status PNG goes to a temp dir, never into the cache: a planner run may be
        # writing there, and its own grid_status.png is its record, not ours to touch.
        png_dir = tempfile.mkdtemp(prefix="grid_visualizer_")
        atexit.register(shutil.rmtree, png_dir, True)
        self.png_path = os.path.join(png_dir, "grid_status.png")
        self.reload()

    # ── main-thread work queue ───────────────────────────────────────────────

    def submit(self, fn, timeout=900.0):
        """Have the main thread run ``fn``, and wait here for its result.

        The one way an HTTP worker is allowed to reach meshcat (see the class
        docstring). The timeout is a backstop for a --realtime playback that outlasts
        the browser's patience rather than an expected outcome; jobs are run in
        submission order, so a second click simply queues behind the first.
        """
        job = _Job(fn)
        self.jobs.put(job)
        if not job.done.wait(timeout):
            return dict(ok=False, message=f"timed out after {timeout:.0f}s "
                                          "waiting for the replay to finish")
        return job.result

    def run_main_loop(self):
        """Run submitted jobs until interrupted. The main thread's job while serving.

        Polls rather than blocking indefinitely so Ctrl-C is always prompt.
        """
        while True:
            try:
                job = self.jobs.get(timeout=0.25)
            except queue.Empty:
                continue
            job.run()

    # ── cache reading ────────────────────────────────────────────────────────

    def reload(self):
        """Re-read status.json and the cache, and redraw the status PNG.

        Worth having as an explicit action: a --precompute run writes point_NN.pkl
        as each point finishes and status.json only at the end, so a viewer opened
        during a run goes stale and a reload is the difference between "that point
        failed" and "that point had not finished yet".
        """
        try:
            self.status = E.load_status(self.cache_dir)
        except (OSError, ValueError) as e:
            # A status.json being rewritten by a live run can be read mid-write.
            print(f"WARNING: could not read status.json ({e}); "
                  "falling back to the cache alone.")
            self.status = {}
        self.summaries = [self._summarise(i) for i in range(self.n_points)]
        # plot_grid_diagram is reused verbatim (it only needs {"<i>": {"status": ...}}),
        # and pointed at a temp dir: this script must never write into a cache
        # directory a planner may be using.
        E.plot_grid_diagram(
            self.grid_points, self.corner_idxs,
            {str(s["index"]): {"status": s["status"]} for s in self.summaries},
            self.png_path)
        self.png_stamp = int(time.time())
        return self.summaries

    def _summarise(self, index):
        """One point's row: status, the key numbers, and whether it can be replayed.

        Prefers status.json (what the harness recorded) and falls back to the cached
        plan's own meta -- via the same evidence_summary() reduction update_status
        uses, so the numbers do not change depending on which source answered.
        """
        bx, by = self.grid_points[index]
        entry = self.status.get(str(index)) or {}
        path = E.cache_path_for(self.cache_dir, index)
        out = dict(
            index=index, ix=index // self.grid_ny, iy=index % self.grid_ny,
            bx=float(bx), by=float(by),
            status=entry.get("status"), stage=entry.get("stage"),
            detail=entry.get("detail", ""), updated_at=entry.get("updated_at"),
            gcp_index=entry.get("gcp_index"), grasp_seed=None,
            worst_clearance_mm=entry.get("worst_clearance_mm"),
            worst_com_margin_mm=entry.get("worst_com_margin_mm"),
            worst_joint_limit_excess_rad=entry.get("worst_joint_limit_excess_rad"),
            total_duration_s=entry.get("total_duration_s"),
            all_legs_ok=entry.get("all_legs_ok"),
            has_plan=False, n_legs=0, legs=[], source="status.json" if entry else "none",
            note="",
        )
        if not os.path.exists(path):
            if out["status"] is None:
                out["note"] = "no cached plan"
            return out

        try:
            _, meta = load_plan(path)
        except Exception as e:
            # A plan being written right now by a live run reads as a truncated
            # pickle. That is a "come back in a minute", not a failure.
            out["note"] = f"cached plan unreadable ({type(e).__name__}: {e})"
            return out

        legs = meta.get("legs") or []
        out.update(
            has_plan=bool(legs), n_legs=len(legs),
            legs=[dict(name=lg.get("name", "?"),
                       duration=float(lg.get("duration", float("nan"))),
                       with_box=bool(lg.get("with_box")),
                       n=int(np.asarray(lg["q"]).shape[0]) if "q" in lg else 0)
                  for lg in legs],
        )
        if out["gcp_index"] is None:
            out["gcp_index"] = meta.get("gcp_index")
        out["grasp_seed"] = meta.get("grasp_seed")
        evidence = meta.get("evidence") or {}
        if evidence and out["worst_clearance_mm"] is None:
            try:
                out.update(E.evidence_summary(evidence))
                out["source"] = "cache"
            except (KeyError, TypeError, ValueError):
                pass   # older evidence dicts may not carry every key
        if out["status"] is None:
            # A cached plan is the harness's own success criterion (it refuses to
            # cache a plan whose legs did not pass the dense guarantee check), so a
            # readable plan with no status entry is a success that status.json has
            # not caught up with.
            out["status"] = "success"
            out["source"] = "cache"
        if not out["total_duration_s"] and out["legs"]:
            out["total_duration_s"] = round(sum(lg["duration"] for lg in out["legs"]), 2)
        return out

    # ── meshcat scenery ──────────────────────────────────────────────────────

    def _add_open_box(self, root, rgba):
        add_open_box(self.meshcat, root, rgba)

    def _set_box_state(self, kind, q23=None, time_in_recording=None):
        """Show exactly the one box state ``kind`` calls for.

        Re-asserted on every recorded frame rather than only at the leg boundaries:
        a meshcat animation is a set of interpolated keyframe tracks, and a boolean
        that only has keyframes at transitions is at the mercy of how the viewer
        interpolates it. Setting it per frame costs nothing here -- recording runs
        with set_visualizations_while_recording=False, so these calls only append to
        an in-memory animation and never hit the socket.

        ``q23`` is the configuration this frame is being posed at; on a carried leg
        it moves the carried box with the gripper. The plant's own welded held box
        is hidden unconditionally -- see CARRIED_ROOT.
        """
        pick_v, placed_v, held_v = SCENE_VISIBILITY[kind]
        self.meshcat.SetProperty(PICK_ROOT, "visible", pick_v,
                                 time_in_recording=time_in_recording)
        self.meshcat.SetProperty(PLACED_ROOT, "visible", placed_v,
                                 time_in_recording=time_in_recording)
        if held_v and q23 is not None:
            self.meshcat.SetTransform(CARRIED_ROOT, self._held_box_pose(q23),
                                      time_in_recording=time_in_recording)
        self.meshcat.SetProperty(CARRIED_ROOT, "visible",
                                 bool(held_v and q23 is not None),
                                 time_in_recording=time_in_recording)
        for path in self.held_paths:
            self.meshcat.SetProperty(path, "visible", False,
                                     time_in_recording=time_in_recording)

    def _held_box_pose(self, q23):
        """World pose of the carried box at configuration ``q23``.

        FK of the held box -- the same composition
        verify_cached_plan._placed_box_pose and render_grid_videos do, against the
        box-attached plant this viewer already holds. Called with the last carried
        configuration it gives the pose the plan released the box at; called per
        frame it carries the box along with the gripper.
        """
        plant, pctx, ctx, _, idxs = self.viz[True]
        q_full = plant.GetPositions(pctx)
        q_full[idxs] = q23
        plant.SetPositions(pctx, q_full)
        return (plant.EvalBodyPoseInWorld(pctx, self.ee_body)
                @ RigidTransform(RollPitchYaw(E.RPY), E.OFFSET))

    def _pose_robot(self, with_box, q23, t=None):
        """Put the robot at ``q23`` in the chosen diagram and publish it."""
        plant, pctx, ctx, _, idxs = self.viz[with_box]
        rec, rctx = self.recorders[with_box]
        q_full = plant.GetPositions(pctx)
        q_full[idxs] = q23
        plant.SetPositions(pctx, q_full)
        if t is not None:
            ctx.SetTime(t)
        rec.ForcedPublish(rctx)

    # ── replay ───────────────────────────────────────────────────────────────

    def static_html_path(self, index):
        """Path to point ``index``'s exported render, or None if it is not there."""
        if not self.html_dir:
            return None
        path = os.path.join(self.html_dir, f"point_{index:02d}.html")
        return path if os.path.exists(path) else None

    def show_point(self, index):
        """Replay grid point ``index``'s cached plan in meshcat.

        The whole of the click path: ``--show`` calls this directly and the HTTP
        handler reaches it through ``submit``, so both run it on the main thread (the
        only thread allowed to touch meshcat). Never raises -- a point with no plan,
        an unreadable pickle or a plan with no legs comes back as
        ``{"ok": False, "message": ...}`` so the caller can say so instead of dying.
        """
        try:
            return self._show_point(index)
        except Exception as e:
            traceback.print_exc()
            return dict(ok=False, index=index,
                        message=f"replaying point {index} failed: "
                                f"{type(e).__name__}: {e}")

    def _show_point(self, index):
        if not (0 <= index < self.n_points):
            return dict(ok=False, index=index,
                        message=f"index {index} is outside the {self.n_points}-point grid")
        summary = self.summaries[index]
        if not summary["has_plan"]:
            why = summary["note"] or "no cached plan"
            if summary["status"] == "failed":
                why = (f"planning failed at stage {summary['stage']!r}"
                       if summary["stage"] else "planning failed")
            hint = summary["detail"] or _log_tail(self.cache_dir, index)
            print(f"[{index:2d}] nothing to replay: {why}"
                  + (f"\n     {hint}" if hint else ""))
            return dict(ok=False, index=index, message=f"point {index}: {why}",
                        detail=hint)

        path = E.cache_path_for(self.cache_dir, index)
        _, meta = load_plan(path)
        legs = [dict(lg, with_box=bool(lg.get("with_box"))) for lg in meta["legs"]]
        bx, by = float(meta.get("bx", summary["bx"])), float(meta.get("by", summary["by"]))
        kinds = leg_scene_kinds(legs)

        # Both static boxes get their pose now, outside the recording: they do not
        # move during a point, so they belong in the live scene, and only their
        # visibility needs to be part of the animation.
        self.meshcat.SetTransform(PICK_ROOT, E.box_pose_for(bx, by))
        carried = [lg for lg in legs if lg["with_box"]]
        if carried:
            self.meshcat.SetTransform(
                PLACED_ROOT, self._held_box_pose(np.asarray(carried[-1]["q"])[-1]))

        total = sum(float(lg.get("duration", 0.0)) for lg in legs)
        print(f"[{index:2d}] box=({bx:+.3f}, {by:+.3f})  {len(legs)} legs, {total:.1f}s"
              f"  gcp={_fmt(summary['gcp_index'])} seed={_fmt(summary['grasp_seed'])}")
        for lg, kind in zip(legs, kinds):
            print(f"       {lg['name']:<14} {lg['duration']:6.2f}s  "
                  f"{np.asarray(lg['q']).shape[0]:4d} samples  box: {kind}")

        t0 = time.perf_counter()
        if self.realtime:
            n_frames = self._play_realtime(legs, kinds)
            how = f"streamed {n_frames} frames in real time"
        else:
            n_frames = self._record(legs, kinds)
            how = f"published a {n_frames}-frame animation at {self.fps:g} fps"
        self.selected = index
        msg = (f"point {index}: {len(legs)} legs, {total:.1f}s -- {how} "
               f"({time.perf_counter() - t0:.1f}s)")
        print(f"[{index:2d}] {msg}")
        return dict(ok=True, index=index, message=msg, n_frames=n_frames,
                    duration_s=round(total, 2), legs=[lg["name"] for lg in legs])

    def _record(self, legs, kinds):
        """Build and publish one meshcat animation covering every leg, in order.

        Frame times are cumulative across legs, so the slider spans the whole plan
        and the legs run back-to-back the way the robot would execute them. The
        stored samples are thinned to about ``--fps`` because meshcat quantises a
        recorded transform to ``round(t * fps)``: publishing 30 Hz samples into a
        15 fps animation would land two samples on one frame and simply discard one
        of them.
        """
        m = self.meshcat
        m.DeleteRecording()          # a click replaces the previous point's animation
        m.StartRecording(frames_per_second=self.fps,
                         set_visualizations_while_recording=False)
        t0, n_frames = 0.0, 0
        for leg, kind in zip(legs, kinds):
            q = np.asarray(leg["q"], dtype=float)
            n = q.shape[0]
            duration = float(leg.get("duration", 0.0))
            dt = duration / max(n - 1, 1)
            stride = max(1, int(round((1.0 / self.fps) / dt))) if dt > 0 else 1
            idxs = list(range(0, n, stride))
            if idxs[-1] != n - 1:
                idxs.append(n - 1)   # never drop the configuration the leg ends at
            for i in idxs:
                t = t0 + i * dt
                self._set_box_state(kind, q[i], time_in_recording=t)
                self._pose_robot(leg["with_box"], q[i], t=t)
                n_frames += 1
            t0 += duration
        m.StopRecording()
        # Leave the live scene on the plan's first configuration rather than
        # whatever the last publish happened to be, so a viewer that has not hit
        # play sees the start of the motion.
        q_first = np.asarray(legs[0]["q"], dtype=float)[0]
        self._set_box_state(kinds[0], q_first)
        self._pose_robot(legs[0]["with_box"], q_first)
        m.PublishRecording()
        return n_frames

    def _play_realtime(self, legs, kinds):
        """Real-time streaming playback, one leg at a time, via preview_legs.

        Per leg rather than in one call so the box state can be switched between
        legs; preview_legs itself already handles the pacing and the empty-handed /
        box-attached diagram choice from each leg's ``with_box`` tag.
        """
        n = 0
        for leg, kind in zip(legs, kinds):
            # Real-time playback streams a whole leg through preview_legs, so the
            # carried box can only be placed once per leg; it sits at the leg's
            # first configuration rather than tracking the gripper. Recorded
            # playback (the one the video renders from) does track it.
            self._set_box_state(kind, np.asarray(leg["q"], dtype=float)[0])
            preview_legs(self.viz, [leg])
            n += np.asarray(leg["q"]).shape[0]
        return n


# ── Web front end ───────────────────────────────────────────────────────────

PAGE_CSS = """
:root { color-scheme: dark; }
* { box-sizing: border-box; }
body { margin: 0; background: #14161a; color: #e8e8e8;
       font: 13px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; }
header { padding: 10px 14px; background: #1c2027; border-bottom: 1px solid #2c313a;
         display: flex; gap: 16px; align-items: baseline; flex-wrap: wrap; }
header h1 { font-size: 15px; margin: 0; font-weight: 600; }
header .meta { color: #9aa4b2; font-size: 12px; }
header code { color: #cfd6e0; }
button { font: inherit; background: #2a3140; color: #e8e8e8; border: 1px solid #3a4354;
         border-radius: 5px; padding: 4px 10px; cursor: pointer; }
button:hover { background: #333c4d; }
main { display: flex; gap: 14px; padding: 14px; align-items: flex-start;
       flex-wrap: wrap; }
.col { flex: 1 1 460px; min-width: 420px; }
#msg { margin: 0 0 10px; padding: 7px 10px; border-radius: 5px; background: #1c2027;
       border: 1px solid #2c313a; min-height: 19px; }
#msg.err { border-color: #7a2f2f; background: #2a1c1c; }
#msg.busy { border-color: #6b5a1f; background: #2a2618; }
.grid { display: grid; gap: 6px; }
.cell { position: relative; padding: 6px 4px; border-radius: 6px; text-align: center;
        border: 1px solid #00000055; cursor: pointer; color: #fff; line-height: 1.25; }
.cell.unknown { color: #23262c; }
.cell b { display: block; font-size: 14px; }
.cell span { display: block; font-size: 10.5px; opacity: 0.92; font-variant-numeric: tabular-nums; }
.cell.corner { outline: 2px solid __CORNER__; outline-offset: 1px; }
.cell.sel { box-shadow: 0 0 0 3px #ffb000; }
.cell:disabled { cursor: not-allowed; }
.axis { color: #9aa4b2; font-size: 11px; text-align: center; }
.axis.y { writing-mode: vertical-rl; transform: rotate(180deg); }
#hover { margin-top: 8px; padding: 7px 9px; background: #1c2027; border-radius: 5px;
         border: 1px solid #2c313a; min-height: 46px; white-space: pre-wrap;
         font-family: ui-monospace, monospace; font-size: 11.5px; color: #cfd6e0; }
table { border-collapse: collapse; width: 100%; margin-top: 12px;
        font-variant-numeric: tabular-nums; }
th, td { padding: 2px 6px; text-align: right; border-bottom: 1px solid #262b33; }
th { color: #9aa4b2; font-weight: 600; text-align: right; }
th:first-child, td:first-child, td.l { text-align: left; }
tr.clickable { cursor: pointer; }
tr.clickable:hover td { background: #1f242c; }
td.ok { color: #4cc76a; }
td.bad { color: #ff7b7b; }
td.unk { color: #9aa4b2; }
#progwrap {{ display: none; position: relative; height: 18px; margin-bottom: 6px;
  background: #14171c; border: 1px solid #2c313a; border-radius: 4px; overflow: hidden; }}
#progwrap.on {{ display: block; }}
#progbar {{ height: 100%; width: 0%; background: #2a78d6; transition: width .15s linear; }}
#progtxt {{ position: absolute; left: 8px; top: 0; line-height: 18px; font-size: 11px;
  color: #dfe4ea; text-shadow: 0 0 3px #000; white-space: nowrap; }}
iframe { width: 100%; height: 620px; border: 1px solid #2c313a; border-radius: 6px;
         background: #000; }
img.diagram { width: 100%; max-width: 520px; border-radius: 6px; background: #fff; }
details { margin-top: 12px; } summary { cursor: pointer; color: #9aa4b2; }
"""

PAGE_JS = """
const POINTS = __POINTS__;
let busy = false;

function setMsg(text, cls) {
  const el = document.getElementById('msg');
  el.textContent = text;
  el.className = cls || '';
}

function describe(i) {
  const p = POINTS[i];
  return p.hover;
}

const STATIC_HTML = __STATIC_HTML__;

function show(i) {
  if (busy) return;
  document.querySelectorAll('.cell').forEach(c => c.classList.remove('sel'));
  const cell = document.querySelector('.cell[data-i="' + i + '"]');
  if (cell) cell.classList.add('sel');
  if (STATIC_HTML) {
    // Self-contained page per point: fetch it with progress, then hand the bytes to
    // the iframe as a blob URL. Setting fr.src directly would be simpler but gives no
    // feedback at all, and these renders are tens of MB -- long enough that a blank
    // frame reads as "broken" rather than "loading".
    loadStatic(i);
    return;
  }
  busy = true;
  setMsg('loading point ' + i + ' into meshcat...', 'busy');
  fetch('show?index=' + i)
    .then(r => r.json())
    .then(d => {
      setMsg(d.message + (d.detail ? '\\n' + d.detail : ''), d.ok ? '' : 'err');
      busy = false;
    })
    .catch(e => { setMsg('request failed: ' + e, 'err'); busy = false; });
}

async function loadStatic(i) {
  const wrap = document.getElementById('progwrap');
  const bar = document.getElementById('progbar');
  const txt = document.getElementById('progtxt');
  const fr = document.getElementById('viewer');
  const mb = b => (b / 1e6).toFixed(1) + ' MB';
  busy = true;
  wrap.classList.add('on');
  bar.style.width = '0%';
  txt.textContent = 'requesting point ' + i + '...';
  setMsg('loading point ' + i + ' (self-contained render, no meshcat needed)...', 'busy');
  try {
    const resp = await fetch('html?index=' + i);
    if (!resp.ok) throw new Error('HTTP ' + resp.status);
    // Uncompressed size, because the reader below sees decompressed bytes while
    // Content-Length is the gzipped size.
    const total = parseInt(resp.headers.get('X-Uncompressed-Bytes') || '0', 10);
    const reader = resp.body.getReader();
    const chunks = [];
    let got = 0;
    for (;;) {
      const {done, value} = await reader.read();
      if (done) break;
      chunks.push(value);
      got += value.length;
      if (total > 0) {
        const pct = Math.min(100, 100 * got / total);
        bar.style.width = pct.toFixed(1) + '%';
        txt.textContent = 'point ' + i + ': ' + mb(got) + ' / ' + mb(total)
                          + '  (' + pct.toFixed(0) + '%)';
      } else {
        txt.textContent = 'point ' + i + ': ' + mb(got) + ' received';
      }
    }
    bar.style.width = '100%';
    txt.textContent = 'point ' + i + ': ' + mb(got) + ' received -- rendering '
                      + '(uploading geometry to the GPU, a few seconds)...';
    const url = URL.createObjectURL(new Blob(chunks, {type: 'text/html'}));
    fr.onload = () => {
      // Revoked on the next tick, not immediately: the iframe still needs the URL
      // while it parses.
      setTimeout(() => URL.revokeObjectURL(url), 60000);
      wrap.classList.remove('on');
      setMsg('point ' + i + ' loaded (' + mb(got) + '). Use the player controls '
             + '(top right of the frame) to play, pause and scrub.', '');
      busy = false;
    };
    fr.src = url;
  } catch (e) {
    wrap.classList.remove('on');
    setMsg('failed to load point ' + i + ': ' + e, 'err');
    busy = false;
  }
}

function reloadCache() {
  setMsg('re-reading the cache...', 'busy');
  fetch('reload').then(r => r.json()).then(() => location.reload())
    .catch(e => setMsg('reload failed: ' + e, 'err'));
}

window.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('[data-i]').forEach(el => {
    const i = parseInt(el.dataset.i, 10);
    el.addEventListener('click', () => show(i));
    el.addEventListener('mouseenter', () => {
      document.getElementById('hover').textContent = describe(i);
    });
  });
  document.getElementById('reload').addEventListener('click', reloadCache);
  const sel = __SELECTED__;
  if (sel !== null) {
    const cell = document.querySelector('.cell[data-i="' + sel + '"]');
    if (cell) cell.classList.add('sel');
  }
});
"""


def _hover_text(s):
    """The multi-line blurb shown when a point is hovered (and its tooltip)."""
    head = (f"[{s['index']:2d}]  box = ({s['bx']:+.3f}, {s['by']:+.3f}) m   "
            f"grid (ix={s['ix']}, iy={s['iy']})")
    if s["status"] == "failed" or (not s["has_plan"] and s["status"] != "success"):
        why = (f"failed at stage {s['stage']!r}" if s["stage"]
               else (s["note"] or "no cached plan"))
        lines = [head, why]
        if s["detail"]:
            lines.append(s["detail"][:300])
        if not s["has_plan"]:
            lines.append("-> nothing to replay")
        return "\n".join(lines)
    legs = ", ".join(f"{lg['name']} {lg['duration']:.1f}s" for lg in s["legs"])
    return "\n".join([
        head,
        f"{s['status']}  ({s['source']})   worst clearance "
        f"{_fmt(s['worst_clearance_mm'], '.3f')} mm   CoM margin "
        f"{_fmt(s['worst_com_margin_mm'], '.2f')} mm",
        f"joint-limit excess {_fmt(s['worst_joint_limit_excess_rad'], '.2e')} rad   "
        f"total {_fmt(s['total_duration_s'], '.1f')} s   "
        f"gcp {_fmt(s['gcp_index'])}   grasp seed {_fmt(s['grasp_seed'])}",
        f"legs: {legs}" if legs else "no legs in the cached plan",
    ])


def _cell_html(s, corner_idxs, clickable):
    """One clickable grid cell: index, and the two numbers worth seeing at a glance."""
    kind = {"success": "success", "failed": "failed"}.get(s["status"], "unknown")
    color = {"success": COLOR_SUCCESS, "failed": COLOR_FAILED}.get(
        s["status"], COLOR_UNKNOWN)
    classes = ["cell", kind] + (["corner"] if s["index"] in corner_idxs else [])
    if s["status"] == "failed" or not s["has_plan"]:
        line1 = html.escape(str(s["stage"] or s["note"] or "no plan"))[:18]
        line2 = "no replay"
    else:
        line1 = f"{_fmt(s['worst_clearance_mm'], '.2f')} mm"
        line2 = f"{_fmt(s['total_duration_s'], '.0f')} s / gcp {_fmt(s['gcp_index'])}"
    return (f'<button class="{" ".join(classes)}" data-i="{s["index"]}" '
            f'style="background:{color}" title="{html.escape(_hover_text(s))}"'
            f'{"" if clickable else " disabled"}>'
            f'<b>{s["index"]:02d}</b><span>{line1}</span><span>{line2}</span></button>')


def meshcat_url_for_request(app, host_header):
    """The meshcat URL as *this* browser should reach it.

    Drake's ``web_url()`` reports ``localhost`` whenever MeshcatParams.host is "*",
    which is the default. That is right for a local session and wrong for every remote
    one: the iframe and link would send the viewer's browser to its own machine rather
    than to this server. Meshcat itself already listens on all interfaces (verified:
    ``*:7000``), so only the advertised hostname needs fixing.

    Taking the hostname from the request's own Host header means whatever name the
    viewer typed -- a fully-qualified host, a bare hostname, an IP, or localhost --
    keeps working with no flag to set and nothing to keep in sync. Only meshcat's port
    is substituted in.
    """
    url = app.meshcat_url
    if not host_header:
        return url
    browser_host = host_header.rsplit(":", 1)[0].strip("[]")
    if not browser_host or browser_host in ("localhost", "127.0.0.1", "::1"):
        return url
    parsed = urlsplit(url)
    port = f":{parsed.port}" if parsed.port else ""
    return urlunsplit((parsed.scheme, f"{browser_host}{port}",
                       parsed.path, parsed.query, parsed.fragment))


def render_page(app, host_header=None):
    """The whole page, rebuilt per request so it always shows the current summary."""
    summaries = app.summaries
    by_index = {s["index"]: s for s in summaries}
    n_ok = sum(1 for s in summaries if s["has_plan"])
    n_failed = sum(1 for s in summaries if s["status"] == "failed")
    n_unknown = len(summaries) - n_ok - n_failed

    # Grid orientation matches the PNG next to it: box x increases to the right
    # (columns are ix), box y increases upwards (rows are iy, printed top-down).
    rows = []
    header_cells = "".join(
        f'<div class="axis">x={app.grid_points[ix * app.grid_ny][0]:+.3f}</div>'
        for ix in range(app.grid_nx))
    rows.append(f'<div class="axis"></div>{header_cells}')
    for iy in reversed(range(app.grid_ny)):
        cells = "".join(
            _cell_html(by_index[ix * app.grid_ny + iy], app.corner_idxs,
                       by_index[ix * app.grid_ny + iy]["has_plan"])
            for ix in range(app.grid_nx))
        rows.append(f'<div class="axis y">y={app.grid_points[iy][1]:+.3f}</div>{cells}')
    grid_html = (f'<div class="grid" style="grid-template-columns: auto repeat('
                 f'{app.grid_nx}, 1fr)">' + "".join(rows) + "</div>")

    trs = []
    for s in summaries:
        cls = {"success": "ok", "failed": "bad"}.get(s["status"], "unk")
        note = s["stage"] or s["note"] or ""
        trs.append(
            f'<tr class="clickable" data-i="{s["index"]}">'
            f'<td class="l">{s["index"]:02d}</td>'
            f'<td class="{cls} l">{html.escape(str(s["status"] or "unknown"))}</td>'
            f'<td>{_fmt(s["worst_clearance_mm"], ".3f")}</td>'
            f'<td>{_fmt(s["worst_com_margin_mm"], ".2f")}</td>'
            f'<td>{_fmt(s["worst_joint_limit_excess_rad"], ".1e")}</td>'
            f'<td>{_fmt(s["total_duration_s"], ".1f")}</td>'
            f'<td>{_fmt(s["gcp_index"])}</td>'
            f'<td>{_fmt(s["grasp_seed"])}</td>'
            f'<td class="l">{html.escape(note)[:40]}</td></tr>')

    # The hover blurbs quote planner exception text verbatim, and this JSON is embedded
    # in a <script> block, so neutralise the one sequence that could close it early.
    points_json = json.dumps(
        {s["index"]: dict(hover=_hover_text(s)) for s in summaries}).replace("</", "<\\/")
    js = (PAGE_JS.replace("__POINTS__", points_json)
                 .replace("__STATIC_HTML__", "true" if app.html_dir else "false")
                 .replace("__SELECTED__", "null" if app.selected is None
                          else str(app.selected)))
    mode = ("real-time streaming" if app.realtime
            else f"recorded animation @ {app.fps:g} fps")
    meshcat_url = meshcat_url_for_request(app, host_header)
    static_html = bool(app.html_dir)
    # In static mode nothing points at the meshcat port: the iframe starts blank
    # and each click loads that point's self-contained file from this same server.
    viewer_src = 'about:blank' if static_html else meshcat_url
    # In static mode the meshcat URL is deliberately never shown. It is a second port
    # the viewer often cannot reach, and offering it as a link sends them straight back
    # to the "No connection to server" banner this mode exists to avoid.
    if static_html:
        source_note = ("self-contained renders &mdash; no meshcat connection needed")
        viewer_note = ("Each point is a standalone HTML render served over this same "
                       "port. Use the player controls (top right of the frame) to play, "
                       "pause and scrub. Open one directly at "
                       "<code>/html?index=N</code>.")
    else:
        source_note = (f'live meshcat <code><a href="{meshcat_url}" '
                       f'target="_blank">{meshcat_url}</a></code>')
        viewer_note = ("Use meshcat's own animation controls (open the controls panel, "
                       "top right) to play, pause and scrub the plan. If the frame above "
                       f'stays blank, open <a href="{meshcat_url}" target="_blank">'
                       f"{meshcat_url}</a> in its own tab.")
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>RBY1 grid plan visualiser</title>
<style>{PAGE_CSS.replace("__CORNER__", COLOR_CORNER)}</style>
<script>{js}</script></head>
<body>
<header>
  <h1>RBY1 grid plan visualiser</h1>
  <span class="meta">cache <code>{html.escape(app.cache_dir)}</code></span>
  <span class="meta">{n_ok} replayable &middot; {n_failed} failed &middot;
    {n_unknown} unknown</span>
  <span class="meta">{source_note}</span>
  <span class="meta">{mode}</span>
  <button id="reload">reload cache</button>
</header>
<main>
  <div class="col">
    <div id="msg">Click a grid point to replay its cached plan in meshcat.</div>
    {grid_html}
    <div id="hover">Hover a point for its numbers; click it to replay.</div>
    <table><thead><tr>
      <th>#</th><th>status</th><th>clr&nbsp;mm</th><th>CoM&nbsp;mm</th>
      <th>jl&nbsp;rad</th><th>dur&nbsp;s</th><th>gcp</th><th>seed</th><th>note</th>
    </tr></thead><tbody>{''.join(trs)}</tbody></table>
    <details><summary>status diagram (plot_grid_diagram)</summary>
      <img class="diagram" src="grid.png?v={app.png_stamp}" alt="grid status diagram">
    </details>
  </div>
  <div class="col">
    <div id="progwrap"><div id="progbar"></div><span id="progtxt"></span></div>
    <iframe id="viewer" src="{viewer_src}" title="render"></iframe>
    <div class="meta" style="margin-top:6px">{viewer_note}</div>
  </div>
</main>
</body></html>"""


class _Handler(BaseHTTPRequestHandler):
    """Four routes: the page, the diagram PNG, show?index=N, and reload.

    ``show`` is the click path and calls straight into ``GridPlanViewer.show_point``;
    the viewer's own lock serialises the Drake work, so a double-click cannot have
    two replays writing the same plant context at once.
    """

    app = None
    protocol_version = "HTTP/1.1"
    server_version = "GridVisualizer/1.0"

    def log_message(self, fmt, *args):
        pass    # the viewer prints its own, more useful, one-liner per replay

    def _send(self, code, ctype, body, extra_headers=None):
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra_headers or {}).items():
            self.send_header(k, str(v))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        route = parsed.path.rstrip("/") or "/"
        query = urllib.parse.parse_qs(parsed.query)
        app = _Handler.app
        try:
            if route == "/":
                self._send(200, "text/html; charset=utf-8",
                           render_page(app, self.headers.get("Host")))
            elif route == "/grid.png":
                with open(app.png_path, "rb") as f:
                    self._send(200, "image/png", f.read())
            elif route == "/show":
                try:
                    index = int(query.get("index", ["?"])[0])
                except ValueError:
                    self._send(400, "application/json",
                               json.dumps(dict(ok=False, message="index must be an int")))
                    return
                # submit(), not show_point(): meshcat may only be driven from the
                # thread that created it, and this is an HTTP worker thread.
                result = app.submit(lambda: app.show_point(index))
                self._send(200, "application/json", json.dumps(result))
            elif route == "/html":
                # Serves a pre-exported self-contained render. Kept deliberately dumb:
                # the file already embeds meshcat, the geometry and the animation, so
                # there is nothing to do but hand it over.
                q = urllib.parse.parse_qs(parsed.query)
                try:
                    idx = int(q.get("index", ["?"])[0])
                except ValueError:
                    self._send(400, "text/plain; charset=utf-8", b"bad index")
                    return
                path = app.static_html_path(idx)
                if path is None:
                    self._send(404, "text/html; charset=utf-8",
                               f"<p>No exported render for point {idx}. Generate with "
                               f"<code>--export-html</code>.</p>".encode())
                    return
                # Prefer the pre-gzipped sibling: these renders are ~85 MB of JSON
                # and base64 mesh data, which gzips to ~21 MB. Compressing once at
                # export time rather than per request keeps serving to a file read,
                # and the browser decompresses transparently.
                #
                # X-Uncompressed-Bytes is what makes an honest progress bar possible:
                # with Content-Encoding: gzip, Content-Length is the *compressed* size
                # while a streaming reader observes *decompressed* bytes, so using
                # Content-Length as the denominator would run the bar to ~400%.
                raw_size = os.path.getsize(path)
                gz = path + ".gz"
                if os.path.exists(gz):
                    with open(gz, "rb") as f:
                        body = f.read()
                    self._send(200, "text/html; charset=utf-8", body,
                               {"Content-Encoding": "gzip",
                                "X-Uncompressed-Bytes": raw_size})
                else:
                    with open(path, "rb") as f:
                        self._send(200, "text/html; charset=utf-8", f.read(),
                                   {"X-Uncompressed-Bytes": raw_size})
            elif route == "/reload":
                app.submit(lambda: dict(ok=True, message="cache re-read",
                                        n=len(app.reload())))
                self._send(200, "application/json",
                           json.dumps(dict(ok=True, message="cache re-read")))
            else:
                self._send(404, "text/plain", "not found\n")
        except BrokenPipeError:
            pass    # browser navigated away mid-response
        except Exception as e:
            traceback.print_exc()
            self._send(500, "application/json",
                       json.dumps(dict(ok=False, message=f"{type(e).__name__}: {e}")))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", default=E.DEFAULT_CACHE_DIR,
                    help="Grid cache to visualise. Defaults to the live one; the "
                         "archived runs (plans/grid_cache_run*/) work too.")
    ap.add_argument("--grid-path", default=E.DEFAULT_GRID_PATH,
                    help="Grid definition (grid_points/grid_nx/grid_ny).")
    ap.add_argument("--port", type=int, default=8899,
                    help="Port for the grid page (meshcat gets its own, usually 7000).")
    ap.add_argument("--host", default="127.0.0.1",
                    help="Interface to serve the grid page on.")
    ap.add_argument("--fps", type=float, default=DEFAULT_FPS,
                    help="Animation frame rate; the stored 30 Hz samples are thinned "
                         "to match. Raise for a finer scrub, lower for a lighter page.")
    ap.add_argument("--realtime", action="store_true",
                    help="Stream each plan to meshcat in real time (preview_legs) "
                         "instead of publishing a scrubbable animation. Note that a "
                         "click then blocks for the plan's full 60-100 s.")
    ap.add_argument("--show", type=int, nargs="*", default=None, metavar="IDX",
                    help="Replay these points immediately at startup.")
    ap.add_argument("--no-serve", action="store_true",
                    help="Do --show and exit without serving the page (scripted use).")
    ap.add_argument("--html-dir", metavar="DIR", default=None,
                    help="Serve pre-exported self-contained renders from DIR instead of "
                         "driving the live meshcat. Clicking a point loads that point's "
                         "HTML into the iframe over THIS port, so the meshcat port is "
                         "never contacted -- use this when the meshcat websocket is "
                         "unreachable. Build the files first with --export-html.")
    ap.add_argument("--export-html", metavar="DIR", default=None,
                    help="Write a self-contained point_NN.html per successfully planned "
                         "point into DIR and exit. Each file embeds meshcat, the robot "
                         "geometry and the point's whole animation, so it opens in any "
                         "browser with no server, no Drake and no cache -- which is what "
                         "makes it the thing to copy off this machine. Expect tens of MB "
                         "each (the geometry dominates); lower --fps to shrink them. "
                         "Combine with --show to export only those points.")
    args = ap.parse_args()

    app = GridPlanViewer(args.cache_dir, args.grid_path,
                         fps=args.fps, realtime=args.realtime,
                         html_dir=args.html_dir)
    n_ok = sum(1 for s in app.summaries if s["has_plan"])
    src = ("status.json present" if app.status else
           "no status.json -- statuses derived from the cache")
    print(f"Cache {args.cache_dir}: {n_ok}/{app.n_points} points have a replayable "
          f"plan ({src}).")
    print(f"Meshcat: {app.meshcat_url}")

    for index in (args.show or []):
        app.show_point(index)

    if args.export_html:
        os.makedirs(args.export_html, exist_ok=True)
        # --show doubles as a filter here: re-exporting a single point after a
        # change to the replay is a minute's work, where the whole grid is a
        # quarter of an hour.
        only = set(args.show) if args.show else None
        wrote, skipped = [], []
        for s in app.summaries:
            idx = s["index"]
            if only is not None and idx not in only:
                continue
            if not s["has_plan"]:
                skipped.append(idx)
                continue
            # show_point publishes the animation into the live meshcat; StaticHtml then
            # serialises whatever meshcat currently holds -- scene plus animation -- so
            # the order matters and the two cannot be swapped.
            res = app.show_point(idx)
            if not res.get("ok"):
                skipped.append(idx)
                continue
            out = os.path.join(args.export_html, f"point_{idx:02d}.html")
            with open(out, "w") as f:
                f.write(app.meshcat.StaticHtml())
            wrote.append(idx)
            print(f"      wrote {out}  ({os.path.getsize(out) / 1e6:.1f} MB)")
        print(f"\nExported {len(wrote)} HTML file(s) to {args.export_html}: {wrote}")
        if skipped:
            print(f"Skipped (no cached plan / replay failed): {skipped}")
        return 0

    if args.no_serve:
        return 0

    _Handler.app = app
    server = ThreadingHTTPServer((args.host, args.port), _Handler)
    server.daemon_threads = True
    print(f"Grid page: http://{args.host}:{args.port}/   (Ctrl-C to stop)")
    print("Click a grid point to replay its cached plan; the meshcat view is "
          "embedded in the same page.")
    # The HTTP server runs on a background thread and the *main* thread runs the
    # replays it asks for -- not the other way round, because meshcat belongs to the
    # main thread (see GridPlanViewer's docstring).
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        app.run_main_loop()
    except KeyboardInterrupt:
        print("\nstopping.")
    finally:
        server.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
