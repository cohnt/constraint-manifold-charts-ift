#!/usr/bin/env python
"""Assemble one paper-ready experiment record per grid point from the artifacts a
run already leaves behind.

Why this exists
---------------
A run scatters its record across four formats, three of which need Drake (or the
IKFast extensions) to open, and none of which answers "what did the robot do, on
one clock, and how long did producing it take":

  * ``plans/grid_cache/point_NN.pkl`` -- the plan: 20 Hz waypoint commands, plus
    ``meta["legs"][i]["q"]``, a ``(N, 23)`` array of joint positions per leg.
    Each leg's samples are *implicitly* uniform over ``[0, duration]``; there is
    no time column, each leg restarts at 0, and the gripper steps between legs
    occupy no planned time at all. So a global "at t seconds the robot was here"
    does not exist in the file.
  * ``plans/grid_cache/timing/run_*.jsonl`` -- the event log, the only honest
    source for how long a plan took to produce (see the folder README).
  * ``plans/grid_cache/status.json`` -- the per-point verdict manifest.
  * ``point_NN_exec_<stamp>.pkl`` -- what the robot measured, when a plan was
    actually executed. Its per-step durations exist only implicitly, inside the
    server's ``command_timestamps``.

This script joins them and writes:

  <out>/run.json                     provenance, config, model constants, aggregates
  <out>/phases.csv                   THE KEYFRAME TABLE: one row per (point, phase)
  <out>/summary.csv                  one row per point
  <out>/points/point_NN.json         the whole per-point record, plan meta included
  <out>/points/point_NN.npz          dense planned trajectory on a global clock
  <out>/points/point_NN_exec_*.npz   dense measured traces, per execution

Phases (what "keyframes" means here) are the plan's own sequence -- reach_approach,
reach_descend, grasp, lift, place, release, home_retreat, home -- each carrying its
start/end time on the global clock, its duration, the configuration at both ends,
and its guarantee evidence and solver provenance.

Two properties are deliberate:

  * **Nothing is recomputed that the run already decided.** Clearances, CoM
    margins and joint-limit excesses are copied from ``meta["evidence"]`` (the
    dense check the plan had to pass to be cached). Re-deriving them here would
    invite a second, subtly different number for the same claim; use
    ``scripts/verify_cached_plan.py`` when the question is whether they still hold.
  * **Kinematics that the run never recorded are computed here** -- end-effector
    poses, the carried box pose, the CoM trace -- because there is no stored
    answer to disagree with. This needs Drake; ``--no-kinematics`` skips it and
    everything else still exports.

Usage
-----
    scripts/export_experiment_record.py                        # plans/grid_cache
    scripts/export_experiment_record.py --cache-dir DIR --out DIR
    scripts/export_experiment_record.py --indices 5 9 --no-kinematics
"""

import argparse
import csv
import datetime
import glob
import json
import os
import pickle
import re
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


import plan_format  # noqa: F401  -- installs the pre-rename module aliases the
                    # committed plan and execution pickles were written with
import timing_report                      # noqa: E402  (event-log accounting)
from analyze_execution_record import (    # noqa: E402
    BODY, CHAINS, _as_logs, analyse_leg, commanded_from_plan, gripper_spread,
)

# The paper-facing grouping of the pipeline's leg names. The leg names themselves
# stay the identity -- every other artifact in the repo keys on them -- but a
# figure caption wants "carry", not "lift + place".
PHASE_GROUPS = {
    "reach_approach": "reach",
    "reach_descend": "reach",
    "grasp": "grasp",
    "lift": "carry",
    "place": "carry",
    "release": "release",
    "home_retreat": "return",
    "home": "return",
}

# Which planner produced each leg, as a property of the leg rather than of a
# particular run: the two constrained legs are the ones planned in the reduced
# 14-D space with the gripper-to-gripper transform frozen.
CONSTRAINED_LEGS = ("lift", "place")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _jsonable(x):
    """Recursively convert numpy/Drake-ish values into JSON-serialisable ones.

    Dense arrays are the one thing NOT flattened into JSON: they go to the npz
    beside it. Anything else in ``meta`` -- and meta is copied verbatim, because
    the whole point is that the record carries the plan's own metadata -- is small
    enough that a list is fine.
    """
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    if isinstance(x, (np.floating, float)):
        v = float(x)
        return v if np.isfinite(v) else None      # JSON has no NaN/Inf
    if isinstance(x, (np.integer, int)):
        return int(x)
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if x is None or isinstance(x, str):
        return x
    return str(x)


def _round(x, n=6):
    return None if x is None else round(float(x), n)


def point_indices(cache_dir, prefix="point"):
    out = []
    for p in sorted(glob.glob(os.path.join(cache_dir, f"{prefix}_[0-9]*.pkl"))):
        m = re.fullmatch(rf"{prefix}_(\d+)", os.path.splitext(os.path.basename(p))[0])
        if m:
            out.append(int(m.group(1)))
    return out


# ── The global clock ──────────────────────────────────────────────────────────

def build_timeline(steps, legs):
    """Phases and dense samples on one clock spanning the whole plan.

    The clock is *planned* time: each trajectory step contributes its own
    ``duration``, and a gripper step contributes zero. A gripper close is not
    instantaneous on the robot, but the plan assigns it no time -- inventing a
    nominal here would put a number in the record that no artifact supports.
    The measured duration of every step, gripper steps included, comes from the
    execution record instead (``exec_summary``).

    Returns ``(phases, t, q, phase_of_sample)``:
      phases  ordered dicts, one per step (trajectory or gripper)
      t       (N,) global planned seconds, non-decreasing across legs
      q       (N, 23) joint positions
      phase_of_sample  (N,) index into ``phases``
    """
    by_name = {}
    for lg in legs:
        by_name.setdefault(lg["name"], lg)

    phases, t_parts, q_parts, owner = [], [], [], []
    clock = 0.0
    gripper_closed = False
    for i, step in enumerate(steps):
        name = step["name"]
        if step["type"] == "gripper":
            gripper_closed = step["action"] == "close"
            phases.append(dict(
                phase_index=len(phases), name=name, kind="gripper",
                group=PHASE_GROUPS.get(name, name),
                t_start_s=clock, t_end_s=clock, planned_duration_s=0.0,
                action=step["action"], grippers=list(step["grippers"]),
                gripper_closed_after=gripper_closed,
                n_samples=0, n_waypoints=0, with_box=None, planner=None,
            ))
            continue

        lg = by_name.get(name)
        if lg is None:
            raise ValueError(
                f"trajectory step '{name}' has no matching entry in meta['legs'] "
                f"(legs: {[l['name'] for l in legs]}). The plan and its leg "
                f"samples disagree; re-save it with plan_grid.")
        q23 = np.asarray(lg["q"], dtype=float)
        dur = float(lg["duration"])
        # The samples are uniform over the leg by construction (sample_leg ->
        # _sample_times, endpoint included), so this reconstructs their times
        # exactly rather than approximating them.
        t_local = np.linspace(0.0, dur, q23.shape[0])
        phases.append(dict(
            phase_index=len(phases), name=name, kind="trajectory",
            group=PHASE_GROUPS.get(name, name),
            t_start_s=clock, t_end_s=clock + dur, planned_duration_s=dur,
            action=None, grippers=None, gripper_closed_after=gripper_closed,
            n_samples=int(q23.shape[0]), n_waypoints=len(step["cmds"]),
            with_box=bool(lg.get("with_box")),
            planner="constrained" if name in CONSTRAINED_LEGS else "unconstrained",
        ))
        t_parts.append(clock + t_local)
        q_parts.append(q23)
        owner.append(np.full(q23.shape[0], len(phases) - 1, dtype=int))
        clock += dur

    t = np.concatenate(t_parts)
    q = np.vstack(q_parts)
    return phases, t, q, np.concatenate(owner)


def commanded_timeline(steps, phases):
    """The 20 Hz commands actually streamed, on the same global clock.

    This is the record of what the robot was *told*, as distinct from the 30 Hz
    ``meta["legs"]`` samples, which are a rendering of the same trajectory at a
    different rate. Both belong in the export: the leg samples are what every
    verification script in the repo re-checks, the commands are what executed.
    """
    t_parts, q_parts, owner = [], [], []
    dt = None
    starts = {p["name"]: p["t_start_s"] for p in phases if p["kind"] == "trajectory"}
    for i, step in enumerate(steps):
        if step["type"] != "trajectory":
            continue
        cmds = step["cmds"]
        step_dt = float(cmds[0]["right"].duration)
        dt = step_dt if dt is None else dt
        q20 = np.array([np.concatenate([c["torso"].target_position,
                                        c["right"].target_position,
                                        c["left"].target_position])
                        for c in cmds])
        t_parts.append(starts[step["name"]] + step_dt * np.arange(len(cmds)))
        q_parts.append(q20)
        idx = next(p["phase_index"] for p in phases
                   if p["name"] == step["name"] and p["kind"] == "trajectory")
        owner.append(np.full(len(cmds), idx, dtype=int))
    return (np.concatenate(t_parts), np.vstack(q_parts),
            np.concatenate(owner), dt)


# ── Kinematics the run never recorded ─────────────────────────────────────────

class Kinematics:
    """FK, carried-box pose and CoM for 23-D configurations.

    One plant for the whole export. Obstacles are irrelevant here -- nothing is
    collision-queried -- so this is the cheap empty-handed model, and the carried
    box pose is computed from the grasping end-effector's frame the same way
    ``verify_cached_plan._placed_box_pose`` does.
    """

    def __init__(self):
        from plan_grid import HAND, OFFSET, RPY
        from pydrake.all import RigidTransform, RollPitchYaw
        from rby1_opt_ik import (
            _com_support_polygon_residuals, _stability_constraint_ub,
            support_polygon_edge_lengths,
        )
        from rby1_planning import (
            Rby1ActiveJointLayout, make_default_rby1_infrastructure,
        )

        self._RigidTransform = RigidTransform
        self._RollPitchYaw = RollPitchYaw
        self._residuals = _com_support_polygon_residuals
        self._ub = _stability_constraint_ub(0.0)      # matches _verify_leg's inset=0
        self._edge_len = support_polygon_edge_lengths

        plant, checker, diagram = make_default_rby1_infrastructure(open_grippers=True)
        # Every returned handle is kept, including the checker this class never
        # queries: they are views into one C++-owned structure, and letting the
        # checker be collected frees the plant out from under `self.plant` --
        # which surfaces as a "pre-finalize call to num_positions()" or a
        # straight segfault on the next FK, not as a Python error at the drop.
        self.plant, self.checker, self.diagram = plant, checker, diagram
        self.ctx = diagram.CreateDefaultContext()
        self.pctx = plant.GetMyContextFromRoot(self.ctx)
        self.layout = Rby1ActiveJointLayout(plant)
        self.idxs = self.layout.plant_idxs

        names = list(plant.GetPositionNames())
        self.joint_names = [names[i] for i in self.idxs]
        self.q_lb = plant.GetPositionLowerLimits()[self.idxs]
        self.q_ub = plant.GetPositionUpperLimits()[self.idxs]

        self.frames = {"right": plant.GetFrameByName("ee_right"),
                       "left": plant.GetFrameByName("ee_left")}
        # The box rides on the grasping arm's ee frame, at the same fixed offset
        # the HeldBox model uses, so its world pose is that frame's pose composed
        # with the offset.
        self.hand = HAND
        self.X_ee_box = RigidTransform(RollPitchYaw(RPY), OFFSET)
        self.com_instances = [plant.GetModelInstanceByName(n) for n in
                              ("base", "torso", "right_arm", "left_arm",
                               "right_gripper", "left_gripper", "head")]

    def at(self, q23):
        """Everything derivable from one configuration."""
        q23 = np.asarray(q23, dtype=float)
        q_full = self.plant.GetPositions(self.pctx)
        q_full[self.idxs] = q23
        self.plant.SetPositions(self.pctx, q_full)

        w = self.plant.world_frame()
        poses = {k: self.plant.CalcRelativeTransform(self.pctx, w, f)
                 for k, f in self.frames.items()}
        mid = 0.5 * (poses["right"].translation() + poses["left"].translation())
        box = poses[self.hand] @ self.X_ee_box

        com = self.plant.CalcCenterOfMassPositionInWorld(self.pctx, self.com_instances)
        resid = self._residuals(com[:2], q23[self.layout.base])
        slack = self._ub - resid
        # Two CoM numbers, because the one the pipeline records is not in metres.
        # _com_support_polygon_residuals uses UNNORMALISED edge normals, so its
        # residual is (distance x edge length); meta["evidence"]["com_margin_m"]
        # and status.json inherit that scaling. The sign -- and therefore every
        # feasibility decision ever made from it -- is unaffected, but the
        # magnitude is not a distance. `com_margin_legacy` reproduces the stored
        # number exactly; `com_margin_m` divides each edge's slack by that edge's
        # length and is the metres figure to quote.
        return dict(
            ee_right_xyz=poses["right"].translation(),
            ee_right_quat_wxyz=poses["right"].rotation().ToQuaternion().wxyz(),
            ee_right_rpy=self._RollPitchYaw(poses["right"].rotation()).vector(),
            ee_left_xyz=poses["left"].translation(),
            ee_left_quat_wxyz=poses["left"].rotation().ToQuaternion().wxyz(),
            ee_left_rpy=self._RollPitchYaw(poses["left"].rotation()).vector(),
            mid_gripper_xyz=mid,
            box_xyz=box.translation(),
            box_rpy=self._RollPitchYaw(box.rotation()).vector(),
            com_xyz=com,
            com_margin_legacy=float(np.min(slack)),
            com_margin_m=float(np.min(slack / self._edge_len)),
            com_binding_edge=int(np.argmin(slack / self._edge_len)),
            joint_limit_headroom_rad=float(np.min(np.minimum(q23 - self.q_lb,
                                                            self.q_ub - q23))),
        )

    def trace(self, q):
        """``at`` over an (N, 23) array, as arrays."""
        rows = [self.at(x) for x in q]
        return {k: np.array([r[k] for r in rows]) for k in rows[0]}


# ── Planning runtime, from the event log ──────────────────────────────────────

def load_runtime(cache_dir, logs=None):
    """``({index: per-point runtime}, [run headers])`` from the event log(s).

    Latency and work come from ``scripts/timing_report``, unchanged, so this
    record and that report can never disagree about a number.
    """
    paths = logs or sorted(glob.glob(os.path.join(cache_dir, "timing", "*.jsonl")))
    paths = [p for p in paths if os.path.exists(p)]
    if not paths:
        return {}, []
    recs, runs = timing_report.load_events(paths)
    children, roots = timing_report.build_tree(recs)
    out = {}
    for p in timing_report.analyse(recs, children, roots):
        # analyse() labels a point by its grid index when the enclosing
        # gcp_branch recorded one, else by its box coordinates.
        try:
            key = int(p["label"])
        except (TypeError, ValueError):
            continue
        lat = {c: 0.0 for c, _ in p["latency"]}
        for (c, _d), s in p["latency"].items():
            lat[c] += s
        work_kept = sum(s for (c, d), s in p["work"].items() if not d)
        work_disc = sum(s for (c, d), s in p["work"].items() if d)
        solver = sum(s for (c, _), s in p["latency"].items()
                     if c in timing_report.SOLVER_CATS)
        out[key] = dict(
            latency_s=_round(p["wall"], 3),
            latency_by_category_s={c: _round(v, 4) for c, v in sorted(
                lat.items(), key=lambda kv: -kv[1]) if v >= 5e-4},
            solver_latency_s=_round(solver, 3),
            solver_share_of_latency=_round(solver / max(p["wall"], 1e-9), 4),
            work_s=_round(work_kept + work_disc, 3),
            work_kept_s=_round(work_kept, 3),
            work_discarded_s=_round(work_disc, 3),
            concurrency_ratio=_round((work_kept + work_disc) / max(p["wall"], 1e-9), 3),
            ik_search_s=_round(p["ik_phase"].get("search", 0.0), 3),
            ik_rank_s=_round(p["ik_phase"].get("rank", 0.0), 3),
            killed_mid_plan=bool(p["killed"]),
            n_records_by_category={c: int(n) for c, n in sorted(p["counts"].items())},
        )
    return out, runs


# ── Execution, from the robot's own logs ──────────────────────────────────────

def exec_records_for(cache_dir, index, prefix="point", results_dirs=()):
    """Execution records for one point, from every directory they may live in.

    Two locations, both real: records written before results/ existed sit beside
    the plans in the cache dir, and current ones go to results/. Searching only
    one silently drops half the runs -- and silently, because a point that was
    never executed and a point whose records moved look identical from here.
    Sorted by timestamp, which the filename carries, so the last entry is the most
    recent execution regardless of which directory it came from.
    """
    seen, out = set(), []
    for d in (cache_dir, *results_dirs):
        if not d:
            continue
        for p in glob.glob(os.path.join(d, f"{prefix}_{index:02d}_exec_*.pkl")):
            real = os.path.realpath(p)
            if real not in seen:            # results/ may be a symlink to the cache
                seen.add(real)
                out.append(p)
    return sorted(out, key=lambda p: os.path.basename(p))


def _step_window(logs):
    """``(t_first_send, t_last_send, n_waypoints)`` over a step's server logs."""
    t0, t1, n = np.inf, -np.inf, 0
    for log in logs:
        ct = np.asarray(log.get("command_timestamps", []), dtype=float)
        if ct.size:
            t0, t1, n = min(t0, ct[0]), max(t1, ct[-1]), n + ct.size
    if not np.isfinite(t0):
        return None
    return float(t0), float(t1), int(n)


def _sample_span(logs):
    """``(first, last)`` state-sample timestamp over a step's server logs.

    The send window is the wrong measure for a step issued as a *single*
    long-duration command -- the premove -- where first send and last send are the
    same instant but the robot moves for seconds afterwards. The state samples are
    what bound that.
    """
    t0, t1 = np.inf, -np.inf
    for log in logs:
        at = np.asarray(log.get("timestamp", []), dtype=float)
        if at.size:
            t0, t1 = min(t0, at[0]), max(t1, at[-1])
    return (None if not np.isfinite(t0) else (float(t0), float(t1)))


def exec_summary(path, plan_path, phases):
    """Measured timing and tracking for one execution of one plan.

    Execution *time* is the thing no artifact currently states, so it is derived
    here three ways and all three are kept, because they answer different
    questions: per-step send window (what the plan's own legs cost), the span
    from the first send to the last state sample (what the operator waited for),
    and the planned total (what the plan said). ``premove`` -- the blocking move
    from wherever the robot was to the plan's start pose -- is reported
    separately: it is real time on the robot but belongs to no phase.
    """
    with open(path, "rb") as f:
        rec = pickle.load(f)

    cmd_by_name, dt = ({}, None)
    if plan_path and os.path.exists(plan_path):
        cmd_by_name, dt = commanded_from_plan(plan_path)

    planned_by_name = {p["name"]: p["planned_duration_s"] for p in phases}
    steps_out, traces = [], []
    t_first, t_last = np.inf, -np.inf

    for entry in rec.get("steps", []):
        logs = _as_logs(entry)
        win = _step_window(logs)
        row = dict(index=entry.get("index"), name=entry.get("name"),
                   type=entry.get("type"),
                   planned_duration_s=planned_by_name.get(entry.get("name")))
        # Records written after plan_io started timing its own calls state the
        # step's duration outright; older ones do not, and it is derived below.
        if entry.get("wall_s") is not None:
            row["call_wall_s"] = _round(entry["wall_s"], 4)
        if win is not None:
            s0, s1, n_wp = win
            t_first, t_last = min(t_first, s0), max(t_last, s1)
            row.update(t_start_epoch=s0, t_end_epoch=s1,
                       measured_duration_s=_round(s1 - s0, 4), n_waypoints=n_wp)
            if row["planned_duration_s"]:
                row["pacing_ratio"] = _round(
                    (s1 - s0) / row["planned_duration_s"], 4)
        for log in logs:
            at = np.asarray(log.get("timestamp", []), dtype=float)
            if at.size:
                t_last = max(t_last, float(at[-1]))

        if entry.get("type") == "gripper":
            # The one number the plan cannot state: gripper steps have no planned
            # duration, so their cost is only ever measured.
            row["gripper_spread_s"] = _round(gripper_spread(entry), 4)
        elif entry.get("name") in cmd_by_name and dt:
            det = analyse_leg(entry["name"], logs[0], cmd_by_name[entry["name"]], dt)
            if det is not None:
                row.update(
                    send_gap_median_ms=_round(det["gap_median_ms"], 2),
                    send_gap_max_ms=_round(det["gap_max_ms"], 2),
                    endpoint_gap_mrad=_round(det["endpoint_gap_mrad"], 3),
                    n_state_samples=det["n_samples"], state_hz=det["state_hz"],
                    tracking={ch: dict(mean_lag_mrad=_round(c["mean_lag_mrad"], 3),
                                       peak_lag_mrad=_round(c["peak_lag_mrad"], 3),
                                       tau_ms=_round(c["tau_ms"], 2),
                                       n_dropouts=len(c["dropouts"]))
                              for ch, c in det["chains"].items()})
        steps_out.append(row)
        traces.append((entry.get("index"), entry.get("name"), logs))

    # The premove's own samples, under step_index -1: it is real motion of the
    # robot (7-8 s of it, measured), and dropping it would leave the trace
    # starting mid-approach with no record of where the robot came from.
    if rec.get("premove"):
        traces.insert(0, (-1, "premove",
                          rec["premove"] if isinstance(rec["premove"], list)
                          else [rec["premove"]]))

    premove = None
    pm = rec.get("premove")
    if pm:
        # The premove is stored as the server log itself, not as a step entry with
        # a "logs" key, so it is wrapped by hand rather than through _as_logs.
        pm_logs = pm if isinstance(pm, list) else [pm]
        win, span = _step_window(pm_logs), _sample_span(pm_logs)
        if win or span:
            start = min(x for x in (win and win[0], span and span[0]) if x is not None)
            end = max(x for x in (win and win[1], span and span[1]) if x is not None)
            premove = dict(t_start_epoch=start,
                           measured_duration_s=_round(end - start, 4),
                           note="blocking move to the plan's start pose; belongs "
                                "to no phase, and is not in planned_total_s")
            t_first, t_last = min(t_first, start), max(t_last, end)

    measured_steps = [r["measured_duration_s"] for r in steps_out
                      if r.get("measured_duration_s") is not None]
    return dict(
        record_path=os.path.relpath(path),
        executed_at=rec.get("executed_at"),
        completed=bool(rec.get("completed")),
        error=rec.get("error"),
        n_steps=len(steps_out),
        premove=premove,
        # As recorded by plan_io (client-side, includes every round-trip), where
        # measured_wall_s below is reconstructed from the server's own timestamps.
        # Both are kept: they bracket the same interval from the two ends of the
        # link, and their difference is the RPC overhead.
        recorded_wall_s=_round(rec.get("wall_s"), 3),
        recorded_premove_wall_s=_round(rec.get("premove_wall_s"), 3),
        planned_total_s=_round(sum(p["planned_duration_s"] for p in phases), 3),
        measured_step_total_s=_round(sum(measured_steps), 3) if measured_steps else None,
        measured_wall_s=(_round(t_last - t_first, 3)
                         if np.isfinite(t_first) and np.isfinite(t_last) else None),
        steps=steps_out,
    ), traces, cmd_by_name


def write_exec_traces(path, traces, t_zero, cmd_by_name=None):
    """Dense measured traces for one execution, one npz.

    Everything the server logged, concatenated across steps with a step index and
    a clock zeroed at the start of the run, alongside the *commanded* waypoints on
    that same clock (``cmd_*``). Both halves are needed for the one plot the
    execution record exists to support: what the robot was told against what it
    did. Alignment is by absolute timestamp -- ``command_timestamps`` is when each
    waypoint was actually sent -- so nothing here assumes the pacing was nominal.

    ``step_names[step_index]`` labels every sample, EXCEPT where ``step_index`` is
    ``-1``: that is the premove, which belongs to no plan step.
    """
    fields = ("joint_position", "joint_velocity", "joint_torque",
              "left_wrist_forces", "left_wrist_torques",
              "right_wrist_forces", "right_wrist_torques",
              "left_gripper", "right_gripper")
    cols = {f: [] for f in fields}
    ts, step_idx, names = [], [], {}
    cmd_ts, cmd_q, cmd_idx = [], [], []
    for idx, name, logs in traces:
        if idx is not None and idx >= 0:    # -1 is the premove; see below
            names[idx] = name or ""
        for log in logs:
            at = np.asarray(log.get("timestamp", []), dtype=float)
            if at.size:
                ts.append(at - t_zero)
                step_idx.append(np.full(at.size, -1 if idx is None else idx, dtype=int))
                for f in fields:
                    v = np.asarray(log.get(f, []), dtype=float)
                    cols[f].append(v if v.size and v.shape[0] == at.size
                                   else np.full((at.size,) + (v.shape[1:] or ()), np.nan))
            ct = np.asarray(log.get("command_timestamps", []), dtype=float)
            planned = (cmd_by_name or {}).get(name)
            if ct.size and planned is not None:
                n = min(ct.size, len(planned))
                cmd_ts.append(ct[:n] - t_zero)
                cmd_q.append(np.asarray(planned)[:n])
                cmd_idx.append(np.full(n, -1 if idx is None else idx, dtype=int))
    if not ts:
        return False
    n_steps = (max(names) + 1) if names else 0
    out = dict(
        t_s=np.concatenate(ts), step_index=np.concatenate(step_idx),
        # Indexed BY step_index, so step_names[step_index] labels every sample.
        step_names=np.array([names.get(i, "") for i in range(n_steps)]),
        body_slice=np.array([BODY.start, BODY.stop]),
        chain_slices=np.array([[s.start, s.stop] for s in CHAINS.values()]),
        chain_names=np.array(list(CHAINS)))
    for f in fields:
        try:
            out[f] = np.concatenate(cols[f])
        except ValueError:
            continue                       # ragged across steps: skip that field
    if cmd_ts:
        out.update(cmd_t_s=np.concatenate(cmd_ts), cmd_q20=np.vstack(cmd_q),
                   cmd_step_index=np.concatenate(cmd_idx),
                   cmd_joint_names=np.array(["torso"] * 6 + ["right"] * 7
                                            + ["left"] * 7))
    np.savez_compressed(path, **out)
    return True


# ── Per-point assembly ────────────────────────────────────────────────────────

def export_point(index, cache_dir, out_dir, *, prefix="point", kin=None,
                 runtime=None, status=None, results_dirs=()):
    plan_path = os.path.join(cache_dir, f"{prefix}_{index:02d}.pkl")
    with open(plan_path, "rb") as f:
        payload = pickle.load(f)
    steps, meta = payload["steps"], payload.get("meta", {}) or {}
    legs = meta.get("legs") or []
    if not legs:
        raise ValueError(f"{plan_path}: meta['legs'] is empty; nothing to export")

    phases, t, q, phase_of_sample = build_timeline(steps, legs)
    cmd_t, cmd_q20, cmd_phase, cmd_dt = commanded_timeline(steps, phases)
    evidence = meta.get("evidence", {}) or {}
    trajopt = meta.get("trajopt", {}) or {}
    trace = kin.trace(q) if kin is not None else None

    # Boundary configurations: the phase table's reason for existing. For a
    # trajectory phase these are the first and last stored samples; for a gripper
    # phase the configuration is held, so both ends are the sample in force.
    for p in phases:
        sel = np.flatnonzero(phase_of_sample == p["phase_index"])
        if sel.size:
            p["q_start"] = [_round(v) for v in q[sel[0]]]
            p["q_end"] = [_round(v) for v in q[sel[-1]]]
            p["sample_slice"] = [int(sel[0]), int(sel[-1]) + 1]
        else:
            j = int(np.searchsorted(t, p["t_start_s"], side="right")) - 1
            j = max(0, min(j, q.shape[0] - 1))
            p["q_start"] = p["q_end"] = [_round(v) for v in q[j]]
            p["sample_slice"] = [j, j]
        if p["kind"] == "trajectory":
            p["evidence"] = _jsonable(evidence.get(p["name"], {}))
            tj = dict(trajopt.get(p["name"], {}) or {})
            tj.pop("rrt_path", None)           # dense; lives in the npz
            tj.pop("s_goal", None)
            p["trajopt"] = _jsonable(tj)
        if kin is not None:
            p["kinematics_start"] = _jsonable(kin.at(p["q_start"]))
            p["kinematics_end"] = _jsonable(kin.at(p["q_end"]))
            # Worst-over-phase, so these are comparable with meta["evidence"]'s
            # numbers (which are worst-over-leg) rather than with the boundary
            # values above. Density differs: evidence sampled the live trajectory
            # 300 times, this walks the 30 Hz record the plan actually stores.
            lo, hi = p["sample_slice"]
            if hi > lo:
                p["worst_over_phase"] = dict(
                    com_margin_m=_round(float(np.min(trace["com_margin_m"][lo:hi]))),
                    com_margin_legacy=_round(
                        float(np.min(trace["com_margin_legacy"][lo:hi]))),
                    joint_limit_headroom_rad=_round(float(np.min(
                        trace["joint_limit_headroom_rad"][lo:hi]))),
                    n_samples_checked=int(hi - lo))

    # Dense arrays for the npz.
    arrays = dict(
        t_s=t, q23=q, phase_index=phase_of_sample,
        phase_names=np.array([p["name"] for p in phases]),
        phase_t_start_s=np.array([p["t_start_s"] for p in phases]),
        phase_t_end_s=np.array([p["t_end_s"] for p in phases]),
        gripper_closed=np.array([phases[i]["gripper_closed_after"]
                                 for i in phase_of_sample]),
        cmd_t_s=cmd_t, cmd_q20=cmd_q20, cmd_phase_index=cmd_phase,
        cmd_dt_s=np.array(cmd_dt if cmd_dt else np.nan),
        joint_names=np.array(kin.joint_names if kin else
                             [f"q{i}" for i in range(q.shape[1])]),
        cmd_joint_names=np.array(["torso"] * 6 + ["right"] * 7 + ["left"] * 7),
    )
    if trace is not None:
        arrays.update(trace)
        arrays["q_lower_limit"] = kin.q_lb
        arrays["q_upper_limit"] = kin.q_ub
    # The BiRRT path each constrained leg was seeded from, when the plan kept it:
    # the only record of what trajopt started from, and unrecoverable otherwise.
    for name, tj in trajopt.items():
        if isinstance(tj, dict) and tj.get("rrt_path") is not None:
            arrays[f"rrt_path__{name}"] = np.asarray(tj["rrt_path"], dtype=float)
        if isinstance(tj, dict) and tj.get("s_goal") is not None:
            arrays[f"s_goal__{name}"] = np.asarray(tj["s_goal"], dtype=float)

    os.makedirs(os.path.join(out_dir, "points"), exist_ok=True)
    npz_path = os.path.join(out_dir, "points", f"{prefix}_{index:02d}.npz")
    np.savez_compressed(npz_path, **arrays)

    # Executions, if this point was ever run on the robot.
    executions = []
    for ep in exec_records_for(cache_dir, index, prefix=prefix,
                               results_dirs=results_dirs):
        summary, traces, cmd_by_name = exec_summary(ep, plan_path, phases)
        stamp = os.path.splitext(os.path.basename(ep))[0].split("_exec_")[-1]
        tz = next((r.get("t_start_epoch") for r in summary["steps"]
                   if r.get("t_start_epoch") is not None), 0.0)
        if summary.get("premove"):
            tz = min(tz, summary["premove"]["t_start_epoch"])
        tpath = os.path.join(out_dir, "points",
                             f"{prefix}_{index:02d}_exec_{stamp}.npz")
        if write_exec_traces(tpath, traces, tz, cmd_by_name):
            summary["trace_npz"] = os.path.relpath(tpath, out_dir)
        summary["t_zero_epoch"] = tz
        executions.append(summary)

    planned_total = sum(p["planned_duration_s"] for p in phases)
    record = dict(
        index=index,
        box_xy=[_round(meta.get("bx")), _round(meta.get("by"))],
        plan_path=os.path.relpath(plan_path),
        plan_format_version=payload.get("format_version"),
        trajectory_npz=os.path.relpath(npz_path, out_dir),
        clock=dict(
            planned_total_s=_round(planned_total, 4),
            n_samples=int(q.shape[0]), sample_hz_effective=_round(
                (q.shape[0] - len(legs)) / planned_total, 3) if planned_total else None,
            n_commands=int(cmd_q20.shape[0]), command_dt_s=_round(cmd_dt, 4),
            note="planned time; gripper steps occupy zero planned seconds",
        ),
        # How far the shipped trajectory's grasp endpoint sits from the grasp
        # configuration the pipeline solved for. Not zero: the leg is retimed and
        # reconstructed after that endpoint was chosen, so this is the fidelity of
        # the thing that actually executes, and nothing else records it.
        grasp_endpoint_gap_rad=_round(float(np.max(np.abs(
            np.asarray(meta["q_grasp"], dtype=float)
            - np.asarray(next(p["q_end"] for p in phases
                              if p["name"] == "reach_descend"), dtype=float))))
        ) if meta.get("q_grasp") is not None and any(
            p["name"] == "reach_descend" for p in phases) else None,
        phases=phases,
        planning_runtime=(runtime or {}).get(index),
        planning_wall_s=_round(meta.get("wall_s"), 3),
        status=_jsonable(status) if status is not None else None,
        executions=executions,
        # Verbatim. The exported record must not become a lossy re-statement of
        # the plan: anything derived above is derived FROM this.
        plan_meta=_jsonable({k: v for k, v in meta.items() if k != "legs"}),
        plan_meta_legs=[dict(name=lg["name"], duration=_round(lg["duration"], 6),
                             with_box=bool(lg.get("with_box")),
                             n_samples=int(np.asarray(lg["q"]).shape[0]))
                        for lg in legs],
    )
    json_path = os.path.join(out_dir, "points", f"{prefix}_{index:02d}.json")
    with open(json_path, "w") as f:
        json.dump(record, f, indent=1)
    return record


# ── Cross-point tables ────────────────────────────────────────────────────────

PHASE_COLUMNS = [
    "point", "phase_index", "phase", "group", "kind", "planner",
    "t_start_s", "t_end_s", "planned_duration_s", "measured_duration_s",
    "pacing_ratio", "n_samples", "n_waypoints", "with_box", "gripper_closed_after",
    "min_clearance_mm", "min_clearance_pair", "com_margin_mm_legacy",
    "com_margin_mm", "joint_limit_excess_rad", "verified_ok",
    "trajopt_info", "trajopt_uncertified", "straight_line",
    "ee_right_x", "ee_right_y", "ee_right_z", "ee_left_x", "ee_left_y", "ee_left_z",
    "mid_x", "mid_y", "mid_z", "box_x", "box_y", "box_z",
]


def phase_rows(record):
    """One row per phase, end-of-phase kinematics -- the keyframe table."""
    measured = {}
    for ex in record["executions"]:
        for s in ex["steps"]:
            if s.get("measured_duration_s") is not None:
                measured.setdefault(s["name"], (s["measured_duration_s"],
                                                s.get("pacing_ratio")))
    rows = []
    for p in record["phases"]:
        ev = p.get("evidence") or {}
        tj = p.get("trajopt") or {}
        k = p.get("kinematics_end") or {}
        wp = p.get("worst_over_phase") or {}
        m, pace = measured.get(p["name"], (None, None))
        rows.append({
            "point": record["index"], "phase_index": p["phase_index"],
            "phase": p["name"], "group": p["group"], "kind": p["kind"],
            "planner": p["planner"],
            "t_start_s": _round(p["t_start_s"], 4), "t_end_s": _round(p["t_end_s"], 4),
            "planned_duration_s": _round(p["planned_duration_s"], 4),
            "measured_duration_s": m, "pacing_ratio": pace,
            "n_samples": p["n_samples"], "n_waypoints": p["n_waypoints"],
            "with_box": p["with_box"],
            "gripper_closed_after": p["gripper_closed_after"],
            "min_clearance_mm": _round(1000 * ev["min_clearance_m"], 3)
            if ev.get("min_clearance_m") is not None else None,
            "min_clearance_pair": ev.get("min_clearance_pair"),
            "com_margin_mm_legacy": _round(1000 * ev["com_margin_m"], 3)
            if ev.get("com_margin_m") is not None else None,
            "com_margin_mm": _round(1000 * wp["com_margin_m"], 3)
            if wp.get("com_margin_m") is not None else None,
            "joint_limit_excess_rad": ev.get("joint_limit_excess_rad"),
            "verified_ok": ev.get("ok"),
            "trajopt_info": tj.get("trajopt_info"),
            "trajopt_uncertified": tj.get("trajopt_uncertified"),
            "straight_line": tj.get("straight_line"),
            **{f"ee_right_{a}": _round(k["ee_right_xyz"][i], 5)
               for i, a in enumerate("xyz") if k.get("ee_right_xyz")},
            **{f"ee_left_{a}": _round(k["ee_left_xyz"][i], 5)
               for i, a in enumerate("xyz") if k.get("ee_left_xyz")},
            **{f"mid_{a}": _round(k["mid_gripper_xyz"][i], 5)
               for i, a in enumerate("xyz") if k.get("mid_gripper_xyz")},
            **{f"box_{a}": _round(k["box_xyz"][i], 5)
               for i, a in enumerate("xyz") if k.get("box_xyz")},
        })
    return rows


SUMMARY_COLUMNS = [
    "point", "box_x", "box_y", "planned_total_s", "n_phases", "n_samples",
    "n_commands", "planning_latency_s", "planning_wall_s", "planning_work_s",
    "planning_work_discarded_s", "concurrency_ratio", "solver_latency_s",
    "solver_share_of_latency", "largest_latency_category", "gcp_index",
    "grasp_seed", "grasp_score", "grasp_n_candidates",
    "worst_clearance_mm", "worst_com_margin_mm_legacy", "worst_com_margin_mm",
    "worst_joint_limit_excess_rad", "all_legs_ok", "trajopt_uncertified_legs",
    "n_executions", "last_execution_at", "last_execution_completed",
    "measured_step_total_s", "measured_wall_s",
]


def summary_row(record, rows):
    rt = record.get("planning_runtime") or {}
    meta = record.get("plan_meta") or {}
    cats = rt.get("latency_by_category_s") or {}
    traj_rows = [r for r in rows if r["kind"] == "trajectory"]

    def worst(col, sign=1):
        vals = [r[col] for r in traj_rows if r.get(col) is not None]
        return (min(vals) if sign > 0 else max(vals)) if vals else None

    ex = record["executions"][-1] if record["executions"] else {}
    return {
        "point": record["index"],
        "box_x": record["box_xy"][0], "box_y": record["box_xy"][1],
        "planned_total_s": record["clock"]["planned_total_s"],
        "n_phases": len(record["phases"]),
        "n_samples": record["clock"]["n_samples"],
        "n_commands": record["clock"]["n_commands"],
        "planning_latency_s": rt.get("latency_s"),
        "planning_wall_s": record.get("planning_wall_s"),
        "planning_work_s": rt.get("work_s"),
        "planning_work_discarded_s": rt.get("work_discarded_s"),
        "concurrency_ratio": rt.get("concurrency_ratio"),
        "solver_latency_s": rt.get("solver_latency_s"),
        "solver_share_of_latency": rt.get("solver_share_of_latency"),
        "largest_latency_category": next(iter(cats), None),
        "gcp_index": meta.get("gcp_index"), "grasp_seed": meta.get("grasp_seed"),
        "grasp_score": meta.get("grasp_score"),
        "grasp_n_candidates": meta.get("grasp_n_candidates"),
        "worst_clearance_mm": worst("min_clearance_mm"),
        "worst_com_margin_mm_legacy": worst("com_margin_mm_legacy"),
        "worst_com_margin_mm": worst("com_margin_mm"),
        "worst_joint_limit_excess_rad": worst("joint_limit_excess_rad", sign=-1),
        "all_legs_ok": all(r.get("verified_ok") for r in traj_rows) or None,
        "trajopt_uncertified_legs": ";".join(
            r["phase"] for r in traj_rows if r.get("trajopt_uncertified")),
        "n_executions": len(record["executions"]),
        "last_execution_at": ex.get("executed_at"),
        "last_execution_completed": ex.get("completed"),
        "measured_step_total_s": ex.get("measured_step_total_s"),
        "measured_wall_s": ex.get("measured_wall_s"),
    }


def write_csv(path, columns, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def model_constants():
    """The scene the numbers mean nothing without, read from the source of truth.

    Returns ``None`` rather than failing when the planning stack will not import:
    importing the constants pulls in Drake, and the rest of this export does not
    need it under --no-kinematics.
    """
    try:
        import plan_grid as ept
    except Exception as e:                  # noqa: BLE001 -- any import failure
        print(f"note: model constants unavailable ({type(e).__name__}: {e})")
        return None
    return dict(
        hand=ept.HAND, box_size_m=list(ept.SIZE),
        box_offset_in_ee_m=list(ept.OFFSET), box_rpy_rad=list(ept.RPY),
        box_wall_thickness_m=ept.WALL_T, box_wall_height_m=ept.WALL_HEIGHT,
        grasp_half_separation_m=ept.GRASP_HALF_SEP, grasp_z_m=ept.GRASP_Z,
        standoff_m=ept.STANDOFF_M,
        table_size_m=list(ept.TABLE_SIZE), table_xyz_m=list(ept.TABLE_XYZ),
        tabletop_z_m=ept.TABLETOP_Z,
        box_bottom_below_ee_m=ept.BOX_BOTTOM_BELOW_EE,
        carry_clearance_m=ept.CARRY_CLEARANCE,
        place_clearance_m=ept.PLACE_CLEARANCE,
        carry_hover_xyz_m=list(ept.T_W_MID_ABOVE.translation()),
        place_target_xyz_m=list(ept.PLACE_TARGET.translation()),
        leg_seeds=list(ept.LEG_SEEDS),
        constrained_legs=list(ept.CONSTRAINED_LEG_NAMES),
        com_note=("CoM sums base, torso, both arms, both grippers and the head; "
                  "the carried box is geometry only and contributes no mass, "
                  "matching the planner's own stability check."),
    )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", default=os.path.join("plans", "grid_cache"))
    ap.add_argument("--out", default=None,
                    help="output directory (default <cache-dir>/experiment_record)")
    ap.add_argument("--prefix", default="point", help="cache file prefix")
    ap.add_argument("--indices", type=int, nargs="*", default=None,
                    help="only these grid points (default: every cached point)")
    ap.add_argument("--log", nargs="*", default=None,
                    help="explicit event log(s) instead of <cache-dir>/timing/*.jsonl")
    ap.add_argument("--results-dir", nargs="*", default=None, metavar="DIR",
                    help="where to look for execution records, in addition to the "
                         "cache dir (older records live there). Default: "
                         "<repo>/results and ./results.")
    ap.add_argument("--no-kinematics", action="store_true",
                    help="skip end-effector/box/CoM derivation (no Drake needed)")
    args = ap.parse_args()

    out_dir = args.out or os.path.join(args.cache_dir, "experiment_record")
    indices = args.indices or point_indices(args.cache_dir, args.prefix)
    if not indices:
        sys.exit(f"no {args.prefix}_NN.pkl plans in {args.cache_dir}")
    os.makedirs(out_dir, exist_ok=True)

    runtime, runs = load_runtime(args.cache_dir, args.log)
    if not runtime:
        print(f"note: no event log under {args.cache_dir}/timing -- planning "
              f"runtime will fall back to meta['wall_s'] only")

    status = {}
    spath = os.path.join(args.cache_dir, "status.json")
    if os.path.exists(spath):
        with open(spath) as f:
            status = json.load(f)

    kin = None
    if not args.no_kinematics:
        print("building the plant for end-effector / box / CoM derivation...")
        kin = Kinematics()

    repo_results = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    results_dirs = (args.results_dir if args.results_dir is not None
                    else [repo_results, "results"])

    records, all_phase_rows, summary_rows = [], [], []
    for idx in indices:
        rec = export_point(idx, args.cache_dir, out_dir, prefix=args.prefix,
                           kin=kin, runtime=runtime, status=status.get(str(idx)),
                           results_dirs=results_dirs)
        rows = phase_rows(rec)
        records.append(rec)
        all_phase_rows += rows
        summary_rows.append(summary_row(rec, rows))
        rt = rec.get("planning_runtime") or {}
        print(f"[{idx:2d}] {len(rec['phases'])} phases, "
              f"{rec['clock']['planned_total_s']:6.2f} s planned, "
              f"{rec['clock']['n_samples']:5d} samples, "
              f"plan latency {rt.get('latency_s') or rec.get('planning_wall_s')} s, "
              f"{len(rec['executions'])} execution(s)")

    write_csv(os.path.join(out_dir, "phases.csv"), PHASE_COLUMNS, all_phase_rows)
    write_csv(os.path.join(out_dir, "summary.csv"), SUMMARY_COLUMNS, summary_rows)

    lats = [r["planning_latency_s"] or r["planning_wall_s"] for r in summary_rows
            if (r["planning_latency_s"] or r["planning_wall_s"])]
    durs = [r["planned_total_s"] for r in summary_rows if r["planned_total_s"]]
    run_json = dict(
        exported_at=datetime.datetime.now().isoformat(timespec="seconds"),
        cache_dir=os.path.relpath(args.cache_dir),
        n_points=len(records),
        points=[r["index"] for r in records],
        kinematics=bool(kin),
        joint_names=(kin.joint_names if kin else None),
        joint_layout="23 active DOFs: base(3), torso(6), right_arm(7), left_arm(7)",
        command_layout="20 commanded DOFs: torso(6), right_arm(7), left_arm(7)",
        phase_order=[p["name"] for p in (records[0]["phases"] if records else [])],
        planning_latency_s=dict(
            mean=_round(float(np.mean(lats)), 3) if lats else None,
            median=_round(float(np.median(lats)), 3) if lats else None,
            min=_round(min(lats), 3) if lats else None,
            max=_round(max(lats), 3) if lats else None),
        planned_duration_s=dict(
            mean=_round(float(np.mean(durs)), 3) if durs else None,
            min=_round(min(durs), 3) if durs else None,
            max=_round(max(durs), 3) if durs else None),
        n_executions=sum(len(r["executions"]) for r in records),
        event_log_runs=_jsonable(runs),
        model_constants=_jsonable(model_constants()),
        caveats=[
            "Planned time only: gripper steps carry no planned duration, so the "
            "planned total is shorter than any real execution by the grasp and "
            "release time plus the premove.",
            "com_margin_mm_legacy reproduces meta['evidence']/status.json, which "
            "is (distance x support-polygon edge length), not millimetres. "
            "com_margin_mm is the normalised distance. Signs agree, magnitudes "
            "do not.",
            "Guarantee numbers are copied from the plan's own dense verification, "
            "not recomputed; run scripts/verify_cached_plan.py to re-derive them.",
        ],
    )
    with open(os.path.join(out_dir, "run.json"), "w") as f:
        json.dump(run_json, f, indent=1)

    n_exec = run_json["n_executions"]
    print(f"\nwrote {out_dir}/")
    print("  run.json      run provenance, constants, aggregates")
    print(f"  phases.csv    {len(all_phase_rows)} phase rows (the keyframe table)")
    print(f"  summary.csv   {len(summary_rows)} point rows")
    print(f"  points/       {len(records)} json + npz"
          + (f", {n_exec} execution npz" if n_exec else ""))


if __name__ == "__main__":
    raise SystemExit(main())
