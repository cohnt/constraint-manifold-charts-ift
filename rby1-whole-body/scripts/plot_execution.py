#!/usr/bin/env python
"""Plot commanded vs. executed joint values from a plan + its execution record.

An execution record (written by the robot client (not part of this release) into ``results/``)
holds, per step, the server's logged *actual* joint state (``joint_position``)
plus ``command_timestamps`` -- the instant each commanded waypoint went out, on
the same wall-clock as ``timestamp``. The commanded *values* live in the plan
(``steps[i]["cmds"]``). This script lines the two up on one timeline and lets
you page through joints interactively.

Usage:
    # point it at a record; the plan is read from the record's plan_path
    .venv/bin/python scripts/plot_execution.py results/2026-08-11/point_00_exec_20260811_171307.pkl

    # or at a plan; the newest matching record in results/ is used
    .venv/bin/python scripts/plot_execution.py plans/grid_cache/point_00.pkl

    # override either explicitly, preselect a joint, or save instead of show
    .venv/bin/python scripts/plot_execution.py REC.pkl --plan PLAN.pkl --joint right_arm_3
    .venv/bin/python scripts/plot_execution.py REC.pkl --save exec_plot.png

Interaction: left/right arrow keys or the Prev/Next buttons or the slider select
the joint. Top axes overlays commanded (stairs) and executed (line); bottom axes
shows the tracking error (executed - commanded) with RMS/peak in the title.
"""

import argparse
import glob
import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
import plan_format  # noqa: F401  -- installs the pre-rename module aliases the
                    # committed plan and execution pickles were written with

# Server state-vector layout (see utilities.parse_state_vector):
#   wheels[0:2], torso[2:8], right_arm[8:15], left_arm[15:22], head[22:24].
JOINT_NAMES = (
    ["right_wheel", "left_wheel"]
    + [f"torso_{i}" for i in range(6)]
    + [f"right_arm_{i}" for i in range(7)]
    + [f"left_arm_{i}" for i in range(7)]
    + [f"head_{i}" for i in range(2)]
)
# state index -> (commanded chain key, offset within that chain), for the joints
# the plan actually commands (torso/right/left). Wheels and head are unmanaged.
CMD_MAP = {}
for i in range(2, 8):
    CMD_MAP[i] = ("torso", i - 2)
for i in range(8, 15):
    CMD_MAP[i] = ("right", i - 8)
for i in range(15, 22):
    CMD_MAP[i] = ("left", i - 15)


def _looks_like_record(obj):
    """True for an execution record from either writer.

    The recorded runs embed the plan's whole meta as ``plan_meta``;
    ``plan_grid.py`` records the plan's path instead and leaves
    the meta on disk (it holds the dense per-leg sample arrays, which would be
    tens of MB duplicated into every run). Keying only on ``plan_meta`` rejected
    the second kind outright, so require the field they actually share -- an
    executed step list -- plus any of the three ways a record names its plan or
    its outcome.
    """
    return (isinstance(obj, dict) and "steps" in obj
            and any(k in obj for k in ("plan_meta", "plan_path", "completed")))


def _looks_like_plan(obj):
    return isinstance(obj, dict) and obj.get("format_version") is not None


def resolve_inputs(path, plan_override):
    """Return (plan_dict, record_dict) from a single record-or-plan path."""
    with open(path, "rb") as f:
        obj = pickle.load(f)

    if _looks_like_record(obj):
        record = obj
        plan_path = plan_override or record.get("plan_path")
        if not plan_path or not os.path.exists(plan_path):
            raise SystemExit(
                f"Could not find the plan for this record (plan_path={plan_path!r}). "
                "Pass it with --plan."
            )
        with open(plan_path, "rb") as f:
            plan = pickle.load(f)
        return plan, record

    if _looks_like_plan(obj):
        plan = obj
        if plan_override:  # user passed a plan positionally AND --plan (a record)
            with open(plan_override, "rb") as f:
                return plan, pickle.load(f)
        stem = os.path.splitext(os.path.basename(path))[0]
        cands = sorted(glob.glob(os.path.join("results", f"{stem}_exec_*.pkl")))
        if not cands:
            raise SystemExit(
                f"No execution record found in results/ for plan '{stem}'. "
                "Pass the path to one of the committed records in results/ instead."
            )
        with open(cands[-1], "rb") as f:
            record = pickle.load(f)
        print(f"Using newest record: {cands[-1]}")
        return plan, record

    raise SystemExit(f"{path!r} is neither a plan nor an execution record.")


def build_timeline(plan, record):
    """Assemble the achieved timeline and the per-chain commanded samples.

    Returns (t0, achieved_t, achieved_q, cmd_t, cmd_v, step_marks) where
    achieved_q is (N, 24), cmd_t/cmd_v are dicts keyed by chain, and step_marks
    is a list of (time, name) for drawing step boundaries.
    """
    rec_steps = record["steps"]
    plan_steps = plan["steps"]
    by_name = {s["name"]: s for s in plan_steps}

    # Global t0 across everything with a timestamp (incl. premove).
    all_t = []
    log_blocks = []
    if record.get("premove") and isinstance(record["premove"], dict):
        log_blocks.append(("premove", record["premove"]))
    for rs in rec_steps:
        logs = rs.get("logs")
        if isinstance(logs, dict):
            log_blocks.append((rs["name"], logs))
    for _, logs in log_blocks:
        ts = np.asarray(logs.get("timestamp", []))
        if ts.size:
            all_t.append(ts)
    if not all_t:
        raise SystemExit("Record has no logged timestamps to plot.")
    t0 = min(a.min() for a in all_t)

    # Achieved: concatenate every block's (timestamp, joint_position) in order.
    ach_t, ach_q, step_marks = [], [], []
    for name, logs in log_blocks:
        ts = np.asarray(logs.get("timestamp", []))
        jp = np.asarray(logs.get("joint_position", []))
        if ts.size == 0 or jp.size == 0:
            continue
        step_marks.append((ts.min() - t0, name))
        ach_t.append(ts - t0)
        ach_q.append(jp)
    ach_t = np.concatenate(ach_t)
    ach_q = np.vstack(ach_q)

    # Commanded: for each trajectory step, pair command_timestamps with cmds.
    cmd_t = {"torso": [], "right": [], "left": []}
    cmd_v = {"torso": [], "right": [], "left": []}
    for rs in rec_steps:
        ps = by_name.get(rs["name"])
        if not ps or ps.get("type") != "trajectory":
            continue
        logs = rs.get("logs") or {}
        ct = np.asarray(logs.get("command_timestamps", []))
        cmds = ps.get("cmds", [])
        n = min(len(ct), len(cmds))
        for k in range(n):
            wp = cmds[k]
            for chain in ("torso", "right", "left"):
                if chain in wp:
                    cmd_t[chain].append(ct[k] - t0)
                    cmd_v[chain].append(np.asarray(wp[chain].target_position))
    for chain in cmd_t:
        cmd_t[chain] = np.asarray(cmd_t[chain])
        cmd_v[chain] = np.asarray(cmd_v[chain]) if cmd_v[chain] else np.empty((0, 0))
    return t0, ach_t, ach_q, cmd_t, cmd_v, step_marks


def commanded_for_joint(idx, cmd_t, cmd_v):
    """Return (times, values) commanded for state-vector joint idx, or (None, None)."""
    if idx not in CMD_MAP:
        return None, None
    chain, off = CMD_MAP[idx]
    t, v = cmd_t[chain], cmd_v[chain]
    if t.size == 0 or v.size == 0:
        return None, None
    return t, v[:, off]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="Execution record (results/*_exec_*.pkl) or a plan pickle.")
    ap.add_argument("--plan", default=None, help="Override the plan pickle path.")
    ap.add_argument("--joint", default=None,
                    help="Preselect a joint by name (e.g. right_arm_3) or index (0-23).")
    ap.add_argument("--save", metavar="PATH", default=None,
                    help="Save the figure to PATH and exit instead of showing it.")
    args = ap.parse_args()

    plan, record = resolve_inputs(args.path, args.plan)
    t0, ach_t, ach_q, cmd_t, cmd_v, step_marks = build_timeline(plan, record)
    n_joints = ach_q.shape[1]

    # Resolve the starting joint.
    start = 8  # right_arm_0: an actuated joint is a more useful default than a wheel
    if args.joint is not None:
        if args.joint.isdigit():
            start = int(args.joint)
        elif args.joint in JOINT_NAMES:
            start = JOINT_NAMES.index(args.joint)
        else:
            raise SystemExit(f"Unknown joint {args.joint!r}. Options: {JOINT_NAMES}")

    fig, (ax, ax_err) = plt.subplots(
        2, 1, figsize=(11, 7), sharex=True, height_ratios=[3, 1])
    plt.subplots_adjust(bottom=0.17, right=0.85, hspace=0.15)
    state = {"idx": start}

    def joint_name(i):
        return JOINT_NAMES[i] if i < len(JOINT_NAMES) else f"joint_{i}"

    def draw():
        idx = state["idx"]
        ax.clear(); ax_err.clear()

        # Step boundaries for context.
        for tm, name in step_marks:
            for a in (ax, ax_err):
                a.axvline(tm, color="0.85", lw=1, zorder=0)
            ax.text(tm, 1.005, name, rotation=90, va="bottom", ha="right",
                    fontsize=7, color="0.5", transform=ax.get_xaxis_transform())

        ax.plot(ach_t, ach_q[:, idx], color="C0", lw=1.4, label="executed")

        ct, cv = commanded_for_joint(idx, cmd_t, cmd_v)
        title = f"[{idx}] {joint_name(idx)}"
        if ct is not None:
            ax.plot(ct, cv, color="C3", lw=1.2, ls="--", drawstyle="steps-post",
                    label="commanded")
            # Tracking error: interpolate commanded onto achieved times, in range.
            m = (ach_t >= ct.min()) & (ach_t <= ct.max())
            if m.any():
                cmd_i = np.interp(ach_t[m], ct, cv)
                err = ach_q[m, idx] - cmd_i
                ax_err.plot(ach_t[m], err, color="C2", lw=1.0)
                ax_err.axhline(0, color="0.7", lw=0.8)
                rms = float(np.sqrt(np.mean(err ** 2)))
                peak = float(np.max(np.abs(err)))
                title += f"   RMS err {np.rad2deg(rms):.2f}°  peak {np.rad2deg(peak):.2f}°"
        else:
            ax_err.text(0.5, 0.5, "(joint not commanded by the plan)",
                        ha="center", va="center", transform=ax_err.transAxes,
                        color="0.5", fontsize=9)

        ax.set_title(title)
        ax.set_ylabel("position (rad)")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax_err.set_ylabel("err (rad)")
        ax_err.set_xlabel("time since first sample (s)")
        ax_err.grid(True, alpha=0.3)
        fig.canvas.draw_idle()

    def set_idx(i):
        state["idx"] = int(np.clip(i, 0, n_joints - 1))
        if abs(slider.val - state["idx"]) > 1e-9:
            slider.set_val(state["idx"])  # triggers on_slide -> draw
        else:
            draw()

    # Slider + Prev/Next buttons + arrow keys.
    ax_slider = plt.axes([0.15, 0.06, 0.5, 0.03])
    slider = Slider(ax_slider, "joint", 0, n_joints - 1, valinit=start,
                    valstep=1, valfmt="%d")
    slider.on_changed(lambda v: (state.__setitem__("idx", int(v)), draw()))

    ax_prev = plt.axes([0.70, 0.055, 0.06, 0.04])
    ax_next = plt.axes([0.78, 0.055, 0.06, 0.04])
    b_prev = Button(ax_prev, "◀ Prev")
    b_next = Button(ax_next, "Next ▶")
    b_prev.on_clicked(lambda e: set_idx(state["idx"] - 1))
    b_next.on_clicked(lambda e: set_idx(state["idx"] + 1))

    def on_key(event):
        if event.key in ("right", "up"):
            set_idx(state["idx"] + 1)
        elif event.key in ("left", "down"):
            set_idx(state["idx"] - 1)
    fig.canvas.mpl_connect("key_press_event", on_key)

    draw()

    if args.save:
        fig.savefig(args.save, dpi=130, bbox_inches="tight")
        print(f"Saved {args.save} (joint {joint_name(state['idx'])})")
    else:
        plt.show()


if __name__ == "__main__":
    main()
