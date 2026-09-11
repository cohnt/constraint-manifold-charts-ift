"""Self-contained, Drake-free plan format shared by the planner and the robot.

A *plan* is an ordered list of **steps** executed back-to-back on the robot:

    - a ``"trajectory"`` step: a list of stiff joint-position waypoint commands
      (as produced by ``trajectory_conversion.joint_traj_to_commands``);
    - a ``"gripper"`` step: open/close one or more grippers.

This lets a full manipulation sequence — e.g. reach → grasp → lift → place →
release — be a single serialisable object that the robot client can run
on the Jetson without importing Drake or the planner. The payload holds only
plain dataclasses (``JointPositionCommand``), numpy arrays, and Python literals.

Format v2 supersedes the v1 ``{reach_cmds, lift_cmds}`` layout;
The loader still reads v1 plans for backward compatibility.

Also holds ``premove_from_start``/``execute_plan_steps``: the "run a loaded
plan against a live robot" logic shared by the robot client
and any other script that executes plan steps on the robot (e.g.
``plan_grid.py``), so that dispatch loop exists in one place.
"""

import pickle
import time

PLAN_FORMAT_VERSION = 2

# The grippers a bare "grasp"/"release" acts on, in command order.
DEFAULT_GRIPPERS = ("right_gripper", "left_gripper")


def trajectory_step(name, traj, layout, *, q_fn=None, hz=30.0, dt=None):
    """Build a trajectory step by converting a Drake trajectory to commands.

    Args mirror ``trajectory_conversion.joint_traj_to_commands``. ``name`` is a
    human label shown during execution (e.g. ``"reach"``).
    """
    # Local import: keeps this module importable on the Jetson (which never
    # builds trajectories) without pulling numpy in at module load for load-only
    # use, and avoids a hard import cycle with trajectory_conversion.
    from .trajectory_conversion import joint_traj_to_commands

    cmds = joint_traj_to_commands(traj, layout, q_fn=q_fn, hz=hz, dt=dt)
    return {
        "type": "trajectory",
        "name": name,
        "cmds": cmds,
        "duration": float(traj.end_time() - traj.start_time()),
    }


def gripper_step(action, *, grippers=DEFAULT_GRIPPERS, name=None):
    """Build a gripper step. ``action`` is ``"open"`` or ``"close"``."""
    if action not in ("open", "close"):
        raise ValueError(f"gripper action must be 'open' or 'close', got {action!r}")
    return {
        "type": "gripper",
        "name": name or action,
        "action": action,
        "grippers": list(grippers),
    }


def save_plan(path, steps, *, meta=None):
    """Pickle a step list to ``path`` as a versioned plan. Returns the payload."""
    if not any(s.get("type") == "trajectory" for s in steps):
        raise ValueError("plan has no trajectory steps")
    payload = {
        "format_version": PLAN_FORMAT_VERSION,
        "steps": list(steps),
        "meta": dict(meta or {}),
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    return payload


def load_plan(path):
    """Load a plan pickle and normalise it to a list of steps.

    Accepts both v2 (native step list) and v1 (``reach_cmds``/``lift_cmds``)
    payloads. Returns ``(steps, meta)``. v1 plans carry no gripper steps; the
    caller may inject a grasp between reach and lift.
    """
    with open(path, "rb") as f:
        payload = pickle.load(f)
    version = payload.get("format_version")
    meta = payload.get("meta", {})

    if version == PLAN_FORMAT_VERSION:
        return payload["steps"], meta

    if version == 1:
        # Legacy: reach + lift trajectories, no interleaved gripper actions.
        steps = []
        if payload.get("reach_cmds"):
            steps.append({"type": "trajectory", "name": "reach",
                          "cmds": payload["reach_cmds"],
                          "duration": meta.get("reach_duration", float("nan"))})
        if payload.get("lift_cmds"):
            steps.append({"type": "trajectory", "name": "lift",
                          "cmds": payload["lift_cmds"],
                          "duration": meta.get("lift_duration", float("nan"))})
        return steps, meta

    raise ValueError(
        f"Plan {path} has format_version {version!r}; this build understands "
        f"versions 1 and {PLAN_FORMAT_VERSION}. Re-save it with the current "
        "current plan_grid.py."
    )


def premove_from_start(start_waypoint, duration):
    """Build a single-waypoint move-to-start command from a trajectory step's
    first waypoint (its start pose), overriding the per-chain duration with the
    blocking approach duration."""
    # Local import: same reasoning as trajectory_step's — keeps this module
    # importable on the Jetson without pulling in numpy at module load for
    # load-only use (numpy is only needed once a command is actually built).
    import numpy as np

    from .commands import JointPositionCommand

    return [{
        chain: JointPositionCommand(
            duration=duration, target_position=np.asarray(cmd.target_position)
        )
        for chain, cmd in start_waypoint.items()
    }]


def execute_plan_steps(client, steps, *, premove=None, log=print, state_hz=None):
    """Run ``premove`` (if given), then each step via ``client``, in order.

    Returns a record dict: ``{"completed": bool, "error": str | None,
    "premove": <server logs or None>, "premove_wall_s": float | None,
    "t_start"/"t_end"/"wall_s": measured seconds for the whole sequence,
    "steps": [{"index", "name", "type", "logs", "t_start", "t_end", "wall_s"},
    ...]}``. Never raises — a failure mid-sequence is captured in
    ``"error"`` (as ``repr(e)``) with ``"completed"`` left ``False`` and
    ``"steps"`` holding whatever ran before the fault, so a caller always gets
    a record back to persist even on failure. What a failure *means* (stop,
    retry, propagate) is left to the caller.

    ``state_hz`` is forwarded to the client so the recorded joint state can be
    sampled faster than the server's 10 Hz default, which is below the 20 Hz
    command rate and so aliases per-waypoint behaviour.
    """
    record = {"completed": False, "error": None, "premove": None, "steps": [],
              "t_start": time.time(), "t_end": None, "wall_s": None,
              "premove_wall_s": None}
    try:
        if premove is not None:
            log("Moving to planned start pose...")
            t0 = time.time()
            record["premove"] = client.follow_joint_trajectory_stiff(
                premove, state_hz=state_hz)
            record["premove_wall_s"] = time.time() - t0

        for i, step in enumerate(steps):
            # How long a step took is otherwise recoverable only by digging the
            # server's command_timestamps out of its logs -- and not at all for a
            # gripper step, whose duration the plan does not state. Timing the call
            # here makes execution time a recorded number, in the same units the
            # plan's own `duration` is in.
            t0 = time.time()
            if step["type"] == "trajectory":
                log(f"Executing '{step['name']}' ({len(step['cmds'])} waypoints)...")
                logs = client.follow_joint_trajectory_stiff(
                    step["cmds"], state_hz=state_hz)
            else:  # gripper
                log(f"Gripper '{step['name']}': {step['action']} "
                    f"{', '.join(step['grippers'])}...")
                # One command for all of the step's grippers so they actuate
                # together; one call per gripper made them move in sequence.
                logs = client.execute_gripper_command(
                    step["grippers"], step["action"], state_hz=state_hz
                )
            record["steps"].append({
                "index": i, "name": step["name"], "type": step["type"], "logs": logs,
                "t_start": t0, "t_end": time.time(), "wall_s": time.time() - t0,
            })
        record["completed"] = True
    except BaseException as e:
        record["error"] = repr(e)
    # In a finally-equivalent position: a run that faulted mid-sequence is exactly
    # when "how far did it get, and how long did that take" matters.
    record["t_end"] = time.time()
    record["wall_s"] = record["t_end"] - record["t_start"]
    return record
