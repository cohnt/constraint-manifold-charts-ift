"""Convert planned Drake joint-space trajectories into robot commands.

The planning side (``rby1_planning.py``) produces Drake trajectories over the
full plant; a 23-D active vector ``[base(3), torso(6), right_arm(7),
left_arm(7)]`` is extracted via ``Rby1ActiveJointLayout`` (the same ordering the
notebook's ``q_fn`` yields). This module turns such a trajectory into the
``Iterable[dict[str, JointPositionCommand]]`` that
the robot client consumed, and offers a lightweight
meshcat playback helper for previewing/replaying a trajectory before execution.

The conversion deliberately avoids importing ``rby1_planning`` so it stays cheap
to import; it only duck-types the trajectory (``start_time``/``end_time``/
``value``).
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np

from .commands import JointPositionCommand

if TYPE_CHECKING:
    # Type-only import: avoids pulling the heavy planning module in at runtime
    # while letting type checkers verify the layout passed by callers.
    from rby1_planning import Rby1ActiveJointLayout


def _sample_times(t0: float, t1: float, dt: float) -> np.ndarray:
    """Evenly spaced sample times on [t0, t1], always including the endpoint."""
    n = max(int(np.ceil((t1 - t0) / dt)), 1)
    return np.linspace(t0, t1, n + 1)


def joint_traj_to_commands(
    traj,
    layout: "Rby1ActiveJointLayout",
    *,
    q_fn=None,
    hz: float = 30.0,
    dt: float | None = None,
    include_torso: bool = True,
    base_tol: float = 1e-3,
) -> list[dict[str, JointPositionCommand]]:
    """Convert a Drake joint-space trajectory into stiff joint commands.

    Args:
        traj:    A Drake trajectory exposing ``start_time()``, ``end_time()`` and
                 ``value(t)``. ``traj.value(t).flatten()`` should yield either a
                 23-D active vector ``[base, torso, right_arm, left_arm]`` or the
                 full plant config (pass ``q_fn`` to map it down).
        layout:  An ``rby1_planning.Rby1ActiveJointLayout`` (or anything exposing
                 ``base``/``torso``/``right_arm``/``left_arm`` slice attributes)
                 describing how to slice the 23-D active vector.
        q_fn:    Optional map from ``traj.value(t).flatten()`` to the 23-D active
                 vector (e.g. ``lambda v: v[pos_idxs_23]``). Identity if ``None``.
        hz:      Sampling rate (samples/sec) when ``dt`` is not given.
        dt:      Explicit timestep in seconds. Takes precedence over ``hz`` when
                 provided. This value becomes each waypoint's
                 ``JointPositionCommand.duration``.
        include_torso: Emit a ``"torso"`` command per waypoint when True.
        base_tol: Maximum allowed base displacement (in any base coordinate)
                  across the trajectory. The planning/stability model assumes a
                  stationary base and ``follow_joint_trajectory_stiff`` rejects
                  base commands, so a moving base is an error here.

    Returns:
        A list of waypoint dicts mapping ``"right"``/``"left"`` (and ``"torso"``)
        to ``JointPositionCommand(duration=dt, target_position=...)`` in radians.
        The first waypoint is the trajectory start; callers should treat it as
        the robot's current pose (e.g. pre-move to it before streaming).

    Raises:
        ValueError: if the base moves more than ``base_tol``.
    """
    step = dt if dt is not None else 1.0 / hz
    if step <= 0:
        raise ValueError(f"timestep must be positive, got {step}")

    ts = _sample_times(traj.start_time(), traj.end_time(), step)

    q23s = []
    for t in ts:
        v = np.asarray(traj.value(t)).flatten()
        q23 = np.asarray(q_fn(v) if q_fn is not None else v).flatten()
        q23s.append(q23)
    q23s = np.array(q23s)

    # Base-stationarity guard: base is never emitted, but a moving base means the
    # planned motion can't be reproduced by stiff joint control alone.
    base_disp = float(np.max(np.abs(q23s[:, layout.base] - q23s[0, layout.base])))
    if base_disp > base_tol:
        raise ValueError(
            f"Base moves by {base_disp:.4f} over the trajectory (tol={base_tol}); "
            "follow_joint_trajectory_stiff cannot drive the base. Use "
            "follow_base_trajectory for base motion, or replan with a fixed base."
        )

    commands: list[dict[str, JointPositionCommand]] = []
    for q23 in q23s:
        wp = {
            "right": JointPositionCommand(
                duration=step, target_position=np.array(q23[layout.right_arm])
            ),
            "left": JointPositionCommand(
                duration=step, target_position=np.array(q23[layout.left_arm])
            ),
        }
        if include_torso:
            wp["torso"] = JointPositionCommand(
                duration=step, target_position=np.array(q23[layout.torso])
            )
        commands.append(wp)
    return commands


def sample_trajectory(traj, idxs, *, hz: float = 30.0):
    """Sample a Drake trajectory into a plain ``(n, len(idxs))`` array + duration.

    The returned array is plain numpy, so unlike ``traj`` itself it reliably
    survives a fork boundary (a forked planning worker shipping it back over a
    pipe) or a pickle/cache round-trip (reloading an already-planned grid
    point). This isn't just conservatism: trajectories built from the RRT/
    shortcut path stage before TOPPRA retiming are ``FunctionHandleTrajectory``-
    backed (directly, or wrapped in a ``PathParameterizedTrajectory``), and
    ``pickle.dumps`` on either raises ``TypeError: cannot pickle
    'pydrake.trajectories.FunctionHandleTrajectory' object`` — confirmed by hand
    before writing this function. Other trajectory types in this codebase
    (plain ``PiecewisePolynomial``-backed ones) do happen to pickle, but relying
    on that would make correctness depend on which planning path produced a
    given trajectory.

    Args:
        traj: A Drake trajectory exposing ``start_time()``/``end_time()``/``value(t)``.
        idxs: Indices into ``traj.value(t).flatten()`` to keep per sample (e.g.
              ``Rby1ActiveJointLayout.plant_idxs``, for the 23-D active vector).
        hz:   Sampling rate (samples/sec).

    Returns:
        ``(q_samples, duration)`` — ``q_samples`` is ``(n, len(idxs))``, sampled
        evenly over ``[traj.start_time(), traj.end_time()]`` (endpoint included);
        ``duration`` is ``traj.end_time() - traj.start_time()``.
    """
    t0, t1 = traj.start_time(), traj.end_time()
    ts = _sample_times(t0, t1, 1.0 / hz)
    q_samples = np.array([np.asarray(traj.value(t)).flatten()[idxs] for t in ts])
    return q_samples, float(t1 - t0)


def sample_leg(name, traj, idxs, with_box, *, hz: float = 30.0):
    """One entry of a ``preview_legs``-ready leg list.

    ``with_box`` tags which of ``preview_legs``' viz diagrams (empty-handed vs.
    with-box) should render this leg — a plain ``(n, len(idxs))`` array carries
    no record of which plant/diagram it came from, unlike a live trajectory
    tied to a specific plant.
    """
    q_samples, duration = sample_trajectory(traj, idxs, hz=hz)
    return dict(name=name, q=q_samples, duration=duration, with_box=with_box)


def preview_legs(viz, legs, *, log=print):
    """Real-time meshcat replay of pre-sampled legs (see ``sample_leg``).

    For legs that may have crossed a fork or cache-reload boundary and so have
    no live Drake trajectory object to call ``.value(t)`` on — ``play_trajectory``
    needs exactly that and so cannot be used here. Playback pacing is derived
    from each leg's own ``duration``/sample-count (mirrors
    ``box_pickup_loop.ipynb``'s ``leg_viz``/``_publish`` convention) rather than
    an independent frame rate, so it stays correct regardless of the ``hz``
    ``sample_leg`` was built with.

    Announces each leg's name and expected duration via ``log`` before playing
    it — some legs (esp. lift, observed 45-155s) take long enough in real time
    to look indistinguishable from a hang without this.

    Args:
        viz:  ``dict[bool, (plant, plant_ctx, diagram_ctx, diagram, idxs)]`` —
              one already-built, already-meshcat-attached diagram per box state
              (``False``: empty-handed, for reach/home legs; ``True``: with-box,
              for lift legs), both writing the same meshcat paths so whichever
              published last poses the robot (the same trick
              ``box_pickup_loop.ipynb``'s "Meshcat Playback" section uses).
        legs: List of dicts as returned by ``sample_leg`` (or an equivalent
              cache reload) — ``name``, ``q`` (``(n, len(idxs))`` array),
              ``duration``, ``with_box``.
        log:  Callable taking a message string, called once per leg before
              playback starts. Defaults to ``print``; pass a no-op to silence.
    """
    for leg in legs:
        plant, plant_ctx, diagram_ctx, diagram, idxs = viz[leg["with_box"]]
        n = len(leg["q"])
        dt = leg["duration"] / max(n - 1, 1)
        log(f"Previewing '{leg['name']}' ({leg['duration']:.1f}s, {n} samples)...")
        for q23 in leg["q"]:
            q_full = plant.GetPositions(plant_ctx)
            q_full[idxs] = q23
            plant.SetPositions(plant_ctx, q_full)
            diagram.ForcedPublish(diagram_ctx)
            time.sleep(dt)


def pose_legs_start(viz, legs):
    """Pose the robot at the first configuration of ``legs`` and publish once.

    The zero-duration counterpart to ``preview_legs``, for callers that want the
    scene to show *this* plan without paying its playback time. Publishing matters
    as much as posing: a viewer left showing the previous plan's final pose is
    worse than showing nothing, because it looks like a valid answer to "what is
    about to run".
    """
    if not legs:
        return
    leg = legs[0]
    plant, plant_ctx, diagram_ctx, diagram, idxs = viz[leg["with_box"]]
    q_full = plant.GetPositions(plant_ctx)
    q_full[idxs] = np.asarray(leg["q"], dtype=float)[0]
    plant.SetPositions(plant_ctx, q_full)
    diagram.ForcedPublish(diagram_ctx)


def play_trajectory(plant, diagram, traj, *, q_fn=None, frame_rate=60.0):
    """Play a trajectory back in meshcat (no plot), suitable for repeated replays.

    Mirrors the realtime playback loop in ``rby1_planning.visualize_trajectory``
    but omits the matplotlib plot and static stability geometry, so it can be
    called in a confirm/replay loop without spawning a figure each time.

    Rendering happens via ``diagram.ForcedPublish``, which pushes to whatever
    ``MeshcatVisualizer`` is already attached to ``diagram`` -- so no meshcat
    handle is needed here.

    Args:
        plant:      The Drake plant containing the robot.
        diagram:    The RobotDiagram bound to ``plant`` and a MeshcatVisualizer.
        traj:       A Drake trajectory; see ``joint_traj_to_commands``.
        q_fn:       Optional map to the 23-D active vector (as above).
        frame_rate: Playback frame rate (Hz).
    """
    # Lazy import: only meshcat playback needs the planning layout, and only when
    # a plant is in hand. Keeps the module's top-level import cheap.
    from rby1_planning import Rby1ActiveJointLayout

    layout = Rby1ActiveJointLayout(plant)
    base_inst = plant.GetModelInstanceByName("base")
    torso_inst = plant.GetModelInstanceByName("torso")
    right_inst = plant.GetModelInstanceByName("right_arm")
    left_inst = plant.GetModelInstanceByName("left_arm")

    viz_ctx = diagram.CreateDefaultContext()
    plant_ctx = plant.GetMyContextFromRoot(viz_ctx)
    frame_delay = 1.0 / frame_rate

    for t in np.arange(traj.start_time(), traj.end_time(), frame_delay):
        v = np.asarray(traj.value(t)).flatten()
        q = np.asarray(q_fn(v) if q_fn is not None else v).flatten()
        plant.SetPositions(plant_ctx, base_inst, q[layout.base])
        plant.SetPositions(plant_ctx, torso_inst, q[layout.torso])
        plant.SetPositions(plant_ctx, right_inst, q[layout.right_arm])
        plant.SetPositions(plant_ctx, left_inst, q[layout.left_arm])
        diagram.ForcedPublish(viz_ctx)
        time.sleep(frame_delay)
