"""High-level motion planning for the RBY1 bimanual manipulator.

Two public planning functions:

    unconstrained_plan(plant, collision_checker, diagram, q_start,
                       right_target, left_target, ...)
        Joint-space reaching: IK across all 8 C-bundle GCPs → BiRRT → shortcut
        → optional trajopt → TOPPRA. Returns (full-plant trajectory, 23-D goal).

    constrained_plan(plant, collision_checker, diagram, q_start, mid_target, ...)
        Constrained motion that preserves the gripper-to-gripper transform fixed
        at q_start. Plans in the 14-D SE(3) × torso × ψ state space using the
        GCP derived from q_start's arms. Returns full-plant trajectory.

And a convenience factory:

    make_default_rby1_infrastructure(meshcat=None)
        Build a plant / collision checker / diagram from the bundled URDF for
        callers that don't already have one.
"""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
import itertools
from itertools import product as iproduct
from typing import Callable, Optional, Tuple

import numpy as np
from pydrake.all import (
    Box,
    BsplineBasis,
    BsplineTrajectory,
    CalcGridPointsOptions,
    CollisionCheckerParams,
    CollisionFilterDeclaration,
    CompositeTrajectory,
    CoulombFriction,
    FunctionHandleTrajectory,
    GeometrySet,
    LoadModelDirectives,
    MeshcatVisualizer,
    MeshcatVisualizerParams,
    Parser,
    PathParameterizedTrajectory,
    PiecewisePolynomial,
    ProcessModelDirectives,
    Rgba,
    RigidTransform,
    RobotDiagramBuilder,
    Role,
    RollPitchYaw,
    SceneGraphCollisionChecker,
    StackedTrajectory,
    Toppra,
)

import common
import rrt
import shortcut
from exps.timing_utils import mark, mark_discarded, record, stage_timer
from rby1_analytic_ik import Rby1IK, gcp_key_of, gcp_of_arm
from rby1_opt_ik import (
    DEFAULT_SUPPORT_POLYGON_INSET,
    MakeRby1Diagram,
    Rby1ProblemOptions,
    Rby1ProblemOptionsNew,
    _com_support_polygon_residuals,
    _stability_constraint_ub,
    check_com_stability,
    inset_support_polygon_xy,
    solve_ik,
    support_polygon_xyzs,
)


# Wall-clock budget for a single BiRRT search, in seconds. The vertex/iteration
# caps stay generous so a hard-but-solvable query keeps its full search budget;
# this only bounds how long a *hopeless* one may run before giving up. A timeout
# returns an empty path exactly as exhausting the caps does, so it surfaces
# through the existing "BiRRT failed to find a path" error.
BIRRT_TIMEOUT_S = 120.0


# Damping for the analytic IK's boundary-projection gradient, used by every
# Rby1IK instance trajopt evaluates constraints through.
#
# A sweep against the direct-reachability formulation measured 14% success at
# 0.0 and 100% across [1e-3, 1.0], with 0.1 the cheapest of the saturated
# values. rby1_opt_ik.Rby1ProblemOptions already defaults to 0.1;
# the constrained legs' trajopt did not, because its Rby1IK took the signature
# default of 0.0. The damping is inert wherever IK solves exactly -- the residual
# it scales is only computed on the boundary branch -- so this changes gradients
# only where they were previously undamped solves of a singular Jacobian.
TRAJOPT_BOUNDARY_DAMPING = 0.1

# SNOPT's QP-subproblem tolerance, held at the solver's own default rather than
# following the major optimality tolerance. See _snopt_options for the measurement.
SNOPT_MINOR_OPTIMALITY_TOL = 1e-6


# Whether the ``trajectories=`` output parameter carries the intermediate
# trajectories themselves, or merely records which stages produced one.
#
# Off, because nothing reads the trajectories. The only consumer of the captured
# stages is scripts/grid_provenance_report.py, and it reads the *key set* alone
# (`stages = set(rec.get("stages") or {})`, then `"trajopt" in stages`) to answer
# "was this leg optimised or is it a raw BiRRT fallback". The sampled arrays
# beside those keys are written to every point's `*_debug.pkl` and never read by
# anything.
#
# Producing them is not free:
#   * ``rrt_raw`` and ``rrt_shortcut`` do not exist until they are captured --
#     each is a fresh CompositeTrajectory fitted over every waypoint of the path,
#     built for the capture and nothing else.
#   * scripts/plan_grid._capture_stage_trajectories then
#     resamples up to four stages per leg at ``viz_hz``. On the constrained legs
#     the ``toppra`` stage is a FunctionHandleTrajectory whose every sample is an
#     analytic IK solve, so that resampling re-runs the IK mapping ~30x per
#     second of trajectory -- duplicating work the leg's own sample_leg entry has
#     already done, for a copy nothing opens.
#
# Set True to get the trajectories back for interactive debugging; the key set,
# and therefore the provenance report, is identical either way.
CAPTURE_STAGE_TRAJECTORIES = False


def _record_stage(trajectories, name, make_traj):
    """Record that stage ``name`` produced a trajectory.

    ``make_traj`` is a callable, not a trajectory, so that stages which have to
    be *built* for the capture are not built at all when capture is off.

    Stores None rather than omitting the key: the key is the payload, and
    ``_capture_stage_trajectories`` treats a None value as "ran, not captured".
    """
    if trajectories is None:
        return
    trajectories[name] = make_traj() if CAPTURE_STAGE_TRAJECTORIES else None


# ── Joint layout ──────────────────────────────────────────────────────────────

class Rby1ActiveJointLayout:
    """Named index slices for the 23 active DOFs [base, torso, right_arm, left_arm].

    Used internally to slice 23-D vectors and 14-D constrained states. Not part
    of the public API; callers thread the plant directly.

    Attributes:
        plant_idxs:  1-D int array — plant position indices for the 23 active DOFs.
        base:        slice(0, 3)
        torso:       slice(3, 9)
        right_arm:   slice(9, 16)
        left_arm:    slice(16, 23)
        psi_right:   int — IKFast free-parameter index within the 23-D vector (= 11).
        psi_left:    int — IKFast free-parameter index within the 23-D vector (= 18).
    """

    def __init__(self, plant):
        base_pi  = _pos_idxs(plant, plant.GetModelInstanceByName("base"))
        torso_pi = _pos_idxs(plant, plant.GetModelInstanceByName("torso"))
        right_pi = _pos_idxs(plant, plant.GetModelInstanceByName("right_arm"))
        left_pi  = _pos_idxs(plant, plant.GetModelInstanceByName("left_arm"))

        self.plant_idxs = np.array(base_pi + torso_pi + right_pi + left_pi)

        n0 = 0;  n1 = n0 + len(base_pi);  n2 = n1 + len(torso_pi)
        n3 = n2 + len(right_pi);           n4 = n3 + len(left_pi)

        self.base      = slice(n0, n1)
        self.torso     = slice(n1, n2)
        self.right_arm = slice(n2, n3)
        self.left_arm  = slice(n3, n4)
        self.psi_right = n2 + 2   # right_arm joint 2 within the 23-D vector
        self.psi_left  = n3 + 2


# ── Public API ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class HeldBox:
    """A box rigidly grasped by a gripper, modelled for collision-aware planning.

    The box is registered as collision (and, by default, visual) geometry on the
    arm's end-effector frame (``ee_left`` / ``ee_right``), which belongs to the
    ``left_arm`` / ``right_arm`` model instance. This matters: the collision
    checker only treats the arm/torso/base instances as the robot (not the
    grippers), so attaching the box to the *arm* ee frame is what makes the
    planner actually avoid collisions with the carried object.

    Geometry must be registered before ``plant.Finalize()``; use this with
    ``make_default_rby1_infrastructure(held_boxes=...)``.

    Attributes:
        hand:   ``"left"`` or ``"right"`` — which gripper holds the box.
        size:   ``(lx, ly, lz)`` full box dimensions in metres.
        offset: ``(x, y, z)`` translation of the box centre in the ee frame.
                The gripper points along -z, so a negative z places the box
                beyond the fingers. Default sits just past the fingertips.
        rpy:    ``(roll, pitch, yaw)`` orientation of the box in the ee frame.
        visual: If True, also add a translucent visual box for meshcat.
        color:  RGBA of the visual box.
        open_top: Build the box as a base slab + 4 walls instead of a solid
                block, so the checker can put things inside it.
        wall_thickness: ``t``, thickness of the base slab and of each wall.
        wall_height: Height of the walls standing on the base. The assembled
                box is ``t + wall_height`` tall, so this defaults to
                ``lz - t`` to make the total match ``size``'s lz. Set it
                explicitly only to model a box whose walls are shorter than
                its nominal height.
    """

    hand: str
    size: Tuple[float, float, float]
    offset: Tuple[float, float, float] = (0.0, 0.0, -0.13)
    rpy: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    visual: bool = True
    color: Tuple[float, float, float, float] = (0.2, 0.6, 1.0, 0.4)
    open_top: bool = False
    wall_thickness: float = 0.01
    wall_height: Optional[float] = None

    def __post_init__(self):
        if self.hand not in ("left", "right"):
            raise ValueError(f"HeldBox.hand must be 'left' or 'right', got {self.hand!r}")


def _add_held_box(plant, box: HeldBox):
    """Register ``box`` geometry on the arm ee frame. Must precede Finalize.

    Returns the ee ``RigidBody`` carrying the box, for later collision filtering.
    """
    arm = f"{box.hand}_arm"
    ee_link = f"ee_{box.hand}"
    ee_body = plant.GetBodyByName(ee_link, plant.GetModelInstanceByName(arm))
    X_BG = RigidTransform(RollPitchYaw(*box.rpy), np.asarray(box.offset, dtype=float))
    
    if not box.open_top:
        shape = Box(*box.size)
        plant.RegisterCollisionGeometry(
            ee_body, X_BG, shape, f"held_box_{box.hand}_collision", CoulombFriction(0.9, 0.8)
        )
        if box.visual:
            plant.RegisterVisualGeometry(
                ee_body, X_BG, shape, f"held_box_{box.hand}_visual",
                np.asarray(box.color, dtype=float),
            )
    else:
        lx, ly, lz = box.size
        t = box.wall_thickness
        lz_wall = box.wall_height if box.wall_height is not None else lz - t

        # 5 pieces: base slab + 4 walls standing on it. The walls' z centre is
        # derived from the base rather than hardcoded: previously it was a
        # literal -0.005, so `size`'s lz positioned only the base and the walls
        # floated independently of it. With lz=0.11/wall_height=0.08 that built
        # a 90 mm box out of a nominal 110 mm one -- 18 mm shorter than the
        # physical box (measured 4.25 in = 108 mm), in the unsafe direction:
        # the checker protected a rim below where the real rim is.
        z_base = -lz / 2 + t / 2            # base slab centre
        z_wall = z_base + t / 2 + lz_wall / 2   # walls stand on the base's top face
        pieces = [
            ("base", Box(lx, ly, t), RigidTransform([0, 0, z_base])),
            ("w1", Box(t, ly, lz_wall), RigidTransform([lx/2 - t/2, 0, z_wall])),
            ("w2", Box(t, ly, lz_wall), RigidTransform([-lx/2 + t/2, 0, z_wall])),
            ("w3", Box(lx - 2*t, t, lz_wall), RigidTransform([0, ly/2 - t/2, z_wall])),
            ("w4", Box(lx - 2*t, t, lz_wall), RigidTransform([0, -ly/2 + t/2, z_wall])),
        ]
        
        for name, shape, X_Local in pieces:
            X_World_Piece = X_BG @ X_Local
            plant.RegisterCollisionGeometry(
                ee_body, X_World_Piece, shape, f"held_box_{box.hand}_{name}_collision", CoulombFriction(0.9, 0.8)
            )
            if box.visual:
                plant.RegisterVisualGeometry(
                    ee_body, X_World_Piece, shape, f"held_box_{box.hand}_{name}_visual",
                    np.asarray(box.color, dtype=float),
                )
    return ee_body


def _held_box_neighbors(plant, box: HeldBox):
    """Bodies the held box is rigidly/legitimately in contact with, and so must
    not be collision-checked against.

    Both grippers are included, not just ``box.hand``'s: the grasp is bimanual,
    so the far hand's fingers are inside the box exactly as much as the near
    hand's (measured: ee_finger_2 penetrates the box by 8 mm on *both* sides
    throughout lift/place). Filtering only the declaring hand left the other one
    reporting a permanent 8 mm overlap.

    Deliberately does NOT include the other arm's links, the torso or the world:
    across lift/place those stay 36 mm (table), 62-100 mm (left arm) and 105 mm
    (torso) clear of the box, so they are real collisions to avoid and filtering
    them would mask a box crashing into the table.
    """
    h = box.hand
    arm_inst = plant.GetModelInstanceByName(f"{h}_arm")
    neighbors = []
    # Lower-arm / wrist links of the carrying arm, which the box rides on.
    ft = "FT_sensor_L" if h == "left" else "FT_sensor_R"
    for name in (f"link_{h}_arm_3", f"link_{h}_arm_4", f"link_{h}_arm_5",
                 f"link_{h}_arm_6", ft):
        neighbors.append(plant.GetBodyByName(name, arm_inst))
    # Both grippers' bodies -- the box is held by both hands.
    for hand in ("right", "left"):
        grip_inst = plant.GetModelInstanceByName(f"{hand}_gripper")
        for name in ("ee_body", "ee_finger_1", "ee_finger_2"):
            neighbors.append(plant.GetBodyByName(name, grip_inst))
    return neighbors


def _filter_held_box(collision_checker, plant, box: HeldBox, ee_body):
    """Filter the held box against the wrist/gripper bodies it rides on, so the
    rigidly-attached box never registers as a (self-)collision.

    Redundant with ``_filter_held_box_model`` (a SceneGraph filter is already
    visible to the checker), but kept so a checker built independently of
    ``make_default_rby1_infrastructure`` still gets the filtering.
    """
    for body in _held_box_neighbors(plant, box):
        collision_checker.SetCollisionFilteredBetween(ee_body.index(), body.index(), True)


def _filter_held_box_model(scene_graph, plant, box: HeldBox, ee_body):
    """Apply the same held-box filtering at SceneGraph *model* level.

    ``_filter_held_box`` alone is not enough: a CollisionChecker filter is a
    table owned by the checker object and is invisible to anything built off the
    plant directly -- in particular trajopt's
    ``MinimumDistanceLowerBoundConstraint``. That left the RRT (checker, box
    filtered) and trajopt (plant, box *not* filtered) disagreeing about the
    model, so every lift/place trajopt solve was infeasible from its first
    iterate: the constraint saw a finger permanently 8 mm inside the box, which
    no distance bound can satisfy, and SNOPT returned an "infeasibilities
    minimized" iterate instead of an optimized path.

    Model level (rather than per-context) so every context created afterwards
    inherits it -- including the ones the CollisionChecker caches internally,
    which is why this must run *before* the checker is constructed.

    Only ever reaches the lift/place infrastructure: this is called from the
    ``held_boxes`` branch, and in the reach/home plant ``ee_body`` carries no
    collision geometry at all (the box there is separate world obstacle
    geometry), so gripper-vs-box avoidance during the reach is untouched.
    """
    box_geoms = list(plant.GetCollisionGeometriesForBody(ee_body))
    if not box_geoms:
        return
    neighbor_geoms = []
    for body in _held_box_neighbors(plant, box):
        neighbor_geoms.extend(plant.GetCollisionGeometriesForBody(body))
    if not neighbor_geoms:
        return
    scene_graph.collision_filter_manager().Apply(
        CollisionFilterDeclaration().ExcludeBetween(
            GeometrySet(box_geoms), GeometrySet(neighbor_geoms))
    )


@dataclass(frozen=True)
class SceneBox:
    """A static box obstacle (e.g. a table) anchored in the world.

    Registered as collision (and visual) geometry on the plant's world body, so
    the collision checker treats it as a fixed environment obstacle the robot
    (and any held box) must avoid. Must be added before ``plant.Finalize()``;
    use via ``make_default_rby1_infrastructure(obstacles=...)``.

    Attributes:
        size:  ``(lx, ly, lz)`` full box dimensions in metres.
        xyz:   ``(x, y, z)`` world position of the box CENTRE. A box of height
               ``lz`` sits on the floor when ``z == lz / 2`` (top at ``lz``).
        rpy:   ``(roll, pitch, yaw)`` orientation in the world frame.
        name:  Geometry name; must be unique across obstacles.
        color: RGBA of the visual box.
    """

    size: Tuple[float, float, float]
    xyz: Tuple[float, float, float]
    rpy: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    name: str = "table"
    color: Tuple[float, float, float, float] = (0.82, 0.71, 0.55, 1.0)


def _add_obstacle(plant, obs: SceneBox):
    """Register a static box obstacle on the world body. Must precede Finalize."""
    X_WB = RigidTransform(RollPitchYaw(*obs.rpy), np.asarray(obs.xyz, dtype=float))
    shape = Box(*obs.size)
    plant.RegisterCollisionGeometry(
        plant.world_body(), X_WB, shape, f"{obs.name}_collision", CoulombFriction(0.9, 0.8)
    )
    plant.RegisterVisualGeometry(
        plant.world_body(), X_WB, shape, f"{obs.name}_visual",
        np.asarray(obs.color, dtype=float),
    )


def make_default_rby1_infrastructure(
    meshcat=None,
    *,
    held_boxes: "Optional[HeldBox | list[HeldBox]]" = None,
    obstacles: "Optional[SceneBox | list[SceneBox]]" = None,
    edge_step_size: float = 0.01,
    open_grippers: bool = False,
    gripper_collision: str = "spheres",
):
    """Build a default RBY1 plant / collision checker / diagram.

    Loads ``models/ruby/rby1_description_drake/add_rby1_sim_with_holonomic_base_actuators.dmd.yaml``
    via the package map, finalizes the plant, and builds a
    ``SceneGraphCollisionChecker`` over [base, torso, right_arm, left_arm].
    If ``meshcat`` is provided, attaches a ``MeshcatVisualizer``.

    Convenience for callers without their own Drake setup. Callers who already
    have a plant/collision_checker/diagram should skip this and pass theirs
    directly into unconstrained_plan / constrained_plan.

    Args:
        meshcat:        Optional Meshcat for visualization.
        held_boxes:     A ``HeldBox`` (or list) describing object(s) grasped by
                        the grippers. Each is added as collision geometry on the
                        corresponding arm ee frame so the planner avoids
                        collisions with the carried object, and filtered against
                        the wrist/gripper it rides on. Use this after a pick to
                        plan with the payload in hand.
        obstacles:      A ``SceneBox`` (or list) of static box obstacles (e.g. a
                        table) anchored in the world; the planner avoids them.
        edge_step_size: Collision-checker edge step size (m).
        gripper_collision:
                        Which collision model to use for the grippers.
                        ``"spheres"`` (default) is the historical
                        ``gripper_sim.urdf``: the ``ee_body`` palm is 5 spheres
                        (3x r=0.04 + 2x r=0.02) whose envelope is
                        0.140 x 0.080 x 0.080. ``"boxes"`` is
                        ``gripper_boxed.urdf``, identical except the palm is one
                        box sized to the ``EE_BODY.obj`` bounding box,
                        0.126 x 0.065 x 0.073. The fingers are boxes in both --
                        only the palm differs. The sphere blob bulges into the
                        volume a grasped box wall occupies, which is what put the
                        palm 4 mm inside the wall it was holding; the box does
                        not, while still strictly containing the visual mesh.
    """
    with record("infra.build", "rby1",
                held_boxes=bool(held_boxes), obstacles=bool(obstacles)):
        return _make_default_rby1_infrastructure(
            meshcat, held_boxes=held_boxes, obstacles=obstacles,
            edge_step_size=edge_step_size, open_grippers=open_grippers,
            gripper_collision=gripper_collision)


def _make_default_rby1_infrastructure(
    meshcat=None,
    *,
    held_boxes=None,
    obstacles=None,
    edge_step_size: float = 0.01,
    open_grippers: bool = False,
    gripper_collision: str = "spheres",
):
    if gripper_collision not in ("spheres", "boxes"):
        raise ValueError("make_default_rby1_infrastructure: gripper_collision must "
                         f"be 'spheres' or 'boxes', got {gripper_collision!r}")
    directives = ("add_rby1_sim_with_holonomic_base_actuators.dmd.yaml"
                  if gripper_collision == "spheres" else
                  "add_rby1_sim_with_holonomic_base_actuators_boxed.dmd.yaml")
    builder = RobotDiagramBuilder(time_step=0.0)
    plant = builder.plant()
    parser = Parser(plant)
    parser.package_map().AddPackageXml(os.path.join(common.RepoDir(), "package.xml"))
    ProcessModelDirectives(
        LoadModelDirectives(
            os.path.join(
                common.RepoDir(),
                "models/ruby/rby1_description_drake", directives,
            )
        ),
        parser,
    )

    # Register any grasped boxes on the arm ee frames (must precede Finalize).
    if held_boxes is None:
        boxes = []
    elif isinstance(held_boxes, HeldBox):
        boxes = [held_boxes]
    else:
        boxes = list(held_boxes)
    box_ee_bodies = [(box, _add_held_box(plant, box)) for box in boxes]

    # Register any static obstacles (e.g. a table) on the world body.
    if obstacles is None:
        obs_list = []
    elif isinstance(obstacles, SceneBox):
        obs_list = [obstacles]
    else:
        obs_list = list(obstacles)
    for obs in obs_list:
        _add_obstacle(plant, obs)

    if open_grippers:
        for hand in ("right", "left"):
            inst = plant.GetModelInstanceByName(f"{hand}_gripper")
            plant.GetJointByName("gripper_finger_1", inst).set_default_translation(-0.05)
            plant.GetJointByName("gripper_finger_2", inst).set_default_translation(0.05)

    if meshcat is not None:
        viz_params = MeshcatVisualizerParams()
        viz_params.delete_on_initialization_event = False
        viz_params.role = Role.kIllustration
        viz_params.prefix = "visual"
        MeshcatVisualizer.AddToBuilder(
            builder.builder(), builder.scene_graph(), meshcat, viz_params
        )
        
        col_viz_params = MeshcatVisualizerParams()
        col_viz_params.delete_on_initialization_event = False
        col_viz_params.role = Role.kProximity
        col_viz_params.prefix = "collision"
        col_viz_params.visible_by_default = False
        MeshcatVisualizer.AddToBuilder(
            builder.builder(), builder.scene_graph(), meshcat, col_viz_params
        )

    coll_params = CollisionCheckerParams()
    # The grippers and head MUST be here. A CollisionChecker only checks pairs
    # where at least one side belongs to a robot model instance; everything else
    # it treats as environment, and environment-vs-environment pairs are never
    # checked at all. With the grippers omitted (as they were), every
    # gripper-vs-world pair was invisible: CheckConfigCollisionFree returned True
    # for configurations with ee_finger_1/ee_finger_2 driven straight through the
    # box walls and the table. Measured on a reach_approach path the checker had
    # just certified as collision-free, the raw SceneGraph query found 58.8 mm of
    # finger-vs-world penetration.
    #
    # unconstrained_plan already built its own checker with the grippers included,
    # which is why the reach leg looked fine while plan_to_config and
    # constrained_plan -- both of which use *this* checker -- silently produced
    # colliding plans. That asymmetry is the single largest source of the invalid
    # trajectories this branch set out to fix.
    coll_params.robot_model_instances = [
        plant.GetModelInstanceByName("base"),
        plant.GetModelInstanceByName("torso"),
        plant.GetModelInstanceByName("right_arm"),
        plant.GetModelInstanceByName("left_arm"),
        plant.GetModelInstanceByName("right_gripper"),
        plant.GetModelInstanceByName("left_gripper"),
        plant.GetModelInstanceByName("head"),
    ]
    plant.Finalize()
    diagram = builder.Build()

    # Must precede SceneGraphCollisionChecker: the checker caches its contexts at
    # construction, so a model-level filter applied afterwards would not reach it.
    scene_graph = diagram.GetSubsystemByName("scene_graph")
    for box, ee_body in box_ee_bodies:
        _filter_held_box_model(scene_graph, plant, box, ee_body)

    coll_params.model = diagram
    coll_params.edge_step_size = edge_step_size
    collision_checker = SceneGraphCollisionChecker(coll_params)

    for box, ee_body in box_ee_bodies:
        _filter_held_box(collision_checker, plant, box, ee_body)

    diagram.ForcedPublish(diagram.CreateDefaultContext())

    return plant, collision_checker, diagram


def unwrap_slip_ring_joints(
    traj,
    plant,
    joint_names: Tuple[str, ...] = ("right_arm_4", "left_arm_4"),
    *,
    n_samples: int = 2000,
):
    """Return a copy of ``traj`` with continuous-rotation joints unwrapped.

    Arm joint index 4 (``right_arm_4`` / ``left_arm_4``) rides on a slip ring,
    so it can rotate continuously with no hard limit. A planned/retimed
    trajectory may contain a ~2π jump in this joint (e.g. when stitched or
    wrapped IK solutions land a full turn apart). This samples the trajectory,
    ``np.unwrap``s those joint columns (period 2π) so the joint takes the
    continuous short path instead, and returns a trajectory that adds the
    resulting piecewise-constant 2π offset to the original. Every other joint
    is passed through unchanged, and the timing is preserved.

    For an already-continuous trajectory this is a no-op. After unwrapping the
    joint may exceed the URDF's (artificial) ±limit, which is correct for the
    slip ring and fine for visualization (Drake does not clamp on SetPositions).

    Args:
        traj:        A position trajectory in full-plant coordinates.
        plant:       The (finalized) plant whose position layout ``traj`` matches.
        joint_names: Names of the continuous joints to unwrap.
        n_samples:   Samples used to detect 2π gaps along the path (default 2000).

    Returns:
        A ``FunctionHandleTrajectory`` over the same time span as ``traj``.
    """
    idxs = [int(plant.GetJointByName(n).position_start()) for n in joint_names]
    t0, t1 = traj.start_time(), traj.end_time()
    ts = np.linspace(t0, t1, n_samples)
    raw = np.array([[traj.value(t).flatten()[i] for i in idxs] for t in ts])  # (n, k)
    offsets = np.unwrap(raw, axis=0) - raw  # piecewise-constant multiples of 2π
    off_traj = PiecewisePolynomial.ZeroOrderHold(ts, offsets.T)
    nq = traj.rows()

    def _value(t):
        q = np.array(traj.value(t)).reshape(-1, 1)
        o = np.atleast_1d(off_traj.value(t).flatten())
        for j, i in enumerate(idxs):
            q[i, 0] += o[j]
        return q

    return FunctionHandleTrajectory(_value, nq, 1, t0, t1)


def visualize_trajectory(
    plant,
    diagram,
    meshcat,
    traj,
    *,
    title: str = "Trajectory",
    q_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    polytope_height: float = 1.6,
    support_polygon_inset: float = DEFAULT_SUPPORT_POLYGON_INSET,
    play: bool = True,
    n_samples: int = 1000,
    frame_rate: float = 60.0,
    plot_name: str = "visualize",
) -> None:
    """Play back a trajectory in meshcat and plot joint angles.

    Additionally renders static stability aids under meshcat path ``stability/``:
      - The base support polygon (yellow wireframe) extruded vertically by
        ``polytope_height`` metres. Assumes a stationary base — uses the base
        pose at t=0.
      - When ``support_polygon_inset > 0``, the conservative support polytope
        (orange wireframe) with the rear edge (back-caster edge) moved inward by
        ``support_polygon_inset`` metres — the boundary the planner's CoM
        stability constraint actually enforces.
      - The robot CoM trace over the full trajectory (magenta polyline). It
        should stay inside the (inset) polytope for the motion to remain
        statically stable.

    Args:
        plant:    The Drake plant containing the robot.
        diagram:  The RobotDiagram bound to ``plant`` and ``meshcat``.
        meshcat:  The pydrake.geometry.Meshcat instance to render into.
        traj:     A Drake trajectory; ``traj.value(t)`` produces either the
                  full plant configuration or a parameterised state (see
                  ``q_fn``).

    Keyword args:
        title:           Title for the matplotlib joint-angle plot.
        q_fn:            Maps a trajectory sample to a 23-D ``q_full`` vector
                         ordered [base(3), torso(6), right_arm(7), left_arm(7)].
                         Pass when ``traj`` lives in a parameterised space
                         (e.g. the 14-D lift state). If ``None``, samples are
                         used directly.
        polytope_height: Extrusion height of the support-polygon wireframe (m).
        support_polygon_inset: Inward inset (m) of the rear support edge for the
                         conservative polytope drawn alongside the nominal one;
                         0 to omit it.
        play:            If False, skip the realtime meshcat playback loop;
                         still produces the joint plot and the static curves.
        n_samples:       Number of evenly-spaced samples used for the joint
                         plot and the CoM curve.
        frame_rate:      Playback frame rate in Hz.
    """
    import matplotlib.pyplot as plt
    import time

    layout = Rby1ActiveJointLayout(plant)

    base_inst  = plant.GetModelInstanceByName("base")
    torso_inst = plant.GetModelInstanceByName("torso")
    right_inst = plant.GetModelInstanceByName("right_arm")
    left_inst  = plant.GetModelInstanceByName("left_arm")
    rg_inst    = plant.GetModelInstanceByName("right_gripper")
    lg_inst    = plant.GetModelInstanceByName("left_gripper")
    head_inst  = plant.GetModelInstanceByName("head")
    com_insts  = [base_inst, torso_inst, right_inst, left_inst,
                  rg_inst, lg_inst, head_inst]

    def _support_polytope_world(base_xyt, height, base_pts=None):
        tx, ty, theta = base_xyt
        c, s = np.cos(theta), np.sin(theta)
        if base_pts is None:
            base_pts = support_polygon_xyzs[:, :2]
        world_xy = (np.array([[c, -s], [s, c]]) @ base_pts.T).T + np.array([tx, ty])
        n = len(world_xy)
        bottom = np.column_stack([world_xy, np.zeros(n)])
        top    = np.column_stack([world_xy, np.full(n, height)])
        bottom_loop = np.asfortranarray(np.vstack([bottom, bottom[0]]).T)
        top_loop    = np.asfortranarray(np.vstack([top,    top[0]]).T)
        edge_starts = np.asfortranarray(bottom.T)
        edge_ends   = np.asfortranarray(top.T)
        return bottom_loop, top_loop, edge_starts, edge_ends

    com_eval_ctx = diagram.CreateDefaultContext()
    com_eval_plant_ctx = plant.GetMyContextFromRoot(com_eval_ctx)

    def _eval_com(q23):
        plant.SetPositions(com_eval_plant_ctx, base_inst,  q23[layout.base])
        plant.SetPositions(com_eval_plant_ctx, torso_inst, q23[layout.torso])
        plant.SetPositions(com_eval_plant_ctx, right_inst, q23[layout.right_arm])
        plant.SetPositions(com_eval_plant_ctx, left_inst,  q23[layout.left_arm])
        return plant.CalcCenterOfMassPositionInWorld(com_eval_plant_ctx, com_insts)

    # Sample the trajectory.
    ts = np.linspace(traj.start_time(), traj.end_time(), n_samples)
    pairs = []
    for t in ts:
        v = traj.value(t).flatten()
        q = q_fn(v) if q_fn is not None else v
        if q is not None:
            pairs.append((t, q))
    ts_plot = np.array([p[0] for p in pairs])
    qs_plot = np.array([p[1] for p in pairs])

    # Joint-angle plot.
    plt.figure(figsize=(20, 12))
    lines = plt.plot(ts_plot, qs_plot)
    if qs_plot.ndim == 2 and qs_plot.shape[1] == 23:
        labels = (["base_x", "base_y", "base_rz"]
                  + [f"torso_{i}" for i in range(6)]
                  + [f"right_arm_{i}" for i in range(7)]
                  + [f"left_arm_{i}" for i in range(7)])
        plt.legend(lines, labels, ncol=4, fontsize=9, loc="upper right")
    plt.title(title)
    plt.xlabel("t")
    plt.ylabel("joint angles (rad)")
    plt.tight_layout()
    import os
    out_path = os.path.join(os.path.dirname(__file__), "..", "scratch", f"{plot_name}_joint_angles.png")
    plt.savefig(out_path)
    print(f"Saved visualization plot to {os.path.abspath(out_path)}")

    # Static stability geometry.
    meshcat.Delete("stability")
    base_xyt0 = qs_plot[0][layout.base]
    bottom_loop, top_loop, edge_starts, edge_ends = _support_polytope_world(
        base_xyt0, polytope_height,
    )
    poly_rgba = Rgba(1.0, 0.85, 0.1, 0.9)
    meshcat.SetLine("stability/polytope/bottom", bottom_loop, 2.0, poly_rgba)
    meshcat.SetLine("stability/polytope/top",    top_loop,    2.0, poly_rgba)
    meshcat.SetLineSegments(
        "stability/polytope/edges", edge_starts, edge_ends, 2.0, poly_rgba,
    )

    # Conservative inset polytope (the boundary the stability constraint enforces).
    if support_polygon_inset > 0:
        c_bottom, c_top, c_starts, c_ends = _support_polytope_world(
            base_xyt0, polytope_height,
            base_pts=inset_support_polygon_xy(support_polygon_inset),
        )
        cons_rgba = Rgba(1.0, 0.45, 0.05, 0.9)
        meshcat.SetLine("stability/polytope_conservative/bottom", c_bottom, 2.0, cons_rgba)
        meshcat.SetLine("stability/polytope_conservative/top",    c_top,    2.0, cons_rgba)
        meshcat.SetLineSegments(
            "stability/polytope_conservative/edges", c_starts, c_ends, 2.0, cons_rgba,
        )
    com_pts = np.asfortranarray(np.array([_eval_com(q) for q in qs_plot]).T)
    meshcat.SetLine("stability/com_path", com_pts, 3.0, Rgba(1.0, 0.2, 0.8, 1.0))

    if not play:
        return

    # Realtime playback.
    viz_ctx     = diagram.CreateDefaultContext()
    plant_ctx   = plant.GetMyContextFromRoot(viz_ctx)
    frame_delay = 1.0 / frame_rate
    for t in np.arange(traj.start_time(), traj.end_time(), frame_delay):
        v = traj.value(t).flatten()
        q = q_fn(v) if q_fn is not None else v
        if q is None:
            continue
        plant.SetPositions(plant_ctx, base_inst,  q[layout.base])
        plant.SetPositions(plant_ctx, torso_inst, q[layout.torso])
        plant.SetPositions(plant_ctx, right_inst, q[layout.right_arm])
        plant.SetPositions(plant_ctx, left_inst,  q[layout.left_arm])
        diagram.ForcedPublish(viz_ctx)
        time.sleep(frame_delay)


def unconstrained_plan(
    plant,
    collision_checker,
    diagram,
    q_start: np.ndarray,
    right_target: RigidTransform,
    left_target: RigidTransform,
    *,
    rng_seed: int = 0,
    do_trajopt: bool = True,
    rrt_only: bool = False,
    allow_birrt_fallback: bool = False,
    n_trajopt_constraint_points: int = 50,
    min_distance_margin: float = 0.05,
    trajopt_min_distance: float = 0.001,
    clearance_margin: float = 0.0,
    trajopt_time_limit: float = 30.0,
    trajopt_optimality_tolerance: float = 1e-3,
    support_polygon_inset: float = DEFAULT_SUPPORT_POLYGON_INSET,
    ik_n_tries_per_gcp: int = 5,
    ik_n_candidates: int = 1,
    ik_options: Optional[Rby1ProblemOptions] = None,
    ik_diagram=None,
    rrt_options: Optional[rrt.RRTOptions] = None,
    toppra_min_points: int = 200,
    toppra_max_iter: int = 2,
    toppra_velocity_scale: float = 1.0,
    toppra_acceleration_scale: float = 1.0,
    timings: Optional[dict] = None,
    trajectories: Optional[dict] = None,
    diagnostics: Optional[dict] = None,
) -> Tuple[PathParameterizedTrajectory, np.ndarray]:
    """Plan an unconstrained reaching motion.

    Searches all 8 C-bundle GCPs for a stable, collision-free IK solution to the
    pair of end-effector targets, then runs BiRRT + shortcutting + optional
    KinematicTrajectoryOptimization + TOPPRA.

    Args:
        plant, collision_checker, diagram:  Drake infrastructure (see
            ``make_default_rby1_infrastructure`` for a convenience factory).
        q_start:       23-D starting configuration.
        right_target:  World-frame target for the right end-effector.
        left_target:   World-frame target for the left end-effector.

    Keyword args:
        rng_seed:                     RNG seed (default 0).
        do_trajopt:                   Run KinematicTrajectoryOptimization after
                                      shortcutting (default True).
        rrt_only:                     Return right after BiRRT + shortcutting
                                      with the unretimed 23-D path trajectory
                                      (no trajopt, no TOPPRA). Use when only
                                      plan feasibility matters (default False).
        n_trajopt_constraint_points:  Trajopt collision/stability constraint
                                      evaluations along the path (default 50).
        min_distance_margin:          Feeds MinimumDistanceLowerBoundConstraint's
                                      influence_distance_offset (default 0.05 m).
                                      NOT the hard floor, and not a penalty
                                      either: influence_distance decides which
                                      pairs are *included* in the constraint's
                                      softmin -- pairs further apart than it are
                                      dropped from the computation. It buys no
                                      clearance. Widening it only means more
                                      distant pairs participate; the distance
                                      actually enforced is
                                      ``trajopt_min_distance`` and nothing else.
                                      The name predates that distinction -- do
                                      not read it as "the minimum distance".
        trajopt_min_distance:         Hard minimum distance trajopt enforces along
                                      the path (default 0.001 m). Must stay at or
                                      below the clearance the *validated path*
                                      already achieves, or trajopt's initial guess
                                      is infeasible before the solver takes a step
                                      and it returns an "infeasibilities minimized"
                                      iterate that can be worse than the path it
                                      started from. Measured on this pipeline: the
                                      reach path grazes to 1.26 mm (ee_finger_2 vs
                                      a box wall) and the lift path to 0.33 mm
                                      arm-vs-arm, because the RRT's validity check
                                      is a penetration test with no margin. Hence
                                      1 mm, not the 20 mm this used to hardcode.
                                      Note what this means: **this bound is the
                                      only thing buying clearance**, so a leg left
                                      at 1 mm comes back at 1 mm. It is not
                                      supplemented by ``min_distance_margin``,
                                      which merely selects which pairs enter the
                                      softmin. To get clearance, pad the RRT
                                      (``clearance_margin``) so the guess has room,
                                      then raise this to spend it -- see
                                      plan_grid's leg profiles.
        clearance_margin:             Clearance the RRT/shortcutter must keep, in
                                      metres (default 0.0 = penetration-only, the
                                      historical behaviour). Implemented as
                                      collision-checker padding, so the existing
                                      boolean check means "clearance >= margin" at
                                      no extra per-check cost. The reach tolerates
                                      0.005 (measured: raises achieved path
                                      clearance from 1.26 mm to 7.59 mm). The
                                      constrained legs do not -- the two arms hold
                                      a 386 mm box between them and pass within
                                      0.33 mm by construction, so any padding at
                                      all made their BiRRT fail. Raise per leg,
                                      never globally.
        trajopt_time_limit:           SNOPT wall-clock limit in seconds (default 30).
        support_polygon_inset:        Inward CoM safety margin (m) on the rear
                                      support edge (back-caster edge) only: the
                                      CoM must keep this clearance from that edge,
                                      enforced in IK, RRT validity, and trajopt.
                                      0 = nominal polygon (default 0.0823).
        ik_n_tries_per_gcp:           IK retries per GCP (default 10).
        ik_n_candidates:              Feasible IK solutions to collect before
                                      picking the best (default 8). Ranking is by
                                      clearance, with arm joint-limit margin as a
                                      tiebreak -- see ``_search_ik``. This is not
                                      a luxury: at one measured pair of grasp
                                      targets, feasible solutions ranged from
                                      1.94 mm arm-vs-arm to >15 mm within the same
                                      GCP, and a 1.94 mm goal leaves trajopt's
                                      minimum-distance constraint infeasible at
                                      s=1, wasting the whole leg. 1 = first
                                      success, which ranks nothing.
        ik_options:                   Override the default Rby1ProblemOptions.
        ik_diagram:                   Pre-built diagram for IK (expensive to build);
                                      if None, MakeRby1Diagram() is called internally.
        rrt_options:                  Override RRT parameters.
        toppra_min_points:            Minimum TOPPRA grid points (default 200).
        toppra_max_iter:              TOPPRA gridpoint iteration cap (default 2).
        toppra_velocity_scale:        Multiplicative factor on the plant's natural
                                      joint velocity limits used by TOPPRA, clipped
                                      to (0, 1] (default 1.0 = full limits).
        toppra_acceleration_scale:    Multiplicative factor on the plant's natural
                                      joint acceleration limits used by TOPPRA,
                                      clipped to (0, 1] (default 1.0 = full limits).
        timings:                      Optional dict; if provided, per-stage
                                      wall-clock seconds are written into it under
                                      keys "ik", "rrt", "trajopt" (None when
                                      do_trajopt=False), and "toppra".

    Returns:
        (full-plant PathParameterizedTrajectory, 23-D goal configuration).
        With ``rrt_only=True`` the trajectory is instead the unretimed 23-D
        CompositeTrajectory of the shortcut path.

    Raises:
        RuntimeError: if IK fails across all 8 GCPs or TOPPRA returns an
            infinite-duration trajectory.
    """
    setup = _Setup.build(plant, collision_checker, diagram)
    rng = np.random.default_rng(rng_seed)

    ik_options = ik_options if ik_options is not None else _default_ik_options()
    ik_diagram = ik_diagram if ik_diagram is not None else MakeRby1Diagram()
    # The public inset is authoritative across IK, RRT validity, and trajopt.
    ik_options.new_formulation_options.support_polygon_inset = support_polygon_inset

    idx = 1
    g = _all_gcps()[idx]

    with stage_timer(timings, "ik"):
        q_goal = _search_ik(
            setup, right_target, left_target,
            gcp_pairs=[(g, g)],
            n_tries=ik_n_tries_per_gcp,
            ik_options=ik_options,
            ik_diagram=ik_diagram,
            rng=rng,
            inset=support_polygon_inset,
            n_candidates=ik_n_candidates,
        )
    if q_goal is None:
        raise RuntimeError("unconstrained_plan: IK failed across all C-bundle GCPs")

    if rrt_options is None:
        rrt_options = rrt.RRTOptions(
            step_size=0.1, check_size=0.001,
            max_vertices=10_000, max_iters=200_000,
            goal_sample_frequency=0.1, always_swap=False,
            timeout=BIRRT_TIMEOUT_S,
        )

    rrt_coll_params = CollisionCheckerParams()
    rrt_coll_params.model = diagram
    rrt_coll_params.robot_model_instances = [
        plant.GetModelInstanceByName("base"),
        plant.GetModelInstanceByName("torso"),
        plant.GetModelInstanceByName("right_arm"),
        plant.GetModelInstanceByName("left_arm"),
        plant.GetModelInstanceByName("right_gripper"),
        plant.GetModelInstanceByName("left_gripper"),
        # The head belongs here for the same reason the grippers do: a checker only
        # tests pairs involving an instance it was given, so omitting it made every
        # head-vs-world and head-vs-arm pair invisible on this leg alone, while the
        # main checker (make_default_rby1_infrastructure) and every verification path
        # did include it. The arms pass within ~29 mm of the head's deliberately
        # conservative sensor sphere at Q_READY, so this is a pair that really occurs.
        plant.GetModelInstanceByName("head"),
    ]
    rrt_coll_params.edge_step_size = collision_checker.edge_step_size()
    rrt_collision_checker = SceneGraphCollisionChecker(rrt_coll_params)

    # Filter collisions between grippers and the box walls so the IK goal is valid.
    # We must use GeometryId for SetCollisionFilteredBetween.
    try:
        sg = diagram.GetSubsystemByName("scene_graph")
        inspector = sg.model_inspector()
        for hand in ["right", "left"]:
            grip_inst = plant.GetModelInstanceByName(f"{hand}_gripper")
            grip_bodies = [plant.GetBodyByName(name, grip_inst) for name in ("ee_body", "ee_finger_1", "ee_finger_2")]
            grip_geoms = []
            for b in grip_bodies:
                grip_geoms.extend(plant.GetCollisionGeometriesForBody(b))
            
            for box_name in ["box_base", "box_w1", "box_w2", "box_w3", "box_w4"]:
                if plant.HasBodyNamed(box_name):
                    box_body = plant.GetBodyByName(box_name)
                    box_geoms = plant.GetCollisionGeometriesForBody(box_body)
                    for wg in box_geoms:
                        for gg in grip_geoms:
                            rrt_collision_checker.SetCollisionFilteredBetween(wg, gg, True)
    except Exception as e:
        print(f"Error setting up collision filters: {e}")

    setup_rrt = _Setup.build(plant, rrt_collision_checker, diagram)
    
    # Enforce strictly no base motion for downstream TrajOpt
    setup_rrt.q_lb[:3] = q_start[:3]
    setup_rrt.q_ub[:3] = q_start[:3]

    # Pad the planner's own checker so the path it returns keeps real clearance
    # rather than grazing; see _checker_padding. This checker is built inside this
    # function, so the padding cannot leak to the caller's.
    if clearance_margin:
        rrt_collision_checker.SetPaddingAllRobotRobotPairs(clearance_margin)
        rrt_collision_checker.SetPaddingAllRobotEnvironmentPairs(clearance_margin)

    plan_ctx = setup_rrt.plant.CreateDefaultContext()
    validity = _make_reach_validity(setup_rrt, plan_ctx, inset=support_polygon_inset)
    if not validity(q_start):
        raise RuntimeError("unconstrained_plan: q_start is not collision-free / stable")
    if not validity(q_goal):
        q_f = setup_rrt.q_full_default.copy()
        q_f[setup_rrt.pos_idxs_23] = q_goal
        diag = setup_rrt.diagram
        diag_ctx = diag.CreateDefaultContext()
        plant_ctx = setup_rrt.plant.GetMyContextFromRoot(diag_ctx)
        setup_rrt.plant.SetPositions(plant_ctx, q_f)
        sg = diag.GetSubsystemByName("scene_graph")
        sg_ctx = sg.GetMyContextFromRoot(diag_ctx)
        query = sg.get_query_output_port().Eval(sg_ctx)
        inspector = sg.model_inspector()
        print("COLLISIONS AT IK GOAL:")
        for p in query.ComputePointPairPenetration():
            nA = inspector.GetName(p.id_A)
            nB = inspector.GetName(p.id_B)
            print(f"  {nA} hits {nB} (depth: {p.depth})")
        raise RuntimeError("unconstrained_plan: IK goal is not collision-free / stable")

    with stage_timer(timings, "rrt"):
        path = _unconstrained_rrt_path(
            q_start, q_goal, setup_rrt, validity, rrt_options, rng_seed,
            trajectories=trajectories,
        )
    if not path:
        raise RuntimeError("unconstrained_plan: BiRRT failed to find a path")

    traj = _path_to_composite_traj(path)
    _record_stage(trajectories, "rrt", lambda: traj)

    if rrt_only:
        if timings is not None:
            timings["trajopt"] = None
            timings["toppra"] = None
        return traj, q_goal
    if do_trajopt:
        with stage_timer(timings, "trajopt"):
            shortcut_traj = traj
            traj, trajopt_success, trajopt_info = _reach_trajopt(
                path, traj, setup_rrt,
                min_dist_margin=min_distance_margin,
                n_constr_pts=n_trajopt_constraint_points,
                time_limit=trajopt_time_limit,
                optimality_tolerance=trajopt_optimality_tolerance,
                support_polygon_inset=support_polygon_inset,
                min_dist_lower_bound=trajopt_min_distance,
            )
            _record_stage(trajectories, "trajopt", lambda: traj)
            if diagnostics is not None:
                diagnostics["trajopt_info"] = trajopt_info
                diagnostics["trajopt_uncertified"] = trajopt_uncertified(trajopt_info)
                diagnostics["trajopt_fell_back"] = False
            if not trajopt_success:
                # trajopt's output on failure is an "infeasibilities minimized"
                # iterate: not collision-free, and empirically capable of being
                # far worse than the path it started from (grid points 14/15/16
                # were once cached with the gripper 90-120 mm inside the torso,
                # base and table this way). The shortcut path, by contrast, was
                # validated sample-by-sample by the RRT's validity checker, so
                # falling back to it keeps the collision/stability guarantees --
                # at the cost of a jerkier, longer motion.
                if not allow_birrt_fallback:
                    raise TrajoptRequired(
                        f"unconstrained_plan: trajopt failed ({trajopt_info}) and "
                        f"allow_birrt_fallback=False")
                print(f"unconstrained_plan: trajopt failed ({trajopt_info}); "
                      f"falling back to the validated shortcut path")
                _tag_fallback(diagnostics, f"solver failed ({trajopt_info})")
                traj = _fallback_traj(path, shortcut_traj)
                if trajectories is not None:
                    # Remove, do not set to None: _capture_stage_trajectories
                    # samples every captured stage and would crash on None.
                    trajectories.pop("trajopt", None)
            else:
                traj, accepted, accept_report = _accept_trajopt_or_fall_back(
                    traj, shortcut_traj, path, setup_rrt, label="unconstrained_plan",
                    support_polygon_inset=support_polygon_inset,
                    allow_birrt_fallback=allow_birrt_fallback)
                if not accepted:
                    _tag_fallback(diagnostics, "dense check rejected trajopt's output "
                                  f"({'; '.join(accept_report.failures)})")
                    if trajectories is not None:
                        trajectories.pop("trajopt", None)
    elif timings is not None:
        timings["trajopt"] = None

    with stage_timer(timings, "toppra"):
        retimed = _toppra_q23(
            traj, setup_rrt, toppra_min_points, toppra_max_iter,
            toppra_velocity_scale, toppra_acceleration_scale,
        )
        _record_stage(trajectories, "toppra", lambda: retimed)
    return retimed, q_goal


def plan_to_config(
    plant,
    collision_checker,
    diagram,
    q_start: np.ndarray,
    q_goal: np.ndarray,
    *,
    rng_seed: int = 0,
    do_trajopt: bool = True,
    allow_birrt_fallback: bool = False,
    n_trajopt_constraint_points: int = 50,
    min_distance_margin: float = 0.05,
    trajopt_min_distance: float = 0.001,
    clearance_margin: float = 0.0,
    clearance_margin_env_only: bool = False,
    trajopt_time_limit: float = 30.0,
    trajopt_optimality_tolerance: float = 1e-3,
    support_polygon_inset: float = DEFAULT_SUPPORT_POLYGON_INSET,
    rrt_options: Optional[rrt.RRTOptions] = None,
    toppra_min_points: int = 200,
    toppra_max_iter: int = 2,
    toppra_velocity_scale: float = 1.0,
    toppra_acceleration_scale: float = 1.0,
    pin_base: bool = True,
    try_straight_line: bool = False,
    timings: Optional[dict] = None,
    trajectories: Optional[dict] = None,
    diagnostics: Optional[dict] = None,
) -> PathParameterizedTrajectory:
    """Plan an unconstrained motion to a known goal configuration.

    ``try_straight_line`` first tests the C-space straight line between the two
    configurations and uses it when it is valid, falling back to the BiRRT otherwise;
    see ``_straight_line_path`` for when that is worth attempting. Off by default,
    because on a leg that has to travel around the box it can only waste a sweep of
    validity checks. Trajopt still runs on whichever path is chosen, so this changes
    how the leg's geometry is *found*, not whether it is optimised.

    Like ``unconstrained_plan`` but the 23-D goal config is supplied directly
    instead of being found by IK from end-effector targets (so no IK stage
    runs). Used for returning to a fixed pose such as a home configuration.
    Runs BiRRT + shortcutting + optional KinematicTrajectoryOptimization +
    TOPPRA.

    Args:
        plant, collision_checker, diagram:  Drake infrastructure.
        q_start:  23-D starting configuration.
        q_goal:   23-D goal configuration.

    Keyword args:
        pin_base: Clamp the base DOFs to the start/goal values for RRT sampling
            and trajopt (as ``unconstrained_plan`` does), so the planned motion
            is executable via arm/torso commands alone. Requires q_start and
            q_goal to agree in the base DOFs. Otherwise same kwargs as
            ``unconstrained_plan`` (minus the IK-specific ones). ``timings``,
            if provided, receives keys "rrt", "trajopt" (None when
            do_trajopt=False), and "toppra"; there is no "ik" key.

    Returns:
        Full-plant PathParameterizedTrajectory.

    Raises:
        RuntimeError: if q_start or q_goal is not collision-free / stable, if
            BiRRT fails, or if TOPPRA returns an infinite-duration trajectory.
    """
    setup = _Setup.build(plant, collision_checker, diagram)

    if pin_base:
        if np.any(np.abs(q_start[:3] - q_goal[:3]) > 1e-3):
            raise ValueError(
                "plan_to_config: pin_base=True but q_start and q_goal differ "
                f"in the base DOFs by {np.abs(q_start[:3] - q_goal[:3])}; "
                "pass pin_base=False to allow base motion."
            )
        setup.q_lb[:3] = np.minimum(q_start[:3], q_goal[:3])
        setup.q_ub[:3] = np.maximum(q_start[:3], q_goal[:3])

    if rrt_options is None:
        rrt_options = rrt.RRTOptions(
            step_size=0.2, check_size=0.005,
            max_vertices=10_000, max_iters=200_000,
            goal_sample_frequency=0.1, always_swap=False,
            timeout=BIRRT_TIMEOUT_S,
        )

    plan_ctx = setup.plant.CreateDefaultContext()
    validity = _make_reach_validity(setup, plan_ctx, inset=support_polygon_inset)
    # The endpoints are given, not solved for, so they are checked unpadded: a
    # standoff or grasp config handed in by the caller is allowed to sit closer to
    # the box than the *path* is required to stay. Padding applies to the search.
    if not validity(q_start):
        raise RuntimeError("plan_to_config: q_start is not collision-free / stable")
    if not validity(q_goal):
        raise RuntimeError("plan_to_config: q_goal is not collision-free / stable")

    straight_line = False
    with stage_timer(timings, "rrt"), _checker_padding(collision_checker,
                                                       clearance_margin,
                                                       clearance_margin_env_only):
        # Timed under "rrt" because it stands in for the BiRRT, so the stage total
        # stays comparable between runs with and without it.
        path = (_straight_line_path(q_start, q_goal, validity, rrt_options.check_size)
                if try_straight_line else None)
        straight_line = path is not None
        if not straight_line:
            path = _unconstrained_rrt_path(
                q_start, q_goal, setup, validity, rrt_options, rng_seed,
                trajectories=trajectories,
            )
    if straight_line:
        print("plan_to_config: straight-line C-space path is valid; BiRRT skipped")
    if diagnostics is not None:
        diagnostics["straight_line"] = straight_line
    if not path:
        raise RuntimeError("plan_to_config: BiRRT failed to find a path")

    traj = _path_to_composite_traj(path)
    _record_stage(trajectories, "rrt", lambda: traj)
    if do_trajopt:
        with stage_timer(timings, "trajopt"):
            shortcut_traj = traj
            traj, trajopt_success, trajopt_info = _reach_trajopt(
                path, traj, setup,
                min_dist_margin=min_distance_margin,
                n_constr_pts=n_trajopt_constraint_points,
                time_limit=trajopt_time_limit,
                optimality_tolerance=trajopt_optimality_tolerance,
                support_polygon_inset=support_polygon_inset,
                min_dist_lower_bound=trajopt_min_distance,
            )
            _record_stage(trajectories, "trajopt", lambda: traj)
            if diagnostics is not None:
                diagnostics["trajopt_info"] = trajopt_info
                diagnostics["trajopt_uncertified"] = trajopt_uncertified(trajopt_info)
                diagnostics["trajopt_fell_back"] = False
            if not trajopt_success:
                if not allow_birrt_fallback:
                    raise TrajoptRequired(
                        f"plan_to_config: trajopt failed ({trajopt_info}) and "
                        f"allow_birrt_fallback=False")
                print(f"plan_to_config: trajopt failed ({trajopt_info}); "
                      f"falling back to the validated shortcut path")
                _tag_fallback(diagnostics, f"solver failed ({trajopt_info})")
                traj = _fallback_traj(path, shortcut_traj)
                if trajectories is not None:
                    # Remove, do not set to None: _capture_stage_trajectories
                    # samples every captured stage and would crash on None.
                    trajectories.pop("trajopt", None)
            else:
                traj, accepted, accept_report = _accept_trajopt_or_fall_back(
                    traj, shortcut_traj, path, setup, label="plan_to_config",
                    support_polygon_inset=support_polygon_inset,
                    allow_birrt_fallback=allow_birrt_fallback)
                if not accepted:
                    _tag_fallback(diagnostics, "dense check rejected trajopt's output "
                                  f"({'; '.join(accept_report.failures)})")
                    if trajectories is not None:
                        trajectories.pop("trajopt", None)
    elif timings is not None:
        timings["trajopt"] = None

    with stage_timer(timings, "toppra"):
        retimed = _toppra_q23(
            traj, setup, toppra_min_points, toppra_max_iter,
            toppra_velocity_scale, toppra_acceleration_scale,
        )
        _record_stage(trajectories, "toppra", lambda: retimed)
    return retimed


def standoff_above(
    plant,
    collision_checker,
    diagram,
    q_base: np.ndarray,
    *,
    standoff: float = 0.08,
    rng_seed: int = 0,
    support_polygon_inset: float = DEFAULT_SUPPORT_POLYGON_INSET,
    ik_n_tries: int = 20,
    ik_n_candidates: int = 3,
    ik_options: Optional[Rby1ProblemOptions] = None,
    ik_diagram=None,
    timings: Optional[dict] = None,
) -> np.ndarray:
    """A validated 23-D config whose end-effectors sit ``standoff`` above ``q_base``'s.

    Both end-effector poses are translated straight up in world z and re-solved
    with full IK (torso free -- the descent between the two configs is planned by
    BiRRT over all 23 DOFs, so there is no reason to pin the torso, and pinning it
    would restrict how far the arms can lift). ``q_base`` is passed as the IK
    reference so the result is the solution nearest it in joint space, which keeps
    the standoff a small motion away rather than an unrelated posture.

    The returned config is checked collision-free, CoM-stable and inside joint
    limits -- ``_search_ik`` guarantees none of those on its own, and handing an
    unvalidated config to ``plan_to_config`` surfaces later as the opaque
    "q_goal is not collision-free / stable".

    Raises:
        RuntimeError: if no valid configuration exists ``standoff`` above
            ``q_base``. A smaller standoff is the thing to try.
    """
    setup = _Setup.build(plant, collision_checker, diagram)
    rng = np.random.default_rng(rng_seed)
    ik_options = ik_options if ik_options is not None else _default_ik_options()
    ik_diagram = ik_diagram if ik_diagram is not None else MakeRby1Diagram()
    ik_options.new_formulation_options.support_polygon_inset = support_polygon_inset

    q_base = np.asarray(q_base, dtype=float)
    fk_ctx = setup.plant.CreateDefaultContext()
    setup.set_q23_into(fk_ctx, q_base)
    w = setup.plant.world_frame()
    X_r = setup.plant.CalcRelativeTransform(
        fk_ctx, w, setup.plant.GetFrameByName("ee_right"))
    X_l = setup.plant.CalcRelativeTransform(
        fk_ctx, w, setup.plant.GetFrameByName("ee_left"))

    up = np.array([0.0, 0.0, float(standoff)])
    T_r = RigidTransform(X_r.rotation(), X_r.translation() + up)
    T_l = RigidTransform(X_l.rotation(), X_l.translation() + up)

    L = setup.layout
    gcp = gcp_of_arm(q_base[L.right_arm], "right")
    validity_ctx = setup.plant.CreateDefaultContext()
    validity = _make_reach_validity(setup, validity_ctx, inset=support_polygon_inset)

    with stage_timer(timings, "ik"):
        q_standoff = _search_ik(
            setup, T_r, T_l,
            gcp_pairs=[(gcp, gcp)],
            n_tries=ik_n_tries,
            ik_options=ik_options,
            ik_diagram=ik_diagram,
            rng=rng,
            inset=support_polygon_inset,
            n_candidates=ik_n_candidates,
            q_reference=q_base,
            q_verifier=validity,
        )
    if q_standoff is None:
        raise RuntimeError(
            f"standoff_above: no valid configuration {standoff*100:.1f} cm above "
            f"the given one (IK found nothing collision-free, stable and within "
            f"joint limits in the same C-bundle). Try a smaller standoff."
        )

    # Snap the base back to q_base's. ik_options.pin_base bounds the base to its
    # initial guess but IPOPT still returns it a few millimetres off, and
    # plan_to_config(pin_base=True) then refuses the pair outright ("q_start and
    # q_goal differ in the base DOFs by ..."). The base is immobile in this
    # pipeline, so the drift is numerical, not a plan -- but it must be validated
    # again after snapping rather than assumed harmless.
    q_standoff = q_standoff.copy()
    q_standoff[L.base] = q_base[L.base]
    if not validity(q_standoff):
        raise RuntimeError(
            f"standoff_above: the {standoff*100:.1f} cm standoff is invalid once "
            f"its base DOFs are snapped back to the start's (IPOPT had drifted "
            f"them by {np.max(np.abs(q_standoff[L.base] - q_base[L.base])):.2e}). "
            f"Try a smaller standoff."
        )
    return q_standoff


def grasp_posture_score(q23: np.ndarray) -> float:
    """Whole-arm left/right asymmetry of a 23-D configuration: lower is better.

    ``|| |q_right| - |q_left| ||`` over the 7 arm joints (active layout: right
    9:16, left 16:23). The absolute values make it invariant to the arms'
    mirrored sign conventions, so it reads as "how differently folded are the
    two arms" -- directly visible in a render.

    Why this feature. Re-derived 2026-08-07 against ground truth measured under
    the exact GCP labelling (`scripts/grasp_candidate_survey.py --variants none
    --probe --seeds 6`, 120 candidates over the 20 grid points, 55 plannable).
    The earlier justification -- 9/10 pairs over historical plans/grid_cache*
    runs -- was discarded twice over: those runs used the superseded labelling,
    *and* the survey's probe had been silently returning no ground truth at all
    since 868ccc9 (fixed in 8bc1274). So this is the first validation of this
    score on data that means anything.

    It survives that re-validation, and nothing beat it:

      selection policy (pick 1 of 6 drawn candidates)   plannable   mean lift len
        first draw, no ranking                            10/20         4.065
        lowest whole-arm asymmetry  (this)                16/20         3.184
        lowest distance-to-Q_READY                        12/20         3.200
        asymmetry + distance-to-Q_READY                   15/20         2.984
        oracle (any of the 6)                             20/20

    Pairwise agreement, same-point candidate pairs: 76.1% on plannability
    (155 pairs; next best 71.6%) and 73.2% on shorter lift (41 pairs).
    Spearman rho against the probe's 23-D lift length is +0.728, far ahead of
    joint-limit margin (+0.576), torso norm (+0.522) and distance-to-Q_READY
    (+0.496). Single-joint shoulder asymmetry is worse than a coin flip on both
    (52.9% / 41.5%, rho -0.271) -- do not substitute it.

    Distance-to-Q_READY edges this out on the lift-length pairs alone (78.0% vs
    73.2%, i.e. 32 vs 30 of 41 -- inside the noise) but is clearly worse at the
    decision that matters, picking a plannable grasp at all. A sum of the two
    also loses. Keep the single interpretable feature.
    """
    q23 = np.asarray(q23, dtype=float)
    d = np.abs(q23[9:16]) - np.abs(q23[16:23])
    return float(np.linalg.norm(d))


def grasp_and_standoff(
    plant,
    collision_checker,
    diagram,
    right_target: RigidTransform,
    left_target: RigidTransform,
    *,
    standoff: float = 0.08,
    rng_seed: int = 0,
    support_polygon_inset: float = DEFAULT_SUPPORT_POLYGON_INSET,
    ik_n_tries_per_gcp: int = 5,
    ik_n_candidates: int = 1,
    n_attempts: int = 4,
    gcp_index: int = 1,
    ik_options: Optional[Rby1ProblemOptions] = None,
    ik_diagram=None,
    timings: Optional[dict] = None,
    grasp_ik_options: Optional[Rby1ProblemOptions] = None,
    score_fn: Optional[Callable] = None,
    info: Optional[dict] = None,
    post_success_wall_time: Optional[float] = None,
    collect_budget_s: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Solve the grasp, then a standoff ``standoff`` metres above it.

    Returns ``(q_standoff, q_grasp)``.

    The ordering matters: solve the grasp first (it is the pose that actually has
    to be achieved), then derive the standoff from it with ``q_grasp`` as the IK
    reference. Solving the standoff first and descending does not work as well --
    IK picks whatever posture suits the pose it is given, and there is no reason
    that posture also suits the grasp 8 cm below.

    Retries with a fresh grasp IK draw if the standoff cannot be found, up to
    ``n_attempts``: the grasp has many IK solutions and not all of them have a
    valid configuration above them.

    ``gcp_index`` selects which of ``_all_gcps()`` the grasp is solved in, and it
    is more consequential than it looks: constrained_plan locks the lift and place
    to ``gcp_of_arm(q_start)``, so the branch chosen here *is* the branch the 14-D
    constrained search projects through for the rest of the pick. A grid point
    whose lift is infeasible in one branch can be perfectly feasible in another,
    which is why this is a parameter and not the hardcoded ``_all_gcps()[1]`` it
    used to be. Evidence: a collaborator's branch with a different GCP labelling
    (hence a different physical branch behind the same index) plans points 14 and
    16 -- both verified here against our own collision model, grippers and head
    included -- while failing 0, 6, 12, 18 and 19, which we plan. The two failure
    sets are near-complementary, and 4/10/11 fail in both.

    ``grasp_ik_options``, when given, is used for the *grasp* solve only; the
    standoff solve keeps ``ik_options``. They are separate because the standoff
    ranks by nearest-to-``q_grasp``, which a posture-cost pull would fight.
    ``score_fn`` ranks the grasp candidates (see ``_search_ik``); ``info``
    receives the grasp search's ``n_found``/``best_score``.
    """
    setup = _Setup.build(plant, collision_checker, diagram)
    rng = np.random.default_rng(rng_seed)
    ik_options = ik_options if ik_options is not None else _default_ik_options()
    ik_diagram = ik_diagram if ik_diagram is not None else MakeRby1Diagram()
    ik_options.new_formulation_options.support_polygon_inset = support_polygon_inset
    if grasp_ik_options is None:
        grasp_ik_options = ik_options
    else:
        grasp_ik_options.new_formulation_options.support_polygon_inset = (
            support_polygon_inset)
    gcps = _all_gcps()
    if not 0 <= gcp_index < len(gcps):
        raise ValueError(f"grasp_and_standoff: gcp_index {gcp_index} out of range "
                         f"[0, {len(gcps)})")
    g = gcps[gcp_index]

    validity_ctx = setup.plant.CreateDefaultContext()
    validity = _make_reach_validity(setup, validity_ctx, inset=support_polygon_inset)

    reasons = []
    with stage_timer(timings, "ik"):
        for attempt in range(max(1, n_attempts)):
            q_grasp = _search_ik(
                setup, right_target, left_target,
                gcp_pairs=[(g, g)],
                n_tries=ik_n_tries_per_gcp,
                ik_options=grasp_ik_options,
                ik_diagram=ik_diagram,
                rng=rng,
                inset=support_polygon_inset,
                n_candidates=ik_n_candidates,
                q_verifier=validity,
                score_fn=score_fn,
                info=info,
                post_success_wall_time=post_success_wall_time,
                collect_budget_s=collect_budget_s,
            )
            if q_grasp is None:
                reasons.append("IK found no valid solution at the grasp pose")
                continue
            try:
                q_standoff = standoff_above(
                    plant, collision_checker, diagram, q_grasp,
                    standoff=standoff, rng_seed=rng_seed + attempt,
                    support_polygon_inset=support_polygon_inset,
                    ik_options=ik_options, ik_diagram=ik_diagram,
                )
            except RuntimeError as e:
                reasons.append(str(e))
                continue
            return q_standoff, q_grasp

    raise RuntimeError(
        f"grasp_and_standoff: no valid grasp/standoff pair in "
        f"{max(1, n_attempts)} attempts. Reasons: " + "; ".join(sorted(set(reasons)))
    )


def constrained_plan(
    plant,
    collision_checker,
    diagram,
    q_start: np.ndarray,
    mid_target: RigidTransform,
    *,
    rng_seed: int = 0,
    do_trajopt: bool = False,
    rrt_only: bool = False,
    allow_birrt_fallback: bool = False,
    birrt_timeout: Optional[float] = None,
    trajopt_optimality_tolerance: Optional[float] = None,
    n_trajopt_constraint_points: int = 50,
    min_distance_margin: float = 0.05,
    trajopt_min_distance: float = 0.001,
    trajopt_pair_min_distance: Optional[float] = None,
    trajopt_pair_geometries: Tuple[str, str] = ("held_box", "table"),
    clearance_margin: float = 0.0,
    trajopt_time_limit: float = 30.0,
    trajopt_joint_space_energy_cost: bool = True,
    support_polygon_inset: float = DEFAULT_SUPPORT_POLYGON_INSET,
    mid_orientation_margin: Optional[float] = None,
    ik_n_tries: int = 20,
    ik_n_candidates: int = 1,
    ik_options: Optional[Rby1ProblemOptions] = None,
    ik_diagram=None,
    rrt_options: Optional[rrt.RRTOptions] = None,
    toppra_min_points: int = 200,
    toppra_max_iter: int = 2,
    toppra_velocity_scale: float = 1.0,
    toppra_acceleration_scale: float = 1.0,
    timings: Optional[dict] = None,
    trajectories: Optional[dict] = None,
    plot_name: str = "plan",   # accepted for compatibility; the debug plots are gone
    diagnostics: Optional[dict] = None,
    goal_ik_post_success_wall_time: Optional[float] = None,
    goal_ik_collect_budget_s: Optional[float] = None,
    warm_start_path: Optional[list] = None,
    warm_start_s_goal: Optional[np.ndarray] = None,
) -> PathParameterizedTrajectory:
    """Plan a constrained motion preserving the gripper-to-gripper transform.

    Derives the grasp's C-bundle GCP from ``q_start``'s arm joints (both arms
    must share the same GCP), captures the relative ``T_mid_right`` /
    ``T_mid_left`` transforms via FK, and plans in the 14-D state space
    [mid_xyz(3), mid_rpy(3), torso(6), ψ_right(1), ψ_left(1)] with IK restricted
    to the start GCP. Reachability at the goal is determined by solving IK for
    the corresponding gripper pair under that GCP.

    Args:
        plant, collision_checker, diagram:  Drake infrastructure.
        q_start:     23-D grasp configuration (both arms holding the object).
        mid_target:  Desired pose for the mid-gripper frame.

    Keyword args:
        rng_seed:                          RNG seed (default 0).
        do_trajopt:                        Run trajopt after shortcutting (default False).
        birrt_timeout:                     Override BIRRT_TIMEOUT_S for this call.
                                           Exists for cheap feasibility probes: with
                                           rrt_only=True and a short timeout this
                                           answers "is a lift plannable from this
                                           grasp at all" in seconds, which is the
                                           screen that picks a grasp (see
                                           plan_grid's grasp
                                           screening). A liftable grasp is found in
                                           ~6-22 s; an unliftable one runs to the
                                           full timeout and finds nothing.
        trajopt_optimality_tolerance:      Override _lift_trajopt's tolerance. Looser
                                           is safe here because feasibility is
                                           enforced separately by the dense check
                                           after the solve, so a looser tolerance can
                                           only cost path quality, never validity —
                                           and it buys wall clock on legs whose
                                           previous failure mode was literally the
                                           solver's time limit.
        rrt_only:                          Return right after constrained BiRRT +
                                           shortcutting with the unretimed 14-D
                                           state-space path trajectory — no
                                           joint-angle plot, no trajopt, no
                                           TOPPRA. Use when only plan feasibility
                                           matters (default False).
        n_trajopt_constraint_points:       Constraint evaluations along path (default 50).
        min_distance_margin:               Minimum collision-free distance (default 0.05 m).
        trajopt_time_limit:                SNOPT wall-clock limit (default 30 s).
        support_polygon_inset:             Inward CoM safety margin (m) on the rear
                                           support edge (back-caster edge) only,
                                           enforced in IK, RRT validity, and
                                           trajopt; 0 = nominal (default 0.0823).
        mid_orientation_margin:            If set (radians), lock the mid-gripper
                                           orientation to the start↔goal interval
                                           widened by this margin, keeping the
                                           carried object at a near-constant
                                           orientation (e.g. level) along the
                                           whole path. None (default) lets the
                                           orientation vary freely (±2π).
        ik_n_tries:                        IK retries for the goal pose (default 20).
        ik_n_candidates:                   Feasible goal IK solutions to collect
                                           before picking the one **nearest to
                                           q_start in joint space** (the
                                           ``q_reference`` ranking in _search_ik;
                                           an earlier version of this docstring
                                           wrongly claimed joint-limit-margin
                                           ranking). Default 1 = first success.
        goal_ik_post_success_wall_time / goal_ik_collect_budget_s:
                                           Optional bounds on the candidate
                                           refinement after the first feasible
                                           goal; see ``_search_ik``.
        ik_options, ik_diagram, rrt_options, toppra_min_points, toppra_max_iter,
        toppra_velocity_scale, toppra_acceleration_scale, timings:
                                           See ``unconstrained_plan``.

    Returns:
        Full-plant PathParameterizedTrajectory. With ``rrt_only=True``, instead
        the unretimed CompositeTrajectory over the 14-D planning state
        [mid_xyz(3), mid_rpy(3), torso(6), ψ_right(1), ψ_left(1)].

    Raises:
        RuntimeError: if the two arms have inconsistent GCPs at ``q_start``, if
            IK fails at the goal pose, if BiRRT fails, or if TOPPRA returns
            ``inf`` (typically a workspace singularity along the path).
    """
    setup = _Setup.build(plant, collision_checker, diagram)
    rng = np.random.default_rng(rng_seed)

    ik_options = ik_options if ik_options is not None else _default_ik_options()
    ik_diagram = ik_diagram if ik_diagram is not None else MakeRby1Diagram()
    # The public inset is authoritative across IK, RRT validity, and trajopt.
    ik_options.new_formulation_options.support_polygon_inset = support_polygon_inset

    lift = _make_lift_helpers(q_start, mid_target, setup)
    L = setup.layout

    plan_ctx = setup.plant.CreateDefaultContext()
    validity = _make_lift_validity(
        setup, plan_ctx, lift.state_to_q23, lift.gcp, inset=support_polygon_inset,
    )

    def _q_goal_verifier(q: np.ndarray) -> bool:
        """Accept a goal config only if its *exact analytic state projection* is
        valid -- collision-free, CoM-stable and inside joint limits.

        Checking joint limits alone (what this did before) let goals through that
        the RRT then immediately rejected, because the state the planner actually
        searches over is the projection, not the IK solution itself.
        """
        s = np.concatenate([
            mid_target.translation(),
            mid_target.rotation().ToRollPitchYaw().vector(),
            q[L.torso],
            [q[L.psi_right]],
            [q[L.psi_left]],
        ])
        return validity(s)

    # Solve IK at the goal mid-frame, restricted to the start GCP -- unless the
    # caller supplied a warm start (a path + goal state this exact problem
    # already produced at RRT level, e.g. by the grasp screen's carry probe),
    # in which case the goal IK is skipped outright.
    T_right_goal = mid_target @ lift.T_mid_right
    T_left_goal  = mid_target @ lift.T_mid_left
    if warm_start_s_goal is not None:
        s_goal = np.asarray(warm_start_s_goal, dtype=float)
        if timings is not None:
            # 0.0, never None: grid_timing_report sums these with Counter +=.
            timings["ik"] = 0.0
    else:
        with stage_timer(timings, "ik"):
            q_goal = _search_ik(
                setup, T_right_goal, T_left_goal,
                gcp_pairs=[(lift.gcp, lift.gcp)],
                n_tries=ik_n_tries,
                ik_options=ik_options,
                ik_diagram=ik_diagram,
                rng=rng,
                inset=support_polygon_inset,
                n_candidates=ik_n_candidates,
                q_reference=q_start,
                q_verifier=_q_goal_verifier,
                # Optional bounds on the post-first-success refinement solves.
                # The pipeline passes None: budgeting these was measured to
                # trade goal quality for IK time and rejected -- see
                # LIFT_GOAL_IK_* in scripts/plan_grid.py.
                post_success_wall_time=goal_ik_post_success_wall_time,
                collect_budget_s=goal_ik_collect_budget_s,
        )
        if q_goal is None:
            raise RuntimeError(
                "constrained_plan: IK failed at mid_target under the start GCP "
                "(goal is unreachable in the same C-bundle as q_start)."
            )

        s_goal = np.concatenate([
            mid_target.translation(),
            mid_target.rotation().ToRollPitchYaw().vector(),
            q_goal[L.torso],
            [q_goal[L.psi_right]],
            [q_goal[L.psi_left]],
        ])
    if lift.state_to_q23(s_goal) is None:
        raise RuntimeError("constrained_plan: goal state failed IK back-computation")

    s_lb, s_ub = _lift_state_bounds(lift.s_start, s_goal, setup,
                                    rpy_margin=mid_orientation_margin)

    if rrt_options is None:
        # The constrained legs are the ones that fail: on the full grid, points 4
        # and 10 both lost to "constrained BiRRT failed to find a path" on all
        # three seeds. They were exhausting these caps rather than timing out
        # (BIRRT_TIMEOUT_S was declared but never wired, so timeout was inf), which
        # means the search was budget-limited and a bigger budget is the fix.
        #
        # Raising caps used to be a dangerous kind of tuning -- it is what took the
        # old pipeline to a false 20/20 -- but that danger came from accepting
        # whatever the search returned. Every leg is now densely verified before it
        # can be cached, so a larger budget can only produce a slower plan, never
        # an invalid one. The wall-clock timeout bounds the downside: search harder
        # *within* 120 s, rather than search forever.
        rrt_options = rrt.RRTOptions(
            step_size=0.1, check_size=0.005,
            # max_iters raised with the vertex cap because it, not the cap, is what
            # actually binds: of 91 constrained invocations in one grid run, 13
            # exhausted max_iters=500_000 and 0 ever reached max_vertices. One
            # invocation spent 500k iterations building 40 vertices -- >99.99% of
            # iterations added nothing, because a sampled state is only usable if both
            # arms' analytic IK lands on the locked GCP branch.
            max_vertices=60_000, max_iters=2_000_000,
            goal_sample_frequency=0.1, always_swap=False,
            timeout=BIRRT_TIMEOUT_S if birrt_timeout is None else birrt_timeout,
        )

    if not validity(lift.s_start):
        raise RuntimeError("constrained_plan: start state is not collision-free / stable")
    if not validity(s_goal):
        raise RuntimeError("constrained_plan: goal state is not collision-free / stable")

    def edge_validator(s1: np.ndarray, s2: np.ndarray) -> bool:
        q1 = lift.state_to_q23(s1)
        q2 = lift.state_to_q23(s2)
        if q1 is None or q2 is None:
            return False
        return np.max(np.abs(q1 - q2)) < 0.2

    if warm_start_path is not None:
        # Reuse a path this problem already produced at RRT level. The start
        # may differ from lift.s_start at solver-tolerance scale (the donor's
        # q_start passed through a trajopt whose endpoint equality is only
        # ~1e-6-exact), so rebase it and re-validate everything the BiRRT
        # would have guaranteed: every waypoint against this call's validity
        # (this call's manifold -- the T_mid transforms differ at the same
        # tolerance scale) and the rebased first edge. Milliseconds, and the
        # dense verify gate downstream is unchanged. Any rejection raises
        # WarmStartRejected, which callers treat as "do the full replan".
        path = [np.asarray(p, dtype=float) for p in warm_start_path]
        if len(path) < 2:
            raise WarmStartRejected("warm-start path has fewer than 2 waypoints")
        if not np.allclose(path[0], lift.s_start, atol=1e-3):
            raise WarmStartRejected(
                f"warm-start path starts {np.max(np.abs(path[0] - lift.s_start)):.2e} "
                f"away from this problem's start state")
        if not np.allclose(path[-1], s_goal, atol=1e-3):
            raise WarmStartRejected("warm-start path does not end at s_goal")
        path[0] = lift.s_start.copy()
        path[-1] = s_goal.copy()
        if not all(validity(p) for p in path):
            raise WarmStartRejected("a warm-start waypoint fails this problem's validity")
        if not edge_validator(path[0], path[1]):
            raise WarmStartRejected("the rebased first edge fails the edge validator")
        if timings is not None:
            timings["rrt"] = 0.0
    else:
        with stage_timer(timings, "rrt"), _checker_padding(collision_checker,
                                                           clearance_margin):
            path = _constrained_rrt_path(
                lift.s_start, s_goal, s_lb, s_ub, validity, rrt_options, rng_seed,
                edge_validator=edge_validator, trajectories=trajectories,
            )
        if not path:
            raise RuntimeError("constrained_plan: constrained BiRRT failed to find a path")

    if diagnostics is not None:
        # Always exported: the RRT-level product of this call, as plain arrays
        # (picklable across the probe fork), so a later call can warm-start
        # from it instead of re-solving the same goal IK + BiRRT.
        diagnostics["rrt_path"] = [np.asarray(p, dtype=float) for p in path]
        diagnostics["s_goal"] = np.asarray(s_goal, dtype=float)

    traj_14 = _path_to_composite_traj(path)
    if rrt_only:
        if timings is not None:
            timings["trajopt"] = None
            timings["toppra"] = None
        return traj_14
    _record_stage(trajectories, "rrt", lambda: traj_14)

    # The per-stage joint-angle debug plots that lived here are gone: each call
    # sampled the trajectory 1000 times through the analytic IK (~seconds,
    # charged to the trajopt/toppra stage timers), every concurrent
    # constrained_plan wrote the SAME two PNG paths (a live file race during the
    # grasp fan-out), and the "toppra" plot sliced a full-plant trajectory with
    # the 14-D state accessor, so it was dimensionally wrong anyway. The cached
    # debug pickles' stage trajectories are the supported way to inspect a leg.

    if do_trajopt:
        with stage_timer(timings, "trajopt"):
            shortcut_traj_14 = traj_14
            traj_14, trajopt_success, trajopt_info = _lift_trajopt(
                path, traj_14, setup,
                lift.state_to_lumped_for_trajopt, lift.gcp, s_lb, s_ub,
                min_dist_margin=min_distance_margin,
                n_constr_pts=n_trajopt_constraint_points,
                time_limit=trajopt_time_limit,
                support_polygon_inset=support_polygon_inset,
                min_dist_lower_bound=trajopt_min_distance,
                pair_min_distance=trajopt_pair_min_distance,
                pair_geometries=trajopt_pair_geometries,
                joint_space_energy_cost=trajopt_joint_space_energy_cost,
                **({} if trajopt_optimality_tolerance is None
                   else dict(optimality_tolerance=trajopt_optimality_tolerance)),
            )
            if diagnostics is not None:
                diagnostics["trajopt_info"] = trajopt_info
                diagnostics["trajopt_uncertified"] = trajopt_uncertified(trajopt_info)
                diagnostics["trajopt_fell_back"] = False
            # Deliberately NOT recorded into ``trajectories`` yet -- only once the
            # solve is known good, below. Recording it here made a leg that went on to
            # raise TrajoptRequired still ship a captured "trajopt" stage, so the
            # provenance report called the actually-failed legs optimised.
            if not trajopt_success:
                if not allow_birrt_fallback:
                    raise TrajoptRequired(
                        f"constrained_plan: trajopt failed ({trajopt_info}) and "
                        f"allow_birrt_fallback=False, so the raw BiRRT+shortcut path "
                        f"is not an acceptable result for this leg")
                print(f"constrained_plan: trajopt failed ({trajopt_info}); "
                      f"falling back to the validated shortcut path")
                _tag_fallback(diagnostics, f"solver failed ({trajopt_info})")
                traj_14 = _fallback_traj(path, shortcut_traj_14)
                if trajectories is not None:
                    # Remove, do not set to None: _capture_stage_trajectories
                    # samples every captured stage and would crash on None.
                    trajectories.pop("trajopt", None)
            else:
                traj_14, accepted, accept_report = _accept_trajopt_or_fall_back(
                    traj_14, shortcut_traj_14, path, setup,
                    label="constrained_plan", q_fn=lift.state_to_q23,
                    support_polygon_inset=support_polygon_inset,
                    allow_birrt_fallback=allow_birrt_fallback)
                if accepted:
                    _record_stage(trajectories, "trajopt", lambda: traj_14)
                else:
                    _tag_fallback(diagnostics, "dense check rejected trajopt's output "
                                  f"({'; '.join(accept_report.failures)})")
                if not accepted and trajectories is not None:
                    # Mirrors the not-trajopt_success branch above. Without this the
                    # captured stage said "trajopt" for a leg that shipped the
                    # fallback, so a dense-check rejection was indistinguishable from
                    # an accepted solve in the cached debug records -- which is why
                    # the 9-of-15 fallback count measured from them is a lower bound.
                    trajectories.pop("trajopt", None)
    elif timings is not None:
        timings["trajopt"] = None

    with stage_timer(timings, "toppra"):
        retimed = _toppra_lift(
            traj_14, setup, lift.lift_ik, lift.state_to_lumped, lift.gcp,
            toppra_min_points, toppra_max_iter,
            toppra_velocity_scale, toppra_acceleration_scale,
        )
        _record_stage(trajectories, "toppra", lambda: retimed)
    return retimed


# ── Internal infrastructure ───────────────────────────────────────────────────

@dataclass
class _Setup:
    """Per-call bundle: plant + checker + diagram + derived layout/bounds."""
    plant: object
    collision_checker: object
    diagram: object
    layout: Rby1ActiveJointLayout
    pos_idxs_23: np.ndarray
    q_lb: np.ndarray
    q_ub: np.ndarray
    q_full_default: np.ndarray
    base_inst: object
    torso_inst: object
    right_inst: object
    left_inst: object

    @classmethod
    def build(cls, plant, collision_checker, diagram) -> "_Setup":
        layout = Rby1ActiveJointLayout(plant)
        pos_idxs = layout.plant_idxs
        q_lb = plant.GetPositionLowerLimits()[pos_idxs]
        q_ub = plant.GetPositionUpperLimits()[pos_idxs]
        # Base x/y are unbounded; clamp to finite fallback for uniform sampling.
        q_lb = np.where(np.isfinite(q_lb), q_lb, -2 * np.pi)
        q_ub = np.where(np.isfinite(q_ub), q_ub,  2 * np.pi)
        return cls(
            plant=plant,
            collision_checker=collision_checker,
            diagram=diagram,
            layout=layout,
            pos_idxs_23=pos_idxs,
            q_lb=q_lb,
            q_ub=q_ub,
            q_full_default=plant.GetDefaultPositions().copy(),
            base_inst=plant.GetModelInstanceByName("base"),
            torso_inst=plant.GetModelInstanceByName("torso"),
            right_inst=plant.GetModelInstanceByName("right_arm"),
            left_inst=plant.GetModelInstanceByName("left_arm"),
        )

    def set_q23_into(self, plant_ctx, q23):
        L = self.layout
        self.plant.SetPositions(plant_ctx, self.base_inst,  q23[L.base])
        self.plant.SetPositions(plant_ctx, self.torso_inst, q23[L.torso])
        self.plant.SetPositions(plant_ctx, self.right_inst, q23[L.right_arm])
        self.plant.SetPositions(plant_ctx, self.left_inst,  q23[L.left_arm])

    def embed_q23(self, q23):
        q = self.q_full_default.copy()
        q[self.pos_idxs_23] = q23
        return q


# ── IK helpers ────────────────────────────────────────────────────────────────

def _pos_idxs(plant, inst):
    out = []
    for jidx in plant.GetJointIndices(inst):
        j = plant.get_joint(jidx)
        if j.num_positions() > 0:
            out += list(range(j.position_start(), j.position_start() + j.num_positions()))
    return out


def _default_ik_options() -> Rby1ProblemOptions:
    """Direct-reachability constraint with no manipulability barrier (neither cost
    nor constraint), flex + stability constraints on, no_collisions=True,
    boundary_fallback=True with the tuned boundary_damping_lambda (0.1 default) so
    out-of-reach requests get smooth, damped gradients, IPOPT with loose acceptable
    tols. Mirrors the shipped Rby1ProblemOptions defaults for the reachability terms."""
    return Rby1ProblemOptions(
        solver="IPOPT",
        new_formulation_options=Rby1ProblemOptionsNew(
            impose_direct_reachability_constraint=True,
            ik_stepback=1e-6,
            impose_flex_constraints=True,
            impose_stability_constraints=True,
        ),
        impose_joint_centering_cost=False,
        no_collisions=True,
        acceptable_constr_viol_tol=1e-3,
        acceptable_tol=1e-4,
        acceptable_iter=1,
        max_wall_time=10.0,
        pin_base=True,
        pin_torso=False,
        influence_distance_offset=1e-4,
        boundary_fallback=True,
    )


# How far out to look for the closest geometry pair when scoring an IK candidate.
# Anything beyond this is equally "clear" for ranking purposes, and a bounded
# search is much cheaper than an unbounded one.
_IK_CLEARANCE_SEARCH_M = 0.05

# Clearance differences below this are treated as ties, and broken on joint-limit
# margin instead. Set at 1 mm: the measured spread between IK solutions at one
# grasp was 1.94 mm vs >15 mm, so real differences are far larger than this, while
# sub-millimetre differences are inside the geometry's own fidelity.
_IK_CLEARANCE_TIE_M = 0.001


def _all_gcps() -> list:
    """The 8 GCP branches, ordered (wrist, elbow, shoulder) with +1 first.

    The index is what ``--gcp-candidates`` and ``gcp_index`` name, so it must
    stay pinned to the branch it has always meant. Correspondence with the
    branchless split solvers' entry points
    (``Solve{Left,Right}Arm_{Wp,Wm}_{Ep,Em}_{Sp,Sm}``, +1 == P == that axis'
    template branch 0):

        0 (+1,+1,+1) Wp_Ep_Sp      4 (-1,+1,+1) Wm_Ep_Sp
        1 (+1,+1,-1) Wp_Ep_Sm      5 (-1,+1,-1) Wm_Ep_Sm
        2 (+1,-1,+1) Wp_Em_Sp      6 (-1,-1,+1) Wm_Em_Sp
        3 (+1,-1,-1) Wp_Em_Sm      7 (-1,-1,-1) Wm_Em_Sm

    Index 1 -- Wp_Ep_Sm -- is the branch every grid point plans in, verified
    directly against all 20 shipped grasps under the exact labelling.
    """
    return [np.array(g, dtype=np.int8) for g in iproduct([1, -1], repeat=3)]


def _random_q_in_gcp(gcp, setup: _Setup, rng, max_samples: int = 2000) -> Optional[np.ndarray]:
    """Rejection-sample a 23-D config whose arm GCPs both match ``gcp``.

    Returns None after ``max_samples`` misses: some GCPs are unreachable under
    the joint limits (reachable ones match within ~20 samples), and an
    unbounded loop would spin forever on them.
    """
    L = setup.layout
    for _ in range(max_samples):
        q = rng.uniform(setup.q_lb, setup.q_ub)
        q[L.base] = 0.0
        if (np.array_equal(gcp_of_arm(q[L.right_arm], "right"), gcp) and
                np.array_equal(gcp_of_arm(q[L.left_arm], "left"), gcp)):
            return q
    return None


def _search_ik(
    setup: _Setup,
    right_target: RigidTransform,
    left_target: RigidTransform,
    *,
    gcp_pairs: list,
    n_tries: int,
    ik_options: Rby1ProblemOptions,
    ik_diagram,
    rng,
    inset: float = 0.0,
    n_candidates: int = 1,
    q_reference: Optional[np.ndarray] = None,
    q_verifier: Optional[Callable] = None,
    min_clearance: float = 0.0,
    prefer_clearance: bool = False,
    score_fn: Optional[Callable] = None,
    info: Optional[dict] = None,
    post_success_wall_time: Optional[float] = None,
    collect_budget_s: Optional[float] = None,
) -> Optional[np.ndarray]:
    """Iterate over (right_gcp, left_gcp) pairs trying up to ``n_tries`` random
    initial configs each. Collects feasible (collision-free, CoM-stable with the
    given inward ``inset`` margin) solutions until ``n_candidates`` are found or
    all attempts are exhausted.

    If ``q_reference`` is provided, the first IK attempt uses it as the initial
    guess (biasing the local optimizer toward a nearby solution), and the best
    candidate is the one with the smallest joint-space distance to q_reference.
    This prevents large joint sweeps when q_reference is the start configuration.

    Otherwise candidates are ranked by minimum arm joint-limit margin.

    ``prefer_clearance=True`` instead ranks by clearance (largest minimum signed
    distance), joint-limit margin breaking ties. **This is off by default, and the
    reason is worth recording.** Clearance among feasible IK solutions at one
    measured pair of grasp targets ranged from 1.94 mm arm-vs-arm
    (link_left_arm_5 vs link_right_arm_3) up to >15 mm within the *same* GCP, so
    ranking on it looks like free safety. It is not: the grasp seeds the whole
    constrained chain, and the clearance-optimal grasp put the torso at
    [-0.036, -0.204, 1.306, 0.569, 0.379, 0.181], from which ``constrained_plan``
    could not reach T_W_MID_ABOVE at all -- BiRRT failed outright -- while the
    1.94 mm first-feasible grasp (torso [-0.014, -0.578, 1.410, 1.044, -0.229,
    0.606]) lifted fine. Same C-bundle in both; the torso posture is what
    differs. Greedily maximising grasp clearance therefore trades a working
    pipeline for a safer first leg. Use it only where nothing downstream depends
    on the posture, or alongside a ``q_verifier`` that checks the next leg.

    ``min_clearance`` additionally *rejects* candidates below that signed distance
    (0.0 = accept anything the collision checker calls free).

    After the first feasible solve, subsequent attempts warm-start from that
    solution (with small noise) rather than from q_reference.  This keeps IPOPT
    in the same IKFast branch (same sign of joint 5), reducing solve times for
    multi-candidate searches.  The sign of joint 5 in ik_options is also locked
    after the first success so that subsequent IPOPT solves are explicitly prevented
    from crossing the wrist singularity (joint 5 = 0).  Because the default IK
    options enable impose_flex_constraints, this tightens existing flex-constraint
    bounds at no extra evaluator cost.

    If ``q_verifier`` is provided, each candidate q must pass ``q_verifier(q)``
    before being counted.  Candidates that fail the verifier are silently
    skipped.  This allows ``constrained_plan`` to gate acceptance on the IK
    back-computation (``state_to_q23``) check so that only reconstructable
    configurations are returned.

    ``score_fn`` (used only when ``q_reference`` is None) replaces the default
    ranking: candidates are ranked by *minimum* ``score_fn(q)``, ties broken by
    discovery order (deterministic under a fixed rng). The grasp path uses this
    with ``grasp_posture_score`` -- see that function for the evidence.

    ``info``, when passed a dict, receives ``n_found`` (feasible candidates
    collected) and ``best_score`` (the winner's score when score_fn is used) --
    observability for the caller without changing the return type.

    ``post_success_wall_time`` / ``collect_budget_s`` bound what multi-candidate
    *ranking* may cost on top of plain feasibility. The search is untouched up
    to the first feasible candidate (same draws, same per-solve wall limit), so
    feasibility can never regress; after it, further tries run with the reduced
    IPOPT wall limit and no new try starts once ``collect_budget_s`` seconds
    have elapsed since that first success. Measured need: failed IPOPT tries
    cost up to the full 10 s wall each, so exhausting 5 tries for ranking cost
    grid point 0 +19 s where the budgeted version costs ~5 s. Under heavy load
    the collection window closes earlier and the result degrades toward
    first-feasible -- the same wall-clock sensitivity the BiRRT timeouts
    already have, and the safe direction.

    Limit-hugging IK branches make poor RRT seeds: the analytic IK around such
    a configuration reconstructs arm joints just past their limits, starving
    tree growth. ``n_candidates=1`` reproduces first-success behaviour.

    Candidates are only collected within the first GCP pair that yields any —
    scanning later pairs would multiply the IK cost (and half the GCPs are
    unreachable under the joint limits) without better goals.
    """
    stab_ctx = setup.plant.CreateDefaultContext()
    L = setup.layout
    # A separate diagram-rooted context, because clearance needs the SceneGraph
    # query output port and stab_ctx is a bare plant context.
    clear_diag_ctx = setup.diagram.CreateDefaultContext()
    clear_ctx = setup.plant.GetMyContextFromRoot(clear_diag_ctx)

    def clearance_of(q23) -> float:
        """Minimum signed distance at ``q23``; negative means penetration."""
        setup.set_q23_into(clear_ctx, q23)
        qobj = setup.plant.get_geometry_query_input_port().Eval(clear_ctx)
        pens = qobj.ComputePointPairPenetration()
        if pens:
            return -float(max(p.depth for p in pens))
        pairs = qobj.ComputeSignedDistancePairwiseClosestPoints(_IK_CLEARANCE_SEARCH_M)
        if not pairs:
            return _IK_CLEARANCE_SEARCH_M
        return float(min(p.distance for p in pairs))

    best_q, best_metric, n_found = None, np.inf if q_reference is not None else -np.inf, 0
    wall_time_0 = ik_options.max_wall_time
    t_first_success = None
    for right_gcp, left_gcp in gcp_pairs:
        ik_options.right_gcp = right_gcp
        ik_options.left_gcp  = left_gcp
        # Reset any sign constraints from prior GCP pairs so the first solve
        # can explore freely; the sign is locked after the first feasible
        # solution (see below).
        ik_options.joint5_sign_right = None
        ik_options.joint5_sign_left  = None
        for attempt in range(n_tries):
            if (t_first_success is not None and collect_budget_s is not None
                    and time.perf_counter() - t_first_success > collect_budget_s):
                break  # a candidate exists; ranking may not spend more wall
            if q_reference is not None:
                # After the first feasible solve, warm-start from that solution
                # instead of q_reference + noise.  Staying near a known-feasible
                # point keeps subsequent IPOPT solves in the same IKFast branch
                # (same GCP / same sign of joint 5), so they converge faster.
                if best_q is not None and attempt > 0:
                    q_initial = best_q + rng.normal(scale=0.05, size=best_q.shape)
                else:
                    q_initial = q_reference.copy()
                    if attempt > 0:
                        q_initial += rng.normal(scale=0.25, size=q_initial.shape)
            else:
                q_initial = _random_q_in_gcp(right_gcp, setup, rng)
            if q_initial is None:
                break  # GCP unreachable by sampling; skip its remaining tries.
            # One record per attempt, with the outcome, because an IK search is a
            # loop over *rejections*: a solve that succeeds and is then thrown out
            # by the CoM, collision or verifier check costs exactly as much as one
            # that ships, and only the outcome field distinguishes them.
            # phase distinguishes the two halves of this loop, which the code
            # itself already treats differently (post_success_wall_time /
            # collect_budget_s bound only the second): everything up to the first
            # feasible candidate is the search, everything after it is ranking --
            # extra solves whose only product is a better choice among candidates.
            # All but one of those are thrown away by construction, but which one
            # wins is not known until the loop ends, so they are labelled rather
            # than flagged discarded.
            with record("ik.attempt", label=None, attempt=attempt,
                        gcp=int(gcp_key_of(right_gcp)),
                        phase="search" if t_first_success is None else "rank",
                        warm_start=bool(best_q is not None and attempt > 0)):
                q = solve_ik(
                    right_target, left_target,
                    q_initial=q_initial,
                    diagram=ik_diagram,
                    options=ik_options,
                )
                outcome = "solved"
                if q is None:
                    outcome = "solver_failed"
                elif not check_com_stability(q, setup.plant, stab_ctx, inset=inset):
                    outcome = "rejected_com"
                else:
                    setup.set_q23_into(stab_ctx, q)
                    if not setup.collision_checker.CheckConfigCollisionFree(
                        setup.plant.GetPositions(stab_ctx)
                    ):
                        outcome = "rejected_collision"
                    elif q_verifier is not None and not q_verifier(q):
                        outcome = "rejected_verifier"
                mark(outcome=outcome)
                if outcome != "solved":
                    mark_discarded()
            if outcome != "solved":
                continue
            # Clearance is a full signed-distance query; only pay for it when a
            # branch below actually reads it. The goal-IK path (q_reference set,
            # min_clearance=0) was computing and discarding it once per feasible
            # candidate -- pure waste at 50 candidates per constrained leg.
            if min_clearance > 0.0 or (prefer_clearance and q_reference is None):
                clear = clearance_of(q)
                if clear < min_clearance:
                    continue
            else:
                clear = 0.0
            if q_reference is not None:
                metric = np.linalg.norm(q - q_reference)
                if metric < best_metric:
                    best_q, best_metric = q, metric
            elif score_fn is not None:
                # Minimise the caller's score; strict < keeps the first-found
                # candidate on ties, so the outcome is deterministic.
                metric = -float(score_fn(q))
                if best_q is None or metric > best_metric:
                    best_q, best_metric = q, metric
            else:
                m = np.minimum(q - setup.q_lb, setup.q_ub - q)
                jl_margin = min(m[L.right_arm].min(), m[L.left_arm].min())
                # Clearance first, joint-limit margin as a tiebreak within
                # _IK_CLEARANCE_TIE_M of the best clearance seen. Ranking on
                # clearance alone would trade a whole radian of joint-limit
                # headroom for a tenth of a millimetre of gap.
                metric = (round(clear / _IK_CLEARANCE_TIE_M) if prefer_clearance
                          else 0.0, jl_margin)
                if best_q is None or metric > best_metric:
                    best_q, best_metric = q, metric
            if t_first_success is None:
                t_first_success = time.perf_counter()
                if post_success_wall_time is not None:
                    # Only the *extra* ranking tries get the tight limit; the
                    # limit is restored before returning (the options object is
                    # shared with later attempts and the standoff solve).
                    ik_options.max_wall_time = min(wall_time_0,
                                                   post_success_wall_time)
            n_found += 1
            if n_found >= n_candidates:
                if info is not None:
                    info["n_found"] = n_found
                    if score_fn is not None and best_metric > -np.inf:
                        info["best_score"] = -best_metric
                ik_options.max_wall_time = wall_time_0
                return best_q
            # After the first feasible solve, lock the sign of arm joint 5 so
            # subsequent IPOPT solves are restricted to the same half-plane.
            # When impose_flex_constraints is True (the default), this tightens
            # the flex-constraint bounds at no extra evaluator cost.
            # Only enforce if joint 5 is far enough from zero (> 0.05 rad) to
            # avoid blocking searches near the singularity crossing.
            _J5_MIN = 0.05  # rad — minimum |q5| to trust the sign
            if n_found == 1:
                if abs(q[L.right_arm][5]) >= _J5_MIN:
                    ik_options.joint5_sign_right = (
                        1 if q[L.right_arm][5] >= 0 else -1
                    )
                if abs(q[L.left_arm][5]) >= _J5_MIN:
                    ik_options.joint5_sign_left = (
                        1 if q[L.left_arm][5] >= 0 else -1
                    )
        if n_found > 0:
            break  # Stay within the first productive GCP pair.
    ik_options.max_wall_time = wall_time_0
    if info is not None:
        info["n_found"] = n_found
        if score_fn is not None and n_found > 0:
            info["best_score"] = -best_metric
    return best_q




# ── Planner clearance margin ──────────────────────────────────────────────────

@contextmanager
def _checker_padding(collision_checker, margin: float, env_only: bool = False):
    """Temporarily inflate collision pairs by ``margin`` metres.

    ``CheckConfigCollisionFree`` is a *penetration* test: it accepts a
    configuration whose geometry is a micron from an obstacle. So the RRT and
    shortcutter happily return paths that graze -- measured on the reach leg,
    1.26 mm between ee_finger_2 and a box wall mid-descent, and 0.33 mm arm-vs-arm
    on the lift. Those paths are then handed to trajopt as its initial guess,
    where any positive minimum-distance floor makes them infeasible before the
    solver takes a step, and trajopt returns an infeasibilities-minimized iterate
    that is worse than the path it started from.

    Padding fixes this at the source and at zero per-check cost: the checker's
    existing boolean test now means "clearance >= margin". That is much cheaper
    than computing signed distances in the validity checker, which runs 25k-35k
    times per leg.

    ``env_only=True`` pads robot-vs-environment pairs and leaves robot-vs-robot
    alone. Prefer it when the margin exists to buy trajopt headroom rather than to
    make the path itself safer: every reach/home leg whose trajopt output was
    dense-rejected over a full grid run was rejected on ``ee_finger_1``/
    ``ee_finger_2`` vs *world*, never on a self-collision, so robot-robot padding
    buys nothing there -- while costing real free space, since a bimanual grasp
    brings the two arms deliberately close and the reach leg has to get them there.

    Padding does not override collision *filters*, so the held-box filtering and
    the model-level filters are unaffected.

    Restores the previous padding matrix on exit, because the checker usually
    belongs to the caller rather than to the planner.
    """
    if not margin:
        yield collision_checker
        return
    # np.array(..., copy=True): GetPaddingMatrix() hands back a view that aliases
    # the checker's live matrix, so without the copy `saved` tracks the very
    # mutations we are about to make and the restore is a no-op.
    saved = np.array(collision_checker.GetPaddingMatrix(), copy=True)
    try:
        if not env_only:
            collision_checker.SetPaddingAllRobotRobotPairs(margin)
        collision_checker.SetPaddingAllRobotEnvironmentPairs(margin)
        yield collision_checker
    finally:
        collision_checker.SetPaddingMatrix(saved)


# ── Trajectory verification ───────────────────────────────────────────────────

@dataclass
class TrajectoryReport:
    """Outcome of ``verify_trajectory``. ``ok`` is the guarantee; the rest is
    evidence."""
    ok: bool
    n_samples: int
    duration: float
    min_clearance: float          # m; <= 0 means penetration
    min_clearance_pair: str
    min_clearance_t: float
    worst_stability_margin: float  # m; < 0 means CoM outside the (inset) polygon
    worst_stability_t: float
    max_joint_limit_violation: float   # rad
    max_ee_error: float           # m (0.0 when no EE constraint was given)
    failures: tuple               # human-readable reasons, empty when ok

    def summary(self) -> str:
        bits = [
            f"{self.n_samples} samples over {self.duration:.3f}s",
            f"clearance {self.min_clearance*1000:.2f} mm "
            f"({self.min_clearance_pair} @ t={self.min_clearance_t:.3f})",
            f"CoM margin {self.worst_stability_margin*1000:.1f} mm",
            f"joint-limit violation {self.max_joint_limit_violation:.2e} rad",
        ]
        if self.max_ee_error:
            bits.append(f"EE error {self.max_ee_error*1000:.3f} mm")
        head = "OK" if self.ok else "FAIL"
        tail = "" if self.ok else "  |  " + "; ".join(self.failures)
        return f"[{head}] " + ", ".join(bits) + tail

    def __str__(self):
        return self.summary()


def verify_trajectory(*args, **kwargs) -> "TrajectoryReport":
    """Timed wrapper around the dense guarantee check; see ``_verify_trajectory``.

    Split out purely so the check appears in the event log. It is not free --
    n_samples analytic-IK reconstructions plus a signed-distance query each --
    and it runs on every leg of every attempt, so leaving it untimed put a real
    slice of the pipeline's runtime outside the accounting entirely.
    """
    with record("verify.leg", kwargs.get("label"),
                n_samples=kwargs.get("n_samples", 200)):
        report = _verify_trajectory(*args, **{k: v for k, v in kwargs.items()
                                              if k != "label"})
        mark(ok=bool(report.ok))
        return report


def _verify_trajectory(
    traj,
    plant,
    collision_checker,
    diagram,
    *,
    q_fn: Optional[Callable] = None,
    n_samples: int = 200,
    support_polygon_inset: float = 0.0,
    required_clearance: float = 0.0,
    # 1e-3 rad ~= 0.06 deg. Tighter than this rejects trajectories over numerical
    # dust: trajopt bounds the B-spline's control points to the joint limits, and
    # the curve then overshoots a limit by ~7e-4 rad between them, which is far
    # below the robot's own repeatability and not a defect worth failing a leg for.
    joint_limit_tol: float = 1e-3,
    ee_target_fn: Optional[Callable] = None,
    ee_tolerance: float = 1e-3,
    # 0.10, not the 0.05 this used to be: with the constrained legs now held to a
    # 50 mm box<->table bound, a 50 mm search saturated on every one of them and
    # reported a flat "50.000 mm" that could not be told apart from "exactly at
    # the bound". The cost is only in the distance query's broad phase, which is
    # not where verification time goes (n_samples x IK is).
    clearance_search_distance: float = 0.10,
) -> TrajectoryReport:
    """Densely check a finished trajectory against the three guarantees.

    Trajopt constrains only ``n_constr_pts`` samples of a B-spline and says
    nothing about the curve between them; TOPPRA then reparameterises it, and a
    fallback path may not have been through trajopt at all. So the only way to
    *know* a returned trajectory is valid is to sample the returned trajectory --
    which is what this does, and what the pipeline previously never did (its only
    gate checked that the stages had run and that the duration was positive).

    Args:
        traj:   the finished trajectory (typically the retimed one).
        q_fn:   maps ``traj.value(t).flatten()`` to the 23-D active-joint vector.
                Defaults to identity for a 23-row trajectory, and to the active
                joint slice for a full-plant one.
        required_clearance: minimum signed distance to demand, in metres. 0.0
                means "no penetration", matching what the RRT's validity checker
                enforces. Raise it only to the clearance the configurations
                involved can actually achieve -- the bimanual grasp pins
                link_left_arm_5 against link_right_arm_3 at about 2 mm, so a
                trajectory that touches the grasp cannot meet more than that.
        ee_target_fn: optional ``t -> RigidTransform`` giving the commanded
                mid-gripper pose, for the constrained legs. When given, the
                achieved mid-frame pose is compared against it and the worst
                translation error is reported.
    """
    setup = _Setup.build(plant, collision_checker, diagram)
    L = setup.layout
    pos_idxs = np.asarray(setup.pos_idxs_23)

    if q_fn is None:
        rows = traj.value(traj.start_time()).flatten().size
        if rows == 23:
            q_fn = lambda v: v
        elif rows == plant.num_positions():
            q_fn = lambda v: v[pos_idxs]
        else:
            raise ValueError(
                f"verify_trajectory: trajectory has {rows} rows, which is neither "
                f"23 nor the plant's {plant.num_positions()}; pass q_fn explicitly."
            )

    diag_ctx = diagram.CreateDefaultContext()
    plant_ctx = plant.GetMyContextFromRoot(diag_ctx)
    stab_ctx = plant.CreateDefaultContext()
    insp = plant.get_geometry_query_input_port().Eval(plant_ctx).inspector()

    def body_name(gid):
        return plant.GetBodyFromFrameId(insp.GetFrameId(gid)).name()

    t0, t1 = traj.start_time(), traj.end_time()
    duration = float(t1 - t0)
    if not np.isfinite(duration) or duration <= 0:
        return TrajectoryReport(
            False, 0, duration, np.nan, "", np.nan, np.nan, np.nan, 0.0,
            (f"non-finite or non-positive duration ({duration})",))

    min_clear, min_pair, min_t = np.inf, "(nothing within search distance)", t0
    worst_stab, worst_stab_t = np.inf, t0
    max_jl = 0.0
    max_ee = 0.0
    failures = []

    for t in np.linspace(t0, t1, n_samples):
        q23 = q_fn(traj.value(t).flatten())
        if q23 is None or not np.all(np.isfinite(q23)):
            failures.append(f"trajectory is not finite at t={t:.3f}")
            break

        # Joint limits.
        lo = float(np.max(setup.q_lb - q23))
        hi = float(np.max(q23 - setup.q_ub))
        max_jl = max(max_jl, lo, hi)

        setup.set_q23_into(plant_ctx, q23)
        q_full = plant.GetPositions(plant_ctx)

        # Clearance / penetration, from SceneGraph so the *reported pair* is
        # nameable -- a bare collision-free bool cannot tell you what was tight.
        qobj = plant.get_geometry_query_input_port().Eval(plant_ctx)
        pens = qobj.ComputePointPairPenetration()
        if pens:
            deepest = max(pens, key=lambda p: p.depth)
            d = -float(deepest.depth)
            pair = f"{body_name(deepest.id_A)} vs {body_name(deepest.id_B)}"
        else:
            pairs = qobj.ComputeSignedDistancePairwiseClosestPoints(
                clearance_search_distance)
            if pairs:
                closest = min(pairs, key=lambda p: p.distance)
                d = float(closest.distance)
                pair = f"{body_name(closest.id_A)} vs {body_name(closest.id_B)}"
            else:
                d, pair = clearance_search_distance, "(all pairs beyond search)"
        if d < min_clear:
            min_clear, min_pair, min_t = d, pair, float(t)

        # Stability: distance from the CoM to the (inset) support polygon edges.
        setup.set_q23_into(stab_ctx, q23)
        com = plant.CalcCenterOfMassPositionInWorld(
            stab_ctx,
            [setup.base_inst, setup.torso_inst, setup.right_inst, setup.left_inst,
             plant.GetModelInstanceByName("right_gripper"),
             plant.GetModelInstanceByName("left_gripper"),
             plant.GetModelInstanceByName("head")],
        )
        resid = _com_support_polygon_residuals(com[:2], q23[L.base])
        margin = float(np.min(_stability_constraint_ub(support_polygon_inset) - resid))
        if margin < worst_stab:
            worst_stab, worst_stab_t = margin, float(t)

        if ee_target_fn is not None:
            X_des = ee_target_fn(t)
            w = plant.world_frame()
            X_r = plant.CalcRelativeTransform(plant_ctx, w, plant.GetFrameByName("ee_right"))
            X_l = plant.CalcRelativeTransform(plant_ctx, w, plant.GetFrameByName("ee_left"))
            mid = 0.5 * (X_r.translation() + X_l.translation())
            max_ee = max(max_ee, float(np.linalg.norm(mid - X_des.translation())))

    if min_clear < required_clearance:
        kind = "penetration" if min_clear <= 0 else "clearance below requirement"
        failures.append(
            f"{kind}: {min_clear*1000:.2f} mm at t={min_t:.3f} ({min_pair}), "
            f"required {required_clearance*1000:.2f} mm")
    if worst_stab < 0:
        failures.append(
            f"CoM outside support polygon by {-worst_stab*1000:.1f} mm at "
            f"t={worst_stab_t:.3f} (inset {support_polygon_inset*1000:.1f} mm)")
    if max_jl > joint_limit_tol:
        failures.append(f"joint limits exceeded by {max_jl:.2e} rad")
    if ee_target_fn is not None and max_ee > ee_tolerance:
        failures.append(
            f"end-effector constraint violated by {max_ee*1000:.3f} mm "
            f"(tolerance {ee_tolerance*1000:.3f} mm)")

    return TrajectoryReport(
        ok=not failures, n_samples=n_samples, duration=duration,
        min_clearance=min_clear, min_clearance_pair=min_pair, min_clearance_t=min_t,
        worst_stability_margin=worst_stab, worst_stability_t=worst_stab_t,
        max_joint_limit_violation=max_jl, max_ee_error=max_ee,
        failures=tuple(failures),
    )


# ── Validity checkers ─────────────────────────────────────────────────────────

def _make_reach_validity(
    setup: _Setup, plant_ctx, inset: float = 0.0,
) -> Callable[[np.ndarray], bool]:
    def checker(q23: np.ndarray) -> bool:
        if np.any(q23 < setup.q_lb) or np.any(q23 > setup.q_ub):
            return False
        setup.set_q23_into(plant_ctx, q23)
        if not setup.collision_checker.CheckConfigCollisionFree(
            setup.plant.GetPositions(plant_ctx)
        ):
            return False
        return check_com_stability(q23, setup.plant, plant_ctx, inset=inset)
    return checker


def _make_lift_validity(
    setup: _Setup, plant_ctx, state_to_q23: Callable, lift_gcp: np.ndarray,
    inset: float = 0.0,
) -> Callable[[np.ndarray], bool]:
    L = setup.layout

    def checker(s: np.ndarray) -> bool:
        q = state_to_q23(s)
        if q is None:
            return False
        if np.any(q < setup.q_lb) or np.any(q > setup.q_ub):
            return False
        if not (np.array_equal(gcp_of_arm(q[L.right_arm], "right"), lift_gcp) and
                np.array_equal(gcp_of_arm(q[L.left_arm], "left"),  lift_gcp)):
            return False
        setup.set_q23_into(plant_ctx, q)
        if not setup.collision_checker.CheckConfigCollisionFree(
            setup.plant.GetPositions(plant_ctx)
        ):
            return False
        return check_com_stability(q, setup.plant, plant_ctx, inset=inset)
    return checker


# ── 14-D lift helpers ─────────────────────────────────────────────────────────

@dataclass
class _LiftHelpers:
    gcp: np.ndarray
    T_mid_right: RigidTransform
    T_mid_left: RigidTransform
    s_start: np.ndarray
    lift_ik: object                          # boundary_fallback=False (validity)
    lift_ik_for_trajopt: object              # boundary_fallback=True  (trajopt)
    state_to_lumped: Callable
    state_to_lumped_for_trajopt: Callable
    state_to_q23: Callable                   # uses lift_ik


def _make_lift_helpers(
    q_start: np.ndarray, mid_target: RigidTransform, setup: _Setup,
) -> _LiftHelpers:
    L = setup.layout

    # GCP from q_start arm joints. Both arms must agree.
    gcp_right = gcp_of_arm(q_start[L.right_arm], "right")
    gcp_left  = gcp_of_arm(q_start[L.left_arm], "left")
    if not np.array_equal(gcp_right, gcp_left):
        raise RuntimeError(
            f"constrained_plan: q_start arm GCPs differ "
            f"(right={list(gcp_right)}, left={list(gcp_left)}); "
            f"both arms must share the same C-bundle."
        )
    lift_gcp = gcp_right

    # FK at q_start to derive the grasp.
    fk_ctx = setup.plant.CreateDefaultContext()
    setup.set_q23_into(fk_ctx, q_start)
    T_W_r = setup.plant.CalcRelativeTransform(
        fk_ctx, setup.plant.world_frame(), setup.plant.GetFrameByName("ee_right"))
    T_W_l = setup.plant.CalcRelativeTransform(
        fk_ctx, setup.plant.world_frame(), setup.plant.GetFrameByName("ee_left"))
    mid_pos_start = (T_W_r.translation() + T_W_l.translation()) / 2.0
    T_W_mid_start = RigidTransform(mid_pos_start)
    T_mid_right = T_W_mid_start.InvertAndCompose(T_W_r)
    T_mid_left  = T_W_mid_start.InvertAndCompose(T_W_l)

    state_to_lumped         = _make_state_to_lumped(T_mid_right, T_mid_left)
    # Trajopt needs the same map, but AutoDiff-aware. Same closure works either way.
    state_to_lumped_trajopt = state_to_lumped

    lift_ik         = Rby1IK(boundary_fallback=False)
    lift_ik_trajopt = Rby1IK(boundary_fallback=True,
                             boundary_damping_lambda=TRAJOPT_BOUNDARY_DAMPING)

    def state_to_q23(s: np.ndarray) -> Optional[np.ndarray]:
        q = lift_ik.compute_ik(state_to_lumped(s), right_gcp=lift_gcp, left_gcp=lift_gcp)
        return None if not np.all(np.isfinite(q)) else q

    s_start = np.concatenate([
        mid_pos_start, np.zeros(3),
        q_start[L.torso],
        [q_start[L.psi_right]],
        [q_start[L.psi_left]],
    ])
    q23_back = state_to_q23(s_start)
    if q23_back is None:
        raise RuntimeError("constrained_plan: start state failed IK back-computation")
    if not np.allclose(q23_back, q_start, atol=1e-3):
        raise RuntimeError(
            "constrained_plan: q_start does not match the IK branch implied by its GCP "
            "(state_to_q23(s_start) differs from q_start beyond 1e-3). Was q_start "
            "produced by solve_ik, or hand-crafted?"
        )

    return _LiftHelpers(
        gcp=lift_gcp,
        T_mid_right=T_mid_right,
        T_mid_left=T_mid_left,
        s_start=s_start,
        lift_ik=lift_ik,
        lift_ik_for_trajopt=lift_ik_trajopt,
        state_to_lumped=state_to_lumped,
        state_to_lumped_for_trajopt=state_to_lumped_trajopt,
        state_to_q23=state_to_q23,
    )


def _make_state_to_lumped(T_mid_right: RigidTransform, T_mid_left: RigidTransform) -> Callable:
    """Build a 14-D → 23-D lumped-vars mapping, AutoDiff-aware.

    Lumped layout matches Rby1IK.compute_ik:
        [base(3)=0, torso(6), right_eef(6), psi_right(1), left_eef(6), psi_left(1)].
    """
    from pydrake.all import (
        AutoDiffXd,
        InitializeAutoDiff,
        RigidTransform_,
        RollPitchYaw_,
        RotationMatrix_,
    )

    def state_to_lumped(s):
        is_ad = isinstance(s.flat[0], AutoDiffXd)
        if is_ad:
            n = s[0].derivatives().size
            base_zeros = InitializeAutoDiff(np.zeros(3), np.zeros((3, n))).flatten()
            T_mid = RigidTransform_[AutoDiffXd](RollPitchYaw_[AutoDiffXd](s[3:6]), s[:3])
            T_r_mat = T_mid.GetAsMatrix4() @ T_mid_right.GetAsMatrix4()
            T_l_mat = T_mid.GetAsMatrix4() @ T_mid_left.GetAsMatrix4()
            T_r = RigidTransform_[AutoDiffXd](
                RotationMatrix_[AutoDiffXd](T_r_mat[:3, :3]), T_r_mat[:3, 3])
            T_l = RigidTransform_[AutoDiffXd](
                RotationMatrix_[AutoDiffXd](T_l_mat[:3, :3]), T_l_mat[:3, 3])
        else:
            base_zeros = np.zeros(3)
            T_mid = RigidTransform(RollPitchYaw(s[3:6]), s[:3])
            T_r = T_mid @ T_mid_right
            T_l = T_mid @ T_mid_left
        reef = np.concatenate([T_r.translation(), T_r.rotation().ToRollPitchYaw().vector()])
        leef = np.concatenate([T_l.translation(), T_l.rotation().ToRollPitchYaw().vector()])
        return np.concatenate([base_zeros, s[6:12], reef, s[12:13], leef, s[13:14]])

    return state_to_lumped


def _lift_state_bounds(s_start: np.ndarray, s_goal: np.ndarray, setup: _Setup,
                       rpy_margin: Optional[float] = None):
    """Derive lift sampling bounds from the start and goal mid-frame positions.

    Position component grows around the start↔goal interval with margin; torso /
    ψ bounds come from the plant joint limits. The mid-frame rpy component is
    wide (±2π) by default, because the path may need to swing through arbitrary
    midframe orientations between an aligned start and goal. If ``rpy_margin``
    is given, the rpy is instead clamped to the start↔goal orientation interval
    widened by that margin (radians) — use a small value to keep the carried
    object at a near-constant orientation (e.g. level) along the whole path.
    """
    L = setup.layout
    margin = 0.4
    mid_lo = np.minimum(s_start[:3], s_goal[:3]) - margin
    mid_hi = np.maximum(s_start[:3], s_goal[:3]) + margin
    mid_lo[2] = max(0.05, mid_lo[2])

    if rpy_margin is None:
        rpy_lo = np.array([-2 * np.pi, -2 * np.pi, -2 * np.pi])
        rpy_hi = np.array([ 2 * np.pi,  2 * np.pi,  2 * np.pi])
    else:
        rpy_lo = np.minimum(s_start[3:6], s_goal[3:6]) - rpy_margin
        rpy_hi = np.maximum(s_start[3:6], s_goal[3:6]) + rpy_margin

    s_lb = np.concatenate([
        mid_lo,
        rpy_lo,
        setup.q_lb[L.torso],
        [setup.q_lb[L.psi_right]],
        [setup.q_lb[L.psi_left]],
    ])
    s_ub = np.concatenate([
        mid_hi,
        rpy_hi,
        setup.q_ub[L.torso],
        [setup.q_ub[L.psi_right]],
        [setup.q_ub[L.psi_left]],
    ])
    return s_lb, s_ub


# ── RRT helpers ───────────────────────────────────────────────────────────────

def _straight_line_path(
    q_start: np.ndarray,
    q_goal: np.ndarray,
    validity: Callable,
    check_size: float,
    n_waypoints: int = 9,
) -> Optional[list]:
    """The C-space straight line from start to goal, or None if anything on it fails.

    A straight line in configuration space is the shortest path under the joint-space
    length that trajopt is minimising, so where it is valid there is nothing for a
    BiRRT to find that is better -- the search only costs time and makes the leg's
    output depend on a random seed. Worth trying only on legs known to be short and
    largely unobstructed; on a long leg that has to get around the box it will fail,
    and then the probe has cost one sweep of validity checks.

    Held to exactly the standard a BiRRT edge is held to: sampled at ``check_size``,
    against the same ``validity`` predicate, under whatever clearance padding the
    caller has in force. The endpoints are skipped because ``plan_to_config`` has
    already checked them, deliberately unpadded -- a grasp or standoff config handed
    in by the caller is allowed to sit closer to the box than the *path* may.

    What comes back is only ``n_waypoints`` evenly spaced configurations, not every
    sample that was checked. The line is straight, so intermediate waypoints carry no
    information, and a path of 200-odd waypoints would hand trajopt a B-spline with
    hundreds of control points -- whose cost grows about quadratically in that count.
    Nine is enough for the guess to be well conditioned.
    """
    delta = np.asarray(q_goal, dtype=float) - np.asarray(q_start, dtype=float)
    span = float(np.linalg.norm(delta))
    if span == 0.0:
        return [np.asarray(q_start, dtype=float).copy()] * 2

    n_chk = max(int(np.ceil(span / max(check_size, 1e-9))), n_waypoints)
    for t in np.linspace(0.0, 1.0, n_chk + 1)[1:-1]:
        if not validity(q_start + t * delta):
            return None
    return [q_start + t * delta for t in np.linspace(0.0, 1.0, n_waypoints)]


def _unconstrained_rrt_path(
    q_start: np.ndarray,
    q_goal: np.ndarray,
    setup: _Setup,
    validity: Callable,
    rrt_options: rrt.RRTOptions,
    rng_seed: int,
    trajectories: Optional[dict] = None,
) -> list:
    L = setup.layout
    margin = 0.4
    q_lo = np.minimum(q_start, q_goal) - margin
    q_hi = np.maximum(q_start, q_goal) + margin

    def random_config() -> np.ndarray:
        q = q_start.copy()
        q[L.torso.start:] = np.random.uniform(q_lo[L.torso.start:], q_hi[L.torso.start:])
        return q

    np.random.seed(rng_seed)
    with record("rrt.birrt", "unconstrained"):
        path = rrt.BiRRT(random_config, validity).plan(q_start, q_goal, rrt_options)
        mark(found=bool(path), n_waypoints=len(path) if path else 0)
    if not path:
        return path
    _record_stage(trajectories, "rrt_raw", lambda: _path_to_composite_traj(path))
    np.random.seed(rng_seed)
    with record("rrt.shortcut", "unconstrained", num_tries=10):
        shortcut_path = shortcut.shortcut(path, validity, num_tries=10, check_size=rrt_options.check_size)
    _record_stage(trajectories, "rrt_shortcut",
                  lambda: _path_to_composite_traj(shortcut_path))
    return shortcut_path


def _constrained_rrt_path(
    s_start: np.ndarray,
    s_goal: np.ndarray,
    s_lb: np.ndarray,
    s_ub: np.ndarray,
    validity: Callable,
    rrt_options: rrt.RRTOptions,
    rng_seed: int,
    edge_validator: Optional[Callable] = None,
    trajectories: Optional[dict] = None,
) -> list:
    def random_state() -> np.ndarray:
        return np.random.uniform(s_lb, s_ub)

    np.random.seed(rng_seed)
    with record("rrt.birrt", "constrained"):
        path = rrt.BiRRT(random_state, validity, EdgeValidator=edge_validator).plan(s_start, s_goal, rrt_options)
        mark(found=bool(path), n_waypoints=len(path) if path else 0)
    if not path:
        return path
    _record_stage(trajectories, "rrt_raw", lambda: _path_to_composite_traj(path))
    np.random.seed(rng_seed)
    with record("rrt.shortcut", "constrained", num_tries=100):
        shortcut_path = shortcut.shortcut(path, validity, num_tries=100, check_size=rrt_options.check_size, EdgeValidator=edge_validator)
    _record_stage(trajectories, "rrt_shortcut",
                  lambda: _path_to_composite_traj(shortcut_path))
    return shortcut_path


# ── Trajectory construction ──────────────────────────────────────────────────

def _path_to_composite_traj(path: list) -> CompositeTrajectory:
    """Twice-differentiable interpolation of an RRT path, in one spline.

    One call over *all* the waypoints, not one call per segment. The per-segment
    version this replaces passed ``np.zeros(n), np.zeros(n)`` as the start and end
    velocities of every two-point piece and then concatenated them, so although each
    piece was C2 in isolation, the composed trajectory came to a **complete stop at
    every waypoint**. TOPPRA retimes along a path and cannot undo a vanishing path
    derivative, so a leg that skips trajopt executed as a series of starts and stops.

    Measured over a full grid run, before this change: the two ``do_trajopt=False``
    legs, ``reach_descend`` and ``home_retreat``, had a median of 5 interior velocity
    dips below 10% of their own mean speed on 19 of 19 points, while every leg that
    does run trajopt had none. Those two sit just before the pick and just after the
    place, which is where the hesitation is visible.

    Zero velocity now applies only at the true start and end of the leg, which is
    correct -- the robot really is at rest there.
    """
    n = len(path[0])
    return CompositeTrajectory([
        PiecewisePolynomial.CubicWithContinuousSecondDerivatives(
            np.arange(len(path), dtype=float),
            np.array(path).T,
            np.zeros(n), np.zeros(n),
        )
    ])


def _path_to_bspline(
    path: list,
    t0: float,
    t1: float,
    spline_order: int = 4,
    eps: float = 1e-4,
    lb: Optional[np.ndarray] = None,
    ub: Optional[np.ndarray] = None,
    multiplicity: Optional[int] = None,
) -> BsplineTrajectory:
    """B-spline through an RRT path, at a chosen control-point multiplicity.

    ``multiplicity`` is how many control points each waypoint contributes;
    ``None`` means ``spline_order - 1``, the full multiplicity.

    At full multiplicity the curve *interpolates* every waypoint. A B-spline does
    not pass through its control points, so feeding the RRT waypoints in directly
    produces a curve that cuts the corners off the path -- and the path is the only
    thing that has actually been verified collision-free, so the guess can violate
    constraints the path satisfies and trajopt starts infeasible through no fault
    of the path. Duplicating to full multiplicity removes that failure mode
    completely: measured over a full grid run, 87 of 87 solves reported a feasible
    guess.

    At multiplicity 1 the curve is the smooth corner-cutting one, with a third of
    the control points. **This is the better guess wherever it is feasible, and by
    a wide margin** -- full multiplicity makes the guess a piecewise-linear path
    with a C0 kink at every waypoint, and the optimiser stays in that basin, so the
    kinks survive into the shipped trajectory. Measured on grid point 0's lift, same
    verified clearance (+25.93 mm) in both cases:

        multiplicity 3:  joint path 5.39 rad, 24 kinks, max turn 21.9 deg
        multiplicity 1:  joint path 2.62 rad,  0 kinks, max turn  2.0 deg

    Hence ``_select_guess_multiplicity``, which tries 1 first and only falls back to
    full multiplicity when 1 leaves the guess infeasible. Do not simply change this
    default -- feasibility of the guess is the thing being traded, and it has to be
    tested per solve rather than assumed.

    The duplicates (multiplicity > 1) are perturbed by ``eps`` along the local path
    direction: exactly coincident control points make the fit numerically unstable
    for the solver, and nudging them along the direction of travel keeps them on the
    path while separating them.

    ``lb``/``ub`` clip the perturbed control points back into the position bounds
    trajopt will impose. Without this the perturbation can push a control point
    ``eps`` past a bound that the *path* respects exactly, and trajopt's
    AddPositionBounds then reports a violation at the guess. That matters more than
    its size suggests: eps is 1e-4 and _constraint_violations' tolerance is also
    1e-4, the constrained legs' mid-frame RPY box is only [-0.05, +0.05] wide, and
    an infeasible guess makes _solve_trajopt_prog skip the IPOPT-aggressive rung
    entirely. So a 1e-4 bookkeeping artefact could cost a leg its second solver.
    """
    # A B-spline of this order needs at least spline_order control points; a very
    # short path (heavily shortcut, or a near-trivial query) may not have that
    # many waypoints on its own.
    path = list(path)
    while len(path) < 2:
        path.append(path[-1])

    n = len(path)
    # ``multiplicity`` may be a scalar (every waypoint contributes the same number
    # of control points) or one count per waypoint. The per-waypoint form is what
    # _shrink_guess walks down: it needs to drop a duplicate from one waypoint
    # without touching the others.
    if multiplicity is None:
        blocks = [spline_order - 1] * n
    elif np.ndim(multiplicity) == 0:
        blocks = [max(1, int(multiplicity))] * n
    else:
        blocks = [max(1, int(m)) for m in multiplicity]
        if len(blocks) != n:
            raise ValueError(f"multiplicity has {len(blocks)} entries for a "
                             f"{n}-waypoint path")
    # The basis needs at least ``spline_order`` control points, and both the
    # multiplicity ladder and _shrink_guess can drive the total below that on a
    # short path (a C-space straight line is two waypoints, so mult=1 gives two
    # control points against order 4). Top up round-robin rather than failing.
    i = 0
    while sum(blocks) < spline_order:
        blocks[i % n] += 1
        i += 1
    starts = np.cumsum([0] + blocks[:-1])
    ctrl = np.array([pt for pt, b in zip(path, blocks) for _ in range(b)]).T

    for i in range(n):
        block = blocks[i]
        s, e = int(starts[i]), int(starts[i]) + block
        if i == 0:
            direction = path[1] - path[0]
        elif i == n - 1:
            direction = path[-1] - path[-2]
        else:
            direction = path[i + 1] - path[i - 1]
        nrm = np.linalg.norm(direction)
        direction = direction / nrm if nrm > 1e-12 else np.zeros_like(direction)
        # Offsets are one-sided at the endpoints so the duplicated ends do not
        # step outside the path, and centred in the interior.
        if i == 0:
            offsets = eps * np.arange(block)
        elif i == n - 1:
            offsets = -eps * np.arange(block)
        else:
            offsets = eps * (np.arange(block) - (block - 1) / 2.0)
        ctrl[:, s:e] += np.outer(direction, offsets)

    if lb is not None:
        ctrl = np.maximum(ctrl, np.asarray(lb, dtype=float).reshape(-1, 1))
    if ub is not None:
        ctrl = np.minimum(ctrl, np.asarray(ub, dtype=float).reshape(-1, 1))

    basis = BsplineBasis(spline_order, ctrl.shape[1],
                         initial_parameter_value=t0, final_parameter_value=t1)
    return BsplineTrajectory(basis, ctrl)


def _traj_row_slice(traj, rows: list):
    if isinstance(traj, BsplineTrajectory):
        ctrl = [traj.control_points()[i][rows, :] for i in range(len(traj.control_points()))]
        return BsplineTrajectory(traj.basis(), ctrl)
    if isinstance(traj, PiecewisePolynomial):
        polys = [traj.getPolynomialMatrix(i)[rows, :] for i in range(traj.get_number_of_segments())]
        return PiecewisePolynomial(polys, traj.get_segment_times())
    if isinstance(traj, CompositeTrajectory):
        segs = [_traj_row_slice(traj.segment(i), rows) for i in range(traj.get_number_of_segments())]
        return CompositeTrajectory(segs)
    raise TypeError(f"_traj_row_slice: unsupported trajectory type {type(traj)}")


def _embed_q23_into_full_traj(traj_q23, setup: _Setup) -> StackedTrajectory:
    """Lift a 23-D trajectory into the full plant configuration space.

    Active joints are non-contiguous in plant ordering, so we walk plant
    positions in order and append either a row-slice of ``traj_q23`` (active
    group) or a constant ``ZeroOrderHold`` (inactive group). All evaluation
    and derivatives stay in C++.
    """
    active = set(int(i) for i in setup.pos_idxs_23)
    nq = setup.plant.num_positions()
    inactive = set(range(nq)) - active
    pos_to_row = {int(pos): row for row, pos in enumerate(setup.pos_idxs_23)}
    t0, t1 = traj_q23.start_time(), traj_q23.end_time()

    full = StackedTrajectory(rowwise=True)
    i = 0
    while i < nq:
        if i in inactive:
            j = i
            while j < nq and j in inactive:
                j += 1
            q_const = setup.q_full_default[i:j].reshape(-1, 1)
            full.Append(
                PiecewisePolynomial.ZeroOrderHold([t0, t1], np.hstack([q_const, q_const]))
            )
        else:
            j = i
            while j < nq and j not in inactive:
                j += 1
            rows = [pos_to_row[k] for k in range(i, j)]
            full.Append(_traj_row_slice(traj_q23, rows))
        i = j
    return full


# ── Trajopt solver infrastructure ─────────────────────────────────────────────

class TrajoptRequired(RuntimeError):
    """Trajopt did not produce a usable trajectory and no fallback is permitted.

    This is the **default** behaviour (``allow_birrt_fallback=False``): a leg whose
    optimised trajectory is unusable is a failed leg, and a run that reports success
    means every leg was optimised. Falling back to the raw BiRRT+shortcut path is
    opt-in, for downstream consumers that would rather have a jerkier motion than no
    motion; see ``_tag_fallback`` for what that opt-in must record.

    A distinct type so a harness can count "trajopt would not hold" separately from
    "BiRRT found no path at all" -- they are different failures with different fixes,
    and lumping them together is what made the grid's five RRT failures and its nine
    silent trajopt failures look like one number.
    """


class WarmStartRejected(RuntimeError):
    """A warm-start path handed to ``constrained_plan`` failed its prechecks.

    A distinct type because the caller's contract is 'reuse is additive':
    on this exception the caller silently falls back to a full replan, whereas
    any other failure means the plan itself failed. Never raised once the
    prechecks pass -- from there the leg follows the ordinary code path and
    fails with the ordinary exceptions.
    """


def _accept_trajopt_or_fall_back(traj, shortcut_traj, path, setup, *, label: str,
                                 q_fn=None, support_polygon_inset: float = 0.0,
                                 n_samples: int = 250,
                                 allow_birrt_fallback: bool = False):
    """Return trajopt's trajectory only if it survives a dense check.

    Trajopt constrains its B-spline at ``n_constr_pts`` sample points and says
    nothing about the curve between them. A *successful* solve can therefore still
    return a trajectory that penetrates between samples -- measured on the
    reach_approach leg, SNOPT reported success while the trajectory dipped 7.3 mm
    into an obstacle. The floor being adaptive makes this more likely, not less:
    where the guess only clears a fraction of a millimetre the floor drops to
    match, leaving trajopt free to graze and then overshoot between samples.

    So the solve's own success flag is necessary but not sufficient, and a solve that
    is rejected here is a trajopt failure exactly as much as one SNOPT itself refused.
    By default that raises ``TrajoptRequired``. The shortcut path, validated by the
    RRT at its ``check_size`` resolution, is the safe thing to keep instead *if* the
    caller has opted into ``allow_birrt_fallback`` -- and then the leg is tagged, not
    quietly downgraded.

    Returns ``(traj_to_use, accepted, report)``. The report is returned rather than
    only logged so callers can record *which* guarantee the optimised curve broke.
    """
    report = verify_trajectory(
        traj, setup.plant, setup.collision_checker, setup.diagram,
        q_fn=q_fn, n_samples=n_samples,
        support_polygon_inset=support_polygon_inset, required_clearance=0.0,
    )
    if report.ok:
        return traj, True, report
    if not allow_birrt_fallback:
        raise TrajoptRequired(
            f"{label}: trajopt solved but its output fails a dense check "
            f"({'; '.join(report.failures)}) and allow_birrt_fallback=False")
    print(f"[{label}] trajopt solved but its output fails a dense check "
          f"({'; '.join(report.failures)}); keeping the validated shortcut path")
    return _fallback_traj(path, shortcut_traj), False, report


def _tag_fallback(diagnostics, reason: str):
    """Record that this leg shipped the BiRRT+shortcut path, not trajopt's output.

    A fallback is a *trajopt failure*: the optimiser did not produce a usable
    trajectory. It happens to be a recoverable one, because the path it fell back to
    carries the same three guarantees -- but "recoverable" is not "did not happen",
    and for a result being reported it is the failure that matters. It used to be
    visible only on stdout, so a cached plan whose reach leg had been dense-rejected
    was indistinguishable from one whose optimised curve shipped: both recorded
    ``SNOPT (1)``. That is how a 9-of-15 fallback count could only ever be quoted as a
    lower bound. This tag is what makes it countable from the artifact alone, and
    ``trajopt_summary`` in the grid pipeline reads it back.
    """
    if diagnostics is None:
        return
    diagnostics["trajopt_fell_back"] = True
    diagnostics["trajopt_fallback_reason"] = reason


def _fallback_traj(path: list, composite_traj):
    """Trajectory to return when trajopt fails, built from the validated path.

    The B-spline through the waypoints, not the ``CompositeTrajectory``. Since
    ``_path_to_bspline`` duplicates each waypoint to full multiplicity it tracks
    the validated path closely rather than cutting its corners, but it is still
    only an *approximation* of it, so the caller must verify the result -- which
    ``verify_trajectory`` does.
    """
    try:
        return _path_to_bspline(path, composite_traj.start_time(),
                                composite_traj.end_time())
    except Exception as e:  # keep the leg alive on a degenerate path
        print(f"[fallback] B-spline construction failed ({type(e).__name__}: {e}); "
              f"using the piecewise trajectory, which TOPPRA may reject")
        return composite_traj


def _path_min_clearance(q_full_samples, plant, plant_ctx) -> float:
    """Minimum signed distance over a set of full-plant configurations.

    Negative means penetration. Used to pick trajopt's distance floor from what
    its initial guess actually achieves, instead of from a constant.
    """
    worst = np.inf
    port = plant.get_geometry_query_input_port()
    for q_full in q_full_samples:
        if q_full is None or not np.all(np.isfinite(q_full)):
            return -np.inf
        plant.SetPositions(plant_ctx, q_full)
        qobj = port.Eval(plant_ctx)
        pens = qobj.ComputePointPairPenetration()
        if pens:
            worst = min(worst, -float(max(p.depth for p in pens)))
            continue
        pairs = qobj.ComputeSignedDistancePairwiseClosestPoints(0.05)
        if pairs:
            worst = min(worst, float(min(p.distance for p in pairs)))
    return 0.05 if worst is np.inf else worst


def _feasible_min_dist_bound(requested: float, path_clearance: float,
                             label: str, slack: float = 1e-4) -> float:
    """Lower trajopt's distance floor to something its initial guess satisfies.

    A floor above the guess's own clearance is infeasible at the guess, so the
    solver starts outside the feasible set and (measured, repeatedly) returns an
    infeasibilities-minimized iterate that is worse than the collision-free path
    it was given. The RRT's validity check is a pure penetration test, so the
    clearance it hands over is whatever the geometry happened to allow -- 1.26 mm
    on the reach, 0.33 mm arm-vs-arm on the lift. Rather than hardcode a constant
    that suits one leg and breaks another, take the floor from the path.

    Clearance is still *improved* by min_distance_margin's smooth penalty, which
    pushes away from obstacles out to its influence distance without making
    anything infeasible. The floor's job is only to stop trajopt making things
    worse.
    """
    if not np.isfinite(path_clearance):
        return 0.0
    usable = max(0.0, path_clearance - slack)
    if usable < requested:
        print(f"[trajopt {label}] lowering min-distance floor "
              f"{requested*1000:.2f} -> {usable*1000:.2f} mm: the initial guess "
              f"only clears {path_clearance*1000:.2f} mm")
        return usable
    return requested


def _constraint_violations(prog, x, tol: float = 1e-4) -> list:
    """(description, max_violation) for every constraint of ``prog`` violated at
    the decision-variable vector ``x``, worst first.

    Used both to label the initial guess and to explain a failed solve. Without
    this a trajopt failure is a bare solver code, and the two cases that matter
    -- "the guess was already infeasible" and "the guess was feasible and the
    solver lost it" -- are indistinguishable.
    """
    out = []
    for binding in prog.GetAllConstraints():
        try:
            val = x[prog.FindDecisionVariableIndices(binding.variables())]
            res = np.asarray(binding.evaluator().Eval(val), dtype=float)
        except Exception as e:
            # Report, do not skip. A binding that *throws* used to be dropped
            # silently, so a program whose worst constraints could not even be
            # evaluated was reported as FEASIBLE -- and "the guess was feasible"
            # is the branch that decides whether _solve_trajopt_prog escalates to
            # IPOPT. An unevaluable constraint is not a satisfied one.
            out.append((f"{binding.evaluator().get_description() or '<unnamed>'} "
                        f"<eval raised {type(e).__name__}>", np.inf))
            continue
        desc = binding.evaluator().get_description() or "<unnamed>"
        if not np.all(np.isfinite(res)):
            out.append((desc, np.inf))
            continue
        viol = np.maximum(binding.evaluator().lower_bound() - res,
                          res - binding.evaluator().upper_bound())

        # The constrained legs stack four feasibility families into one binding so
        # they can share a single analytic IK solve, so report the violation per
        # family rather than as one anonymous maximum. "which family" is the whole
        # diagnostic value here -- a guess that violates lift_min_dist (the path
        # cuts a corner into the box) and one that violates lift_ee_residual (the
        # solver has wandered out of the reachable set) call for opposite fixes.
        if desc.startswith(LIFT_FEASIBILITY_DESC):
            suffix = desc[len(LIFT_FEASIBILITY_DESC):]
            for name, start, stop in LIFT_FEASIBILITY_BLOCKS:
                m = float(np.max(viol[start:stop]))
                if m > tol:
                    out.append((f"{name}{suffix}", m))
            continue

        m = float(np.max(viol))
        if m > tol:
            out.append((desc, m))
    return sorted(out, key=lambda t: -t[1])


def _guess_is_feasible(prog) -> bool:
    """``_report_guess_feasibility`` without the logging.

    The shrink search calls this once per candidate removal -- hundreds of times
    per leg -- so it must not print, or the per-leg log becomes unreadable and
    the one line that matters (the summary) is lost in it.
    """
    return not _constraint_violations(prog, prog.initial_guess())


def _report_guess_feasibility(prog, label: str) -> bool:
    """Log whether trajopt's initial guess is feasible. Returns True if it is."""
    viols = _constraint_violations(prog, prog.initial_guess())
    if not viols:
        print(f"[trajopt {label}] initial guess FEASIBLE")
        return True
    worst = viols[0]
    print(f"[trajopt {label}] initial guess INFEASIBLE: {len(viols)} constraint(s), "
          f"worst {worst[0]} by {worst[1]:.2e}")
    for desc, m in viols[:4]:
        print(f"[trajopt {label}]   {desc}: {m:.2e}")
    return False


# Control-point multiplicities to try for the B-spline guess, smoothest first. See
# _path_to_bspline for what the ends of this mean and the measurements behind
# preferring 1; the fallback exists because full multiplicity is the only
# multiplicity guaranteed to interpolate the verified path, and a guess that starts
# outside the feasible set is the one case Harel's ladder says not to escalate from.
#
# 2 is in the middle for a reason beyond tracking the path more closely than 1: the
# constrained legs' joint-space energy cost runs one analytic IK per control point per
# evaluation, under AutoDiff whose width is the whole decision vector, so its cost
# grows roughly *quadratically* in the control-point count. Jumping 1 -> 3 triples the
# control points and so costs ~9x per evaluation, which is what makes the mult=3 rung
# exit on SNOPT's time (34) and iteration (31) limits. 2 costs ~4x instead, so it can
# keep both a feasible guess and a convergent solve where 3 keeps neither.
GUESS_MULTIPLICITIES = (1, 2, 3)


def _select_guess_multiplicity(build, label: str, multiplicities=None):
    """Build the trajopt program at the smoothest guess whose iterate is feasible.

    ``build(multiplicity)`` must return a fully-constrained
    ``KinematicTrajectoryOptimization``. Multiplicities are tried in order and the
    first feasible guess wins; the last is used unconditionally, so this always
    returns a program even when every guess is infeasible (which is the pre-existing
    behaviour -- ``_solve_trajopt_prog`` skips its IPOPT rung in that case).

    Building the program twice costs only constraint bookkeeping, not a solve: the
    bindings are added but never evaluated beyond the one pass
    ``_report_guess_feasibility`` makes over them.

    Returns ``(trajopt, guess_feasible, multiplicity)``.
    """
    # Resolved here rather than as a default argument. A default is bound once at
    # definition time, so `multiplicities=GUESS_MULTIPLICITIES` silently ignored any
    # later rebinding of the module global -- the constant looked configurable while
    # being frozen, which cost one wasted experiment before it was noticed.
    if multiplicities is None:
        multiplicities = GUESS_MULTIPLICITIES
    last = len(multiplicities) - 1
    for i, m in enumerate(multiplicities):
        trajopt = build(m)
        feasible = _report_guess_feasibility(trajopt.prog(), f"{label} mult={m}")
        if feasible or i == last:
            if not feasible:
                print(f"[trajopt {label}] no multiplicity in {list(multiplicities)} "
                      f"gives a feasible guess; solving from mult={m} anyway")
            return trajopt, feasible, m


# Guess-shrinking: start at full multiplicity and remove control points while the
# guess stays feasible *as a program*, so trajopt solves the smallest program that
# still starts inside the feasible set.
#
# Why it should pay: the constrained legs' joint-space energy cost runs one
# analytic IK per control point per evaluation under AutoDiff over the whole
# decision vector, so evaluation cost is roughly quadratic in the control-point
# count. GUESS_MULTIPLICITIES can only pick one of three uniform rungs; this can
# go below the lowest of them by dropping waypoints outright.
#
# Why it might not: every candidate removal costs a program rebuild plus one pass
# over every binding, and there are O(n) candidates. GUESS_SHRINK_MAX_TESTS caps
# that so a pathological path cannot swamp a leg, and every leg logs what the
# search cost so the trade is measurable rather than assumed.
GUESS_SHRINK = True
GUESS_SHRINK_MAX_TESTS = 400


def _choose_guess(build, path, label: str):
    """Pick trajopt's initial guess: shrink search, or the fixed ladder.

    ``build(path, multiplicity)`` is the one interface both take, so flipping
    GUESS_SHRINK swaps the strategy without touching either trajopt builder.
    """
    if GUESS_SHRINK:
        return _shrink_guess(build, path, label)
    return _select_guess_multiplicity(lambda m: build(path, m), label)


def _shrink_guess(build, path, label: str):
    """Smallest feasible-guess program reachable by removing control points.

    ``build(path, multiplicity)`` returns a fully-constrained
    ``KinematicTrajectoryOptimization`` over ``path`` at a per-waypoint
    ``multiplicity``.

    Starts at full multiplicity -- the only setting that interpolates the
    validated RRT path exactly, and measured feasible on 87/87 guesses -- then:

      * **Phase A** drops duplicate control points, one step at a time, per
        waypoint, 3 -> 2 -> 1.
      * **Phase B** drops whole waypoints, going below what any uniform
        multiplicity can reach. It runs **regardless of how far Phase A got**: a
        waypoint that still carries duplicates is still a removal candidate.

    Endpoints are never dropped. Both trajopt builders pin s=0 and s=1 with
    ``AddPathPositionConstraint(bspline.value(...))``, so removing the first or
    last waypoint would move the leg's endpoints -- a different problem, not a
    smaller one.

    Ordered and incremental, never random: ``archive/ruby_demo`` reduced control
    points randomly and shipped 8/20 with loosened tolerances.

    Returns ``(trajopt, guess_feasible, n_control_points)``.
    """
    t0 = time.perf_counter()
    n_tests = 0

    def attempt(p, m):
        """Build at (p, m) and report whether its guess is feasible."""
        nonlocal n_tests
        n_tests += 1
        # Split, because the two halves respond to different changes: the build
        # is constraint bookkeeping (n_constr_pts bindings), the probe is one
        # evaluation pass over all of them -- and on the constrained legs each
        # such evaluation is an analytic IK solve.
        with record("trajopt.build", label):
            prog = build(p, m)
        with record("trajopt.probe", label):
            feasible = _guess_is_feasible(prog.prog())
        return prog, feasible

    path = list(path)
    mult = [3] * len(path)
    trajopt, feasible = attempt(path, mult)
    n_full = trajopt.num_control_points()
    if not feasible:
        # Full multiplicity interpolates the path the BiRRT validated, so an
        # infeasible guess here is a model-level disagreement between the RRT's
        # validity predicate and trajopt's constraints -- not something a smaller
        # program fixes. Report it and hand back the same program the old ladder
        # would have ended on.
        _report_guess_feasibility(trajopt.prog(), f"{label} mult=3")
        print(f"[trajopt {label}] guess shrink: skipped, full multiplicity is "
              f"already infeasible ({n_full} control points)")
        return trajopt, False, n_full

    best = trajopt

    def budget_left():
        return n_tests < GUESS_SHRINK_MAX_TESTS

    # Phase A -- thin the duplicates.
    for target in (2, 1):
        for i in range(len(path)):
            if not budget_left():
                break
            if mult[i] <= target:
                continue
            trial = list(mult)
            trial[i] = target
            prog, ok = attempt(path, trial)
            if ok:
                mult, best = trial, prog

    # Phase B -- drop whole waypoints, interior only.
    i = 1
    while i < len(path) - 1 and budget_left():
        trial_path = path[:i] + path[i + 1:]
        trial_mult = mult[:i] + mult[i + 1:]
        prog, ok = attempt(trial_path, trial_mult)
        if ok:
            # Do not advance i: the next waypoint has shifted into this slot.
            path, mult, best = trial_path, trial_mult, prog
        else:
            i += 1

    n_final = best.num_control_points()
    print(f"[trajopt {label}] guess shrink: {n_full} -> {n_final} control points "
          f"({100.0 * (1 - n_final / max(n_full, 1)):.0f}% smaller) in {n_tests} "
          f"feasibility tests, {time.perf_counter() - t0:.1f}s"
          + ("  [TEST BUDGET EXHAUSTED]" if not budget_left() else ""))
    return best, True, n_final


# When set to a directory, every trajopt solve writes its solver's own iteration log
# there. Off by default: the logs are large and per-solve, and both solvers are
# otherwise silenced (SNOPT via kPrintToConsole, IPOPT via print_level 0), which means
# a stuck solve normally leaves nothing behind but an exit code.
#
# The question these logs exist to answer is what the iterates *did*: diverge
# immediately, or oscillate between feasible and infeasible before stalling, and whether
# any late iterate was nearly feasible. An exit code cannot distinguish those, and they
# call for different fixes.
# Marks a solve that the solver declared a failure but whose returned iterate violates
# no constraint. One constant, used both to build the info string and to detect it, so
# the producer and the consumer cannot drift apart.
TRAJOPT_UNCERTIFIED_TAG = "[feasible, optimality not certified]"


def trajopt_uncertified(info: Optional[str]) -> bool:
    """True if this trajopt info string describes a feasible-but-uncertified solve."""
    return bool(info) and TRAJOPT_UNCERTIFIED_TAG in info


TRAJOPT_SOLVER_LOG_DIR: Optional[str] = None
_solver_log_counter = itertools.count()


def _solver_log_path(solver: str, label: str) -> Optional[str]:
    """Unique log path for one solve, or None when logging is off."""
    if not TRAJOPT_SOLVER_LOG_DIR:
        return None
    os.makedirs(TRAJOPT_SOLVER_LOG_DIR, exist_ok=True)
    n = next(_solver_log_counter)
    return os.path.join(TRAJOPT_SOLVER_LOG_DIR,
                        f"{n:04d}_{label}_{solver}_pid{os.getpid()}.log")


def _ipopt_aggressive_options(time_limit: float, optimality_tolerance: float,
                              label: str = "solve"):
    """IPOPT options tuned to *hold on to feasibility* on these problems.

    IPOPT strays too far from feasibility and gets
    stuck on this class of problem with its default filter line search. The
    penalty line search keeps it near the feasible set at the cost of worse
    globalization, which is the trade we want when the initial guess is already
    feasible and we mainly need not to lose it.
    """
    from pydrake.all import CommonSolverOption, IpoptSolver, SolverOptions

    opts = SolverOptions()
    opts.SetOption(CommonSolverOption.kPrintToConsole, False)
    ip = IpoptSolver().solver_id()
    log = _solver_log_path("ipopt", label)
    if log:
        # print_level governs the *console*; what lands in the file is governed by
        # file_print_level, which Drake does not set. Without it the file gets the
        # problem-size header and the final summary but not the per-iteration table --
        # and that table (its inf_pr column especially) is the only record of whether
        # the iterates stayed feasible, drifted, or oscillated.
        #
        # print_user_options echoes back the options IPOPT actually received, which is
        # how we verify the aggressive settings below are in force rather than assumed.
        opts.SetOption(CommonSolverOption.kPrintFileName, log)
        opts.SetOption(ip, "print_level", 5)
        opts.SetOption(ip, "file_print_level", 5)
        opts.SetOption(ip, "print_frequency_iter", 1)
        opts.SetOption(ip, "print_user_options", "yes")
    else:
        opts.SetOption(ip, "print_level", 0)
    opts.SetOption(ip, "max_wall_time", float(time_limit))
    # Accept once feasible and the cost stops improving materially.
    opts.SetOption(ip, "acceptable_tol", max(optimality_tolerance, 1e-2))
    opts.SetOption(ip, "acceptable_iter", 5)
    opts.SetOption(ip, "acceptable_constr_viol_tol", 1e-6)
    # Keep IPOPT from relaxing the constraint bounds to buy progress.
    opts.SetOption(ip, "bound_relax_factor", 1e-12)
    opts.SetOption(ip, "honor_original_bounds", "yes")
    # The reason these options exist at all.
    opts.SetOption(ip, "line_search_method", "penalty")
    # Works around a Drake/IPOPT interaction; see main.ipynb.
    opts.SetOption(ip, "watchdog_shortened_iter_trigger", 0)
    return opts


def _snopt_options(time_limit: float, optimality_tolerance: float,
                   label: str = "solve"):
    from pydrake.all import CommonSolverOption, SnoptSolver, SolverOptions

    opts = SolverOptions()
    opts.SetOption(CommonSolverOption.kPrintToConsole, False)
    sn = SnoptSolver().solver_id()
    log = _solver_log_path("snopt", label)
    if log:
        # Major print level 1 gives one line per major iteration, including the
        # Feasibl and Optimal columns -- the per-iteration constraint violation.
        opts.SetOption(CommonSolverOption.kPrintFileName, log)
        opts.SetOption(sn, "Major print level", 1)
        opts.SetOption(sn, "Minor print level", 0)
    # "Timing level" must be enabled for "Time Limit" to have any effect at all
    # Without it the limit is silently inert and a
    # stuck solve runs unbounded -- a direct cause of the planner's occasional
    # multi-minute stalls.
    opts.SetOption(sn, "Timing level", 3)
    opts.SetOption(sn, "Time Limit", float(time_limit))
    opts.SetOption(sn, "Major optimality tolerance", optimality_tolerance)
    # The MINOR tolerance is deliberately NOT tied to optimality_tolerance. It governs
    # the QP subproblem that produces each major iteration's search direction, so
    # loosening it does not buy a looser stopping test -- it degrades the direction and
    # the solve burns major iterations without converging. Measured: tying both to 1e-1
    # took grid point 0 from 119 s to 390 s of planning (lift trajopt 20 s -> 218 s)
    # and exited SNOPT 31 (iteration limit) twice, which is the opposite of the intent.
    # Loosening the major tolerance alone is the knob that trades path quality for
    # runtime; this one only trades away convergence.
    opts.SetOption(sn, "Minor optimality tolerance", SNOPT_MINOR_OPTIMALITY_TOL)
    opts.SetOption(sn, "Major feasibility tolerance", 1e-4)
    return opts


def _solve_trajopt_prog(prog, *, label: str, time_limit: float,
                        optimality_tolerance: float, guess_feasible: bool):
    """Solve a trajopt program, escalating on failure.

    SNOPT first (it needs no option fiddling to behave on these problems). If it
    fails *and the initial guess was feasible*, retry with IPOPT on the
    feasibility-preserving options: a solver handed a feasible guess and losing
    it is a solver problem worth a second opinion, whereas a solver handed an
    infeasible guess failing is expected and re-solving will not fix it.

    **A solver's failure code is not the same as an unusable answer.** SNOPT
    routinely stops on a resource limit (exit 34) or on numerical difficulties
    (exit 41) while holding an iterate that satisfies every constraint of the
    program -- its own log reports ``Nonlinear constraint violn 0.0E+00`` and our
    independent ``_constraint_violations`` agrees. Measured on grid point 18's
    lift, both codes came back with a fully feasible iterate. The exit code
    reports whether optimality was *certified*, and discarding a feasible
    trajectory for want of a certificate is what previously turned solver
    stalls into leg failures and then into re-rolled seeds and branches.

    So a failed solve whose returned point is feasible is accepted, and the
    trajectory still has to pass the dense ``verify_trajectory`` gate downstream
    like any other -- that check, not the solver's certificate, is what decides
    whether a leg ships. The acceptance is announced so it stays visible rather
    than becoming a silent lowering of the bar.

    Returns ``(result, info)``.
    """
    from pydrake.all import IpoptSolver, SnoptSolver

    with record("trajopt.solve", label, solver="SNOPT", n_vars=prog.num_vars()):
        result = SnoptSolver().Solve(
            prog, None, _snopt_options(time_limit, optimality_tolerance, label))
        mark(info=result.get_solver_details().info,
             success=bool(result.is_success()))
    info = f"SNOPT ({result.get_solver_details().info})"
    if result.is_success():
        return result, info, True

    print(f"[trajopt {label}] {info} failed")
    with record("trajopt.check", label):
        viols = _constraint_violations(prog, result.GetSolution())
    for desc, m in viols[:5]:
        print(f"[trajopt {label}]   violated: {desc} by {m:.2e}")
    if not viols:
        print(f"[trajopt {label}] {info} did not certify optimality, but its "
              f"returned iterate satisfies every constraint -- accepting it "
              f"(the dense check still gates the leg)")
        return result, f"{info} {TRAJOPT_UNCERTIFIED_TAG}", True

    if not guess_feasible:
        # Expected: the guess was already infeasible. Escalating would just spend
        # another time_limit rediscovering that.
        return result, info + " [guess was infeasible]", False

    print(f"[trajopt {label}] guess was feasible -- retrying with IPOPT "
          f"(penalty line search)")
    with record("trajopt.solve_retry", label, solver="IPOPT"):
        ip_result = IpoptSolver().Solve(
            prog, None, _ipopt_aggressive_options(time_limit, optimality_tolerance, label))
        mark(info=ip_result.get_solver_details().status,
             success=bool(ip_result.is_success()))
    ip_info = f"IPOPT ({ip_result.get_solver_details().status})"
    if ip_result.is_success():
        return ip_result, ip_info, True
    with record("trajopt.check", label):
        ip_viols = _constraint_violations(prog, ip_result.GetSolution())
    if not ip_viols:
        print(f"[trajopt {label}] {ip_info} did not certify optimality, but its "
              f"iterate satisfies every constraint -- accepting it")
        return ip_result, f"{ip_info} {TRAJOPT_UNCERTIFIED_TAG}", True
    print(f"[trajopt {label}] {ip_info} also failed")
    return result, f"{info} then {ip_info}", False


# ── Trajopt: reach (23-D) ─────────────────────────────────────────────────────

def _reach_trajopt(
    path: list,
    initial_traj,
    setup: _Setup,
    *,
    min_dist_margin: float,
    n_constr_pts: int,
    time_limit: float,
    support_polygon_inset: float = 0.0,
    optimality_tolerance: float = 1e-3,
    min_dist_lower_bound: float = 0.001,
):
    """KinematicTrajectoryOptimization in the 23-D joint space.

    ``min_dist_margin`` sets only the width of the smooth collision-penalty zone;
    ``min_dist_lower_bound`` is the hard floor the solver must satisfy. Keeping
    them separate matters because the floor has to sit at or below the clearance
    the *endpoints* actually achieve, or the problem is infeasible at s=0 or s=1
    and the solver returns an "infeasibilities minimized" iterate that can be far
    worse than the collision-free path it started from.

    Returns ``(traj, success, info)``.
    """
    from pydrake.all import (
        AutoDiffXd, ExtractValue,
        KinematicTrajectoryOptimization, MinimumDistanceLowerBoundConstraint,
        PyFunctionConstraint,
    )
    from rby1_opt_ik import _com_support_polygon_residuals, support_polygon_xyzs

    p = setup.plant
    L = setup.layout
    pos_idxs = np.asarray(setup.pos_idxs_23)

    rg_inst   = p.GetModelInstanceByName("right_gripper")
    lg_inst   = p.GetModelInstanceByName("left_gripper")
    head_inst = p.GetModelInstanceByName("head")

    with record("trajopt.setup", "reach"):
        p_ad     = p.ToAutoDiffXd()
        base_ad  = p_ad.GetModelInstanceByName("base")
        torso_ad = p_ad.GetModelInstanceByName("torso")
        right_ad = p_ad.GetModelInstanceByName("right_arm")
        left_ad  = p_ad.GetModelInstanceByName("left_arm")
        rg_ad    = p_ad.GetModelInstanceByName("right_gripper")
        lg_ad    = p_ad.GetModelInstanceByName("left_gripper")
        head_ad  = p_ad.GetModelInstanceByName("head")

        coll_diag_ctx  = setup.diagram.CreateDefaultContext()
        coll_plant_ctx = p.GetMyContextFromRoot(coll_diag_ctx)
        stab_ctx       = p.CreateDefaultContext()
        stab_ctx_ad    = p_ad.CreateDefaultContext()

    # Take the floor from what the guess actually clears -- see
    # _feasible_min_dist_bound. Measured on the path's own waypoints, which is
    # what the B-spline guess tracks.
    measure_ctx = p.GetMyContextFromRoot(setup.diagram.CreateDefaultContext())

    def _q_full_of(q23):
        q_f = setup.q_full_default.copy()
        q_f[pos_idxs] = np.asarray(q23, dtype=float)
        return q_f

    with record("trajopt.bound", "reach"):
        min_dist_lower_bound = _feasible_min_dist_bound(
            min_dist_lower_bound,
            _path_min_clearance([_q_full_of(q) for q in path], p, measure_ctx),
            "reach",
        )

    min_dist_constr = MinimumDistanceLowerBoundConstraint(
        p, min_dist_lower_bound, coll_plant_ctx, None,
        max(min_dist_margin - min_dist_lower_bound, 1e-3),
    )
    n_poly = len(support_polygon_xyzs)

    def q23_to_q_plant(q23):
        is_ad = isinstance(q23.flat[0], AutoDiffXd)
        q_f = setup.q_full_default.copy()
        if is_ad:
            q_f[pos_idxs] = ExtractValue(q23).flatten()
            n_d = q23[0].derivatives().size
            q_plant = np.array([AutoDiffXd(v, np.zeros(n_d)) for v in q_f])
            for j, idx in enumerate(pos_idxs):
                q_plant[idx] = q23[j]
            return q_plant
        else:
            q_f[pos_idxs] = q23
            return q_f

    def min_dist_callable(q23):
        return min_dist_constr.Eval(q23_to_q_plant(q23))

    def stability_callable(q23):
        is_ad = isinstance(q23.flat[0], AutoDiffXd)
        if is_ad:
            if not np.all(np.isfinite(ExtractValue(q23))):
                return np.array([AutoDiffXd(np.inf, np.zeros(q23[0].derivatives().size))] * n_poly)
            p_ad.SetPositions(stab_ctx_ad, base_ad,  q23[L.base])
            p_ad.SetPositions(stab_ctx_ad, torso_ad, q23[L.torso])
            p_ad.SetPositions(stab_ctx_ad, right_ad, q23[L.right_arm])
            p_ad.SetPositions(stab_ctx_ad, left_ad,  q23[L.left_arm])
            com = p_ad.CalcCenterOfMassPositionInWorld(
                stab_ctx_ad,
                [base_ad, torso_ad, right_ad, left_ad, rg_ad, lg_ad, head_ad],
            )
        else:
            if not np.all(np.isfinite(q23)):
                return np.full(n_poly, np.inf)
            p.SetPositions(stab_ctx, setup.base_inst,  q23[L.base])
            p.SetPositions(stab_ctx, setup.torso_inst, q23[L.torso])
            p.SetPositions(stab_ctx, setup.right_inst, q23[L.right_arm])
            p.SetPositions(stab_ctx, setup.left_inst,  q23[L.left_arm])
            com = p.CalcCenterOfMassPositionInWorld(
                stab_ctx,
                [setup.base_inst, setup.torso_inst, setup.right_inst, setup.left_inst,
                 rg_inst, lg_inst, head_inst],
            )
        return _com_support_polygon_residuals(com[:2], q23[L.base])

    min_dist_constraint = PyFunctionConstraint(
        23, min_dist_callable, [-np.inf], [1.0], "reach_min_dist",
    )
    stability_constraint = PyFunctionConstraint(
        23, stability_callable,
        np.full(n_poly, -np.inf), _stability_constraint_ub(support_polygon_inset),
        "reach_stability",
    )

    t0, t1 = initial_traj.start_time(), initial_traj.end_time()

    def build(pth, multiplicity):
        bspline = _path_to_bspline(pth, t0, t1, lb=setup.q_lb, ub=setup.q_ub,
                                   multiplicity=multiplicity)
        trajopt = KinematicTrajectoryOptimization(bspline)
        trajopt.AddPositionBounds(setup.q_lb, setup.q_ub)
        trajopt.AddPathPositionConstraint(bspline.value(bspline.start_time()),
                                          bspline.value(bspline.start_time()), 0)
        trajopt.AddPathPositionConstraint(bspline.value(bspline.end_time()),
                                          bspline.value(bspline.end_time()), 1)
        # Label each per-s binding: Drake names them all "path position constraint at
        # s=..." otherwise, which makes a failure report say *where* it failed but not
        # *what* failed -- and min-distance vs stability call for opposite fixes.
        for s in np.linspace(0, 1, n_constr_pts):
            for c, nm in ((min_dist_constraint, "reach_min_dist"),
                          (stability_constraint, "reach_stability")):
                b = trajopt.AddPathPositionConstraint(c, s)
                try:
                    b.evaluator().set_description(f"{nm}@s={s:.3f}")
                except AttributeError:
                    pass
        trajopt.AddPathEnergyCost()
        return trajopt

    # optimality_tolerance defaults to 1e-3 (matches _lift_trajopt's tolerance):
    # the originally-relaxed 1e-1 was intended to keep SNOPT from spending minutes
    # micro-optimizing in the 23-DOF null-space, but empirically (grid point 16's
    # reach leg) it instead let SNOPT settle into a substantially worse local
    # optimum -- trajopt ballooned the shortcut path's length by +132% at 1e-1,
    # vs. +45% at 1e-3, while also solving ~15x faster (0.3s vs 4.9s). Tolerances
    # from 1e-4 to 5e-2 all converged to the same solution in that test.
    # Program construction (one KinematicTrajectoryOptimization per rung of the
    # multiplicity ladder, each registering 2 * n_constr_pts bindings) plus the
    # feasibility probe of the guess. Separated from the solve because it is the
    # part a formulation change moves, and the part a solver change does not.
    with record("trajopt.guess", "reach"):
        trajopt, guess_feasible, _ = _choose_guess(build, path, "reach")
        mark(guess_feasible=bool(guess_feasible))
    result, info, usable = _solve_trajopt_prog(
        trajopt.prog(), label="reach", time_limit=time_limit,
        optimality_tolerance=optimality_tolerance, guess_feasible=guess_feasible,
    )
    with record("trajopt.reconstruct", "reach"):
        traj = trajopt.ReconstructTrajectory(result)
    return traj, usable, info


# ── Trajopt: lift (14-D) ──────────────────────────────────────────────────────

# Output layout of the merged constrained-leg feasibility constraint, as
# (family, start, stop) half-open slices into its stacked output vector.
#
# The four families -- collision clearance, CoM stability, arm joint limits and the
# end-effector residual -- used to be four separate PyFunctionConstraints, and every
# one of them called the analytic IK on the *same* 14-D state. At n_constr_pts=50 that
# is 200 bindings and 200 AutoDiff IK solves per constraint sweep where 50 suffice,
# on the legs whose trajopt is 61% of the pipeline's runtime. Merged, one IK solve
# feeds all four, and one SetPositions feeds both the CoM and the FK residual (which
# previously wrote the same context twice). Modelled on FullFeasibilityConstraint in
# cpp_parameterization/cpp/iiwa_ik/constraints.cc, which stacks its parameterized
# families the same way.
#
# _constraint_violations reads this layout back to name which family a violation
# belongs to. Without it, merging would trade four precise diagnostics for one
# anonymous "lift_full_feasibility" -- and *which* family a guess violates is what the
# guess-multiplicity ladder and every past trajopt diagnosis have turned on.
_N_SUPPORT_POLYGON = len(support_polygon_xyzs)
LIFT_FEASIBILITY_DESC = "lift_full_feasibility"
LIFT_FEASIBILITY_BLOCKS = (
    ("lift_min_dist",         0,                       1),
    ("lift_stability",        1,                       1 + _N_SUPPORT_POLYGON),
    ("lift_arm_joint_limits", 1 + _N_SUPPORT_POLYGON, 15 + _N_SUPPORT_POLYGON),
    ("lift_ee_residual",     15 + _N_SUPPORT_POLYGON, 17 + _N_SUPPORT_POLYGON),
    # Held-box-vs-table only. Always present so the block layout is fixed; when
    # no pair bound is asked for it evaluates a constant that trivially holds.
    ("lift_pair_min_dist",   17 + _N_SUPPORT_POLYGON, 18 + _N_SUPPORT_POLYGON),
)
LIFT_FEASIBILITY_N_OUT = LIFT_FEASIBILITY_BLOCKS[-1][2]


def _restricted_min_distance_constraint(setup, bound, a_match, b_match,
                                        influence_offset=0.01):
    """MinimumDistanceLowerBoundConstraint seeing *only* the a_match<->b_match pairs.

    ``MinimumDistanceLowerBoundConstraint`` applies one scalar bound to every
    candidate pair, so it cannot express "5 cm between these two things, 1 mm
    elsewhere" -- and its bound is dragged down to whatever the tightest pair in
    the initial guess allows. The way to aim it at one pair is not a different
    constraint but a different *context*: collision filters are context-level
    state, so a private diagram context with everything filtered out except the
    pairs of interest yields a constraint that measures only those.

    Verified: filtering takes the candidate set from 6060 pairs to 5 (the held
    box's five geometries against the one table geometry), and the constraint
    evaluates differently from the unrestricted one wherever some other pair is
    the global minimum -- i.e. the filter is honoured at evaluation time, not
    baked in at construction from the unfiltered model.

    ``a_match``/``b_match`` are substrings matched against geometry names.
    Returns ``(constraint, plant_context)``; the context must be kept alive for
    as long as the constraint is used.
    """
    from pydrake.geometry import CollisionFilterDeclaration, GeometrySet
    from pydrake.multibody.inverse_kinematics import (
        MinimumDistanceLowerBoundConstraint)

    sg = setup.diagram.GetSubsystemByName("scene_graph")
    insp = sg.model_inspector()
    every, a_ids, b_ids = [], [], []
    for gid in insp.GetAllGeometryIds():
        if insp.GetProximityProperties(gid) is None:
            continue
        every.append(gid)
        name = insp.GetName(gid)
        if a_match in name:
            a_ids.append(gid)
        if b_match in name:
            b_ids.append(gid)
    if not a_ids or not b_ids:
        raise ValueError(
            f"_restricted_min_distance_constraint: no geometries matched "
            f"{a_match!r} ({len(a_ids)}) / {b_match!r} ({len(b_ids)})")

    ctx = setup.diagram.CreateDefaultContext()
    mgr = sg.collision_filter_manager(sg.GetMyContextFromRoot(ctx))
    mgr.Apply(CollisionFilterDeclaration().ExcludeWithin(GeometrySet(every)))
    mgr.Apply(CollisionFilterDeclaration().AllowBetween(
        GeometrySet(a_ids), GeometrySet(b_ids)))
    plant_ctx = setup.plant.GetMyContextFromRoot(ctx)
    return (MinimumDistanceLowerBoundConstraint(
        setup.plant, bound, plant_ctx, None, influence_offset), ctx)


def _lift_trajopt(
    path: list,
    initial_traj,
    setup: _Setup,
    state_to_lumped: Callable,
    lift_gcp: np.ndarray,
    s_lb: np.ndarray,
    s_ub: np.ndarray,
    *,
    min_dist_margin: float,
    n_constr_pts: int,
    time_limit: float,
    support_polygon_inset: float = 0.0,
    min_dist_lower_bound: float = 0.001,
    optimality_tolerance: float = 1e-2,
    ee_residual_threshold: float = 1e-2,
    joint_space_energy_cost: bool = True,
    pair_min_distance: Optional[float] = None,
    pair_geometries: Tuple[str, str] = ("held_box", "table"),
):
    """KinematicTrajectoryOptimization in the 14-D constrained state space.

    See ``_reach_trajopt`` for the ``min_dist_margin`` / ``min_dist_lower_bound``
    distinction. ``ee_residual_threshold`` bounds the per-arm FK residual that
    keeps the solver inside the reachable set -- see ``lift_ee_residual_callable``.
    ``joint_space_energy_cost`` picks the objective: see ``joint_path_energy_cost``
    below for why the joint-space one is the right default and what it costs.
    Returns ``(traj, success, info)``.
    """
    from pydrake.all import (
        AutoDiffXd, ExtractValue,
        KinematicTrajectoryOptimization, MinimumDistanceLowerBoundConstraint,
        PyFunctionConstraint, RigidTransform_, RollPitchYaw_,
    )
    from rby1_opt_ik import _com_support_polygon_residuals, support_polygon_xyzs

    p = setup.plant
    L = setup.layout
    pos_idxs = np.asarray(setup.pos_idxs_23)

    rg_inst   = p.GetModelInstanceByName("right_gripper")
    lg_inst   = p.GetModelInstanceByName("left_gripper")
    head_inst = p.GetModelInstanceByName("head")

    with record("trajopt.setup", "lift"):
        p_ad     = p.ToAutoDiffXd()
        base_ad  = p_ad.GetModelInstanceByName("base")
        torso_ad = p_ad.GetModelInstanceByName("torso")
        right_ad = p_ad.GetModelInstanceByName("right_arm")
        left_ad  = p_ad.GetModelInstanceByName("left_arm")
        rg_ad    = p_ad.GetModelInstanceByName("right_gripper")
        lg_ad    = p_ad.GetModelInstanceByName("left_gripper")
        head_ad  = p_ad.GetModelInstanceByName("head")

        col_diag_ctx  = setup.diagram.CreateDefaultContext()
        col_plant_ctx = p.GetMyContextFromRoot(col_diag_ctx)
        stab_ctx      = p.CreateDefaultContext()
        stab_ctx_ad   = p_ad.CreateDefaultContext()

    # Trajopt uses Rby1IK with boundary_fallback=True (smooth gradients near
    # workspace boundary); validity uses False (hard NaN on infeasibility).
    #
    # "Smooth gradients" was only half true until the damping was set. With
    # lambda=0 the boundary branch solves an undamped least-squares problem
    # against del_reef_del_qright_aug, which is singular exactly where the
    # boundary fallback engages -- so the constraint callables below were handing
    # SNOPT a pseudo-inverse of a rank-deficient Jacobian and calling it a
    # gradient. See Rby1IK.__init__ for the measured cost of leaving it at zero.
    lift_ik = Rby1IK(boundary_fallback=True,
                     boundary_damping_lambda=TRAJOPT_BOUNDARY_DAMPING)

    # Floor taken from the guess's own clearance -- see _feasible_min_dist_bound.
    # The constrained legs need this more than the reach does: the two arms hold a
    # 386 mm box between them, so they pass within a third of a millimetre by
    # construction and no fixed floor suits both legs.
    measure_ctx = p.GetMyContextFromRoot(setup.diagram.CreateDefaultContext())

    def _q_full_of_s14(s14):
        q23 = lift_ik.compute_ik(state_to_lumped(np.asarray(s14, dtype=float)),
                                 right_gcp=lift_gcp, left_gcp=lift_gcp)
        if q23 is None:
            return None
        q_f = setup.q_full_default.copy()
        q_f[pos_idxs] = np.asarray(q23, dtype=float)
        return q_f

    with record("trajopt.bound", "constrained"):
        min_dist_lower_bound = _feasible_min_dist_bound(
            min_dist_lower_bound,
            _path_min_clearance([_q_full_of_s14(s) for s in path], p, measure_ctx),
            # NOT "lift": this function serves both constrained legs, so labelling
            # every clamp as the lift's actively misleads. It read as "place is never
            # clamped" for a whole debugging session while place was in fact being
            # clamped from 20 mm to 2.52 mm. Naming the specific leg would mean
            # threading it down from constrained_plan's caller; until then, say what
            # is actually known.
            "constrained",
        )

    min_dist_constr = MinimumDistanceLowerBoundConstraint(
        p, min_dist_lower_bound, col_plant_ctx, None,
        max(min_dist_margin - min_dist_lower_bound, 1e-3),
    )

    # A second, pair-restricted bound. Deliberately NOT run through
    # _feasible_min_dist_bound: that clamp exists because a *global* bound above
    # the guess's clearance makes the program infeasible at its start, and the
    # guess's global clearance is set by whatever pair happens to be tightest
    # (arm-vs-arm, within 0.33 mm by construction). This bound sees only the held
    # box against the table, which the BiRRT is free to keep well clear of, so it
    # can be asked for outright rather than negotiated down to the guess.
    pair_constr = pair_ctx = None
    if pair_min_distance:
        pair_constr, pair_ctx = _restricted_min_distance_constraint(
            setup, pair_min_distance, pair_geometries[0], pair_geometries[1])

    n_poly = len(support_polygon_xyzs)

    def _embed_lift_q23(q23, is_ad):
        """Place a 23-D active-DOF solution into a full-plant position vector.

        Split out of lift_parameterization so a caller that has already solved the
        IK for this state can embed *that* solution instead of solving it again.
        lift_ee_residual_callable did exactly that -- one compute_ik of its own, then
        another inside lift_parameterization on the same s14 -- which is two of the
        five AutoDiff IK solves the four constraint families spend per constrained
        sample. At ~0.5 ms per AD solve and 50 samples a sweep, that duplicate alone
        was ~25 ms of every constraint sweep for no new information.
        """
        q_def = setup.q_full_default.copy()
        if is_ad:
            q_def[pos_idxs] = ExtractValue(q23).flatten()
            n_d = q23[0].derivatives().size
            q_plant = np.array([AutoDiffXd(v, np.zeros(n_d)) for v in q_def])
            for j, idx in enumerate(pos_idxs):
                q_plant[idx] = q23[j]
            return q_plant
        q_def[pos_idxs] = q23
        return q_def

    # Cached frames: GetFrameByName is a name lookup, and the FK residual below runs
    # it twice per constrained sample per solver evaluation.
    ee_right_ad = p_ad.GetFrameByName("ee_right")
    ee_left_ad  = p_ad.GetFrameByName("ee_left")
    ee_right    = p.GetFrameByName("ee_right")
    ee_left     = p.GetFrameByName("ee_left")

    def lift_full_feasibility_callable(s14):
        """All four constrained-leg feasibility families from one analytic IK solve.

        Stacked output, laid out by ``LIFT_FEASIBILITY_BLOCKS``:

            [0]     collision clearance   (min_dist_constr, 1)
            [1:5]   CoM stability         (support-polygon residuals, n_poly)
            [5:19]  arm joint limits      (right then left, 14)
            [19:21] end-effector residual (per-arm FK error, 2)

        Each family used to be its own PyFunctionConstraint calling
        ``lift_ik.compute_ik`` on the same ``s14`` -- four AutoDiff IK solves per
        constrained sample where one suffices, and the IK dominates the evaluation.
        Everything below is the union of those four callables with the redundant
        solves and the duplicate ``SetPositions`` removed; the bounds, the residual
        formulas and the infeasibility signalling are unchanged, so this is a
        speed change and not a model change.

        On the end-effector residual specifically (unchanged, but the reason it
        exists is easy to mistake for redundancy): the 14-D parameterization is an
        exact encoding of the end-effector constraint only *within* the reachable
        set. ``lift_ik`` is deliberately built with boundary_fallback=True because
        trajopt needs smooth gradients near the workspace boundary, so on an
        out-of-reach request compute_ik returns a nearby *feasible* configuration
        whose FK does not reach the requested pose. Without this residual trajopt
        would optimise happily against a state that no longer means what the
        constraint says it means, with nothing detecting it.
        """
        is_ad = isinstance(s14.flat[0], AutoDiffXd)
        n_d = s14[0].derivatives().size if is_ad else 0

        def infeasible():
            """Every family unevaluable, signalled exactly as the four callables did.

            Note this is the ``inf``-barrier-with-no-gradient behaviour that strangles
            both line searches (see the notes on _solve_trajopt_prog); it is preserved
            verbatim here so that merging the bindings changes only their number.
            Replacing it with a large finite penalty is a separate change.
            """
            if is_ad:
                return np.array([AutoDiffXd(np.inf, np.zeros(n_d))]
                                * LIFT_FEASIBILITY_N_OUT)
            return np.full(LIFT_FEASIBILITY_N_OUT, np.inf)

        lumped = state_to_lumped(s14)
        q23 = lift_ik.compute_ik(lumped, right_gcp=lift_gcp, left_gcp=lift_gcp)
        if q23 is None:
            return infeasible()
        if not np.all(np.isfinite(ExtractValue(q23) if is_ad else q23)):
            return infeasible()

        # One embed, one SetPositions, shared by the clearance, CoM and FK families.
        # q_plant is q23 copied into q_full_default, so q23 finite => q_plant finite
        # and the old per-family finiteness re-checks are subsumed by the one above.
        q_plant = _embed_lift_q23(q23, is_ad)
        if is_ad:
            p_ad.SetPositions(stab_ctx_ad, q_plant)
            com = p_ad.CalcCenterOfMassPositionInWorld(
                stab_ctx_ad,
                [base_ad, torso_ad, right_ad, left_ad, rg_ad, lg_ad, head_ad],
            )
            X_r_fk = ee_right_ad.CalcPoseInWorld(stab_ctx_ad).GetAsMatrix4()
            X_l_fk = ee_left_ad.CalcPoseInWorld(stab_ctx_ad).GetAsMatrix4()
            X_r_des = RigidTransform_[AutoDiffXd](
                RollPitchYaw_[AutoDiffXd](lumped[12:15]), lumped[9:12]).GetAsMatrix4()
            X_l_des = RigidTransform_[AutoDiffXd](
                RollPitchYaw_[AutoDiffXd](lumped[19:22]), lumped[16:19]).GetAsMatrix4()
        else:
            p.SetPositions(stab_ctx, q_plant)
            com = p.CalcCenterOfMassPositionInWorld(
                stab_ctx,
                [setup.base_inst, setup.torso_inst, setup.right_inst, setup.left_inst,
                 rg_inst, lg_inst, head_inst],
            )
            X_r_fk = ee_right.CalcPoseInWorld(stab_ctx).GetAsMatrix4()
            X_l_fk = ee_left.CalcPoseInWorld(stab_ctx).GetAsMatrix4()
            X_r_des = RigidTransform(
                RollPitchYaw(lumped[12:15]), lumped[9:12]).GetAsMatrix4()
            X_l_des = RigidTransform(
                RollPitchYaw(lumped[19:22]), lumped[16:19]).GetAsMatrix4()

        d_r = X_r_fk - X_r_des
        d_l = X_l_fk - X_l_des
        # The pair block reuses q_plant, so it costs one distance query and no
        # extra IK solve -- the IK dominates this callable (see its docstring).
        if pair_constr is None:
            pair_val = (np.array([AutoDiffXd(0.0, np.zeros(n_d))]) if is_ad
                        else np.zeros(1))
        else:
            pair_val = np.asarray(pair_constr.Eval(q_plant)).flatten()
        return np.concatenate([
            np.asarray(min_dist_constr.Eval(q_plant)).flatten(),
            _com_support_polygon_residuals(com[:2], q23[L.base]),
            q23[L.right_arm], q23[L.left_arm],
            np.array([100.0 * np.sum(d_r * d_r), 100.0 * np.sum(d_l * d_l)]),
            pair_val,
        ])

    lift_feasibility_constraint = PyFunctionConstraint(
        14, lift_full_feasibility_callable,
        np.concatenate([
            [-np.inf],
            np.full(n_poly, -np.inf),
            setup.q_lb[L.right_arm], setup.q_lb[L.left_arm],
            [-np.inf, -np.inf],
            [-np.inf],
        ]),
        np.concatenate([
            [1.0],
            _stability_constraint_ub(support_polygon_inset),
            setup.q_ub[L.right_arm], setup.q_ub[L.left_arm],
            [ee_residual_threshold, ee_residual_threshold],
            # Same <=1 convention as the global block: the constraint returns a
            # smoothed penalty that reaches 1 at the bound. When no pair bound is
            # requested the callable emits 0, which holds trivially.
            [1.0],
        ]),
        LIFT_FEASIBILITY_DESC,
    )

    t0, t1 = initial_traj.start_time(), initial_traj.end_time()

    def joint_path_energy_cost(trajopt):
        """sum ||dq_23||^2 between consecutive control points, through analytic IK.

        Path energy in *joint* space rather than in the 14-D lumped state space.
        This is the quantity that decides whether the motion looks smooth: the map
        from the lumped state to joint angles is nonlinear and near-singular in
        places, so a state-space-optimal path is not a joint-space-smooth one.
        Measured on grid point 5's lift, both at multiplicity 3 and identical
        verified clearance (+50.00 mm):

            AddPathEnergyCost (14-D):   joint path 2.32 rad, 47 kinks, max turn 30.7 deg
            this cost (joint space):    joint path 1.97 rad,  0 kinks, max turn  1.9 deg

        It costs one analytic IK per control point per evaluation, under AutoDiff
        whose width is the whole decision vector, so it is roughly quadratic in the
        control-point count. That is why it was gated off: at multiplicity 3 it
        measured 22 ms per evaluation and no time limit tried (30 s, 120 s) was
        enough. Two later changes made it affordable -- the SNOPT minor-optimality
        fix, which cut a lift solve from 20.2 s to 5.6 s, and the multiplicity-1
        guess, which cuts the control-point count (and hence the IK calls) by 3x.
        Re-measured on point 5's lift at multiplicity 1: 17 s, well inside
        CONSTRAINED_TRAJOPT_TIME_LIMIT.
        """
        n_s = trajopt.num_positions()
        n_ctrl = trajopt.num_control_points()

        def cost(control_points_flat):
            is_ad = isinstance(control_points_flat.flat[0], AutoDiffXd)
            cp_mat = control_points_flat.reshape(n_s, n_ctrl)

            # Initialize total cost with the correct type (AutoDiffXd or float)
            total = control_points_flat[0] * 0.0

            def safe_extract(q):
                if q is None:
                    return False
                v = ExtractValue(q) if is_ad else q
                return np.all(np.isfinite(v))

            q_prev = lift_ik.compute_ik(state_to_lumped(cp_mat[:, 0]),
                                        right_gcp=lift_gcp, left_gcp=lift_gcp)
            if not safe_extract(q_prev):
                return control_points_flat[0] * 0.0 + np.inf

            for i in range(1, n_ctrl):
                q_curr = lift_ik.compute_ik(state_to_lumped(cp_mat[:, i]),
                                            right_gcp=lift_gcp, left_gcp=lift_gcp)
                if not safe_extract(q_curr):
                    return control_points_flat[0] * 0.0 + np.inf

                delta = q_curr - q_prev
                # delta * delta, not delta ** 2: numpy's object-dtype `power` path
                # emits "RuntimeWarning: divide by zero encountered in power" on every
                # call here for AutoDiffXd operands. The product is identical and
                # warning-free.
                total = total + np.sum(delta * delta)
                q_prev = q_curr

            return total

        return cost

    def build(pth, multiplicity):
        bspline = _path_to_bspline(pth, t0, t1, lb=s_lb, ub=s_ub,
                                   multiplicity=multiplicity)
        trajopt = KinematicTrajectoryOptimization(bspline)
        trajopt.AddPositionBounds(s_lb, s_ub)
        trajopt.AddPathPositionConstraint(bspline.value(bspline.start_time()),
                                          bspline.value(bspline.start_time()), 0)
        trajopt.AddPathPositionConstraint(bspline.value(bspline.end_time()),
                                          bspline.value(bspline.end_time()), 1)
        # Labelled per-s bindings -- see the same loop in _reach_trajopt. One binding
        # per sample now, not four: the four feasibility families share a single
        # analytic IK solve inside lift_full_feasibility_callable, and
        # _constraint_violations splits the stacked output back into per-family
        # diagnostics via LIFT_FEASIBILITY_BLOCKS.
        for sv in np.linspace(0, 1, n_constr_pts):
            b = trajopt.AddPathPositionConstraint(lift_feasibility_constraint, sv)
            try:
                b.evaluator().set_description(f"{LIFT_FEASIBILITY_DESC}@s={sv:.3f}")
            except AttributeError:
                pass
        if joint_space_energy_cost:
            trajopt.prog().AddCost(
                joint_path_energy_cost(trajopt),
                trajopt.control_points().flatten(),
                "lift_joint_path_energy",
            )
        else:
            # Drake's native path-energy cost, evaluated in C++ over the 14-D state.
            trajopt.AddPathEnergyCost()
        return trajopt

    with record("trajopt.guess", "lift"):
        trajopt, guess_feasible, _ = _choose_guess(build, path, "lift")
        mark(guess_feasible=bool(guess_feasible))
    result, info, usable = _solve_trajopt_prog(
        trajopt.prog(), label="lift", time_limit=time_limit,
        optimality_tolerance=optimality_tolerance, guess_feasible=guess_feasible,
    )
    with record("trajopt.reconstruct", "lift"):
        traj = trajopt.ReconstructTrajectory(result)
    return traj, usable, info


# ── TOPPRA retiming ───────────────────────────────────────────────────────────

def _toppra_q23(
    traj_q23, setup: _Setup, min_points: int, max_iter: int,
    velocity_scale: float = 1.0, acceleration_scale: float = 1.0,
) -> PathParameterizedTrajectory:
    """TOPPRA for a trajectory already in 23-D joint space.

    Embeds into the full plant via StackedTrajectory so EvalDerivative runs in C++.
    """
    full = _embed_q23_into_full_traj(traj_q23, setup)
    return _run_toppra(full, setup, min_points, max_iter, velocity_scale, acceleration_scale)


def _toppra_lift(
    traj_14, setup: _Setup, lift_ik, state_to_lumped: Callable,
    lift_gcp: np.ndarray, min_points: int, max_iter: int,
    velocity_scale: float = 1.0, acceleration_scale: float = 1.0,
) -> PathParameterizedTrajectory:
    """TOPPRA for the constrained lift trajectory.

    First derivative: AutoDiff via the IK inverse function theorem (one IK call
    per evaluation, exact). Second derivative: 5-point central-difference *first*-
    derivative stencil applied to that first-derivative function (four IK calls per
    evaluation). The stencil differentiates q̇ once, not twice -- differentiating it
    twice yields jerk, which is what this used to compute.
    """
    from pydrake.all import ExtractGradient, InitializeAutoDiff

    nq = setup.plant.num_positions()
    pos_idxs = setup.pos_idxs_23
    q_fallback23 = setup.q_full_default[pos_idxs]
    t0_lift = traj_14.start_time()
    t1_lift = traj_14.end_time()

    # Counts samples where the 14-D trajectory left the reachable set and traj_fn
    # substituted the default configuration. Every such substitution splices the
    # robot's *home pose* into the middle of the retimed trajectory -- a joint-space
    # discontinuity that verify_trajectory cannot see, because the default pose is
    # collision-free and CoM-stable. Nothing detected this before; the resampling
    # route further down re-raises on the same event, so the two routes disagreed.
    # Reported rather than raised for now: raising here fails the leg outright, and
    # whether this fires at all is still unmeasured.
    splices = {"n": 0, "n_calls": 0, "t_first": None}

    def traj_fn(t: float):
        t_c = np.clip(t, t0_lift, t1_lift)
        s = traj_14.value(t_c).flatten()
        q23 = lift_ik.compute_ik(state_to_lumped(s), right_gcp=lift_gcp, left_gcp=lift_gcp)
        splices["n_calls"] += 1
        if q23 is None or not np.all(np.isfinite(q23)):
            splices["n"] += 1
            if splices["t_first"] is None:
                splices["t_first"] = float(t_c)
            q23 = q_fallback23
        return setup.embed_q23(q23).reshape(-1, 1)

    def deriv1(t: float):
        t_c = np.clip(t, t0_lift, t1_lift)
        s = traj_14.value(t_c).flatten()
        ds_dt = traj_14.EvalDerivative(t_c, 1).flatten()
        s_ad = InitializeAutoDiff(s, ds_dt.reshape(-1, 1)).flatten()
        q23_ad = lift_ik.compute_ik(state_to_lumped(s_ad), right_gcp=lift_gcp, left_gcp=lift_gcp)
        dq23_dt = ExtractGradient(q23_ad).flatten()
        dq_full = np.zeros(nq)
        dq_full[pos_idxs] = dq23_dt
        return dq_full.reshape(-1, 1)

    def deriv(t: float, order: int, dt: float = 1e-5):
        if order == 1:
            return deriv1(t)
        if order == 2:
            fp2 = deriv1(t + 2 * dt)
            fp1 = deriv1(t +     dt)
            fm1 = deriv1(t -     dt)
            fm2 = deriv1(t - 2 * dt)
            return (-fp2 + 8 * fp1 - 8 * fm1 + fm2) / (12 * dt)
        raise RuntimeError(f"_toppra_lift: derivative order {order} not supported")

    def _report_splices(route: str):
        if splices["n"]:
            print(f"[toppra] WARNING: the default configuration was spliced into "
                  f"{splices['n']} of {splices['n_calls']} sampled configurations "
                  f"({route} route), first at t={splices['t_first']:.4f} -- the 14-D "
                  f"trajectory leaves the reachable set there and the retimed "
                  f"trajectory contains a joint-space discontinuity that "
                  f"verify_trajectory cannot detect.")

    # The constrained legs retime through the IK mapping, as a
    # FunctionHandleTrajectory, and only that way. There is deliberately no
    # fallback here.
    #
    # There used to be one: on a RuntimeError this resampled the constrained
    # trajectory into joint space and retimed a CubicShapePreserving through
    # those samples. That is not an equivalent retiming and must not come back.
    # Every sample is a genuine IK solution, so the carry constraint holds
    # exactly *at* the knots -- but the cubic between them is interpolated in
    # joint space, where the constrained manifold is curved, so the retimed
    # trajectory leaves the manifold between knots. Nothing downstream catches
    # it: verify_trajectory checks collision, CoM and joint limits, none of
    # which see a carried box that has tilted.
    #
    # On the severity, corrected against the run's own logs: it fired 14 times
    # on the 20-point grid and **all 14 were the carry probe's lift**, not a
    # shipped leg. The probe consumes only the RRT path and
    # lift_traj.value(end_time) (an exact IK solution at the final knot), so no
    # off-manifold trajectory ever reached the robot. It is deleted because it
    # masked failures and would have shipped an off-manifold leg the first time
    # a real leg needed it -- not because it had already done so.
    #
    # A failure here is a real failure and must surface as one. The rung that
    # handles genuine numerical difficulty is the constraint relaxation in
    # _run_toppra (see TOPPRA_RELAXATION_LADDER), which perturbs the LP's
    # tolerance rather than the trajectory's representation.
    full = FunctionHandleTrajectory(traj_fn, nq, 1, t0_lift, t1_lift)
    full.set_derivative(deriv)
    out = _run_toppra(full, setup, min_points, max_iter,
                      velocity_scale, acceleration_scale)
    _report_splices("exact")
    return out


# Retry ladder for _run_toppra, as a forward-pass constraint relaxation.
#
# Drake's TOPPRA is numerically brittle: the backward pass can land a solution
# exactly on the boundary of the feasible set, so the forward-pass LP then fails at
# some knot ("Toppra failed to find the maximum path acceleration at knot N/M") and
# SolvePathParameterization() returns None -- on a path that is perfectly fine
# physically. Passing that None straight to PathParameterizedTrajectory raises an
# opaque TypeError rather than reporting the real failure.
#
# `Toppra::set_constraint_relaxation` (Drake PR #24798) is the matching fix, and
# the pinned nightly exposes it. Drake's own docs: "This is a problem tolerance,
# not a solver tolerance... useful for numerically difficult trajectories where
# the backward pass solver finds a solution on the boundary of the feasible set,
# causing the forward pass to fail due to solver precision limits. A typical
# value is 1e-4 or 1e-5."
#
# 0.0 is first and carries every leg that reaches TOPPRA with a trajopt-smoothed
# B-spline. Measured on the 20-point grid: 13 relaxations fired, and all 13 were
# the carry probe's lift -- a path that goes straight from the shortcutter into
# TOPPRA with do_trajopt=False. Every shipped lift and place retimed at 0.0.
#
# So what is hard to retime is the *jagged shortcut path*, not the IK mapping as
# such; a smoothed B-spline through the same mapping is fine. (An earlier version
# of this comment claimed relaxation was "systematic for retiming through the IK
# mapping", from a count that double-globbed each point's .log and .log.*.live
# and so counted every event twice. Count with point_*.log only.)
#
# This REPLACES an earlier ladder over grid density and limit scale, which was
# the wrong lever twice over. It never once recovered a leg: measured on the
# 20-point grid, every leg that failed the first rung went on to fail all five,
# including grid x4 and limits x0.25 -- a denser grid cannot fix a
# boundary-precision failure in the forward-pass LP. It was also actively
# harmful: its lower rungs scale the requested velocity/acceleration limits
# down, so a leg "rescued" there ships slower than asked for.
TOPPRA_RELAXATION_LADDER: Tuple[float, ...] = (
    0.0,      # no relaxation -- the normal path
    1e-5,     # occasional numerical difficulty
    1e-4,     # Drake's other documented typical value
)


def _run_toppra(
    full_traj, setup: _Setup, min_points: int, max_iter: int,
    velocity_scale: float = 0.1, acceleration_scale: float = 0.1,
) -> PathParameterizedTrajectory:
    # Drake's plant limits are the natural ceiling, so a scale > 1 would request
    # motion the plant can't deliver; clip to (0, 1].
    velocity_scale = float(np.clip(velocity_scale, 0.0, 1.0))
    acceleration_scale = float(np.clip(acceleration_scale, 0.0, 1.0))

    if not hasattr(Toppra, "set_constraint_relaxation"):
        raise RuntimeError(
            "This Drake build does not expose Toppra.set_constraint_relaxation "
            "(Drake PR #24798), which TOPPRA_RELAXATION_LADDER requires. Pin a "
            "nightly that has it -- the superseded grid/limit-scale ladder is not "
            "an acceptable substitute: it never recovered a leg on the 20-point "
            "grid, and its lower rungs silently shipped legs slower than requested."
        )

    # Gridpoint selection is not a rounding error next to the solve: it samples
    # the path adaptively, and on a constrained leg every sample is an analytic
    # IK evaluation through the FunctionHandleTrajectory.
    with record("toppra.gridpoints", min_points=min_points, max_iter=max_iter):
        gridpts = Toppra.CalcGridPoints(
            full_traj,
            CalcGridPointsOptions(max_iter=max_iter, min_points=2 * min_points),
        )
        mark(n_gridpoints=int(len(gridpts)))

    failures = []
    for relaxation in TOPPRA_RELAXATION_LADDER:
        with record("toppra.build", relaxation=relaxation):
            toppra = Toppra(full_traj, setup.plant, gridpts)
            toppra.AddJointVelocityLimit(
                velocity_scale * setup.plant.GetVelocityLowerLimits(),
                velocity_scale * setup.plant.GetVelocityUpperLimits(),
            )
            toppra.AddJointAccelerationLimit(
                acceleration_scale * setup.plant.GetAccelerationLowerLimits(),
                acceleration_scale * setup.plant.GetAccelerationUpperLimits(),
            )
            # 0.0 is Drake's default (no relaxation); set it explicitly anyway so
            # the rung is unambiguous rather than relying on the default.
            toppra.set_constraint_relaxation(relaxation)

        # Every rung that fails is logged as it happens, not just summarised in
        # the final exception: a leg that needed relaxation is materially
        # different from one that retimed cleanly, and the difference must be
        # visible even though both end in a returned trajectory. Drake's own
        # per-knot "Toppra failed to find the maximum path acceleration" line is
        # NOT this signal -- it is a diagnostic that is frequently emitted on
        # solves that then succeed.
        def _rung_failed(reason: str) -> None:
            failures.append(f"relaxation {relaxation:g}: {reason}")
            print(f"[toppra] relaxation {relaxation:g} FAILED: {reason}; "
                  f"trying the next rung")

        with record("toppra.solve", relaxation=relaxation, rung=len(failures)):
            time_traj = toppra.SolvePathParameterization()
            mark(solved=time_traj is not None)
            if time_traj is None:
                mark_discarded()
        if time_traj is None:
            _rung_failed("no solution")
            continue
        retimed = PathParameterizedTrajectory(full_traj, time_traj)
        if not np.isfinite(retimed.end_time() - retimed.start_time()):
            _rung_failed("infinite duration")
            continue
        if relaxation != 0.0:
            # Deliberately does NOT claim the leg is within the requested limits.
            # Drake: "This is a problem tolerance, not a solver tolerance,
            # meaning it slightly relaxes the physical problem bounds." So the
            # velocity/acceleration bounds ARE relaxed, by `relaxation`, and
            # nothing downstream re-checks them -- verify_trajectory covers
            # collision, CoM and joint limits only. At 1e-5 the overshoot is
            # negligible in practice, but it is unverified, so say so rather
            # than reassure.
            print(f"[toppra] RELAXATION ENGAGED: retimed with constraint "
                  f"relaxation {relaxation:g} after {len(failures)} failed "
                  f"rung(s). This relaxes the physical velocity/acceleration "
                  f"bounds by that tolerance and nothing downstream re-checks "
                  f"them -- see the Drake error above.")
        return retimed

    raise RuntimeError(
        "TOPPRA failed to retime the path at every rung of "
        "TOPPRA_RELAXATION_LADDER. " + "; ".join(failures)
        + ". Either the path has a workspace singularity along it, or it is too "
        "jagged to retime -- a path that fell back from a failed trajopt solve is "
        "the usual culprit, since it keeps every RRT waypoint's zero-velocity stop."
    )
