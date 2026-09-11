"""Generate meshcat HTML for the IIWA bimanual experiment.

Runs BiRRT + shortcutting + TOPPRA on the bimanual IIWA setup, records the
trajectory in meshcat, and exports a StaticHtml file (which
``blender_render_iiwa.py`` then renders).

The scene, the parameterization and the planning stack here are the same ones
used by ``../iiwa-bimanual/notebooks/main_cpp.ipynb`` (the
authoritative ICRA-2024 demo).  Everything below the constants is exposed as
functions so the renderers show *this* trajectory rather than inventing their
own.

Usage:
    .venv/bin/python scripts/video/generate_iiwa_meshcat.py
    .venv/bin/python scripts/video/generate_iiwa_meshcat.py --replan
"""

import argparse
import os
import sys
import time

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
IIWA_REPO = os.path.join(REPO, os.pardir, "iiwa-bimanual")
sys.path.insert(0, os.path.join(IIWA_REPO, "src"))

import iiwa_analytic_ik
import rrt
import shortcut as shortcut_mod

from pydrake.all import (
    StartMeshcat, CollisionCheckerParams, RobotDiagramBuilder,
    MeshcatVisualizerParams, Role, MeshcatVisualizer, Parser,
    LoadModelDirectives, ProcessModelDirectives,
    SceneGraphCollisionChecker, AutoDiffXd, RigidTransform_,
    PiecewisePolynomial, CompositeTrajectory,
    FunctionHandleTrajectory, InitializeAutoDiff, ExtractGradient,
    Toppra, PathParameterizedTrajectory, CalcGridPointsOptions,
    DiagramBuilder, TrajectorySource, InverseDynamicsController,
    Demultiplexer, Simulator,
    HPolyhedron,
    Box, RigidTransform, RotationMatrix, RollPitchYaw,
    SpatialInertia, UnitInertia, CoulombFriction,
)

import common

directives_file = os.path.join(common.RepoDir(), "models/old_shelves.dmd.yaml")
grasp_distance = 0.6
GC2, GC4, GC6 = 1, 1, -1

# The notebook's two hand-written carry poses.  They are kept because they
# document the demo this segment comes from, and because q_tilde_bottom is the
# reference the pick keyframe is measured against -- but they are NOT the
# endpoints of the motion any more; see PLANK_POSE_* below.  Neither of them is
# a placement: the notebook carries an implied object from one mid-air pose to
# another and stops.
q_tilde_bottom = np.array([-0.6430910102907225, 1.9156121024586796, -1.7968254667817805,
                            1.2945447141185198, -0.023834531305537934, -0.876966810663043,
                            -1.7041643160834519, 1.45])
q_tilde_top = np.array([-0.1994994216078726, 0.9140739951190965, -2.236618320862171,
                         0.5238879195899456, 0.7998441913611017, -1.3575398006936048,
                         -1.0153092816310436, 2.41])

# ---------------------------------------------------------------------------
# The carried plank
# ---------------------------------------------------------------------------
# Neither the notebook nor old_shelves.dmd.yaml models a manipuland: the demo's
# constrained bimanual carry is implied by `grasp_distance`, with the two
# grippers facing each other and nothing in between.  For the video we need to
# *see* what is being carried, so a plank is welded into the left gripper.
#
# Geometry rationale, all in the wsg_left::body frame:
#   * the two fingers of one gripper occupy |x| in [0.05, 0.09], so the pinch
#     gap is |x| <= 0.05 and a plank 0.09 deep is gripped with ~5 mm to spare;
#   * X_LR is a constant [0, 0.42, 0] with a 180 deg roll, and each gripper's
#     fingers span y in [0.023, 0.187] of their own frame, so a plank centred at
#     y = 0.21 and 0.33 long (y in [0.045, 0.375]) is inside both grippers'
#     fingers while staying clear of the gripper bodies (|y| <= 0.036);
#   * the fingers span |z| <= 0.019.  The plank's *top* face sits at z = +0.0275,
#     8 mm proud of them, and the plank then hangs *downwards* out of the pinch.
#
# That last point is the one that matters, and it is not cosmetic.  The iiwa's
# forearm (iiwa_link_5) hangs about 62 mm below the wrist, and the wrists are
# coaxial with the plank -- X_L7_plank is a pure translation along the plank's
# long axis, so both wrists sit at the plank's own height.  Measured by sweeping
# all 8 leader GC branches x 72 leader elbow angles x 72 follower elbow angles at
# each candidate height: an object gripped at *mid* height can never get its
# underside closer than ~35 mm to the table, and can never be lowered towards a
# shelf plate at all (the forearm reaches over the plate's front edge and
# penetrates it 30+ mm before the object arrives).  A plank gripped near its top
# edge puts its underside 132.5 mm below the wrists, i.e. well below where the
# forearm bottoms out, which is what makes a real placement kinematically
# possible.
CARRIED_PLANK_SIZE = (0.09, 0.33, 0.16)
PLANK_TOP_IN_GRIPPER_Z = 0.0275
CARRIED_PLANK_OFFSET_IN_GRIPPER = (
    0.0, 0.21, PLANK_TOP_IN_GRIPPER_Z - CARRIED_PLANK_SIZE[2] / 2.0)
# Green: distinct from the KUKA orange/grey, the tan shelves and the blue
# fingers, so the carried object is unambiguous on screen.
CARRIED_PLANK_COLOR = (0.15, 0.65, 0.30, 1.0)

# ---------------------------------------------------------------------------
# The task
# ---------------------------------------------------------------------------
# Surfaces, in world z.  old_shelves is welded at world (0.8, 0.3825, 0); its
# two plates are 0.014 thick and centred at z = 0.3 and z = 0.6, and they span
# x in [0.6, 1.0], y in [-0.1175, 0.8825].  table_wide's collision box is a
# half-space whose top face is z = 0.
TABLE_TOP_Z = 0.0
SHELF_PLATE_TOP_Z = 0.607

# Explicit clearances.  Resting the plank *on* a surface is contact, which the
# collision checker rejects by construction, so both the pick pose and the place
# pose leave a few mm of air.  Nothing is filtered against the shelves or the
# table: the plank is a full member of the collision model (see build_scene).
PICK_CLEARANCE = 0.004
PLACE_CLEARANCE = 0.005
# Margin the *arms and grippers* keep from the environment while planning; see
# build_scene's docstring for why it is not zero.
PLAN_ENV_PADDING = 0.008

# Keyframes, given as (x, y, z_of_plank_underside) of the plank in world.  The
# plank is level (zero roll/pitch/yaw) at all three, so it lies flat.
#   pick      the plank sitting on the table in front of the shelf unit, clear
#             of the lower plate's footprint (x <= 0.6) so it can be lifted
#             straight up;
#   standoff  high and well in front of the unit, plank underside 120 mm above
#             the top plate.  The last leg is then one in-and-down move of
#             +0.30 m in x and -0.115 m in z; a standoff nearer the shelf makes
#             that move so small that, from any camera that also shows the pick,
#             the placement stops reading as a placement;
#   place     over the top plate, plank underside PLACE_CLEARANCE above it.
PLANK_POSE_PICK = (0.50, 0.40, TABLE_TOP_Z + PICK_CLEARANCE)
PLANK_POSE_STANDOFF = (0.40, 0.40, SHELF_PLATE_TOP_Z + 0.120)
PLANK_POSE_PLACE = (0.70, 0.40, SHELF_PLATE_TOP_Z + PLACE_CLEARANCE)

PATH_CACHE = os.path.join(REPO, "video", "v2_iiwa_bimanual_path_place.npy")

# Playback speed for the recorded/rendered motion; see lift_and_retime.  At 1.0
# (TOPPRA's own optimum, 7.4 s) the carry is brisk enough that the plank looks
# thrown; 0.9 plus the two holds lands the segment at ~11 s, close to the 10 s
# the assembled overview video was cut for.
PLAYBACK_SPEED = 0.9

analytic_ik_obj = iiwa_analytic_ik.Analytic_IK_7DoF(
    iiwa_analytic_ik.iiwa_alpha, iiwa_analytic_ik.iiwa_d,
    iiwa_analytic_ik.iiwa_limits_lower, iiwa_analytic_ik.iiwa_limits_upper)


def q_to_ee_target(q):
    ad = isinstance(q[0], AutoDiffXd)
    T = AutoDiffXd if ad else float
    tf_goal = analytic_ik_obj.FK(q)
    ang = (180 - 2. * 68.) * np.pi / 180.
    c, s = np.cos(ang), np.sin(ang)
    tf_goal[:-1, :-1] = tf_goal[:-1, :-1] @ np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]) @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    tf_goal[:-1, -1] += tf_goal[:-1, :-1] @ np.array([0, 0, -grasp_distance])
    tf_goal[:-1, -1] += np.array([0, -0.765, 0])
    return RigidTransform_[T](tf_goal)


def parameterization(q_tilde):
    q_full = np.zeros(14, dtype=type(q_tilde[0]))
    q_full[:7] = q_tilde[:7]
    tf_goal = q_to_ee_target(q_tilde[:7])
    psi = q_tilde[7]
    q_follower = analytic_ik_obj.IK(tf_goal, [GC2, GC4, GC6], psi)
    q_full[7:] = q_follower
    return q_full


def X_L7_plank():
    """The (constant) pose of the plank in iiwa_left::iiwa_link_7.

    Composed from the model directives rather than read off a context so the
    keyframe solver does not need a plant.  ``build_scene`` asserts it against
    the plant it just built.
    """
    X_L7_gripper = RigidTransform(
        RollPitchYaw(np.radians(90.0), 0.0, np.radians(68.0)), [0.0, 0.0, 0.09])
    X_gripper_plank = RigidTransform(np.asarray(CARRIED_PLANK_OFFSET_IN_GRIPPER))
    return X_L7_gripper @ X_gripper_plank


def add_carried_plank(plant):
    """Weld the carried plank into wsg_left::body.  Returns the new body.

    Welding (rather than adding a floating body) keeps the plant at 14
    positions, so the same SetPositions calls work with and without the plank,
    and makes the plank-to-left-gripper transform rigid by construction.  The
    plank-to-*right*-gripper transform is then constant exactly when the
    parameterization's fixed gripper-to-gripper constraint holds, which is what
    render_iiwa_experiment.py measures.
    """
    instance = plant.AddModelInstance("carried_plank")
    body = plant.AddRigidBody(
        "carried_plank", instance,
        SpatialInertia(1.0, np.zeros(3), UnitInertia.SolidBox(*CARRIED_PLANK_SIZE)))
    X_GB = RigidTransform(np.asarray(CARRIED_PLANK_OFFSET_IN_GRIPPER))
    plant.WeldFrames(plant.GetFrameByName("body", plant.GetModelInstanceByName("wsg_left")),
                     body.body_frame(), X_GB)
    shape = Box(*CARRIED_PLANK_SIZE)
    plant.RegisterVisualGeometry(body, RigidTransform(), shape, "carried_plank_visual",
                                 np.array(CARRIED_PLANK_COLOR))
    plant.RegisterCollisionGeometry(body, RigidTransform(), shape, "carried_plank_collision",
                                    CoulombFriction(0.9, 0.8))
    return body


_DIAGRAM_KEEPALIVE = []


def build_scene(meshcat=None, with_collision_checker=True, with_carried_plank=False,
                env_padding=0.0):
    """Build the notebook's scene, optionally with the carried plank.

    Returns (diagram, plant, checker).  `checker` is None when
    with_collision_checker is False.

    Two things here are deliberately *stricter* than the notebook, which set
    ``robot_model_instances = [iiwa_left, iiwa_right]`` only:

    * the two grippers are in the robot set.  In the notebook they are
      environment bodies welded to the robot, and SceneGraphCollisionChecker
      ignores environment-environment pairs -- so nothing ever checked a gripper
      against a shelf, and a plan was free to sweep one through a plate;
    * the plank, when present, is in the robot set too, with its pair against
      wsg_right filtered (it is held by that gripper, so that contact is the
      grasp, not a collision).  Its pairs against the shelves and the table are
      *not* filtered.

    Both changes only remove configurations, and the placement claim needs them:
    a placement is only meaningful if the thing being placed is in the
    collision model.

    `env_padding` (metres) inflates every robot-environment pair *except* the
    plank's own pairs against the shelves and the table.  Planning at
    env_padding=0 produced a path that came within -17 um of old_shelves at one
    instant: the polyline edges are only checked every 0.01 in the 8-D reduced
    space, and a straightened edge can graze a plate between two checks.  So the
    planner runs padded and the verifier runs at env_padding=0, which is the
    honest number.  The plank's own pairs stay unpadded because approaching a
    surface to within a few mm is the whole point of the placement -- it is
    still never allowed to touch.
    """
    params = CollisionCheckerParams()
    builder = RobotDiagramBuilder(time_step=0.0)

    if meshcat is not None:
        vis_params = MeshcatVisualizerParams()
        vis_params.delete_on_initialization_event = False
        vis_params.role = Role.kIllustration
        vis_params.prefix = "visual"
        MeshcatVisualizer.AddToBuilder(builder.builder(), builder.scene_graph(),
                                       meshcat, vis_params)

    plant = builder.plant()
    parser = Parser(plant)
    parser.package_map().AddPackageXml(os.path.join(common.RepoDir(), "package.xml"))
    directives = LoadModelDirectives(directives_file)
    ProcessModelDirectives(directives, parser)

    plank_body = add_carried_plank(plant) if with_carried_plank else None

    robot_instances = [
        plant.GetModelInstanceByName("iiwa_left"),
        plant.GetModelInstanceByName("iiwa_right"),
        plant.GetModelInstanceByName("wsg_left"),
        plant.GetModelInstanceByName("wsg_right"),
    ]
    if plank_body is not None:
        robot_instances.append(plant.GetModelInstanceByName("carried_plank"))
    params.robot_model_instances = robot_instances
    plant.Finalize()

    builder.builder().ExportInput(plant.get_actuation_input_port(), "actuation")
    builder.builder().ExportOutput(plant.get_state_output_port(), "state")

    diagram = builder.Build()

    if plank_body is not None:
        # The analytic X_L7_plank used by the keyframe solver must agree with
        # the plant, or keyframes would be solved for the wrong body.
        ctx = plant.CreateDefaultContext()
        X_num = plant.CalcRelativeTransform(
            ctx, plant.GetFrameByName("iiwa_link_7", plant.GetModelInstanceByName("iiwa_left")),
            plank_body.body_frame())
        assert np.allclose(X_num.GetAsMatrix4(), X_L7_plank().GetAsMatrix4(), atol=1e-12), (
            f"X_L7_plank mismatch:\n{X_num.GetAsMatrix4()}\nvs\n{X_L7_plank().GetAsMatrix4()}")

    # The returned plant is owned by the diagram.  Callers that only want the
    # plant (to hand to TOPPRA, say) used to drop the diagram on the floor and
    # then crash inside GetAccelerationLowerLimits with a garbage JointIndex, so
    # keep a reference alive for the life of the process.
    _DIAGRAM_KEEPALIVE.append(diagram)

    checker = None
    if with_collision_checker:
        params.model = diagram
        params.edge_step_size = 0.01
        checker = SceneGraphCollisionChecker(params)
        if plank_body is not None:
            wsg_right = plant.GetModelInstanceByName("wsg_right")
            for bi in plant.GetBodyIndices(wsg_right):
                checker.SetCollisionFilteredBetween(plank_body.index(), bi, True)
        if env_padding:
            checker.SetPaddingAllRobotEnvironmentPairs(env_padding)
            if plank_body is not None:
                for name in ("old_shelves", "table"):
                    for bi in plant.GetBodyIndices(plant.GetModelInstanceByName(name)):
                        checker.SetPaddingBetween(plank_body.index(), bi, 0.0)

    return diagram, plant, checker


def make_validity_checker(checker):
    """The notebook's validity predicate on the 8-D parameterized space."""

    def unclipped_vals(q_tilde):
        tf_goal = q_to_ee_target(q_tilde[:7])
        psi = q_tilde[7]
        return analytic_ik_obj.IK(tf_goal, [GC2, GC4, GC6], psi, return_unclipped_vals=True)

    def ValidityChecker(q_tilde):
        if np.any(unclipped_vals(q_tilde) > np.ones(4)):
            return False
        if np.any(unclipped_vals(q_tilde) < -np.ones(4)):
            return False
        q_full = parameterization(q_tilde)
        q_sub = q_full[7:]
        if np.any(q_sub < iiwa_analytic_ik.iiwa_limits_lower):
            return False
        if np.any(q_sub > iiwa_analytic_ik.iiwa_limits_upper):
            return False
        if np.any(np.abs(q_sub[[1, 3, 5]]) < 1e-2):
            return False
        return checker.CheckConfigCollisionFree(q_full)

    return ValidityChecker


# ---------------------------------------------------------------------------
# Keyframes
# ---------------------------------------------------------------------------
# The plank's world pose is a function of the leader (left) arm's 7 joints
# alone, because the plank is welded into the left gripper.  So a keyframe is
# solved by running the *same* analytic IK the parameterization uses, on the
# left arm, for the link_7 pose that puts the plank where we want it -- then
# sweeping the follower's elbow angle for a configuration the validity checker
# accepts.  Nothing here is hand-typed into joint space.

_GC_BRANCHES = [(a, b, c) for a in (1, -1) for b in (1, -1) for c in (1, -1)]


def plank_pose_to_X_W_L7(x, y, z_underside, rpy_deg=(0.0, 0.0, 0.0)):
    R = RotationMatrix(RollPitchYaw(*np.radians(rpy_deg)))
    z_centre = z_underside + CARRIED_PLANK_SIZE[2] / 2.0
    X_W_plank = RigidTransform(R, [x, y, z_centre])
    return X_W_plank @ X_L7_plank().inverse()


def solve_keyframe(plank_pose, validity_checker, reference=None,
                   n_psi_leader=145, n_psi_follower=145, verbose=True):
    """Solve one keyframe in the 8-D parameterized space.

    plank_pose is (x, y, z_underside).  Sweeps the leader's 8 GC branches and
    elbow angle and the follower's elbow angle, keeps every configuration the
    validity checker accepts, and returns the one closest to `reference` (so
    consecutive keyframes stay on the same branch and the legs stay short).
    """
    X_W_L7 = plank_pose_to_X_W_L7(*plank_pose)
    lower = iiwa_analytic_ik.iiwa_limits_lower
    upper = iiwa_analytic_ik.iiwa_limits_upper

    candidates = []
    n_reach = n_lim = 0
    for gc in _GC_BRANCHES:
        for psi_l in np.linspace(-np.pi, np.pi, n_psi_leader)[:-1]:
            q_leader = analytic_ik_obj.IK(X_W_L7, list(gc), psi_l)
            # IK clips its arccos arguments, so an unreachable pose comes back
            # as a silently wrong configuration.  Re-run FK and reject.
            fk = analytic_ik_obj.FK(q_leader)
            if (np.linalg.norm(fk[:3, 3] - X_W_L7.translation()) > 1e-7
                    or np.max(np.abs(fk[:3, :3] - X_W_L7.rotation().matrix())) > 1e-6):
                n_reach += 1
                continue
            if np.any(q_leader < lower) or np.any(q_leader > upper):
                n_lim += 1
                continue
            for psi_f in np.linspace(0.0, 2.0 * np.pi, n_psi_follower)[:-1]:
                q_tilde = np.concatenate([q_leader, [psi_f]])
                if validity_checker(q_tilde):
                    candidates.append(q_tilde)

    if verbose:
        print(f"  keyframe at plank {np.round(plank_pose, 4)}: {len(candidates)} valid "
              f"configurations ({n_reach} leader poses unreachable, {n_lim} past a joint limit)")
    if not candidates:
        raise RuntimeError(f"no valid configuration for plank pose {plank_pose}")

    candidates = np.asarray(candidates)
    if reference is None:
        reference = q_tilde_bottom
    d = np.linalg.norm(candidates - np.asarray(reference), axis=1)
    return candidates[int(np.argmin(d))]


def solve_task_keyframes(validity_checker, verbose=True):
    """pick -> standoff -> place, each chained to the previous one."""
    if verbose:
        print("Solving keyframes...")
    # The pick keyframe is the one anchored to the notebook: of the thousands of
    # valid configurations it takes the one nearest q_tilde_bottom, so the shot
    # opens on the demo's own pose.  The other two then chain off it, which keeps
    # each leg short and the elbows from wandering mid-carry.
    pick = solve_keyframe(PLANK_POSE_PICK, validity_checker,
                          reference=q_tilde_bottom, verbose=verbose)
    standoff = solve_keyframe(PLANK_POSE_STANDOFF, validity_checker,
                              reference=pick, verbose=verbose)
    place = solve_keyframe(PLANK_POSE_PLACE, validity_checker,
                           reference=standoff, verbose=verbose)
    return [pick, standoff, place]


def plan_reduced_path(validity_checker, start, goal, seed=0, label=""):
    """BiRRT + shortcut in the 8-D parameterized space.

    start/goal are required.  They used to default to the notebook's
    q_tilde_bottom/q_tilde_top, which is now a trap: with the plank in the
    collision model both of those configurations are *in collision* (the plank
    would be 32 mm into the table at one and 10 mm into the shelf plate at the
    other), so the default would fail the validity checker on its first call.
    """

    domain_lower = np.hstack((iiwa_analytic_ik.iiwa_limits_lower, [0.0]))
    domain_upper = np.hstack((iiwa_analytic_ik.iiwa_limits_upper, [2.0 * np.pi]))

    def RandomConfig():
        return np.random.uniform(low=domain_lower, high=domain_upper)

    print(f"Running BiRRT{' (' + label + ')' if label else ''}...")
    rrt_options = rrt.RRTOptions(
        step_size=2e-1, check_size=1e-2, max_vertices=1e4,
        max_iters=1e6, goal_sample_frequency=0.01, always_swap=False,
    )
    rrt_planner = rrt.BiRRT(RandomConfig, validity_checker)
    np.random.seed(seed)
    path = rrt_planner.plan(start, goal, rrt_options)
    if len(path) == 0:
        raise RuntimeError(f"BiRRT failed for leg {label!r}")
    print(f"  RRT path: {len(path)} waypoints")

    print("Shortcutting...")
    np.random.seed(seed)
    shortcut_path = shortcut_mod.shortcut(path.copy(), validity_checker,
                                          num_tries=100, check_size=rrt_options.check_size)
    print(f"  Shortcut path: {len(shortcut_path)} waypoints")
    return np.asarray(shortcut_path)


def _edge_valid(a, b, validity_checker, check_size):
    d = float(np.linalg.norm(b - a))
    if d < 1e-12:
        return True
    u = (b - a) / d
    for k in range(1, int(d / check_size) + 1):
        if not validity_checker(a + u * (check_size * k)):
            return False
    return bool(validity_checker(b))


def prune_waypoints(path, validity_checker, check_size=1e-2):
    """Drop interior waypoints whose neighbours can be joined directly.

    ``lift_and_retime`` turns each waypoint pair into a cubic with zero end
    velocities, so the traced path is exactly the polyline the shortcutter
    validated -- but the motion comes to a full stop at *every* waypoint.  A
    33-waypoint path therefore stutters 31 times.  This is the cheap
    complement to the random shortcutter: it keeps the path a polyline of
    validity-checked straight edges (so nothing about the collision guarantee
    changes) and just removes the corners that were not buying anything.
    """
    path = [np.asarray(p, dtype=float) for p in path]
    changed = True
    while changed and len(path) > 2:
        changed = False
        i = 1
        while i < len(path) - 1:
            if _edge_valid(path[i - 1], path[i + 1], validity_checker, check_size):
                del path[i]
                changed = True
            else:
                i += 1
    return np.asarray(path)


def plan_task(validity_checker, seed=0):
    """The whole motion: carry the plank up, then set it on the top plate.

    Each leg is planned and shortcut separately and the legs are concatenated.
    Shortcutting the concatenation would cut the corner at the standoff and turn
    the in-and-down placement into one diagonal sweep through the shelf's front
    edge, which is exactly the beat that has to stay legible.
    """
    keyframes = solve_task_keyframes(validity_checker)
    legs = []
    for i, label in enumerate(("carry", "place")):
        leg = plan_reduced_path(validity_checker, seed=seed,
                                start=keyframes[i], goal=keyframes[i + 1], label=label)
        n_before = len(leg)
        leg = prune_waypoints(leg, validity_checker)
        print(f"  Pruned path: {n_before} -> {len(leg)} waypoints")
        legs.append(leg if i == 0 else leg[1:])
    path = np.vstack(legs)
    print(f"Full path: {len(path)} waypoints "
          f"({' + '.join(str(len(l)) for l in legs)})")
    return path


def get_reduced_path(replan=False, seed=0):
    """The cached concatenated path, or a fresh plan."""
    if not replan and os.path.exists(PATH_CACHE):
        path = np.load(PATH_CACHE)
        print(f"Loaded cached reduced path ({len(path)} waypoints) from {PATH_CACHE}")
        return path
    _, _, checker = build_scene(with_collision_checker=True, with_carried_plank=True,
                                env_padding=PLAN_ENV_PADDING)
    path = plan_task(make_validity_checker(checker), seed=seed)
    os.makedirs(os.path.dirname(PATH_CACHE), exist_ok=True)
    np.save(PATH_CACHE, path)
    print(f"Wrote {PATH_CACHE}")
    return path


def lift_and_retime(shortcut_path, plant, speed=1.0):
    """Lift the reduced path to 14-D and retime it with TOPPRA.

    `speed` < 1 stretches TOPPRA's time scaling s(t) for playback.  It only
    touches the *parameterization*: the geometric path through configuration
    space is untouched, so every clearance measured on the speed-1 trajectory
    holds unchanged.  The motion at the joint limits is 5.2 s, which is too
    quick for the gripper-to-gripper constraint to read on video.
    """
    rrt_traj_segments = [
        PiecewisePolynomial.CubicWithContinuousSecondDerivatives(
            np.array([float(i) - 1.0, float(i)]),
            np.array([shortcut_path[i - 1], shortcut_path[i]]).T,
            np.zeros(8), np.zeros(8),
        )
        for i in range(1, len(shortcut_path))
    ]
    rrt_traj = CompositeTrajectory(rrt_traj_segments)

    print("Lifting to full config space...")
    traj_to_show = rrt_traj.Clone()
    traj_function = lambda t: parameterization(traj_to_show.value(t).flatten())
    full_traj = FunctionHandleTrajectory(traj_function, 14, 1,
                                         traj_to_show.start_time(), traj_to_show.end_time())

    def full_traj_derivative(t, order, dt=1e-6):
        if order == 1:
            x_val = traj_to_show.value(t).flatten()
            xdot_val = traj_to_show.EvalDerivative(t, 1).flatten()
            x_ad = InitializeAutoDiff(x_val).flatten()
            y_ad = parameterization(x_ad)
            J = ExtractGradient(y_ad)
            return J @ xdot_val.reshape(-1, 1)
        elif order == 2:
            f0 = full_traj_derivative(t, 1)
            f1 = full_traj_derivative(t + dt, 1)
            f2 = full_traj_derivative(t - dt, 1)
            f3 = full_traj_derivative(t + 2 * dt, 1)
            f4 = full_traj_derivative(t - 2 * dt, 1)
            return (-f3 + 16 * f1 - 30 * f0 + 16 * f2 - f4) / (12 * dt ** 2)
        raise RuntimeError(f"Unsupported order={order}")

    full_traj.set_derivative(full_traj_derivative)

    print("Running TOPPRA...")
    gridpoints = Toppra.CalcGridPoints(full_traj, CalcGridPointsOptions(max_iter=2, min_points=200))
    # Retiming *through* the analytic IK mapping is ill-conditioned: dq/ds blows
    # up wherever the follower's elbow passes near a singularity, and the forward
    # pass then finds itself on the wrong side of a bound it has already
    # saturated ("failed to find the maximum path acceleration at knot k/256").
    # Constraint relaxation is the documented lever for exactly this; escalate it
    # until the solve lands, and say which rung was used so a silently-relaxed
    # trajectory cannot be mistaken for a clean one.  It relaxes the velocity and
    # acceleration bounds only -- the geometric path, and so every clearance, is
    # untouched.
    time_traj = None
    for relaxation in (0.0, 1e-4, 1e-3, 1e-2, 5e-2):
        toppra = Toppra(full_traj, plant, gridpoints)
        toppra.AddJointVelocityLimit(plant.GetVelocityLowerLimits(),
                                     plant.GetVelocityUpperLimits())
        toppra.AddJointAccelerationLimit(plant.GetAccelerationLowerLimits(),
                                        plant.GetAccelerationUpperLimits())
        if relaxation:
            toppra.set_constraint_relaxation(relaxation)
        time_traj = toppra.SolvePathParameterization()
        if time_traj is not None:
            if relaxation:
                print(f"  TOPPRA needed constraint_relaxation={relaxation:g}")
            break
    if time_traj is None:
        raise RuntimeError("TOPPRA failed at every relaxation level")
    if speed != 1.0:
        # pchip, so s(t) stays monotone and C1.
        ts = np.linspace(time_traj.start_time(), time_traj.end_time(), 801)
        ss = np.array([[float(np.asarray(time_traj.value(t)).flatten()[0]) for t in ts]])
        time_traj = PiecewisePolynomial.CubicShapePreserving(ts / speed, ss)
    retimed_full_traj = PathParameterizedTrajectory(full_traj, time_traj)
    print(f"  Trajectory duration: {retimed_full_traj.end_time() - retimed_full_traj.start_time():.2f}s"
          + (f" (TOPPRA optimum stretched to {speed:g}x)" if speed != 1.0 else ""))
    return retimed_full_traj


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--replan", action="store_true", help="re-run the planner")
    ap.add_argument("--hold", type=float, default=2.0,
                    help="seconds to hold the placement at the end of the recording")
    ap.add_argument("--speed", type=float, default=PLAYBACK_SPEED,
                    help="playback speed multiplier (<1 slows the motion down)")
    args = ap.parse_args()

    out_path = os.path.join(REPO, "video", "v2_iiwa_bimanual.html")

    print("Starting meshcat...")
    meshcat = StartMeshcat()

    print("Building environment...")
    diagram, plant, _ = build_scene(meshcat=meshcat, with_collision_checker=False,
                                    with_carried_plank=True)
    # The plan is made against a second, identical scene that carries the
    # collision checker; the recorded scene only needs geometry.
    _, plan_plant, _ = build_scene(with_collision_checker=False, with_carried_plank=True)

    shortcut_path = get_reduced_path(replan=args.replan)
    retimed_full_traj = lift_and_retime(shortcut_path, plan_plant, speed=args.speed)

    print("Setting up simulation for recording...")
    sim_builder = DiagramBuilder()
    traj_source = TrajectorySource(retimed_full_traj, 2)
    sim_builder.AddSystem(traj_source)

    nq = diagram.plant().num_positions()
    controller = InverseDynamicsController(
        diagram.plant(), np.full(nq, 10000.0), np.full(nq, 1.0), np.full(nq, 20.0),
        has_reference_acceleration=True,
    )
    sim_builder.AddSystem(controller)

    demux = Demultiplexer([28, 14])
    sim_builder.AddSystem(demux)
    sim_builder.AddSystem(diagram)

    sim_builder.Connect(traj_source.get_output_port(), demux.get_input_port())
    sim_builder.Connect(demux.get_output_port(0), controller.get_input_port_desired_state())
    sim_builder.Connect(demux.get_output_port(1), controller.get_input_port_desired_acceleration())
    sim_builder.Connect(diagram.GetOutputPort("state"), controller.get_input_port_estimated_state())
    sim_builder.Connect(controller.get_output_port_control(), diagram.GetInputPort("actuation"))

    sim_diagram = sim_builder.Build()
    sim_ctx = sim_diagram.CreateDefaultContext()
    plant.SetPositions(plant.GetMyContextFromRoot(sim_ctx),
                       retimed_full_traj.value(retimed_full_traj.start_time()))

    recorder = diagram.GetSubsystemByName("meshcat_visualizer(visual)")
    simulator = Simulator(sim_diagram, sim_ctx)
    simulator.set_target_realtime_rate(0)

    meshcat.SetCameraPose(
        camera_in_world=[-1.2, -0.5, 1.2],
        target_in_world=[0.0, 0.38, 0.5],
    )

    print("Recording...")
    recorder.StartRecording()
    simulator.AdvanceTo(retimed_full_traj.end_time() + args.hold)
    recorder.PublishRecording()

    print("Exporting StaticHtml...")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(meshcat.StaticHtml())
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"Wrote {out_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
