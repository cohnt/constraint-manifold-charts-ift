"""
run_boundary_sweep.py

Sensitivity study for the boundary-reachability threshold tau: runs the same
pipeline as run_full_comparison.py across a range of tau values, so the choice
derived by scripts/analysis/calibrate_boundary_threshold.py can be checked
against its neighbours.

This is a supporting study, not part of the paper's main table. Note that tau is
DERIVED from kinematics and must never be tuned against downstream success rate:
the constraint is an inequality b <= tau, so any search rewarded for success
drives tau upward until it stops constraining anything.

Usage (from the repository root):

    python3 scripts/experiments/run_boundary_sweep.py
"""

import os
import sys
import numpy as np
import time
import json
import zipfile
import io

# ── Path setup ────────────────────────────────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_dir   = os.path.abspath(os.path.join(script_dir, "../.."))
sys.path.append(os.path.join(repo_dir, "cpp_parameterization/python"))
sys.path.append(repo_dir)

from iiwa_ik import (
    BimanualConfig,
    MakeParameterization,
    IiwaBimanualReachableConstraint,
    OldStyleReachableConstraint,
    IiwaBimanualJointLimitConstraint,
    IiwaBimanualCollisionFreeConstraint,
    FullFeasibilityConstraint,
    IiwaBimanualPsiSingularityConstraint,
    IiwaBimanualPathCost,
    IftSingularityHandling,
    AutoDiffConfig,
    ReachabilityType,
)

from src.reach_constraints import make_reach_constraint, reach_constraint_satisfied
from src.iiwa_analytic_ik import iiwa_limits_lower, iiwa_limits_upper
import src.common as common
import src.rrt as rrt
import src.shortcut as shortcut

from pydrake.all import (
    CollisionCheckerParams,
    RobotDiagramBuilder,
    Parser,
    LoadModelDirectives,
    ProcessModelDirectives,
    SceneGraphCollisionChecker,
    IrisNp2Options,
    IrisNp2,
    MathematicalProgram,
    HPolyhedron,
    Hyperellipsoid,
    GraphOfConvexSetsOptions,
    Point,
    PiecewisePolynomial,
    CompositeTrajectory,
    SnoptSolver,
    GcsTrajectoryOptimization,
    ComputePairwiseIntersections,
    StartMeshcat,
    MeshcatVisualizer,
    MeshcatVisualizerParams,
    Role,
    Simulator,
    TrajectorySource,
    InverseDynamicsController,
    Demultiplexer,
    DiagramBuilder,
    BsplineBasis,
    BsplineTrajectory,
    KinematicTrajectoryOptimization,
    MinimumDistanceLowerBoundConstraint,
    SolverOptions,
    CommonSolverOption,
    FunctionHandleTrajectory,
    Toppra,
    CalcGridPointsOptions,
    PathParameterizedTrajectory,
    InitializeAutoDiff,
    ExtractGradient,
)

# ── Shared waypoints and seeds ────────────────────────────────────────────────
Q_TILDE_BOTTOM = np.array([-0.6430910102907225, 1.9156121024586796, -1.7968254667817805,
                             1.2945447141185198, -0.023834531305537934, -0.876966810663043,
                             -1.7041643160834519, 1.45])
Q_TILDE_MIDDLE = np.array([-0.5997312520566763, 1.489780849654964, -1.4739679827359913,
                             1.2905366081785483, -0.04421061906813227, -0.8793712572715165,
                             -1.1603461715511334, 1.45])
Q_TILDE_TOP    = np.array([-0.1994994216078726, 0.9140739951190965, -2.236618320862171,
                             0.5238879195899456, 0.7998441913611017, -1.3575398006936048,
                             -1.0153092816310436, 2.41])

SEEDS = [
    Q_TILDE_BOTTOM,
    np.array([-0.7341522021700233, 1.9192492722970935, -1.849050540687353,  1.4690188979347225, -0.022913995470214974, -0.7839567180379224, -1.735834076048031,  1.45]),
    np.array([-0.816394667473979,  1.9228828117510568, -1.9042766014076622, 1.6254903325102958, -0.020884458583263387, -0.6994788210824544, -1.773950224396859,  1.45]),
    np.array([-0.9076736984240236, 1.7999568628541147, -1.8278258357789336, 1.8976493299850326, -0.032028511314404574, -0.5492230012012871, -1.624933169711267,  1.45]),
    np.array([-0.90780384835653,   1.5443282072400564, -1.480882097408486,  1.9741581801564516, -0.07059018895327443,  -0.5065618808846135, -1.1610690777465094, 1.45]),
    np.array([-0.877792385473089,  1.283945692440691,  -1.1673903163525974, 1.7986279782674526, -0.08798686286997325,  -0.605914842625335,  -0.7496023024205761, 1.45]),
    np.array([-0.7363360141869535, 1.0835790623705088, -1.102219288049605,  1.3727471630916555, -0.07210415656873102,  -0.8362237374759414, -0.6008766712030682, 1.45]),
    np.array([-0.7093225760311644, 0.8650840325295542, -1.4794100092984808, 1.2099934253928784,  0.44173726212402287,  -0.9673197772349095, -0.9450827150678346, 2.0]),
    np.array([-0.5237049267440886, 0.7086764066165658, -1.9872212610757156, 1.045742737284787,   0.8594286107005795,   -1.171705603794283,  -1.1435157398017397, 2.41]),
    np.array([-0.37540312953312194,0.7958305227244739, -2.112215906760149,  0.8433434932970723,  0.8316630398644385,   -1.2430896040746857, -1.1077155278001196, 2.41]),
    Q_TILDE_TOP,
    np.array([-0.7686406052800139, 1.504938625148829,  -1.4584578152597332, 1.655937158932382,  -0.055175677810583384,-0.6834840454669682, -1.1418310479792013, 1.45]),
    Q_TILDE_MIDDLE,
    # Bridging seeds
    np.array([-0.7754, 1.8578, -1.8123, 1.5961, -0.0279, -0.7131, -1.6645, 1.45]),   # Bridge 0-3
    np.array([-0.7228, 0.9743, -1.2908, 1.2914,  0.1848, -0.9018, -0.773,  1.725]),  # Bridge 6-7
    np.array([-0.6165, 0.7869, -1.7333, 1.1279,  0.6506, -1.0695, -1.0443, 2.205]), # Bridge 7-8
    np.array([-0.2875, 0.855,  -2.1744, 0.6836,  0.8158, -1.3003, -1.0615, 2.41]),   # Bridge 9-10
]

DOMAIN_LOWER = np.hstack((iiwa_limits_lower, [0.0]))
DOMAIN_UPPER = np.hstack((iiwa_limits_upper, [2.0 * np.pi]))


# ── Helpers ───────────────────────────────────────────────────────────────────

def build_checker_and_plant(meshcat):
    """
    Build a SceneGraphCollisionChecker and associated plant/diagram.
    Adds both a visual and a collision MeshcatVisualizer, and exports the
    actuation input and state output ports.
    """
    directives_file = os.path.join(common.RepoDir(), "models/old_shelves.dmd.yaml")
    params  = CollisionCheckerParams()
    # A discrete-time plant (time_step > 0) is used because numerically
    # challenging trajectories can hang a continuous-time simulation.
    builder = RobotDiagramBuilder(time_step=0.001)

    meshcat_visual_params = MeshcatVisualizerParams()
    meshcat_visual_params.delete_on_initialization_event = False
    meshcat_visual_params.role = Role.kIllustration
    meshcat_visual_params.prefix = "visual"
    meshcat_visual = MeshcatVisualizer.AddToBuilder(
        builder.builder(), builder.scene_graph(), meshcat, meshcat_visual_params)

    meshcat_collision_params = MeshcatVisualizerParams()
    meshcat_collision_params.delete_on_initialization_event = False
    meshcat_collision_params.role = Role.kProximity
    meshcat_collision_params.prefix = "collision"
    meshcat_collision_params.visible_by_default = False
    meshcat_collision = MeshcatVisualizer.AddToBuilder(
        builder.builder(), builder.scene_graph(), meshcat, meshcat_collision_params)

    plant  = builder.plant()
    parser = Parser(plant)
    parser.package_map().AddPackageXml(os.path.join(common.RepoDir(), "package.xml"))
    ProcessModelDirectives(LoadModelDirectives(directives_file), parser)

    params.robot_model_instances = [
        plant.GetModelInstanceByName("iiwa_left"),
        plant.GetModelInstanceByName("iiwa_right"),
    ]
    plant.Finalize()

    builder.builder().ExportInput(plant.get_actuation_input_port(), "actuation")
    builder.builder().ExportOutput(plant.get_state_output_port(), "state")

    diagram = builder.Build()
    params.model = diagram
    params.edge_step_size = 0.01
    checker = SceneGraphCollisionChecker(params)
    return checker, plant, diagram


def save_meshcat_static(meshcat, base_filename):
    """
    Save a meshcat snapshot as HTML + shared assets under out/, merging
    the assets directory to save space.
    """
    os.makedirs("out", exist_ok=True)
    zip_data = meshcat.StaticZip()
    if isinstance(zip_data, str):
        zip_data = zip_data.encode("utf-8")
    with zipfile.ZipFile(io.BytesIO(zip_data)) as z:
        for info in z.infolist():
            if info.filename == "meshcat.html":
                html_path = f"out/{base_filename}.html"
                with z.open(info) as src, open(html_path, "wb") as dst:
                    dst.write(src.read())
                print(f"  Saved HTML to {html_path}")
            else:
                z.extract(info, "out")


def run_toppra_and_save(source_traj, iris_options, plant, diagram, meshcat,
                         label, config_short_name, test_timings,
                         exit_on_toppra_failure=False):
    """
    Run TOPPRA on *source_traj* (in parameterised space), simulate, and save
    an HTML meshcat snapshot.  Non-fatal unless exit_on_toppra_failure=True.

    Details:
      - second-order finite-difference for the second derivative
      - segment boundary gridpoints added for piecewise trajectories
      - torque limits added
      - full simulation + InverseDynamicsController before saving
      - backup un-retimed trajectory on TOPPRA failure
    """
    try:
        print(f"  TOPPRA retiming [{label}]...")
        t_toppra_start = time.time()

        t_min = source_traj.start_time()
        t_max = source_traj.end_time()

        def full_traj_derivative(t, order, dt=1e-6):
            if order == 1:
                # Clamp t so we never evaluate source_traj outside its valid
                # domain. CompositeTrajectory may extrapolate unreliably for
                # t < start_time, which causes the 5-pt stencil below to
                # return extreme accelerations at the first/last gridpoints,
                # causing TOPPRA to fail with "cannot find controllable set".
                # tc    = float(np.clip(t, t_min, t_max))
                tc = t # Unclear if this helps or not
                x_val = source_traj.value(tc).flatten()
                xdot  = source_traj.EvalDerivative(tc, 1).flatten()
                x_ad  = InitializeAutoDiff(x_val).flatten()
                y_ad  = iris_options.parameterization.get_parameterization_autodiff()(x_ad)
                J     = ExtractGradient(y_ad)
                return J @ xdot.reshape(-1, 1)
            elif order == 2:
                # 4th-order central FIRST-derivative stencil applied to q-dot,
                # which gives q-double-dot. Using the second-derivative stencil
                # here would return jerk: the input is already a first derivative.
                # The centre sample has weight zero for an odd-order derivative,
                # so it is not evaluated. Note the divisor is 12*dt, not 12*dt**2.
                f1 = full_traj_derivative(t + dt,   order=1)
                f2 = full_traj_derivative(t - dt,   order=1)
                f3 = full_traj_derivative(t + 2*dt, order=1)
                f4 = full_traj_derivative(t - 2*dt, order=1)
                return (f4 - 8*f2 + 8*f1 - f3) / (12 * dt)

        traj_fn  = lambda t: iris_options.parameterization.get_parameterization_double()(source_traj.value(t).flatten())
        full_traj = FunctionHandleTrajectory(traj_fn, 14, 1,
                                             source_traj.start_time(),
                                             source_traj.end_time())
        full_traj.set_derivative(full_traj_derivative)

        gridpoints = Toppra.CalcGridPoints(full_traj, CalcGridPointsOptions(max_iter=4, min_points=1000))
        # Add segment boundary gridpoints for piecewise trajectories
        try:
            n_seg = source_traj.get_number_of_segments()
            seg_times = np.array([
                source_traj.start_time() + i * (source_traj.end_time() - source_traj.start_time()) / n_seg
                for i in range(n_seg + 1)
            ])
            gridpoints = np.unique(np.sort(np.concatenate([gridpoints, seg_times])))
        except AttributeError:
            pass  # not a piecewise trajectory

        toppra = Toppra(full_traj, plant, gridpoints)
        toppra.AddJointVelocityLimit(plant.GetVelocityLowerLimits(), plant.GetVelocityUpperLimits())
        toppra.AddJointAccelerationLimit(plant.GetAccelerationLowerLimits(), plant.GetAccelerationUpperLimits())
        toppra.AddJointTorqueLimit(plant.GetEffortLowerLimits(), plant.GetEffortUpperLimits())
        time_traj = toppra.SolvePathParameterization()
        t_toppra_end = time.time()

        if "toppra" not in test_timings:
            test_timings["toppra"] = {}
        if "toppra_durations" not in test_timings:
            test_timings["toppra_durations"] = {}
        test_timings["toppra"][label] = t_toppra_end - t_toppra_start

        if time_traj is None:
            raise RuntimeError("TOPPRA returned None")
        retimed  = PathParameterizedTrajectory(full_traj, time_traj)
        duration = retimed.end_time() - retimed.start_time()
        if not np.isfinite(duration):
            raise RuntimeError(f"TOPPRA returned non-finite duration: {duration}")
        print(f"  TOPPRA [{label}] finished. Duration: {duration:.2f}s "
              f"(Solve time: {t_toppra_end - t_toppra_start:.3f}s)")
        test_timings["toppra_durations"][label] = duration

        _simulate_and_save(retimed, plant, diagram, meshcat,
                           label, config_short_name)

    except Exception as toppra_e:
        print(f"  WARNING: TOPPRA failed for [{label}]: {toppra_e}")
        print(f"  Attempting backup un-retimed trajectory saving...")
        if "toppra_backup" not in test_timings:
            test_timings["toppra_backup"] = {}
        test_timings["toppra_backup"][label] = True
        try:
            backup_duration   = 5.0
            scaling_breaks    = np.array([0.0, backup_duration])
            scaling_samples   = np.array([[source_traj.start_time(), source_traj.end_time()]])
            time_scaling      = PiecewisePolynomial.FirstOrderHold(scaling_breaks, scaling_samples)
            retimed_backup    = PathParameterizedTrajectory(full_traj, time_scaling)
            print(f"  Backup [{label}] duration: {backup_duration:.2f}s")
            _simulate_and_save(retimed_backup, plant, diagram, meshcat,
                               label, config_short_name, suffix="_backup")
        except Exception as backup_e:
            print(f"  Backup simulation also failed: {backup_e}")

        if exit_on_toppra_failure:
            raise


def _simulate_and_save(retimed, plant, diagram, meshcat,
                        label, config_short_name, suffix=""):
    """
    Build a fresh simulation diagram with an InverseDynamicsController,
    advance to end of trajectory, and save a meshcat HTML snapshot.
    """
    print(f"  Simulating [{label}]{suffix}...")
    meshcat.Delete()

    sim_checker, sim_plant, sim_diagram = build_checker_and_plant(meshcat)

    sim_builder = DiagramBuilder()
    traj_src    = TrajectorySource(retimed, 2)
    sim_builder.AddSystem(traj_src)

    nq = sim_plant.num_positions()
    kp = np.full(nq, 10000.0)
    ki = np.full(nq, 1.0)
    kd = np.full(nq, 20.0)
    controller = InverseDynamicsController(sim_plant, kp, ki, kd,
                                           has_reference_acceleration=True)
    sim_builder.AddSystem(controller)

    demux = Demultiplexer([28, 14])
    sim_builder.AddSystem(demux)
    sim_builder.AddSystem(sim_diagram)

    sim_builder.Connect(traj_src.get_output_port(),          demux.get_input_port())
    sim_builder.Connect(demux.get_output_port(0),            controller.get_input_port_desired_state())
    sim_builder.Connect(demux.get_output_port(1),            controller.get_input_port_desired_acceleration())
    sim_builder.Connect(sim_diagram.GetOutputPort("state"),  controller.get_input_port_estimated_state())
    sim_builder.Connect(controller.get_output_port_control(),sim_diagram.GetInputPort("actuation"))

    outer_diagram  = sim_builder.Build()
    sim_context    = outer_diagram.CreateDefaultContext()
    sim_plant.SetPositions(
        sim_plant.GetMyContextFromRoot(sim_context),
        retimed.value(retimed.start_time()),
    )

    recorder  = sim_diagram.GetSubsystemByName("meshcat_visualizer(visual)")
    simulator = Simulator(outer_diagram, sim_context)
    recorder.StartRecording()
    simulator.AdvanceTo(retimed.end_time() + 1.0)
    recorder.PublishRecording()

    safe_label  = label.replace(" ", "_")
    file_prefix = f"{config_short_name}_{safe_label}_trajectory{suffix}"
    save_meshcat_static(meshcat, file_prefix)


# ── Main pipeline ────────────────────────────────────────────────────────────

def run_pipeline(cfg, meshcat, all_timing_results):
    """
    Run the full IRIS → GCS → RRT → TrajOpt → TOPPRA pipeline for one
    configuration dict.  Results are appended to all_timing_results.
    """
    config_name  = cfg["name"]
    short_name   = cfg.get("short_name", config_name)
    use_ift      = cfg["use_ift"]
    handling     = cfg.get("ift_handling", IftSingularityHandling.kPseudoinverse)
    old_reach    = cfg["old_reach"]
    use_psi      = cfg["use_psi"]
    lmbda        = cfg.get("lambda", 0.0)
    svt_eps      = cfg.get("svt_epsilon", None)
    svt_l_max    = cfg.get("svt_lambda_max", None)
    exit_on_fail = cfg.get("exit_on_failure", False)
    use_aniso    = cfg.get("use_anisotropic_damping", False)
    reach_type   = cfg.get("reach_type", ReachabilityType.kProbing if not old_reach else ReachabilityType.kDirect)

    test_timings = {"config": config_name}
    print(f"\n{'='*57}")
    print(f"Testing Configuration: {config_name}")
    print(f"{'='*57}\n")

    checker, plant, diagram = build_checker_and_plant(meshcat)

    grasp_distance = 0.6
    b_thresh     = cfg.get("boundary_threshold", 2.46)
    b_eps        = cfg.get("boundary_epsilon", 1e-6)
    config = BimanualConfig(shoulder_up=True, elbow_up=True, wrist_up=False,
                            grasp_distance=grasp_distance,
                            clipping_margin=1e-4, clipping_margin_psi=1e-4,
                            boundary_threshold=b_thresh, boundary_epsilon=b_eps)

    # ── IRIS ──────────────────────────────────────────────────────────────────
    iris_options = IrisNp2Options()
    ad_config_iris = AutoDiffConfig(use_ift=use_ift, ift_handling=handling, lambda_=lmbda,
                                   svt_epsilon=svt_eps, svt_lambda_max=svt_l_max,
                                   use_anisotropic_damping=use_aniso)
    iris_options.parameterization = MakeParameterization(config, ad_config_iris)
    iris_options.sampled_iris_options.random_seed               = 2
    iris_options.sampled_iris_options.verbose                   = True
    iris_options.sampled_iris_options.max_iterations            = 1
    iris_options.sampled_iris_options.relax_margin              = True
    iris_options.sampled_iris_options.epsilon                   = 0.01
    iris_options.sampled_iris_options.delta                     = 0.01
    iris_options.sampled_iris_options.sample_particles_in_parallel = False
    iris_options.add_hyperplane_if_solve_fails                  = True
    iris_options.solver_options.SetOption(SnoptSolver().solver_id(),
                                          "Major iterations limit", 200)

    iris_prog   = MathematicalProgram()
    q_tilde_vars = iris_prog.NewContinuousVariables(8, "q_tilde")
    iris_options.sampled_iris_options.prog_with_additional_constraints = iris_prog

    # Constraints on the IRIS programme
    ad_config = AutoDiffConfig(use_ift=use_ift, ift_handling=handling, lambda_=lmbda,
                                   svt_epsilon=svt_eps, svt_lambda_max=svt_l_max,
                                   use_anisotropic_damping=use_aniso)
    reach_con = make_reach_constraint(reach_type, config, ad_config)
    iris_prog.AddConstraint(reach_con, q_tilde_vars)

    jl_con = IiwaBimanualJointLimitConstraint(iiwa_limits_lower, iiwa_limits_upper, config, ad_config)
    iris_prog.AddConstraint(jl_con, q_tilde_vars)

    # if use_psi:
    #     psi_iris = IiwaBimanualPsiSingularityConstraint(config, ad_config)
    #     iris_prog.AddConstraint(psi_iris, q_tilde_vars)

    domain = HPolyhedron.MakeBox(DOMAIN_LOWER, DOMAIN_UPPER)

    print("Generating IRIS Regions...")
    t_iris_start = time.time()
    regions = []
    for seed in SEEDS:
        region = IrisNp2(checker, Hyperellipsoid.MakeHypersphere(1e-2, seed),
                         domain, iris_options)
        regions.append(region.ReduceInequalities())
    test_timings["iris_total_time"] = time.time() - t_iris_start
    print(f"Generated {len(regions)} regions.")

    start_in_region = [i for i, r in enumerate(regions) if r.PointInSet(Q_TILDE_BOTTOM)]
    goal_in_region  = [i for i, r in enumerate(regions) if r.PointInSet(Q_TILDE_TOP)]
    print(f"  q_tilde_bottom in regions: {start_in_region} "
          f"({'OK' if start_in_region else 'MISSING - GCS will fail!'})")
    print(f"  q_tilde_top    in regions: {goal_in_region}  "
          f"({'OK' if goal_in_region else 'MISSING - GCS will fail!'})")

    # Connectivity diagnostics
    intersections, _ = ComputePairwiseIntersections(regions, [])
    adjacency = set()
    for (i, j) in intersections:
        adjacency.add((i, j))
        adjacency.add((j, i))
    print(f"  Pairwise intersections: {len(intersections)} edges")
    start_r = start_in_region[0] if start_in_region else None
    goal_r  = goal_in_region[0]  if goal_in_region  else None
    if start_r is not None and goal_r is not None:
        visited = {start_r}
        queue   = [start_r]
        while queue:
            node = queue.pop(0)
            for (a, b) in adjacency:
                if a == node and b not in visited:
                    visited.add(b)
                    queue.append(b)
        if goal_r in visited:
            print(f"  Connectivity: start {start_r} -> goal {goal_r}: CONNECTED")
        else:
            print(f"  Connectivity: start {start_r} -> goal {goal_r}: DISCONNECTED")
            print(f"  Reachable from start: {sorted(visited)}")
            goal_comp = {goal_r}
            q2 = [goal_r]
            while q2:
                node = q2.pop(0)
                for (a, b) in adjacency:
                    if a == node and b not in goal_comp:
                        goal_comp.add(b)
                        q2.append(b)
            print(f"  Reachable from goal: {sorted(goal_comp)}")
    else:
        print(f"  Cannot check connectivity: start_r={start_r}, goal_r={goal_r}")

    # ── GCS ───────────────────────────────────────────────────────────────────
    print("Running GCS Optimization...")
    gcs = GcsTrajectoryOptimization(8)
    gcs.AddPathContinuityConstraints(1)
    gcs.AddPathContinuityConstraints(2)
    main_graph  = gcs.AddRegions(regions, 3, h_min=0.01, h_max=100, name="")
    start_graph = gcs.AddRegions([Point(Q_TILDE_BOTTOM)], 0)
    goal_graph  = gcs.AddRegions([Point(Q_TILDE_TOP)], 0)
    gcs.AddEdges(start_graph, main_graph)
    gcs.AddEdges(main_graph, goal_graph)
    gcs.AddPathLengthCost()
    gcs.AddPathEnergyCost()
    gcs.AddTimeCost()
    gcs.AddVelocityBounds(-np.ones(8), np.ones(8))

    gcs_options = GraphOfConvexSetsOptions()
    gcs_options.max_rounding_trials = 1000
    gcs_options.max_rounded_paths   = 1000
    gcs_options.convex_relaxation   = True

    t_gcs_start = time.time()
    gcs_traj, gcs_result = gcs.SolvePath(start_graph, goal_graph, gcs_options)
    test_timings["gcs_solve_time"] = time.time() - t_gcs_start
    print(f"GCS Solved: {gcs_result.is_success()}")
    if not gcs_result.is_success() and exit_on_fail:
        print("GCS failed and exit_on_failure=True. Aborting pipeline.")
        all_timing_results.append(test_timings)
        return

    # ── RRT ───────────────────────────────────────────────────────────────────
    print("Running BiRRT...")

    def random_config():
        return np.random.uniform(low=DOMAIN_LOWER, high=DOMAIN_UPPER)

    psi_sing_rrt = IiwaBimanualPsiSingularityConstraint(config, ad_config) if use_psi else None

    def validity_checker(q_tilde):
        if not reach_constraint_satisfied(reach_con, reach_type, config, q_tilde):
            return False

        q_full = iris_options.parameterization.get_parameterization_double()(q_tilde)
        q_sub  = q_full[7:]
        if np.any(q_sub < iiwa_limits_lower): return False
        if np.any(q_sub > iiwa_limits_upper): return False
        if np.any(np.abs(q_sub[[1, 3, 5]]) < 1e-2): return False
        if use_psi:
            psi_val = psi_sing_rrt.Eval(q_tilde)
            if np.any(np.abs(psi_val) > 1.0 - config.clipping_margin_psi): return False
        return checker.CheckConfigCollisionFree(q_full)

    rrt_options = rrt.RRTOptions(step_size=2e-1, check_size=1e-2,
                                 max_vertices=1e4, max_iters=1e6,
                                 goal_sample_frequency=0.01)
    rrt_planner = rrt.BiRRT(random_config, validity_checker)
    np.random.seed(0)
    t_rrt_start = time.time()
    path = rrt_planner.plan(Q_TILDE_BOTTOM, Q_TILDE_TOP, rrt_options)
    test_timings["rrt_plan_time"] = time.time() - t_rrt_start

    shortcut_path = None
    if path is not None:
        print(f"RRT Succeeded! Found path with {len(path)} waypoints")
        np.random.seed(0)
        t_sc_start = time.time()
        shortcut_path = shortcut.shortcut(path.copy(), validity_checker,
                                          num_tries=1e2,
                                          check_size=rrt_options.check_size)
        test_timings["rrt_shortcut_time"] = time.time() - t_sc_start
    else:
        print("RRT Failed.")

    # ── Trajectory optimisation + TOPPRA (inside try/except like reference) ──
    print("Running KinematicTrajectoryOptimization...")
    try:
        # GCS → TOPPRA
        if gcs_result.is_success():
            valid_segs = []
            for i in range(gcs_traj.get_number_of_segments()):
                s = gcs_traj.segment(i)
                if s.end_time() - s.start_time() > 1e-4:
                    t_new_start = float(len(valid_segs))
                    t_new_end   = t_new_start + 1.0
                    scaling     = PiecewisePolynomial.FirstOrderHold(
                        [t_new_start, t_new_end], [[s.start_time(), s.end_time()]])
                    valid_segs.append(PathParameterizedTrajectory(s, scaling))
            if valid_segs:
                try:
                    clean_gcs_traj = CompositeTrajectory(valid_segs)
                    run_toppra_and_save(clean_gcs_traj, iris_options, plant, diagram,
                                        meshcat, "GCS", short_name, test_timings,
                                        exit_on_toppra_failure=exit_on_fail)
                except Exception as ce:
                    print(f"  WARNING: Failed to construct clean GCS composite: {ce}")
                    run_toppra_and_save(gcs_traj, iris_options, plant, diagram,
                                        meshcat, "GCS-Original", short_name, test_timings,
                                        exit_on_toppra_failure=exit_on_fail)
            else:
                print("  WARNING: All GCS segments were zero-length! Skipping TOPPRA for GCS.")

        # RRT shortcut → TOPPRA
        if shortcut_path is not None:
            rrt_traj_segments = [
                PiecewisePolynomial.CubicWithContinuousSecondDerivatives(
                    np.array([float(i) - 1.0, float(i)]),
                    np.array([shortcut_path[i-1], shortcut_path[i]]).T,
                    np.zeros(8), np.zeros(8),
                )
                for i in range(1, len(shortcut_path))
            ]
            rrt_traj = CompositeTrajectory(rrt_traj_segments)
            run_toppra_and_save(rrt_traj, iris_options, plant, diagram,
                                meshcat, "RRT", short_name, test_timings,
                                exit_on_toppra_failure=exit_on_fail)

        # TrajOpt (initialised from RRT shortcut) → TOPPRA
        if shortcut_path is not None:
            spline_order = 4
            t0 = rrt_traj.start_time()
            t1 = rrt_traj.end_time()
            ctrl_pts = np.array([pt for pt in shortcut_path]).T
            basis        = BsplineBasis(spline_order, ctrl_pts.shape[1],
                                        initial_parameter_value=t0,
                                        final_parameter_value=t1)
            initial_traj = BsplineTrajectory(basis, ctrl_pts)
            trajopt      = KinematicTrajectoryOptimization(initial_traj)

            # Bound the controlled arm on every control point. psi is left
            # unbounded on purpose: no physical joint corresponds to it, and it
            # is limited indirectly through the subordinate arm's joint limits,
            # which FullFeasibilityConstraint already enforces.
            trajopt.AddPositionBounds(
                np.hstack((iiwa_limits_lower, [-np.inf])),
                np.hstack((iiwa_limits_upper, [ np.inf])))

            trajopt.AddPathPositionConstraint(
                initial_traj.value(initial_traj.start_time()),
                initial_traj.value(initial_traj.start_time()), 0)
            trajopt.AddPathPositionConstraint(
                initial_traj.value(initial_traj.end_time()),
                initial_traj.value(initial_traj.end_time()), 1)

            min_distance  = 0.001
            infl_distance = 0.05
            mdc_context   = diagram.CreateDefaultContext()
            mdc_plant_ctx = plant.GetMyContextFromRoot(mdc_context)
            min_dist_con  = MinimumDistanceLowerBoundConstraint(
                plant, min_distance, mdc_plant_ctx, None,
                infl_distance - min_distance)

            full_feas_con = FullFeasibilityConstraint(
                iiwa_limits_lower, iiwa_limits_upper, config, ad_config,
                min_dist_con, reach_type, use_psi)

            for s in np.linspace(0, 1, 50):
                trajopt.AddPathPositionConstraint(full_feas_con, s)

            path_energy = IiwaBimanualPathCost(
                trajopt.num_positions(), trajopt.num_control_points(),
                config, ad_config, True)
            trajopt.prog().AddCost(path_energy,
                                   trajopt.control_points().flatten(order='F'))

            trajopt_opts = SolverOptions()
            trajopt_opts.SetOption(CommonSolverOption.kPrintToConsole, False)
            trajopt_opts.SetOption(CommonSolverOption.kPrintFileName, "snopt.log")
            trajopt_opts.SetOption(SnoptSolver().solver_id(), "Major print level", 1)
            trajopt_opts.SetOption(SnoptSolver().solver_id(), "Timing level", 3)
            trajopt_opts.SetOption(SnoptSolver().solver_id(), "Time Limit", 60)
            trajopt_opts.SetOption(SnoptSolver().solver_id(),
                                   "Major optimality tolerance", 1e-1)
            trajopt_opts.SetOption(SnoptSolver().solver_id(), "Iterations limit", 100000)
            trajopt_opts.SetOption(SnoptSolver().solver_id(), "Minor iterations limit", 100000)

            solver = SnoptSolver()
            t_to_start = time.time()
            
                    # We don't actually visualize, but use it as a generic callback to store feasible iterates.
            # iterates = []
            # def track_iterates(x):
            #     iterates.append(x)
            # trajopt.prog().AddVisualizationCallback(track_iterates, trajopt.control_points().flatten(order='F'))
            
            to_result  = solver.Solve(trajopt.prog(), None, trajopt_opts)
            test_timings["trajopt_solve_time"] = time.time() - t_to_start
            test_timings["trajopt_solve_success"] = to_result.is_success()
            print(f"Trajectory Optimization Solved: {to_result.is_success()}")

            if to_result.is_success():
                trajopt_traj = trajopt.ReconstructTrajectory(to_result)
            else:
                print("  WARNING: TrajOpt failed. Falling back to INITIAL (RRT) trajectory.")
                # We fall back to the initial trajectory because it's guaranteed feasible by the RRT planner.
                trajopt_traj = initial_traj
            
            run_toppra_and_save(trajopt_traj, iris_options, plant, diagram,
                                meshcat, "Trajopt", short_name, test_timings,
                                exit_on_toppra_failure=exit_on_fail)

        all_timing_results.append(test_timings)

    except Exception as e:
        import traceback
        print(f"Pipeline error for '{config_name}': {e}")
        traceback.print_exc()
        all_timing_results.append(test_timings)

    print("\nConfiguration complete!")

    # This sweep writes under its own directory. It used to share
    # out/timing_{short_name}.json and out/timing_results_full_comparison.json
    # with run_full_comparison.py, so running it after the main benchmark
    # silently overwrote the inputs to the paper's runtime table -- and
    # --skip-existing would then happily reuse the overwritten files.
    os.makedirs(OUT_DIR, exist_ok=True)

    # Per-config JSON so individual runs can be collected by downstream code.
    per_config_path = os.path.join(OUT_DIR, f"timing_{short_name}.json")
    with open(per_config_path, "w") as f:
        json.dump(test_timings, f, indent=4)
    print(f"  Per-config results saved to {per_config_path}")

    # Intermediate aggregate save after every config.
    with open(os.path.join(OUT_DIR, "timing_results_boundary_sweep.json"), "w") as f:
        json.dump(all_timing_results, f, indent=4)


# ── Entry point ───────────────────────────────────────────────────────────────

# Kept separate from run_full_comparison.py's out/ so the two cannot collide.
OUT_DIR = os.environ.get("BOUNDARY_SWEEP_OUT_DIR", "out/boundary_sweep")

def main():
    # ──────────────────────────────────────────────────────────────────────────
    # CONFIGURATIONS
    # Comment out any entry to skip that configuration.
    # ──────────────────────────────────────────────────────────────────────────
    CONFIGS = [
        # tau values are DERIVED, not tuned: each is the threshold implied by a
        # task speed the arm must be able to realise (sigma* = v_task / qdot_max),
        # from scripts/analysis/calibrate_boundary_threshold.py. This sweep probes
        # sensitivity around the derived operating point (tau = 2.46 at 0.1 m/s);
        # it must NOT be used to pick tau by downstream success rate, which would
        # drive tau upward until the constraint stops constraining.
        # ── AD Sweep ─────────────────────────────────────────────────────────
        {
            "name":        "AD, Boundary tau=2.46 (v_task=0.100 m/s)",
            "use_ift":     False,
            "old_reach":   False,
            "reach_type":  ReachabilityType.kBoundary,
            "use_psi":     False,
            "short_name":  "ad_tau2p46",
            "boundary_threshold": 2.46,
        },
        {
            "name":        "AD, Boundary tau=3.35 (v_task=0.065 m/s)",
            "use_ift":     False,
            "old_reach":   False,
            "reach_type":  ReachabilityType.kBoundary,
            "use_psi":     False,
            "short_name":  "ad_tau3p35",
            "boundary_threshold": 3.35,
        },
        {
            "name":        "AD, Boundary tau=4.25 (v_task=0.042 m/s)",
            "use_ift":     False,
            "old_reach":   False,
            "reach_type":  ReachabilityType.kBoundary,
            "use_psi":     False,
            "short_name":  "ad_tau4p25",
            "boundary_threshold": 4.25,
        },
        {
            "name":        "AD, Boundary tau=5.73 (v_task=0.021 m/s)",
            "use_ift":     False,
            "old_reach":   False,
            "reach_type":  ReachabilityType.kBoundary,
            "use_psi":     False,
            "short_name":  "ad_tau5p73",
            "boundary_threshold": 5.73,
        },
        # ── IFT Res Aniso Sweep ─────────────────────────────────────────────
        {
            "name":        "IFT Aniso, Boundary tau=2.46 (v_task=0.100 m/s)",
            "use_ift":     True,
            "ift_handling": IftSingularityHandling.kResidualDamping,
            "use_anisotropic_damping": True,
            "old_reach":   False,
            "reach_type":  ReachabilityType.kBoundary,
            "use_psi":     False,
            "lambda":      10.0,
            "short_name":  "ift_aniso_tau2p46",
            "boundary_threshold": 2.46,
        },
        {
            "name":        "IFT Aniso, Boundary tau=3.35 (v_task=0.065 m/s)",
            "use_ift":     True,
            "ift_handling": IftSingularityHandling.kResidualDamping,
            "use_anisotropic_damping": True,
            "old_reach":   False,
            "reach_type":  ReachabilityType.kBoundary,
            "use_psi":     False,
            "lambda":      10.0,
            "short_name":  "ift_aniso_tau3p35",
            "boundary_threshold": 3.35,
        },
        {
            "name":        "IFT Aniso, Boundary tau=4.25 (v_task=0.042 m/s)",
            "use_ift":     True,
            "ift_handling": IftSingularityHandling.kResidualDamping,
            "use_anisotropic_damping": True,
            "old_reach":   False,
            "reach_type":  ReachabilityType.kBoundary,
            "use_psi":     False,
            "lambda":      10.0,
            "short_name":  "ift_aniso_tau4p25",
            "boundary_threshold": 4.25,
        },
        {
            "name":        "IFT Aniso, Boundary tau=5.73 (v_task=0.021 m/s)",
            "use_ift":     True,
            "ift_handling": IftSingularityHandling.kResidualDamping,
            "use_anisotropic_damping": True,
            "old_reach":   False,
            "reach_type":  ReachabilityType.kBoundary,
            "use_psi":     False,
            "lambda":      10.0,
            "short_name":  "ift_aniso_tau5p73",
            "boundary_threshold": 5.73,
        },
    ]
    # ──────────────────────────────────────────────────────────────────────────

    meshcat           = StartMeshcat()
    all_timing_results = []

    print(f"\n\n{'#'*80}")
    print(f"RUNNING COMPARATIVE BENCHMARK ({len(CONFIGS)} CONFIGURATIONS)")
    print(f"{'#'*80}\n")

    for cfg in CONFIGS:
        run_pipeline(cfg, meshcat, all_timing_results)

    # Final summary table
    max_name_len = max(len(cfg["name"]) for cfg in CONFIGS)
    table_width = max_name_len + 75 
    
    print(f"\n\n{'='*table_width}")
    print(f"{'CONFIGURATION':<{max_name_len}} | {'IRIS':<8} | {'GCS':<8} | {'G-TOP':<8} | "
          f"{'TrjOpt':<8} | {'T-TOP':<8} | {'RRT':<8} | {'R-TOP':<8}")
    print(f"{'-'*table_width}")
    
    baselines = all_timing_results[:3]
    experimentals = sorted(all_timing_results[3:], 
                           key=lambda x: x.get('iris_total_time', 0.0), 
                           reverse=True)
    
    def print_res(res):
        iris     = f"{res.get('iris_total_time',   0.0):.2f}"
        gcs      = f"{res.get('gcs_solve_time',    0.0):.2f}"
        traj     = f"{res.get('trajopt_solve_time',0.0):.2f}"
        rrt_time = f"{res.get('rrt_plan_time',     0.0):.2f}"
        toppra   = res.get("toppra", {})
        g_top    = f"{toppra.get('GCS',    0.0):.2f}"
        t_top    = f"{toppra.get('Trajopt',0.0):.2f}"
        r_top    = f"{toppra.get('RRT',    0.0):.2f}"
        print(f"{res['config']:<{max_name_len}} | {iris:<8} | {gcs:<8} | {g_top:<8} | "
              f"{traj:<8} | {t_top:<8} | {rrt_time:<8} | {r_top:<8}")

    for res in baselines:
        print_res(res)
    
    print(f"{'-'*table_width}") # Horizontal rule separating baselines
    
    for res in experimentals:
        print_res(res)
    print(f"{'='*table_width}\n")

    # Final save
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "timing_results_boundary_sweep.json")
    with open(out_path, "w") as f:
        json.dump(all_timing_results, f, indent=4)
    print(f"Full timing results saved to {out_path}")
    print("\nAll comparative benchmark runs completed successfully!")


if __name__ == "__main__":
    main()
