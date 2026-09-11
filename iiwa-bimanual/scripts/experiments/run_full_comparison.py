"""
run_full_comparison.py

The downstream benchmark behind the paper's runtime table: for each entry in
the CONFIGS list in main(), run IRIS-NP2 -> GCS -> RRT + shortcut -> kinematic
trajectory optimization -> TOPPRA, and record the runtime and outcome of every
stage.

Each configuration pairs a gradient strategy (bespoke autodiff, or the IFT with
one of five singularity-handling modes) with a reachability formulation. Results
are written to out/timing_<short_name>.json plus the aggregate
out/timing_results_full_comparison.json, which scripts/analysis/
plot_pipeline_comparison.py turns into the table and plots.

Usage (from the repository root):

    python3 scripts/experiments/run_full_comparison.py --num-rrt-trials 10

To run only a subset of configurations, comment out entries in CONFIGS. Note
that the analysis script expects all 17 and will warn if it finds fewer.
"""

import os
import sys
import numpy as np
import time
import json
import zipfile
import io
import argparse
import statistics

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
    BoundaryReachabilityConstraint,
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

# Every key a CONFIGS entry may set. Kept next to the readers below so that
# adding a knob without wiring it through fails loudly at startup (see the
# check in main()).
KNOWN_CONFIG_KEYS = {
    "name", "short_name", "exit_on_failure",
    "use_ift", "ift_handling", "lambda", "svt_epsilon", "svt_lambda_max",
    "use_anisotropic_damping",
    "old_reach", "reach_type", "boundary_threshold", "boundary_epsilon",
    "use_psi",
}


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
                         exit_on_toppra_failure=False, save_html=True):
    """
    Run TOPPRA on *source_traj* (in parameterised space), simulate, and save
    an HTML meshcat snapshot if save_html=True. Non-fatal unless exit_on_toppra_failure=True.

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
        # Record the outcome next to the runtime. A solve that returns None or a
        # non-finite duration still consumed wall time, so the runtime above is
        # real -- but it must not be averaged in as if it had succeeded.
        test_timings.setdefault("toppra_success", {})[label] = False

        if time_traj is None:
            raise RuntimeError("TOPPRA returned None")
        retimed  = PathParameterizedTrajectory(full_traj, time_traj)
        duration = retimed.end_time() - retimed.start_time()
        if not np.isfinite(duration):
            raise RuntimeError(f"TOPPRA returned non-finite duration: {duration}")
        print(f"  TOPPRA [{label}] finished. Duration: {duration:.2f}s "
              f"(Solve time: {t_toppra_end - t_toppra_start:.3f}s)")
        test_timings["toppra_durations"][label] = duration
        test_timings["toppra_success"][label] = True

        if save_html:
            _simulate_and_save(retimed, plant, diagram, meshcat,
                               label, config_short_name)

    except Exception as toppra_e:
        print(f"  WARNING: TOPPRA failed for [{label}]: {toppra_e}")
        print(f"  Attempting backup un-retimed trajectory saving...")
        if "toppra_backup" not in test_timings:
            test_timings["toppra_backup"] = {}
        test_timings["toppra_backup"][label] = True
        # Also covers failures raised before the runtime was recorded (gridpoint
        # computation, Toppra construction), where no `toppra` entry exists.
        test_timings.setdefault("toppra_success", {})[label] = False
        test_timings.setdefault("toppra_error", {})[label] = str(toppra_e)
        try:
            backup_duration   = 5.0
            scaling_breaks    = np.array([0.0, backup_duration])
            scaling_samples   = np.array([[source_traj.start_time(), source_traj.end_time()]])
            time_scaling      = PiecewisePolynomial.FirstOrderHold(scaling_breaks, scaling_samples)
            retimed_backup    = PathParameterizedTrajectory(full_traj, time_scaling)
            print(f"  Backup [{label}] duration: {backup_duration:.2f}s")
            if save_html:
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


def run_iris_and_gcs(cfg, meshcat, checker, plant, diagram):
    """
    Run the IRIS and GCS stages of the pipeline.
    """
    use_ift      = cfg["use_ift"]
    handling     = cfg.get("ift_handling", IftSingularityHandling.kPseudoinverse)
    old_reach    = cfg["old_reach"]
    lmbda        = cfg.get("lambda", 0.0)
    svt_eps      = cfg.get("svt_epsilon", None)
    svt_l_max    = cfg.get("svt_lambda_max", None)
    exit_on_fail = cfg.get("exit_on_failure", False)
    use_aniso    = cfg.get("use_anisotropic_damping", False)
    reach_type   = cfg.get("reach_type", ReachabilityType.kProbing if not old_reach else ReachabilityType.kDirect)

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

    ad_config = AutoDiffConfig(use_ift=use_ift, ift_handling=handling, lambda_=lmbda,
                                   svt_epsilon=svt_eps, svt_lambda_max=svt_l_max,
                                   use_anisotropic_damping=use_aniso)
    reach_con = make_reach_constraint(reach_type, config, ad_config)
    iris_prog.AddConstraint(reach_con, q_tilde_vars)

    jl_con = IiwaBimanualJointLimitConstraint(iiwa_limits_lower, iiwa_limits_upper, config, ad_config)
    iris_prog.AddConstraint(jl_con, q_tilde_vars)

    domain = HPolyhedron.MakeBox(DOMAIN_LOWER, DOMAIN_UPPER)

    print("Generating IRIS Regions...")
    t_iris_start = time.time()
    regions = []
    for seed in SEEDS:
        region = IrisNp2(checker, Hyperellipsoid.MakeHypersphere(1e-2, seed),
                         domain, iris_options)
        regions.append(region.ReduceInequalities())
    iris_total_time = time.time() - t_iris_start
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
    regions_connected = None
    if start_r is not None and goal_r is not None:
        visited = {start_r}
        queue   = [start_r]
        while queue:
            node = queue.pop(0)
            for (a, b) in adjacency:
                if a == node and b not in visited:
                    visited.add(b)
                    queue.append(b)
        regions_connected = goal_r in visited
        print(f"  Connectivity: start {start_r} -> goal {goal_r}: "
              f"{'CONNECTED' if regions_connected else 'DISCONNECTED'}")
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
    gcs_solve_time = time.time() - t_gcs_start
    print(f"GCS Solved: {gcs_result.is_success()}")

    # Everything printed above is also recorded. A runtime with no recorded
    # outcome is not interpretable -- gcs_solve_time used to be stored without
    # gcs_success, so the tables reported a GCS time that could not be checked
    # against whether GCS had actually solved.
    timings_partial = {
        "iris_total_time": iris_total_time,
        # Number of seed points the IRIS total covers. Recorded so downstream
        # reporting can show a per-seed figure without hardcoding the seed count.
        "iris_num_seeds": len(SEEDS),
        "iris_num_regions": len(regions),
        "iris_start_in_regions": start_in_region,
        "iris_goal_in_regions": goal_in_region,
        "iris_pairwise_intersections": len(intersections),
        "regions_connected": regions_connected,
        "gcs_solve_time": gcs_solve_time,
        "gcs_success": bool(gcs_result.is_success()),
    }
    return iris_options, regions, gcs_traj, gcs_result, timings_partial


def run_rrt_shortcut_trajopt_toppra(cfg, iris_options, gcs_traj, gcs_result,
                                     checker, plant, diagram, meshcat,
                                     rng_seed, save_meshcat):
    """
    Run the RRT, Shortcut, TrajOpt, and TOPPRA stages of the pipeline.
    """
    config_name  = cfg["name"]
    short_name   = cfg.get("short_name", config_name)
    use_ift      = cfg["use_ift"]
    handling     = cfg.get("ift_handling", IftSingularityHandling.kPseudoinverse)
    old_reach    = cfg["old_reach"]
    # No benchmark row enables the psi-singularity constraint; the knob is
    # kept so a configuration can opt in without editing the pipeline.
    use_psi      = cfg.get("use_psi", False)
    lmbda        = cfg.get("lambda", 0.0)
    svt_eps      = cfg.get("svt_epsilon", None)
    svt_l_max    = cfg.get("svt_lambda_max", None)
    exit_on_fail = cfg.get("exit_on_failure", False)
    use_aniso    = cfg.get("use_anisotropic_damping", False)
    reach_type   = cfg.get("reach_type", ReachabilityType.kProbing if not old_reach else ReachabilityType.kDirect)

    grasp_distance = 0.6
    b_thresh     = cfg.get("boundary_threshold", 2.46)
    b_eps        = cfg.get("boundary_epsilon", 1e-6)
    config = BimanualConfig(shoulder_up=True, elbow_up=True, wrist_up=False,
                            grasp_distance=grasp_distance,
                            clipping_margin=1e-4, clipping_margin_psi=1e-4,
                            boundary_threshold=b_thresh, boundary_epsilon=b_eps)
    ad_config = AutoDiffConfig(use_ift=use_ift, ift_handling=handling, lambda_=lmbda,
                                   svt_epsilon=svt_eps, svt_lambda_max=svt_l_max,
                                   use_anisotropic_damping=use_aniso)

    trial_timings = {}

    # ── RRT ───────────────────────────────────────────────────────────────────
    print(f"Running BiRRT (seed {rng_seed})...")

    def random_config():
        return np.random.uniform(low=DOMAIN_LOWER, high=DOMAIN_UPPER)

    reach_con = make_reach_constraint(reach_type, config, ad_config)
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
    np.random.seed(rng_seed)
    t_rrt_start = time.time()
    path = rrt_planner.plan(Q_TILDE_BOTTOM, Q_TILDE_TOP, rrt_options)
    trial_timings["rrt_plan_time"] = time.time() - t_rrt_start
    trial_timings["rrt_success"] = path is not None

    shortcut_path = None
    if path is not None:
        print(f"RRT Succeeded! Found path with {len(path)} waypoints")
        np.random.seed(rng_seed)
        t_sc_start = time.time()
        shortcut_path = shortcut.shortcut(path.copy(), validity_checker,
                                          num_tries=1e2,
                                          check_size=rrt_options.check_size)
        trial_timings["rrt_shortcut_time"] = time.time() - t_sc_start
    else:
        print("RRT Failed.")

    # ── Trajectory optimisation + TOPPRA ──────────────────────────────────────
    print("Running KinematicTrajectoryOptimization...")
    try:
        # GCS → TOPPRA (only if it's the first trial, since GCS is deterministic)
        if rng_seed == 0 and gcs_result.is_success():
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
                                        meshcat, "GCS", short_name, trial_timings,
                                        exit_on_toppra_failure=exit_on_fail,
                                        save_html=save_meshcat)
                except Exception as ce:
                    print(f"  WARNING: Failed to construct clean GCS composite: {ce}")
                    # Retime the original trajectory, but keep the "GCS" label so
                    # the record still has a GCS entry. Using a separate label here
                    # made a fallback look like a *missing* GCS row instead of a
                    # recorded one; the flag below is what marks the difference.
                    trial_timings["gcs_toppra_used_original"] = True
                    trial_timings["gcs_toppra_clean_composite_error"] = str(ce)
                    run_toppra_and_save(gcs_traj, iris_options, plant, diagram,
                                        meshcat, "GCS", short_name, trial_timings,
                                        exit_on_toppra_failure=exit_on_fail,
                                        save_html=save_meshcat)
            else:
                print("  WARNING: All GCS segments were zero-length! Skipping TOPPRA for GCS.")
                trial_timings.setdefault("toppra_success", {})["GCS"] = False
                trial_timings.setdefault("toppra_error", {})["GCS"] = (
                    "all GCS segments were zero-length")

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
                                meshcat, "RRT", short_name, trial_timings,
                                exit_on_toppra_failure=exit_on_fail,
                                save_html=save_meshcat)

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
            to_result  = solver.Solve(trajopt.prog(), None, trajopt_opts)
            trial_timings["trajopt_solve_time"] = time.time() - t_to_start
            trial_timings["trajopt_solve_success"] = to_result.is_success()
            print(f"Trajectory Optimization Solved: {to_result.is_success()}")

            if to_result.is_success():
                trajopt_traj = trajopt.ReconstructTrajectory(to_result)
            else:
                print("  WARNING: TrajOpt failed. Falling back to INITIAL (RRT) trajectory.")
                trajopt_traj = initial_traj
            
            run_toppra_and_save(trajopt_traj, iris_options, plant, diagram,
                                meshcat, "Trajopt", short_name, trial_timings,
                                exit_on_toppra_failure=exit_on_fail,
                                save_html=save_meshcat)

    except Exception as e:
        import traceback
        print(f"RRT-chain error for trial {rng_seed}: {e}")
        traceback.print_exc()

    return trial_timings


def run_pipeline(cfg, meshcat, all_timing_results, num_trials=1, meshcat_save="first"):
    """
    Run the full IRIS → GCS → RRT → TrajOpt → TOPPRA pipeline for one
    configuration dict.  Results are appended to all_timing_results.
    """
    config_name  = cfg["name"]
    short_name   = cfg.get("short_name", config_name)
    exit_on_fail = cfg.get("exit_on_failure", False)

    print(f"\n{'='*57}")
    print(f"Testing Configuration: {config_name}")
    print(f"Number of RRT trials: {num_trials}")
    print(f"{'='*57}\n")

    checker, plant, diagram = build_checker_and_plant(meshcat)

    # 1. IRIS and GCS (Deterministic)
    iris_options, regions, gcs_traj, gcs_result, test_timings = run_iris_and_gcs(
        cfg, meshcat, checker, plant, diagram)

    if not gcs_result.is_success() and exit_on_fail:
        print("GCS failed and exit_on_failure=True. Aborting pipeline.")
        all_timing_results.append(test_timings)
        return

    # 2. RRT, Shortcut, TrajOpt, TOPPRA (Stochastic trials)
    rrt_trials = []
    for i in range(num_trials):
        try:
            save_meshcat = (meshcat_save == "all") or (meshcat_save == "first" and i == 0)
            trial_results = run_rrt_shortcut_trajopt_toppra(
                cfg, iris_options, gcs_traj, gcs_result, checker, plant, diagram, meshcat,
                rng_seed=i, save_meshcat=save_meshcat)
            rrt_trials.append(trial_results)
        except Exception as e:
            print(f"\n[ERROR] Trial {i} failed with error: {e}")
            rrt_trials.append({
                "rrt_success": False,
                "trajopt_solve_success": False,
                "error": str(e)
            })

    # ── Statistical Aggregation ───────────────────────────────────────────────
    def get_stats(data_list, key):
        vals = [d[key] for d in data_list if key in d]
        if not vals: return 0.0, 0.0, 0.0, 0.0
        return float(np.mean(vals)), float(np.std(vals)), float(np.min(vals)), float(np.max(vals))

    # Timing stats
    timing_keys = ["rrt_plan_time", "rrt_shortcut_time", "trajopt_solve_time"]
    for k in timing_keys:
        mean_v, std_v, min_v, max_v = get_stats(rrt_trials, k)
        test_timings[f"{k}_mean"] = mean_v
        test_timings[f"{k}_std"]  = std_v
        # For backward compatibility and summary table, use trial 0 if num_trials=1, else mean
        test_timings[k] = rrt_trials[0].get(k, 0.0) if num_trials == 1 else mean_v

    # TOPPRA timings
    toppra_labels = ["GCS", "RRT", "Trajopt"]
    test_timings["toppra"] = {}
    for label in toppra_labels:
        # NOTE: `toppra_durations` is the DURATION of the retimed trajectory, and
        # `toppra` is the wall time TOPPRA took to compute it. These used to be
        # conflated here -- the aggregate wrote durations into test_timings["toppra"],
        # which the tables label as a runtime. That is why the reported "TOPPRA"
        # column tracked trajopt failure rate: on failure the code retimes the RRT
        # spline instead, whose duration is ~7.4 s against ~2.0 s for a trajopt
        # trajectory, while the actual solve time is ~0.9 s either way.
        # Both are now aggregated, under distinct names.
        #
        # Only successful solves are averaged. A TOPPRA call that returned None
        # or a non-finite duration still recorded a runtime, and including it
        # would report a solve time for a solve that produced nothing.
        # Trials written before `toppra_success` existed encode the outcome only
        # by key presence, so fall back to that rather than discarding them.
        def _succeeded(d):
            if label in d.get("toppra_success", {}):
                return d["toppra_success"][label]
            return (label in d.get("toppra", {})
                    and not d.get("toppra_backup", {}).get(label, False))

        attempted = [d for d in rrt_trials
                     if label in d.get("toppra_success", {})
                     or label in d.get("toppra", {})
                     or label in d.get("toppra_backup", {})]
        succeeded = [d for d in attempted if _succeeded(d)]
        if attempted:
            test_timings.setdefault("toppra_num_attempts", {})[label] = len(attempted)
            test_timings.setdefault("toppra_num_success", {})[label] = len(succeeded)
            test_timings.setdefault("toppra_success_rate", {})[label] = (
                len(succeeded) / len(attempted))

        solve_vals = [d["toppra"][label] for d in succeeded if label in d.get("toppra", {})]
        if solve_vals:
            test_timings["toppra"][label] = float(np.mean(solve_vals))
            test_timings[f"toppra_{label}_std"] = float(np.std(solve_vals))

        dur_vals = [d["toppra_durations"][label] for d in succeeded
                    if label in d.get("toppra_durations", {})]
        if dur_vals:
            test_timings.setdefault("toppra_durations", {})[label] = float(np.mean(dur_vals))
            test_timings[f"toppra_duration_{label}_std"] = float(np.std(dur_vals))

    # Success rates
    test_timings["rrt_success_rate"]     = float(np.mean([d.get("rrt_success", False) for d in rrt_trials]))
    test_timings["trajopt_success_rate"] = float(np.mean([d.get("trajopt_solve_success", False) for d in rrt_trials]))
    
    # Single trial success flags (for trial 0 compatibility)
    test_timings["rrt_success"] = rrt_trials[0].get("rrt_success", False)
    test_timings["trajopt_solve_success"] = rrt_trials[0].get("trajopt_solve_success", False)

    # Store all trials for full detail
    test_timings["rrt_trials"] = rrt_trials
    test_timings["config"] = config_name

    all_timing_results.append(test_timings)

    print("\nConfiguration complete!")

    os.makedirs("out", exist_ok=True)
    per_config_path = f"out/timing_{short_name}.json"
    with open(per_config_path, "w") as f:
        json.dump(test_timings, f, indent=4)
    print(f"  Per-config results saved to {per_config_path}")

    with open("out/timing_results_full_comparison.json", "w") as f:
        json.dump(all_timing_results, f, indent=4)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run full comparative benchmark.")
    parser.add_argument("--num-rrt-trials", type=int, default=1, help="Number of RRT + Shortcut + TrajOpt trials to run (default: 1)")
    parser.add_argument("--meshcat-save", type=str, choices=["all", "first", "none"], default="first", help="Which Meshcat HTML snapshots to save (default: first)")
    parser.add_argument("--skip-existing", action="store_true", help="Skip configurations that already have a JSON result file in out/")
    args = parser.parse_args()
    # ──────────────────────────────────────────────────────────────────────────
    # CONFIGURATIONS
    # Comment out any entry to skip that configuration.
    # ──────────────────────────────────────────────────────────────────────────
    CONFIGS = [
        # ── Baselines ─────────────────────────────────────────────────────────
        {
            "name":        "Current best (AD, New Reach)",
            "use_ift":     False,
            "old_reach":   False,
            "short_name":  "current_best",
        },
        {
            "name":        "Baseline old (AD, Old Reach)",
            "use_ift":     False,
            "old_reach":   True,
            "short_name":  "baseline_old",
        },
        {
            "name":        "AD, Boundary Reach",
            "use_ift":     False,
            "old_reach":   False,
            "reach_type":  ReachabilityType.kBoundary,
            "short_name":  "boundary_reach",
            "boundary_threshold": 2.46,
        },

        # ── IFT Zero Gradients ────────────────────────────────────────────────
        {
            "name":        "IFT Zero, Direct Reach",
            "use_ift":     True,
            "ift_handling": IftSingularityHandling.kZero,
            "old_reach":   True,
            "short_name":  "ift_zero_direct",
        },
        {
            "name":        "IFT Zero, Boundary Reach",
            "use_ift":     True,
            "ift_handling": IftSingularityHandling.kZero,
            "old_reach":   False,
            "reach_type":  ReachabilityType.kBoundary,
            "short_name":  "ift_zero_boundary",
            "boundary_threshold": 2.46,
        },

        # ── IFT Pseudoinverse ─────────────────────────────────────────────────
        {
            "name":        "IFT Pseudo, Direct Reach",
            "use_ift":     True,
            "ift_handling": IftSingularityHandling.kPseudoinverse,
            "old_reach":   True,
            "short_name":  "ift_pseudo_direct",
        },
        {
            "name":        "IFT Pseudo, Boundary Reach",
            "use_ift":     True,
            "ift_handling": IftSingularityHandling.kPseudoinverse,
            "old_reach":   False,
            "reach_type":  ReachabilityType.kBoundary,
            "short_name":  "ift_pseudo_boundary",
            "boundary_threshold": 2.46,
        },

        # ── IFT Levenberg-Marquardt (Constant) ────────────────────────────────
        {
            "name":        "IFT LM Const, Direct Reach",
            "use_ift":     True,
            "ift_handling": IftSingularityHandling.kLevenbergMarquardt,
            "old_reach":   True,
            "lambda":      1e-5,
            "short_name":  "ift_lm_const_direct",
        },
        {
            "name":        "IFT LM Const, Boundary Reach",
            "use_ift":     True,
            "ift_handling": IftSingularityHandling.kLevenbergMarquardt,
            "old_reach":   False,
            "reach_type":  ReachabilityType.kBoundary,
            "lambda":      1e-5,
            "short_name":  "ift_lm_const_boundary",
            "boundary_threshold": 2.46,
        },

        # ── IFT Levenberg-Marquardt (SVT) ─────────────────────────────────────
        {
            "name":        "IFT LM SVT, Direct Reach",
            "use_ift":     True,
            "ift_handling": IftSingularityHandling.kLevenbergMarquardt,
            "old_reach":   True,
            "svt_epsilon": 0.01,
            "svt_lambda_max": 0.005,
            "short_name":  "ift_lm_svt_direct",
        },
        {
            "name":        "IFT LM SVT, Boundary Reach",
            "use_ift":     True,
            "ift_handling": IftSingularityHandling.kLevenbergMarquardt,
            "old_reach":   False,
            "reach_type":  ReachabilityType.kBoundary,
            "svt_epsilon": 0.01,
            "svt_lambda_max": 0.005,
            "short_name":  "ift_lm_svt_boundary",
            "boundary_threshold": 2.46,
        },

        # ── IFT Full Newton ───────────────────────────────────────────────────
        {
            "name":        "IFT Newton, Direct Reach",
            "use_ift":     True,
            "ift_handling": IftSingularityHandling.kFullNewton,
            "old_reach":   True,
            "lambda":      1e-5,
            "short_name":  "ift_newton_direct",
        },
        {
            "name":        "IFT Newton, Boundary Reach",
            "use_ift":     True,
            "ift_handling": IftSingularityHandling.kFullNewton,
            "old_reach":   False,
            "reach_type":  ReachabilityType.kBoundary,
            "lambda":      1e-5,
            "short_name":  "ift_newton_boundary",
            "boundary_threshold": 2.46,
        },

        # ── IFT Residual Damping (Standard) ───────────────────────────────────
        {
            "name":        "IFT Res Std, Direct Reach",
            "use_ift":     True,
            "ift_handling": IftSingularityHandling.kResidualDamping,
            "old_reach":   True,
            "lambda":      10.0,
            "short_name":  "ift_res_std_direct",
        },
        {
            "name":        "IFT Res Std, Boundary Reach",
            "use_ift":     True,
            "ift_handling": IftSingularityHandling.kResidualDamping,
            "old_reach":   False,
            "reach_type":  ReachabilityType.kBoundary,
            "lambda":      10.0,
            "short_name":  "ift_res_std_boundary",
            "boundary_threshold": 2.46,
        },

        # ── IFT Residual Damping (Anisotropic) ────────────────────────────────
        {
            "name":        "IFT Res Aniso, Direct Reach",
            "use_ift":     True,
            "ift_handling": IftSingularityHandling.kResidualDamping,
            "use_anisotropic_damping": True,
            "old_reach":   True,
            "lambda":      10.0,
            "short_name":  "ift_res_aniso_direct",
        },
        {
            "name":        "IFT Res Aniso, Boundary Reach",
            "use_ift":     True,
            "ift_handling": IftSingularityHandling.kResidualDamping,
            "use_anisotropic_damping": True,
            "old_reach":   False,
            "reach_type":  ReachabilityType.kBoundary,
            "lambda":      10.0,
            "short_name":  "ift_res_aniso_boundary",
            "boundary_threshold": 2.46,
        },
    ]
    # ──────────────────────────────────────────────────────────────────────────

    meshcat           = StartMeshcat()
    all_timing_results = []

    print(f"\n\n{'#'*80}")
    print(f"RUNNING COMPARATIVE BENCHMARK ({len(CONFIGS)} CONFIGURATIONS)")
    print(f"RRT TRIALS PER CONFIG: {args.num_rrt_trials}")
    print(f"{'#'*80}\n")

    # Guard against the failure mode this benchmark has hit three times: a knob
    # declared in a CONFIGS entry but never read by the code that builds
    # BimanualConfig/AutoDiffConfig, so the row is labelled one thing and
    # measures another. Nothing else validates these hand-written dicts.
    for cfg in CONFIGS:
        unknown = set(cfg) - KNOWN_CONFIG_KEYS
        if unknown:
            raise KeyError(
                f"Configuration {cfg.get('name', '<unnamed>')!r} declares key(s) "
                f"{sorted(unknown)} that no code reads. Either wire them through "
                f"or remove them -- a silently ignored key means this row does not "
                f"measure what its label claims.")

    for cfg in CONFIGS:
        short_name = cfg.get("short_name", cfg["name"])
        if args.skip_existing and os.path.exists(f"out/timing_{short_name}.json"):
            print(f"Skipping {cfg['name']} (results already exist in out/timing_{short_name}.json)")
            # Load existing results to include in final summary
            with open(f"out/timing_{short_name}.json", "r") as f:
                all_timing_results.append(json.load(f))
            continue
            
        run_pipeline(cfg, meshcat, all_timing_results, 
                     num_trials=args.num_rrt_trials, 
                     meshcat_save=args.meshcat_save)

    # Final summary table
    max_name_len = max(len(cfg["name"]) for cfg in CONFIGS)
    table_width = max_name_len + 110 
    
    print(f"\n\n{'='*table_width}")
    print(f"{'CONFIGURATION':<{max_name_len}} | {'IRIS':<8} | {'GCS':<8} | {'G-TOP':<8} | "
          f"{'TrjOpt':<15} | {'T-TOP':<15} | {'RRT':<15} | {'R-TOP':<15}")
    print(f"{'-'*table_width}")
    
    baselines = all_timing_results[:3]
    experimentals = sorted(all_timing_results[3:], 
                           key=lambda x: x.get('iris_total_time', 0.0), 
                           reverse=True)
    
    def print_res(res):
        iris     = f"{res.get('iris_total_time',   0.0):.2f}"
        gcs      = f"{res.get('gcs_solve_time',    0.0):.2f}"
        
        traj_val = res.get('trajopt_solve_time', 0.0)
        traj_std = res.get('trajopt_solve_time_std', 0.0)
        traj = f"{traj_val:.2f}" + (f" (±{traj_std:.2f})" if traj_std > 1e-3 else "")
        
        rrt_val = res.get('rrt_plan_time', 0.0)
        rrt_std = res.get('rrt_plan_time_std', 0.0)
        rrt_time = f"{rrt_val:.2f}" + (f" (±{rrt_std:.2f})" if rrt_std > 1e-3 else "")
        
        toppra   = res.get("toppra", {})
        g_top    = f"{toppra.get('GCS',    0.0):.2f}"
        
        t_top_val = toppra.get('Trajopt', 0.0)
        t_top_std = res.get('toppra_Trajopt_std', 0.0)
        t_top = f"{t_top_val:.2f}" + (f" (±{t_top_std:.2f})" if t_top_std > 1e-3 else "")
        
        r_top_val = toppra.get('RRT', 0.0)
        r_top_std = res.get('toppra_RRT_std', 0.0)
        r_top = f"{r_top_val:.2f}" + (f" (±{r_top_std:.2f})" if r_top_std > 1e-3 else "")
        
        s_rate = res.get('trajopt_success_rate', 1.0)
        s_str  = f" [{s_rate*100:.0f}% succ]" if s_rate < 1.0 else ""
        
        print(f"{res['config']:<{max_name_len}} | {iris:<8} | {gcs:<8} | {g_top:<8} | "
              f"{traj:<15} | {t_top:<15} | {rrt_time:<15} | {r_top:<15}{s_str}")

    for res in baselines:
        print_res(res)
    
    print(f"{'-'*table_width}") # Horizontal rule separating baselines
    
    for res in experimentals:
        print_res(res)
    print(f"{'='*table_width}\n")

    # Final save
    os.makedirs("out", exist_ok=True)
    out_path = "out/timing_results_full_comparison.json"
    with open(out_path, "w") as f:
        json.dump(all_timing_results, f, indent=4)
    print(f"Full timing results saved to {out_path}")
    print("\nAll comparative benchmark runs completed successfully!")


if __name__ == "__main__":
    main()
