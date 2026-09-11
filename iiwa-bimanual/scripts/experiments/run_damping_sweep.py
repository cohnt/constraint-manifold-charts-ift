"""
run_damping_sweep.py

Performs a grid search over various IFT damping and regularization parameters
to optimize IRIS runtime performance while ensuring GCS and TrajOpt success.

Scope: direct reachability only. This script never sets `reach_type`, so every
configuration here uses the default (kDirect for old_reach, kProbing otherwise).
Boundary-reachability tuning -- in particular the threshold tau -- lives in
run_boundary_sweep.py.
"""

import os
import sys
import numpy as np
import time
import json
import argparse
# Path setup
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
)

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
)

# Shared waypoints and seeds
Q_TILDE_BOTTOM = np.array([-0.6430910102907225, 1.9156121024586796, -1.7968254667817805,
                             1.2945447141185198, -0.023834531305537934, -0.876966810663043,
                             -1.7041643160834519, 1.45])
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
    Q_TILDE_MIDDLE := np.array([-0.5997312520566763, 1.489780849654964, -1.4739679827359913,
                             1.2905366081785483, -0.04421061906813227, -0.8793712572715165,
                             -1.1603461715511334, 1.45]),
]

DOMAIN_LOWER = np.hstack((iiwa_limits_lower, [0.0]))
DOMAIN_UPPER = np.hstack((iiwa_limits_upper, [2.0 * np.pi]))

def build_checker_and_plant(meshcat):
    directives_file = os.path.join(common.RepoDir(), "models/old_shelves.dmd.yaml")
    params  = CollisionCheckerParams()
    builder = RobotDiagramBuilder(time_step=0.001)

    plant  = builder.plant()
    parser = Parser(plant)
    parser.package_map().AddPackageXml(os.path.join(common.RepoDir(), "package.xml"))
    ProcessModelDirectives(LoadModelDirectives(directives_file), parser)

    params.robot_model_instances = [
        plant.GetModelInstanceByName("iiwa_left"),
        plant.GetModelInstanceByName("iiwa_right"),
    ]
    plant.Finalize()
    diagram = builder.Build()
    params.model = diagram
    params.edge_step_size = 0.01
    checker = SceneGraphCollisionChecker(params)
    return checker, plant, diagram

def run_pipeline(cfg, meshcat, skip_trajopt=False):
    config_name  = cfg["name"]
    use_ift      = cfg["use_ift"]
    handling     = cfg.get("ift_handling", IftSingularityHandling.kPseudoinverse)
    old_reach    = cfg["old_reach"]
    use_psi      = cfg["use_psi"]
    lmbda        = cfg.get("lambda", 0.0)
    svt_eps      = cfg.get("svt_epsilon", None)
    svt_l_max    = cfg.get("svt_lambda_max", None)
    use_aniso    = cfg.get("use_anisotropic_damping", False)

    test_timings = {"config": config_name}
    print(f"Testing: {config_name}")

    checker, plant, diagram = build_checker_and_plant(meshcat)
    grasp_distance = 0.6
    config = BimanualConfig(shoulder_up=True, elbow_up=True, wrist_up=False,
                            grasp_distance=grasp_distance)

    # IRIS
    iris_options = IrisNp2Options()
    ad_config_iris = AutoDiffConfig(use_ift=use_ift, ift_handling=handling, lambda_=lmbda,
                                   svt_epsilon=svt_eps, svt_lambda_max=svt_l_max,
                                   use_anisotropic_damping=use_aniso)
    iris_options.parameterization = MakeParameterization(config, ad_config_iris)
    iris_options.sampled_iris_options.random_seed               = 2
    iris_options.sampled_iris_options.max_iterations            = 1
    iris_options.sampled_iris_options.relax_margin              = True
    iris_options.sampled_iris_options.epsilon                   = 0.01
    iris_options.sampled_iris_options.delta                     = 0.01
    iris_options.add_hyperplane_if_solve_fails                  = True
    iris_options.solver_options.SetOption(SnoptSolver().solver_id(), "Major iterations limit", 200)

    iris_prog   = MathematicalProgram()
    q_tilde_vars = iris_prog.NewContinuousVariables(8, "q_tilde")
    iris_options.sampled_iris_options.prog_with_additional_constraints = iris_prog

    ad_config = AutoDiffConfig(use_ift=use_ift, ift_handling=handling, lambda_=lmbda,
                                   svt_epsilon=svt_eps, svt_lambda_max=svt_l_max,
                                   use_anisotropic_damping=use_aniso)
    reach_con = (OldStyleReachableConstraint(config, ad_config, 1e-4)
                 if old_reach else IiwaBimanualReachableConstraint(config))
    iris_prog.AddConstraint(reach_con, q_tilde_vars)
    jl_con = IiwaBimanualJointLimitConstraint(iiwa_limits_lower, iiwa_limits_upper, config, ad_config)
    iris_prog.AddConstraint(jl_con, q_tilde_vars)

    domain = HPolyhedron.MakeBox(DOMAIN_LOWER, DOMAIN_UPPER)

    t_iris_start = time.time()
    regions = []
    for seed in SEEDS:
        region = IrisNp2(checker, Hyperellipsoid.MakeHypersphere(1e-2, seed), domain, iris_options)
        regions.append(region.ReduceInequalities())
    test_timings["iris_total_time"] = time.time() - t_iris_start

    # GCS
    gcs = GcsTrajectoryOptimization(8)
    gcs.AddPathContinuityConstraints(1)
    gcs.AddPathContinuityConstraints(2)
    main_graph  = gcs.AddRegions(regions, 3, h_min=0.01, h_max=100)
    start_graph = gcs.AddRegions([Point(Q_TILDE_BOTTOM)], 0)
    goal_graph  = gcs.AddRegions([Point(Q_TILDE_TOP)], 0)
    gcs.AddEdges(start_graph, main_graph)
    gcs.AddEdges(main_graph, goal_graph)
    gcs.AddPathLengthCost()
    gcs.AddPathEnergyCost()
    gcs.AddTimeCost()
    gcs.AddVelocityBounds(-np.ones(8), np.ones(8))

    gcs_options = GraphOfConvexSetsOptions()
    gcs_options.convex_relaxation = True

    t_gcs_start = time.time()
    gcs_traj, gcs_result = gcs.SolvePath(start_graph, goal_graph, gcs_options)
    test_timings["gcs_solve_time"] = time.time() - t_gcs_start
    test_timings["gcs_success"] = gcs_result.is_success()

    if skip_trajopt or not gcs_result.is_success():
        test_timings["trajopt_solve_time"] = 0.0
        test_timings["trajopt_solve_success"] = False
        return test_timings

    # RRT (for TrajOpt seed)
    def random_config(): return np.random.uniform(low=DOMAIN_LOWER, high=DOMAIN_UPPER)
    def validity_checker(q_tilde):
        if np.any(reach_con.Eval(q_tilde) > np.ones(4)): return False
        if np.any(reach_con.Eval(q_tilde) < -np.ones(4)): return False
        q_full = iris_options.parameterization.get_parameterization_double()(q_tilde)
        q_sub  = q_full[7:]
        if np.any(q_sub < iiwa_limits_lower) or np.any(q_sub > iiwa_limits_upper): return False
        return checker.CheckConfigCollisionFree(q_full)

    rrt_options = rrt.RRTOptions(step_size=2e-1, check_size=1e-2, max_vertices=1e4, max_iters=1e6)
    rrt_planner = rrt.BiRRT(random_config, validity_checker)
    np.random.seed(0)
    path = rrt_planner.plan(Q_TILDE_BOTTOM, Q_TILDE_TOP, rrt_options)
    
    if path is not None:
        shortcut_path = shortcut.shortcut(path, validity_checker, num_tries=100, check_size=rrt_options.check_size)
        
        # TrajOpt
        spline_order = 4
        ctrl_pts = np.array([pt for pt in shortcut_path]).T
        basis = BsplineBasis(spline_order, ctrl_pts.shape[1], initial_parameter_value=0, final_parameter_value=1)
        initial_traj = BsplineTrajectory(basis, ctrl_pts)
        trajopt = KinematicTrajectoryOptimization(initial_traj)
        # Controlled arm only; psi has no physical joint limit (it is bounded
        # indirectly via the subordinate arm, inside FullFeasibilityConstraint).
        trajopt.AddPositionBounds(np.hstack((iiwa_limits_lower, [-np.inf])),
                                  np.hstack((iiwa_limits_upper, [ np.inf])))
        trajopt.AddPathPositionConstraint(Q_TILDE_BOTTOM, Q_TILDE_BOTTOM, 0)
        trajopt.AddPathPositionConstraint(Q_TILDE_TOP, Q_TILDE_TOP, 1)

        min_distance = 0.001
        mdc_context = diagram.CreateDefaultContext()
        mdc_plant_ctx = plant.GetMyContextFromRoot(mdc_context)
        min_dist_con = MinimumDistanceLowerBoundConstraint(plant, min_distance, mdc_plant_ctx, None, 0.05)
        full_feas_con = FullFeasibilityConstraint(iiwa_limits_lower, iiwa_limits_upper, config, ad_config, min_dist_con, old_reach, use_psi)
        for s in np.linspace(0, 1, 50):
            trajopt.AddPathPositionConstraint(full_feas_con, s)

        trajopt.prog().AddCost(IiwaBimanualPathCost(8, trajopt.num_control_points(), config, ad_config, True), trajopt.control_points().flatten(order='F'))
        
        trajopt_opts = SolverOptions()
        trajopt_opts.SetOption(SnoptSolver().solver_id(), "Major optimality tolerance", 1e-1)
        trajopt_opts.SetOption(SnoptSolver().solver_id(), "Time Limit", 30)

        t_to_start = time.time()
        to_result = SnoptSolver().Solve(trajopt.prog(), None, trajopt_opts)
        test_timings["trajopt_solve_time"] = time.time() - t_to_start
        test_timings["trajopt_solve_success"] = to_result.is_success()
    else:
        test_timings["trajopt_solve_time"] = 0.0
        test_timings["trajopt_solve_success"] = False

    return test_timings

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="Include TrajOpt in sweep")
    args = parser.parse_args()

    meshcat = StartMeshcat()
    
    # Baselines
    BASELINES = [
        {"name": "Current best (AD, New Reach)", "use_ift": False, "old_reach": False, "use_psi": False, "short_name": "base_ad_new"},
        {"name": "Baseline old (AD, Old Reach)", "use_ift": False, "old_reach": True, "use_psi": False, "short_name": "base_ad_old"},
        {"name": "Sanity check (IFT, New Reach, NO Psi)", "use_ift": True, "ift_handling": IftSingularityHandling.kPseudoinverse, "old_reach": False, "use_psi": False, "short_name": "base_ift_new"},
    ]

    SWEEP_CONFIGS = []
    
    # 1. LM Constant
    for l in [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]:
        SWEEP_CONFIGS.append({
            "name": f"LM Const λ={l:.1e}", "use_ift": True, "old_reach": True, "use_psi": False,
            "ift_handling": IftSingularityHandling.kLevenbergMarquardt, "lambda": l, "short_name": f"lm_const_{l:.1e}"
        })
    
    # 2. LM SVT
    epsilons = [0.005, 0.01, 0.02, 0.05, 0.10]
    lambda_maxes = [0.001, 0.005, 0.01, 0.05, 0.1]
    for e in epsilons:
        for lm in lambda_maxes:
            SWEEP_CONFIGS.append({
                "name": f"LM SVT ε={e}, λmax={lm}", "use_ift": True, "old_reach": True, "use_psi": False,
                "ift_handling": IftSingularityHandling.kLevenbergMarquardt, "svt_epsilon": e, "svt_lambda_max": lm, "short_name": f"lm_svt_e{e}_l{lm}"
            })
            
    # 3. Residual Std
    for l0 in [0.1, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0]:
        SWEEP_CONFIGS.append({
            "name": f"Res Std λ0={l0}", "use_ift": True, "old_reach": True, "use_psi": False,
            "ift_handling": IftSingularityHandling.kResidualDamping, "lambda": l0, "use_anisotropic_damping": False, "short_name": f"res_std_{l0}"
        })

    # 4. Residual Aniso
    for alpha in [0.5, 1.0, 5.0, 10.0, 25.0, 50.0, 100.0]:
        SWEEP_CONFIGS.append({
            "name": f"Res Aniso α={alpha}", "use_ift": True, "old_reach": True, "use_psi": False,
            "ift_handling": IftSingularityHandling.kResidualDamping, "lambda": alpha, "use_anisotropic_damping": True, "short_name": f"res_aniso_{alpha}"
        })

    # 5. Full Newton
    for l_floor in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]:
        SWEEP_CONFIGS.append({
            "name": f"Full Newton λ={l_floor:.1e}", "use_ift": True, "old_reach": True, "use_psi": False,
            "ift_handling": IftSingularityHandling.kFullNewton, "lambda": l_floor, "short_name": f"newton_{l_floor:.1e}"
        })

    all_results = []
    
    print(f"Starting sweep with {len(BASELINES)} baselines and {len(SWEEP_CONFIGS)} configs...")
    
    for cfg in BASELINES + SWEEP_CONFIGS:
        res = run_pipeline(cfg, meshcat, skip_trajopt=not args.full)
        all_results.append(res)
        
        # Intermediate save
        os.makedirs("out/sweep", exist_ok=True)
        with open(f"out/sweep/timing_{cfg['short_name']}.json", "w") as f:
            json.dump(res, f, indent=4)

    # Final summary table
    print("\n\n" + "="*90)
    print(f"{'Configuration':<45} | {'IRIS':<8} | {'GCS':<5} | {'TrjOpt':<6}")
    print("-" * 90)
    
    sorted_results = sorted(all_results, key=lambda x: x.get('iris_total_time', 1000.0))
    for res in sorted_results:
        iris = f"{res.get('iris_total_time', 0.0):.2f}"
        gcs = "OK" if res.get('gcs_success') else "FAIL"
        traj = "OK" if res.get('trajopt_solve_success') else "FAIL"
        if not args.full: traj = "N/A"
        print(f"{res['config']:<45} | {iris:<8} | {gcs:<5} | {traj:<6}")
    print("="*90)

    with open("out/damping_sweep_results.json", "w") as f:
        json.dump(all_results, f, indent=4)

if __name__ == "__main__":
    main()
