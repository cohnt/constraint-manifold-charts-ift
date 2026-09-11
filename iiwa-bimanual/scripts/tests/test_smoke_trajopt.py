import os
import sys
import numpy as np
import time
import unittest

# Add local path to finding the bindings
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_dir = os.path.abspath(os.path.join(script_dir, "../.."))
sys.path.append(os.path.join(repo_dir, "cpp_parameterization/python"))
sys.path.append(repo_dir)

try:
    from iiwa_ik import (
        BimanualConfig,
        MakeParameterization,
        IiwaBimanualReachableConstraint,
        OldStyleReachableConstraint,
        IiwaBimanualJointLimitConstraint,
        FullFeasibilityConstraint,
        IiwaBimanualPsiSingularityConstraint,
        IiwaBimanualPathCost,
        IftSingularityHandling,
        AutoDiffConfig,
        ReachabilityType
    )
except ImportError:
    sys.path.append(os.path.join(os.getcwd(), "cpp_parameterization/python"))
    sys.path.append(os.getcwd())
    from iiwa_ik import (
        BimanualConfig,
        MakeParameterization,
        IiwaBimanualReachableConstraint,
        OldStyleReachableConstraint,
        IiwaBimanualJointLimitConstraint,
        FullFeasibilityConstraint,
        IiwaBimanualPsiSingularityConstraint,
        IiwaBimanualPathCost,
        IftSingularityHandling,
        AutoDiffConfig,
        ReachabilityType
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
    MathematicalProgram,
    BsplineBasis,
    BsplineTrajectory,
    KinematicTrajectoryOptimization,
    MinimumDistanceLowerBoundConstraint,
    SolverOptions,
    SnoptSolver,
    Toppra,
    CalcGridPointsOptions,
    InitializeAutoDiff,
    ExtractGradient,
    FunctionHandleTrajectory
)

class TestTrajOptSmoke(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Build checker once
        directives_file = os.path.join(common.RepoDir(), "models/old_shelves.dmd.yaml")
        params = CollisionCheckerParams()
        builder = RobotDiagramBuilder(time_step=0.001)
        
        plant = builder.plant()
        parser = Parser(plant)
        package_xml_path = os.path.join(common.RepoDir(), "package.xml")
        parser.package_map().AddPackageXml(package_xml_path)
        directives = LoadModelDirectives(directives_file)
        ProcessModelDirectives(directives, parser)

        params.robot_model_instances = [
            plant.GetModelInstanceByName("iiwa_left"),
            plant.GetModelInstanceByName("iiwa_right")
        ]
        plant.Finalize()
        diagram = builder.Build()
        params.model = diagram
        params.edge_step_size = 0.01
        cls.checker = SceneGraphCollisionChecker(params)
        cls.plant = plant
        cls.diagram = diagram

        cls.q_tilde_bottom = np.array([-0.6430910102907225, 1.9156121024586796, -1.7968254667817805, 1.2945447141185198, -0.023834531305537934, -0.876966810663043, -1.7041643160834519, 1.45])
        cls.q_tilde_middle = np.array([-0.5997312520566763, 1.489780849654964, -1.4739679827359913, 1.2905366081785483, -0.04421061906813227, -0.8793712572715165, -1.1603461715511334, 1.45])

        # Pre-plan RRT once
        print("Pre-planning RRT path...")
        def RandomConfig():
            domain_lower = np.hstack((iiwa_limits_lower, [0.0]))
            domain_upper = np.hstack((iiwa_limits_upper, [2.0 * np.pi]))
            return np.random.uniform(low=domain_lower, high=domain_upper)

        config_rrt = BimanualConfig(shoulder_up=True, elbow_up=True, wrist_up=False, grasp_distance=0.6)
        ad_config_rrt = AutoDiffConfig(use_ift=False)
        param_rrt = MakeParameterization(config_rrt, ad_config_rrt)
        reach_rrt = IiwaBimanualReachableConstraint(config_rrt)

        def ValidityCheckerRRT(q_tilde):
            if np.any(reach_rrt.Eval(q_tilde) > np.ones(4) + 1e-4): return False
            if np.any(reach_rrt.Eval(q_tilde) < -np.ones(4) - 1e-4): return False
            q_full = param_rrt.get_parameterization_double()(q_tilde)
            q_sub = q_full[7:]
            if np.any(q_sub < iiwa_limits_lower - 1e-4): return False
            if np.any(q_sub > iiwa_limits_upper + 1e-4): return False
            if np.any(np.abs(q_sub[[1, 3, 5]]) < 1e-2): return False
            return cls.checker.CheckConfigCollisionFree(q_full)

        rrt_options = rrt.RRTOptions(step_size=0.1, check_size=0.01, max_vertices=5000, max_iters=100000)
        rrt_planner = rrt.BiRRT(RandomConfig, ValidityCheckerRRT)
        np.random.seed(0)
        path = rrt_planner.plan(cls.q_tilde_bottom, cls.q_tilde_middle, rrt_options)
        if path is None:
            raise RuntimeError("BiRRT failed to find a path.")
        cls.shortcut_path = shortcut.shortcut(path, ValidityCheckerRRT, num_tries=50, check_size=0.01)

    def run_trajopt_smoke_test(self, config_name, parameterization_factory, reachability_constraint_factory, ad_config, use_psi, old_style_reach):
        grasp_distance = 0.6
        shoulder_up = True
        elbow_up = True
        wrist_up = False
        config = BimanualConfig(shoulder_up=shoulder_up, elbow_up=elbow_up, wrist_up=wrist_up, grasp_distance=grasp_distance)
        parameterization = parameterization_factory(config)

        # 1. Trajopt
        control_points_matrix = np.array(self.shortcut_path).T
        basis = BsplineBasis(4, control_points_matrix.shape[1], initial_parameter_value=0.0, final_parameter_value=10.0)
        initial_traj = BsplineTrajectory(basis, control_points_matrix)
        trajopt = KinematicTrajectoryOptimization(initial_traj)

        trajopt.AddPathPositionConstraint(self.q_tilde_bottom, self.q_tilde_bottom, 0)
        trajopt.AddPathPositionConstraint(self.q_tilde_middle, self.q_tilde_middle, 1)

        min_dist_context = self.diagram.CreateDefaultContext()
        plant_context = self.plant.GetMyContextFromRoot(min_dist_context)
        min_dist_constraint = MinimumDistanceLowerBoundConstraint(self.plant, 0.001, plant_context, None, 0.049)
        reach_type = ReachabilityType.kDirect if old_style_reach else ReachabilityType.kProbing
        full_feasibility = FullFeasibilityConstraint(iiwa_limits_lower, iiwa_limits_upper, config, ad_config, min_dist_constraint, reach_type, use_psi)

        for s in np.linspace(0, 1, 50):
            trajopt.AddPathPositionConstraint(full_feasibility, s)

        trajopt.prog().AddCost(IiwaBimanualPathCost(8, trajopt.num_control_points(), config, ad_config, True), trajopt.control_points().flatten(order='F'))

        solver = SnoptSolver()
        options = SolverOptions()
        options.SetOption(solver.solver_id(), "Major optimality tolerance", 1e-1)
        options.SetOption(solver.solver_id(), "Time Limit", 60)
        options.SetOption(solver.solver_id(), "Iterations limit", 100000)
        options.SetOption(solver.solver_id(), "Minor iterations limit", 100000)
        
        result = solver.Solve(trajopt.prog(), None, options)
        if not result.is_success():
            print(f"DEBUG: Solver failed for {config_name}")
            print(f"  Solver ID: {result.get_solver_id().name()}")
            print(f"  Success: {result.is_success()}")
            try:
                details = result.get_solver_details()
                print(f"  SNOPT Status: {details.info}")
            except: pass
        self.assertTrue(result.is_success())
        
        # 2. TOPPRA
        trajopt_traj = trajopt.ReconstructTrajectory(result)
        def traj_fn(t): return parameterization.get_parameterization_double()(trajopt_traj.value(t).flatten())
        
        def full_traj_derivative(t, order, dt=1e-5):
             if order == 1:
                 x_val = trajopt_traj.value(t).flatten()
                 xdot_val = trajopt_traj.EvalDerivative(t, 1).flatten()
                 x_ad = InitializeAutoDiff(x_val).flatten()
                 y_ad = parameterization.get_parameterization_autodiff()(x_ad)
                 J = ExtractGradient(y_ad)
                 return J @ xdot_val.reshape(-1, 1)
             return np.zeros((14,1))

        full_traj = FunctionHandleTrajectory(traj_fn, 14, 1, trajopt_traj.start_time(), trajopt_traj.end_time())
        full_traj.set_derivative(full_traj_derivative)
        
        gridpoints = Toppra.CalcGridPoints(full_traj, CalcGridPointsOptions(max_iter=2, min_points=200))
        toppra = Toppra(full_traj, self.plant, gridpoints)
        toppra.AddJointVelocityLimit(self.plant.GetVelocityLowerLimits(), self.plant.GetVelocityUpperLimits())
        toppra.AddJointAccelerationLimit(self.plant.GetAccelerationLowerLimits(), self.plant.GetAccelerationUpperLimits())
        
        time_traj = toppra.SolvePathParameterization()
        self.assertIsNotNone(time_traj)

    def test_configs(self):
        configs = [
            {"name": "Current best (AD, New Reach)", "use_ift": False, "old_reach": False, "use_psi": False},
            {"name": "Baseline old (AD, Old Reach)", "use_ift": False, "old_reach": True, "use_psi": False},
            {"name": "Sanity check (IFT, New Reach, with Psi)", "use_ift": True, "old_reach": False, "use_psi": True, "ift_handling": IftSingularityHandling.kPseudoinverse},
            {"name": "Sanity check (IFT, New Reach, NO Psi)", "use_ift": True, "old_reach": False, "use_psi": False, "ift_handling": IftSingularityHandling.kPseudoinverse},
            {"name": "IFT Pseudoinverse (Old Reach, with Psi)", "use_ift": True, "old_reach": True, "use_psi": True, "ift_handling": IftSingularityHandling.kPseudoinverse},
            {"name": "IFT Pseudoinverse (Old Reach, NO Psi)", "use_ift": True, "old_reach": True, "use_psi": False, "ift_handling": IftSingularityHandling.kPseudoinverse},
            {"name": "IFT Zero gradients (Old Reach, with Psi)", "use_ift": True, "old_reach": True, "use_psi": True, "ift_handling": IftSingularityHandling.kZero},
            {"name": "IFT Zero gradients (Old Reach, NO Psi)", "use_ift": True, "old_reach": True, "use_psi": False, "ift_handling": IftSingularityHandling.kZero},
        ]

        for cfg in configs:
            with self.subTest(config=cfg["name"]):
                handling = cfg.get("ift_handling", IftSingularityHandling.kPseudoinverse)
                ad_cfg = AutoDiffConfig(use_ift=cfg["use_ift"], ift_handling=handling)
                self.run_trajopt_smoke_test(
                    cfg["name"],
                    lambda config: MakeParameterization(config, ad_cfg),
                    lambda config: (OldStyleReachableConstraint(config, ad_cfg, 1e-4) if cfg["old_reach"] else IiwaBimanualReachableConstraint(config)),
                    ad_cfg, False, cfg["old_reach"] # Force use_psi=False
                )

if __name__ == "__main__":
    unittest.main()
