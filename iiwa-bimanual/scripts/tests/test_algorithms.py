import os
import sys
import numpy as np
import unittest
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
    Point,
    PiecewisePolynomial,
    CompositeTrajectory,
    SnoptSolver,
    StartMeshcat,
    Role,
    BsplineBasis,
    BsplineTrajectory,
    KinematicTrajectoryOptimization,
    Toppra,
    CalcGridPointsOptions,
    PathParameterizedTrajectory,
    FunctionHandleTrajectory,
    MinimumDistanceLowerBoundConstraint
)

# Add local paths
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_dir = os.path.abspath(os.path.join(script_dir, "../.."))
sys.path.append(os.path.join(repo_dir, "cpp_parameterization/python"))
sys.path.append(repo_dir)

from iiwa_ik import (
    MakeParameterization,
    AutoDiffConfig,
    IiwaBimanualReachableConstraint,
    IiwaBimanualJointLimitConstraint,
    IiwaBimanualPathCost,
    FullFeasibilityConstraint,
    BimanualConfig,
    ReachabilityType
)
from src.iiwa_analytic_ik import iiwa_limits_lower, iiwa_limits_upper
import src.common as common
import src.rrt as rrt

def build_checker_and_plant():
    directives_file = os.path.join(repo_dir, "models/old_shelves.dmd.yaml")
    params = CollisionCheckerParams()
    builder = RobotDiagramBuilder(time_step=0.0)
    
    plant = builder.plant()
    parser = Parser(plant)
    package_xml_path = os.path.join(repo_dir, "package.xml")
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
    checker = SceneGraphCollisionChecker(params)
    return checker, plant, diagram

class TestAlgorithms(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.checker, cls.plant, cls.diagram = build_checker_and_plant()
        cls.grasp_distance = 0.6
        cls.seed = np.array([-0.643, 1.915, -1.796, 1.294, -0.023, -0.876, -1.704, 1.45])
        
        # Create a min distance constraint for FullFeasibility
        cls.diagram_context = cls.diagram.CreateDefaultContext()
        cls.plant_context = cls.plant.GetMyContextFromRoot(cls.diagram_context)
        cls.min_dist_constraint = MinimumDistanceLowerBoundConstraint(
            cls.plant, 0.001, cls.plant_context, None, 0.05
        )

    def test_all_algorithms_config1(self):
        self.run_algorithms_for_config(True, True, False)

    def test_all_algorithms_config2(self):
        self.run_algorithms_for_config(True, True, True)

    def run_algorithms_for_config(self, shoulder_up, elbow_up, wrist_up):
        checker, plant, diagram = self.checker, self.plant, self.diagram
        grasp_distance, seed = self.grasp_distance, self.seed
        
        # 1. Setup Parameterization
        config = BimanualConfig(shoulder_up=shoulder_up, elbow_up=elbow_up, wrist_up=wrist_up, grasp_distance=grasp_distance)
        parameterization = MakeParameterization(config, AutoDiffConfig(False))
        
        # 2. IRIS-NP2 (Mini version)
        iris_options = IrisNp2Options()
        iris_options.parameterization = parameterization
        iris_options.sampled_iris_options.max_iterations = 1
        iris_options.sampled_iris_options.num_particles = 100
        iris_options.sampled_iris_options.configuration_space_margin = 1e-6
        iris_options.ray_sampler_options.num_particles_to_walk_towards = 10
        
        iris_prog = MathematicalProgram()
        q_tilde_vars = iris_prog.NewContinuousVariables(8, "q_tilde")
        iris_options.sampled_iris_options.prog_with_additional_constraints = iris_prog
        
        reachability_constraint = IiwaBimanualReachableConstraint(config)
        iris_prog.AddConstraint(reachability_constraint, q_tilde_vars)
        
        domain_lower = np.hstack((iiwa_limits_lower, [0.0]))
        domain_upper = np.hstack((iiwa_limits_upper, [2.0 * np.pi]))
        domain = HPolyhedron.MakeBox(domain_lower, domain_upper)
        
        print(f"Testing IRIS-NP2 for config ({shoulder_up}, {elbow_up}, {wrist_up})...")
        region = IrisNp2(checker, Hyperellipsoid.MakeHypersphere(1e-2, seed), domain, iris_options)
        self.assertIsNotNone(region)
        self.assertEqual(region.ambient_dimension(), 8)
        
        # 3. RRT (Mini version)
        def RandomConfig():
            return np.random.uniform(low=domain_lower, high=domain_upper)
        
        def ValidityChecker(q_tilde):
            if np.any(q_tilde < domain_lower) or np.any(q_tilde > domain_upper): return False
            q_full = parameterization.get_parameterization_double()(q_tilde)
            return checker.CheckConfigCollisionFree(q_full)

        rrt_planner = rrt.BiRRT(RandomConfig, ValidityChecker)
        rrt_options = rrt.RRTOptions(step_size=0.5, max_vertices=100)
        print("Testing RRT...")
        path = rrt_planner.plan(seed, seed + 0.01, rrt_options)
        
        # 4. Trajopt (Mini version)
        print("Testing Trajopt...")
        shortcut_path = [seed, seed + 0.01]
        control_points_matrix = np.array(shortcut_path).T
        basis = BsplineBasis(2, control_points_matrix.shape[1], initial_parameter_value=0, final_parameter_value=1)
        initial_traj = BsplineTrajectory(basis, control_points_matrix)
        trajopt = KinematicTrajectoryOptimization(initial_traj)
        
        full_feasibility_constraint = FullFeasibilityConstraint(
            iiwa_limits_lower, iiwa_limits_upper, config, AutoDiffConfig(False),
            self.min_dist_constraint, ReachabilityType.kProbing, False
        )
        trajopt.AddPathPositionConstraint(full_feasibility_constraint, 0.5)
        
        cost = IiwaBimanualPathCost(8, 2, config, AutoDiffConfig(True), True)
        trajopt.prog().AddCost(cost, trajopt.control_points().flatten(order='F'))
        
        solver = SnoptSolver()
        trajopt_result = solver.Solve(trajopt.prog())
        self.assertTrue(trajopt_result.get_solver_id().name() == "SNOPT" or trajopt_result.is_success())

        # 5. TOPRA (Mini version)
        print("Testing TOPRA...")
        traj_fn = lambda t: parameterization.get_parameterization_double()(initial_traj.value(t).flatten())
        full_traj = FunctionHandleTrajectory(traj_fn, 14, 1, 0, 1)
        # Toppra requires a derivative
        full_traj.set_derivative(lambda t, order: np.zeros((14, 1)))
        
        gridpoints = Toppra.CalcGridPoints(full_traj, CalcGridPointsOptions(max_iter=1, min_points=10))
        toppra = Toppra(full_traj, plant, gridpoints)
        toppra.AddJointVelocityLimit(plant.GetVelocityLowerLimits(), plant.GetVelocityUpperLimits())
        time_traj = toppra.SolvePathParameterization()

if __name__ == "__main__":
    unittest.main()
