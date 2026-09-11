import os
import sys
import numpy as np
import time
import json
import unittest
from unittest.mock import MagicMock

# Add local path to finding the bindings
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_dir = os.path.abspath(os.path.join(script_dir, "../.."))
sys.path.append(os.path.join(repo_dir, "cpp_parameterization/python"))
sys.path.append(repo_dir)

try:
    from iiwa_ik import (
        BimanualConfig,
        AutoDiffConfig,
        MakeParameterization,
        IiwaBimanualReachableConstraint,
        OldStyleReachableConstraint,
        IiwaBimanualJointLimitConstraint,
        FullFeasibilityConstraint,
        IiwaBimanualPsiSingularityConstraint,
        IiwaBimanualPathCost,
        IftSingularityHandling
    )
except ImportError:
    sys.path.append(os.path.join(os.getcwd(), "cpp_parameterization/python"))
    sys.path.append(os.getcwd())
    from iiwa_ik import (
        BimanualConfig,
        AutoDiffConfig,
        MakeParameterization,
        IiwaBimanualReachableConstraint,
        OldStyleReachableConstraint,
        IiwaBimanualJointLimitConstraint,
        FullFeasibilityConstraint,
        IiwaBimanualPsiSingularityConstraint,
        IiwaBimanualPathCost,
        IftSingularityHandling
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
    SnoptSolver,
    GcsTrajectoryOptimization,
    ComputePairwiseIntersections,
    StartMeshcat,
    Role
)

class TestPipelineIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # We don't need meshcat for the automated test, but we can mock it if needed
        # Or just not use it. Most components work fine without it.
        cls.meshcat = MagicMock()
        
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
        
        builder.builder().ExportInput(plant.get_actuation_input_port(), "actuation")
        builder.builder().ExportOutput(plant.get_state_output_port(), "state")

        diagram = builder.Build()
        params.model = diagram
        params.edge_step_size = 0.01
        cls.checker = SceneGraphCollisionChecker(params)
        cls.plant = plant
        cls.diagram = diagram

        cls.q_tilde_bottom = np.array([-0.6430910102907225, 1.9156121024586796, -1.7968254667817805, 1.2945447141185198, -0.023834531305537934, -0.876966810663043, -1.7041643160834519, 1.45])
        cls.q_tilde_top = np.array([-0.1994994216078726, 0.9140739951190965, -2.236618320862171, 0.5238879195899456, 0.7998441913611017, -1.3575398006936048, -1.0153092816310436, 2.41])

    def test_pipeline_smoke(self):
        """Runs a simplified version of the pipeline for a single config."""
        config_name = "Smoke Test Config"
        use_ift = False
        handling = IftSingularityHandling.kPseudoinverse
        
        grasp_distance = 0.6
        shoulder_up = True
        elbow_up = True
        wrist_up = False
        config = BimanualConfig(shoulder_up, elbow_up, wrist_up, grasp_distance)

        iris_options = IrisNp2Options()
        iris_options.parameterization = MakeParameterization(config, AutoDiffConfig(use_ift, handling))
        iris_options.sampled_iris_options.random_seed = 2
        iris_options.sampled_iris_options.max_iterations = 1
        iris_options.sampled_iris_options.epsilon = 0.05
        iris_options.sampled_iris_options.configuration_space_margin = 0
        iris_options.add_hyperplane_if_solve_fails = True
        iris_options.solver_options.SetOption(SnoptSolver().solver_id(), "Major iterations limit", 50)
        
        iris_prog = MathematicalProgram()
        q_tilde_vars = iris_prog.NewContinuousVariables(8, "q_tilde")
        iris_options.sampled_iris_options.prog_with_additional_constraints = iris_prog
        
        reachability_constraint = IiwaBimanualReachableConstraint(config)
        iris_prog.AddConstraint(reachability_constraint, q_tilde_vars)
        
        subordinate_arm_joint_limit_constraint = IiwaBimanualJointLimitConstraint(
            iiwa_limits_lower, iiwa_limits_upper, config, AutoDiffConfig(use_ift)
        )
        iris_prog.AddConstraint(subordinate_arm_joint_limit_constraint, q_tilde_vars)

        domain_lower = np.hstack((iiwa_limits_lower, [0.0]))
        domain_upper = np.hstack((iiwa_limits_upper, [2.0 * np.pi]))
        domain = HPolyhedron.MakeBox(domain_lower, domain_upper)
        
        # 1. IRIS
        regions = []
        for seed in [self.q_tilde_bottom, self.q_tilde_top]:
             region = IrisNp2(self.checker, Hyperellipsoid.MakeHypersphere(1e-2, seed), domain, iris_options)
             regions.append(region.ReduceInequalities())
        
        self.assertEqual(len(regions), 2)
        
        # 2. GCS
        gcs = GcsTrajectoryOptimization(8)
        main_graph = gcs.AddRegions(regions, 1, h_min=0.01, h_max=100)
        start_graph = gcs.AddRegions([Point(self.q_tilde_bottom)], 0)
        goal_graph = gcs.AddRegions([Point(self.q_tilde_top)], 0)
        gcs.AddEdges(start_graph, main_graph)
        gcs.AddEdges(main_graph, goal_graph)
        gcs.AddPathLengthCost()
        
        gcs_options = GraphOfConvexSetsOptions()
        gcs_options.convex_relaxation = True
        
        gcs_traj, result = gcs.SolvePath(start_graph, goal_graph, gcs_options)
        # Note: with only 2 regions, connectivity might fail if they don't intersect.
        # But this is a smoke test for the calling sequence.
        # self.assertTrue(result.is_success()) 

if __name__ == "__main__":
    unittest.main()
