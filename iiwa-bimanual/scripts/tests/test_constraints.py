import os
import sys
import numpy as np
import unittest
from pydrake.all import (
    InitializeAutoDiff,
    ExtractValue,
    ExtractGradient,
    CollisionCheckerParams,
    RobotDiagramBuilder,
    Parser,
    LoadModelDirectives,
    ProcessModelDirectives,
    SceneGraphCollisionChecker,
    MinimumDistanceLowerBoundConstraint
)

# Add local paths
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_dir = os.path.abspath(os.path.join(script_dir, "../.."))
sys.path.append(os.path.join(repo_dir, "cpp_parameterization/python"))
sys.path.append(repo_dir)

from iiwa_ik import (
    IiwaBimanualCollisionFreeConstraint,
    FullFeasibilityConstraint,
    BimanualConfig,
    AutoDiffConfig,
    ReachabilityType
)
from src.iiwa_analytic_ik import iiwa_limits_lower, iiwa_limits_upper

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

class TestConstraints(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.checker, cls.plant, cls.diagram = build_checker_and_plant()
        cls.grasp_distance = 0.6
        cls.x = np.array([-0.643, 1.915, -1.796, 1.294, -0.023, -0.876, -1.704, 1.45])
        
        # Create a min distance constraint for constraints
        cls.diagram_context = cls.diagram.CreateDefaultContext()
        cls.plant_context = cls.plant.GetMyContextFromRoot(cls.diagram_context)
        cls.min_dist_constraint = MinimumDistanceLowerBoundConstraint(
            cls.plant, 0.001, cls.plant_context, None, 0.05
        )

    def test_collision_free_constraint(self):
        grasp_distance, x = self.grasp_distance, self.x
        shoulder_up, elbow_up, wrist_up = True, True, False
        c = IiwaBimanualCollisionFreeConstraint(
            BimanualConfig(shoulder_up, elbow_up, wrist_up, grasp_distance),
            AutoDiffConfig(use_ift=False), # Example setting
            self.min_dist_constraint)
        
        # 1. Eval double
        y = c.Eval(x)
        self.assertEqual(len(y), 1)
        self.assertTrue(np.isfinite(y[0]))
        
        # 2. Eval AutoDiff and check gradient
        x_ad = InitializeAutoDiff(x)
        y_ad = c.Eval(x_ad)
        grad_ad = ExtractGradient(y_ad)
        
        self.assertEqual(grad_ad.shape, (1, 8))
        self.assertTrue(np.all(np.isfinite(grad_ad)))

    def test_full_feasibility_constraint(self):
        grasp_distance, x = self.grasp_distance, self.x
        shoulder_up, elbow_up, wrist_up = True, True, False
        
        c = FullFeasibilityConstraint(
            iiwa_limits_lower, iiwa_limits_upper, BimanualConfig(shoulder_up, elbow_up, wrist_up, grasp_distance),
            AutoDiffConfig(use_ift=False),
            self.min_dist_constraint, ReachabilityType.kProbing, False
        )
        
        # 1. Eval double
        y = c.Eval(x)
        self.assertGreater(len(y), 0)
        self.assertTrue(np.all(np.isfinite(y)))
        
        # 2. Eval AutoDiff
        x_ad = InitializeAutoDiff(x)
        y_ad = c.Eval(x_ad)
        grad_ad = ExtractGradient(y_ad)
        
        self.assertEqual(grad_ad.shape, (len(y), 8))
        self.assertTrue(np.all(np.isfinite(grad_ad)))

        # 3. Finite difference check for a few dimensions
        def f(x_in):
            return c.Eval(x_in)
        
        eps = 1e-6
        J_fd = np.zeros((len(y), 8))
        for i in range(8):
            x_p = x.copy(); x_p[i] += eps
            x_m = x.copy(); x_m[i] -= eps
            J_fd[:, i] = (f(x_p) - f(x_m)) / (2.0 * eps)
        
        # Check if gradient matches FD
        self.assertTrue(np.allclose(grad_ad, J_fd, rtol=1e-3, atol=1e-3))

if __name__ == "__main__":
    unittest.main()
