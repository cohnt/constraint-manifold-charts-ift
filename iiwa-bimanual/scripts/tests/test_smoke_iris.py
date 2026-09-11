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
        AutoDiffConfig,
        MakeParameterization,
        IiwaBimanualReachableConstraint,
        OldStyleReachableConstraint,
        IiwaBimanualJointLimitConstraint,
        IiwaBimanualPsiSingularityConstraint,
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
        IiwaBimanualPsiSingularityConstraint,
        IftSingularityHandling
    )

from src.iiwa_analytic_ik import iiwa_limits_lower, iiwa_limits_upper
import src.common as common
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
    SnoptSolver
)

class TestIrisSmoke(unittest.TestCase):
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

    def run_iris_smoke_test(self, config_name, parameterization_factory, reachability_constraint_factory, use_ift, use_psi):
        grasp_distance = 0.6
        shoulder_up = True
        elbow_up = True
        wrist_up = False
        config = BimanualConfig(shoulder_up=shoulder_up, elbow_up=elbow_up, wrist_up=wrist_up, grasp_distance=grasp_distance)

        # Interior seed for smoke test
        seed = np.array([-0.5997312520566763, 1.489780849654964, -1.4739679827359913, 1.2905366081785483, -0.04421061906813227, -0.8793712572715165, -1.1603461715511334, 1.45])

        iris_options = IrisNp2Options()
        iris_options.parameterization = parameterization_factory(config)
        iris_options.sampled_iris_options.random_seed = 2
        iris_options.sampled_iris_options.max_iterations = 1
        iris_options.sampled_iris_options.epsilon = 0.05
        iris_options.sampled_iris_options.configuration_space_margin = 0
        iris_options.add_hyperplane_if_solve_fails = True

        iris_options.solver_options.SetOption(SnoptSolver().solver_id(), "Major iterations limit", 100)
        
        iris_prog = MathematicalProgram()
        q_tilde_vars = iris_prog.NewContinuousVariables(8, "q_tilde")
        iris_options.sampled_iris_options.prog_with_additional_constraints = iris_prog
        
        reachability_constraint = reachability_constraint_factory(config)
        iris_prog.AddConstraint(reachability_constraint, q_tilde_vars)
        
        subordinate_arm_joint_limit_constraint = IiwaBimanualJointLimitConstraint(
            iiwa_limits_lower, iiwa_limits_upper, config, AutoDiffConfig(use_ift)
        )
        iris_prog.AddConstraint(subordinate_arm_joint_limit_constraint, q_tilde_vars)

        if use_psi and use_ift:
            psi_sing_iris = IiwaBimanualPsiSingularityConstraint(config, AutoDiffConfig(use_ift))
            iris_prog.AddConstraint(psi_sing_iris, q_tilde_vars)
        
        domain_lower = np.hstack((iiwa_limits_lower, [0.0]))
        domain_upper = np.hstack((iiwa_limits_upper, [2.0 * np.pi]))
        domain = HPolyhedron.MakeBox(domain_lower, domain_upper)
        
        # This will raise an exception if it fails, which unittest will catch
        region = IrisNp2(self.checker, Hyperellipsoid.MakeHypersphere(1e-2, seed), domain, iris_options)
        self.assertIsNotNone(region)

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
                self.run_iris_smoke_test(
                    cfg["name"],
                    lambda config: MakeParameterization(config, AutoDiffConfig(cfg["use_ift"], handling)),
                    lambda config: (OldStyleReachableConstraint(config, AutoDiffConfig(cfg["use_ift"]), 1e-4) if cfg["old_reach"] else IiwaBimanualReachableConstraint(config)),
                    cfg["use_ift"], cfg["use_psi"]
                )

if __name__ == "__main__":
    unittest.main()
