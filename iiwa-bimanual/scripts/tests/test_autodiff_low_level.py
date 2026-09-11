import sys
import os
import time
import numpy as np
import unittest
from pydrake.all import InitializeAutoDiff, AutoDiffXd, ExtractValue, ExtractGradient

# Add local path to finding the bindings
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_dir = os.path.abspath(os.path.join(script_dir, "../.."))
sys.path.append(os.path.join(repo_dir, "cpp_parameterization/python"))

try:
    from iiwa_ik import (
        MakeParameterization,
        IftSingularityHandling,
        BimanualConfig,
        AutoDiffConfig
    )
except ImportError:
    sys.path.append(os.path.join(os.getcwd(), "cpp_parameterization/python"))
    from iiwa_ik import (
        MakeParameterization,
        IftSingularityHandling,
        BimanualConfig,
        AutoDiffConfig
    )

# LBR IIWA 14 R820 limits (approximate for the 7 joints)
IIWA_LOWER_LIMITS = np.array([-2.96, -2.09, -2.96, -2.09, -2.96, -2.09, -3.05])
IIWA_UPPER_LIMITS = np.array([ 2.96,  2.09,  2.96,  2.09,  2.96,  2.09,  3.05])

class TestAutoDiffLowLevel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(42)

    def test_all_configs(self):
        num_vars = 8
        grasp_distance = 0.6
        
        # Test all 8 configurations
        for shoulder_up in [True, False]:
            for elbow_up in [True, False]:
                for wrist_up in [True, False]:
                    with self.subTest(s=shoulder_up, e=elbow_up, w=wrist_up):
                        # Rejection sampling to ensure reachability
                        config = BimanualConfig(shoulder_up, elbow_up, wrist_up, grasp_distance)
                        param_ad = MakeParameterization(
                            config=config,
                            ad_config=AutoDiffConfig(use_ift=False)
                        )

                        attempts = 0
                        max_attempts = 1000
                        found_reachable = False
                        while attempts < max_attempts:
                            q_c = np.random.uniform(IIWA_LOWER_LIMITS, IIWA_UPPER_LIMITS)
                            psi = np.random.uniform(0.0, 2 * np.pi, size=(1,))
                            q_and_psi = np.concatenate([q_c, psi])
                            
                            V = np.random.randn(num_vars, num_vars)
                            q_and_psi_ad = InitializeAutoDiff(q_and_psi, V)
                            
                            out_ad = param_ad.get_parameterization_autodiff()(q_and_psi_ad)
                            grad_ad = ExtractGradient(out_ad)
                            
                            # Check reachability (unclipped outputs)
                            clipped = False
                            for row in range(7, 14):
                                if np.max(np.abs(grad_ad[row])) < 1e-6:
                                    clipped = True
                                    break
                            
                            if not clipped:
                                found_reachable = True
                                break
                            attempts += 1
                        
                        self.assertTrue(found_reachable, f"Could not find reachable configuration for s={shoulder_up}, e={elbow_up}, w={wrist_up}")
                        
                        val_ad = ExtractValue(out_ad)
                        
                        # 2. Analytic IFT Gradients (Pseudoinverse)
                        param_ift = MakeParameterization(
                            config=config,
                            ad_config=AutoDiffConfig(
                                use_ift=True,
                                ift_handling=IftSingularityHandling.kPseudoinverse
                            )
                        )
                        
                        out_ift = param_ift.get_parameterization_autodiff()(q_and_psi_ad)
                        val_ift = ExtractValue(out_ift)
                        grad_ift = ExtractGradient(out_ift)
                    
                        # 3. Analytic IFT Gradients (kZero)
                        param_ift_zero = MakeParameterization(
                            config=config,
                            ad_config=AutoDiffConfig(
                                use_ift=True,
                                ift_handling=IftSingularityHandling.kZero
                            )
                        )
                        out_ift_zero = param_ift_zero.get_parameterization_autodiff()(q_and_psi_ad)
                        grad_ift_zero = ExtractGradient(out_ift_zero)
                    
                        # Assertions
                        np.testing.assert_allclose(val_ad, val_ift, err_msg="Values do not match!")
                        
                        grad_diff = np.linalg.norm(grad_ad - grad_ift)
                        grad_diff_zero = np.linalg.norm(grad_ad - grad_ift_zero)
                        
                        self.assertLess(grad_diff, 1e-10, f"IFT gradients (Pseudoinverse) mismatch: {grad_diff}")
                        self.assertLess(grad_diff_zero, 1e-10, f"IFT gradients (kZero) mismatch: {grad_diff_zero}")

if __name__ == "__main__":
    unittest.main()
