import os
import sys
import numpy as np
import unittest
import itertools
from pydrake.all import InitializeAutoDiff, ExtractGradient

# Add local paths
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_dir = os.path.abspath(os.path.join(script_dir, "../.."))
sys.path.append(os.path.join(repo_dir, "cpp_parameterization/python"))
sys.path.append(repo_dir)

from iiwa_ik import (
    BimanualConfig,
    AutoDiffConfig,
    IftSingularityHandling,
    MakeParameterization,
    IiwaBimanualPathCost,
    IiwaBimanualJointLimitConstraint,
    OldStyleReachableConstraint,
    IiwaBimanualPsiSingularityConstraint
)

class TestAllConfigs(unittest.TestCase):
    def test_all_combinations(self):
        shoulder_ups = [True, False]
        elbow_ups = [True, False]
        wrist_ups = [True, False]
        use_ifts = [True, False]
        ift_handlings = [IftSingularityHandling.kPseudoinverse, IftSingularityHandling.kZero]
        
        grasp_distance = 0.6
        x = np.array([-0.353, -0.957, -1.232, 1.976, 0.517, 1.387, -2.770, 0.5])
        
        combinations = list(itertools.product(shoulder_ups, elbow_ups, wrist_ups, use_ifts, ift_handlings))
        print(f"\nTesting {len(combinations)} configuration combinations...")
        
        for s, e, w, ift, handling in combinations:
            with self.subTest(s=s, e=e, w=w, ift=ift, handling=handling):
                config = BimanualConfig(s, e, w, grasp_distance)
                ad_config = AutoDiffConfig(ift, handling)
                
                # 1. Parameterization
                param = MakeParameterization(config, ad_config)
                q_full = param.get_parameterization_double()(x)
                self.assertEqual(len(q_full), 14)
                
                # 2. Costs
                cost = IiwaBimanualPathCost(8, 2, config, ad_config, True)
                y_cost = cost.Eval(np.tile(x, 2))
                self.assertEqual(len(y_cost), 1)
                
                # 3. Constraints (Gradients)
                constraints = [
                    IiwaBimanualJointLimitConstraint(np.ones(7)*-3, np.ones(7)*3, config, ad_config),
                    OldStyleReachableConstraint(config, ad_config),
                    IiwaBimanualPsiSingularityConstraint(config, ad_config)
                ]
                
                for c in constraints:
                    # Double eval
                    y = c.Eval(x)
                    self.assertGreater(len(y), 0)
                    
                    # AutoDiff eval (Gradient check)
                    x_ad = InitializeAutoDiff(x)
                    y_ad = c.Eval(x_ad)
                    grad_ad = ExtractGradient(y_ad)
                    
                    # Finite difference
                    def f(x_in): return c.Eval(x_in)
                    eps = 1e-6
                    J_fd = np.zeros((len(y), 8))
                    for i in range(8):
                        x_p = x.copy(); x_p[i] += eps
                        x_m = x.copy(); x_m[i] -= eps
                        J_fd[:, i] = (f(x_p) - f(x_m)) / (2.0 * eps)
                    
                    self.assertTrue(np.allclose(grad_ad, J_fd, rtol=1e-3, atol=1e-3),
                                    f"Gradient mismatch for {type(c).__name__} in config s={s}, e={e}, w={w}, ift={ift}, handling={handling}")

if __name__ == "__main__":
    unittest.main()
