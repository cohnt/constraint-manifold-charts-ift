import sys
import os
import numpy as np
import unittest
from pydrake.all import InitializeAutoDiff, ExtractValue, ExtractGradient

# Add local path to finding the bindings
script_dir = os.path.dirname(os.path.abspath(__file__))
# Note: when running from the repo root, this path needs to reach cpp_parameterization/python
repo_dir = os.path.abspath(os.path.join(script_dir, "../.."))
sys.path.append(os.path.join(repo_dir, "cpp_parameterization/python"))

try:
    from iiwa_ik import MakeParameterization, OldStyleReachableConstraint, BimanualConfig, AutoDiffConfig, IftSingularityHandling
except ImportError:
    # Fallback for different execution environments
    sys.path.append(os.path.join(os.getcwd(), "cpp_parameterization/python"))
    from iiwa_ik import MakeParameterization, OldStyleReachableConstraint, BimanualConfig, AutoDiffConfig, IftSingularityHandling

class TestBindings(unittest.TestCase):
    def test_config_struct_members(self):
        # Every field of both config structs must be readable and writable from
        # Python; a field that silently fails to bind is how benchmark
        # configurations have drifted from their labels in the past.
        config = BimanualConfig(shoulder_up=True, elbow_up=True, wrist_up=True, grasp_distance=0.6)
        self.assertTrue(config.shoulder_up)
        self.assertEqual(config.grasp_distance, 0.6)
        
        config.clipping_margin = 1e-5
        self.assertEqual(config.clipping_margin, 1e-5)
        
        config.clipping_margin_psi = 1e-7
        self.assertEqual(config.clipping_margin_psi, 1e-7)

        # AutoDiffConfig members
        ad = AutoDiffConfig()
        self.assertTrue(ad.use_ift)
        self.assertEqual(ad.ift_handling, IftSingularityHandling.kPseudoinverse)
        
        ad.use_ift = False
        self.assertFalse(ad.use_ift)
        ad.ift_handling = IftSingularityHandling.kZero
        self.assertEqual(ad.ift_handling, IftSingularityHandling.kZero)

        # Exposed as `lambda_`: `lambda` is a Python reserved word and could
        # not be passed by keyword.
        ad.lambda_ = 1e-5
        self.assertEqual(ad.lambda_, 1e-5)
        self.assertEqual(AutoDiffConfig(lambda_=2e-5).lambda_, 2e-5)

    def test_old_style_reachable_binding(self):
        config = BimanualConfig(shoulder_up=True, elbow_up=True, wrist_up=True, grasp_distance=0.6)
        # Test with use_ift=True
        c_ift = OldStyleReachableConstraint(config, AutoDiffConfig(use_ift=True))
        # Test with use_ift=False
        c_ad = OldStyleReachableConstraint(config, AutoDiffConfig(use_ift=False))
        
        x = np.array([-0.353, -0.957, -1.232, 1.976, 0.517, 1.387, -2.770, 0.012])
        
        y_ift = c_ift.Eval(x)
        y_ad = c_ad.Eval(x)
        
        np.testing.assert_allclose(y_ift, y_ad, atol=1e-6)

if __name__ == "__main__":
    unittest.main()
