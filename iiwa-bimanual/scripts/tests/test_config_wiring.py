"""
Regression tests for the ReachabilityType -> constraint wiring.

Every "Boundary Reach" row of the benchmark once fell through to the
probing-function constraint in IRIS, and to no reachability check at all in the
RRT validity checker, because the selection logic was an inline two-way branch
(kDirect vs everything-else) with no kBoundary case. These tests pin the mapping
so that cannot recur silently.
"""

import os
import sys
import unittest

import numpy as np

repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(os.path.join(repo_dir, "cpp_parameterization/python"))
sys.path.append(repo_dir)

import pydrake  # noqa: F401  (registers the Constraint base class)

from iiwa_ik import (
    AutoDiffConfig,
    BimanualConfig,
    BoundaryReachabilityConstraint,
    IftSingularityHandling,
    IiwaBimanualReachableConstraint,
    OldStyleReachableConstraint,
    ReachabilityType,
)

from src.reach_constraints import make_reach_constraint, reach_constraint_satisfied
from src.iiwa_analytic_ik import iiwa_limits_lower, iiwa_limits_upper

# Matches the benchmark's scenario constants.
GRASP_DISTANCE = 0.6
BOUNDARY_THRESHOLD = 60.0
BOUNDARY_EPSILON = 1e-3

# A configuration that is reachable (direct residual ~1e-31).
Q_TILDE_REACHABLE = np.array([
    -0.6430910102907225, 1.9156121024586796, -1.7968254667817805,
    1.2945447141185198, -0.023834531305537934, -0.876966810663043,
    -1.7041643160834519, 1.45])

DOMAIN_LOWER = np.hstack((iiwa_limits_lower, [0.0]))
DOMAIN_UPPER = np.hstack((iiwa_limits_upper, [2.0 * np.pi]))

ALL_REACH_TYPES = (ReachabilityType.kDirect,
                   ReachabilityType.kProbing,
                   ReachabilityType.kBoundary)


def make_config(boundary_threshold=BOUNDARY_THRESHOLD):
    return BimanualConfig(shoulder_up=True, elbow_up=True, wrist_up=False,
                          grasp_distance=GRASP_DISTANCE,
                          clipping_margin=1e-4, clipping_margin_psi=1e-4,
                          boundary_threshold=boundary_threshold,
                          boundary_epsilon=BOUNDARY_EPSILON)


def make_ad_config():
    return AutoDiffConfig(use_ift=False,
                          ift_handling=IftSingularityHandling.kPseudoinverse,
                          lambda_=0.0, svt_epsilon=None, svt_lambda_max=None,
                          use_anisotropic_damping=False)


class TestReachConstraintWiring(unittest.TestCase):

    def setUp(self):
        self.config = make_config()
        self.ad_config = make_ad_config()

    def test_each_reach_type_maps_to_its_own_constraint(self):
        """The bug: kBoundary silently produced the probing constraint."""
        expected = {
            ReachabilityType.kDirect: OldStyleReachableConstraint,
            ReachabilityType.kProbing: IiwaBimanualReachableConstraint,
            ReachabilityType.kBoundary: BoundaryReachabilityConstraint,
        }
        for reach_type, cls in expected.items():
            with self.subTest(reach_type=reach_type):
                con = make_reach_constraint(reach_type, self.config,
                                            self.ad_config)
                self.assertIsInstance(con, cls)

    def test_unhandled_reach_type_raises(self):
        """A new ReachabilityType must not fall through to a default."""
        with self.assertRaises(ValueError):
            make_reach_constraint("not a reach type", self.config,
                                  self.ad_config)
        con = make_reach_constraint(ReachabilityType.kDirect, self.config,
                                    self.ad_config)
        with self.assertRaises(ValueError):
            reach_constraint_satisfied(con, "not a reach type", self.config,
                                       Q_TILDE_REACHABLE)

    def test_reachable_config_accepted_by_every_formulation(self):
        for reach_type in ALL_REACH_TYPES:
            with self.subTest(reach_type=reach_type):
                con = make_reach_constraint(reach_type, self.config,
                                            self.ad_config)
                self.assertTrue(reach_constraint_satisfied(
                    con, reach_type, self.config, Q_TILDE_REACHABLE))

    def test_boundary_threshold_is_actually_consulted(self):
        """
        The RRT validity checker previously ignored kBoundary entirely, so every
        sample passed. Driving the threshold to its extremes must flip the
        verdict; if it does not, the constraint is not being evaluated.
        """
        permissive = make_config(boundary_threshold=1e6)
        con = make_reach_constraint(ReachabilityType.kBoundary, permissive,
                                    self.ad_config)
        self.assertTrue(reach_constraint_satisfied(
            con, ReachabilityType.kBoundary, permissive, Q_TILDE_REACHABLE))

        strict = make_config(boundary_threshold=-1e6)
        con = make_reach_constraint(ReachabilityType.kBoundary, strict,
                                    self.ad_config)
        self.assertFalse(reach_constraint_satisfied(
            con, ReachabilityType.kBoundary, strict, Q_TILDE_REACHABLE))

    def test_unreachable_samples_are_rejected(self):
        """
        Direct and probing must reject unreachable configurations outright.
        The boundary formulation is a conservative proximity-to-singularity
        proxy, so it is checked separately in
        test_boundary_value_separates_reachable_from_unreachable.
        """
        rng = np.random.RandomState(0)
        direct = make_reach_constraint(ReachabilityType.kDirect, self.config,
                                       self.ad_config)
        probing = make_reach_constraint(ReachabilityType.kProbing, self.config,
                                        self.ad_config)
        checked = 0
        for _ in range(2000):
            q = rng.uniform(DOMAIN_LOWER, DOMAIN_UPPER)
            if direct.Eval(q)[0] <= 1e-4:
                continue
            checked += 1
            self.assertFalse(reach_constraint_satisfied(
                direct, ReachabilityType.kDirect, self.config, q))
            self.assertFalse(reach_constraint_satisfied(
                probing, ReachabilityType.kProbing, self.config, q))
            if checked >= 50:
                break
        self.assertGreaterEqual(checked, 50)

    def test_boundary_value_separates_reachable_from_unreachable(self):
        """
        -log det(J J^T + eps I) must be larger on unreachable configurations,
        which is what makes it usable as a reachability proxy at all. Stated as a
        distributional claim because the two populations overlap: reachable
        configurations near the workspace boundary legitimately score high.
        """
        rng = np.random.RandomState(1)
        direct = make_reach_constraint(ReachabilityType.kDirect, self.config,
                                       self.ad_config)
        boundary = make_reach_constraint(ReachabilityType.kBoundary,
                                         self.config, self.ad_config)
        reachable, unreachable = [], []
        for _ in range(4000):
            q = rng.uniform(DOMAIN_LOWER, DOMAIN_UPPER)
            target = (reachable if direct.Eval(q)[0] <= 1e-4 else unreachable)
            if len(target) < 100:
                target.append(boundary.Eval(q)[0])
            if len(reachable) >= 100 and len(unreachable) >= 100:
                break
        self.assertGreaterEqual(len(reachable), 100)
        self.assertGreaterEqual(len(unreachable), 100)
        self.assertGreater(np.median(unreachable), np.median(reachable))


if __name__ == "__main__":
    unittest.main()
