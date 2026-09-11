"""
Single source of truth for mapping a ReachabilityType onto a constraint object
and onto the bounds used to accept/reject a configuration.

This used to be an inline two-way branch (`kDirect ? OldStyle : Probing`)
duplicated at four call sites across run_full_comparison.py and
run_boundary_sweep.py. Because none of those copies had a kBoundary case,
every "Boundary Reach" configuration silently fell through to the
probing-function constraint in IRIS, and to *no reachability check at all* in
the RRT validity checker. Keep the mapping here so a new ReachabilityType
cannot be added without every consumer seeing it.
"""

import numpy as np

from iiwa_ik import (
    BoundaryReachabilityConstraint,
    IiwaBimanualReachableConstraint,
    OldStyleReachableConstraint,
    ReachabilityType,
)

# Tolerance for the direct ("old style") reachability residual.
DIRECT_REACH_TOLERANCE = 1e-4


def make_reach_constraint(reach_type, config, ad_config):
    """
    Build the reachability constraint for `reach_type`.

    The boundary threshold and epsilon are read off `config` rather than a raw
    config dict, so BimanualConfig stays the single source of truth --
    FullFeasibilityConstraint already reads them from there, and a second source
    would reopen the same class of silent drift.

    Note that IiwaBimanualReachableConstraint (the probing formulation) takes no
    AutoDiffConfig and is always differentiated with bespoke autodiff, so probing
    rows are unaffected by the IFT settings of the configuration they belong to.
    """
    if reach_type == ReachabilityType.kDirect:
        return OldStyleReachableConstraint(config, ad_config,
                                           DIRECT_REACH_TOLERANCE)
    if reach_type == ReachabilityType.kBoundary:
        return BoundaryReachabilityConstraint(
            config, ad_config,
            threshold=config.boundary_threshold,
            epsilon=config.boundary_epsilon)
    if reach_type == ReachabilityType.kProbing:
        return IiwaBimanualReachableConstraint(config)
    raise ValueError(f"Unhandled ReachabilityType: {reach_type}")


def reach_constraint_satisfied(reach_con, reach_type, config, q_tilde):
    """
    Evaluate `reach_con` at `q_tilde` and report whether the configuration is
    reachable under the bounds appropriate to `reach_type`.
    """
    val = reach_con.Eval(q_tilde)
    if reach_type == ReachabilityType.kProbing:
        return bool(np.all(val <= np.ones(4)) and np.all(val >= -np.ones(4)))
    if reach_type == ReachabilityType.kDirect:
        return bool(val[0] <= DIRECT_REACH_TOLERANCE)
    if reach_type == ReachabilityType.kBoundary:
        return bool(val[0] <= config.boundary_threshold)
    raise ValueError(f"Unhandled ReachabilityType: {reach_type}")
