#!/usr/bin/env python3
"""
calibrate_boundary_threshold.py

Derive the boundary reachability threshold from robot kinematics, with no reference to
solver outcomes.

The constraint is

    y(q) = -log det( J(q) J(q)ᵀ + eps * I )  <=  threshold,

with J length-scaled so its entries are in metres (see BoundaryReachConstraint).  We
choose the threshold from a velocity-amplification criterion: realising a task-space
speed v requires joint rates up to v / sigma_min, and the URDF caps joint velocity at
qdot_max, so the smallest acceptable singular value is

    sigma* = v_task / qdot_max.

A worst-case sufficient bound on y is far too conservative to be usable (it bounds the
other five singular values by sigma_max, giving negative thresholds against a median y
of ~7.7).  Instead we calibrate empirically against the sigma_min level set: sample the
configuration space, keep configurations whose sigma_min is near sigma*, and report the
median y there.  This is still purely kinematic -- it never runs the optimiser.

Usage:
    python scripts/calibrate_boundary_threshold.py [--samples N] [--v-task V]
"""

import argparse
import os
import sys

import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
repo_dir = os.path.abspath(os.path.join(script_dir, ".."))
sys.path.insert(0, repo_dir)

from pydrake.all import JacobianWrtVariable

from src.boundary_reach_constraint import BoundaryReachConstraint
from src.util import BuildEnv, RepoDir

# UR5e joint velocity limit, from the URDF <limit velocity=...> fields.
QDOT_MAX = np.pi


def sample_singular_values(n_samples, length_scale, seed=0):
    """Return an (n_samples, 6) array of singular values of the length-scaled Jacobian."""
    diagram = BuildEnv(
        None,
        directives_file=os.path.join(RepoDir(), "models/ur5e_collision_simple.yaml"),
        visualize=False,
    )
    plant = diagram.GetSubsystemByName("plant")
    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyContextFromRoot(context)
    arm = plant.GetModelInstanceByName("ur5e")
    frame_E = plant.GetFrameByName("tool0", arm)
    frame_W = plant.world_frame()
    row_scale = np.diag([length_scale] * 3 + [1.0] * 3)

    rng = np.random.default_rng(seed)
    out = np.empty((n_samples, 6))
    for i in range(n_samples):
        plant.SetPositions(plant_context, arm, rng.uniform(-np.pi, np.pi, 6))
        J = plant.CalcJacobianSpatialVelocity(
            plant_context, JacobianWrtVariable.kV, frame_E,
            np.zeros(3), frame_W, frame_W,
        )
        out[i] = np.linalg.svd(row_scale @ J, compute_uv=False)
    return out


def calibrate(sigma, epsilon, sigma_star, band=0.15):
    """Median y over configurations whose sigma_min lies within `band` of sigma_star."""
    y = -np.sum(np.log(sigma ** 2 + epsilon), axis=1)
    s_min = sigma[:, -1]
    in_band = (s_min > sigma_star * (1 - band)) & (s_min < sigma_star * (1 + band))
    if in_band.sum() < 20:
        return None, int(in_band.sum())
    return float(np.median(y[in_band])), int(in_band.sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples", type=int, default=20000)
    ap.add_argument("--epsilon", type=float,
                    default=BoundaryReachConstraint.__init__.__defaults__[0])
    ap.add_argument("--length-scale", type=float,
                    default=BoundaryReachConstraint.DEFAULT_LENGTH_SCALE)
    ap.add_argument("--v-task", type=float, default=0.1,
                    help="Task-space speed the arm should be able to realise (m/s).")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print(f"Sampling {args.samples} configurations (length scale L = {args.length_scale} m) …")
    sigma = sample_singular_values(args.samples, args.length_scale, args.seed)
    print(f"  sigma_max: median {np.median(sigma[:, 0]):.4f}  max {sigma[:, 0].max():.4f}")
    print(f"  sigma_min: median {np.median(sigma[:, -1]):.5f}")
    print(f"  => epsilon must be << {np.median(sigma[:, -1])**2:.2e} (median sigma_min^2) "
          f"and >> {sigma[:, 0].max()**2 * np.finfo(float).eps:.2e} (float floor)")

    print(f"\nCalibration at epsilon = {args.epsilon:.0e}:")
    print(f"{'sigma*':>10} {'v_task (m/s)':>14} {'threshold':>11} {'n in band':>11}")
    for sigma_star in (0.01, 0.0159, 0.0318, 0.05, 0.0796):
        thr, n = calibrate(sigma, args.epsilon, sigma_star)
        v = sigma_star * QDOT_MAX
        thr_s = f"{thr:11.2f}" if thr is not None else f"{'--':>11}"
        print(f"{sigma_star:10.4f} {v:14.3f} {thr_s} {n:11d}")

    sigma_star = args.v_task / QDOT_MAX
    thr, n = calibrate(sigma, args.epsilon, sigma_star)
    print(f"\nFor v_task = {args.v_task} m/s at qdot_max = {QDOT_MAX:.4f} rad/s:")
    print(f"  sigma* = {sigma_star:.4f}  ->  threshold = {thr:.2f}  (n = {n})")


if __name__ == "__main__":
    main()
