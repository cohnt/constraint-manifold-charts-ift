#!/usr/bin/env python3
"""
calibrate_boundary_threshold.py

Derive the boundary reachability constraint's epsilon and threshold from robot
kinematics, with no reference to solver outcomes.

WHY NOT TUNE AGAINST SUCCESS RATE: the boundary constraint is an inequality
`b <= tau`, so any search that rewards downstream success will happily drive tau
upward until the constraint stops constraining anything. That failure mode was hit
in this repo (tau = 60.0 rejected 0% of unreachable samples) and independently in
the EAIK sibling experiment. Both parameters must come from kinematics and
numerics instead.

The constraint is

    b(q) = -log det( S J(q) (S J(q))^T + eps * I )  <=  tau,
    S    = diag(L, L, L, 1, 1, 1),

where J is the geometric Jacobian [angular; linear] of the subordinate arm. The
angular rows are in rad/s and the linear rows in m/s, so without the row scaling S
the determinant mixes units and no single epsilon is coherent. L is the maximum
|p_ee| over the joint-limit box, which puts every entry of S*J in metres.

Choosing epsilon -- three brackets:
  * far below a typical sigma_min^2, so it does not perturb well-conditioned
    configurations;
  * far above the floating-point floor sigma_max^2 * eps_machine;
  * large enough to cap the constraint's gradient, which is O(1/eps) at an exact
    singularity.

Choosing tau -- velocity amplification: realising a task-space speed v_task
requires joint rates up to v_task / sigma_min, and the URDF caps joint velocity at
qdot_max, so the smallest acceptable singular value is

    sigma* = v_task / qdot_max.

A worst-case sufficient bound on b is far too conservative to be usable, so
calibrate against the sigma_min level set: sample the configuration space, keep
configurations whose sigma_min is near sigma*, and report the median b there. This
is still purely kinematic -- it never runs the optimiser or the planner.

Usage:
    python3 scripts/analysis/calibrate_boundary_threshold.py [--samples N] [--v-task V]
"""

import argparse
import os
import sys

import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
repo_dir = os.path.abspath(os.path.join(script_dir, "../.."))
sys.path.append(os.path.join(repo_dir, "cpp_parameterization/python"))
sys.path.append(repo_dir)

import pydrake  # noqa: F401
from pydrake.all import MultibodyPlant, Parser

from iiwa_ik import ComputePoseJacobianGeometric
import src.common as common
from src.iiwa_analytic_ik import iiwa_limits_lower, iiwa_limits_upper

# Benchmark scenario constant (run_full_comparison.py).
GRASP_DISTANCE = 0.6


def build_plant():
    plant = MultibodyPlant(0.0)
    parser = Parser(plant)
    parser.package_map().AddPackageXml(os.path.join(common.RepoDir(), "package.xml"))
    parser.AddModels(os.path.join(
        common.RepoDir(), "models/iiwa14_convex_decimated_collision.urdf"))
    plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("base"))
    plant.Finalize()
    return plant


def measure_length_scale(plant, n_samples, grasp_distance, seed=0):
    """L = max |p_ee| over the joint-limit box, in metres."""
    ctx = plant.CreateDefaultContext()
    W = plant.world_frame()
    F = plant.GetFrameByName("iiwa_link_7")
    rng = np.random.default_rng(seed)
    best = 0.0
    for _ in range(n_samples):
        q = rng.uniform(iiwa_limits_lower, iiwa_limits_upper)
        plant.SetPositions(ctx, q)
        X = plant.CalcRelativeTransform(ctx, W, F)
        p_ee = X.translation() + X.rotation().matrix()[:, 2] * grasp_distance
        best = max(best, float(np.linalg.norm(p_ee)))
    return best


def sample_singular_values(n_samples, length_scale, grasp_distance, seed=0):
    """Singular values of the length-scaled geometric Jacobian, shape (n, 6)."""
    row_scale = np.diag([length_scale] * 3 + [1.0] * 3)
    rng = np.random.default_rng(seed)
    out = np.empty((n_samples, 6))
    for i in range(n_samples):
        q = rng.uniform(iiwa_limits_lower, iiwa_limits_upper)
        J = ComputePoseJacobianGeometric(q, grasp_distance)
        out[i] = np.linalg.svd(row_scale @ J, compute_uv=False)
    return out


def calibrate(sigma, epsilon, sigma_star, band=0.15):
    """Median b over configurations whose sigma_min is within `band` of sigma*."""
    b = -np.sum(np.log(sigma ** 2 + epsilon), axis=1)
    s_min = sigma[:, -1]
    in_band = ((s_min > sigma_star * (1 - band)) &
               (s_min < sigma_star * (1 + band)))
    if in_band.sum() < 20:
        return None, int(in_band.sum())
    return float(np.median(b[in_band])), int(in_band.sum())


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples", type=int, default=20000)
    ap.add_argument("--length-samples", type=int, default=50000)
    ap.add_argument("--epsilon", type=float, default=1e-6)
    ap.add_argument("--length-scale", type=float, default=None,
                    help="Override L (m). Default: measured from the URDF.")
    ap.add_argument("--grasp-distance", type=float, default=GRASP_DISTANCE)
    ap.add_argument("--v-task", type=float, default=0.1,
                    help="Task-space speed the arm should realise (m/s).")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    plant = build_plant()
    qdot_max = float(plant.GetVelocityUpperLimits().min())

    L = args.length_scale
    if L is None:
        print(f"Measuring L over {args.length_samples} configurations ...")
        L = measure_length_scale(plant, args.length_samples,
                                 args.grasp_distance, args.seed)
    print(f"  L = {L:.4f} m   (max |p_ee|, grasp_distance = {args.grasp_distance})")
    print(f"  qdot_max = {qdot_max:.4f} rad/s   (binding joint from the URDF)")

    print(f"\nSampling {args.samples} configurations ...")
    sigma = sample_singular_values(args.samples, L, args.grasp_distance, args.seed)
    s_max_max = float(sigma[:, 0].max())
    s_min_med = float(np.median(sigma[:, -1]))
    print(f"  sigma_max: median {np.median(sigma[:, 0]):.4f}   max {s_max_max:.4f}")
    print(f"  sigma_min: median {s_min_med:.5f}   min {sigma[:, -1].min():.3e}")
    print(f"  => epsilon must be << {s_min_med**2:.2e} (median sigma_min^2)")
    print(f"     and              >> {s_max_max**2 * np.finfo(float).eps:.2e} (float floor)")

    print(f"\nCalibration at epsilon = {args.epsilon:.0e}:")
    print(f"{'sigma*':>10} {'v_task (m/s)':>14} {'threshold':>11} {'n in band':>11}")
    for sigma_star in (0.01, 0.0159, 0.0318, 0.05, 0.0796):
        thr, n = calibrate(sigma, args.epsilon, sigma_star)
        thr_s = f"{thr:11.2f}" if thr is not None else f"{'--':>11}"
        print(f"{sigma_star:10.4f} {sigma_star * qdot_max:14.3f} {thr_s} {n:11d}")

    sigma_star = args.v_task / qdot_max
    thr, n = calibrate(sigma, args.epsilon, sigma_star)
    print(f"\nFor v_task = {args.v_task} m/s at qdot_max = {qdot_max:.4f} rad/s:")
    if thr is None:
        print(f"  sigma* = {sigma_star:.4f}  ->  too few samples in band (n = {n})")
    else:
        print(f"  sigma* = {sigma_star:.4f}  ->  threshold = {thr:.2f}  (n = {n})")


if __name__ == "__main__":
    main()
