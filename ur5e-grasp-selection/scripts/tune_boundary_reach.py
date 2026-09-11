"""
tune_boundary_reach.py

Script for tuning the boundary reachability constraint hyperparameters (epsilon and threshold).
Samples configurations, classifies their reachability, and searches for the optimal hyperparameters
that cut off all non-reachable targets while keeping as many reachable ones as possible.
"""

import os
import sys
import argparse
import numpy as np

# ── Path Setup ────────────────────────────────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_dir   = os.path.abspath(os.path.join(script_dir, ".."))
sys.path.insert(0, repo_dir)

from pydrake.all import (
    DiagramBuilder,
    AddMultibodyPlantSceneGraph,
    Parser,
    ProcessModelDirectives,
    LoadModelDirectives,
    RigidTransform,
    RotationMatrix,
    JacobianWrtVariable,
)

from src.util import RepoDir
from src.eaik_ik import EaikIK
from src.ur_experiments import ur5e_solver_limits_lower, ur5e_solver_limits_upper


def build_env():
    directives_file = os.path.join(RepoDir(), "models/ur5e_collision.yaml")
    
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.01)
    
    parser = Parser(plant, scene_graph)
    package_xml_path = os.path.join(RepoDir(), "package.xml")
    parser.package_map().AddPackageXml(package_xml_path)
    ur_description_xml = os.path.join(RepoDir(), "models/universal_robots/ur_description/package.xml")
    if os.path.exists(ur_description_xml):
        parser.package_map().AddPackageXml(ur_description_xml)
        
    ProcessModelDirectives(LoadModelDirectives(directives_file), plant, parser)
    plant.Finalize()
    
    diagram = builder.Build()
    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyContextFromRoot(context)
    arm_instance = plant.GetModelInstanceByName("ur5e")
    ee_frame = plant.GetFrameByName("tool0", arm_instance)
    
    return plant, plant_context, arm_instance, ee_frame


def compute_jacobian(plant, plant_context, arm_instance, ee_frame, q):
    plant.SetPositions(plant_context, arm_instance, q)
    J = plant.CalcJacobianSpatialVelocity(
        plant_context,
        JacobianWrtVariable.kV,
        ee_frame,
        np.zeros(3),
        plant.world_frame(),
        plant.world_frame(),
    )
    
    # Extract columns corresponding to the arm velocities
    indices = []
    for idx in plant.GetJointIndices():
        joint = plant.get_joint(idx)
        if joint.model_instance() == arm_instance:
            for k in range(joint.num_velocities()):
                indices.append(joint.velocity_start() + k)
    return J[:, indices]


def compute_boundary_value(J, epsilon):
    A = J @ J.T + epsilon * np.eye(6)
    sign, logdet = np.linalg.slogdet(A)
    if sign <= 0:
        return 1e10
    return -logdet


def main():
    parser = argparse.ArgumentParser(description="Tune Boundary Reachability Constraint Hyperparameters")
    parser.add_argument("--num-samples", type=int, default=50, help="Number of base configurations to sample")
    parser.add_argument("--step-size", type=float, default=0.02, help="Perturbation step size along radial direction")
    parser.add_argument("--tolerance", type=float, default=1e-6, help="FK pose residual tolerance for reachability")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    print("Building environment...")
    plant, plant_context, arm_instance, ee_frame = build_env()
    ik_solver = EaikIK()

    # We will collect a dataset of (q, J, is_reachable)
    # J is stored to speed up the grid search
    reachable_samples = []
    unreachable_samples = []

    print(f"Sampling configurations (num_samples={args.num_samples}, tolerance={args.tolerance})...")
    
    sampled_count = 0
    attempts = 0
    max_attempts = args.num_samples * 1000

    while sampled_count < args.num_samples and attempts < max_attempts:
        attempts += 1
        # Sample base configuration within the solver joint box (+/-pi), matching
        # the range EAIK's atan2 roots and the benchmark's initial guesses live in.
        q_base = rng.uniform(ur5e_solver_limits_lower, ur5e_solver_limits_upper)
        
        # Check forward kinematics
        plant.SetPositions(plant_context, arm_instance, q_base)
        X_base = ee_frame.CalcPoseInWorld(plant_context)
        t_base = X_base.translation()
        
        # Verify it is well within reasonable workspace limits
        dist_from_base = np.linalg.norm(t_base)
        if dist_from_base < 0.2 or dist_from_base > 0.8:
            continue
            
        u = t_base / dist_from_base
        
        # Sweep radial perturbation d from -0.15m to 0.25m
        sweep_ds = np.arange(-0.15, 0.25, args.step_size)
        
        for d in sweep_ds:
            t_target = t_base + d * u
            X_target = RigidTransform(X_base.rotation(), t_target)
            X_target_mat = X_target.GetAsMatrix4()
            
            # Solve with EAIK
            sols = ik_solver.solve(X_target_mat)
            if not sols:
                continue
                
            # Evaluate FK residual for each solution
            best_q = None
            best_res = np.inf
            
            for q in sols:
                # Check joint limits
                if np.any(q < ur5e_solver_limits_lower) or np.any(q > ur5e_solver_limits_upper):
                    continue
                X_actual = ik_solver.fk(q)
                res = np.sum((X_actual - X_target_mat) ** 2)
                if res < best_res:
                    best_res = res
                    best_q = q
                    
            if best_q is None:
                continue
                
            # Classify
            is_reachable = best_res <= args.tolerance
            J = compute_jacobian(plant, plant_context, arm_instance, ee_frame, best_q)
            
            if is_reachable:
                reachable_samples.append((best_q, J))
            else:
                unreachable_samples.append((best_q, J))
                
        sampled_count += 1

    print(f"Generated {len(reachable_samples)} reachable and {len(unreachable_samples)} non-reachable configurations.")
    if not reachable_samples or not unreachable_samples:
        print("Error: Could not generate both reachable and non-reachable configurations. Try different parameters.")
        return

    # Sweep epsilons and select optimal threshold
    epsilons = [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 5e-3, 1e-2]
    
    print("\nGrid Search over Epsilon:")
    print(f"{'Epsilon':<12} {'Threshold':<15} {'Reachable Kept':<18} {'Unreachable Cut':<18}")
    print("-" * 67)
    
    best_eps = None
    best_threshold = None
    best_fpr = 101.0 # Fraction of reachable configurations cut off (want to minimize, in %)
    
    for eps in epsilons:
        # Compute boundary reachability values
        Y_reach = [compute_boundary_value(J, eps) for _, J in reachable_samples]
        Y_unreach = [compute_boundary_value(J, eps) for _, J in unreachable_samples]
        
        # Threshold: to guarantee cutting off all non-reachable targets, we set threshold strictly below min(Y_unreach)
        threshold = np.min(Y_unreach) - 1e-5
        
        # Calculate rates
        unreachable_cut = np.sum(np.array(Y_unreach) > threshold)
        unreachable_cut_pct = (unreachable_cut / len(unreachable_samples)) * 100.0
        
        reachable_kept = np.sum(np.array(Y_reach) <= threshold)
        reachable_kept_pct = (reachable_kept / len(reachable_samples)) * 100.0
        
        reachable_cut_pct = 100.0 - reachable_kept_pct
        
        print(f"{eps:<12.1e} {threshold:<15.4f} {reachable_kept_pct:<16.1f}% ({reachable_kept}/{len(reachable_samples)})  "
              f"{unreachable_cut_pct:<16.1f}% ({unreachable_cut}/{len(unreachable_samples)})")
              
        # We want to minimize the fraction of reachable targets cut off (which is equivalent to maximizing those kept)
        if reachable_cut_pct < best_fpr:
            best_fpr = reachable_cut_pct
            best_eps = eps
            best_threshold = threshold

    print("\n" + "=" * 60)
    print("RECOMMENDED HYPERPARAMETERS:")
    print(f"  Best Epsilon:   {best_eps:.1e}")
    print(f"  Best Threshold: {best_threshold:.4f}")
    print(f"  This keeps {100.0 - best_fpr:.1f}% of reachable targets while cutting off 100.0% of unreachable targets.")
    print("=" * 60)


if __name__ == "__main__":
    main()
