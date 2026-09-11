import sys
import time
import numpy as np
import pandas as pd
import os
from pydrake.all import InitializeAutoDiff, ExtractValue, ExtractGradient
import pydrake.math

# Setup paths
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_dir = os.path.abspath(os.path.join(script_dir, "../.."))
sys.path.append(os.path.join(repo_dir, "cpp_parameterization/python"))
sys.path.append(repo_dir)

from iiwa_ik import (
    BimanualConfig,
    MakeParameterization,
    AutoDiffConfig,
    IftSingularityHandling,
    OldStyleReachableConstraint,
    IiwaBimanualReachableConstraint,
    IiwaBimanualPsiSingularityConstraint
)
from src.iiwa_analytic_ik import Analytic_IK_7DoF, iiwa_alpha, iiwa_d

# Constants
FILTER_PSI_SINGULARITIES = False
IIWA_LOWER_LIMITS = np.array([-2.96, -2.09, -2.96, -2.09, -2.96, -2.09, -3.05])
IIWA_UPPER_LIMITS = np.array([ 2.96,  2.09,  2.96,  2.09,  2.96,  2.09,  3.05])


def compute_psi_ad(thetas, s_up, e_up, w_up, config):
    # Local AutoDiff-compatible psi calculation
    def safe_norm(x):
        return pydrake.math.sqrt(dot_ad(x, x))
    
    def dot_ad(a, b):
        res = a[0]*b[0]
        for i in range(1, len(a)):
            res += a[i]*b[i]
        return res

    def matmul_ad(A, B):
        C = np.zeros((4,4), dtype=object)
        for i in range(4):
            for j in range(4):
                C[i,j] = A[i,0]*B[0,j] + A[i,1]*B[1,j] + A[i,2]*B[2,j] + A[i,3]*B[3,j]
        return C

    def cross_ad(a, b):
        return np.array([
            a[1]*b[2] - a[2]*b[1],
            a[2]*b[0] - a[0]*b[2],
            a[0]*b[1] - a[1]*b[0]
        ])

    def get_T(ti, ai, di):
        t = ti[0] if isinstance(ti, (np.ndarray, list)) else ti
        return np.array([
            [pydrake.math.cos(t), -pydrake.math.sin(t)*pydrake.math.cos(ai), pydrake.math.sin(t)*pydrake.math.sin(ai), 0],
            [pydrake.math.sin(t), pydrake.math.cos(t)*pydrake.math.cos(ai), -pydrake.math.cos(t)*pydrake.math.sin(ai), 0],
            [0, pydrake.math.sin(ai), pydrake.math.cos(ai), di],
            [0, 0, 0, 1]
        ], dtype=object)

    GC4 = 1 if e_up else -1
    eval_Ts = [get_T(thetas[i], iiwa_alpha[i], iiwa_d[i]) for i in range(7)]
    
    T_07 = eval_Ts[0]
    for i in range(1, 7):
        T_07 = matmul_ad(T_07, eval_Ts[i])
    
    p_07 = T_07[:-1, -1]
    R_07 = T_07[:-1, :-1]

    d_bs, d_se, d_ew, d_wf = iiwa_d[0], iiwa_d[2], iiwa_d[4], iiwa_d[6]
    p_02 = np.array([0, 0, d_bs])
    p_67 = np.array([0, 0, d_wf])

    p_26 = p_07 - p_02 - np.array([
        R_07[0,0]*p_67[0] + R_07[0,1]*p_67[1] + R_07[0,2]*p_67[2],
        R_07[1,0]*p_67[0] + R_07[1,1]*p_67[1] + R_07[1,2]*p_67[2],
        R_07[2,0]*p_67[0] + R_07[2,1]*p_67[1] + R_07[2,2]*p_67[2]
    ])
    
    T_04 = matmul_ad(matmul_ad(matmul_ad(eval_Ts[0], eval_Ts[1]), eval_Ts[2]), eval_Ts[3])
    T_06 = matmul_ad(matmul_ad(T_04, eval_Ts[4]), eval_Ts[5])
    p_04 = T_04[:-1, -1]
    p_06 = T_06[:-1, -1]

    p26_sq = dot_ad(p_26, p_26)
    margin = config.clipping_margin
    arccos_in_4v = (p26_sq - d_se**2 - d_ew**2) / (2 * d_se * d_ew)
    arccos_in_4v = pydrake.math.max(-(1.0 - margin), pydrake.math.min(1.0 - margin, arccos_in_4v))
    theta_4v = GC4 * pydrake.math.arccos(arccos_in_4v)
    theta_1v = pydrake.math.arctan2(p_26[1], p_26[0]) 
    
    arccos_in_phi = (d_se**2 + p26_sq - d_ew**2) / (2 * d_se * safe_norm(p_26))
    arccos_in_phi = pydrake.math.max(-(1.0 - margin), pydrake.math.min(1.0 - margin, arccos_in_phi))
    phi = pydrake.math.arccos(arccos_in_phi)
    theta_2v = pydrake.math.arctan2(safe_norm(p_26[:2]), p_26[2]) + (GC4 * phi)
    
    T_v0 = get_T(theta_1v, iiwa_alpha[0], iiwa_d[0])
    T_v1 = get_T(theta_2v, iiwa_alpha[1], iiwa_d[1])
    T_v2 = get_T(0.0, iiwa_alpha[2], iiwa_d[2])
    T_v3 = get_T(theta_4v, iiwa_alpha[3], iiwa_d[3])
    
    T_02_v = matmul_ad(T_v0, T_v1)
    T_04_v = matmul_ad(matmul_ad(T_02_v, T_v2), T_v3)
    p_02_v = T_02_v[:-1, -1]
    p_04_v = T_04_v[:-1, -1]
    p_06_v = p_06

    v_se_v = (p_04_v - p_02_v) / safe_norm(p_04_v - p_02_v)
    v_sw_v = (p_06_v - p_02_v) / safe_norm(p_06_v - p_02_v)
    v_sew_v = cross_ad(v_se_v, v_sw_v)
    v_sew_v_hat = v_sew_v / safe_norm(v_sew_v)

    v_se = (p_04 - p_02) / safe_norm(p_04 - p_02)
    v_sw = (p_06 - p_02) / safe_norm(p_06 - p_02)
    v_sew = cross_ad(v_se, v_sw)
    v_sew_hat = v_sew / safe_norm(v_sew)

    dot_val = dot_ad(v_sew_v_hat, v_sew_hat)
    margin_psi = config.clipping_margin_psi
    dot_val = pydrake.math.max(-(1.0 - margin_psi), pydrake.math.min(1.0 - margin_psi, dot_val))
    
    v_sew_v_hat_f = np.array([x.value() if hasattr(x, 'value') else x for x in v_sew_v_hat])
    v_sew_hat_f = np.array([x.value() if hasattr(x, 'value') else x for x in v_sew_hat])
    p_26_f = np.array([x.value() if hasattr(x, 'value') else x for x in p_26])
    
    sg_psi = np.sign(np.dot(np.cross(v_sew_v_hat_f, v_sew_hat_f), p_26_f))
    if sg_psi == 0: sg_psi = 1.0
    
    psi = sg_psi * pydrake.math.arccos(dot_val)
    return psi

def main():
    print("Running Gradient Accuracy and Runtime Study...")
    np.random.seed(42)
    num_samples = 10000
    partials_sizes = [2**n for n in range(0, 10+1)]
    
    s_up, e_up, w_up = True, True, True
    config = BimanualConfig(s_up, e_up, w_up, 0.6, 1e-4, 1e-12)
    param_ad = MakeParameterization(config, AutoDiffConfig(use_ift=False))
    param_ift = MakeParameterization(config, AutoDiffConfig(use_ift=True, ift_handling=IftSingularityHandling.kPseudoinverse))
    reach_con_old = OldStyleReachableConstraint(config, AutoDiffConfig(use_ift=False))
    reach_con_new = IiwaBimanualReachableConstraint(config)
    psi_singularity_con = IiwaBimanualPsiSingularityConstraint(config, AutoDiffConfig(use_ift=False))
    ik_analytic = Analytic_IK_7DoF(iiwa_alpha, iiwa_d)

    out_file = os.path.join(repo_dir, "out/gradient_study_results.csv")
    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    rows = []
    cached_samples = []
    
    print(f"Sampling {num_samples} configurations...")
    count = 0
    while count < num_samples:
        q_c = np.random.uniform(IIWA_LOWER_LIMITS, IIWA_UPPER_LIMITS)
        psi = np.random.uniform(0.0, 2 * np.pi, size=(1,))
        q_and_psi = np.concatenate([q_c, psi])

        if reach_con_old.Eval(q_and_psi)[0] > 1e-6: continue
        reach_vals = reach_con_new.Eval(q_and_psi)
        dist_to_boundary = 1.0 - np.max(np.abs(reach_vals))
        
        if FILTER_PSI_SINGULARITIES:
            if np.any(np.abs(psi_singularity_con.Eval(q_and_psi)) > (1.0 - 1e-4)): continue

        q_and_psi_ad = InitializeAutoDiff(q_and_psi, np.eye(8))
        out_ad = param_ad.get_parameterization_autodiff()(q_and_psi_ad)
        cond_param = np.linalg.cond(ExtractGradient(out_ad))
        q_sub_val = ExtractValue(out_ad)[7:14].flatten()
        
        eval_Ts = [ik_analytic.Ts[i](float(q_sub_val[i])) for i in range(7)]
        Ts_accum = [eval_Ts[0]]
        for i in range(1, 7): Ts_accum.append(Ts_accum[-1] @ eval_Ts[i])
        p_n = Ts_accum[-1][:3, 3]
        J_w, J_v = np.zeros((3, 7)), np.zeros((3, 7))
        J_w[:, 0], J_v[:, 0] = [0,0,1], np.cross([0,0,1], p_n)
        for i in range(1, 7):
            z_i, p_i = Ts_accum[i-1][:3, 2], Ts_accum[i-1][:3, 3]
            J_w[:, i], J_v[:, i] = z_i, np.cross(z_i, p_n - p_i)
        J_kin = np.vstack([J_w, J_v])
        
        q_sub_ad_for_psi = InitializeAutoDiff(q_sub_val, np.eye(7))
        psi_ad_val = compute_psi_ad(q_sub_ad_for_psi, s_up, e_up, w_up, config)
        if hasattr(psi_ad_val, 'derivatives') and psi_ad_val.derivatives().size == 7:
            grad_psi = psi_ad_val.derivatives()
        else:
            grad_psi = np.zeros(7)
        J_aug = np.vstack([J_kin, grad_psi.reshape(1, 7)])
        
        cached_samples.append({
            "q_and_psi": q_and_psi,
            "dist_to_boundary": dist_to_boundary,
            "cond_kin": min(np.linalg.cond(J_kin), 1e16),
            "cond_aug": min(np.linalg.cond(J_aug), 1e16),
            "cond_param": min(cond_param, 1e16)
        })
        count += 1
        if count % 200 == 0: print(f"  Sampled {count}/{num_samples}...")

    for sz in partials_sizes:
        print(f"Benchmarking Partials Size: {sz}")
        for i, sample in enumerate(cached_samples):
            V = np.random.randn(8, sz)
            q_and_psi_ad = InitializeAutoDiff(sample["q_and_psi"], V)
            
            t0 = time.perf_counter()
            grad_ad = ExtractGradient(param_ad.get_parameterization_autodiff()(q_and_psi_ad))
            ad_dt = time.perf_counter() - t0
            
            t0 = time.perf_counter()
            grad_ift = ExtractGradient(param_ift.get_parameterization_autodiff()(q_and_psi_ad))
            ift_dt = time.perf_counter() - t0
            
            diff = np.max(np.abs(grad_ad - grad_ift))
            base_row = {
                "Partials Size": sz, "Max Error": diff, "Boundary Distance": sample["dist_to_boundary"],
                "Cond Kinematic": sample["cond_kin"], "Cond Augmented": sample["cond_aug"], "Cond Parameterization": sample["cond_param"]
            }
            rows.append({**base_row, "Runtime (s)": ad_dt, "Method": "Standard AD"})
            rows.append({**base_row, "Runtime (s)": ift_dt, "Method": "IFT AD"})

    pd.DataFrame(rows).to_csv(out_file, index=False)
    print(f"Results saved to {out_file}")

if __name__ == "__main__":
    main()
