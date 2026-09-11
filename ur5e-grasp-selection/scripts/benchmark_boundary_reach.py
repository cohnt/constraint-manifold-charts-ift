"""
benchmark_boundary_reach.py

Three-way benchmark of UR5e IK formulations:
  1. Old   — full q-space IK with EE pose constraint (UrIKProblemOldFormulation)
  2. Direct — minimal coords + direct reachability (residual norm ‖FK(IK(X))−X‖²)
  3. Boundary — minimal coords + boundary reachability (−log det(JJᵀ + εI))

Usage:
    python scripts/benchmark_boundary_reach.py [options]

Options:
    --num-targets N      Number of random feasible target poses (default: 10)
    --num-guesses M      Number of random initial guesses per target (default: 100)
    --max-wall-time T    Solver wall-time limit per solve in seconds (default: 10.0)
    --solver S           Solver to use: SNOPT, IPOPT or NLOPT (default: SNOPT)
    --tune               Diagnostic sweep of (epsilon, threshold) against success rate.
                         NOT the source of truth: those parameters are derived from
                         kinematics and numerics in calibrate_boundary_threshold.py.
    --tune-targets N     Targets used during tuning (default: 5)
    --tune-guesses M     Guesses per target during tuning (default: 10)
    --epsilon E          Gram-matrix regularisation in m^2 (default: 1e-6, derived)
    --threshold T        Boundary constraint threshold (default: 10.0, derived)
    --v-task V           Task speed the threshold was derived from (default: 0.1 m/s)
    --out FILE           Output JSON path (default: logs/boundary_benchmark_<ts>.json)
    --seed S             Global random seed (default: 42)
"""

import os
import sys
import time
import json
import argparse
import datetime
import statistics as stats

import numpy as np

# ── Path setup ────────────────────────────────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_dir   = os.path.abspath(os.path.join(script_dir, ".."))
sys.path.insert(0, repo_dir)

from pydrake.all import (
    RigidTransform, RollPitchYaw, DiagramBuilder, AddMultibodyPlantSceneGraph,
    Parser, ProcessModelDirectives, LoadModelDirectives, LoadModelDirectivesFromString,
    ApplyVisualizationConfig, VisualizationConfig
)

from src.util import BuildEnv, RepoDir
from src.boundary_reach_constraint import BoundaryReachConstraint
from src.ift_gradients import IftGradient
from src.ur_experiments import (
    UrProblemOptions,
    UrIKProblemOldFormulation,
    UrIKProblemNewFormulation,
)
from src.bench_stats import (
    compute_stats,
    compute_all_stats,
    mutual_stats,
    paired_success_stats,
    bootstrap_ci,
    cap_hit_stats,
    failure_reason_histogram,
    multistart_stats,
)

METHODS = ("old", "direct", "boundary")
# NOTE: the joint limits are deliberately not imported here.  Targets and initial guesses
# are sampled directly in [-pi, pi] (sample_targets / sample_initial_guesses), which is
# the solver box every formulation is now restricted to; see ur5e_solver_limits_* in
# src/ur_experiments.py.

# ── Defaults ──────────────────────────────────────────────────────────────────
# The boundary constraint's epsilon and threshold are DERIVED, not tuned against success
# rate: see scripts/calibrate_boundary_threshold.py.  epsilon comes from the numerics of
# the length-scaled Jacobian, and the threshold from the velocity-amplification criterion
# sigma* = v_task / qdot_max.  The sweep below remains only as a diagnostic; do not treat
# its output as the source of truth.
TUNE_EPSILONS    = [1e-6, 1e-4]
TUNE_THRESHOLDS  = [9.4, 14.0]


# ── Environment helpers ───────────────────────────────────────────────────────

def build_env(meshcat=None, target_pose=None, no_obstacles=False):
    yaml_name = "ur5e_collision_simple.yaml" if no_obstacles else "ur5e_collision.yaml"
    directives_file = os.path.join(RepoDir(), f"models/{yaml_name}")
    
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.01)
    
    parser = Parser(plant, scene_graph)
    package_xml_path = os.path.join(RepoDir(), "package.xml")
    parser.package_map().AddPackageXml(package_xml_path)
    ur_description_xml = os.path.join(RepoDir(), "models/universal_robots/ur_description/package.xml")
    if os.path.exists(ur_description_xml):
        parser.package_map().AddPackageXml(ur_description_xml)
        
    ProcessModelDirectives(LoadModelDirectives(directives_file), plant, parser)
    
    if target_pose is not None:
        rpy = target_pose.rotation().ToRollPitchYaw().vector()
        rpy_deg = np.degrees(rpy)
        xyz = target_pose.translation()
        yaml_str = f"""
directives:
  - add_model:
      name: target_mug
      file: package://eaik_ift_experiment/models/mug/mug_simple_red.urdf
  - add_weld:
      parent: world
      child: target_mug::mug_body_link
      X_PC:
        translation: [{xyz[0]}, {xyz[1]}, {xyz[2]}]
        rotation: !Rpy {{ deg: [{rpy_deg[0]}, {rpy_deg[1]}, {rpy_deg[2]}] }}
"""
        ProcessModelDirectives(LoadModelDirectivesFromString(yaml_str), plant, parser)
    
    plant.Finalize()
    
    if meshcat is not None:
        vis_config = VisualizationConfig()
        vis_config.publish_illustration = True
        vis_config.publish_proximity = True
        vis_config.publish_inertia = False
        vis_config.delete_on_initialization_event = True
        ApplyVisualizationConfig(vis_config, builder, meshcat=meshcat)

    diagram = builder.Build()
    return diagram, plant




def make_problems(diagram, epsilon, threshold):
    urdf_path = os.path.join(
        RepoDir(),
        "models/universal_robots/ur_description/urdf/ur5e_drake_collision.urdf",
    )
    old_prob  = UrIKProblemOldFormulation(diagram, urdf_path)
    new_prob  = UrIKProblemNewFormulation(diagram, use_boundary_reach=False)
    brc_prob  = UrIKProblemNewFormulation(diagram, use_boundary_reach=True, epsilon=epsilon,
                                          threshold=threshold)
    return old_prob, new_prob, brc_prob


# ── Target generation ─────────────────────────────────────────────────────────

def sample_targets(old_prob, n_targets, rng):
    """
    Sample n_targets guaranteed-feasible target poses by forward kinematics from
    random collision-free configurations.  Returns list of (name, X_WM, q_gold).
    """
    targets = []
    attempts = 0
    max_attempts = n_targets * 5000
    while len(targets) < n_targets and attempts < max_attempts:
        attempts += 1
        q_gold = rng.uniform(-np.pi, np.pi, 6)
        if old_prob.EvalCollision(q_gold) == 0:
            old_prob.plant.SetPositions(
                old_prob.plant_context, old_prob.arm_instance, q_gold
            )
            X_WE = old_prob.ee_frame.CalcPoseInWorld(old_prob.plant_context)
            X_WM = X_WE.multiply(old_prob.X_EG)
            targets.append((f"target_{len(targets):04d}", X_WM, q_gold))
    if len(targets) < n_targets:
        print(f"  WARNING: Only found {len(targets)}/{n_targets} feasible targets "
              f"after {attempts} attempts.")
    return targets


def sample_initial_guesses(old_prob, X_WM, n_guesses, rng, mug_height=0.08):
    """
    Sample n_guesses tuples of (q_init, p_init) uniformly in joint space.

    q_init : a random collision-free joint configuration.  Also serves as the branch
             reference for the parameter-space formulations (see SelectBranch).
    p_init : the grasp-frame pose of q_init expressed in the target mug frame, so that
             the parameter-space formulations start from exactly the same configuration
             as the joint-space baseline.

    X_WM must be the target mug pose from sample_targets (which already includes X_EG);
    recomputing it here from forward kinematics is what previously desynchronized the
    joint-space and parameter-space initial guesses.
    """
    guesses = []
    attempts = 0
    max_attempts = n_guesses * 10000

    while len(guesses) < n_guesses and attempts < max_attempts:
        attempts += 1
        q = rng.uniform(-np.pi, np.pi, 6)
        if old_prob.EvalCollision(q) == 0:
            guesses.append((q, old_prob.GraspParamsFromQ(q, X_WM)))

    if len(guesses) < n_guesses:
        print(f"  WARNING: Only found {len(guesses)}/{n_guesses} initial guesses "
              f"after {attempts} attempts.")
    return guesses



# ── Solve helpers ─────────────────────────────────────────────────────────────

def verify_solution(prob, res, options, X_WM, is_parameter_space=False):
    """
    Unified success verification.  Returns (ok, reason), where reason names the gate that
    rejected the run and is None on success:

      "nan"        — the recovered configuration is not finite
      "constraint" — infeasible for the mathematical program at tol 1e-6
      "task_error" — the grasp frame missed its target by more than 1 cm
      "mug_xy"     — (joint-space only) the grasp frame is not on the mug axis

    The reason is recorded per run so a failure histogram can say *why* a formulation
    fails, which a bare success rate cannot.
    """
    q_opt = prob.GetQ(res.get_x_val()) if is_parameter_space else res.get_x_val()
    if np.any(np.isnan(q_opt)):
        return False, "nan"

    # Check feasibility for the mathematical program up to tolerance 1e-6
    if not prob.prog.CheckSatisfied(prob.prog.GetAllConstraints(), res.GetSolution(), tol=1e-6):
        return False, "constraint"

    # Check task-space reachability (with ~1cm tolerance)
    prob.plant.SetPositions(prob.plant_context, prob.arm_instance, q_opt)
    X_WE_actual = prob.ee_frame.CalcPoseInWorld(prob.plant_context)
    X_WG_actual = X_WE_actual.multiply(prob.X_EG)

    if is_parameter_space:
        p_val = res.get_x_val()
        X_MG_desired = RigidTransform(RollPitchYaw(p_val[3:]), p_val[:3])
        X_WG_target = X_WM.multiply(X_MG_desired)
        err_trans = np.linalg.norm(X_WG_actual.translation() - X_WG_target.translation())
        if err_trans > 1e-2:
            return False, "task_error"
    else:
        X_MG_actual = X_WM.inverse().multiply(X_WG_actual)
        dist_xy = np.linalg.norm(X_MG_actual.translation()[:2])
        if dist_xy > 1e-2:
            return False, "mug_xy"

    return True, None


def solver_diagnostics(res):
    """
    (status, solver_time) off the solver details, or (None, nan) if unavailable.

    For SNOPT the status is its INFO code, which is the only reliable way to tell a
    time-limit exit from a genuine infeasibility: the time limit is soft, checked between
    major iterations, so a solve can overshoot the cap by seconds and wall time alone
    misclassifies it.
    """
    try:
        details = res.get_solver_details()
    except Exception:
        return None, float("nan")
    status = getattr(details, "info", None)
    if status is None:
        status = getattr(details, "status", None)
    solve_time = getattr(details, "solve_time", float("nan"))
    try:
        status = int(status) if status is not None else None
    except (TypeError, ValueError):
        status = None
    try:
        solve_time = float(solve_time)
    except (TypeError, ValueError):
        solve_time = float("nan")
    return status, solve_time


def eval_cost(prob, res, options, is_parameter_space=False):
    """Unified centering cost: multiplier * sum(w_i * (q_i - q_target_i)^2)."""
    # Scored against prob.q_target and prob.cost_weights, which _ResolveCostReference sets
    # from the same options object all three formulations are configured with.  That is
    # what makes the scored objective the minimised one under every cost_reference; with
    # the default "nominal" and uniform weights this is exactly ||q - q_nominal||^2, so
    # the numbers are unchanged from before the variants existed.
    q_opt = prob.GetQ(res.get_x_val()) if is_parameter_space else res.get_x_val()
    q_err = q_opt - prob.q_target
    cost = float(options["joint_centering_cost_multiplier"]
                 * np.sum(prob.cost_weights * q_err ** 2))

    # When a manipulability objective is active it is part of what was minimised, so it
    # must be part of what is scored, or the reported cost is not the solver's objective.
    w = options.get("manipulability_cost_weight", 0.0)
    if w != 0.0:
        brc = getattr(prob, "_eval_cost_brc", None)
        if brc is None:
            brc = BoundaryReachConstraint(prob, options.get("manipulability_epsilon", 1e-6))
            prob._eval_cost_brc = brc
        y, _ = brc.value_and_grad_q(np.asarray(q_opt, dtype=float))
        cost += float(w * y)
    return cost


def _record(prob, res, options, X_WM, elapsed, is_parameter_space):
    """
    Assemble the per-run result dict shared by all three formulations.

    Everything here is read off the result object, so it costs no extra solving.  The
    solver status is what distinguishes a time-limit exit from a genuine infeasibility,
    and the failure reason is what turns a bare success rate into an explanation.
    """
    ok, reason = verify_solution(prob, res, options, X_WM,
                                 is_parameter_space=is_parameter_space)
    status, solver_time = solver_diagnostics(res)
    return {
        "ok":            bool(ok),
        "cost":          eval_cost(prob, res, options, is_parameter_space) if ok
                         else float("nan"),
        "time":          elapsed,
        "fail_reason":   reason,
        "solver_status": status,
        "solver_time":   solver_time,
    }


FAILED_RUN = {
    "ok": False, "cost": float("nan"), "time": float("nan"),
    "fail_reason": "exception", "solver_status": None, "solver_time": float("nan"),
}


def solve_old(old_prob, X_WM, q_init, options, meshcat=None):
    """Run UrIKProblemOldFormulation and return the per-run result dict."""
    opts = UrProblemOptions(
        solver=options["solver"],
        max_wall_time=options["max_wall_time"],
        major_optimality_tol=options["major_optimality_tol"],
        minor_optimality_tol=options["minor_optimality_tol"],
        major_feasibility_tol=options["major_feasibility_tol"],
        avoid_collisions=options["avoid_collisions"],
        impose_joint_centering_cost=options["impose_joint_centering_cost"],
        joint_centering_cost_multiplier=options["joint_centering_cost_multiplier"],
        cost_reference=options["cost_reference"],
        joint_cost_weights=options["joint_cost_weights"],
        manipulability_cost_weight=options["manipulability_cost_weight"],
        target_mug=X_WM,
        q_initial=q_init,
        mug_height=options["mug_height"],
        minimum_distance=options["minimum_distance"],
    )
    old_prob.ApplyOptions(opts)
    old_prob.prog.SetInitialGuess(old_prob.ik.q(), q_init)

    t0  = time.time()
    res = old_prob.Solve()
    elapsed = time.time() - t0

    out = _record(old_prob, res, options, X_WM, elapsed, is_parameter_space=False)

    if meshcat is not None:
        q_opt = res.get_x_val()
        print(f"    [Visualizing] Old Formulation - Success: {out['ok']}")
        if not np.any(np.isnan(q_opt)):
            old_prob.plant.SetPositions(old_prob.plant_context, old_prob.arm_instance, q_opt)
            old_prob.diagram.ForcedPublish(old_prob.diagram_context)
            time.sleep(1.0)

    return out


def solve_new(new_prob, X_WM, q_init, p_init, options, meshcat=None):
    """Run UrIKProblemNewFormulation and return the per-run result dict."""
    opts = UrProblemOptions(
        solver=options["solver"],
        max_wall_time=options["max_wall_time"],
        major_optimality_tol=options["major_optimality_tol"],
        minor_optimality_tol=options["minor_optimality_tol"],
        major_feasibility_tol=options["major_feasibility_tol"],
        avoid_collisions=options["avoid_collisions"],
        impose_joint_centering_cost=options["impose_joint_centering_cost"],
        joint_centering_cost_multiplier=options["joint_centering_cost_multiplier"],
        cost_reference=options["cost_reference"],
        joint_cost_weights=options["joint_cost_weights"],
        manipulability_cost_weight=options["manipulability_cost_weight"],
        # Needed by cost_reference="initial"; harmless otherwise.  The minimal-coordinate
        # formulations seed the program from p_init rather than q_init, so this is the
        # objective's reference only, not the starting point.
        q_initial=q_init,
        ift_strategy=options.get("ift_strategy", "residual"),
        ift_damping_lam=options.get("ift_damping_lam", 1e-4),
        square_frobenius_norm=options.get("square_frobenius_norm", False),
        joint_limits=options.get("joint_limits", True),
        target_mug=X_WM,
        mug_height=options["mug_height"],
        minimum_distance=options["minimum_distance"],
    )
    new_prob.ApplyOptions(opts)
    new_prob.prog.SetInitialGuess(new_prob.p, p_init)

    t0  = time.time()
    res = new_prob.Solve()
    elapsed = time.time() - t0

    out = _record(new_prob, res, options, X_WM, elapsed, is_parameter_space=True)

    if meshcat is not None:
        print(f"    [Visualizing] Direct Formulation - Success: {out['ok']}")
        p_opt = res.get_x_val()
        if not np.any(np.isnan(p_opt)):
            q_res = new_prob.GetQ(p_opt)
            new_prob.plant.SetPositions(new_prob.plant_context, new_prob.arm_instance, q_res)
            new_prob.diagram.ForcedPublish(new_prob.diagram_context)
            time.sleep(1.0)

    return out


def solve_boundary(brc_prob, X_WM, q_init, p_init, options, meshcat=None):
    """Run UrIKProblemNewFormulation (Boundary) and return the per-run result dict."""
    opts = UrProblemOptions(
        solver=options["solver"],
        max_wall_time=options["max_wall_time"],
        major_optimality_tol=options["major_optimality_tol"],
        minor_optimality_tol=options["minor_optimality_tol"],
        major_feasibility_tol=options["major_feasibility_tol"],
        avoid_collisions=options["avoid_collisions"],
        impose_joint_centering_cost=options["impose_joint_centering_cost"],
        joint_centering_cost_multiplier=options["joint_centering_cost_multiplier"],
        cost_reference=options["cost_reference"],
        joint_cost_weights=options["joint_cost_weights"],
        manipulability_cost_weight=options["manipulability_cost_weight"],
        # Needed by cost_reference="initial"; harmless otherwise.  The minimal-coordinate
        # formulations seed the program from p_init rather than q_init, so this is the
        # objective's reference only, not the starting point.
        q_initial=q_init,
        ift_strategy=options.get("ift_strategy", "residual"),
        ift_damping_lam=options.get("ift_damping_lam", 1e-6),
        joint_limits=options.get("joint_limits", True),
        target_mug=X_WM,
        mug_height=options["mug_height"],
        minimum_distance=options["minimum_distance"],
    )
    brc_prob.ApplyOptions(opts)
    brc_prob.prog.SetInitialGuess(brc_prob.p, p_init)

    t0  = time.time()
    res = brc_prob.Solve()
    elapsed = time.time() - t0

    out = _record(brc_prob, res, options, X_WM, elapsed, is_parameter_space=True)

    if meshcat is not None:
        print(f"    [Visualizing] Boundary Formulation - Success: {out['ok']}")
        p_opt = res.get_x_val()
        if not np.any(np.isnan(p_opt)):
            q_res = brc_prob.GetQ(p_opt)
            brc_prob.plant.SetPositions(brc_prob.plant_context, brc_prob.arm_instance, q_res)
            brc_prob.diagram.ForcedPublish(brc_prob.diagram_context)
            time.sleep(1.0)

    return out



# ── Tuning ────────────────────────────────────────────────────────────────────

def run_tune(targets_tune, guesses_tune, base_options, rng, meshcat=None, no_obstacles=False):
    """
    Sweep epsilon × threshold and return the best (epsilon, threshold) pair
    by success rate of the boundary formulation.
    """
    print("\n" + "=" * 60)
    print("TUNING PHASE: sweeping epsilon × threshold")
    print(f"  targets={len(targets_tune)}  guesses_per_target={len(guesses_tune[0])}")
    print("=" * 60)

    best_rate = -1.0
    best_eps  = TUNE_EPSILONS[0]
    best_thr  = TUNE_THRESHOLDS[0]

    # base_options carries ift_strategy="unused" (each formulation picks its own); the
    # sweep is of the boundary formulation, so give it the boundary knobs explicitly.
    base_options = base_options.copy()
    base_options["ift_strategy"]    = base_options["boundary_strategy"]
    base_options["ift_damping_lam"] = base_options["boundary_lam"]

    header = f"{'epsilon':>10} {'threshold':>10} {'succ_rate':>12}"
    print(header)
    print("-" * len(header))

    for eps in TUNE_EPSILONS:
        for thr in TUNE_THRESHOLDS:
            n_ok = 0
            n_total = 0
            for (_, X_WM, q_gold), gs in zip(targets_tune, guesses_tune):
                diagram, _ = build_env(meshcat=meshcat, target_pose=X_WM, no_obstacles=no_obstacles)
                _, _, brc = make_problems(diagram, eps, thr)

                for (q_init, p_init) in gs:
                    try:
                        brc.q_branch_ref = q_init
                        out = solve_boundary(brc, X_WM, q_init, p_init, base_options,
                                             meshcat=meshcat)
                        n_ok += int(out["ok"])
                    except Exception:
                        pass
                    n_total += 1

            rate = n_ok / max(n_total, 1)
            print(f"{eps:>10.1e} {thr:>10.1f} {rate:>12.3f}")

            if rate > best_rate:
                best_rate = rate
                best_eps  = eps
                best_thr  = thr

    print(f"\n  Best: epsilon={best_eps:.1e}  threshold={best_thr:.1f}"
          f"  (success rate {best_rate:.3f})")
    return best_eps, best_thr


# ── Main benchmark ────────────────────────────────────────────────────────────

def build_summary(all_records, base_options, epsilon, threshold,
                  n_targets, n_guesses_per_target):
    """
    All statistics, from the records alone.  Kept separate from run_benchmark so a
    checkpoint file can be re-scored without re-solving, and so old logs can be fed
    through the current metric set.
    """
    summary = {
        "epsilon":   epsilon,
        "threshold": threshold,
        "length_scale": BoundaryReachConstraint.DEFAULT_LENGTH_SCALE,
        "n_targets": n_targets,
        "n_guesses_per_target": n_guesses_per_target,
        "n_total_runs": len(all_records),
    }
    if not all_records:
        return summary

    # ── Primary lens: single start ────────────────────────────────────────────
    for m in METHODS:
        summary[m] = compute_all_stats(all_records, m)

    summary["mutual_all_three"] = mutual_stats(all_records, METHODS)
    for a, b in (("old", "direct"), ("old", "boundary"), ("direct", "boundary")):
        summary[f"mutual_{a}_{b}"]   = mutual_stats(all_records, (a, b))
        summary[f"mcnemar_{a}_{b}"]  = paired_success_stats(all_records, a, b)

    # Cluster bootstrap: resample whole targets, since a target's guesses are correlated.
    summary["bootstrap"] = {
        m: {
            "success_rate": bootstrap_ci(all_records, f"{m}_ok",   stat="rate"),
            "cost_success": bootstrap_ci(all_records, f"{m}_cost", stat="mean"),
            "time_success": bootstrap_ci(all_records, f"{m}_time", stat="mean"),
        }
        for m in METHODS
    }

    # ── Diagnostics ───────────────────────────────────────────────────────────
    summary["cap_hits"] = cap_hit_stats(all_records, METHODS,
                                        base_options["max_wall_time"])
    summary["failure_reasons"] = failure_reason_histogram(all_records, METHODS)

    # ── Secondary lens: multi start ───────────────────────────────────────────
    summary["multistart"] = multistart_stats(all_records, METHODS)
    return summary


def run_benchmark(targets, guesses, base_options, epsilon, threshold, meshcat=None,
                  no_obstacles=False, checkpoint_path=None):
    """
    Full three-way benchmark.  Returns list of per-run records and summary.

    If checkpoint_path is given, the records so far are written there after every target.
    A 100x10 run is over half an hour; without this, an interruption loses everything.
    """
    all_records = []
    n_targets   = len(targets)

    direct_opts = base_options.copy()
    direct_opts["ift_strategy"]    = base_options["direct_strategy"]
    direct_opts["ift_damping_lam"] = base_options["direct_lam"]

    boundary_opts = base_options.copy()
    boundary_opts["ift_strategy"]    = base_options["boundary_strategy"]
    boundary_opts["ift_damping_lam"] = base_options["boundary_lam"]

    for t_idx, (t_name, X_WM, q_gold) in enumerate(targets):
        print(f"\n  Target {t_idx + 1}/{n_targets}: {t_name}")

        diagram, _ = build_env(meshcat=meshcat, target_pose=X_WM, no_obstacles=no_obstacles)
        old_prob, new_prob, brc_prob = make_problems(diagram, epsilon, threshold)

        for g_idx, (q_init, p_init) in enumerate(guesses[t_idx]):
            rec = {
                "target": t_name,
                "guess_idx": g_idx,
            }
            try:
                res_o = solve_old(old_prob, X_WM, q_init, base_options, meshcat=meshcat)
            except Exception as e:
                res_o = dict(FAILED_RUN)
                print(f"    [WARN] Old solver error on g{g_idx}: {e}")

            try:
                new_prob.q_branch_ref = q_init
                res_n = solve_new(new_prob, X_WM, q_init, p_init, direct_opts, meshcat=meshcat)
            except Exception as e:
                res_n = dict(FAILED_RUN)
                print(f"    [WARN] Direct solver error on g{g_idx}: {e}")

            try:
                brc_prob.q_branch_ref = q_init
                res_b = solve_boundary(brc_prob, X_WM, q_init, p_init, boundary_opts, meshcat=meshcat)
            except Exception as e:
                res_b = dict(FAILED_RUN)
                print(f"    [WARN] Boundary solver error on g{g_idx}: {e}")

            for m, out in zip(METHODS, (res_o, res_n, res_b)):
                for field, value in out.items():
                    rec[f"{m}_{field}"] = value
            all_records.append(rec)

        # Per-target mini-summary
        tgt_recs = [r for r in all_records if r["target"] == t_name]
        def sr(key): return np.mean([r[key] for r in tgt_recs])
        print(f"    Succ rates — Old: {sr('old_ok'):.2f}  "
              f"Direct: {sr('direct_ok'):.2f}  Boundary: {sr('boundary_ok'):.2f}")

        if checkpoint_path is not None:
            _write_checkpoint(checkpoint_path, all_records, t_idx + 1, n_targets)

    summary = build_summary(all_records, base_options, epsilon, threshold,
                            n_targets, len(guesses[0]) if guesses else 0)
    return all_records, summary


def _write_checkpoint(path, records, n_done, n_total):
    """Atomically overwrite the checkpoint: write a sibling temp file, then rename."""
    tmp = path + ".tmp"
    try:
        with open(tmp, "w") as f:
            json.dump({"targets_done": n_done, "targets_total": n_total,
                       "records": records}, f, default=str)
        os.replace(tmp, path)
    except Exception as e:  # a failed checkpoint must never kill an overnight run
        print(f"    [WARN] checkpoint write failed: {e}")


# ── Pretty print ──────────────────────────────────────────────────────────────

def _fmt(val, fmt=".4f"):
    return f"{val:{fmt}}" if val is not None else "N/A"


def print_summary(summary):
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print(f"  epsilon={summary['epsilon']:.1e}  threshold={summary['threshold']:.1f}")
    print(f"  targets={summary['n_targets']}  guesses/target={summary['n_guesses_per_target']}")
    print(f"  total runs={summary['n_total_runs']}")
    print("=" * 80)

    labels = ["old", "direct", "boundary"]
    col_w  = 22

    # Success rates
    print(f"\n{'Metric':<30}", end="")
    for lab in labels:
        print(f"{lab:>{col_w}}", end="")
    print()
    print("-" * (30 + col_w * len(labels)))

    def row(name, getter):
        print(f"{name:<30}", end="")
        for lab in labels:
            try:
                v = getter(summary[lab])
                print(f"{_fmt(v, '.4f'):>{col_w}}", end="")
            except Exception:
                print(f"{'N/A':>{col_w}}", end="")
        print()

    print("\n-- Single start (primary) --")
    row("Success rate",            lambda s: s["success_rate"])
    row("Cost (success) mean",     lambda s: s["cost_success"]["mean"])
    row("Cost (success) median",   lambda s: s["cost_success"]["median"])
    row("Cost (success) std",      lambda s: s["cost_success"]["std"])
    row("Time (success) mean [s]", lambda s: s["time_success"]["mean"])
    row("Time (success) median[s]",lambda s: s["time_success"]["median"])
    row("Time (success) p90 [s]",  lambda s: s["time_success"]["p90"])
    row("Time (success) p99 [s]",  lambda s: s["time_success"]["p99"])
    row("Time (all) mean [s]",     lambda s: s["time_all"]["mean"])
    row("Time (all) median [s]",   lambda s: s["time_all"]["median"])

    # Bootstrap intervals resample whole targets, so they respect the correlation among a
    # target's guesses; this is what says whether a cost gap is real or noise.
    boot = summary.get("bootstrap")
    if boot:
        print("\n-- 95% CI (cluster bootstrap over targets) --")
        for name, field in (("Success rate", "success_rate"),
                            ("Cost (success)", "cost_success"),
                            ("Time (success)", "time_success")):
            print(f"{name:<30}", end="")
            for lab in labels:
                ci = boot.get(lab, {}).get(field, {})
                txt = (f"[{_fmt(ci.get('lo'), '.3f')}, {_fmt(ci.get('hi'), '.3f')}]"
                       if ci.get("lo") is not None else "N/A")
                print(f"{txt:>{col_w}}", end="")
            print()

    print("\n-- Paired success (McNemar exact) --")
    for a, b in (("old", "direct"), ("old", "boundary"), ("direct", "boundary")):
        mc = summary.get(f"mcnemar_{a}_{b}")
        if not mc:
            continue
        print(f"  {a} vs {b}: {a}-only={mc[f'{a}_only']}  {b}-only={mc[f'{b}_only']}  "
              f"both={mc['both']}  neither={mc['neither']}  p={mc['mcnemar_exact_p']:.3g}")

    print("\n-- Mutual-success subsets (paired, same instances) --")
    for a, b in (("old", "direct"), ("old", "boundary"), ("direct", "boundary")):
        ms = summary.get(f"mutual_{a}_{b}", {})
        n_mut = ms.get("n_mutual", 0)
        print(f"  {a} vs {b}: n_mutual={n_mut}")
        if not n_mut:
            continue
        for lab in (a, b):
            s = ms.get(lab, {})
            print(f"    {lab:<9} cost mean={_fmt(s['cost']['mean'])} "
                  f"median={_fmt(s['cost']['median'])}   "
                  f"time mean={_fmt(s['time']['mean'])} "
                  f"median={_fmt(s['time']['median'])}")
        pr = ms.get("paired", {})
        if pr:
            print(f"    paired: {a} cheaper {pr[f'{a}_cheaper_frac']:.3f}, "
                  f"faster {pr[f'{a}_faster_frac']:.3f}, "
                  f"dominates {pr[f'{a}_dominates_frac']:.3f}   "
                  f"wilcoxon p cost={_fmt(pr['cost_wilcoxon_p'], '.3g')} "
                  f"time={_fmt(pr['time_wilcoxon_p'], '.3g')}")

    caps = summary.get("cap_hits")
    if caps:
        print(f"\n-- Time-limit hits (cap {caps['max_wall_time']:.1f}s) --")
        for lab in labels:
            c = caps.get(lab, {})
            print(f"  {lab:<9} over {caps['wall_frac']:.0%} of cap: "
                  f"{c.get('n_over_wall_frac', 0)} ({c.get('frac_over_wall', 0):.3f})   "
                  f"solver time limit: {c.get('n_solver_timeout', 0)}   "
                  f"iteration limit: {c.get('n_iteration_limit', 0)}")

    fails = summary.get("failure_reasons")
    if fails:
        print("\n-- Failure reasons --")
        for lab in labels:
            hist = fails.get(lab, {})
            body = "  ".join(f"{k}={v}" for k, v in sorted(hist.items())) or "none"
            print(f"  {lab:<9} {body}")

    ms = summary.get("multistart")
    if ms:
        print("\n-- Multi start (secondary): each target's guesses as a restart sequence --")
        def mrow(name, getter):
            print(f"{name:<30}", end="")
            for lab in labels:
                try:
                    print(f"{_fmt(getter(ms[lab]), '.4f'):>{col_w}}", end="")
                except Exception:
                    print(f"{'N/A':>{col_w}}", end="")
            print()
        mrow("Any-success rate",     lambda s: s["any_success_rate"])
        mrow("Best cost mean",       lambda s: s["best_cost"]["mean"])
        mrow("Best cost median",     lambda s: s["best_cost"]["median"])
        mrow("Median guesses to 1st",lambda s: s["median_guesses_to_first_success"])
        mrow("Time to 1st succ mean", lambda s: s["time_to_first_success"]["mean"])
        for b in ms.get("budgets", []):
            mrow(f"Solved within {b} guess(es)",
                 lambda s, b=b: s["solved_within"][str(b)])
        for pair, w in ms.get("best_cost_wins", {}).items():
            if w.get("n"):
                a, b = pair.split("_vs_")
                print(f"  best cost {pair}: n={w['n']}  {a}={w[f'{a}_wins']}  "
                      f"{b}={w[f'{b}_wins']}  ties={w['ties']}  "
                      f"wilcoxon p={_fmt(w['wilcoxon_p'], '.3g')}")
    print("=" * 80)


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser(
        description="Benchmark three UR5e IK formulations: old, direct, boundary."
    )
    ap.add_argument("--num-targets",    type=int,   default=100)
    ap.add_argument("--num-guesses",    type=int,   default=10)
    ap.add_argument("--max-wall-time",  type=float, default=10.0)
    ap.add_argument("--solver",         type=str,
                    choices=["SNOPT", "IPOPT", "NLOPT"], default="SNOPT",
                    help="Solver to use (default: SNOPT). Applied identically to all three "
                         "formulations, so the comparison stays fair.")
    ap.add_argument("--square-frobenius-norm", action="store_true",
                    help="Direct formulation: use ||.||_F^2 instead of ||.||_F for the FK "
                         "residual.  The un-squared norm has a kink exactly at the solution "
                         "(residual = 0), which is where the solver spends its time.")
    ap.add_argument("--cost-reference", type=str, choices=["nominal", "initial"],
                    default="nominal",
                    help="What the centering cost is measured against: 'nominal' is the "
                         "posture prior ||q - q_nominal||^2 (default, what every published "
                         "number uses), 'initial' is ||q - q_initial||^2, the re-planning "
                         "objective.  Applied identically to all three formulations.")
    ap.add_argument("--joint-cost-weights", type=float, nargs=6, default=None,
                    metavar="W",
                    help="Six per-joint weights for the centering cost (default: uniform). "
                         "E.g. --joint-cost-weights 4 4 2 1 1 1 makes the proximal joints "
                         "expensive relative to the wrist.")
    ap.add_argument("--manipulability-cost-weight", type=float, default=0.0,
                    help="Weight on a -log det(J J^T + eps I) objective (default: 0, off). "
                         "Uses the Boundary formulation's own constraint expression as a "
                         "cost, so all three formulations optimise the same measure.")
    ap.add_argument("--major-optimality-tol",  type=float, default=1e-6)
    ap.add_argument("--minor-optimality-tol",  type=float, default=1e-6)
    ap.add_argument("--major-feasibility-tol", type=float, default=1e-6)
    ap.add_argument("--tune",           action="store_true",
                    help="Sweep epsilon/threshold before main benchmark.")
    ap.add_argument("--tune-targets",   type=int,   default=5)
    ap.add_argument("--tune-guesses",   type=int,   default=10)
    ap.add_argument("--epsilon",        type=float, default=1e-6,
                    help="Gram-matrix regularisation, in m^2 (the Jacobian is length-scaled). "
                         "Derived in scripts/calibrate_boundary_threshold.py, not tuned.")
    ap.add_argument("--threshold",      type=float, default=10.0,
                    help="Boundary constraint threshold. Derived from the velocity-amplification "
                         "criterion sigma* = v_task / qdot_max (v_task = 0.1 m/s) via "
                         "scripts/calibrate_boundary_threshold.py, not tuned.")
    ap.add_argument("--v-task",         type=float, default=0.1,
                    help="Task speed the threshold was derived from; recorded for provenance.")
    ap.add_argument("--direct-strategy",   type=str,   default="residual",
                    choices=list(IftGradient.STRATEGIES),
                    help="Gradient approximation strategy for Direct formulation. Optimal: residual.")
    ap.add_argument("--direct-lam",    type=float, default=1e-5,
                    help="Damping lambda for Direct formulation. Optimal: 1e-5.")
    ap.add_argument("--boundary-strategy",   type=str,   default="residual",
                    choices=list(IftGradient.STRATEGIES),
                    help="Gradient approximation strategy for Boundary formulation. Optimal: residual.")
    ap.add_argument("--boundary-lam",    type=float, default=1e-3,
                    help="Damping lambda for Boundary formulation. Optimal: 1e-3.")
    ap.add_argument("--out",            type=str,   default=None)
    ap.add_argument("--seed",           type=int,   default=42)
    ap.add_argument("--visualize",      action="store_true",
                    help="Enable Meshcat visualization.")
    ap.add_argument("--no-obstacles",   action="store_true",
                    help="Run without shelves and extra mugs (uses ur5e_collision_simple.yaml)")
    return ap


def main():
    args = parse_args().parse_args()
    rng  = np.random.default_rng(args.seed)

    # Default output path
    ts       = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = args.out or os.path.join(
        repo_dir, "logs", f"boundary_benchmark_{ts}.json"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    base_options = {
        "solver":                       args.solver,
        "max_wall_time":                args.max_wall_time,
        "square_frobenius_norm":        args.square_frobenius_norm,
        "major_optimality_tol":         args.major_optimality_tol,
        "minor_optimality_tol":         args.minor_optimality_tol,
        "major_feasibility_tol":        args.major_feasibility_tol,
        "avoid_collisions":             True,
        "impose_joint_centering_cost":  True,
        "joint_centering_cost_multiplier": 1.0,
        "cost_reference":               args.cost_reference,
        "joint_cost_weights":           args.joint_cost_weights,
        "manipulability_cost_weight":   args.manipulability_cost_weight,
        "direct_strategy": args.direct_strategy,
        "direct_lam": args.direct_lam,
        "boundary_strategy": args.boundary_strategy,
        "boundary_lam": args.boundary_lam,
        "ift_strategy":                 "unused",
        "joint_limits":                 True,
        "mug_height":                   0.08,
        "minimum_distance":             0.001,
    }

    print("Building environment …")
    meshcat = None
    if args.visualize:
        from pydrake.geometry import StartMeshcat
        meshcat = StartMeshcat()
        
    diagram_ref, plant_ref = build_env(meshcat=meshcat, no_obstacles=args.no_obstacles)

    # We need a reference old_prob for target sampling (no EAIK needed)
    ref_urdf = os.path.join(
        RepoDir(),
        "models/universal_robots/ur_description/urdf/ur5e_drake_collision.urdf",
    )
    ref_old  = UrIKProblemOldFormulation(diagram_ref, ref_urdf)

    # ── Determine epsilon/threshold ───────────────────────────────────────────
    epsilon   = args.epsilon
    threshold = args.threshold
    fixed     = epsilon is not None and threshold is not None

    if not fixed or args.tune:
        # Generate a smaller set for tuning
        n_tune_tgt = args.tune_targets
        n_tune_gs  = args.tune_guesses
        print(f"\nGenerating {n_tune_tgt} tuning targets …")
        tune_targets = sample_targets(ref_old, n_tune_tgt, rng)
        tune_guesses = [
            sample_initial_guesses(ref_old, X_WM, n_tune_gs, rng,
                                   mug_height=base_options["mug_height"])
            for (_, X_WM, _) in tune_targets
        ]
        best_eps, best_thr = run_tune(
            tune_targets, tune_guesses, base_options, rng, meshcat=meshcat, no_obstacles=args.no_obstacles
        )
        if epsilon is None:
            epsilon = best_eps
        if threshold is None:
            threshold = best_thr
    else:
        print(f"Using fixed epsilon={epsilon:.1e}  threshold={threshold:.1f}")

    # ── Generate main benchmark targets / guesses ─────────────────────────────
    n_tgt = args.num_targets
    n_gs  = args.num_guesses
    print(f"\nGenerating {n_tgt} benchmark targets …")
    targets = sample_targets(ref_old, n_tgt, rng)
    print(f"\nGenerating {n_gs} initial guesses per target …")
    guesses = [
        sample_initial_guesses(ref_old, X_WM, n_gs, rng,
                               mug_height=base_options["mug_height"])
        for (_, X_WM, _) in targets
    ]

    # ── Run benchmark ─────────────────────────────────────────────────────────
    print(f"\n{'#' * 70}")
    print(f"RUNNING BENCHMARK  |  epsilon={epsilon:.1e}  threshold={threshold:.1f}")
    print(f"  targets={n_tgt}  guesses/target={n_gs}  "
          f"total={n_tgt * n_gs} solves × 3 formulations")
    print(f"{'#' * 70}\n")
    t_bench_start = time.time()

    checkpoint_path = out_path + ".partial.json"
    all_records, summary = run_benchmark(
        targets, guesses, base_options, epsilon, threshold, meshcat=meshcat,
        no_obstacles=args.no_obstacles, checkpoint_path=checkpoint_path,
    )

    bench_time = time.time() - t_bench_start
    summary["total_benchmark_wall_time_s"] = bench_time
    print(f"\nBenchmark completed in {bench_time:.1f}s")

    # ── Print and save ────────────────────────────────────────────────────────
    print_summary(summary)

    output = {
        "args": vars(args),
        "constraint_provenance": {
            "length_scale_m": BoundaryReachConstraint.DEFAULT_LENGTH_SCALE,
            "epsilon": epsilon,
            "threshold": threshold,
            "v_task_m_per_s": args.v_task,
            "qdot_max_rad_per_s": float(np.pi),
            "sigma_star": args.v_task / float(np.pi),
            "derivation": "scripts/calibrate_boundary_threshold.py",
        },
        "summary": summary,
        "records": all_records,
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {out_path}")

    # The full file supersedes the checkpoint.
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)


if __name__ == "__main__":
    main()
