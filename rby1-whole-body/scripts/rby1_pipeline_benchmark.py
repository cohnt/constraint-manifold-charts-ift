#!/usr/bin/env python
import argparse
import sys
import os
import time
import datetime
import traceback
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from pydrake.all import RigidTransform, RollPitchYaw
from rby1_planning import (
    make_default_rby1_infrastructure,
    unconstrained_plan,
    constrained_plan,
    Rby1ActiveJointLayout,
    HeldBox,
    SceneBox,
)

# Constants from the notebook
HAND   = "right"
SIZE   = (0.386, 0.264, 0.108)
OFFSET = (-0.193, 0, -0.166)   # see plan_grid: box sits on the ground
RPY    = (0.0, 0.0, 0.0)

# Measured 2026-08-10, same as plan_grid's copies.
TABLE_SIZE = [1.20, 0.60, 0.7334]
TABLE_XYZ  = [0.32, -0.79, 0.3667]
TABLE_RGBA = [0.82, 0.71, 0.55, 1.0]

def parse_args():
    parser = argparse.ArgumentParser(description="Run RBY1 planning pipeline benchmark.")
    parser.add_argument("--num-targets", type=int, default=5, help="Number of different targets to generate.")
    parser.add_argument("--seeds-per-target", type=int, default=3, help="Number of random seeds to try per target.")
    parser.add_argument("--randomize-poses", action="store_true", default=True, help="Randomize start and goal poses.")
    parser.add_argument("--no-randomize-poses", action="store_false", dest="randomize_poses")
    parser.add_argument("--log-file", type=str, default="scratch/benchmark_results.log", help="File to log results.")
    return parser.parse_args()

def save_plot(traj, pos_idxs, name):
    n_samples = 1000
    ts = np.linspace(traj.start_time(), traj.end_time(), n_samples)
    pairs = []
    for t in ts:
        v = traj.value(t).flatten()
        q = v[pos_idxs] if len(v) > len(pos_idxs) else v
        pairs.append((t, q))
    ts_plot = np.array([p[0] for p in pairs])
    qs_plot = np.array([p[1] for p in pairs])

    plt.figure(figsize=(20, 12))
    lines = plt.plot(ts_plot, qs_plot)
    if qs_plot.ndim == 2 and qs_plot.shape[1] == 23:
        labels = (["base_x", "base_y", "base_rz"]
                  + [f"torso_{i}" for i in range(6)]
                  + [f"right_arm_{i}" for i in range(7)]
                  + [f"left_arm_{i}" for i in range(7)])
        plt.legend(lines, labels, ncol=4, fontsize=9, loc="upper right")
    plt.title(name)
    plt.xlabel("t")
    plt.ylabel("joint angles (rad)")
    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), "..", "scratch", f"{name}_joint_angles.png")
    plt.savefig(out_path)
    plt.close()
    return out_path

def classify_error(err_str):
    err_str = err_str.lower()
    if "ik failed" in err_str or "ik back-computation" in err_str:
        return "IK Failure"
    if "is not collision-free" in err_str or "stable" in err_str:
        return "Validation Failure"
    if "birrt failed" in err_str:
        return "RRT Failure"
    if "maximum velocity/acceleration" in err_str or "toppra" in err_str:
        return "TOPPRA Failure"
    return "Unknown/Trajopt Failure"

def run_trial(trial_idx, target_idx, seed, target_dx, target_dy, target_dyaw, args, log_f):
    rng = np.random.default_rng(seed)
    
    # 1. Use the pre-generated pose offsets for this target
    dx, dy, dyaw = target_dx, target_dy, target_dyaw

    T_W_MidStart = RigidTransform(RollPitchYaw([0, 0, dyaw]), [0.50 + dx, dy, 0.22])
    right_target = T_W_MidStart @ RigidTransform(RollPitchYaw([0, 0, -np.pi/2]), [0, -0.193, 0])
    left_target  = T_W_MidStart @ RigidTransform(RollPitchYaw([0, 0, np.pi/2]), [0, 0.193, 0])

    # Re-create infrastructure for empty-handed reach
    box_x_W = right_target @ RigidTransform(RollPitchYaw(RPY), OFFSET)
    lx, ly, lz = SIZE
    t = 0.01
    box_color = [0.2, 0.6, 1.0, 0.4]

    def make_wall(name, size, local_offset):
        X_Box_Wall = RigidTransform(RollPitchYaw(0, 0, 0), local_offset)
        X_W_Wall = box_x_W @ X_Box_Wall
        return SceneBox(
            size=size,
            xyz=X_W_Wall.translation(),
            rpy=RollPitchYaw(X_W_Wall.rotation()).vector(),
            name=name,
            color=box_color
        )

    # Walls stand on the base slab, derived rather than hardcoded -- see
    # plan_grid.WALL_SPECS and _add_held_box.
    lz_wall = lz - t
    z_base = -lz/2 + t/2
    z_wall = z_base + t/2 + lz_wall/2
    base = make_wall("box_base", (lx, ly, t), [0, 0, z_base])
    wall1 = make_wall("box_w1", (t, ly, lz_wall), [lx/2 - t/2, 0, z_wall])
    wall2 = make_wall("box_w2", (t, ly, lz_wall), [-lx/2 + t/2, 0, z_wall])
    wall3 = make_wall("box_w3", (lx - 2*t, t, lz_wall), [0, ly/2 - t/2, z_wall])
    wall4 = make_wall("box_w4", (lx - 2*t, t, lz_wall), [0, -ly/2 + t/2, z_wall])

    held = HeldBox(
        hand=HAND, size=SIZE, offset=OFFSET, rpy=RPY, visual=True,
        open_top=True, wall_thickness=t, wall_height=lz_wall
    )
    table = SceneBox(size=TABLE_SIZE, xyz=TABLE_XYZ, color=TABLE_RGBA, name="table")
    open_box_obstacles = [table, base, wall1, wall2, wall3, wall4]

    # Empty infrastructure
    coll_plant, collision_checker, diagram = make_default_rby1_infrastructure(
        None, obstacles=open_box_obstacles, open_grippers=True
    )
    layout = Rby1ActiveJointLayout(coll_plant)
    pos_idxs_23 = layout.plant_idxs

    q_full_start = np.concatenate([
        np.zeros(3),
        np.zeros(6),
        np.deg2rad([0.0, -135.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        np.deg2rad([0.0,  135.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    ])

    results = {}

    log_f.write(f"--- Trial {trial_idx} (seed {seed}) ---\n")
    log_f.write(f"Pose offsets: dx={dx:.4f}, dy={dy:.4f}, dyaw={dyaw:.4f}\n")
    log_f.flush()

    # PHASE 1: Reach
    try:
        reach_timings = {}
        reach_trajectories = {}
        reach_retimed_traj, q_full_goal = unconstrained_plan(
            coll_plant, collision_checker, diagram,
            q_full_start, right_target, left_target,
            rng_seed=seed,
            timings=reach_timings,
            trajectories=reach_trajectories,
        )
        for stage, traj in reach_trajectories.items():
            save_plot(traj, pos_idxs_23, f"trial_{trial_idx}_reach_{stage}")
        results["reach"] = {"status": "success", "timings": reach_timings}
    except Exception as e:
        err_msg = str(e)
        err_type = classify_error(err_msg)
        log_f.write(f"Reach failed [{err_type}]: {err_msg}\n")
        log_f.write(traceback.format_exc() + "\n")
        results["reach"] = {"status": "failed", "type": err_type, "error": err_msg}
        return results

    # Infrastructure WITH box
    coll_plant_box, collision_checker_box, diagram_box = make_default_rby1_infrastructure(
        None, held_boxes=held, obstacles=table
    )
    layout_box = Rby1ActiveJointLayout(coll_plant_box)
    pos_idxs_box = layout_box.plant_idxs

    ctx_box = diagram_box.CreateDefaultContext()
    pctx_box = coll_plant_box.GetMyContextFromRoot(ctx_box)
    q_pick = coll_plant_box.GetPositions(pctx_box)
    q_pick[pos_idxs_box] = q_full_goal
    coll_plant_box.SetPositions(pctx_box, q_pick)

    ee_body = coll_plant_box.GetBodyByName(f"ee_{HAND}", coll_plant_box.GetModelInstanceByName(f"{HAND}_arm"))
    qobj = coll_plant_box.get_geometry_query_input_port().Eval(pctx_box)
    insp = qobj.inspector()
    for pp in qobj.ComputePointPairPenetration():
        bA = coll_plant_box.GetBodyFromFrameId(insp.GetFrameId(pp.id_A))
        bB = coll_plant_box.GetBodyFromFrameId(insp.GetFrameId(pp.id_B))
        if ee_body.index() in (bA.index(), bB.index()):
            other = bB if bA.index() == ee_body.index() else bA
            collision_checker_box.SetCollisionFilteredBetween(ee_body.index(), other.index(), True)

    # PHASE 2: Lift
    tabletop_z = TABLE_XYZ[2] + TABLE_SIZE[2] / 2.0
    T_W_mid_above = RigidTransform(RollPitchYaw([0.0, 0.0, 0.0]), [0.5, 0.0, tabletop_z + 0.20])
    try:
        lift_timings = {}
        lift_retimed_traj = constrained_plan(
            coll_plant_box, collision_checker_box, diagram_box,
            q_full_goal, T_W_mid_above,
            support_polygon_inset=0,
            rng_seed=seed,
            do_trajopt=True,
            ik_n_candidates=50,
            ik_n_tries=50,
            plot_name=f"trial_{trial_idx}_lift",
            timings=lift_timings,
        )
        results["lift"] = {"status": "success", "timings": lift_timings}
    except Exception as e:
        err_msg = str(e)
        err_type = classify_error(err_msg)
        log_f.write(f"Lift failed [{err_type}]: {err_msg}\n")
        log_f.write(traceback.format_exc() + "\n")
        results["lift"] = {"status": "failed", "type": err_type, "error": err_msg}
        return results

    # PHASE 3: Place
    PLACE_TARGET = RigidTransform(
        RollPitchYaw([0.0, 0.0, -np.pi/2 + dyaw]),
        [TABLE_XYZ[0] - 0.3 + dx,                      
         TABLE_XYZ[1] + TABLE_SIZE[1] / 2 - 0.10 + dy, 
         tabletop_z + 0.20],                      
    )
    q_place_start = lift_retimed_traj.value(lift_retimed_traj.end_time()).flatten()[pos_idxs_box]

    try:
        place_timings = {}
        place_retimed_traj = constrained_plan(
            coll_plant_box, collision_checker_box, diagram_box,
            q_place_start, PLACE_TARGET, support_polygon_inset=0,
            mid_orientation_margin=0.05,
            rng_seed=seed,
            do_trajopt=True,
            ik_n_candidates=50,
            ik_n_tries=50,
            plot_name=f"trial_{trial_idx}_place",
            timings=place_timings,
        )
        results["place"] = {"status": "success", "timings": place_timings}
    except Exception as e:
        err_msg = str(e)
        err_type = classify_error(err_msg)
        log_f.write(f"Place failed [{err_type}]: {err_msg}\n")
        log_f.write(traceback.format_exc() + "\n")
        results["place"] = {"status": "failed", "type": err_type, "error": err_msg}
        return results

    return results

def main():
    args = parse_args()
    
    os.makedirs("scratch", exist_ok=True)
    
    summary = {
        "reach": {"success": 0, "failures": {}},
        "lift": {"success": 0, "failures": {}},
        "place": {"success": 0, "failures": {}},
    }

    print(f"Running {args.num_targets} targets, {args.seeds_per_target} seeds per target...")
    with open(args.log_file, "w") as log_f:
        log_f.write(f"Benchmark started at {datetime.datetime.now()}\n")
        
        trial_idx = 0
        target_rng = np.random.default_rng(42) # Fixed seed for target generation so runs are comparable
        
        for tgt_idx in range(args.num_targets):
            # Generate target pose offsets
            if args.randomize_poses:
                tgt_dx = target_rng.uniform(-0.05, 0.05)
                tgt_dy = target_rng.uniform(-0.05, 0.05)
                tgt_dyaw = target_rng.uniform(-np.deg2rad(15), np.deg2rad(15))
            else:
                tgt_dx, tgt_dy, tgt_dyaw = 0.0, 0.0, 0.0
                
            for s_idx in range(args.seeds_per_target):
                seed = 1000 + tgt_idx * args.seeds_per_target + s_idx
                print(f"Trial {trial_idx} (Target {tgt_idx}, Seed {seed})... ", end="", flush=True)
                res = run_trial(trial_idx, tgt_idx, seed, tgt_dx, tgt_dy, tgt_dyaw, args, log_f)
                
                trial_failed = False
                for phase in ["reach", "lift", "place"]:
                    if phase in res:
                        if res[phase]["status"] == "success":
                            summary[phase]["success"] += 1
                        else:
                            err_type = res[phase]["type"]
                            summary[phase]["failures"][err_type] = summary[phase]["failures"].get(err_type, 0) + 1
                            print(f"FAILED at {phase} ({err_type})")
                            trial_failed = True
                            break
                if not trial_failed:
                    print("SUCCESS")
                    log_f.write(f"Trial {trial_idx} SUCCESS\n")
                log_f.write("-" * 40 + "\n")
                trial_idx += 1

        # Write summary
        summary_str = "\n" + "="*40 + "\nBENCHMARK SUMMARY\n" + "="*40 + "\n"
        for phase in ["reach", "lift", "place"]:
            succ = summary[phase]["success"]
            total_reached = succ + sum(summary[phase]["failures"].values())
            summary_str += f"\n--- {phase.upper()} PHASE ---\n"
            summary_str += f"Success: {succ} / {total_reached}\n"
            for etype, cnt in summary[phase]["failures"].items():
                summary_str += f"  - {etype}: {cnt}\n"
        
        print(summary_str)
        log_f.write(summary_str)

if __name__ == "__main__":
    main()
