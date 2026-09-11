"""Plan reach -> grasp -> lift -> place -> release -> home for each point of
the box-placement grid (``data/box_placement_grid.npz``),
and preview in meshcat.

Planning only: this release does not include the robot client, so plans are
computed, verified and cached but never streamed to hardware.

Per grid point, six legs:

  reach_approach  ready pose -> a standoff STANDOFF_M above the grasp   (trajopt)
  reach_descend   standoff -> grasp                    (straight line or BiRRT)
  lift            grasp -> fixed raise pose over the workspace   (constrained)
  place           -> fixed place pose over the table            (constrained)
  home_retreat    release pose -> standoff above it    (straight line or BiRRT)
  home            standoff -> ready pose                              (trajopt)

The reach and home are split at a standoff rather than run straight to and from
the grasp because a trajopt leg whose endpoint sits ~15 mm from the box is
squeezed against its own minimum-distance floor, which makes the problem
infeasible exactly at the endpoint and leaves the solver returning an
"infeasibilities minimized" iterate. The two short legs adjacent to the box skip
trajopt entirely -- they are short enough that BiRRT + shortcutting produces a
clean path in 0.1-0.3 s -- so trajopt handles only the open middle of the motion.

Every leg's returned trajectory is then densely sampled and checked for the three
guarantees (collision-free, CoM inside the support polygon, joint limits) before
it can be cached; the resulting numbers ride along in the cache and in
status.json, so a cached plan can be audited without replanning it. A leg that
fails the check fails the point -- previously the only gate was whether the
trajopt and TOPPRA *stages had run*, which let colliding plans cache as
successes.

``--mark-corners`` runs a separate, lighter pass instead: just the reach leg
(no grasp/lift/home box logic — descend to the grasp pose and return, box
never leaves the floor) at the grid's 4 corner indices, so you can physically
mark/verify the grid's boundary with the robot before laying out all 20 box
positions, and interpolate the interior points from those two known corners
and the grid's fixed pitch.

Planning for each grid point runs in a forked child process : every planning
call in this pipeline leaks ~0.5-1.5 GB of unreclaimable pydrake memory per
trial, so a 20-point loop planned in-process risks the exact crash that leak
was root-caused from earlier. Meshcat preview stays in the parent process
across the whole run.

A successfully planned point is cached (``plan_io`` v2) so re-running the script doesn't replan
already-computed points. Failed points are not cached — retrying a failure on
the next run is cheap, and this session's own experience is that failures can
be transient (a `worker_died` grid-feasibility rejection succeeded cleanly on
retry). Every planning attempt (hit, fresh success, or failure) updates a
persistent status manifest (``<cache-dir>/status.json``) and redraws
``<cache-dir>/grid_status.png`` — a durable, always-current record of which
indices are known-good vs known-bad, for exactly the case where you're moving
the physical box between grid points and need to know which ones to skip
without having to watch the console the whole time.

Run (dry, no robot, plans/previews only):
    .venv/bin/python scripts/plan_grid.py --precompute

Run a single point against the robot:
    .venv/bin/python scripts/plan_grid.py --indices 0 --replan

Mark the grid's 4 corners (reach-only, no grasp) before laying out the grid:
    .venv/bin/python scripts/plan_grid.py --mark-corners

Just check current status (no planning, no meshcat, no robot):
    .venv/bin/python scripts/plan_grid.py --status
"""

import argparse
import atexit
import contextlib
import datetime
import inspect
import io
import json
import multiprocessing as mp
import os
import pickle
import resource
import sys
import time
import traceback
from itertools import product as iproduct

import numpy as np
from pydrake.all import RigidTransform, RollPitchYaw, StartMeshcat

import common
from exps.timing_utils import (
    emit_meta,
    mark,
    mark_discarded,
    record,
    start_recording,
    stop_recording,
)
from rby1_planning import (
    HeldBox,
    Rby1ActiveJointLayout,
    SceneBox,
    TrajoptRequired,
    constrained_plan,
    grasp_and_standoff,
    grasp_posture_score,
    make_default_rby1_infrastructure,
    plan_to_config,
    standoff_above,
    unconstrained_plan,
    verify_trajectory,
)

from plan_format.plan_io import (
    gripper_step,
    load_plan,
    save_plan,
    trajectory_step,
)
from plan_format.trajectory_conversion import (
    pose_legs_start, preview_legs, sample_leg, sample_trajectory,
)

# --- Grasped box (same as box_reachability_sampling.ipynb / box_pickup_loop.ipynb) ---
HAND   = "right"                # arm the HeldBox rides on while carried
SIZE   = (0.386, 0.264, 0.108)  # (lx, ly, lz) full box dimensions [m]
# OFFSET z is pinned by two facts, not tuned: the pickup pose at GRASP_Z is known
# to work on the robot, and the box sits flat on the ground. Together those force
# the box centre to lz/2, i.e. OFFSET_z = lz/2 - GRASP_Z = -0.166. The previous
# -0.13 left the box floating 36 mm above the floor, which (once the box was
# corrected to its true 108 mm) drove the gripper's ee_body collision spheres
# 4 mm into the very wall they grasp and made all 20 grid points unplannable.
OFFSET = (-0.193, 0, -0.166)    # box centre in the ee frame; -z points past the fingers
RPY    = (0.0, 0.0, 0.0)        # box orientation in the ee frame [rad]
WALL_T      = 0.01              # open-box wall thickness [m]
# lz and WALL_HEIGHT are the physical box measured 2026-08-10: 4.25 in = 108 mm
# overall, as a 10 mm base slab plus 98 mm walls. The previous 0.11/0.08 pair
# built a 90 mm box (see _add_held_box), 18 mm short of the real one.
WALL_HEIGHT = 0.098             # open-box wall height [m]

# --- Grasp geometry -----------------------------------------------------------
GRASP_HALF_SEP = 0.193          # gripper offset from box centre along world y [m]
GRASP_Z        = 0.22           # gripper height at the grasp [m]

# --- Static table + fixed mid-gripper raise target (same as rby1_model_test) ---
# Height measured 2026-08-10: 28.875 in = 733.4 mm. The previous 0.72 put the
# modelled tabletop 13.4 mm BELOW the real one, so every clearance over the
# table was optimistic by that much.
TABLE_SIZE = [1.20, 0.60, 0.7334]
TABLE_XYZ  = [0.32, -0.79, 0.3667]
TABLE = SceneBox(size=TABLE_SIZE, xyz=TABLE_XYZ,
                 color=[0.82, 0.71, 0.55, 1.0], name="table")
TABLETOP_Z = TABLE_XYZ[2] + TABLE_SIZE[2] / 2.0

# How far the carried box's underside hangs below the mid-gripper frame. Both
# hover heights below are derived from this rather than tuned, because a tuned
# gripper height silently stops meaning what it says the moment the box geometry
# changes -- which is exactly what happened. `TABLETOP_Z + 0.20` was chosen when
# the box hung 0.184 below the frame, putting its underside 16 mm ABOVE the
# tabletop; with the corrected 0.220 hang and the 13.4 mm taller table the same
# expression puts it 20 mm BELOW the tabletop, so the place leg had to drag the
# box up across the table's height band and grazed its near edge doing so
# (measured: box base corner vs table top edge, 0.255 mm, point 3).
BOX_BOTTOM_BELOW_EE = -(OFFSET[2] - SIZE[2] / 2.0)      # 0.220 m
CARRY_CLEARANCE = 0.05          # box underside above the tabletop while carried [m]
# Strictly greater than BOX_TABLE_MIN_DISTANCE: the place target is a *pinned*
# endpoint of the place leg, so if it sits exactly on the box<->table bound the
# program is infeasible there before the solver takes a step (measured: all 4
# trial points failed with lift_pair_min_dist violated near s=1).
PLACE_CLEARANCE = 0.08          # box underside above the tabletop at the place hover [m]
T_W_MID_ABOVE = RigidTransform(
    [0.5, 0.0, TABLETOP_Z + BOX_BOTTOM_BELOW_EE + CARRY_CLEARANCE])
# Fixed place target (same for every grid point, same as rby1_model_test.ipynb):
# mid-gripper pose hovering above the table -- touching it counts as a collision,
# so this hovers rather than rests. z is PLACE_CLEARANCE of *box underside* over
# the tabletop, not a tuned gripper height: the old TABLETOP_Z + 0.25 left the
# underside only 30 mm clear once the box's true hang was accounted for.
PLACE_TARGET = RigidTransform(
    RollPitchYaw([0.0, 0.0, -np.pi / 2]),
    [TABLE_XYZ[0] - 0.3, TABLE_XYZ[1] + TABLE_SIZE[1] / 2 - 0.10,
     TABLETOP_Z + BOX_BOTTOM_BELOW_EE + PLACE_CLEARANCE],
)

HELD = HeldBox(hand=HAND, size=SIZE, offset=OFFSET, rpy=RPY, visual=False,
               open_top=True, wall_thickness=WALL_T, wall_height=WALL_HEIGHT)

# Standoff height (m) for the split reach/home legs. The reach is planned to a
# configuration this far above the grasp with trajopt, then descends to the grasp
# with BiRRT + shortcutting only; the home leg mirrors it after the release.
# Keeping trajopt away from an endpoint that sits ~15 mm from the box is the point
# -- a trajopt leg ending there is squeezed against its own minimum-distance floor.
STANDOFF_M = 0.08

# Seed offsets tried per leg before giving up. BiRRT is randomised and the
# constrained legs in particular fail on some trees and succeed on others -- the
# first full grid run lost point 4 at 'place' to a single unlucky draw. Re-seeding
# costs nothing on legs that succeed first time.
LEG_SEEDS = (0, 1, 2)

# Legs planned in the 14-D constrained state space. A failure in one of these is a
# failure of the constrained manifold the grasp's GCP branch selected, so it is the
# only kind of failure worth retrying in a different branch.
CONSTRAINED_LEG_NAMES = ("lift", "place")

# Clearance headroom for the two 23-D trajopt legs, and why they need any.
#
# trajopt constrains its B-spline at n_trajopt_constraint_points samples and says
# nothing about the curve between them, so the leg is only shippable if the floor it
# holds at those samples exceeds whatever it bulges between them. At the planner's
# 1 mm default it does not: over the previous full grid run, reach_approach's
# optimised curve was thrown away on **19 of 20 points** -- 15 dense-rejected for
# penetrating 2-16 mm between samples (always ee_finger_1/2 vs world), 4 for an
# outright solver failure -- and every one of those points shipped the raw
# BiRRT+shortcut path instead. That is the entire reason the reach looks
# piecewise-linear while `home`, whose solve does survive, comes out at a 0.3 deg
# maximum turn.
#
# Raising the floor alone is not enough, because _feasible_min_dist_bound clamps it
# down to what the guess achieves and the guess is a BiRRT path validated by a pure
# penetration test -- 7 of 20 reach paths cleared under 6 mm. Point 11 measures the
# distinction exactly: floor 15 mm with an unpadded RRT is still rejected (the floor
# clamps to 3.8 mm), while padding the RRT first takes it through.
#
# So the padding is what creates the headroom and the floor is what spends it.
# Measured per leg, baseline -> padded, on the four tightest reach points:
#
#   pt14   turn 83.7 deg, 16 kinks, 189 s  ->  0.3 deg, 0 kinks, 4 s
#   pt3    turn 23.7 deg,  4 kinks,   4 s  ->  0.5 deg, 0 kinks, 5 s
#
# and the shipped trajectory ends up *further* from the obstacles, not closer
# (+1.5 mm at floor-only, +6 to +8 mm padded), because trajopt is now optimising
# inside an open corridor rather than along a grazing path.
UNCONSTRAINED_CLEARANCE_MARGIN = 0.025
UNCONSTRAINED_TRAJOPT_FLOOR = 0.020

# Same floor idea for the constrained legs, which had none: they ran at the
# planner's 1 mm default and duly came back at 1 mm, because
# MinimumDistanceLowerBoundConstraint enforces exactly its bound. The
# influence distance (min_distance_margin, 5 cm) only decides which pairs enter
# the softmin -- it buys no clearance, so nothing was ever pushing these legs
# further from the table than 1 mm.
#
# No RRT padding to go with it: clearance_margin pads *every* pair, and the two
# arms pass within 0.33 mm of each other by construction while holding a 386 mm
# box, so padding the constrained BiRRT makes it fail outright (measured, see
# clearance_margin's docstring). The floor alone is safe to raise because
# _feasible_min_dist_bound clamps it down to whatever the BiRRT's path already
# achieves rather than handing trajopt an infeasible guess -- so the worst case
# is that this has no effect on a leg whose guess grazes, not that the leg fails.
#
# 20 mm, not the 10 mm first tried. The bound is enforced at
# n_trajopt_constraint_points samples, so the B-spline is free to dip between
# them -- measured at 10 mm, point 3's place leg solved cleanly (SNOPT (1), not
# uncertified, floor never clamped) and still verified at 0.36 mm. Bounding the
# inter-point excursion is an open problem; raising the floor buys margin against
# it rather than eliminating it, so the verified clearance is what to read, never
# the floor.
CONSTRAINED_TRAJOPT_FLOOR = 0.020

# A hard minimum between the *carried box* and the *table* specifically, on top
# of the global floor above. The global floor cannot express this: it applies one
# scalar to every candidate pair and _feasible_min_dist_bound then clamps it down
# to whatever the guess's tightest pair allows -- which is arm-vs-arm at
# sub-millimetre by construction, so a global bound can never be raised far. This
# one sees only held-box-vs-table (5 geometries against 1), a pair the BiRRT is
# free to keep clear of, so it is asked for outright and never clamped.
BOX_TABLE_MIN_DISTANCE = 0.05

# The bound above is imposed at this many points, spaced uniformly in the
# B-spline's path parameter. Overridable from the CLI so the spacing can be
# swept: the place leg's tight stretch is short in path-parameter terms (TOPPRA
# slows down there, so it looks long in time-sampled data) and 50 points can
# step over it entirely -- point 3 satisfied a 20 mm bound at its constraint
# points and still verified at 0.364 mm. Measured on that point, everything else
# held fixed: 50 pts -> 0.364 mm, 100 pts -> 2.522 mm, which is exactly the
# enforced floor. The bound is only as good as the sampling that imposes it.
CONSTRAINED_CONSTRAINT_POINTS = int(os.environ.get("RBY1_CONSTR_PTS", 50))

# Parameter profiles tried, in order, for a leg that runs trajopt. The first profile
# that yields a leg passing _verify_leg wins.
#
# The second profile is the planner's own defaults -- no padding, 1 mm floor -- and
# it is a safety net, not a preference: padding shrinks the free space the BiRRT
# searches, so a point that can only be reached through a gap narrower than
# UNCONSTRAINED_CLEARANCE_MARGIN would lose its reach leg entirely. Keeping the old
# settings as a fallback means the change cannot cost a point that plans today; the
# worst it can do is spend the padded profile's seeds first and then produce exactly
# what it produces now.
UNCONSTRAINED_LEG_PROFILES = (
    dict(clearance_margin=UNCONSTRAINED_CLEARANCE_MARGIN,
         trajopt_min_distance=UNCONSTRAINED_TRAJOPT_FLOOR),
    dict(),
)

# Deliberately no cost-function ladder for the constrained legs.
#
# There was one: on a lift+place failure it retried the pair with the joint-space energy
# cost turned off. It is removed because re-rolling the *objective* to dodge a solver
# failure is not a fix, it is a way of hiding one -- and it hid this one. If the BiRRT
# produced a path and the B-spline guess over it is feasible as a mathematical program,
# then the manifold is feasible by construction and a solve that fails there is a
# trajopt or solver defect. Changing the cost, the RRT seed, the grasp or the GCP branch
# all relabel that defect as search luck, and measurement then chases the wrong thing:
# grid point 18 was diagnosed four different wrong ways before the solver logs showed a
# feasible guess with a badly conditioned solve.
#
# The one ladder that stays is the guess *multiplicity* (GUESS_MULTIPLICITIES), because
# it does not change the problem's feasibility question -- the path is fixed and only the
# B-spline's fidelity to it varies, up to multiplicity 3 where the guess is the
# piecewise-linear RRT path itself.


# Whether the two short heuristic legs (reach_descend, home_retreat) run trajopt.
#
# They were left un-optimised because their far endpoint is the grasp / release pose,
# ~15 mm from the box, and trajopt there is squeezed against its own distance floor.
# Two things have changed since: _feasible_min_dist_bound lowers the floor to what the
# guess actually achieves, so the solve no longer starts infeasible, and these legs
# permit the BiRRT fallback, so a failed solve costs nothing but time.
#
# The reason to want it is TOPPRA. After fixing _path_to_composite_traj to stop pinning
# every waypoint to zero velocity, these two legs still showed a median 2.5 and 3.5
# interior velocity dips because TOPPRA is time-optimal subject to acceleration limits
# and must slow at high path curvature -- and an un-optimised BiRRT path through a
# 0.24-1.13 rad descent still turns 47-61 deg at its corners. Optimising the geometry
# is the only thing that removes those; every leg that runs trajopt has zero dips.
#
# Measured on points 0 and 5: trajopt survives the dense check on both legs on both
# points with no rejection and no fallback, the dips go to 0, and the clearance is
# *unchanged* at exactly 15.00 mm (ee_body vs world) -- that number is the standoff
# geometry, not something the optimiser was holding back. The legs also get much
# shorter: reach_descend 6.07 -> 3.20 s and home_retreat 5.57 -> 2.77 s on point 0.
SHORT_LEG_TRAJOPT = True

# The short legs never take the padded profile even when they do run trajopt -- see
# SHORT_LEG_TRAJOPT for why 25 mm of padding is not available to them.
SHORT_LEG_PROFILES = ({},)

# These same two legs pass try_straight_line=True to plan_to_config, so the C-space
# straight line between their endpoints is tried before any BiRRT is built. They are
# the only legs where that is worth attempting: both are short (0.24-1.13 rad) and run
# between a standoff and the pose directly below or above it, so the line is usually
# clear, whereas reach_approach and home have to travel across the workspace. Trajopt
# still runs on whichever path is found, so the "fully optimised" gate is unaffected.
# See _straight_line_path for why the line needs no search when it is valid: it is
# already the shortest path under the joint-space length trajopt minimises.


def _profile_label(profile):
    """Short human-readable name for a profile, for the retry diagnostics."""
    if not profile:
        return "planner defaults"
    return ", ".join(f"{k}={v}" for k, v in sorted(profile.items()))

# Trajopt budget for the constrained legs, overriding constrained_plan's 30 s default.
#
# Measured over one full grid run: of 52 trajopt failures, **42 were SNOPT exit 34 (time
# limit) and 8 were exit 31 (iteration limit)**. Zero were infeasibility codes. And the
# BiRRT+shortcut paths those solves were given are not marginal -- the seeds trajopt gave
# up on were 100% valid at 32-62 mm clearance, while the trajectories the run *accepted*
# clear 0.114-15 mm. So the legs were being failed on a solver resource limit, not on
# anything about the problem, and the 30 s default is simply too small for a 14-D program
# with 200 PyFunctionConstraint bindings each running analytic IK.
#
# The alternative fix -- accept the BiRRT path when it densely verifies -- is exactly the
# fallback Harel ruled out for the load-carrying legs, so the budget is the lever.
CONSTRAINED_TRAJOPT_TIME_LIMIT = 180.0

# SNOPT/IPOPT optimality tolerance for the constrained legs, loosened from
# _lift_trajopt's 1e-2 default. Feasibility is enforced *after* the solve by a dense
# verify_trajectory check that the leg cannot pass without, so a looser optimality
# tolerance can only cost path quality, never validity -- and the failure mode it
# targets was literally the solver's clock (42 of 52 failures were SNOPT exit 34).
CONSTRAINED_TRAJOPT_OPTIMALITY_TOL = 1e-1

# Grasp candidates to draw when screening for a plannable lift, and the wall clock
# each probe gets. The probe distinguishes decisively: measured liftable grasps are
# found in 5.7-21.6 s while unliftable ones find nothing in 120 s, so 30 s separates
# them with margin while keeping a fully-screened-out point affordable.
GRASP_CANDIDATE_SEEDS = (0, 1, 2, 3, 4, 5)
GRASP_PROBE_TIMEOUT_S = 30.0
GRASP_PROBE_IK_CANDIDATES = 5

# Feasible grasp IK candidates to collect and rank by `grasp_posture_score`
# before the carry probe. Set above ik_n_tries_per_gcp (5) so the ranking uses
# every solve the existing try budget already allows -- the budget is never
# extended for ranking's sake. Cost: the 4 extra IPOPT solves after the first
# feasible one, ~1-4 s per point.
#
# Why ranking is worth that, re-measured 2026-08-07 under the exact GCP
# labelling (120 surveyed candidates, 6 per grid point): taking the first draw
# lands on a plannable grasp at 10/20 points, ranking by whole-arm asymmetry at
# 16/20, against an oracle of 20/20 -- and the ranked pick's downstream 23-D
# lift is shorter on average (3.18 vs 4.07 rad). See grasp_posture_score's
# docstring for the full comparison. (The previous justification here, a
# point-18 anecdote from the old labelling, was measured on a manifold that no
# longer exists.)
GRASP_IK_CANDIDATES = 6

# NOT DONE, but measured and worth knowing. P(probe passes | asymmetry score) is
# monotone and reaches zero: [0.0,0.5) 87%, [1.0,1.5) ~46%, [2.0,2.5) 20%,
# [2.5,3.0) 10%, >=3.0 0/6. So a cutoff that refuses to probe a high-scoring
# candidate would skip 31 of 120 probes at T=2.0 while still leaving every one
# of the 20 points a plannable candidate -- attractive, because rejected probes
# at 40-120 s each are exactly what the GCP fix made more common.
#
# It is not implemented because it makes the *ranking* a plannability gate, which
# is the failure mode recorded in _search_ik's docstring (the clearance-ranking
# experiment). Two specific reasons to distrust the number before acting on it:
# the survey draws one first-feasible candidate per seed, whereas each pipeline
# seed is already the ranked winner of up to GRASP_IK_CANDIDATES draws, so the
# score distributions differ; and "0 points lost" rests on 20 points, which is
# thin margin for a mechanism that can only ever lose points. Validate on a full
# grid before believing it.

# Bounds on what the ranking may spend AFTER the first feasible candidate: each
# extra try's IPOPT solve gets GRASP_RANK_WALL_S (a feasible grasp solves in
# ~0.5-2 s; only failures use the full 10 s default, and a failed ranking try
# teaches nothing worth 10 s), and no new try starts once GRASP_RANK_BUDGET_S
# has elapsed since that first success. The search up to the first candidate is
# untouched, so feasibility cannot regress. Unbudgeted, ranking cost point 0
# +19 s of standoff-ik; budgeted it is ~5 s worst case.
GRASP_RANK_WALL_S = 2.0
GRASP_RANK_BUDGET_S = 5.0

# Budgeting the constrained legs' goal IK was tried and REJECTED by
# measurement (2026-08-07, --jobs 1 bench on points 0/5/10/14/18). Goal
# *quality*, not IK time, is what the refinement solves buy, and every budget
# variant perturbed which goal wins on some point, cascading downstream:
#   - wall cap 2 s: point 18's place ik 20.0 -> 5.2 s (the intended win), but
#     point 0's changed goal cost +33 s of place trajopt + toppra (42 -> 75 s).
#   - window-only 10 s: point 0 restored exactly, but point 18's place seed-0
#     attempt now failed dense verify, the seed-1 goal moved q_place_end, and
#     the release standoff_above spent ~2 min on the new config (67 -> 204 s).
# The savings these budgets chased are delivered deterministically by the
# probe-reuse path instead (the goal IK runs once, in the probe, and the legs
# consume its s_goal). The constants stay as OFF switches for that plumbing.
LIFT_GOAL_IK_WALL_S = None
LIFT_GOAL_IK_BUDGET_S = None

# Wall-clock ceiling for one grid point, across every GCP branch it tries.
#
# The retry loops are nested and multiply: GRASP_CANDIDATE_SEEDS (6) x
# GCP_CANDIDATES (8) x LEG_SEEDS (3) x two solver rungs, each with a raised budget
# (120 s BiRRT, 180 s trajopt). Worst case per branch is ~180 s of grasp screening
# plus 3 x (120 + 180 + 180) for the lift and again for the place -- about 51 min --
# so a point no branch can plan would spend ~6.8 h before reporting failure. Every
# individual budget in that product is justified on its own; the product is not.
#
# Checked between GCP branches (the outermost and largest multiplier), so a branch
# already running is never interrupted mid-solve -- the bound is on how many further
# branches get started, not on the work in flight.
POINT_TIME_BUDGET_S = 2400.0

# Failure stages attributable to the grasp's GCP branch, and therefore worth
# retrying in a different one.
#
# 'lift'/'place' are the constrained legs, locked to the branch the grasp was solved
# in. 'standoff' belongs here too and is easy to miss: grasp_and_standoff forces the
# requested branch, so a branch that cannot reach this box at all fails there, before
# any leg is planned -- and treating that as non-retryable would silently truncate the
# sweep at the first unreachable branch rather than skipping it. 'pickup_collision' is
# likewise a property of the grasp posture the branch produced.
#
# Deliberately excluded: reach_*/home* (planned in joint space, no GCP lock) and the
# worker_* rejections (a dead child says nothing about the manifold).
GCP_RETRY_STAGES = ("standoff", "pickup_collision", "lift", "place")

# A dead or timed-out forked worker says nothing about the GCP branch -- but it says
# nothing about the *point* either, so it must not abort the sweep. It did: one point's
# branch-4 worker died and branches 5, 6 and 7 were consequently never tried, on two
# separate points.
WORKER_FAILURE_STAGES = ("worker_error", "worker_died", "worker_timeout")

# How far through plan_grid_point a failure stage represents, so a failed GCP sweep can
# report the branch that got furthest instead of the branch that happened to go first.
_STAGE_ORDER = ("worker_died", "worker_timeout", "worker_error", "standoff",
                "pickup_collision", "reach_approach", "reach_descend", "lift", "place",
                "home_standoff", "home_retreat", "home")


def _stage_progress(stage):
    try:
        return _STAGE_ORDER.index(stage)
    except ValueError:
        return -1                       # unknown/None stage: least informative

# Legs whose final trajectory may NOT be a raw BiRRT+shortcut path: a trajopt
# failure there fails the leg (after all LEG_SEEDS) instead of silently shipping the
# unoptimised path. "Failure" means *the trajectory is unusable* -- both an outright
# solver failure and a successful solve whose curve the dense check rejects, which
# are the same event as far as a reported result is concerned.
#
# Every leg, by default. This used to be ("lift", "place"), with reach_approach and
# home left out because reach_approach's trajopt output had been dense-rejected on
# both canary points and gating it "would turn two working points into failures".
# That reasoning was backwards: those points were not working, they were failing
# silently and shipping an unoptimised reach. A result that says "20/20 planned" has
# to mean 20/20 optimised, or the number is not reportable. If widening this turns
# points red, that is the measurement, and the fix is a better reach -- not a
# quieter one.
#
# reach_descend/home_retreat are planned with do_trajopt=False and never reach a
# fallback site, so their membership here is a no-op kept for completeness.
#
# --allow-trajopt-fallback opts back out, per leg or everywhere, for downstream use
# where a jerkier motion beats no motion. Those legs are then tagged
# trajopt_fell_back in the cached plan (see rby1_planning._tag_fallback) so the
# concession is countable from the artifact.
REQUIRE_TRAJOPT_LEGS = ("reach_approach", "reach_descend", "lift", "place",
                        "home_retreat", "home")

# Grasp GCP branches to try, in order. Index 1 first because it is what the
# pipeline was tuned against and every currently-planning point uses it; the rest
# follow so a point whose lift is infeasible in branch 1 still gets asked whether
# another branch works. Branches with no valid grasp IK cost only the IK stage.
#
# Measured motivation: 14 and 16 fail our lift in branch 1, yet a collaborator's
# branch -- same index, different GCP labelling, hence a different physical branch
# -- produces lift and place legs for both that pass our own dense check with the
# grippers and head in the model. So those two points were never infeasible; the
# search was locked to the one branch that could not reach them.
#
# Amendment: the labelling has since been corrected to an exact bijection with
# IKFast's branches (see rby1_analytic_ik.gcp_key). Two consequences for this
# list. First, index 1 still means what it always meant -- all 20 shipped grasps
# were verified to be genuinely (1,1,-1) = Wp_Ep_Sm under both labellings -- so
# leading with it is unchanged. Second, the other seven entries now name seven
# *distinct* branches. Under the old labelling the elbow bit was constant across
# the whole reachable joint range, so indices differing from each other only in
# that bit were not separable and the sweep had fewer real candidates than it
# appeared to.
GCP_CANDIDATES = (1, 0, 2, 3, 4, 5, 6, 7)

# Ready ("home") pose from RainbowRobotics/rby1-sdk 09_demo_motion.py
Q_READY = np.concatenate([
    np.zeros(3),                                          # base
    np.zeros(6),                                          # torso
    np.deg2rad([0.0, -135.0, 0.0, 0.0, 0.0, 0.0, 0.0]),   # right arm
    np.deg2rad([0.0,  135.0, 0.0, 0.0, 0.0, 0.0, 0.0]),   # left arm
])

_lx, _ly, _lz = SIZE
# Must stay the same assembly as rby1_planning._add_held_box's open_top branch:
# this is the box as a *scene obstacle* (on the floor before the grasp, and at
# the placed pose), that one is the same box *carried*. The wall z was a
# hardcoded -0.005 in both, independent of lz; it is derived from the base slab
# here so the two stay consistent when SIZE changes.
_Z_BASE = -_lz / 2 + WALL_T / 2
_Z_WALL = _Z_BASE + WALL_T / 2 + WALL_HEIGHT / 2
WALL_SPECS = [
    ("base", (_lx, _ly, WALL_T),        [0, 0, _Z_BASE]),
    ("w1", (WALL_T, _ly, WALL_HEIGHT),  [_lx / 2 - WALL_T / 2, 0, _Z_WALL]),
    ("w2", (WALL_T, _ly, WALL_HEIGHT),  [-_lx / 2 + WALL_T / 2, 0, _Z_WALL]),
    ("w3", (_lx - 2 * WALL_T, WALL_T, WALL_HEIGHT), [0, _ly / 2 - WALL_T / 2, _Z_WALL]),
    ("w4", (_lx - 2 * WALL_T, WALL_T, WALL_HEIGHT), [0, -_ly / 2 + WALL_T / 2, _Z_WALL]),
]
BOX_COLOR = [0.2, 0.6, 1.0, 0.4]

DEFAULT_GRID_PATH = os.path.join(common.RepoDir(), "data", "box_placement_grid.npz")
DEFAULT_CACHE_DIR = os.path.join(common.RepoDir(), "plans", "grid_cache")
# Anchored at the repo root rather than the cwd, so a run launched from anywhere
# writes to the same tracked-but-ignored results/ folder the robot client
# uses (that one defaults to a relative "results", i.e. cwd-dependent).
DEFAULT_RESULTS_DIR = os.path.join(common.RepoDir(), "results")
STATUS_FILENAME = "status.json"
DIAGRAM_FILENAME = "grid_status.png"


# ── Scene helpers (verbatim conventions from box_pickup_loop.ipynb) ─────────

def grasp_targets_for_box(bx, by):
    """(right, left) gripper grasp poses for a box centred at (bx, by)."""
    right = RigidTransform(RollPitchYaw([0, 0, -np.pi / 2]),
                           [bx, by - GRASP_HALF_SEP, GRASP_Z])
    left = RigidTransform(RollPitchYaw([0, 0, np.pi / 2]),
                          [bx, by + GRASP_HALF_SEP, GRASP_Z])
    return right, left


def box_pose_for(bx, by):
    """World pose of the box when grasped at the nominal targets."""
    right_target, _ = grasp_targets_for_box(bx, by)
    return right_target @ RigidTransform(RollPitchYaw(RPY), OFFSET)


def open_box_walls(X_W_box, prefix="box"):
    """Open-top box (base + 4 walls) as SceneBox obstacles at a world pose."""
    walls = []
    for name, size, local in WALL_SPECS:
        X_W_wall = X_W_box @ RigidTransform(local)
        walls.append(SceneBox(size=size, xyz=X_W_wall.translation(),
                              rpy=RollPitchYaw(X_W_wall.rotation()).vector(),
                              name=f"{prefix}_{name}", color=BOX_COLOR))
    return walls


def _qfn(idxs):
    return lambda v: v[idxs]


def _verify_leg(traj, plant, checker, diagram, *, q_fn=None, inset=0.0,
                n_samples=300):
    """Check a leg's returned trajectory against the three guarantees.

    Replaces an earlier check that only asked whether the trajopt and TOPPRA
    *stages had run* and whether the duration was finite and positive. That let
    invalid plans cache as successes: a stage can run, report success, and still
    return a trajectory that penetrates an obstacle between the samples trajopt
    constrained. This samples the trajectory that is actually being cached.

    Returns ``(ok, detail, report)``.
    """
    report = verify_trajectory(
        traj, plant, checker, diagram, q_fn=q_fn, n_samples=n_samples,
        support_polygon_inset=inset, required_clearance=0.0,
    )
    return report.ok, ("" if report.ok else "; ".join(report.failures)), report


def _leg_evidence(report):
    """Per-leg guarantee numbers, as plain picklable data for the cache."""
    return dict(
        duration=float(report.duration),
        min_clearance_m=float(report.min_clearance),
        min_clearance_pair=report.min_clearance_pair,
        min_clearance_t=float(report.min_clearance_t),
        com_margin_m=float(report.worst_stability_margin),
        joint_limit_excess_rad=float(report.max_joint_limit_violation),
        ok=bool(report.ok),
    )


def _pose_params(target):
    """Decompose a RigidTransform target into a plain, readable dict for the
    debug record -- xyz translation + rpy, rather than the RigidTransform
    object itself (keeps the debug record simple plain-data, not a Drake
    math-object round-trip dependency)."""
    return dict(xyz=tuple(target.translation()),
               rpy=tuple(RollPitchYaw(target.rotation()).vector()))


def _capture_stage_trajectories(stage_trajs, hz):
    """Sample every intermediate trajectory captured via constrained_plan's/
    unconstrained_plan's/plan_to_config's ``trajectories=`` output param into
    plain, picklable ``(q, duration)`` arrays (see ``sample_trajectory`` --
    the RRT/shortcut/trajopt-stage trajectories are ``FunctionHandleTrajectory``-
    backed and don't survive a fork/pickle round-trip otherwise).

    A stage whose value is None ran but was not captured, which is what
    ``rby1_planning.CAPTURE_STAGE_TRAJECTORIES = False`` (the default) produces.
    **The key set is the part anything reads** -- grid_provenance_report.py asks
    only whether "trajopt" is among the keys, to tell an optimised leg from a
    raw BiRRT fallback -- so the report is identical either way and only the
    unread arrays disappear. Keep the key, drop the value.

    Keys present depend on what the planner populated: ``rrt_raw`` (path
    straight out of BiRRT, before shortcutting), ``rrt_shortcut`` (after
    shortcut.shortcut), ``trajopt`` (after SNOPT), ``toppra`` (after retiming
    -- the same trajectory the leg's final sample_leg entry already carries,
    included here too so all 4 stages sit side by side for comparison).

    Each stage keeps its own natural coordinate space rather than being
    projected into a shared layout: for the unconstrained legs (reach, home)
    that's the 23-D active-joint vector; for the constrained legs (lift,
    place) rrt_raw/rrt_shortcut/trajopt are the 14-D mid-frame planning state
    ``[mid_xyz(3), mid_rpy(3), torso(6), psi_right(1), psi_left(1)]`` while
    toppra is the full-plant joint vector -- because that's what each stage
    actually operates over, and forcing a common layout would just be wrong
    for whichever stages don't share one.
    """
    return {
        stage_name: (None if traj is None else
                     dict(zip(("q", "duration"),
                              sample_trajectory(traj, slice(None), hz=hz))))
        for stage_name, traj in stage_trajs.items()
    }


def _filter_grasp_ee_contacts(plant_box, checker_box, diagram_box, layout_box,
                              ee_body, q_grasp):
    """Filter exactly the ee<->box contacts present at ``q_grasp``, on the checker.

    Split out of plan_grid_point so the grasp-screening probe uses the identical
    model the real lift will use. Measured to filter 0 pairs in practice -- the
    model-level ``_filter_held_box_model`` inside make_default_rby1_infrastructure
    already covers the fingers-holding-the-wall pairs -- but it must be kept in step
    with the real leg either way, or the probe would screen against a different
    model than the one that has to succeed.

    Returns True if ``q_grasp`` is collision-free after filtering.
    """
    ctx = diagram_box.CreateDefaultContext()
    pctx = plant_box.GetMyContextFromRoot(ctx)
    q_full = plant_box.GetPositions(pctx)
    q_full[layout_box.plant_idxs] = q_grasp
    plant_box.SetPositions(pctx, q_full)
    qobj = plant_box.get_geometry_query_input_port().Eval(pctx)
    insp = qobj.inspector()
    for pp in qobj.ComputePointPairPenetration():
        bA = plant_box.GetBodyFromFrameId(insp.GetFrameId(pp.id_A))
        bB = plant_box.GetBodyFromFrameId(insp.GetFrameId(pp.id_B))
        if ee_body.index() in (bA.index(), bB.index()):
            other = bB if bA.index() == ee_body.index() else bA
            checker_box.SetCollisionFilteredBetween(ee_body.index(), other.index(), True)
    return bool(checker_box.CheckConfigCollisionFree(q_full))


def _traj_path_length(traj, idxs=None) -> float:
    """Polyline joint-space length of a trajectory, sampled at 200 uniform knots
    (dense enough for a diagnostic; the probes' paths have <40 waypoints)."""
    ts = np.linspace(traj.start_time(), traj.end_time(), 200)
    qs = np.array([traj.value(t).flatten() for t in ts])
    if idxs is not None:
        qs = qs[:, idxs]
    return float(np.linalg.norm(np.diff(qs, axis=0), axis=1).sum())


def _carry_is_plannable(plant_box, checker_box, diagram_box, layout_box, ee_body,
                       q_grasp, toppra, rng_seed=0,
                       lift_ik_candidates=None):
    """Probe: can the whole carry -- lift *and* place -- be planned from ``q_grasp``?
    And if so, hand the RRT-level work back so the real legs need not redo it.

    **Both legs, not just the lift.** Screening on the lift alone selected grasps that
    lift well and then cannot place: with lift-only screening, grid points 4 and 16 each
    had exactly the branches whose grasp passed the screen go on to reach the lift and
    then die at `place` -- point 4 in branch 1, point 16 in branches 1 and 5, every time
    with "constrained BiRRT failed to find a path" at place. The place leg is the harder
    of the two (0.77 m of travel plus a 90 degree rotation of a 386 mm box, ending
    ~30 mm above the table, against the lift's near-pure translation in free space), so a
    screen that ignores it is screening on the easy half.

    The place is probed from the lift's *end* configuration, which is exactly where the
    real place leg starts -- hence the lift probe runs with do_trajopt=False rather than
    rrt_only=True, so it returns a full-plant retimed trajectory whose final
    configuration can be handed on.

    The probe runs at the SAME fidelity as the real legs' first attempt: same
    ``rng_seed`` (this used to be hardcoded 0 -- correct only for the default
    --rng-seed), same goal-IK candidate count (this used to be 5 vs the legs'
    50, an inconsistency the old docstring here defended on cost grounds and
    which meant the probe's goal was not the leg's goal). Identical inputs make
    the probe's product exactly the seed-0 RRT stage, so the legs can consume
    it via ``constrained_plan(warm_start_...)`` instead of re-solving the same
    goal IK and re-running the same BiRRT -- the probe is no longer thrown
    away, which is what pays for its higher fidelity.

    Returns ``(ok, why_not, diag, payload)``. ``diag`` records screen evidence
    (path lengths, durations) for ``grasp_screen``; nothing gates on it.
    ``payload`` (None unless ok) carries plain, fork-picklable arrays:
    lift/place 14-D waypoint lists, their s_goals, and per-stage timings.
    """
    if lift_ik_candidates is None:
        lift_ik_candidates = GRASP_PROBE_IK_CANDIDATES
    diag = dict(lift_len23=None, lift_dur_s=None, place_len14=None)
    if not _filter_grasp_ee_contacts(plant_box, checker_box, diagram_box, layout_box,
                                     ee_body, q_grasp):
        return False, "grasp pose in collision with the attached box", diag, None
    lift_diag, place_diag = {}, {}
    tm_lift, tm_place = {}, {}
    try:
        lift_traj = constrained_plan(
            plant_box, checker_box, diagram_box, q_grasp, T_W_MID_ABOVE,
            support_polygon_inset=0, rng_seed=rng_seed, do_trajopt=False,
            mid_orientation_margin=0.05,
            ik_n_candidates=lift_ik_candidates,
            ik_n_tries=lift_ik_candidates,
            birrt_timeout=GRASP_PROBE_TIMEOUT_S,
            timings=tm_lift, diagnostics=lift_diag, **toppra)
    except Exception as e:
        return False, f"lift: {type(e).__name__}: {str(e)[:70]}", diag, None
    diag["lift_len23"] = round(_traj_path_length(lift_traj, layout_box.plant_idxs), 3)
    diag["lift_dur_s"] = round(lift_traj.end_time() - lift_traj.start_time(), 2)
    try:
        q_lift_end = lift_traj.value(
            lift_traj.end_time()).flatten()[layout_box.plant_idxs]
        place_traj = constrained_plan(
            plant_box, checker_box, diagram_box, q_lift_end, PLACE_TARGET,
            support_polygon_inset=0, rng_seed=rng_seed, rrt_only=True, do_trajopt=False,
            mid_orientation_margin=0.05,
            ik_n_candidates=lift_ik_candidates,
            ik_n_tries=lift_ik_candidates,
            birrt_timeout=GRASP_PROBE_TIMEOUT_S,
            timings=tm_place, diagnostics=place_diag, **toppra)
        diag["place_len14"] = round(_traj_path_length(place_traj), 3)
        payload = dict(
            lift_path=lift_diag.get("rrt_path"),
            s_goal_lift=lift_diag.get("s_goal"),
            place_path=place_diag.get("rrt_path"),
            s_goal_place=place_diag.get("s_goal"),
            # Nones stripped: grid_timing_report's Counter += crashes on None
            # (rrt_only legs record trajopt/toppra as None).
            tm_lift={k: v for k, v in tm_lift.items() if v is not None},
            tm_place={k: v for k, v in tm_place.items() if v is not None},
        )
        return True, "", diag, payload
    except Exception as e:
        return False, f"place: {type(e).__name__}: {str(e)[:70]}", diag, None


def _grasp_candidate_child(conn, seed, make_grasp, probe):
    """Fork target: build one grasp candidate and probe its lift. Never raises.

    Runs in a nested fork of the already-forked planning worker, which is why it takes
    callables rather than Drake objects -- under the "fork" start method the child
    inherits the parent's fully-built plant, checker and diagram copy-on-write, so
    nothing has to be (or can be) pickled across the boundary. Only plain data comes
    back: the verdict plus the two configuration vectors.

    The child's own stdout is discarded. Six children probing concurrently would
    otherwise interleave their progress bars into the same tee'd log and make it
    unreadable; the parent prints the results in candidate order instead.
    """
    out = dict(seed=int(seed), ok=False, why="", secs=0.0, q_standoff=None,
               q_grasp=None, score=None, n_ik_candidates=None, diag=None,
               payload=None)
    t0 = time.perf_counter()
    try:
        with contextlib.redirect_stdout(io.StringIO()), \
             contextlib.redirect_stderr(io.StringIO()), \
             record("grasp.candidate", f"seed+{seed}", seed=int(seed)):
            ik_info = {}
            with record("grasp.ik", f"seed+{seed}", seed=int(seed)):
                q_standoff, q_grasp = make_grasp(seed, ik_info)
                mark(n_found=ik_info.get("n_found"))
            with record("grasp.probe", f"seed+{seed}", seed=int(seed)):
                ok, why, diag, payload = probe(q_grasp)
                mark(plannable=bool(ok), why=why)
                if not ok:
                    mark_discarded()
        out.update(ok=bool(ok), why=why, diag=diag, payload=payload,
                   score=round(grasp_posture_score(q_grasp), 4),
                   n_ik_candidates=ik_info.get("n_found"),
                   q_standoff=np.asarray(q_standoff), q_grasp=np.asarray(q_grasp))
    except Exception as e:
        out["why"] = f"{type(e).__name__}: {str(e)[:100]}"
    out["secs"] = round(time.perf_counter() - t0, 2)
    try:
        conn.send(out)
    except Exception:
        pass
    conn.close()


def _screen_grasps_parallel(seeds, make_grasp, probe):
    """Probe several grasp candidates concurrently; return every verdict.

    The candidates are completely independent -- a different IK draw, a different
    manifold, a throwaway probe -- so screening them one at a time was leaving the
    machine idle. Measured on a hard point: five serial probes at 18-40 s each is
    ~2.5 min per GCP branch, and with eight branches to try that dominated the point's
    entire budget while 22 of 24 cores did nothing.

    Every candidate is still probed rather than stopping at the first success, because
    they run in parallel anyway and the full set is what tells us whether a point is
    liftable from most grasps or just one -- which is the difference between "unlucky
    draw" and "nearly infeasible". The caller picks the winner in seed order, so the
    choice is deterministic and identical to what the serial version would have made.
    """
    procs = []
    for s in seeds:
        parent_conn, child_conn = _MP.Pipe(duplex=False)
        p = _MP.Process(target=_grasp_candidate_child,
                        args=(child_conn, s, make_grasp, probe))
        p.start()
        child_conn.close()
        procs.append((s, p, parent_conn))

    results = []
    for s, p, conn in procs:
        # Generous per-child ceiling: the probe bounds its own BiRRT with
        # GRASP_PROBE_TIMEOUT_S and the grasp IK is bounded by its retry counts, so
        # this only catches a wedged child.
        budget = GRASP_PROBE_TIMEOUT_S * 4 + 120.0
        got = None
        if conn.poll(budget):
            try:
                got = conn.recv()
            except EOFError:
                got = None
        conn.close()
        p.join(timeout=10)
        if p.is_alive():
            p.kill()
            p.join()
        results.append(got or dict(seed=int(s), ok=False, secs=0.0,
                                   why="probe child produced no result",
                                   q_standoff=None, q_grasp=None,
                                   score=None, n_ik_candidates=None, diag=None,
                                   payload=None))
    return results


def corner_indices(grid_nx, grid_ny):
    """Indices of the 4 corners of the (grid_nx x grid_ny) grid.

    grid_points is built x-major, y-minor (index = ix * grid_ny + iy, see
    box_reachability_sampling.ipynb), so the corners are the first/last index
    of the first/last x-column.
    """
    return [0, grid_ny - 1, (grid_nx - 1) * grid_ny, grid_nx * grid_ny - 1]


# ── Per-point planners (pure planning; no meshcat, no robot) ────────────────

def plan_grid_point(bx, by, *, rng_seed=0, lift_ik_candidates=50, hz=20.0,
                    viz_hz=30.0, standoff=STANDOFF_M, gcp_index=1,
                    require_trajopt_legs=REQUIRE_TRAJOPT_LEGS,
                    leg_parallel=False):
    """Plan the six-leg pick-and-place for a box at (bx, by).

    reach_approach -> reach_descend -> grasp -> lift -> place -> release ->
    home_retreat -> home.

    The reach and home are each split at a standoff configuration ``standoff``
    metres above the grasp / release pose. The two short legs adjacent to the box
    (reach_descend, home_retreat) are planned with BiRRT + shortcutting only, no
    trajectory optimisation: trajopt's minimum-distance constraint is squeezed
    against its own floor at an endpoint that close to the box, and the leg is
    short enough that shortcutting alone produces a clean path (measured 0.1-0.3 s
    to solve). trajopt then handles only the open middle of the motion.

    Pure planning: builds no meshcat, talks to no robot. Everything returned is
    plain, picklable data -- safe to ship across the fork boundary in
    plan_isolated, or into a cache file.

    Returns a dict with keys: bx, by, success, stage (None or the failing leg
    name), detail, steps, legs, timings, debug, evidence. ``evidence`` carries the
    per-leg guarantee numbers from verify_trajectory (worst clearance and the pair
    that achieved it, worst CoM margin, joint-limit excess) so a cached plan can
    be audited without replanning it.
    """
    result = dict(bx=float(bx), by=float(by), success=False, stage=None, detail="",
                  steps=[], legs=[], timings={}, debug={}, evidence={},
                  gcp_index=int(gcp_index), grasp_screen=[], grasp_seed=None,
                  grasp_score=None, grasp_n_candidates=None, q_grasp=None,
                  trajopt={},
                  # Carried out so the cache can state the rates its arrays are on
                  # rather than leaving them to be inferred from array lengths.
                  hz=float(hz), viz_hz=float(viz_hz))

    def finish(**kw):
        result.update(kw)
        return result

    right_target, left_target = grasp_targets_for_box(bx, by)
    toppra = dict(toppra_acceleration_scale=0.1, toppra_velocity_scale=0.1)

    # --- Empty-handed infrastructure, with the open box as a static obstacle ---
    plant, checker, diagram = make_default_rby1_infrastructure(
        obstacles=[TABLE] + open_box_walls(box_pose_for(bx, by)), open_grippers=True)
    layout = Rby1ActiveJointLayout(plant)

    # --- Box-attached infrastructure, built here so the grasp can be screened ---
    # (Also used for the lift and place legs below; the model-level held-box filters
    #  are applied at construction and do not depend on the configuration.)
    plant_box, checker_box, diagram_box = make_default_rby1_infrastructure(
        held_boxes=HELD, obstacles=TABLE)
    layout_box = Rby1ActiveJointLayout(plant_box)
    ee_body = plant_box.GetBodyByName(
        f"ee_{HAND}", plant_box.GetModelInstanceByName(f"{HAND}_arm"))

    # Make the BiRRT respect the same box<->table bound trajopt is given, so the
    # path handed over already satisfies it. Without this the guess is free to
    # graze the table, trajopt starts outside its own feasible set, and the solve
    # returns an infeasibilities-minimised iterate -- the failure mode
    # _feasible_min_dist_bound exists to dodge for the *global* floor, which it
    # does by lowering the bound rather than by fixing the path.
    #
    # Padding is per body pair, and this pair is exactly the one we mean: in this
    # infrastructure `ee_{HAND}` carries only the five held-box geometries and the
    # world body carries only `table_collision` (verified -- there is no ground
    # plane here). So this pads box-against-table and nothing else, unlike
    # clearance_margin, whose uniform padding the constrained BiRRT cannot take.
    #
    # Note what this is and is not for. It constrains the *path* -- the BiRRT and
    # the shortcutter, which are free to route the box wherever they like between
    # the endpoints, and did route it 0.255 mm from the table's near edge. It is
    # NOT what makes the constrained legs' *endpoints* safe, and the goal IK does
    # not need it: that IK solves to a fixed target pose (T_W_MID_ABOVE,
    # PLACE_TARGET), the box is welded to the ee frame, so every IK solution for a
    # given target puts the box in exactly the same place. The box-vs-table
    # distance at an endpoint is therefore a property of the target, not of the
    # configuration the IK happens to return, and no bound applied during the IK
    # search could change it. The targets are derived from BOX_BOTTOM_BELOW_EE
    # plus an explicit clearance, so they satisfy this bound by construction --
    # PLACE_CLEARANCE 80 mm against BOX_TABLE_MIN_DISTANCE 50 mm. The padding does
    # still apply to the IK's validity check (it shares this checker), but there
    # it is inert: it can only ever pass or fail uniformly, decided entirely by
    # the target.
    #
    # That guarantee lives or dies with the targets being derived rather than
    # tuned. `TABLETOP_Z + 0.20` was a tuned gripper height that silently became
    # 20 mm *below* the tabletop when the box geometry was corrected, and nothing
    # caught it, because no constraint was ever checking what the target implied
    # for the box.
    checker_box.SetPaddingBetween(
        ee_body, plant_box.world_body(), BOX_TABLE_MIN_DISTANCE)

    # --- Grasp + standoff configs, SCREENED BY LIFT PLANNABILITY ---------------
    #
    # The grasp is not just a pose to reach -- constrained_plan locks the lift and
    # place to `gcp_of_arm(q_grasp)` and builds the 14-D manifold from the grasp's
    # own FK, so **the grasp configuration determines the entire constrained problem
    # the lift is then searched in.** Taking the first IK draw that happens to be
    # valid therefore gambles the whole point on an arbitrary choice.
    #
    # Measured: at grid points 14 and 16 our first-valid grasp yields a lift that
    # finds nothing in 120 s, while a collaborator's grasp -- *in the same GCP
    # branch*, (1,1,-1) = _all_gcps()[1], both arms matching -- lifts in 5.7 s and
    # 21.6 s respectively. So those points were never infeasible and it was never a
    # GCP-branch problem; it was this one unexamined choice.
    #
    # So the grasp is chosen in two stages, both selective:
    #   1. `grasp_and_standoff` collects up to GRASP_IK_CANDIDATES feasible IK
    #      solutions within its existing try budget and keeps the one with the
    #      lowest `grasp_posture_score` (whole-arm asymmetry -- see its docstring
    #      for the measured 9/10 pairwise evidence). Cost: ~1-4 s of extra IPOPT
    #      solves. This is what stops a point-18: the first-feasible draw there
    #      scored 3.22 and shipped a 150 s lift when ~6 s alternatives existed.
    #   2. The winner is probed for lift+place plannability exactly as before
    #      (the score ranks, the probe gates). If it fails, remaining seeds are
    #      probed in parallel and the best-scoring plannable one wins.
    # Most points are liftable from their ranked draw, and for those this costs
    # one ~5 s probe and forks nothing; only the points that actually need
    # screening pay for the fan-out.
    # Per candidate, not one shared dict: every seed's grasp_and_standoff used to
    # write into the same `timings["standoff"]`, so each candidate silently
    # overwrote the previous one's stage times and only the last survived.
    result["timings"]["standoff"] = {}
    q_standoff = q_grasp = None
    _so_reasons = []

    def _make_grasp(seed, ik_info=None):
        tm = result["timings"].setdefault(
            "standoff" if seed == GRASP_CANDIDATE_SEEDS[0] else f"standoff_seed{seed}",
            {})
        return grasp_and_standoff(
            plant, checker, diagram, right_target, left_target,
            standoff=standoff, rng_seed=rng_seed + seed,
            gcp_index=gcp_index, timings=tm,
            ik_n_candidates=GRASP_IK_CANDIDATES,
            score_fn=grasp_posture_score, info=ik_info,
            post_success_wall_time=GRASP_RANK_WALL_S,
            collect_budget_s=GRASP_RANK_BUDGET_S)

    def _probe(q):
        # Same rng_seed and goal-IK fidelity as the real legs' first attempt,
        # so a passing probe's RRT product is exactly the seed-0 stage the
        # legs would otherwise recompute -- see _carry_is_plannable.
        return _carry_is_plannable(plant_box, checker_box, diagram_box, layout_box,
                                  ee_body, q, toppra, rng_seed=rng_seed,
                                  lift_ik_candidates=lift_ik_candidates)

    def _record(seed, ok, secs, why, score=None, n_ik=None, diag=None):
        # Recorded per candidate, not just the winner: which candidates were rejected,
        # each one's posture score and probe path lengths, and how long each probe
        # took are the evidence that the selection is doing anything -- and the
        # dataset a future score improvement gets validated against.
        result["grasp_screen"].append(
            dict(seed=int(seed), plannable=bool(ok), probe_s=round(secs, 2), why=why,
                 score=score, n_ik_candidates=n_ik, **(diag or {})))
        print(f"      grasp probe seed+{seed}: "
              f"{'plannable' if ok else 'NOT plannable'} in {secs:.1f}s"
              f"{f' score={score:.3f}' if score is not None else ''}"
              f"{'' if ok else f' ({why})'}")
        if not ok:
            _so_reasons.append(f"grasp seed+{seed}: {why}")

    def unconstrained_leg(name, q_from, q_to, do_trajopt, pl, ck, dg, lay,
                          profiles=None, try_straight_line=False):
        """Plan, verify, and record one 23-D leg. Returns the trajectory or None.

        Retried across ``LEG_SEEDS``: BiRRT is randomised, so a leg that finds no
        path (or whose trajopt output fails the dense check) on one seed often
        succeeds on the next. Re-seeding costs one extra solve on the legs that
        need it and nothing on the legs that do not.
        """
        tmg = result["timings"][name] = {}
        reasons = []
        # Clearance profiles are per leg, not per do_trajopt: the short legs end ~15 mm
        # from the box, so padding them by 25 mm would make them unplannable however
        # they are optimised. Callers that want trajopt without the padding pass
        # ``profiles=({},)`` explicitly.
        if profiles is None:
            profiles = UNCONSTRAINED_LEG_PROFILES if do_trajopt else ({},)
        for attempt, (profile, seed) in enumerate(iproduct(profiles, LEG_SEEDS)):
            # Name the profile in every diagnostic: "padded" failing and "planner
            # defaults" failing mean different things (no room for the margin vs. no
            # path at all), and without the label the two are indistinguishable in a
            # cached log.
            tag = (f"seed+{seed}/"
                   f"{'padded' if profile else 'planner-defaults'}")
            stage_trajs = {}
            params = dict(rng_seed=rng_seed + seed, do_trajopt=do_trajopt,
                          allow_birrt_fallback=name not in require_trajopt_legs,
                          try_straight_line=try_straight_line,
                          **profile, **toppra)
            diag = {}
            # One record per ATTEMPT, not per leg: a leg that succeeds on its
            # third seed cost three plans, and the two that failed are exactly
            # the work the old per-leg timings folded away.
            with record("leg", name, attempt=attempt, seed=seed,
                        profile="padded" if profile else "planner-defaults",
                        do_trajopt=bool(do_trajopt)):
                try:
                    traj = plan_to_config(pl, ck, dg, q_from, q_to,
                                          trajectories=stage_trajs, timings=tmg,
                                          diagnostics=diag, **params)
                except Exception as e:
                    mark(outcome=f"{type(e).__name__}")
                    mark_discarded()
                    reasons.append(f"{tag}: {type(e).__name__}: {e}")
                    continue
                finally:
                    result["debug"][name] = dict(
                        params=params,
                        stages=_capture_stage_trajectories(stage_trajs, viz_hz))
                if diag.get("trajopt_info"):
                    result["trajopt"][name] = diag
                ok, detail, report = _verify_leg(traj, pl, ck, dg)
                result["evidence"][name] = _leg_evidence(report)
                mark(outcome="ok" if ok else "failed_verify")
                if not ok:
                    mark_discarded()
            if ok:
                if attempt:
                    print(f"      {name}: succeeded on {tag} "
                          f"(after {attempt} failed attempt(s))")
                return traj
            reasons.append(f"{tag}: {detail}")
        result.update(stage=name, detail=" | ".join(reasons[-3:]))
        return None

    def _reach_worker(conn, q_so, q_gr):
        """Fork target: plan reach_approach + reach_descend from (q_so, q_gr).

        Runs concurrently with the winner's carry probe (the reach legs depend
        only on the grasp pair, not on the probe verdict). Everything shipped
        back is plain data: the step/leg dicts are sampled arrays, and the
        per-leg timings/debug/trajopt/evidence slices are read back out of the
        child's copy-on-write ``result``. Stdout is captured and returned so
        the parent can print it at the canonical position in the log.
        """
        out = dict(ok=False, stage="reach_approach", detail="", steps=None,
                   legs=None, timings={}, debug={}, trajopt={}, evidence={},
                   log="")
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                ra = unconstrained_leg("reach_approach", Q_READY, q_so,
                                       True, plant, checker, diagram, layout)
                rd = None
                if ra is not None:
                    rd = unconstrained_leg("reach_descend", q_so, q_gr,
                                           SHORT_LEG_TRAJOPT, plant, checker,
                                           diagram, layout,
                                           profiles=SHORT_LEG_PROFILES,
                                           try_straight_line=True)
                if rd is not None:
                    out["ok"] = True
                    out["stage"] = None
                    out["steps"] = [
                        trajectory_step("reach_approach", ra, layout,
                                        q_fn=_qfn(layout.plant_idxs), hz=hz),
                        trajectory_step("reach_descend", rd, layout,
                                        q_fn=_qfn(layout.plant_idxs), hz=hz),
                    ]
                    out["legs"] = [
                        sample_leg("reach_approach", ra, layout.plant_idxs,
                                   False, hz=viz_hz),
                        sample_leg("reach_descend", rd, layout.plant_idxs,
                                   False, hz=viz_hz),
                    ]
                else:
                    out["stage"] = result.get("stage") or out["stage"]
                    out["detail"] = result.get("detail", "")
            for key in ("timings", "debug", "trajopt", "evidence"):
                for name in ("reach_approach", "reach_descend"):
                    if name in result[key]:
                        out[key][name] = result[key][name]
        except Exception as e:
            out["detail"] = out["detail"] or f"{type(e).__name__}: {str(e)[:150]}"
        out["log"] = buf.getvalue()
        try:
            conn.send(out)
        except Exception:
            pass
        conn.close()

    def _fork_reach(q_so, q_gr):
        parent_conn, child_conn = _MP.Pipe(duplex=False)
        p = _MP.Process(target=_reach_worker, args=(child_conn, q_so, q_gr))
        p.start()
        child_conn.close()
        return p, parent_conn

    def _join_reach(p, conn, budget=GRASP_PROBE_TIMEOUT_S * 4 + 240.0):
        got = None
        if conn.poll(budget):
            try:
                got = conn.recv()
            except EOFError:
                got = None
        conn.close()
        p.join(timeout=10)
        if p.is_alive():
            p.kill()
            p.join()
        return got or dict(ok=False, stage="reach_approach",
                           detail="reach child produced no result",
                           steps=None, legs=None, timings={}, debug={},
                           trajopt={}, evidence={}, log="")

    carry_payload = None
    reach_result = None
    reach_proc = None
    first = GRASP_CANDIDATE_SEEDS[0]
    t0 = time.perf_counter()
    _ik_info = {}
    try:
        with record("grasp.ik", f"seed+{first}", seed=int(first)):
            cand_standoff, cand_grasp = _make_grasp(first, _ik_info)
            mark(n_found=_ik_info.get("n_found"))
        if leg_parallel:
            # The reach legs need only (q_standoff, q_grasp); overlap them
            # with the ~15-40 s carry probe. If the probe rejects this grasp
            # the child's work is discarded -- wasted CPU, zero latency cost.
            reach_proc = _fork_reach(cand_standoff, cand_grasp)
        with record("grasp.probe", f"seed+{first}", seed=int(first)):
            ok, why, diag, payload = _probe(cand_grasp)
            mark(plannable=bool(ok), why=why)
            if not ok:
                mark_discarded()
        score = round(grasp_posture_score(cand_grasp), 4)
    except Exception as e:
        ok, why, cand_standoff, cand_grasp = False, f"{type(e).__name__}: {e}", None, None
        score, diag, payload = None, None, None
    _record(first, ok, time.perf_counter() - t0, why,
            score=score, n_ik=_ik_info.get("n_found"), diag=diag)
    if reach_proc is not None:
        # Join right after the probe verdict: the probe dominates the reach
        # legs (~30 s vs ~7 s), so this costs no median latency, and failure
        # ordering stays identical to the sequential code (a reach failure
        # surfaces before any carry leg is planned).
        if ok:
            # Blocked-on-child time, named: with the reach legs overlapping the
            # probe this is usually ~0, and when it is not, the reach child --
            # not the probe -- is what set the point's latency.
            with record("fork.wait", "reach"):
                reach_result = _join_reach(*reach_proc)
        else:
            with record("fork.kill", "reach"):
                mark_discarded()
                p, conn = reach_proc
                conn.close()
                p.kill()
                p.join()
        reach_proc = None
    if ok:
        q_standoff, q_grasp = cand_standoff, cand_grasp
        carry_payload = payload
        result["grasp_seed"] = int(first)
        result["grasp_score"] = score
        result["grasp_n_candidates"] = _ik_info.get("n_found")
    elif len(GRASP_CANDIDATE_SEEDS) > 1:
        rest = GRASP_CANDIDATE_SEEDS[1:]
        print(f"      grasp: candidate seed+{first} carry not plannable; probing "
              f"{len(rest)} more candidates in parallel")
        with record("grasp.screen", n_seeds=len(rest)):
            probed = _screen_grasps_parallel(rest, _make_grasp, _probe)
            mark(n_plannable=sum(1 for r in probed if r.get("ok")))
        for r in probed:
            _record(r["seed"], r["ok"], r["secs"], r["why"],
                    score=r.get("score"), n_ik=r.get("n_ik_candidates"),
                    diag=r.get("diag"))
        # Among plannable candidates, lowest posture score wins (seed breaks
        # ties, so the choice stays deterministic). The configurations come back
        # from the child that probed them, so the grasp committed to is exactly
        # the one that was screened -- re-deriving it here would repeat the
        # grasp IK for nothing.
        winners = [r for r in probed if r["ok"] and r["q_grasp"] is not None]
        if winners:
            best = min(winners, key=lambda r: (
                r["score"] if r.get("score") is not None else float("inf"),
                r["seed"]))
            q_standoff, q_grasp = best["q_standoff"], best["q_grasp"]
            carry_payload = best.get("payload")
            result["grasp_seed"] = int(best["seed"])
            result["grasp_score"] = best.get("score")
            result["grasp_n_candidates"] = best.get("n_ik_candidates")
            print(f"      grasp: using candidate seed+{best['seed']} "
                  f"(best posture score among plannable candidates)")
    if q_grasp is None:
        # Every candidate either had no valid grasp or no plannable lift. Report as
        # `standoff` so the GCP sweep treats it as branch-attributable.
        return finish(stage="standoff", detail=" | ".join(_so_reasons[-3:]))
    result["q_grasp"] = [round(float(v), 6) for v in np.asarray(q_grasp)]
    if carry_payload is not None:
        # The probe's own stage costs, as pseudo-legs: the RRT-level work the
        # real lift/place no longer redo lives here, so per-point stage sums
        # stay comparable across caches even though the legs record 0.0 for
        # the stages they reuse.
        result["timings"]["carry_probe_lift"] = carry_payload.get("tm_lift", {})
        result["timings"]["carry_probe_place"] = carry_payload.get("tm_place", {})

    # --- Legs 1+2: reach_approach + reach_descend ------------------------------
    # Either spliced in from the parallel reach child (forked alongside the
    # winner's carry probe -- see the grasp block above) or planned here,
    # sequentially, exactly as before.
    if reach_result is not None:
        for key in ("timings", "debug", "trajopt", "evidence"):
            result[key].update(reach_result[key])
        if reach_result["log"]:
            print(reach_result["log"], end="")
        if not reach_result["ok"]:
            return finish(stage=reach_result["stage"],
                          detail=reach_result["detail"])
        reach_steps = reach_result["steps"]
        reach_leg_samples = reach_result["legs"]
    else:
        reach_approach = unconstrained_leg("reach_approach", Q_READY, q_standoff,
                                           True, plant, checker, diagram, layout)
        if reach_approach is None:
            return result
        reach_descend = unconstrained_leg("reach_descend", q_standoff, q_grasp,
                                          SHORT_LEG_TRAJOPT, plant, checker, diagram,
                                          layout, profiles=SHORT_LEG_PROFILES,
                                          try_straight_line=True)
        if reach_descend is None:
            return result
        reach_steps = [
            trajectory_step("reach_approach", reach_approach, layout,
                            q_fn=_qfn(layout.plant_idxs), hz=hz),
            trajectory_step("reach_descend", reach_descend, layout,
                            q_fn=_qfn(layout.plant_idxs), hz=hz),
        ]
        reach_leg_samples = [
            sample_leg("reach_approach", reach_approach, layout.plant_idxs, False,
                       hz=viz_hz),
            sample_leg("reach_descend", reach_descend, layout.plant_idxs, False,
                       hz=viz_hz),
        ]

    # The box-attached model and its grasp-pose ee<->box filters were built and
    # applied during grasp screening above, against this exact q_grasp -- screening
    # against a different model than the one that has to succeed would be pointless.
    if not _filter_grasp_ee_contacts(plant_box, checker_box, diagram_box, layout_box,
                                     ee_body, q_grasp):
        return finish(stage="pickup_collision",
                      detail="grasp pose in collision with the attached box")

    def constrained_leg(name, q_from, mid_target, warm_start=None):
        """Plan, verify, and record one constrained (14-D) leg.

        Seeds are re-rolled for **one** reason: the constrained BiRRT found no path.
        That is a randomised search coming up empty, and a different tree is the
        honest response -- on the first grid run it cost point 4 at ``place`` with
        "constrained BiRRT failed to find a path", and re-seeding recovered it.

        A **trajopt** failure is not re-seeded. If the BiRRT produced a path and the
        guess over it is feasible as a mathematical program, the manifold is feasible
        by construction, so the failure belongs to the optimiser and a fresh tree
        cannot fix it -- it only converts a reproducible solver defect into an
        intermittent one. ``TrajoptRequired`` is raised for exactly this case and
        breaks the loop immediately, so the failure is reported where it happened.

        ``allow_birrt_fallback=False`` is the default policy for every leg, and these
        two are where it started: the lift and place are the long constrained motions
        with a 386 mm box between the grippers, and an unoptimised RRT path there is not
        something to cache as a success. Auditing an earlier cached run: 9 of 15
        "successful" points had a lift or place built from the fallback, and the
        pipeline reported all 15 identically. See
        the folder README's note on trajectory-optimization failures.
        """
        tmg = result["timings"][name] = {}
        reasons = []

        def _attempt(params, tag, warm=None):
            """One constrained_plan attempt + dense verify.

            Returns (traj_or_None, fatal): ``fatal`` is True only for a
            TrajoptRequired raised by a *plain* attempt -- for the warm-start
            attempt every failure, that one included, just falls through to
            the ordinary seed loop, so reuse can never fail a point today's
            code would pass (the loop then judges its own guesses exactly as
            before).
            """
            stage_trajs = {}
            diag = {}
            kw = dict(params)
            if warm is not None:
                kw.update(warm_start_path=warm["path"],
                          warm_start_s_goal=warm["s_goal"])
            # See the unconstrained legs: one record per attempt, so a re-seeded
            # leg's discarded attempts stay visible instead of being summed into
            # the attempt that happened to succeed.
            with record("leg", name, attempt=tag, warm_start=warm is not None):
                try:
                    traj = constrained_plan(plant_box, checker_box, diagram_box,
                                            q_from, mid_target,
                                            trajectories=stage_trajs, timings=tmg,
                                            diagnostics=diag, **kw)
                except TrajoptRequired as e:
                    # The search did its job and the optimiser did not. On a plain
                    # attempt, report it rather than spending the remaining seeds
                    # rediscovering the same solver behaviour on a different tree.
                    mark(outcome="TrajoptRequired")
                    mark_discarded()
                    reasons.append(f"{tag}: TrajoptRequired: {e}")
                    return None, warm is None
                except Exception as e:
                    mark(outcome=type(e).__name__)
                    mark_discarded()
                    reasons.append(f"{tag}: {type(e).__name__}: {e}")
                    return None, False
                finally:
                    result["debug"][name] = dict(
                        params=dict(kw, mid_target=_pose_params(mid_target),
                                    attempt=tag),
                        stages=_capture_stage_trajectories(stage_trajs, viz_hz))
                if diag.get("trajopt_info"):
                    result["trajopt"][name] = diag
                ok, detail, report = _verify_leg(traj, plant_box, checker_box,
                                                 diagram_box)
                result["evidence"][name] = _leg_evidence(report)
                mark(outcome="ok" if ok else "failed_verify")
                if not ok:
                    mark_discarded()
            if ok:
                return traj, False
            reasons.append(f"{tag}: {detail}")
            return None, False

        def _params(seed):
            return dict(support_polygon_inset=0, rng_seed=rng_seed + seed,
                        do_trajopt=True, mid_orientation_margin=0.05,
                        trajopt_min_distance=CONSTRAINED_TRAJOPT_FLOOR,
                        n_trajopt_constraint_points=CONSTRAINED_CONSTRAINT_POINTS,
                        trajopt_pair_min_distance=BOX_TABLE_MIN_DISTANCE,
                        allow_birrt_fallback=name not in require_trajopt_legs,
                        trajopt_time_limit=CONSTRAINED_TRAJOPT_TIME_LIMIT,
                        trajopt_optimality_tolerance=CONSTRAINED_TRAJOPT_OPTIMALITY_TOL,
                        ik_n_candidates=lift_ik_candidates,
                        ik_n_tries=lift_ik_candidates,
                        goal_ik_post_success_wall_time=LIFT_GOAL_IK_WALL_S,
                        goal_ik_collect_budget_s=LIFT_GOAL_IK_BUDGET_S,
                        **toppra)

        # Attempt "-1": reuse the carry probe's RRT-level product (same
        # rng_seed, same goal-IK fidelity, same manifold) so this leg goes
        # straight to trajopt. Purely additive: any failure -- including
        # WarmStartRejected's prechecks and TrajoptRequired -- falls through
        # to the ordinary seed loop below, which then behaves verbatim as it
        # did before reuse existed.
        if warm_start is not None and warm_start.get("path") is not None:
            traj, _ = _attempt(_params(LEG_SEEDS[0]), "probe-reuse", warm=warm_start)
            if traj is not None:
                return traj

        for attempt, seed in enumerate(LEG_SEEDS):
            traj, fatal = _attempt(_params(seed), f"seed+{seed}")
            if fatal:
                result.update(stage=name, detail=" | ".join(reasons[-3:]),
                              trajopt_failure=True)
                return None
            if traj is not None:
                if attempt or reasons:
                    print(f"      {name}: succeeded on seed+{seed} "
                          f"(after {len(reasons)} failed attempt(s))")
                return traj
        result.update(stage=name, detail=" | ".join(reasons[-3:]))
        return None

    # --- Leg 3: lift -- raise the box to the fixed pose over the workspace ------
    lift_traj = constrained_leg(
        "lift", q_grasp, T_W_MID_ABOVE,
        warm_start=None if carry_payload is None else dict(
            path=carry_payload.get("lift_path"),
            s_goal=carry_payload.get("s_goal_lift")))
    if lift_traj is None:
        return result
    q_lift_end = lift_traj.value(lift_traj.end_time()).flatten()[layout_box.plant_idxs]

    # --- Leg 4: place -- lower the box to the fixed place pose over the table ---
    #
    # place starts from wherever lift ended, so the two are coupled: q_lift_end is read
    # off the lift's finished trajectory. That coupling is worth knowing about but is
    # not something to retry around -- if place cannot be planned from a validly
    # optimised lift, that is a finding to report, not a reason to re-plan the lift
    # differently until it stops being a problem.
    #
    # A related gap, documented rather than papered over: the grasp screening that chose
    # q_grasp probes lift->place with do_trajopt=False, so it validates the RRT-level
    # chain and does not predict the trajopt-modified one it stands in for. That is why
    # a candidate can probe "plannable" and then fail for real.
    place_traj = constrained_leg(
        "place", q_lift_end, PLACE_TARGET,
        warm_start=None if carry_payload is None else dict(
            path=carry_payload.get("place_path"),
            s_goal=carry_payload.get("s_goal_place")))
    if place_traj is None:
        return result
    q_place_end = place_traj.value(place_traj.end_time()).flatten()[layout_box.plant_idxs]

    # --- Release: FK the box's placed pose, rebuild empty-handed infra with the
    #     placed box as an obstacle for the retreat.
    ctx_ret = diagram_box.CreateDefaultContext()
    pctx_ret = plant_box.GetMyContextFromRoot(ctx_ret)
    q_fk = plant_box.GetPositions(pctx_ret)
    q_fk[layout_box.plant_idxs] = q_place_end
    plant_box.SetPositions(pctx_ret, q_fk)
    box_x_W_placed = (plant_box.EvalBodyPoseInWorld(pctx_ret, ee_body)
                      @ RigidTransform(RollPitchYaw(RPY), OFFSET))
    plant_ret, checker_ret, diagram_ret = make_default_rby1_infrastructure(
        obstacles=[TABLE] + open_box_walls(box_x_W_placed, prefix="placed"),
        open_grippers=True)
    layout_ret = Rby1ActiveJointLayout(plant_ret)

    # --- Leg 5: home_retreat -- lift clear of the placed box (no trajopt) ------
    # Symmetric with reach_descend: the release pose sits as close to the box as
    # the grasp did, so a trajopt leg starting there is squeezed against its own
    # distance floor. A short BiRRT + shortcut lift covers it instead.
    tm = result["timings"]["home_standoff"] = {}
    try:
        # timings= was silently dropped before standoff_above grew the
        # parameter, leaving this stage invisible -- and it can be huge: one
        # measured run spent ~2 minutes here after a place-goal change moved
        # q_place_end into a hard corner (2026-08-07 goal-IK budget bench).
        q_release_standoff = standoff_above(
            plant_ret, checker_ret, diagram_ret, q_place_end,
            standoff=standoff, rng_seed=rng_seed, timings=tm)
    except Exception as e:
        return finish(stage="home_standoff", detail=f"{type(e).__name__}: {e}")

    home_retreat = unconstrained_leg("home_retreat", q_place_end, q_release_standoff,
                                     SHORT_LEG_TRAJOPT, plant_ret, checker_ret,
                                     diagram_ret, layout_ret,
                                     profiles=SHORT_LEG_PROFILES,
                                     try_straight_line=True)
    if home_retreat is None:
        return result

    # --- Leg 6: home -- empty-handed return to the ready pose (trajopt on) -----
    home_traj = unconstrained_leg("home", q_release_standoff, Q_READY,
                                  True, plant_ret, checker_ret, diagram_ret,
                                  layout_ret)
    if home_traj is None:
        return result

    # --- Assemble plan_io steps (robot-executable) + preview legs (meshcat) ---
    steps = [
        *reach_steps,
        gripper_step("close", name="grasp"),
        trajectory_step("lift", lift_traj, layout_box,
                        q_fn=_qfn(layout_box.plant_idxs), hz=hz),
        trajectory_step("place", place_traj, layout_box,
                        q_fn=_qfn(layout_box.plant_idxs), hz=hz),
        gripper_step("open", name="release"),
        trajectory_step("home_retreat", home_retreat, layout_ret,
                        q_fn=_qfn(layout_ret.plant_idxs), hz=hz),
        trajectory_step("home", home_traj, layout_ret,
                        q_fn=_qfn(layout_ret.plant_idxs), hz=hz),
    ]
    legs = [
        *reach_leg_samples,
        sample_leg("lift", lift_traj, layout_box.plant_idxs, True, hz=viz_hz),
        sample_leg("place", place_traj, layout_box.plant_idxs, True, hz=viz_hz),
        sample_leg("home_retreat", home_retreat, layout_ret.plant_idxs, False, hz=viz_hz),
        sample_leg("home", home_traj, layout_ret.plant_idxs, False, hz=viz_hz),
    ]
    # stage/detail are cleared explicitly: a leg records its failure into the result as
    # it happens, and since the lift+place pair is retried across profiles, a point can
    # now reach here having recorded an earlier profile's failure. Without this reset a
    # recovered point reports success=True alongside stage='place', which reads as a
    # contradiction and would mislead the GCP-retry bookkeeping.
    return finish(success=True, stage=None, detail="", steps=steps, legs=legs)


def plan_reach_only(bx, by, *, rng_seed=0, hz=20.0, viz_hz=30.0, **_ignored):
    """Plan reach -> home for a box at (bx, by), with no grasp/lift.

    For marking/verifying a box position (e.g. the grid's corner points) with
    the robot: descend to the grasp pose as if picking the box up, then return
    -- the box is never actually grasped, so it never moves, and both legs
    reuse the same empty-handed infrastructure (no need for the with-box
    plant, no release/FK'd-obstacle step). ``**_ignored`` absorbs kwargs (e.g.
    ``lift_ik_candidates``) shared with plan_grid_point's call sites but not
    applicable here.

    Returns a dict with keys: bx, by, success, stage (None | "reach" |
    "home"), detail, steps, legs, timings.
    """
    result = dict(bx=float(bx), by=float(by), success=False, stage=None, detail="",
                  steps=[], legs=[], timings={},
                  hz=float(hz), viz_hz=float(viz_hz))

    def finish(**kw):
        result.update(kw)
        return result

    right_target, left_target = grasp_targets_for_box(bx, by)
    plant, checker, diagram = make_default_rby1_infrastructure(
        obstacles=[TABLE] + open_box_walls(box_pose_for(bx, by)), open_grippers=True)
    layout = Rby1ActiveJointLayout(plant)

    tm = result["timings"]["reach"] = {}
    try:
        reach_traj, q_grasp = unconstrained_plan(
            plant, checker, diagram, Q_READY, right_target, left_target,
            rng_seed=rng_seed, do_trajopt=True, timings=tm,
            toppra_acceleration_scale=0.1, toppra_velocity_scale=0.1)
    except Exception as e:
        return finish(stage="reach", detail=f"{type(e).__name__}: {e}")
    ok, detail, _ = _verify_leg(reach_traj, plant, checker, diagram)
    if not ok:
        return finish(stage="reach", detail=detail)

    tm = result["timings"]["home"] = {}
    try:
        home_traj = plan_to_config(
            plant, checker, diagram, q_grasp, Q_READY,
            support_polygon_inset=0, rng_seed=rng_seed, do_trajopt=True,
            timings=tm, toppra_acceleration_scale=0.1, toppra_velocity_scale=0.1)
    except Exception as e:
        return finish(stage="home", detail=f"{type(e).__name__}: {e}")
    ok, detail, _ = _verify_leg(home_traj, plant, checker, diagram)
    if not ok:
        return finish(stage="home", detail=detail)

    steps = [
        trajectory_step("reach", reach_traj, layout, q_fn=_qfn(layout.plant_idxs), hz=hz),
        trajectory_step("home", home_traj, layout, q_fn=_qfn(layout.plant_idxs), hz=hz),
    ]
    legs = [
        sample_leg("reach", reach_traj, layout.plant_idxs, False, hz=viz_hz),
        sample_leg("home", home_traj, layout.plant_idxs, False, hz=viz_hz),
    ]
    return finish(success=True, steps=steps, legs=legs)


# ── Process isolation (mirrors box_pickup_loop.ipynb's compile_iteration_isolated) ---

_MP = mp.get_context("fork")

# Per-point address-space cap. The pydrake leak scales with planning work, so a
# hard point can grow far past a nominal trial's footprint. Under the cap a
# runaway allocation raises MemoryError/bad_alloc in the child instead, which
# the worker's except-clause reports as a rejection rather than drawing the
# OOM killer against something else on the machine.
WORKER_MEM_CAP = 8 * 2**30      # [bytes]


class _TeeToDisk:
    """Capture into memory *and* stream to a file, flushed at every newline.

    The child's output previously existed only in an in-memory StringIO that is
    handed back through the pipe after ``planner_fn`` returns. So the two cases where
    the log matters most -- a child killed by the memory cap, and one that outruns
    plan_isolated's timeout -- returned ``log=""`` and left no evidence at all. And a
    point still running was completely opaque: nothing reached disk until it finished,
    which for a hard point is tens of minutes of silence.

    Flushing per newline rather than per write keeps tqdm's carriage-return spam from
    turning this into thousands of fsyncs while still guaranteeing that any completed
    line survives a SIGKILL.
    """

    def __init__(self, buf, path):
        self._buf = buf
        self._fh = None
        if path:
            try:
                self._fh = open(path, "w", buffering=1)
            except OSError:
                self._fh = None          # never let logging break planning

    def write(self, s):
        self._buf.write(s)
        if self._fh is not None:
            try:
                self._fh.write(s)
                if "\n" in s:
                    self._fh.flush()
            except (OSError, ValueError):
                self._fh = None
        return len(s)

    def flush(self):
        if self._fh is not None:
            try:
                self._fh.flush()
            except (OSError, ValueError):
                self._fh = None

    def close(self):
        if self._fh is not None:
            try:
                self._fh.flush()
                self._fh.close()
            except (OSError, ValueError):
                pass
            self._fh = None


def _isolated_worker(conn, planner_fn, bx, by, kwargs, log_path=None):
    resource.setrlimit(resource.RLIMIT_AS, (WORKER_MEM_CAP, WORKER_MEM_CAP))
    buf = io.StringIO()
    sink = _TeeToDisk(buf, log_path)
    try:
        t0 = time.perf_counter()
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink), \
                record("point", f"{bx:+.3f},{by:+.3f}"):
            res = planner_fn(bx, by, **kwargs)
        # Wall latency of this point, distinct from the stage-second sums in
        # res["timings"]: with the reach legs planned in a parallel child, the
        # stage sums are CPU cost and this is what a robot would wait.
        res["wall_s"] = round(time.perf_counter() - t0, 2)
    except Exception as e:                      # never leave the parent hanging
        # The traceback goes into the captured buffer, not just repr(e) into detail.
        # Without it a worker_error left no evidence at all: the buffer simply stopped
        # mid-stage and the reason existed only in `detail`, which is dropped entirely
        # when a later GCP branch goes on to succeed. A MemoryError from WORKER_MEM_CAP
        # and a genuine planner bug are indistinguishable from a truncated log.
        # Written straight to the buffer, not printed: the redirect_stdout context has
        # already exited by the time this handler runs, so a print here would go to the
        # real stdout of a forked child that nothing is reading.
        sink.write(f"\n[worker_error] {type(e).__name__}: {e}\n{traceback.format_exc()}")
        res = dict(bx=float(bx), by=float(by), success=False, stage="worker_error",
                   detail=repr(e), steps=[], legs=[], timings={})
    sink.close()
    res["log"] = buf.getvalue()
    conn.send(res)
    conn.close()


def plan_isolated(planner_fn, bx, by, *, timeout_s=1800.0, log_path=None, **kwargs):
    """Run ``planner_fn(bx, by, **kwargs)`` in a forked child; leaked pydrake
    memory dies with it. ``planner_fn`` must return a plain, picklable dict, as
    both plan_grid_point and plan_reach_only do.

    ``log_path`` streams the child's stdout+stderr to disk as it is produced, so the
    output survives the child being killed and is inspectable while it is still
    running. On a kill or timeout the pipe carries nothing, so the on-disk file is
    read back to populate ``log`` -- otherwise the only two failures that leave no
    return value would also leave no diagnostics.
    """
    parent_conn, child_conn = _MP.Pipe(duplex=False)
    proc = _MP.Process(target=_isolated_worker,
                       args=(child_conn, planner_fn, bx, by, kwargs, log_path))
    proc.start()
    child_conn.close()
    if parent_conn.poll(timeout_s):             # readable OR at EOF (child died)
        try:
            res = parent_conn.recv()
        except EOFError:                        # child died before sending
            proc.join(timeout=10)
            res = dict(bx=float(bx), by=float(by), success=False, stage="worker_died",
                       detail=f"child exited with code {proc.exitcode} "
                              "(negative = signal; -9 usually means OOM-killed)",
                       steps=[], legs=[], timings={}, log="")
    else:
        res = dict(bx=float(bx), by=float(by), success=False, stage="worker_timeout",
                   detail=f"no result within {timeout_s} s",
                   steps=[], legs=[], timings={}, log="")
    if not res.get("log") and log_path and os.path.exists(log_path):
        # Killed or timed-out child: nothing came back through the pipe, but the tee
        # already flushed everything up to the last complete line.
        try:
            with open(log_path) as f:
                res["log"] = f.read()
        except OSError:
            pass
    parent_conn.close()
    proc.join(timeout=10)
    if proc.is_alive():
        proc.kill()
        proc.join()
    return res


# ── Per-point caching (plan_io v2) ---

def cache_path_for(cache_dir, index, prefix="point"):
    return os.path.join(cache_dir, f"{prefix}_{index:02d}.pkl")


def debug_path_for(cache_dir, index, prefix="point"):
    """Where a fresh planning attempt's per-leg diagnostics are written --
    see plan_grid_point's ``debug`` return key. Separate from the main plan
    cache since it's not robot-executable data, just for post-hoc analysis
    (e.g. comparing RRT/shortcut/trajopt/toppra trajectories for a point that
    planned something visibly off)."""
    return os.path.join(cache_dir, f"{prefix}_{index:02d}_debug.pkl")


def log_path_for(cache_dir, index, prefix="point"):
    """Where the forked planning worker's captured stdout+stderr is written.

    Plain text alongside the debug pickle, because it is the only durable record
    of *why* a stage failed, and grepping it is the first thing anyone does.
    """
    return os.path.join(cache_dir, f"{prefix}_{index:02d}.log")


def plan_or_load(index, bx, by, cache_dir, *, replan=False, planner_fn=plan_grid_point,
                 cache_prefix="point", gcp_candidates=GCP_CANDIDATES, **plan_kwargs):
    """Load a cached plan for grid point ``index``, or plan it fresh (isolated
    in a forked child, via ``planner_fn``) and cache on success.

    ``gcp_candidates`` are the grasp GCP branches to try, in order, when the point
    fails at a stage in ``GCP_RETRY_STAGES``. constrained_plan locks the lift and
    place to the branch the grasp was solved in, so a lift that is infeasible in one
    branch may be feasible in another -- and the branch was hardcoded, so a point
    that failed was never asked whether another branch would have worked. Failures
    that are not attributable to the branch (reach, home, a dead worker) do not
    retry, because re-solving the grasp would not address them.

    The retries are cheap relative to what they replace. A doomed constrained
    BiRRT already burns LEG_SEEDS x BIRRT_TIMEOUT_S before giving up, and a branch
    with no valid grasp IK is rejected in the IK stage without any search at all.

    ``cache_prefix`` keeps the main experiment's cache (``point_NN.pkl``)
    separate from e.g. the corner-marking pass's (``corner_NN.pkl``), since a
    reach-only plan there is not a substitute for (or interchangeable with) a
    full reach+grasp+lift+release+home plan at the same index.

    Failed points are never cached: plan_io.save_plan requires >=1 trajectory
    step, and retrying a failure on the next run is cheap -- this session's own
    experience is that some failures are transient (a worker_died rejection
    during grid-feasibility checking earlier succeeded cleanly on retry).

    A fresh attempt (success or failure) always writes its per-leg debug data
    (whichever ``planner_fn`` populated, e.g. plan_grid_point's ``debug`` key)
    to ``debug_path_for(...)``, independent of whether the plan itself got
    cached -- a cache hit skips planning entirely and leaves whatever debug
    file is already there untouched.
    """
    path = cache_path_for(cache_dir, index, prefix=cache_prefix)
    if os.path.exists(path) and not replan:
        steps, meta = load_plan(path)
        return dict(bx=float(bx), by=float(by), success=True, stage=None, detail="",
                   steps=steps, legs=meta["legs"], timings=meta["timings"],
                   evidence=meta.get("evidence", {}), cached=True,
                   gcp_index=meta.get("gcp_index"),
                   grasp_seed=meta.get("grasp_seed"),
                   grasp_score=meta.get("grasp_score"))

    # Try each GCP branch until one plans, keeping the first attempt's result if
    # they all fail (that is the branch the pipeline was tuned against, so its
    # failure is the informative one to report).
    supports_gcp = "gcp_index" in inspect.signature(planner_fn).parameters
    candidates = list(gcp_candidates) if supports_gcp else [None]
    res = None
    attempts, logs = [], []
    t_point = time.perf_counter()
    for n, gcp in enumerate(candidates):
        if n and time.perf_counter() - t_point > POINT_TIME_BUDGET_S:
            spent = time.perf_counter() - t_point
            msg = (f"GCP sweep stopped after {n} of {len(candidates)} branches: "
                   f"point budget {POINT_TIME_BUDGET_S:.0f}s exceeded ({spent:.0f}s). "
                   f"Branches not tried: {candidates[n:]}")
            print(f"[{index:2d}] {msg}")
            logs.append(f"===== sweep truncated =====\n{msg}\n")
            break
        kw = dict(plan_kwargs)
        if gcp is not None:
            kw["gcp_index"] = gcp
        os.makedirs(cache_dir, exist_ok=True)
        live = log_path_for(cache_dir, index, prefix=cache_prefix) + (
            f".gcp{gcp}.live" if gcp is not None else ".live")
        # One record per GCP branch attempted. Losing branches are flagged rather
        # than dropped: `res` below keeps only the branch that got furthest, so
        # until now a point that failed in branch 1 and succeeded in branch 5
        # reported branch 5's seconds as if branch 1 had never run.
        with record("gcp_branch", f"gcp{gcp}", gcp=gcp, order=n, index=index):
            res = plan_isolated(planner_fn, bx, by, log_path=live, **kw)
            mark(success=bool(res.get("success")), stage=res.get("stage"))
            if not res.get("success"):
                mark_discarded()
        attempts.append(res)
        logs.append(f"===== gcp_index={gcp} =====\n{res.get('log', '')}")
        # Persist what we have after every branch. Previously the whole point's log
        # was written only once, at the end, so a point on branch 6 showed nothing on
        # disk for the five branches it had already finished.
        try:
            with open(log_path_for(cache_dir, index, prefix=cache_prefix), "w") as f:
                f.write("\n".join(logs))
        except OSError:
            pass
        if res["success"]:
            if n:
                print(f"[{index:2d}] planned in GCP branch {gcp} "
                      f"(branch {candidates[0]} failed at {attempts[0].get('stage')})")
            break
        if res.get("stage") not in GCP_RETRY_STAGES + WORKER_FAILURE_STAGES:
            break                       # tells us nothing about the GCP branch
        if res.get("trajopt_failure"):
            # The BiRRT found a path in this branch and the guess over it was feasible;
            # trajopt then failed. That says the branch is feasible and the optimiser
            # is not working, so trying a different branch would hide a solver defect
            # behind a manifold change -- and it did exactly that on grid point 18,
            # whose branch-1 lift was written off as a branch-5 place-BiRRT failure.
            print(f"[{index:2d}] trajopt failed in GCP branch {gcp} on a feasible "
                  f"guess -- not retrying other branches; this is a solver/formulation "
                  f"failure to fix, not a branch to avoid")
            break
        if n + 1 < len(candidates):
            print(f"[{index:2d}] failed at {res.get('stage')} in GCP branch {gcp}; "
                  f"retrying in branch {candidates[n + 1]}")
    if not res["success"]:
        # Report the attempt that got FURTHEST, not the first one. Reporting the first
        # branch's failure hid the real story on a point whose lift BiRRT succeeded
        # instantly in a later branch and only then failed in trajopt: it was filed as
        # "constrained BiRRT failed to find a path", which sent the diagnosis after the
        # wrong subsystem entirely.
        res = max(attempts, key=lambda r: _stage_progress(r.get("stage")))
    res["log"] = "\n".join(logs)
    res["cached"] = False
    if res.get("debug") or res.get("log"):
        os.makedirs(cache_dir, exist_ok=True)
        with open(debug_path_for(cache_dir, index, prefix=cache_prefix), "wb") as f:
            pickle.dump(dict(index=index, bx=bx, by=by,
                             debug=res.get("debug", {}), log=res.get("log", "")), f)
        # Also as plain text. _isolated_worker redirects the child's stdout AND
        # stderr into res["log"], which is the only place Drake's own C++-level
        # messages exist ("Toppra failed to find the maximum path acceleration at
        # knot N/M") and the only place the planner's own diagnostics land
        # (_report_guess_feasibility, _feasible_min_dist_bound, _solve_trajopt_prog,
        # _accept_trajopt_or_fall_back). Nothing read it back, so every one of those
        # lines for every point planned so far was produced and discarded -- which
        # is why the cached run cannot say whether a trajopt failure started from an
        # infeasible guess or wandered off a feasible one.
        if res.get("log"):
            with open(log_path_for(cache_dir, index, prefix=cache_prefix), "w") as f:
                f.write(res["log"])
    if res["success"]:
        os.makedirs(cache_dir, exist_ok=True)
        # evidence rides in meta so a cached plan can be audited for the three
        # guarantees without replanning it.
        save_plan(path, res["steps"], meta=dict(
            index=index, bx=res["bx"], by=res["by"],
            timings=res["timings"], legs=res["legs"],
            evidence=res.get("evidence", {}),
            gcp_index=res.get("gcp_index"),
            grasp_seed=res.get("grasp_seed"),
            grasp_score=res.get("grasp_score"),
            grasp_n_candidates=res.get("grasp_n_candidates"),
            q_grasp=res.get("q_grasp"),
            wall_s=res.get("wall_s"),
            grasp_screen=res.get("grasp_screen", []),
            provenance=_plan_provenance(),
            # The two rates the stored arrays are on. Both were previously implicit
            # -- meta["legs"]["q"] is uniform over [0, duration] at viz_hz and the
            # commands at hz -- so a consumer had to infer them from array lengths.
            sample_hz=res.get("viz_hz"),
            command_hz=res.get("hz"),
            # Per-leg record of what the optimiser actually reported, so a cached plan
            # says whether each leg's trajopt certified optimality or merely returned a
            # feasible point. Without it, "the leg is valid" and "the solve converged"
            # are indistinguishable after the fact -- and they call for different work.
            trajopt=res.get("trajopt", {})))
    return res


# ── Persistent status manifest (durable "which points can I skip" record) ---

def status_path_for(cache_dir):
    return os.path.join(cache_dir, STATUS_FILENAME)


def load_status(cache_dir):
    path = status_path_for(cache_dir)
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def _cached_trajopt(cache_dir, index, prefix="point"):
    """The per-leg trajopt record from a cached plan's meta, or {}.

    Needed because a point loaded from cache has no live ``res["trajopt"]`` -- the
    solve happened in an earlier run. Reading it back from meta keeps a cached point's
    status honest about which legs were never certified, instead of silently dropping
    the flag on exactly the runs where nothing was replanned.
    """
    try:
        _, meta = load_plan(cache_path_for(cache_dir, index, prefix=prefix))
    except Exception:
        return {}
    return meta.get("trajopt") or {}


def trajopt_summary(trajopt):
    """Which legs the optimiser did not certify, and why it stopped.

    A leg whose solver reported failure but returned a constraint-satisfying iterate is
    accepted (see ``_solve_trajopt_prog``) -- that is the right call, because the dense
    verification gate decides what ships and a certificate is not evidence of validity.
    But it must not be invisible: an accepted-uncertified leg is a standing flag that
    the optimiser is not converging on that problem, which is work to do rather than a
    result to be happy with. Recording the solver's own exit reason alongside it is what
    makes the flag actionable instead of merely alarming.
    """
    unc = {leg: d.get("trajopt_info", "")
           for leg, d in (trajopt or {}).items() if d.get("trajopt_uncertified")}
    # A fell-back leg is a trajopt *failure* that was survivable, and it can only
    # occur under an explicit --allow-trajopt-fallback. It is recorded next to the
    # uncertified legs rather than folded in with them because the two say different
    # things: uncertified means "shipped trajopt's curve without a certificate",
    # fell_back means "did not ship trajopt's curve at all".
    fell = {leg: d.get("trajopt_fallback_reason", "")
            for leg, d in (trajopt or {}).items() if d.get("trajopt_fell_back")}
    return {
        "trajopt_uncertified_legs": sorted(unc),
        "trajopt_uncertified_detail": unc,
        "trajopt_fell_back_legs": sorted(fell),
        "trajopt_fell_back_detail": fell,
    }


def evidence_summary(evidence):
    """Worst case across a point's legs, as status.json records it.

    Split out of update_status so a reader that has only the cached plan -- e.g.
    scripts/grid_visualizer.py while a --precompute run is still in flight and has
    not written status.json yet -- reports the *same* numbers from meta["evidence"]
    instead of its own slightly different reduction of the same per-leg dicts.
    """
    return {
        "worst_clearance_mm": round(
            1000 * min(e["min_clearance_m"] for e in evidence.values()), 3),
        "worst_com_margin_mm": round(
            1000 * min(e["com_margin_m"] for e in evidence.values()), 2),
        "worst_joint_limit_excess_rad": max(
            e["joint_limit_excess_rad"] for e in evidence.values()),
        "total_duration_s": round(
            sum(e["duration"] for e in evidence.values()), 2),
        "all_legs_ok": all(e["ok"] for e in evidence.values()),
    }


def update_status(cache_dir, key, status, *, stage=None, detail="", evidence=None,
                  gcp_index=None, trajopt=None, grasp_seed=None, grasp_score=None):
    """Persist a point's latest planning status to <cache_dir>/status.json.

    A human moving the physical box between grid points may not be watching
    the console the whole time -- this gives them a durable, always-current
    reference of which indices are known-good vs known-bad instead. ``key`` is
    e.g. ``"3"`` for a main-experiment point or ``"corner_3"`` for a
    corner-marking pass, keeping the two namespaces distinct in one file.
    """
    os.makedirs(cache_dir, exist_ok=True)
    data = load_status(cache_dir)
    entry = dict(status=status, stage=stage, detail=str(detail)[:300],
                 updated_at=datetime.datetime.now().isoformat(timespec="seconds"))
    if gcp_index is not None:
        # Which grasp GCP branch produced this plan. Worth recording per point: it
        # is no longer a constant, and it determines the constrained manifold the
        # lift and place were searched in.
        entry["gcp_index"] = int(gcp_index)
    if grasp_seed is not None:
        entry["grasp_seed"] = int(grasp_seed)
    if grasp_score is not None:
        # The winning grasp's posture score (whole-arm asymmetry, lower is
        # better) -- the at-a-glance answer to "did this point get a balanced
        # grasp", next to the durations it predicts.
        entry["grasp_score"] = float(grasp_score)
    if evidence:
        entry.update(evidence_summary(evidence))
    if trajopt:
        entry.update(trajopt_summary(trajopt))
    data[key] = entry
    with open(status_path_for(cache_dir), "w") as f:
        json.dump(data, f, indent=2)


def _gcp_candidates(args):
    """Resolve --gcp-candidates, keeping the default when the flag is absent."""
    if getattr(args, "gcp_candidates", None) is None:
        return GCP_CANDIDATES
    return tuple(args.gcp_candidates)


def _discard_empty_timing_log(path, owner_pid):
    """Delete an event log that recorded no point, at exit.

    A run that dies before it plans anything -- a bare invocation stopped at the
    meshcat prompt, a typo'd --indices, a Ctrl-C during infrastructure build --
    still writes a log holding a run header and a couple of infra.build records.
    That file is not a measurement of anything, and since the cache's timing/ is
    no longer gitignored it shows up as an untracked file that reads like a
    result. Deleting it here means the only logs that survive are the ones with a
    point in them, so an untracked log in plans/grid_cache/timing/ always means
    "a real run finished and its log is waiting to be committed".

    Only the process that opened the log cleans up: forked point workers inherit
    the atexit registry and would otherwise each race to delete the file their
    parent is still writing to. A SIGKILLed run leaves its log behind, since
    atexit does not run then.
    """
    if os.getpid() != owner_pid:
        return
    stop_recording()
    try:
        with open(path) as fh:
            for line in fh:
                if '"cat": "point"' in line:
                    return
        os.remove(path)
        parent = os.path.dirname(path)
        if not os.listdir(parent):
            os.rmdir(parent)
    except OSError:
        pass                            # never let cleanup break the exit path


def _plan_provenance():
    """Which build produced a plan, stamped into the plan itself.

    This exists in the event log's run header already, but a plan pickle outlives
    the log it was planned under (and is often copied to the Jetson without one),
    so a cached trajectory could not say which commit or which Drake produced it.
    Cheap: two git calls, once per cached point.
    """
    import platform
    import subprocess

    def _git(*a):
        try:
            return subprocess.run(("git",) + a, cwd=common.RepoDir(),
                                  capture_output=True, text=True,
                                  timeout=10).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""

    try:
        import pydrake
        drake_path = os.path.dirname(pydrake.__file__)
    except Exception:
        drake_path = ""
    return dict(
        planned_at=datetime.datetime.now().isoformat(timespec="seconds"),
        git_commit=_git("rev-parse", "HEAD"),
        git_dirty=bool(_git("status", "--porcelain")),
        hostname=platform.node(),
        python=platform.python_version(),
        drake_path=drake_path,
    )


def _run_config(args, timing_log):
    """Machine + invocation identity, recorded so two caches can be compared.

    Two runs' numbers are only differenceable if these agree: SNOPT's and the
    BiRRT's limits are wall-clock, so core count and concurrency change what the
    solvers are allowed to do, not merely how long the run takes.
    """
    import platform
    import subprocess

    def _git(*a):
        try:
            return subprocess.run(("git",) + a, cwd=common.RepoDir(),
                                  capture_output=True, text=True,
                                  timeout=10).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""

    # A source build of Drake reports its version as "unknown", so record the
    # install path too -- with a source build that is the only thing that
    # identifies which Drake produced the numbers.
    try:
        import importlib.metadata
        drake_version = importlib.metadata.version("drake")
    except Exception:
        drake_version = ""
    try:
        import pydrake
        drake_path = os.path.dirname(pydrake.__file__)
    except Exception:
        drake_path = ""

    return dict(
        argv=sys.argv,
        jobs=int(getattr(args, "jobs", 1)),
        sequential=bool(getattr(args, "sequential", False)),
        leg_parallel=bool(_leg_parallel(args)),
        rng_seed=int(getattr(args, "rng_seed", 0)),
        lift_ik_candidates=int(getattr(args, "lift_ik_candidates", 0)),
        cache_dir=getattr(args, "cache_dir", ""),
        timing_log=timing_log,
        git_commit=_git("rev-parse", "HEAD"),
        git_dirty=bool(_git("status", "--porcelain")),
        hostname=platform.node(),
        cpu_count=os.cpu_count(),
        python=platform.python_version(),
        drake_version=drake_version,
        drake_path=drake_path,
        omp_num_threads=os.environ.get("OMP_NUM_THREADS"),
        mkl_num_threads=os.environ.get("MKL_NUM_THREADS"),
        t_start=time.time(),
    )


def _leg_parallel(args):
    """Resolve --leg-parallel: 'auto' enables it only for --jobs 1."""
    mode = getattr(args, "leg_parallel", "auto")
    if mode == "on":
        return True
    if mode == "off":
        return False
    return getattr(args, "jobs", 1) == 1


def _require_trajopt_legs(args):
    """Resolve --allow-trajopt-fallback into the leg-name tuple plan_grid_point wants.

    ``None`` (flag absent) is the strict default: trajopt failure fails the leg, on
    every leg. The flag present with no names means "allow the fallback everywhere",
    which argparse gives us as ``[]`` and which must not be confused with absent --
    that distinction is the whole reason this is a function and not a default=.
    """
    allow = getattr(args, "allow_trajopt_fallback", None)
    if allow is None:
        return REQUIRE_TRAJOPT_LEGS
    if not allow:                       # flag with no names: allow it everywhere
        return ()
    return tuple(leg for leg in REQUIRE_TRAJOPT_LEGS if leg not in set(allow))


def _skip_known_failed(args, known_status, key, idx, bx, by, *, label=None):
    """True (and prints a notification) if ``--skip-known-failed`` is set,
    ``key`` was already recorded as failed in ``known_status`` (a snapshot of
    status.json taken at the start of this run), and ``--replan`` wasn't also
    passed (which explicitly asks for a retry regardless)."""
    if not args.skip_known_failed or args.replan:
        return False
    entry = known_status.get(key)
    if not entry or entry.get("status") != "failed":
        return False
    tag = label or f"{idx:2d}"
    print(f"[{tag}] box=({bx:+.3f}, {by:+.3f}) skipped (--skip-known-failed; "
          f"previously failed at stage={entry.get('stage')!r}: "
          f"{entry.get('detail', '')[:150]})")
    return True


def plot_grid_diagram(grid_points, corner_idxs, status, out_path):
    """Save a PNG: every grid index, colored by known status (green = success,
    red = failed, gray = not yet attempted), corner-marking indices ringed,
    thin arrows connecting consecutive indices in the default iteration order
    -- a single reference for both "what order do points run in" and "which
    can I skip right now".
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    color_by_status = {"success": "#008300", "failed": "#e34948"}
    colors = [color_by_status.get(status.get(str(i), {}).get("status"), "0.7")
              for i in range(len(grid_points))]

    fig, ax = plt.subplots(figsize=(7, 8))
    for i in range(len(grid_points) - 1):
        ax.annotate("", xy=grid_points[i + 1], xytext=grid_points[i], zorder=1,
                   arrowprops=dict(arrowstyle="->", color="0.6", lw=1.0,
                                   shrinkA=10, shrinkB=10))
    ax.scatter(grid_points[:, 0], grid_points[:, 1], c=colors, s=260, zorder=3,
              edgecolors="0.2", linewidths=1)
    corner_pts = grid_points[corner_idxs]
    ax.scatter(corner_pts[:, 0], corner_pts[:, 1], s=460, facecolors="none",
              edgecolors="#2a78d6", linewidths=2.5, zorder=2)
    for i, (x, y) in enumerate(grid_points):
        ax.annotate(str(i), (x, y), ha="center", va="center", fontsize=8, zorder=4,
                   color="white" if colors[i] != "0.7" else "black")

    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#008300",
              markersize=12, label="success"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#e34948",
              markersize=12, label="failed"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="0.7",
              markersize=12, label="not yet attempted"),
        Line2D([0], [0], marker="o", color="#2a78d6", markerfacecolor="none",
              markersize=14, markeredgewidth=2.5, label="grid corner"),
        Line2D([0], [0], color="0.6", lw=1.0, marker=">", markersize=6,
              label="default iteration order"),
    ]
    ax.legend(handles=handles, loc="upper left", framealpha=0.9, fontsize=8)
    ax.set_xlabel("box centre x [m]")
    ax.set_ylabel("box centre y [m]")
    ax.set_title("Grid point order + status")
    ax.set_aspect("equal")
    ax.grid(True, color="0.9", lw=0.7)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=110)
    plt.close(fig)


# ── Meshcat preview infrastructure ────────────────────────────────────────────

def build_viz(meshcat):
    """Two persistent, meshcat-attached diagrams (empty-handed / with-box), for
    preview_legs to select between per leg via its with_box tag -- mirrors
    box_pickup_loop.ipynb's Meshcat Playback dual-diagram convention (both
    write the same meshcat paths, so whichever published last poses the robot).
    """
    plant_v, _, diagram_v = make_default_rby1_infrastructure(
        meshcat, obstacles=TABLE, open_grippers=True)
    held_vis = HeldBox(hand=HAND, size=SIZE, offset=OFFSET, rpy=RPY, visual=True,
                       open_top=True, wall_thickness=WALL_T, wall_height=WALL_HEIGHT)
    plant_vb, _, diagram_vb = make_default_rby1_infrastructure(
        meshcat, held_boxes=held_vis, obstacles=TABLE)

    layout_v = Rby1ActiveJointLayout(plant_v)
    layout_vb = Rby1ActiveJointLayout(plant_vb)
    ctx_v = diagram_v.CreateDefaultContext()
    ctx_vb = diagram_vb.CreateDefaultContext()
    pctx_v = plant_v.GetMyContextFromRoot(ctx_v)
    pctx_vb = plant_vb.GetMyContextFromRoot(ctx_vb)

    return {
        False: (plant_v, pctx_v, ctx_v, diagram_v, layout_v.plant_idxs),
        True: (plant_vb, pctx_vb, ctx_vb, diagram_vb, layout_vb.plant_idxs),
    }


# ── Shared preview/confirm/execute loop (used by both run modes) ────────────

def preview_confirm(idx, res, viz, args, *, name_prefix="point"):
    """Preview res["legs"] in meshcat and prompt once. Returns "continue" or "stop".

    ``--no-preview`` skips only the *first* playback and still poses the robot at
    the plan's start configuration, so the prompt is never answered against a
    stale meshcat scene. [r]eplay is unaffected -- the point of the flag is to
    stop paying 30-45 s of real-time playback per point before being allowed to
    answer, not to give up the ability to watch a leg.

    The original version of this function could also stream the plan to the
    robot. That path depended on the robot client, which is not part of this
    release; the plan is cached either way.
    """
    skip = getattr(args, "no_preview", False)
    while True:
        if skip:
            pose_legs_start(viz, res["legs"])
            print(f"[{idx:2d}] --no-preview set; showing the start pose only "
                  f"({sum(lg['duration'] for lg in res['legs']):.1f} s of motion "
                  f"across {len(res['legs'])} legs). Press r to play it.")
            skip = False
        else:
            preview_legs(viz, res["legs"])
        choice = input(f"[{idx:2d}] [c]ontinue / [r]eplay / [a]bort? ").strip().lower()
        if choice in ("c", "continue", ""):
            return "continue"
        if choice in ("a", "abort", "q"):
            sub = input(f"[{idx:2d}] [s]top / [c]ontinue to next point? ").strip().lower()
            return "stop" if sub in ("s", "stop") else "continue"
        # Anything else (including "r"): loop back and replay.


def run_main_experiment(indices, grid_points, grid_feasible, corner_idxs, viz, args, plan_kwargs):
    known_status = load_status(args.cache_dir)
    for idx in indices:
        bx, by = grid_points[idx]
        if args.skip_known_infeasible and not grid_feasible[idx]:
            print(f"[{idx:2d}] box=({bx:+.3f}, {by:+.3f}) skipped "
                  "(--skip-known-infeasible, grid_feasible=False)")
            continue
        if _skip_known_failed(args, known_status, str(idx), idx, bx, by):
            continue

        print(f"[{idx:2d}] box=({bx:+.3f}, {by:+.3f}) planning...")
        res = plan_or_load(idx, bx, by, args.cache_dir, replan=args.replan,
                           planner_fn=plan_grid_point, cache_prefix="point",
                           gcp_candidates=_gcp_candidates(args), **plan_kwargs)
        if not res["success"]:
            update_status(args.cache_dir, str(idx), "failed",
                          stage=res["stage"], detail=res["detail"])
            print(f"[{idx:2d}] planning FAILED at stage={res['stage']!r}: "
                  f"{res['detail'][:200]}")
            print(f"[{idx:2d}] skipping to the next grid point.")
            continue
        update_status(args.cache_dir, str(idx), "success",
                      evidence=res.get("evidence"), gcp_index=res.get("gcp_index"),
                      trajopt=res.get("trajopt"),
                      grasp_seed=res.get("grasp_seed"),
                      grasp_score=res.get("grasp_score"))
        print(f"[{idx:2d}] {'loaded from cache' if res['cached'] else 'freshly planned'}.")

        action = preview_confirm(idx, res, viz, args, name_prefix="point")
        if action == "stop":
            print(f"[{idx:2d}] grid loop stopped; skipping remaining points.")
            break

    diagram_path = os.path.join(args.cache_dir, DIAGRAM_FILENAME)
    plot_grid_diagram(grid_points, corner_idxs, load_status(args.cache_dir), diagram_path)
    print(f"Status diagram updated: {diagram_path}")
    print("Grid loop finished.")


def run_corner_marking(indices, grid_points, corner_idxs, viz, args, plan_kwargs):
    known_status = load_status(args.cache_dir)
    for idx in indices:
        bx, by = grid_points[idx]
        if _skip_known_failed(args, known_status, f"corner_{idx}", idx, bx, by,
                              label=f"corner {idx:2d}"):
            continue

        print(f"[corner {idx:2d}] box=({bx:+.3f}, {by:+.3f}) planning reach-only...")
        res = plan_or_load(idx, bx, by, args.cache_dir, replan=args.replan,
                           planner_fn=plan_reach_only, cache_prefix="corner", **plan_kwargs)
        if not res["success"]:
            update_status(args.cache_dir, f"corner_{idx}", "failed",
                          stage=res["stage"], detail=res["detail"])
            print(f"[corner {idx:2d}] planning FAILED at stage={res['stage']!r}: "
                  f"{res['detail'][:200]}")
            print(f"[corner {idx:2d}] skipping to the next corner.")
            continue
        update_status(args.cache_dir, f"corner_{idx}", "success")
        print(f"[corner {idx:2d}] {'loaded from cache' if res['cached'] else 'freshly planned'}.")

        action = preview_confirm(idx, res, viz, args, name_prefix="corner")
        if action == "stop":
            print(f"[corner {idx:2d}] corner-marking pass stopped.")
            break

    diagram_path = os.path.join(args.cache_dir, DIAGRAM_FILENAME)
    plot_grid_diagram(grid_points, corner_idxs, load_status(args.cache_dir), diagram_path)
    print(f"Status diagram updated: {diagram_path}")
    print("Corner-marking pass finished.")


def _plan_many_parallel(todo, args, plan_kwargs, planner_fn, cache_prefix, jobs):
    """Plan several grid points concurrently, ``jobs`` at a time.

    Grid points are completely independent -- separate plants, separate caches,
    separate RNG streams -- so this is embarrassingly parallel. The pipeline was
    serial only because plan_isolated forks one child at a time to contain the
    ~0.5-1.5 GB of pydrake memory each planning call leaks; running N at once costs
    N times that, which is the only reason to bound the pool rather than launch all
    20. Measured peak is ~1.3 GB per child, so the default 8 needs ~11 GB.

    Cache writes stay collision-free because each point owns its own
    ``point_NN.pkl``. status.json is *not* touched here -- the parent writes it
    from the returned results, as it did serially -- so there is no shared file to
    race on.

    Returns ``[(idx, tag, result), ...]`` in the input order.
    """
    import threading

    out = {}
    lock = threading.Lock()
    sem = threading.Semaphore(jobs)

    def work(idx, bx, by, tag):
        with sem:
            print(f"[{tag}] box=({bx:+.3f}, {by:+.3f}) planning...", flush=True)
            res = plan_or_load(idx, bx, by, args.cache_dir, replan=args.replan,
                               planner_fn=planner_fn, cache_prefix=cache_prefix,
                               gcp_candidates=_gcp_candidates(args), **plan_kwargs)
            with lock:
                out[idx] = (tag, res)

    # Threads, not processes: each one only waits on plan_isolated's forked child,
    # so the GIL is irrelevant and the real work is already in separate processes.
    threads = [threading.Thread(target=work, args=(idx, bx, by, tag), daemon=True)
               for idx, bx, by, tag in todo]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return [(idx, out[idx][0], out[idx][1]) for idx, _, _, _ in todo if idx in out]


def run_precompute(indices, grid_points, grid_feasible, corner_idxs, args, plan_kwargs, *,
                   planner_fn=plan_grid_point, cache_prefix="point", label_prefix=""):
    """Plan (or load) every requested point and cache it, with no meshcat, no
    preview, no robot -- just the fork-isolated planning + caching half of
    run_main_experiment (or run_corner_marking, via ``planner_fn``/
    ``cache_prefix``/``label_prefix``). For populating plans/grid_cache/ ahead
    of time so a later interactive run (or a replay on the robot)
    never has to wait on the planner.
    """
    n_success = n_failed = n_cached = n_skipped = 0
    known_status = load_status(args.cache_dir)
    status_key = (lambda idx: f"{cache_prefix}_{idx}") if cache_prefix != "point" else str
    todo = []
    for idx in indices:
        bx, by = grid_points[idx]
        tag = f"{label_prefix}{idx:2d}"
        if cache_prefix == "point" and args.skip_known_infeasible and not grid_feasible[idx]:
            print(f"[{tag}] box=({bx:+.3f}, {by:+.3f}) skipped "
                  "(--skip-known-infeasible, grid_feasible=False)")
            n_skipped += 1
            continue
        if _skip_known_failed(args, known_status, status_key(idx), idx, bx, by, label=tag):
            n_skipped += 1
            continue

        todo.append((idx, bx, by, tag))

    jobs = max(1, int(getattr(args, "jobs", 1)))
    if jobs > 1 and len(todo) > 1:
        print(f"planning {len(todo)} point(s) with {jobs} worker(s) in parallel")
        results = _plan_many_parallel(todo, args, plan_kwargs, planner_fn,
                                     cache_prefix, jobs)
    else:
        results = []
        for idx, bx, by, tag in todo:
            print(f"[{tag}] box=({bx:+.3f}, {by:+.3f}) planning...")
            results.append((idx, tag, plan_or_load(
                idx, bx, by, args.cache_dir, replan=args.replan,
                planner_fn=planner_fn, cache_prefix=cache_prefix,
                gcp_candidates=_gcp_candidates(args), **plan_kwargs)))

    for idx, tag, res in results:
        if not res["success"]:
            update_status(args.cache_dir, status_key(idx), "failed",
                          stage=res["stage"], detail=res["detail"],
                          gcp_index=res.get("gcp_index"))
            n_failed += 1
            print(f"[{tag}] planning FAILED at stage={res['stage']!r}: "
                  f"{res['detail'][:200]}")
            continue
        update_status(args.cache_dir, status_key(idx), "success",
                      evidence=res.get("evidence"), gcp_index=res.get("gcp_index"),
                      trajopt=res.get("trajopt") or _cached_trajopt(args.cache_dir, idx),
                      grasp_seed=res.get("grasp_seed"),
                      grasp_score=res.get("grasp_score"))
        n_success += 1
        n_cached += res["cached"]
        summary = trajopt_summary(
            res.get("trajopt") or _cached_trajopt(args.cache_dir, idx))
        unc = sorted(summary["trajopt_uncertified_detail"].items())
        fell = sorted(summary["trajopt_fell_back_detail"].items())
        print(f"[{tag}] {'loaded from cache' if res['cached'] else 'freshly planned'}."
              + (f"  UNCERTIFIED trajopt: "
                 + "; ".join(f"{leg} ({info})" for leg, info in unc) if unc else "")
              + (f"  TRAJOPT FAILED (fallback shipped): "
                 + "; ".join(f"{leg} ({why})" for leg, why in fell) if fell else ""))

    diagram_path = os.path.join(args.cache_dir, DIAGRAM_FILENAME)
    plot_grid_diagram(grid_points, corner_idxs, load_status(args.cache_dir), diagram_path)
    print(f"Status diagram updated: {diagram_path}")
    print(f"Precompute finished: {n_success} succeeded ({n_cached} already cached, "
          f"{n_success - n_cached} freshly planned), {n_failed} failed, "
          f"{n_skipped} skipped, out of {len(indices)} requested.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-path", default=DEFAULT_GRID_PATH,
                        help="Path to the grid saved by box_reachability_sampling.ipynb.")
    parser.add_argument("--indices", type=int, nargs="*", default=None,
                        help="Grid point indices to run (default: all; for "
                             "--mark-corners, default: all 4 corners).")
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR,
                        help="Where per-point plans and the status manifest/diagram "
                             "are cached. Execution records go to --results-dir.")
    parser.add_argument("--results-dir", metavar="DIR", default=DEFAULT_RESULTS_DIR,
                        help="Directory for execution records (created if needed). "
                             "Default: results/ at the repo root, matching "
                             "a hardware replay. Keeping them out of the plan "
                             "cache matters: that cache's 20 plan pickles are "
                             "committed deliverables, and a run's robot logs are "
                             "tens of MB of diagnostics, not results.")
    parser.add_argument("--replan", action="store_true",
                        help="Recompute even if a cached plan exists for a point. "
                             "NOTE: on the canonical cache this OVERWRITES a "
                             "committed plan pickle -- that is what makes trajectory "
                             "changes show up in `git status`, but it is not what you "
                             "want if you only meant to look at a cached plan.")
    parser.add_argument("--skip-known-infeasible", action="store_true",
                        help="Skip points the grid file's grid_feasible marks infeasible "
                             "(informational only -- from a weaker rrt_only-only check). "
                             "Ignored for --mark-corners.")
    parser.add_argument("--skip-known-failed", action="store_true",
                        help="Skip points/corners status.json already recorded as "
                             "'failed', printing a notification instead of "
                             "replanning them. Overridden by --replan (which explicitly "
                             "asks for a retry).")
    parser.add_argument("--mark-corners", action="store_true",
                        help="Run only the reach leg (no grasp/lift/home-box-logic) at "
                             "the grid's 4 corner indices, to physically mark/verify the "
                             "grid's boundary before laying out all 20 box positions. "
                             "Runs once and exits; does not run the main experiment.")
    parser.add_argument("--status", action="store_true",
                        help="Just reprint and redraw the current status diagram from "
                             "the cache/status manifest, then exit -- no planning, no "
                             "meshcat, no robot connection.")
    parser.add_argument("--precompute", action="store_true",
                        help="Plan (or load) and cache every requested point with no "
                             "meshcat, no preview, no robot connection -- for populating "
                             "plans/grid_cache/ ahead of time. Combine with --mark-corners "
                             "to precompute the reach-only corner plans instead.")
    parser.add_argument("--hz", type=float, default=20.0,
                        help="Robot-command streaming rate (samples/sec).")
    parser.add_argument("--viz-hz", type=float, default=30.0,
                        help="Meshcat preview sampling rate (samples/sec).")
    parser.add_argument("--no-preview", action="store_true",
                        help="Skip the real-time meshcat playback before each "
                             "point's prompt, posing the robot at the plan's start "
                             "configuration instead. Playback is real time (30-45 s "
                             "per point), so this is the difference between "
                             "inspecting a cached grid and waiting out 20 replays. "
                             "Press r at the prompt to play a point after all.")
    parser.add_argument("--jobs", type=int, default=10,
                        help="plan this many grid points concurrently during "
                             "--precompute. Points are independent, so this scales "
                             "nearly linearly; each worker peaks around 1.4 GB of "
                             "leaked pydrake memory, so size it against free RAM "
                             "(10 needs ~14 GB). Default 10. Each worker is "
                             "effectively single-threaded, so also keep jobs well "
                             "under the core count: measured on 24 cores, 10 workers "
                             "sit at ~34%% CPU, while 20 oversubscribe (load ~35), "
                             "which eats into the *wall-clock* SNOPT and BiRRT time "
                             "limits and so costs solve quality, not just latency.")
    parser.add_argument("--rng-seed", type=int, default=0,
                        help="RNG seed passed to the planner (shared across all points).")
    parser.add_argument("--leg-parallel", choices=("auto", "on", "off"), default="auto",
                        help="Plan the two reach legs in a forked child concurrently "
                             "with the winner's carry probe (they depend only on the "
                             "grasp pair, not the probe verdict). 'auto' (default) "
                             "enables it only at --jobs 1 -- the single-point-latency "
                             "case it exists for -- because at --jobs 10 every extra "
                             "fork feeds the measured memory-bandwidth contention "
                             "that inflates all wall-clock solver budgets.")
    parser.add_argument("--lift-ik-candidates", type=int, default=50,
                        help="Feasible IK solutions to collect for the lift goal; the "
                             "one with the largest minimum joint-limit margin is used. "
                             "Unused for --mark-corners (no lift).")
    parser.add_argument("--allow-trajopt-fallback", nargs="*", default=None,
                        metavar="LEG",
                        help="Legs allowed to ship the raw BiRRT+shortcut path when "
                             "trajopt's trajectory is unusable (solver failure OR a "
                             "solve the dense check rejects). Off by default on every "
                             "leg: a trajopt failure fails the leg, which is what it "
                             "is. Pass with no names to allow it everywhere, or e.g. "
                             "'reach_approach home' for those legs only. Legs that "
                             "take the fallback are tagged trajopt_fell_back in the "
                             "cached plan and reported by grid_provenance_report.")
    parser.add_argument("--gcp-candidates", nargs="+", type=int, default=None,
                        metavar="IDX",
                        help="Grasp GCP branches to try in order when a constrained "
                             f"leg fails. Default {list(GCP_CANDIDATES)}.")
    parser.add_argument("--sequential", action="store_true",
                        help="Measurement mode: plan one grid point at a time "
                             "(--jobs 1) so each point's wall time is the latency "
                             "of producing that plan alone, uncontended. "
                             "Parallelism WITHIN a point (--leg-parallel, grasp "
                             "screening) is deliberately left on -- it is part of "
                             "what one plan costs. Errors if --jobs > 1 was also "
                             "passed. Pair with scripts/timing_report.py.")
    parser.add_argument("--timing-log", default=None, metavar="PATH",
                        help="Write the runtime event log here (default: "
                             "<cache-dir>/timing/run_<timestamp>.jsonl). Every "
                             "process in the run, including forked children that "
                             "are later killed, appends to this one file.")
    parser.add_argument("--no-timing-log", action="store_true",
                        help="Disable the event log entirely.")
    args = parser.parse_args()

    if args.sequential:
        if args.jobs != parser.get_default("jobs") and args.jobs > 1:
            parser.error(f"--sequential means one point at a time, but --jobs "
                         f"{args.jobs} was also passed; drop one of them.")
        args.jobs = 1

    if not args.no_timing_log:
        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        timing_log = args.timing_log or os.path.join(
            args.cache_dir, "timing", f"run_{stamp}.jsonl")
        start_recording(timing_log)
        emit_meta("run", **_run_config(args, timing_log))
        args.timing_log = timing_log
        print(f"Timing log: {timing_log}")
        atexit.register(_discard_empty_timing_log, timing_log, os.getpid())

    grid = np.load(args.grid_path)
    grid_points = grid["grid_points"]
    grid_nx, grid_ny = int(grid["grid_nx"]), int(grid["grid_ny"])
    corner_idxs = corner_indices(grid_nx, grid_ny)
    grid_feasible = (grid["grid_feasible"] if "grid_feasible" in grid.files
                     else np.ones(len(grid_points), dtype=bool))

    if args.status:
        status = load_status(args.cache_dir)
        diagram_path = os.path.join(args.cache_dir, DIAGRAM_FILENAME)
        plot_grid_diagram(grid_points, corner_idxs, status, diagram_path)
        print(f"Status diagram: {diagram_path}")
        for i in range(len(grid_points)):
            st = status.get(str(i), {}).get("status", "not yet attempted")
            tag = " (corner)" if i in corner_idxs else ""
            print(f"  [{i:2d}]{tag} {st}")
        return

    print(f"Loaded {len(grid_points)}-point grid from {args.grid_path}.")

    if args.precompute and args.mark_corners:
        indices = [i for i in corner_idxs if args.indices is None or i in args.indices]
        print(f"--precompute --mark-corners set; planning/caching reach-only at corner "
              f"indices {indices} (no meshcat, no robot).")
        plan_kwargs = dict(rng_seed=args.rng_seed, hz=args.hz, viz_hz=args.viz_hz)
        run_precompute(indices, grid_points, grid_feasible, corner_idxs, args, plan_kwargs,
                       planner_fn=plan_reach_only, cache_prefix="corner", label_prefix="corner ")
        return

    if args.precompute:
        indices = args.indices if args.indices else list(range(len(grid_points)))
        print(f"--precompute set; planning/caching indices {indices} (no meshcat, no robot).")
        plan_kwargs = dict(rng_seed=args.rng_seed, lift_ik_candidates=args.lift_ik_candidates,
                           hz=args.hz, viz_hz=args.viz_hz,
                           require_trajopt_legs=_require_trajopt_legs(args),
                           leg_parallel=_leg_parallel(args))
        run_precompute(indices, grid_points, grid_feasible, corner_idxs, args, plan_kwargs)
        return

    meshcat = StartMeshcat()
    print(f"Meshcat: {meshcat.web_url()}")
    viz = build_viz(meshcat)

    if args.mark_corners:
        indices = [i for i in corner_idxs if args.indices is None or i in args.indices]
        print(f"--mark-corners set; running reach-only at corner indices {indices}.")
        plan_kwargs = dict(rng_seed=args.rng_seed, hz=args.hz, viz_hz=args.viz_hz)
        run_corner_marking(indices, grid_points, corner_idxs, viz, args, plan_kwargs)
        return

    indices = args.indices if args.indices else list(range(len(grid_points)))
    print(f"Running indices {indices}.")
    plan_kwargs = dict(rng_seed=args.rng_seed, lift_ik_candidates=args.lift_ik_candidates,
                       hz=args.hz, viz_hz=args.viz_hz,
                       require_trajopt_legs=_require_trajopt_legs(args),
                       leg_parallel=_leg_parallel(args))
    run_main_experiment(indices, grid_points, grid_feasible, corner_idxs, viz, args, plan_kwargs)


if __name__ == "__main__":
    main()
