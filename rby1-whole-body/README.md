# RB-Y1 Whole-Body Constrained Planning

The **RB-Y1 whole-body experiment** for *Planning along Differentiable Charts of Constraint
Manifolds with General-Purpose IK Solvers*, and the paper's hardware demonstration.

A 23-DOF Rainbow Robotics RB-Y1 — holonomic base (3), torso (6), two 7-DOF arms — picks up a box
and places it on a table, at each of 20 positions on a grid. While the box is carried, the
gripper-to-gripper transform is frozen: a kinematic equality constraint on the full 23-DOF
configuration. Planning happens in a reduced space built from **IKFast** analytic IK, and the
constraint is eliminated by construction rather than enforced.

IKFast is the point. It is an automated meta-solver whose generated code is not written for
differentiability and is impractical to modify — exactly the case the paper's inverse-function-
theorem approach exists to handle.

**Executed on hardware 2026-08-11: 20/20 grid points planned, 20/20 executed cleanly.**

---

## What this folder produces

| Paper artifact | Command | Runtime |
| --- | --- | --- |
| Full-body IK table — solve time | `python scripts/timing_report.py --log plans/grid_cache/timing/run_20260810-222505.jsonl` | seconds |
| — per-leg / per-stage breakdown | `python scripts/planning_time_by_leg.py` | seconds |
| — constraint violation | `python scripts/ee_constraint_report.py` | seconds |
| — success rate and the three guarantees | `python scripts/grid_guarantee_report.py` | seconds |
| Guarantees re-derived from scratch | `python scripts/verify_cached_plan.py plans/grid_cache/point_*.pkl` | ~10 min |
| Replan the whole grid | `python scripts/plan_grid.py --precompute --sequential --replan` | ~19 min |
| Supplementary video (simulation cut) | `python scripts/video/build_video.py overview` | ~1 h |
| Paper teaser figure | committed at `scripts/figures/sweep_p7_med.png`; `render_options.sh` regenerates it, but needs the top-down recording, which is not distributable — see that folder's README | ~3 min |

The report scripts read the **committed** plan cache and hardware records, so every reported
number regenerates in seconds without replanning and without a robot.

## Headline numbers

| Success rate | Solve time (s) | Constraint violation |
| --- | --- | --- |
| 20/20 planned, 20/20 executed | 43.6 median (56.1 mean) | 0.55 mm mean, 2.23 mm max |

- **Success rate** — all 20 grid points produced a plan passing the dense three-guarantee check,
  and all 20 executed to completion on hardware, 8 steps each, with no error.
- **Solve time** — per-point planning latency from the event log, which reconciles against
  measured wall clock with a 0.000 % residual. Range 34.1–135.0 s, sequential (`--jobs 1`), so
  this is uncontended latency rather than throughput.
- **Constraint violation** — drift of the gripper-to-gripper transform on the constrained legs
  *as measured on the robot*. The planned trajectory's own violation is 6×10⁻¹⁵ m, i.e. machine
  precision. Read the caveats below before quoting either figure.

Worst case across all 20 cached plans:

| Guarantee | Worst |
| --- | --- |
| Collision | +8.99 mm clearance, no penetration |
| Static stability | +8.63 mm CoM margin, inside the support polygon |
| Joint limits | 7.95×10⁻⁵ rad excess |

> **Two caveats that must travel with these numbers.**
>
> The joint-limit excess is trajectory optimization's between-knot overshoot on legs that passed
> dense verification. 4 of 20 points needed a client-side clamp, all in the `place` leg, all on a
> wrist joint.
>
> The CoM margin is on an **unnormalised** scaling: the residual uses unnormalised edge normals,
> so the stored value is (distance × edge length). It is not in metres and should not be quoted
> as such.

---

## The idea

**The parameterization.** `src/rby1_analytic_ik.py` wraps the two IKFast extensions and works over
both `float` and `AutoDiffXd`. A **GCP** (global configuration parameter) is the
(wrist, elbow, shoulder) sign triple naming which of IKFast's 8 branches a configuration lives on.
Branch selection has to be an exact bijection with the branch IKFast actually took — otherwise
trajectories teleport between branches. Near the workspace boundary, IK falls back to a damped
projection (`_damped_lstsq`, `boundary_damping_lambda` default 0.1).

**Optimization on the chart.** `src/rby1_opt_ik.py` optimizes over
`(base_xyt, torso, right_eef_xyzrpy, ψ_right, left_eef_xyzrpy, ψ_left)`; the analytic IK
reconstructs the full 23-DOF configuration inside every evaluation. It also owns the
centre-of-mass / support-polygon stability constraint, which is what keeps a mobile base with a
6-DOF torso from planning itself over.

**Planning.** `src/rby1_planning.py` has two entry points: `unconstrained_plan` (joint-space
reach — IK across all 8 GCPs → BiRRT → shortcut → optional trajopt → TOPPRA) and
`constrained_plan` (carries the object with the gripper-to-gripper transform frozen, planning in
the 14-D SE(3) × torso × ψ space).

**Six legs per grid point:**

| Leg | Motion | Method |
| --- | --- | --- |
| `reach_approach` | ready pose → standoff above the grasp | trajopt |
| `reach_descend` | standoff → grasp | straight line or BiRRT |
| `lift` | grasp → fixed raise pose | **constrained** |
| `place` | → fixed place pose over the table | **constrained** |
| `home_retreat` | release → standoff above it | straight line or BiRRT |
| `home` | standoff → ready pose | trajopt |

Reach and home are split at a standoff rather than run straight to the grasp because a trajopt leg
whose endpoint sits ~15 mm from the box is squeezed against its own minimum-distance floor, making
the problem infeasible exactly at the endpoint. The two short legs adjacent to the box skip trajopt
entirely — BiRRT plus shortcutting clears them in 0.1–0.3 s — so trajopt handles only the open
middle of the motion.

**The three guarantees.** Every leg's trajectory is densely sampled and checked for collision
freedom, CoM inside the support polygon, and joint limits *before* it can be cached. A leg that
fails the check fails the point. The resulting numbers ride along in the cache and in
`status.json`, so a cached plan can be audited without replanning it.

---

## Install

Route 1 of [`../docs/INSTALL.md`](../docs/INSTALL.md) — the Drake pip wheel — is enough for
everything here except two stages of the video, which need a Drake installation. This release is
pinned to **Drake 1.56.0**.

**Python 3.12 exactly**: the IKFast extensions build as `cpython-312`.

```bash
cd rby1-whole-body
python3.12 -m venv .venv && . .venv/bin/activate
pip install -e .        # also compiles rainbow_{left,right}_arm_ik -- takes seconds
python -c "import rainbow_left_arm_ik, rainbow_right_arm_ik; print('IKFast ok')"
```

A container with this already built is available — see [`../docs/DOCKER.md`](../docs/DOCKER.md).

---

## What ships, and what does not

**This folder ships planning, verification, reporting and the video pipeline. It does not ship the
robot client.** The code that talked to the RB-Y1 over the network — connection management, the
on-robot server, the gripper driver, camera capture — is not part of this release. Consequently:

- `scripts/plan_grid.py` plans, verifies and caches; it cannot execute. The execution paths are
  removed, not stubbed.
- `src/plan_format/` is the Drake-free, network-free remainder: the plan file format, the
  conversion to command waypoints, and the robot's published joint limits. It is what lets a
  cached plan be loaded, verified, previewed and rendered without a robot.

> **Reading the pickles from outside this folder.** The plan cache and execution records were
> written before this package was renamed, so they name a module (`rby1_interface.utilities`)
> that no longer exists. Import `plan_format` first — it registers the historical names — and a
> plain `pickle.load` then works:
>
> ```python
> import sys; sys.path.insert(0, "src")
> import plan_format          # registers the pre-rename module aliases
> import pickle; plan = pickle.load(open("plans/grid_cache/point_00.pkl", "rb"))
> ```
>
> The records were left as they are rather than rewritten: they are the experiment's result, and
> rewriting them to tidy a rename is not a trade worth making.

**The hardware run cannot be repeated, so its records ship as data.**
`results/2026-08-11/*.pkl` (20 execution records) and `plans/grid_cache/` (20 plan pickles plus
the event log) are committed. For anything quoted as an execution result, **the records are the
result** — the robot was in that state once.

**The raw hardware capture is not distributable.** The 1.9 GB of video from the 2026-08-11 session
is not in this repository. `videos/trim_points.json` — the derived sync table — is.

**What that means for the video.** `scripts/video/build_video.py` is complete and shipped.

- `build_video.py overview` — the explainer / simulation cut (~175 s) rebuilds end to end from
  this repository.
- `build_video.py ral` — the hardware supplementary (~178 s) **cannot** be rebuilt, because the
  raw clips are the footage. The finished cut is published at
  [https://www.youtube.com/watch?v=ADF4g3iQsuY](https://www.youtube.com/watch?v=ADF4g3iQsuY). It detects the missing clips and refuses rather than substituting
  other files, because the clips are addressed positionally by filename and a silent mismatch
  would pair the wrong footage with the wrong plan. That refusal is correct behaviour, not a
  failure.

`build_video.py --dry-run` names every missing prerequisite without doing any work.

---

## Reproducing the results

All commands run from this folder.

### 1. Verify the committed plan cache

```bash
python scripts/verify_cached_plan.py plans/grid_cache/point_05.pkl   # one point, dense
python scripts/verify_plan_commands.py                               # streamed waypoints
python scripts/grid_guarantee_report.py                              # cached evidence, all 20
```

### 2. The runtime numbers

```bash
python scripts/timing_report.py --log plans/grid_cache/timing/run_20260810-222505.jsonl
python scripts/timing_report.py --log plans/grid_cache/timing/run_20260810-222505.jsonl --per-point   # one row per grid point
python scripts/planning_time_by_leg.py
```

Expect 20 points, mean 56.1 s, median 43.6 s, range 34.1–135.0 s, and an event-log residual
against measured wall clock of 0.000 %.

> **Name the log explicitly.** `plans/grid_cache/timing/` holds two event logs: the canonical
> 20-point sequential run (`run_20260810-222505.jsonl`), which is where the reported numbers come
> from, and a later single-point re-run of point 00 (`run_20260812-001224.jsonl`). With no
> `--log`, the report globs both, counts point 00 twice, and reports 21 points at mean 58.1 s
> instead of 20 at 56.1 s. Both logs ship because both are part of the record; only the first is
> the result.

### 3. The constraint violation

```bash
python scripts/ee_constraint_report.py
```

Reads the hardware records. Expect 0.55 mm mean / 2.23 mm max measured, ~6×10⁻¹⁵ m planned.

### 4. Re-derive the guarantees rather than trusting them

```bash
python scripts/verify_cached_plan.py plans/grid_cache/point_*.pkl    # ~10 min, all 20
```

This is the single most valuable check here: `grid_guarantee_report.py` prints the evidence each
plan *cached* at planning time, whereas `verify_cached_plan.py` replays the stored trajectories
through the collision, stability and joint-limit checks again. It therefore proves that the plans,
the models, the `package://` paths and the pinned Drake all still agree, rather than trusting a
verdict recorded earlier.

### 5. Replan from scratch (optional)

```bash
python scripts/plan_grid.py --precompute --sequential --replan --cache-dir plans/grid_cache_repro
python scripts/grid_guarantee_report.py --cache-dir plans/grid_cache_repro
python scripts/verify_cached_plan.py plans/grid_cache_repro/point_*.pkl
```

~19 min sequential; use `--jobs 10` for throughput rather than latency. Planning is stochastic, so
a replan will not reproduce the cached trajectories exactly — it should reproduce 20/20 success
and comparable latencies.

### 6. Inspect a plan interactively

```bash
python scripts/grid_visualizer.py            # meshcat on :7000
```

---

## Two things that surprise people

**pydrake leaks 0.5–1.5 GB per planning call.** It is unreclaimable, so a 20-point loop planned
in-process will exhaust memory. Every grid point is therefore planned in a forked child; the
meshcat preview stays in the parent. This is why the pipeline forks rather than loops.

**There are up to four nested levels of fork parallelism** — point, leg, grasp candidate, IK
candidate. `--jobs` bounds the outermost. On a 20-core machine `--jobs 10` is a reasonable
default; the reported latencies were measured with `--jobs 1` deliberately, because latency and
throughput are different claims.

---

## Repository layout

| Path | Contents |
| --- | --- |
| `src/rby1_analytic_ik.py` | The parameterization: IKFast wrapper, GCP branch tracking, damped boundary projection. |
| `src/rby1_opt_ik.py` | Optimization IK on the chart; CoM / support-polygon stability; `MakeRby1Diagram`. |
| `src/rby1_planning.py` | `unconstrained_plan`, `constrained_plan`, `verify_trajectory`, scene construction. |
| `src/rrt.py`, `src/shortcut.py` | Bidirectional RRT and randomized shortcutting. |
| `src/exps/timing_utils.py` | The event log the runtime numbers are derived from. |
| `src/plan_format/` | Plan file format, command conversion, published joint limits. |
| `cpp_parameterization/cpp/rby1_ik/` | The IKFast sources. Built by `setup.py`; needs no Drake. |
| `scripts/plan_grid.py` | The end-to-end experiment. |
| `scripts/video/` | The video pipeline; `build_video.py` orchestrates it. |
| `scripts/figures/` | The paper's teaser figure: a multiple-exposure composite built from a top-down recording of the hardware session. |
| `models/` | Robot and scene descriptions. See [`THIRD_PARTY.md`](THIRD_PARTY.md). |
| `plans/grid_cache/` | The 20 committed plans plus the event log. |
| `results/2026-08-11/` | The 20 hardware execution records. |

---

## Licence and third-party content

MIT for this project's own source — see [`../LICENSE`](../LICENSE). The RB-Y1 description, the
gripper, the scene geometry and the IKFast sources each carry their own terms, listed in
[`THIRD_PARTY.md`](THIRD_PARTY.md).

## Citation

See the [top-level README](../README.md#citation).

## Questions

Please open an issue on this repository.
