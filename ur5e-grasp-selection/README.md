# UR5e Grasp Selection with a General-Purpose Analytic IK Solver

The **UR5e grasp-selection experiment** for *Planning along Differentiable Charts of Constraint
Manifolds with General-Purpose IK Solvers*. An analytic IK optimization pipeline for the UR5e,
using **EAIK** as the analytic solver — a general-purpose one, not written for differentiability —
and the **Inverse Function Theorem** to recover its gradients.

It produces the paper's **grasp-IK table** and both of its UR5e figures: the grasp-selection
figure (three arms grasping the same mug on three distinct IK branches) and the target-scene
figure (one sampled benchmark target in the obstacle scene the benchmark actually runs in).

The paper's rendered PNGs are not redistributed here — both regenerate from the pinned
configurations committed alongside the scripts. See
[Replicating the Grasp Selection Figure](#replicating-the-grasp-selection-figure) and
[Replicating the Target Scene Figure](#replicating-the-target-scene-figure).

## Key Features
- **Minimal Coordinate Optimization**: Optimizes grasp parameters (6-DOF) relative to a target object (mug), rather than full joint angles (6-DOF arm).
- **IFT-based Gradients**: Propagates gradients through the black-box analytical IK solver (EAIK) using the kinematic Jacobian and IFT.
- **Three-Formulation Benchmark**: Compares the traditional full-space NLP (Drake IK) against two reduced-space IFT formulations that differ in how they enforce reachability.

## The three formulations

| Name | Decision variables | Reachability enforced by |
|---|---|---|
| **Old** | the 6 joint angles $q$ | a Drake `InverseKinematics` pose constraint |
| **Direct** | the 6-DOF grasp parameters $p$ of the grasp frame in the mug frame | the FK residual $\|FK(IK(X(p))) - X(p)\|_F = 0$ |
| **Boundary** | the same $p$ | $-\log\det(JJ^\top + \epsilon I) \le \text{threshold}$ |

Direct and Boundary recover $q$ by calling the analytic EAIK solver inside the constraint
and cost evaluations, and differentiate through it with the Inverse Function Theorem,
$dq/dp = J^{-1}\,dpose/dp$.

## Repository Structure
- `src/eaik_ik.py`: Wrapper for the EAIK solver, including the Drake `tool0` ↔ EAIK flange convention.
- `src/ift_gradients.py`: IFT gradient logic ($dq/dpose = J^{-1}$) and the damped-pseudo-inverse strategies.
- `src/boundary_reach_constraint.py`: The $-\log\det(JJ^\top + \epsilon I)$ reachability constraint, also reusable as a manipulability cost.
- `src/ur_experiments.py`: The three optimization problem definitions and the shared `UrProblemOptions` knob bag.
- `src/bench_stats.py`: All benchmark statistics — McNemar's exact test, Wilcoxon signed-rank, target-level bootstrap, multi-start scoring.
- `src/util.py`: Builds the Drake diagram from the model-directives YAMLs.
- `models/`: URDF and scene description files for the UR5e environment.

### Scripts

Reproducible entry points:
- `scripts/benchmark_boundary_reach.py` — **the main benchmark**: three-way Old/Direct/Boundary comparison, writes a JSON summary to `logs/`.
- `scripts/tests/test_eaik_ift.py` — the test suite (EAIK round-trip, IFT vs finite differences, end-to-end solve).
- `scripts/calibrate_boundary_threshold.py` — derives $\epsilon$ and the boundary threshold from the robot's kinematics.
- `scripts/reproduce_grasp_figure.sh` — regenerates the grasp-selection figure with every parameter pinned.
- `scripts/visualize_grasp_selection.py`, `scripts/render_grasp_figure.{py,sh}` — the grasp-selection figure pipeline (see below).
- `scripts/visualize_target_scene.py`, `scripts/render_target_scene.{py,sh}` — the target-scene figure pipeline (see below); its target is pinned in `scripts/target_scene_configuration.json`.

Diagnostics, not sources of reported results:
- `scripts/tune_boundary_reach.py` — samples configurations and searches $(\epsilon, \text{threshold})$. Kept as the record for how the solver box was chosen.

## Getting Started

Requires Python 3.12.

1. Create an environment and install the dependencies:
   ```bash
   python3.12 -m venv venv        # conda / uv work too; see the note below
   source venv/bin/activate
   pip install -r requirements.txt
   ```
   This installs Drake 1.56.0 from PyPI — the wheel is sufficient for this experiment,
   which only ever imports `pydrake`. See [`../docs/INSTALL.md`](../docs/INSTALL.md) for
   the other routes.

   The figure wrappers (`scripts/*.sh`) activate `venv/` only if it exists and no other
   environment is already active, so conda and uv users are not forced into a venv named
   `venv`. Set `$PYTHON` to choose the interpreter explicitly. Drake's precompiled binaries bundle a private SNOPT
   build that needs no separate licence when invoked through `SnoptSolver`, so the
   default solver configuration works out of the box. (The reported results were produced
   against a local Drake source build, but the code uses no source-build-specific API. If
   you would rather not depend on SNOPT at all, pass `--solver IPOPT`.)

2. Run the tests:
   ```bash
   python scripts/tests/test_eaik_ift.py
   ```

3. Run the benchmark — start small, since the full configuration takes ~45 minutes:
   ```bash
   python scripts/benchmark_boundary_reach.py --num-targets 5 --num-guesses 3 --seed 42
   ```

All scripts are run from this folder (`ur5e-grasp-selection/`); each appends it to `sys.path` so that
`import src.*` resolves. Results land in `logs/`; figures land in `out/`. Both directories
are gitignored and are created on demand.

## Replicating the Boundary Reachability Benchmark

To replicate the results comparing the **Old** (joint-space), **Direct** (parameter-space EAIK), and **Boundary** (parameter-space EAIK + BRC) formulations on the UR5e robot:

1. **Run the Benchmark**:
   Run the benchmark script with the tuned IFT damping (Direct $\lambda = 10^{-5}$, Boundary $\lambda = 10^{-3}$) and the *derived* boundary-constraint parameters ($\epsilon = 10^{-6}$, threshold $= 10.0$), using 100 targets and 10 initial guesses per target. The solver utilizes SNOPT with $10^{-6}$ tolerances and uses the un-squared Frobenius norm for exact reachability. **Obstacles are on** in this configuration, which is the one the paper's grasp-IK table reports.
   ```bash
   python scripts/benchmark_boundary_reach.py --num-targets 100 --num-guesses 10 --max-wall-time 10.0 --seed 42 --out logs/boundary_benchmark_100t_10g_final.json
   ```
   Add `--no-obstacles` for the secondary, obstacle-free experiment. Use `python -u` if you want to watch progress, since stdout is block-buffered when redirected.

2. **Where $\epsilon$ and the threshold come from**:
   They are *derived from the robot's kinematics and numerics*, not tuned against success rate. `scripts/calibrate_boundary_threshold.py` reproduces the derivation: the Jacobian is length-scaled by $L = 1.12\,\mathrm{m}$ so $\det(JJ^\top)$ has coherent units, $\epsilon = 10^{-6}$ follows from the numerics of that scaled Jacobian, and the threshold follows from the velocity-amplification criterion $\sigma^\* = v_\text{task} / \dot q_\max$.
   ```bash
   python scripts/calibrate_boundary_threshold.py
   ```
   The `--tune` flag on the benchmark sweeps $(\epsilon, \text{threshold})$ against success rate. It is a **diagnostic only** — `verify_solution` re-checks task-space error independently, so a success-rate search can loosen the constraint until it stops constraining.

Results will be printed to stdout and saved in JSON format under the `logs/` directory.

**Two caveats when quoting numbers from a run.** The solver matters: it changes the size of
the minimal-coordinate advantage, so any reported table has to say which solver produced
it. And the wall-clock cap makes success rate and all timings load-dependent — run
anything you intend to quote as a single process on an idle machine, and check `cap_hits`
in the JSON summary to see whether the cap bound at all. Only cost-conditional-on-success
is immune to machine load.

## Replicating the Grasp Selection Figure

The figure shows three UR5e arms grasping the same mug on distinct IK branches, rendered in
Blender with Cycles. It is reproduced by a single command:

```bash
./scripts/reproduce_grasp_figure.sh
```

This pins every parameter that affects the image — the three grasps, the transparencies,
the camera, the resolution and the sample count — and writes `out/grasp_selection.png`
(3000×1688), reproducing the published figure to within render noise. The version in the
paper is a **manual crop** of it (2422×1320); the crop is the one step that is not
scripted.

> **Why the grasps are pinned as configurations rather than as a seed.** The published
> figure was solved by the revision at commit `cd0a354`. Branch selection and the solver's
> joint box both changed afterwards, so **no seed reproduces it** — today's solver
> converges to a different, side-on set of grasps for essentially every seed (measured
> over 45 seeds, all but a few landing on the same azimuths). The three configurations
> were recovered from the archived Meshcat scene by fitting Drake's forward kinematics to
> the exported link transforms, to a residual below 3e-8, and live in
> `scripts/paper_figure_configuration.json`. Drop `--configurations` from
> `reproduce_grasp_figure.sh` to solve fresh grasps with the current solver — that
> produces a *new* figure, not this one.

> **Where the UR link colours come from.** Drake's Meshcat visualizer merges all of a link's
> visual geometries into a single Meshcat object, so the one-`<visual>`-per-colour split in
> `ur5e_drake_collision_obj.urdf` collapses on export and each arm arrives in Blender as seven
> flat-grey link meshes. `assign_ur_link_colors` in `render_grasp_figure.py` rebuilds the split
> in Blender: the merge preserves vertices exactly, so each polygon of the merged mesh is
> matched by centroid to a face of one of the per-colour OBJs on disk and takes that OBJ's
> colour. (Face *order* is not preserved, so the groups cannot be recovered from face counts
> alone — the geometry is what identifies a face.) Watch the `[ur-colours]` log line — it
> reports how many link meshes were recoloured and names any it had to leave grey.
>
> The same export also welds coincident vertices, which averages away the faceted normals the
> `.dae` meshes are authored with and rounds off every crease — the engraved UR logo renders as
> an inflated ridge. `restore_ur_flat_shading` puts the authored shading back; see the
> `[ur-shading]` log line.

### Prerequisites

- **Blender 5.0.1** with nepfaff's `meshcat_html_importer` extension installed (part of
  [`drake-blender-tools`](https://github.com/nepfaff/drake-blender-tools), which installs
  under `~/.config/blender/5.0/extensions/user_default/`).
  If Blender lives somewhere else, set `BLENDER=/path/to/blender`.
- The project venv, for the solve step.

### How the pipeline works

1. `scripts/visualize_grasp_selection.py --no-interactive` solves the three grasps and exports
   the Drake scene as a Meshcat static page (`meshcat.StaticHtml()`, ~58 MB) to
   `out/grasp_selection.html`. `StaticHtml` is the interchange format because the importer maps
   meshcat's `opacity`/`transparent` onto the Principled BSDF `Alpha`, so the arm transparency
   survives the round trip.
2. `scripts/render_grasp_figure.py` runs *inside Blender's own Python*
   (`blender --background --python`), imports that page, restyles the materials, and renders.

### Iterating on the figure

Use `scripts/render_grasp_figure.sh` rather than the reproduce script when you are changing
things — it takes every default from an environment variable:

| Variable | Meaning | Default |
|---|---|---|
| `BLENDER` | Blender executable | `~/opt/blender-5.0.1-linux-x64/blender` |
| `HTML` | input Meshcat scene | `out/grasp_selection.html` |
| `OUT` | output PNG | `out/grasp_selection.png` |
| `SAMPLES` | Cycles samples | `256` |
| `RESOLUTION` | `"WIDTH HEIGHT"` | `"3000 1688"` |
| `FRAME_RADIUS` | metres around the mug to fit | `0.30` |
| `DEVICE` | `GPU` or `CPU` | `GPU` |

```bash
# Re-render the existing scene, no re-solve
./scripts/render_grasp_figure.sh

# Re-solve first, then render
./scripts/render_grasp_figure.sh --solve

# Cheap preview while adjusting framing
RESOLUTION="800 450" SAMPLES=16 OUT=out/preview.png ./scripts/render_grasp_figure.sh
```

`DEVICE=GPU` means OptiX where available, then CUDA/HIP/oneAPI, falling back to CPU with a
printed warning; the `[device] …` line in the log always states what actually rendered. The full
3000×1688 / 256-sample render takes ~50 s on an RTX 3080 Ti Laptop.

Camera placement is a parallax problem — use `--camera-azimuth` / `--camera-elevation` (which
orbit about the mug) rather than raw positions, passing them through after `--`:

```bash
./scripts/render_grasp_figure.sh -- --camera-azimuth 100 --camera-elevation 20
```

### Trying a different set of grasps

Run the visualizer interactively:

```bash
python scripts/visualize_grasp_selection.py
```

It starts a local web server and prints a URL (usually `http://localhost:7000`); click and drag
to orbit. The **"Solve New Random Seed"** button in the Meshcat control panel re-solves for three
distinct grasp branches and updates the view, printing the seed to the terminal. To adopt a new
seed, change `SEED` in `scripts/reproduce_grasp_figure.sh` and drop its `--configurations`
argument, which otherwise pins the published grasps and skips the solve entirely.

## Replicating the Target Scene Figure

The second figure shows what a benchmark *target* is: the full obstacle scene, the arm at a
sampled goal configuration `q_gold`, and the target mug welded at the pose that
configuration induces, `X_WM = FK_tool0(q_gold) · X_EG`. It is a separate pipeline from the
grasp figure, with the same two stages and the same Blender prerequisites
([above](#prerequisites)).

```bash
./scripts/render_target_scene.sh --solve
```

That poses the pinned target, exports the scene, and writes `out/target_scene.png`
(2332×1242, cropped from a 3000×1688 frame). Unlike the grasp figure, this crop *is*
scripted — it is applied as a Cycles render border — so the output needs no manual step.

### How a target is chosen

`scripts/visualize_target_scene.py` draws targets exactly as the benchmark does —
`sample_targets` from `scripts/benchmark_boundary_reach.py`, which rejection-samples
`q_gold ~ U(-π, π)^6` until the arm is collision-free and then takes the mug pose from
forward kinematics — and then applies figure-only filters: the mug must not rise above a
shelf top, and every pair of models except gripper-vs-target-mug must clear 5 mm. That
second filter closes a gap in the benchmark's sampler, which screens the *arm* only,
because the mug is not in the plant when a target is drawn.

```bash
# Draw a fresh target into a scratch pin beside the HTML; the committed pin is untouched.
SEED=7 ./scripts/render_target_scene.sh --resample

# Browse several candidates interactively, then pin the one you want.
python scripts/visualize_target_scene.py --num-scenes 4
```

> **The target is pinned as a configuration, not as a seed.** A seed does not identify a
> target: `--resample` keeps the *first* target its seed accepts, while the committed
> figure is the fourth from seed 42. The configuration lives in
> `scripts/target_scene_configuration.json` and is committed; `--solve` poses it and fails
> if it is missing. `--resample` writes its draw to `out/target_scene.json` and prints the
> `cp` that would promote it — deliberately, because `out/` is gitignored and a pin left
> there is lost on the next clone. That is how the grasp figure was lost once already.

### Iterating on the target-scene figure

`scripts/render_target_scene.sh` takes its defaults from environment variables:

| Variable | Meaning | Default |
|---|---|---|
| `BLENDER` | Blender executable | `~/opt/blender-5.0.1-linux-x64/blender` |
| `HTML` | input Meshcat scene | `out/target_scene.html` |
| `OUT` | output PNG | `out/target_scene.png` |
| `CONFIG` | pinned target | `scripts/target_scene_configuration.json` |
| `SAMPLES` | Cycles samples | `256` |
| `RESOLUTION` | `"WIDTH HEIGHT"` of the uncropped frame | `"3000 1688"` |
| `FRAME_RADIUS` | metres around the target to fit | `0.72` |
| `SEED` | target draw, `--resample` only | `42` |
| `DEVICE` | `GPU` or `CPU` | `GPU` |

```bash
# Re-render the existing scene, no re-solve
./scripts/render_target_scene.sh

# Cheap preview while adjusting framing
RESOLUTION="800 450" SAMPLES=16 OUT=out/preview.png ./scripts/render_target_scene.sh

# A different viewpoint, and the uncropped frame
./scripts/render_target_scene.sh -- --camera-azimuth 95 --camera-elevation 25 --no-crop
```

The default camera is azimuth +169.94°, elevation +17.98°, 3.402 m from `(0.15, 0, 0.42)`.
Unlike the grasp figure, **the crop is scripted**: `--crop` takes left/top/right/bottom as
fractions of the frame, so it survives a change of `RESOLUTION`, and it is applied as a
Cycles render border with `use_crop_to_border` — the border is all that gets rendered, so
cropping is free rather than wasted pixels. `--no-crop` returns the full frame.

The obstacle scene is `models/ur5e_collision_obj.yaml`: the benchmark's
`models/ur5e_collision.yaml` with the arm and gripper swapped for the per-colour `_obj`
variants, which is what carries the UR link colours into Blender. The target mug is not in
the YAML — it is welded in at runtime at the sampled pose.

## What a fresh render writes

Both pipelines write only into `out/`, which is gitignored:

| Pipeline | Files written |
|---|---|
| Grasp selection | `out/grasp_selection.html` (~146 MB), `out/grasp_selection.png` |
| Target scene | `out/target_scene.html` (~52 MB), `out/target_scene.json`, `out/target_scene.png` |

The paper's rendered PNGs are not committed here; both regenerate from the pinned
configurations (`scripts/paper_figure_configuration.json`,
`scripts/target_scene_configuration.json`), which are what make the published grasps and
target reproducible rather than resampled. Both visualizer scripts also rewrite their scene YAML
(`models/ur5e_single_arm_on_table.yaml`, `models/ur5e_three_arms.yaml`) from a heredoc and
regenerate the per-colour OBJs and `ur5e_drake_collision_obj.urdf` if they are missing; all
of those are committed and byte-identical to what a run produces, so a render leaves the
working tree clean.

## Licence and third-party content

This project's own source is released under the MIT Licence — see
[`../LICENSE`](../LICENSE). Several assets under `models/` are third party and keep their
own terms; [`THIRD_PARTY.md`](THIRD_PARTY.md) lists each one with its origin and licence.

## Citation

See the [top-level README](../README.md#citation).

## Questions

Please open an issue on this repository.
