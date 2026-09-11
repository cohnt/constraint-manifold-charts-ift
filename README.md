<h1 align="center">Planning along Differentiable Charts of Constraint Manifolds<br>with General-Purpose IK Solvers</h1>

<p align="center">
  Thomas Cohn*, Seiji Shaw*, Harel Biggie, Travis Manderson, Nicholas Roy, Russ Tedrake
  <br>
  <em>Computer Science and Artificial Intelligence Laboratory, MIT</em>
  <br>
  <sub>* equal contribution</sub>
</p>

<p align="center">
  <a href="https://tommycohn.com/inverse-function-theorem-parameterization/">Project page</a> ·
  <a href="https://arxiv.org/abs/2609.10905">arXiv</a> ·
  <a href="https://www.youtube.com/watch?v=ADF4g3iQsuY">Video</a>
</p>

> Planning trajectories for robot manipulators under kinematic equality constraints restricts
> feasible motions to a measure-zero submanifold of the configuration space, requiring special
> algorithmic treatment. A promising strategy is parametrizing the set of feasible configurations
> using analytic inverse kinematics (IK). Bespoke analytic IK functions can be written to be
> differentiable, a necessary property for gradient-based trajectory optimization. But the vast
> majority of IK functions are computed by automated meta-solvers like IKFast, and are difficult to
> modify for differentiability. We present a new approach for computing gradients of analytic IK
> parameterizations: we leverage the inverse function theorem to recover the desired gradients from
> the ordinary forward kinematic Jacobian. Furthermore, we present a least-squares domain extension
> and an optimization-amenable description of the reachability constraint, which preserves gradient
> signal outside the reachable workspace. We demonstrate the efficacy of our approach through
> numerical experiments and downstream tasks, including a hardware demonstration of an RB-Y1
> picking up a box and placing it on a table.

**Status.** This paper is under review. Code is released under the MIT license; the third-party
robot descriptions and mesh geometry redistributed alongside it carry their own terms — see
[`LICENSE`](LICENSE) and [`THIRD_PARTY.md`](THIRD_PARTY.md).

---

## What's here

Three experiments, one folder each. Each folder is a **self-contained working root** — `cd` into
it before running anything.

| Paper artifact | Folder | Command | Wall time |
| --- | --- | --- | --- |
| **Downstream runtime table** — 17 gradient × reachability configurations through IRIS-NP2 → GCS → RRT+shortcut → trajopt → TOPPRA | `iiwa-bimanual` | `python3 scripts/experiments/run_full_comparison.py --num-rrt-trials 10`<br>`python3 scripts/analysis/plot_pipeline_comparison.py` | 1–1.5 h |
| **Forward-mode autodiff comparison** — error and runtime vs. number of partials, 10 000 samples | `iiwa-bimanual` | `python3 scripts/experiments/run_gradient_study.py`<br>`python3 scripts/analysis/plot_gradient_study.py` | ~10 min |
| **Boundary-reachability parameters** (τ, ε, L) — derived, not tuned | `iiwa-bimanual` | `python3 scripts/analysis/calibrate_boundary_threshold.py` | seconds |
| **Swept-volume figure** | `iiwa-bimanual` | `python3 scripts/figures/render_swept_volume.py` | ~1 min/orbit |
| **Grasp-IK table** — Old / Direct / Boundary on the UR5e, 100 targets × 10 guesses | `ur5e-grasp-selection` | `python scripts/benchmark_boundary_reach.py --num-targets 100 --num-guesses 10 --max-wall-time 10.0 --seed 42` | ~45 min |
| **Grasp-selection figure** | `ur5e-grasp-selection` | `./scripts/reproduce_grasp_figure.sh` | ~2 min |
| **Target-scene figure** | `ur5e-grasp-selection` | `./scripts/render_target_scene.sh --solve` | ~2 min |
| **Full-body IK table** — success rate, solve time and constraint violation on RB-Y1 hardware | `rby1-whole-body` | `python scripts/timing_report.py --log plans/grid_cache/timing/run_20260810-222505.jsonl`<br>`python scripts/ee_constraint_report.py`<br>`python scripts/grid_guarantee_report.py` | ~2 min (reads committed artifacts) |
| **Supplementary video** | `rby1-whole-body` | `python scripts/video/build_video.py overview` | ~1 h |

## Install

Read [`docs/INSTALL.md`](docs/INSTALL.md). There is one decision — which kind of Drake install you
need — and it depends on which experiment you want. Everything is pinned to **Drake 1.56.0**.

Or skip the decision entirely and use the prebuilt container, which has all three experiments
already built and needs no Drake install of your own:

```bash
docker pull cohnt/constraint-manifold-charts-ift:v1
docker run --rm -it cohnt/constraint-manifold-charts-ift:v1
```

`:v1` is the immutable tag matching arXiv v1; `:latest` moves with this repository. Building the
same image yourself from a clone is `docker build -t cohnt/constraint-manifold-charts-ift .`. The
image is `linux/amd64`; on Apple Silicon see the platform note in
[`docs/DOCKER.md`](docs/DOCKER.md), which also covers what does and does not work in the
container.

## The three experiments

**[`iiwa-bimanual/`](iiwa-bimanual/)** — Two 7-DOF IIWA arms holding a common object, so their
end-effector transform is fixed. An 8-dimensional chart `q̃ = [q_controlled(7), ψ]` replaces the
14-dimensional configuration space, and the equality constraint is eliminated by construction.
This folder carries the C++ core: the analytic IK, the chart and its derivatives, the constraints
and the costs. It produces the downstream runtime table, the forward-mode autodiff comparison,
and the swept-volume figure.

**[`ur5e-grasp-selection/`](ur5e-grasp-selection/)** — A UR5e selecting a grasp on a mug, with IK
supplied by [EAIK](https://github.com/OstermD/EAIK) rather than by a hand-written differentiable
function. Compares three formulations — the unparameterized baseline, the direct IFT gradient, and
the boundary-reachability formulation — on 100 targets × 10 initial guesses.

**[`rby1-whole-body/`](rby1-whole-body/)** — A 23-DOF Rainbow Robotics RB-Y1 (holonomic base,
6-DOF torso, two 7-DOF arms) picking up a box and placing it on a table, with the gripper-to-gripper
transform frozen while carrying. IK comes from IKFast, which cannot be modified for
differentiability. Executed on hardware: 20/20 grid points planned, 20/20 executed cleanly.

## Reproduction notes

- **No commercial solver is required.** Drake sends the conic solves to Mosek when licensed and
  to Clarabel otherwise; the paper's timings were measured with Mosek, so absolute numbers will
  differ without it. One IIWA test does need Mosek — see [`docs/INSTALL.md`](docs/INSTALL.md).
- Absolute runtimes are machine- and build-dependent; a 20–30% spread from hardware alone is
  normal. **The ratios between rows are the reproducible quantity, not the absolute seconds.**
- Run benchmarks as a single process on an idle machine. Success rates and timings are
  load-dependent; only cost-conditional-on-success is not.
- The RB-Y1 hardware run cannot be repeated. Its 20 execution records ship in
  `rby1-whole-body/results/`, and for anything quoted as an execution result those records *are*
  the result.
- The hardware half of the supplementary video is composited from 1.9 GB of raw capture that is
  not distributable and is not in this repository. The simulation cut rebuilds fully; the hardware
  cut cannot. See that folder's README.

## Citation

```bibtex
@article{cohn2026planning,
  title   = {Planning along Differentiable Charts of Constraint Manifolds
             with General-Purpose IK Solvers},
  author  = {Cohn, Thomas and Shaw, Seiji and Biggie, Harel and
             Manderson, Travis and Roy, Nicholas and Tedrake, Russ},
  journal = {arXiv preprint arXiv:2609.10905},
  year    = {2026},
  note    = {Thomas Cohn and Seiji Shaw contributed equally.}
}
```

## License and third-party content

MIT for this project's own source — see [`LICENSE`](LICENSE). The robot descriptions, mesh
geometry, vendored Blender add-on and IKFast-generated sources redistributed here each carry their
own terms, listed in [`THIRD_PARTY.md`](THIRD_PARTY.md) along with the items whose licensing we
have not been able to resolve.

## Acknowledgements

Built on [Drake](https://drake.mit.edu). Analytic IK from [EAIK](https://github.com/OstermD/EAIK)
and [IKFast](http://openrave.org/docs/latest_stable/openravepy/ikfast/). Robot descriptions from
[Rainbow Robotics](https://www.rainbow-robotics.com/), [ROS-Industrial's
`universal_robot`](https://github.com/ros-industrial/universal_robot) and
[`iiwa_stack`](https://github.com/IFL-CAMP/iiwa_stack). Meshcat→Blender import via
[`drake-blender-tools`](https://github.com/nepfaff/drake-blender-tools).

This work was supported by the National Science Foundation Graduate Research Fellowship Program
under Grant No. 2141064, MIT Siegel Family Quest for Intelligence, the Natural Sciences and
Engineering Research Council of Canada (NSERC), and the Army Research Laboratory under Cooperative
Agreement Number W911NF-17-2-0181. Any opinions, findings, and conclusions or recommendations
expressed in this material are those of the authors and do not necessarily reflect the views of
the National Science Foundation or the other sponsors acknowledged in this work.

## Questions

Please open an issue on this repository.
