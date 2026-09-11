# Reproducing in Docker

One image contains all three experiments, with Drake 1.56.0 installed, the IIWA C++ extension
already compiled, and the RB-Y1 IKFast extensions already built.

A prebuilt image is published on Docker Hub, so nothing needs compiling locally:

```bash
docker pull cohnt/constraint-manifold-charts-ift:v1
docker run --rm -it cohnt/constraint-manifold-charts-ift:v1
```

Two tags are published, from
[hub.docker.com/r/cohnt/constraint-manifold-charts-ift](https://hub.docker.com/r/cohnt/constraint-manifold-charts-ift):

| Tag | Meaning |
| --- | --- |
| `:v1` | Immutable. The image the paper's arXiv v1 was verified against — use this to reproduce. |
| `:latest` | Moves with the repository. |

Building it yourself from a clone gives the same thing and takes roughly 15 minutes:

```bash
docker build -t cohnt/constraint-manifold-charts-ift .
docker run --rm -it cohnt/constraint-manifold-charts-ift
```

## Platform: x86-64 only, so far

The published image is `linux/amd64`, and x86-64 is the only architecture any of this has been
tested on — the paper's numbers included. On an Apple Silicon Mac or another arm64 host:

- **`docker pull` works, but the image runs emulated.** The report scripts, which mostly read the
  committed caches, may well be fine. The benchmarks will be far slower than native, and their
  timings are meaningless as measurements — do not compare them against the paper's.
- **`docker build` fails.** The Drake step fetches `drake-${DRAKE_VERSION}-noble.tar.gz`, which is
  the x86-64 tarball. Drake publishes `drake-${DRAKE_VERSION}-noble-arm64.tar.gz` next to it, so
  swapping that one filename is likely the entire fix — it simply has not been tried here.

If you attempt either, please [open an
issue](https://github.com/cohnt/constraint-manifold-charts-ift/issues) and say how it went. **A
report that arm64 works is just as valuable as a report that it doesn't** — if it turns out to be
fine, we would rather document that than leave this caveat standing.

The image is built from your local checkout — the Docker build context — rather than by cloning
inside the container. It therefore contains whatever is in your working tree, minus what
`.dockerignore` excludes, not necessarily what is committed. Build from a clean checkout if you
want the image to correspond to a published commit.

Dependency installs, the IIWA C++ build and the IKFast build each sit behind their own narrow
`COPY`, so editing documentation or a Python script rebuilds in seconds rather than redoing the
multi-minute LTO compile.

Inside the container, `use` switches between the three experiments:

```
use iiwa    # bimanual IIWA        -> /release/iiwa-bimanual
use ur5e    # UR5e grasp selection -> /release/ur5e-grasp-selection
use rby1    # RB-Y1 whole body     -> /release/rby1-whole-body
```

Each has its own virtualenv. All three share the single Drake installation at `/opt/drake`, which
is on `PYTHONPATH`. That is deliberate: with exactly one `pydrake` in the image, the pybind11 ABI
mismatch described in [INSTALL.md](INSTALL.md) cannot happen.

**Run `use` before anything else.** It is what puts that experiment's interpreter first on `PATH`.
The figure wrappers (`ur5e-grasp-selection/scripts/*.sh`) invoke `$PYTHON`, defaulting to
`python3`, so without `use` they pick up the system interpreter and fail with
`ModuleNotFoundError: No module named 'eaik'`. Either `use ur5e` first, or set the interpreter
explicitly:

```bash
PYTHON=/opt/venv/ur5e/bin/python ./scripts/reproduce_grasp_figure.sh
```

## The cheap checks first

```bash
use iiwa && ctest --test-dir cpp_parameterization/build     # 232 tests, ~5 s
use ur5e && python scripts/tests/test_eaik_ift.py           # 7 tests
use rby1 && python scripts/grid_guarantee_report.py         # reads the committed plan cache
```

## Build options

| Build arg | Default | Effect |
| --- | --- | --- |
| `WITH_BLENDER` | `true` | Blender 5.0.1, needed for every figure. Adds ~700 MB. |
| `WITH_VIDEO` | `false` | LaTeX + manim + nbconvert, needed **only** to rebuild the supplementary video. Adds several GB, because manim renders every equation through LaTeX. |
| `DRAKE_VERSION` | `1.56.0` | Do not change without re-running the verification in [INSTALL.md](INSTALL.md). |
| `BLENDER_VERSION` / `BLENDER_SHA256` | `5.0.1` / its published sha256 | Change both together, or set the checksum empty to skip verification. |

```bash
docker build --build-arg WITH_VIDEO=true -t cohnt/constraint-manifold-charts-ift:video .    # if you want the video stack
docker build --build-arg WITH_BLENDER=false -t cohnt/constraint-manifold-charts-ift:slim .  # benchmarks/tables only
```

Blender is fetched from `download.blender.org` if that host will serve it and from one of Blender's
listed mirrors otherwise — the official host sits behind a Cloudflare challenge that answers `403`
to non-browser clients, which would otherwise break the build outright. Whichever mirror serves it,
the archive is checked against the officially published sha256 before it is unpacked.

## What works in the container, and what does not

**Runs normally.** Every benchmark, table, report and verification script for the UR5e and RB-Y1
experiments, and the IIWA autodiff study, boundary-parameter derivation and swept-volume figure.
These are CPU-only and need no display.

**One test needs a Mosek licence.** The image ships none, so Drake uses Clarabel — fine for every
experiment, but `scripts/tests/test_pipeline_integration.py::test_pipeline_smoke` fails here with
`Solver Clarabel failed to solve the maximum inscribed ellipse problem`, leaving 24/25. Timings
also differ from the paper's, which were measured with Mosek. Mount a licence if you have one:

```bash
docker run --rm -it \
  -v "$HOME/mosek/mosek.lic:/opt/mosek/mosek.lic:ro" \
  -e MOSEKLM_LICENSE_FILE=/opt/mosek/mosek.lic cohnt/constraint-manifold-charts-ift
```

See [INSTALL.md](INSTALL.md) for why.

**Runs, but slowly.** All the Blender figures. Cycles falls back to CPU rendering unless the
container has GPU access, because the image ships no CUDA/OptiX runtime. The output is correct —
just much slower than the ~1 min/orbit quoted for a discrete GPU. For the swept-volume figure CPU
Cycles is in any case the *correct* path, since the figure is stacked semi-transparent geometry
and EEVEE cannot composite it properly.

To try for GPU rendering, pass the NVIDIA runtime through:

```bash
docker run --rm -it --gpus all cohnt/constraint-manifold-charts-ift
```

Whether Cycles then finds OptiX depends on your host driver and the NVIDIA container toolkit; the
render scripts print the engine and device they actually selected, so check that line rather than
assuming. A silent CPU fallback looks exactly like success.

**Cannot be rebuilt at all: the hardware half of the supplementary video.** It is composited from
1.9 GB of raw RB-Y1 capture from the 2026-08-11 session, which is not distributable and is not in
this repository. `build_video.py ral` will detect the missing clips and refuse rather than
substitute other footage — that refusal is the correct behaviour, not a failure. The simulation
cut (`build_video.py overview`) rebuilds fully. See the RB-Y1 folder's README.

## Results out of the container

Everything is written inside `/release`, which is lost when the container exits. Mount a host
directory if you want to keep results:

```bash
docker run --rm -it -v "$PWD/results:/release/out" cohnt/constraint-manifold-charts-ift
```

Meshcat previews serve on port 7000; add `-p 7000:7000` and open the printed URL if you want to
inspect a plan interactively.
