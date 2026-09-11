# Installation

Everything here depends on [Drake](https://drake.mit.edu). This release is pinned to
**Drake 1.56.0**. There is one real decision to make — *which kind* of Drake install you need —
and it depends on which experiment you want to run.

If you would rather not make that decision at all, use the Docker image: it is one command and
contains all three experiments, already built. See [DOCKER.md](DOCKER.md).

---

## Which route do I need?

| I want to… | Route |
| --- | --- |
| Run the UR5e grasp-selection benchmark | **1** (pip wheel) |
| Run RB-Y1 planning, verification and the report scripts | **1** (pip wheel) |
| Build `_iiwa_ik` and run the bimanual IIWA experiments | **2** (binary tarball) or **3** (source) |
| Rebuild the supplementary video | **2** or **3**, and the *same* install `_iiwa_ik` was built against |
| Reproduce everything from one install | **2** |

The bimanual IIWA experiment compiles a pybind11 extension against Drake's C++ library. That
needs a Drake *installation* carrying `lib/cmake/drake/drake-config.cmake` — which the pip wheel
does not provide. The other two experiments only ever `import pydrake`, so the wheel is enough.

---

## Route 1 — pip wheel

Serves the UR5e experiment and the RB-Y1 planning/report path. Not the IIWA C++, not the video.

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install 'drake==1.56.0'
python -c "import pydrake; from pydrake.planning import IrisNp2Options; print('ok', pydrake.__file__)"
```

## Route 2 — binary tarball (recommended; serves all three)

Gives you `drake-config.cmake` *and* a matching `pydrake`, so one install covers everything.

```bash
# Pick the tarball matching your Ubuntu VERSION_CODENAME (noble = 24.04).
CODENAME=$(. /etc/os-release; echo "$VERSION_CODENAME")
curl -fsSL -o /tmp/drake.tar.gz \
  "https://github.com/RobotLocomotion/drake/releases/download/v1.56.0/drake-1.56.0-${CODENAME}.tar.gz"
sudo tar -xzf /tmp/drake.tar.gz -C /opt          # -> /opt/drake
sudo /opt/drake/share/drake/setup/install_prereqs

export DRAKE_INSTALL_DIR=/opt/drake
export PYTHONPATH="$DRAKE_INSTALL_DIR/lib/python3.12/site-packages${PYTHONPATH:+:$PYTHONPATH}"

test -f "$DRAKE_INSTALL_DIR/lib/cmake/drake/drake-config.cmake" && echo "install OK"
python3 -c "import pydrake; print(pydrake.__file__)"   # must be under $DRAKE_INSTALL_DIR
```

Note that `VERSION_CODENAME` is read rather than `ID`: on an Ubuntu derivative, `ID` is the
derivative's name and there is no tarball under it.

## Route 3 — source build

What the paper's numbers were originally produced with. Serves everything, takes a while.

```bash
git clone https://github.com/RobotLocomotion/drake.git && cd drake
git checkout v1.56.0
sudo ./setup/ubuntu/install_prereqs.sh
mkdir ../drake-build && cd ../drake-build
cmake -DCMAKE_INSTALL_PREFIX=$HOME/opt/drake -DCMAKE_BUILD_TYPE=Release ../drake
make -j$(nproc) install

export DRAKE_INSTALL_DIR=$HOME/opt/drake
export PYTHONPATH="$DRAKE_INSTALL_DIR/lib/python3.12/site-packages${PYTHONPATH:+:$PYTHONPATH}"
```

A source build reports `importlib.metadata.version("drake") == "unknown"` — the install path is
the only identifier it carries.

---

## Do not mix the two kinds of Drake

> If `$DRAKE_INSTALL_DIR` is a source or binary install, put **its** `site-packages` on
> `PYTHONPATH` and do **not** also have a `drake` wheel installed in the active environment. An
> install on `PYTHONPATH` silently shadows the wheel, and an extension compiled against one Drake
> and imported alongside a different `pydrake` fails with:
>
> ```
> ImportError: generic_type: type "IiwaBimanualReachableConstraint"
>              referenced unknown base type "drake::solvers::Constraint"
> ```
>
> That is two pybind11 type registries in one process, not a build error. Check with
> `python3 -c "import pydrake; print(pydrake.__file__)"` and confirm the path is the one you
> expect.

---

## Per-experiment setup

Each experiment folder is a **self-contained working root** — `cd` into it before running
anything. Paths inside each are resolved relative to the folder, so commands run from elsewhere
will not find the models.

```bash
# Bimanual IIWA  (needs route 2 or 3)
cd iiwa-bimanual
python3.12 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt          # note: this does NOT install drake, by design
cmake -S cpp_parameterization/cpp -B cpp_parameterization/build \
      -DCMAKE_PREFIX_PATH=$DRAKE_INSTALL_DIR -DOPTIMIZED_BUILD=ON
cmake --build cpp_parameterization/build -j$(nproc)
ctest --test-dir cpp_parameterization/build          # 232 tests

# UR5e grasp selection  (route 1 is enough)
cd ../ur5e-grasp-selection
python3.12 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python scripts/tests/test_eaik_ift.py

# RB-Y1 whole body  (route 1 is enough; Python 3.12 exactly)
cd ../rby1-whole-body
python3.12 -m venv .venv && . .venv/bin/activate
pip install -e .                          # also compiles the two IKFast extensions
python -c "import rainbow_left_arm_ik, rainbow_right_arm_ik; print('IKFast ok')"
```

`OPTIMIZED_BUILD=ON` is what the paper's runtime numbers were produced with. Do **not** add
`-march=native`: it breaks Eigen's alignment assumptions against pybind11. Rebuild after any C++
change — a stale `.so` will happily benchmark the old behaviour.

---

## Solvers, and where Mosek fits in

Drake's binary distribution bundles **SNOPT** under a licence that needs nothing extra from you
when it is invoked through `SnoptSolver`, and bundles **Clarabel** and **IPOPT** as open
alternatives. Everything in this release runs on those alone — no commercial licence is required
to reproduce any experiment.

**Mosek is optional and interchangeable with Clarabel.** Drake dispatches the conic solves — the
maximum-volume inscribed ellipsoid inside IRIS-NP2, and the GCS relaxations — to Mosek when a
licence is present and to Clarabel when it is not. The paper's numbers were produced on a machine
with a Mosek licence, so **timings will differ without one**; the pipeline itself runs either way.
This is a specific case of the general caveat that absolute runtimes are not the reproducible
quantity — the ratios between rows are.

If you do have a licence (Mosek offers [free academic
ones](https://www.mosek.com/products/academic-licenses/)), point `MOSEKLM_LICENSE_FILE` at it:

```bash
python3 -c "from pydrake.solvers import MosekSolver; print(MosekSolver().enabled())"
```

### One test needs Mosek

`scripts/tests/test_pipeline_integration.py::test_pipeline_smoke` sets
`configuration_space_margin = 0`, which can leave IRIS with a region that has no interior. Mosek
tolerates it; Clarabel rejects it:

```
RuntimeError: Solver Clarabel failed to solve the maximum inscribed ellipse problem;
it terminated with SolutionResult SolverSpecificError).
```

So without a Mosek licence that one test errors and the other 24 pass. The experiments themselves
are unaffected — `run_full_comparison.py` uses `relax_margin = True` with a nonzero margin and
completes on Clarabel, as does `scripts/tests/test_smoke_iris.py`, which runs a full IRIS region.

---

## Blender (figures and video only)

The swept-volume figure, both UR5e figures, and the video render through Blender **5.0.x**. It is
found via `$BLENDER`, then `blender` on `PATH`. The Meshcat→Blender add-on is vendored at
`third_party/meshcat_html_importer` and is installed into Blender's extension directory by:

```bash
python3 iiwa-bimanual/scripts/figures/install_meshcat_importer.py          # install
python3 iiwa-bimanual/scripts/figures/install_meshcat_importer.py --check  # verify it matches
```

Cycles uses a GPU where one is available and falls back to CPU where it is not. On CPU the
renders are slow but correct — and for the swept-volume figure specifically, CPU Cycles is
*required* for a correct result, because the figure is stacked semi-transparent geometry and
EEVEE has no equivalent of `transparent_max_bounces`.

---

## A note on the version pin

Drake 1.56.0 is the version this release is validated against: the IIWA C++ extension builds
against it and its 232 C++ tests pass.

The original results were produced against untagged local source builds (the IIWA and UR5e
experiments) and a Drake nightly (the RB-Y1 experiment), so 1.56.0 is a reconstruction rather
than the pin that was in use at the time. The RB-Y1 project's own re-baseline recorded that
nightly as behaviourally identical to the neighbouring stable release, reproducing three canary
points' durations exactly.

Absolute runtimes are machine- and build-dependent — a 20–30% spread from hardware alone is
normal. **The ratios between rows are the reproducible quantity, not the absolute seconds.**
