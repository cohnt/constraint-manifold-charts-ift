# Building and Testing

To build this C++ project, you will have to download and use the [compiled Drake binaries](https://github.com/RobotLocomotion/drake/releases) or [compile Drake from source](https://drake.mit.edu/from_source.html).
But it's worth it for the major speedups!

All commands in this README are run from the **repository root**, matching the
[top-level README](../README.md); there is a single build directory,
`cpp_parameterization/build`.

## Source Code Structure

The C++ implementation is organized into several key components:

### Core Implementation (`cpp/iiwa_ik/`)
| File | Description |
| --- | --- |
| `iiwa_analytic_ik.h` / `.cc` | Analytic IK for a single IIWA arm, including $\psi$ calculation and gradient-stable `arccos` clipping. |
| `parameterization.h` / `.cc` | Bimanual parameterization mapping (8-DOF → 14-DOF). Supports standard AutoDiff and IFT-based gradients. |
| `constraints.h` / `.cc` | Planning constraints (reachability, joint limits, collision-free) in parameterized space. |
| `costs.h` / `.cc` | Quadratic path costs in parameterized space. |
| `bindings.cc` | Pybind11 bindings for exposing all components to Python. |

### Configuration Structs
- **`BimanualConfig`**: Centralizes the physical and numerical parameters of the bimanual system:
  - `shoulder_up`, `elbow_up`, `wrist_up`: Boolean flags for the analytic IK solution branch.
  - `grasp_distance`: Relative distance between end-effectors.
  - `clipping_margin`: Numerical safety margin for joint `arccos` operations.
  - `clipping_margin_psi`: Numerical safety margin for $\psi$ `arccos` operations.
- **`AutoDiffConfig`**: Configures the automatic differentiation strategy:
  - `use_ift`: Whether to use the Implicit Function Theorem for gradients.
  - `ift_handling`: Singularity handling mode -- `kZero`, `kPseudoinverse`,
    `kLevenbergMarquardt`, `kResidualDamping`, or `kFullNewton`.
  - `lambda_`, `svt_epsilon`, `svt_lambda_max`, `use_anisotropic_damping`: damping parameters for
    those modes. Note the trailing underscore on `lambda_`: `lambda` is a reserved word in Python and
    could not otherwise be passed by keyword.

### C++ Unit Tests (`cpp/tests/`)
Build the `tests` target to produce these binaries in `build/tests/`:
| Binary | Tests |
| --- | --- |
| `test_iiwa_analytic_ik` | Forward and Inverse Kinematics for a single arm. |
| `test_parameterization` | The 8-DOF bimanual mapping. |
| `test_gradients` | Gradient accuracy across all IK configurations. |
| `test_ift` | Implicit Function Theorem logic and refinement. |
| `test_constraints` | Reachability and joint limit constraint evaluations. |
| `test_costs` | Path cost evaluations and gradients. |
| `test_regularization` | Singularity handling and damping behavior. |
| `test_newton_hessian` | Analytical Hessian implementation for Full Newton IFT. |

## Building

The build system offers two modes, controlled by the `OPTIMIZED_BUILD` CMake option.

### Basic Build

Builds with the default compiler flags (`RelWithDebInfo`). Good for development and debugging.

```bash
export DRAKE_INSTALL_DIR=/path/to/drake/installation
cmake -S cpp_parameterization/cpp -B cpp_parameterization/build -DCMAKE_PREFIX_PATH=$DRAKE_INSTALL_DIR
cmake --build cpp_parameterization/build --target _iiwa_ik -j$(nproc)
```

### Optimized Build

Enables aggressive compiler optimizations for maximum runtime speed.
Pass `-DOPTIMIZED_BUILD=ON` to enable LTO, `-ffast-math`, vectorization, and related flags.

```bash
export DRAKE_INSTALL_DIR=/path/to/drake/installation
cmake -S cpp_parameterization/cpp -B cpp_parameterization/build -DCMAKE_PREFIX_PATH=$DRAKE_INSTALL_DIR -DOPTIMIZED_BUILD=ON
cmake --build cpp_parameterization/build --target _iiwa_ik -j$(nproc)
```

This is the configuration the paper's runtimes were measured with.

**Warning:** Do not use `-march=native` — it can cause errors related to Eigen allocation and pybind11.

## Testing

### Python Integration Tests

After building the bindings, run the Python test suite from the **root directory** to verify bindings and algorithms:

```bash
# From the root directory
python3 -m unittest discover scripts/tests
```

Or run individual test files:
```bash
python3 -m unittest scripts.tests.test_bindings
python3 -m unittest scripts.tests.test_autodiff_low_level
python3 -m unittest scripts.tests.test_constraints
```

### C++ Unit Tests

The project uses [Google Test](https://github.com/google/googletest) for C++ unit testing.
Build all targets, then run the full suite with `ctest`:

```bash
export DRAKE_INSTALL_DIR=/path/to/drake/installation
cmake -S cpp_parameterization/cpp -B cpp_parameterization/build -DCMAKE_PREFIX_PATH=$DRAKE_INSTALL_DIR
cmake --build cpp_parameterization/build -j$(nproc)
ctest --test-dir cpp_parameterization/build --output-on-failure
```

Individual test binaries are also available in `cpp_parameterization/build/tests/` if you want to run
a specific suite; they are the eight listed in the table above.

# A Complete Build-and-Run Recipe

This assumes you have just cloned the repository, you have an appropriate Drake installation, and are starting at the **root directory**.

```bash
# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Set the Drake installation path
export DRAKE_INSTALL_DIR=/path/to/drake/installation

# Install remaining Python dependencies
pip install numpy scipy matplotlib pandas tqdm networkx pyyaml jupyter ipywidgets pydot

# Point PYTHONPATH to Drake's Python bindings
export PYTHONPATH=$DRAKE_INSTALL_DIR/lib/python3.$(python3 -c 'import sys; print(sys.version_info.minor)')/site-packages:$PYTHONPATH

# Build the C++ project (OPTIMIZED_BUILD=ON is what the paper's numbers used)
cmake -S cpp_parameterization/cpp -B cpp_parameterization/build \
      -DCMAKE_PREFIX_PATH=$DRAKE_INSTALL_DIR -DOPTIMIZED_BUILD=ON
cmake --build cpp_parameterization/build -j$(nproc)

# Verify everything is working
ctest --test-dir cpp_parameterization/build --output-on-failure
python3 -m unittest discover scripts/tests

# Launch the walkthrough notebook
jupyter notebook notebooks/main_cpp.ipynb
```

From here, open `notebooks/main_cpp.ipynb` and all code should run.