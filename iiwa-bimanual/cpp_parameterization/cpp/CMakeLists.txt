cmake_minimum_required(VERSION 3.18)
project(iiwa_ik)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type" FORCE)
endif()

# Option: enable aggressive compiler optimizations for maximum speed.
# Usage: cmake -S cpp -B build -DCMAKE_PREFIX_PATH=... -DOPTIMIZED_BUILD=ON
# See README.md for details and caveats.
option(OPTIMIZED_BUILD "Enable aggressive compiler optimizations (LTO, fast-math, vectorization, etc.)" OFF)

if(OPTIMIZED_BUILD)
  message(STATUS "OPTIMIZED_BUILD=ON: enabling aggressive optimization flags.")
  set(OPT_FLAGS "-O3 -g -flto -fstack-protector-strong -D_FORTIFY_SOURCE=2 -ffast-math -fno-math-errno -funroll-loops -finline-small-functions -fprefetch-loop-arrays -fstrict-aliasing")
  set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   ${OPT_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OPT_FLAGS} -DEIGEN_NO_DEBUG -DEIGEN_VECTORIZE")
endif()

# User must supply the Drake installation via CMAKE_PREFIX_PATH.
# Example:
#   cmake -S cpp -B build -DCMAKE_PREFIX_PATH=/path/to/drake/install
find_package(drake REQUIRED)
message(STATUS "drake found at: ${drake_DIR}")

# Ensure cmake finds Drake's version of pybind11.
list(PREPEND CMAKE_PREFIX_PATH ${drake_DIR}/../)
find_package(pybind11 REQUIRED)
message(STATUS "pybind11 found at: ${pybind11_DIR}")

# Build module named _iiwa_ik (not iiwa_ik) so Python package exports it cleanly
pybind11_add_module(_iiwa_ik
    iiwa_ik/bindings.cc
    iiwa_ik/parameterization.cc
    iiwa_ik/constraints.cc
    iiwa_ik/costs.cc
    iiwa_ik/iiwa_analytic_ik.cc
)

target_link_libraries(_iiwa_ik PRIVATE drake::drake)

# Output the shared library directly into the Python package directory:
#   <repo-root>/cpp_parameterization/python/iiwa_ik/
set(PY_PACKAGE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../python/iiwa_ik)

set_target_properties(_iiwa_ik
    PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${PY_PACKAGE_DIR}
)

enable_testing()
add_subdirectory(tests)