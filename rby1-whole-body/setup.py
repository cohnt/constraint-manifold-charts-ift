"""
References:
-----------
- cmake_example
    - https://github.com/pybind/cmake_example
    - https://github.com/pybind/cmake_example/blob/835e1a81b01d06097ccbb7b8f214ef9bd2d0c159/setup.py
- libigl
    - https://stackoverflow.com/q/75430008/1814274
    - https://github.com/libigl/libigl-python-bindings
- Other resources on StackOverflow:
    - https://stackoverflow.com/q/47599162/1814274
    - https://stackoverflow.com/a/20548189/1814274

Run:
----
- pdm:
    - For development: [[ `pdm install -v` ]]
        - Produces a `build` directory in the root.
        - Produces `.so` files in `src`.
        - Doesn't rebuild an already built `build` directory.
    - For development: [[ `pdm install --prod -v` ]]
        - Produces a `build` directory in the root.
        - Produces `.so` files in `src`.
        - Doesn't rebuild an already built `build` directory.
- pip
    - For development: [[ `pip install -e . -v` ]]
        - Produces a `build` directory in the root.
        - Produces `.so` files in source directories.
        - Doesn't rebuild an already built `build` directory.
    - For production: [[ `pip install . -v` ]]
        - Doesn't produces a build directory -- uses a /tmp directory.
        - Produces `.egg-info` files in `src`.
        - Rebuilds in a /tmp directory every time.
"""

import os
from pathlib import Path

from setuptools import Extension, setup


# A CMakeExtension needs a sourcedir instead of a file list.
# The name must be the _single_ output extension from the CMake build.
class CMakeExtension(Extension):
    def __init__(self, name: str, sourcedir: str = "", sources=[]) -> None:
        super().__init__(name, sources=sources)
        self.sourcedir = os.fspath(Path(sourcedir).resolve())



# Compile logic for Rainbow IKFast cpp files.
setup(
    ext_modules=[
        CMakeExtension(
            name="rainbow_left_arm_ik",
            sourcedir="cpp_parameterization/cpp/rby1_ik/rainbow_left_arm_ik.cpp",
            sources=[
                "cpp_parameterization/cpp/rby1_ik/rainbow_left_arm_ik.cpp"
            ],
        ),
        CMakeExtension(
            name="rainbow_right_arm_ik",
            sourcedir="cpp_parameterization/cpp/rby1_ik/rainbow_right_arm_ik.cpp",
            sources=[
                "cpp_parameterization/cpp/rby1_ik/rainbow_right_arm_ik.cpp"
            ],
        ),
    ]
)
