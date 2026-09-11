"""Unpickling support for plans written before this package was renamed.

The committed plan cache and hardware execution records were pickled by code in
which this package was called ``rby1_interface`` and this module's contents
lived in ``rby1_interface.utilities``. Pickle stores the module path by name, so
those files name a module that no longer exists.

Rather than rewrite the records -- they are the experiment's result, and
rewriting them to make a rename tidy is not a trade worth making -- this
registers the historical names as aliases. Importing ``plan_format`` installs
them, so ``plan_format.plan_io.load_plan`` just works.
"""

import sys

from . import commands, plan_io, trajectory_conversion


def install() -> None:
    """Register the pre-rename module names so old pickles resolve."""
    pkg = sys.modules[__name__.rsplit(".", 1)[0]]
    for legacy, target in (
        ("rby1_interface", pkg),
        ("rby1_interface.utilities", commands),
        ("rby1_interface.plan_io", plan_io),
        ("rby1_interface.trajectory_conversion", trajectory_conversion),
    ):
        sys.modules.setdefault(legacy, target)
