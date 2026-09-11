"""Plan file format for RB-Y1 whole-body plans.

This package is the Drake-free, network-free subset of the original robot
interface: it is what lets a cached plan be loaded, converted to command
waypoints, verified and rendered without a robot and without Drake.

The robot client itself -- connection management, the on-robot server, the
gripper driver and camera capture -- is not part of this release.
"""

from . import _legacy as _legacy_pickle_compat

_legacy_pickle_compat.install()
