"""Robot joint limits and the command dataclasses used by the plan format.

Extracted from the robot interface's utilities module for this release: only
the pieces the planning, verification and rendering paths need are kept. The
networking constants, camera/odometry types and image helpers that lived
alongside them are part of the robot client and are not released.
"""

from dataclasses import dataclass

import numpy as np

M4Transformation = np.ndarray  # 4x4 homogeneous transformation matrix


# Joint position limits the ROBOT enforces, per chain, in joint order.
#
# Copied from the vendor's own model (rby1-sdk `models/rby1a/urdf/model.urdf`),
# NOT from our Drake model: this is the bound the robot validates a command
# against and silently refuses the whole update for. Our planning URDF is
# tighter or equal everywhere (see models/ruby/rby1_description_drake/), so a
# plan that respects the planner cannot trip this -- the guard exists to catch
# the case where it drifts apart again, which has happened once and cost a
# 69-degree tracking error nobody noticed until the robot's web log was read.
#
# Duplicated rather than parsed because this module must stay importable on the
# Jetson with no Drake and no URDF parsing.
ROBOT_JOINT_LIMITS = {
    "torso": [
        (-0.261799388, 0.261799388), (-0.523598776, 1.570796327),
        (-2.617993878, 1.570796327), (-0.785398163, 1.570796327),
        (-0.523598776, 0.523598776), (-2.356194490, 2.356194490),
    ],
    "right_arm": [
        (-3.141592654, 3.141592654), (-3.141592654, 0.017453293),
        (-3.141592654, 3.141592654), (-2.617993878, 0.017453293),
        (-3.141592654, 3.141592654), (-1.570796327, 1.919862177),
        (-2.705260340, 2.705260340),
    ],
    "left_arm": [
        (-3.141592654, 3.141592654), (-0.017453293, 3.141592654),
        (-3.141592654, 3.141592654), (-2.617993878, 0.017453293),
        (-3.141592654, 3.141592654), (-1.570796327, 1.919862177),
        (-2.705260340, 2.705260340),
    ],
    "head": [(-0.523, 0.523), (-0.350, 1.570)],
}
# The chain aliases follow_joint_trajectory_stiff accepts, mapped to the above.
_CHAIN_ALIASES = {"right": "right_arm", "left": "left_arm",
                  "right_arm": "right_arm", "left_arm": "left_arm",
                  "torso": "torso", "head": "head"}
# Tolerance for float noise only. Deliberately not a safety margin: the robot
# was observed to reject a value sitting exactly ON the bound, so the margin
# has to come from the planner staying inside, not from slack here.
JOINT_LIMIT_EPS = 1e-6

# How far past a limit a commanded position may be and still be silently pulled
# back onto it (the robot client re-checked this). 1e-4 rad = 0.1 mrad =
# 0.006 deg, which is well under the robot's own repeatability and is the scale of
# trajopt's between-knot overshoot: it constrains the B-spline at its control
# points and says nothing about the curve between them, so a leg verified inside
# the limits still renders a waypoint or two a few hundredths of a milliradian
# out. That is the case this exists for. Anything larger is a real planning
# defect and must keep failing loudly -- a clamp big enough to hide those would
# silently change the geometry the plan was verified at.
JOINT_LIMIT_CLAMP_TOL = 1e-4

# Clamped values land this far INSIDE the bound rather than on it, because the
# robot rejects a target sitting exactly on the limit (the same observation
# JOINT_LIMIT_EPS records). Clamping to the bound itself would leave the update
# refused and the chain stalled -- the exact failure the clamp is meant to avoid.
JOINT_LIMIT_CLAMP_INSET = 1e-6


def joint_limit_violations(chain, positions):
    """[(index, value, lower, upper)] for entries outside the robot's limits.

    ``chain`` may be any alias ``follow_joint_trajectory_stiff`` accepts.
    Unknown chains (grippers, base) have no limits here and return [].
    """
    key = _CHAIN_ALIASES.get(chain)
    if key is None:
        return []
    limits = ROBOT_JOINT_LIMITS[key]
    out = []
    for i, q in enumerate(np.asarray(positions).flatten()):
        if i >= len(limits):
            break
        lo, hi = limits[i]
        if q < lo - JOINT_LIMIT_EPS or q > hi + JOINT_LIMIT_EPS:
            out.append((i, float(q), lo, hi))
    return out


class ControlCommandBase(object):
    """Base class for control commands."""


@dataclass
class ChainCommand(ControlCommandBase):
    """Joint command for the robot arm."""

    duration: float


@dataclass
class JointSpaceChainCommand(ChainCommand):
    """Joint space command for the robot arm."""


@dataclass
class JointPositionCommand(JointSpaceChainCommand):
    """Joint position command for the robot arm."""

    target_position: np.ndarray


@dataclass
class JointImpedanceCommand(JointSpaceChainCommand):
    """Joint impedance control command for the robot arm."""

    target_position: np.ndarray
    stiffness: list[float] | None = None
    damping_ratio: float = 1.0
    torque_limit: list[float] | None = None


@dataclass
class RobotConfigurationCommand(ControlCommandBase):
    """Robot configuration command for multiple robot chains"""

    target_configuration: dict[str, JointPositionCommand]


@dataclass
class CartesianSpaceChainCommand(ChainCommand):
    """Cartesian space command for the robot arm."""


@dataclass
class CartesianPoseCommand(CartesianSpaceChainCommand):
    """Cartesian pose command for the robot arm."""

    target_pose: M4Transformation
    target_joint_position: np.ndarray


@dataclass
class CartesianPositionCommand(CartesianSpaceChainCommand):
    """Cartesian position command for a chain. The base transformation
    is the base link of the chain."""

    target_pose: M4Transformation


@dataclass
class GripperCommand(ControlCommandBase):
    """Gripper command for the robot arm."""


@dataclass
class GripperPositionCommand(GripperCommand):
    target_width: float
    target_force: float


@dataclass
class GripperStateCommand(GripperCommand):
    """Gripper state command for the robot arm."""

    close: bool


@dataclass
class RobotBaseCommand(ControlCommandBase):
    """Robot base command for the robot base."""


@dataclass
class RobotBaseXYThetaCommand(RobotBaseCommand):
    """An XY-theta command for the robot base."""

    x: float
    y: float
    theta: float


@dataclass
class RobotBaseXYThetaDirCommand(RobotBaseCommand):
    """An XY-theta command for the robot base,
    including a binary direction for translational components."""

    x: float
    y: float
    theta: float
    dir: int  # 1 for forward, -1 for backward, 0 for in-place rotation

