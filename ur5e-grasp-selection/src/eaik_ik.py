import numpy as np
import os
from eaik.IK_URDF import UrdfRobot
from src.util import RepoDir

# Drake's tool0 frame differs from EAIK's flange convention by this fixed change of basis.
# It is an involution (X_OFFSET @ X_OFFSET == I), so the inverse is the matrix itself; it
# used to be rebuilt and re-inverted with np.linalg.inv on every single IK call, which the
# profile showed costing more than the branch selection.
X_OFFSET = np.array([[-1, 0, 0, 0],
                     [0, 0, 1, 0],
                     [0, 1, 0, 0],
                     [0, 0, 0, 1]], dtype=float)
X_OFFSET_INV = X_OFFSET  # involution; asserted in scripts/tests/test_eaik_ift.py


class EaikIK:
    def __init__(self, urdf_path=None):
        """
        Initialize EAIK robot from URDF.
        If urdf_path is None, defaults to the UR5e model.
        """
        if urdf_path is None:
            urdf_path = os.path.join(RepoDir(), "models/universal_robots/ur_description/urdf/ur5e_drake_collision.urdf")
        
        # UR5e has 6 joints. We don't need to fix any joints for it.
        self.robot = UrdfRobot(urdf_path)

    def solve_all(self, target_pose_drake_4x4):
        """
        Return (solutions, is_ls) for a target pose in Drake's tool0 convention.

        `is_ls[i]` marks solution i as least-squares: EAIK substitutes a single
        least-squares root wherever a subproblem's discriminant goes non-positive, so
        such a row does NOT reach the requested pose.  These rows are the domain
        extension of IK (they are what make it defined outside the reachable workspace)
        and must be kept, but an exact row is preferable whenever one exists.

        `is_ls` is a solve diagnostic used for branch tracking, not a constraint.
        Reachability is enforced by the reachability constraints, which are
        differentiable functions of the decision variables.
        """
        # Bake in the offset: X_eaik = X_drake * X_offset^-1
        target_pose_eaik = target_pose_drake_4x4 @ X_OFFSET_INV

        result = self.robot.IK(target_pose_eaik)
        if result.Q is None:
            return [], np.zeros(0, dtype=bool)
        Q = [np.array(q).flatten() for q in result.Q]
        is_ls = np.asarray(result.is_LS).flatten().astype(bool)
        return Q, is_ls

    def solve(self, target_pose_drake_4x4) -> list[np.ndarray]:
        """
        Return all IK solutions as a list of 6-element joint arrays, least-squares rows
        included.  Use solve_all() when the least-squares flags matter.
        """
        return self.solve_all(target_pose_drake_4x4)[0]

    def fk(self, q) -> np.ndarray:
        """
        Forward kinematics in Drake's tool0 convention. Returns 4x4.
        """
        X_eaik = self.robot.fwdKin(q)
        return X_eaik @ X_OFFSET
