# SPDX-License-Identifier: MIT
"""Transform utilities for meshcat scene data."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class Transform:
    """Decomposed transform with translation, rotation, and scale."""

    translation: tuple[float, float, float]
    rotation: tuple[float, float, float, float]  # Quaternion (x, y, z, w)
    scale: tuple[float, float, float]

    @classmethod
    def identity(cls) -> Transform:
        """Create an identity transform."""
        return cls(
            translation=(0.0, 0.0, 0.0),
            rotation=(0.0, 0.0, 0.0, 1.0),
            scale=(1.0, 1.0, 1.0),
        )

    def to_matrix(self) -> np.ndarray:
        """Convert to a 4x4 transformation matrix."""
        # Build rotation matrix from quaternion
        x, y, z, w = self.rotation
        rot = np.array(
            [
                [
                    1 - 2 * y * y - 2 * z * z,
                    2 * x * y - 2 * z * w,
                    2 * x * z + 2 * y * w,
                    0,
                ],
                [
                    2 * x * y + 2 * z * w,
                    1 - 2 * x * x - 2 * z * z,
                    2 * y * z - 2 * x * w,
                    0,
                ],
                [
                    2 * x * z - 2 * y * w,
                    2 * y * z + 2 * x * w,
                    1 - 2 * x * x - 2 * y * y,
                    0,
                ],
                [0, 0, 0, 1],
            ],
            dtype=np.float64,
        )

        # Apply scale
        scale_mat = np.diag([self.scale[0], self.scale[1], self.scale[2], 1.0])

        # Apply translation
        trans_mat = np.eye(4, dtype=np.float64)
        trans_mat[0, 3] = self.translation[0]
        trans_mat[1, 3] = self.translation[1]
        trans_mat[2, 3] = self.translation[2]

        return trans_mat @ rot @ scale_mat


def parse_transform_matrix(matrix_data: list[float] | np.ndarray) -> np.ndarray:
    """Parse a column-major 4x4 matrix from meshcat format.

    Args:
        matrix_data: 16 floats in column-major order

    Returns:
        4x4 numpy array
    """
    if isinstance(matrix_data, np.ndarray):
        data = matrix_data.flatten()
    else:
        data = matrix_data

    if len(data) != 16:
        raise ValueError(f"Expected 16 matrix elements, got {len(data)}")

    # Column-major to row-major conversion
    matrix = np.array(data, dtype=np.float64).reshape(4, 4).T
    return matrix


def matrix_to_trs(matrix: np.ndarray) -> Transform:
    """Decompose a 4x4 matrix into translation, rotation, scale.

    Args:
        matrix: 4x4 transformation matrix

    Returns:
        Transform with decomposed TRS
    """
    # Extract translation
    translation = (float(matrix[0, 3]), float(matrix[1, 3]), float(matrix[2, 3]))

    # Extract scale from column magnitudes
    sx = np.linalg.norm(matrix[:3, 0])
    sy = np.linalg.norm(matrix[:3, 1])
    sz = np.linalg.norm(matrix[:3, 2])
    scale = (float(sx), float(sy), float(sz))

    # Normalize to get rotation matrix
    rot_matrix = np.zeros((3, 3), dtype=np.float64)
    rot_matrix[:, 0] = matrix[:3, 0] / sx if sx > 1e-10 else matrix[:3, 0]
    rot_matrix[:, 1] = matrix[:3, 1] / sy if sy > 1e-10 else matrix[:3, 1]
    rot_matrix[:, 2] = matrix[:3, 2] / sz if sz > 1e-10 else matrix[:3, 2]

    # Convert rotation matrix to quaternion
    rotation = rotation_matrix_to_quaternion(rot_matrix)

    return Transform(
        translation=translation,
        rotation=rotation,
        scale=scale,
    )


def rotation_matrix_to_quaternion(
    rot: np.ndarray,
) -> tuple[float, float, float, float]:
    """Convert a 3x3 rotation matrix to quaternion (x, y, z, w).

    Args:
        rot: 3x3 rotation matrix

    Returns:
        Quaternion as (x, y, z, w)
    """
    # Shepperd's method for numerical stability
    trace = rot[0, 0] + rot[1, 1] + rot[2, 2]

    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (rot[2, 1] - rot[1, 2]) * s
        y = (rot[0, 2] - rot[2, 0]) * s
        z = (rot[1, 0] - rot[0, 1]) * s
    elif rot[0, 0] > rot[1, 1] and rot[0, 0] > rot[2, 2]:
        s = 2.0 * math.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2])
        w = (rot[2, 1] - rot[1, 2]) / s
        x = 0.25 * s
        y = (rot[0, 1] + rot[1, 0]) / s
        z = (rot[0, 2] + rot[2, 0]) / s
    elif rot[1, 1] > rot[2, 2]:
        s = 2.0 * math.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2])
        w = (rot[0, 2] - rot[2, 0]) / s
        x = (rot[0, 1] + rot[1, 0]) / s
        y = 0.25 * s
        z = (rot[1, 2] + rot[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1])
        w = (rot[1, 0] - rot[0, 1]) / s
        x = (rot[0, 2] + rot[2, 0]) / s
        y = (rot[1, 2] + rot[2, 1]) / s
        z = 0.25 * s

    return (float(x), float(y), float(z), float(w))


def quaternion_multiply(
    q1: tuple[float, float, float, float],
    q2: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """Multiply two quaternions (x, y, z, w format).

    Args:
        q1: First quaternion
        q2: Second quaternion

    Returns:
        Product quaternion
    """
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2

    return (
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    )


def combine_transforms(parent: Transform, child: Transform) -> Transform:
    """Combine parent and child transforms into a single world transform.

    Args:
        parent: Parent transform (already in world space)
        child: Child transform (local to parent)

    Returns:
        Combined transform in world space
    """
    # Convert to matrices and multiply
    parent_mat = parent.to_matrix()
    child_mat = child.to_matrix()
    combined_mat = parent_mat @ child_mat

    # Decompose back to TRS
    return matrix_to_trs(combined_mat)
