"""Camera model shared between the Blender segment renders and their overlays.

The problem this solves: the three "explainer" segments of the overview video
(domain extension, boundary reachability, static stability) draw annotations that
are *anchored to 3D points* -- a residual line between two gripper positions, the
support polygon on the floor, the CoM marker.  Those were originally drawn on VTK
frames and projected with the VTK camera.  Once the 3D render comes from Blender
instead, projecting with the old camera would put every annotation in the wrong
place, so the two halves have to agree on one camera.

They agree by construction here:

* ``blender_camera_matrix`` builds the Blender ``matrix_world`` from the same
  eye/target look-at that Drake's ``_look_at`` helpers used, so the framing of a
  converted segment matches the VTK one it replaces.
* ``lens_for_focal`` turns a pinhole focal length in pixels into the Blender lens
  in millimetres, for ``sensor_fit = 'HORIZONTAL'``.
* the Blender script writes the camera it actually rendered with to
  ``camera.json``; ``SegmentCamera.load`` reads *that file* rather than
  recomputing, so an overlay can never silently disagree with the render.

Blender vs Drake conventions, since they are the whole reason this file exists:

    Drake camera:   +X right, +Y down,  +Z forward (into the scene)
    Blender camera: +X right, +Y up,    -Z forward (looks down its own -Z)

so ``R_blender = [right | up | -forward]`` where ``[right | down | forward]`` is
Drake's.  With ``focal_x == focal_y`` and no lens shift the two projections are
algebraically identical, which ``selftest`` checks numerically.
"""

from __future__ import annotations

import json
import math
import os

import numpy as np

# Blender's default camera sensor width [mm].  Fixed here rather than read back
# so lens computation and projection cannot drift apart.
SENSOR_WIDTH_MM = 36.0


def look_at_basis(eye, target, up=(0.0, 0.0, 1.0)):
    """Drake-convention camera basis (right, down, forward) for a look-at.

    Identical to the ``_look_at`` helper duplicated across the VTK render
    scripts, factored out so the Blender side provably uses the same view.
    """
    eye = np.asarray(eye, dtype=float)
    target = np.asarray(target, dtype=float)
    up = np.asarray(up, dtype=float)

    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(-up, forward)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    return right, down, forward


def blender_camera_matrix(eye, target, up=(0.0, 0.0, 1.0)):
    """4x4 ``matrix_world`` for a Blender camera at ``eye`` looking at ``target``."""
    eye = np.asarray(eye, dtype=float)
    right, down, forward = look_at_basis(eye, target, up)
    M = np.eye(4)
    M[:3, 0] = right
    M[:3, 1] = -down          # Blender's +Y is up
    M[:3, 2] = -forward       # Blender looks along its own -Z
    M[:3, 3] = eye
    return M


def focal_px_for_vfov(vfov_rad, height):
    """Pinhole focal length in pixels for a vertical field of view.

    Matches Drake's ``CameraInfo(width, height, fov_y)``, which sets
    ``focal_x == focal_y == height / (2 tan(fov_y / 2))``.
    """
    return (height / 2.0) / math.tan(vfov_rad / 2.0)


def lens_for_focal(focal_px, width, sensor_width_mm=SENSOR_WIDTH_MM):
    """Blender lens [mm] giving ``focal_px`` at ``width`` px, sensor_fit HORIZONTAL."""
    return focal_px * sensor_width_mm / float(width)


class SegmentCamera:
    """Projects world points to pixels exactly as the Blender render did."""

    def __init__(self, matrix_world, lens_mm, sensor_width_mm, width, height):
        self.matrix_world = np.asarray(matrix_world, dtype=float).reshape(4, 4)
        self.lens_mm = float(lens_mm)
        self.sensor_width_mm = float(sensor_width_mm)
        self.width = int(width)
        self.height = int(height)

        self.R = self.matrix_world[:3, :3]
        self.eye = self.matrix_world[:3, 3]
        # sensor_fit HORIZONTAL: the sensor width maps onto resolution_x.
        self.fx = self.lens_mm / self.sensor_width_mm * self.width
        self.fy = self.fx                      # square pixels (pixel aspect 1:1)
        self.cx = self.width / 2.0
        self.cy = self.height / 2.0

    # ── construction ────────────────────────────────────────────────────────

    @classmethod
    def from_look_at(cls, eye, target, width, height, vfov_rad=None, lens_mm=None,
                     up=(0.0, 0.0, 1.0)):
        if (vfov_rad is None) == (lens_mm is None):
            raise ValueError("give exactly one of vfov_rad / lens_mm")
        if lens_mm is None:
            lens_mm = lens_for_focal(focal_px_for_vfov(vfov_rad, height), width)
        return cls(blender_camera_matrix(eye, target, up), lens_mm,
                   SENSOR_WIDTH_MM, width, height)

    @classmethod
    def load(cls, path):
        """Read the camera the Blender render actually used."""
        with open(path) as f:
            d = json.load(f)
        return cls(d["matrix_world"], d["lens_mm"], d["sensor_width_mm"],
                   d["width"], d["height"])

    def to_dict(self):
        return dict(matrix_world=self.matrix_world.tolist(),
                    lens_mm=self.lens_mm, sensor_width_mm=self.sensor_width_mm,
                    width=self.width, height=self.height,
                    sensor_fit="HORIZONTAL",
                    eye=self.eye.tolist())

    # ── projection ──────────────────────────────────────────────────────────

    def project(self, p_world, clip_to_frame=True):
        """Pixel coordinates of a world point, or None if it is not visible.

        Returns floats; callers that draw with PIL round them.  ``clip_to_frame``
        mirrors the old VTK helper, which returned None for anything off-frame.
        """
        d = np.asarray(p_world, dtype=float) - self.eye
        p_cam = self.R.T @ d
        depth = -p_cam[2]                      # Blender looks down -Z
        if depth <= 1e-9:
            return None
        u = self.cx + self.fx * p_cam[0] / depth
        v = self.cy - self.fy * p_cam[1] / depth
        if clip_to_frame and not (0 <= u < self.width and 0 <= v < self.height):
            return None
        return float(u), float(v)

    def project_int(self, p_world, clip_to_frame=True):
        p = self.project(p_world, clip_to_frame=clip_to_frame)
        return None if p is None else (int(round(p[0])), int(round(p[1])))


def selftest():
    """Check the Blender projection against Drake's VTK projection.

    Run as ``.venv/bin/python scripts/video/segment_camera.py``.  Needs pydrake
    only for this check, never for rendering.
    """
    from pydrake.all import CameraInfo, RigidTransform, RotationMatrix

    width, height, vfov = 1920, 1080, 0.8
    eye = np.array([1.2, -1.4, 1.0])
    target = np.array([0.4, 0.0, 0.55])

    cam = SegmentCamera.from_look_at(eye, target, width, height, vfov_rad=vfov)

    right, down, forward = look_at_basis(eye, target)
    X_WC = RigidTransform(RotationMatrix(np.column_stack([right, down, forward])), eye)
    info = CameraInfo(width, height, vfov)

    def vtk_project(p):
        p_c = X_WC.inverse() @ np.asarray(p, dtype=float)
        if p_c[2] <= 0:
            return None
        return (info.focal_x() * p_c[0] / p_c[2] + info.center_x(),
                info.focal_y() * p_c[1] / p_c[2] + info.center_y())

    rng = np.random.default_rng(0)
    worst = 0.0
    n = 0
    for _ in range(2000):
        p = target + rng.normal(scale=0.5, size=3)
        a, b = cam.project(p, clip_to_frame=False), vtk_project(p)
        if a is None or b is None:
            continue
        worst = max(worst, abs(a[0] - b[0]), abs(a[1] - b[1]))
        n += 1
    print(f"compared {n} points; worst |Blender - VTK| = {worst:.4f} px")
    print(f"lens = {cam.lens_mm:.4f} mm, fx = {cam.fx:.2f} px")
    # The only difference should be Drake's half-pixel principal point
    # (center = W/2 - 0.5) versus Blender's exact sensor centre.
    assert worst < 0.51, f"projection mismatch of {worst:.3f} px"
    print("OK")


if __name__ == "__main__":
    selftest()
