# Third-party content

This project's own source is MIT licensed — see [`LICENSE`](LICENSE). The robot descriptions,
mesh geometry, vendored add-ons and generated inverse-kinematics sources redistributed alongside
it are third party and keep their own terms.

Each experiment folder documents the assets it ships. This file covers what is shared across
them, and keeps the register of items whose licensing we have not been able to resolve.

| Folder | Notices |
| --- | --- |
| `iiwa-bimanual/` | [`iiwa-bimanual/THIRD_PARTY.md`](iiwa-bimanual/THIRD_PARTY.md) |
| `ur5e-grasp-selection/` | [`ur5e-grasp-selection/THIRD_PARTY.md`](ur5e-grasp-selection/THIRD_PARTY.md) |
| `rby1-whole-body/` | [`rby1-whole-body/THIRD_PARTY.md`](rby1-whole-body/THIRD_PARTY.md) |

---

## 1. Shared vendored code — `third_party/meshcat_html_importer/`

A Blender add-on that imports Drake Meshcat static HTML pages, used by the figure and video
pipelines in `iiwa-bimanual/` and `rby1-whole-body/`.

**Upstream:** [nepfaff/drake-blender-tools](https://github.com/nepfaff/drake-blender-tools), tag
`v0.1.3`, commit `8f3687f99db96a4b242f18a2b85a3f76fe588de0`. © 2024 Nicholas Pfaff.
**Licence:** BSD 2-Clause — full text at
[`third_party/LICENSE.meshcat_html_importer.TXT`](third_party/LICENSE.meshcat_html_importer.TXT).

Vendored byte-identically to upstream, with no local patches. It is shared rather than duplicated
per experiment so that the two installers verify against a single pinned copy. See
[`third_party/README.md`](third_party/README.md) for the re-vendoring procedure and for the
upstream SPDX-header inconsistencies, which are inherited and deliberately left unmodified.

## 2. Dependencies, resolved on your machine and not redistributed

| Dependency | Licence | Role |
| --- | --- | --- |
| [Drake](https://drake.mit.edu) 1.56.0 | BSD 3-Clause | Multibody dynamics, optimization, IRIS, GCS, the collision checker. Also supplies `drake_models`, which several scenes reference at parse time. |
| [Blender](https://www.blender.org) 5.0.x | GPL-2.0-or-later (used as a tool) | Cycles rendering for every figure and for the video. |
| [EAIK](https://github.com/OstermD/EAIK) 1.2.1 | BSD 3-Clause | The analytic IK solver in the UR5e experiment. Its authors request citation; see that folder's notices. |
| [BlenderKit](https://www.blenderkit.com) materials | see below | Two materials optionally used by the swept-volume figure, read from your own cache. |

Nothing in this repository vendors or redistributes any of these.

## 3. Robot and scene geometry, by origin

Summarised here; each entry is documented in full in the owning folder's notices.

| Origin | Licence | Where |
| --- | --- | --- |
| [RainbowRobotics/rby1-sdk](https://github.com/RainbowRobotics/rby1-sdk) | Apache-2.0 | RB-Y1 description |
| [RobotLocomotion/models](https://github.com/RobotLocomotion/models) (`drake_models`) | BSD 3-Clause | iiwa 14 collision geometry, WSG 50 body |
| [IFL-CAMP/iiwa_stack](https://github.com/IFL-CAMP/iiwa_stack) | BSD 3-Clause | iiwa 14 original geometry, © 2015 Krug & Stoyanov, Örebro University |
| [ros-industrial/universal_robot](https://github.com/ros-industrial/universal_robot) | Apache-2.0 at the repository root; the `ur_description` manifest still declares BSD (see below) | UR5e description |
| [nepfaff/scene_gen_policy](https://github.com/nepfaff/scene_gen_policy) | MIT | WSG 50 Fin Ray fingers |
| [RobotLocomotion/gcs-science-robotics](https://github.com/RobotLocomotion/gcs-science-robotics) | BSD 3-Clause | Table geometry |
| [Field-Robotics-Japan/UnitySensors](https://github.com/Field-Robotics-Japan/UnitySensors) | Apache-2.0 | Livox Mid-360 visual mesh |
| [OpenRAVE](https://github.com/rdiankov/openrave) | Apache-2.0 | `ikfast.h` and the generated RB-Y1 IK solvers |

> **On IKFast.** The IKFast *generator* (`ikfast.py`) is LGPL-3.0, but the solver code it
> *generates* carries an explicit Apache-2.0 grant that the generator writes into every file it
> emits. We redistribute generated solvers and `ikfast.h`, not the generator.

---

## Unresolved register

Items whose licensing we have **not** been able to establish to our own satisfaction. They are
listed here rather than quietly shipped.

| Item | Where | Status |
| --- | --- | --- |
| RB-Y1 mesh format conversions, and the camera-mount parts | `rby1-whole-body/models/ruby/rby1_description_drake/meshes/` | The underlying geometry is Apache-2.0 from Rainbow Robotics, which permits redistribution of derivatives. The DAE→OBJ conversion set and the in-house mount parts reached this project through an intermediate repository that publishes no licence file. |
| `rby1_wrist_camera.obj` | same | An Autodesk-exported model of the wrist camera body whose own `mtllib` names a file absent from the tree. The mounts around it are in-house parts; this file is the vendor camera and its origin is not established. |
| Two Fin Ray texture maps | `rby1-whole-body/models/wsg_finray/`, `ur5e-grasp-selection/models/wsg_finray/` | `_normal.png` and `_occlusion_roughness_metallic.png` are byte-identical to copies in a repository that publishes no licence. The MIT grant from `scene_gen_policy` is our licence of record; the earlier provenance is not established. |
| Table texture maps | `iiwa-bimanual/models/old_table/`, `rby1-whole-body/models/old_table/` | Authoring metadata indicates a commercial interior asset pack predating the upstream repository. We redistribute under that repository's BSD 3-Clause grant; the pack's own terms are not recorded upstream of it. |
| BlenderKit materials | not redistributed | The per-asset licence is held in BlenderKit's online database rather than in the downloaded `.blend`. This is why the paper's rendered swept-volume PNGs are **not** shipped — a rendered image is a derived work of the materials used. Rendering with `--table-material procedural --shelf-material procedural` avoids them entirely. |

### A licence discrepancy we did not reconcile

`ros-industrial/universal_robot` is licensed Apache-2.0 at the repository root, and that is the
grant we rely on and the text vendored at
`ur5e-grasp-selection/models/universal_robots/LICENSE.txt`. Its `ur_description/package.xml`,
however, still declares `BSD` — a legacy artifact of that project's relicensing. We ship the
manifest as upstream wrote it rather than silently rewriting it, and record both here.

---

If you are the rights holder for any of the above and something here is wrong, please open an
issue and we will correct or remove it.
