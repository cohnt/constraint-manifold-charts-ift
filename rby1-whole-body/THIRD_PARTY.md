# Third-party content — RB-Y1 whole-body experiment

This folder's own source is MIT licensed (see [`../LICENSE`](../LICENSE)). The robot
description, scene geometry and generated inverse-kinematics sources redistributed alongside it
are third party and keep their own terms, listed below.

Where a file is described as **byte-identical** to an upstream file, that was established by
comparing cryptographic hashes against the upstream object, not by resemblance.

---

## 1. Rainbow Robotics RB-Y1 description — `models/ruby/rby1_description_drake/`

**Upstream:** [RainbowRobotics/rby1-sdk](https://github.com/RainbowRobotics/rby1-sdk),
`models/rby1a/urdf/model.urdf` and its meshes.
**Licence:** Apache License 2.0 —
[LICENSE](https://github.com/RainbowRobotics/rby1-sdk/blob/main/LICENSE).

The kinematics, link masses and inertia tensors in `base.urdf`, `torso.urdf`, `left_arm.urdf`,
`right_arm.urdf`, `head*.urdf` and the `add_rby1_*.dmd.yaml` directives derive from Rainbow's
official model. The visual meshes (`BASE`, `LINK_0`–`LINK_20`, `EE_BODY`, `EE_FINGER`,
`FT_SENSOR_L/R`, `WHEEL`, `PAN_TILT_1`–`3`) are format conversions of Rainbow's originals.

**Notice of modification (Apache 2.0, §4(b)).** These files have been modified from the Rainbow
originals for use with Drake:

- the single upstream URDF was split into separate base / torso / arm / head descriptions,
- collision geometry was replaced throughout with primitive spheres and boxes,
- several link masses were changed from upstream values (notably `link_torso_5`, and four
  `left_arm` links) to reflect the as-built robot,
- Drake model-directive (`.dmd.yaml`) files were added, which have no upstream counterpart.

`head_with_mit_camera_mount.urdf` is Rainbow's head chain with a locally designed camera mount
added; the added section is marked inline in the file.

## 2. KUKA LBR iiwa 14 collision geometry — `models/meshes/`, `models/iiwa14_convex_decimated_collision.urdf`

**Upstream:** [RobotLocomotion/models](https://github.com/RobotLocomotion/models)
(`drake_models`), `iiwa_description/meshes/iiwa14/collision/`, which in turn took the geometry
from [IFL-CAMP/iiwa_stack](https://github.com/IFL-CAMP/iiwa_stack) at SHA `2fa5bd5`.
**Licence:** BSD 3-Clause. Original geometry © 2015 Robert Krug & Todor Stoyanov, AASS Research
Center, Örebro University, Sweden. Drake's modifications are covered by Drake's BSD 3-Clause
terms.

`link_0.obj` … `link_7.obj` are **unmodified** Drake originals, byte-identical to upstream.
`link_*_convex.obj` and `link_*_convex_decimated.obj` are convex decompositions computed from
them by this project (MIT); the underlying geometry remains BSD 3-Clause. The URDF is derived
from Drake's `iiwa14_primitive_collision.urdf.xacro`.

## 3. Schunk WSG 50 with Fin Ray fingers — `models/wsg_finray/`

**Upstream:** [nepfaff/scene_gen_policy](https://github.com/nepfaff/scene_gen_policy),
`models/wsg_finray/`.
**Licence:** MIT —
[LICENSE](https://github.com/nepfaff/scene_gen_policy/blob/main/LICENSE).

`one_piece_wide_grasp_finray_finger.obj`, its `.mtl` and its three texture maps are
byte-identical to upstream. Unlike this release's UR5e experiment, this copy vendors no
`wsg_body.obj` — the SDF references Drake's own
`package://drake_models/wsg_50_description/meshes/wsg_body.gltf`, which Drake supplies at parse
time under BSD 3-Clause.

> Two of the texture maps (`_normal.png`, `_occlusion_roughness_metallic.png`) also appear,
> byte-identically, in [real-stanford/scalingup](https://github.com/real-stanford/scalingup),
> which publishes no licence file. The MIT grant from `scene_gen_policy` is our licence of
> record; the earlier provenance of those two textures is not established.

## 4. Table geometry — `models/old_table/`

**Upstream:** [RobotLocomotion/gcs-science-robotics](https://github.com/RobotLocomotion/gcs-science-robotics),
`models/table/`.
**Licence:** BSD 3-Clause —
[LICENSE.TXT](https://github.com/RobotLocomotion/gcs-science-robotics/blob/main/LICENSE.TXT).

Nine of the ten files are byte-identical to upstream; `table_wide.sdf` is locally modified.

> The texture maps in this set carry authoring metadata indicating they originate in a
> commercial interior asset pack that predates the upstream repository. We redistribute them
> under the BSD 3-Clause grant made by that repository; the pack's own original terms are not
> recorded anywhere upstream of it.

## 5. Livox Mid-360 LiDAR visual mesh — `models/ruby/rby1_description_drake/meshes/mid-360.{obj,mtl}`

**Upstream:** [Field-Robotics-Japan/UnitySensors](https://github.com/Field-Robotics-Japan/UnitySensors),
`Packages/UnitySensors/Runtime/RawData/LivoxModels/mid-360.obj`.
**Licence:** Apache License 2.0 —
[LICENSE](https://github.com/Field-Robotics-Japan/UnitySensors/blob/master/LICENSE).

**Notice of modification (Apache 2.0, §4(b)).** The mesh has been modified from the upstream file:
rescaled from millimetres to metres, translated by (-3.93, -4.13, 0.00) mm so its bounding box is
centred on the link origin (matching the collision box in the RB-Y1 URDFs), given computed vertex
normals at a 40° crease angle (upstream carries none), and given per-face material groups. No
vertex positions were otherwise altered. The accompanying `mid-360.mtl` is authored for this
release.

Note that Livox's own repositories do not publish a Mid-360 mesh — their
`livox_laser_simulation` package substitutes the Mid-40 model — so this is the licensed source we
were able to identify for this sensor's geometry.

## 6. IKFast — `cpp_parameterization/cpp/rby1_ik/`

**Upstream:** [OpenRAVE](https://github.com/rdiankov/openrave), `python/ikfast.h` and the
`ikfast.py` solver generator. © 2012–2014 Rosen Diankov.
**Licence:** Apache License 2.0.

`ikfast.h` carries Apache 2.0 in its own header. `rainbow_left_arm_ik.cpp` and
`rainbow_right_arm_ik.cpp` are **generated output** of OpenRAVE's IKFast (version `0x1000004a`,
`transform6d` solver, generated 2024-10-11) and each carries the Apache 2.0 grant that the
generator writes into every solver it emits.

> A common misconception: the IKFast *generator* (`ikfast.py`) is LGPL-3.0, but the code it
> *generates* is explicitly Apache 2.0 — the licence block is written into the emitter for that
> purpose. We do not redistribute the generator.

`ikfast.h` has been modified from the OpenRAVE original (a `#include "Python.h"` block required
by the extension entry points); the change is marked inline with an Apache 2.0 §4(b) notice.

## 7. This project's own content

`models/shelves/`, `models/old_shelves.sdf`, `models/new_shelves.dmd.yaml`,
`models/metal_table.sdf`, `models/wooden_table.sdf`, `models/wall.sdf`, the `.dmd.yaml` scene
directives, and all Python and C++ outside the files listed above are MIT licensed — see
[`../LICENSE`](../LICENSE).

The vendored Meshcat→Blender add-on used by the video pipeline lives at
[`../third_party/`](../third_party/) and is documented in
[`../THIRD_PARTY.md`](../THIRD_PARTY.md).

---

## Unresolved

Items whose licensing we have **not** been able to establish. See
[`../THIRD_PARTY.md`](../THIRD_PARTY.md) for the release-wide register.

| Item | Status |
| --- | --- |
| `meshes/rby1_wrist_camera.obj` | An Autodesk-exported CAD model of the wrist camera body, whose own `mtllib` line names a file that is not in the tree — indicating it was copied out of a directory we no longer have. The camera *mounts* alongside it are in-house parts; this file is the vendor camera itself and we have not identified its origin. |
| The DAE→OBJ mesh conversions in `models/ruby/rby1_description_drake/meshes/` | The underlying geometry is Apache 2.0 from Rainbow Robotics, which permits redistribution of derivatives. The conversion set itself reached this project through an intermediate repository that publishes no licence file. |
| `rby1_camera_mount_realsense.obj`, `rby1_camera_mount_v1.obj` | Locally designed brackets (Onshape export), reaching this project through the same intermediate repository, which publishes no licence file. |
