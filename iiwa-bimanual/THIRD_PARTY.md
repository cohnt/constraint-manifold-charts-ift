# Third-party notices

The MIT licence in [`../LICENSE`](../LICENSE) covers the code and documentation written for this project. It
does **not** cover the third-party material listed here, which is either vendored into the tree,
embedded in a committed data file, or baked into a committed rendered image. Each item keeps its own
terms.

## 1. Vendored code

### `../third_party/meshcat_html_importer/`

The Blender add-on that imports a Drake Meshcat static HTML page. Upstream
<https://github.com/nepfaff/drake-blender-tools> at tag `v0.1.3`
(`8f3687f99db96a4b242f18a2b85a3f76fe588de0`), author Nicholas Pfaff.

**BSD-2-Clause**, text in `../third_party/LICENSE.meshcat_html_importer.TXT`. Two inconsistencies
inherited from upstream and left unmodified: the individual `.py` files carry
`SPDX-License-Identifier: MIT` headers, and the vendored `_msgpack/` carries `Apache-2.0`. See
`../third_party/README.md`.

## 2. Robot and scene model geometry

`models/` contains a KUKA LBR iiwa 14 description and a Schunk WSG 50 gripper with fin-ray fingers.
The visual meshes are referenced from Drake's `drake_models` package
(`package://drake_models/iiwa_description/...`); the decimated convex collision meshes under
`models/meshes/` are derived from the same source.

- **iiwa 14** — originally from [`IFL-CAMP/iiwa_stack`](https://github.com/IFL-CAMP/iiwa_stack)
  at SHA `2fa5bd5`, **BSD-3-Clause**, Copyright (c) 2015 Robert Krug & Todor Stoyanov, AASS Research
  Center, Örebro University, Sweden. Substantially modified for Drake (mesh format conversion, mass
  and inertia properties, collision models); those modifications are under Drake's terms.
- **Schunk WSG 50** — **BSD-3-Clause**, Copyright 2020 Toyota Research Institute.
- **Drake** and `drake_models` — **BSD-3-Clause**, Copyright 2012-2025 Robot Locomotion Group @
  CSAIL.
- **Table** (`models/old_table/`) — from
  [`RobotLocomotion/gcs-science-robotics`](https://github.com/RobotLocomotion/gcs-science-robotics),
  `models/table/`, **BSD-3-Clause**
  ([LICENSE.TXT](https://github.com/RobotLocomotion/gcs-science-robotics/blob/main/LICENSE.TXT)).
  The `table_wide.*` files here are byte-identical to upstream. Note that the texture maps carry
  authoring metadata indicating they originate in a commercial interior asset pack predating that
  repository; we redistribute them under the BSD-3-Clause grant made there, and the pack's own
  original terms are not recorded upstream of it.

The upstream licence texts ship with Drake, in `iiwa_description/LICENSE.TXT`,
`iiwa_description/iiwa_stack.LICENSE.txt` and `wsg_50_description/LICENSE` inside the `drake_models`
package.

### `data/swept_volume_trajectory.html`

**This file embeds the above mesh geometry.** It is a Meshcat static page — a self-contained
serialisation of the whole visual scene, so the iiwa and WSG meshes are inside it as data, not as
references. Committing it therefore redistributes that geometry, under the terms above. It is
committed deliberately, so the swept-volume figure reproduces from a clone without Drake; see
`README.md`.

## 3. Materials used by the rendered figures

The paper's rendered `swept_volume_*.png` figures use two materials from
[BlenderKit](https://www.blenderkit.com):

| Role | Asset | BlenderKit base ID |
| --- | --- | --- |
| Table | *Old Metal* | `9fff7c54-fbc7-490b-9312-5adc941ecdbe` |
| Shelves | *Old plywood* | `29dfd92a-f797-46b6-8e9b-34d082e56c7c` |

Neither the assets nor any image rendered with them is redistributed here: the materials are read
from the user's own BlenderKit cache at render time and are absent from a clone, and the paper's
rendered PNGs are deliberately **not** committed to this release.

> The per-asset licence is held in BlenderKit's online database rather than in the downloaded
> `.blend` metadata, so it could not be determined offline. That is why the rendered figures are
> not shipped — a rendered image is a derived work of the materials it was rendered with. If you
> render your own from your own BlenderKit cache, check each asset's licence page before
> redistributing the result. Rendering with `--table-material procedural --shelf-material
> procedural` uses only code from this repository and produces images with no third-party asset
> content.

## 4. Not redistributed

Drake itself, Blender, and the BlenderKit assets are dependencies resolved on the user's machine.
Nothing in this repository vendors or redistributes them.
