# Vendored third-party code

## `meshcat_html_importer`

The Blender add-on that imports a Drake Meshcat static HTML page into Blender. It is the
interchange step in this repo's figure pipeline:

    Drake -> Meshcat StaticHtml -> meshcat_html_importer -> Cycles

It is vendored rather than fetched at render time so that
`scripts/figures/render_swept_volume.py` reproduces from a fresh checkout with no network.

| | |
|---|---|
| upstream | <https://github.com/nepfaff/drake-blender-tools>, path `blender_addons/meshcat_html_importer` |
| pin | tag `v0.1.3` = commit `8f3687f99db96a4b242f18a2b85a3f76fe588de0` |
| add-on version | 0.1.3 (`blender_manifest.toml`) |
| requires | Blender >= 5.0.0 (declared `blender_version_min`) |
| author | Nicholas Pfaff \<nepfaff@mit.edu\> |
| size | 144 KB |

The tree here is byte-identical to upstream at that commit; there are no local patches. To
re-vendor, clone upstream at the pinned commit and copy
`blender_addons/meshcat_html_importer` over this directory.

**The upstream repository was renamed.** It used to be `nepfaff/drake-blender-recorder`, which
still 301-redirects, so older references keep working; cite the current name.

### Licensing

`LICENSE.meshcat_html_importer.TXT` is upstream's root `LICENSE.TXT`, **BSD-2-Clause**, which is
also what `blender_manifest.toml` declares (`license = ["SPDX:BSD-2-Clause"]`). Upstream keeps no
license file inside the add-on directory itself, so it has to be copied separately — a plain
`cp -r` of the add-on ships no license text at all.

Two inconsistencies inherited from upstream, recorded here rather than silently normalised:
the individual `.py` files carry `SPDX-License-Identifier: MIT` headers, and the vendored
`_msgpack/` carries `Apache-2.0`.

### Dependencies

Self-contained apart from **numpy**, which it imports unconditionally and does not vendor. That is
satisfied by the numpy Blender bundles (1.26.4 in Blender 5.0.1), so nothing needs installing --
but it does mean the add-on only imports inside Blender's Python, never a plain `python3`.
It vendors its own msgpack (`_msgpack/`) because Blender bundles none.
