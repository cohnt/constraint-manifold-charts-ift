#!/usr/bin/env python3
"""Install (or verify) the vendored meshcat_html_importer Blender add-on.

The add-on lives in `third_party/meshcat_html_importer`, pinned to upstream
v0.1.3 (see `third_party/README.md`). Blender will not see it there: it loads
user extensions from `~/.config/blender/5.0/extensions/user_default`, so the
tree has to be copied into place once per machine.

    python3 scripts/figures/install_meshcat_importer.py            # install
    python3 scripts/figures/install_meshcat_importer.py --check    # verify only
    python3 scripts/figures/install_meshcat_importer.py --force    # no .bak

`--check` exits non-zero on any mismatch, which is what makes it usable as a
precondition: `render_swept_volume.py` runs the same comparison before it
launches Blender, because a stale installed copy renders a scene that looks
plausible and is not the one this repo pins.

Plain `python3` -- this runs outside Blender and imports nothing but stdlib.
"""

import argparse
import filecmp
import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]  # release root (third_party/ is shared)
SOURCE = REPO_ROOT / "third_party" / "meshcat_html_importer"

# Hardcoded to Blender 5.0 to match blender_paths.EXTENSIONS and the add-on's
# own `blender_version_min = "5.0.0"`.
EXTENSIONS = Path(os.path.expanduser("~/.config/blender/5.0/extensions/user_default"))
TARGET = EXTENSIONS / "meshcat_html_importer"

IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc")


def _relevant_files(root: Path):
    """Every file under `root`, relative, skipping bytecode."""
    out = set()
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        for name in filenames:
            if name.endswith(".pyc"):
                continue
            out.add((Path(dirpath) / name).relative_to(root))
    return out


def compare():
    """Return (missing, extra, differing) between SOURCE and TARGET."""
    if not TARGET.exists():
        return sorted(_relevant_files(SOURCE)), [], []
    want, have = _relevant_files(SOURCE), _relevant_files(TARGET)
    differing = [
        rel for rel in sorted(want & have)
        if not filecmp.cmp(SOURCE / rel, TARGET / rel, shallow=False)
    ]
    return sorted(want - have), sorted(have - want), differing


def check(verbose=True):
    missing, extra, differing = compare()
    if not (missing or extra or differing):
        if verbose:
            print(f"[addon] up to date: {TARGET}")
        return True
    if verbose:
        if not TARGET.exists():
            print(f"[addon] not installed: {TARGET}")
        for rel in missing:
            print(f"[addon] missing:   {rel}")
        for rel in extra:
            print(f"[addon] extra:     {rel}")
        for rel in differing:
            print(f"[addon] differs:   {rel}")
        print("[addon] run: python3 scripts/figures/install_meshcat_importer.py")
    return False


def install(force=False):
    if not SOURCE.is_dir():
        raise SystemExit(f"Vendored add-on missing: {SOURCE}")
    EXTENSIONS.mkdir(parents=True, exist_ok=True)
    if TARGET.exists():
        if force:
            shutil.rmtree(TARGET)
        else:
            backup = TARGET.with_suffix(TARGET.suffix + ".bak")
            if backup.exists():
                shutil.rmtree(backup)
            shutil.move(str(TARGET), str(backup))
            print(f"[addon] moved existing install to {backup}")
    shutil.copytree(SOURCE, TARGET, ignore=IGNORE)
    print(f"[addon] installed {SOURCE} -> {TARGET}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true",
                    help="compare only; exit non-zero if the install is missing or stale")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing install instead of moving it to .bak")
    args = ap.parse_args()

    if args.check:
        sys.exit(0 if check() else 1)
    install(force=args.force)
    sys.exit(0 if check() else 1)


if __name__ == "__main__":
    main()
