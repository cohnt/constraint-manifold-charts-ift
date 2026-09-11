#!/usr/bin/env python3
"""Install (or verify) the vendored meshcat_html_importer Blender add-on.

The add-on lives in Blender's user extensions directory, outside this repo, so
nothing in git normally proves that the copy Blender is running matches the copy
this repo pins.  That matters: it is the only thing that turns the exported
meshcat HTML into a Blender scene, so a silently different version changes what
every simulation render looks like, and a lost one blocks re-rendering entirely.

    ...install_meshcat_importer.py --check     compare installed vs vendored
    ...install_meshcat_importer.py             install, backing up any existing
    ...install_meshcat_importer.py --force     overwrite without a backup

The vendored tree is byte-identical to upstream v0.1.3; see third_party/README.md.
"""

from __future__ import annotations

import argparse
import filecmp
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]  # release root (third_party/ is shared)
VENDORED = REPO_ROOT / "third_party" / "meshcat_html_importer"

# Blender 5.0's user extensions directory.  `user_default` is the repository
# name Blender gives to locally installed (non-remote) extensions.
INSTALL_DIR = (
    Path.home() / ".config" / "blender" / "5.0" / "extensions" / "user_default"
)
INSTALLED = INSTALL_DIR / "meshcat_html_importer"

# Compiled bytecode is written into the installed copy by Blender itself and is
# never part of the comparison.
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo")


def _relative_files(root: Path) -> set[Path]:
    return {
        p.relative_to(root)
        for p in root.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts and p.suffix not in {".pyc", ".pyo"}
    }


def compare(vendored: Path, installed: Path) -> tuple[list[Path], list[Path], list[Path]]:
    """Return (missing, extra, differing) paths, relative to the add-on root."""
    want, have = _relative_files(vendored), _relative_files(installed)
    missing = sorted(want - have)
    extra = sorted(have - want)
    differing = sorted(
        rel
        for rel in want & have
        # shallow=False: compare contents, not just size and mtime.  An add-on
        # restored from a copy can easily match on stat and differ in bytes.
        if not filecmp.cmp(vendored / rel, installed / rel, shallow=False)
    )
    return missing, extra, differing


def do_check() -> int:
    if not INSTALLED.exists():
        print(f"NOT INSTALLED: {INSTALLED}")
        print("Run this script without --check to install it.")
        return 1

    missing, extra, differing = compare(VENDORED, INSTALLED)
    if not (missing or extra or differing):
        n = len(_relative_files(VENDORED))
        print(f"OK: installed add-on matches the vendored tree ({n} files).")
        print(f"  vendored:  {VENDORED}")
        print(f"  installed: {INSTALLED}")
        return 0

    print(f"MISMATCH between {VENDORED} and {INSTALLED}:")
    for label, paths in (
        ("missing from the install", missing),
        ("present only in the install", extra),
        ("differing contents", differing),
    ):
        for rel in paths:
            print(f"  {label}: {rel}")
    print("\nRe-run without --check to reinstall the vendored version.")
    return 1


def do_install(force: bool) -> int:
    INSTALL_DIR.mkdir(parents=True, exist_ok=True)

    if INSTALLED.exists():
        if force:
            shutil.rmtree(INSTALLED)
        else:
            backup = INSTALLED.with_name(INSTALLED.name + ".bak")
            if backup.exists():
                shutil.rmtree(backup)
            shutil.move(str(INSTALLED), str(backup))
            print(f"Moved the existing add-on aside -> {backup}")

    shutil.copytree(VENDORED, INSTALLED, ignore=IGNORE)
    print(f"Installed {len(_relative_files(INSTALLED))} files -> {INSTALLED}")
    print(
        "\nRestart Blender, then enable 'Meshcat HTML Importer' under\n"
        "Edit > Preferences > Add-ons if it is not already enabled."
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--check",
        action="store_true",
        help="only compare the installed add-on against the vendored tree; "
        "exit nonzero if they differ or it is missing",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="overwrite an existing install without keeping a .bak copy",
    )
    args = ap.parse_args()

    if not VENDORED.is_dir():
        print(f"Vendored add-on not found at {VENDORED}", file=sys.stderr)
        return 2

    return do_check() if args.check else do_install(args.force)


if __name__ == "__main__":
    raise SystemExit(main())
