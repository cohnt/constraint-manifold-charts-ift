"""Locate the Blender executable and the meshcat importer add-on.

Resolution order for the executable is `$BLENDER`, then the expected 5.0.1
tarball location under `~/opt`, then whatever is on `PATH`.

Driver-side only. The scripts that run *inside* Blender cannot import this
module (no repo on `sys.path` there); they expand `~` inline instead.
"""

import os
import shutil

# Blender's user extensions directory. `user_default` is the repository name
# Blender gives to locally installed (non-remote) extensions. Must agree with
# install_meshcat_importer.py, which installs the pinned add-on here.
EXTENSIONS = os.path.expanduser("~/.config/blender/5.0/extensions/user_default")

_DEFAULT_BLENDER = os.path.expanduser("~/opt/blender-5.0.1-linux-x64/blender")


def find_blender():
    """Return a path to the Blender executable, or raise with what was tried."""
    candidates = [os.environ.get("BLENDER"), _DEFAULT_BLENDER]
    for path in candidates:
        if path and os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    found = shutil.which("blender")
    if found:
        return found
    raise FileNotFoundError(
        "Could not find Blender. Set $BLENDER to a 5.0.x executable. Tried: "
        + ", ".join(repr(c) for c in candidates if c)
        + ", and 'blender' on PATH."
    )


BLENDER = (
    os.environ.get("BLENDER")
    or (_DEFAULT_BLENDER if os.path.isfile(_DEFAULT_BLENDER) else None)
    or shutil.which("blender")
    or _DEFAULT_BLENDER
)
