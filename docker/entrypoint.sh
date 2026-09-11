#!/usr/bin/env bash
# Entrypoint for the IFT-IK reproduction image.
#
# Each experiment has its own virtualenv. `use` switches between them; every
# environment shares the single Drake installation at /opt/drake via PYTHONPATH,
# so there is never a second pydrake for a compiled extension to collide with.
set -euo pipefail

export DRAKE_INSTALL_DIR=/opt/drake
export PYTHONPATH=/opt/drake/lib/python3.12/site-packages

cat >/etc/profile.d/ift.sh <<'PROFILE'
use() {
  case "$1" in
    iiwa) export VIRTUAL_ENV=/opt/venv/iiwa; cd /release/iiwa-bimanual ;;
    ur5e) export VIRTUAL_ENV=/opt/venv/ur5e; cd /release/ur5e-grasp-selection ;;
    rby1) export VIRTUAL_ENV=/opt/venv/rby1; cd /release/rby1-whole-body ;;
    *) echo "usage: use {iiwa|ur5e|rby1}" >&2; return 1 ;;
  esac
  export PATH="$VIRTUAL_ENV/bin:$PATH"
  echo "[$1] $(pwd)  python=$(command -v python3)"
}
PROFILE

if [ "${1:-bash}" = "bash" ] && [ -t 0 ]; then
  cat <<'BANNER'

  IFT-IK reproduction image  --  Drake 1.56.0 at /opt/drake

    use iiwa    bimanual IIWA          (Table I, autodiff study, swept volume)
    use ur5e    UR5e grasp selection   (grasp-IK table, both figures)
    use rby1    RB-Y1 whole body       (full-body IK table, video)

  Start with the cheap checks:
    use iiwa && ctest --test-dir cpp_parameterization/build
    use ur5e && python scripts/tests/test_eaik_ift.py
    use rby1 && python scripts/grid_guarantee_report.py

  Figures and video need Blender; Cycles falls back to CPU without a GPU
  runtime, which is slow but correct. See docs/DOCKER.md.

BANNER
  exec bash --rcfile <(echo '. /etc/profile.d/ift.sh; . /etc/bash.bashrc 2>/dev/null || true')
fi

exec "$@"
