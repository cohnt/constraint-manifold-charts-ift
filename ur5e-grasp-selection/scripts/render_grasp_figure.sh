#!/usr/bin/env bash
# Render the grasp-selection figure with Blender.
#
#   ./scripts/render_grasp_figure.sh [--solve] [-- <extra render args>]
#
# By default this renders whatever scene is already in out/grasp_selection.html.
# Pass --solve to re-run the IK first and export a fresh scene.
#
# Every default below is overridable by environment variable:
#
#   BLENDER       path to the Blender executable
#   HTML          input Meshcat scene            (out/grasp_selection.html)
#   OUT           output PNG                     (out/grasp_selection.png)
#   SAMPLES       Cycles samples                 (256)
#   RESOLUTION    "WIDTH HEIGHT", unquoted-split (3000 1688)
#   FRAME_RADIUS  metres around the mug to fit   (0.30)
#   DEVICE        GPU or CPU                     (GPU)
#
# GPU means OptiX where available, then CUDA/HIP/oneAPI, falling back to CPU with a
# printed warning; the log line "[device] ..." always states what actually rendered.
#
# e.g. a fast low-resolution preview:
#
#   RESOLUTION="1200 675" SAMPLES=64 ./scripts/render_grasp_figure.sh
#
# Render time scales with pixel count at fixed samples, so the 3000x1688 default costs
# ~2.25x the old 2000x1125.
#
# Requires Blender with the meshcat_html_importer extension installed (nepfaff's
# drake-blender-tools); override the interpreter with BLENDER=/path/to/blender.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

# Activate the project venv only if it exists AND no environment is already
# active. Under `set -euo pipefail` an unconditional `source venv/bin/activate`
# aborts the whole script for anyone using conda, uv, or a venv by another name.
# Override the interpreter with $PYTHON.
PYTHON="${PYTHON:-python3}"
activate_env() {
    if [[ -z "${VIRTUAL_ENV:-}${CONDA_PREFIX:-}" && -f venv/bin/activate ]]; then
        # shellcheck disable=SC1091
        source venv/bin/activate
    fi
}

BLENDER="${BLENDER:-$HOME/opt/blender-5.0.1-linux-x64/blender}"
HTML="${HTML:-out/grasp_selection.html}"
OUT="${OUT:-out/grasp_selection.png}"
SAMPLES="${SAMPLES:-256}"
RESOLUTION="${RESOLUTION:-3000 1688}"
FRAME_RADIUS="${FRAME_RADIUS:-0.30}"
DEVICE="${DEVICE:-GPU}"

SOLVE=0
EXTRA=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --solve) SOLVE=1; shift ;;
        --) shift; EXTRA=("$@"); break ;;
        *) EXTRA+=("$1"); shift ;;
    esac
done

if [[ ! -x "$BLENDER" ]]; then
    echo "Blender not found at: $BLENDER" >&2
    echo "Set BLENDER=/path/to/blender and re-run." >&2
    exit 1
fi

if [[ "$SOLVE" -eq 1 ]]; then
    echo "==> Solving grasp IK and exporting the Meshcat scene"
    activate_env
    "$PYTHON" scripts/visualize_grasp_selection.py --no-interactive --html-out "$HTML"
fi

if [[ ! -f "$HTML" ]]; then
    echo "No scene at $HTML. Re-run with --solve to generate one." >&2
    exit 1
fi

echo "==> Rendering $HTML -> $OUT"
# shellcheck disable=SC2086
"$BLENDER" --background --python scripts/render_grasp_figure.py -- \
    --html "$HTML" \
    --out "$OUT" \
    --samples "$SAMPLES" \
    --resolution $RESOLUTION \
    --frame-radius "$FRAME_RADIUS" \
    --device "$DEVICE" \
    ${EXTRA[@]+"${EXTRA[@]}"}

echo "==> Wrote $OUT"
