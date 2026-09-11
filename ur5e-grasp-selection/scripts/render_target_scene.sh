#!/usr/bin/env bash
# Render the target-scene figure with Blender.
#
#   ./scripts/render_target_scene.sh [--solve|--resample] [-- <extra render args>]
#
# By default this renders whatever scene is already in out/target_scene.html.
#
#   --solve     re-export the scene from the pinned configuration in $CONFIG, so this is
#               deterministic and cannot lose the figure.
#   --resample  draw a fresh target from SEED into a scratch pin beside $HTML.  It does
#               not touch $CONFIG: promoting a new target to *the* figure is a deliberate
#               copy, printed at the end of the run.  Note SEED alone does not identify a
#               target -- --resample keeps the first one it accepts, and the committed pin
#               is the fourth from seed 42 -- so pin configurations, never seeds.
#
# Every default below is overridable by environment variable:
#
#   BLENDER       path to the Blender executable
#   HTML          input Meshcat scene            (out/target_scene.html)
#   OUT           output PNG                     (out/target_scene.png)
#   SAMPLES       Cycles samples                 (256)
#   RESOLUTION    "WIDTH HEIGHT", unquoted-split (3000 1688)
#   FRAME_RADIUS  metres around the target to fit (0.72)
#   CONFIG        pinned target configuration    (scripts/target_scene_configuration.json)
#   SEED          target-draw seed, --resample only  (42)
#   DEVICE        GPU or CPU                     (GPU)
#
# GPU means OptiX where available, then CUDA/HIP/oneAPI, falling back to CPU with a
# printed warning; the log line "[device] ..." always states what actually rendered.
#
# Camera placement is a parallax problem -- use --camera-azimuth / --camera-elevation,
# which orbit about --camera-target, rather than raw positions:
#
#   ./scripts/render_target_scene.sh -- --camera-azimuth 95 --camera-elevation 25
#
# The render is cropped to the figure by default (see --crop in the render script, which
# takes left/top/right/bottom as fractions of the frame).  Cycles renders the border only,
# so the crop is free; --no-crop gives the full frame back:
#
#   ./scripts/render_target_scene.sh -- --no-crop
#
# A cheap contact sheet to pick that angle before committing to a full render:
#
#   for az in 60 95 130 165; do
#     RESOLUTION="800 450" SAMPLES=16 OUT=out/preview_az${az}.png \
#       ./scripts/render_target_scene.sh -- --camera-azimuth $az --camera-elevation 25
#   done
#
# The figure's target lives in $CONFIG and is committed.  out/ is gitignored, so a pin
# left there is lost on the next clone -- which is how the published grasp figure was
# lost once already.
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
HTML="${HTML:-out/target_scene.html}"
OUT="${OUT:-out/target_scene.png}"
SAMPLES="${SAMPLES:-256}"
RESOLUTION="${RESOLUTION:-3000 1688}"
FRAME_RADIUS="${FRAME_RADIUS:-0.72}"
DEVICE="${DEVICE:-GPU}"
SEED="${SEED:-42}"
CONFIG="${CONFIG:-scripts/target_scene_configuration.json}"

SOLVE=0
RESAMPLE=0
EXTRA=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --solve) SOLVE=1; shift ;;
        --resample) SOLVE=1; RESAMPLE=1; shift ;;
        --) shift; EXTRA=("$@"); break ;;
        *) EXTRA+=("$1"); shift ;;
    esac
done

if [[ ! -x "$BLENDER" ]]; then
    echo "Blender not found at: $BLENDER" >&2
    echo "Set BLENDER=/path/to/blender and re-run." >&2
    exit 1
fi

SCRATCH_PIN="${HTML%.html}.json"

if [[ "$SOLVE" -eq 1 ]]; then
    activate_env
    if [[ "$RESAMPLE" -eq 0 && -f "$CONFIG" ]]; then
        echo "==> Re-posing the pinned target in $CONFIG"
        "$PYTHON" scripts/visualize_target_scene.py --no-interactive \
            --configuration "$CONFIG" --html-out "$HTML"
    elif [[ "$RESAMPLE" -eq 0 ]]; then
        echo "No pinned configuration at $CONFIG. Re-run with --resample to draw one." >&2
        exit 1
    else
        echo "==> Drawing a target from seed $SEED into $SCRATCH_PIN ($CONFIG is untouched)"
        "$PYTHON" scripts/visualize_target_scene.py --no-interactive \
            --num-scenes 1 --seed "$SEED" --html-out "$HTML"
        echo "==> To make this the figure's target:  cp $SCRATCH_PIN $CONFIG"
    fi
fi

if [[ ! -f "$HTML" ]]; then
    echo "No scene at $HTML. Re-run with --solve to generate one." >&2
    exit 1
fi

echo "==> Rendering $HTML -> $OUT"
# shellcheck disable=SC2086
"$BLENDER" --background --python scripts/render_target_scene.py -- \
    --html "$HTML" \
    --out "$OUT" \
    --samples "$SAMPLES" \
    --resolution $RESOLUTION \
    --frame-radius "$FRAME_RADIUS" \
    --device "$DEVICE" \
    ${EXTRA[@]+"${EXTRA[@]}"}

echo "==> Wrote $OUT"
