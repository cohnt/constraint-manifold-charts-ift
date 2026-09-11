#!/usr/bin/env bash
# Reproduce the grasp-selection figure exactly as it appears in the paper.
#
#   ./scripts/reproduce_grasp_figure.sh
#
# One command, no arguments, no environment dependence: every parameter that affects the
# image is pinned below, including the IK seed.  This is the script to run when the goal
# is "regenerate the committed figure"; use scripts/render_grasp_figure.sh directly when
# the goal is to iterate on framing, resolution or materials, since that one honours
# environment overrides and this one deliberately does not.
#
# Chain:
#   visualize_grasp_selection.py --no-interactive  ->  out/grasp_selection.html
#   render_grasp_figure.py (inside Blender)        ->  out/grasp_selection.png
#
# The version of the figure in the paper is a manual crop of out/grasp_selection.png
# (2422x1320 out of 3000x1688); the crop is the one step that is not scripted.
#
# Requires Blender 5.0.1 with nepfaff's meshcat_html_importer extension installed.
# Override the interpreter with BLENDER=/path/to/blender if it is not in the default spot.
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

# ── Pinned parameters ─────────────────────────────────────────────────────────
# The three grasps are pinned as explicit joint configurations rather than as a seed.
# The published figure was solved by the revision at commit cd0a354; branch selection and
# the solver's joint box both changed afterwards, so no seed reproduces it -- today's
# solver converges to a different, side-on set of grasps for essentially every seed
# (measured over 45 seeds).  The configurations were recovered from the archived Meshcat
# scene by fitting Drake's forward kinematics to the exported link transforms, and
# reproduce the figure to within render noise.  Delete --configurations below to solve
# fresh grasps instead; that is a *new* figure, not this one.
CONFIGURATIONS="scripts/paper_figure_configuration.json"
# The grasp-sector bounds that separate the three arms.  The mug handle makes perfectly
# upright grasps infeasible at every yaw, so the three grasps are tilted and split into
# yaw sectors instead.  They no longer affect the pinned path, but are kept so that
# dropping --configurations reproduces the intended search.
SEED=692332
ROLL_PITCH_DEG=40.0
YAW_CENTERS=(-140.0 0.0 140.0)
YAW_SECTOR=40.0
ARM_ALPHA=0.4
GRIPPER_ALPHA=0.5

# Render: 16:9 at 3000 px wide, Cycles on the GPU.  The camera position corresponds to
# azimuth 95 deg / elevation 25 deg orbited about the mug.
HTML="out/grasp_selection.html"
PNG="out/grasp_selection.png"
SAMPLES=256
RESOLUTION=(3000 1688)
FRAME_RADIUS=0.30
LENS=50.0
VIEW_TRANSFORM="Standard"
DEVICE="GPU"

BLENDER="${BLENDER:-$HOME/opt/blender-5.0.1-linux-x64/blender}"

if [[ ! -x "$BLENDER" ]]; then
    echo "Blender not found at: $BLENDER" >&2
    echo "Set BLENDER=/path/to/blender and re-run." >&2
    exit 1
fi

echo "==> Posing the three pinned grasps and exporting the Meshcat scene"
activate_env
"$PYTHON" scripts/visualize_grasp_selection.py \
    --no-interactive \
    --configurations "$CONFIGURATIONS" \
    --seed "$SEED" \
    --alpha "$GRIPPER_ALPHA" \
    --arm-alpha "$ARM_ALPHA" \
    --roll-pitch-deg "$ROLL_PITCH_DEG" \
    --yaw-centers "${YAW_CENTERS[@]}" \
    --yaw-sector "$YAW_SECTOR" \
    --html-out "$HTML"

echo "==> Rendering $HTML -> $PNG"
"$BLENDER" --background --python scripts/render_grasp_figure.py -- \
    --html "$HTML" \
    --out "$PNG" \
    --samples "$SAMPLES" \
    --resolution "${RESOLUTION[@]}" \
    --frame-radius "$FRAME_RADIUS" \
    --lens "$LENS" \
    --view-transform "$VIEW_TRANSFORM" \
    --device "$DEVICE"
# Note: --arm-alpha is deliberately NOT passed to the renderer.  The visualizer already
# bakes ARM_ALPHA into the exported scene; the renderer's own --arm-alpha overrides that
# with a slightly different value (its restyle default is 0.42), which is what the
# committed figure was rendered with.

echo "==> Wrote $PNG"
echo "    The paper's figure is a manual crop of this image."
