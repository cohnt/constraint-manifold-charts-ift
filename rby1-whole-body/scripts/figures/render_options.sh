#!/usr/bin/env bash
# Renders the four sweep options next to this script.
# Args to step3_compose.py:
#   <poses> <output> <arm_lo> <arm_hi> <box_lo> <box_hi> <arm_final> <box_final>
set -e
cd "$(dirname "$0")"

# step2_prep.py builds the background plate and the sharpness table. They are
# derived files - rebuild them if they are not here (~2 min).
if [ ! -f plate.png ] || [ ! -f sharp.npy ]; then
    echo "cache missing - running step2_prep.py"
    python3 step2_prep.py
fi

#            poses  output            arm_lo arm_hi box_lo box_hi arm_fin box_fin
python3 step3_compose.py 7 sweep_p7_low.png 0.20   0.30   0.72   0.86   0.52    0.94
python3 step3_compose.py 7 sweep_p7_med.png 0.30   0.42   0.80   0.92   0.65    0.97
python3 step3_compose.py 9 sweep_p9_low.png 0.20   0.30   0.72   0.86   0.52    0.94
python3 step3_compose.py 9 sweep_p9_med.png 0.30   0.42   0.80   0.92   0.65    0.97

echo "wrote sweep_p{7,9}_{low,med}.png"
