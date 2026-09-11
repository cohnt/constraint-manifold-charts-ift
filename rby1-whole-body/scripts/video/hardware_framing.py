"""How the RB-Y1 hardware footage is cropped, and the pane it is composed into.

One definition, imported by every consumer, because these numbers are only
correct *relative to each other*: the crop's aspect ratio, the pane the crop is
scaled into, and the resolution the Drake sim renders at all have to agree or
the side-by-side panes stop lining up.

## The crop

The 2026-08-11 capture is 1920x1080 from a phone that never moved, so one crop
serves all 20 clips. It used to be ``1200:800:250:250``, which was mostly static
background: a wall, a kitchenette on the left and a bank of desks on the right,
with the robot occupying about a third of the width.

``940:890:450:190`` is fitted to where the footage actually moves. Accumulating
a per-pixel max-minus-min over every clip at 1 fps puts the robot's swept region
at x 520..1360, y 230..1080; the crop clears that by ~70 px on the left and
includes the table it works over, whose left edge runs off the frame the same
way it does in the reference framing. Verified against the extremes -- both arms
fully raised, the leftmost reach across the table, and the box on the floor.

Note the aspect is ~1.06, not 3:2. **A 3:2 crop cannot frame this scene
tightly**: the robot's vertical extent is ~850 px, which at 3:2 forces a width
of ~1275 px and drags the desks and the passer-by on the right back into shot.
That is the whole reason the old crop wasted so much of the frame, and it is why
the pane below is near-square rather than 3:2.

Do not derive any of this from ``detect_motion.py``. That script is superseded
for deriving *trim points*; the accumulation described above is a spatial
question about a fixed camera, which is a different thing entirely.
"""

# Crop rectangle within the 1920x1080 source: width, height, x, y.
HW_CROP_W, HW_CROP_H = 940, 890
HW_CROP_X, HW_CROP_Y = 450, 190

# ffmpeg `crop=` argument.
HW_CROP = f"{HW_CROP_W}:{HW_CROP_H}:{HW_CROP_X}:{HW_CROP_Y}"

# The pane each half of the side-by-side is composed into. Its aspect matches
# the crop's to within 0.1%, so scaling the crop into it neither letterboxes nor
# distorts. Two panes stack to 1920x910 and the *stack* is padded once to
# 1920x1080, which keeps the background band continuous across the full width
# instead of giving each half its own bars.
PANE_W, PANE_H = 960, 910

# The Drake sim half renders at exactly the pane size, so the simulated and real
# robots come out the same size on screen (blender_render_grid.py).
SIM_RENDER_W, SIM_RENDER_H = PANE_W, PANE_H
