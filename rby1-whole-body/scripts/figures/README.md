# The paper's teaser figure

A multiple-exposure composite of the RB-Y1 picking up the box, built from a static top-down
recording of the hardware session rather than from simulation. The ghosts are real frames of the
real robot, drawn over an empty-room background plate recovered from the same recording.

```
step1_track_box.py   Lucas-Kanade box track over the transport window   -> box_track.json
step2_prep.py        median background plate + per-frame sharpness      -> plate.png, sharp.npy
step3_compose.py     pick poses, matte, and composite the exposures     -> sweep_p<N>_<level>.png
render_options.sh    renders the four published variants
```

The four rendered images are committed here — `sweep_p7_{low,med}.png` and
`sweep_p9_{low,med}.png` — because the recording they were built from is not distributable, so
they cannot be regenerated from a clone. The paper's teaser is `sweep_p7_med.png`.

`render_options.sh` reproduces exactly those four images, given the recording:

```bash
SWEEP_VIDEO=/path/to/recording.mp4 ./render_options.sh
# -> sweep_p7_low.png  sweep_p7_med.png  sweep_p9_low.png  sweep_p9_med.png
```

`p7` / `p9` are the number of ghost poses drawn; `low` / `med` are two opacity ramps. The paper
uses `sweep_p7_med.png`. The full argument list is
`<poses> <output> <arm_lo> <arm_hi> <box_lo> <box_hi> <arm_final> <box_final>`, and the exact
values used for each variant are in `render_options.sh`.

`box_track.json` is committed, so `step1_track_box.py` does not need to be re-run; the track was
seeded by hand against a verified frame and is what pins the composite. `plate.png` and
`sharp.npy` are derived and are rebuilt automatically by `render_options.sh` (~2 min).

## The input recording is not part of this release

The pipeline reads `20260811_174657.mp4`, a top-down recording of the 2026-08-11 hardware
session. Like the rest of the raw capture from that session it is not distributable and is not in
this repository, so **the teaser figure cannot be regenerated from a clone.** The scripts are
included because they are the method — the tracking window, the seed box, the plate construction
and the exposure ramps are all recorded here rather than lost.

Point `$SWEEP_VIDEO` at the recording if you have it. The hard-coded timings
(`WIN_T0`/`WIN_T1` in step 1, `T0`/`T1` in step 3, the 44 s plate start in step 2) are specific to
that recording and would need re-deriving for any other footage.

## Licence

The four rendered stills are photographs of the authors' own robot and lab, composited by the
scripts here, and are covered by this project's MIT licence along with the code. See
[`../../../LICENSE`](../../../LICENSE).

## Dependencies

`opencv-python` and `numpy`. OpenCV is not a dependency of the rest of this folder, so install it
separately if you want to run these:

```bash
pip install opencv-python
```
