"""Assemble the hardware supplementary video.

Concatenates: title → segment1 → segment2. Two cuts share every segment and
differ only in the title card -- anonymous by default, named with --named --
so their default output names differ too and cannot be confused for one
another.

Usage:
    .venv/bin/python scripts/video/assemble_v1.py            # -> hardware_supplementary_anonymous.mp4
    .venv/bin/python scripts/video/assemble_v1.py --named    # -> hardware_supplementary_non_anonymous.mp4
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ral_annotations  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
VIDEO_DIR = os.path.join(REPO, "video")

BG_HEX = "0x1a1a2e"
FPS = 30

# No closing results card: this video is about the RB-Y1 hardware runs, and a
# static summary frame at the end was restating numbers the footage already
# shows. video/results.mp4 is still on disk if it is ever wanted back.
#
# The title card is chosen in main(): the two cuts differ only there.
SEGMENTS = [
    ("segment1", os.path.join(VIDEO_DIR, "segment1_point00.mp4"), None),
    ("segment2", os.path.join(VIDEO_DIR, "segment2_all20.mp4"), None),
]
TITLE_CARDS  = {False: "title_anonymous.mp4", True: "title_named.mp4"}
OUTPUT_STEMS = {False: "hardware_supplementary_anonymous",
                True:  "hardware_supplementary_non_anonymous"}


def normalize_segment(name, path, out_path, max_duration=None, fade=False):
    """Scale/pad/fps-normalize `path` (the `name`'d segment) to 1920x1080,
    optionally fading the first/last segment in/out, and write `out_path`.

    `name` exists so this can ask ral_annotations.py whether `name` has any
    assembly-time overlays (see that module's docstring for why the RA-L
    cut's new artwork lives there rather than in the segment renderers).
    When it does, the ffmpeg graph switches from a plain `-vf` chain to
    `-filter_complex` with the overlay PNGs as extra looped inputs, composited
    AFTER this function's own scale/pad/fps normalization (an annotation has
    to land on the final 1920x1080 frame, not the pre-scale source) and
    BEFORE the fade this function applies to the first/last segment (so the
    fade still covers the annotations along with everything else). Segments
    with no overlays (title, today) take the original `-vf` path completely
    unchanged -- byte-for-byte, not just equivalent -- so this refactor
    cannot itself perturb a segment that isn't supposed to change.
    """
    dur = float(subprocess.check_output([
        "ffprobe", "-v", "quiet", "-show_entries", "format=duration",
        "-of", "csv=p=0", path,
    ]).decode().strip())

    if max_duration and dur > max_duration:
        dur = max_duration

    fade_out_start = max(0, dur - 0.3)
    base_vf = (
        f"scale=1920:1080:force_original_aspect_ratio=decrease,"
        f"pad=1920:1080:(ow-iw)/2:(oh-ih)/2:color={BG_HEX},"
        f"fps={FPS}"
    )

    overlay_paths = ral_annotations.overlay_inputs(name)
    if overlay_paths:
        cmd = ["ffmpeg", "-y", "-i", path]
        for p in overlay_paths:
            # -r FPS: a still image input defaults to 25fps if not told
            # otherwise, which does not evenly divide `dur` the way this
            # segment's own (now fps=30-normalized) frames do -- `overlay`'s
            # framesync waits for BOTH inputs to reach EOF before the output
            # ends, so a 25fps-vs-30fps rounding mismatch of a fraction of a
            # frame let the overlaid output run ~2 frames (0.067s) longer
            # than the un-annotated baseline. Measured empirically (isolated
            # single-overlay test, 2026-09-07): without -r, 128.30s ->
            # 128.333s; with it, exactly 128.300s. -t dur, not left to loop
            # forever: filter_complex needs every input to have a known end,
            # and dur (this segment's own, already-clamped-to-max_duration
            # duration) is exactly how long the base video stream this gets
            # composited onto lasts.
            cmd += ["-loop", "1", "-r", str(FPS), "-t", f"{dur:.2f}", "-i", p]
        filter_complex = f"[0:v]{base_vf}[scaled];"
        filter_complex += ral_annotations.overlay_filter(name, "[scaled]")
        out_label = "[annotated]"
        if fade:
            filter_complex += (
                f";{out_label}fade=in:0:9,"
                f"fade=out:st={fade_out_start:.2f}:d=0.3[faded]"
            )
            out_label = "[faded]"
        cmd += ["-filter_complex", filter_complex, "-map", out_label]
        cmd += ["-c:v", "libx264", "-crf", "18", "-preset", "medium",
                "-pix_fmt", "yuv420p", "-an"]
        if max_duration:
            cmd.extend(["-t", f"{max_duration:.2f}"])
        cmd.append(out_path)
    else:
        vf = base_vf
        if fade:
            vf += f",fade=in:0:9,fade=out:st={fade_out_start:.2f}:d=0.3"
        cmd = [
            "ffmpeg", "-y", "-i", path,
            "-vf", vf,
            "-c:v", "libx264", "-crf", "18", "-preset", "medium",
            "-pix_fmt", "yuv420p", "-an",
        ]
        if max_duration:
            cmd.extend(["-t", f"{max_duration:.2f}"])
        cmd.append(out_path)

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ffmpeg error for {path}:\n{result.stderr[-300:]}")
        raise RuntimeError(f"normalize failed for {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", default=None)
    ap.add_argument("--no-title", action="store_true",
                    help="drop the opening title card (for embedding somewhere "
                         "that supplies its own)")
    ap.add_argument("--named", action="store_true",
                    help="use the named title card and write the non-anonymous "
                         "cut. Off by default: the anonymous card is the safe "
                         "state for a double-blind submission.")
    args = ap.parse_args()

    segments = list(SEGMENTS)
    if not args.no_title:
        segments.insert(0, ("title",
                            os.path.join(VIDEO_DIR, TITLE_CARDS[args.named]), 3.0))

    if args.output is None:
        stem = OUTPUT_STEMS[args.named] + ("_notitle" if args.no_title else "")
        args.output = os.path.join(VIDEO_DIR, f"{stem}.mp4")

    missing = [(n, p) for n, p, _ in segments if not os.path.exists(p)]
    if missing:
        print("Missing segments:")
        for name, path in missing:
            print(f"  {name}: {path}")
        print("\nRender the missing segments first.")
        sys.exit(1)

    with tempfile.TemporaryDirectory() as tmpdir:
        norm_paths = []
        for i, (name, path, max_dur) in enumerate(segments):
            out = os.path.join(tmpdir, f"{i:02d}_{name}.mp4")
            # Whatever ends up first still gets the fade-in, so dropping the
            # title does not leave the video opening on a hard cut.
            is_first = (i == 0)
            is_last = (i == len(segments) - 1)
            print(f"Normalizing {name}...")
            normalize_segment(name, path, out, max_duration=max_dur,
                              fade=is_first or is_last)
            norm_paths.append(out)

        concat_file = os.path.join(tmpdir, "segments.txt")
        with open(concat_file, "w") as f:
            for p in norm_paths:
                f.write(f"file '{p}'\n")

        print("Concatenating segments...")
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", concat_file,
            "-c:v", "libx264", "-crf", "18", "-preset", "medium",
            "-pix_fmt", "yuv420p",
            args.output,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Concat failed:\n{result.stderr[-500:]}")
            sys.exit(1)

    dur = float(subprocess.check_output([
        "ffprobe", "-v", "quiet", "-show_entries", "format=duration",
        "-of", "csv=p=0", args.output,
    ]).decode().strip())
    print(f"\nFinal video: {args.output}")
    print(f"Duration: {dur:.1f}s (limit: 180s)")
    if dur > 180:
        print("WARNING: exceeds 3-minute RA-L limit!")


if __name__ == "__main__":
    main()
