"""Assemble Video 2: promotional/overview video.

Concatenates all segments with fade transitions.

The per-segment explanatory captions (a hold on the segment's last frame,
then drawtext lines faded in over it) are drawn *here*, in normalize_segment,
rather than baked into the segment scripts that render domain_extension,
boundary_reach, iiwa_bimanual, iiwa_iris, rby1_stability and rby1_hardware.
Two of those are a full Cycles render and the rest go through a Blender
frame-cache pipeline that only sometimes hits, so wording lives in
overview_captions.py and gets applied at assembly instead: editing a caption
marks only this stage (`assemble`) stale, an ffmpeg pass over frames that are
already rendered, not a re-render.

Usage:
    .venv/bin/python scripts/video/assemble_v2.py
    .venv/bin/python scripts/video/assemble_v2.py --skip-missing
"""

import argparse
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import overview_captions  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
VIDEO_DIR = os.path.join(REPO, "video")
MANIM_DIR = os.path.join(REPO, "media", "videos", "manim_scenes_v2", "1080p30")

BG_HEX = "0x1a1a2e"
FPS = 30

SEGMENTS = [
    ("title", os.path.join(VIDEO_DIR, "v2_title.mp4"), None),
    ("manim", os.path.join(MANIM_DIR, "IKParameterizationScene.mp4"), None),
    ("domain_extension", os.path.join(VIDEO_DIR, "v2_domain_extension.mp4"), None),
    ("boundary_reach", os.path.join(VIDEO_DIR, "v2_boundary_reach.mp4"), None),
    ("iiwa_bimanual", os.path.join(VIDEO_DIR, "v2_iiwa_bimanual.mp4"), None),
    ("iiwa_iris", os.path.join(VIDEO_DIR, "v2_iiwa_iris.mp4"), None),
    ("rby1_pipeline", os.path.join(MANIM_DIR, "RBY1PipelineScene.mp4"), None),
    ("rby1_stability", os.path.join(VIDEO_DIR, "v2_rby1_stability.mp4"), None),
    ("rby1_hardware", os.path.join(VIDEO_DIR, "v2_rby1_hardware.mp4"), None),
    ("results", os.path.join(VIDEO_DIR, "v2_results.mp4"), None),
    ("closing", os.path.join(VIDEO_DIR, "v2_closing.mp4"), None),
]


def normalize_segment(path, out_path, max_duration=None, fade=False, name=None):
    dur = float(subprocess.check_output([
        "ffprobe", "-v", "quiet", "-show_entries", "format=duration",
        "-of", "csv=p=0", path,
    ]).decode().strip())

    if max_duration and dur > max_duration:
        dur = max_duration

    vf = (
        f"scale=1920:1080:force_original_aspect_ratio=decrease,"
        f"pad=1920:1080:(ow-iw)/2:(oh-ih)/2:color={BG_HEX},"
        f"fps={FPS}"
    )
    hold_vf = overview_captions.hold_filter(name)
    if hold_vf:
        vf += f",{hold_vf}"
    caption_vf = overview_captions.caption_filter(name)
    if caption_vf:
        vf += f",{caption_vf}"

    # The hold pads the segment with cloned frames, so the fade-out has to
    # start relative to the *held* duration -- computed before the fade is
    # appended, or the fade-out would land partway through the hold instead
    # of on the segment's last held frame.
    dur += overview_captions.hold_seconds(name)
    fade_out_start = max(0, dur - 0.3)
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
    ap.add_argument("--skip-missing", action="store_true",
                    help="Skip missing segments instead of failing")
    ap.add_argument("--no-title", action="store_true",
                    help="drop the opening title card (for embedding somewhere "
                         "that supplies its own)")
    args = ap.parse_args()

    final_segments = [s for s in SEGMENTS
                      if not (args.no_title and s[0] == "title")]

    if args.output is None:
        stem = "promo_video_notitle" if args.no_title else "promo_video"
        args.output = os.path.join(VIDEO_DIR, f"{stem}.mp4")

    missing = [(n, p) for n, p, _ in final_segments if not os.path.exists(p)]
    if missing:
        if args.skip_missing:
            print("Skipping missing segments:")
            for name, path in missing:
                print(f"  {name}: {path}")
            final_segments = [(n, p, d) for n, p, d in final_segments if os.path.exists(p)]
        else:
            print("Missing segments:")
            for name, path in missing:
                print(f"  {name}: {path}")
            print("\nRender the missing segments first, or use --skip-missing.")
            sys.exit(1)

    print(f"Assembling {len(final_segments)} segments...")

    with tempfile.TemporaryDirectory() as tmpdir:
        norm_paths = []
        for i, (name, path, max_dur) in enumerate(final_segments):
            out = os.path.join(tmpdir, f"{i:02d}_{name}.mp4")
            is_first = (i == 0)
            is_last = (i == len(final_segments) - 1)
            print(f"  Normalizing {name}...")
            normalize_segment(path, out, max_duration=max_dur,
                              fade=is_first or is_last, name=name)
            norm_paths.append(out)

        concat_file = os.path.join(tmpdir, "segments.txt")
        with open(concat_file, "w") as f:
            for p in norm_paths:
                f.write(f"file '{p}'\n")

        print("  Concatenating...")
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
    print(f"Duration: {dur:.1f}s")
    segments_str = " -> ".join(n for n, _, _ in final_segments)
    print(f"Segments: {segments_str}")


if __name__ == "__main__":
    main()
