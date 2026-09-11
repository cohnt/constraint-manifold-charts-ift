"""Compose the side-by-side Drake + hardware grid for the supplementary video.

Layout: 2 rows x 4 columns (+ row labels)
  Top row:    Drake renders for seeds 0, 3, 11, 16
  Bottom row: Real hardware videos for the same seeds

Each Drake render is time-stretched to match its corresponding hardware video
so that leg transitions stay synchronized. Then both are sped up by a common
factor to hit the target duration.

Usage:
    .venv/bin/python scripts/video/compose_grid.py
    .venv/bin/python scripts/video/compose_grid.py --seeds 0 3 11 16 --target-duration 21
"""

import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hardware_framing import HW_CROP  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
VIDEO_DIR = os.path.join(REPO, "video")
HW_DIR = os.path.join(REPO, "videos")


def build_seed_map(hw_dir):
    files = sorted(f for f in os.listdir(hw_dir) if f.endswith(".mp4"))
    return {i: f for i, f in enumerate(files)}


def probe_duration(path):
    return float(subprocess.check_output([
        "ffprobe", "-v", "quiet", "-show_entries", "format=duration",
        "-of", "csv=p=0", path,
    ]).decode().strip())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 3, 11, 16])
    ap.add_argument("--target-duration", type=float, default=21.0)
    ap.add_argument("--trim-json", default=os.path.join(HW_DIR, "trim_points.json"))
    ap.add_argument("--output", default=os.path.join(VIDEO_DIR, "grid_side_by_side.mp4"))
    args = ap.parse_args()

    seed_map = build_seed_map(HW_DIR)
    seeds = args.seeds
    assert len(seeds) == 4, "Need exactly 4 seeds for the 2x4 grid"

    if os.path.exists(args.trim_json):
        with open(args.trim_json) as f:
            trim = json.load(f)
    else:
        trim = {}

    # Get durations for each seed
    drake_durs = []
    hw_durs = []
    for seed in seeds:
        drake_path = os.path.join(VIDEO_DIR, f"drake_blender_{seed:02d}.mp4")
        drake_durs.append(probe_duration(drake_path))

        fname = seed_map[seed]
        if fname in trim:
            hw_durs.append(trim[fname]["duration_s"])
        else:
            hw_durs.append(probe_duration(os.path.join(HW_DIR, fname)))

    # Per-seed sync: stretch Drake to match hardware, then common speedup
    # Drake speed factor = drake_dur / hw_dur (makes Drake take same time as hw)
    # Then both get sped up by common_speed = max_hw_dur / target_duration
    max_hw_dur = max(hw_durs)
    common_speed = max(max_hw_dur / args.target_duration, 1.0)

    print(f"Per-seed sync:")
    for i, seed in enumerate(seeds):
        drake_stretch = drake_durs[i] / hw_durs[i]
        total_drake_speed = drake_stretch * common_speed
        print(f"  Seed {seed}: Drake {drake_durs[i]:.1f}s -> stretch {drake_stretch:.3f}x "
              f"-> total {total_drake_speed:.2f}x | HW {hw_durs[i]:.1f}s -> {common_speed:.2f}x")
    print(f"Common speedup: {common_speed:.2f}x")

    # Build inputs
    inputs = []
    for seed in seeds:
        drake_path = os.path.join(VIDEO_DIR, f"drake_blender_{seed:02d}.mp4")
        inputs.extend(["-i", drake_path])

    for seed in seeds:
        fname = seed_map[seed]
        hw_path = os.path.join(HW_DIR, fname)
        t = trim.get(fname, {})
        start = t.get("start_s", 0)
        end = t.get("end_s", None)
        if end:
            inputs.extend(["-ss", str(start), "-to", str(end), "-i", hw_path])
        else:
            inputs.extend(["-i", hw_path])

    grid_ny = 5
    grid_coords = {s: (s // grid_ny, s % grid_ny) for s in seeds}
    cell_w, cell_h = 480, 540

    filters = []
    for i, seed in enumerate(seeds):
        ix, iy = grid_coords[seed]
        # Drake: stretch to match hw duration, then apply common speedup
        drake_stretch = drake_durs[i] / hw_durs[i]
        total_drake_speed = drake_stretch * common_speed
        filters.append(
            f"[{i}:v]scale={cell_w}:{cell_h},setpts=PTS/{total_drake_speed:.4f},"
            f"drawtext=text='Point {seed}  ({ix}, {iy})':"
            f"fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:"
            f"fontsize=16:fontcolor=white:borderw=1:bordercolor=black:"
            f"x=6:y=6[d{i}]"
        )
        # Hardware: common speedup only
        hw_idx = i + 4
        filters.append(
            f"[{hw_idx}:v]crop={HW_CROP},scale={cell_w}:{cell_h},setpts=PTS/{common_speed:.4f},"
            f"drawtext=text='Point {seed}  ({ix}, {iy})':"
            f"fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:"
            f"fontsize=16:fontcolor=white:borderw=1:bordercolor=black:"
            f"x=6:y=6[h{i}]"
        )

    # Row labels
    filters.append(
        f"color=c=0x1a1a2e:s=1920x30:d={args.target_duration},"
        f"drawtext=text='Simulation (Drake)':"
        f"fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:"
        f"fontsize=18:fontcolor=#4fc3f7:x=(w-text_w)/2:y=4[rtop]"
    )
    filters.append(
        f"color=c=0x1a1a2e:s=1920x30:d={args.target_duration},"
        f"drawtext=text='Hardware':"
        f"fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:"
        f"fontsize=18:fontcolor=#66bb6a:x=(w-text_w)/2:y=4[rbot]"
    )

    filters.append("[d0][d1][d2][d3]hstack=inputs=4[drake_row]")
    filters.append("[h0][h1][h2][h3]hstack=inputs=4[hw_row]")
    filters.append("[rtop][drake_row][rbot][hw_row]vstack=inputs=4[stacked]")
    # Speed label in top-right corner
    filters.append(
        f"[stacked]drawtext=text='{common_speed:.0f}×':"
        f"fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:"
        f"fontsize=22:fontcolor=white:borderw=2:bordercolor=black:"
        f"x=w-text_w-12:y=8[out]"
    )

    filter_str = ";\n".join(filters)

    cmd = [
        "ffmpeg", "-y",
        *inputs,
        "-filter_complex", filter_str,
        "-map", "[out]",
        "-c:v", "libx264", "-crf", "20", "-preset", "medium",
        "-pix_fmt", "yuv420p",
        "-t", str(args.target_duration),
        "-an",
        args.output,
    ]

    print(f"\nRunning ffmpeg grid composition...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ffmpeg stderr:\n{result.stderr[-1000:]}")
        sys.exit(1)

    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
