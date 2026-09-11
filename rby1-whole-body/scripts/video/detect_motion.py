"""Detect first/last motion frame in each hardware video for trimming.

SUPERSEDED for the 2026-08-11 grid run -- do not re-run it over those videos, it
will overwrite videos/trim_points.json with worse numbers. Use instead:

    .venv/bin/python scripts/video/check_video_sync.py --write-trim

which derives the cut from the execution records' absolute timestamps and each
video's container creation_time. That is exact; this is a detector, and on that
dataset it was late by +0.5..+2.2s per seed (a threshold crossing always lags a
robot accelerating from rest) and fell through to `return 0.0, duration` for
point 17, leaving it 9.5s out of sync. validate_motion_sync.py shows the logged
window bracketing the measured motion profile for all 20 seeds.

Still useful when there is no execution record to align against -- footage from a
run that was not logged, or a camera whose clock cannot be trusted.

Outputs a JSON file mapping each video filename to {start_s, end_s} trim points.
Uses frame differencing with Gaussian blur to detect when the robot starts/stops moving.

Usage:
    .venv/bin/python scripts/video/detect_motion.py
    .venv/bin/python scripts/video/detect_motion.py --video-dir videos --output videos/trim_points.json
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

SUBSAMPLE_FPS = 5
BLUR_KERNEL = (21, 21)
# Fraction of frame to keep (center crop) to exclude static table edges / background
ROI_FRAC = 0.70
# Threshold on mean absolute diff per pixel — tuned for 1080p robot motion
MOTION_THRESHOLD = 2.5
# Seconds of padding around detected motion
PAD_S = 0.5


def detect_motion_bounds(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    stride = max(1, round(fps / SUBSAMPLE_FPS))

    # ROI crop bounds
    rx = int(w * (1 - ROI_FRAC) / 2)
    ry = int(h * (1 - ROI_FRAC) / 2)
    rw = int(w * ROI_FRAC)
    rh = int(h * ROI_FRAC)

    prev_gray = None
    motion_frames = []

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % stride == 0:
            roi = frame[ry:ry+rh, rx:rx+rw]
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, BLUR_KERNEL, 0)
            if prev_gray is not None:
                diff = cv2.absdiff(gray, prev_gray)
                mean_diff = diff.mean()
                motion_frames.append((frame_idx / fps, mean_diff))
            prev_gray = gray
        frame_idx += 1
    cap.release()

    if not motion_frames:
        return 0.0, total_frames / fps

    times = np.array([t for t, _ in motion_frames])
    diffs = np.array([d for _, d in motion_frames])

    moving = diffs > MOTION_THRESHOLD
    if not moving.any():
        # No clear motion detected — use full video
        return 0.0, total_frames / fps

    first_idx = np.argmax(moving)
    last_idx = len(moving) - 1 - np.argmax(moving[::-1])

    start_s = max(0.0, times[first_idx] - PAD_S)
    end_s = min(total_frames / fps, times[last_idx] + PAD_S)

    return round(start_s, 2), round(end_s, 2)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video-dir", default=os.path.join(REPO, "videos"))
    ap.add_argument("--output", default=os.path.join(REPO, "videos", "trim_points.json"))
    args = ap.parse_args()

    videos = sorted(f for f in os.listdir(args.video_dir) if f.endswith(".mp4"))
    if not videos:
        print("No .mp4 files found in", args.video_dir)
        sys.exit(1)

    results = {}
    for i, fname in enumerate(videos):
        path = os.path.join(args.video_dir, fname)
        start_s, end_s = detect_motion_bounds(path)
        duration = end_s - start_s
        results[fname] = {"start_s": start_s, "end_s": end_s, "duration_s": round(duration, 2)}
        print(f"[{i:2d}] {fname}: {start_s:.2f} - {end_s:.2f}  ({duration:.1f}s trimmed)")

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
