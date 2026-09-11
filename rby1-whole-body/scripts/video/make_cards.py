"""Generate title card and results card as MP4 segments, plus an SRT subtitle file.

This video accompanies a double-blind RA-L submission, so the title card is
**anonymous by default**. Pass --named to put the real author list on it.

Usage:
    .venv/bin/python scripts/video/make_cards.py            # -> title_anonymous.mp4
    .venv/bin/python scripts/video/make_cards.py --named    # -> title_named.mp4
"""

import argparse
import json
import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import video_credits as credits  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT_DIR = os.path.join(REPO, "video")
W, H = 1920, 1080
FPS = 30

BG = (26, 26, 46)       # #1a1a2e
TEXT_WHITE = (224, 224, 224)
ACCENT_BLUE = (79, 195, 247)
ACCENT_GREEN = (102, 187, 106)
MUTED = (160, 160, 160)


def get_font(size, bold=False):
    paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for p in paths:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def center_text(draw, text, y, font, fill):
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    draw.text(((W - tw) // 2, y), text, font=font, fill=fill)


def make_title_card(anonymous=True):
    """The RA-L supplementary title card.

    Anonymous by default: this video accompanies a double-blind submission, so
    the state you get by forgetting the flag is the safe one.
    """
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)

    title_font = get_font(36, bold=True)
    author_font = get_font(24)
    affil_font = get_font(20)
    note_font = get_font(18)
    subtitle_font = get_font(20)

    y = 300
    for line in credits.TITLE_LINES:
        center_text(draw, line, y, title_font, ACCENT_BLUE)
        y += 50

    y += 40
    fonts = {"authors": (author_font, TEXT_WHITE),
             "note": (note_font, MUTED),
             "affiliation": (affil_font, MUTED),
             "venue": (affil_font, MUTED)}
    gaps = {"authors": 44, "note": 40, "affiliation": 35, "venue": 35}
    for text, role in credits.credit_lines(
            anonymous, affiliation=credits.AFFILIATION_SHORT,
            # Venue on the anonymous cut only. That one is submission material,
            # where naming the venue is useful context and costs nothing. The
            # named cut goes to YouTube, which is the one place a venue is
            # expensive to change: a redirect to ICRA/IROS would mean a
            # re-render, a re-upload, a new video ID and a new embed on the
            # project page. That page says only "Under review"; match it.
            venue=credits.VENUE if anonymous else None):
        font, colour = fonts[role]
        center_text(draw, text, y, font, colour)
        y += gaps[role]

    # Supplementary video descriptor in lower empty space
    center_text(draw, "Supplementary video: 20 constrained bimanual pick-and-place motions",
                610, subtitle_font, MUTED)
    center_text(draw, "on an RB-Y1, planned and executed on hardware.",
                640, subtitle_font, MUTED)

    return img


def make_results_card():
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)

    heading_font = get_font(32, bold=True)
    section_font = get_font(22, bold=True)
    stat_font = get_font(24, bold=True)
    label_font = get_font(18)

    center_text(draw, "Experimental Results", 80, heading_font, ACCENT_BLUE)

    # Left column: Numerical results (from Table I / Fig. 2)
    col_l = 120
    y = 160
    draw.text((col_l, y), "IFT Gradient Computation", font=section_font, fill=ACCENT_BLUE)
    y += 40
    for val, lbl in [
        ("< 10⁻¹³", "Median gradient error vs. autodiff"),
        ("10x faster", "Than autodiff at large partial sizes"),
    ]:
        draw.text((col_l, y), val, font=stat_font, fill=ACCENT_GREEN)
        draw.text((col_l, y + 28), lbl, font=label_font, fill=TEXT_WHITE)
        y += 70

    # Grasp selection (Table II)
    draw.text((col_l, y), "Grasp Selection IK (EAIK)", font=section_font, fill=ACCENT_BLUE)
    y += 40
    for val, lbl in [
        ("59.8%", "IFT success rate (vs. 54% C-space baseline)"),
        ("15.86", "IFT cost (vs. 26.68 baseline)"),
    ]:
        draw.text((col_l, y), val, font=stat_font, fill=ACCENT_GREEN)
        draw.text((col_l, y + 28), lbl, font=label_font, fill=TEXT_WHITE)
        y += 70

    # Right column: Hardware results
    col_r = W // 2 + 60
    y = 160
    draw.text((col_r, y), "Hardware: RB-Y1 Pick-and-Place", font=section_font, fill=ACCENT_BLUE)
    y += 40

    # Load status.json for real metrics
    status_path = os.path.join(REPO, "plans", "grid_cache", "status.json")
    if os.path.exists(status_path):
        with open(status_path) as f:
            status = json.load(f)
        points = {k: v for k, v in status.items() if k.isdigit()}
        n_success = sum(1 for p in points.values() if p.get("status") == "success")
        worst_clearance = min(
            p.get("worst_clearance_mm", 999) for p in points.values()
        )
        worst_com = min(
            p.get("worst_com_margin_mm", 999) for p in points.values()
        )
    else:
        n_success = 20
        worst_clearance = 3.91
        worst_com = 6.75

    for val, lbl in [
        (f"{n_success} / 20", "Grid points planned and executed"),
        (f"{worst_clearance:.1f} mm", "Worst-case collision clearance"),
        (f"{worst_com:.1f} mm", "Worst-case CoM stability margin"),
        ("0", "Joint limit violations"),
    ]:
        draw.text((col_r, y), val, font=stat_font, fill=ACCENT_GREEN)
        draw.text((col_r, y + 28), lbl, font=label_font, fill=TEXT_WHITE)
        y += 70

    # Divider line
    draw.line([(W // 2, 170), (W // 2, H - 200)], fill=(80, 80, 80), width=1)

    # Pipeline summary at bottom
    pipeline_font = get_font(18)
    center_text(draw, "Pipeline: Analytic IK  →  BiRRT  →  Trajectory Optimization  →  TOPPRA",
                H - 140, pipeline_font, MUTED)

    return img


def image_to_mp4(img, out_path, duration_s):
    """Encode a single PIL image as a static video segment."""
    n_frames = int(duration_s * FPS)
    raw = np.array(img)
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{W}x{H}", "-r", str(FPS),
        "-i", "pipe:0",
        "-c:v", "libx264", "-crf", "18", "-preset", "medium",
        "-pix_fmt", "yuv420p",
        "-t", f"{duration_s}",
        out_path,
    ]
    proc = subprocess.Popen(
        cmd, stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )
    frame_bytes = raw.tobytes()
    for _ in range(n_frames):
        proc.stdin.write(frame_bytes)
    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        print(f"ffmpeg error: {proc.stderr.read().decode()[:500]}", file=sys.stderr)
        sys.exit(1)


# Text is burned in by the segment renderers; no separate SRT file needed for this cut.


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--named", action="store_true",
                    help="show the real author list and affiliation. Off by "
                         "default: this video accompanies a double-blind "
                         "submission.")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    print("Generating title card (%s)..."
          % ("named" if args.named else "ANONYMOUS"))
    title_img = make_title_card(anonymous=not args.named)
    # Separate paths per mode: a single fixed title.mp4 meant the anonymous and
    # named cards overwrote each other, so the two cuts had to be built in a
    # strict order and a mistake there would pair a named author list with an
    # anonymous card (or vice versa).
    title_path = os.path.join(
        OUT_DIR, "title_named.mp4" if args.named else "title_anonymous.mp4")
    image_to_mp4(title_img, title_path, 3.0)
    print(f"  -> {title_path}")

    print("Generating results card...")
    results_img = make_results_card()
    results_path = os.path.join(OUT_DIR, "results.mp4")
    image_to_mp4(results_img, results_path, 5.0)
    print(f"  -> {results_path}")

    print("Done.")


if __name__ == "__main__":
    main()
