"""Generate title, closing and results cards for Video 2 (promotional).

Usage:
    .venv/bin/python scripts/video/make_cards_v2.py
    .venv/bin/python scripts/video/make_cards_v2.py --anonymous
    .venv/bin/python scripts/video/make_cards_v2.py --backdrop B
    .venv/bin/python scripts/video/make_cards_v2.py --all-variants
    .venv/bin/python scripts/video/make_cards_v2.py --alpha-sweep
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import video_credits as credits  # noqa: E402

# Private by name only. Renaming them public would edit ral_explainers.py, which
# marks the RA-L segment renders stale and forces a re-render of an already
# shipped video, so the promo card reaches in instead.
from ral_explainers import _load_grid_counts, _load_measured_violation  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
VIDEO_DIR = os.path.join(REPO, "video")
REVIEW_DIR = os.path.join(REPO, "scratch", "overview_review")

W, H = 1920, 1080
FPS = 30
BG = (0x1a, 0x1a, 0x2e)
BG_HEX = "0x1a1a2e"
WHITE = (0xe0, 0xe0, 0xe0)
BLUE = (0x4f, 0xc3, 0xf7)
MUTED = (0x88, 0x88, 0x99)
# MUTED was picked against the flat #1a1a2e card background and nearly
# vanishes over the title card's video backdrop -- used only there, only
# for the note/affiliation lines. The closing card is still flat and keeps
# plain MUTED.
MUTED_ON_VIDEO = (0xb8, 0xb8, 0xc8)
GREEN = (0x66, 0xbb, 0x6a)

FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def image_to_mp4(img, out_path, duration_s):
    arr = np.array(img.convert("RGB"))
    n_frames = int(duration_s * FPS)
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{W}x{H}", "-pix_fmt", "rgb24",
        "-r", str(FPS),
        "-i", "-",
        "-c:v", "libx264", "-crf", "18", "-preset", "medium",
        "-pix_fmt", "yuv420p", "-an",
        "-t", str(duration_s),
        out_path,
    ]
    pipe = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    frame_bytes = arr[:, :, :3].tobytes()
    for _ in range(n_frames):
        pipe.stdin.write(frame_bytes)
    pipe.stdin.close()
    pipe.wait()


def _centre(draw, text, y, font, fill, stroke_width=0, stroke_fill=None):
    bbox = draw.textbbox((0, 0), text, font=font, stroke_width=stroke_width)
    draw.text(((W - (bbox[2] - bbox[0])) // 2, y), text, fill=fill, font=font,
              stroke_width=stroke_width, stroke_fill=stroke_fill)


def _draw_credits(draw, anonymous, for_video=False):
    """Draw the title + credit block onto `draw`.

    make_title is the only caller: the closing card is a byte-for-byte copy of
    the title card, so there is one credits layout and one render, and the
    video opens and closes on the same image -- the bookend property
    the video pipeline's own stage table records.

    `for_video=True` adds a thin opaque-black stroke to
    every line and lightens the note/affiliation colour to MUTED_ON_VIDEO --
    both needed only because those lines sit over a photographic backdrop
    instead of the flat card background the plain colours were chosen
    against. The stroke is drawn in the alpha channel too (the canvas is
    RGBA with a transparent background), so it survives compositing rather
    than only showing up in the standalone PNG.
    """
    try:
        font_title = ImageFont.truetype(FONT_BOLD_PATH, 36)
        font_authors = ImageFont.truetype(FONT_PATH, 20)
        font_note = ImageFont.truetype(FONT_PATH, 15)
        font_affil = ImageFont.truetype(FONT_PATH, 16)
    except OSError:
        font_title = font_authors = font_note = font_affil = \
            ImageFont.load_default()

    stroke_width = 1 if for_video else 0
    stroke_fill = (0, 0, 0, 255) if for_video else None
    muted = MUTED_ON_VIDEO if for_video else MUTED

    y = 280
    for line in credits.TITLE_LINES_NARROW:
        _centre(draw, line, y, font_title, BLUE, stroke_width, stroke_fill)
        y += 50

    y += 40
    fonts = {"authors": (font_authors, WHITE),
             "note": (font_note, muted),
             "affiliation": (font_affil, muted)}
    gaps = {"authors": 34, "note": 32, "affiliation": 30}
    # venue=None: the promo deliberately does not name a submission venue.
    # video_credits.credit_lines() already guards on `if venue:`, so this is
    # the only change needed -- the RA-L card (make_cards.py) keeps its own
    # venue line by continuing to pass the default.
    for text, role in credits.credit_lines(anonymous, venue=None, separator=",  "):
        font, colour = fonts[role]
        _centre(draw, text, y, font, colour, stroke_width, stroke_fill)
        y += gaps[role]


def make_title(anonymous=False):
    """Credits rendered onto a transparent RGBA canvas, for both cards.

    Only the glyphs are opaque -- title_with_backdrop() composites this over
    the sped-up video backdrop with a plain ffmpeg overlay, so everywhere
    that isn't text shows the backdrop through untouched.
    """
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    _draw_credits(draw, anonymous, for_video=True)
    return img


def make_results():
    """Flat results card: headline numbers loaded, not typed.

    `_load_grid_counts` / `_load_measured_violation` read
    `plans/grid_cache/status.json` and `video/error_data.pkl` and fall back to
    the frozen published literals on any data hiccup, so
    this card cannot silently drift from the RA-L montage panel that uses the
    same two loaders.
    """
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)

    font_h1 = ImageFont.truetype(FONT_BOLD_PATH, 40)
    font_big = ImageFont.truetype(FONT_BOLD_PATH, 54)
    font_body = ImageFont.truetype(FONT_PATH, 24)
    font_h2 = ImageFont.truetype(FONT_BOLD_PATH, 24)
    font_stat = ImageFont.truetype(FONT_BOLD_PATH, 32)
    font_small = ImageFont.truetype(FONT_PATH, 18)

    n_success, n_total = _load_grid_counts()
    v = _load_measured_violation()

    _centre(draw, "Results", 250, font_h1, BLUE)
    _centre(draw, f"{n_success} / {n_total}", 340, font_big, GREEN)
    _centre(draw, "grid points planned and executed on hardware", 412, font_body, WHITE)

    _centre(draw, "Constraint violation, measured", 505, font_h2, BLUE)
    # 2 dp, not 3: at 3 dp the live pool prints 0.554 mm mean against the
    # paper's 0.555 (ee_constraint_report.py windows each leg to its command
    # interval; the live pool here is very slightly different), which would
    # contradict the published numbers on screen. 2 dp is also exactly
    # what ral_explainers.results_panel shows in the montage panel later in
    # this same cut, so the two agree.
    _centre(draw,
            f"{v['pos_mean_mm']:.2f} mm mean          {v['pos_max_mm']:.2f} mm max",
            558, font_stat, WHITE)
    _centre(draw,
            f"{v['rot_mean_mrad']:.2f} mrad mean        {v['rot_max_mrad']:.2f} mrad max",
            603, font_stat, WHITE)
    _centre(draw, "position and orientation are reported separately —", 655, font_small, MUTED)
    _centre(draw, "metres cannot be added to radians", 682, font_small, MUTED)

    _centre(draw, "Planning time", 740, font_h2, BLUE)
    # Frozen paper values, not loaded: the folder README's headline table and the
    # paper's full-body IK table both report this same 20-point sequential run as
    # 43.6 s median / 56.1 s mean.
    _centre(draw, "43.6 s median          56.1 s mean", 790, font_stat, WHITE)

    return img


# ------------------------------------------------------- title backdrop


def _ffprobe_duration(path):
    out = subprocess.check_output([
        "ffprobe", "-v", "quiet", "-show_entries", "format=duration",
        "-of", "csv=p=0", path,
    ])
    return float(out.decode().strip())


def _ffprobe_resolution(path):
    out = subprocess.check_output([
        "ffprobe", "-v", "quiet", "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=p=0:s=x", path,
    ])
    return out.decode().strip().splitlines()[0]


TRIM_JSON_KEY = "20260811_171207.mp4"

# Backdrop opacity for the title card: a legibility-vs-presence tradeoff,
# chosen by eye from rendered frames, not derived from anything.
TITLE_DURATION_S = 4.0
# Equal on purpose: the video opens and closes on the same image, so the two
# cards are the same file. See the copy in main().
CLOSING_DURATION_S = TITLE_DURATION_S

BACKDROP_ALPHA_DEFAULT = 0.15

# variant -> (dir under REPO, filename, held-window length in seconds).
# A held-window of None means "the whole clip" (C) or "the full point-0
# window from trim_points.json" (A). Only B's 16.0 is a genuine hardcoded
# design choice -- a calmer sub-window of the same clip, starting at the same
# point; everything else (A's window, C's length) is read off disk in
# _variant_window so a re-trim or re-render upstream cannot silently drift
# the promo's speed-up out of sync with the source it names.
TITLE_BACKDROP_VARIANTS = {
    "A": ("videos", TRIM_JSON_KEY, None),
    "B": ("videos", TRIM_JSON_KEY, 16.0),
    "C": ("video", "drake_blender_00.mp4", None),
}


def _variant_window(variant, trim, duration_s=TITLE_DURATION_S):
    """Resolve a backdrop variant to (path, start_s, end_s, speed).

    Speed is always (window_length or clip_length) / duration_s -- derived,
    never typed, so the design doc's "11.6x / 4x / 9.46x" figures are a
    consequence of the clip lengths, not independent numbers that could drift
    out of sync with them. The closing card passes the shorter duration and
    therefore plays the same window slightly faster.
    """
    subdir, clip, hold_s = TITLE_BACKDROP_VARIANTS[variant]
    path = os.path.join(REPO, subdir, clip)
    if clip == TRIM_JSON_KEY:
        window = trim[TRIM_JSON_KEY]
        start_s = window["start_s"]
        end_s = start_s + hold_s if hold_s is not None else window["end_s"]
    else:
        start_s = 0.0
        end_s = hold_s if hold_s is not None else _ffprobe_duration(path)
    speed = (end_s - start_s) / duration_s
    return path, start_s, end_s, speed


def _title_filter_complex(speed, alpha):
    """The backdrop-composite filter graph, parameterised on speed (from
    _variant_window) and alpha (backdrop opacity, --backdrop-alpha):

        color                                             [base]
        <backdrop>, scale+crop, hue, format, colorchannelmixer   [bed]
        [base][bed] overlay=shortest=0                    [bg]
        [bg][2:v] overlay                                 [out]

    scale=...:force_original_aspect_ratio=increase + crop=1920:1080 fills the
    frame with no bars regardless of the source aspect -- variant C is
    960x910 and has to be upscaled to do this, which is expected and is the
    point of showing it in the review set.
    """
    return (
        f"[1:v]setpts=PTS/{speed:.6f},"
        f"scale=1920:1080:force_original_aspect_ratio=increase,"
        f"crop=1920:1080,hue=s=0.35,format=rgba,"
        f"colorchannelmixer=aa={alpha:.4f}[bed];"
        f"[0:v][bed]overlay=shortest=0[bg];"
        f"[bg][2:v]overlay[out]"
    )


def card_with_backdrop(credits_png, out_path, variant, trim,
                       alpha=BACKDROP_ALPHA_DEFAULT,
                       duration_s=TITLE_DURATION_S):
    """Composite the credits PNG over a sped-up, desaturated, low-opacity
    backdrop clip and write a `duration_s` clip to out_path.

    Used for both the title card and the closing card -- they are the same
    image over the same backdrop, differing only in how long they hold.
    """
    path, start_s, end_s, speed = _variant_window(variant, trim, duration_s)
    filter_complex = _title_filter_complex(speed, alpha)
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", f"color=c={BG_HEX}:s=1920x1080:d={duration_s:g}:r={FPS}",
        "-ss", str(start_s), "-to", str(end_s), "-i", path,
        "-loop", "1", "-t", f"{duration_s:g}", "-i", credits_png,
        "-filter_complex", filter_complex,
        "-map", "[out]",
        "-t", f"{duration_s:g}", "-r", str(FPS),
        "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-an",
        out_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ffmpeg error:\n{result.stderr[-1500:]}")
        sys.exit(1)

    dur = _ffprobe_duration(out_path)
    res = _ffprobe_resolution(out_path)
    if abs(dur - duration_s) > 0.05 or res != "1920x1080":
        print(f"warning: {out_path} is {dur:.3f}s @ {res}, "
              f"expected {duration_s:.2f}s @ 1920x1080")


# The frame offset (into the 4 s composited clip) used for every alpha-sweep
# still, so the four are lined up on the same backdrop content and only
# alpha differs. Chosen by eye: the robot (mid pick-and-place) is clearly in
# shot at variant A's default speed, matching the frame make_cards_v2.py's
# own review stills have used for variant A ("t=2" in the review workflow).
ALPHA_SWEEP_OFFSET_S = 2.0
ALPHA_SWEEP_VALUES = (0.08, 0.12, 0.15, 0.20)


def title_still(credits_png, out_path, variant, trim, alpha, offset_s):
    """Render a single composited frame (PNG, not a clip) at `offset_s` into
    the backdrop-composite timeline -- the alpha sweep only needs one frame
    per alpha to compare legibility, not four more 4 s clips."""
    path, start_s, end_s, speed = _variant_window(variant, trim)
    filter_complex = _title_filter_complex(speed, alpha)
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"color=c={BG_HEX}:s=1920x1080:d=4:r={FPS}",
        "-ss", str(start_s), "-to", str(end_s), "-i", path,
        "-loop", "1", "-t", "4", "-i", credits_png,
        "-filter_complex", filter_complex,
        "-map", "[out]",
        "-ss", str(offset_s), "-frames:v", "1",
        out_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ffmpeg error:\n{result.stderr[-1500:]}")
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--anonymous", action="store_true",
                    help="replace the author list and affiliation with "
                         "anonymous placeholders")
    ap.add_argument("--backdrop", choices=["A", "B", "C"], default="A",
                    help="title card backdrop variant, see TITLE_BACKDROP_VARIANTS")
    ap.add_argument("--all-variants", action="store_true",
                    help="write all three backdrop variants to "
                         "scratch/overview_review/ for review, instead of "
                         "writing video/v2_title.mp4")
    ap.add_argument("--backdrop-alpha", type=float, default=BACKDROP_ALPHA_DEFAULT,
                    help="backdrop opacity fed to colorchannelmixer=aa= "
                         "(default %(default)s)")
    ap.add_argument("--alpha-sweep", action="store_true",
                    help="write variant-A stills at ALPHA_SWEEP_VALUES to "
                         "scratch/overview_review/ for review, instead of "
                         "writing video/v2_title.mp4")
    ap.add_argument("--trim-json", default=os.path.join(REPO, "videos", "trim_points.json"))
    args = ap.parse_args()

    os.makedirs(VIDEO_DIR, exist_ok=True)
    with open(args.trim_json) as f:
        trim = json.load(f)

    if args.alpha_sweep:
        os.makedirs(REVIEW_DIR, exist_ok=True)
        with tempfile.TemporaryDirectory() as tmp:
            credits_png = os.path.join(tmp, "credits.png")
            make_title(anonymous=args.anonymous).save(credits_png)
            for alpha in ALPHA_SWEEP_VALUES:
                tag = f"{round(alpha * 100):03d}"
                out_path = os.path.join(REVIEW_DIR, f"title_alpha_{tag}.png")
                print(f"Rendering alpha={alpha} still...")
                title_still(credits_png, out_path, "A", trim, alpha,
                            ALPHA_SWEEP_OFFSET_S)
                print(f"  {out_path}")
        return

    if args.all_variants:
        os.makedirs(REVIEW_DIR, exist_ok=True)
        with tempfile.TemporaryDirectory() as tmp:
            credits_png = os.path.join(tmp, "credits.png")
            make_title(anonymous=args.anonymous).save(credits_png)
            for variant in ("A", "B", "C"):
                out_path = os.path.join(REVIEW_DIR, f"title_variant_{variant}.mp4")
                print(f"Generating title backdrop variant {variant}...")
                card_with_backdrop(credits_png, out_path, variant, trim,
                                   alpha=args.backdrop_alpha)
                print(f"  {out_path}")
        return

    print("Generating title card (%s, backdrop %s, alpha %s)..."
          % ("ANONYMOUS" if args.anonymous else "named", args.backdrop,
             args.backdrop_alpha))
    with tempfile.TemporaryDirectory() as tmp:
        credits_png = os.path.join(tmp, "credits.png")
        make_title(anonymous=args.anonymous).save(credits_png)
        title_path = os.path.join(VIDEO_DIR, "v2_title.mp4")
        card_with_backdrop(credits_png, title_path, args.backdrop, trim,
                           alpha=args.backdrop_alpha)
        print(f"  {title_path}")

        # The closing card IS the title card: same credits, same backdrop,
        # same hold. Rendering it a second time was verified to produce a
        # byte-identical file (matching sha256 on 2026-09-06), so this copies
        # instead -- which makes the bookend structural rather than a property
        # that happens to hold, and halves the card cost. Give CLOSING_DURATION_S
        # its own value if the two ever need to differ again; card_with_backdrop
        # already takes duration_s and derives the backdrop speed from it.
        print("Generating closing card...")
        closing_path = os.path.join(VIDEO_DIR, "v2_closing.mp4")
        if CLOSING_DURATION_S == TITLE_DURATION_S:
            shutil.copyfile(title_path, closing_path)
        else:
            card_with_backdrop(credits_png, closing_path, args.backdrop, trim,
                               alpha=args.backdrop_alpha,
                               duration_s=CLOSING_DURATION_S)
        print(f"  {closing_path}")

    print("Generating results card...")
    results_img = make_results()
    results_path = os.path.join(VIDEO_DIR, "v2_results.mp4")
    image_to_mp4(results_img, results_path, 6.0)
    print(f"  {results_path}")


if __name__ == "__main__":
    main()
