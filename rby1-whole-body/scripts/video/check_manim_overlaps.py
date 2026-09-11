"""Render a still of every settled composition in the manim scenes and report
text that collides with other elements.

A *settled* composition is the scene contents at the moment a self.wait()
begins -- exactly the frames the viewer has time to read, and the only ones
where an overlap matters. For each one this

  1. writes a 1920x1080 PNG for eyeball review, and
  2. renders every text mobject and every non-text mobject in *isolation* and
     intersects their ink masks, so a collision is reported per pair with the
     number of colliding pixels.

Ink intersection rather than bounding-box intersection, because a bbox test
calls every label near the curved manifold a collision and misses a caption
that clips a thin stroke.

A label centred inside a filled container (the pipeline stage boxes) is a
deliberate composition, not an overlap, so those pairs are suppressed --
see CONTAINER_FILL_MIN.

Usage:
    .venv/bin/python scripts/video/check_manim_overlaps.py
    .venv/bin/python scripts/video/check_manim_overlaps.py --outdir /tmp/frames
    .venv/bin/python scripts/video/check_manim_overlaps.py --scene RBY1PipelineScene
"""

import argparse
import json
import os
import sys

import numpy as np
from manim import *

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BG_COLOR = "#1a1a2e"

TEXT_TYPES = (Text, MarkupText, MathTex, Tex, SingleStringMathTex)

# A pixel counts as inked if it differs from the background by more than this
# in any channel. Low enough to catch a 2px stroke's antialiased edge.
INK_TOL = 12

# A graphic with at least this much fill that fully contains a text mobject is
# treated as that text's container.
CONTAINER_FILL_MIN = 0.01

SCENES = ["IKParameterizationScene", "RBY1PipelineScene"]


def classify(mobjects):
    """Split a scene's mobject list into (text_nodes, other_leaf_nodes).

    Text mobjects are kept whole -- their per-glyph submobjects are not
    descended into, so a collision is reported against the caption a viewer
    sees rather than against the letter "g".
    """
    texts, others = [], []

    def walk(m):
        if isinstance(m, TEXT_TYPES):
            texts.append(m)
            return
        kids = list(m.submobjects)
        if kids:
            for k in kids:
                walk(k)
        elif getattr(m, "points", None) is not None and len(m.points):
            others.append(m)

    for m in mobjects:
        walk(m)
    return texts, others


def to_mask(img, bg_rgb):
    arr = np.array(img.convert("RGB")).astype(np.int16)
    return np.abs(arr - bg_rgb[None, None, :]).max(axis=2) > INK_TOL


def contains(outer, inner, pad=0.02):
    return (outer.get_left()[0] - pad <= inner.get_left()[0]
            and outer.get_right()[0] + pad >= inner.get_right()[0]
            and outer.get_bottom()[1] - pad <= inner.get_bottom()[1]
            and outer.get_top()[1] + pad >= inner.get_top()[1])


def is_container(graphic, text):
    fill = graphic.get_fill_opacity() if hasattr(graphic, "get_fill_opacity") else 0
    return fill >= CONTAINER_FILL_MIN and contains(graphic, text)


def label_of(m):
    if isinstance(m, TEXT_TYPES):
        raw = getattr(m, "text", None) or getattr(m, "tex_string", None) or ""
        raw = " ".join(str(raw).split())
        if len(raw) > 46:
            raw = raw[:43] + "..."
        return f"{type(m).__name__}('{raw}')"
    return type(m).__name__


def geom_of(m):
    c = m.get_center()
    return f"c=({c[0]:+.2f},{c[1]:+.2f}) w={m.width:.2f} h={m.height:.2f}"


class OverlapCheckMixin:
    """Snapshot and analyse the composition at the start of every wait()."""

    shot_prefix = "scene"
    report = None      # shared list, injected by main()
    outdir = "."

    def setup(self):
        self._shot_i = 0
        self._cam = Camera(background_color=BG_COLOR)
        self._bg_rgb = None
        super().setup()

    def wait(self, duration=DEFAULT_WAIT_TIME, *args, **kwargs):
        self._shoot(duration)
        return super().wait(duration, *args, **kwargs)

    def _render(self, mobject_list):
        self._cam.reset()
        self._cam.capture_mobjects(mobject_list)
        return self._cam.get_image()

    def _shoot(self, duration):
        self._shot_i += 1
        tag = f"{self.shot_prefix}_{self._shot_i:02d}"

        if self._bg_rgb is None:
            self._bg_rgb = np.array(self._render([]).convert("RGB"))[0, 0].copy()
        bg = self._bg_rgb

        mobs = list(self.mobjects)
        texts, others = classify(mobs)

        png = os.path.join(self.outdir, f"{tag}.png")
        self._render(mobs).save(png)

        tmasks = [to_mask(self._render([t]), bg) for t in texts]
        omasks = [to_mask(self._render([o]), bg) for o in others]

        collisions = []
        for i, t in enumerate(texts):
            ti = tmasks[i]
            npix = max(int(ti.sum()), 1)
            for j, o in enumerate(others):
                if is_container(o, t):
                    continue
                n = int((ti & omasks[j]).sum())
                if n:
                    collisions.append({
                        "kind": "text-vs-graphic", "pixels": n,
                        "frac_of_a": round(n / npix, 4),
                        "a": label_of(t), "a_geom": geom_of(t),
                        "b": label_of(o), "b_geom": geom_of(o),
                    })
            for j in range(i + 1, len(texts)):
                n = int((ti & tmasks[j]).sum())
                if n:
                    collisions.append({
                        "kind": "text-vs-text", "pixels": n,
                        "frac_of_a": round(n / npix, 4),
                        "a": label_of(t), "a_geom": geom_of(t),
                        "b": label_of(texts[j]), "b_geom": geom_of(texts[j]),
                    })

        half_w, half_h = config.frame_width / 2, config.frame_height / 2
        offframe = []
        for t in texts:
            left, right = t.get_left()[0], t.get_right()[0]
            bottom, top = t.get_bottom()[1], t.get_top()[1]
            if left < -half_w or right > half_w or bottom < -half_h or top > half_h:
                offframe.append({"text": label_of(t), "geom": geom_of(t)})

        collisions.sort(key=lambda c: -c["pixels"])
        self.report.append({
            "tag": tag, "png": png, "wait_s": duration,
            "n_text": len(texts), "n_graphic": len(others),
            "collisions": collisions, "offframe": offframe,
        })

        flag = ""
        if collisions:
            flag += f"  <-- {len(collisions)} collision(s), worst {collisions[0]['pixels']}px"
        if offframe:
            flag += f"  <-- {len(offframe)} off-frame"
        print(f"[{tag}] wait={duration:.1f}s text={len(texts)} gfx={len(others)}{flag}",
              flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--module", default="manim_scenes_v2",
                    help="scene module in scripts/video/ (default: manim_scenes_v2)")
    ap.add_argument("--scene", action="append", dest="scenes",
                    help="scene class to check; repeatable (default: both v2 scenes)")
    ap.add_argument("--outdir", default=None,
                    help="where to write the PNGs (default: <repo>/video/manim_frames)")
    args = ap.parse_args()

    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    outdir = args.outdir or os.path.join(repo, "video", "manim_frames")
    os.makedirs(outdir, exist_ok=True)

    module = __import__(args.module)
    scene_names = args.scenes or SCENES
    report = []

    # frame_rate is low on purpose: the settled compositions do not depend on
    # it, and it cuts the pass from minutes to seconds. TracedPath trails come
    # out coarser than in the real render, which only affects how much ink a
    # trail contributes -- never where the text sits.
    with tempconfig({
        "pixel_width": 1920, "pixel_height": 1080, "frame_rate": 10,
        "write_to_movie": False, "dry_run": True,
        "disable_caching": True, "verbosity": "ERROR",
    }):
        for name in scene_names:
            base = getattr(module, name)
            checker = type(
                f"Check{name}", (OverlapCheckMixin, base),
                {"shot_prefix": name, "report": report, "outdir": outdir},
            )
            print(f"\n=== {name} ===", flush=True)
            checker().render()

    with open(os.path.join(outdir, "report.json"), "w") as f:
        json.dump(report, f, indent=2)

    print("\n==================== SUMMARY ====================")
    bad = [e for e in report if e["collisions"] or e["offframe"]]
    for e in bad:
        print(f"\n{e['tag']}  ({e['png']})  wait={e['wait_s']}s")
        for c in e["collisions"]:
            print(f"   {c['kind']:17s} {c['pixels']:6d}px  ({c['frac_of_a'] * 100:.1f}% of A)")
            print(f"       A: {c['a']}  {c['a_geom']}")
            print(f"       B: {c['b']}  {c['b_geom']}")
        for o in e["offframe"]:
            print(f"   OFF-FRAME  {o['text']}  {o['geom']}")

    print(f"\n{len(bad)}/{len(report)} settled states have problems.")
    print(f"Stills: {outdir}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
