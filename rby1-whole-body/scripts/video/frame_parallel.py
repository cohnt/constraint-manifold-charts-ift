"""Render an ordered frame sequence across processes.

The overlay half of the meshcat -> Blender pipeline is pure CPU work -- open a
PNG, draw a matplotlib panel and some PIL text on it, hand the pixels to x264 --
and it was single-threaded, so a 570-frame segment sat on one core for minutes
while the other nineteen idled.  Every frame is an independent function of its
index, so the loop parallelises exactly.

Usage mirrors the loops it replaces::

    frames = parallel_frames(lambda: (lambda i: render_one(i)), n)
    encode(frames, out, W, H)

``make_renderer`` is called *once per worker* and returns the per-frame
function.  That indirection exists because the per-frame work usually reuses
expensive state -- a matplotlib Figure, loaded fonts -- which must not be shared
across processes but should not be rebuilt per frame either.

Two properties are load-bearing:

* **Order is preserved.**  Blocks come back in submission order, so the encoder
  sees frame i before frame i+1.  A frame sequence delivered out of order is not
  a crash, it is a silently scrambled video.
* **The pool is forked eagerly, before this returns.**  Callers pass the result
  straight into ``encode``, which has already opened a pipe to ffmpeg; a worker
  forked after that would inherit the write end of that pipe and ffmpeg would
  never see EOF.  Do not turn this function into a generator.
"""

from __future__ import annotations

import multiprocessing as mp
import os

import numpy as np

# Set in the parent immediately before the pool forks, and read in the child
# through fork inheritance.  Passing the factory as an argument instead would
# require it to be picklable, which rules out exactly the closures every caller
# here uses.
_MAKE_RENDERER = None
_RENDER = None


def frame_workers(cap=None):
    """How many processes to composite with.  ``VIDEO_JOBS`` overrides.

    One per hardware thread less one, so the machine keeps a thread for
    everything else -- in particular for the encoder on the consuming end of
    the frames these workers produce, which must not be starved by them.
    """
    env = os.environ.get("VIDEO_JOBS", "").strip()
    if env:
        return max(1, int(env))
    n = max(1, (os.cpu_count() or 1) - 1)
    return max(1, min(n, cap)) if cap else n


def _init():
    global _RENDER
    _RENDER = _MAKE_RENDERER()


def _render_block(bounds):
    start, stop = bounds
    return [np.ascontiguousarray(_RENDER(i), dtype=np.uint8)
            for i in range(start, stop)]


def parallel_frames(make_renderer, n_frames, *, workers=None, block=2,
                    progress_every=60, label="compositing"):
    """Iterator over frames ``0..n_frames-1``, rendered across processes.

    ``block`` frames are rendered per task and at most ``3 * workers`` tasks are
    in flight, which bounds the pixels held in memory (a 1080p frame is 6 MB, so
    the default window is a few hundred MB) while keeping every worker fed.
    """
    global _MAKE_RENDERER

    workers = workers or frame_workers()
    if workers == 1 or n_frames <= block:
        render = make_renderer()
        def serial():
            for i in range(n_frames):
                if progress_every and i % progress_every == 0:
                    print(f"  {label} frame {i}/{n_frames}", flush=True)
                yield np.ascontiguousarray(render(i), dtype=np.uint8)
        return serial()

    bounds = [(s, min(s + block, n_frames)) for s in range(0, n_frames, block)]
    _MAKE_RENDERER = make_renderer
    # fork, not spawn: the child inherits the factory above and the module state
    # the caller has already built (annotation arrays, the camera).
    pool = mp.get_context("fork").Pool(workers, initializer=_init)
    print(f"  {label} {n_frames} frames across {workers} processes", flush=True)

    def drain():
        pending = []
        emitted = 0
        try:
            for b in bounds:
                pending.append(pool.apply_async(_render_block, (b,)))
                while len(pending) >= 3 * workers:
                    for arr in pending.pop(0).get():
                        if progress_every and emitted % progress_every == 0:
                            print(f"  {label} frame {emitted}/{n_frames}",
                                  flush=True)
                        emitted += 1
                        yield arr
            while pending:
                for arr in pending.pop(0).get():
                    if progress_every and emitted % progress_every == 0:
                        print(f"  {label} frame {emitted}/{n_frames}", flush=True)
                    emitted += 1
                    yield arr
        finally:
            pool.terminate()
            pool.join()

    return drain()
