"""Runtime instrumentation: a fork-safe event recorder for the planning pipeline.

The pipeline's runtime cannot be described by a dict of stage sums, because a
"stage" here is one wall-clock interval that may wrap a whole fork wave (up to
five grasp-candidate children plus a reach child inside a single grid point).
Summing such intervals produces a number that is neither latency (the children
overlap) nor CPU cost (killed children vanish).

So this module records a *timeline* instead: one event per unit of work, with an
id, its parent's id, absolute start/end timestamps, and CPU deltas. Sums, the
critical path, and the overlap between them are all derivable from that; none of
them is recoverable from a sum. See the folder README for the
schema and the two accounting views built on it.

Usage::

    from exps.timing_utils import record, mark, start_recording

    start_recording("plans/grid_cache/timing/point_05.jsonl")
    with record("trajopt.solve", "lift"):
        result = SnoptSolver().Solve(prog)
        mark(solver_info=result.get_solver_details().info)

``stage_timer`` is kept as a shim over the same machinery so the legacy
``meta["timings"]`` layout (and everything that reads it) is unchanged.
"""

from __future__ import annotations

import json
import os
import resource
import threading
import time
from contextlib import contextmanager
from typing import Optional


# ── The recorder ──────────────────────────────────────────────────────────────
#
# Records are appended to a JSONL file rather than accumulated and returned,
# because the processes whose cost we most need to see are exactly the ones that
# cannot return anything: the reach child that is killed when its grasp loses
# (plan_grid._fork_reach), a wedged grasp child, a point worker
# killed by WORKER_MEM_CAP. A file that each process appends to survives all
# three, so discarded work reconciles against wall time instead of leaving a gap.
#
# Every write is a single os.write() of one line to a fd opened O_APPEND. POSIX
# makes such a write atomic with respect to other appenders up to PIPE_BUF
# (4096 on Linux), so concurrent children interleave whole lines, never partial
# ones. _MAX_LINE enforces that budget rather than trusting it.

_MAX_LINE = 4000

_fd: Optional[int] = None       # append fd, or None when recording is off
_counter: int = 0
_prefix: str = ""               # per-process id prefix, re-seeded after fork


class _Local(threading.local):
    """Open records, per thread.

    Thread-local rather than global because the grid driver plans several points
    concurrently in *threads* of one process (`_plan_many_parallel`), each
    thread waiting on its own forked child. A shared stack would interleave
    those threads' blocks and mis-parent every record. Under `fork` only the
    forking thread exists in the child, and it inherits exactly the stack that
    was open at the fork site -- which is what links the child's tree to its
    parent's.
    """

    def __init__(self):
        self.stack = []
        self.pending = {}


_tl = _Local()


_id_lock = threading.Lock()


def _new_id() -> str:
    """A process-unique id. Locked because ids are handed out across threads."""
    global _counter
    with _id_lock:
        _counter += 1
        return f"{_prefix}.{_counter}"


def _cpu() -> tuple:
    """(self, children) CPU seconds so far: utime + stime from getrusage."""
    s = resource.getrusage(resource.RUSAGE_SELF)
    c = resource.getrusage(resource.RUSAGE_CHILDREN)
    return (s.ru_utime + s.ru_stime, c.ru_utime + c.ru_stime)


def _reseed(path: Optional[str] = None) -> None:
    """Reset this process's id namespace (and optionally reopen the log)."""
    global _counter, _prefix, _fd
    _counter = 0
    _prefix = f"p{os.getpid()}"
    if path is not None:
        _fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)


def start_recording(path: str) -> None:
    """Begin recording events to ``path`` (created if absent, appended if not).

    Idempotent per process: calling it again reopens the log, which is what a
    forked child wants if it is redirected to a different file. Recording is off
    until this is called, so every ``record`` block below is a no-op by default
    and importing this module costs nothing.
    """
    _tl.stack = []
    _tl.pending = {}
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    _reseed(path)


def stop_recording() -> None:
    global _fd
    if _fd is not None:
        try:
            os.close(_fd)
        except OSError:
            pass
        _fd = None


def recording() -> bool:
    return _fd is not None


def _child_after_fork() -> None:
    """Give the child its own id namespace and no inherited open records.

    The id prefix is the pid, so ids stay unique across the whole forest without
    coordination. The *stack* is deliberately kept: the child's first record
    should hang off whatever record was open at the fork site in the parent, and
    that is exactly what stitches the per-process trees into one.

    ``_pending`` is cleared instead: those entries belong to the parent's blocks,
    which this process will never close, and mark()-ing them here would write a
    duplicate line for a record the parent is also going to write.
    """
    _tl.pending = {}
    if _fd is not None:
        _reseed()


os.register_at_fork(after_in_child=_child_after_fork)


def _emit(rec: dict) -> None:
    if _fd is None:
        return
    try:
        line = json.dumps(rec, default=str)
        if len(line) > _MAX_LINE:                       # keep the append atomic
            rec = dict(rec, extra={"truncated": True})
            line = json.dumps(rec, default=str)
        os.write(_fd, (line + "\n").encode())
    except (OSError, TypeError, ValueError):
        pass                                            # never break a planner run


@contextmanager
def record(cat: str, label: Optional[str] = None, **extra):
    """Time one unit of work and emit an event for it.

    ``cat`` is the category the analysis aggregates on (``"trajopt.solve"``,
    ``"ik.attempt"``, ...); ``label`` names the instance (the leg, the solver).
    Parentage is taken from the enclosing ``record`` block in this process, so
    nesting needs no plumbing.

    The event is written in a ``finally``, so work that raises is still recorded
    (with ``error`` set) -- "trajopt failed after 40 s" and "trajopt failed
    immediately" are different facts and both need to survive.
    """
    if _fd is None:
        yield None
        return
    rid = _new_id()
    rec = {
        "id": rid,
        "parent": _tl.stack[-1] if _tl.stack else None,
        "pid": os.getpid(),
        "cat": cat,
        "label": label,
        "t0": time.time(),
        "extra": dict(extra),
    }
    cpu0 = _cpu()
    _tl.stack.append(rid)
    _tl.pending[rid] = rec
    # An "open" line, written before the work starts, so a process that is
    # KILLED mid-block still leaves evidence of the frame it died in. Without it
    # the child's completed inner records point at a parent id that was never
    # written, and the killed frame -- which is precisely the discarded work we
    # are trying to account for -- would be unnameable. The close line below
    # carries the same id; the analysis keeps whichever is more complete.
    _emit({"id": rid, "parent": rec["parent"], "pid": rec["pid"], "cat": cat,
           "label": label, "t0": rec["t0"], "open": True})
    try:
        yield rec
    except BaseException as e:
        rec["error"] = f"{type(e).__name__}: {str(e)[:120]}"
        raise
    finally:
        rec["t1"] = time.time()
        cpu1 = _cpu()
        rec["cpu_self"] = round(cpu1[0] - cpu0[0], 6)
        rec["cpu_children"] = round(cpu1[1] - cpu0[1], 6)
        # Defensive: start_recording() resets the stack, so a block that spans
        # a (re)start would otherwise pop someone else's frame or an empty list.
        # The recorder must never be the thing that breaks a planning run.
        if _tl.stack and _tl.stack[-1] == rid:
            _tl.stack.pop()
        elif rid in _tl.stack:
            _tl.stack.remove(rid)
        _tl.pending.pop(rid, None)
        _emit(rec)


def mark(**extra) -> None:
    """Attach fields to the innermost open record (solver status, counts, ...).

    A no-op when recording is off or no record is open, so call sites need no
    guard.
    """
    if _fd is None or not _tl.stack:
        return
    rec = _tl.pending.get(_tl.stack[-1])
    if rec is not None:
        rec["extra"].update(extra)


def mark_discarded(value: bool = True) -> None:
    """Flag the innermost open record as work that did not reach the plan.

    Losing grasp seeds, failed GCP branches, IK draws rejected by the CoM or
    collision check, leg attempts that failed their dense verify. The analysis
    reports these separately rather than dropping them: they cost CPU, and (when
    they are on the critical path) latency too.

    Not for work that succeeded but was not selected -- which candidate wins is
    generally unknown while it is running. Label those instead; see the `phase`
    field on ``ik.attempt``.
    """
    if _fd is None or not _tl.stack:
        return
    rec = _tl.pending.get(_tl.stack[-1])
    if rec is not None:
        rec["discarded"] = bool(value)


def current_id() -> Optional[str]:
    """Id of the innermost open record, for callers that need to cross-link."""
    return _tl.stack[-1] if _tl.stack else None


def emit_meta(cat: str, **extra) -> None:
    """Emit a zero-duration event (run configuration, machine identity, notes)."""
    if _fd is None:
        return
    now = time.time()
    _emit({"id": _new_id(), "parent": _tl.stack[-1] if _tl.stack else None,
           "pid": os.getpid(), "cat": cat, "label": None,
           "t0": now, "t1": now, "cpu_self": 0.0, "cpu_children": 0.0,
           "extra": dict(extra)})


# ── Legacy stage timer ────────────────────────────────────────────────────────

@contextmanager
def stage_timer(timings: Optional[dict], key: str):
    """Record wall-clock seconds for a pipeline stage into ``timings[key]``.

    Everything before the ``yield`` runs on entering the ``with`` block;
    everything after runs on exit. The elapsed time is written in a ``finally``
    block, so a stage that raises still records its duration before the
    exception propagates (useful for distinguishing "stage X failed quickly"
    from "stage X ran long then failed").

    If ``timings`` is None this is a no-op timer, so callers that don't want
    instrumentation can pay no overhead -- but the event is still emitted when
    recording is on, because the event log's completeness must not depend on
    whether a particular caller happened to pass a dict.

    ``meta["timings"]`` keeps its old shape so grid_timing_report.py,
    grid_provenance_report.py and every already-cached plan still read. It is
    superseded for anything latency-shaped: see scripts/timing_report.py.

    Example:
        timings = {}
        with stage_timer(timings, "ik"):
            q_goal = solve_ik(...)
        # timings == {"ik": 0.0123}
    """
    t0 = time.perf_counter()
    # cat is "stage.<key>", so the analysis can report what a stage spent
    # *outside* every instrumented leaf under it as its own self time.
    with record(f"stage.{key}", key):
        try:
            yield
        finally:
            if timings is not None:
                timings[key] = time.perf_counter() - t0
