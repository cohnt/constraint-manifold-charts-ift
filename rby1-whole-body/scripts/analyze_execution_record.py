"""Diagnose an execution record: what the robot did vs. what it was asked.

Reads the ``<plan>_exec_<timestamp>.pkl`` written by the robot client (not part of this release)
and the plan it came from, aligns actual joint state against the commanded
waypoints using ``command_timestamps``, and reports which of a fixed set of
failure signatures the run matches. The point is to replace "the arms looked
out of sync" with a number.

The signatures, and what each would mean:

``pacing``
    Leg wall-clock duration vs. the plan's own. A leg running short means
    waypoints were issued faster than the plan's timing -- the whole motion is
    rushed, uniformly, both arms together.

``lag``
    Tracking error regressed against *commanded* joint velocity, per chain. A
    first-order lag (a command filter, or servo dynamics) gives
    ``error ~= -tau * velocity``, so the slope estimates an effective ``tau``
    and hence a cutoff frequency. Crucially, compare ``tau`` between the two
    arms: **equal** tau is a common-mode effect, and the arms differ visibly
    only because they move at different speeds; **unequal** tau is something
    chain-specific.

``dropout``
    A chain sitting still while the plan says it should be moving, then
    surging. That is control ending for that chain -- the one mechanism that
    produces *visible* desync rather than a few milliradians of it.

``stall``
    Gaps between consecutive waypoint sends far above the streaming period.
    Blocking I/O in the control loop (stdout over SSH), GC, or scheduling.

``gripper``
    How far apart two grippers in one step actually actuated.

Usage:
    python scripts/analyze_execution_record.py run_exec_20260810_120000.pkl
    python scripts/analyze_execution_record.py REC --plan plans/grid_cache/point_00.pkl
    python scripts/analyze_execution_record.py REC --plot out.png
    python scripts/analyze_execution_record.py --self-test
"""

import argparse
import os
import pickle
import sys

import numpy as np
import plan_format  # noqa: F401  -- installs the pre-rename module aliases the
                    # committed plan and execution pickles were written with

# Body slice of the server state vector (utilities.parse_state_vector):
# wheels [0:2], torso [2:8], right arm [8:15], left arm [15:22], head [22:24].
# 2:22 is torso+right+left, contiguous and in the same order as the commanded
# whole-body vector, so no reordering is needed.
BODY = slice(2, 22)
CHAINS = {"torso": slice(0, 6), "right": slice(6, 13), "left": slice(13, 20)}

# A chain is "moving" (for dropout purposes) above this commanded speed, and
# "stopped" below this measured speed.
MOVING_RAD_S = 0.02
STOPPED_RAD_S = 0.002
DROPOUT_MIN_S = 0.15          # ignore blips shorter than this
STALL_FACTOR = 3.0            # a send gap this many x the median counts as a stall


# --------------------------------------------------------------------- load

def load_record(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def commanded_from_plan(plan_path):
    """{step name: (n_waypoints, 20) commanded body positions} + the waypoint dt."""
    with open(plan_path, "rb") as f:
        payload = pickle.load(f)
    out, dt = {}, None
    for step in payload.get("steps", []):
        if step.get("type") != "trajectory" or not step.get("cmds"):
            continue
        cmds = step["cmds"]
        out[step["name"]] = np.array([
            np.concatenate([c["torso"].target_position,
                            c["right"].target_position,
                            c["left"].target_position]) for c in cmds])
        dt = dt or float(cmds[0]["right"].duration)
    return out, dt


def _as_logs(entry):
    """Server responses for a step: a dict, or a list of them (one per gripper
    when the client still sent grippers as separate calls)."""
    logs = entry.get("logs")
    if logs is None:
        return []
    return logs if isinstance(logs, list) else [logs]


# ----------------------------------------------------------------- analysis

def align(actual_t, command_t):
    """Index of the waypoint in force at each actual sample (-1 before the first)."""
    return np.searchsorted(command_t, actual_t, side="right") - 1


def fit_tau(err, vel):
    """Least-squares tau in ``err ~= -tau * vel``, plus the correlation.

    A pure first-order lag between command and response shows up as tracking
    error proportional to commanded velocity; the slope is the time constant.
    """
    mask = np.abs(vel) > 1e-4
    if mask.sum() < 20:
        return None, None
    v, e = vel[mask], err[mask]
    tau = -float(np.dot(v, e) / np.dot(v, v))
    denom = np.linalg.norm(v) * np.linalg.norm(e)
    corr = float(np.dot(v, e) / denom) if denom > 0 else 0.0
    return tau, corr


def windowed_speed(q, t, win_s=DROPOUT_MIN_S):
    """Per-joint |displacement|/elapsed over a sliding window, shape as ``q``.

    Used instead of a pointwise gradient because the commanded signal is a
    staircase -- 20 Hz waypoints sampled at 100 Hz -- so a difference of
    adjacent samples reads zero on every tread and spikes on every riser.
    A window wider than the waypoint period measures motion, not sampling.
    """
    n = len(t)
    w = max(int(round(win_s / np.median(np.diff(t)))), 2)
    lo = np.clip(np.arange(n) - w // 2, 0, n - 1)
    hi = np.clip(np.arange(n) + w // 2, 0, n - 1)
    dt = np.maximum(t[hi] - t[lo], 1e-6)
    return np.abs(q[hi] - q[lo]) / dt[:, None]


def find_dropouts(t, actual_vel, commanded_speed, valid):
    """Spans where a chain is ~stopped while the plan says it should be moving.

    ``valid`` gates the search to samples where a dropout would actually be
    *this chain's* fault: not inside a send stall (nothing was commanded, so
    everything is legitimately still) and only while some other chain is
    measurably moving. Without that gate a global pause reads as every chain
    dropping out at once, which is the opposite of the desync being hunted.
    """
    idle = ((np.abs(actual_vel) < STOPPED_RAD_S)
            & (commanded_speed > MOVING_RAD_S) & valid)
    spans, start = [], None
    for i, flag in enumerate(idle):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            if t[i - 1] - t[start] >= DROPOUT_MIN_S:
                spans.append((t[start], t[i - 1]))
            start = None
    if start is not None and t[-1] - t[start] >= DROPOUT_MIN_S:
        spans.append((t[start], t[-1]))
    return spans


def analyse_leg(name, log, cmd, dt):
    """Metrics for one trajectory step. Returns None if the log is unusable."""
    ct = np.asarray(log.get("command_timestamps", []), dtype=float)
    at = np.asarray(log.get("timestamp", []), dtype=float)
    pos = np.asarray(log.get("joint_position", []), dtype=float)
    if ct.size < 2 or at.size < 2 or pos.ndim != 2:
        return None

    n_wp = min(len(ct), len(cmd))
    ct, cmd = ct[:n_wp], cmd[:n_wp]
    body = pos[:, BODY]

    # Restrict to samples taken while the leg was actually streaming.
    live = (at >= ct[0]) & (at <= ct[-1] + dt)
    at, body = at[live], body[live]
    if len(at) < 5:
        return None

    idx = np.clip(align(at, ct), 0, n_wp - 1)
    target = cmd[idx]
    err = body - target

    # Rate the commanded target actually advances, from the *measured* send
    # intervals rather than the plan's dt. They differ whenever pacing is off,
    # and using the plan's dt would inflate every tau estimate by exactly the
    # pacing ratio -- which would then be misread as a slower command filter.
    send_dt = np.maximum(np.diff(ct), 1e-6)
    cmd_vel_wp = np.vstack([np.zeros((1, cmd.shape[1])),
                            np.diff(cmd, axis=0) / send_dt[:, None]])
    cmd_vel = cmd_vel_wp[idx]

    gaps = np.diff(ct)
    res = {
        "name": name,
        "n_waypoints": n_wp,
        "n_samples": len(at),
        "state_hz": log.get("state_hz"),
        "planned_s": (len(cmd) - 1) * dt,
        "wall_s": float(ct[-1] - ct[0]),
        "gap_median_ms": float(np.median(gaps) * 1000),
        "gap_p95_ms": float(np.percentile(gaps, 95) * 1000),
        "gap_max_ms": float(gaps.max() * 1000),
        "stalls": [(float(ct[i]), float(g * 1000))
                   for i, g in enumerate(gaps)
                   if g > STALL_FACTOR * np.median(gaps)],
        "endpoint_gap_mrad": float(np.abs(err[-1]).max() * 1000),
        "chains": {},
    }
    res["pacing_ratio"] = res["wall_s"] / res["planned_s"] if res["planned_s"] else float("nan")

    # Samples that fall inside an abnormally long gap between sends: no new
    # command was in flight, so stillness there says nothing about a chain.
    stall_lo = np.median(gaps) * STALL_FACTOR
    in_stall = np.zeros(len(at), dtype=bool)
    for i, g in enumerate(gaps):
        if g > stall_lo:
            in_stall |= (at >= ct[i]) & (at <= ct[i + 1])

    wspeed = {ch: windowed_speed(body[:, sl], at) for ch, sl in CHAINS.items()}

    for chain, sl in CHAINS.items():
        e, v = err[:, sl], cmd_vel[:, sl]
        tau, corr = fit_tau(e.ravel(), v.ravel())
        others_moving = np.any(
            [wspeed[o].max(axis=1) > MOVING_RAD_S for o in CHAINS if o != chain],
            axis=0)
        valid = others_moving & ~in_stall
        drops = []
        for j in range(sl.stop - sl.start):
            drops += find_dropouts(at, wspeed[chain][:, j], np.abs(v[:, j]), valid)
        res["chains"][chain] = {
            "mean_lag_mrad": float(np.abs(e).mean() * 1000),
            "peak_lag_mrad": float(np.abs(e).max() * 1000),
            "tau_ms": None if tau is None else tau * 1000,
            "corr": corr,
            "implied_fc_hz": (None if not tau or tau <= 0
                              else 1.0 / (2 * np.pi * tau)),
            "dropouts": drops,
        }
    return res


def gripper_spread(entry):
    """Seconds between the first sends of a step's gripper commands."""
    stamps = []
    for log in _as_logs(entry):
        ct = np.asarray(log.get("command_timestamps", []), dtype=float)
        if ct.size:
            stamps.append(float(ct[0]))
    return (max(stamps) - min(stamps)) if len(stamps) > 1 else 0.0


# ------------------------------------------------------------------ report

def report(results, gripper_steps):
    verdicts = []
    for r in results:
        print(f"\n=== {r['name']} ===")
        print(f"  {r['n_waypoints']} waypoints, {r['n_samples']} state samples "
              f"@ {r['state_hz']} Hz")
        print(f"  duration      {r['wall_s']:6.2f}s vs {r['planned_s']:6.2f}s planned "
              f"({r['pacing_ratio']:.3f}x)")
        print(f"  send interval median {r['gap_median_ms']:6.1f}ms  "
              f"p95 {r['gap_p95_ms']:6.1f}ms  max {r['gap_max_ms']:7.1f}ms")
        print(f"  endpoint gap  {r['endpoint_gap_mrad']:.2f} mrad")
        for chain, c in r["chains"].items():
            tau = "  n/a" if c["tau_ms"] is None else f"{c['tau_ms']:6.1f}ms"
            fc = "" if not c["implied_fc_hz"] else f" (~{c['implied_fc_hz']:.1f} Hz)"
            print(f"    {chain:6s} lag mean {c['mean_lag_mrad']:7.2f} "
                  f"peak {c['peak_lag_mrad']:7.2f} mrad   tau {tau}{fc}"
                  f"  r={c['corr']:+.2f}")
            if c["dropouts"]:
                t0 = r.get("_t0", 0)
                spans = ", ".join(f"{a-t0:.2f}-{b-t0:.2f}s" for a, b in c["dropouts"][:4])
                print(f"           DROPOUT: {chain} idle while commanded: {spans}")

        # --- signatures ---
        if abs(r["pacing_ratio"] - 1.0) > 0.05:
            verdicts.append(
                f"[pacing]  '{r['name']}' ran at {r['pacing_ratio']:.2f}x its planned "
                f"duration ({r['wall_s']:.2f}s vs {r['planned_s']:.2f}s).")
        if r["stalls"]:
            worst = max(g for _, g in r["stalls"])
            verdicts.append(
                f"[stall]   '{r['name']}' had {len(r['stalls'])} send gap(s) over "
                f"{STALL_FACTOR:.0f}x the median, worst {worst:.0f}ms — blocking I/O "
                f"in the control loop.")
        dropped = [ch for ch, c in r["chains"].items() if c["dropouts"]]
        if dropped:
            verdicts.append(
                f"[dropout] '{r['name']}': {', '.join(dropped)} stopped while the plan "
                f"was still commanding motion. This is the signature that produces "
                f"visible desync.")
        taus = {ch: c["tau_ms"] for ch, c in r["chains"].items()
                if c["tau_ms"] and c["corr"] < -0.3}
        if "right" in taus and "left" in taus:
            tr, tl = taus["right"], taus["left"]
            rel = abs(tr - tl) / max(tr, tl)
            if max(tr, tl) > 20:
                kind = ("COMMON-MODE — both arms lag equally; they look different only "
                        "because they move at different speeds"
                        if rel < 0.25 else
                        "CHAIN-SPECIFIC — the arms lag by different amounts")
                verdicts.append(
                    f"[lag]     '{r['name']}': tau right {tr:.0f}ms / left {tl:.0f}ms. "
                    f"{kind}.")

    for name, spread in gripper_steps:
        print(f"\n=== {name} (gripper) ===")
        print(f"  grippers actuated {spread*1000:.0f} ms apart")
        if spread > 0.1:
            verdicts.append(
                f"[gripper] '{name}': grippers actuated {spread*1000:.0f} ms apart — "
                f"serialised into separate round-trips.")

    print("\n" + "=" * 72)
    if verdicts:
        print("SIGNATURES DETECTED")
        for v in verdicts:
            print("  " + v)
    else:
        print("No signature fired: pacing within 5%, no stalls, no dropouts, "
              "no velocity-proportional lag above 20ms.")
    return verdicts


def plot(results, records, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(results), 1, figsize=(11, 3.1 * len(results)),
                             squeeze=False)
    for ax, (r, (at, err)) in zip(axes[:, 0], zip(results, records)):
        t = at - at[0]
        for chain, sl in CHAINS.items():
            ax.plot(t, np.abs(err[:, sl]).mean(axis=1) * 1000, label=chain, lw=1.2)
        ax.set_title(f"{r['name']} — tracking error "
                     f"({r['pacing_ratio']:.2f}x planned duration)", fontsize=10)
        ax.set_ylabel("mean |error| (mrad)")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(alpha=0.3)
    axes[-1, 0].set_xlabel("time within leg (s)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"\nWrote {path}")


# --------------------------------------------------------------- self-test

def _synth(signature, n_wp=120, dt=0.05, hz=100.0):
    """A record exhibiting one known signature, for validating the analyser."""
    t_wp = np.arange(n_wp) * dt
    cmd = np.zeros((n_wp, 20))
    for j in range(20):
        cmd[:, j] = 0.3 * np.sin(2 * np.pi * (0.15 + 0.02 * j) * t_wp)
    if signature == "quiet_left":            # left arm barely moves
        cmd[:, CHAINS["left"]] *= 0.05

    pace = 0.80 * dt if signature == "pacing" else 0.95 * dt
    ct = 1000.0 + np.arange(n_wp) * pace
    if signature == "stall":
        ct[n_wp // 2:] += 0.9

    at = np.arange(ct[0], ct[-1], 1.0 / hz)
    idx = np.clip(np.searchsorted(ct, at, side="right") - 1, 0, n_wp - 1)
    body = cmd[idx].copy()

    if signature in ("lag", "quiet_left"):   # first-order lag, tau = 40ms
        tau, a = 0.040, (1.0 / hz) / (0.040 + 1.0 / hz)
        y = body[0].copy()
        for i in range(len(body)):
            y = y + a * (body[i] - y)
            body[i] = y
    if signature == "dropout":               # right arm freezes, then jumps
        lo, hi = len(at) // 3, len(at) // 3 + int(0.6 * hz)
        body[lo:hi, CHAINS["right"]] = body[lo, CHAINS["right"]]

    pos = np.zeros((len(at), 24))
    pos[:, BODY] = body
    return ct, at, pos, cmd, dt


def self_test():
    print("Self-test: synthetic records with known signatures\n")
    # A stall legitimately stretches the leg, so it fires 'pacing' too -- that
    # is a true observation, not a false positive. It must NOT fire 'dropout':
    # nothing was commanded during the gap.
    expect = {"clean": set(), "pacing": {"pacing"}, "stall": {"stall", "pacing"},
              "dropout": {"dropout"}, "lag": {"lag"},
              "quiet_left": {"lag"}}
    failures = []
    for sig, want in expect.items():
        ct, at, pos, cmd, dt = _synth(sig)
        log = {"command_timestamps": ct, "timestamp": at,
               "joint_position": pos, "state_hz": 100.0}
        r = analyse_leg(sig, log, cmd, dt)
        if r is None:
            failures.append(f"{sig}: analyse_leg returned None")
            continue
        got = set()
        if abs(r["pacing_ratio"] - 1.0) > 0.05:
            got.add("pacing")
        if r["stalls"]:
            got.add("stall")
        if any(c["dropouts"] for c in r["chains"].values()):
            got.add("dropout")
        taus = {ch: c["tau_ms"] for ch, c in r["chains"].items()
                if c["tau_ms"] and c["corr"] < -0.3 and c["tau_ms"] > 20}
        if {"right", "left"} <= set(taus):
            got.add("lag")
        ok = got == want
        detail = ""
        if sig in ("lag", "quiet_left"):
            tr = r["chains"]["right"]["tau_ms"]
            tl = r["chains"]["left"]["tau_ms"]
            detail = f"  tau R={tr:.0f}ms L={tl:.0f}ms (injected 40ms)"
        print(f"  {'pass' if ok else 'FAIL'}  {sig:11s} detected={sorted(got) or '[]'}"
              f"{detail}")
        if not ok:
            failures.append(f"{sig}: expected {sorted(want)}, got {sorted(got)}")

    print()
    if failures:
        for f in failures:
            print("  " + f)
        return 1
    print("All self-tests passed.")
    return 0


# -------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("record", nargs="?", help="Execution record pickle.")
    ap.add_argument("--plan", help="Plan pickle. Defaults to the record's plan_path, "
                                   "which may not resolve off the robot.")
    ap.add_argument("--plot", metavar="PNG", help="Also write a tracking-error plot.")
    ap.add_argument("--self-test", action="store_true",
                    help="Validate the analyser against synthetic records.")
    args = ap.parse_args()

    if args.self_test:
        return self_test()
    if not args.record:
        ap.error("a record path is required (or --self-test)")

    rec = load_record(args.record)
    plan_path = args.plan or rec.get("plan_path")
    if not plan_path or not os.path.exists(plan_path):
        ap.error(f"plan not found at {plan_path!r}; pass --plan explicitly")

    commanded, dt = commanded_from_plan(plan_path)
    print(f"record   {args.record}")
    print(f"plan     {plan_path}")
    print(f"executed {rec.get('executed_at')}  completed={rec.get('completed')}"
          + (f"  error={rec['error']}" if rec.get("error") else ""))

    results, series, grippers = [], [], []
    for entry in rec.get("steps", []):
        if entry.get("type") == "gripper":
            grippers.append((entry["name"], gripper_spread(entry)))
            continue
        cmd = commanded.get(entry["name"])
        if cmd is None:
            print(f"  (skipping '{entry['name']}': not in the plan)")
            continue
        for log in _as_logs(entry):
            r = analyse_leg(entry["name"], log, cmd, dt)
            if r is None:
                print(f"  (skipping '{entry['name']}': log lacks "
                      f"command_timestamps/timestamp — recorded before the "
                      f"logging branch?)")
                continue
            at = np.asarray(log["timestamp"], dtype=float)
            ct = np.asarray(log["command_timestamps"], dtype=float)
            r["_t0"] = float(ct[0])
            live = (at >= ct[0]) & (at <= ct[-1] + dt)
            body = np.asarray(log["joint_position"], dtype=float)[live][:, BODY]
            idx = np.clip(align(at[live], ct[:min(len(ct), len(cmd))]),
                          0, min(len(ct), len(cmd)) - 1)
            results.append(r)
            series.append((at[live], body - cmd[idx]))

    if not results and not grippers:
        print("\nNothing analysable in this record.")
        return 1
    report(results, grippers)
    if args.plot and results:
        plot(results, series, args.plot)
    return 0


if __name__ == "__main__":
    sys.exit(main())
