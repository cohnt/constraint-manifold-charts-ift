"""Physical comparison of retimed trajectories: length, energy, duration, profiles.

The quantity that most directly exposes stop-start motion is mean speed relative to
peak speed, i.e. the duty factor L / (T * v_peak). TOPPRA is time-optimal subject to
scaled velocity and acceleration limits, so a well-conditioned path rides its limit
for most of the leg and the duty factor approaches 1. A trajectory that repeatedly
decelerates and re-accelerates covers the same arc length in more time at the same
peak, so the duty factor falls -- which is the "same path length, longer duration"
signature.

Reported per leg, all in the 20 torso+arm DOFs:
    L        joint-space path length (rad)
    T        duration (s)
    L/T      mean speed (rad/s)
    v_peak   peak speed (rad/s)
    duty     L / (T * v_peak), 1.0 = constant speed at the peak
    energy   integral ||qdot||^2 dt (rad^2/s)
    a_max    peak acceleration magnitude (rad/s^2)
"""
import argparse
import os
import pickle
import sys

import numpy as np

# Resolve src/ so this runs without `pip install -e .`, like its sibling reports.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import plan_format  # noqa: F401  -- installs the pre-rename module aliases the
                    # committed plan and execution pickles were written with

DOFS = slice(3, 23)


def leg_kinematics(q, duration):
    q = np.asarray(q, dtype=float)[:, DOFS]
    n = q.shape[0]
    T = max(float(duration), 1e-9)
    dt = T / (n - 1)
    v = np.diff(q, axis=0) / dt
    sp = np.linalg.norm(v, axis=1)
    a = np.diff(v, axis=0) / dt
    ac = np.linalg.norm(a, axis=1)
    L = float(np.linalg.norm(np.diff(q, axis=0), axis=1).sum())
    v_peak = float(sp.max()) if sp.size else 0.0
    return dict(
        L=L, T=T, mean_speed=L / T,
        v_peak=v_peak,
        duty=(L / (T * v_peak)) if v_peak > 0 else 0.0,
        energy=float((sp**2).sum() * dt),
        a_max=float(ac.max()) if ac.size else 0.0,
        a_rms=float(np.sqrt((ac**2).mean())) if ac.size else 0.0,
    )


def profile(q, duration, width=64, deriv=1):
    q = np.asarray(q, dtype=float)[:, DOFS]
    n = q.shape[0]
    dt = max(float(duration), 1e-9) / (n - 1)
    v = np.diff(q, axis=0) / dt
    sig = np.linalg.norm(v, axis=1) if deriv == 1 else \
        np.linalg.norm(np.diff(v, axis=0) / dt, axis=1)
    if sig.size < 2:
        return ""
    idx = np.linspace(0, len(sig) - 1, width).astype(int)
    s = sig[idx] / (sig.max() or 1)
    return "".join(" .:-=+*#%@"[min(9, int(x * 9.99))] for x in s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", required=True,
                    help="LABEL:cache_dir pairs")
    ap.add_argument("--profiles", type=int, default=None,
                    help="also print velocity/acceleration profiles for this point")
    args = ap.parse_args()

    per_set = {}
    for spec in args.sets:
        label, d = spec.split(":", 1)
        legs, plans = {}, []
        import glob
        for p in sorted(glob.glob(f"{d}/point_[0-9][0-9].pkl")):
            with open(p, "rb") as f:
                meta = pickle.load(f).get("meta", {}) or {}
            tot_L = tot_T = 0.0
            for lg in meta.get("legs") or []:
                k = leg_kinematics(lg["q"], lg["duration"])
                legs.setdefault(lg["name"], []).append(k)
                tot_L += k["L"]; tot_T += k["T"]
            if tot_T:
                plans.append((tot_L, tot_T))
        per_set[label] = (legs, plans)

    print("=== WHOLE PLAN (all legs summed), medians")
    print(f"{'set':<16}{'plans':>6}{'L (rad)':>10}{'T (s)':>9}{'L/T':>9}")
    for label, (legs, plans) in per_set.items():
        if not plans:
            continue
        L = np.median([a for a, _ in plans]); T = np.median([b for _, b in plans])
        print(f"{label:<16}{len(plans):>6}{L:>10.2f}{T:>9.1f}{L/T:>9.3f}")

    print("\n=== PER LEG, medians")
    print(f"{'set':<16}{'leg':<16}{'n':>4}{'L':>7}{'T':>7}{'L/T':>8}"
          f"{'v_peak':>8}{'duty':>7}{'energy':>9}{'a_max':>8}{'a_rms':>8}")
    for label, (legs, _) in per_set.items():
        for name, ks in legs.items():
            m = lambda k: np.median([x[k] for x in ks])
            print(f"{label:<16}{name:<16}{len(ks):>4}{m('L'):>7.2f}{m('T'):>7.1f}"
                  f"{m('mean_speed'):>8.3f}{m('v_peak'):>8.3f}{m('duty'):>7.2f}"
                  f"{m('energy'):>9.3f}{m('a_max'):>8.3f}{m('a_rms'):>8.3f}")
        print()

    if args.profiles is not None:
        i = args.profiles
        print(f"=== PROFILES, point {i}   ' '=0  '@'=peak (normalised per leg)")
        for label, _ in ((l, None) for l in per_set):
            d = dict(s.split(":", 1) for s in args.sets)[label]
            try:
                with open(f"{d}/point_{i:02d}.pkl", "rb") as f:
                    meta = pickle.load(f)["meta"]
            except FileNotFoundError:
                continue
            print(f"\n{label}")
            for lg in meta["legs"]:
                print(f"  {lg['name']:<15} |v| {profile(lg['q'], lg['duration'], deriv=1)}")
                print(f"  {'':<15} |a| {profile(lg['q'], lg['duration'], deriv=2)}")


if __name__ == "__main__":
    raise SystemExit(main())
