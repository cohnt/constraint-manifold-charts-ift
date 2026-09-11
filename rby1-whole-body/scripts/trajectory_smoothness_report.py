"""Quantify how piecewise-linear a cached plan's legs are.

Reads the sampled (n, 23) leg records straight out of the plan pickles, so it
works cross-branch without replanning. Two families of metric:

  kink  -- geometric, time-independent. Turning angle between consecutive
           joint-space segment directions. A piecewise-linear path is flat
           (0 deg) along each segment and spikes at the breakpoints; a smooth
           curve spreads a small turn over every sample. So max-turn and the
           count of samples above a threshold separate the two cleanly, and
           are immune to whatever TOPPRA did to the timing.
  jerk  -- dimensionless jerk, RMS(|d3q/dt3|) * T^3.5 / L, the standard
           movement-smoothness scalar. Sensitive to timing, reported for
           completeness.
"""
import argparse
import os
import pickle

import numpy as np
import sys

# Resolve src/ so this runs without `pip install -e .`, like its sibling reports.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import plan_format  # noqa: F401  -- installs the pre-rename module aliases the
                    # committed plan and execution pickles were written with

# Active-DOF layout inside the 23-vector: base(3) torso(6) right_arm(7) left_arm(7)
SLICES = {"all": slice(0, 23), "arms": slice(9, 23), "torso_arms": slice(3, 23)}


def leg_metrics(q, duration, dofs="torso_arms", kink_deg=5.0):
    q = np.asarray(q, dtype=float)[:, SLICES[dofs]]
    n = q.shape[0]
    T = max(float(duration), 1e-9)
    d = np.diff(q, axis=0)
    seg = np.linalg.norm(d, axis=1)
    L = float(seg.sum())
    live = seg > 1e-9
    u = d[live] / seg[live][:, None]
    if len(u) >= 2:
        cos = np.clip(np.einsum("ij,ij->i", u[:-1], u[1:]), -1.0, 1.0)
        turn = np.degrees(np.arccos(cos))
    else:
        turn = np.zeros(1)
    dt = T / (n - 1)
    d3 = np.diff(q, n=3, axis=0) / dt**3
    jerk_rms = float(np.sqrt(np.mean(np.sum(d3**2, axis=1)))) if len(d3) else 0.0
    dj = jerk_rms * T**3.5 / L if L > 1e-9 else 0.0
    return dict(
        n=n, T=T, L=L,
        turn_max=float(turn.max()), turn_sum=float(turn.sum()),
        turn_p99=float(np.percentile(turn, 99)),
        kinks=int((turn > kink_deg).sum()),
        # Turning per unit path length: the scale-free "how bendy" number.
        turn_per_m=float(turn.sum() / L) if L > 1e-9 else 0.0,
        dimensionless_jerk=float(dj),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pkl", nargs="+")
    ap.add_argument("--dofs", default="torso_arms", choices=list(SLICES))
    ap.add_argument("--kink-deg", type=float, default=5.0)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    per_leg = {}
    print(f"{'point':<7}{'leg':<15}{'n':>5}{'T(s)':>7}{'L(rad)':>8}"
          f"{'turnmax':>9}{'turnsum':>9}{'turn/m':>9}{'kinks':>7}{'DJ':>11}")
    for p in sorted(args.pkl):
        with open(p, "rb") as f:
            meta = pickle.load(f).get("meta", {}) or {}
        for lg in meta.get("legs") or []:
            m = leg_metrics(lg["q"], lg["duration"], args.dofs, args.kink_deg)
            per_leg.setdefault(lg["name"], []).append(m)
            print(f"{meta.get('index','?'):<7}{lg['name']:<15}{m['n']:>5}{m['T']:>7.1f}"
                  f"{m['L']:>8.2f}{m['turn_max']:>9.1f}{m['turn_sum']:>9.0f}"
                  f"{m['turn_per_m']:>9.1f}{m['kinks']:>7}{m['dimensionless_jerk']:>11.2e}")

    print(f"\n=== medians per leg {args.label} (dofs={args.dofs}, kink>{args.kink_deg} deg)")
    print(f"{'leg':<15}{'#':>4}{'L(rad)':>8}{'turnmax':>9}{'turnsum':>9}{'turn/m':>9}{'kinks':>7}{'DJ':>11}")
    for name, ms in per_leg.items():
        med = lambda k: np.median([m[k] for m in ms])
        print(f"{name:<15}{len(ms):>4}{med('L'):>8.2f}{med('turn_max'):>9.1f}"
              f"{med('turn_sum'):>9.0f}{med('turn_per_m'):>9.1f}"
              f"{med('kinks'):>7.0f}{med('dimensionless_jerk'):>11.2e}")


if __name__ == "__main__":
    raise SystemExit(main())
