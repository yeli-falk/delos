"""Report the synthetic observation-noise sweep: latent- vs input-space prediction.

Reads ``outputs/synth_sweep/<arm>_n<noise>_s<seed>/test_metrics.json`` and
prints forecast MAE per (noise, arm), mean +- std over seeds, plus the
headline statistic: the PAIRED jepa_input - jepa_latent margin at each noise
level, computed within a seed.

Why paired: every arm in a (noise, seed) cell shares one pretrained encoder, so
the per-seed difference removes the encoder draw entirely and has far less
variance than either arm's absolute MAE.

What supports the claim: the margin should be ~0 at node_noise=0 and GROW as
observation noise rises. A flat margin is a null result and should be reported
as one -- the sweep is designed so that outcome is visible rather than
hideable. NOTE this only holds when the sweep was run with
``--synth_target_noise`` set (the default in slurm_synth_noise_sweep.sh):
without it the persistence baseline is immune to the noise axis, total
achievable headroom decays to ~0, and a flat margin says nothing about the
methods.

`skill` = 1 - MAE/MAE_persist, the fraction by which the arm beats copy-last.
Positive means the model beats persistence; negative means it does not, which
is itself informative.

Significance: the per-noise margin is reported with a standard error, a paired
two-sided t-test against 0, and a percentile bootstrap CI over seeds. With 5
noise levels tested, treat p<0.01 rather than p<0.05 as the bar (Bonferroni at
5 tests), and read the TREND across noise as the real evidence -- a single
significant cell surrounded by nulls is what multiplicity looks like.

Usage:
    python scripts/synthetic/aggregate_synth_sweep.py
    python scripts/synthetic/aggregate_synth_sweep.py --horizon 0 --metric MAE
    python scripts/synthetic/aggregate_synth_sweep.py --ref jepa_latent_ema
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics as st
from pathlib import Path

DIR_RE = re.compile(r"^(?P<arm>[a-z_]+)_n(?P<noise>[0-9.]+)_s(?P<seed>\d+)$")
ARM_ORDER = ["jepa_latent", "jepa_latent_ema", "jepa_input", "enc_only",
             "cldg", "ddgcl", "idol"]
BOOT = 10_000


def _cell(vals, places=4):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "-"
    m = st.mean(vals)
    s = st.stdev(vals) if len(vals) > 1 else 0.0
    return f"{m:.{places}f}±{s:.{places}f}"


def _t_sf(t: float, df: int) -> float:
    """Two-sided p-value for Student's t, via the regularised incomplete beta.

    Hand-rolled so the aggregator stays dependency-free (numpy/scipy are not
    guaranteed on the login node where this gets run).
    """
    if df <= 0:
        return float("nan")
    x = df / (df + t * t)
    return _betainc(df / 2.0, 0.5, x)


def _betainc(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta I_x(a, b) by continued fraction (NR 6.4)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
             + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(lbeta) * _betacf(a, b, x) / a
    return 1.0 - math.exp(lbeta) * _betacf(b, a, 1.0 - x) / b


def _betacf(a: float, b: float, x: float, itmax: int = 200, eps: float = 3e-12):
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1.0 / d
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        c = 1.0 + aa / c
        if abs(d) < 1e-30:
            d = 1e-30
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        c = 1.0 + aa / c
        if abs(d) < 1e-30:
            d = 1e-30
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _boot_ci(d, lo=2.5, hi=97.5, seed=0):
    """Percentile bootstrap CI of the mean. Fixed seed: reruns must agree."""
    if len(d) < 2:
        return None
    rng = _Lcg(seed)
    n = len(d)
    means = sorted(sum(d[rng.idx(n)] for _ in range(n)) / n for _ in range(BOOT))
    return (means[int(lo / 100 * BOOT)], means[min(int(hi / 100 * BOOT), BOOT - 1)])


class _Lcg:
    """Tiny deterministic RNG so the bootstrap needs no numpy."""

    def __init__(self, seed=0):
        self.s = (seed * 2 + 1) & 0xFFFFFFFF

    def idx(self, n):
        self.s = (1103515245 * self.s + 12345) & 0x7FFFFFFF
        return self.s % n


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="outputs/synth_sweep")
    p.add_argument("--horizon", default="0", help="Forecast step key to report.")
    p.add_argument("--metric", default="MAE", choices=["MAE", "RMSE", "MAE_inflect"])
    p.add_argument("--ref", default="jepa_latent",
                   help="Arm the paired margin is measured against "
                        "(margin = --vs MAE - --ref MAE; positive favours --ref).")
    p.add_argument("--vs", default="jepa_input", help="Arm compared to --ref.")
    args = p.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"{root} does not exist — has the sweep run yet?")

    # (noise, arm) -> {seed: (metric, persist)}
    runs: dict = {}
    for f in sorted(root.glob("*/test_metrics.json")):
        m = DIR_RE.match(f.parent.name)
        if not m:
            continue
        blk = json.loads(f.read_text()).get(args.horizon)
        if not blk:
            continue
        pk = ("MAE_persist_inflect" if args.metric == "MAE_inflect"
              else f"{args.metric.split('_')[0]}_persist")
        runs.setdefault((float(m.group("noise")), m.group("arm")), {})[
            int(m.group("seed"))] = (blk.get(args.metric), blk.get(pk))

    if not runs:
        raise SystemExit(f"no test_metrics.json with horizon '{args.horizon}' under {root}")

    # --- tripwire: every arm in a cell must have seen the SAME test data -----
    # MAE_persist is a property of the dataset, not the model, so it must agree
    # across arms within a (noise, seed). When it does not, some arm rebuilt a
    # different dataset (a stale checkpoint's saved args silently dropping a
    # flag is the known way this happens) and its row is not comparable.
    bad = []
    for n in sorted({k[0] for k in runs}):
        by_seed: dict = {}
        for (nn, arm), per_seed in runs.items():
            if nn != n:
                continue
            for s, (_, pers) in per_seed.items():
                if pers is not None:
                    by_seed.setdefault(s, {})[arm] = pers
        for s, arm_pers in by_seed.items():
            lo, hi = min(arm_pers.values()), max(arm_pers.values())
            if hi - lo > 1e-6 * max(abs(hi), 1.0):
                bad.append((n, s, arm_pers))
    if bad:
        print("\n" + "=" * 78)
        print("REFUSING TO COMPARE: persistence baseline differs across arms")
        print("=" * 78)
        for n, s, arm_pers in bad[:10]:
            print(f"  noise={n:.2f} seed={s}: " + "  ".join(
                f"{a}={v:.4f}" for a, v in sorted(arm_pers.items())))
        if len(bad) > 10:
            print(f"  ... and {len(bad) - 10} more cells")
        print("\nMAE_persist depends only on the data, so arms that disagree on"
              " it were\nprobed on DIFFERENT datasets and their MAEs are not "
              "comparable. Usual\ncause: an encoder-only arm rebuilt the "
              "dataset from a stale checkpoint's\nsaved args. Re-run those arms"
              " with the dataset flags passed explicitly.\n")
        raise SystemExit(2)

    noises = sorted({k[0] for k in runs})
    arms = ([a for a in ARM_ORDER if any(k[1] == a for k in runs)]
            + sorted({k[1] for k in runs} - set(ARM_ORDER)))

    print(f"\nSynthetic observation-noise sweep — {args.metric} at horizon "
          f"{args.horizon} (lower is better)\n")
    hdr = f"{'node_noise':>11}" + "".join(f"{a:>20}" for a in arms) + f"{'persist':>16}"
    print(hdr)
    print("-" * len(hdr))
    for n in noises:
        row = f"{n:>11.2f}"
        for a in arms:
            row += f"{_cell([v[0] for v in runs.get((n, a), {}).values()]):>20}"
        pers = [v[1] for k, vv in runs.items() if k[0] == n for v in vv.values()]
        row += f"{_cell(pers):>16}"
        print(row)

    # --- skill: how far each arm beats copy-last -----------------------------
    print(f"\nskill = 1 - {args.metric}/{args.metric}_persist  "
          f"(positive = beats copy-last)\n")
    print(hdr[:-16])
    print("-" * (len(hdr) - 16))
    for n in noises:
        row = f"{n:>11.2f}"
        for a in arms:
            sk = [1 - v[0] / v[1] for v in runs.get((n, a), {}).values()
                  if v[0] is not None and v[1]]
            row += f"{_cell(sk, places=3):>20}"
        print(row)

    # --- the headline: PAIRED margin, within seed ----------------------------
    # In skill units (margin / that seed's persistence MAE), not raw MAE:
    #   * the target's scale differs per seed (the generator redraws community
    #     anchors), so a raw margin weights the large-scale seeds more;
    #   * persistence MAE grows along the noise axis, so a margin that is
    #     constant in relative terms still grows in raw MAE, which is the
    #     pattern the claim predicts. The raw trend would confirm the claim
    #     from a scale artifact alone.
    print(f"\nPAIRED margin  ({args.vs} - {args.ref}), per seed, "
          f"as a fraction of that seed's persistence {args.metric}")
    print(f"positive = {args.ref} is BETTER (lower error). "
          f"0.01 = 1 skill point.\n")
    print(f"{'node_noise':>11}{'margin':>12}{'SE':>10}{'95% CI':>22}"
          f"{'p':>9}{'n':>5}{'rawMAE':>10}   verdict")
    print("-" * 100)
    margins = []
    for n in noises:
        ref = runs.get((n, args.ref), {})
        vs = runs.get((n, args.vs), {})
        seeds = [s for s in sorted(set(ref) & set(vs))
                 if ref[s][0] is not None and vs[s][0] is not None]
        raw = [vs[s][0] - ref[s][0] for s in seeds]
        # normalise each seed by its own persistence error
        d = [(vs[s][0] - ref[s][0]) / ref[s][1] for s in seeds if ref[s][1]]
        if not d:
            print(f"{n:>11.2f}{'-':>12}{'-':>10}{'-':>22}{'-':>9}{0:>5}{'-':>10}")
            continue
        m, mraw = st.mean(d), st.mean(raw)
        margins.append((n, m))
        if len(d) < 2:
            print(f"{n:>11.2f}{m:>12.4f}{'-':>10}{'-':>22}{'-':>9}{1:>5}"
                  f"{mraw:>10.4f}   1 seed — no inference possible")
            continue
        se = st.stdev(d) / math.sqrt(len(d))
        t = m / se if se > 0 else float("inf")
        pv = _t_sf(t, len(d) - 1) if se > 0 else 0.0
        ci = _boot_ci(d, seed=int(n * 100))
        ci_s = f"[{ci[0]:+.4f}, {ci[1]:+.4f}]" if ci else "-"
        v = ("null (CI spans 0)" if ci and ci[0] <= 0 <= ci[1] else
             f"{args.ref} better" if m > 0 else f"{args.vs} better")
        if pv > 0.01:
            v += "  [not sig. at Bonferroni 0.01]"
        print(f"{n:>11.2f}{m:>12.4f}{se:>10.4f}{ci_s:>22}{pv:>9.4f}{len(d):>5}"
              f"{mraw:>10.4f}   {v}")

    if len(margins) >= 2:
        (n0, m0), (n1, m1) = margins[0], margins[-1]
        print(f"\ntrend: margin goes {m0:+.4f} at noise={n0:.2f} -> {m1:+.4f} at "
              f"noise={n1:.2f}  ({m1 - m0:+.4f})   [skill units]")
        print("the CLAIM predicts this grows. If it is flat or shrinks, the")
        print("result is null and should be reported that way.")
        print("Before reading a null as evidence against the claim, check the")
        print("[synthetic_dyn] persistence line in the run logs: if MAE_persist")
        print("is identical at every noise level, --synth_target_noise was off")
        print("and the sweep had no headroom to detect anything with.")


if __name__ == "__main__":
    main()
