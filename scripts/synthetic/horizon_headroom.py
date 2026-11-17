"""How much can a ROLLOUT gain over a direct head, per horizon, on a given
generator setting? Answered from the TRUE latents, before training anything.

Two oracles per horizon h, both linear (ridge) on clean latents, both decoded
with the generator's nearest-anchor rule:

  history oracle   z(t+h) from [z(t), z(t-1), ..., z(t-H+1)]   -- knows the
                   hidden velocity through the trajectory (what a rollout
                   can at best exploit)
  position oracle  z(t+h) from z(t) alone                       -- what a
                   direct head on a perfect single-snapshot readout can do

history minus position on SWITCH rows (state at t+h != state at t) is the
CEILING of the rollout's advantage at that horizon. If it is small or vanishes
after a few steps, no method can show a long-horizon gain on that setting,
whatever the arms score. Persistence is 0 on switch rows by definition and is
printed for the overall column.

Usage (repo root):
    python scripts/synthetic/horizon_headroom.py \
        --momentum 0.8 0.9 0.95 --theta 0.15 0.05 --sigma 0.35 --horizons 1 3 6 9 12
Other generator knobs default to scripts/dblp/_dataset_config.sh (synthetic_dyn).
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dynamic_graph_generator import generate_dynamic_graph  # noqa: E402


def _ridge(X, Y, lam=1e-3):
    Xb = np.c_[X, np.ones(len(X))]
    return np.linalg.solve(Xb.T @ Xb + lam * np.eye(Xb.shape[1]), Xb.T @ Y)


def _pred(W, X):
    return np.c_[X, np.ones(len(X))] @ W


def oracles(data, H, horizons, test_frac=0.3):
    z, c, anchors = data["latents"], data["labels"], data["anchors"]
    T, N, D = z.shape
    out = {}
    for h in horizons:
        ts = np.arange(H - 1, T - h)                      # last observed step t
        t_tr = ts[: int(len(ts) * (1 - test_frac))]
        t_te = ts[int(len(ts) * (1 - test_frac)):]

        def rows(tt, hist):
            X = np.concatenate([z[tt - j].reshape(-1, D) for j in range(hist)], axis=1)
            Y = z[tt + h].reshape(-1, D)
            c_now, c_fut = c[tt].reshape(-1), c[tt + h].reshape(-1)
            return X, Y, c_now, c_fut

        res = {}
        for name, hist in (("history", H), ("position", 1)):
            Xtr, Ytr, _, _ = rows(t_tr, hist)
            Xte, _, c_now, c_fut = rows(t_te, hist)
            zhat = _pred(_ridge(Xtr, Ytr), Xte)
            dec = np.linalg.norm(zhat[:, None, :] - anchors[None], axis=-1).argmin(1)
            sw = c_fut != c_now
            res[name] = (float((dec == c_fut).mean()),
                         float((dec[sw] == c_fut[sw]).mean()) if sw.any() else float("nan"))
            if name == "position":
                res["persist"] = float((c_now == c_fut).mean())
                res["switch_rate"] = float(sw.mean())
        out[h] = res
    return out


def velocity_r2(data, H, theta, test_frac=0.3, lam=1e-3):
    """R^2 to the hidden velocity v(t) from the TRUE position z(t) alone vs
    from the true history [z(t) .. z(t-H+1)]: the single-snapshot ceiling and
    the trajectory ceiling of the probe's `vel_r2_test`. A method whose
    representation lands at the position ceiling carries no hidden state."""
    z, c, anchors = data["latents"], data["labels"], data["anchors"]
    T, N, D = z.shape
    ts = np.arange(H - 1, T)
    v = z[ts] - z[ts - 1] - theta * (anchors[c[ts]] - z[ts - 1])       # (n_t, N, D)
    Y = v.reshape(-1, D)
    n_tr = int(len(ts) * (1 - test_frac)) * N
    out = {}
    for name, hist in (("position", 1), ("history", H)):
        X = np.concatenate([z[ts - j].reshape(-1, D) for j in range(hist)], axis=1)
        W = _ridge(X[:n_tr], Y[:n_tr], lam)
        pred = _pred(W, X[n_tr:])
        sse = ((Y[n_tr:] - pred) ** 2).sum()
        sst = ((Y[n_tr:] - Y[n_tr:].mean(0)) ** 2).sum()
        out[name] = 1.0 - sse / sst
    return out


def feature_velocity_r2(data, H, theta, test_frac=0.3, lam=1e-2):
    """Same as velocity_r2 but from the OBSERVED node features x(t) (one snapshot
    vs the H-snapshot history), plus position R^2 from x(t). This is the linear
    ceiling of the probe's vel_r2_test for ANY encoder: what the observation
    map lets through."""
    z, c, anchors, x = data["latents"], data["labels"], data["anchors"], data["node_features"]
    T, N, D = z.shape
    F = x.shape[-1]
    ts = np.arange(H - 1, T)
    v = (z[ts] - z[ts - 1] - theta * (anchors[c[ts]] - z[ts - 1])).reshape(-1, D)
    n_tr = int(len(ts) * (1 - test_frac)) * N

    def r2(X, Y):
        mu, sd = X[:n_tr].mean(0), X[:n_tr].std(0) + 1e-6
        X = (X - mu) / sd
        W = _ridge(X[:n_tr], Y[:n_tr], lam)
        P = _pred(W, X[n_tr:])
        return 1.0 - ((Y[n_tr:] - P) ** 2).sum() / ((Y[n_tr:] - Y[n_tr:].mean(0)) ** 2).sum()

    out = {"position": r2(x[ts].reshape(-1, F), v),
           "history": r2(np.concatenate([x[ts - j].reshape(-1, F) for j in range(H)], 1), v),
           "pos": r2(x[ts].reshape(-1, F), z[ts].reshape(-1, D))}
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--momentum", type=float, nargs="+", default=[0.8])
    p.add_argument("--theta", type=float, nargs="+", default=[0.15])
    p.add_argument("--sigma", type=float, nargs="+", default=[0.35])
    p.add_argument("--anchor_scale", type=float, default=1.0)
    p.add_argument("--nodes", type=int, default=200)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--history", type=int, default=3)
    p.add_argument("--horizons", type=int, nargs="+", default=[1, 3, 6, 9, 12])
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43])
    p.add_argument("--readout", choices=["tanh", "linear"], default="tanh")
    p.add_argument("--diffusion", type=float, default=0.3)
    p.add_argument("--node_noise", type=float, default=0.1)
    a = p.parse_args()
    gen_kw = dict(n_nodes=a.nodes, n_steps=a.steps, d_latent=8, d_node_feat=16, n_communities=4,
                  anchor_scale=a.anchor_scale, label_mode="nearest_anchor", hysteresis=0.0,
                  switch_mode="latent_driven", feature_readout=a.readout,
                  diffusion_alpha=a.diffusion, node_noise=a.node_noise)

    print(f"{'rho':>5} {'theta':>6} {'sigma':>6} | {'h':>3} {'switch%':>8} {'persist':>8} "
          f"{'pos.oracle':>11} {'hist.oracle':>12} | {'pos.switch':>11} {'hist.switch':>12} {'GAIN':>7}")
    for rho, th, sg in itertools.product(a.momentum, a.theta, a.sigma):
        acc = {}
        vr, fr = [], []
        for seed in a.seeds:
            data = generate_dynamic_graph(ou_theta=th, ou_sigma=sg, momentum=rho, seed=seed, **gen_kw)
            for h, r in oracles(data, a.history, a.horizons).items():
                acc.setdefault(h, []).append(r)
            vr.append(velocity_r2(data, a.history, th))
            fr.append(feature_velocity_r2(data, a.history, th))
        print(f"{rho:>5} {th:>6} {sg:>6} | velocity R^2 from TRUE latents: position-only "
              f"{np.mean([r['position'] for r in vr]):.3f}   history "
              f"{np.mean([r['history'] for r in vr]):.3f}")
        print(f"{rho:>5} {th:>6} {sg:>6} | velocity R^2 from raw FEATURES ({a.readout}, diffusion "
              f"{a.diffusion}, noise {a.node_noise}): x(t) {np.mean([r['position'] for r in fr]):.3f}   "
              f"history {np.mean([r['history'] for r in fr]):.3f}   [position R^2 from x(t): "
              f"{np.mean([r['pos'] for r in fr]):.3f}]  <- what a LINEAR probe on any encoder can reach")
        for h in a.horizons:
            rs = acc[h]
            m = lambda f: float(np.nanmean([f(r) for r in rs]))
            gain = m(lambda r: r["history"][1]) - m(lambda r: r["position"][1])
            print(f"{rho:>5} {th:>6} {sg:>6} | {h:>3} {100 * m(lambda r: r['switch_rate']):>8.1f} "
                  f"{m(lambda r: r['persist']):>8.3f} {m(lambda r: r['position'][0]):>11.3f} "
                  f"{m(lambda r: r['history'][0]):>12.3f} | {m(lambda r: r['position'][1]):>11.3f} "
                  f"{m(lambda r: r['history'][1]):>12.3f} {gain:>+7.3f}")
        print()


if __name__ == "__main__":
    main()
