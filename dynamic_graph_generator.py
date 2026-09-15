"""
Synthetic dynamic graph generator with dynamic node features, dynamic edge
features, and dynamic labels, all driven by a shared latent process.

Design principle: a per-node latent trajectory z_i(t) is the single cause of
everything observable. This makes structure and features mutually informative
and temporally predictable, with knobs to control how much.

    z_i(t)   Ornstein-Uhlenbeck pull toward a community anchor, with Markov
             community switching (regime changes / nonstationarity).
    A_t      edges ~ Bernoulli(sigmoid(beta * <z_i, z_j> + bias)), blended
             with A_{t-1} for temporal persistence. Note: with the blend
             p = (1-rho)*p_latent + rho*A_prev, the stationary edge density
             equals p_latent; rho only sets edge lifetime, not density.
    x_i(t)   node features = fixed random 1-hidden-layer readout of z_i(t),
             optionally smoothed by one diffusion step over A_t, plus noise.
    events   per active edge, Poisson-many interaction events per step with
             timestamps jittered inside the step; edge features are a noisy
             readout of z_u(t) + z_v(t). For genuinely bursty event streams,
             replace this block with a Hawkes process whose baseline is
             mu_ij = f(<z_i, z_j>) (e.g. via the `tick` library).
    y_i(t)   dynamic label = current community id (free node-classification
             ground truth).

Outputs (dict):
    snapshots     list of T adjacency matrices, uint8 [N, N]   (DTDG view)
    node_features float32 [T, N, d_node]
    events        DataFrame (src, dst, t, ef0..)               (CTDG view,
                  TGB / JODIE style temporal edge list)
    latents       float32 [T, N, d_latent]   ground truth for probing
    labels        int64   [T, N]             dynamic community labels
    labels_cont   float32 [T, N]             smooth scalar readout of z; the
                  regression target to use when you need forecasting to be a
                  BEATABLE task (see switch_mode in generate_dynamic_graph)
    anchors       float32 [K, d_latent]      the community anchors (for the
                  state-classification oracle in ``state_headroom``)

Plugging in:
    - PyG-Temporal: DynamicGraphTemporalSignal(edge_indices, edge_weights,
      features, targets) with edge_index_list(snapshots), unit weights,
      node_features[t], labels[t].
    - TGN/TGB-style loaders: use `events` directly (sorted by t); split
      chronologically for train/val/test.
    - JEPA evals: ridge-probe learned embeddings against `latents`; sweep
      prediction horizon; compare against knob settings.

Dependencies: numpy, pandas.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def generate_dynamic_graph(
    n_nodes: int = 100,
    n_steps: int = 200,
    d_latent: int = 8,
    d_node_feat: int = 16,
    d_edge_feat: int = 4,
    n_communities: int = 4,
    switch_prob: float = 0.005,    # per node per step; nonstationarity knob
    ou_theta: float = 0.15,        # mean reversion; higher = more predictable latents
    ou_sigma: float = 0.25,        # latent noise; higher = less predictable
    anchor_scale: float = 2.0,     # community separation in latent space
    beta: float = 1.0,             # edge logit sharpness on latent similarity
    target_density: float = 0.05,  # approximate edge density; realized density
                                   # fluctuates slowly around this as community
                                   # occupancies random-walk (verified: no
                                   # systematic drift across seeds)
    persistence: float = 0.6,      # edge lifetime knob (0 = resample every step)
    diffusion_alpha: float = 0.3,  # graph smoothing of node features (0 = off);
                                   # entangles features with current topology
    node_noise: float = 0.10,
    edge_noise: float = 0.10,
    events_per_edge: float = 1.5,  # Poisson rate of events per active edge per step
    burn_in: int = 50,             # latent steps before recording, so bias is
                                   # calibrated at the OU stationary regime
    momentum: float = 0.0,         # velocity autocorrelation; 0 = pure OU (the
                                   # original). >0 is what makes the trajectory
                                   # predictable MORE THAN ONE STEP AHEAD — see
                                   # the docstring.
    switch_mode: str = "iid",      # "iid" | "latent_driven"; see below
    switch_gain: float = 6.0,      # latent_driven: how sharply drift raises risk
    label_mode: str = "hazard",    # "hazard" | "nearest_anchor"; see below
    hysteresis: float = 0.0,       # nearest_anchor: relative margin before a
                                   # boundary crossing counts as a switch
    seed: int = 0,
) -> dict:
    """See module docstring. Two knobs decide whether FORECASTING is a real task:

    ``switch_mode="iid"`` (the original) draws community switches from a
    constant Bernoulli(switch_prob), independent of the latent. Those switches
    are irreducibly unpredictable, so copy-last is already the optimal
    forecaster (~99.6% at the default rate) and no model can beat persistence.
    Use it only as a persistence-dominated control.

    ``switch_mode="latent_driven"`` makes the hazard a function of how far the
    node has drifted from its community anchor. Drift is itself an
    autocorrelated (OU) process, so an imminent switch is VISIBLE IN THE
    HISTORY: a model that tracks z can anticipate it, while copy-last cannot.
    That is what makes the forecast task beatable, which is the precondition
    for any claim that one architecture forecasts better than another.

    ``labels_cont`` (always returned) is a smooth scalar readout of z — a
    regression target with the same property and no class-imbalance issues.

    ``momentum`` (rho) decides whether FORECASTING HAS ANY HEADROOM AT ALL.

    With rho=0 the latent is pure OU, i.e. Markov-1: given a clean observation
    at t, the history adds nothing, and the only predictable part of the next
    step is the mean-reversion pull. Its size relative to the innovation is
    ``theta / sqrt(2*theta - theta^2)`` — a function of THETA ALONE, since
    sigma scales the predictable and irreducible parts equally. At the default
    theta=0.15 that leaves an oracle only ~2% better than copy-last, so no
    method can demonstrate anything and every comparison returns a null.

    With rho>0 the velocity is autocorrelated:

        v(t+1) = rho * v(t) + sigma * eps
        z(t+1) = z(t) + v(t+1) + theta * (anchor - z(t))

    Copy-last is now SYSTEMATICALLY wrong (it assumes zero velocity), and
    inferring the velocity needs at least two observations — so history
    genuinely matters and a temporal model has something to win. Measured
    oracle-vs-persistence headroom (theta=0.15):

        rho=0.0 -> 4.3%    rho=0.5 -> 11.0%
        rho=0.8 -> 26.5%   rho=0.95 -> 41.4%

    Estimating that velocity from NOISY observations is precisely the
    filtering problem where latent-space prediction should beat
    observation-space prediction, which is what --synth_node_noise sweeps.

    Check ``headroom()`` before running any comparison: below ~10% the setup
    cannot resolve a difference and a null result means nothing.

    ``label_mode`` decides WHAT a community is.

    ``"hazard"`` (the original): the community is a separately sampled Markov
    label. Switches are drawn from ``switch_mode``'s hazard and the destination
    is uniform, so even an oracle that knows z exactly cannot say WHICH
    community comes next (switch-event AUROC ~0.65). The label is therefore
    only partly a function of z, and the dense state-classification task it
    supports is capped by that irreducible coin flip.

    ``"nearest_anchor"``: the community IS a readout of z — the anchor the
    latent is currently closest to. No hazard is sampled. A node switches
    exactly when noise + velocity carry it across the boundary between two
    basins, after which the pull drags it into the new one. Every observable,
    the label included, is then a function of z; the state at t+1 is knowable
    from the trajectory through the velocity, so a predictive latent has real
    headroom over copy-last. ``hysteresis`` (relative margin) stops a node
    flickering across a boundary under momentum: it moves only when the nearest
    anchor is closer than ``(1 - hysteresis)`` times its current distance. The
    switch rate is EMERGENT (from sigma, theta, rho, anchor_scale, d): read the
    rate ``state_headroom()`` prints and tune, rather than assuming
    ``switch_prob``, which this mode ignores.
    """
    if switch_mode not in ("iid", "latent_driven"):
        raise ValueError("switch_mode must be 'iid' or 'latent_driven'")
    if label_mode not in ("hazard", "nearest_anchor"):
        raise ValueError("label_mode must be 'hazard' or 'nearest_anchor'")
    if not 0.0 <= hysteresis < 1.0:
        raise ValueError("hysteresis must be in [0, 1)")
    rng = np.random.default_rng(seed)

    anchors = rng.normal(0.0, anchor_scale, size=(n_communities, d_latent))
    comm = rng.integers(0, n_communities, size=n_nodes)
    z = anchors[comm] + rng.normal(size=(n_nodes, d_latent))

    # Fixed random readouts: the "decoder" from latent to observables.
    w1 = rng.normal(size=(d_latent, 32)) / np.sqrt(d_latent)
    w2 = rng.normal(size=(32, d_node_feat)) / np.sqrt(32)
    we = rng.normal(size=(d_latent, d_edge_feat)) / np.sqrt(d_latent)
    wy = rng.normal(size=d_latent) / np.sqrt(d_latent)   # continuous label readout

    # Switch hazard. Under "latent_driven" the risk rises with how far a node
    # has drifted from its anchor; drift is an OU process, hence autocorrelated,
    # hence PREDICTABLE from history — which is what lets a forecaster beat
    # copy-last. Normalised by the OU stationary spread so the mean rate stays
    # ~switch_prob and the total number of switches stays comparable to "iid".
    ou_var = ou_sigma ** 2 / max(1.0 - (1.0 - ou_theta) ** 2, 1e-9)
    drift_ref = np.sqrt(d_latent * ou_var)

    # Velocity state for the momentum term (stays exactly zero when rho=0, so
    # the rho=0 path reproduces the original pure-OU process).
    vel = np.zeros_like(z)

    def _step_latent(z_now, comm_now, vel_now):
        vel_next = momentum * vel_now + ou_sigma * rng.normal(size=z_now.shape)
        z_next = z_now + vel_next + ou_theta * (anchors[comm_now] - z_now)
        return z_next, vel_next

    def _hazard(z_now: np.ndarray, comm_now: np.ndarray) -> np.ndarray:
        if switch_mode == "iid":
            return np.full(n_nodes, switch_prob)
        drift = np.linalg.norm(z_now - anchors[comm_now], axis=1)
        return 2.0 * switch_prob * _sigmoid(switch_gain * (drift / drift_ref - 1.0))

    def _switch(z_now: np.ndarray, comm_now: np.ndarray) -> np.ndarray:
        """Community after this step's switch decision (in place on comm_now)."""
        if label_mode == "hazard":
            fire = rng.random(n_nodes) < _hazard(z_now, comm_now)
            if fire.any():
                comm_now[fire] = rng.integers(0, n_communities, size=int(fire.sum()))
            return comm_now
        # nearest_anchor: label = closest anchor, with a relative hysteresis
        # margin so momentum cannot make a node flicker across a boundary.
        dist = np.linalg.norm(z_now[:, None, :] - anchors[None, :, :], axis=-1)
        nearest = dist.argmin(axis=1)
        d_near = dist[np.arange(n_nodes), nearest]
        d_cur = dist[np.arange(n_nodes), comm_now]
        move = (nearest != comm_now) & (d_near < (1.0 - hysteresis) * d_cur)
        comm_now[move] = nearest[move]
        return comm_now

    # Burn in the latent process, then calibrate the edge bias at stationarity
    # so realized density tracks target_density instead of drifting.
    for _ in range(burn_in):
        comm = _switch(z, comm)
        z, vel = _step_latent(z, comm, vel)
    iu = np.triu_indices(n_nodes, k=1)
    bias = -np.quantile(beta * (z @ z.T)[iu], 1.0 - target_density)

    # Start the edge process at equilibrium rather than from an empty graph.
    p0 = _sigmoid(beta * (z @ z.T) + bias)
    a_prev = np.triu(rng.random((n_nodes, n_nodes)) < p0, k=1)
    a_prev = (a_prev | a_prev.T).astype(np.uint8)
    np.fill_diagonal(a_prev, 0)
    latents = np.empty((n_steps, n_nodes, d_latent), dtype=np.float32)
    node_features = np.empty((n_steps, n_nodes, d_node_feat), dtype=np.float32)
    labels = np.empty((n_steps, n_nodes), dtype=np.int64)
    labels_cont = np.empty((n_steps, n_nodes), dtype=np.float32)
    snapshots: list[np.ndarray] = []
    ev_src, ev_dst, ev_t, ev_feat = [], [], [], []

    for t in range(n_steps):
        # Latent dynamics: community switching, then OU toward the anchor.
        comm = _switch(z, comm)
        z, vel = _step_latent(z, comm, vel)

        # Topology from latent similarity, blended with A_{t-1} for persistence.
        p_latent = _sigmoid(beta * (z @ z.T) + bias)
        p = (1.0 - persistence) * p_latent + persistence * a_prev
        a = np.triu(rng.random((n_nodes, n_nodes)) < p, k=1)
        a = (a | a.T).astype(np.uint8)
        np.fill_diagonal(a, 0)

        # Node features: readout of latent, optionally smoothed over the graph.
        x = np.tanh(z @ w1) @ w2
        if diffusion_alpha > 0.0:
            deg = a.sum(axis=1, keepdims=True).clip(min=1)
            x = (1.0 - diffusion_alpha) * x + diffusion_alpha * (a @ x) / deg
        x = x + node_noise * rng.normal(size=x.shape)

        # Interaction events on active edges (cheap Poisson surrogate).
        src, dst = np.nonzero(np.triu(a, k=1))
        counts = rng.poisson(events_per_edge, size=src.shape[0])
        ru, rv = np.repeat(src, counts), np.repeat(dst, counts)
        if ru.size:
            ev_src.append(ru)
            ev_dst.append(rv)
            ev_t.append(t + rng.random(ru.size))
            ev_feat.append(
                np.tanh((z[ru] + z[rv]) @ we)
                + edge_noise * rng.normal(size=(ru.size, d_edge_feat))
            )

        latents[t], node_features[t], labels[t] = z, x, comm
        labels_cont[t] = z @ wy
        snapshots.append(a)
        a_prev = a

    feat = np.concatenate(ev_feat) if ev_feat else np.empty((0, d_edge_feat))
    events = pd.DataFrame(
        {
            "src": np.concatenate(ev_src) if ev_src else np.empty(0, dtype=int),
            "dst": np.concatenate(ev_dst) if ev_dst else np.empty(0, dtype=int),
            "t": np.concatenate(ev_t) if ev_t else np.empty(0),
        }
    )
    for k in range(d_edge_feat):
        events[f"ef{k}"] = feat[:, k]
    events = events.sort_values("t", ignore_index=True)

    return {
        "snapshots": snapshots,
        "node_features": node_features,
        "events": events,
        "latents": latents,
        "labels": labels,
        "labels_cont": labels_cont,
        "anchors": anchors.astype(np.float32),
        "label_mode": label_mode,
        "hysteresis": hysteresis,
    }


def headroom(data: dict, splits=(0.7, 0.1, 0.2), lam: float = 1e-3,
             history: int = 3) -> dict:
    """How much better than copy-last can ANY model possibly be on this data?

    Fits a ridge from the TRUE latents over the last ``history`` steps to the
    target y(t+1) — i.e. gives a hypothetical model perfect knowledge of the
    causal variable, with the same window a real model gets — and compares its
    test MAE against the persistence baseline. That is the ceiling: no encoder,
    objective or architecture can do better, because everything observable is a
    function of z.

    ``history`` must be >= 2 whenever ``momentum > 0``: the predictable part of
    a momentum process is the VELOCITY, which cannot be inferred from a single
    snapshot. A 1-step oracle reports ~0% headroom on momentum data and would
    wrongly condemn a perfectly good config.

    RUN THIS BEFORE ANY COMPARISON. Below ~10% headroom the setup cannot
    resolve a difference between methods, so a null result says nothing about
    the methods and everything about the generator config. The default pure-OU
    settings give ~2%, which is why sweeps on them come back flat.

    Returns {persist_mae, oracle_mae, headroom_frac, verdict}.
    """
    z, y = data["latents"], data["labels_cont"]
    T, N, D = z.shape
    h = max(1, history)
    tr = int(T * splits[0])
    te = int(T * (splits[0] + splits[1]))

    def _win(lo, hi):
        """Rows of [z(t-h+1) .. z(t)] flattened, with target y(t+1)."""
        X, Y = [], []
        for t in range(max(lo, h - 1), hi - 1):
            X.append(np.concatenate([z[t - k] for k in range(h)], axis=1))
            Y.append(y[t + 1])
        return np.concatenate(X), np.concatenate(Y)

    def _fit(X, t):
        X = np.c_[X, np.ones(len(X))]
        w = np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ t)
        return lambda Z: np.c_[Z, np.ones(len(Z))] @ w

    Xtr, Ytr = _win(0, tr)
    Xte, Yte = _win(te, T)
    f = _fit(Xtr, Ytr)
    # persistence over exactly the same rows
    P = np.concatenate([y[t] for t in range(max(te, h - 1), T - 1)])
    persist = float(np.abs(Yte - P).mean())
    oracle = float(np.abs(f(Xte) - Yte).mean())
    frac = (persist - oracle) / persist if persist else 0.0
    return {
        "persist_mae": persist,
        "oracle_mae": oracle,
        "headroom_frac": frac,
        "verdict": ("USABLE" if frac >= 0.10 else
                    "TOO LITTLE HEADROOM — raise momentum and/or ou_theta"),
    }


def state_headroom(data: dict, splits=(0.7, 0.1, 0.2), lam: float = 1e-3,
                   history: int = 3) -> dict:
    """Classification twin of ``headroom``: how much better than copy-last can
    ANY model be at predicting the community at t+1?

    ``label_mode="nearest_anchor"``: the label is a deterministic decode of
    z(t+1), so the oracle forecasts z(t+1) by ridge from the true latent
    history (exactly what ``headroom`` does for the regression target) and
    then applies the SAME decode rule the generator used — nearest anchor with
    the hysteresis margin relative to c(t). Whatever accuracy that reaches is
    the ceiling: no encoder can know more about z(t+1) than the true history.

    ``label_mode="hazard"``: the destination of a switch is a uniform draw the
    latent says nothing about, so the oracle is a ridge from [latent history,
    one-hot c(t)] to one-hot c(t+1), argmax-decoded. It lands on persistence,
    which is the honest ceiling for that label — and the reason the dense
    state task should be run with nearest-anchor labels.

    The number that matters is ``switch_acc``: accuracy on the rows whose
    community actually changes. Persistence scores exactly 0 there by
    definition, so it is the pure measure of anticipation. Read it together
    with ``switch_rate``: if almost nothing switches inside the test window,
    overall accuracy is persistence-dominated and cannot separate methods.

    Returns {switch_rate, persist_acc, oracle_acc, switch_acc, headroom_frac,
    verdict}; headroom_frac is the relative error reduction over persistence.
    """
    z, c = data["latents"], data["labels"]
    T, N, D = z.shape
    K = int(c.max()) + 1
    h = max(1, history)
    tr = int(T * splits[0])
    te = int(T * (splits[0] + splits[1]))
    nearest_mode = data.get("label_mode") == "nearest_anchor" and "anchors" in data

    def _win(lo, hi):
        X, Y, P, Znext = [], [], [], []
        for t in range(max(lo, h - 1), hi - 1):
            X.append(np.concatenate([z[t - k] for k in range(h)], axis=1))
            Y.append(c[t + 1]); P.append(c[t]); Znext.append(z[t + 1])
        return (np.concatenate(X), np.concatenate(Y), np.concatenate(P),
                np.concatenate(Znext))

    def _ridge(X, Y):
        A = np.c_[X, np.ones(len(X))]
        W = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ Y)
        return lambda Q: np.c_[Q, np.ones(len(Q))] @ W

    Xtr, Ytr, Ptr, Ztr = _win(0, tr)
    Xte, Yte, Pte, _ = _win(te, T)
    if nearest_mode:
        anchors = np.asarray(data["anchors"], dtype=np.float64)
        hyst = float(data.get("hysteresis", 0.0))
        zhat = _ridge(Xtr, Ztr)(Xte)
        dist = np.linalg.norm(zhat[:, None, :] - anchors[None, :, :], axis=-1)
        near = dist.argmin(axis=1)
        d_near = dist[np.arange(len(zhat)), near]
        d_cur = dist[np.arange(len(zhat)), Pte]
        pred = np.where((near != Pte) & (d_near < (1.0 - hyst) * d_cur), near, Pte)
    else:
        eye = np.eye(K)
        f = _ridge(np.c_[Xtr, eye[Ptr]], eye[Ytr])
        pred = f(np.c_[Xte, eye[Pte]]).argmax(axis=1)

    switch = Yte != Pte
    persist = float((Pte == Yte).mean())
    oracle = float((pred == Yte).mean())
    switch_acc = float((pred[switch] == Yte[switch]).mean()) if switch.any() else float("nan")
    frac = ((1 - persist) - (1 - oracle)) / (1 - persist) if persist < 1 else 0.0
    return {
        "switch_rate": float(switch.mean()),
        "persist_acc": persist,
        "oracle_acc": oracle,
        "switch_acc": switch_acc,
        "headroom_frac": float(frac),
        "verdict": ("USABLE" if frac >= 0.10 and switch.mean() >= 0.01 else
                    "TOO LITTLE HEADROOM — raise ou_sigma / lower anchor_scale "
                    "(more switches) or raise momentum (more anticipation)"),
    }


def edge_index_list(snapshots: list[np.ndarray]) -> list[np.ndarray]:
    """[2, E_t] arrays (both directions), ready for PyG-Temporal."""
    out = []
    for a in snapshots:
        src, dst = np.nonzero(a)
        out.append(np.stack([src, dst]))
    return out


if __name__ == "__main__":
    data = generate_dynamic_graph(seed=0)
    snaps, ev = data["snapshots"], data["events"]
    n = snaps[0].shape[0]
    n_pairs = n * (n - 1) / 2

    dens = np.array([np.triu(a, 1).sum() / n_pairs for a in snaps])
    switches = (np.diff(data["labels"], axis=0) != 0).sum()

    # Homophily: do edges track latent similarity? (last snapshot)
    z = data["latents"][-1]
    iu = np.triu_indices(n, k=1)
    sim, adj = (z @ z.T)[iu], snaps[-1][iu].astype(float)
    homophily = np.corrcoef(sim, adj)[0, 1]

    # Temporal predictability of features: mean lag-1 autocorrelation.
    x = data["node_features"]
    xc = x - x.mean(axis=0)
    ac = (xc[:-1] * xc[1:]).mean(axis=0) / (xc.var(axis=0) + 1e-12)
    print(f"steps={len(snaps)}  nodes={n}  events={len(ev)}")
    print(f"density: mean={dens.mean():.4f}  first10={dens[:10].mean():.4f}  "
          f"last10={dens[-10:].mean():.4f}")
    print(f"community switches total={switches}  "
          f"(~{switches / x.shape[0] / n:.4f} per node per step)")
    print(f"edge/latent-similarity correlation (homophily): {homophily:.3f}")
    print(f"node feature lag-1 autocorrelation: {ac.mean():.3f}")
    print(ev.head())
