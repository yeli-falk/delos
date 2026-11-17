"""Task-adjustable linear probes for DELOS (Block 4).

One interface, three downstream tasks. ``build_probe(task, in_dim, ...)``
returns the right readout head; ``fit_probe`` / ``evaluate_probe`` dispatch to
the task-specific train/eval loop:

  - ``link_pred``   pairwise edge readout (concat + Hadamard -> MLP), trained
                    with BCE vs sampled negatives; reports MRR / Hits@k / AUC.
                    (DBLP, JODIE, Reddit.)
  - ``regression``  per-node scalar (latent -> speed); MSE; reports MAE / RMSE /
                    MAPE per horizon. (METR-LA.)
  - ``node_class``  per-node multiclass (latent -> logits); cross-entropy;
                    reports accuracy. (structural / community probes.)
  - ``bilinear``    z_i^T W z_j edge readout (for heterogeneous visit-code).

The pairwise/regression fit+eval bodies are the proven implementations that
used to live in ``train_cjepa_dblp`` / ``train_cjepa_metr_la``; they are moved
here unchanged so the unified trainer and the (legacy) per-dataset trainers
share one source of truth.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Probe heads + ranking metrics still live in delos_encoder (imported by
# many scripts); reuse them here rather than duplicating.
from src.world_models.delos_encoder import (
    LinearProbe,
    LinkPredictionProbe,
    link_ranking_metrics,
    normalize_adjacency,
)


# ---------------------------------------------------------------------------
# Missing-history / partial-observability ablation
# ---------------------------------------------------------------------------
# Test the hypothesis that causal JEPA (trained with node masking) predicts a
# node from GRAPH CONTEXT when the node's own past is unavailable, whereas
# augmentation-based SSL leans on the node's own history. At eval time we hide a
# fraction of nodes' history features (keeping them active in the graph so
# message-passing still reaches them) and score prediction on those nodes.
def sample_shared_masked_nodes(active_hist: torch.Tensor, frac: float,
                               generator: "torch.Generator | None" = None
                               ) -> torch.Tensor:
    """Pick a fraction of nodes active across the whole history to hide.

    active_hist: (B, T_hist, N) bool. Returns a 1-D LongTensor of node indices
    that are active in every (b, t) of the batch (V-JEPA shared mask), so the
    hidden set is well-defined for every example.
    """
    N = active_hist.shape[-1]
    all_active = active_hist.reshape(-1, N).all(dim=0)
    idx = all_active.nonzero(as_tuple=True)[0]
    k = int(round(frac * idx.numel()))
    if k <= 0:
        return torch.empty(0, dtype=torch.long, device=active_hist.device)
    if generator is None:
        perm = torch.randperm(idx.numel(), device=idx.device)
    else:
        perm = torch.randperm(idx.numel(), generator=generator, device=idx.device)
    return idx[perm[:k]]


def sample_masked_per_example(active_ref: torch.Tensor, frac: float,
                              generator: "torch.Generator | None" = None
                              ) -> torch.Tensor:
    """Per-example hidden-node mask — the eval-time analog of node masking.

    active_ref: (B, N) bool, the nodes eligible to hide in each window (e.g.
    active at the last history step, so they have a past to hide). For each
    example independently, hide ``frac`` of its eligible nodes. Independent
    per-example sampling keeps the hidden set large and label-representative
    even on sparse graphs where few nodes are active in *every* window (which
    a batch-shared mask would shrink to almost nothing).

    Returns (B, N) bool: True where that node's history is hidden.
    """
    B, N = active_ref.shape
    out = torch.zeros(B, N, dtype=torch.bool, device=active_ref.device)
    if frac <= 0:
        return out
    for b in range(B):
        idx = active_ref[b].nonzero(as_tuple=True)[0]
        k = int(round(frac * idx.numel()))
        if k <= 0:
            continue
        if generator is None:
            perm = torch.randperm(idx.numel(), device=idx.device)
        else:
            perm = torch.randperm(idx.numel(), generator=generator, device=idx.device)
        out[b, idx[perm[:k]]] = True
    return out


def corrupt_history_bn(x_hist: torch.Tensor, mask_bn: torch.Tensor,
                       fill: "torch.Tensor | float" = 0.0) -> torch.Tensor:
    """Replace masked (B, N) positions' history features across all history
    steps with ``fill``. x_hist: (B, T_hist, N, F); mask_bn: (B, N) bool."""
    if not bool(mask_bn.any()):
        return x_hist
    m = mask_bn.view(mask_bn.shape[0], 1, mask_bn.shape[1], 1)
    if isinstance(fill, torch.Tensor):
        fill_t = fill.to(x_hist.dtype).to(x_hist.device).view(1, 1, 1, -1)
        return torch.where(m, fill_t.expand_as(x_hist), x_hist)
    return torch.where(m, torch.full_like(x_hist, float(fill)), x_hist)


def corrupt_history(x_hist: torch.Tensor, masked_nodes: torch.Tensor,
                    fill: "torch.Tensor | float" = 0.0) -> torch.Tensor:
    """Replace masked nodes' history features with ``fill`` (mean/zero token).

    x_hist: (B, T_hist, N, F). ``fill`` is broadcast over (B, T, F). Returns a
    fresh tensor; the graph (adj/active) is untouched, so the encoder still
    aggregates neighbour messages onto the hidden nodes.
    """
    if masked_nodes.numel() == 0:
        return x_hist
    x = x_hist.clone()
    if isinstance(fill, torch.Tensor):
        x[:, :, masked_nodes, :] = fill.to(x.dtype).to(x.device)
    else:
        x[:, :, masked_nodes, :] = fill
    return x


class BilinearLinkProbe(nn.Module):
    """z_i^T W z_j edge logit. Used for heterogeneous (visit, code) readout."""

    def __init__(self, in_dim: int, hidden_dim: int = 0):
        super().__init__()
        self.W = nn.Bilinear(in_dim, in_dim, 1)

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        return self.W(z_i, z_j).squeeze(-1)


class NodeClassProbe(nn.Module):
    """Per-node multiclass readout (linear or one-hidden-layer MLP)."""

    def __init__(self, in_dim: int, num_classes: int, hidden_dim: int = 0):
        super().__init__()
        if hidden_dim > 0:
            self.head = nn.Sequential(
                nn.Linear(in_dim, hidden_dim), nn.GELU(),
                nn.Linear(hidden_dim, num_classes),
            )
        else:
            self.head = nn.Linear(in_dim, num_classes)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.head(z)


def build_probe(task: str, in_dim: int, out_dim: int = 1, hidden_dim: int = 64):
    """Construct the readout head for ``task``.

    ``out_dim`` is the number of classes for ``node_class`` (ignored otherwise).
    """
    if task == "link_pred":
        return LinkPredictionProbe(in_dim=in_dim, hidden_dim=hidden_dim)
    if task == "regression":
        return LinearProbe(in_dim=in_dim, hidden_dim=hidden_dim)
    if task == "node_class":
        return NodeClassProbe(in_dim=in_dim, num_classes=out_dim,
                              hidden_dim=hidden_dim)
    if task == "bilinear":
        return BilinearLinkProbe(in_dim=in_dim, hidden_dim=hidden_dim)
    raise ValueError(
        f"unknown probe task {task!r}; expected link_pred, regression, "
        f"node_class, or bilinear.")


# ---------------------------------------------------------------------------
# Link prediction (DBLP / JODIE / Reddit) — moved verbatim from train_cjepa_dblp
# ---------------------------------------------------------------------------
def score_negatives_chunked(probe, z: torch.Tensor, src: torch.Tensor,
                            neg_dst: torch.Tensor, max_elems: int = 1 << 26
                            ) -> torch.Tensor:
    """``probe(z[src][:, None], z[neg_dst])`` -> (P, k), scored in chunks of
    positives. The probe concatenates [z_i, z_j, z_i*z_j], i.e. P*k*3D floats
    at once: on a GossipCop snapshot (~150k edges, k=99, D=128) that single
    tensor is 22 GB and OOMs a 32 GB card. ``max_elems`` bounds it (256 MB)."""
    P, k = neg_dst.shape
    D = z.shape[-1]
    step = max(1, max_elems // max(k * 3 * D, 1))
    out = []
    for s in range(0, P, step):
        nz = z[neg_dst[s:s + step]]                                 # (p, k, D)
        out.append(probe(z[src[s:s + step]].unsqueeze(1).expand_as(nz), nz))
    return torch.cat(out, dim=0) if out else z.new_zeros(0, k)


def sample_neg_edges(adj_t: torch.Tensor, active_t: torch.Tensor,
                     k_per_pos: int, seed: int = 0) -> torch.Tensor:
    """For each positive (i, j) edge at time t, sample k_per_pos negatives
    that share node i and target an active-but-not-connected j.

    Returns (P, k) long tensor of negative j-indices for each positive edge.
    Used at eval time to compute MRR.
    """
    g = torch.Generator(device=adj_t.device).manual_seed(seed)
    src, dst = adj_t.nonzero(as_tuple=True)
    P = src.shape[0]
    if P == 0:
        return torch.empty((0, k_per_pos), dtype=torch.long, device=adj_t.device)
    candidates = active_t.clone()
    neg = torch.zeros(P, k_per_pos, dtype=torch.long, device=adj_t.device)
    for p in range(P):
        valid = candidates & ~adj_t[src[p]]
        valid[src[p]] = False
        idx = valid.nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            neg[p] = src[p]                              # fallback
        else:
            choice = torch.randint(0, idx.numel(), (k_per_pos,), generator=g, device=adj_t.device)
            neg[p] = idx[choice]
    return neg


def fit_link_probe(predictor, encoder, train_loader, history_size,
                   num_preds, device, epochs=3, lr=1e-3, hidden_dim=64,
                   neg_per_pos=4):
    z_dim = predictor.node_dim
    probe = LinkPredictionProbe(in_dim=z_dim, hidden_dim=hidden_dim).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr)
    predictor.eval()

    for epoch in range(epochs):
        losses, n = 0.0, 0
        for batch in train_loader:
            x_hist = batch["x_hist"].to(device)
            adj_b = batch["adj"].to(device)
            active_b = batch["active"].to(device)
            B, T_total, N = active_b.shape
            adj_norm = normalize_adjacency(adj_b)
            with torch.no_grad():
                z_hist = encoder(x_hist, adj_norm[:, : history_size])
                pred = predictor.inference(z_hist, adj_b, active_b)   # (B, T_pred, N, D)

            loss = 0.0
            tot_pairs = 0
            for b in range(B):
                for tp in range(pred.shape[1]):
                    t_global = history_size + tp
                    adj_t = adj_b[b, t_global]
                    active_t = active_b[b, t_global]
                    if active_t.sum() < 2 or adj_t.sum() == 0:
                        continue
                    src, dst = adj_t.nonzero(as_tuple=True)
                    if src.numel() == 0:
                        continue
                    z = pred[b, tp]                                    # (N, D)
                    pos_logits = probe(z[src], z[dst])
                    neg_dst = torch.randint(
                        0, N, (src.numel(), neg_per_pos), device=device,
                    )
                    neg_z = z[neg_dst]
                    neg_logits = probe(
                        z[src].unsqueeze(1).expand_as(neg_z),
                        neg_z,
                    )
                    pos_y = torch.ones_like(pos_logits)
                    neg_y = torch.zeros_like(neg_logits)
                    loss = loss + F.binary_cross_entropy_with_logits(
                        torch.cat([pos_logits, neg_logits.reshape(-1)]),
                        torch.cat([pos_y, neg_y.reshape(-1)]),
                    )
                    tot_pairs += pos_logits.numel() + neg_logits.numel()
            if isinstance(loss, float):
                continue
            loss = loss / max(tot_pairs, 1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses += loss.item() * B
            n += B
        print(f"  [probe epoch {epoch+1}] bce={losses/max(n,1):.6f}")
    return probe


@torch.no_grad()
def evaluate_link_prediction(predictor, encoder, probe, loader, history_size,
                             device, k_per_pos=99):
    """MRR / Hits@k for two positive subsets (all edges; new-this-window edges)."""
    predictor.eval()
    probe.eval()
    per_horizon_all: dict = {}
    per_horizon_new: dict = {}
    for batch_idx, batch in enumerate(loader):
        x_hist = batch["x_hist"].to(device)
        adj_b = batch["adj"].to(device)
        active_b = batch["active"].to(device)
        B = x_hist.shape[0]
        adj_norm = normalize_adjacency(adj_b)
        z_hist = encoder(x_hist, adj_norm[:, : history_size])
        pred = predictor.inference(z_hist, adj_b, active_b)         # (B, T_pred, N, D)

        for b in range(B):
            for tp in range(pred.shape[1]):
                t_global = history_size + tp
                adj_t = adj_b[b, t_global]
                active_t = active_b[b, t_global]
                if adj_t.sum() == 0 or active_t.sum() < 2:
                    continue
                src, dst = adj_t.nonzero(as_tuple=True)
                z = pred[b, tp]
                pos_scores = probe(z[src], z[dst])                  # (P,)
                # Same seed formula as analyze_jepa.eval_link_pred and
                # run_baselines.py: every evaluator must rank against the
                # identical negative set, and the batch index must enter the
                # seed or one RNG stream is reused for every window.
                neg_dst = sample_neg_edges(
                    adj_t, active_t, k_per_pos=k_per_pos,
                    seed=batch_idx * 1000 + b * 100 + tp)
                if neg_dst.numel() == 0:
                    continue
                neg_scores = score_negatives_chunked(probe, z, src, neg_dst)  # (P, k)
                m_all = link_ranking_metrics(pos_scores, neg_scores)
                per_horizon_all.setdefault(tp, []).append(m_all)

                prior_seen = adj_b[b, :t_global].any(dim=0)         # (N, N) bool
                is_new = ~prior_seen[src, dst]
                if int(is_new.sum()) == 0:
                    continue
                m_new = link_ranking_metrics(
                    pos_scores[is_new], neg_scores[is_new],
                )
                per_horizon_new.setdefault(tp, []).append(m_new)

    def _aggregate(per_horizon):
        out = {}
        for tp, ms in per_horizon.items():
            keys = ms[0].keys()
            out[tp] = {k: float(np.mean([m[k] for m in ms])) for k in keys}
        return out

    return {"all": _aggregate(per_horizon_all),
            "new": _aggregate(per_horizon_new)}


# ---------------------------------------------------------------------------
# Regression (METR-LA) — moved verbatim from train_cjepa_metr_la
# ---------------------------------------------------------------------------
def fit_regression_probe(predictor, encoder, train_loader, adj_dense, adj_norm,
                         history_size, num_preds, device, epochs=3, lr=1e-3,
                         hidden_dim=0):
    probe = LinearProbe(in_dim=predictor.node_dim, hidden_dim=hidden_dim).to(device)
    n_probe = sum(p.numel() for p in probe.parameters())
    head = "linear" if hidden_dim == 0 else f"MLP h={hidden_dim}"
    print(f"  [probe] {head} ({n_probe:,} params)")
    opt = torch.optim.AdamW(probe.parameters(), lr=lr)
    predictor.eval()

    mean = train_loader.dataset.mean
    std = train_loader.dataset.std
    for epoch in range(epochs):
        losses = 0.0
        n = 0
        for batch in train_loader:
            x_hist = batch["x_hist"].to(device)
            speeds = batch["speeds"].to(device)             # (B, T_total, N) un-normalised
            B = x_hist.shape[0]
            T_total = history_size + num_preds
            adj_bool = adj_dense.bool().unsqueeze(0).unsqueeze(0).expand(
                B, T_total, predictor.num_nodes, predictor.num_nodes,
            )
            active = torch.ones(
                B, T_total, predictor.num_nodes, dtype=torch.bool, device=device,
            )
            with torch.no_grad():
                z_hist = encoder(x_hist, adj_norm)
                pred = predictor.inference(z_hist, adj_bool, active)  # (B, T_pred, N, D)
            future_speeds_norm = (speeds[:, history_size:] - mean) / std
            pred_speed_norm = probe(pred)
            loss = F.mse_loss(pred_speed_norm, future_speeds_norm)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses += loss.item() * B
            n += B
        print(f"  [probe epoch {epoch+1}] mse={losses/max(n,1):.4f}")
    return probe


@torch.no_grad()
def evaluate_regression(predictor, encoder, probe, loader, adj_dense, adj_norm,
                        history_size, num_preds, device, horizons=(2, 5, 11)):
    """MAE/RMSE/MAPE at several future horizons (in steps)."""
    predictor.eval()
    probe.eval()
    mean = loader.dataset.mean
    std = loader.dataset.std

    sums = {h: {"abs": 0.0, "sq": 0.0, "ape": 0.0, "n": 0.0} for h in horizons}
    for batch in loader:
        x_hist = batch["x_hist"].to(device)
        speeds = batch["speeds"].to(device)
        B, T_hist, N, _ = x_hist.shape
        T_total = history_size + num_preds
        z_hist = encoder(x_hist, adj_norm)
        adj_bool = adj_dense.bool().unsqueeze(0).unsqueeze(0).expand(B, T_total, N, N)
        active = torch.ones(B, T_total, N, dtype=torch.bool, device=device)
        pred = predictor.inference(z_hist, adj_bool, active)            # (B, T_pred, N, D)
        pred_speed = probe(pred) * std + mean                            # (B, T_pred, N)

        for h in horizons:
            if h >= num_preds:
                continue
            true = speeds[:, history_size + h]                           # (B, N)
            est = pred_speed[:, h]
            valid = (true > 0).float()
            diff = est - true
            sums[h]["abs"] += (diff.abs() * valid).sum().item()
            sums[h]["sq"] += ((diff ** 2) * valid).sum().item()
            sums[h]["ape"] += (
                (diff.abs() / true.abs().clamp(min=1.0)) * valid
            ).sum().item()
            sums[h]["n"] += valid.sum().item()

    out = {}
    for h, s in sums.items():
        if s["n"] == 0:
            continue
        out[h] = {
            "MAE": s["abs"] / s["n"],
            "RMSE": (s["sq"] / s["n"]) ** 0.5,
            "MAPE": s["ape"] / s["n"],
        }
    return out


# ---------------------------------------------------------------------------
# Regression on dynamic-topology graphs (England COVID)
# ---------------------------------------------------------------------------
# Same protocol as the METR-LA regression probe, but the adjacency changes
# every snapshot, so we read per-batch `adj` / `active` from the loader (like
# the dynamic link-pred path) instead of broadcasting one static graph.
def hist_adj(adj_norm, history_size):
    """The history-window slice of a normalised adjacency.

    A no-op for a fixed-topology graph, which has no time axis to slice.
    """
    return adj_norm[:, :history_size] if adj_norm.dim() == 4 else adj_norm


def graph_inputs(batch, dataset, device, history_size, num_preds):
    """``(adj_bool, adj_norm, active)`` for one batch, whatever the topology.

    Two dataset families feed the regression probes and they disagree on where
    the graph lives:

    * **dynamic topology** (England COVID, synthetic_dyn) -- every window
      carries its own ``adj`` (B, T_total, N, N) and ``active`` (B, T_total, N).
    * **fixed topology** (METR-LA) -- the graph never changes, so the dataset
      holds a single ``adj`` (N, N) and the item carries no ``adj`` at all.
      Materialising the (T_total, N, N) copy per window would dominate collate
      time and GPU memory (~4 MB/window at N=207, T=24) for no information.

    Resolving that difference HERE, once, is what puts both datasets through
    the identical probe, metric and persistence/inflection accounting. The
    alternative -- a second regression probe for fixed graphs -- is what the
    repo had, and it silently reported a different metric schema (no
    ``MAE_persist``, no inflection subset), so METR-LA numbers could not be
    placed in the same table as England COVID ones.

    The fixed branch expands with ``Tensor.expand`` (a stride-0 view, no copy)
    and caches the normalised adjacency on the dataset, so the per-batch cost
    is a view construction rather than an N^2 normalisation.

    The returned ``adj_norm`` spans the FULL window -- (B, T_total, N, N) when
    dynamic, (N, N) when fixed. Slice it with :func:`hist_adj`, which knows not
    to index a static graph on a time axis it does not have.
    """
    if "adj" in batch:
        adj = batch["adj"].to(device)
        adj_norm = normalize_adjacency(adj)        # bool in -> one float copy
        return adj, adj_norm, batch["active"].to(device)

    fixed = getattr(dataset, "adj", None)
    if fixed is None:
        raise ValueError(
            f"{type(dataset).__name__} yields no per-batch 'adj' and exposes no "
            f"dataset-level .adj, so the graph is undefined for this probe.")
    cache = getattr(dataset, "_adj_norm_cache", None)
    if cache is None:
        cache = {}
        dataset._adj_norm_cache = cache
    key = str(device)
    if key not in cache:
        adj2d = torch.as_tensor(fixed).float().to(device)          # (N, N)
        cache[key] = (adj2d.bool(), normalize_adjacency(adj2d))
    adj_bool2d, adj_norm2d = cache[key]

    B, _, N = batch["x_hist"].shape[:3]
    T_total = history_size + num_preds
    adj_bool = adj_bool2d[None, None].expand(B, T_total, N, N)
    if "active" in batch:
        active = batch["active"].to(device)
    else:
        active = torch.ones(B, T_total, N, dtype=torch.bool, device=device)
    # The encoder accepts either a (N, N) static graph or a (B, T, N, N) stack;
    # handing it the 2-D form keeps the fixed-topology path allocation-free.
    return adj_bool, adj_norm2d, active


def _with_base(feats, base, mean, std, enabled):
    """Append the (normalised) persistence base to the probe's input.

    ``feats`` (B, T, N, D) or (B, N, D); ``base`` (B, N). Returns ``feats``
    unchanged when disabled.

    Why this exists: with a NOISY base, ``persist_residual`` alone pins the
    forecast to base + f(latent), which re-injects the base's own noise and
    makes it irreducible — the probe cannot subtract what it cannot see, so it
    converges to outputting ~0 and lands EXACTLY on the persistence MAE no
    matter how long it trains. Feeding it the base lets it learn the shrinkage
    between the noisy last observation and its latent-derived estimate (the
    Kalman gain), which is the whole point of a filtering task. Measured at
    node_noise=0.5 with a matched noisy base, best linear estimator vs
    persistence: anchored -0.1%, base-blind 7.3%, base-visible 30.9%.
    """
    if not enabled:
        return feats
    b = (base - mean) / std                                   # (B, N)
    if feats.dim() == 4:                                      # (B, T, N, D)
        b = b.unsqueeze(1).unsqueeze(-1).expand(*feats.shape[:-1], 1)
    else:                                                     # (B, N, D)
        b = b.unsqueeze(-1)
    return torch.cat([feats, b], dim=-1)


def fit_regression_probe_dynamic(predictor, encoder, train_loader, history_size,
                                 num_preds, device, epochs=3, lr=1e-3,
                                 hidden_dim=0, persist_residual=False,
                                 probe_sees_base=False):
    """Fit the case-count probe on the frozen predictor's rollouts.

    persist_residual: if True, the probe predicts the DELTA over the last
    observed value (signal[history_size-1]) rather than the absolute signal —
    so persistence is the zero-output floor and the probe only has to model
    the change. The regression analog of the latent residual head; without it
    a learned latent→signal probe can (and historically did) land far below
    persistence on this smooth, high-autocorrelation signal.

    The residual base comes from ``signal_obs`` when the dataset provides it
    (the NOISY observation of the history) and the target always comes from
    ``signal`` (the clean value). When the two differ, beating persistence
    requires denoising the base as well as forecasting — which is the point of
    an observation-noise sweep. Datasets without ``signal_obs`` fall back to
    ``signal`` and behave exactly as before.
    """
    probe = LinearProbe(in_dim=predictor.node_dim + int(probe_sees_base),
                        hidden_dim=hidden_dim).to(device)
    n_probe = sum(p.numel() for p in probe.parameters())
    head = "linear" if hidden_dim == 0 else f"MLP h={hidden_dim}"
    print(f"  [probe] {head} ({n_probe:,} params)"
          f"{'  +persist_residual' if persist_residual else ''}"
          f"{'  +base_input' if probe_sees_base else ''}")
    opt = torch.optim.AdamW(probe.parameters(), lr=lr)
    predictor.eval()

    mean = train_loader.dataset.mean
    std = train_loader.dataset.std
    for epoch in range(epochs):
        losses = 0.0
        n = 0
        for batch in train_loader:
            x_hist = batch["x_hist"].to(device)
            signal = batch["signal"].to(device)             # (B, T_total, N) clean
            sig_obs = batch.get("signal_obs", batch["signal"]).to(device)
            B = x_hist.shape[0]
            adj, adj_norm, active = graph_inputs(batch, train_loader.dataset,
                                                 device, history_size, num_preds)
            with torch.no_grad():
                z_hist = encoder(x_hist, hist_adj(adj_norm, history_size))
                pred = predictor.inference(z_hist, adj, active)   # (B, T_pred, N, D)
            base = sig_obs[:, history_size - 1]                     # (B, N)
            if persist_residual:
                target_norm = (signal[:, history_size:] - base.unsqueeze(1)) / std
            else:
                target_norm = (signal[:, history_size:] - mean) / std
            pred_norm = probe(_with_base(pred, base, mean, std, probe_sees_base))
            loss = F.mse_loss(pred_norm, target_norm)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses += loss.item() * B
            n += B
        print(f"  [probe epoch {epoch+1}] mse={losses/max(n,1):.4f}")
    return probe


# --- shared metric accounting (predictor & encoder-only regression evals) ---
def _blank_reg_sums():
    """Per-horizon accumulators: forecast + persistence, full set + inflection."""
    return {k: 0.0 for k in (
        "abs", "sq", "ape", "n",
        "p_abs", "p_sq", "n_full",          # persistence, full set
        "i_abs", "i_sq", "i_n",             # forecast, inflection subset
        "pi_abs", "pi_sq",                  # persistence, inflection subset
    )}


def _accum_reg(s, est, true, last_obs, prev_obs, can_inflect, valid=None):
    """Accumulate one horizon's forecast + persistence errors into ``s``.

    est/true/last_obs/prev_obs: (B, N). ``last_obs`` is the persistence
    prediction (last observed value); the inflection subset is where the
    true signal reverses its recent trend, so persistence is provably wrong.

    ``valid`` (B, N) selects which entries are scored. Callers on a
    SIGNED target MUST pass one (normally the ``active`` mask): the legacy
    default ``true > 0`` was written for England-COVID case counts, and on a
    zero-mean target it silently drops ~60% of the test set and conditions the
    metric on the outcome variable — which can flip the sign of the verdict.
    """
    valid = (true > 0).float() if valid is None else valid.float()
    diff = est - true
    pdiff = last_obs - true
    s["abs"] += (diff.abs() * valid).sum().item()
    s["sq"] += ((diff ** 2) * valid).sum().item()
    s["ape"] += ((diff.abs() / true.abs().clamp(min=1.0)) * valid).sum().item()
    s["n"] += valid.sum().item()
    s["p_abs"] += (pdiff.abs() * valid).sum().item()
    s["p_sq"] += ((pdiff ** 2) * valid).sum().item()
    s["n_full"] += valid.sum().item()
    if can_inflect:
        turn = (torch.sign(true - last_obs)
                * torch.sign(last_obs - prev_obs)) < 0
        infl = valid * turn.float()
        s["i_abs"] += (diff.abs() * infl).sum().item()
        s["i_sq"] += ((diff ** 2) * infl).sum().item()
        s["i_n"] += infl.sum().item()
        s["pi_abs"] += (pdiff.abs() * infl).sum().item()
        s["pi_sq"] += ((pdiff ** 2) * infl).sum().item()


def _finalize_reg(sums, report_mape=True):
    """``report_mape``: percentage error is meaningless on a signed, zero-mean
    target (the clamp(min=1) denominator is arbitrary there), so the signed-
    target callers drop the field rather than publish a number nobody can
    interpret."""
    out = {}
    for h, s in sums.items():
        if s["n"] == 0:
            continue
        entry = {
            "MAE": s["abs"] / s["n"],
            "RMSE": (s["sq"] / s["n"]) ** 0.5,
            "MAE_persist": s["p_abs"] / max(s["n_full"], 1),
            "RMSE_persist": (s["p_sq"] / max(s["n_full"], 1)) ** 0.5,
            "n_scored": int(s["n"]),
        }
        if report_mape:
            entry["MAPE"] = s["ape"] / s["n"]
        if s["i_n"] > 0:
            entry.update({
                "n_inflect": int(s["i_n"]),
                "MAE_inflect": s["i_abs"] / s["i_n"],
                "MAE_persist_inflect": s["pi_abs"] / s["i_n"],
            })
        out[h] = entry
    return out


@torch.no_grad()
def evaluate_regression_dynamic(predictor, encoder, probe, loader, history_size,
                                num_preds, device, horizons=None,
                                persist_residual=False, valid_mode="positive",
                                probe_sees_base=False):
    """MAE/RMSE per future horizon on a dynamic-topology regression set.

    Reports, per horizon, the JEPA forecast alongside the **persistence
    baseline** (predict the last observed value, signal_obs[history_size-1]) —
    the regression analog of the copy-last column. Additionally scores an
    **inflection subset**: (node, step) pairs where the true signal *reverses*
    its recent trend. There persistence is provably wrong, so it's where a real
    spatiotemporal forecaster has to win.

    ``valid_mode``: which entries are scored. ``"positive"`` (default, England
    COVID) keeps ``true > 0``; ``"active"`` keeps every live node and drops
    MAPE — required for signed targets, where ``true > 0`` conditions the
    metric on the outcome. See ``_accum_reg``.
    """
    if horizons is None:
        horizons = tuple(range(num_preds))
    predictor.eval()
    probe.eval()
    mean = loader.dataset.mean
    std = loader.dataset.std
    can_inflect = history_size >= 2
    sums = {h: _blank_reg_sums() for h in horizons}
    for batch in loader:
        x_hist = batch["x_hist"].to(device)
        signal = batch["signal"].to(device)
        sig_obs = batch.get("signal_obs", batch["signal"]).to(device)
        adj, adj_norm, active = graph_inputs(batch, loader.dataset, device,
                                             history_size, num_preds)
        z_hist = encoder(x_hist, hist_adj(adj_norm, history_size))
        pred = predictor.inference(z_hist, adj, active)             # (B, T_pred, N, D)
        last_obs = sig_obs[:, history_size - 1]                     # (B, N) persistence
        out = probe(_with_base(pred, last_obs, mean, std, probe_sees_base))
        if persist_residual:
            pred_signal = last_obs.unsqueeze(1) + out * std          # (B, T_pred, N)
        else:
            pred_signal = out * std + mean
        prev_obs = sig_obs[:, history_size - 2] if can_inflect else None
        for h in horizons:
            if h >= num_preds:
                continue
            valid = (active[:, history_size + h] if valid_mode == "active" else None)
            _accum_reg(sums[h], pred_signal[:, h], signal[:, history_size + h],
                       last_obs, prev_obs, can_inflect, valid=valid)
    return _finalize_reg(sums, report_mape=(valid_mode == "positive"))


class RegressionForecastHead(nn.Module):
    """Per-node history embedding -> per-horizon scalar forecast.

    For encoder-only baselines (CLDG / DDGCL / IDOL) that have no forecasting
    predictor: one embedding of the observed history, num_preds horizon-
    specific readouts. Gives each baseline a fair learned forecast from its
    frozen representation, evaluated with the same persistence/inflection
    protocol as the DELOS predictor rollout.
    """

    def __init__(self, in_dim: int, num_preds: int, hidden_dim: int = 0):
        super().__init__()
        if hidden_dim > 0:
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim), nn.GELU(),
                nn.Linear(hidden_dim, num_preds),
            )
        else:
            self.net = nn.Linear(in_dim, num_preds)

    def forward(self, z):                       # z: (B, N, D) -> (B, T_pred, N)
        return self.net(z).permute(0, 2, 1)


def fit_regression_probe_encoder_only(encoder, train_loader, history_size,
                                      num_preds, device, epochs=3, lr=1e-3,
                                      hidden_dim=0, persist_residual=False,
                                      probe_sees_base=False,
                                      train_encoder=False, encoder_lr=None,
                                      val_loader=None, early_stop_patience=0,
                                      early_stop_min_delta=0.0):
    """Fit a forecast head on the encoder's last-history embedding.

    The encoder-only analog of fit_regression_probe_dynamic: no predictor
    rollout, so the head reads z_hist[:, -1] and emits all horizons. Same
    persist_residual option (predict the delta over the last observed value),
    and the same ``signal_obs``-as-base / ``signal``-as-target split as
    fit_regression_probe_dynamic.

    ``train_encoder`` switches this between the two arms it has to serve:

    * ``False`` (default) -- the encoder is frozen and only the head is fit.
      This is the *frozen* arm and the probe used for CLDG / DDGCL / IDOL: it
      measures what the self-supervised representation already contains.
    * ``True`` -- encoder and head are optimised jointly on the forecast loss.
      This is the *supervised end-to-end* arm, the reference every SSL method
      is trying to reach without labels. ``encoder_lr`` defaults to ``lr``.

    Both arms share this one loop on purpose: an end-to-end baseline written
    separately would drift on normalisation, the persistence base or the valid
    mask, and then its gap to the SSL arms would not be attributable to
    pretraining. ``val_loader`` enables best-val-loss weight restoration and
    early stopping, matching what every other trainer in the repo does -- an
    end-to-end arm that kept its last epoch while the JEPA arms restored their
    best-val weights would be handicapped for a reason unrelated to method.
    """
    encoder.train(train_encoder)
    for p_ in encoder.parameters():
        p_.requires_grad_(train_encoder)
    mean = train_loader.dataset.mean
    std = train_loader.dataset.std
    # Infer the encoder's output dim from one forward.
    head = None
    opt = None
    tracker = None

    def _forward(batch):
        x_hist = batch["x_hist"].to(device)
        signal = batch["signal"].to(device)
        sig_obs = batch.get("signal_obs", batch["signal"]).to(device)
        _, adj_norm, _ = graph_inputs(batch, train_loader.dataset, device,
                                      history_size, num_preds)
        with torch.set_grad_enabled(train_encoder):
            z_hist = encoder(x_hist, hist_adj(adj_norm, history_size))
        base = sig_obs[:, history_size - 1]                  # (B, N)
        z_last = _with_base(z_hist[:, -1], base, mean, std,   # (B, N, D[+1])
                            probe_sees_base)
        if persist_residual:
            target = (signal[:, history_size:] - base.unsqueeze(1)) / std
        else:
            target = (signal[:, history_size:] - mean) / std
        return z_last, target, x_hist.shape[0]

    for epoch in range(epochs):
        encoder.train(train_encoder)
        losses, n = 0.0, 0
        for batch in train_loader:
            z_last, target, B = _forward(batch)
            if head is None:
                head = RegressionForecastHead(
                    z_last.shape[-1], num_preds, hidden_dim=hidden_dim).to(device)
                n_h = sum(p.numel() for p in head.parameters())
                mode = "end-to-end" if train_encoder else "encoder-only"
                print(f"  [probe] {mode} forecast head "
                      f"({n_h:,} params)"
                      f"{'  +persist_residual' if persist_residual else ''}"
                      f"{'  +base_input' if probe_sees_base else ''}")
                groups = [{"params": head.parameters(), "lr": lr}]
                if train_encoder:
                    groups.append({"params": encoder.parameters(),
                                   "lr": encoder_lr if encoder_lr else lr})
                opt = torch.optim.AdamW(groups, lr=lr)
                if val_loader is not None:
                    from src.baselines.common import BestValTracker
                    tracker = BestValTracker(early_stop_patience,
                                             early_stop_min_delta)
            out = head(z_last)                                 # (B, T_pred, N)
            loss = F.mse_loss(out, target)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses += loss.item() * B
            n += B
        msg = f"  [probe epoch {epoch+1}] mse={losses/max(n,1):.4f}"

        if tracker is not None:
            encoder.eval(); head.eval()
            vl, vn = 0.0, 0
            with torch.no_grad():
                for batch in val_loader:
                    z_last, target, B = _forward(batch)
                    vl += F.mse_loss(head(z_last), target).item() * B
                    vn += B
            val_loss = vl / max(vn, 1)
            # Select on the pair actually used at eval time. Tracking only the
            # head would restore a head that was never validated against the
            # restored encoder in the end-to-end arm.
            bundle = nn.ModuleDict({"encoder": encoder, "head": head})
            improved, stop = tracker.update(val_loss, bundle)
            msg += f"  val_mse={val_loss:.4f}{'  (best)' if improved else ''}"
            print(msg, flush=True)
            if stop:
                print(f"  early stopping: no val improvement for "
                      f"{early_stop_patience} epochs", flush=True)
                break
        else:
            print(msg, flush=True)

    if tracker is not None and tracker.restore(
            nn.ModuleDict({"encoder": encoder, "head": head})):
        print(f"  [best] restored encoder+head from best val epoch "
              f"(val={tracker.best:.4f})", flush=True)
    return head


@torch.no_grad()
def evaluate_regression_encoder_only(encoder, head, loader, history_size,
                                     num_preds, device, horizons=None,
                                     persist_residual=False,
                                     valid_mode="positive",
                                     probe_sees_base=False):
    """MAE/RMSE + persistence + inflection for a frozen encoder + head.

    Mirrors evaluate_regression_dynamic exactly (same accounting helpers,
    same ``valid_mode`` and ``signal_obs`` handling) but the forecast comes
    from head(z_hist[:, -1]) instead of a predictor rollout, so baseline
    encoders sit in the same table as DELOS.
    """
    if horizons is None:
        horizons = tuple(range(num_preds))
    encoder.eval()
    head.eval()
    mean = loader.dataset.mean
    std = loader.dataset.std
    can_inflect = history_size >= 2
    sums = {h: _blank_reg_sums() for h in horizons}
    for batch in loader:
        x_hist = batch["x_hist"].to(device)
        signal = batch["signal"].to(device)
        sig_obs = batch.get("signal_obs", batch["signal"]).to(device)
        _, adj_norm, active = graph_inputs(batch, loader.dataset, device,
                                           history_size, num_preds)
        z_hist = encoder(x_hist, hist_adj(adj_norm, history_size))
        last_obs = sig_obs[:, history_size - 1]
        out = head(_with_base(z_hist[:, -1], last_obs, mean, std,
                              probe_sees_base))                # (B, T_pred, N)
        if persist_residual:
            pred_signal = last_obs.unsqueeze(1) + out * std
        else:
            pred_signal = out * std + mean
        prev_obs = sig_obs[:, history_size - 2] if can_inflect else None
        for h in horizons:
            if h >= num_preds:
                continue
            valid = (active[:, history_size + h] if valid_mode == "active" else None)
            _accum_reg(sums[h], pred_signal[:, h], signal[:, history_size + h],
                       last_obs, prev_obs, can_inflect, valid=valid)
    return _finalize_reg(sums, report_mape=(valid_mode == "positive"))
