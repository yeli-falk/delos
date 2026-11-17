"""Shared DyG-JEPA training/eval loops.

:func:`train_jepa_epoch` / :func:`evaluate_jepa` handle EVERY data source —
dynamic-graph (DBLP / JODIE / Reddit / EHRshot / England COVID) and
fixed-topology (METR-LA) alike. They resolve the graph through
``probes.graph_inputs``, which reads per-snapshot adjacency when the batch
carries it and falls back to the dataset's single static graph when it does not.

:func:`train_jepa_epoch_fixed` / :func:`evaluate_jepa_fixed` are the older
METR-LA-only loops. They are kept for the legacy per-dataset trainers, but they
support NONE of the encoder-training variants (no EMA target, no VICReg, no
``--train_encoder``, no change-weighting), so METR-LA runs through them could
only ever produce the frozen-encoder arm. New code should use the unified pair
above; ``train_dyg_jepa.py`` does.
"""
from __future__ import annotations

import copy
import math
import time

import numpy as np
import torch

from src.cjepa_graph_predictor import compute_graph_jepa_loss
from src.world_models.graph_jepa_encoder import normalize_adjacency
from src.world_models.probes import graph_inputs, hist_adj


class EMAEncoder:
    """Exponential moving-average copy of the online encoder.

    Provides a stable target for the JEPA predictor that decouples
    target-encoder gradients from the predictor's optimisation. Without this
    asymmetry the trivial fixed point (encoder + predictor both collapse to
    one direction) minimises the loss, which is exactly what we saw on DBLP
    (eff_rank 1.02). See I-JEPA / V-JEPA / BYOL for the canonical setup.

    Momentum schedule:  theta_target <- m * theta_target + (1 - m) * theta_online

    ``momentum_end`` + ``total_steps`` turn on the I-JEPA / V-JEPA cosine
    anneal of m from ``momentum`` to ``momentum_end`` (canonically 1.0) over
    ``total_steps`` updates; the target is noisy early, when the online
    encoder moves fast, and frozen late, when the predictor should converge on
    a fixed target. ``total_steps`` must be a horizon the run actually
    reaches: under early stopping the whole-run horizon is rarely reached and
    the momentum never leaves its start value (train_dyg_jepa's
    ``--ema_anneal_epochs``). Leaving both unset keeps the legacy constant
    momentum.
    """

    def __init__(self, online_encoder: torch.nn.Module, momentum: float = 0.996,
                 momentum_end: "float | None" = None,
                 total_steps: "int | None" = None):
        self.target = copy.deepcopy(online_encoder)
        self.target.eval()
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.momentum = momentum
        self.momentum_end = momentum_end
        self.total_steps = total_steps
        self.step = 0
        # ``paused`` = the online encoder is frozen (--encoder_freeze_epochs):
        # the target is identical to it, so an update would be a no-op, and
        # the anneal clock must not run while nothing moves.
        self.paused = False

    def current_momentum(self) -> float:
        if self.momentum_end is None or not self.total_steps:
            return self.momentum
        frac = min(self.step / self.total_steps, 1.0)
        return self.momentum_end - (self.momentum_end - self.momentum) \
            * 0.5 * (1.0 + math.cos(math.pi * frac))

    @torch.no_grad()
    def update(self, online_encoder: torch.nn.Module) -> None:
        if self.paused:
            return
        m = self.current_momentum()
        self.step += 1
        for p_target, p_online in zip(
            self.target.parameters(), online_encoder.parameters(),
        ):
            p_target.data.mul_(m).add_(p_online.data, alpha=1.0 - m)
        # Update buffers too (e.g. running mean/var for any BN/GN inside).
        for b_target, b_online in zip(
            self.target.buffers(), online_encoder.buffers(),
        ):
            b_target.data.copy_(b_online.data)

    @torch.no_grad()
    def __call__(self, *args, **kwargs):
        return self.target(*args, **kwargs)

    def state_dict(self):
        return self.target.state_dict()

    def load_state_dict(self, sd):
        self.target.load_state_dict(sd)


def compute_louvain_communities(ds, seed: int = 0):
    """Static node->community assignment on the time-aggregated adjacency.

    Used by community-aware structured masking: instead of picking
    `num_masked_nodes` random nodes, sample a whole community per step.
    Forces the predictor to reconstruct a structural unit from outside it,
    breaking the degree shortcut that random masking allows.

    Returns: (N,) int64 array; communities[i] is the Louvain community of
    node i, or -1 if the node never appears or sits alone in a singleton.
    """
    import networkx as nx
    N = ds.num_nodes
    agg = ds.adj.any(axis=0)
    active_any = ds.active.any(axis=0)
    triu = np.triu(agg, k=1)
    src, dst = triu.nonzero()
    G = nx.Graph()
    G.add_nodes_from(range(N))
    G.add_edges_from(zip(src.tolist(), dst.tolist()))
    communities = nx.algorithms.community.louvain_communities(G, seed=seed)
    labels = np.full(N, -1, dtype=np.int64)
    for c_id, members in enumerate(communities):
        if len(members) >= 2:                      # drop singletons
            for v in members:
                labels[v] = c_id
    labels[~active_any] = -1
    n_nontrivial = labels.max() + 1 if (labels >= 0).any() else 0
    sizes = np.bincount(labels[labels >= 0]) if n_nontrivial else np.array([])
    print(f"[community] louvain: {n_nontrivial} communities  "
          f"sizes min/median/max = {sizes.min() if sizes.size else 0}/"
          f"{int(np.median(sizes)) if sizes.size else 0}/"
          f"{sizes.max() if sizes.size else 0}  "
          f"coverage = {(labels >= 0).mean():.2f}", flush=True)
    return labels


def sample_community_mask(community_labels: np.ndarray,
                          active_at_0: torch.Tensor,
                          max_mask_size: int,
                          generator: "np.random.Generator | None" = None):
    """Pick one community at random; return the indices of its members
    that are active across the whole batch. Shared across the batch
    (V-JEPA style) and resampled every forward pass.

    Falls back to None if no usable community exists at this step (e.g.
    too small after filtering inactive nodes); caller should then let the
    predictor sample randomly.
    """
    device = active_at_0.device
    # Nodes active in every example of the batch (V-JEPA shared mask).
    all_active = active_at_0.all(dim=0).cpu().numpy()
    labelled = community_labels >= 0
    valid_node = all_active & labelled
    if not valid_node.any():
        return None
    valid_labels = community_labels[valid_node]
    avail = np.unique(valid_labels)
    if avail.size == 0:
        return None
    if generator is None:
        generator = np.random.default_rng()
    c = int(generator.choice(avail))
    members = np.where((community_labels == c) & valid_node)[0]
    if members.size > max_mask_size:
        members = generator.choice(members, size=max_mask_size, replace=False)
    return torch.from_numpy(members.astype(np.int64)).to(device)


def encode_window(encoder, x, adj_norm, train_encoder: bool = False):
    """Encode (B, T, N, F) -> (B, T, N, D). If `train_encoder` is True the
    encoder's parameters receive gradients; otherwise we wrap in no_grad so
    the autograd graph stops at the encoder (the historical default)."""
    if train_encoder:
        return encoder(x, adj_norm)
    with torch.no_grad():
        return encoder(x, adj_norm)


def mask_encoder_context(x_hist, adj_hist, masked_idx):
    """Hide the masked nodes from the ONLINE (context) encoder.

    Without this, the encoder runs on the full history and, through message
    passing, writes every masked node's features into its neighbours'
    latents before the predictor ever sees a mask token -- the
    masked-history term is then largely readable from context. I-JEPA /
    V-JEPA give the context encoder only the visible tokens; this is the
    graph analogue. Mirrors ``DyGJEPAPredictor.prepare_input``: t = 0 stays
    visible (it is the anchor), t >= 1 is hidden -- features zeroed and all
    incident edges removed, so the node is an isolated self-loop the encoder
    cannot read or propagate. The TARGET encoder still sees everything.

    Args:
      x_hist:   (B, T_hist, N, F)
      adj_hist: (B, T_hist, N, N) bool -- history slice of the raw adjacency.
      masked_idx: 1-D LongTensor of node indices.
    Returns (x_masked, adj_norm_masked) with adj_norm_masked (B, T_hist, N, N).
    """
    x = x_hist.clone()
    adj = adj_hist.clone()
    if masked_idx.numel() > 0 and x.shape[1] > 1:
        x[:, 1:, masked_idx] = 0.0
        adj[:, 1:, masked_idx, :] = False
        adj[:, 1:, :, masked_idx] = False
    return x, normalize_adjacency(adj)


def _sample_hidden_edges_for_jepa(adj, active, mask_rate, neg_per_pos):
    """Sample a bool `hidden_edges` mask + supervised labels for the
    predictor's edge head during JEPA training.

    Returns:
      hidden_edges: (B, T, N, N) bool -- positives at hidden real edges plus
                    `neg_per_pos` sampled non-edges per positive.
      labels:       1-D float tensor of 0/1 labels aligned with the
                    (b,t,i,j) ordering of hidden_edges.nonzero().
    """
    device = adj.device
    B, T, N, _ = adj.shape
    triu = torch.triu(torch.ones(N, N, device=device, dtype=torch.bool), diagonal=1)
    pos_mask = adj & triu.view(1, 1, N, N)
    # Per-(b,t) Bernoulli on real edges, drawn ONLY over the existing edges: a
    # dense rand over (B, T, N, N) is two ~3 GB float tensors at full-N MOOC.
    eb, et, ei, ej = pos_mask.nonzero(as_tuple=True)
    keep = torch.rand(eb.numel(), device=device) < mask_rate
    pos_hidden = torch.zeros_like(pos_mask)
    pos_hidden[eb[keep], et[keep], ei[keep], ej[keep]] = True
    # Symmetrise so the edge head sees both (i,j) and (j,i) for undirected.
    hidden = pos_hidden | pos_hidden.transpose(-1, -2)

    # Sampled negatives: for each positive, sample neg_per_pos non-edges at
    # the same (b, t). Simple uniform sampling; rejection sampling for the
    # "is a real edge" / "is the diagonal" cases.
    pb, pt, pi, pj = pos_hidden.nonzero(as_tuple=True)
    P = pb.numel()
    if P > 0 and neg_per_pos > 0:
        Q = P * neg_per_pos
        rep_b = pb.repeat_interleave(neg_per_pos)
        rep_t = pt.repeat_interleave(neg_per_pos)
        rand_i = torch.randint(0, N, (Q,), device=device)
        rand_j = torch.randint(0, N, (Q,), device=device)
        for _ in range(4):
            bad = (
                (rand_i == rand_j)
                | adj[rep_b, rep_t, rand_i, rand_j]
                | ~active[rep_b, rep_t, rand_i]
                | ~active[rep_b, rep_t, rand_j]
            )
            if not bad.any():
                break
            rand_i = torch.where(bad,
                                 torch.randint(0, N, (Q,), device=device), rand_i)
            rand_j = torch.where(bad,
                                 torch.randint(0, N, (Q,), device=device), rand_j)
        # Mark negatives in `hidden` and remember their positions for labels.
        hidden[rep_b, rep_t, rand_i, rand_j] = True
        hidden[rep_b, rep_t, rand_j, rand_i] = True

    # Labels: 1 for true edges in `hidden`, 0 elsewhere. Read in the same
    # (b,t,i,j) order as `hidden.nonzero()` so they align with edge_out.
    nz_b, nz_t, nz_i, nz_j = hidden.nonzero(as_tuple=True)
    labels = adj[nz_b, nz_t, nz_i, nz_j].to(torch.float32)
    return hidden, labels


def train_jepa_epoch(predictor, encoder, loader, optimizer, history_size,
                     device, log_every=50, scheduler=None, max_steps=None,
                     edge_mask_rate: float = 0.0, edge_weight: float = 1.0,
                     edge_neg_per_pos: int = 1,
                     ema_target: "EMAEncoder | None" = None,
                     train_encoder: bool = False,
                     vicreg_var_weight: float = 0.0,
                     vicreg_cov_weight: float = 0.0,
                     vicreg_gamma: float = 1.0,
                     change_weight: float = 0.0,
                     horizon_norm: bool = False,
                     community_labels: "np.ndarray | None" = None,
                     community_max_mask_size: int = 30,
                     community_rng: "np.random.Generator | None" = None,
                     decoder: "torch.nn.Module | None" = None,
                     mask_encoder_input: bool = False,
                     target_norm: bool = False,
                     target_grad: bool = False,
                     sigreg_weight: float = 0.0,
                     sigreg_slices: int = 256,
                     sigreg_knots: int = 17):
    """One JEPA training epoch over per-batch dynamic adjacency.

    JEPA variants (all default off, so legacy runs are bit-for-bit
    reproducible -- when off, none of them touches the RNG stream):

    * ``mask_encoder_input`` -- sample the node mask BEFORE encoding and hide
      those nodes from the online encoder (:func:`mask_encoder_context`).
    * ``target_norm`` -- LayerNorm the target latents (I-JEPA / V-JEPA); the
      predictor must have been built with ``latent_norm=True``.
    * ``target_grad`` -- LeJEPA: no stop-gradient on the online-encoder target
      (needs ``train_encoder``, no EMA, no decoder).
    * ``sigreg_weight`` -- LeJEPA's SIGReg on predictor outputs and, when the
      encoder trains, its latents. Replaces VICReg's var/cov pair.

    ``decoder`` switches the prediction space. Default (None) is JEPA proper:
    the predictor's output is matched against the ENCODER's latents. Passing a
    decoder (node_dim -> feature_dim) instead matches it against the RAW
    OBSERVATIONS ``x`` — i.e. ordinary observation-space forecasting, with the
    same encoder, predictor and masking. That is the control the JEPA claim
    needs: latent-space vs input-space prediction, everything else held fixed.
    """
    predictor.train()
    if train_encoder:
        encoder.train()
    log_keys = ("loss", "loss_masked_history", "loss_future", "loss_edge",
                "loss_vicreg_var", "loss_vicreg_cov", "loss_sigreg")
    log = {k: 0.0 for k in log_keys}
    log["n"] = 0
    t0 = time.time()
    use_edge_loss = edge_mask_rate > 0.0
    use_vicreg = vicreg_var_weight > 0 or vicreg_cov_weight > 0
    use_sigreg = sigreg_weight > 0
    if target_grad and (not train_encoder or ema_target is not None
                        or decoder is not None):
        raise ValueError("target_grad needs train_encoder=True and no "
                         "EMA target / observation decoder")
    for step, batch in enumerate(loader):
        if max_steps is not None and step >= max_steps:
            break
        x_hist = batch["x_hist"].to(device, non_blocking=True)
        x_fut = batch["x_fut"].to(device, non_blocking=True)
        # Dynamic datasets carry per-snapshot adjacency; fixed-topology ones
        # (METR-LA) resolve to the dataset's single graph. See
        # probes.graph_inputs — routing both through it is what lets METR-LA use
        # this loop, and hence honour --train_encoder / --ema_target / VICReg.
        adj, adj_norm, active = graph_inputs(
            batch, loader.dataset, device, history_size, x_fut.shape[1])

        forced_idx = None
        if community_labels is not None:
            forced_idx = sample_community_mask(
                community_labels, active[:, 0],
                max_mask_size=community_max_mask_size,
                generator=community_rng,
            )
        if mask_encoder_input:
            # Draw the mask now (same sampler / RNG draw the predictor would
            # make) so the context encoder can be blinded to it.
            if forced_idx is None:
                forced_idx = predictor._mask_indices(active[:, 0], device)
            x_ctx, adj_norm_ctx = mask_encoder_context(
                x_hist, adj[:, :history_size], forced_idx)
        else:
            x_ctx, adj_norm_ctx = x_hist, hist_adj(adj_norm, history_size)
        z_hist = encode_window(
            encoder, x_ctx, adj_norm_ctx, train_encoder=train_encoder,
        )
        # Target: EMA copy if available, otherwise the online encoder under
        # no_grad. The historical default (target == online) is recoverable by
        # passing ema_target=None and train_encoder=False.
        x_full = torch.cat([x_hist, x_fut], dim=1)
        if decoder is not None:
            # Observation-space target: the raw features themselves. No encoder
            # pass, so nothing about the target can drift or collapse.
            z_target = x_full.detach()
        elif ema_target is not None:
            z_target = ema_target(x_full, adj_norm).detach()
        elif target_grad:
            # LeJEPA: gradients flow into the target too; SIGReg (not a
            # stop-grad) is what keeps this from collapsing.
            z_target = encode_window(encoder, x_full, adj_norm,
                                     train_encoder=True)
        else:
            z_target = encode_window(encoder, x_full, adj_norm,
                                     train_encoder=False).detach()

        # Optional edge-masking term: hide a fraction of edges, ask the
        # predictor's edge head to identify them against random non-edges.
        # This gives the predictor a direct structural-prediction signal
        # alongside the latent-MSE JEPA loss.
        hidden_edges = edge_labels = None
        if use_edge_loss:
            hidden_edges, edge_labels = _sample_hidden_edges_for_jepa(
                adj, active, edge_mask_rate, edge_neg_per_pos,
            )

        cond = None
        if getattr(predictor, "hint", None) is not None:
            # Two-level run: the frozen coarse model must hide the same nodes
            # as the fine predictor, so the mask is drawn here -- at the point
            # of the RNG stream where the predictor would draw it itself.
            if forced_idx is None:
                forced_idx = predictor._mask_indices(active[:, 0], device)
            cond = predictor.make_cond(x_hist, adj, active, x_fut=x_fut,
                                       masked_idx=forced_idx)
        pred, masked_idx, edge_out = predictor(
            z_hist, adj, active, hidden_edges=hidden_edges,
            forced_masked_idx=forced_idx, **({"cond": cond} if cond is not None else {}),
        )
        if decoder is not None:
            pred = decoder(pred)          # (B,T,N,D) -> (B,T,N,F)
        sigreg_rows = None
        if use_sigreg and train_encoder:
            # Regularise what the encoder emits, not just what the predictor
            # writes: LeJEPA applies SIGReg to the embeddings themselves.
            sigreg_rows = [z_hist[active[:, :history_size]]]
            if target_grad:
                sigreg_rows.append(z_target[active])
        losses = compute_graph_jepa_loss(
            pred, z_target, active, masked_idx, history_size,
            edge_out=edge_out if use_edge_loss else None,
            edge_labels=edge_labels if use_edge_loss else None,
            edge_weight=edge_weight,
            vicreg_var_weight=vicreg_var_weight,
            vicreg_cov_weight=vicreg_cov_weight,
            vicreg_gamma=vicreg_gamma,
            change_weight=change_weight,
            horizon_norm=horizon_norm,
            target_norm=target_norm and decoder is None,
            sigreg_weight=sigreg_weight,
            sigreg_slices=sigreg_slices,
            sigreg_knots=sigreg_knots,
            sigreg_extra_rows=sigreg_rows,
        )

        loss = losses["loss"]
        if not torch.isfinite(loss):
            print(f"  [step {step+1}] non-finite loss, skipping")
            optimizer.zero_grad(set_to_none=True)
            continue
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        clip_params = list(predictor.parameters())
        if train_encoder:
            clip_params += list(encoder.parameters())
        gnorm = torch.nn.utils.clip_grad_norm_(clip_params, 1.0)
        if not torch.isfinite(gnorm):
            optimizer.zero_grad(set_to_none=True)
            continue
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        if ema_target is not None and train_encoder:
            ema_target.update(encoder)

        bs = x_hist.shape[0]
        for k in ("loss", "loss_masked_history", "loss_future"):
            log[k] += losses[k].item() * bs
        if "loss_edge" in losses:
            log["loss_edge"] += losses["loss_edge"].item() * bs
        if "loss_vicreg_var" in losses:
            log["loss_vicreg_var"] += losses["loss_vicreg_var"].item() * bs
            log["loss_vicreg_cov"] += losses["loss_vicreg_cov"].item() * bs
        if "loss_sigreg" in losses:
            log["loss_sigreg"] += losses["loss_sigreg"].item() * bs
        log["n"] += bs
        if (step + 1) % log_every == 0:
            n = log["n"]
            cur_lr = optimizer.param_groups[0]["lr"]
            edge_str = f" edge={log['loss_edge']/n:.4f}" if use_edge_loss else ""
            vic_str = (f" var={log['loss_vicreg_var']/n:.4f}"
                       f" cov={log['loss_vicreg_cov']/n:.4f}") if use_vicreg else ""
            sig_str = f" sig={log['loss_sigreg']/n:.4f}" if use_sigreg else ""
            print(f"  [step {step+1:>5}] loss={log['loss']/n:.4f} "
                  f"mh={log['loss_masked_history']/n:.4f} "
                  f"fut={log['loss_future']/n:.4f}{edge_str}{vic_str}{sig_str} "
                  f"lr={cur_lr:.2e} ({time.time()-t0:.1f}s)")
    n = max(log["n"], 1)
    return {k: v / n for k, v in log.items() if k != "n"}


@torch.no_grad()
def evaluate_jepa(predictor, encoder, loader, history_size, device,
                  decoder: "torch.nn.Module | None" = None,
                  ema_target: "EMAEncoder | None" = None,
                  change_weight: float = 0.0,
                  mask_encoder_input: bool = False,
                  target_norm: bool = False,
                  horizon_norm: bool = False):
    """JEPA validation loss over per-batch dynamic adjacency.

    ``mask_encoder_input`` / ``target_norm`` must match training so the val
    loss measures the same task (the predictor's deterministic eval mask is
    drawn explicitly here so the context encoder can be blinded to it).
    """
    predictor.eval()
    encoder.eval()
    total = {"loss": 0.0, "loss_masked_history": 0.0, "loss_future": 0.0, "n": 0}
    for batch in loader:
        x_hist = batch["x_hist"].to(device)
        x_fut = batch["x_fut"].to(device)
        adj, adj_norm, active = graph_inputs(
            batch, loader.dataset, device, history_size, x_fut.shape[1])
        forced_idx = None
        if mask_encoder_input:
            forced_idx = predictor._mask_indices(active[:, 0], device)
            x_ctx, adj_norm_ctx = mask_encoder_context(
                x_hist, adj[:, :history_size], forced_idx)
        else:
            x_ctx, adj_norm_ctx = x_hist, hist_adj(adj_norm, history_size)
        cond = None
        if getattr(predictor, "hint", None) is not None:
            if forced_idx is None:
                forced_idx = predictor._mask_indices(active[:, 0], device)
            cond = predictor.make_cond(x_hist, adj, active, x_fut=x_fut,
                                       masked_idx=forced_idx)
        z_hist = encoder(x_ctx, adj_norm_ctx)
        x_full = torch.cat([x_hist, x_fut], dim=1)
        if decoder is not None:
            z_target = x_full
        elif ema_target is not None:
            z_target = ema_target(x_full, adj_norm)
        else:
            z_target = encoder(x_full, adj_norm)
        pred, masked_idx, _ = predictor(z_hist, adj, active,
                                        forced_masked_idx=forced_idx,
                                        **({"cond": cond} if cond is not None else {}))
        if decoder is not None:
            pred = decoder(pred)
        losses = compute_graph_jepa_loss(pred, z_target, active, masked_idx,
                                         history_size,
                                         change_weight=change_weight,
                                         horizon_norm=horizon_norm,
                                         target_norm=target_norm and decoder is None)
        bs = x_hist.shape[0]
        for k in ("loss", "loss_masked_history", "loss_future"):
            total[k] += losses[k].item() * bs
        total["n"] += bs
    n = max(total["n"], 1)
    return {k: v / n for k, v in total.items() if k != "n"}


# ---------------------------------------------------------------------------
# Fixed-topology variant (METR-LA): one static adjacency broadcast per batch.
# ---------------------------------------------------------------------------
def train_jepa_epoch_fixed(predictor, encoder, loader, optimizer, adj_dense,
                           adj_norm, history_size, device, log_every=50,
                           max_steps=None, scheduler=None):
    """One JEPA training epoch over a fixed (broadcast) adjacency."""
    predictor.train()
    log = {"loss": 0.0, "loss_masked_history": 0.0, "loss_future": 0.0, "n": 0}
    t0 = time.time()

    for step, batch in enumerate(loader):
        if max_steps is not None and step >= max_steps:
            break
        x_hist = batch["x_hist"].to(device, non_blocking=True)
        x_fut = batch["x_fut"].to(device, non_blocking=True)
        full = torch.cat([x_hist, x_fut], dim=1)        # (B, T_total, N, F)

        z_hist = encode_window(encoder, x_hist, adj_norm)            # (B, T_hist, N, D)
        z_target = encode_window(encoder, full, adj_norm)            # (B, T_total, N, D)

        B, T_total, N, _ = z_target.shape
        adj_bool = adj_dense.bool().unsqueeze(0).unsqueeze(0).expand(B, T_total, N, N)
        active = torch.ones(B, T_total, N, dtype=torch.bool, device=device)

        pred, masked_idx, _ = predictor(z_hist, adj_bool, active)
        losses = compute_graph_jepa_loss(
            pred, z_target.detach(), active, masked_idx, history_size,
        )

        loss_value = losses["loss"]
        if not torch.isfinite(loss_value):
            print(f"  [step {step+1}] non-finite loss ({loss_value.item()}), skipping batch")
            optimizer.zero_grad(set_to_none=True)
            continue

        optimizer.zero_grad(set_to_none=True)
        loss_value.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
        if not torch.isfinite(gnorm):
            print(f"  [step {step+1}] non-finite grad norm, skipping step")
            optimizer.zero_grad(set_to_none=True)
            continue
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        bs = x_hist.shape[0]
        for k in ("loss", "loss_masked_history", "loss_future"):
            log[k] += losses[k].item() * bs
        log["n"] += bs

        if (step + 1) % log_every == 0:
            n = log["n"]
            cur_lr = optimizer.param_groups[0]["lr"]
            print(
                f"  [step {step+1:>5}] loss={log['loss']/n:.4f} "
                f"mh={log['loss_masked_history']/n:.4f} "
                f"fut={log['loss_future']/n:.4f} "
                f"lr={cur_lr:.2e} "
                f"({(time.time()-t0):.1f}s)"
            )

    n = max(log["n"], 1)
    return {k: v / n for k, v in log.items() if k != "n"}


@torch.no_grad()
def evaluate_jepa_fixed(predictor, encoder, loader, adj_dense, adj_norm,
                        history_size, device):
    """JEPA validation loss over a fixed (broadcast) adjacency."""
    predictor.eval()
    total = {"loss": 0.0, "loss_masked_history": 0.0, "loss_future": 0.0, "n": 0}
    for batch in loader:
        x_hist = batch["x_hist"].to(device)
        x_fut = batch["x_fut"].to(device)
        full = torch.cat([x_hist, x_fut], dim=1)
        z_hist = encoder(x_hist, adj_norm)
        z_target = encoder(full, adj_norm)
        B, T_total, N, _ = z_target.shape
        adj_bool = adj_dense.bool().unsqueeze(0).unsqueeze(0).expand(B, T_total, N, N)
        active = torch.ones(B, T_total, N, dtype=torch.bool, device=device)

        pred, masked_idx, _ = predictor(z_hist, adj_bool, active)
        losses = compute_graph_jepa_loss(pred, z_target, active, masked_idx, history_size)
        bs = x_hist.shape[0]
        for k in ("loss", "loss_masked_history", "loss_future"):
            total[k] += losses[k].item() * bs
        total["n"] += bs
    n = max(total["n"], 1)
    return {k: v / n for k, v in total.items() if k != "n"}
