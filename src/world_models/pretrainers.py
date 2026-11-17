"""Self-supervised pretext heads for DyG-JEPA encoder pretraining.

Encoders + the encoder registry. These wrap a (trainable) encoder with a small decoder/head
that drives a pretext objective; after pretraining only ``encoder.state_dict()``
is kept and the head is discarded.

Three pretexts, selected by ``--pretext`` in ``pretrain_encoder.py``:
  - ``masked_signal``  -> :class:`EncoderPretrainer`     (mask channel-0 values,
        reconstruct from spatial context; used for METR-LA speeds).
  - ``feature_mask``   -> :class:`FeatureMaskPretrainer` (mask the semantic
        feature channels and reconstruct; needs --with_edge_features).
  - ``edge_mask``      -> :class:`EdgePretrainer`        (hide edges, predict
        their existence vs sampled negatives; the canonical DTDG pretext).

``graph_jepa_encoder`` re-exports every public name here for backward
compatibility, so existing ``from ...graph_jepa_encoder import EdgePretrainer``
imports keep working.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class EncoderPretrainer(nn.Module):
    """Encoder + small reconstruction decoder for GraphMAE-style pretraining.

    Random-masks a fraction of `(t, n)` speed values, replaces them with a
    learned mask token, runs the (un-frozen) encoder, and asks a 2-layer MLP
    decoder to reconstruct the masked speeds from the encoder's latents.

    Once trained, you save `encoder.state_dict()` and load it into the frozen
    encoder slot of the C-JEPA training loop. The decoder + mask token are
    discarded -- they were only there to drive the encoder.
    """

    def __init__(self, encoder: nn.Module, latent_dim: int,
                 decoder_hidden: int = 64, speed_channel: int = 0):
        super().__init__()
        self.encoder = encoder
        self.speed_channel = speed_channel
        # Only the speed channel is masked; we replace it with a learned
        # scalar so the encoder has a stable "this is hidden" signal.
        # Time-of-day features stay visible -- in deployment you always know
        # the clock, so blanking it during pretraining just makes the task
        # artificially harder without any matching real-world payoff.
        self.mask_token = nn.Parameter(torch.zeros(1))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, decoder_hidden),
            nn.GELU(),
            nn.Linear(decoder_hidden, 1),
        )

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor | None,
                mask: torch.Tensor) -> torch.Tensor:
        # x: (B, T, N, F)  mask: (B, T, N) bool — True = hidden, train target
        speed = x[..., self.speed_channel]
        masked_speed = torch.where(mask, self.mask_token.expand_as(speed), speed)
        x_in = x.clone()
        x_in[..., self.speed_channel] = masked_speed
        z = self.encoder(x_in, adj_norm)             # (B, T, N, D)
        return self.decoder(z).squeeze(-1)           # (B, T, N) predicted speed_norm


class FeatureMaskPretrainer(nn.Module):
    """Masked-feature reconstruction pretrainer for semantic node features.

    Mechanism: mask a random subset of (t, n) positions over the *semantic*
    feature channels only (channels `[mask_channels_lo, F)`); replace with
    a learned per-channel mask token; encode; decode the masked channels
    from the encoder's latents; minimise L2 (or, optionally, cosine) loss.
    The graph-derived channels `[0, mask_channels_lo)` stay visible so the
    encoder still sees degree info but isn't *only* trained on it.
    """

    def __init__(self, encoder: nn.Module, in_dim: int, latent_dim: int,
                 mask_channels_lo: int, decoder_hidden: int = 64):
        super().__init__()
        self.encoder = encoder
        self.mask_channels_lo = mask_channels_lo
        self.in_dim = in_dim
        self.n_masked = in_dim - mask_channels_lo
        assert self.n_masked > 0, (
            f"FeatureMaskPretrainer requires semantic features "
            f"(in_dim={in_dim} <= mask_channels_lo={mask_channels_lo}). "
            "Did you set --with_edge_features?"
        )
        # Per-channel learned mask token (one scalar per masked channel).
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, self.n_masked))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, decoder_hidden),
            nn.GELU(),
            nn.Linear(decoder_hidden, self.n_masked),
        )

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor,
                mask: torch.Tensor):
        """
        x:        (B, T, N, F)        node features (graph stats + semantic)
        adj_norm: normalised adjacency in any of the dispatched shapes.
        mask:     (B, T, N) bool — True where the SEMANTIC channels should be
                  hidden and supervised. Graph channels are always visible.

        Returns:
          pred_masked: (M, n_masked) decoder output at masked positions.
          target_masked: (M, n_masked) ground-truth values at the same positions.
        """
        x_in = x.clone()
        lo = self.mask_channels_lo
        # Broadcast the per-channel token across (b, t, n) at masked positions.
        token = self.mask_token.expand_as(x[..., lo:])
        x_in[..., lo:] = torch.where(mask.unsqueeze(-1), token, x[..., lo:])
        z = self.encoder(x_in, adj_norm)                          # (B,T,N,D)
        pred = self.decoder(z[mask])                              # (M, n_masked)
        target = x[..., lo:][mask]                                # (M, n_masked)
        return pred, target


def sample_feature_mask(active: torch.Tensor, mask_rate: float = 0.25,
                        generator=None) -> torch.Tensor:
    """Per-(b, t, n) bool mask over ACTIVE positions, at the given rate.

    Returns same shape as `active`; True ⇒ hide the semantic features at
    this position and supervise the decoder there.
    """
    if generator is None:
        rnd = torch.rand_like(active.float())
    else:
        rnd = torch.empty_like(active.float())
        rnd.uniform_(0.0, 1.0, generator=generator)
    return active & (rnd < mask_rate)


class EdgePretrainer(nn.Module):
    """Edge-masking pretraining for dynamic graphs (the right DTDG pretext).

    Hide a random subset of edges from the encoder's input adjacency, then
    ask a pairwise head to predict their existence from the encoder's
    latents. Both positives (hidden real edges) and negatives (sampled
    non-edges) are supervised via BCE so the encoder is forced to learn local
    structural regularities (clustering, preferential attachment, temporal
    carry-over). After training, save `encoder.state_dict()` and discard the
    head.
    """

    def __init__(self, encoder: nn.Module, latent_dim: int,
                 decoder_hidden: int = 64):
        super().__init__()
        self.encoder = encoder
        # Concat + Hadamard pairwise representation; same shape as the
        # downstream LinkPredictionProbe head so latent geometry transfers.
        self.head = nn.Sequential(
            nn.Linear(3 * latent_dim, decoder_hidden),
            nn.GELU(),
            nn.Linear(decoder_hidden, 1),
        )

    def _pair_logits(self, z: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        # idx: (K, 4) int [b, t, i, j]
        b, t, i, j = idx.unbind(-1)
        zi, zj = z[b, t, i], z[b, t, j]
        return self.head(torch.cat([zi, zj, zi * zj], dim=-1)).squeeze(-1)

    def forward(self, x: torch.Tensor, adj_masked_norm: torch.Tensor,
                pos_idx: torch.Tensor, neg_idx: torch.Tensor) -> tuple:
        """
        x:                 (B, T, N, F)        node features
        adj_masked_norm:   (B, T, N, N) float  pre-normalised adjacency with
                                               hidden positives zeroed out
        pos_idx, neg_idx:  (K, 4) int          [b, t, i, j] for each edge
        Returns:
          pos_logits (P,), neg_logits (Q,)
        """
        z = self.encoder(x, adj_masked_norm)
        return self._pair_logits(z, pos_idx), self._pair_logits(z, neg_idx)


def sample_masked_edges(adj: torch.Tensor, active: torch.Tensor,
                        mask_rate: float = 0.15, neg_per_pos: int = 1,
                        generator: Optional[torch.Generator] = None
                        ) -> tuple:
    """Pick a random subset of real edges to hide, plus sampled negatives.

    Args:
      adj:       (B, T, N, N) bool / int adjacency
      active:    (B, T, N)   bool node-existence mask
      mask_rate: fraction of real edges to hide
      neg_per_pos: number of negatives sampled per positive

    Returns:
      adj_masked:  (B, T, N, N) bool with hidden positives zeroed
      pos_idx:     (P, 4) long [b, t, i, j]  -- the hidden edges
      neg_idx:     (P * neg_per_pos, 4) long -- random non-edges at same (b, t)
    """
    device = adj.device
    B, T, N, _ = adj.shape
    # Upper triangle so we don't double-count (i, j) and (j, i) for undirected graphs.
    triu = torch.triu(torch.ones(N, N, dtype=torch.bool, device=device), diagonal=1)
    # nonzero() refuses tensors with more than INT_MAX (2^31) elements, which
    # (B, T, N, N) exceeds at full-N MOOC/Reddit with batch 8. Enumerate per
    # snapshot instead: same row-major (b, t, i, j) order, so the edge list --
    # and hence the seeded randperm below -- is identical, and the (B, T, N, N)
    # boolean temporary is never materialised.
    eb, et, ei, ej = [], [], [], []
    for b in range(B):
        for t in range(T):
            i, j = (adj[b, t].bool() & triu).nonzero(as_tuple=True)
            eb.append(torch.full_like(i, b)); et.append(torch.full_like(i, t))
            ei.append(i); ej.append(j)
    edges_b, edges_t, edges_i, edges_j = (torch.cat(eb), torch.cat(et),
                                          torch.cat(ei), torch.cat(ej))
    n_edges = edges_b.numel()
    if n_edges == 0:
        empty = torch.empty((0, 4), dtype=torch.long, device=device)
        return adj.clone(), empty, empty

    n_mask = max(1, int(mask_rate * n_edges))
    perm = torch.randperm(n_edges, generator=generator, device=device)[:n_mask]
    pos_b, pos_t, pos_i, pos_j = (edges_b[perm], edges_t[perm],
                                  edges_i[perm], edges_j[perm])
    pos_idx = torch.stack([pos_b, pos_t, pos_i, pos_j], dim=-1)

    adj_masked = adj.clone()
    adj_masked[pos_b, pos_t, pos_i, pos_j] = False
    adj_masked[pos_b, pos_t, pos_j, pos_i] = False

    # Negatives: sample (i, j) at the same (b, t) such that the pair is NOT
    # currently an edge and both nodes are active. Cheap rejection sampling.
    N = adj.shape[-1]
    n_neg = n_mask * neg_per_pos
    rep_b = pos_b.repeat_interleave(neg_per_pos)
    rep_t = pos_t.repeat_interleave(neg_per_pos)
    rand_i = torch.randint(0, N, (n_neg,), generator=generator, device=device)
    rand_j = torch.randint(0, N, (n_neg,), generator=generator, device=device)
    
    for _ in range(4):
        bad = (
            (rand_i == rand_j)
            | adj_masked[rep_b, rep_t, rand_i, rand_j]
            | ~active[rep_b, rep_t, rand_i]
            | ~active[rep_b, rep_t, rand_j]
        )
        if not bad.any():
            break
        rand_i = torch.where(
            bad, torch.randint(0, N, (n_neg,), generator=generator, device=device),
            rand_i,
        )
        rand_j = torch.where(
            bad, torch.randint(0, N, (n_neg,), generator=generator, device=device),
            rand_j,
        )
    neg_idx = torch.stack([rep_b, rep_t, rand_i, rand_j], dim=-1)
    return adj_masked, pos_idx, neg_idx
