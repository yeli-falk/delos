"""Trajectory-window tokenisation for V-JEPA-style dynamic-graph SSL.

Replaces the per-(timestep, node) tokens used by the current DELOS
predictor with per-(window, node) tokens, where a window is W contiguous
timesteps. This yields K = T // W tokens per node along the time axis
(instead of T) -- each token summarises a node's evolution over W
timesteps rather than its state at one instant.

Why this matters
----------------
Per-(timestep, node) tokens admit a degree shortcut: predicting a single
timestep's adjacency is solvable by degree-ordering, and our multi-task
experiments show the JEPA predictor in that setting consistently mirrors
the encoder rather than enriching it. Per-window tokens force the model
to encode HOW a node's neighbourhood evolves (rate, direction, pattern
of new connections) -- not a 1D quantity, and not recoverable from degree
alone.

Two pieces here:
  - `TrajectoryWindowPooler`:    (B, T, N, D) -> (B, K, N, D)
  - `sample_trajectory_mask_blocks`:  V-JEPA-style multi-block mask in the
                                       (K, N) grid.

A third piece (the trajectory-aware predictor) will live in
src/delos_predictor.py as a variant of MaskedNodePredictor, in step 2.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


class TrajectoryWindowPooler(nn.Module):
    """Pool per-timestep latents into per-(window, node) tokens.

    Args:
      window_size: W, number of contiguous timesteps per window.
      pool:        'mean' (default, parameter-free)
                   'gru'  (small per-node GRU; richer but adds latent_dim^2 params)
      latent_dim:  D, dimension of input latents. Only used for 'gru' mode.

    Forward:
      x: (B, T, N, D) -- typically the encoder's output per timestep.

    Returns:
      (B, K, N, D)  where  K = T // window_size.
      Any tail timesteps not divisible by window_size are dropped.

    Notes:
      We pool over the encoder's output (which has already seen graph
      structure via message passing), not over raw features. So a "window
      token" carries an aggregate of structurally-aware per-timestep
      latents rather than a featurewise summary of inputs.
    """

    def __init__(self, window_size: int, pool: str = "mean",
                 latent_dim: int = 128):
        super().__init__()
        if pool not in ("mean", "gru"):
            raise ValueError(f"pool must be 'mean' or 'gru'; got {pool!r}")
        if window_size < 1:
            raise ValueError(f"window_size must be >= 1; got {window_size}")
        self.window_size = window_size
        self.pool = pool
        self.latent_dim = latent_dim
        if pool == "gru":
            # Per-(B, n) GRU over the W timesteps of one window. Output is
            # the final hidden state of each window, which acts as the
            # token representation.
            self.gru = nn.GRU(
                input_size=latent_dim, hidden_size=latent_dim,
                num_layers=1, batch_first=True,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"expected (B, T, N, D), got shape {tuple(x.shape)}")
        B, T, N, D = x.shape
        W = self.window_size
        K = T // W
        if K == 0:
            raise ValueError(
                f"window_size={W} exceeds T={T}; can't form any windows."
            )
        # Drop any timesteps not fitting cleanly into K full windows.
        x = x[:, : K * W]
        # Reshape to (B, K, W, N, D) so we can reduce dim=2 (the W axis).
        x = x.reshape(B, K, W, N, D)
        if self.pool == "mean":
            return x.mean(dim=2)                                # (B, K, N, D)

        # GRU path: collapse (B, K, N) into a batch dim, run W steps per chunk.
        # x: (B, K, W, N, D) -> (B, K, N, W, D) -> ((B*K*N), W, D)
        x = x.permute(0, 1, 3, 2, 4).contiguous().reshape(B * K * N, W, D)
        _, h = self.gru(x)                                       # h: (1, B*K*N, D)
        return h.squeeze(0).view(B, K, N, D)


def sample_trajectory_mask_blocks(
    K: int,
    N: int,
    n_blocks: int = 4,
    block_k_range: Tuple[int, int] = (1, 3),
    block_n_frac: Tuple[float, float] = (0.10, 0.30),
    generator: Optional[torch.Generator] = None,
    device: "torch.device | str" = "cpu",
) -> torch.Tensor:
    """V-JEPA-style multi-block mask in the (window, node) grid.

    Samples `n_blocks` random rectangles; for each:
      - k_extent:   random integer in [block_k_range[0], block_k_range[1]]
                    (number of contiguous windows to mask).
      - n_count:    random integer ~ Uniform(block_n_frac * N) -- how many
                    nodes the block touches.
      - k_start:    random uniform in [0, K - k_extent].
      - nodes:      random subset of N (not contiguous: graph nodes have
                    no inherent ordering, unlike image patches).

    The R blocks' union forms the final mask. Block overlap is allowed
    (this is V-JEPA's setup; gives a useful asymmetric mask distribution).

    Returns:
      (K, N) bool tensor; True = masked (the predictor must reconstruct
      these tokens from the unmasked ones).
    """
    # The generator's device is the source of truth: every torch.randint /
    # rand / randperm call must use it, otherwise we hit:
    #   RuntimeError: Expected a 'cpu' device type for generator but found 'cuda'
    if generator is None:
        generator = torch.Generator(device=device).manual_seed(0)
    g_device = generator.device
    mask = torch.zeros(K, N, dtype=torch.bool, device=g_device)

    k_lo, k_hi = block_k_range
    n_lo, n_hi = block_n_frac
    if k_lo < 1 or k_hi < k_lo:
        raise ValueError(f"bad block_k_range {block_k_range}")
    if not (0.0 <= n_lo <= n_hi <= 1.0):
        raise ValueError(f"bad block_n_frac {block_n_frac}")

    for _ in range(n_blocks):
        k_extent = int(torch.randint(
            k_lo, k_hi + 1, (1,), generator=generator, device=g_device,
        ).item())
        k_extent = min(k_extent, K)
        k_start = int(torch.randint(
            0, K - k_extent + 1, (1,), generator=generator, device=g_device,
        ).item())
        frac = n_lo + (n_hi - n_lo) * float(torch.rand(
            1, generator=generator, device=g_device,
        ).item())
        n_count = max(1, min(N, int(N * frac)))
        node_idx = torch.randperm(
            N, generator=generator, device=g_device,
        )[:n_count]
        mask[k_start : k_start + k_extent].index_fill_(1, node_idx, True)

    # Caller may have requested a different output device than the generator's.
    if str(device) != str(g_device):
        mask = mask.to(device)
    return mask


def mask_to_tokens(
    tokens: torch.Tensor,
    mask: torch.Tensor,
    mask_token: torch.Tensor,
) -> torch.Tensor:
    """Replace masked positions in a (B, K, N, D) token grid with the
    learned mask token. Returns a fresh tensor (input is not modified).

    Args:
      tokens:     (B, K, N, D) token grid.
      mask:       (K, N) bool, True at masked positions. Shared across
                  the batch (V-JEPA style).
      mask_token: (D,) or broadcastable to (1, 1, 1, D).
    """
    if tokens.dim() != 4 or mask.dim() != 2:
        raise ValueError(
            f"expected tokens (B, K, N, D) and mask (K, N); got "
            f"{tuple(tokens.shape)} and {tuple(mask.shape)}"
        )
    B, K, N, D = tokens.shape
    if mask.shape != (K, N):
        raise ValueError(
            f"mask shape {tuple(mask.shape)} != ({K}, {N})"
        )
    out = tokens.clone()
    mask_b = mask.view(1, K, N, 1).expand(B, K, N, D)
    token_b = mask_token.view(1, 1, 1, -1).expand(B, K, N, D)
    out = torch.where(mask_b, token_b, out)
    return out
