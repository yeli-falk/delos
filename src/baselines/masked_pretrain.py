"""Masked time-series pre-training primitives shared by STEP and STD-MAE.

STD-MAE forked STEP's codebase, so these four pieces are byte-identical in the
two reference repositories:

  :func:`masked_mae`       BasicTS's null-value-aware MAE, the training loss of
                           both pre-training stages and both forecasters.
  :class:`PatchEmbedding`  patchify a long series along time: a strided Conv2d
                           over ``[B, N, C, P*L] -> [B, N, d, P]``.
  :class:`TransformerLayers`  post-norm transformer encoder applied to the
                           last-but-one axis of a ``[B, N, L, D]`` tensor.
  :class:`MaskGenerator`   uniform random token mask.

They live here so the two arms cannot drift apart in them: a difference between
their rows should be the pre-training objective, not the patchifier.

What is NOT shared, because the papers genuinely differ: the positional
encoding (STEP learns a table, STD-MAE uses a 2-D sin/cos one over the
node x patch grid) and the masking axis (STEP masks time patches only; STD-MAE
decouples into a node-masking and a patch-masking branch).
"""
from __future__ import annotations

import math
import random

import torch
from torch import nn


def masked_mae(preds: torch.Tensor, labels: torch.Tensor,
               null_val: float = 0.0) -> torch.Tensor:
    """BasicTS's ``masked_mae``: MAE with ``null_val`` entries excluded.

    ``preds``/``labels`` are in the ORIGINAL (un-normalised) scale -- BasicTS
    re-scales before computing the loss, and so do the callers here, because
    ``null_val=0.0`` only means "missing reading" in raw units.
    """
    if null_val != null_val:                                   # NaN sentinel
        mask = ~torch.isnan(labels)
    else:
        # atol matches the reference; without it float round-trip error through
        # the (x-mean)/std normalisation would leave a "missing" reading a few
        # 1e-6 away from 0 and it would be scored as a real target.
        mask = ~torch.isclose(
            labels, torch.tensor(null_val, dtype=labels.dtype,
                                 device=labels.device).expand_as(labels),
            atol=5e-5, rtol=0.0)
    mask = mask.float()
    mask = mask / torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels) * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


class PatchEmbedding(nn.Module):
    """Patchify time series along time: [B,N,C,P*L] -> [B,N,d,P]."""

    def __init__(self, patch_size: int, in_channel: int, embed_dim: int,
                 norm_layer=None):
        super().__init__()
        self.len_patch = patch_size
        self.input_channel = in_channel
        self.output_channel = embed_dim
        self.input_embedding = nn.Conv2d(
            in_channel, embed_dim,
            kernel_size=(self.len_patch, 1), stride=(self.len_patch, 1))
        self.norm_layer = norm_layer if norm_layer is not None else nn.Identity()

    def forward(self, long_term_history: torch.Tensor) -> torch.Tensor:
        batch_size, num_nodes, num_feat, len_time_series = long_term_history.shape
        x = long_term_history.reshape(
            batch_size * num_nodes, num_feat, len_time_series, 1)
        out = self.norm_layer(self.input_embedding(x))
        out = out.squeeze(-1).view(batch_size, num_nodes, self.output_channel, -1)
        assert out.shape[-1] == len_time_series // self.len_patch
        return out                                              # B, N, d, P


class TransformerLayers(nn.Module):
    """Post-norm transformer encoder over the last-but-one axis of [B,N,L,D]."""

    def __init__(self, hidden_dim: int, nlayers: int, mlp_ratio: int,
                 num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.d_model = hidden_dim
        layer = nn.TransformerEncoderLayer(
            hidden_dim, num_heads, hidden_dim * mlp_ratio, dropout)
        # enable_nested_tensor=False: the nested-tensor fast path is a no-op
        # here (no padding mask is ever passed) and only emits a warning.
        self.transformer_encoder = nn.TransformerEncoder(
            layer, nlayers, enable_nested_tensor=False)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        B, N, L, D = src.shape
        src = src * math.sqrt(self.d_model)
        src = src.contiguous().view(B * N, L, D).transpose(0, 1)
        out = self.transformer_encoder(src, mask=None)
        return out.transpose(0, 1).view(B, N, L, D)


class MaskGenerator(nn.Module):
    """Uniform random token mask. Draws from python's ``random``, which
    ``src.train.engine.set_deterministic`` seeds, so a run is reproducible."""

    def __init__(self, num_tokens: int, mask_ratio: float):
        super().__init__()
        self.num_tokens = num_tokens
        self.mask_ratio = mask_ratio
        self.sort = True

    def uniform_rand(self):
        mask = list(range(int(self.num_tokens)))
        random.shuffle(mask)
        mask_len = int(self.num_tokens * self.mask_ratio)
        self.masked_tokens = mask[:mask_len]
        self.unmasked_tokens = mask[mask_len:]
        if self.sort:
            self.masked_tokens = sorted(self.masked_tokens)
            self.unmasked_tokens = sorted(self.unmasked_tokens)
        return self.unmasked_tokens, self.masked_tokens

    def forward(self):
        return self.uniform_rand()
