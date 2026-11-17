"""STD-MAE architecture, ported from https://github.com/Jimmy-7664/STD-MAE.

Two pieces, kept structurally faithful to the reference so the numbers stay
attributable to the method rather than to a re-derivation:

  :class:`Mask`          the masked autoencoder. ``spatial=False`` masks whole
                         time patches (T-MAE); ``spatial=True`` masks whole
                         nodes (S-MAE). Same module, same weights layout, both
                         stages -- exactly as in the reference.
  :class:`STDMAE`        frozen S-MAE + T-MAE encoders feeding the shared
                         :class:`src.baselines.graphwavenet.GraphWaveNet`, with
                         STD-MAE's two-stream hidden-state injection
                         (``hidden_streams=2``).

Changes from the reference, all of them dependency removals or shape
generalisations -- never changes to the objective:

  * ``timm.trunc_normal_``  -> ``torch.nn.init.trunc_normal_`` (identical).
  * ``positional_encodings.PositionalEncoding2D`` -> :class:`PositionalEncoding`
    below, a line-for-line reimplementation of that package's 2-D sin/cos
    table, cached per (grid, dtype, device) instead of rebuilt every forward.
  * the Graph WaveNet backend lives in :mod:`src.baselines.graphwavenet`,
    shared with the STEP arm; see that module for its two mechanical changes.
  * ``STDMAE`` takes already-constructed ``Mask`` modules rather than
    checkpoint paths, so the training script owns loading and device placement.

The patchifier, transformer stack, mask generator and ``masked_mae`` loss are
byte-identical to STEP's, so they live in :mod:`src.baselines.masked_pretrain`
and are imported here (and re-exported, so this module still reads as the whole
architecture).
"""
from __future__ import annotations

import math
from typing import List, Optional, Sequence

import torch
from torch import nn

from src.baselines.graphwavenet import GraphWaveNet
from src.baselines.masked_pretrain import (
    MaskGenerator,
    PatchEmbedding,
    TransformerLayers,
    masked_mae,
)


__all__ = ["Mask", "STDMAE", "GraphWaveNet", "PositionalEncoding", "masked_mae",
           "MaskGenerator", "PatchEmbedding", "TransformerLayers"]


# -- masked autoencoder pieces (the rest come from masked_pretrain) --
def _sin_cos(sin_inp: torch.Tensor) -> torch.Tensor:
    """``get_emb`` from the ``positional_encodings`` package."""
    return torch.flatten(torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1), -2, -1)


class PositionalEncoding(nn.Module):
    """2-D sin/cos encoding over the (node, patch) grid.

    Reimplements ``positional_encodings.torch_encodings.PositionalEncoding2D``
    exactly -- half the channels encode the node index, half the patch index --
    but caches the table per (X, Y, channels, dtype, device). The reference
    rebuilt it on every forward, twice.

    Returns ``(input + table, table)``. The second value is STD-MAE's
    ``pos_mat``: the decoder re-adds the positions of the tokens it dropped.
    """

    def __init__(self):
        super().__init__()
        self._cache: dict = {}

    def _table(self, x: int, y: int, channels: int, device, dtype) -> torch.Tensor:
        key = (x, y, channels, str(device), str(dtype))
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        c = int(math.ceil(channels / 4) * 2)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, c, 2, device=device,
                                                 dtype=torch.float32) / c))
        pos_x = torch.arange(x, device=device, dtype=torch.float32)
        pos_y = torch.arange(y, device=device, dtype=torch.float32)
        emb_x = _sin_cos(torch.einsum("i,j->ij", pos_x, inv_freq)).unsqueeze(1)
        emb_y = _sin_cos(torch.einsum("i,j->ij", pos_y, inv_freq)).unsqueeze(0)
        emb = torch.zeros((x, y, c * 2), device=device, dtype=torch.float32)
        emb[:, :, :c] = emb_x
        emb[:, :, c:2 * c] = emb_y
        table = emb[None, :, :, :channels].to(dtype)
        self._cache[key] = table
        return table

    def forward(self, input_data: torch.Tensor):
        batch_size, num_nodes, num_patches, num_feat = input_data.shape
        table = self._table(num_nodes, num_patches, num_feat,
                            input_data.device, input_data.dtype)
        pos = table.expand(batch_size, -1, -1, -1)
        return input_data + pos, pos


class Mask(nn.Module):
    """One STD-MAE branch: S-MAE when ``spatial=True``, T-MAE otherwise.

    ``mode="pre-train"``  -> forward returns ``(reconstructed_masked_tokens,
                             ground_truth_masked_tokens)`` for the MAE loss.
    ``mode="forecasting"`` -> forward returns the full hidden states
                             ``[B, N, P, d]``, nothing masked. This is the
                             representation the backend consumes.
    """

    def __init__(self, patch_size: int, in_channel: int, embed_dim: int,
                 num_heads: int, mlp_ratio: int, dropout: float,
                 mask_ratio: float, encoder_depth: int, decoder_depth: int,
                 spatial: bool = False, mode: str = "pre-train"):
        super().__init__()
        assert mode in ("pre-train", "forecasting"), f"bad mode {mode!r}"
        self.patch_size = patch_size
        self.in_channel = in_channel
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mask_ratio = mask_ratio
        self.encoder_depth = encoder_depth
        self.mode = mode
        self.mlp_ratio = mlp_ratio
        self.spatial = spatial
        self.selected_feature = 0
        self.pos_mat: Optional[torch.Tensor] = None

        self.encoder_norm = nn.LayerNorm(embed_dim)
        self.decoder_norm = nn.LayerNorm(embed_dim)
        self.patch_embedding = PatchEmbedding(patch_size, in_channel, embed_dim,
                                              norm_layer=None)
        self.positional_encoding = PositionalEncoding()
        self.encoder = TransformerLayers(embed_dim, encoder_depth, mlp_ratio,
                                         num_heads, dropout)
        self.enc_2_dec_emb = nn.Linear(embed_dim, embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, embed_dim))
        self.decoder = TransformerLayers(embed_dim, decoder_depth, mlp_ratio,
                                         num_heads, dropout)
        self.output_layer = nn.Linear(embed_dim, patch_size)
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    # -- encoder ----------------------------------------------------------
    def encoding(self, long_term_history: torch.Tensor, mask: bool = True):
        """``long_term_history``: [B, N, C, P*L]. Returns
        ``(hidden_states, unmasked_token_index, masked_token_index)``."""
        batch_size, num_nodes = long_term_history.shape[0], long_term_history.shape[1]
        patches = self.patch_embedding(long_term_history).transpose(-1, -2)
        patches, self.pos_mat = self.positional_encoding(patches)   # B, N, P, d
        num_time = patches.shape[2]

        if not mask:
            encoder_input = patches
            if self.spatial:
                encoder_input = encoder_input.transpose(-2, -3)     # B, P, N, d
            hidden = self.encoder(encoder_input)
            if self.spatial:
                hidden = hidden.transpose(-2, -3)                   # B, N, P, d
            hidden = self.encoder_norm(hidden).view(
                batch_size, num_nodes, -1, self.embed_dim)
            return hidden, None, None

        if self.spatial:
            # S-MAE: drop whole NODES, keep every time patch.
            gen = MaskGenerator(patches.shape[1], self.mask_ratio)
            unmasked_token_index, masked_token_index = gen.uniform_rand()
            encoder_input = patches[:, unmasked_token_index, :, :].transpose(-2, -3)
            hidden = self.encoder(encoder_input)
            hidden = self.encoder_norm(hidden).view(
                batch_size, num_time, -1, self.embed_dim)           # B, P, N', d
        else:
            # T-MAE: drop whole time PATCHES, keep every node.
            gen = MaskGenerator(patches.shape[2], self.mask_ratio)
            unmasked_token_index, masked_token_index = gen.uniform_rand()
            encoder_input = patches[:, :, unmasked_token_index, :]
            hidden = self.encoder(encoder_input)
            hidden = self.encoder_norm(hidden).view(
                batch_size, num_nodes, -1, self.embed_dim)          # B, N, P', d
        return hidden, unmasked_token_index, masked_token_index

    # -- decoder ----------------------------------------------------------
    def decoding(self, hidden_states_unmasked: torch.Tensor,
                 masked_token_index: List[int]) -> torch.Tensor:
        hidden_states_unmasked = self.enc_2_dec_emb(hidden_states_unmasked)

        if self.spatial:
            batch_size, num_time, num_unmasked, _ = hidden_states_unmasked.shape
            unmasked_token_index = [
                i for i in range(0, len(masked_token_index) + num_unmasked)
                if i not in masked_token_index]
            hidden_states_masked = self.pos_mat[:, masked_token_index, :, :]
            hidden_states_masked = hidden_states_masked.transpose(-2, -3)
            hidden_states_masked = hidden_states_masked + self.mask_token.expand(
                batch_size, num_time, len(masked_token_index), self.embed_dim)
            hidden_states_unmasked = hidden_states_unmasked + \
                self.pos_mat[:, unmasked_token_index, :, :].transpose(-2, -3)
            hidden_states_full = torch.cat(
                [hidden_states_unmasked, hidden_states_masked], dim=-2)
            hidden_states_full = self.decoder_norm(self.decoder(hidden_states_full))
            return self.output_layer(hidden_states_full.view(
                batch_size, num_time, -1, self.embed_dim))          # B, P, N, L

        batch_size, num_nodes, num_unmasked, _ = hidden_states_unmasked.shape
        unmasked_token_index = [
            i for i in range(0, len(masked_token_index) + num_unmasked)
            if i not in masked_token_index]
        hidden_states_masked = self.pos_mat[:, :, masked_token_index, :]
        hidden_states_masked = hidden_states_masked + self.mask_token.expand(
            batch_size, num_nodes, len(masked_token_index), self.embed_dim)
        hidden_states_unmasked = hidden_states_unmasked + \
            self.pos_mat[:, :, unmasked_token_index, :]
        hidden_states_full = torch.cat(
            [hidden_states_unmasked, hidden_states_masked], dim=-2)
        hidden_states_full = self.decoder_norm(self.decoder(hidden_states_full))
        return self.output_layer(hidden_states_full.view(
            batch_size, num_nodes, -1, self.embed_dim))             # B, N, P, L

    # -- loss targets -----------------------------------------------------
    def get_reconstructed_masked_tokens(self, reconstruction_full, real_value_full,
                                        unmasked_token_index, masked_token_index):
        """Line up reconstruction and ground truth on the MASKED tokens only.

        ``decoding`` concatenates unmasked-then-masked, so the masked block is
        always the tail; the label is gathered from ``real_value_full`` by the
        masked indices, in the same (sorted) order.
        """
        label_full = real_value_full.permute(0, 3, 1, 2).unfold(
            1, self.patch_size, self.patch_size
        )[:, :, :, self.selected_feature, :].transpose(1, 2)        # B, N, P, L

        if self.spatial:
            batch_size, num_time, _, _ = reconstruction_full.shape
            recon = reconstruction_full[:, :, len(unmasked_token_index):, :]
            recon = recon.reshape(batch_size, num_time, -1)
            label = label_full[:, masked_token_index, :, :].transpose(1, 2)
            label = label.contiguous().view(batch_size, num_time, -1)
            return recon, label

        batch_size, num_nodes, _, _ = reconstruction_full.shape
        recon = reconstruction_full[:, :, len(unmasked_token_index):, :]
        recon = recon.reshape(batch_size, num_nodes, -1).transpose(1, 2)
        label = label_full[:, :, masked_token_index, :].contiguous()
        label = label.view(batch_size, num_nodes, -1).transpose(1, 2)
        return recon, label

    def forward(self, history_data: torch.Tensor, **kwargs):
        """``history_data``: [B, L, N, C] (C == ``in_channel``)."""
        history_data = history_data.permute(0, 2, 3, 1)             # B, N, C, L
        if self.mode == "pre-train":
            hidden, unmasked_idx, masked_idx = self.encoding(history_data)
            reconstruction_full = self.decoding(hidden, masked_idx)
            return self.get_reconstructed_masked_tokens(
                reconstruction_full, history_data, unmasked_idx, masked_idx)
        hidden, _, _ = self.encoding(history_data, mask=False)
        return hidden                                               # B, N, P, d


# -- the full model --
class STDMAE(nn.Module):
    """Frozen S-MAE + T-MAE representations feeding a Graph WaveNet forecaster.

    Both MAEs must already be in ``mode="forecasting"`` and carry their
    pre-trained weights; :meth:`freeze_maes` puts them in eval mode with
    ``requires_grad=False``, which is what makes this arm "pre-train then train
    the backend" rather than a joint end-to-end model.
    """

    def __init__(self, tmae: Mask, smae: Mask, backend: GraphWaveNet):
        super().__init__()
        self.tmae = tmae
        self.smae = smae
        self.smae.spatial = True
        self.tmae.spatial = False
        self.backend = backend
        self.freeze_maes()

    def freeze_maes(self) -> None:
        for p in self.tmae.parameters():
            p.requires_grad = False
        for p in self.smae.parameters():
            p.requires_grad = False
        self.tmae.eval()
        self.smae.eval()

    def train(self, mode: bool = True):
        """Keep the frozen MAEs in eval mode: their dropout must not fire and
        their LayerNorms must not drift when the backend switches to train."""
        super().train(mode)
        self.tmae.eval()
        self.smae.eval()
        return self

    def forward(self, history_data: torch.Tensor, long_history_data: torch.Tensor,
                supports: Optional[Sequence[torch.Tensor]] = None) -> torch.Tensor:
        """``history_data``: [B, L, N, C] short forecast history.
        ``long_history_data``: [B, L*P, N, 1] the MAEs' long input window.
        Returns [B, out_dim, N, 1]."""
        with torch.no_grad():
            hidden_t = self.tmae(long_history_data[..., [0]])       # B, N, P, d
            hidden_s = self.smae(long_history_data[..., [0]])       # B, N, P, d
        hidden_states = torch.cat((hidden_t, hidden_s), -1)[:, :, -1, :]
        y_hat = self.backend(history_data, hidden_states=hidden_states,
                             supports=supports)                     # B, N, out
        return y_hat.transpose(1, 2).unsqueeze(-1)                  # B, out, N, 1
