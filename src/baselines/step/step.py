"""STEP architecture, ported from https://github.com/GestaltCogTeam/STEP.

Three pieces, kept structurally faithful to the reference:

  :class:`TSFormer`               the masked autoencoder. Patchify a long series
                                  per node, drop 75% of the time patches, encode
                                  the rest, re-insert mask tokens, decode, and
                                  score the dropped patches. Unlike STD-MAE
                                  there is no spatial branch -- every node is
                                  encoded independently.
  :class:`DiscreteGraphLearning`  learns a *discrete* dependency graph. Node
                                  embeddings come from a CNN over each node's
                                  whole training series; every ordered pair gets
                                  a Bernoulli logit, and a hard Gumbel-softmax
                                  sample is the adjacency the backend uses. A
                                  kNN graph over the frozen TSFormer's
                                  representations is the prior it is regularised
                                  towards (:func:`step_loss`).
  :class:`STEP`                   frozen TSFormer + that learned graph feeding
                                  the shared Graph WaveNet backend.

Changes from the reference, all of them dependency removals, de-hard-coding, or
memory work -- never changes to the objective:

  * ``timm.trunc_normal_`` -> ``torch.nn.init.trunc_normal_`` (identical).
  * ``DiscreteGraphLearning`` looked up ``num_nodes``, ``train_length``,
    ``dim_fc`` and ``dim_fc_mean`` in per-dataset dicts and read the node
    features off a BasicTS pickle path. All four are derived here from the
    data that is passed in, so the module works on any dataset (``dim_fc`` is
    the conv output width).
  * ``rel_rec`` / ``rel_send`` were ``(N*N, N)`` one-hot matrices multiplied
    into the node features to build every ordered pair. They are exactly
    ``repeat_interleave`` and ``repeat``, which is what is used here.
  * the global node embedding is recomputed every forward in the reference.
    That is kept for training; under ``no_grad`` it is cached for the pass
    (see :meth:`DiscreteGraphLearning._global_feat`).
  * ``GraphWaveNet`` lives in :mod:`src.baselines.graphwavenet`, shared with
    STD-MAE (both papers ship near-identical forks of it).

Note the reference has ALREADY disabled the dynamic (TSFormer-derived) half of
the graph logits -- see the commented-out ``dynamic_feat`` line and its comment,
"it could lead to instability in the learning of the underlying graph
structure". The port matches the current reference: the learned graph comes
from the global series embedding, and TSFormer enters through the kNN prior and
through the representation injected into the backend.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from src.baselines.graphwavenet import GraphWaveNet
from src.baselines.masked_pretrain import (
    MaskGenerator,
    PatchEmbedding,
    TransformerLayers,
    masked_mae,
)

__all__ = ["TSFormer", "DiscreteGraphLearning", "STEP", "step_loss",
           "GraphWaveNet", "masked_mae", "random_walk_supports"]


# -- TSFormer --
class PositionalEncoding(nn.Module):
    """STEP's LEARNED positional table (STD-MAE uses a 2-D sin/cos one).

    ``index`` selects rows, which is how the decoder gives each re-inserted
    mask token the position of the patch it stands in for.
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.1, max_len: int = 1000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.position_embedding = nn.Parameter(
            torch.empty(max_len, hidden_dim), requires_grad=True)

    def forward(self, input_data: torch.Tensor, index=None, abs_idx=None):
        batch_size, num_nodes, num_patches, num_feat = input_data.shape
        x = input_data.view(batch_size * num_nodes, num_patches, num_feat)
        if index is None:
            pe = self.position_embedding[:x.size(1), :].unsqueeze(0)
        else:
            pe = self.position_embedding[index].unsqueeze(0)
        x = self.dropout(x + pe)
        return x.view(batch_size, num_nodes, num_patches, num_feat)


class TSFormer(nn.Module):
    """STEP's masked pre-training model over long per-node time series.

    ``mode="pre-train"``  -> forward returns ``(reconstructed_masked_tokens,
                             ground_truth_masked_tokens)`` for the MAE loss.
    ``mode="forecasting"`` -> forward returns the full hidden states
                             ``[B, N, P, d]``, nothing masked.
    """

    def __init__(self, patch_size: int, in_channel: int, embed_dim: int,
                 num_heads: int, mlp_ratio: int, dropout: float,
                 num_token: int, mask_ratio: float, encoder_depth: int,
                 decoder_depth: int, mode: str = "pre-train"):
        super().__init__()
        assert mode in ("pre-train", "forecasting"), f"bad mode {mode!r}"
        self.patch_size = patch_size
        self.in_channel = in_channel
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_token = int(num_token)
        self.mask_ratio = mask_ratio
        self.encoder_depth = encoder_depth
        self.mode = mode
        self.mlp_ratio = mlp_ratio
        self.selected_feature = 0

        self.encoder_norm = nn.LayerNorm(embed_dim)
        self.decoder_norm = nn.LayerNorm(embed_dim)
        self.patch_embedding = PatchEmbedding(patch_size, in_channel, embed_dim,
                                              norm_layer=None)
        # The reference pins max_len=1000; keep that floor but never let a long
        # window silently index past the end of the table.
        self.positional_encoding = PositionalEncoding(
            embed_dim, dropout=dropout, max_len=max(1000, self.num_token))
        self.mask = MaskGenerator(self.num_token, mask_ratio)
        self.encoder = TransformerLayers(embed_dim, encoder_depth, mlp_ratio,
                                         num_heads, dropout)
        self.enc_2_dec_emb = nn.Linear(embed_dim, embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, embed_dim))
        self.decoder = TransformerLayers(embed_dim, decoder_depth, mlp_ratio,
                                         num_heads, dropout)
        self.output_layer = nn.Linear(embed_dim, patch_size)

        nn.init.uniform_(self.positional_encoding.position_embedding, -0.02, 0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def encoding(self, long_term_history: torch.Tensor, mask: bool = True):
        """``long_term_history``: [B, N, C, P*L]."""
        batch_size, num_nodes, _, _ = long_term_history.shape
        patches = self.patch_embedding(long_term_history).transpose(-1, -2)
        patches = self.positional_encoding(patches)             # B, N, P, d

        if mask:
            unmasked_token_index, masked_token_index = self.mask()
            encoder_input = patches[:, :, unmasked_token_index, :]
        else:
            unmasked_token_index, masked_token_index = None, None
            encoder_input = patches

        hidden = self.encoder(encoder_input)
        hidden = self.encoder_norm(hidden).view(
            batch_size, num_nodes, -1, self.embed_dim)
        return hidden, unmasked_token_index, masked_token_index

    def decoding(self, hidden_states_unmasked: torch.Tensor,
                 masked_token_index: List[int]) -> torch.Tensor:
        batch_size, num_nodes, _, _ = hidden_states_unmasked.shape
        hidden_states_unmasked = self.enc_2_dec_emb(hidden_states_unmasked)
        # Mask tokens carry only the position of the patch they replace.
        hidden_states_masked = self.positional_encoding(
            self.mask_token.expand(batch_size, num_nodes,
                                   len(masked_token_index),
                                   hidden_states_unmasked.shape[-1]),
            index=masked_token_index)
        hidden_states_full = torch.cat(
            [hidden_states_unmasked, hidden_states_masked], dim=-2)
        hidden_states_full = self.decoder_norm(self.decoder(hidden_states_full))
        return self.output_layer(hidden_states_full.view(
            batch_size, num_nodes, -1, self.embed_dim))

    def get_reconstructed_masked_tokens(self, reconstruction_full, real_value_full,
                                        unmasked_token_index, masked_token_index):
        """Line up reconstruction and ground truth on the MASKED tokens only.

        ``decoding`` concatenates unmasked-then-masked, so the masked block is
        always the tail; the label is gathered by the masked indices, in the
        same (sorted) order.
        """
        batch_size, num_nodes, _, _ = reconstruction_full.shape
        recon = reconstruction_full[:, :, len(unmasked_token_index):, :]
        recon = recon.reshape(batch_size, num_nodes, -1).transpose(1, 2)

        label_full = real_value_full.permute(0, 3, 1, 2).unfold(
            1, self.patch_size, self.patch_size
        )[:, :, :, self.selected_feature, :].transpose(1, 2)        # B, N, P, L
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


# -- discrete graph learning --
def batch_cosine_similarity(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    l2_x = torch.norm(x, dim=2, p=2) + 1e-7
    l2_y = torch.norm(y, dim=2, p=2) + 1e-7
    l2_m = torch.matmul(l2_x.unsqueeze(dim=2), l2_y.unsqueeze(dim=2).transpose(1, 2))
    return torch.matmul(x, y.transpose(1, 2)) / l2_m


def gumbel_softmax(logits: torch.Tensor, temperature: float, hard: bool = False,
                   eps: float = 1e-10) -> torch.Tensor:
    """Sample from Gumbel-Softmax; ``hard`` straight-throughs to a one-hot."""
    uniform = torch.rand(logits.size(), device=logits.device)
    gumbel = -torch.log(-torch.log(uniform + eps) + eps)
    y_soft = F.softmax((logits + gumbel) / temperature, dim=-1)
    if not hard:
        return y_soft
    index = y_soft.max(-1, keepdim=True)[1]
    y_hard = torch.zeros_like(logits).scatter_(-1, index, 1.0)
    return (y_hard - y_soft).detach() + y_soft


class DiscreteGraphLearning(nn.Module):
    """STEP's learned dependency graph.

    ``node_feats`` is the TRAIN region of the normalised signal, ``(T_train, N)``
    -- the reference reads exactly that off the BasicTS pickle and slices it to
    the train length. Nothing later than the train split may be used here: the
    learned adjacency is a model parameter, and feeding it val/test signal would
    leak held-out data into every forecast.
    """

    def __init__(self, node_feats: np.ndarray, k: int = 10,
                 embedding_dim: int = 100):
        super().__init__()
        feats = np.asarray(node_feats, dtype=np.float32)
        train_length, num_nodes = feats.shape
        # conv1(k=10) then conv2(k=10), both stride 1, no padding.
        conv_out = train_length - 18
        if conv_out < 1:
            raise SystemExit(
                f"STEP's graph learner convolves each node's whole training "
                f"series with two width-10 kernels, so it needs at least 19 "
                f"train timesteps; this dataset has {train_length}. Bin the "
                f"timeline into more steps, or drop the STD-MAE/STEP arms for "
                f"it.")
        self.k = k
        self.num_nodes = num_nodes
        self.train_length = train_length
        self.embedding_dim = embedding_dim
        self.dim_fc = 16 * conv_out
        # (T_train, N) -> (N, 1, T_train) once, as a buffer so .to(device) moves it.
        self.register_buffer("node_feats",
                             torch.from_numpy(feats).transpose(1, 0).contiguous()
                             .view(num_nodes, 1, -1))

        self.conv1 = nn.Conv1d(1, 8, 10, stride=1)
        self.conv2 = nn.Conv1d(8, 16, 10, stride=1)
        self.fc = nn.Linear(self.dim_fc, embedding_dim)
        self.bn1 = nn.BatchNorm1d(8)
        self.bn2 = nn.BatchNorm1d(16)
        self.bn3 = nn.BatchNorm1d(embedding_dim)

        self.fc_cat = nn.Linear(embedding_dim, 2)
        self.fc_out = nn.Linear(embedding_dim * 2, embedding_dim)
        # (The reference also builds a `self.dropout = nn.Dropout(0.5)` here and
        # never calls it; dropping dead state rather than carrying it.)
        self._global_cache: Optional[torch.Tensor] = None

    def train(self, mode: bool = True):
        # Any mode switch invalidates the cached embedding: it is only valid
        # for the exact weights and BatchNorm mode it was computed under.
        self._global_cache = None
        return super().train(mode)

    def _global_feat(self) -> torch.Tensor:
        """``(N, embedding_dim)`` embedding of each node's whole train series.

        Independent of the batch, but its weights move every step, so training
        recomputes it exactly as the reference does. Under ``no_grad`` the
        weights are frozen and BatchNorm uses running stats, so caching it for
        the pass is exactly equivalent and removes what is otherwise the
        dominant cost of an evaluation.
        """
        if not self.training and self._global_cache is not None:
            return self._global_cache
        x = self.bn1(F.relu(self.conv1(self.node_feats)))
        x = self.bn2(F.relu(self.conv2(x)))
        x = self.bn3(F.relu(self.fc(x.view(self.num_nodes, -1))))
        if not self.training:
            self._global_cache = x.detach()
        else:
            self._global_cache = None
        return x

    def get_k_nn_neighbor(self, data: torch.Tensor, k: int) -> torch.Tensor:
        """Binary kNN graph over cosine similarity, top-``k`` GLOBALLY.

        Note this is top-k over the flattened ``(N*N)`` similarity matrix, not
        per row -- the reference's behaviour, so a hub node can take many edges.
        """
        sim = batch_cosine_similarity(data, data)                  # B, N, N
        batch_size, num_nodes, _ = sim.shape
        flat = sim.view(batch_size, num_nodes * num_nodes)
        res = torch.zeros_like(flat)
        top_k, indices = torch.topk(flat, min(k, flat.shape[-1]), dim=-1)
        res.scatter_(-1, indices, top_k)
        adj = torch.where(res != 0, 1.0, 0.0).detach().clone()
        return adj.view(batch_size, num_nodes, num_nodes)

    def forward(self, long_term_history: torch.Tensor, tsformer: nn.Module):
        """``long_term_history``: [B, P*L, N, C].

        Returns ``(bernoulli_unnorm [B,N*N,2], hidden_states [B,N,P,d],
        adj_knn [B,N,N], sampled_adj [B,N,N])``.
        """
        batch_size, _, num_nodes, _ = long_term_history.shape
        node_feat = self._global_feat().unsqueeze(0).expand(
            batch_size, num_nodes, -1)                              # Gi, Eq. (2)

        hidden_states = tsformer(long_term_history[..., [0]])

        # Every ordered pair (i, j): receivers repeat each node N times,
        # senders tile the node list N times.
        receivers = node_feat.repeat_interleave(num_nodes, dim=1)
        senders = node_feat.repeat(1, num_nodes, 1)
        edge_feat = torch.relu(self.fc_out(torch.cat([senders, receivers], dim=-1)))
        bernoulli_unnorm = self.fc_cat(edge_feat)                   # Theta, Eq. (2)

        # Differentiable discrete sampling, Eq. (4).
        sampled_adj = gumbel_softmax(bernoulli_unnorm, temperature=0.5, hard=True)
        sampled_adj = sampled_adj[..., 0].clone().reshape(batch_size, num_nodes, -1)
        mask = torch.eye(num_nodes, device=sampled_adj.device).unsqueeze(0).bool()
        sampled_adj.masked_fill_(mask, 0)

        # Prior graph: kNN over the frozen TSFormer representations.
        adj_knn = self.get_k_nn_neighbor(
            hidden_states.reshape(batch_size, num_nodes, -1),
            k=self.k * self.num_nodes)
        adj_knn.masked_fill_(mask, 0)
        return bernoulli_unnorm, hidden_states, adj_knn, sampled_adj


def random_walk_supports(adj: torch.Tensor) -> List[torch.Tensor]:
    """``[rw(A), rw(Aᵀ)]`` with ``rw(A) = rownorm(A + I)`` -- the reference's
    ``_calculate_random_walk_matrix`` applied to the sampled graph."""
    def rw(a):
        a = a + torch.eye(a.shape[-1], device=a.device, dtype=a.dtype)
        d_inv = 1.0 / a.sum(-1, keepdim=True)
        d_inv = torch.where(torch.isinf(d_inv), torch.zeros_like(d_inv), d_inv)
        return a * d_inv
    return [rw(adj), rw(adj.transpose(-1, -2))]


# -- the full model + loss --
class STEP(nn.Module):
    """Frozen TSFormer + a learned discrete graph feeding Graph WaveNet."""

    def __init__(self, tsformer: TSFormer, backend: GraphWaveNet,
                 dgl: DiscreteGraphLearning):
        super().__init__()
        self.tsformer = tsformer
        self.backend = backend
        self.discrete_graph_learning = dgl
        self.freeze_tsformer()

    def freeze_tsformer(self) -> None:
        for p in self.tsformer.parameters():
            p.requires_grad = False
        self.tsformer.eval()

    def train(self, mode: bool = True):
        """Keep the frozen TSFormer in eval mode: its dropout must not fire and
        its LayerNorms must not drift when the rest switches to train."""
        super().train(mode)
        self.tsformer.eval()
        return self

    def forward(self, history_data: torch.Tensor, long_history_data: torch.Tensor):
        """``history_data``: [B, L, N, C]. ``long_history_data``: [B, L*P, N, 1].

        Returns ``(y_hat [B, out, N, 1], theta [B, N, N], adj_knn [B, N, N])``,
        where ``theta`` is the edge probability :func:`step_loss` regularises
        towards ``adj_knn``.
        """
        batch_size, _, num_nodes, _ = history_data.shape
        # The graph learner calls the TSFormer itself, as in the reference.
        # Every TSFormer parameter has requires_grad=False and the input is
        # plain data, so no graph is built for it and no activations are kept.
        bernoulli_unnorm, hidden_states, adj_knn, sampled_adj = \
            self.discrete_graph_learning(long_history_data, self.tsformer)
        hidden_states = hidden_states[:, :, -1, :]
        y_hat = self.backend(history_data, hidden_states=hidden_states,
                             supports=random_walk_supports(sampled_adj))
        theta = bernoulli_unnorm.softmax(-1)[..., 0].clone().reshape(
            batch_size, num_nodes, num_nodes)
        return y_hat.transpose(1, 2).unsqueeze(-1), theta, adj_knn


def gsl_coefficient(epoch: Optional[int]) -> float:
    """Weight on the graph-structure loss: ``1 / (epoch // 6 + 1)``.

    The reference decays it so the learned graph is pulled hard towards the kNN
    prior early and is progressively freed to depart from it.

    ``epoch`` is 1-INDEXED, matching the reference: EasyTorch hands its runner a
    1-indexed epoch and STEP applies this formula to it directly (unlike the
    curriculum, which explicitly subtracts one first). Callers with a 0-indexed
    loop must pass ``epoch + 1``.
    """
    return 0.0 if epoch is None else 1.0 / (int(epoch / 6) + 1)


def step_loss(prediction: torch.Tensor, real_value: torch.Tensor,
              theta: torch.Tensor, priori_adj: torch.Tensor,
              coefficient: float, null_val: float = 0.0) -> torch.Tensor:
    """``masked_mae`` on the forecast + BCE pulling the learned graph to the prior."""
    batch_size = theta.shape[0]
    loss_graph = F.binary_cross_entropy(theta.reshape(batch_size, -1),
                                        priori_adj.reshape(batch_size, -1))
    loss_pred = masked_mae(prediction, real_value, null_val=null_val)
    return loss_pred + loss_graph * coefficient
