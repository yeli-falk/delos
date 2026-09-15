"""JODIE (Kumar, Zhang & Leskovec, KDD 2019) -- faithful compact reimplementation.

Reference: "Predicting Dynamic Embedding Trajectory in Temporal Interaction
Networks", https://github.com/claws-lab/jodie

JODIE maintains a dynamic embedding per user and per item on a bipartite
event stream, with three mechanisms (all kept here, matching the reference):

  1. Coupled recurrent updates: on interaction (u, i, t, f) each side updates
     from the other side's DYNAMIC embedding (statics are not RNN inputs in
     the reference either):
         u <- RNN_u([i_dyn, f, dt_u]),   i <- RNN_i([u_dyn, f, dt_i])
  2. The temporal PROJECTION operator that forecasts an embedding forward in
     time without observing new events:
         u_hat(t + dt) = (1 + w(dt)) * u(t)
     This is JODIE's forecasting mechanism -- the analogue of DyG-JEPA's
     predictor -- and provides the `jepa` column in our state probe.
  3. Self-supervised next-item-embedding prediction, statics included exactly
     as in the reference:
         input  = [projected user dyn ; user one-hot ;
                   previous item dyn  ; previous item one-hot]
         target = [current item dyn (detached) ; current item one-hot]
     The one-hot target anchors the objective (it is a which-item-comes-next
     classification inside an MSE), so the dynamic embeddings cannot collapse.
     Plus the reference's embedding-drift L2 regularizers.

  One-hot Linear terms are implemented as nn.Embedding lookups -- a Linear
  applied to a one-hot IS a column lookup of its weight matrix, so this is
  mathematically identical to the reference and avoids materialising
  (batch, n_users) one-hots. The "no previous item" case uses padding_idx
  (a frozen zero row), matching the reference's null previous item.

Remaining deviation from the reference (document in the paper): standard
nn.RNNCell initialisation instead of the reference's custom NormalLinear.

The objective is self-supervised (state labels are never seen), so JODIE
slots into the same protocol as every other baseline: frozen embeddings ->
the identical downstream probe (`run_block_probe.py --jodie_ckpt`). Only the
DYNAMIC embedding (dim matched to node_dim) is ever probed -- probing the
one-hot statics would let the probe memorise user identity.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class JODIE(nn.Module):
    """Coupled user/item RNNs + temporal projection + next-item prediction.

    The embedding banks are NOT module parameters -- they are recurrent
    *state*, managed functionally by the trainer (initialised from the learned
    ``init_user`` / ``init_item`` vectors each epoch and updated out-of-place
    with ``index_copy`` so gradients flow within a backprop chunk).
    """

    def __init__(self, n_users: int, n_items: int, emb_dim: int = 128,
                 feat_dim: int = 0):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.emb_dim = emb_dim
        self.feat_dim = feat_dim
        out_dim = emb_dim + n_items                 # [item dyn ; item one-hot]

        # Shared learned initial embeddings (reference: one vector per side).
        self.init_user = nn.Parameter(F.normalize(torch.rand(emb_dim), dim=0))
        self.init_item = nn.Parameter(F.normalize(torch.rand(emb_dim), dim=0))

        # Coupled recurrent updates. Input = [other-side dyn emb, edge feats,
        # normalised time-since-last-interaction].
        in_dim = emb_dim + feat_dim + 1
        self.user_rnn = nn.RNNCell(in_dim, emb_dim)
        self.item_rnn = nn.RNNCell(in_dim, emb_dim)

        # Projection operator: u_hat = (1 + W dt) * u  (W: 1 -> emb_dim).
        self.project_w = nn.Linear(1, emb_dim)
        nn.init.normal_(self.project_w.weight, std=0.02)
        nn.init.zeros_(self.project_w.bias)

        # Prediction head, reference-faithful. The dense part consumes the two
        # dynamic embeddings; the one-hot parts are Embedding lookups
        # (== Linear-on-one-hot columns). n_items + 1 rows: last row is the
        # frozen-zero "no previous item" token.
        self.predict_dyn = nn.Linear(2 * emb_dim, out_dim)
        self.predict_ustat = nn.Embedding(n_users, out_dim)
        self.predict_istat = nn.Embedding(n_items + 1, out_dim,
                                          padding_idx=n_items)
        nn.init.normal_(self.predict_ustat.weight, std=0.02)
        nn.init.normal_(self.predict_istat.weight, std=0.02)
        with torch.no_grad():                       # keep the null row zero
            self.predict_istat.weight[n_items].zero_()

    def project(self, emb: torch.Tensor, dt_norm: torch.Tensor) -> torch.Tensor:
        """Forecast ``emb`` forward by normalised time gap ``dt_norm`` (B,)."""
        return emb * (1.0 + self.project_w(dt_norm.unsqueeze(-1)))

    def step(self, u_emb, i_emb, feats, dt_u, dt_i):
        """One vectorised interaction batch (disjoint users/items within it).

        Args:
          u_emb, i_emb: (B, D) current dynamic embeddings.
          feats:        (B, F) edge features (or (B, 0)).
          dt_u, dt_i:   (B,) normalised time since each side's last event.
        Returns (u_new, i_new).
        """
        u_in = torch.cat([i_emb, feats, dt_u.unsqueeze(-1)], dim=-1)
        i_in = torch.cat([u_emb, feats, dt_i.unsqueeze(-1)], dim=-1)
        u_new = self.user_rnn(u_in, u_emb)
        i_new = self.item_rnn(i_in, i_emb)
        return u_new, i_new

    def prediction_loss(self, u_proj, user_idx, prev_i_emb, prev_item_idx,
                        i_emb_target, item_idx):
        """Reference eq.: predict [i_dyn ; i_onehot] of the CURRENT item from
        the projected user state and the previous item, statics included.

        Args:
          u_proj:        (B, D) projected user dynamic embedding.
          user_idx:      (B,) user ids (for the user one-hot term).
          prev_i_emb:    (B, D) dynamic embedding of each user's previous item
                         (zeros when none).
          prev_item_idx: (B,) previous item ids, n_items when none.
          i_emb_target:  (B, D) current item dynamic embedding (pre-update).
          item_idx:      (B,) current item ids.
        """
        pred = (self.predict_dyn(torch.cat([u_proj, prev_i_emb], dim=-1))
                + self.predict_ustat(user_idx)
                + self.predict_istat(prev_item_idx))
        onehot = F.one_hot(item_idx, self.n_items).float()
        target = torch.cat([i_emb_target.detach(), onehot], dim=-1)
        return F.mse_loss(pred, target)
