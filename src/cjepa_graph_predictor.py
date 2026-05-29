"""
C-JEPA-DG: dynamic-graph analog of MaskedSlotPredictor.

Differences vs src/cjepa_predictor.py:
  - slots -> nodes with explicit identity
  - full attention -> masked attention over a spatio-temporal graph
  - adds optional edge masking + edge-prediction head
  - supports variable node set via per-timestep `active` masks
  - supports irregular time via continuous time embedding

Tensor shapes:
  x      : (B, T_hist, N, D)        node features, zero-padded to N
  adj    : (B, T_total, N, N) bool  per-timestep adjacency
  active : (B, T_total, N)    bool  True where node exists
  times  : (B, T_total) float       optional continuous timestamps
  hidden_edges : (B, T_total, N, N) bool  edges hidden from attention
                                          (and supervised by edge head)
"""
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class GraphMaskedAttention(nn.Module):
    def __init__(self, dim, heads, dim_head=64, dropout=0.0):
        super().__init__()
        inner = heads * dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, 3 * inner, bias=False)
        self.to_out = nn.Linear(inner, dim)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (rearrange(t, "b l (h d) -> b h l d", h=self.heads) for t in qkv)
        # SDPA uses memory-efficient kernels (no materialised (L, L) matrix),
        # which lets large graph sequences (T*N ~ 5k) fit on small GPUs.
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.attn_drop.p if self.training else 0.0,
        )
        out = rearrange(out, "b h l d -> b l (h d)")
        return self.resid_drop(self.to_out(out))


class GraphTransformerLayer(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = GraphMaskedAttention(dim, heads, dim_head, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim), nn.Dropout(dropout),
        )

    def forward(self, x, attn_mask=None):
        x = x + self.attn(self.norm1(x), attn_mask=attn_mask)
        x = x + self.mlp(self.norm2(x))
        return x


class GraphSpacetimeTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([
            GraphTransformerLayer(dim, heads, dim_head, mlp_dim, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, attn_mask=None):
        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask)
        return self.norm(x)


def build_spacetime_mask(
    adj: torch.Tensor,
    active: torch.Tensor,
    hidden_edges: Optional[torch.Tensor] = None,
    allow_temporal_self: bool = True,
    allow_temporal_neighbors: bool = False,
) -> torch.Tensor:
    """Build an additive attention mask of shape (B, 1, T*N, T*N).

    Token (t, i) attends to (t', j) iff both are active AND one of:
      - t == t' and (i, j) is a visible edge (self-loops always allowed),
      - i == j and allow_temporal_self,
      - t != t' and (i, j) is a visible edge at time t, and allow_temporal_neighbors.
    Hidden (masked) edges are removed from `adj` before building the mask.
    """
    B, T, N, _ = adj.shape
    device = adj.device

    adj_vis = adj & ~hidden_edges if hidden_edges is not None else adj
    eye_N = torch.eye(N, dtype=torch.bool, device=device)
    intra = adj_vis | eye_N.view(1, 1, N, N)

    allow = torch.zeros(B, T, N, T, N, dtype=torch.bool, device=device)
    for t in range(T):
        allow[:, t, :, t, :] = intra[:, t]

    if allow_temporal_self:
        eye_T = torch.eye(T, dtype=torch.bool, device=device)
        self_edges = eye_N.view(1, 1, N, 1, N) & ~eye_T.view(1, T, 1, T, 1)
        allow = allow | self_edges

    if allow_temporal_neighbors:
        cross = adj_vis.view(B, T, N, 1, N).expand(B, T, N, T, N)
        allow = allow | cross

    active_flat = rearrange(active, "b t n -> b (t n)")
    allow = rearrange(allow, "b t1 n1 t2 n2 -> b (t1 n1) (t2 n2)")
    allow = allow & active_flat.unsqueeze(1) & active_flat.unsqueeze(2)

    # Guarantee every row has a valid key so softmax never sees all -inf.
    # Self-loop on the full TN diagonal — inactive tokens attend only to themselves,
    # and their outputs are masked out of the loss anyway.
    L = allow.shape[-1]
    diag = torch.arange(L, device=device)
    allow[:, diag, diag] = True

    # Return a bool mask (True = attend). SDPA accepts this natively and avoids
    # the 4x memory blow-up from the additive-float form.
    return allow.unsqueeze(1)


class ContinuousTimeEmbedding(nn.Module):
    def __init__(self, dim, num_freqs=8, max_period=10000.0):
        super().__init__()
        freqs = torch.exp(torch.linspace(0.0, -np.log(max_period), num_freqs))
        self.register_buffer("freqs", freqs)
        self.proj = nn.Linear(2 * num_freqs, dim)

    def forward(self, t):
        x = t.unsqueeze(-1) * self.freqs
        return self.proj(torch.cat([x.sin(), x.cos()], dim=-1))


class MaskedNodePredictor(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        node_dim: int,
        history_frames: int,
        pred_frames: int = 1,
        num_masked_nodes: int = 2,
        seed: int = 42,
        depth: int = 6,
        heads: int = 8,
        dim_head: int = 64,
        mlp_dim: int = 2048,
        dropout: float = 0.1,
        use_continuous_time: bool = False,
        allow_temporal_neighbors: bool = False,
        use_edge_head: bool = False,
        use_attention_mask: bool = True,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.node_dim = node_dim
        self.history_frames = history_frames
        self.pred_frames = pred_frames
        self.total_frames = history_frames + pred_frames
        self.num_masked_nodes = num_masked_nodes
        self.seed = seed
        self.allow_temporal_neighbors = allow_temporal_neighbors
        self.use_attention_mask = use_attention_mask

        self.mask_token = nn.Parameter(torch.zeros(1, 1, node_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        # Used as the anchor feature for nodes that are absent at t=0 but appear later.
        self.new_node_token = nn.Parameter(torch.zeros(1, 1, node_dim))
        nn.init.trunc_normal_(self.new_node_token, std=0.02)

        if use_continuous_time:
            self.time_embed = ContinuousTimeEmbedding(node_dim)
            self.time_pos_embed = None
        else:
            self.time_embed = None
            self.time_pos_embed = nn.Parameter(
                torch.randn(1, self.total_frames, 1, node_dim)
            )

        self.id_projector = nn.Linear(node_dim, node_dim)
        self.transformer = GraphSpacetimeTransformer(
            dim=node_dim, depth=depth, heads=heads,
            dim_head=dim_head, mlp_dim=mlp_dim, dropout=dropout,
        )
        self.to_out = nn.Linear(node_dim, node_dim)

        self.edge_head = None
        if use_edge_head:
            self.edge_head = nn.Sequential(
                nn.Linear(2 * node_dim, node_dim), nn.GELU(),
                nn.Linear(node_dim, 1),
            )

    def _time_emb(self, B, T, times):
        if self.time_embed is not None:
            assert times is not None, "continuous time requires `times`"
            return self.time_embed(times).unsqueeze(2)
        return self.time_pos_embed[:, :T].expand(B, T, 1, self.node_dim)

    def _mask_indices(self, active_at_0, device):
        """Pick which nodes to hide. Shared across the batch (V-JEPA style)
        but resampled on every forward pass during training -- node identity
        on a graph is geographic, so fixing the seed would pin the masked
        sensors to the same 30 forever. Eval keeps the deterministic seed
        so val/test loss is comparable across runs."""
        if self.num_masked_nodes <= 0:
            return torch.tensor([], dtype=torch.long, device=device)
        valid = active_at_0.all(dim=0).nonzero(as_tuple=True)[0]
        pool = valid if valid.numel() >= self.num_masked_nodes \
            else torch.arange(self.num_nodes, device=device)
        if self.training:
            perm = torch.randperm(pool.numel(), device=device)[:self.num_masked_nodes]
            return pool[perm]
        rng = np.random.RandomState(self.seed)
        idx = rng.choice(pool.cpu().numpy(), self.num_masked_nodes, replace=False)
        return torch.from_numpy(idx).to(device)

    def prepare_input(self, x, active, times=None):
        B, T_hist, N, D = x.shape
        T_total = self.total_frames
        device = x.device

        masked_idx = self._mask_indices(active[:, 0], device)
        is_masked = torch.zeros(N, dtype=torch.bool, device=device)
        is_masked[masked_idx] = True

        active_at_0 = active[:, 0].unsqueeze(-1)
        anchor_feat = torch.where(
            active_at_0, x[:, 0], self.new_node_token.expand(B, N, D)
        )
        anchor_queries = self.id_projector(anchor_feat)

        time_emb = self._time_emb(B, T_total, times)
        time_grid = time_emb.expand(B, T_total, N, D)
        tokens_grid = self.mask_token.expand(B, T_total, N, D)
        anchor_grid = anchor_queries.unsqueeze(1).expand(B, T_total, N, D)

        out = tokens_grid + time_grid + anchor_grid

        # t=0 visible for every active node
        vis0 = active[:, 0:1].unsqueeze(-1)
        out = out.clone()
        out[:, 0:1] = torch.where(vis0, x[:, 0:1] + time_emb[:, 0:1], out[:, 0:1])

        # t in [1, T_hist) visible for active & unmasked nodes
        if T_hist > 1:
            unmasked = (~is_masked).view(1, 1, N, 1)
            vis = active[:, 1:T_hist].unsqueeze(-1) & unmasked
            out[:, 1:T_hist] = torch.where(
                vis, x[:, 1:T_hist] + time_emb[:, 1:T_hist], out[:, 1:T_hist]
            )

        return out, masked_idx

    def forward(self, x, adj, active, times=None, hidden_edges=None):
        B, T_hist, N, D = x.shape
        T_total = self.total_frames

        x_input, masked_idx = self.prepare_input(x, active, times)
        x_flat = rearrange(x_input, "b t n d -> b (t n) d")

        if self.use_attention_mask:
            attn_mask = build_spacetime_mask(
                adj, active,
                hidden_edges=hidden_edges,
                allow_temporal_self=True,
                allow_temporal_neighbors=self.allow_temporal_neighbors,
            )
        else:
            attn_mask = None

        h_flat = self.transformer(x_flat, attn_mask=attn_mask)
        h = rearrange(h_flat, "b (t n) d -> b t n d", t=T_total, n=N)
        pred = self.to_out(h)

        edge_out = None
        if self.edge_head is not None and hidden_edges is not None:
            edge_out = self._predict_edges(h, hidden_edges)

        return pred, masked_idx, edge_out

    def _predict_edges(self, node_emb, hidden_edges):
        b, t, i, j = hidden_edges.nonzero(as_tuple=True)
        h_ij = torch.cat([node_emb[b, t, i], node_emb[b, t, j]], dim=-1)
        logits = self.edge_head(h_ij).squeeze(-1)
        return (b, t, i, j), logits

    @torch.no_grad()
    def inference(self, x, adj, active, times=None):
        saved = self.num_masked_nodes
        self.num_masked_nodes = 0
        try:
            pred, _, _ = self.forward(x, adj, active, times=times)
        finally:
            self.num_masked_nodes = saved
        return pred[:, self.history_frames:]


def compute_graph_jepa_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    active: torch.Tensor,
    masked_idx: torch.Tensor,
    T_hist: int,
    edge_out=None,
    edge_labels: Optional[torch.Tensor] = None,
    edge_weight: float = 1.0,
):
    B, T_total, N, D = pred.shape
    diff2 = (pred - target).pow(2).mean(dim=-1)

    node_sel = torch.zeros(N, dtype=torch.bool, device=pred.device)
    node_sel[masked_idx] = True

    mh = torch.zeros_like(active)
    mh[:, 1:T_hist] = True
    mh = mh & active & node_sel.view(1, 1, N)
    fut = torch.zeros_like(active)
    fut[:, T_hist:] = True
    fut = fut & active

    loss_mh = (diff2 * mh.float()).sum() / mh.sum().clamp(min=1)
    loss_fut = (diff2 * fut.float()).sum() / fut.sum().clamp(min=1)
    losses = {"loss_masked_history": loss_mh, "loss_future": loss_fut}
    total = loss_mh + loss_fut

    if edge_out is not None and edge_labels is not None:
        _, logits = edge_out
        loss_edge = F.binary_cross_entropy_with_logits(logits, edge_labels.float())
        losses["loss_edge"] = loss_edge
        total = total + edge_weight * loss_edge

    losses["loss"] = total
    return losses


if __name__ == "__main__":
    torch.manual_seed(0)
    B, T_hist, T_pred, N, D = 2, 3, 1, 6, 32
    T_total = T_hist + T_pred

    model = MaskedNodePredictor(
        num_nodes=N, node_dim=D, history_frames=T_hist, pred_frames=T_pred,
        num_masked_nodes=2, depth=2, heads=4, dim_head=16, mlp_dim=64,
        use_continuous_time=True, use_edge_head=True,
    )

    x = torch.randn(B, T_hist, N, D)
    adj = (torch.rand(B, T_total, N, N) > 0.6)
    adj = adj | adj.transpose(-1, -2)
    active = torch.ones(B, T_total, N, dtype=torch.bool)
    active[:, :, -1] = False
    active[:, T_hist:, -2] = True
    times = torch.linspace(0, 1, T_total).unsqueeze(0).expand(B, -1)

    hidden_edges = adj & (torch.rand_like(adj, dtype=torch.float) < 0.2)
    edge_labels = adj[hidden_edges]

    pred, masked_idx, edge_out = model(x, adj, active, times=times, hidden_edges=hidden_edges)
    target = torch.randn_like(pred)
    losses = compute_graph_jepa_loss(
        pred, target, active, masked_idx, T_hist,
        edge_out=edge_out, edge_labels=edge_labels,
    )
    print("masked nodes:", masked_idx.tolist())
    print("pred shape:", tuple(pred.shape))
    print({k: round(v.item(), 4) for k, v in losses.items()})

    fut = model.inference(x, adj, active, times=times)
    print("future shape:", tuple(fut.shape))
