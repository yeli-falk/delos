"""
DELOS predictor.

  - nodes with explicit identity
  - masked attention over a spatio-temporal graph
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
from torch.utils.checkpoint import checkpoint


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
        """attn_mask is one of
          - None: full attention (SDPA flash/mem-efficient kernel);
          - (B, 1, L, L) bool: dense structural mask (SDPA, quadratic memory);
          - (b_idx, q_idx, k_idx) LongTensor triple from
            :func:`build_spacetime_edge_index`: the SAME structural mask as an
            edge list, evaluated with gather/scatter so memory is linear in the
            number of allowed pairs instead of (T*N)^2. Numerically identical
            to the dense path (one softmax per query over the same key set).
        """
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (rearrange(t, "b l (h d) -> b h l d", h=self.heads) for t in qkv)
        if isinstance(attn_mask, tuple):
            p_drop = self.attn_drop.p if self.training else 0.0
            if torch.is_grad_enabled() and x.requires_grad:
                out = checkpoint(
                    sparse_attention, q, k, v, *attn_mask,
                    scale=self.scale, dropout_p=p_drop, use_reentrant=False,
                )
            else:
                out = sparse_attention(q, k, v, *attn_mask,
                                       scale=self.scale, dropout_p=p_drop)
        else:
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
        out = rearrange(out, "b h l d -> b l (h d)")
        return self.resid_drop(self.to_out(out))


def sparse_attention(q, k, v, b_idx, q_idx, k_idx, scale, dropout_p=0.0):
    """Multi-head attention restricted to an explicit list of (query, key)
    token pairs.

    q, k, v : (B, H, L, d)
    b_idx, q_idx, k_idx : (P,) long -- pair p lets token (b, q_idx) attend to
        token (b, k_idx). Every query token must appear in at least one pair
        (the spacetime builder guarantees the diagonal), otherwise its output
        row is zero.

    Softmax is taken per query over exactly its listed keys, so this equals
    ``F.scaled_dot_product_attention`` with the equivalent boolean mask. The
    per-query max used for numerical stability is a constant shift and is
    detached. Half-precision inputs (autocast) do the score/probability
    arithmetic in float32; float32/64 inputs keep their dtype.
    Returns (B, H, L, d).
    """
    B, H, L, d = q.shape
    acc = torch.float32 if q.dtype in (torch.float16, torch.bfloat16) else q.dtype
    qg = q[b_idx, :, q_idx]                               # (P, H, d)
    kg = k[b_idx, :, k_idx]
    vg = v[b_idx, :, k_idx].to(acc)
    s = (qg * kg).sum(-1).to(acc) * scale                 # (P, H)
    seg = b_idx * L + q_idx                               # (P,) query row id
    seg_h = seg.unsqueeze(1).expand(-1, H)
    m = torch.full((B * L, H), float("-inf"), device=s.device, dtype=s.dtype)
    m = m.scatter_reduce(0, seg_h, s.detach(), reduce="amax", include_self=True)
    m = torch.where(torch.isfinite(m), m, torch.zeros_like(m))   # rows w/o pairs
    e = torch.exp(s - m[seg])                             # (P, H)
    denom = torch.zeros(B * L, H, device=s.device, dtype=s.dtype)
    denom = denom.index_add(0, seg, e)
    p = e / denom[seg].clamp(min=torch.finfo(s.dtype).tiny)
    if dropout_p > 0.0:
        p = F.dropout(p, dropout_p, training=True)
    out = torch.zeros(B * L, H, d, device=q.device, dtype=s.dtype)
    out = out.index_add(0, seg, p.unsqueeze(-1) * vg)
    return out.to(q.dtype).view(B, L, H, d).permute(0, 2, 1, 3)


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
    return allow.unsqueeze(1) # Return a bool mask (True = attend)


def build_spacetime_edge_index(
    adj: torch.Tensor,
    active: torch.Tensor,
    hidden_edges: Optional[torch.Tensor] = None,
    allow_temporal_self: bool = True,
    allow_temporal_neighbors: bool = False,
) -> tuple:
    """Edge-list form of :func:`build_spacetime_mask`.

    Returns ``(b_idx, q_idx, k_idx)`` long tensors, one entry per allowed
    (query, key) token pair in the flattened (B, T*N) token space, with the
    SAME semantics as the dense builder (same-time visible neighbours, the
    same node at other times, optional cross-time neighbours, both endpoints
    active, and the full diagonal always allowed). The (T*N)^2 grid is never
    materialised: peak memory is a few tensors of size ``P`` (number of allowed
    pairs), which for sparse graphs is ~ sum_t nnz(adj_t) + T^2 * N.

    Candidate pairs from the different sources can overlap (e.g. adjacency
    self-loops vs. the diagonal), so they are de-duplicated -- a duplicated key
    would otherwise be counted twice in the softmax.
    """
    B, T, N, _ = adj.shape
    device = adj.device
    L = T * N

    adj_vis = adj & ~hidden_edges if hidden_edges is not None else adj
    act_q = active.unsqueeze(-1)                      # (B, T, N, 1)
    act_k = active.unsqueeze(-2)                      # (B, T, 1, N)

    bs, qs, ks = [], [], []

    # 1) same timestep, visible edge, both active.
    for bi in range(B):
        t, i, j = (adj_vis[bi] & act_q[bi] & act_k[bi]).nonzero(as_tuple=True)
        bs.append(torch.full_like(t, bi)); qs.append(t * N + i); ks.append(t * N + j)

    # 2) same node, different timesteps, both active
    if allow_temporal_self:
        both = active.unsqueeze(2) & active.unsqueeze(1)          # (B, T, T, N)
        both = both & ~torch.eye(T, dtype=torch.bool, device=device).view(1, T, T, 1)
        b, t1, t2, i = both.nonzero(as_tuple=True)
        bs.append(b); qs.append(t1 * N + i); ks.append(t2 * N + i)

    # 3) neighbours at time t, attended at every other time t'
    if allow_temporal_neighbors:
        parts = [adj_vis[bi].nonzero(as_tuple=True) for bi in range(B)]
        b = torch.cat([torch.full_like(p[0], bi) for bi, p in enumerate(parts)])
        t, i, j = (torch.cat([p[k] for p in parts]) for k in range(3))
        P0 = b.numel()
        t2 = torch.arange(T, device=device).repeat(P0)              # (P0*T,)
        b = b.repeat_interleave(T); t = t.repeat_interleave(T)
        i = i.repeat_interleave(T); j = j.repeat_interleave(T)
        keep = (t2 != t) & active[b, t, i] & active[b, t2, j]
        bs.append(b[keep]); qs.append((t * N + i)[keep]); ks.append((t2 * N + j)[keep])

    # 4) full diagonal (every token, active or not, sees itself)
    diag = torch.arange(L, device=device)
    bs.append(torch.arange(B, device=device).repeat_interleave(L))
    qs.append(diag.repeat(B)); ks.append(diag.repeat(B))

    b_idx = torch.cat(bs); q_idx = torch.cat(qs); k_idx = torch.cat(ks)
    lin = (b_idx * L + q_idx) * L + k_idx
    lin = torch.unique(lin)
    k_idx = lin % L
    rest = lin // L
    q_idx = rest % L
    b_idx = rest // L
    return b_idx, q_idx, k_idx


class ContinuousTimeEmbedding(nn.Module):
    def __init__(self, dim, num_freqs=8, max_period=10000.0):
        super().__init__()
        freqs = torch.exp(torch.linspace(0.0, -np.log(max_period), num_freqs))
        self.register_buffer("freqs", freqs)
        self.proj = nn.Linear(2 * num_freqs, dim)

    def forward(self, t):
        x = t.unsqueeze(-1) * self.freqs
        return self.proj(torch.cat([x.sin(), x.cos()], dim=-1))


class DELOSPredictor(nn.Module):
    """DELOS spatio-temporal predictor over (timestep, node) tokens.

    Configurable masking scheme:
      - ``node_mask`` (default): hide ``num_masked_nodes`` whole nodes across
        history AND mask the future — the model reconstructs masked-history and
        forecasts future latents.
      - ``future_forecast``: mask only the future (no node hiding); a pure
        causal forecaster. Equivalent to ``node_mask`` with 0 masked nodes.

    The trajectory-block scheme is served by :class:`TrajectoryJEPAPredictor`
    (different token granularity); use :func:`build_predictor` to pick.
    """

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
        mask_scheme: str = "node_mask",
        residual_head: bool = False,
        latent_norm: bool = False,
        attn_impl: str = "sparse",
        mask_frac: float = 0.0,
        input_delta: bool = False,
        linear_skip: bool = False,
        hier_cfg: Optional[dict] = None,
        hier_hint: str = "forecast",
    ):
        super().__init__()
        if attn_impl not in ("sparse", "dense"):
            raise ValueError(f"attn_impl must be 'sparse' or 'dense', got {attn_impl!r}")
        self.attn_impl = attn_impl
        if mask_scheme not in ("node_mask", "future_forecast"):
            raise ValueError(
                f"DELOSPredictor mask_scheme must be 'node_mask' or "
                f"'future_forecast', got {mask_scheme!r}. For "
                f"'trajectory_block' use TrajectoryJEPAPredictor / build_predictor.")
        self.mask_scheme = mask_scheme
        if mask_scheme == "future_forecast":
            num_masked_nodes = 0
            mask_frac = 0.0
        self.num_nodes = num_nodes
        self.node_dim = node_dim
        self.history_frames = history_frames
        self.pred_frames = pred_frames
        self.total_frames = history_frames + pred_frames
        self.num_masked_nodes = num_masked_nodes
        if not 0.0 <= mask_frac < 1.0:
            raise ValueError(f"mask_frac must be in [0, 1), got {mask_frac}")
        self.mask_frac = mask_frac
        self.seed = seed
        self.allow_temporal_neighbors = allow_temporal_neighbors
        self.use_attention_mask = use_attention_mask
        self.latent_norm = latent_norm
        self.input_delta = input_delta
        self.delta_proj = None
        if input_delta:
            self.delta_proj = nn.Linear(node_dim, node_dim, bias=False)
            nn.init.zeros_(self.delta_proj.weight)

        self.linear_skip = linear_skip
        self.skip = None
        if linear_skip:
            self.skip = nn.Linear(history_frames * node_dim, pred_frames * node_dim)
            nn.init.zeros_(self.skip.weight)
            nn.init.zeros_(self.skip.bias)

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

        self.residual_head = residual_head
        if residual_head:
            nn.init.zeros_(self.to_out.weight)
            nn.init.zeros_(self.to_out.bias)

        self.edge_head = None
        if use_edge_head:
            self.edge_head = nn.Sequential(
                nn.Linear(2 * node_dim, node_dim), nn.GELU(),
                nn.Linear(node_dim, 1),
            )

        self.hint = None
        self.cond_proj = None
        if hier_cfg:
            from src.world_models.hier import CoarseHint
            rng_state = torch.get_rng_state()
            self.hint = CoarseHint(hier_cfg, num_nodes, hint=hier_hint)
            if self.hint.pred * self.hint.w != pred_frames \
                    or self.hint.hist * self.hint.w != history_frames:
                raise ValueError(
                    f"coarse model covers {self.hint.hist}x{self.hint.w} history / "
                    f"{self.hint.pred}x{self.hint.w} future fine steps, the fine "
                    f"predictor has {history_frames} / {pred_frames}")
            self.cond_proj = nn.Linear(self.hint.out_dim, node_dim, bias=False)
            nn.init.zeros_(self.cond_proj.weight)
            torch.set_rng_state(rng_state)

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
        empty = torch.tensor([], dtype=torch.long, device=device)
        if self.num_masked_nodes <= 0 and self.mask_frac <= 0:
            return empty
        valid = active_at_0.all(dim=0).nonzero(as_tuple=True)[0]
        if self.mask_frac > 0:
            # Fraction of the maskable pool; never falls back to inactive nodes.
            k = int(round(self.mask_frac * valid.numel()))
            if k <= 0:
                return empty
            pool = valid
        else:
            k = self.num_masked_nodes
            pool = valid if valid.numel() >= k \
                else torch.arange(self.num_nodes, device=device)
        if self.training:
            perm = torch.randperm(pool.numel(), device=device)[:k]
            return pool[perm]
        rng = np.random.RandomState(self.seed)
        idx = rng.choice(pool.cpu().numpy(), k, replace=False)
        return torch.from_numpy(idx).to(device)

    def make_cond(self, x_hist, adj, active, x_fut=None, masked_idx=None):
        """Top-down hint for one batch, or None for a flat predictor.

        Takes the RAW history features (the coarse model has its own
        encoder), so it is computed by the caller, which holds them, and
        passed to forward()/inference() as ``cond``. ``masked_idx`` are the
        nodes this forward pass will hide (pass the same indices as
        ``forced_masked_idx``); omit at inference."""
        if self.hint is None:
            return None
        self.hint._owner_training = self.training
        return self.hint(x_hist, adj, active, x_fut=x_fut, masked_idx=masked_idx)

    def prepare_input(self, x, active, times=None, forced_masked_idx=None,
                      cond=None):
        B, T_hist, N, D = x.shape
        T_total = self.total_frames
        device = x.device

        if forced_masked_idx is not None:
            masked_idx = forced_masked_idx.to(device).long()
        else:
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

        x_tok = x
        if self.delta_proj is not None and T_hist > 1:
            both = (active[:, 1:T_hist] & active[:, :T_hist - 1]).unsqueeze(-1)
            delta = torch.zeros_like(x)
            delta[:, 1:T_hist] = torch.where(
                both, x[:, 1:T_hist] - x[:, :T_hist - 1], torch.zeros_like(x[:, 1:T_hist]))
            x_tok = x + self.delta_proj(delta)

        # t=0 visible for every active node
        vis0 = active[:, 0:1].unsqueeze(-1)
        out = out.clone()
        out[:, 0:1] = torch.where(vis0, x_tok[:, 0:1] + time_emb[:, 0:1], out[:, 0:1])

        # t in [1, T_hist) visible for active & unmasked nodes
        if T_hist > 1:
            unmasked = (~is_masked).view(1, 1, N, 1)
            vis = active[:, 1:T_hist].unsqueeze(-1) & unmasked
            out[:, 1:T_hist] = torch.where(
                vis, x_tok[:, 1:T_hist] + time_emb[:, 1:T_hist], out[:, 1:T_hist]
            )

        if self.cond_proj is not None:
            # Future query tokens: + projected coarse hint. Hidden nodes get
            # none: the coarse model saw their history, and the hint would
            # hand it to the masked-history reconstruction.
            c = self.cond_proj(cond)                              # (B, P, N, D)
            keep = (~is_masked).view(1, 1, N, 1).to(c.dtype)
            out[:, T_hist:] = out[:, T_hist:] + c * keep

        return out, masked_idx

    def forward(self, x, adj, active, times=None, hidden_edges=None,
                forced_masked_idx=None, cond=None):
        B, T_hist, N, D = x.shape
        T_total = self.total_frames
        if self.latent_norm:
            x = F.layer_norm(x, (D,))
        if self.cond_proj is not None and cond is None:
            raise ValueError(
                "this predictor was trained with a coarse hint (--hier_ckpt); "
                "pass cond=predictor.make_cond(x_hist, adj, active, ...)")

        x_input, masked_idx = self.prepare_input(
            x, active, times, forced_masked_idx=forced_masked_idx, cond=cond,
        )
        x_flat = rearrange(x_input, "b t n d -> b (t n) d")

        if self.use_attention_mask:
            builder = (build_spacetime_edge_index if self.attn_impl == "sparse"
                       else build_spacetime_mask)
            attn_mask = builder(
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

        if self.residual_head:
            # Anchor = last latent the predictor was allowed to see: the final
            # history step for visible nodes, t=0 for masked nodes.
            anchor = x[:, -1]
            if masked_idx.numel() > 0:
                anchor = anchor.clone()
                anchor[:, masked_idx] = x[:, 0][:, masked_idx]
            pred = pred + anchor.unsqueeze(1)

        if self.skip is not None:
            H, P = self.history_frames, self.pred_frames
            B_, _, N_, D_ = x.shape
            hist = x[:, :H].permute(0, 2, 1, 3).reshape(B_, N_, H * D_)      # (B,N,H*D)
            disp = self.skip(hist).view(B_, N_, P, D_).permute(0, 2, 1, 3)   # (B,P,N,D)
            if masked_idx.numel() > 0:
                keep = torch.ones(N_, device=x.device, dtype=disp.dtype)
                keep[masked_idx] = 0.0
                disp = disp * keep.view(1, 1, N_, 1)
            pred = torch.cat([pred[:, :H], pred[:, H:] + disp], dim=1)

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
    def inference(self, x, adj, active, times=None, intervention_idx=None,
                  cond=None):
        """Predict future-step latents with no nodes masked (observational
        forecast). Pass `intervention_idx` (a 1-D LongTensor of node indices)
        to replace those nodes' trajectories past t=0 with the learned mask
        token, a do(.) intervention. Predictions are returned for all nodes,
        including the intervened ones.

        Future graph structure is withheld: adjacency and activity at forecast
        steps (t >= history_frames) are overwritten with the last history
        step's values, so the predictor can never peek at future edges
        regardless of the --no_attention_mask flag. Known future timestamps
        (`times`) are left intact."""
        H = self.history_frames
        if adj is not None and adj.shape[1] > H:
            adj = adj.clone()
            adj[:, H:] = adj[:, H - 1:H]          # carry last-seen edges forward
        if active is not None and active.shape[1] > H:
            active = active.clone()
            active[:, H:] = active[:, H - 1:H]     # carry last-seen activity forward
        saved, saved_frac = self.num_masked_nodes, self.mask_frac
        if intervention_idx is None:
            self.num_masked_nodes = 0
            self.mask_frac = 0.0
        try:
            pred, _, _ = self.forward(
                x, adj, active, times=times,
                forced_masked_idx=intervention_idx, cond=cond,
            )
        finally:
            self.num_masked_nodes, self.mask_frac = saved, saved_frac
        return pred[:, self.history_frames:]


def vicreg_terms(z: torch.Tensor, mask: torch.Tensor,
                 gamma: float = 1.0, eps: float = 1e-4) -> tuple:
    """VICReg variance + covariance regularizers on a node-latent tensor.

    The MSE-style invariance term is already in `compute_graph_jepa_loss`; this
    function provides the two complementary losses from VICReg (Bardes, Ponce
    & LeCun, 2022) needed to prevent collapse:

      - variance: per-dimension std should stay above gamma. Penalty:
            mean_d( relu(gamma - std(z_d)) )
        keeps no single dim from shrinking to a delta.
      - covariance: off-diagonal entries of cov(z) should be zero. Penalty:
            sum_{i != j} cov_ij^2 / D
        decorrelates dims so dim-count != effective-rank.

    Args:
      z:    (B, T, N, D) predictor (or encoder) output.
      mask: (B, T, N) bool — rows to include (active nodes at future steps).
      gamma: variance floor. 1.0 is the VICReg default.

    Returns (var_loss, cov_loss) as 0-d tensors. If fewer than 2 valid rows
    are available the function returns zeros (no signal possible).
    """
    rows = z[mask]
    if rows.shape[0] < 2:
        zero = z.sum() * 0.0
        return zero, zero
    std = torch.sqrt(rows.var(dim=0, unbiased=False) + eps)         # (D,)
    var_loss = F.relu(gamma - std).mean()
    rows_c = rows - rows.mean(dim=0, keepdim=True)
    cov = (rows_c.T @ rows_c) / max(rows.shape[0] - 1, 1)           # (D, D)
    D = cov.shape[0]
    off2 = cov.pow(2).sum() - cov.diagonal().pow(2).sum()
    cov_loss = off2 / D
    return var_loss, cov_loss


def sigreg_loss(rows: torch.Tensor, num_slices: int = 256, knots: int = 17,
                t_max: float = 5.0, max_rows: int = 8192,
                generator: "torch.Generator | None" = None) -> torch.Tensor:
    """SIGReg (Sketched Isotropic Gaussian Regularisation) from LeJEPA
    (Balestriero & LeCun, 2025).

    Pushes the empirical distribution of ``rows`` toward an isotropic
    standard Gaussian, which rules out every collapse mode (total,
    dimensional, and the anisotropic ones VICReg's var/cov terms miss).
    The test is one-dimensional: project onto ``num_slices`` random unit
    directions and, per direction, compute the Epps-Pulley statistic

        int |phi_n(t) - exp(-t^2/2)|^2 exp(-t^2/2) dt

    between the empirical characteristic function phi_n and that of N(0, 1),
    on a trapezoid grid of ``knots`` points over [0, t_max] (the integrand is
    even, so the negative half is a constant factor). The target is the
    standard normal, so the scale is pinned and no separate variance floor
    is needed.

    Rows are sub-sampled to ``max_rows`` and slices are chunked to bound the
    (n, M, K) cos/sin tensor. The classical n-factor on the statistic is
    dropped: the caller's weight absorbs it and this keeps the loss
    batch-size-agnostic.

    Args:
      rows: (n, D) embeddings (predictor outputs and/or encoder latents).
    Returns a 0-d tensor (mean statistic over slices).
    """
    n, D = rows.shape
    if n < 2:
        return rows.sum() * 0.0
    if n > max_rows:
        sel = torch.randperm(n, device=rows.device, generator=generator)[:max_rows]
        rows = rows[sel]
        n = max_rows
    A = torch.randn(D, num_slices, device=rows.device, dtype=rows.dtype,
                    generator=generator)
    A = A / A.norm(dim=0, keepdim=True).clamp(min=1e-6)
    t = torch.linspace(0.0, t_max, knots, device=rows.device, dtype=rows.dtype)
    w = torch.exp(-0.5 * t * t)                       # = phi_N(0,1)(t) = weight
    dt = t_max / max(knots - 1, 1)
    chunk = max(1, (1 << 24) // (n * knots))          # slices per chunk
    stats = []
    for s0 in range(0, num_slices, chunk):
        proj = rows @ A[:, s0:s0 + chunk]              # (n, m)
        tx = proj.unsqueeze(-1) * t                    # (n, m, K)
        re = torch.cos(tx).mean(0)                     # (m, K)
        im = torch.sin(tx).mean(0)
        err = (re - w).pow(2) + im.pow(2)
        stats.append((err * w).sum(-1) * dt)           # (m,)
    return torch.cat(stats).mean()


class TrajectoryJEPAPredictor(nn.Module):
    """V-JEPA-style predictor over trajectory-window tokens.

    Differs from DELOSPredictor in three ways:
      1. Token granularity is (window, node), not (timestep, node). The
         caller passes already-pooled (B, K, N, D) tokens from
         `TrajectoryWindowPooler`; this module just predicts.
      2. Mask is a (K, N) bool grid (positions), not a node-index set.
         No past/future split -- masking is bidirectional, V-JEPA style.
      3. No anchor-query / new-node-token machinery. The pooled tokens
         already carry node identity via the encoder; we add learned
         window-position and node-position embeddings on top.

    The transformer backbone (GraphSpacetimeTransformer) is shared with
    DELOSPredictor.
    """

    def __init__(self, num_windows: int, num_nodes: int, node_dim: int,
                 depth: int = 4, heads: int = 4, dim_head: int = 32,
                 mlp_dim: int = 512, dropout: float = 0.1,
                 node_pos_mode: str = "off"):
        """
        node_pos_mode:
          - "off" (default): no per-node positional embedding; node identity
            flows through the pooled tokens themselves.
          - "learned": free (N, D) lookup table. It lets the model memorise
            per-node attributes (e.g. the block ID in an SBM) during training,
            which contaminates downstream probes.
          - "from_input": project the encoder's first-window pooled token
            through a linear layer to produce the per-node anchor, as
            DELOSPredictor does.
        """
        super().__init__()
        if node_pos_mode not in ("off", "learned", "from_input"):
            raise ValueError(f"unknown node_pos_mode {node_pos_mode!r}")
        self.num_windows = num_windows
        self.num_nodes = num_nodes
        self.node_dim = node_dim
        self.node_pos_mode = node_pos_mode

        # Single learned mask token; broadcast at masked positions.
        self.mask_token = nn.Parameter(torch.zeros(node_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        # Window-position embedding: (1, K, 1, D). 
        self.window_pos = nn.Parameter(
            0.02 * torch.randn(1, num_windows, 1, node_dim)
        )

        # Node-position embedding: configurable to avoid the memorisation
        # confound (see node_pos_mode docstring above).
        if node_pos_mode == "learned":
            self.node_pos = nn.Parameter(
                0.02 * torch.randn(1, 1, num_nodes, node_dim)
            )
        elif node_pos_mode == "from_input":
            self.id_projector = nn.Linear(node_dim, node_dim)

        self.transformer = GraphSpacetimeTransformer(
            dim=node_dim, depth=depth, heads=heads,
            dim_head=dim_head, mlp_dim=mlp_dim, dropout=dropout,
        )
        self.to_out = nn.Linear(node_dim, node_dim)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor):
        """
        tokens: (B, K, N, D) -- output of TrajectoryWindowPooler. Caller
                must ensure K <= self.num_windows and N <= self.num_nodes.
        mask:   (K, N) bool -- True at masked positions (V-JEPA style,
                shared across the batch).

        Returns:
          pred:   (B, K, N, D) predicted tokens. Loss should be computed
                  at masked positions; unmasked positions are not
                  supervised but are output for completeness.
        """
        if tokens.dim() != 4:
            raise ValueError(
                f"expected tokens (B, K, N, D); got shape {tuple(tokens.shape)}"
            )
        B, K, N, D = tokens.shape
        if mask.shape != (K, N):
            raise ValueError(
                f"mask shape {tuple(mask.shape)} != ({K}, {N})"
            )

        # Replace masked positions with the learned mask token.
        mask_b = mask.view(1, K, N, 1).expand(B, K, N, D)
        mtok = self.mask_token.view(1, 1, 1, D).expand(B, K, N, D)
        x = torch.where(mask_b, mtok, tokens)

        x = x + self.window_pos[:, :K]

        # Node position: only if configured. Node identity comes from the
        # tokens themselves in 'off' mode; from a projection of the first
        # visible window's token in 'from_input' mode.
        if self.node_pos_mode == "learned":
            x = x + self.node_pos[:, :, :N]
        elif self.node_pos_mode == "from_input":
            anchor = self.id_projector(tokens[:, 0])             # (B, N, D)
            x = x + anchor.unsqueeze(1)                          # broadcast over K

        # Spacetime transformer over the flattened (K*N) token sequence, with
        # no attention mask: every token attends to every other.
        x_flat = rearrange(x, "b k n d -> b (k n) d")
        h_flat = self.transformer(x_flat, attn_mask=None)
        h = rearrange(h_flat, "b (k n) d -> b k n d", k=K, n=N)
        return self.to_out(h)


# Backward-compatible aliases for the predictor's earlier names.
MaskedNodePredictor = DELOSPredictor
DyGJEPAPredictor = DELOSPredictor


class PlainForecaster(nn.Module):
    """Graph-free, mask-free temporal forecaster: the "generic predictor"
    control for the masked spacetime transformer.

    It maps each node's history latents to its future latents independently
    (no cross-node attention, no adjacency, no masking objective) and is
    trained on the identical future-latent MSE loss (masked_idx is empty, so
    ``compute_graph_jepa_loss`` supervises only the future positions). Two
    backbones:
      - ``mlp`` : flatten the H history latents per node -> MLP -> P*D.
      - ``gru`` : GRU over the H history steps per node -> last hidden -> P*D.

    Capacity is set by ``mlp_dim`` (hidden width) and ``depth`` (layers),
    the same CLI flags as the transformer, so parameter counts can be
    matched. ``residual_head`` makes the forecast start at copy-last, as in
    DELOSPredictor.

    Drop-in for DELOSPredictor: same ``forward`` / ``inference`` signatures,
    returns ``(pred, empty_masked_idx, None)``. ``adj`` / ``active`` are
    accepted for interface parity and deliberately ignored, so future graph
    structure can never leak.
    """

    def __init__(self, num_nodes, node_dim, history_frames, pred_frames=1,
                 kind="mlp", depth=2, mlp_dim=512, dropout=0.1,
                 residual_head=False, **_ignored):
        super().__init__()
        if kind not in ("mlp", "gru"):
            raise ValueError(f"PlainForecaster kind must be 'mlp' or 'gru', "
                             f"got {kind!r}")
        self.num_nodes = num_nodes
        self.node_dim = node_dim
        self.history_frames = history_frames
        self.pred_frames = pred_frames
        self.total_frames = history_frames + pred_frames
        self.num_masked_nodes = 0            # interface parity (never used)
        self.kind = kind
        self.residual_head = residual_head
        D, H, P = node_dim, history_frames, pred_frames
        n_layers = max(1, depth)

        if kind == "gru":
            self.gru = nn.GRU(input_size=D, hidden_size=mlp_dim,
                              num_layers=n_layers, batch_first=True,
                              dropout=dropout if n_layers > 1 else 0.0)
        else:
            dims = [H * D] + [mlp_dim] * n_layers
            layers = []
            for a, b in zip(dims[:-1], dims[1:]):
                layers += [nn.Linear(a, b), nn.GELU(), nn.Dropout(dropout)]
            self.body = nn.Sequential(*layers)
        self.to_out = nn.Linear(mlp_dim, P * D)

        # Residual parameterisation parity with DELOSPredictor: zero-init the
        # head so the forecast STARTS at copy-last (anchor = last history
        # latent) and can only add dynamics on top. 
        if residual_head:
            nn.init.zeros_(self.to_out.weight)
            nn.init.zeros_(self.to_out.bias)

    def _forecast(self, x):
        """x: (B, H, N, D) -> future latents (B, P, N, D). Per-node only."""
        B, H, N, D = x.shape
        P = self.pred_frames
        xn = x.permute(0, 2, 1, 3).reshape(B * N, H, D)        # (B*N, H, D)
        if self.kind == "gru":
            out, _ = self.gru(xn)                              # (B*N, H, hid)
            h = out[:, -1]
        else:
            h = self.body(xn.reshape(B * N, H * D))            # (B*N, hid)
        fut = self.to_out(h).view(B, N, P, D).permute(0, 2, 1, 3)   # (B,P,N,D)
        if self.residual_head:
            fut = fut + x[:, -1].unsqueeze(1)                  # anchor copy-last
        return fut

    def forward(self, x, adj=None, active=None, times=None,
                hidden_edges=None, forced_masked_idx=None):
        fut = self._forecast(x)                                # (B, P, N, D)
        pred = torch.cat([x, fut], dim=1)                      # (B, H+P, N, D)
        empty = torch.empty(0, dtype=torch.long, device=x.device)
        return pred, empty, None                               # no masking, no edges

    @torch.no_grad()
    def inference(self, x, adj=None, active=None, times=None,
                  intervention_idx=None):
        return self._forecast(x)                               # (B, P, N, D)


def predictor_flags_from_args(saved: "dict | None") -> dict:
    """Predictor constructor flags that a checkpoint's weights cannot recover.

    Every script that REBUILDS a trained predictor (state probe, analysis,
    fine-tuning) must pass these, read from the training ``args`` stored in
    the checkpoint:

    * ``linear_skip`` adds the ``skip`` weights (linear-dynamics prior).
    * ``input_delta`` adds the ``delta_proj`` weights (velocity channel);
      omitting it makes ``load_state_dict`` fail on an ``--input_delta`` ckpt.
    * ``residual_head`` / ``latent_norm`` have NO parameters, so a mismatch is
      silent: ``load_state_dict`` succeeds and the predictor then runs a
      different function from the one that was trained (e.g. a ``--target_norm``
      model probed without its input LayerNorm).
    * ``use_edge_head`` adds the ``edge_head.*`` weights; omitting it makes
      ``load_state_dict`` fail on any ``--edge_mask_rate > 0`` checkpoint.
    * ``attn_impl`` only changes how the same attention is computed.
    * ``mask_frac`` is parameter-free too; inference() never masks, but the
      validation loss and the masked analyses do.

    ``latent_norm`` follows ``--target_norm`` exactly as the trainer sets it
    (``train_delos.build_jepa_predictor``). PlainForecaster ignores the keys
    it does not use, so the result is safe for every ``mask_scheme``.
    """
    saved = saved or {}
    hier = {}
    if saved.get("hier_cfg"):
        # Two-level run: the frozen coarse model is a submodule of the
        # predictor, so it must be rebuilt before load_state_dict.
        hier = dict(hier_cfg=saved["hier_cfg"],
                    hier_hint=saved.get("hier_hint", "forecast"))
    return dict(
        **hier,
        residual_head=bool(saved.get("residual_head", False)),
        input_delta=bool(saved.get("input_delta", False)),
        linear_skip=bool(saved.get("linear_skip", False)),
        latent_norm=bool(saved.get("target_norm", False)),
        use_edge_head=float(saved.get("edge_mask_rate", 0.0) or 0.0) > 0.0,
        attn_impl=saved.get("attn_impl", "sparse"),
        mask_frac=float(saved.get("mask_frac", 0.0) or 0.0),
    )


def build_predictor(mask_scheme: str, **cfg):
    """Construct the DELOS predictor for a masking scheme.

    - ``node_mask`` / ``future_forecast`` -> :class:`DELOSPredictor`
      (expects cfg: num_nodes, node_dim, history_frames, pred_frames, ...).
    - ``mlp_forecast`` / ``gru_forecast``  -> :class:`PlainForecaster`
      (the graph-free ablation control; transformer-only cfg keys such
      as heads/dim_head/use_attention_mask are accepted and ignored).
    - ``trajectory_block``                -> :class:`TrajectoryJEPAPredictor`
      (expects cfg: num_windows, num_nodes, node_dim, ...).
    """
    if mask_scheme in ("node_mask", "future_forecast"):
        return DELOSPredictor(mask_scheme=mask_scheme, **cfg)
    if mask_scheme in ("mlp_forecast", "gru_forecast"):
        kind = "mlp" if mask_scheme == "mlp_forecast" else "gru"
        return PlainForecaster(kind=kind, **cfg)
    if mask_scheme == "trajectory_block":
        return TrajectoryJEPAPredictor(**cfg)
    raise ValueError(
        f"unknown mask_scheme {mask_scheme!r}; expected node_mask, "
        f"future_forecast, mlp_forecast, gru_forecast, or trajectory_block.")


def compute_trajectory_jepa_loss(pred: torch.Tensor, target: torch.Tensor,
                                 mask: torch.Tensor) -> torch.Tensor:
    """MSE in latent space, restricted to masked (predicted) positions.

    pred, target: (B, K, N, D)
    mask:         (K, N) bool, True at masked positions to supervise.

    Returns a 0-d tensor. Returns the all-zero positions' MSE if the
    mask is empty, to avoid NaN on degenerate batches.
    """
    B, K, N, D = pred.shape
    diff2 = (pred - target).pow(2).mean(dim=-1)                  # (B, K, N)
    mask_b = mask.view(1, K, N).expand(B, K, N).float()
    n_masked = mask_b.sum().clamp(min=1.0)
    return (diff2 * mask_b).sum() / n_masked


def compute_graph_jepa_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    active: torch.Tensor,
    masked_idx: torch.Tensor,
    T_hist: int,
    edge_out=None,
    edge_labels: Optional[torch.Tensor] = None,
    edge_weight: float = 1.0,
    vicreg_var_weight: float = 0.0,
    vicreg_cov_weight: float = 0.0,
    vicreg_gamma: float = 1.0,
    change_weight: float = 0.0,
    target_norm: bool = False,
    sigreg_weight: float = 0.0,
    sigreg_slices: int = 256,
    sigreg_knots: int = 17,
    sigreg_extra_rows: "list[torch.Tensor] | None" = None,
    horizon_norm: bool = False,
):
    """Latent-MSE JEPA loss (masked-history + future) with optional extras.

    horizon_norm divides every FUTURE step's squared error by the copy-last
    squared error at that step (mean over the batch's active future tokens of
    ||z_{T_hist-1} - z_t||^2 on the TARGET latents, no gradient). Without it
    the summed future loss is dominated by the longest horizons purely
    through their variance, most of which is irreducible, so the MSE-optimal
    predictor shrinks its update toward the mean at long horizons. With it
    every horizon contributes on the same scale and the future loss reads as
    error RELATIVE TO PERSISTENCE: 1.0 = a zero update, < 1 = better than
    copy-last. Masked-history terms are unchanged.

    target_norm applies a parameter-free LayerNorm over the channel dim of
    the target (I-JEPA / V-JEPA) so no node/dim with a large latent norm can
    dominate the average and the loss scale stays fixed as an EMA target
    drifts. The predictor must be built with ``latent_norm=True`` so its input
    (and residual anchor) live in the same space.

    sigreg_weight > 0 adds LeJEPA's SIGReg on the predictor output at the
    supervised (masked-history + future) rows, averaged with any
    ``sigreg_extra_rows`` (e.g. the online encoder's latents when it is
    being trained). See :func:`sigreg_loss`.

    change_weight > 0 turns on change-weighted MSE: each (b, t, n) term is
    weighted by 1 + change_weight * ||z_t - z_{t-1}|| / mean(||.||), computed
    on the TARGET latents. Transitions are rare, so the unweighted average is
    dominated by static nodes — for which copying is optimal — and the
    predictor never trains on the rows that matter. Weighted means keep the
    loss scale comparable across change_weight settings.
    """
    B, T_total, N, D = pred.shape
    if target_norm:
        target = F.layer_norm(target, (D,))
    diff2 = (pred - target).pow(2).mean(dim=-1)

    node_sel = torch.zeros(N, dtype=torch.bool, device=pred.device)
    node_sel[masked_idx] = True

    mh = torch.zeros_like(active)
    mh[:, 1:T_hist] = True
    mh = mh & active & node_sel.view(1, 1, N)
    fut = torch.zeros_like(active)
    fut[:, T_hist:] = True
    fut = fut & active

    if horizon_norm and T_total > T_hist:
        with torch.no_grad():
            last = target[:, T_hist - 1:T_hist]                          # (B,1,N,D)
            cl = (last - target[:, T_hist:]).pow(2).mean(dim=-1)         # (B,T_pred,N)
            f = fut[:, T_hist:].float()
            scale = (cl * f).sum(dim=(0, 2)) / f.sum(dim=(0, 2)).clamp(min=1.0)
            scale = scale.clamp(min=1e-6).view(1, -1, 1)                 # (1,T_pred,1)
        diff2 = torch.cat([diff2[:, :T_hist], diff2[:, T_hist:] / scale], dim=1)

    if change_weight > 0:
        with torch.no_grad():
            delta = (target[:, 1:] - target[:, :-1]).pow(2).mean(dim=-1).sqrt()
            delta = torch.cat([torch.zeros_like(delta[:, :1]), delta], dim=1)
            sup = mh | fut
            norm = (delta[sup].mean() if sup.any()
                    else delta.mean()).clamp(min=1e-6)
            w = 1.0 + change_weight * delta / norm                 # (B, T, N)
        loss_mh = (diff2 * w * mh.float()).sum() \
            / (w * mh.float()).sum().clamp(min=1e-6)
        loss_fut = (diff2 * w * fut.float()).sum() \
            / (w * fut.float()).sum().clamp(min=1e-6)
    else:
        loss_mh = (diff2 * mh.float()).sum() / mh.sum().clamp(min=1)
        loss_fut = (diff2 * fut.float()).sum() / fut.sum().clamp(min=1)
    losses = {"loss_masked_history": loss_mh, "loss_future": loss_fut}
    total = loss_mh + loss_fut

    if edge_out is not None and edge_labels is not None:
        _, logits = edge_out
        loss_edge = F.binary_cross_entropy_with_logits(logits, edge_labels.float())
        losses["loss_edge"] = loss_edge
        total = total + edge_weight * loss_edge

    # VICReg on predictor output at future + masked-history positions: that's
    # where the predictor is forced to actually populate latents (elsewhere it
    # just echoes visible inputs).
    if vicreg_var_weight > 0 or vicreg_cov_weight > 0:
        reg_mask = mh | fut
        var_loss, cov_loss = vicreg_terms(pred, reg_mask, gamma=vicreg_gamma)
        losses["loss_vicreg_var"] = var_loss
        losses["loss_vicreg_cov"] = cov_loss
        total = total + vicreg_var_weight * var_loss + vicreg_cov_weight * cov_loss

    if sigreg_weight > 0:
        groups = [pred[mh | fut]] + list(sigreg_extra_rows or [])
        terms = [sigreg_loss(g, num_slices=sigreg_slices, knots=sigreg_knots)
                 for g in groups if g.shape[0] >= 2]
        sig = torch.stack(terms).mean() if terms else pred.sum() * 0.0
        losses["loss_sigreg"] = sig
        total = total + sigreg_weight * sig

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
