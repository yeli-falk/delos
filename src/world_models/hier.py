"""Two-level (coarse -> fine) DELOS: coarse time scale + top-down hint.

Stage 1 trains an ordinary DELOS on a COARSE view of every window
(``--coarse_window w``): the fine window of ``w * (H + P)`` snapshots is
reduced to ``H + P`` coarse steps by :func:`coarsen_window`. Stage 2 trains
the ordinary fine DELOS with that coarse model frozen inside the predictor
(``--hier_ckpt``): its forecast for the coarse window containing fine step
``t`` is added to the future query token of step ``t`` (see
``DyGJEPAPredictor.prepare_input``).

Both stages coarsen with the SAME function over the SAME fine windows, so the
coarse model meets at stage 2 exactly the input distribution it was trained
on (all ``w`` phase offsets included), and its history never reaches past the
last observed fine bin.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

COARSE_MODES = ("last", "mean")
HINT_KINDS = ("forecast", "true", "shuffle")

# Keys of the coarse run's training args that are needed to rebuild it.
_CFG_KEYS = (
    "data_source", "encoder_kind", "encoder_hidden", "node_dim", "bidirectional",
    "mask_scheme", "history_size", "num_preds", "num_masked_nodes", "depth",
    "heads", "mlp_dim", "dropout", "no_attention_mask",
    "allow_temporal_neighbors", "residual_head", "input_delta", "linear_skip",
    "target_norm", "edge_mask_rate", "attn_impl", "mask_frac",
    "coarse_window", "coarse_mode", "ema_target", "seed",
)


def coarsen_window(x, adj, active, w: int, mode: str):
    """Reduce a window of ``T = K * w`` fine snapshots to ``K`` coarse steps.

    x (..., T, N, F), adj (..., T, N, N) bool, active (..., T, N) bool; any
    leading dims (batched or not).

    * ``last``: keep the LAST snapshot of every block of ``w`` (subsampling).
      The coarse state is the fine state at the end of the block, so a coarse
      forecast is a waypoint for the fine predictor.
    * ``mean``: re-bin. Features are averaged over the steps where the node is
      active, edges are the union, a node is active if it is active anywhere
      in the block.
    """
    if mode not in COARSE_MODES:
        raise ValueError(f"coarse mode must be one of {COARSE_MODES}, got {mode!r}")
    T = x.shape[-3]
    if w < 1 or T % w != 0:
        raise ValueError(f"window of {T} snapshots is not divisible by coarse_window={w}")
    if w == 1:
        return x, adj, active
    if mode == "last":
        return (x[..., w - 1::w, :, :], adj[..., w - 1::w, :, :],
                active[..., w - 1::w, :])
    K = T // w
    xr = x.unflatten(-3, (K, w))                       # (..., K, w, N, F)
    ar = active.unflatten(-2, (K, w))                  # (..., K, w, N)
    wt = ar.unsqueeze(-1).to(x.dtype)
    x_c = (xr * wt).sum(dim=-3) / wt.sum(dim=-3).clamp(min=1.0)
    adj_c = adj.unflatten(-3, (K, w)).any(dim=-3)
    return x_c, adj_c, ar.any(dim=-2)


class CoarseWindowDataset(Dataset):
    """Coarse view of a fine window dataset (stage 1 training data).

    ``base`` must have been built with ``history_size * w`` / ``num_preds * w``
    fine snapshots. Items keep the canonical keys with coarse lengths; every
    other attribute (num_nodes, feature_dim, starts, ...) is the base's. The
    splits are the fine chronological splits, so nothing crosses them.

    The item's ``year`` stays the FINE start index and the label matrices are
    the fine ones, so the state probe must not be run on a coarse checkpoint.
    """

    def __init__(self, base, w: int, mode: str):
        if base.history_size % w or base.num_preds % w:
            raise ValueError("fine history / horizon not divisible by coarse_window")
        self.base = base
        self.coarse_window = w
        self.coarse_mode = mode
        self.history_size = base.history_size // w
        self.num_preds = base.num_preds // w
        self.window = self.history_size + self.num_preds

    def __getattr__(self, name):
        # Only reached for attributes not set on the wrapper itself. `base`
        # must not recurse while the object is being unpickled in a worker.
        if name == "base":
            raise AttributeError(name)
        return getattr(self.base, name)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = dict(self.base[idx])
        x = torch.cat([item["x_hist"], item["x_fut"]], dim=0)
        x_c, adj_c, act_c = coarsen_window(
            x, item["adj"], item["active"], self.coarse_window, self.coarse_mode)
        item["x_hist"] = x_c[: self.history_size]
        item["x_fut"] = x_c[self.history_size:]
        item["adj"] = adj_c
        item["active"] = act_c
        return item


def hier_cfg_from_ckpt(ckpt: dict, feature_dim: int) -> dict:
    """The slim, JSON-serialisable description of a coarse run that the fine
    run stores in its own args (``hier_cfg``), so every script that rebuilds
    the fine predictor can rebuild the coarse model inside it."""
    saved = ckpt.get("args") or {}
    w = int(saved.get("coarse_window", 1) or 1)
    if w <= 1:
        raise SystemExit("--hier_ckpt must point at a run trained with "
                         "--coarse_window > 1 (the coarse level)")
    if saved.get("hier_cfg"):
        raise SystemExit("--hier_ckpt is itself a hierarchical run; only two "
                         "levels are supported")
    cfg = {k: saved[k] for k in _CFG_KEYS if k in saved}
    cfg["feature_dim"] = int(feature_dim)
    cfg["has_target_encoder"] = "ema_encoder" in ckpt
    return cfg


class CoarseHint(nn.Module):
    """Frozen coarse DELOS (encoder + predictor) that turns the FINE window's
    raw history into a per-(future step, node) hint for the fine predictor.

    hint kinds
      forecast  the coarse model's forecast (the method).
      true      DIAGNOSTIC ceiling: the coarse target encoder's latent of the
                TRUE future. Uses future snapshots; never a results row.
      shuffle   DIAGNOSTIC capacity control: the coarse forecast with the node
                axis permuted, so each node receives another node's hint (the
                graph-level statistics of the hint survive, per-node
                information does not).

    The output is LayerNorm-ed (parameter-free) so its scale does not depend
    on how the coarse model was trained, and repeated ``w`` times along time:
    (B, P_fine, N, D_coarse).
    """

    def __init__(self, cfg: dict, num_nodes: int, hint: str = "forecast"):
        super().__init__()
        if hint not in HINT_KINDS:
            raise ValueError(f"hier hint must be one of {HINT_KINDS}, got {hint!r}")
        # Imported here: cjepa_graph_predictor builds this class lazily.
        from src.cjepa_graph_predictor import build_predictor, predictor_flags_from_args
        from src.world_models.graph_jepa_encoder import build_encoder

        self.kind = hint
        self.w = int(cfg["coarse_window"])
        self.mode = cfg.get("coarse_mode", "last")
        self.hist = int(cfg["history_size"])            # coarse steps
        self.pred = int(cfg["num_preds"])
        self.out_dim = int(cfg["node_dim"])
        self.target_norm = bool(cfg.get("target_norm", False))
        self.seed = int(cfg.get("seed", 0))

        def _enc():
            return build_encoder(
                cfg.get("encoder_kind", "tgcn"), in_dim=cfg["feature_dim"],
                hidden_dim=cfg.get("encoder_hidden", 128), out_dim=self.out_dim,
                freeze=True, bidirectional=cfg.get("bidirectional", True))

        self.encoder = _enc()
        self.target_encoder = _enc() if cfg.get("has_target_encoder") else None
        heads = cfg.get("heads", 4)
        self.predictor = build_predictor(
            cfg.get("mask_scheme", "node_mask"),
            num_nodes=num_nodes, node_dim=self.out_dim,
            history_frames=self.hist, pred_frames=self.pred,
            num_masked_nodes=cfg.get("num_masked_nodes", 30),
            depth=cfg.get("depth", 4), heads=heads, dim_head=self.out_dim // heads,
            mlp_dim=cfg.get("mlp_dim", 512), dropout=cfg.get("dropout", 0.1),
            allow_temporal_neighbors=cfg.get("allow_temporal_neighbors", False),
            use_attention_mask=not cfg.get("no_attention_mask", False),
            **predictor_flags_from_args(cfg))
        for p in self.parameters():
            p.requires_grad_(False)
        super().train(False)

    def train(self, mode: bool = True):
        """Always eval: the coarse model is frozen (no dropout, no masking),
        whatever the fine predictor that owns it is doing."""
        return super().train(False)

    def load_weights(self, ckpt: dict):
        self.encoder.load_state_dict(ckpt["encoder"])
        if self.target_encoder is not None:
            self.target_encoder.load_state_dict(ckpt["ema_encoder"])
        self.predictor.load_state_dict(ckpt["predictor"])

    @torch.no_grad()
    def forward(self, x_hist, adj, active, x_fut=None, masked_idx=None):
        """x_hist (B, H_fine, N, F) raw features; adj / active span the full
        fine window (only the history part is read unless kind == 'true').
        ``masked_idx``: the fine predictor's hidden nodes; the coarse
        predictor hides the same ones past its first step."""
        from src.world_models.graph_jepa_encoder import normalize_adjacency
        H = self.hist * self.w
        if x_hist.shape[1] != H:
            raise ValueError(f"coarse hint expects {H} fine history steps, "
                             f"got {x_hist.shape[1]}")
        if self.kind == "true":
            if x_fut is None:
                raise ValueError("hier hint 'true' needs x_fut")
            x_c, adj_c, _ = coarsen_window(
                torch.cat([x_hist, x_fut], dim=1), adj, active, self.w, self.mode)
            enc = self.target_encoder if self.target_encoder is not None else self.encoder
            z = enc(x_c, normalize_adjacency(adj_c))[:, self.hist:]
            if self.target_norm:
                z = F.layer_norm(z, (z.shape[-1],))
        else:
            x_c, adj_c, act_c = coarsen_window(
                x_hist, adj[:, :H], active[:, :H], self.w, self.mode)
            z_hist = self.encoder(x_c, normalize_adjacency(adj_c))
            # Future graph: carry the last observed coarse step forward (what
            # inference() does anyway), so no future edge is ever read.
            adj_f = torch.cat([adj_c, adj_c[:, -1:].expand(-1, self.pred, -1, -1)], dim=1)
            act_f = torch.cat([act_c, act_c[:, -1:].expand(-1, self.pred, -1)], dim=1)
            idx = masked_idx if masked_idx is not None and masked_idx.numel() > 0 else None
            z = self.predictor.inference(z_hist, adj_f, act_f, intervention_idx=idx)
            if self.kind == "shuffle":
                N = z.shape[2]
                if self._owner_training:
                    perm = torch.randperm(N, generator=self._gen())
                else:
                    perm = torch.randperm(
                        N, generator=torch.Generator().manual_seed(self.seed))
                z = z[:, :, perm.to(z.device)]
        z = F.layer_norm(z, (z.shape[-1],))
        return z.repeat_interleave(self.w, dim=1)

    # The hint itself is always in eval mode, so the owning predictor tells it
    # whether this is a training step (fresh permutation) or an eval one
    # (fixed permutation, comparable across epochs).
    _owner_training = False
    _generator = None

    def _gen(self):
        # Private generator: the permutation must not consume the global RNG
        # stream the rest of training draws from.
        if self._generator is None:
            self._generator = torch.Generator().manual_seed(self.seed + 1)
        return self._generator
