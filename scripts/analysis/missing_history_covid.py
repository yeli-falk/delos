"""Missing-history (partial-observability) probe for England COVID forecasting.

Tests the causal-JEPA hypothesis: when a node's own past is hidden, can the
model still forecast it from GRAPH CONTEXT (neighbours) rather than its own
history? At eval time we hide a fraction of nodes' history features (they stay
active in the graph, so message-passing still reaches them) and score the
forecast on exactly those hidden nodes.

Naive self-history baselines (persistence / window-mean) are undefined for a
node with no history, so the parameter-free comparator is RELATIONAL
PERSISTENCE: predict a hidden node's future = mean of its active neighbours'
last observed value. A method beats it only by using the graph beyond a
one-hop average.

Works on a full DyG-JEPA checkpoint (encoder + predictor rollout) or a bare
encoder checkpoint (CLDG/DDGCL/IDOL; encoder + forecast head). Heads are fit on
CLEAN train data with absolute prediction (persist_residual is meaningless here
since the node's own last value is the thing being hidden).

Usage:
    python scripts/analysis/missing_history_covid.py \
        --ckpt outputs/ec_dygjepa_s42/best.ckpt \
        --history_size 8 --num_preds 6 --fracs 0 0.25 0.5 0.75 1.0
    python scripts/analysis/missing_history_covid.py \
        --ckpt outputs/ec_cldg_s42/encoder.ckpt --history_size 8 --num_preds 6
"""
from __future__ import annotations

import argparse
import json
import types
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.cjepa_graph_predictor import MaskedNodePredictor, predictor_flags_from_args
from src.custom_codes.dataset_factory import build_dynamic_dataset
from src.world_models.graph_jepa_encoder import build_encoder, normalize_adjacency
from src.world_models.probes import (
    RegressionForecastHead,
    corrupt_history_bn,
    fit_regression_probe_dynamic,
    sample_masked_per_example,
)


def _feat_mean(loader, device):
    """Per-feature mean over active train positions — the history fill token."""
    tot, n = None, 0.0
    for b in loader:
        x = b["x_hist"].to(device)
        m = b["active"][:, : x.shape[1]].to(device).unsqueeze(-1).float()
        s = (x * m).sum(dim=(0, 1, 2))
        tot = s if tot is None else tot + s
        n += m.sum().item()
    return tot / max(n, 1.0)


def _relational_persistence(adj, signal, history_size):
    """(B, N) mean of active neighbours' last-observed signal (context persist)."""
    A = adj[:, history_size - 1].float()                       # (B, N, N)
    last = signal[:, history_size - 1]                          # (B, N)
    num = torch.bmm(A, last.unsqueeze(-1)).squeeze(-1)          # (B, N)
    den = A.sum(-1).clamp(min=1.0)
    return num / den


@torch.no_grad()
def evaluate(encoder, predictor, head, probe, loader, history_size, num_preds,
             device, fracs, fill, mask_seed=0):
    encoder.eval()
    mean = loader.dataset.mean
    std = loader.dataset.std
    out = {}
    for frac in fracs:
        # forecast MAE on the hidden nodes; relational-persistence on the same.
        sums = {h: {"m": 0.0, "r": 0.0, "n": 0.0} for h in range(num_preds)}
        g = torch.Generator(device=device).manual_seed(mask_seed)
        for batch in loader:
            x_hist = batch["x_hist"].to(device)
            adj = batch["adj"].to(device)
            active = batch["active"].to(device)
            signal = batch["signal"].to(device)
            B, N = active.shape[0], active.shape[-1]
            # Hide a fraction of each window's nodes that have a past to hide.
            active_ref = active[:, history_size - 1]               # (B, N)
            mask_bn = sample_masked_per_example(active_ref, frac, g)  # (B, N)
            x_in = corrupt_history_bn(x_hist, mask_bn, fill) if frac > 0 else x_hist
            adj_norm = normalize_adjacency(adj)
            z_hist = encoder(x_in, adj_norm[:, :history_size])
            if predictor is not None:
                pred = predictor.inference(z_hist, adj, active)     # (B,Tp,N,D)
                est = probe(pred) * std + mean                      # absolute
            else:
                est = head(z_hist[:, -1]) * std + mean              # (B,Tp,N)
            rel = _relational_persistence(adj, signal, history_size)  # (B,N)
            # Scored set: hidden nodes (frac>0) or all nodes (frac==0 reference).
            sel_bn = mask_bn if frac > 0 else torch.ones(B, N, dtype=torch.bool,
                                                         device=device)
            for h in range(num_preds):
                true = signal[:, history_size + h]                 # (B,N)
                valid = (true > 0) & active[:, history_size + h] & sel_bn
                v = valid.float()
                sums[h]["m"] += ((est[:, h] - true).abs() * v).sum().item()
                sums[h]["r"] += ((rel - true).abs() * v).sum().item()
                sums[h]["n"] += v.sum().item()
        out[frac] = {
            h: {"MAE_model": s["m"] / max(s["n"], 1),
                "MAE_relpersist": s["r"] / max(s["n"], 1),
                "n": int(s["n"])}
            for h, s in sums.items() if s["n"] > 0
        }
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--history_size", type=int, default=8)
    p.add_argument("--num_preds", type=int, default=6)
    p.add_argument("--fracs", type=float, nargs="+",
                   default=[0.0, 0.25, 0.5, 0.75, 1.0])
    p.add_argument("--probe_epochs", type=int, default=10)
    p.add_argument("--probe_hidden", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    torch.manual_seed(args.seed)
    device = args.device

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    saved = ck.get("args", {}) or {}
    ns = types.SimpleNamespace(**saved)
    ns.data_source = "england_covid"
    ns.history_size = args.history_size
    ns.num_preds = args.num_preds
    ns.seed = args.seed
    train_ds = build_dynamic_dataset(ns, "train")
    test_ds = build_dynamic_dataset(ns, "test")
    test_ds.mean, test_ds.std = train_ds.mean, train_ds.std
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers)

    enc_kind = ck.get("encoder_kind", saved.get("encoder_kind", "tgcn"))
    enc_hidden = ck.get("encoder_hidden", saved.get("encoder_hidden", 128))
    node_dim = ck.get("node_dim", saved.get("node_dim", 128))
    enc_bidir = ck.get("bidirectional", saved.get("bidirectional", False))
    encoder = build_encoder(enc_kind, in_dim=train_ds.feature_dim,
                            hidden_dim=enc_hidden, out_dim=node_dim,
                            freeze=True, bidirectional=enc_bidir).to(device)
    encoder.load_state_dict(ck["encoder"])
    fill = _feat_mean(train_loader, device)

    predictor = head = probe = None
    if "predictor" in ck:
        predictor = MaskedNodePredictor(
            num_nodes=train_ds.num_nodes, node_dim=node_dim,
            history_frames=args.history_size, pred_frames=args.num_preds,
            num_masked_nodes=saved.get("num_masked_nodes", 20),
            depth=saved.get("depth", 6), heads=saved.get("heads", 8),
            dim_head=node_dim // saved.get("heads", 8),
            mlp_dim=saved.get("mlp_dim", 512), dropout=saved.get("dropout", 0.1),
            use_attention_mask=not saved.get("no_attention_mask", False),
            **predictor_flags_from_args(saved),
        ).to(device)
        predictor.load_state_dict(ck["predictor"])
        # Fit an ABSOLUTE-prediction probe on clean train (persist_residual off).
        probe = fit_regression_probe_dynamic(
            predictor, encoder, train_loader, args.history_size, args.num_preds,
            device, epochs=args.probe_epochs, hidden_dim=args.probe_hidden,
            persist_residual=False)
        tag = Path(args.ckpt).parent.name + " (predictor)"
    else:
        # Bare encoder: fit an absolute forecast head on clean train.
        head = RegressionForecastHead(node_dim, args.num_preds,
                                      hidden_dim=args.probe_hidden).to(device)
        opt = torch.optim.AdamW(head.parameters(), lr=1e-3)
        mean, std = train_ds.mean, train_ds.std
        for _ in range(args.probe_epochs):
            for b in train_loader:
                x = b["x_hist"].to(device); adj = b["adj"].to(device)
                sig = b["signal"].to(device)
                an = normalize_adjacency(adj)
                with torch.no_grad():
                    z = encoder(x, an[:, :args.history_size])
                tgt = (sig[:, args.history_size:] - mean) / std
                loss = torch.nn.functional.mse_loss(head(z[:, -1]), tgt)
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        tag = Path(args.ckpt).parent.name + " (encoder)"

    res = evaluate(encoder, predictor, head, probe, test_loader,
                   args.history_size, args.num_preds, device, args.fracs, fill,
                   mask_seed=args.seed)

    print(f"\n[{tag}]  missing-history forecast MAE ($\\downarrow$) on hidden nodes")
    print(f"{'frac':>6} | " + " ".join(f"h{h}" for h in range(args.num_preds))
          + "   | model_avg  relpersist_avg  n")
    for frac in args.fracs:
        r = res.get(frac, {})
        if not r:
            continue
        maes = [r[h]["MAE_model"] for h in sorted(r)]
        rels = [r[h]["MAE_relpersist"] for h in sorted(r)]
        cells = " ".join(f"{m:.2f}" for m in maes)
        nn = r[sorted(r)[0]]["n"]
        note = "  (frac=0: all nodes, full history — reference)" if frac == 0 else ""
        print(f"{frac:>6.2f} | {cells}   | {np.mean(maes):>8.3f}  "
              f"{np.mean(rels):>13.3f}  {nn}{note}")
    out_path = Path(args.ckpt).parent / "missing_history.json"
    with open(out_path, "w") as f:
        json.dump({str(k): v for k, v in res.items()}, f, indent=2)
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
