"""Targeted-intervention evaluation for a trained C-JEPA model on DBLP.

The trained predictor accepts arbitrary node-level masks at inference. This
script swaps the random-mask sampler at eval time for structured
interventions:

  random       baseline; matches training-time distribution
  hub          mask the top-K highest-degree nodes (greatest cascade risk)
  peripheral   mask the bottom-K lowest-degree nodes (controls for "did
               masking high-degree nodes matter, or just *masking K nodes*?")
  none         no intervention (observational forecast baseline)

For each strategy and each K, we compute MRR / Hits@k for link prediction on
the *un-intervened* nodes -- i.e. the model is asked to predict the rest of
the graph's future under the do(.) intervention, and we score it against
ground-truth future edges among the surviving nodes.

This is the experiment that distinguishes our framework from supervised
baselines structurally: the supervised models cannot answer counterfactual
queries because they were never trained to.

Usage:
    python src/train/intervention_eval_dblp.py \\
        --resume outputs/dblp_cjepa/best.ckpt \\
        --encoder_ckpt outputs/dblp_encoder/encoder.ckpt \\
        --data_path dataset/dblp/edges.csv \\
        --strategies random hub peripheral none \\
        --num_masked_nodes 30 100 300
"""
from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.cjepa_graph_predictor import MaskedNodePredictor
from src.custom_codes.dblp_dataset import DBLPDynamicGraphDataset
from src.world_models.probes import (
    fit_link_probe,
    sample_neg_edges,
)
from src.world_models.graph_jepa_encoder import (
    LinkPredictionProbe,
    build_encoder,
    link_ranking_metrics,
    normalize_adjacency,
)


def make_loader(ds, batch_size, num_workers):
    return DataLoader(
        ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=True, drop_last=False,
    )


# ---------------------------------------------------------------------------
# Intervention strategies — return (B, K) long tensor of masked indices per
# batch element. Shared across batch within a strategy/K for simplicity
# (matches the V-JEPA-style "same mask across batch" convention).
# ---------------------------------------------------------------------------
def pick_intervention(strategy: str, K: int, adj_t0: torch.Tensor,
                      active_t0: torch.Tensor, seed: int = 0) -> torch.Tensor:
    """adj_t0: (N, N) bool — adjacency at t=0 of the window.
       active_t0: (N,) bool — node existence at t=0.
       Returns: (K,) long tensor of node indices to intervene on."""
    valid = active_t0.nonzero(as_tuple=True)[0]
    if valid.numel() < K:
        K = valid.numel()
    if K == 0:
        return torch.empty(0, dtype=torch.long, device=adj_t0.device)

    if strategy == "none":
        return torch.empty(0, dtype=torch.long, device=adj_t0.device)
    if strategy == "random":
        g = torch.Generator(device=adj_t0.device).manual_seed(seed)
        perm = torch.randperm(valid.numel(), generator=g, device=adj_t0.device)
        return valid[perm[:K]]
    if strategy == "hub":
        deg = adj_t0.float().sum(-1)
        deg_valid = deg[valid]
        top_local = deg_valid.topk(K).indices
        return valid[top_local]
    if strategy == "peripheral":
        deg = adj_t0.float().sum(-1)
        deg_valid = deg[valid]
        bot_local = (-deg_valid).topk(K).indices
        return valid[bot_local]
    raise ValueError(f"Unknown strategy: {strategy}")


# ---------------------------------------------------------------------------
# Evaluation under intervention
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_under_intervention(predictor, encoder, probe, loader,
                                history_size, strategy, K, device,
                                k_per_pos=99):
    """Run link-prediction eval with the given intervention strategy.

    Important: we restrict scoring to (i, j) edges where *neither* i nor j is
    in the intervened set. The model's predictions for intervened nodes are
    counterfactual and we have no ground-truth to score them against, so
    those edges are excluded from the metric."""
    predictor.eval(); probe.eval()
    per_horizon: dict = {}
    for batch_idx, batch in enumerate(loader):
        x_hist = batch["x_hist"].to(device)
        adj_b = batch["adj"].to(device)
        active_b = batch["active"].to(device)
        B = x_hist.shape[0]
        adj_norm = normalize_adjacency(adj_b)
        z_hist = encoder(x_hist, adj_norm[:, : history_size])

        for b in range(B):
            # Pick intervention indices for this batch element.
            interv = pick_intervention(
                strategy, K, adj_b[b, 0], active_b[b, 0],
                seed=batch_idx * 100 + b,
            )
            pred = predictor.inference(
                z_hist[b:b+1], adj_b[b:b+1], active_b[b:b+1],
                intervention_idx=interv if interv.numel() > 0 else None,
            )  # (1, T_pred, N, D)

            interv_set = set(interv.cpu().tolist())
            for tp in range(pred.shape[1]):
                t_global = history_size + tp
                adj_t = adj_b[b, t_global]
                active_t = active_b[b, t_global]
                if adj_t.sum() == 0 or active_t.sum() < 2:
                    continue
                src, dst = adj_t.nonzero(as_tuple=True)
                # Drop edges that touch an intervened node — we have no ground
                # truth for counterfactual edges, so they're un-scoreable.
                keep = torch.tensor(
                    [s.item() not in interv_set and d.item() not in interv_set
                     for s, d in zip(src, dst)],
                    device=device, dtype=torch.bool,
                )
                src, dst = src[keep], dst[keep]
                if src.numel() == 0:
                    continue
                z = pred[0, tp]
                pos_scores = probe(z[src], z[dst])
                neg_dst = sample_neg_edges(
                    adj_t, active_t, k_per_pos=k_per_pos,
                    seed=batch_idx * 1000 + b * 100 + tp,
                )
                # Realign neg_dst to only the kept positives.
                neg_dst = neg_dst[keep.nonzero(as_tuple=True)[0]]
                if neg_dst.numel() == 0:
                    continue
                src_rep = z[src].unsqueeze(1).expand(-1, neg_dst.shape[1], -1)
                neg_scores = probe(src_rep, z[neg_dst])
                m = link_ranking_metrics(pos_scores, neg_scores)
                per_horizon.setdefault(tp, []).append(m)
    out = {}
    for tp, ms in per_horizon.items():
        keys = ms[0].keys()
        out[tp] = {k: float(np.mean([m[k] for m in ms])) for k in keys}
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--resume", required=True,
                   help="Path to C-JEPA best.ckpt (predictor + encoder).")
    p.add_argument("--encoder_ckpt", default=None,
                   help="If best.ckpt's encoder is unreliable, force-load "
                        "from a separate pretrained encoder.ckpt.")
    p.add_argument("--data_path", default=None)
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--synthetic_nodes", type=int, default=80)
    p.add_argument("--synthetic_years", type=int, default=24)
    p.add_argument("--history_size", type=int, default=3)
    p.add_argument("--num_preds", type=int, default=1)
    p.add_argument("--encoder_kind", choices=["linear", "gcn", "tgcn"], default="tgcn")
    p.add_argument("--node_dim", type=int, default=128)
    p.add_argument("--encoder_hidden", type=int, default=128)
    p.add_argument("--probe_hidden", type=int, default=64)
    p.add_argument("--strategies", nargs="+",
                   default=["none", "random", "hub", "peripheral"])
    p.add_argument("--num_masked_nodes", nargs="+", type=int,
                   default=[30, 100, 300],
                   help="Intervention sizes K to evaluate.")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--probe_epochs", type=int, default=3,
                   help="How long to refit the probe (we don't ship it in "
                        "best.ckpt yet, so we refit on train).")
    p.add_argument("--out_dir", default="outputs/dblp_intervention_eval")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    device = torch.device(args.device)

    # --- Load checkpoint metadata to pick the right architecture
    ck = torch.load(args.resume, map_location=device, weights_only=False)
    saved_args = ck.get("args", {})
    enc_kind = saved_args.get("encoder_kind", args.encoder_kind)
    enc_hidden = saved_args.get("encoder_hidden", args.encoder_hidden)
    enc_out = saved_args.get("node_dim", args.node_dim)
    history_size = saved_args.get("history_size", args.history_size)
    num_preds = saved_args.get("num_preds", args.num_preds)
    print(f"[load] {args.resume}  epoch={ck.get('epoch','?')}  "
          f"val_loss={ck.get('val_loss','?'):.4f}", flush=True)
    print(f"[load] encoder={enc_kind} D={enc_out} H={enc_hidden}  "
          f"history={history_size} preds={num_preds}", flush=True)

    # --- Data
    ds_kwargs = dict(
        data_path=args.data_path, history_size=history_size,
        num_preds=num_preds, synthetic=args.synthetic,
        synthetic_nodes=args.synthetic_nodes,
        synthetic_years=args.synthetic_years,
    )
    train_ds = DBLPDynamicGraphDataset(split="train", **ds_kwargs)
    test_ds = DBLPDynamicGraphDataset(split="test", **ds_kwargs)
    print(f"[data] train={len(train_ds)} test={len(test_ds)} "
          f"N={train_ds.num_nodes} F={train_ds.feature_dim}", flush=True)
    train_loader = make_loader(train_ds, args.batch_size, args.num_workers)
    test_loader = make_loader(test_ds, args.batch_size, args.num_workers)

    # --- Build encoder + predictor + load weights
    encoder = build_encoder(
        enc_kind, in_dim=train_ds.feature_dim,
        hidden_dim=enc_hidden, out_dim=enc_out, freeze=False,
    ).to(device)
    if args.encoder_ckpt:
        enc_ck = torch.load(args.encoder_ckpt, map_location=device, weights_only=False)
        encoder.load_state_dict(enc_ck["encoder"])
        print(f"[encoder] loaded from {args.encoder_ckpt}", flush=True)
    else:
        encoder.load_state_dict(ck["encoder"])
        print(f"[encoder] loaded from best.ckpt", flush=True)
    encoder.eval()
    for q in encoder.parameters():
        q.requires_grad_(False)

    predictor = MaskedNodePredictor(
        num_nodes=train_ds.num_nodes,
        node_dim=enc_out,
        history_frames=history_size,
        pred_frames=num_preds,
        num_masked_nodes=saved_args.get("num_masked_nodes", 30),
        seed=args.seed,
        depth=saved_args.get("depth", 4),
        heads=saved_args.get("heads", 4),
        dim_head=enc_out // saved_args.get("heads", 4),
        mlp_dim=saved_args.get("mlp_dim", 512),
        dropout=saved_args.get("dropout", 0.1),
        use_attention_mask=not saved_args.get("no_attention_mask", True),
        residual_head=saved_args.get("residual_head", False),
    ).to(device)
    predictor.load_state_dict(ck["predictor"])
    predictor.eval()
    print(f"[predictor] loaded; params={sum(p.numel() for p in predictor.parameters()):,}",
          flush=True)

    # --- Fit a fresh link probe (we don't save the probe in best.ckpt yet).
    print("[probe] refitting link-prediction probe on train", flush=True)
    probe = fit_link_probe(
        predictor, encoder, train_loader, history_size, num_preds, device,
        epochs=args.probe_epochs, hidden_dim=args.probe_hidden,
    )

    # --- Evaluate under each (strategy, K) combination
    results = {}
    for strategy, K in product(args.strategies, args.num_masked_nodes):
        if strategy == "none" and K != args.num_masked_nodes[0]:
            continue                                # observational doesn't depend on K
        label = f"{strategy}_K{K}" if strategy != "none" else "none"
        print(f"\n[eval] strategy={strategy}  K={K}", flush=True)
        metrics = evaluate_under_intervention(
            predictor, encoder, probe, test_loader, history_size,
            strategy=strategy, K=K, device=device,
        )
        for tp, m in sorted(metrics.items()):
            line = f"  +{tp+1:>2}y: " + " ".join(f"{k}={v:.4f}" for k, v in m.items())
            print(line, flush=True)
        results[label] = metrics

    # --- Save aggregated table
    with open(out_dir / "intervention_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[saved] {out_dir / 'intervention_results.json'}", flush=True)

    # --- Print a final summary table
    print("\n=== Summary (MRR / Hits@10 averaged across horizons) ===")
    print(f"  {'strategy':<25}  {'MRR':>7}  {'Hits@10':>8}")
    for label, metrics in results.items():
        mrr = float(np.mean([m["MRR"] for m in metrics.values()])) if metrics else 0.0
        h10 = float(np.mean([m["Hits@10"] for m in metrics.values()])) if metrics else 0.0
        print(f"  {label:<25}  {mrr:7.4f}  {h10:8.4f}")


if __name__ == "__main__":
    main()
