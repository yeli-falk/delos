"""Representation diagnostic for trajectory-JEPA checkpoints.

Mirrors `analyze_jepa.py` (part 1) exactly but extracts latents from the
trajectory-JEPA stack (encoder + window pooler + trajectory predictor)
instead of the node-level JEPA stack. The downstream analysis
(effective rank, structural probes, CKA, optional Louvain block probe)
is reused unchanged.

Per-node representations:
  encoder rep:  pooler(encoder(x_full))[:, -1, :, :]   -- last visible window
  jepa rep:     predictor(tokens, mask_last_window)[:, -1, :, :]  -- the
                predictor's reconstruction of the last window when that
                window is the only one masked. Analogous to "predict the
                future window from the visible history".

Structural target timestep is the LAST timestep of the long sequence
(the end of the last window), since that's the time the encoder/predictor
representations should describe.

Usage:
    python src/train/diagnose_trajectory_jepa.py \\
        --jepa_ckpt outputs/sbm8_traj_s42_cjepa/best.ckpt \\
        --data_source dblp --synthetic --synthetic_kind sbm \\
        --synthetic_blocks 8 \\
        --history_size 20 --num_preds 4 \\
        --out_dir outputs/sbm8_traj_s42_diagnostic
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.cjepa_graph_predictor import TrajectoryJEPAPredictor
from src.custom_codes.dataset_factory import (
    add_dataset_cli_flags,
    build_dynamic_dataset,
)
from src.train.analyze_jepa import (
    _pagerank,
    _kcore,
    compute_block_labels,
    run_diagnostic,
)
from src.world_models.graph_jepa_encoder import build_encoder, normalize_adjacency
from src.world_models.trajectory_jepa import TrajectoryWindowPooler


@torch.no_grad()
def collect_representations_trajectory(encoder, pooler, predictor, loader,
                                       device):
    """Per-node latents from the trajectory-JEPA stack.

    For each window in the loader:
      tokens = pooler(encoder(x_full))           # (B, K, N, D)
      enc_rep[b, n]   = tokens[b, -1, n]         # last visible window
      mask = zeros((K, N)); mask[-1] = True       # mask last window only
      pred = predictor(tokens, mask)
      jepa_rep[b, n]  = pred[b, -1, n]            # predictor's recon of it

    Structural targets (degree, PageRank, k-core) are computed at the
    last timestep of the long sequence; that's the time-of-reference
    for the per-node representation we just extracted.
    """
    enc_rows, pred_rows = [], []
    deg_rows, pr_rows, kc_rows = [], [], []
    win_rows, node_rows = [], []

    encoder.eval(); pooler.eval(); predictor.eval()
    for wid, batch in enumerate(loader):
        x = torch.cat([batch["x_hist"], batch["x_fut"]], dim=1).to(device)
        adj = batch["adj"].to(device)              # (B, T_long, N, N) bool
        active = batch["active"].to(device)         # (B, T_long, N) bool

        adj_norm = normalize_adjacency(adj)
        z = encoder(x, adj_norm)                    # (B, T_long, N, D)
        tokens = pooler(z)                          # (B, K, N, D)

        B, K, N, D = tokens.shape
        mask = torch.zeros(K, N, dtype=torch.bool, device=device)
        mask[-1, :] = True                          # mask the last window
        pred = predictor(tokens, mask)              # (B, K, N, D)

        t_last = adj.shape[1] - 1                   # final timestep index
        for b in range(B):
            act = active[b, t_last]
            if act.sum() < 2:
                continue
            idx = act.nonzero(as_tuple=True)[0]
            A = adj[b, t_last].cpu().numpy().astype(np.float64)
            act_np = act.cpu().numpy()
            deg = A.sum(axis=-1)
            pr = _pagerank(A, act_np)
            kc = _kcore(A, act_np)

            idx_np = idx.cpu().numpy()
            enc_rows.append(tokens[b, -1, idx].cpu())
            pred_rows.append(pred[b, -1, idx].cpu())
            deg_rows.append(torch.from_numpy(deg[idx_np]))
            pr_rows.append(torch.from_numpy(pr[idx_np]))
            kc_rows.append(torch.from_numpy(kc[idx_np]))
            win_rows.append(torch.full((idx.numel(),), wid * 1000 + b))
            node_rows.append(torch.from_numpy(idx_np.astype(np.int64)))

    Z_enc = torch.cat(enc_rows, dim=0).numpy().astype(np.float64)
    Z_pred = torch.cat(pred_rows, dim=0).numpy().astype(np.float64)
    degree = torch.cat(deg_rows, dim=0).numpy().astype(np.float64)
    pagerank = torch.cat(pr_rows, dim=0).numpy().astype(np.float64)
    kcore = torch.cat(kc_rows, dim=0).numpy().astype(np.float64)
    window = torch.cat(win_rows, dim=0).numpy()
    node_id = torch.cat(node_rows, dim=0).numpy()
    return dict(
        Z_enc=Z_enc, Z_pred=Z_pred, degree=degree,
        pagerank=pagerank, kcore=kcore,
        window=window, t_offset=np.zeros(len(degree), dtype=np.int64),
        node_id=node_id,
    )


def make_loader(ds, batch_size, num_workers):
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True, drop_last=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--jepa_ckpt", required=True,
                   help="Trajectory-JEPA best.ckpt produced by "
                        "train_trajectory_jepa.py")
    p.add_argument("--history_size", type=int, default=20)
    p.add_argument("--num_preds", type=int, default=4)
    p.add_argument("--split", choices=["val", "test"], default="test")
    p.add_argument("--data_path", default=None)
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--synthetic_nodes", type=int, default=100)
    p.add_argument("--synthetic_years", type=int, default=30)
    p.add_argument("--synthetic_kind", choices=["dblp", "sbm"], default="sbm")
    p.add_argument("--synthetic_blocks", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out_dir", default="outputs/trajectory_diagnostic")
    p.add_argument("--block_probe", choices=["none", "louvain", "dataset"],
                   default="none")
    add_dataset_cli_flags(p)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    device = torch.device(args.device)

    # Load checkpoint first; lots of architecture comes from its args.
    ckpt = torch.load(args.jepa_ckpt, map_location=device, weights_only=False)
    saved = ckpt.get("args", {}) or {}
    window_size = saved.get("window_size", 4)
    pool = saved.get("pool", "mean")
    enc_kind = saved.get("encoder_kind", "tgcn")
    enc_hidden = saved.get("encoder_hidden", 128)
    node_dim = saved.get("node_dim", 128)
    enc_bidir = saved.get("bidirectional", True)
    depth = saved.get("depth", 4)
    heads = saved.get("heads", 4)
    mlp_dim = saved.get("mlp_dim", 512)
    dropout = saved.get("dropout", 0.1)
    node_pos_mode = saved.get("node_pos_mode", "learned")  # legacy ckpts

    # Force the dataset window to match what the checkpoint was trained on.
    args.history_size = saved.get("history_size", args.history_size)
    args.num_preds = saved.get("num_preds", args.num_preds)

    ds = build_dynamic_dataset(args, split=args.split)
    loader = make_loader(ds, args.batch_size, args.num_workers)
    T_long = args.history_size + args.num_preds
    K = T_long // window_size
    print(f"[data] {args.split}: {len(ds)} windows  N={ds.num_nodes}  "
          f"T_long={T_long}  W={window_size}  K={K}", flush=True)

    encoder = build_encoder(
        enc_kind, in_dim=ds.feature_dim, hidden_dim=enc_hidden,
        out_dim=node_dim, freeze=True, bidirectional=enc_bidir,
    ).to(device)
    if "encoder" in ckpt:
        encoder.load_state_dict(ckpt["encoder"])
        print(f"[encoder] loaded {enc_kind} from jepa_ckpt", flush=True)
    else:
        print(f"[encoder] random init {enc_kind} (no encoder in ckpt)",
              flush=True)

    pooler = TrajectoryWindowPooler(
        window_size=window_size, pool=pool, latent_dim=node_dim,
    ).to(device)
    if "pooler" in ckpt:
        pooler.load_state_dict(ckpt["pooler"])

    predictor = TrajectoryJEPAPredictor(
        num_windows=K, num_nodes=ds.num_nodes, node_dim=node_dim,
        depth=depth, heads=heads, dim_head=node_dim // heads,
        mlp_dim=mlp_dim, dropout=dropout,
        node_pos_mode=node_pos_mode,
    ).to(device)
    predictor.load_state_dict(ckpt["predictor"])
    print(f"[jepa] loaded trajectory predictor (epoch={ckpt.get('epoch','?')}, "
          f"val_loss={ckpt.get('val_loss','?')})", flush=True)

    data = collect_representations_trajectory(
        encoder, pooler, predictor, loader, device,
    )

    block_labels = None
    if args.block_probe != "none":
        try:
            block_labels = compute_block_labels(ds, source=args.block_probe)
        except Exception as e:
            print(f"[warn] block_probe={args.block_probe} failed: {e}",
                  flush=True)
    run_diagnostic(data, out_dir, block_labels=block_labels)


if __name__ == "__main__":
    main()
