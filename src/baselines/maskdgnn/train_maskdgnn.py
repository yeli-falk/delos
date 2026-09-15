"""Train the MaskDGNN baseline, save a standard encoder checkpoint, and score
its own edge head end-to-end.

MaskDGNN (see ``maskdgnn.py``) masks low-activeness edges and trains the
encoder + an edge MLP to predict the masked edges and the target snapshot's
edges against sampled negatives. This script trains on the TRAIN portion of
the snapshot sequence, selects on the val-window loss, then

  1. saves ``encoder.ckpt`` in the ``pretrain_encoder.py`` schema, so the
     encoder is scored through the SAME frozen probes as CLDG / DDGCL / IDOL
     (link MRR, state AUROC, forecast MAE);
  2. saves ``maskdgnn.ckpt`` (encoder + edge head) and, unless
     ``--e2e_eval none``, scores that pair on the ``--e2e_eval`` split through
     the shared 99-negative link evaluator (``analyze_jepa.eval_link_pred``)
     -> ``link_e2e.json`` / ``link_e2e_val.json``. That row is the paper's own
     protocol (no separate probe); it sits next to "Supervised (end-to-end)".

    # 1. train (paper defaults: mask 10-40%, alpha 0.7, lam 1, eta 2, gamma -0.5)
    python -m src.baselines.maskdgnn.train_maskdgnn --data_source jodie_mooc \
        --encoder_kind maskdgnn --encoder_hidden 128 --node_dim 128 \
        --mask_ratio 0.2 --alpha 0.7 --epochs 100 --device cuda \
        --out_dir outputs/maskdgnn_mooc

    # 2. frozen-encoder probe, identical to every other baseline
    python -m src.train.analyze_jepa --skip_diagnostic --link_sources encoder \
        --data_source jodie_mooc --encoder_ckpt outputs/maskdgnn_mooc/encoder.ckpt

Windows: ``--window`` consecutive train snapshots are encoded as one sequence
(default = ``--history_size``, so the frequency filter sees the same length at
pretrain and probe time); ``--target next`` additionally needs the following
snapshot as the target. Supports the global dynamic-graph datasets (dblp /
jodie_* / reddit_*) and the fixed-topology signal datasets (metr_la /
england_covid) -- on a constant graph the dynamics score is identically zero,
so masking is driven by PageRank alone; the trainer says so.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from src.baselines.common import BestValTracker, ConstantAdjacency, split_snapshots
from src.baselines.maskdgnn.maskdgnn import ActivenessCache, MaskDGNN
from src.custom_codes.dataset_factory import (
    add_dataset_cli_flags,
    build_dynamic_dataset,
)
from src.train import cli
from src.train.engine import make_loader, save_encoder_ckpt, set_deterministic, setup_output_dir
from src.world_models.graph_jepa_encoder import build_encoder


def _window_geometry(T: int, want: int, extra: int, what: str) -> int:
    """Input-window length usable on a split of ``T`` snapshots (``want``
    inputs + ``extra`` target snapshots), shrunk with a warning when the split
    is too short. Returns 0 if not even one input + target fits."""
    if T < 1 + extra:
        return 0
    L = min(want, T - extra)
    if L < want:
        print(f"[warn] {what}: only {T} snapshots; window shrunk from {want} "
              f"to {L} input snapshot(s) (+{extra} target).", flush=True)
    return L


def run_epoch(model, snaps, cache, L, args, device, rng, n_steps, train: bool,
              opt=None):
    """``n_steps`` windows of ``L`` input snapshots (+ target) sampled from
    ``snaps``. ``train=False`` scores the identical objective under
    ``no_grad`` with the caller's fixed-seed ``rng``, so the val estimate is
    comparable across epochs (same windows, same masks, same negatives)."""
    adj, active, feats = snaps
    T = adj.shape[0]
    extra = model.extra_target_snapshots
    need = L + extra
    if L == 0:
        return float("nan"), {}
    agg, parts_agg, n = 0.0, {"ssl": 0.0, "pos": 0.0, "neg": 0.0}, 0
    for _ in range(n_steps):
        s = int(rng.integers(0, T - need + 1))
        adj_w = np.stack([np.asarray(adj[t]) for t in range(s, s + need)])
        active_w = np.asarray(active[s:s + need])
        feats_w = np.asarray(feats[s:s + L])
        act_scores = torch.stack([cache.get(t) for t in range(s, s + L)])
        gen = torch.Generator(device=device).manual_seed(int(rng.integers(2**31 - 1)))
        with contextlib.nullcontext() if train else torch.no_grad():
            loss, parts = model.window_loss(feats_w, adj_w, active_w, act_scores,
                                            device, generator=gen, return_parts=True)
        if train:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        agg += float(loss.detach()); n += 1
        for k in parts_agg:
            parts_agg[k] += parts[k]
    return agg / max(n, 1), {k: v / max(n, 1) for k, v in parts_agg.items()}


def train_maskdgnn(model, train_snaps, val_snaps, args, device):
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    rng = np.random.default_rng(args.seed)
    extra = model.extra_target_snapshots
    want = args.window if args.window > 0 else args.history_size
    L_tr = _window_geometry(train_snaps[0].shape[0], want, extra, "train")
    L_va = _window_geometry(val_snaps[0].shape[0], want, extra, "val")
    if L_tr == 0:
        raise SystemExit(f"MaskDGNN needs >= {1 + extra} train snapshots; got "
                         f"{train_snaps[0].shape[0]}.")
    if L_va == 0:
        raise SystemExit(f"MaskDGNN needs >= {1 + extra} val snapshots for model "
                         f"selection; got {val_snaps[0].shape[0]}. Bin the "
                         f"timeline into more steps.")
    kw = dict(alpha=args.alpha, lam=args.lam, eta=args.eta, gamma=args.gamma)
    cache_tr = ActivenessCache(train_snaps[0], train_snaps[1], device, **kw)
    cache_va = ActivenessCache(val_snaps[0], val_snaps[1], device, **kw)
    tracker = BestValTracker(args.early_stop_patience, args.early_stop_min_delta)

    n_epochs = 1 if args.smoke else args.epochs
    for epoch in range(n_epochs):
        model.train()
        t0 = time.time()
        n_steps = 3 if args.smoke else args.steps_per_epoch
        train_loss, tp = run_epoch(model, train_snaps, cache_tr, L_tr, args, device,
                                   rng, n_steps, train=True, opt=opt)
        if args.smoke:
            print(f"[smoke] ok  loss={train_loss:.4f} parts={tp}"); return float("inf")

        model.eval()
        val_loss, vp = run_epoch(model, val_snaps, cache_va, L_va, args, device,
                                 np.random.default_rng(args.seed + 10_000),
                                 args.val_steps, train=False)
        improved, stop = tracker.update(val_loss, model)
        print(f"[epoch {epoch+1}/{n_epochs}] loss_maskdgnn={train_loss:.4f} "
              f"(ssl={tp['ssl']:.3f} pos={tp['pos']:.3f} neg={tp['neg']:.3f}) "
              f"val_loss_maskdgnn={val_loss:.4f}"
              f"{'  (best)' if improved else ''} ({time.time()-t0:.1f}s)", flush=True)
        if stop:
            print(f"  early stopping: no val improvement for "
                  f"{args.early_stop_patience} epochs", flush=True)
            break

    if tracker.restore(model):
        print(f"[best] restored encoder + edge head from best val epoch "
              f"(val={tracker.best:.4f})", flush=True)
    return tracker.best


@torch.no_grad()
def evaluate_end_to_end(model, args, split: str, device) -> dict:
    """Score encoder + edge head on ``split`` with the shared link evaluator:
    history-last latents of the FULL (unmasked) window, every positive edge of
    each future step ranked against 99 same-source negatives, ``all`` and
    ``new`` subsets. Same windows, negatives and metrics as every other row."""
    # Lazy: analyze_jepa imports the JEPA predictor (einops); training alone
    # should not need it.
    from src.train.analyze_jepa import eval_link_pred
    ds = build_dynamic_dataset(args, split=split)
    loader = make_loader(ds, args.e2e_batch_size, False, args.num_workers)
    model.eval()
    return eval_link_pred("encoder", model.encoder, None, model.edge_head, loader,
                          args.history_size, device)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    cli.add_dblp_synthetic_flags(p)
    add_dataset_cli_flags(p)
    cli.add_window_flags(p)          # dataset construction + default window length
    cli.add_encoder_flags(p, default_kind="maskdgnn", node_dim=128, encoder_hidden=128)
    cli.add_run_flags(p, out_dir="outputs/maskdgnn")
    cli.add_early_stop_flags(p, patience=20, min_delta=0.0)
    # MaskDGNN-specific (paper Section 5.3 defaults)
    p.add_argument("--mask_ratio", type=float, default=0.2,
                   help="Fraction of each input snapshot's edges masked (paper: "
                        "10-40%% per dataset, best at 10-30%%).")
    p.add_argument("--alpha", type=float, default=0.7,
                   help="Weight of the dynamics score vs PageRank in the "
                        "activeness (paper: 0.7).")
    p.add_argument("--lam", type=float, default=1.0, help="Eq. 2 lambda (paper 1.0).")
    p.add_argument("--eta", type=float, default=2.0, help="Eq. 2 eta (paper 2.0).")
    p.add_argument("--gamma", type=float, default=-0.5, help="Eq. 2 gamma (paper -0.5).")
    p.add_argument("--window", type=int, default=0,
                   help="Input snapshots per training window (the sequence the "
                        "frequency layer filters). 0 = --history_size, so the "
                        "filter sees the same length at pretrain and probe time. "
                        "The paper's gradient-aggregation window is 4-8.")
    p.add_argument("--target", choices=["next", "same"], default="next",
                   help="Which snapshot's edges the latents of input snapshot t "
                        "predict: t+1 (paper eq. 1 / WinGNN's outer objective) "
                        "or t itself (masked-edge reconstruction).")
    p.add_argument("--neg_per_pos", type=int, default=1,
                   help="Sampled non-edges per positive edge (WinGNN/ROLAND: 1).")
    p.add_argument("--max_pos_per_snapshot", type=int, default=0,
                   help="Cap on positive edges scored per snapshot (0 = all).")
    p.add_argument("--edge_hidden", type=int, default=0,
                   help="Hidden width of the edge MLP (0 = node_dim).")
    p.add_argument("--dropout", type=float, default=0.1,
                   help="Dropout in the frequency layer's residual branch and "
                        "the edge MLP (paper 0.1).")
    p.add_argument("--steps_per_epoch", type=int, default=20,
                   help="Training windows per epoch (same budget as CLDG/DDGCL/IDOL).")
    p.add_argument("--val_steps", type=int, default=10,
                   help="Val windows scored per epoch (fixed seed).")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=4e-3)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--e2e_eval", choices=["test", "val", "none"], default="test",
                   help="Split on which encoder + edge head are scored end-to-end "
                        "(link_e2e[_val].json). 'val' for sweeps, 'none' to skip "
                        "(classification / regression pipelines).")
    p.add_argument("--e2e_batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=2)
    args = p.parse_args()

    set_deterministic(args.seed)
    out_dir = setup_output_dir(args.out_dir, args)
    device = torch.device(args.device)
    print(f"[setup] MaskDGNN baseline  data={args.data_source} "
          f"encoder={args.encoder_kind} mask_ratio={args.mask_ratio} "
          f"alpha={args.alpha} target={args.target} window={args.window or args.history_size}",
          flush=True)
    if args.encoder_kind != "maskdgnn":
        print(f"[warn] MaskDGNN's native encoder is 'maskdgnn' (GCN + adaptive "
              f"frequency layer); got '{args.encoder_kind}' -- the 'w/o FE' "
              f"ablation if this is 'gcn'.", flush=True)

    train_ds = build_dynamic_dataset(args, split="train")
    train_snaps, val_snaps = split_snapshots(train_ds, "MaskDGNN", args)
    print(f"[data] train snapshots={train_snaps[0].shape[0]} "
          f"val snapshots={val_snaps[0].shape[0]} N={train_ds.num_nodes} "
          f"F={train_ds.feature_dim}", flush=True)
    if isinstance(train_snaps[0], ConstantAdjacency):
        print("[note] fixed topology: the dynamics score is identically 0, so "
              "activeness = PageRank only and masking is PageRank-weighted.",
              flush=True)

    encoder = build_encoder(
        args.encoder_kind, in_dim=train_ds.feature_dim,
        hidden_dim=args.encoder_hidden, out_dim=args.node_dim, freeze=False,
        bidirectional=args.bidirectional).to(device)
    freq = getattr(encoder, "freq", None)
    if freq is not None:
        freq.dropout.p = args.dropout
    model = MaskDGNN(encoder, latent_dim=args.node_dim, mask_ratio=args.mask_ratio,
                     target=args.target, neg_per_pos=args.neg_per_pos,
                     edge_hidden=args.edge_hidden, dropout=args.dropout,
                     max_pos_per_snapshot=args.max_pos_per_snapshot).to(device)
    n_params = sum(p_.numel() for p_ in model.parameters() if p_.requires_grad)
    print(f"[model] MaskDGNN trainable={n_params:,}", flush=True)

    best = train_maskdgnn(model, train_snaps, val_snaps, args, device)
    if args.smoke:
        return
    ckpt_path = out_dir / "encoder.ckpt"
    save_encoder_ckpt(ckpt_path, model.encoder, args,
                      feature_dim=train_ds.feature_dim,
                      pretrain_objective="maskdgnn", best_loss=best, val_loss=best)
    torch.save({"model": model.state_dict(), "args": vars(args),
                "feature_dim": train_ds.feature_dim, "val_loss": best},
               out_dir / "maskdgnn.ckpt")
    print(f"[done] best val loss_maskdgnn = {best:.4f}  -> {ckpt_path}\n"
          f"[eval] probe it with: python -m src.train.analyze_jepa "
          f"--skip_diagnostic --link_sources encoder --encoder_ckpt {ckpt_path} "
          f"<same --data_* flags>", flush=True)

    if args.e2e_eval != "none":
        print(f"\n--- end-to-end link prediction (encoder + edge head, "
              f"{args.e2e_eval} split) ---", flush=True)
        try:
            metrics = evaluate_end_to_end(model, args, args.e2e_eval, device)
        except Exception as e:                       # noqa: BLE001
            print(f"[warn] end-to-end eval failed ({e}); no link_e2e written",
                  flush=True)
            return
        suffix = "" if args.e2e_eval == "test" else "_val"
        path = out_dir / f"link_e2e{suffix}.json"
        with open(path, "w") as f:
            json.dump(metrics, f, indent=2)
        for subset in ("all", "new"):
            for tp, m in sorted(metrics.get(subset, {}).items()):
                print(f"  [e2e] {subset:<4} +{tp+1:>2}: "
                      + " ".join(f"{k}={v:.4f}" for k, v in m.items()), flush=True)
        print(f"[done] wrote {path}", flush=True)


if __name__ == "__main__":
    main()
