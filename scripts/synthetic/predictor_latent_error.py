"""Does the predictor beat copy-last on ITS OWN objective, per horizon?

Before any downstream number can show a rollout advantage, the predictor's
forecast latent has to be closer to the target latent than the last observed
latent is. This script measures exactly that on a trained DELOS checkpoint,
per forecast horizon, on the held-out windows:

  mse_pred / mse_last   mean squared error of the forecast vs of copy-last,
                        against the encoder's latent at the target step; the
                        ratio < 1 means the predictor moves the latent in a
                        useful direction. With the residual head the floor is
                        ratio = 1 (a zero update IS copy-last).
  disp_r2               R^2 of the PREDICTED displacement (pred - last) against
                        the TRUE displacement (target - last), pooled over
                        latent dims. 0 = the predictor explains none of the
                        movement, 1 = all of it. The most direct "learned the
                        dynamics" number.
  disp_cos              mean cosine between predicted and true displacement.
  disp_mag              ||pred - last|| / ||target - last||: shrinkage. A
                        predictor that has learned the direction but hedges
                        shows cos > 0 with mag << 1.

The target is the ONLINE encoder at the target step with the full window
visible (the same latent the state probe's oracle column reads). For an EMA
arm the training target was the EMA encoder; at the end of training the two
are close, and the comparison forecast-vs-copy-last is unaffected because
both are scored against the same target.

Usage:
    python scripts/synthetic/predictor_latent_error.py \
        --jepa_ckpt outputs/synth_horizon/NPRED_12_rho095_th005/jepa_ema_s42/best.ckpt
    python scripts/synthetic/predictor_latent_error.py \
        --pattern "outputs/synth_horizon/NPRED_12_rho095_th005/jepa_ema*_s*/best.ckpt"
Writes <ckpt_dir>/latent_error.json next to each checkpoint.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.dblp.run_block_probe import (  # noqa: E402
    _make_args_namespace, collect_test_latents)
from src.delos_predictor import build_predictor, predictor_flags_from_args  # noqa: E402
from src.custom_codes.dataset_factory import build_dynamic_dataset  # noqa: E402
from src.world_models.delos_encoder import build_encoder  # noqa: E402


def load_run(jepa_ckpt: Path, device: str, split: str, batch_size: int, num_workers: int):
    ckpt = torch.load(jepa_ckpt, map_location="cpu", weights_only=False)
    saved = ckpt.get("args", {})
    if not saved:
        raise ValueError(f"{jepa_ckpt} has no 'args'")
    args = _make_args_namespace(saved)
    args.device, args.batch_size, args.num_workers, args.split = device, batch_size, num_workers, split
    dev = torch.device(device)
    ds = build_dynamic_dataset(args, split=split)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    node_dim = saved.get("node_dim", 128)
    encoder = build_encoder(saved.get("encoder_kind", "tgcn"), in_dim=ds.feature_dim,
                            hidden_dim=saved.get("encoder_hidden", 128), out_dim=node_dim,
                            freeze=True, bidirectional=saved.get("bidirectional", True)).to(dev)
    encoder.load_state_dict(ckpt["encoder"])
    # EMA-trained runs also store the target encoder: the predictor was
    # trained to emit its latents, so the error is measured against it; the
    # online encoder still provides the predictor's inputs.
    target_encoder = None
    if "ema_encoder" in ckpt:
        target_encoder = build_encoder(saved.get("encoder_kind", "tgcn"), in_dim=ds.feature_dim,
                                       hidden_dim=saved.get("encoder_hidden", 128), out_dim=node_dim,
                                       freeze=True, bidirectional=saved.get("bidirectional", True)).to(dev)
        target_encoder.load_state_dict(ckpt["ema_encoder"])
    encoder.target_encoder = target_encoder
    predictor = build_predictor(
        saved.get("mask_scheme", "node_mask"), num_nodes=ds.num_nodes, node_dim=node_dim,
        history_frames=saved.get("history_size", 3), pred_frames=saved.get("num_preds", 1),
        num_masked_nodes=saved.get("num_masked_nodes", 30), depth=saved.get("depth", 4),
        heads=saved.get("heads", 4), dim_head=node_dim // saved.get("heads", 4),
        mlp_dim=saved.get("mlp_dim", 512), dropout=saved.get("dropout", 0.1),
        use_attention_mask=not saved.get("no_attention_mask", True),
        **predictor_flags_from_args(saved)).to(dev)
    predictor.load_state_dict(ckpt["predictor"])
    return ds, loader, encoder, predictor, saved, dev


@torch.no_grad()
def linear_oracle(encoder, loader, history_size, device, test_frac=0.3, lam=1e-2):
    """Per-horizon latent-error ratio of a LINEAR ridge from the flattened
    history latents [z_t, z_{t-1}, ..., z_{t-H+1}] (the predictor's own
    inputs, in ENCODER space) to the target-step latent, vs copy-last. Fit on
    the first (1 - test_frac) windows, scored on the rest. This is the floor
    any predictor that sees the same latents should reach; a trained
    predictor far above it is underfitting a linear map of its inputs."""
    from src.world_models.delos_encoder import normalize_adjacency
    X, Y, L, Hh = [], [], [], []
    encoder.eval()
    for batch in loader:
        x_hist = batch["x_hist"].to(device); x_fut = batch["x_fut"].to(device)
        adj = batch["adj"].to(device); active = batch["active"].to(device)
        adj_norm = normalize_adjacency(adj)
        z_full = encoder(torch.cat([x_hist, x_fut], 1), adj_norm)
        z_hist = encoder(x_hist, adj_norm[:, :history_size])
        B, T_total, N = active.shape
        for b in range(B):
            for tp in range(T_total - history_size):
                idx = active[b, history_size + tp].nonzero(as_tuple=True)[0]
                if idx.numel() < 2:
                    continue
                X.append(z_hist[b, :, idx].permute(1, 0, 2).reshape(idx.numel(), -1).cpu())
                Y.append(z_full[b, history_size + tp, idx].cpu())
                L.append(z_hist[b, -1, idx].cpu())
                Hh.append(torch.full((idx.numel(),), tp))
    X = torch.cat(X).double().numpy(); Y = torch.cat(Y).double().numpy()
    L = torch.cat(L).double().numpy(); Hh = torch.cat(Hh).numpy()
    out = {}
    for k in np.unique(Hh):
        m = Hh == k
        Xk, Yk, Lk = X[m], Y[m], L[m]
        ntr = int(len(Xk) * (1 - test_frac))
        mu, sd = Xk[:ntr].mean(0), Xk[:ntr].std(0) + 1e-6
        Xs = np.c_[(Xk - mu) / sd, np.ones(len(Xk))]
        # predict the DISPLACEMENT from copy-last (the residual head's parameterisation)
        W = np.linalg.solve(Xs[:ntr].T @ Xs[:ntr] + lam * np.eye(Xs.shape[1]), Xs[:ntr].T @ (Yk[:ntr] - Lk[:ntr]))
        P = Lk[ntr:] + Xs[ntr:] @ W
        out[f"h{int(k)}"] = float(((P - Yk[ntr:]) ** 2).sum(1).mean() / max(((Lk[ntr:] - Yk[ntr:]) ** 2).sum(1).mean(), 1e-12))
    return out


def latent_error(Z_enc, Z_last, Z_pred, horizon):
    out = {}
    for k in np.unique(horizon):
        m = horizon == k
        tgt, last, pred = Z_enc[m], Z_last[m], Z_pred[m]
        d_true, d_pred = tgt - last, pred - last
        mse_pred = float(((pred - tgt) ** 2).sum(1).mean())
        mse_last = float(((last - tgt) ** 2).sum(1).mean())
        sse = float(((d_true - d_pred) ** 2).sum())
        sst = float(((d_true - d_true.mean(0)) ** 2).sum())
        nt, npd = np.linalg.norm(d_true, axis=1), np.linalg.norm(d_pred, axis=1)
        cos = (d_true * d_pred).sum(1) / np.maximum(nt * npd, 1e-12)
        out[f"h{int(k)}"] = {
            "mse_pred": mse_pred, "mse_last": mse_last,
            "ratio": mse_pred / max(mse_last, 1e-12),
            "disp_r2": 1.0 - sse / max(sst, 1e-12),
            "disp_cos": float(cos.mean()),
            "disp_mag": float(npd.mean() / max(nt.mean(), 1e-12)),
            "n": int(m.sum()),
        }
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--jepa_ckpt", nargs="*", default=[])
    p.add_argument("--pattern", default=None, help="glob over best.ckpt files")
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--linear_oracle", action="store_true",
                   help="also fit a per-horizon ridge from the flattened history latents "
                        "to the target latent (encoder space): the linear floor the "
                        "predictor should reach")
    a = p.parse_args()
    if a.device.startswith("cuda") and not torch.cuda.is_available():
        print("[device] no CUDA (login node?) -> cpu", flush=True)
        a.device = "cpu"
    ckpts = [Path(c) for c in a.jepa_ckpt] + ([Path(c) for c in sorted(glob.glob(a.pattern))] if a.pattern else [])
    if not ckpts:
        raise SystemExit("give --jepa_ckpt or --pattern")

    pooled = {}
    for ck in ckpts:
        ds, loader, encoder, predictor, saved, dev = load_run(ck, a.device, a.split, a.batch_size, a.num_workers)
        Z_enc, Z_last, Z_pred, window, node_id, degree, deg_hist, abs_bin, horizon, _ = collect_test_latents(
            encoder, predictor, loader, history_size=saved.get("history_size", 3), device=dev)
        target_space = "online encoder"
        if getattr(encoder, "target_encoder", None) is not None:
            # same rows (same active masks): only the target column changes
            Z_enc = collect_test_latents(encoder.target_encoder, predictor, loader,
                                         history_size=saved.get("history_size", 3), device=dev)[0]
            target_space = "EMA target encoder"
        res = latent_error(Z_enc, Z_last, Z_pred, horizon)
        if a.linear_oracle:
            orc = linear_oracle(encoder, loader, saved.get("history_size", 3), dev)
            for k, v in orc.items():
                res[k]["oracle_ratio"] = v
        res["_ckpt"] = str(ck)
        res["_flags"] = {k: saved.get(k) for k in ("mask_scheme", "residual_head", "change_weight",
                                                    "train_encoder", "ema_target", "num_preds")}
        with open(ck.parent / f"latent_error{'_val' if a.split == 'val' else ''}.json", "w") as f:
            json.dump(res, f, indent=2)
        res["_target_space"] = target_space
        print(f"\n== {ck}   {res['_flags']}   [error measured against the {target_space}]")
        print(f"{'h':>4} {'mse_pred':>10} {'mse_last':>10} {'ratio':>7} {'disp_r2':>8} {'disp_cos':>9} {'disp_mag':>9} {'n':>7}"
              + (f" {'lin.oracle':>11}" if a.linear_oracle else ""))
        for key in sorted((k for k in res if k.startswith("h")), key=lambda k: int(k[1:])):
            r = res[key]
            print(f"{int(key[1:]) + 1:>4} {r['mse_pred']:>10.4f} {r['mse_last']:>10.4f} {r['ratio']:>7.3f} "
                  f"{r['disp_r2']:>8.3f} {r['disp_cos']:>9.3f} {r['disp_mag']:>9.3f} {r['n']:>7}"
                  + (f" {r['oracle_ratio']:>11.3f}" if 'oracle_ratio' in r else ""))
            pooled.setdefault(key, []).append(r)

    if len(ckpts) > 1:
        print(f"\n== mean over {len(ckpts)} checkpoints")
        print(f"{'h':>4} {'ratio':>7} {'disp_r2':>8} {'disp_cos':>9} {'disp_mag':>9}")
        for key in sorted(pooled, key=lambda k: int(k[1:])):
            rs = pooled[key]
            print(f"{int(key[1:]) + 1:>4} {np.mean([r['ratio'] for r in rs]):>7.3f} "
                  f"{np.mean([r['disp_r2'] for r in rs]):>8.3f} {np.mean([r['disp_cos'] for r in rs]):>9.3f} "
                  f"{np.mean([r['disp_mag'] for r in rs]):>9.3f}")
    print("\nReading: ratio < 1 and disp_r2 > 0 = the predictor learned dynamics; ratio ~ 1 with "
          "disp_mag ~ 0 = a zero update (copy-last); disp_cos > 0 with disp_mag << 1 = right "
          "direction, hedged magnitude.")


if __name__ == "__main__":
    main()
