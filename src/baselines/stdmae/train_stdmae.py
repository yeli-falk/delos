"""Stage 2 of the STD-MAE baseline: train the forecaster on frozen MAEs.

Loads the S-MAE and T-MAE checkpoints written by ``pretrain_stdmae.py``,
freezes both, and trains a Graph WaveNet backend that reads the short forecast
history plus the two MAEs' last-patch representations. This is the arm that
lands in the results table.

    python -m src.baselines.stdmae.train_stdmae \
        --data_source metr_la --history_size 12 --num_preds 12 \
        --smae_ckpt outputs/stdmae_pre_s42/smae.ckpt \
        --tmae_ckpt outputs/stdmae_pre_s42/tmae.ckpt \
        --probe_split test --out_dir outputs/reg_metr_la/stdmae_s42

Writes ``test_metrics.json`` in the schema every other arm writes — it is
scored through ``src/world_models/probes.py``'s ``_accum_reg`` /
``_finalize_reg``, on the same windows, with the same valid mask, the same
persistence reference and the same inflection subset — so
``aggregate_regression.py`` merges it straight into the table.

Two things about this arm that a reader of the table should know:

  * Unlike CLDG / DDGCL / IDOL, STD-MAE does NOT plug into the shared frozen
    forecast head. Its published downstream stage IS a full Graph WaveNet
    trained on the forecast loss, and cutting that out would not be STD-MAE.
    It sits closest to the ``supervised`` row: same label budget, but its
    representations came from self-supervised pre-training.
  * On a dynamic-topology dataset (England COVID) the backend gets the window's
    OWN mobility graph per batch rather than one static adjacency; see
    ``long_history.batch_supports``.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.baselines.common import BestValTracker
from src.baselines.long_history import (
    batch_supports,
    build_long_history_datasets,
    static_supports,
)
from src.baselines.graphwavenet import GraphWaveNet
from src.baselines.stdmae.stdmae import STDMAE, Mask, masked_mae
from src.custom_codes.dataset_factory import add_dataset_cli_flags
from src.train import cli
from src.train.engine import set_deterministic, setup_output_dir
from src.world_models.probes import _accum_reg, _blank_reg_sums, _finalize_reg


def curriculum_length(epoch: int, num_preds: int, warm_up: int, cl_epochs: int,
                      step: int = 1) -> int:
    """BasicTS's curriculum: forecast 1 step, then 2, ... every ``cl_epochs``.

    ``cl_epochs <= 0`` disables it and every epoch trains on the full horizon.
    ``epoch`` is 0-indexed here; BasicTS's is 1-indexed and subtracts one.
    """
    if cl_epochs <= 0:
        return num_preds
    if epoch < warm_up:
        return num_preds
    return min(((epoch - warm_up) // cl_epochs + 1) * step, num_preds)


def load_mae(path: str, device, expect_spatial: bool):
    """Rebuild one pre-trained MAE in forecasting mode from its checkpoint."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    for key in ("model", "mask_args", "spatial"):
        if key not in ck:
            raise SystemExit(
                f"{path} is not an STD-MAE pre-training checkpoint (no "
                f"'{key}'). Produce it with "
                f"python -m src.baselines.stdmae.pretrain_stdmae.")
    if bool(ck["spatial"]) != expect_spatial:
        want = "spatial (S-MAE)" if expect_spatial else "temporal (T-MAE)"
        raise SystemExit(
            f"{path} holds a "
            f"{'spatial' if ck['spatial'] else 'temporal'} branch but was "
            f"passed as the {want} checkpoint. Swapping the two silently "
            f"trains the backend on the wrong pair of representations.")
    model = Mask(spatial=bool(ck["spatial"]), **ck["mask_args"])
    model.load_state_dict(ck["model"])
    return model.to(device), ck


def make_supports(batch, dataset, history_size, device, static):
    """Graph WaveNet's supports for one batch: static graph, or the window's."""
    if static is not None:
        return None                      # the backend uses its own buffers
    adj = batch.get("adj")
    if adj is None:
        raise SystemExit(
            f"{type(dataset).__name__} carries neither a dataset-level 2-D "
            f".adj nor a per-window 'adj', so Graph WaveNet has no graph.")
    return batch_supports(adj[:, :history_size].to(device))


def forward_batch(model, batch, dataset, args, device, static):
    x_hist = batch["x_hist"].to(device, non_blocking=True)
    long_hist = batch["long_history"].to(device, non_blocking=True)
    sup = make_supports(batch, dataset, args.history_size, device, static)
    pred = model(x_hist, long_hist, supports=sup)          # B, T_pred, N, 1
    return pred.squeeze(-1)                                # B, T_pred, N


def run_epoch(model, loader, args, device, static, mean, std, cl_len,
              train: bool, opt=None) -> float:
    model.train(train)
    total, n = 0.0, 0
    for batch in loader:
        signal = batch["signal"].to(device, non_blocking=True)
        with torch.set_grad_enabled(train):
            out = forward_batch(model, batch, loader.dataset, args, device, static)
            pred_signal = out * std + mean
            true = signal[:, args.history_size:]
            # Loss in ORIGINAL units with null_val=0.0 -- the reference's
            # convention, and the one that drops METR-LA's missing readings.
            loss = masked_mae(pred_signal[:, :cl_len], true[:, :cl_len], 0.0)
        if train:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                args.clip_grad)
            opt.step()
        bs = signal.shape[0]
        total += float(loss.detach()) * bs
        n += bs
    return total / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, args, device, static, mean, std, valid_mode):
    """Same accounting as every other arm (probes._accum_reg/_finalize_reg)."""
    model.eval()
    horizons = tuple(range(args.num_preds))
    can_inflect = args.history_size >= 2
    sums = {h: _blank_reg_sums() for h in horizons}
    for batch in loader:
        signal = batch["signal"].to(device, non_blocking=True)
        sig_obs = batch.get("signal_obs", batch["signal"]).to(device)
        out = forward_batch(model, batch, loader.dataset, args, device, static)
        pred_signal = out * std + mean
        last_obs = sig_obs[:, args.history_size - 1]
        prev_obs = sig_obs[:, args.history_size - 2] if can_inflect else None
        active = batch["active"].to(device) if "active" in batch else None
        for h in horizons:
            valid = (active[:, args.history_size + h]
                     if (valid_mode == "active" and active is not None) else None)
            _accum_reg(sums[h], pred_signal[:, h],
                       signal[:, args.history_size + h],
                       last_obs, prev_obs, can_inflect, valid=valid)
    return _finalize_reg(sums, report_mape=(valid_mode == "positive"))


def _print_table(metrics):
    print("[stdmae] horizon  MAE  MAE_persist  |  infl: MAE  MAE_persist  (n)")
    for h, m in sorted(metrics.items()):
        line = f"  +{h:>2}: {m['MAE']:.3f}     {m['MAE_persist']:.3f}"
        if "MAE_inflect" in m:
            line += (f"     |  {m['MAE_inflect']:.3f}     "
                     f"{m['MAE_persist_inflect']:.3f}  (n={m['n_inflect']})")
        print(line + ("  <-- beats persistence"
                      if m["MAE"] < m["MAE_persist"] else ""))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    add_dataset_cli_flags(p)
    cli.add_window_flags(p, history_size=12, num_preds=12)
    cli.add_run_flags(p, out_dir="outputs/stdmae")
    cli.add_early_stop_flags(p, patience=20, min_delta=0.0)
    p.add_argument("--smae_ckpt", required=True)
    p.add_argument("--tmae_ckpt", required=True)
    p.add_argument("--probe_split", choices=["test", "val"], default="test",
                   help="Split to report on. 'val' during hyperparameter "
                        "search so nothing is ranked on test.")
    # --- backend (Graph WaveNet) -------------------------------------------
    p.add_argument("--residual_channels", type=int, default=32)
    p.add_argument("--dilation_channels", type=int, default=32)
    p.add_argument("--skip_channels", type=int, default=256)
    p.add_argument("--end_channels", type=int, default=512)
    p.add_argument("--kernel_size", type=int, default=2)
    p.add_argument("--blocks", type=int, default=4)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--backend_dropout", type=float, default=0.3)
    # --- optimisation (reference's STDMAE_* config) ------------------------
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--milestones", type=int, nargs="*",
                   default=[1, 18, 36, 54, 72])
    p.add_argument("--gamma", type=float, default=0.5)
    p.add_argument("--clip_grad", type=float, default=3.0)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--cl_epochs", type=int, default=0,
                   help="Curriculum learning: extend the trained horizon by one "
                        "step every k epochs (reference: 6 on METR-LA). 0 "
                        "trains the full horizon from epoch 1 — the right "
                        "choice on a short horizon such as England COVID's 3.")
    p.add_argument("--cl_warm_epochs", type=int, default=0)
    args = p.parse_args()

    set_deterministic(args.seed)
    out_dir = setup_output_dir(args.out_dir, args)
    device = torch.device(args.device)

    tmae, t_ck = load_mae(args.tmae_ckpt, device, expect_spatial=False)
    smae, s_ck = load_mae(args.smae_ckpt, device, expect_spatial=True)
    long_history = int(t_ck["long_history"])
    if int(s_ck["long_history"]) != long_history:
        raise SystemExit(
            f"the two branches were pre-trained on different window lengths "
            f"(T-MAE {long_history}, S-MAE {s_ck['long_history']}); they must "
            f"see the same input to be concatenated.")
    embed_dim = int(t_ck["mask_args"]["embed_dim"])
    if int(s_ck["mask_args"]["embed_dim"]) != embed_dim:
        raise SystemExit("the two branches have different embed_dim; the "
                         "backend splits their concatenation in half.")
    for ck, tag in ((t_ck, "T-MAE"), (s_ck, "S-MAE")):
        src = (ck.get("args") or {}).get("data_source")
        if src and src != args.data_source:
            print(f"[warn] {tag} was pre-trained on --data_source {src} but "
                  f"this run is {args.data_source}.", flush=True)

    print(f"[setup] STD-MAE forecasting  data={args.data_source} "
          f"long_history={long_history} embed_dim={embed_dim} "
          f"hist={args.history_size} horizon={args.num_preds}", flush=True)

    ds, _ = build_long_history_datasets(args, long_history)
    train_ds, val_ds, eval_ds = ds["train"], ds["val"], ds[args.probe_split]
    mean, std = train_ds.mean, train_ds.std

    def loader(view, shuffle):
        return DataLoader(view, batch_size=args.batch_size, shuffle=shuffle,
                          num_workers=args.num_workers, pin_memory=True)

    train_loader, val_loader = loader(train_ds, True), loader(val_ds, False)
    eval_loader = loader(eval_ds, False)

    fixed_adj = getattr(train_ds, "adj", None)
    static = (np.asarray(fixed_adj) if fixed_adj is not None
              and np.asarray(fixed_adj).ndim == 2 else None)
    backend = GraphWaveNet(
        num_nodes=train_ds.num_nodes,
        supports=static_supports(static) if static is not None else None,
        num_supports=0 if static is not None else 2,
        dropout=args.backend_dropout, gcn_bool=True, addaptadj=True,
        aptinit=None, in_dim=train_ds.feature_dim, out_dim=args.num_preds,
        residual_channels=args.residual_channels,
        dilation_channels=args.dilation_channels,
        skip_channels=args.skip_channels, end_channels=args.end_channels,
        kernel_size=args.kernel_size, blocks=args.blocks, layers=args.layers,
        hidden_dim=embed_dim, hidden_streams=2)   # T-MAE ++ S-MAE
    model = STDMAE(tmae, smae, backend).to(device)
    n_train = sum(q.numel() for q in model.parameters() if q.requires_grad)
    n_froz = sum(q.numel() for q in model.parameters() if not q.requires_grad)
    print(f"[model] graph={'static' if static is not None else 'per-window'}  "
          f"backend trainable={n_train:,}  frozen MAEs={n_froz:,}", flush=True)

    opt = torch.optim.Adam([q for q in model.parameters() if q.requires_grad],
                           lr=args.lr, weight_decay=args.weight_decay, eps=1e-8)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=args.milestones,
                                                 gamma=args.gamma)
    tracker = BestValTracker(args.early_stop_patience, args.early_stop_min_delta)

    n_epochs = 1 if args.smoke else args.epochs
    for epoch in range(n_epochs):
        t0 = time.time()
        cl_len = curriculum_length(epoch, args.num_preds, args.cl_warm_epochs,
                                   args.cl_epochs)
        train_loss = run_epoch(model, train_loader, args, device, static, mean,
                               std, cl_len, train=True, opt=opt)
        with torch.no_grad():
            val_loss = run_epoch(model, val_loader, args, device, static, mean,
                                 std, args.num_preds, train=False)
        sched.step()
        improved, stop = tracker.update(val_loss, model)
        cl_note = f" cl={cl_len}/{args.num_preds}" if args.cl_epochs > 0 else ""
        print(f"[epoch {epoch+1}/{n_epochs}] mae={train_loss:.4f}{cl_note} "
              f"val_mae={val_loss:.4f}{'  (best)' if improved else ''} "
              f"({time.time()-t0:.1f}s)", flush=True)
        if stop:
            print(f"  early stopping: no val improvement for "
                  f"{args.early_stop_patience} epochs", flush=True)
            break

    if tracker.restore(model):
        print(f"[best] restored from best val epoch (val_mae={tracker.best:.4f})",
              flush=True)
    if args.smoke:
        print("[smoke] ok")
        return

    valid_mode = "active" if getattr(eval_ds, "signed_target", False) else "positive"
    metrics = evaluate(model, eval_loader, args, device, static, mean, std,
                       valid_mode)
    _print_table(metrics)
    out_path = Path(out_dir) / "test_metrics.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    torch.save({"model": model.state_dict(), "args": vars(args),
                "val_loss": tracker.best}, Path(out_dir) / "best.ckpt")
    print(f"[done] {args.probe_split} metrics -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
