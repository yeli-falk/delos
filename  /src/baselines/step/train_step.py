"""Stage 2 of the STEP baseline: train the graph learner + forecaster.

Loads the TSFormer checkpoint written by ``pretrain_tsformer.py``, freezes it,
and trains a discrete graph learner plus a Graph WaveNet backend on the
forecast loss. This is the arm that lands in the results table.

    python -m src.baselines.step.train_step \
        --data_source metr_la --history_size 12 --num_preds 12 \
        --tsformer_ckpt outputs/step_pre_s42/tsformer.ckpt \
        --probe_split test --out_dir outputs/reg_metr_la/step_s42

Writes ``test_metrics.json`` in the schema every other arm writes -- it is
scored through ``src/world_models/probes.py``'s ``_accum_reg`` /
``_finalize_reg``, on the same windows, with the same valid mask, the same
persistence reference and the same inflection subset -- so
``aggregate_regression.py`` merges it straight into the table.

Three things about this arm that a reader of the table should know:

  * Like STD-MAE and unlike CLDG / DDGCL / IDOL, STEP does NOT plug into the
    shared frozen forecast head. Its published downstream stage IS a full Graph
    WaveNet trained on the forecast loss. Read the row next to ``supervised``
    and ``stdmae``.
  * STEP does not use the dataset's graph at all. It LEARNS a discrete
    adjacency, regularised towards a kNN graph over the frozen TSFormer's
    representations. On METR-LA that means the sensor adjacency every other arm
    is given is simply unused; on England COVID the same is true of the
    mobility graph. That is the method's claim, not an omission.
  * The graph learner reads every node's whole TRAIN signal as a feature. It is
    given exactly the train region of the timeline -- never val or test -- since
    the learned adjacency is a parameter that persists into evaluation.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.baselines.common import BestValTracker
from src.baselines.graphwavenet import GraphWaveNet
from src.baselines.long_history import build_long_history_datasets
from src.baselines.step.step import (
    STEP,
    DiscreteGraphLearning,
    TSFormer,
    gsl_coefficient,
    step_loss,
)
from src.custom_codes.dataset_factory import add_dataset_cli_flags
from src.train import cli
from src.train.engine import set_deterministic
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


def load_tsformer(path: str, device):
    """Rebuild the pre-trained TSFormer in forecasting mode from its checkpoint."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    for key in ("model", "tsformer_args", "long_history"):
        if key not in ck:
            raise SystemExit(
                f"{path} is not a TSFormer pre-training checkpoint (no "
                f"'{key}'). Produce it with "
                f"python -m src.baselines.step.pretrain_tsformer.")
    model = TSFormer(**ck["tsformer_args"])
    model.load_state_dict(ck["model"])
    return model.to(device), ck


def forward_batch(model, batch, device):
    x_hist = batch["x_hist"].to(device, non_blocking=True)
    long_hist = batch["long_history"].to(device, non_blocking=True)
    y_hat, theta, adj_knn = model(x_hist, long_hist)       # B, T_pred, N, 1
    return y_hat.squeeze(-1), theta, adj_knn               # B, T_pred, N


def run_epoch(model, loader, args, device, mean, std, cl_len, coef,
              train: bool, opt=None) -> float:
    model.train(train)
    total, n = 0.0, 0
    for batch in loader:
        signal = batch["signal"].to(device, non_blocking=True)
        with torch.set_grad_enabled(train):
            out, theta, adj_knn = forward_batch(model, batch, device)
            pred_signal = out * std + mean
            true = signal[:, args.history_size:]
            # Forecast loss in ORIGINAL units with null_val=0.0 (the reference's
            # convention, and the one that drops METR-LA's missing readings),
            # plus the BCE pulling the learned graph towards the kNN prior.
            loss = step_loss(pred_signal[:, :cl_len], true[:, :cl_len],
                             theta, adj_knn, coef, null_val=0.0)
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
def evaluate(model, loader, args, device, mean, std, valid_mode):
    """Same accounting as every other arm (probes._accum_reg/_finalize_reg)."""
    model.eval()
    horizons = tuple(range(args.num_preds))
    can_inflect = args.history_size >= 2
    sums = {h: _blank_reg_sums() for h in horizons}
    for batch in loader:
        signal = batch["signal"].to(device, non_blocking=True)
        sig_obs = batch.get("signal_obs", batch["signal"]).to(device)
        out, _, _ = forward_batch(model, batch, device)
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
    print("[step] horizon  MAE  MAE_persist  |  infl: MAE  MAE_persist  (n)")
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
    cli.add_run_flags(p, out_dir="outputs/step")
    cli.add_early_stop_flags(p, patience=20, min_delta=0.0)
    p.add_argument("--tsformer_ckpt", required=True)
    p.add_argument("--probe_split", choices=["test", "val"], default="test",
                   help="Split to report on. 'val' during hyperparameter "
                        "search so nothing is ranked on test.")
    # --- discrete graph learning -------------------------------------------
    p.add_argument("--dgl_k", type=int, default=10,
                   help="kNN degree of the prior graph; the top k*N entries of "
                        "the similarity matrix become edges (reference: 10).")
    p.add_argument("--dgl_embedding_dim", type=int, default=100)
    # --- backend (Graph WaveNet) -------------------------------------------
    p.add_argument("--residual_channels", type=int, default=32)
    p.add_argument("--dilation_channels", type=int, default=32)
    p.add_argument("--skip_channels", type=int, default=256)
    p.add_argument("--end_channels", type=int, default=512)
    p.add_argument("--kernel_size", type=int, default=2)
    p.add_argument("--blocks", type=int, default=4)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--backend_dropout", type=float, default=0.3)
    # --- optimisation (reference's STEP_* config) --------------------------
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--milestones", type=int, nargs="*",
                   default=[1, 18, 36, 54, 72])
    p.add_argument("--gamma", type=float, default=0.5)
    p.add_argument("--clip_grad", type=float, default=3.0)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--cl_epochs", type=int, default=0,
                   help="Curriculum learning: extend the trained horizon by one "
                        "step every k epochs (reference: 6 on METR-LA). 0 "
                        "trains the full horizon from epoch 1 -- the right "
                        "choice on a short horizon such as England COVID's 3.")
    p.add_argument("--cl_warm_epochs", type=int, default=0)
    args = p.parse_args()

    set_deterministic(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    device = torch.device(args.device)

    tsformer, ck = load_tsformer(args.tsformer_ckpt, device)
    long_history = int(ck["long_history"])
    embed_dim = int(ck["tsformer_args"]["embed_dim"])
    src = (ck.get("args") or {}).get("data_source")
    if src and src != args.data_source:
        print(f"[warn] TSFormer was pre-trained on --data_source {src} but "
              f"this run is {args.data_source}.", flush=True)

    print(f"[setup] STEP forecasting  data={args.data_source} "
          f"long_history={long_history} embed_dim={embed_dim} "
          f"hist={args.history_size} horizon={args.num_preds} k={args.dgl_k}",
          flush=True)

    ds, info = build_long_history_datasets(args, long_history)
    train_ds, val_ds, eval_ds = ds["train"], ds["val"], ds[args.probe_split]
    mean, std = train_ds.mean, train_ds.std

    def loader(view, shuffle):
        return DataLoader(view, batch_size=args.batch_size, shuffle=shuffle,
                          num_workers=args.num_workers, pin_memory=True)

    train_loader, val_loader = loader(train_ds, True), loader(val_ds, False)
    eval_loader = loader(eval_ds, False)

    # Train region ONLY: the learned adjacency is a parameter that persists
    # into evaluation, so val/test signal must never reach it.
    dgl = DiscreteGraphLearning(info.normalised_train_region(), k=args.dgl_k,
                                embedding_dim=args.dgl_embedding_dim)
    backend = GraphWaveNet(
        num_nodes=train_ds.num_nodes, supports=None, num_supports=2,
        dropout=args.backend_dropout, gcn_bool=True, addaptadj=True,
        aptinit=None, in_dim=train_ds.feature_dim, out_dim=args.num_preds,
        residual_channels=args.residual_channels,
        dilation_channels=args.dilation_channels,
        skip_channels=args.skip_channels, end_channels=args.end_channels,
        kernel_size=args.kernel_size, blocks=args.blocks, layers=args.layers,
        hidden_dim=embed_dim, hidden_streams=1)   # TSFormer only
    model = STEP(tsformer, backend, dgl).to(device)
    n_dgl = sum(q.numel() for q in dgl.parameters() if q.requires_grad)
    n_back = sum(q.numel() for q in backend.parameters() if q.requires_grad)
    n_froz = sum(q.numel() for q in tsformer.parameters())
    print(f"[model] graph=learned (train region {info.train_length} steps, "
          f"dim_fc={dgl.dim_fc})  backend={n_back:,}  graph learner={n_dgl:,}  "
          f"frozen TSFormer={n_froz:,}", flush=True)

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
        coef = gsl_coefficient(epoch + 1)     # gsl_coefficient wants 1-indexed
        train_loss = run_epoch(model, train_loader, args, device, mean, std,
                               cl_len, coef, train=True, opt=opt)
        with torch.no_grad():
            # Always validated on the FULL horizon at the SAME graph-loss
            # weight, so the selection signal does not change under the
            # curriculum or the decaying coefficient.
            val_loss = run_epoch(model, val_loader, args, device, mean, std,
                                 args.num_preds, 0.0, train=False)
        sched.step()
        improved, stop = tracker.update(val_loss, model)
        notes = f" cl={cl_len}/{args.num_preds}" if args.cl_epochs > 0 else ""
        print(f"[epoch {epoch+1}/{n_epochs}] loss={train_loss:.4f}{notes} "
              f"gsl={coef:.3f} val_mae={val_loss:.4f}"
              f"{'  (best)' if improved else ''} ({time.time()-t0:.1f}s)",
              flush=True)
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
    metrics = evaluate(model, eval_loader, args, device, mean, std, valid_mode)
    _print_table(metrics)
    out_path = out_dir / "test_metrics.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    torch.save({"model": model.state_dict(), "args": vars(args),
                "val_loss": tracker.best}, out_dir / "best.ckpt")
    print(f"[done] {args.probe_split} metrics -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
