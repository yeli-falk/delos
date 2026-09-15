"""Unified DyG-JEPA trainer (Block 3 entry point).

Pick the predictor masking scheme and the downstream task; the data source flows through the universal loader and the readout flows
through the task-adjustable probe (Block 4):

  --mask_scheme     node_mask | future_forecast | trajectory_block
  --downstream_task link_pred | regression | none

  node_mask / future_forecast + link_pred  -> DBLP / JODIE / Reddit
  node_mask / future_forecast + regression -> METR-LA (fixed-topology)
  trajectory_block + none                   -> V-JEPA-style pretext (diagnostic)


Run:
    python src/train/train_dyg_jepa.py --data_source dblp --synthetic \
        --mask_scheme node_mask --downstream_task link_pred --encoder_ckpt ...
    python src/train/train_dyg_jepa.py --data_source metr_la --metr_la_synthetic \
        --mask_scheme node_mask --downstream_task regression --history_size 12 --num_preds 12
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from src.cjepa_graph_predictor import (
    build_predictor,
    compute_trajectory_jepa_loss,
)
from src.custom_codes.dataset_factory import (
    add_dataset_cli_flags,
    build_dynamic_dataset,
)
from src.train import cli
from src.train.engine import (
    cosine_warmup_lr,
    make_loader,
    set_deterministic,
    setup_output_dir,
)
# Shared JEPA loops (dynamic-graph + fixed-topology variants).
from src.train.jepa import (
    EMAEncoder,
    compute_louvain_communities,
    evaluate_jepa as dblp_eval_jepa,
    evaluate_jepa_fixed as metr_eval_jepa,
    train_jepa_epoch as dblp_train_epoch,
    train_jepa_epoch_fixed as metr_train_epoch,
)
from src.train import probe as probe_stage
from src.world_models.graph_jepa_encoder import build_encoder, normalize_adjacency
from src.world_models.trajectory_jepa import (
    TrajectoryWindowPooler,
    sample_trajectory_mask_blocks,
)

# Node-signal forecasting sources. All of them run the unified regression path
# (run_regression_dynamic): England COVID and synthetic_dyn carry per-snapshot
# adjacency, METR-LA resolves to its single static graph inside
# probes.graph_inputs. Sharing the path is what puts them in one table — it is
# also the only way METR-LA gets the encoder-training variants (frozen / naive /
# EMA / VICReg), which the legacy fixed-topology loop does not implement.
DYNAMIC_REGRESSION_SOURCES = {"england_covid", "synthetic_dyn", "metr_la"}


# ---------------------------------------------------------------------------
# Encoder / predictor construction (shared)
# ---------------------------------------------------------------------------
def build_encoder_for(args, feature_dim, device):
    enc_kind, enc_hidden, enc_out = args.encoder_kind, args.encoder_hidden, args.node_dim
    enc_bidir = getattr(args, "bidirectional", True)
    ckpt = None
    if args.encoder_ckpt:
        if not os.path.exists(args.encoder_ckpt):
            raise FileNotFoundError(
                f"--encoder_ckpt {args.encoder_ckpt!r} does not exist. Run the "
                f"encoder pretraining first (src/train/pretrain_encoder.py, e.g. "
                f"scripts/england_covid/pretrain_encoder.sh) to produce it, or "
                f"drop --encoder_ckpt to train with a random-init encoder.")
        ckpt = torch.load(args.encoder_ckpt, map_location=device, weights_only=False)
        enc_kind = ckpt.get("encoder_kind", enc_kind)
        enc_hidden = ckpt.get("encoder_hidden", enc_hidden)
        enc_out = ckpt.get("node_dim", enc_out)
        enc_bidir = ckpt.get("bidirectional", enc_bidir)
    encoder = build_encoder(
        enc_kind, in_dim=feature_dim, hidden_dim=enc_hidden, out_dim=enc_out,
        freeze=False, bidirectional=enc_bidir).to(device)
    if ckpt is not None:
        encoder.load_state_dict(ckpt["encoder"])
        print(f"[encoder] loaded {enc_kind} from {args.encoder_ckpt}", flush=True)
    else:
        print(f"[encoder] random init {enc_kind}", flush=True)
    args.node_dim = enc_out
    train_enc = getattr(args, "train_encoder", False)
    encoder.train(train_enc)
    for p_ in encoder.parameters():
        p_.requires_grad_(train_enc)
    return encoder


def build_jepa_predictor(args, num_nodes, device):
    return build_predictor(
        args.mask_scheme,
        num_nodes=num_nodes, node_dim=args.node_dim,
        history_frames=args.history_size, pred_frames=args.num_preds,
        num_masked_nodes=args.num_masked_nodes, seed=args.seed,
        depth=args.depth, heads=args.heads, dim_head=args.node_dim // args.heads,
        mlp_dim=args.mlp_dim, dropout=args.dropout,
        allow_temporal_neighbors=args.allow_temporal_neighbors,
        use_attention_mask=not args.no_attention_mask,
        attn_impl=args.attn_impl,
        use_edge_head=(args.edge_mask_rate > 0),
        residual_head=args.residual_head,
        latent_norm=args.target_norm,
    ).to(device)


def build_ema(args, encoder, device, steps_per_epoch):
    """EMA target encoder, or None. ``--ema_schedule cosine`` anneals the
    momentum from --ema_momentum to --ema_momentum_end over the whole run
    (I-JEPA / V-JEPA); ``const`` is the legacy fixed momentum."""
    if not args.ema_target:
        return None
    if args.ema_schedule == "cosine":
        ema = EMAEncoder(encoder, momentum=args.ema_momentum,
                         momentum_end=args.ema_momentum_end,
                         total_steps=max(1, args.epochs * steps_per_epoch))
    else:
        ema = EMAEncoder(encoder, momentum=args.ema_momentum)
    ema.target.to(device)
    return ema


def jepa_variant_kwargs(args):
    """Train-loop kwargs for the opt-in JEPA variants (all off by default)."""
    return dict(
        mask_encoder_input=args.mask_encoder_input,
        target_norm=args.target_norm,
        target_grad=args.target_grad,
        sigreg_weight=args.sigreg,
        sigreg_slices=args.sigreg_slices,
        sigreg_knots=args.sigreg_knots,
    )


def build_obs_decoder(args, feature_dim, device):
    """Readout from latent space back to observation space for
    ``--predict_space input``, else None.

    Deliberately a SINGLE linear layer: the latent-vs-input comparison must be
    about where the loss lives, not about handing one arm extra capacity.
    """
    if getattr(args, "predict_space", "latent") != "input":
        return None
    dec = torch.nn.Linear(args.node_dim, feature_dim).to(device)
    print(f"[predict_space] input — decoding {args.node_dim} -> {feature_dim} "
          f"and matching raw node features", flush=True)
    return dec


def _scheduler(opt, args, steps_per_epoch):
    total = max(1, args.epochs * steps_per_epoch)
    if args.epochs == 0:
        return None
    warmup = int(args.warmup_epochs * steps_per_epoch)
    return torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: cosine_warmup_lr(s, warmup, total, args.lr_min_ratio))


# ---------------------------------------------------------------------------
# Shared: best-checkpoint save + optional early stopping
# ---------------------------------------------------------------------------
def _append_history(out_dir, record):
    """Append one epoch's losses to ``history.json``.

    ``best.ckpt`` records only the single best val loss, which cannot show
    whether a run converged, plateaued or DIVERGED -- the difference between
    "needs more epochs" and "the LR is too high" is the whole loss trajectory,
    and it was previously only recoverable by scraping SLURM logs. Rewritten in
    full each epoch so a killed job still leaves a readable file.
    """
    path = out_dir / "history.json"
    hist = []
    if path.exists():
        try:
            with open(path) as f:
                hist = json.load(f)
        except (json.JSONDecodeError, OSError):
            hist = []          # a torn write must not kill the training run
    hist.append(record)
    with open(path, "w") as f:
        json.dump(hist, f, indent=2)


def _update_best(vl, best_val, no_improve, epoch, args, predictor, encoder, out_dir):
    """Save best.ckpt on improvement and decide whether to early-stop.

    Returns ``(best_val, no_improve, improved, stop)``. An improvement requires
    the val loss to drop by more than ``--early_stop_min_delta``.
    """
    improved = vl["loss"] < best_val - args.early_stop_min_delta
    if improved:
        best_val = vl["loss"]
        no_improve = 0
        torch.save({"predictor": predictor.state_dict(),
                    "encoder": encoder.state_dict(), "epoch": epoch,
                    "val_loss": best_val, "args": vars(args)},
                   out_dir / "best.ckpt")
    else:
        no_improve += 1
    stop = (args.early_stop_patience > 0
            and no_improve >= args.early_stop_patience)
    return best_val, no_improve, improved, stop


# ---------------------------------------------------------------------------
# Task: link prediction (dynamic graphs)
# ---------------------------------------------------------------------------
def run_link_pred(args, device, out_dir):
    train_ds = build_dynamic_dataset(args, "train")
    val_ds = build_dynamic_dataset(args, "val")
    test_ds = build_dynamic_dataset(args, "test")
    print(f"[data] train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} "
          f"N={train_ds.num_nodes} F={train_ds.feature_dim}", flush=True)
    train_loader = make_loader(train_ds, args.batch_size, True, args.num_workers)
    val_loader = make_loader(val_ds, args.batch_size, False, args.num_workers)
    test_loader = make_loader(test_ds, args.batch_size, False, args.num_workers)

    encoder = build_encoder_for(args, train_ds.feature_dim, device)
    ema = build_ema(args, encoder, device, max(1, len(train_loader)))
    predictor = build_jepa_predictor(args, train_ds.num_nodes, device)
    decoder = build_obs_decoder(args, train_ds.feature_dim, device)

    pred_params = list(predictor.parameters())
    if decoder is not None:
        pred_params += list(decoder.parameters())
    if args.train_encoder:
        enc_lr = args.encoder_lr if args.encoder_lr is not None else args.lr
        opt = torch.optim.AdamW([
            {"params": pred_params, "lr": args.lr},
            {"params": encoder.parameters(), "lr": enc_lr},
        ], weight_decay=args.weight_decay)
    else:
        opt = torch.optim.AdamW(pred_params, lr=args.lr,
                                weight_decay=args.weight_decay)
    sched = _scheduler(opt, args, max(1, len(train_loader)))

    community_labels = community_rng = None
    if args.community_masking:
        community_labels = compute_louvain_communities(train_ds, seed=args.seed)
        community_rng = np.random.default_rng(args.seed)
    train_kwargs = dict(
        edge_mask_rate=args.edge_mask_rate, edge_weight=args.edge_weight,
        edge_neg_per_pos=args.edge_neg_per_pos, ema_target=ema,
        train_encoder=args.train_encoder, vicreg_var_weight=args.vicreg_var,
        vicreg_cov_weight=args.vicreg_cov, vicreg_gamma=args.vicreg_gamma,
        community_labels=community_labels,
        community_max_mask_size=args.num_masked_nodes,
        community_rng=community_rng,
        decoder=decoder,
        change_weight=args.change_weight,
        **jepa_variant_kwargs(args))

    if args.smoke:
        dblp_train_epoch(predictor, encoder, train_loader, opt, args.history_size,
                         device, log_every=1, max_steps=3, scheduler=sched,
                         **train_kwargs)
        print("[smoke] ok"); return

    best_val = float("inf")
    no_improve = 0
    for epoch in range(args.epochs):
        print(f"[epoch {epoch+1}/{args.epochs}]", flush=True)
        tr = dblp_train_epoch(predictor, encoder, train_loader, opt,
                              args.history_size, device, scheduler=sched, **train_kwargs)
        vl = dblp_eval_jepa(predictor, encoder, val_loader, args.history_size,
                            device, decoder=decoder, ema_target=ema,
                            change_weight=args.change_weight,
                            mask_encoder_input=args.mask_encoder_input,
                            target_norm=args.target_norm)
        best_val, no_improve, improved, stop = _update_best(
            vl, best_val, no_improve, epoch, args, predictor, encoder, out_dir)
        _append_history(out_dir, {
            "epoch": epoch + 1, "train_loss": tr["loss"], "val_loss": vl["loss"],
            "lr": opt.param_groups[0]["lr"], "best": improved})
        print(f"  train loss={tr['loss']:.4f}  val loss={vl['loss']:.4f}"
              f"{'  (best)' if improved else ''}", flush=True)
        if stop:
            print(f"  early stopping: no val improvement for "
                  f"{no_improve} epochs", flush=True)
            break

    # Block 4: downstream probe stage (fit + evaluate + report).
    # --probe_split val during a hyperparameter sweep, so configurations are
    # ranked without ever reading the test split; test for the final table.
    eval_loader = val_loader if args.probe_split == "val" else test_loader
    probe_stage.probe_link_pred(predictor, encoder, train_loader, eval_loader,
                                args, device, out_dir)


# ---------------------------------------------------------------------------
# Task: regression (METR-LA, fixed topology)
# ---------------------------------------------------------------------------
def run_regression(args, device, out_dir):
    train_ds = build_dynamic_dataset(args, "train")
    val_ds = build_dynamic_dataset(args, "val")
    test_ds = build_dynamic_dataset(args, "test")
    for ds in (val_ds, test_ds):
        ds.mean, ds.std = train_ds.mean, train_ds.std
    print(f"[data] train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} "
          f"N={train_ds.num_nodes} F={train_ds.feature_dim}", flush=True)
    train_loader = make_loader(train_ds, args.batch_size, True, args.num_workers)
    val_loader = make_loader(val_ds, args.batch_size, False, args.num_workers)
    test_loader = make_loader(test_ds, args.batch_size, False, args.num_workers)

    adj_dense = torch.from_numpy(train_ds.adj).to(device)
    adj_norm = normalize_adjacency(adj_dense)

    encoder = build_encoder_for(args, train_ds.feature_dim, device)
    encoder.eval()
    for p_ in encoder.parameters():
        p_.requires_grad_(False)
    predictor = build_jepa_predictor(args, train_ds.num_nodes, device)
    opt = torch.optim.AdamW(predictor.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    sched = _scheduler(opt, args, max(1, len(train_loader)))

    if args.smoke:
        metr_train_epoch(predictor, encoder, train_loader, opt, adj_dense,
                         adj_norm, args.history_size, device, log_every=1, max_steps=3)
        print("[smoke] ok"); return

    best_val = float("inf")
    no_improve = 0
    for epoch in range(args.epochs):
        print(f"[epoch {epoch+1}/{args.epochs}]", flush=True)
        tr = metr_train_epoch(predictor, encoder, train_loader, opt, adj_dense,
                              adj_norm, args.history_size, device, scheduler=sched)
        vl = metr_eval_jepa(predictor, encoder, val_loader, adj_dense, adj_norm,
                            args.history_size, device)
        best_val, no_improve, improved, stop = _update_best(
            vl, best_val, no_improve, epoch, args, predictor, encoder, out_dir)
        _append_history(out_dir, {
            "epoch": epoch + 1, "train_loss": tr["loss"], "val_loss": vl["loss"],
            "lr": opt.param_groups[0]["lr"], "best": improved})
        print(f"  train loss={tr['loss']:.4f}  val loss={vl['loss']:.4f}"
              f"{'  (best)' if improved else ''}", flush=True)
        if stop:
            print(f"  early stopping: no val improvement for "
                  f"{no_improve} epochs", flush=True)
            break

    # Block 4: downstream probe stage (fit + evaluate + report).
    probe_stage.probe_regression(predictor, encoder, train_loader, test_loader,
                                 adj_dense, adj_norm, args, device, out_dir)


# ---------------------------------------------------------------------------
# Task: regression on a dynamic-topology graph (England COVID)
# ---------------------------------------------------------------------------
# Same probe/metrics as run_regression, but the mobility graph changes every
# snapshot, so the JEPA loop and probe consume per-batch adjacency (the dynamic
# DBLP-style loops) instead of one broadcast static graph.
def run_regression_dynamic(args, device, out_dir):
    train_ds = build_dynamic_dataset(args, "train")
    val_ds = build_dynamic_dataset(args, "val")
    test_ds = build_dynamic_dataset(args, "test")
    for ds in (val_ds, test_ds):
        ds.mean, ds.std = train_ds.mean, train_ds.std
    print(f"[data] train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} "
          f"N={train_ds.num_nodes} F={train_ds.feature_dim}", flush=True)
    train_loader = make_loader(train_ds, args.batch_size, True, args.num_workers)
    val_loader = make_loader(val_ds, args.batch_size, False, args.num_workers)
    test_loader = make_loader(test_ds, args.batch_size, False, args.num_workers)

    encoder = build_encoder_for(args, train_ds.feature_dim, device)
    ema = build_ema(args, encoder, device, max(1, len(train_loader)))
    if not args.train_encoder:
        encoder.eval()
        for p_ in encoder.parameters():
            p_.requires_grad_(False)
    predictor = build_jepa_predictor(args, train_ds.num_nodes, device)
    decoder = build_obs_decoder(args, train_ds.feature_dim, device)
    pred_params = list(predictor.parameters())
    if decoder is not None:
        pred_params += list(decoder.parameters())
    if args.train_encoder:
        enc_lr = args.encoder_lr if args.encoder_lr is not None else args.lr
        opt = torch.optim.AdamW([
            {"params": pred_params, "lr": args.lr},
            {"params": encoder.parameters(), "lr": enc_lr},
        ], weight_decay=args.weight_decay)
    else:
        opt = torch.optim.AdamW(pred_params, lr=args.lr,
                                weight_decay=args.weight_decay)
    sched = _scheduler(opt, args, max(1, len(train_loader)))
    train_kwargs = dict(change_weight=args.change_weight, ema_target=ema,
                        train_encoder=args.train_encoder, decoder=decoder,
                        vicreg_var_weight=args.vicreg_var,
                        vicreg_cov_weight=args.vicreg_cov,
                        vicreg_gamma=args.vicreg_gamma,
                        **jepa_variant_kwargs(args))

    if args.smoke:
        dblp_train_epoch(predictor, encoder, train_loader, opt, args.history_size,
                         device, log_every=1, max_steps=3, scheduler=sched,
                         **train_kwargs)
        print("[smoke] ok"); return

    best_val = float("inf")
    no_improve = 0
    for epoch in range(args.epochs):
        print(f"[epoch {epoch+1}/{args.epochs}]", flush=True)
        tr = dblp_train_epoch(predictor, encoder, train_loader, opt,
                              args.history_size, device, scheduler=sched,
                              **train_kwargs)
        vl = dblp_eval_jepa(predictor, encoder, val_loader, args.history_size,
                            device, decoder=decoder, ema_target=ema,
                            change_weight=args.change_weight,
                            mask_encoder_input=args.mask_encoder_input,
                            target_norm=args.target_norm)
        best_val, no_improve, improved, stop = _update_best(
            vl, best_val, no_improve, epoch, args, predictor, encoder, out_dir)
        _append_history(out_dir, {
            "epoch": epoch + 1, "train_loss": tr["loss"], "val_loss": vl["loss"],
            "lr": opt.param_groups[0]["lr"], "best": improved})
        print(f"  train loss={tr['loss']:.4f}  val loss={vl['loss']:.4f}"
              f"{'  (best)' if improved else ''}", flush=True)
        if stop:
            print(f"  early stopping: no val improvement for "
                  f"{no_improve} epochs", flush=True)
            break

    # Block 4: downstream probe stage (fit + evaluate + report).
    # --probe_split val during a hyperparameter sweep, so configurations are
    # ranked without ever reading the test split; test for the final table.
    eval_loader = val_loader if args.probe_split == "val" else test_loader
    probe_stage.probe_regression_dynamic(predictor, encoder, train_loader,
                                         eval_loader, args, device, out_dir)


# ---------------------------------------------------------------------------
# Task: trajectory-block pretext (diagnostic; no downstream probe)
# ---------------------------------------------------------------------------
def run_trajectory(args, device, out_dir):
    train_ds = build_dynamic_dataset(args, "train")
    val_ds = build_dynamic_dataset(args, "val")
    print(f"[data] train={len(train_ds)} val={len(val_ds)} "
          f"N={train_ds.num_nodes} F={train_ds.feature_dim}", flush=True)
    train_loader = make_loader(train_ds, args.batch_size, True, args.num_workers)

    encoder = build_encoder_for(args, train_ds.feature_dim, device)
    encoder.eval()
    for p_ in encoder.parameters():
        p_.requires_grad_(False)
    pooler = TrajectoryWindowPooler(window_size=args.window_size, pool=args.pool,
                                    latent_dim=args.node_dim).to(device)
    K = (args.history_size + args.num_preds) // args.window_size
    predictor = build_predictor(
        "trajectory_block", num_windows=K, num_nodes=train_ds.num_nodes,
        node_dim=args.node_dim, depth=args.depth, heads=args.heads,
        dim_head=args.node_dim // args.heads, mlp_dim=args.mlp_dim,
        dropout=args.dropout, node_pos_mode=args.node_pos_mode).to(device)
    opt = torch.optim.AdamW(
        list(predictor.parameters()) + list(pooler.parameters()), lr=args.lr,
        weight_decay=args.weight_decay)

    n_steps = 3 if args.smoke else None
    for epoch in range(1 if args.smoke else args.epochs):
        predictor.train()
        tot, n = 0.0, 0
        for step, batch in enumerate(train_loader):
            if n_steps is not None and step >= n_steps:
                break
            x = torch.cat([batch["x_hist"], batch["x_fut"]], dim=1).to(device)
            adj = batch["adj"].to(device)
            with torch.no_grad():
                z = encoder(x, normalize_adjacency(adj))
            tokens = pooler(z)                                  # (B, K, N, D)
            Kc, Nc = tokens.shape[1], tokens.shape[2]
            gen = torch.Generator(device=device).manual_seed(args.seed + step)
            mask = sample_trajectory_mask_blocks(
                Kc, Nc, n_blocks=args.n_blocks,
                block_k_range=tuple(args.block_k_range),
                block_n_frac=tuple(args.block_n_frac),
                generator=gen, device=device)
            pred = predictor(tokens, mask)
            loss = compute_trajectory_jepa_loss(pred, tokens.detach(), mask)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            tot += loss.item() * x.shape[0]; n += x.shape[0]
        print(f"[epoch {epoch+1}] traj loss={tot/max(n,1):.4f}", flush=True)
        if args.smoke:
            print("[smoke] ok"); return
    torch.save({"predictor": predictor.state_dict(), "pooler": pooler.state_dict(),
                "encoder": encoder.state_dict(), "args": vars(args)},
               out_dir / "best.ckpt")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    cli.add_dblp_synthetic_flags(p)
    add_dataset_cli_flags(p)
    cli.add_window_flags(p)
    cli.add_encoder_flags(p)
    cli.add_optim_flags(p, lr=5e-4)
    cli.add_run_flags(p, out_dir="outputs/dyg_jepa")
    # scheme / task
    p.add_argument("--mask_scheme",
                   choices=["node_mask", "future_forecast",
                            "mlp_forecast", "gru_forecast", "trajectory_block"],
                   default="node_mask",
                   help="node_mask = masked spacetime transformer (the model). "
                        "future_forecast = same transformer, no masking. "
                        "mlp_forecast / gru_forecast = graph-free generic "
                        "forecaster (Tier-2 ablation control).")
    p.add_argument("--downstream_task",
                   choices=["link_pred", "regression", "none"], default="link_pred")
    # predictor
    p.add_argument("--num_masked_nodes", type=int, default=10)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--mlp_dim", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--no_attention_mask", action="store_true")
    p.add_argument("--attn_impl", choices=["sparse", "dense"], default="sparse",
                   help="How the structural attention mask is evaluated. "
                        "sparse = edge-list gather/scatter attention, memory "
                        "linear in #allowed pairs (fits full-N MOOC). dense = "
                        "materialised (B,1,TN,TN) mask through SDPA, quadratic "
                        "memory; kept for parity checks. Identical function.")
    p.add_argument("--allow_temporal_neighbors", action="store_true")
    p.add_argument("--residual_head", action="store_true",
                   help="pred = last-visible-latent + to_out(h), with to_out "
                        "zero-initialised: copy-last becomes the training "
                        "floor instead of something the predictor must "
                        "re-learn (and often under-learns).")
    p.add_argument("--predict_space", choices=["latent", "input"], default="latent",
                   help="What the predictor is trained to match. 'latent' (the "
                        "JEPA objective) matches the encoder's latents. 'input' "
                        "adds a node_dim->feature_dim head and matches the RAW "
                        "node features instead — ordinary observation-space "
                        "forecasting with the SAME encoder, predictor and "
                        "masking. This is the control the JEPA claim needs: it "
                        "isolates the prediction SPACE and nothing else. Under "
                        "heavy observation noise 'latent' should win, because "
                        "'input' must spend capacity on noise it cannot predict.")
    # encoder ckpt / EMA / joint training
    p.add_argument("--encoder_ckpt", default=None)
    p.add_argument("--train_encoder", action="store_true")
    p.add_argument("--encoder_lr", type=float, default=None)
    p.add_argument("--ema_target", action="store_true")
    p.add_argument("--ema_momentum", type=float, default=0.996)
    p.add_argument("--ema_schedule", choices=["const", "cosine"], default="const",
                   help="'cosine' anneals the EMA momentum from --ema_momentum "
                        "to --ema_momentum_end over the run (I-JEPA / V-JEPA). "
                        "'const' = legacy fixed momentum.")
    p.add_argument("--ema_momentum_end", type=float, default=1.0)
    # JEPA variants (all off by default => legacy behaviour is reproducible)
    p.add_argument("--mask_encoder_input", action="store_true",
                   help="Hide the masked nodes from the ONLINE encoder too "
                        "(features zeroed, incident edges dropped for t>=1), "
                        "so their information cannot reach the predictor via "
                        "neighbours' latents. I-JEPA's context-encoder rule; "
                        "the target encoder still sees the full graph.")
    p.add_argument("--target_norm", action="store_true",
                   help="LayerNorm the target latents over the channel dim "
                        "(I-JEPA / V-JEPA). Also normalises the predictor's "
                        "input latents so the residual anchor and every "
                        "downstream inference() call live in the same space. "
                        "Ignored under --predict_space input.")
    p.add_argument("--target_grad", action="store_true",
                   help="LeJEPA: no stop-gradient on the online-encoder target. "
                        "Requires --train_encoder, no --ema_target, latent "
                        "space; pair with --sigreg.")
    p.add_argument("--sigreg", type=float, default=0.0,
                   help="Weight of LeJEPA's SIGReg (isotropic-Gaussian "
                        "regulariser) on predictor outputs and, with "
                        "--train_encoder, on encoder latents. 0 = off. "
                        "Intended as a replacement for --vicreg_var/--vicreg_cov.")
    p.add_argument("--sigreg_slices", type=int, default=256,
                   help="Random projection directions per SIGReg evaluation.")
    p.add_argument("--sigreg_knots", type=int, default=17,
                   help="Quadrature points on [0, 5] for the Epps-Pulley integral.")
    # loss extras
    p.add_argument("--edge_mask_rate", type=float, default=0.0)
    p.add_argument("--edge_weight", type=float, default=1.0)
    p.add_argument("--edge_neg_per_pos", type=int, default=1)
    p.add_argument("--vicreg_var", type=float, default=0.0)
    p.add_argument("--vicreg_cov", type=float, default=0.0)
    p.add_argument("--vicreg_gamma", type=float, default=1.0)
    p.add_argument("--change_weight", type=float, default=0.0,
                   help="Weight per-(t,n) latent MSE by 1 + change_weight * "
                        "||z_t - z_{t-1}|| / mean: upweights the rare "
                        "transition-carrying nodes the unweighted average "
                        "drowns out. 0 = off (legacy behaviour).")
    p.add_argument("--community_masking", action="store_true")
    # lr schedule
    p.add_argument("--warmup_epochs", type=float, default=1.0)
    p.add_argument("--lr_min_ratio", type=float, default=0.05)
    # early stopping (shared with pretrain_encoder + every baseline)
    cli.add_early_stop_flags(p, patience=20, min_delta=0.0)
    # probe
    p.add_argument("--probe_epochs", type=int, default=3)
    p.add_argument("--probe_hidden", type=int, default=64)
    p.add_argument("--probe_split", choices=["test", "val"], default="test",
                   help="Split the downstream probe is EVALUATED on. Use 'val' "
                        "for hyperparameter search so configurations are never "
                        "ranked on test; 'test' (default) for the final table.")
    p.add_argument("--persist_residual", action="store_true",
                   help="Regression only: the probe predicts the delta over "
                        "the last observed value, so persistence is the "
                        "zero-output floor. Pairs with --residual_head to "
                        "floor the forecast at persistence in signal space.")
    # trajectory-block extras
    p.add_argument("--window_size", type=int, default=2)
    p.add_argument("--pool", choices=["mean", "gru"], default="mean")
    p.add_argument("--node_pos_mode", choices=["off", "learned", "from_input"],
                   default="off")
    p.add_argument("--n_blocks", type=int, default=2)
    p.add_argument("--block_k_range", type=int, nargs=2, default=[1, 2])
    p.add_argument("--block_n_frac", type=float, nargs=2, default=[0.2, 0.5])
    args = p.parse_args()

    set_deterministic(args.seed)
    out_dir = setup_output_dir(args.out_dir, args)
    device = torch.device(args.device)
    print(f"[setup] data={args.data_source} mask_scheme={args.mask_scheme} "
          f"task={args.downstream_task} device={args.device}", flush=True)

    if args.target_norm and args.mask_scheme in ("mlp_forecast", "gru_forecast"):
        raise SystemExit("--target_norm needs a predictor with latent_norm "
                         "(node_mask / future_forecast); the plain forecasters "
                         "would see un-normalised inputs against a normalised "
                         "target.")
    if args.target_grad and (not args.train_encoder or args.ema_target
                             or args.predict_space != "latent"):
        raise SystemExit("--target_grad requires --train_encoder, no "
                         "--ema_target and --predict_space latent.")
    if args.mask_scheme == "trajectory_block":
        run_trajectory(args, device, out_dir)
    elif args.downstream_task == "regression":
        if args.data_source in DYNAMIC_REGRESSION_SOURCES:
            run_regression_dynamic(args, device, out_dir)
        else:
            run_regression(args, device, out_dir)
    elif args.downstream_task == "link_pred":
        run_link_pred(args, device, out_dir)
    else:
        raise ValueError(
            f"downstream_task={args.downstream_task!r} with mask_scheme="
            f"{args.mask_scheme!r} is not a supported combination. Use "
            f"trajectory_block for pretext-only, or pick link_pred/regression.")


if __name__ == "__main__":
    main()
