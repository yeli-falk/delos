"""Unified DELOS trainer.

Pick the predictor masking scheme and the downstream task; the data source
flows through the universal loader and the readout through the
task-adjustable probe:
  --mask_scheme     node_mask | future_forecast | mlp_forecast | gru_forecast | trajectory_block
  --downstream_task link_pred | regression | none

  node_mask / future_forecast + link_pred  -> DBLP / JODIE / Reddit
  node_mask / future_forecast + regression -> METR-LA (fixed-topology)
  trajectory_block + none                   -> V-JEPA-style pretext (diagnostic)


Run:
    python src/train/train_delos.py --data_source dblp --data_path dataset/dblp/edges_top2k.csv \
        --mask_scheme node_mask --downstream_task link_pred --encoder_ckpt ...
    python src/train/train_delos.py --data_source metr_la --metr_la_synthetic \
        --mask_scheme node_mask --downstream_task regression --history_size 12 --num_preds 12
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from src.delos_predictor import (
    build_predictor,
    compute_trajectory_jepa_loss,
)
from src.custom_codes.dataset_factory import (
    add_dataset_cli_flags,
    build_dynamic_dataset,
)
from src.train import cli
from src.train.engine import (
    check_encoder_ckpt_batch_size,
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
from src.world_models.delos_encoder import build_encoder, normalize_adjacency
from src.world_models.trajectory_jepa import (
    TrajectoryWindowPooler,
    sample_trajectory_mask_blocks,
)

# Node-signal forecasting sources. All of them run the unified regression path
# (run_regression_dynamic): England COVID and synthetic_dyn carry per-snapshot
# adjacency, METR-LA resolves to its single static graph inside
# probes.graph_inputs. The shared path is the only one that gives METR-LA the
# encoder-training variants (frozen / naive / EMA / VICReg).
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
        check_encoder_ckpt_batch_size(ckpt, args, args.encoder_ckpt)
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


def build_jepa_predictor(args, num_nodes, device, feature_dim=None):
    """``--hier_ckpt``: the frozen coarse model is built inside the predictor
    and its weights loaded here; ``args.hier_cfg`` (saved with the fine
    checkpoint) is what lets every other script rebuild it."""
    hier_ckpt = None
    if getattr(args, "hier_ckpt", None):
        from src.world_models.hier import hier_cfg_from_ckpt
        if feature_dim is None:
            raise SystemExit("--hier_ckpt is wired for --downstream_task "
                             "link_pred / none only")
        if not os.path.exists(args.hier_ckpt):
            raise FileNotFoundError(f"--hier_ckpt {args.hier_ckpt!r} does not exist "
                                    f"(train the coarse level first)")
        hier_ckpt = torch.load(args.hier_ckpt, map_location=device, weights_only=False)
        args.hier_cfg = hier_cfg_from_ckpt(hier_ckpt, feature_dim)
        if args.hier_cfg.get("data_source") != args.data_source:
            raise SystemExit(f"--hier_ckpt was trained on "
                             f"{args.hier_cfg.get('data_source')!r}, this run is "
                             f"{args.data_source!r}")
    predictor = _build_jepa_predictor(args, num_nodes, device)
    if hier_ckpt is not None:
        predictor.hint.load_weights(hier_ckpt)
        c = args.hier_cfg
        print(f"[hier] coarse level from {args.hier_ckpt}: window={c['coarse_window']} "
              f"mode={c.get('coarse_mode')} {c['history_size']}+{c['num_preds']} coarse "
              f"steps, hint={args.hier_hint} (frozen, "
              f"{sum(q.numel() for q in predictor.hint.parameters())} params)", flush=True)
    return predictor


def _build_jepa_predictor(args, num_nodes, device):
    hier = {}
    if getattr(args, "hier_cfg", None):
        hier = dict(hier_cfg=args.hier_cfg, hier_hint=args.hier_hint)
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
        mask_frac=args.mask_frac,
        input_delta=args.input_delta,
        linear_skip=args.linear_skip,
        **hier,
    ).to(device)


def build_ema(args, encoder, device, steps_per_epoch):
    """EMA target encoder, or None. ``--ema_schedule cosine`` anneals the
    momentum from --ema_momentum to --ema_momentum_end over
    ``--ema_anneal_epochs`` epochs (0 = the whole ``--epochs`` budget, the
    I-JEPA / V-JEPA convention); ``const`` is a fixed momentum.

    The horizon matters under early stopping: a whole-run anneal barely
    moves the momentum before the run stops, so the cosine arm degenerates
    to the constant one. A horizon shorter than a typical run lets the
    schedule finish (target frozen at --ema_momentum_end) while the
    predictor still trains. The anneal clock counts encoder updates, so it
    starts when --encoder_freeze_epochs ends."""
    if not args.ema_target:
        return None
    if args.ema_schedule == "cosine":
        horizon = args.ema_anneal_epochs if args.ema_anneal_epochs > 0 else args.epochs
        ema = EMAEncoder(encoder, momentum=args.ema_momentum,
                         momentum_end=args.ema_momentum_end,
                         total_steps=max(1, horizon * steps_per_epoch))
    else:
        ema = EMAEncoder(encoder, momentum=args.ema_momentum)
    ema.target.to(device)
    return ema


def smoke_done(args, train_ds=None):
    """End a ``--smoke`` run with the peak GPU memory and the window geometry.

    Three steps are enough to reach the peak, which is cheaper than sizing
    the batch by resubmitting a full run. The dominant term is predictable: a
    dynamic-topology window carries a dense ``(B, T, N, N)`` adjacency plus its
    normalised float32 copy, so memory is linear in the batch AND in
    ``T = history_size + num_preds``. The printed estimate is that term alone;
    the measured peak includes the encoder, the predictor's activations and the
    autograd graph.
    """
    import torch as _t
    line = "[smoke] ok"
    if train_ds is not None:
        N = getattr(train_ds, "num_nodes", None)
        T = args.history_size + args.num_preds
        if N:
            gb = args.batch_size * T * N * N * 5 / 1024 ** 3   # bool + float32 copy
            line += (f"  N={N} T={T} (hist {args.history_size} + preds {args.num_preds}) "
                     f"batch={args.batch_size} -> dense adjacency ~{gb:.1f} GiB")
    if _t.cuda.is_available():
        line += (f"  peak allocated={_t.cuda.max_memory_allocated() / 1024 ** 3:.1f} GiB"
                 f"  reserved={_t.cuda.max_memory_reserved() / 1024 ** 3:.1f} GiB")
    print(line, flush=True)


def set_encoder_frozen(args, epoch, encoder, ema):
    """Apply ``--encoder_freeze_epochs`` at the start of ``epoch`` (0-based).

    For the first K epochs of a --train_encoder run the encoder's parameters
    get no gradient (AdamW skips them, weight decay included) and the EMA
    target neither updates nor advances its anneal clock. The predictor
    therefore first fits a FIXED, pretrained target, and the encoder only
    starts moving once the predictor's gradients carry signal rather than
    its initial noise; otherwise early encoder updates reshape the latents
    to be easier to predict, not more informative. Returns True while frozen.
    """
    k = getattr(args, "encoder_freeze_epochs", 0)
    if not args.train_encoder or k <= 0:
        return False
    frozen = epoch < k
    for p_ in encoder.parameters():
        p_.requires_grad_(not frozen)
    if ema is not None:
        ema.paused = frozen
    if epoch == k:
        print(f"  [encoder] unfrozen at epoch {epoch + 1}: encoder lr and "
              f"EMA updates now active", flush=True)
    return frozen


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
    whether a run converged, plateaued or DIVERGED. Rewritten in full each
    epoch so a killed job still leaves a readable file.
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


def _ckpt_payload(predictor, encoder, epoch, val_loss, args, ema=None):
    """best.ckpt / snapshot schema. ``encoder`` is the ONLINE encoder (what
    every probe reads). With an EMA target the predictor was trained to emit
    the TARGET encoder's latents, so that encoder is saved too
    (``ema_encoder``): the latent-error diagnostic scores the forecast
    against it, since online and target can drift apart. Older checkpoints
    lack the key."""
    out = {"predictor": predictor.state_dict(), "encoder": encoder.state_dict(),
           "epoch": epoch, "val_loss": val_loss, "args": vars(args)}
    if ema is not None and getattr(ema, "target", None) is not None:
        out["ema_encoder"] = ema.target.state_dict()
    return out


def _update_best(vl, best_val, no_improve, epoch, args, predictor, encoder, out_dir,
                 ema=None):
    """Save best.ckpt on improvement and decide whether to early-stop.

    Returns ``(best_val, no_improve, improved, stop)``. An improvement requires
    the val loss to drop by more than ``--early_stop_min_delta``.
    """
    improved = vl["loss"] < best_val - args.early_stop_min_delta
    if improved:
        best_val = vl["loss"]
        no_improve = 0
        torch.save(_ckpt_payload(predictor, encoder, epoch, best_val, args, ema),
                   out_dir / "best.ckpt")
    else:
        no_improve += 1
    stop = (args.early_stop_patience > 0
            and no_improve >= args.early_stop_patience)
    return best_val, no_improve, improved, stop


def _save_snapshot(epoch, vl, args, predictor, encoder, out_dir, ema=None):
    """Periodic checkpoint for probe-based model selection.

    ``best.ckpt`` is chosen on the validation JEPA loss. With an EMA target that
    loss is measured against a MOVING target, so it is not comparable across
    epochs and only loosely tied to what the downstream probe reads. With
    ``--snapshot_every N`` every N-th epoch is also kept as
    ``snapshots/ep<NNNN>/best.ckpt`` (same schema, so every ckpt consumer can
    read it); ``scripts/classification/select_ckpt_by_probe.py`` then probes
    them on the VALIDATION split and promotes the best one. One directory per
    snapshot because the probe writes its json next to the checkpoint.
    """
    every = getattr(args, "snapshot_every", 0)
    # --snapshot_first K additionally keeps each of the first K epochs: latent
    # collapse, when it happens, happens in the first few epochs, long before
    # the first periodic snapshot (scripts/collapse/collapse_metrics.py).
    early = (epoch + 1) <= getattr(args, "snapshot_first", 0)
    if not early and (every <= 0 or (epoch + 1) % every != 0):
        return
    d = out_dir / "snapshots" / f"ep{epoch + 1:04d}"
    d.mkdir(parents=True, exist_ok=True)
    torch.save(_ckpt_payload(predictor, encoder, epoch, vl["loss"], args, ema), d / "best.ckpt")


@torch.no_grad()
def init_linear_skip_ridge(predictor, encoder, loader, history_size, device, lam=1e-2):
    """Set the predictor's linear skip (``--linear_skip``) to the closed-form
    ridge solution on the TRAINING windows: per future step k, the map from
    the flattened history latents [z_{t-H+1} .. z_t] (plus bias) to the
    displacement z_{t+k} - z_t, in the encoder's own latent space.

    A zero-initialised skip trained by gradient descent does not reach this
    linear floor before early stopping. Solving it exactly puts the predictor
    AT the floor before the first step; the transformer then only has to
    learn the nonlinear / structural residual. Valid for a frozen encoder
    (the latents are fixed); with --train_encoder it is a warm start that
    goes stale as the encoder moves. ``lam`` is relative to the mean diagonal
    of X^T X so it is scale-free. Masked nodes are irrelevant here: the skip
    is applied to visible-history nodes only at run time.
    """
    skip = getattr(predictor, "skip", None)
    if skip is None:
        raise SystemExit("--linear_skip_init ridge needs --linear_skip")
    encoder.eval()
    H, P = predictor.history_frames, predictor.pred_frames
    D = predictor.node_dim
    F = H * D
    XtX = torch.zeros(F + 1, F + 1, dtype=torch.float64, device=device)
    XtY = torch.zeros(P, F + 1, D, dtype=torch.float64, device=device)
    n_rows = 0
    for batch in loader:
        x_hist = batch["x_hist"].to(device); x_fut = batch["x_fut"].to(device)
        adj = batch["adj"].to(device); active = batch["active"].to(device)
        adj_norm = normalize_adjacency(adj)
        z_full = encoder(torch.cat([x_hist, x_fut], dim=1), adj_norm)        # (B,T,N,D)
        z_hist = encoder(x_hist, adj_norm[:, :H])                              # (B,H,N,D)
        B, T_total, N = active.shape
        # rows = nodes active at EVERY future step (one X row serves all k)
        act = active[:, H:].all(dim=1)                                         # (B,N)
        Xb = z_hist.permute(0, 2, 1, 3).reshape(B, N, F)[act].double()         # (R,F)
        Xb = torch.cat([Xb, torch.ones(Xb.shape[0], 1, dtype=Xb.dtype, device=device)], 1)
        last = z_hist[:, -1][act].double()                                     # (R,D)
        XtX += Xb.T @ Xb
        for k in range(P):
            Yk = z_full[:, H + k][act].double() - last
            XtY[k] += Xb.T @ Yk
        n_rows += Xb.shape[0]
    reg = lam * XtX.diagonal()[:-1].mean()
    A = XtX + reg * torch.eye(F + 1, dtype=XtX.dtype, device=device)
    A[-1, -1] -= reg                                                            # no penalty on the bias
    W = torch.linalg.solve(A, XtY.reshape(P, F + 1, D).permute(1, 0, 2).reshape(F + 1, P * D))
    W = W.float()                                                              # (F+1, P*D)
    skip.weight.copy_(W[:-1].T)                                                # (P*D, F)
    skip.bias.copy_(W[-1])
    print(f"[skip] ridge-initialised linear skip from {n_rows} training rows "
          f"(F={F} -> P*D={P * D}, lam={lam})", flush=True)


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
    predictor = build_jepa_predictor(args, train_ds.num_nodes, device,
                                     feature_dim=train_ds.feature_dim)
    decoder = build_obs_decoder(args, train_ds.feature_dim, device)
    if args.linear_skip and args.linear_skip_init == "ridge":
        if args.train_encoder:
            print("[skip] WARNING: ridge init with --train_encoder is a warm start only; "
                  "the encoder will move away from the latents it was fitted on", flush=True)
        init_linear_skip_ridge(predictor, encoder, train_loader, args.history_size, device,
                               lam=args.linear_skip_lam)
    if args.linear_skip and args.linear_skip_freeze:
        for q in predictor.skip.parameters():
            q.requires_grad_(False)
        print("[skip] linear skip frozen", flush=True)

    pred_params = [q for q in predictor.parameters() if q.requires_grad]
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
                            horizon_norm=args.horizon_norm,
        **jepa_variant_kwargs(args))

    if args.smoke:
        dblp_train_epoch(predictor, encoder, train_loader, opt, args.history_size,
                         device, log_every=1, max_steps=3, scheduler=sched,
                         **train_kwargs)
        smoke_done(args, train_ds); return

    best_val = float("inf")
    no_improve = 0
    for epoch in range(args.epochs):
        print(f"[epoch {epoch+1}/{args.epochs}]", flush=True)
        set_encoder_frozen(args, epoch, encoder, ema)
        tr = dblp_train_epoch(predictor, encoder, train_loader, opt,
                              args.history_size, device, scheduler=sched, **train_kwargs)
        vl = dblp_eval_jepa(predictor, encoder, val_loader, args.history_size,
                            device, decoder=decoder, ema_target=ema,
                            change_weight=args.change_weight,
                            horizon_norm=args.horizon_norm,
                            mask_encoder_input=args.mask_encoder_input,
                            target_norm=args.target_norm)
        best_val, no_improve, improved, stop = _update_best(
            vl, best_val, no_improve, epoch, args, predictor, encoder, out_dir, ema=ema)
        _save_snapshot(epoch, vl, args, predictor, encoder, out_dir, ema=ema)
        _append_history(out_dir, {
            "epoch": epoch + 1, "train_loss": tr["loss"], "val_loss": vl["loss"],
            "lr": opt.param_groups[0]["lr"], "best": improved})
        print(f"  train loss={tr['loss']:.4f}  val loss={vl['loss']:.4f}"
              f"{'  (best)' if improved else ''}", flush=True)
        if stop:
            print(f"  early stopping: no val improvement for "
                  f"{no_improve} epochs", flush=True)
            break

    # Downstream probe stage (fit + evaluate + report).
    # --probe_split val during a hyperparameter sweep, so configurations are
    # ranked without ever reading the test split; test for the final table.
    if args.downstream_task == "none":
        # Pretext-only run (e.g. the node-state horizon showcase, whose probe
        # is scripts/dblp/run_block_probe.py on best.ckpt): skip the link
        # probe but still write test_metrics.json, the launchers' marker that
        # training reached its end.
        print("[probe] downstream_task=none -> no link-prediction probe", flush=True)
        probe_stage._save_metrics({"downstream_task": "none"}, out_dir)
        return
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
        smoke_done(args, train_ds); return

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
        _save_snapshot(epoch, vl, args, predictor, encoder, out_dir)
        _append_history(out_dir, {
            "epoch": epoch + 1, "train_loss": tr["loss"], "val_loss": vl["loss"],
            "lr": opt.param_groups[0]["lr"], "best": improved})
        print(f"  train loss={tr['loss']:.4f}  val loss={vl['loss']:.4f}"
              f"{'  (best)' if improved else ''}", flush=True)
        if stop:
            print(f"  early stopping: no val improvement for "
                  f"{no_improve} epochs", flush=True)
            break

    # Downstream probe stage (fit + evaluate + report).
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
    train_kwargs = dict(change_weight=args.change_weight,
                            horizon_norm=args.horizon_norm, ema_target=ema,
                        train_encoder=args.train_encoder, decoder=decoder,
                        vicreg_var_weight=args.vicreg_var,
                        vicreg_cov_weight=args.vicreg_cov,
                        vicreg_gamma=args.vicreg_gamma,
                        **jepa_variant_kwargs(args))

    if args.smoke:
        dblp_train_epoch(predictor, encoder, train_loader, opt, args.history_size,
                         device, log_every=1, max_steps=3, scheduler=sched,
                         **train_kwargs)
        smoke_done(args, train_ds); return

    best_val = float("inf")
    no_improve = 0
    for epoch in range(args.epochs):
        print(f"[epoch {epoch+1}/{args.epochs}]", flush=True)
        set_encoder_frozen(args, epoch, encoder, ema)
        tr = dblp_train_epoch(predictor, encoder, train_loader, opt,
                              args.history_size, device, scheduler=sched,
                              **train_kwargs)
        vl = dblp_eval_jepa(predictor, encoder, val_loader, args.history_size,
                            device, decoder=decoder, ema_target=ema,
                            change_weight=args.change_weight,
                            horizon_norm=args.horizon_norm,
                            mask_encoder_input=args.mask_encoder_input,
                            target_norm=args.target_norm)
        best_val, no_improve, improved, stop = _update_best(
            vl, best_val, no_improve, epoch, args, predictor, encoder, out_dir, ema=ema)
        _save_snapshot(epoch, vl, args, predictor, encoder, out_dir, ema=ema)
        _append_history(out_dir, {
            "epoch": epoch + 1, "train_loss": tr["loss"], "val_loss": vl["loss"],
            "lr": opt.param_groups[0]["lr"], "best": improved})
        print(f"  train loss={tr['loss']:.4f}  val loss={vl['loss']:.4f}"
              f"{'  (best)' if improved else ''}", flush=True)
        if stop:
            print(f"  early stopping: no val improvement for "
                  f"{no_improve} epochs", flush=True)
            break

    # Downstream probe stage (fit + evaluate + report).
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
            smoke_done(args, train_ds); return
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
    p.add_argument("--mask_frac", type=float, default=0.0,
                   help="Mask this FRACTION of the maskable nodes (active at the "
                        "first history step in every window of the batch) "
                        "instead of --num_masked_nodes. 0 = off (use the count). "
                        "I-JEPA / V-JEPA mask half or more of the tokens; a "
                        "count of 20-50 is under 1%% of a 7k-node graph.")
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
    p.add_argument("--linear_skip_init", choices=["zero", "ridge"], default="zero",
                   help="How the linear skip starts: zero (learned by gradient descent, "
                        "legacy) or ridge (closed-form least squares on the training "
                        "windows' latents; see init_linear_skip_ridge).")
    p.add_argument("--linear_skip_lam", type=float, default=1e-2,
                   help="Ridge strength for --linear_skip_init ridge, relative to the mean "
                        "diagonal of X^T X.")
    p.add_argument("--linear_skip_freeze", action="store_true",
                   help="Keep the linear skip fixed during training (with the ridge init: "
                        "the transformer learns only the residual).")
    p.add_argument("--linear_skip", action="store_true",
                   help="Linear-dynamics prior: a zero-init per-node linear map from the "
                        "flattened history latents to every future step's displacement, "
                        "added to the predictor output (masked nodes excluded). Saved in "
                        "args and restored by predictor_flags_from_args.")
    p.add_argument("--input_delta", action="store_true",
                   help="Velocity channel: add a learned (zero-init) projection of "
                        "z_t - z_{t-1} to every visible history token, so the "
                        "predictor receives consecutive-latent differences instead "
                        "of having to assemble them through attention. Saved in "
                        "args and restored by predictor_flags_from_args.")
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
    p.add_argument("--ema_anneal_epochs", type=int, default=0,
                   help="Epochs over which the cosine momentum anneal runs "
                        "(0 = the whole --epochs budget). Under early "
                        "stopping the whole-run horizon is rarely reached, "
                        "so set this below a typical run length for the "
                        "schedule to actually finish. Counted in encoder "
                        "updates, so it starts after --encoder_freeze_epochs.")
    p.add_argument("--encoder_freeze_epochs", type=int, default=0,
                   help="With --train_encoder: keep the encoder frozen (no "
                        "gradient, no EMA update) for the first K epochs so "
                        "the predictor fits the pretrained target before the "
                        "encoder starts moving. 0 = off.")
    # JEPA variants (all off by default)
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
    p.add_argument("--horizon_norm", action="store_true",
                   help="Divide each future step's latent MSE by the copy-last MSE at "
                        "that step (batch statistic, no gradient), so every horizon "
                        "weighs the same and the future loss is error relative to "
                        "persistence (1 = zero update). Off = raw MSE (legacy).")
    p.add_argument("--change_weight", type=float, default=0.0,
                   help="Weight per-(t,n) latent MSE by 1 + change_weight * "
                        "||z_t - z_{t-1}|| / mean: upweights the rare "
                        "transition-carrying nodes the unweighted average "
                        "drowns out. 0 = off (legacy behaviour).")
    p.add_argument("--community_masking", action="store_true")
    # two-level (coarse -> fine) DELOS, src/world_models/hier.py
    p.add_argument("--coarse_window", type=int, default=1,
                   help="Train the COARSE level: every window is reduced to one "
                        "step per W fine snapshots, and --history_size / "
                        "--num_preds count COARSE steps (the data window is W "
                        "times longer). 1 = off (ordinary run).")
    p.add_argument("--coarse_mode", choices=["last", "mean"], default="last",
                   help="How W fine snapshots become one coarse step: last = keep "
                        "the last one (subsample); mean = average features, union "
                        "of edges (re-bin).")
    p.add_argument("--hier_ckpt", default=None,
                   help="best.ckpt of a --coarse_window run on the same data. Its "
                        "frozen forecast for the coarse window containing each "
                        "future step is added to that step's query token "
                        "(zero-initialised projection). This run's window must be "
                        "the coarse run's window times its --coarse_window.")
    p.add_argument("--hier_hint", choices=["forecast", "true", "shuffle"],
                   default="forecast",
                   help="forecast = the coarse model's forecast (the method). "
                        "DIAGNOSTICS: true = the coarse encoding of the TRUE "
                        "future (ceiling; reads future snapshots); shuffle = the "
                        "forecast with nodes permuted (capacity control).")
    p.add_argument("--snapshot_every", type=int, default=0,
                   help="Also keep every N-th epoch as snapshots/ep<NNNN>/"
                        "best.ckpt so the checkpoint can be chosen on the "
                        "VALIDATION PROBE instead of the val JEPA loss "
                        "(scripts/classification/select_ckpt_by_probe.py). "
                        "0 = off.")
    p.add_argument("--snapshot_first", type=int, default=0,
                   help="Also keep each of the first K epochs as a snapshot "
                        "(dense early coverage for the collapse-dynamics "
                        "curves). 0 = off.")
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
    if args.encoder_freeze_epochs > 0 and not args.train_encoder:
        raise SystemExit("--encoder_freeze_epochs only applies with "
                         "--train_encoder (a frozen encoder is the default).")
    if args.encoder_freeze_epochs >= args.epochs > 0:
        raise SystemExit(f"--encoder_freeze_epochs {args.encoder_freeze_epochs} "
                         f">= --epochs {args.epochs}: the encoder would never "
                         f"train; use the frozen arm instead.")
    if args.target_grad and (not args.train_encoder or args.ema_target
                             or args.predict_space != "latent"):
        raise SystemExit("--target_grad requires --train_encoder, no "
                         "--ema_target and --predict_space latent.")
    if args.hier_ckpt and args.coarse_window > 1:
        raise SystemExit("--hier_ckpt with --coarse_window: only two levels "
                         "are supported")
    if args.hier_ckpt and args.mask_scheme not in ("node_mask", "future_forecast"):
        raise SystemExit("--hier_ckpt needs --mask_scheme node_mask / future_forecast")
    if args.mask_scheme == "trajectory_block":
        run_trajectory(args, device, out_dir)
    elif args.downstream_task == "regression":
        if args.data_source in DYNAMIC_REGRESSION_SOURCES:
            run_regression_dynamic(args, device, out_dir)
        else:
            run_regression(args, device, out_dir)
    elif args.downstream_task in ("link_pred", "none"):
        run_link_pred(args, device, out_dir)      # none: same loop, no probe
    else:
        raise ValueError(
            f"downstream_task={args.downstream_task!r} with mask_scheme="
            f"{args.mask_scheme!r} is not a supported combination. Use "
            f"trajectory_block for pretext-only, or pick link_pred/regression.")


if __name__ == "__main__":
    main()
