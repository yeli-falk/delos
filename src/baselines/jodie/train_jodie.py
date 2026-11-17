"""Train the JODIE baseline on a JODIE CSV, then dump per-bin embeddings.

Protocol (matched to every other baseline for direct comparability):
  - SAME node set: the full dataset, ``bipartite_node_ids`` -> users are
    node ids [0, n_users), items [n_users, n_users+n_items) -- identical to
    ``bin_bipartite`` and to the state-label matrix.
  - SAME bin edges: equal-width over all event timestamps (identical to the
    graph's snapshots and labels).
  - SAME training range: only events in bins [0, 0.7*num_bins) are trained on
    (the chronological train split every method uses); no state labels are
    ever seen (self-supervised).
  - SAME model selection: the val bins [0.7, 0.8) score every epoch and the
    best-val weights are restored, with the shared --early_stop_patience rule
    -- identical to DELOS and to the CLDG / DDGCL / IDOL baselines.
  - SAME evaluation: after training, the event stream is replayed with frozen
    weights and the DYNAMIC embeddings are snapshotted at the END of every
    bin. ``run_block_probe.py --jodie_ckpt`` then builds the standard columns
    -- embedding at the last history bin (persistence), the PROJECTED
    embedding at the target bin (JODIE's forecast = its `jepa` column), and
    the embedding at the end of the target bin (oracle) -- and fits the
    identical deterministic probe, writing the same state_probe.json.

Usage:
    python -m src.baselines.jodie.train_jodie \
        --data_source jodie_mooc --jodie_data_path dataset/jodie/mooc.csv \
        --jodie_num_bins 30 \
        --with_edge_features --node_dim 128 --epochs 30 --device cuda \
        --out_dir outputs/jodie_mooc

    python scripts/dblp/run_block_probe.py --jodie_ckpt outputs/jodie_mooc/jodie.ckpt \
        --task binary --state_label_mode state
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F

from src.baselines.common import SPLITS_FALLBACK, BestValTracker
from src.baselines.jodie.jodie import JODIE
from src.custom_codes.dataset_factory import add_dataset_cli_flags
from src.custom_codes.dtdg.binning import bipartite_node_ids
from src.custom_codes.jodie_dataset import (
    _parse_jodie_csv,
    _parse_jodie_csv_with_features,
)
from src.train import cli
from src.train.engine import set_deterministic, setup_output_dir


# ---------------------------------------------------------------------------
# Event preparation (mirrors bin_bipartite's node ids + bin edges EXACTLY)
# ---------------------------------------------------------------------------
def load_events(args):
    """Return time-sorted dense-indexed events + bin metadata.

    Returns dict with users, items (dense ids), times, feats (E, F), bins,
    n_users, n_items, t_lo, span, num_bins.
    """
    from pathlib import Path
    path = Path(args.jodie_data_path).expanduser()
    if args.with_edge_features:
        users, items, times, feats = _parse_jodie_csv_with_features(path)
    else:
        users, items, times = _parse_jodie_csv(path)
        feats = None

    keep_u_arr, keep_i_arr = bipartite_node_ids(users, items)
    u = np.searchsorted(keep_u_arr, users)
    i = np.searchsorted(keep_i_arr, items)
    t = times
    f = feats if feats is not None else np.zeros((len(t), 0), np.float32)

    order = np.argsort(t, kind="stable")
    u, i, t, f = u[order], i[order], t[order], f[order]

    num_bins = args.jodie_num_bins
    t_lo = float(t.min())
    span = max(float(t.max()) - t_lo, 1.0)
    bins = np.clip(((t - t_lo) / span * num_bins).astype(np.int64),
                   0, num_bins - 1)
    return dict(users=u, items=i, times=t, feats=f, bins=bins,
                n_users=len(keep_u_arr), n_items=len(keep_i_arr),
                t_lo=t_lo, span=span, num_bins=num_bins)


def mean_gap(entity_ids: np.ndarray, times: np.ndarray, n: int) -> float:
    """Mean positive inter-event time gap per entity (JODIE's dt normaliser)."""
    order = np.argsort(entity_ids, kind="stable")   # times already sorted
    e_s, t_s = entity_ids[order], times[order]
    same = e_s[1:] == e_s[:-1]
    gaps = (t_s[1:] - t_s[:-1])[same]
    gaps = gaps[gaps > 0]
    return float(gaps.mean()) if gaps.size else 1.0


def tbatch_groups(u: np.ndarray, i: np.ndarray, n_users: int, n_items: int):
    """Greedy t-batching (reference algorithm) over time-sorted events.

    batch(e) = max(last_batch[user_e], last_batch[item_e]) + 1, so events in
    one group touch disjoint users AND items -> the vectorised update equals
    sequential processing, and each entity's events stay chronological.
    Returns a list of index arrays (one per t-batch, in order).
    """
    last = np.full(n_users + n_items, -1, dtype=np.int64)
    bid = np.empty(len(u), dtype=np.int64)
    for e in range(len(u)):
        b = max(last[u[e]], last[n_users + i[e]]) + 1
        bid[e] = b
        last[u[e]] = b
        last[n_users + i[e]] = b
    order = np.argsort(bid, kind="stable")
    bounds = np.flatnonzero(np.diff(bid[order])) + 1
    return np.split(order, bounds)


# ---------------------------------------------------------------------------
# One pass over an event range (shared by train / frozen replay)
# ---------------------------------------------------------------------------
def run_stream(model, ev, sel, device, args, train=False, opt=None,
               snapshot_bins=None, loss_from_bin=0):
    """Process events ``sel`` (a boolean mask over ev) bin by bin, t-batched.

    train=True: chunked BPTT -- accumulate loss over ``args.chunk`` t-batches,
    then backward/step and detach the embedding banks.
    snapshot_bins: if a dict of output arrays is given, record the banks and
    per-user last-interaction times at the END of every bin (frozen replay).
    loss_from_bin: only bins >= this index contribute to the returned mean.
    Used for val scoring -- the train bins are replayed to warm the embedding
    banks (JODIE is stateful, so a cold start would not be the model's real
    val behaviour) but only the held-out bins are scored.
    Returns the mean loss per t-batch over the scored bins.
    """
    n_u, n_i = ev["n_users"], ev["n_items"]
    D = model.emb_dim
    bank_u = model.init_user.unsqueeze(0).expand(n_u, D)
    bank_i = model.init_item.unsqueeze(0).expand(n_i, D)
    if not train:
        bank_u, bank_i = bank_u.detach().clone(), bank_i.detach().clone()
    last_t_u = np.full(n_u, ev["t_lo"])
    last_t_i = np.full(n_i, ev["t_lo"])
    prev_item = torch.full((n_u,), n_i, dtype=torch.long, device=device)  # null

    gap_u = max(ev["mean_gap_u"], 1e-9)
    gap_i = max(ev["mean_gap_i"], 1e-9)
    agg, nb, pending = 0.0, 0, 0
    loss_acc = torch.zeros((), device=device)

    for b in range(ev["num_bins"]):
        in_bin = sel & (ev["bins"] == b)
        idx_bin = np.flatnonzero(in_bin)
        for grp in (tbatch_groups(ev["users"][idx_bin], ev["items"][idx_bin],
                                  n_u, n_i) if idx_bin.size else []):
            g = idx_bin[grp]
            ue = torch.from_numpy(ev["users"][g]).long().to(device)
            ie = torch.from_numpy(ev["items"][g]).long().to(device)
            te = ev["times"][g]
            fe = torch.from_numpy(ev["feats"][g]).float().to(device)
            dt_u = torch.from_numpy(
                (te - last_t_u[ev["users"][g]]) / gap_u).float().to(device)
            dt_i = torch.from_numpy(
                (te - last_t_i[ev["items"][g]]) / gap_i).float().to(device)

            u_emb, i_emb = bank_u[ue], bank_i[ie]
            u_proj = model.project(u_emb, dt_u)
            pidx = prev_item[ue]
            prev_dyn = torch.where(
                (pidx == n_i).unsqueeze(-1),
                torch.zeros_like(u_emb),
                bank_i[pidx.clamp(max=n_i - 1)],
            )
            loss = model.prediction_loss(u_proj, ue, prev_dyn, pidx, i_emb, ie)
            u_new, i_new = model.step(u_emb, i_emb, fe, dt_u, dt_i)
            loss = loss + args.reg * (F.mse_loss(u_new, u_emb.detach())
                                      + F.mse_loss(i_new, i_emb.detach()))

            bank_u = bank_u.index_copy(0, ue, u_new)
            bank_i = bank_i.index_copy(0, ie, i_new)
            last_t_u[ev["users"][g]] = te
            last_t_i[ev["items"][g]] = te
            prev_item[ue] = ie

            if train:
                loss_acc = loss_acc + loss
                pending += 1
                if pending >= args.chunk:
                    opt.zero_grad(set_to_none=True)
                    (loss_acc / pending).backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                    bank_u, bank_i = bank_u.detach(), bank_i.detach()
                    loss_acc = torch.zeros((), device=device)
                    pending = 0
            if b >= loss_from_bin:
                agg += float(loss.detach())
                nb += 1
            if args.smoke and nb >= 5:
                break
        if snapshot_bins is not None:
            snapshot_bins["user_emb"][b] = bank_u.detach().cpu().numpy()
            snapshot_bins["item_emb"][b] = bank_i.detach().cpu().numpy()
            snapshot_bins["last_t_u"][b] = last_t_u
        if args.smoke and nb >= 5 and snapshot_bins is None:
            break
    if train and pending > 0:                       # flush the tail chunk
        opt.zero_grad(set_to_none=True)
        (loss_acc / pending).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return agg / max(nb, 1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    add_dataset_cli_flags(p)
    cli.add_window_flags(p)                 # recorded for the probe's dataset
    cli.add_run_flags(p, out_dir="outputs/jodie")
    cli.add_early_stop_flags(p, patience=20, min_delta=0.0)
    p.add_argument("--node_dim", type=int, default=128,
                   help="Dynamic embedding dim (match every other method).")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--reg", type=float, default=1.0,
                   help="Embedding-drift regularizer weight (reference: 1.0).")
    p.add_argument("--chunk", type=int, default=64,
                   help="t-batches per BPTT chunk before detaching the banks.")
    p.add_argument("--val_every", type=int, default=1,
                   help="Score the val bins every N epochs. Val is a second "
                        "full stream pass (~2.2x the epoch cost on MOOC), so "
                        "N=3-5 makes a sweep affordable. The final epoch is "
                        "always scored. With N>1, --early_stop_patience counts "
                        "evaluations rather than epochs.")
    args = p.parse_args()
    if args.jodie_data_path is None:
        ds_name = args.data_source.replace("jodie_", "")
        args.jodie_data_path = f"dataset/jodie/{ds_name}.csv"

    set_deterministic(args.seed)
    out_dir = setup_output_dir(args.out_dir, args)
    device = torch.device(args.device)

    ev = load_events(args)
    # Chronological 70/10/20, identical to DTDGDataset.SPLITS so JODIE trains
    # on the same bins as every snapshot method and selects on the same
    # held-out bins.
    splits = SPLITS_FALLBACK
    train_end_bin = max(1, int(ev["num_bins"] * splits[0]))
    val_end_bin = min(ev["num_bins"],
                      train_end_bin + int(ev["num_bins"] * splits[1]))
    if val_end_bin <= train_end_bin:
        raise SystemExit(
            f"JODIE: the val split is empty (num_bins={ev['num_bins']}, "
            f"train_end_bin={train_end_bin}). Model selection needs at least "
            f"one val bin — raise --jodie_num_bins.")
    train_sel = ev["bins"] < train_end_bin
    val_sel = ev["bins"] < val_end_bin           # replay train, score val bins
    all_sel = np.ones(len(ev["users"]), dtype=bool)
    ev["mean_gap_u"] = mean_gap(ev["users"][train_sel], ev["times"][train_sel],
                                ev["n_users"])
    ev["mean_gap_i"] = mean_gap(ev["items"][train_sel], ev["times"][train_sel],
                                ev["n_items"])
    print(f"[data] kept events={len(ev['users'])} (train={int(train_sel.sum())}, "
          f"bins<{train_end_bin}; val bins [{train_end_bin},{val_end_bin}))  "
          f"n_users={ev['n_users']} n_items={ev['n_items']} "
          f"gap_u={ev['mean_gap_u']:.1f} gap_i={ev['mean_gap_i']:.1f}", flush=True)

    model = JODIE(ev["n_users"], ev["n_items"], emb_dim=args.node_dim,
                  feat_dim=ev["feats"].shape[1]).to(device)
    n_params = sum(q.numel() for q in model.parameters() if q.requires_grad)
    print(f"[model] JODIE trainable={n_params:,}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)

    tracker = BestValTracker(args.early_stop_patience, args.early_stop_min_delta)
    n_epochs = 1 if args.smoke else args.epochs
    for epoch in range(n_epochs):
        model.train()
        t0 = time.time()
        train_loss = run_stream(model, ev, train_sel, device, args,
                                train=True, opt=opt)

        is_last = (epoch + 1) == n_epochs
        if not (is_last or (epoch + 1) % args.val_every == 0):
            print(f"[epoch {epoch+1}/{n_epochs}] loss={train_loss:.4f} "
                  f"(no val this epoch) ({time.time()-t0:.1f}s)", flush=True)
            continue
        model.eval()
        with torch.no_grad():
            val_loss = run_stream(model, ev, val_sel, device, args,
                                  loss_from_bin=train_end_bin)
        improved, stop = tracker.update(val_loss, model)
        print(f"[epoch {epoch+1}/{n_epochs}] loss={train_loss:.4f} "
              f"val_loss={val_loss:.4f}{'  (best)' if improved else ''} "
              f"({time.time()-t0:.1f}s)", flush=True)
        if stop:
            print(f"  early stopping: no val improvement for "
                  f"{args.early_stop_patience} evaluations", flush=True)
            break
    best = tracker.best
    if tracker.restore(model):
        print(f"[best] restored best-val weights (val_loss={best:.4f})", flush=True)

    model.eval()
    snaps = dict(
        user_emb=np.zeros((ev["num_bins"], ev["n_users"], args.node_dim), np.float32),
        item_emb=np.zeros((ev["num_bins"], ev["n_items"], args.node_dim), np.float32),
        last_t_u=np.zeros((ev["num_bins"], ev["n_users"]), np.float64),
    )
    with torch.no_grad():
        run_stream(model, ev, all_sel, device, args, snapshot_bins=snaps)

    ckpt_path = out_dir / "jodie.ckpt"
    torch.save({
        "model": model.state_dict(),
        "n_users": ev["n_users"], "n_items": ev["n_items"],
        "emb_dim": args.node_dim, "feat_dim": ev["feats"].shape[1],
        "user_emb_bins": snaps["user_emb"], "item_emb_bins": snaps["item_emb"],
        "last_t_user_bins": snaps["last_t_u"],
        "t_lo": ev["t_lo"], "span": ev["span"], "num_bins": ev["num_bins"],
        "mean_gap_user": ev["mean_gap_u"], "mean_gap_item": ev["mean_gap_i"],
        "args": vars(args), "pretrain_objective": "jodie", "best_loss": best,
        "val_loss": best,
    }, ckpt_path)
    print(f"[done] best val loss={best:.4f} -> {ckpt_path}\n"
          f"[eval] probe: python scripts/dblp/run_block_probe.py "
          f"--jodie_ckpt {ckpt_path} --task binary --state_label_mode state",
          flush=True)


if __name__ == "__main__":
    main()
