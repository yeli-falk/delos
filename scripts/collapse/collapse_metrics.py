"""Representation-collapse metrics for every run of a results root.

"No collapse" is three separate claims on a dynamic graph, and each gets its
own numbers here, all measured on held-out windows (test split by default):

  1. not a point        the embeddings of different nodes differ
                        -> spread, cos
  2. not low-rank       they span many directions, not one
                        -> rankme, eff_rank_cov, spectrum, dead_frac
  3. not static in time a node's embedding moves between snapshots, and the
                        predictor does more than copy the last one
                        -> within_frac, step_ratio, forecast (ratio / disp_*)

Rows are (window, forecast step, node) with the node ACTIVE at that step, so
isolated nodes (which every encoder maps to the same vector) never enter and
cannot fake a collapse.

Three views of each run:
  enc    encoder at the target step with the full window visible (the latent
         the JEPA objective regresses onto); horizon 0 only
  last   encoder at the last observed step (what the frozen probe reads for
         every baseline: `encoder_last`); horizon 0 only
  pred   the predictor's forecast latent, all horizons (DELOS runs only)

Metrics per view (Z = rows x D, at most --max_rows rows, fixed subsample):
  rankme        exp(entropy of s / sum(s)), s = singular values of the
                centred Z (Garrido et al., ICML'23). 1 = one direction,
                D = isotropic. Centred, so a large shared mean vector does not
                count as "the" direction; a point collapse is caught by spread.
  eff_rank_cov  the same with s^2 (Roy & Vetterli; the eff_rank that
                src/train/analyze_jepa.py reports)
  spectrum      s / s[0], all D values (the log-spectrum figure)
  spread        ||Z - mean||_F / ||Z||_F in [0, 1]: share of the embedding
                energy that is NOT the common mean. 0 = every row is the same
                point. Scale-free, so a LayerNorm cannot hide a collapse.
  std           mean per-dimension std (raw scale; compare within a method)
  cos           mean cosine between random row pairs (1 = one direction)
  dead_frac     share of dimensions whose std is < 1% of the largest one
  snap_rankme   rankme computed inside each snapshot (one window + step,
                across its nodes) and averaged: pooling all rows could hide a
                per-snapshot collapse behind drift between snapshots
Temporal (enc view, nodes seen in >= 2 windows):
  within_frac   within-node (over time) share of the total variance.
                ~0 = the embedding is a static node code that ignores time.
  step_ratio    E||z(target) - z(last observed)||^2 / E||z_i - z_j||^2 over
                random pairs: how far a node moves in one step relative to
                the distance between two random nodes.
Forecast (DELOS only, per horizon; scripts/synthetic/predictor_latent_error.py):
  ratio         mse(forecast, target) / mse(copy-last, target); 1 = the
                predictor returns its input unchanged
  disp_mag      ||forecast - last|| / ||target - last||; 0 = pure copy
  disp_cos      cosine between predicted and true displacement

Reference rows (written to <root>/_collapse/<name>_s<seed>/), per seed:
  ref_pretrain  the shared edge-mask encoder every DELOS arm starts from
  ref_random    the same architecture at random initialisation (a random
                network is high-rank too: rank alone is not evidence of
                learning, so read it next to the probe score)
  ref_input     the raw node features at the last observed step (D = feature
                dim; the rank the data itself has)

Training dynamics: for a run with snapshots/ep<NNNN>/best.ckpt (trainer flags
--snapshot_every / --snapshot_first) the scalar metrics are computed at every
snapshot and written to <run>/collapse_dynamics.json.

Usage:
    python scripts/collapse/collapse_metrics.py --root outputs/collapse/mooc
    python scripts/collapse/collapse_metrics.py --root outputs/cls_mooc --force
Writes <run>/collapse.json (+ collapse_dynamics.json) next to each checkpoint;
tables and figures come from scripts/collapse/collapse_report.py.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.dblp.run_block_probe import (  # noqa: E402
    _make_args_namespace, collect_encoder_latents, collect_test_latents)
from scripts.synthetic.predictor_latent_error import latent_error  # noqa: E402
from src.delos_predictor import build_predictor, predictor_flags_from_args  # noqa: E402
from src.custom_codes.dataset_factory import build_dynamic_dataset  # noqa: E402
from src.world_models.delos_encoder import build_encoder  # noqa: E402

RUN_RE = re.compile(r"^(?P<stem>.+)_s(?P<seed>\d+)$")
SPLIT_SUFFIX = {"test": "", "val": "_val"}


# --- metrics ---
def _subsample(n: int, max_rows: int, rng) -> np.ndarray:
    return np.arange(n) if n <= max_rows else np.sort(rng.choice(n, max_rows, replace=False))


def _rank_from_sv(s: np.ndarray) -> tuple[float, float]:
    """(rankme, eff_rank_cov) from singular values; (1, 1) for a point."""
    if s.size == 0 or s.sum() < 1e-12:
        return 1.0, 1.0
    out = []
    for p in (s / s.sum(), s ** 2 / (s ** 2).sum()):
        out.append(float(np.exp(-(p * np.log(np.clip(p, 1e-12, None))).sum())))
    return out[0], out[1]


def view_metrics(Z: np.ndarray, group: np.ndarray, rng, max_rows: int,
                 n_pairs: int = 100_000, max_groups: int = 64,
                 spectrum: bool = True) -> dict:
    """Collapse metrics of one representation matrix (rows x D).

    `group` labels the snapshot each row belongs to (for snap_rankme)."""
    n, D = Z.shape
    keep = _subsample(n, max_rows, rng)
    Zs, gs = Z[keep], group[keep]
    mu = Zs.mean(0, keepdims=True)
    Zc = Zs - mu
    s = np.linalg.svd(Zc, compute_uv=False)
    rankme, eff_cov = _rank_from_sv(s)
    std = Zc.std(0)
    norm = np.linalg.norm(Zs)
    i, j = rng.integers(0, len(Zs), n_pairs), rng.integers(0, len(Zs), n_pairs)
    ok = i != j
    a, b = Zs[i[ok]], Zs[j[ok]]
    cos = (a * b).sum(1) / np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-12)

    snap = []
    ids = np.unique(gs)
    if len(ids) > max_groups:
        ids = ids[np.linspace(0, len(ids) - 1, max_groups).astype(int)]
    for g in ids:
        Zg = Zs[gs == g]
        if len(Zg) >= 8:
            snap.append(_rank_from_sv(np.linalg.svd(Zg - Zg.mean(0), compute_uv=False))[0])

    out = {
        "n_rows": int(n), "dim": int(D),
        "rankme": rankme, "eff_rank_cov": eff_cov,
        "spread": float(np.linalg.norm(Zc) / max(norm, 1e-12)),
        "std": float(std.mean()),
        "cos": float(cos.mean()),
        "dead_frac": float((std < 1e-2 * max(std.max(), 1e-12)).mean()),
        "snap_rankme": float(np.mean(snap)) if snap else None,
        "pair_dist2": float(((a - b) ** 2).sum(1).mean()),
    }
    if spectrum:
        sn = np.zeros(D)
        sn[:len(s)] = s[:D] / max(s[0], 1e-12)
        out["spectrum"] = [float(x) for x in sn]
    return out


def temporal_metrics(Z_enc, Z_last, node_id, pair_dist2: float) -> dict:
    """Between-node vs within-node-over-time variance of the encoder latent,
    and the one-step displacement relative to the node-pair distance."""
    order = np.argsort(node_id, kind="stable")
    Z, nid = Z_enc[order], node_id[order]
    uniq, start, counts = np.unique(nid, return_index=True, return_counts=True)
    multi = counts >= 2
    if not multi.any():
        return {"within_frac": None, "step_ratio": None, "n_nodes_multi": 0}
    sums = np.add.reduceat(Z, start, axis=0)
    means = sums / counts[:, None]
    rows = np.repeat(multi, counts)
    Zm, mean_rows = Z[rows], np.repeat(means, counts, axis=0)[rows]
    within = float(((Zm - mean_rows) ** 2).sum())
    total = float(((Zm - Zm.mean(0)) ** 2).sum())
    step = float(((Z_enc - Z_last) ** 2).sum(1).mean())
    return {
        "within_frac": within / max(total, 1e-12) if total > 1e-12 else None,
        "step_ratio": step / pair_dist2 if pair_dist2 > 1e-12 else None,
        "n_nodes_multi": int(multi.sum()),
    }


def run_metrics(lat: dict, rng, max_rows: int, spectrum: bool = True) -> dict:
    """All views + temporal + forecast blocks from collected latents."""
    h0 = lat["horizon"] == 0
    grp = lat["window"] * 1000 + lat["horizon"]
    res = {
        "enc": view_metrics(lat["Z_enc"][h0], grp[h0], rng, max_rows, spectrum=spectrum),
        "last": view_metrics(lat["Z_last"][h0], grp[h0], rng, max_rows, spectrum=False),
    }
    res["temporal"] = temporal_metrics(lat["Z_enc"][h0], lat["Z_last"][h0],
                                       lat["node_id"][h0], res["enc"]["pair_dist2"])
    if lat.get("Z_pred") is not None:
        res["pred"] = view_metrics(lat["Z_pred"], grp, rng, max_rows, spectrum=spectrum)
        res["forecast"] = latent_error(lat["Z_enc"], lat["Z_last"], lat["Z_pred"], lat["horizon"])
    return res


# --- checkpoint loading (mirrors run_block_probe.run_one / run_one_encoder) ---
def build_data(saved: dict, split: str, batch_size: int, num_workers: int, device: str):
    args = _make_args_namespace(saved)
    args.device, args.batch_size, args.num_workers, args.split = device, batch_size, num_workers, split
    ds = build_dynamic_dataset(args, split=split)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return ds, loader


def build_enc(ckpt: dict, ds, dev, load: bool = True, seed: int = 0):
    saved = ckpt.get("args", {})
    kind = ckpt.get("encoder_kind", saved.get("encoder_kind", "gcn"))
    if not load:
        torch.manual_seed(seed)
    enc = build_encoder(
        kind, in_dim=ds.feature_dim,
        hidden_dim=ckpt.get("encoder_hidden", saved.get("encoder_hidden", 128)),
        out_dim=ckpt.get("node_dim", saved.get("node_dim", 128)), freeze=True,
        bidirectional=ckpt.get("bidirectional", saved.get("bidirectional", True))).to(dev)
    if load:
        enc.load_state_dict(ckpt["encoder"])
    return enc


def build_pred(ckpt: dict, ds, dev):
    saved = ckpt["args"]
    node_dim = saved.get("node_dim", 128)
    pred = build_predictor(
        saved.get("mask_scheme", "node_mask"), num_nodes=ds.num_nodes, node_dim=node_dim,
        history_frames=saved.get("history_size", 3), pred_frames=saved.get("num_preds", 1),
        num_masked_nodes=saved.get("num_masked_nodes", 30), depth=saved.get("depth", 4),
        heads=saved.get("heads", 4), dim_head=node_dim // saved.get("heads", 4),
        mlp_dim=saved.get("mlp_dim", 512), dropout=saved.get("dropout", 0.1),
        use_attention_mask=not saved.get("no_attention_mask", True),
        **predictor_flags_from_args(saved)).to(dev)
    pred.load_state_dict(ckpt["predictor"])
    return pred


def collect(encoder, predictor, loader, hist: int, dev) -> dict:
    if predictor is not None:
        Z_enc, Z_last, Z_pred, window, node_id, _, _, _, horizon, _ = collect_test_latents(
            encoder, predictor, loader, history_size=hist, device=dev)
    else:
        Z_enc, Z_last, window, node_id, _, _, _, horizon, _ = collect_encoder_latents(
            encoder, loader, history_size=hist, device=dev)
        Z_pred = None
    return dict(Z_enc=Z_enc, Z_last=Z_last, Z_pred=Z_pred, window=window,
                node_id=node_id, horizon=horizon)


@torch.no_grad()
def collect_input(loader) -> dict:
    """Raw node features as the 'representation': target-step features as the
    enc view, last-observed-step features as the last view (horizon 0 only)."""
    enc, last, win, nid = [], [], [], []
    for wid, batch in enumerate(loader):
        x_hist, x_fut, active = batch["x_hist"], batch["x_fut"], batch["active"]
        H = x_hist.shape[1]
        for b in range(active.shape[0]):
            idx = active[b, H].nonzero(as_tuple=True)[0]
            if idx.numel() < 2:
                continue
            enc.append(x_fut[b, 0, idx]); last.append(x_hist[b, -1, idx])
            win.append(torch.full((idx.numel(),), wid * 1000 + b)); nid.append(idx.long())
    n = sum(len(w) for w in win)
    return dict(Z_enc=torch.cat(enc).numpy().astype(np.float64),
                Z_last=torch.cat(last).numpy().astype(np.float64), Z_pred=None,
                window=torch.cat(win).numpy(), node_id=torch.cat(nid).numpy(),
                horizon=np.zeros(n, dtype=np.int64))


# --- per-run drivers ---
def _write(path: Path, blob) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(blob, f)


def _line(tag: str, res: dict) -> str:
    e, t, p = res["enc"], res.get("temporal", {}), res.get("pred")
    f = lambda v: "   -  " if v is None else f"{v:6.3f}"  # noqa: E731
    s = (f"{tag:<34} rankme {e['rankme']:6.1f}/{e['dim']:<4} spread {e['spread']:.3f} "
         f"cos {e['cos']:+.3f} within {f(t.get('within_frac'))}")
    if p is not None:
        s += f" | pred rankme {p['rankme']:6.1f} spread {p['spread']:.3f}"
    return s


def process_run(run: Path, a, dev) -> None:
    out = run / f"collapse{SPLIT_SUFFIX[a.split]}.json"
    dyn_out = run / f"collapse_dynamics{SPLIT_SUFFIX[a.split]}.json"
    is_jepa = (run / "best.ckpt").exists()
    ck_path = run / ("best.ckpt" if is_jepa else "encoder.ckpt")
    snaps = sorted(run.glob("snapshots/ep*/best.ckpt")) if is_jepa and not a.no_dynamics else []
    need_main = a.force or not out.exists()
    need_dyn = bool(snaps) and (a.force or not dyn_out.exists())
    if not (need_main or need_dyn):
        print(f"[skip] {run.name}", flush=True)
        return
    ckpt = torch.load(ck_path, map_location="cpu", weights_only=False)
    saved = ckpt.get("args", {})
    if not saved or "encoder" not in ckpt:
        print(f"[skip] {run.name}: checkpoint has no args / encoder", flush=True)
        return
    hist = saved.get("history_size", 3)
    ds, loader = build_data(saved, a.split, a.batch_size, a.num_workers, a.device)
    rng = np.random.default_rng(0)

    if need_main:
        enc = build_enc(ckpt, ds, dev)
        pred = build_pred(ckpt, ds, dev) if is_jepa else None
        res = run_metrics(collect(enc, pred, loader, hist, dev), rng, a.max_rows)
        res["_meta"] = {"ckpt": str(ck_path), "split": a.split, "kind": "jepa" if is_jepa else "encoder",
                        "epoch": ckpt.get("epoch"),
                        "flags": {k: saved.get(k) for k in (
                            "train_encoder", "ema_target", "ema_schedule", "target_norm", "target_grad",
                            "sigreg", "vicreg_var", "vicreg_cov", "residual_head", "mask_scheme",
                            "predict_space", "encoder_ckpt", "encoder_kind", "num_preds")}}
        _write(out, res)
        print(_line(run.name, res), flush=True)

    if need_dyn:
        rows = []
        for sp in snaps:
            ck = torch.load(sp, map_location="cpu", weights_only=False)
            r = run_metrics(collect(build_enc(ck, ds, dev), build_pred(ck, ds, dev), loader, hist, dev),
                            np.random.default_rng(0), a.max_rows, spectrum=False)
            r["epoch"] = int(ck.get("epoch", -1)) + 1
            r["val_loss"] = ck.get("val_loss")
            rows.append(r)
            print(f"   ep {r['epoch']:>4} enc rankme {r['enc']['rankme']:6.1f} spread {r['enc']['spread']:.3f}"
                  f" | pred rankme {r['pred']['rankme']:6.1f} spread {r['pred']['spread']:.3f}", flush=True)
        _write(dyn_out, sorted(rows, key=lambda r: r["epoch"]))


def process_refs(root: Path, seed: str, a, dev) -> None:
    """ref_pretrain / ref_random / ref_input for one seed, from the shared
    edge-mask encoder checkpoint of that seed (it carries the data config)."""
    cks = sorted((root / f"_shared_encoder_s{seed}").glob("encoder_*.ckpt"))
    if not cks:
        print(f"[refs] no shared encoder for seed {seed} -> no reference rows", flush=True)
        return
    name = f"collapse{SPLIT_SUFFIX[a.split]}.json"
    outs = {k: root / "_collapse" / f"ref_{k}_s{seed}" / name for k in ("pretrain", "random", "input")}
    if not a.force and all(p.exists() for p in outs.values()):
        print(f"[skip] reference rows s{seed}", flush=True)
        return
    ckpt = torch.load(cks[0], map_location="cpu", weights_only=False)
    saved = ckpt.get("args", {})
    hist = saved.get("history_size", 3)
    ds, loader = build_data(saved, a.split, a.batch_size, a.num_workers, a.device)
    for key, lat in (
            ("pretrain", lambda: collect(build_enc(ckpt, ds, dev), None, loader, hist, dev)),
            ("random", lambda: collect(build_enc(ckpt, ds, dev, load=False, seed=int(seed)), None, loader, hist, dev)),
            ("input", lambda: collect_input(loader))):
        res = run_metrics(lat(), np.random.default_rng(0), a.max_rows)
        res["_meta"] = {"ckpt": str(cks[0]), "split": a.split, "kind": f"ref_{key}"}
        _write(outs[key], res)
        print(_line(f"ref_{key}_s{seed}", res), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True, help="results root holding <stem>_s<seed>/ run directories")
    p.add_argument("--stems", nargs="*", default=None, help="only these stems (default: every run found)")
    p.add_argument("--seeds", nargs="*", default=None, help="only these seeds (default: every seed found)")
    p.add_argument("--split", choices=["test", "val"], default="test")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--max_rows", type=int, default=20000,
                   help="rows kept per view for the SVD / cosine statistics (fixed subsample)")
    p.add_argument("--no_dynamics", action="store_true", help="skip the per-snapshot curves")
    p.add_argument("--no_refs", action="store_true", help="skip the reference rows")
    p.add_argument("--force", action="store_true", help="recompute existing collapse*.json")
    a = p.parse_args()
    if a.device.startswith("cuda") and not torch.cuda.is_available():
        print("[device] no CUDA -> cpu", flush=True)
        a.device = "cpu"
    dev = torch.device(a.device)
    root = Path(a.root)

    runs, seeds = [], set()
    for d in sorted(root.iterdir()):
        m = RUN_RE.match(d.name)
        if not m or d.name.startswith("_") or not d.is_dir():
            continue
        if not ((d / "best.ckpt").exists() or (d / "encoder.ckpt").exists()):
            continue
        if a.stems and m["stem"] not in a.stems:
            continue
        if a.seeds and m["seed"] not in a.seeds:
            continue
        runs.append(d); seeds.add(m["seed"])
    if not runs:
        raise SystemExit(f"no <stem>_s<seed>/(best|encoder).ckpt under {root}")
    print(f"[collapse] {len(runs)} runs, seeds {sorted(seeds)}, split {a.split}, device {a.device}", flush=True)

    failed = []
    if not a.no_refs:
        for s in sorted(seeds):
            try:
                process_refs(root, s, a, dev)
            except Exception as e:  # noqa: BLE001  one broken run must not stop the sweep
                failed.append((f"refs s{s}", repr(e)))
    for d in runs:
        try:
            process_run(d, a, dev)
        except Exception as e:  # noqa: BLE001
            failed.append((d.name, repr(e)))
            print(f"[FAILED] {d.name}: {e!r}", flush=True)
    if failed:
        print(f"\n[collapse] {len(failed)} FAILED:")
        for n, e in failed:
            print(f"  {n}: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
