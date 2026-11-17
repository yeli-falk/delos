"""Parameter-free link-prediction baselines on the same test setup as the
DELOS evaluator. No training, no checkpoints required.

Baselines implemented:
  - EdgeBank-inf  : memorise any pair seen at any prior history timestep
                    in the same window. The TGB paper showed this often
                    beats trained methods on bipartite CTDGs.
  - EdgeBank-tw   : same but with a sliding window of `tw_bins` recent steps.
  - Pref. Attach. : score = deg_src x deg_dst at last history step.
  - Common Neigh. : |N(src) ∩ N(dst)| at last history step.
  - Adamic-Adar   : sum_{c in N(src) ∩ N(dst)} 1 / log(1 + deg(c))

Each baseline scores positive edges and 99 sampled negatives per source,
exactly mirroring `analyze_jepa.eval_link_pred` (same window loop, same
`sample_neg_edges` seed, same aggregation) so the numbers drop straight into
the table beside the trained methods. We report MRR/Hits@k on both the full
positive set and the new-edge subset (positives absent at every prior history
step in the window).

Usage:
    python scripts/dblp/run_baselines.py \\
        --data_source reddit_body \\
        --reddit_data_path dataset/reddit/soc-redditHyperlinks-body.tsv \\
        --reddit_top_k 1000 --reddit_num_bins 36 \\
        --out_dir outputs/baselines_reddit_body

    python scripts/dblp/run_baselines.py \\
        --data_source jodie_reddit \\
        --jodie_data_path dataset/jodie/reddit.csv \\
        --out_dir outputs/baselines_jodie_reddit
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.custom_codes.dataset_factory import (
    add_dataset_cli_flags,
    build_dynamic_dataset,
)
from src.train.engine import set_deterministic
from src.world_models.probes import sample_neg_edges
from src.world_models.delos_encoder import link_ranking_metrics


# Baselines: each scores a batch of (src, dst) pairs at time t_global, given
# the window's adjacency history.
def _edgebank_inf(adj_window, t_history, src, dst):
    """1 if (src, dst) appears at ANY prior history step."""
    # adj_window: (T, N, N) bool; t_history: int; src, dst: (P,) long
    seen = adj_window[:t_history, src, dst]                 # (t_history, P)
    return seen.any(dim=0).float()


def _edgebank_tw(adj_window, t_history, src, dst, tw_bins: int):
    """1 if (src, dst) appears in the last `tw_bins` of history."""
    lo = max(0, t_history - tw_bins)
    seen = adj_window[lo:t_history, src, dst]
    return seen.any(dim=0).float()


def _pref_attach(adj_window, t_history, src, dst):
    """deg(src) * deg(dst) at the last history step."""
    A = adj_window[t_history - 1].float()
    deg = A.sum(dim=-1)
    return deg[src] * deg[dst]


def _common_neighbors(adj_window, t_history, src, dst):
    """|N(src) ∩ N(dst)| at the last history step."""
    A = adj_window[t_history - 1]
    # vectorised: for each pair, AND the rows and sum
    common = A[src].float() * A[dst].float()                # (P, N)
    return common.sum(dim=-1)


def _adamic_adar(adj_window, t_history, src, dst):
    """sum_{c in N(src) ∩ N(dst)} 1 / log(1 + deg(c))."""
    A = adj_window[t_history - 1].float()
    deg = A.sum(dim=-1)
    inv_log = 1.0 / (torch.log1p(deg) + 1e-6)
    common = A[src] * A[dst]                                # (P, N) float
    return (common * inv_log.unsqueeze(0)).sum(dim=-1)


BASELINES = {
    "edgebank_inf":        lambda w, th, s, d: _edgebank_inf(w, th, s, d),
    "edgebank_tw3":        lambda w, th, s, d: _edgebank_tw(w, th, s, d, tw_bins=3),
    "pref_attach":         lambda w, th, s, d: _pref_attach(w, th, s, d),
    "common_neighbors":    lambda w, th, s, d: _common_neighbors(w, th, s, d),
    "adamic_adar":         lambda w, th, s, d: _adamic_adar(w, th, s, d),
}


# Evaluator: mirrors the window loop of analyze_jepa.eval_link_pred.
@torch.no_grad()
def evaluate_baseline(name, score_fn, loader, history_size, k_per_pos=99):
    per_horizon_all: dict = {}
    per_horizon_new: dict = {}
    for batch_idx, batch in enumerate(loader):
        adj_b = batch["adj"]                          # (B, T, N, N) bool
        active_b = batch["active"]                    # (B, T, N) bool
        B = adj_b.shape[0]
        for b in range(B):
            adj_w = adj_b[b]                          # (T, N, N) bool
            for tp in range((adj_w.shape[0] - history_size)):
                t_global = history_size + tp
                adj_t = adj_w[t_global]
                active_t = active_b[b, t_global]
                if adj_t.sum() == 0 or active_t.sum() < 2:
                    continue
                src, dst = adj_t.nonzero(as_tuple=True)
                if src.numel() == 0:
                    continue

                pos =score_fn(adj_w, history_size + tp, src, dst)     # (P,)
                # Sample negatives (same source) with the SAME sampler AND the
                # same per-window seed as analyze_jepa.eval_link_pred, so every
                # method is ranked against a byte-identical negative set.
                # (A seed of just `tp + b` would reuse one RNG stream for every
                # batch, correlating the negatives across windows.)
                neg_dst = sample_neg_edges(adj_t, active_t,
                                            k_per_pos=k_per_pos,
                                            seed=batch_idx * 1000 + b * 100 + tp)
                if neg_dst.numel() == 0:
                    continue
                P, K = neg_dst.shape
                neg_src_flat = src.unsqueeze(1).expand(P, K).reshape(-1)
                neg_dst_flat = neg_dst.reshape(-1)
                # Chunk to bound the peak memory of the (chunk, N) intermediates
                # in CN / Adamic-Adar, which would not fit at DBLP scale.
                chunk = 4096
                neg_parts = []
                for i in range(0, neg_src_flat.numel(), chunk):
                    neg_parts.append(
                        score_fn(adj_w, history_size + tp,
                                 neg_src_flat[i:i + chunk],
                                 neg_dst_flat[i:i + chunk]))
                neg = torch.cat(neg_parts, dim=0).reshape(P, K)
                m_all = link_ranking_metrics(pos, neg)
                per_horizon_all.setdefault(tp, []).append(m_all)

                # New-edge subset
                prior_seen = adj_w[:t_global].any(dim=0)              # (N, N) bool
                is_new = ~prior_seen[src, dst]
                if int(is_new.sum()) == 0:
                    continue
                m_new = link_ranking_metrics(pos[is_new], neg[is_new])
                per_horizon_new.setdefault(tp, []).append(m_new)

    def _aggregate(per_horizon):
        out = {}
        for tp, ms in per_horizon.items():
            keys = ms[0].keys()
            out[tp] = {k: float(np.mean([m[k] for m in ms])) for k in keys}
        return out

    return {"all": _aggregate(per_horizon_all),
            "new": _aggregate(per_horizon_new)}


def make_loader(ds, batch_size, num_workers):
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=False, drop_last=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--history_size", type=int, default=3)
    p.add_argument("--num_preds", type=int, default=1)
    p.add_argument("--data_path", default=None)
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--synthetic_nodes", type=int, default=500)
    p.add_argument("--synthetic_years", type=int, default=30)
    p.add_argument("--synthetic_kind", choices=["dblp", "sbm"], default="dblp")
    p.add_argument("--synthetic_blocks", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--out_dir", default="outputs/baselines")
    p.add_argument("--baselines", nargs="+",
                   default=list(BASELINES.keys()),
                   help="Subset of baselines to run.")
    add_dataset_cli_flags(p)
    args = p.parse_args()

    set_deterministic(args.seed)
    test_ds = build_dynamic_dataset(args, split="test")
    print(f"[baselines] test: {len(test_ds)} windows, N={test_ds.num_nodes}",
          flush=True)
    loader = make_loader(test_ds, args.batch_size, args.num_workers)

    results = {}
    for name in args.baselines:
        if name not in BASELINES:
            print(f"[skip] unknown baseline {name!r}"); continue
        print(f"[run] {name}", flush=True)
        results[name] = evaluate_baseline(
            name, BASELINES[name], loader, args.history_size,
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "baselines.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(out_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # Also write each baseline as its own pseudo-cjepa directory so the
    # existing dump_runs.py picks them up alongside the trained methods.
    # Each contains just test_metrics.json + a stub args.json; no encoder
    # or diagnostic since baselines are parameter-free.
    parent = out_dir.parent
    for name, blk in results.items():
        bdir = parent / f"{out_dir.name}_{name}_cjepa"
        bdir.mkdir(parents=True, exist_ok=True)
        with open(bdir / "test_metrics.json", "w") as f:
            json.dump(blk, f, indent=2)
        stub_args = {**vars(args), "baseline": name}
        with open(bdir / "args.json", "w") as f:
            json.dump(stub_args, f, indent=2)

    # Console summary table
    print(f"\n{'baseline':<22}{'MRR_all':>10}{'Hits@10_all':>14}{'MRR_new':>10}{'Hits@10_new':>14}")
    print("-" * 70)
    for name, blk in results.items():
        all0 = blk.get("all", {}).get(0, {}) or blk.get("all", {}).get("0", {})
        new0 = blk.get("new", {}).get(0, {}) or blk.get("new", {}).get("0", {})
        print(f"{name:<22}"
              f"{all0.get('MRR', float('nan')):>10.4f}"
              f"{all0.get('Hits@10', float('nan')):>14.4f}"
              f"{new0.get('MRR', float('nan')):>10.4f}"
              f"{new0.get('Hits@10', float('nan')):>14.4f}")
    print(f"\n[done] wrote {out_dir/'baselines.json'}", flush=True)


if __name__ == "__main__":
    main()
