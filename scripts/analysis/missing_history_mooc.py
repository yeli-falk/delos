"""Missing-history (partial-observability) probe for MOOC node-state prediction.

Same hypothesis as the COVID version, on a node-classification task: when a
student's own interaction history is hidden, can the model still predict their
state (dropout) from GRAPH CONTEXT? At eval we hide a fraction of nodes'
history features (they stay active in the graph) and score state AUROC on
exactly those hidden nodes, for two representations of the SAME frozen encoder:

  - encoder-last : the last-history-step embedding (copy-last). For a hidden
                   node this is only whatever message-passing delivered.
  - jepa         : the predictor's forecast latent (trained with node masking,
                   so it should reconstruct a hidden node from context).

A linear probe is fit ONCE on clean train latents; each frac re-collects the
test latents under corruption and scores the probe on the hidden nodes. The
claim: jepa AUROC degrades more slowly than encoder-last as the hidden
fraction grows.

Usage:
    python scripts/analysis/missing_history_mooc.py \
        --jepa_ckpt outputs/<TAG>_cjepa/best.ckpt \
        --state_label_mode state --fracs 0 0.25 0.5 0.75
"""
from __future__ import annotations

import argparse
import json
import types
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.delos_predictor import MaskedNodePredictor, predictor_flags_from_args
from src.custom_codes.dataset_factory import build_dynamic_dataset
from src.train.analyze_jepa import train_node_probe
from src.world_models.delos_encoder import build_encoder, normalize_adjacency
from src.world_models.probes import corrupt_history_bn, sample_masked_per_example

# reuse the dynamic-label resolver from the block-probe script
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "_rbp", str(Path(__file__).resolve().parents[1] / "dblp" / "run_block_probe.py"))
_rbp = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_rbp)
_per_row_labels = _rbp._per_row_labels


def _feat_mean(loader, history_size, device):
    tot, n = None, 0.0
    for b in loader:
        x = b["x_hist"].to(device)
        m = b["active"][:, :history_size].to(device).unsqueeze(-1).float()
        s = (x * m).sum(dim=(0, 1, 2))
        tot = s if tot is None else tot + s
        n += m.sum().item()
    return tot / max(n, 1.0)


@torch.no_grad()
def collect(encoder, predictor, loader, history_size, device, frac, fill,
            seed=0):
    """Per-row (encoder-last, jepa) latents at the target step, plus which rows
    were hidden. Corrupts a shared fraction of active-history nodes per batch."""
    enc, jep, node, abin, msk = [], [], [], [], []
    encoder.eval(); predictor.eval()
    g = torch.Generator(device=device).manual_seed(seed)
    for wid, batch in enumerate(loader):
        x_hist = batch["x_hist"].to(device)
        adj = batch["adj"].to(device); active = batch["active"].to(device)
        year = batch["year"]
        # Hide a fraction of each window's nodes that have a past to hide.
        active_ref = active[:, history_size - 1]                   # (B, N)
        mask_bn = sample_masked_per_example(active_ref, frac, g)   # (B, N)
        x_in = corrupt_history_bn(x_hist, mask_bn, fill)
        adj_norm = normalize_adjacency(adj)
        z_hist = encoder(x_in, adj_norm[:, :history_size])
        z_pred = predictor.inference(z_hist, adj, active)          # (B,Tp,N,D)
        B, T_pred = active.shape[0], z_pred.shape[1]
        for b in range(B):
            for tp in range(T_pred):
                tg = history_size + tp
                act = active[b, tg]
                if act.sum() < 2:
                    continue
                idx = act.nonzero(as_tuple=True)[0]
                enc.append(z_hist[b, -1, idx].cpu())
                jep.append(z_pred[b, tp, idx].cpu())
                node.append(idx.cpu().long())
                abin.append(torch.full((idx.numel(),), int(year[b]) + tg))
                msk.append(mask_bn[b, idx].cpu())
    return (torch.cat(enc).numpy().astype(np.float64),
            torch.cat(jep).numpy().astype(np.float64),
            torch.cat(node).numpy(), torch.cat(abin).numpy(),
            torch.cat(msk).numpy())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--jepa_ckpt", required=True)
    p.add_argument("--state_label_mode", default="state",
                   choices=["ever", "event", "state"])
    p.add_argument("--fracs", type=float, nargs="+",
                   default=[0.0, 0.25, 0.5, 0.75])
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    torch.manual_seed(args.seed)
    device = args.device

    ck = torch.load(args.jepa_ckpt, map_location="cpu", weights_only=False)
    saved = ck.get("args", {}) or {}
    ns = types.SimpleNamespace(**saved)
    ns.device = device; ns.batch_size = args.batch_size
    ns.num_workers = args.num_workers
    ns.with_state_labels = True
    ns.jodie_state_label_mode = args.state_label_mode
    HS = saved.get("history_size", 3)

    train_ds = build_dynamic_dataset(ns, "train")
    test_ds = build_dynamic_dataset(ns, "test")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers)

    node_dim = saved.get("node_dim", 128)
    encoder = build_encoder(
        saved.get("encoder_kind", "tgcn"), in_dim=train_ds.feature_dim,
        hidden_dim=saved.get("encoder_hidden", 128), out_dim=node_dim,
        freeze=True, bidirectional=saved.get("bidirectional", True)).to(device)
    encoder.load_state_dict(ck["encoder"])
    predictor = MaskedNodePredictor(
        num_nodes=train_ds.num_nodes, node_dim=node_dim,
        history_frames=HS, pred_frames=saved.get("num_preds", 1),
        num_masked_nodes=saved.get("num_masked_nodes", 30),
        depth=saved.get("depth", 4), heads=saved.get("heads", 4),
        dim_head=node_dim // saved.get("heads", 4),
        mlp_dim=saved.get("mlp_dim", 512), dropout=saved.get("dropout", 0.1),
        use_attention_mask=not saved.get("no_attention_mask", True),
        **predictor_flags_from_args(saved)).to(device)
    predictor.load_state_dict(ck["predictor"])
    fill = _feat_mean(train_loader, HS, device)

    # Clean train latents (fit the probe on these) + their labels.
    Ze_tr, Zj_tr, node_tr, abin_tr, _ = collect(
        encoder, predictor, train_loader, HS, device, 0.0, fill, seed=args.seed)
    y_tr, _, _, _ = _per_row_labels(train_ds, node_tr, abin_tr, "dataset", "binary")

    print(f"\n[{Path(args.jepa_ckpt).parent.name}]  missing-history state AUROC "
          f"($\\uparrow$) on hidden nodes")
    print(f"{'frac':>6} | {'enc-last':>9} {'jepa':>7} | n_hidden  pos")
    res = {}
    for frac in args.fracs:
        Ze_te, Zj_te, node_te, abin_te, msk_te = collect(
            encoder, predictor, test_loader, HS, device, frac, fill,
            seed=args.seed)
        y_te, _, _, _ = _per_row_labels(test_ds, node_te, abin_te, "dataset",
                                        "binary")
        # Concat clean-train + corrupted-test; probe fits on train, scores test.
        def _probe(Ztr, Zte):
            Z = np.concatenate([Ztr, Zte], 0)
            y = np.concatenate([y_tr, y_te], 0).astype(np.int64)
            n_tr = len(Ztr)
            tr = np.zeros(len(Z), bool); tr[:n_tr] = True
            te = ~tr
            valid = y >= 0
            # scored subset = hidden test rows (or all test rows at frac=0)
            sub = np.zeros(len(Z), bool)
            sub[n_tr:] = msk_te if frac > 0 else True
            r = train_node_probe(Z, y, tr & valid, te & valid, "binary",
                                 device=device, test_subsets={"hidden": sub & valid})
            s = r.get("subset_hidden", {})
            return s.get("auroc_test", r["auroc_test"]), s.get("n", 0), s.get("n_pos", 0)
        a_enc, n_h, n_pos = _probe(Ze_tr, Ze_te)
        a_jep, _, _ = _probe(Zj_tr, Zj_te)
        note = "  (frac=0: all nodes, full history)" if frac == 0 else ""
        print(f"{frac:>6.2f} | {a_enc:>9.4f} {a_jep:>7.4f} | {n_h:>7} {n_pos:>4}{note}")
        res[str(frac)] = {"enc_last_auroc": a_enc, "jepa_auroc": a_jep,
                          "n_hidden": int(n_h), "n_pos": int(n_pos)}
    out = Path(args.jepa_ckpt).parent / "missing_history.json"
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
