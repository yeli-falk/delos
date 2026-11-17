"""Rescore saved DELOS runs with the extended link-prediction metric set
(adds AUC and exposes Hits@1, which the JSON dumps drop by default).

The original `test_metrics.json` only stores aggregated MRR / Hits@10 — the
raw (pos, neg) score tensors are not persisted. To get the new metrics
without retraining, we rebuild the model from `best.ckpt` (which embeds the
training args), refit the link probe on the train split, and rerun
`evaluate_link_prediction`. Output is written alongside the original as
`test_metrics_v2.json` so the v1 numbers stay intact.

Usage:
    # Rescore one run.
    python scripts/dblp/rescore_link_metrics.py \
        --delos_dir outputs/reddit_body_long_s42_cjepa \
        --encoder_dir outputs/reddit_body_long_s42_encoder

    # Rescore every long-training run.
    for d in outputs/reddit_body_long_s*_cjepa; do
        tag=${d%_cjepa}
        python scripts/dblp/rescore_link_metrics.py \
            --delos_dir "$d" --encoder_dir "${tag}_encoder"
    done

The probe is refit fresh (its state isn't checkpointed), so MRR numbers may
differ from `test_metrics.json` at the 3rd-4th decimal — that's expected
and within seed noise. Relative ordering across runs is preserved.
"""
from __future__ import annotations

import argparse
import json
import types
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.custom_codes.dataset_factory import build_dynamic_dataset
from src.world_models.delos_encoder import build_encoder
from src.delos_predictor import MaskedNodePredictor
from src.train.engine import set_deterministic
from src.world_models.probes import (
    evaluate_link_prediction,
    fit_link_probe,
)


def _ns_from_dict(d: dict) -> types.SimpleNamespace:
    """Reconstitute an argparse.Namespace-like object from a saved args dict
    so the dataset factory and probe builder see the same flags as training."""
    return types.SimpleNamespace(**d)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--delos_dir", required=True,
                   help="outputs/<TAG>_cjepa containing best.ckpt + args.json")
    p.add_argument("--encoder_dir", required=True,
                   help="outputs/<TAG>_encoder containing encoder.ckpt")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out_name", default="test_metrics_v2.json",
                   help="Filename written under --delos_dir.")
    p.add_argument("--probe_epochs", type=int, default=None,
                   help="Override saved probe_epochs (None = use saved).")
    args = p.parse_args()

    delos_dir = Path(args.delos_dir)
    enc_dir = Path(args.encoder_dir)
    ckpt_path = delos_dir / "best.ckpt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"missing {ckpt_path}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[rescore] delos_dir={delos_dir}  device={device}")

    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    saved_args = ck["args"]
    # Patch the encoder ckpt path so build_encoder + state_dict load match.
    saved_args["encoder_ckpt"] = str(enc_dir / "encoder.ckpt")
    if args.probe_epochs is not None:
        saved_args["probe_epochs"] = args.probe_epochs
    ns = _ns_from_dict(saved_args)

    # Seed as in training so the probe init is identical across reruns of
    # this script.
    set_deterministic(getattr(ns, "seed", 42))

    train_ds = build_dynamic_dataset(ns, split="train")
    test_ds = build_dynamic_dataset(ns, split="test")
    print(f"[rescore] train windows={len(train_ds)}  test windows={len(test_ds)}"
          f"  N={train_ds.num_nodes}  feat_dim={train_ds.feature_dim}")

    train_loader = DataLoader(train_ds, batch_size=ns.batch_size, shuffle=True,
                              num_workers=0, pin_memory=False, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=ns.batch_size, shuffle=False,
                             num_workers=0, pin_memory=False, drop_last=False)

    enc_kind = getattr(ns, "encoder_kind", "tgcn")
    enc_hidden = getattr(ns, "encoder_hidden", 128)
    enc_out = getattr(ns, "node_dim", 128)
    enc_bidir = getattr(ns, "encoder_bidirectional", True)
    encoder = build_encoder(
        enc_kind, in_dim=train_ds.feature_dim, hidden_dim=enc_hidden,
        out_dim=enc_out, freeze=True, bidirectional=enc_bidir,
    ).to(device)
    encoder.load_state_dict(ck["encoder"])
    encoder.eval()

    predictor = MaskedNodePredictor(
        num_nodes=train_ds.num_nodes,
        node_dim=enc_out,
        history_frames=ns.history_size,
        pred_frames=ns.num_preds,
        num_masked_nodes=ns.num_masked_nodes,
        seed=ns.seed,
        depth=ns.depth,
        heads=ns.heads,
        dim_head=enc_out // ns.heads,
        mlp_dim=ns.mlp_dim,
        dropout=getattr(ns, "dropout", 0.0),
        use_attention_mask=not getattr(ns, "no_attention_mask", False),
        use_edge_head=(getattr(ns, "edge_mask_rate", 0.0) > 0),
        residual_head=getattr(ns, "residual_head", False),
    ).to(device)
    predictor.load_state_dict(ck["predictor"])
    predictor.eval()

    print(f"[rescore] refitting probe for {ns.probe_epochs} epochs")
    probe = fit_link_probe(
        predictor, encoder, train_loader, ns.history_size, ns.num_preds,
        device, epochs=ns.probe_epochs,
        hidden_dim=getattr(ns, "probe_hidden", 0),
    )

    print("[rescore] running evaluate_link_prediction with extended metrics")
    metrics = evaluate_link_prediction(
        predictor, encoder, probe, test_loader, ns.history_size, device,
    )
    for subset_name in ("all", "new"):
        print(f"[rescore] {subset_name} edges (horizon 0):")
        for tp, m in sorted(metrics.get(subset_name, {}).items()):
            print(f"  +{tp+1:>2}: " + " ".join(f"{k}={v:.4f}" for k, v in m.items()))

    out_path = delos_dir / args.out_name
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[rescore] wrote {out_path}")


if __name__ == "__main__":
    main()
