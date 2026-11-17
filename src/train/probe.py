"""Downstream probe stage for DELOS.

Once the JEPA predictor is trained (``src/train/jepa.py``), a task-specific
supervised probe is fit on the frozen predictor's rollouts and evaluated on
the test split: reload the best predictor, fit the probe, evaluate, print,
and write ``test_metrics.json``.

The probe heads and their fit/eval loops live in
``src/world_models/probes.py``; this module only orchestrates them and
persists metrics.
"""
from __future__ import annotations

import json

import torch

from src.world_models.probes import (
    evaluate_link_prediction,
    evaluate_regression,
    evaluate_regression_dynamic,
    fit_link_probe,
    fit_regression_probe,
    fit_regression_probe_dynamic,
)


def _load_best_predictor(predictor, args, device, out_dir, encoder=None):
    """Reload the best-val checkpoint before probing (if one exists).

    When ``--train_encoder`` is on, the encoder is part of what early stopping
    selected, so it must be restored too — otherwise the probe runs the
    best predictor against the LAST-epoch encoder, i.e. a pairing that was
    never validated.
    """
    best = out_dir / "best.ckpt"
    if args.epochs > 0 and best.exists():
        ck = torch.load(best, map_location=device, weights_only=False)
        predictor.load_state_dict(ck["predictor"])
        if encoder is not None and getattr(args, "train_encoder", False) \
                and "encoder" in ck:
            encoder.load_state_dict(ck["encoder"])


def _valid_mode(loader):
    """`active` for signed targets, `positive` otherwise. See
    src/world_models/probes.py::_accum_reg."""
    return "active" if getattr(loader.dataset, "signed_target", False) else "positive"


def _sees_base(loader):
    """Give the probe the persistence base as an input iff that base is NOISY.

    With a clean base (England COVID, METR-LA) persist_residual alone is the
    right parameterisation and this stays off. With a noisy base it is
    mandatory: anchoring to an unobservable noisy value caps the achievable
    improvement at about 0%. See probes.py::_with_base.
    """
    return getattr(loader.dataset, "target_noise_sigma", 0.0) > 0.0


def _save_metrics(metrics, out_dir):
    with open(out_dir / "test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)


# ---------------------------------------------------------------------------
# Link prediction (dynamic graphs: DBLP / JODIE / Reddit)
# ---------------------------------------------------------------------------
def probe_link_pred(predictor, encoder, train_loader, test_loader, args, device,
                    out_dir):
    _load_best_predictor(predictor, args, device, out_dir, encoder=encoder)
    print("[probe] fitting link-prediction probe", flush=True)
    probe = fit_link_probe(predictor, encoder, train_loader, args.history_size,
                           args.num_preds, device, epochs=args.probe_epochs,
                           hidden_dim=args.probe_hidden)
    metrics = evaluate_link_prediction(predictor, encoder, probe, test_loader,
                                       args.history_size, device)
    for subset in ("all", "new"):
        print(f"[test] link prediction ({subset} edges, per horizon)")
        for tp, m in sorted(metrics.get(subset, {}).items()):
            print(f"  +{tp+1:>2}: " + " ".join(f"{k}={v:.4f}" for k, v in m.items()))
    _save_metrics(metrics, out_dir)
    return metrics


# ---------------------------------------------------------------------------
# Regression on a fixed-topology graph (METR-LA)
# ---------------------------------------------------------------------------
def probe_regression(predictor, encoder, train_loader, test_loader, adj_dense,
                     adj_norm, args, device, out_dir):
    _load_best_predictor(predictor, args, device, out_dir, encoder=encoder)
    print("[probe] fitting regression probe -> speed", flush=True)
    probe = fit_regression_probe(predictor, encoder, train_loader, adj_dense,
                                 adj_norm, args.history_size, args.num_preds,
                                 device, epochs=args.probe_epochs,
                                 hidden_dim=args.probe_hidden)
    metrics = evaluate_regression(predictor, encoder, probe, test_loader,
                                  adj_dense, adj_norm, args.history_size,
                                  args.num_preds, device)
    print("[test] horizon-step  MAE  RMSE  MAPE")
    for h, m in metrics.items():
        print(f"  +{h:>2}: {m['MAE']:.3f}  {m['RMSE']:.3f}  {m['MAPE']:.3f}")
    _save_metrics(metrics, out_dir)
    return metrics


# ---------------------------------------------------------------------------
# Regression: node-signal forecasting (England COVID, METR-LA, synthetic_dyn)
# ---------------------------------------------------------------------------
# Serves both topologies. Dynamic graphs supply per-snapshot adjacency in the
# batch; fixed-topology ones (METR-LA) resolve to the dataset's static graph
# inside probes.graph_inputs. One probe means one metric schema — MAE / RMSE /
# MAPE plus the persistence reference and the inflection subset — so every
# dataset's rows can sit in the same table.
def probe_regression_dynamic(predictor, encoder, train_loader, test_loader, args,
                             device, out_dir):
    _load_best_predictor(predictor, args, device, out_dir, encoder=encoder)
    signal_name = {"england_covid": "case count", "metr_la": "speed"}.get(
        getattr(args, "data_source", ""), "node signal")
    print(f"[probe] fitting regression probe -> {signal_name}", flush=True)
    pr = getattr(args, "persist_residual", False)
    sb = _sees_base(train_loader)
    probe = fit_regression_probe_dynamic(predictor, encoder, train_loader,
                                         args.history_size, args.num_preds,
                                         device, epochs=args.probe_epochs,
                                         hidden_dim=args.probe_hidden,
                                         persist_residual=pr, probe_sees_base=sb)
    metrics = evaluate_regression_dynamic(predictor, encoder, probe, test_loader,
                                          args.history_size, args.num_preds, device,
                                          persist_residual=pr,
                                          valid_mode=_valid_mode(test_loader),
                                          probe_sees_base=sb)
    # forecast vs persistence, overall and on the inflection subset (where the
    # signal reverses — persistence is provably wrong there, so a real
    # forecaster must win). "win" = JEPA MAE below persistence MAE.
    print("[test] horizon  MAE_jepa  MAE_persist  |  infl: MAE_jepa  MAE_persist  (n)")
    for h, m in sorted(metrics.items()):
        line = (f"  +{h:>2}: {m['MAE']:.3f}     {m['MAE_persist']:.3f}")
        if "MAE_inflect" in m:
            line += (f"     |  {m['MAE_inflect']:.3f}     "
                     f"{m['MAE_persist_inflect']:.3f}  (n={m['n_inflect']})")
        win = "  <-- JEPA beats persistence" if m['MAE'] < m['MAE_persist'] else ""
        print(line + win)
    _save_metrics(metrics, out_dir)
    return metrics
