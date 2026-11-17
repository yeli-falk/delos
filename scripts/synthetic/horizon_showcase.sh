#!/usr/bin/env bash
# HORIZON SHOWCASE on the synthetic node-state task: does the latent rollout
# stay useful further into the future than a probe on the same embedding?
#
# One run per (method, seed), trained and probed with a LONG forecast window
# (NPRED=12 by default). The probe fits one head PER HORIZON on every column
# (run_block_probe.py::_fit_per_horizon), so a single run yields the whole
# curve h = 1..NPRED:
#
#   DELOS row    `jepa`          the predictor's rolled-out latent at h
#                   `encoder_last`  the SAME (EMA-trained) encoder at the
#                                   last observed bin + a head trained for h
#                                   (the direct-forecast control; the paired
#                                   difference isolates the rollout)
#   baselines       `encoder_last`  their embedding at the last observed bin
#                                   + a head trained for h (the only way an
#                                   encoder-only method can forecast)
#   floor           `label_persistence`  the label at the last observed bin
#                                   still holds (0 on switch rows)
#
# Headline metric: accuracy on SWITCH rows (label at t+h differs from the
# last observed one), where persistence is 0 by construction. Overall
# accuracy is the honest second panel (JEPA pays on stay rows).
#
# This is the multi-horizon protocol of the traffic / COVID forecasting
# tables (train once for the longest horizon, report every horizon), so it
# is the classification twin of the England COVID result. The older
# "one run per NPRED value" design is still available as
# `AXIS=NPRED VALUES="1 3 6 9 12" bash scripts/synthetic/sweep_state.sh`
# and is the robustness check that the h=1 point does not depend on the
# training window.
#
# Usage (from the repo root):
#   bash scripts/synthetic/horizon_showcase.sh                # submit 5 seeds
#   bash scripts/synthetic/horizon_showcase.sh --aggregate    # per-horizon table
#   bash scripts/synthetic/horizon_showcase.sh --plot         # figures + LaTeX
#
# Knobs (env): NPRED (12), HIST (3), SEEDS ("42 43 44 45 46"), ARMS (the six
# temporal baselines + jepa_ema), ROOT (outputs/synth_horizon/NPRED_<n>),
# HORIZONS ("1 3 6 9 12", steps ahead, for the table), TAG (root suffix for
# a second generator setting), PARTITION (sbatch
# partition override, e.g. gpu-best on margaret), GRES (sbatch --gres override,
# e.g. gpu:ampere:1 to pin a card type on margaret's mixed partition; the
# job is small enough for any card, so leave it unset: pinning only adds queue
# time, unless per-epoch timings must be comparable), SYN_* generator knobs and
# everything slurm_all_methods.sh reads (BASE_HP, LR, ...).
# Linear feature readout (2026-09-24): the legacy tanh readout caps the hidden
# velocity at R^2 ~0.4 from the feature history, so no encoder can carry it.
# The showcase setting is therefore
#   SYN_MOMENTUM=0.95 SYN_THETA=0.05 SYN_READOUT=linear SYN_DIFFUSION=0 TAG=lin_rho095 bash ...
# (check first: python scripts/synthetic/horizon_headroom.py --readout linear --diffusion 0 ...)
# Root-weight encoder (2026-09-24): the plain GCN averages each node's own
# features into its neighbourhood (velocity R^2 ceiling 0.99 -> 0.37 after one
# aggregation, at ANY density), so the shared encoder for the showcase is
# ENC_KIND=gcn_root (GraphSAGE-style root path; MaskDGNN follows automatically):
#   SYN_MOMENTUM=0.95 SYN_THETA=0.05 SYN_READOUT=linear SYN_DIFFUSION=0 ENC_KIND=gcn_root TAG=lin_rho095_root bash ...
# Predictor variants as extra JEPA rows in the SAME root (baselines + shared
# encoder are reused; the plot takes them via --jepa_arms):
#   ARMS=jepa_ema MASK_SCHEME=future_forecast JEPA_TAG=ff TAG=rho095_th005 bash ...
#   ARMS=jepa_ema CHANGE_W=4 JEPA_TAG=cw4 TAG=rho095_th005 bash ...
#   ARMS=jepa_ema INPUT_DELTA=1 JEPA_TAG=vel ...             # velocity channel (2026-09-29)
#   ARMS=jepa_ema INPUT_DELTA=1 MASK_SCHEME=future_forecast JEPA_TAG=vel_ff ...
#   ... INPUT_DELTA=1 MASK_SCHEME=future_forecast HORIZON_NORM=1 JEPA_TAG=vel_ff_hn ...   # + per-horizon loss norm (no gain)
#   ... INPUT_DELTA=1 MASK_SCHEME=future_forecast LINEAR_SKIP=1 JEPA_TAG=vel_ff_ls ...    # + linear-dynamics prior (2026-09-30, zero-init: no gain)
#   ... LINEAR_SKIP=1 LINEAR_SKIP_INIT=ridge LINEAR_SKIP_FREEZE=1 JEPA_TAG=ridge ...     # closed-form linear floor + residual transformer (2026-10-05)
# Window encoder under the predictor (MaskDGNN's own encoder kind, root-weight
# spatial block), separate root:
#   ENC_KIND=maskdgnn_root TAG=lin_rho095_mdgnn_T1000 ARMS="jepa_frozen maskdgnn cldg ddgcl dysubc" ...
#   NPRED=12 TAG=rho095_th005 bash ... --plot --jepa_arms jepa_ema_ff jepa_ema_cw4
# Diagnose a trained predictor's own latent error vs copy-last per horizon:
#   python scripts/synthetic/predictor_latent_error.py --jepa_ckpt <root>/jepa_ema_s42/best.ckpt
# Split a seed over several jobs with ARMS="jepa_ema" / ARMS="cldg ddgcl" etc.;
# the shared encoder is locked, so arms of one seed may run concurrently.
#
# Logs: logs/hz-<NPRED>-s<seed>-<jobid>.out
set -uo pipefail

NPRED=${NPRED:-12}
HIST=${HIST:-3}
SEEDS=${SEEDS:-"42 43 44 45 46"}
# jepa_ema (encoder trained jointly with an EMA target) vs the temporal SSL
# baselines. No frozen arm, no static (GRACE / GraphMAE / BGRL) block.
ARMS=${ARMS:-"jepa_ema cldg ddgcl idol maskdgnn dysubc dyg2vec dygwm sgjepa summary"}
# The JEPA trainer's link probe and the collapse check are link-prediction
# probes this experiment does not use; the state probe on best.ckpt is the
# result. Exported into slurm_all_methods.sh.
export DOWNSTREAM=${DOWNSTREAM:-none}
export COLLAPSE_CHECK=${COLLAPSE_CHECK:-0}
# TAG names a second generator setting so it never shares a root with the
# reference run, e.g.  SYN_MOMENTUM=0.95 SYN_THETA=0.05 TAG=rho095_th005 bash ...
# (scan settings first: python scripts/synthetic/horizon_headroom.py).
TAG=${TAG:-}
ROOT=${ROOT:-outputs/synth_horizon/NPRED_${NPRED}${TAG:+_$TAG}}
FIG=${FIG:-figures/synth_horizon/NPRED_${NPRED}${TAG:+_$TAG}}
HORIZONS=${HORIZONS:-"1 3 6 9 12"}
PARTITION=${PARTITION:-}
GRES=${GRES:-}

if [[ "${1:-}" == "--aggregate" ]]; then
    # subset_h<k> is 0-indexed inside state_probe.json; the table speaks in
    # steps ahead (1-indexed).
    SW=""; OV=""
    for h in $HORIZONS; do
        k=$((h - 1))
        (( k < NPRED )) || { echo "[warn] horizon $h > NPRED=$NPRED, skipped" >&2; continue; }
        SW="$SW subset_h${k}.switch_acc_test"; OV="$OV subset_h${k}.acc_test"
    done
    echo; echo "################ SWITCH-row accuracy per horizon (steps ahead: $HORIZONS)   ($ROOT)"
    python scripts/classification/aggregate_classification.py \
        --dataset synthetic_dyn --root "$ROOT" --seeds $SEEDS --metrics $SW \
        --json_out "$ROOT/summary_switch.json" 2>&1 | grep -v '^\[skip\]'
    echo; echo "################ OVERALL accuracy per horizon (steps ahead: $HORIZONS)"
    python scripts/classification/aggregate_classification.py \
        --dataset synthetic_dyn --root "$ROOT" --seeds $SEEDS --metrics $OV \
        --json_out "$ROOT/summary_overall.json" 2>&1 | grep -v '^\[skip\]'
    exit 0
fi

if [[ "${1:-}" == "--plot" ]]; then
    shift
    python scripts/synthetic/plot_horizon.py --root "$ROOT" --seeds $SEEDS \
        --table_horizons $HORIZONS --out "$FIG" --latex "$@"
    exit 0
fi

mkdir -p logs "$ROOT"
for s in $SEEDS; do
    LOG="logs/hz-${NPRED}-s${s}-%j"
    echo "sbatch: NPRED=$NPRED HIST=$HIST seed=$s -> $ROOT   (log: $LOG.out)"
    env NPRED="$NPRED" HIST="$HIST" DATASET=synthetic_dyn SEED="$s" ROOT="$ROOT" ARMS="$ARMS" \
        sbatch ${PARTITION:+--partition="$PARTITION"} ${GRES:+--gres="$GRES"} \
               --job-name="hz-${NPRED}-s${s}" --output="$LOG.out" --error="$LOG.err" \
        scripts/classification/slurm_all_methods.sh
done
echo "When done:  NPRED=$NPRED TAG=$TAG bash $0 --aggregate   |   NPRED=$NPRED TAG=$TAG bash $0 --plot"
