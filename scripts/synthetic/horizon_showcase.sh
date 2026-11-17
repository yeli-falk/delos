#!/usr/bin/env bash
# Horizon showcase on the synthetic node-state task: does the latent rollout
# stay useful further into the future than a probe on the same embedding?
#
# One run per (method, seed), trained and probed with a long forecast window
# (NPRED=12 by default). The probe fits one head per horizon on every column
# (run_block_probe.py::_fit_per_horizon), so a single run yields the whole
# curve h = 1..NPRED:
#
#   DELOS row   `jepa`               the predictor's rolled-out latent at h
#               `encoder_last`       the same encoder at the last observed bin
#                                    + a head trained for h (the paired
#                                    difference isolates the rollout)
#   baselines   `encoder_last`       their embedding at the last observed bin
#                                    + a head trained for h
#   floor       `label_persistence`  the label at the last observed bin still
#                                    holds (0 on switch rows)
#
# Headline metric: accuracy on switch rows (the label at t+h differs from the
# last observed one), where persistence is 0 by construction. Overall accuracy
# is the second panel. `AXIS=NPRED bash scripts/synthetic/sweep_state.sh`
# retrains per horizon instead and is the robustness check for h=1.
#
# Usage (from the repo root):
#   bash scripts/synthetic/horizon_showcase.sh                # submit 5 seeds
#   bash scripts/synthetic/horizon_showcase.sh --aggregate    # per-horizon table
#   bash scripts/synthetic/horizon_showcase.sh --plot         # figures + LaTeX
#
# Knobs (env):
#   NPRED (12), HIST (3), SEEDS ("42 43 44 45 46")
#   ARMS       Default: jepa_ema and the temporal SSL baselines. Arms of one
#              seed may be split over several jobs (the shared encoder is locked).
#   ROOT       Default outputs/synth_horizon/NPRED_<n>.
#   TAG        Root suffix for a second generator setting.
#   HORIZONS   Steps ahead shown in the table. Default "1 3 6 9 12".
#   PARTITION  sbatch partition override (e.g. gpu-best on margaret).
#   GRES       sbatch --gres override (e.g. gpu:ampere:1). The job fits on any
#              card, so pin one only if per-epoch timings must be comparable.
#   SYN_* generator knobs and everything slurm_all_methods.sh reads (BASE_HP, LR, ...).
#
# Showcase generator setting. The linear readout keeps the hidden velocity
# recoverable from the feature history, and the root-weight encoder
# (ENC_KIND=gcn_root) keeps each node's own features out of the neighbourhood
# average; check headroom first with scripts/synthetic/horizon_headroom.py.
#   SYN_MOMENTUM=0.95 SYN_THETA=0.05 SYN_READOUT=linear SYN_DIFFUSION=0 \
#     ENC_KIND=gcn_root TAG=lin_rho095_root bash scripts/synthetic/horizon_showcase.sh
#
# Predictor variants as extra JEPA rows in the same root (baselines and shared
# encoder are reused; the plot takes them via --jepa_arms):
#   ARMS=jepa_ema MASK_SCHEME=future_forecast JEPA_TAG=ff TAG=rho095_th005 bash ...
#   ARMS=jepa_ema CHANGE_W=4 JEPA_TAG=cw4 TAG=rho095_th005 bash ...
#   ARMS=jepa_ema INPUT_DELTA=1 JEPA_TAG=vel ...                                      # velocity channel
#   ... LINEAR_SKIP=1 LINEAR_SKIP_INIT=ridge LINEAR_SKIP_FREEZE=1 JEPA_TAG=ridge ...  # linear floor + residual transformer
#   NPRED=12 TAG=rho095_th005 bash ... --plot --jepa_arms jepa_ema_ff jepa_ema_cw4
# Window encoder under the predictor (separate root):
#   ENC_KIND=maskdgnn_root TAG=lin_rho095_mdgnn_T1000 ARMS="jepa_frozen maskdgnn cldg ddgcl dysubc" ...
# Latent error of a trained predictor against copy-last, per horizon:
#   python scripts/synthetic/predictor_latent_error.py --jepa_ckpt <root>/jepa_ema_s42/best.ckpt
#
# Logs: logs/hz-<NPRED>-s<seed>-<jobid>.out
set -uo pipefail

NPRED=${NPRED:-12}
HIST=${HIST:-3}
SEEDS=${SEEDS:-"42 43 44 45 46"}
# jepa_ema against the temporal SSL baselines: no frozen arm and no static
# (GRACE / GraphMAE / BGRL) block.
ARMS=${ARMS:-"jepa_ema cldg ddgcl idol maskdgnn dysubc dyg2vec dygwm sgjepa summary"}
# The trainer's link probe is not used by this experiment; the state probe on
# best.ckpt is the result. Exported into slurm_all_methods.sh.
export DOWNSTREAM=${DOWNSTREAM:-none}
export COLLAPSE_CHECK=${COLLAPSE_CHECK:-0}
# TAG names a second generator setting so that it never shares a root with
# the reference run.
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
