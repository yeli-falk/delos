#!/usr/bin/env bash
#SBATCH --job-name=dblp-cldg
#SBATCH --output=logs/dblp-cldg-%j.out
#SBATCH --error=logs/dblp-cldg-%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --time=10:00:00
#SBATCH --partition=gpu
#
# CLDG baseline (Contrastive Learning on Dynamic Graphs, Xu et al., ICDE'23),
# two-stage so it is directly comparable to the DELOS pipeline:
#   1) train the CLDG encoder with temporal contrastive learning
#   2) freeze it and fit the SAME link-prediction probe DELOS uses
#      (ablations_dblp --mode pretrained_probe)
#
# Outputs:
#   outputs/cldg_<TAG>/encoder.ckpt
#   outputs/cldg_<TAG>_probe/test_metrics.json
#
# Run:    DATASET=dblp sbatch scripts/dblp/slurm_cldg.sh
# Knobs:  DATASET=reddit_body  SEED=43  TAG=<name>  ENC_KIND=gcn  STRATEGY=random
#         Link-pred datasets: dblp, jodie_reddit, jodie_wikipedia, reddit_body,
#         reddit_title (or synthetic). Stage 1 + stage 2 share these flags.
#
# Note: CLDG is a static-GNN-per-view method, so ENC_KIND=gcn is its natural
# encoder. For an apples-to-apples comparison, run DELOS with the same
# --encoder_kind.

set -uo pipefail
mkdir -p logs outputs

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"

if [[ -d "$HOME/.conda/envs/cjepa" ]]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate cjepa
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo "[slurm] repo_root=$REPO_ROOT  node=$(hostname)"

DATASET=${DATASET:-dblp}
# Seed is part of TAG so a multi-seed sweep does not reuse seed 42's checkpoints.
TAG=${TAG:-${DATASET}_s${SEED:-42}}
STRATEGY=${STRATEGY:-random}

# Link-prediction datasets only: these baselines need a global DTDG snapshot
# sequence (stage 1) + a link-pred probe (stage 2). Sets DATA_ARG / WINDOW_ARG /
# ENCODER_ARG / EARLY_STOP_ARG / SEED from the shared config, so both stages —
# and every other method — construct an identical node set.
source "$REPO_ROOT/scripts/dblp/_linkpred_data_arg.sh"
echo "[config] TAG=$TAG  STRATEGY=$STRATEGY"

# -----------------------------------------------------------------------------
# Stage 1: train the CLDG encoder (temporal contrastive learning)
# -----------------------------------------------------------------------------
ENC_DIR="outputs/cldg_${TAG}"
if [[ ! -f "$ENC_DIR/encoder.ckpt" ]]; then
    echo
    echo "================================================================="
    echo "  Stage 1/2: CLDG contrastive pretrain -> $ENC_DIR"
    echo "================================================================="
    srun --export=ALL python -m src.baselines.cldg.train_cldg $DATA_ARG \
        $ENCODER_ARG $WINDOW_ARG $EARLY_STOP_ARG \
        --snapshots 4 --views 2 --strategy "$STRATEGY" --tau 0.07 \
        --node_batch_size 256 --steps_per_epoch 20 --val_steps 10 \
        --device cuda --epochs 100 --lr 4e-3 \
        --seed "$SEED" --out_dir "$ENC_DIR"
    enc_rc=$?
    if [[ $enc_rc -ne 0 || ! -f "$ENC_DIR/encoder.ckpt" ]]; then
        echo "[FATAL] Stage 1 failed (rc=$enc_rc)" >&2; exit 11
    fi
else
    echo "[stage 1] $ENC_DIR/encoder.ckpt exists, skipping CLDG pretrain"
fi

# -----------------------------------------------------------------------------
# Stage 2: freeze the CLDG encoder, fit the shared link-prediction probe
# -----------------------------------------------------------------------------
PROBE_DIR="outputs/cldg_${TAG}_probe"
echo
echo "================================================================="
echo "  Stage 2/2: frozen-encoder + link-prediction probe -> $PROBE_DIR"
echo "================================================================="
srun --export=ALL python -m src.train.analyze_jepa --skip_diagnostic \
    --link_sources encoder $DATA_ARG $WINDOW_ARG \
    --encoder_ckpt "$ENC_DIR/encoder.ckpt" \
    --num_workers ${SLURM_CPUS_PER_TASK:-8} \
    --device cuda --seed "$SEED" --out_dir "$PROBE_DIR"

echo
echo "[done] CLDG baseline complete ($DATASET)."
echo "       encoder : $ENC_DIR/encoder.ckpt"
echo "       metrics : $PROBE_DIR/analysis.json  (link_pred.encoder.{all,new}; compare to DELOS)"
