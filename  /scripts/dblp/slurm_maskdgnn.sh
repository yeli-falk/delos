#!/usr/bin/env bash
#SBATCH --job-name=dblp-maskdgnn
#SBATCH --output=logs/dblp-maskdgnn-%j.out
#SBATCH --error=logs/dblp-maskdgnn-%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --time=10:00:00
#SBATCH --partition=gpu
#
# MaskDGNN baseline (Self-Supervised Dynamic GNNs with Activeness-aware Temporal
# Masking, He et al., IJCAI'25), two-stage so it is directly comparable to the
# DyG-JEPA pipeline:
#   1) train MaskDGNN (activeness-aware edge masking + masked/next-edge BCE on
#      a GCN with an adaptive frequency layer). The trainer also scores its own
#      encoder + edge head end-to-end on the test split (the paper's protocol)
#      -> link_e2e.json
#   2) freeze the encoder and fit the SAME link-prediction probe DyG-JEPA uses
#
# Outputs:
#   outputs/maskdgnn_<TAG>/encoder.ckpt, maskdgnn.ckpt, link_e2e.json
#   outputs/maskdgnn_<TAG>_probe/analysis.json
#
# Run:    DATASET=dblp sbatch scripts/dblp/slurm_maskdgnn.sh
# Knobs:  DATASET=reddit_body  SEED=43  TAG=<name>  MASK_RATIO=0.2  ALPHA=0.7
#         TARGET=next|same  WINDOW=0 (0 = history_size)
#         Link-pred datasets: dblp, jodie_reddit, jodie_wikipedia, reddit_body,
#         reddit_title (or synthetic). Stage 1 + stage 2 share these flags.
#
# Dataset / encoder / window / early-stop config comes from
# scripts/dblp/_dataset_config.sh so it is identical to every other method.
# The encoder kind is overridden to `maskdgnn` (the shared GCN + the method's
# adaptive frequency layer); see the note in _dataset_config.sh.

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
MASK_RATIO=${MASK_RATIO:-0.2}
ALPHA=${ALPHA:-0.7}
TARGET=${TARGET:-next}
WINDOW=${WINDOW:-0}

# Link-prediction datasets only: sets DATA_ARG / WINDOW_ARG / ENCODER_ARG /
# EARLY_STOP_ARG / SEED from the shared config, so both stages -- and every
# other method -- construct an identical node set.
source "$REPO_ROOT/scripts/dblp/_linkpred_data_arg.sh"
echo "[config] TAG=$TAG  MASK_RATIO=$MASK_RATIO  ALPHA=$ALPHA  TARGET=$TARGET  WINDOW=$WINDOW"

# -----------------------------------------------------------------------------
# Stage 1: train MaskDGNN (+ its own end-to-end link evaluation on TEST)
# -----------------------------------------------------------------------------
ENC_DIR="outputs/maskdgnn_${TAG}"
if [[ ! -f "$ENC_DIR/encoder.ckpt" ]]; then
    echo
    echo "================================================================="
    echo "  Stage 1/2: MaskDGNN pretrain -> $ENC_DIR"
    echo "================================================================="
    srun --export=ALL python -m src.baselines.maskdgnn.train_maskdgnn $DATA_ARG \
        $ENCODER_ARG --encoder_kind maskdgnn $WINDOW_ARG $EARLY_STOP_ARG \
        --mask_ratio "$MASK_RATIO" --alpha "$ALPHA" --target "$TARGET" \
        --window "$WINDOW" --neg_per_pos 1 \
        --steps_per_epoch 20 --val_steps 10 \
        --device cuda --epochs 100 --lr 4e-3 \
        --e2e_eval test --num_workers ${SLURM_CPUS_PER_TASK:-8} \
        --seed "$SEED" --out_dir "$ENC_DIR"
    enc_rc=$?
    if [[ $enc_rc -ne 0 || ! -f "$ENC_DIR/encoder.ckpt" ]]; then
        echo "[FATAL] Stage 1 failed (rc=$enc_rc)" >&2; exit 11
    fi
else
    echo "[stage 1] $ENC_DIR/encoder.ckpt exists, skipping MaskDGNN pretrain"
fi

# -----------------------------------------------------------------------------
# Stage 2: freeze the MaskDGNN encoder, fit the shared link-prediction probe
# -----------------------------------------------------------------------------
PROBE_DIR="outputs/maskdgnn_${TAG}_probe"
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
echo "[done] MaskDGNN baseline complete ($DATASET)."
echo "       encoder    : $ENC_DIR/encoder.ckpt"
echo "       end-to-end : $ENC_DIR/link_e2e.json      (paper protocol)"
echo "       probe      : $PROBE_DIR/analysis.json  (link_pred.encoder.{all,new}; compare to DyG-JEPA)"
