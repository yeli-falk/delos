#!/usr/bin/env bash
#SBATCH --job-name=dblp-dygwm
#SBATCH --output=logs/dblp-dygwm-%j.out
#SBATCH --error=logs/dblp-dygwm-%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --time=10:00:00
#SBATCH --partition=gpu
#
# DyG-WM baseline (Dynamic Graph World Modelling, Li et al., 2026; arXiv
# 2610.19364), SSL-only core, two-stage so it is directly comparable to the
# DyG-JEPA pipeline:
#   1) pre-train its context encoder (--encoder_kind dygwm: shared GCN + node /
#      neighbourhood trajectory GRUs) to predict an EMA target encoder's
#      NEXT-bin node + relation latents (no link / rank loss)
#   2) freeze it and fit the SAME link-prediction probe DyG-JEPA uses
#
# Outputs:
#   outputs/dygwm_<TAG>/encoder.ckpt
#   outputs/dygwm_<TAG>_probe/analysis.json
#
# Run:    DATASET=dblp sbatch scripts/dblp/slurm_dygwm.sh
# Knobs:  DATASET=reddit_body  SEED=43  TAG=<name>  EMA=0.999  LR=3e-4  WINDOW=0 (0 = history_size)
#         Link-pred datasets: dblp, jodie_reddit, jodie_wikipedia, reddit_body,
#         reddit_title (or synthetic). Stage 1 + stage 2 share these flags.
#
# Dataset / encoder / window / early-stop config comes from
# scripts/dblp/_dataset_config.sh so it is identical to every other method.

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
EMA=${EMA:-0.999}
LR=${LR:-3e-4}
WINDOW=${WINDOW:-0}

# Sets DATA_ARG / WINDOW_ARG / ENCODER_ARG / EARLY_STOP_ARG / SEED from the
# shared config, so both stages -- and every other method -- construct an
# identical node set.
source "$REPO_ROOT/scripts/dblp/_linkpred_data_arg.sh"
echo "[config] TAG=$TAG  EMA=$EMA  LR=$LR  WINDOW=$WINDOW"

# -----------------------------------------------------------------------------
# Stage 1: pre-train the DyG-WM encoder
# -----------------------------------------------------------------------------
ENC_DIR="outputs/dygwm_${TAG}"
if [[ ! -f "$ENC_DIR/encoder.ckpt" ]]; then
    echo
    echo "================================================================="
    echo "  Stage 1/2: DyG-WM pretrain -> $ENC_DIR"
    echo "================================================================="
    srun --export=ALL python -m src.baselines.dygwm.train_dygwm $DATA_ARG \
        $ENCODER_ARG --encoder_kind dygwm $WINDOW_ARG $EARLY_STOP_ARG \
        --ema_momentum "$EMA" --n_ctx 32 --decay 0.8 --cn_bonus 1.0 \
        --lam_node 1 --lam_rel 1 --lam_var 0.25 --lam_cov 0.025 \
        --window "$WINDOW" --node_batch_size 256 --pair_batch_size 256 \
        --steps_per_epoch 20 --val_steps 10 \
        --device cuda --epochs 100 --lr "$LR" \
        --seed "$SEED" --out_dir "$ENC_DIR"
    enc_rc=$?
    if [[ $enc_rc -ne 0 || ! -f "$ENC_DIR/encoder.ckpt" ]]; then
        echo "[FATAL] Stage 1 failed (rc=$enc_rc)" >&2; exit 11
    fi
else
    echo "[stage 1] $ENC_DIR/encoder.ckpt exists, skipping DyG-WM pretrain"
fi

# -----------------------------------------------------------------------------
# Stage 2: freeze the DyG-WM encoder, fit the shared link-prediction probe
# -----------------------------------------------------------------------------
PROBE_DIR="outputs/dygwm_${TAG}_probe"
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
echo "[done] DyG-WM baseline complete ($DATASET)."
echo "       encoder : $ENC_DIR/encoder.ckpt"
echo "       metrics : $PROBE_DIR/analysis.json  (link_pred.encoder.{all,new}; compare to DyG-JEPA)"
