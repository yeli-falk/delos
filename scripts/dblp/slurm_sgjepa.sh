#!/usr/bin/env bash
#SBATCH --job-name=dblp-sgjepa
#SBATCH --output=logs/dblp-sgjepa-%j.out
#SBATCH --error=logs/dblp-sgjepa-%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --time=10:00:00
#SBATCH --partition=gpu
#
# SG-JEPA baseline (Joint Spiking Embedding Predictive Architecture, Zhang et
# al., 2026; arXiv 2607.18412), two-stage so it is directly comparable to the
# DyG-JEPA pipeline:
#   1) train the shared GCN through a spiking (PLIF) context encoder whose
#      spike counts predict the window's LAST-snapshot stop-gradient latent
#      (per-node InfoNCE, other target nodes as negatives)
#   2) freeze it and fit the SAME link-prediction probe DyG-JEPA uses
#
# Outputs:
#   outputs/sgjepa_<TAG>/encoder.ckpt
#   outputs/sgjepa_<TAG>_probe/analysis.json
#
# Run:    DATASET=dblp sbatch scripts/dblp/slurm_sgjepa.sh
# Knobs:  DATASET=reddit_body  SEED=43  TAG=<name>  TAU=0.1  LR=1e-3  WINDOW=0 (0 = history_size)
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
TAU=${TAU:-0.1}
LR=${LR:-1e-3}
WINDOW=${WINDOW:-0}

# Sets DATA_ARG / WINDOW_ARG / ENCODER_ARG / EARLY_STOP_ARG / SEED from the
# shared config, so both stages -- and every other method -- construct an
# identical node set.
source "$REPO_ROOT/scripts/dblp/_linkpred_data_arg.sh"
echo "[config] TAG=$TAG  TAU=$TAU  LR=$LR  WINDOW=$WINDOW"

# -----------------------------------------------------------------------------
# Stage 1: pre-train the SG-JEPA encoder
# -----------------------------------------------------------------------------
ENC_DIR="outputs/sgjepa_${TAG}"
if [[ ! -f "$ENC_DIR/encoder.ckpt" ]]; then
    echo
    echo "================================================================="
    echo "  Stage 1/2: SG-JEPA pretrain -> $ENC_DIR"
    echo "================================================================="
    srun --export=ALL python -m src.baselines.sgjepa.train_sgjepa $DATA_ARG \
        $ENCODER_ARG $WINDOW_ARG $EARLY_STOP_ARG \
        --snn_layers 1 --tau "$TAU" --init_tau 2.0 --v_th 1.0 \
        --window "$WINDOW" --node_batch_size 256 \
        --steps_per_epoch 20 --val_steps 10 \
        --device cuda --epochs 100 --lr "$LR" \
        --seed "$SEED" --out_dir "$ENC_DIR"
    enc_rc=$?
    if [[ $enc_rc -ne 0 || ! -f "$ENC_DIR/encoder.ckpt" ]]; then
        echo "[FATAL] Stage 1 failed (rc=$enc_rc)" >&2; exit 11
    fi
else
    echo "[stage 1] $ENC_DIR/encoder.ckpt exists, skipping SG-JEPA pretrain"
fi

# -----------------------------------------------------------------------------
# Stage 2: freeze the SG-JEPA encoder, fit the shared link-prediction probe
# -----------------------------------------------------------------------------
PROBE_DIR="outputs/sgjepa_${TAG}_probe"
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
echo "[done] SG-JEPA baseline complete ($DATASET)."
echo "       encoder : $ENC_DIR/encoder.ckpt"
echo "       metrics : $PROBE_DIR/analysis.json  (link_pred.encoder.{all,new}; compare to DyG-JEPA)"
