#!/usr/bin/env bash
#SBATCH --job-name=cjepa-dblp
#SBATCH --output=logs/cjepa-dblp-%j.out
#SBATCH --error=logs/cjepa-dblp-%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --time=10:00:00
#SBATCH --partition=gpu
#
# DBLP coauthorship pipeline (encoder pretrain -> C-JEPA -> diagnostic).
# Mirrors slurm_reddit_pipeline.sh but for DBLP. Used here primarily for
# cross-dataset verification of the long-training rank-gap result.
#
# Data:
#   - Looks for dataset/dblp/edges.csv (the parsed coauthorship dump).
#   - Falls back to synthetic DBLP if SYNTHETIC=1 (smoke / sanity only).
#
# Long-training (the verification sweep):
#   for S in 42 43 44 45 46; do
#     SEED=$S TAG=dblp_long_s$S \
#       ENC_EPOCHS=100 JEPA_EPOCHS=300 PROBE_EPOCHS=20 PATIENCE=20 \
#       sbatch --time=12:00:00 scripts/dblp/slurm_dblp_pipeline.sh
#   done
#
# Submit defaults (50 ep, matches the original 5-seed numbers):
#   SEED=42 sbatch scripts/dblp/slurm_dblp_pipeline.sh

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

SEED=${SEED:-42}
SYNTHETIC=${SYNTHETIC:-0}
SYNTHETIC_NODES=${SYNTHETIC_NODES:-500}
SYNTHETIC_YEARS=${SYNTHETIC_YEARS:-30}
ENC_EPOCHS=${ENC_EPOCHS:-30}
JEPA_EPOCHS=${JEPA_EPOCHS:-50}
PROBE_EPOCHS=${PROBE_EPOCHS:-3}
ENC_LR=${ENC_LR:-1e-3}
JEPA_LR=${JEPA_LR:-5e-4}
PATIENCE=${PATIENCE:-0}
TAG=${TAG:-dblp_s${SEED}}
echo "[config] SEED=$SEED  SYNTHETIC=$SYNTHETIC  TAG=$TAG"
echo "[config] ENC_EPOCHS=$ENC_EPOCHS  JEPA_EPOCHS=$JEPA_EPOCHS  PROBE_EPOCHS=$PROBE_EPOCHS  PATIENCE=$PATIENCE"

if [[ "$SYNTHETIC" == "1" ]]; then
    DATA_ARG="--synthetic --synthetic_kind dblp --synthetic_nodes $SYNTHETIC_NODES --synthetic_years $SYNTHETIC_YEARS"
    echo "[data] using synthetic DBLP (N=$SYNTHETIC_NODES years=$SYNTHETIC_YEARS)"
else
    DBLP_CSV="dataset/dblp/edges.csv"
    if [[ ! -f "$DBLP_CSV" ]]; then
        echo "[FATAL] $DBLP_CSV missing." >&2
        echo "  Either upload the DBLP edges CSV (src,dst,year columns)," >&2
        echo "  or pass SYNTHETIC=1 for the synthetic-data fallback." >&2
        exit 10
    fi
    DATA_ARG="--data_path $DBLP_CSV"
    echo "[data] using real DBLP CSV at $DBLP_CSV"
fi

ENC_DIR="outputs/${TAG}_encoder"
CJEPA_DIR="outputs/${TAG}_cjepa"
DIAG_DIR="outputs/${TAG}_diagnostic"

# -----------------------------------------------------------------------------
# Stage 1: encoder pretraining (edge-masking pretext)
# -----------------------------------------------------------------------------
if [[ ! -f "$ENC_DIR/encoder.ckpt" ]]; then
    echo
    echo "================================================================="
    echo "  Stage 1/3: encoder pretrain ($TAG) -> $ENC_DIR"
    echo "================================================================="
    srun --export=ALL python src/train/pretrain_encoder.py $DATA_ARG \
        --encoder_kind tgcn --encoder_hidden 128 --node_dim 128 \
        --decoder_hidden 64 --mask_rate 0.25 \
        --batch_size 8 --num_workers ${SLURM_CPUS_PER_TASK:-8} \
        --device cuda --epochs "$ENC_EPOCHS" --lr "$ENC_LR" \
        --early_stop_patience "$PATIENCE" \
        --history_size 3 --num_preds 1 \
        --seed "$SEED" --out_dir "$ENC_DIR"
    enc_rc=$?
    if [[ $enc_rc -ne 0 || ! -f "$ENC_DIR/encoder.ckpt" ]]; then
        echo "[FATAL] Stage 1 failed (rc=$enc_rc)" >&2
        exit 11
    fi
else
    echo "[stage 1] $ENC_DIR/encoder.ckpt exists, skipping pretrain"
fi

# -----------------------------------------------------------------------------
# Stage 2: C-JEPA training
# -----------------------------------------------------------------------------
echo
echo "================================================================="
echo "  Stage 2/3: C-JEPA train ($TAG) -> $CJEPA_DIR"
echo "================================================================="
srun --export=ALL python src/train/train_dyg_jepa.py --no_attention_mask $DATA_ARG \
    --encoder_ckpt "$ENC_DIR/encoder.ckpt" \
    --history_size 3 --num_preds 1 \
    --encoder_kind tgcn --encoder_hidden 128 --node_dim 128 \
    --num_masked_nodes 30 --depth 4 --heads 4 --mlp_dim 512 \
    --batch_size 8 --num_workers ${SLURM_CPUS_PER_TASK:-8} \
    --device cuda --epochs "$JEPA_EPOCHS" --lr "$JEPA_LR" --probe_epochs "$PROBE_EPOCHS" \
    --early_stop_patience "$PATIENCE" \
    --seed "$SEED" --out_dir "$CJEPA_DIR"
cjepa_rc=$?
if [[ $cjepa_rc -ne 0 || ! -f "$CJEPA_DIR/best.ckpt" ]]; then
    echo "[FATAL] Stage 2 failed (rc=$cjepa_rc, best.ckpt exists=$( [[ -f "$CJEPA_DIR/best.ckpt" ]] && echo y || echo n ))" >&2
    exit 12
fi

# -----------------------------------------------------------------------------
# Stage 3: diagnostic (with Louvain block probe)
# -----------------------------------------------------------------------------
echo
echo "================================================================="
echo "  Stage 3/3: diagnostic ($TAG) -> $DIAG_DIR"
echo "================================================================="
srun --export=ALL python src/train/diagnose_jepa_dblp.py $DATA_ARG \
    --encoder_ckpt "$ENC_DIR/encoder.ckpt" \
    --jepa_ckpt "$CJEPA_DIR/best.ckpt" \
    --split test --device cuda \
    --batch_size 4 --num_workers ${SLURM_CPUS_PER_TASK:-8} \
    --block_probe louvain \
    --out_dir "$DIAG_DIR"

echo
echo "[done] $TAG complete."
echo "  link prediction:  cat $CJEPA_DIR/test_metrics.json"
echo "  diagnostic:       cat $DIAG_DIR/diagnostic.json"
