#!/usr/bin/env bash
#SBATCH --job-name=reg-lrprobe
#SBATCH --output=logs/reg-lrprobe-%A_%a.out
#SBATCH --error=logs/reg-lrprobe-%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=05:00:00
#SBATCH --partition=gpu
#SBATCH --array=0-5
#
# LR range probe: run this before the hyperparameter search on METR-LA.
#
# With a frozen encoder the targets are static, so a rising train loss is
# optimiser divergence. This probe finds the highest stable learning rate so
# that the search grid can be centred below it.
#
# Design choices:
#   * --lr_min_ratio 1.0 disables the cosine decay, so the LR is constant after
#     warmup; with decay on, a too-high LR can be rescued late by the ramp-down.
#   * Stride 1 and batch 64 as in the real runs, because LR stability depends
#     on the batch size.
#   * A few epochs are enough for a constant LR to show divergence.
#   * One shared pretrained encoder, frozen, for every cell.
#
# Usage:
#   DATASET=metr_la sbatch scripts/regression/slurm_lr_probe.sh
#   python scripts/regression/aggregate_lr_probe.py --root outputs/reg_lrprobe/metr_la
#
# Override the ladder with:  LRS="1e-5 3e-5 1e-4" DATASET=metr_la sbatch --array=0-2 ...

set -uo pipefail
mkdir -p logs

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"
if [[ -d "$HOME/.conda/envs/cjepa" ]]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate cjepa
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

DATASET="${DATASET:-metr_la}"
SEED="${SEED:-42}"
NW="${SLURM_CPUS_PER_TASK:-8}"
PROBE_EPOCHS="${PROBE_EPOCHS:-0}"     # no probe: this measures the JEPA loss only
EPOCHS="${EPOCHS:-4}"

# Log-spaced ladder reaching well below the default search grid.
read -r -a LR_LIST <<< "${LRS:-1e-5 3e-5 1e-4 3e-4 5e-4 1e-3}"
IDX="${SLURM_ARRAY_TASK_ID:-0}"
if (( IDX >= ${#LR_LIST[@]} )); then
    echo "[error] array index $IDX >= ${#LR_LIST[@]} LRs" >&2; exit 2
fi
LR="${LR_LIST[$IDX]}"

TASK=regression
source "$REPO_ROOT/scripts/dblp/_dataset_config.sh"
BATCH="${BATCH:-$REG_BATCH}"
ENC_ARGS="$ENCODER_ARG --bidirectional false"
ATTN_FLAG=""
[[ "$DATASET" == "metr_la" ]] && ATTN_FLAG="--no_attention_mask"

ROOT="outputs/reg_lrprobe/${DATASET}"
OUT="$ROOT/lr_${LR}_s${SEED}"
mkdir -p "$OUT"
echo "[probe $IDX] dataset=$DATASET lr=$LR epochs=$EPOCHS batch=$BATCH -> $OUT"

# Shared frozen encoder, same as the search uses.
ENC_DIR="$ROOT/_shared_encoder_s${SEED}"
ENC_CKPT="$ENC_DIR/encoder_${DATASET}_${ENC_KIND}_masked_signal.ckpt"
mkdir -p "$ENC_DIR"
exec 9>"$ENC_DIR/.pretrain.lock"
flock -w 7200 9 || { echo "[FATAL] encoder lock timeout" >&2; exit 10; }
if [[ ! -f "$ENC_CKPT" ]]; then
    echo "=== shared encoder pretrain (lock held by probe $IDX) ==="
    srun --export=ALL python src/train/pretrain_encoder.py $DATA_ARG \
        --pretext masked_signal $ENC_ARGS --mask_rate 0.25 --decoder_hidden 64 \
        $WINDOW_ARG $EARLY_STOP_ARG \
        --batch_size "$BATCH" --epochs "${ENC_EPOCHS:-300}" --lr 1e-3 \
        --seed "$SEED" --num_workers "$NW" --device cuda --out_dir "$ENC_DIR"
    if [[ ! -f "$ENC_CKPT" ]]; then
        flock -u 9; echo "[FATAL] encoder pretrain failed" >&2; exit 11
    fi
else
    echo "[skip] shared encoder exists"
fi
flock -u 9; exec 9>&-

# --probe_epochs 0 keeps this to the JEPA objective; the question here is
# whether the optimiser is stable, not what the forecast MAE is yet.
srun --export=ALL python src/train/train_delos.py $DATA_ARG $WINDOW_ARG \
    $EARLY_STOP_ARG $ENC_ARGS $ATTN_FLAG \
    --downstream_task regression --mask_scheme node_mask \
    --encoder_ckpt "$ENC_CKPT" \
    --residual_head --persist_residual --change_weight 2.0 \
    --depth "${DEPTH:-6}" --heads "${HEADS:-8}" --mlp_dim "${MLP_DIM:-512}" \
    --num_masked_nodes "${NMASK:-20}" \
    --batch_size "$BATCH" --epochs "$EPOCHS" --lr "$LR" --weight_decay 1e-4 \
    --warmup_epochs 1.0 --lr_min_ratio 1.0 \
    --probe_epochs "$PROBE_EPOCHS" --probe_split val \
    --early_stop_patience 0 \
    --seed "$SEED" --num_workers "$NW" --device cuda --out_dir "$OUT"

echo "[done] lr=$LR -> $OUT/history.json"
