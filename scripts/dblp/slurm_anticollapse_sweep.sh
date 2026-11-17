#!/usr/bin/env bash
#SBATCH --job-name=dblp-anticollapse
#SBATCH --output=logs/dblp-anticollapse-%A_%a.out
#SBATCH --error=logs/dblp-anticollapse-%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --time=10:00:00
#SBATCH --partition=gpu
#SBATCH --array=0-3
#
# 2x2 ablation: {EMA target encoder} x {VICReg var+cov regularization}.
# Tests whether latent rank collapse on DBLP is undone by the anti-collapse
# machinery from I-JEPA / V-JEPA / VICReg.
#
# Array indices:
#   0: baseline           (frozen encoder, no VICReg)
#   1: EMA only           (EMA target encoder, no VICReg)
#   2: VICReg only        (frozen encoder + variance/covariance on predictor)
#   3: full               (EMA + VICReg)
#
# All four reuse the same pretrained encoder ckpt (ENC_DIR, default
# outputs/dblp_encoder/). Cells 1 and 3 enable --train_encoder so the EMA
# target has something to track; cells 0 and 2 keep the encoder frozen. Each
# cell then runs the diagnostic (eff_rank, CKA, probe scores).
#
# Submit:    sbatch scripts/dblp/slurm_anticollapse_sweep.sh
# Single:    sbatch --array=3 scripts/dblp/slurm_anticollapse_sweep.sh

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
echo "[slurm] repo_root=$REPO_ROOT  node=$(hostname)  array_id=${SLURM_ARRAY_TASK_ID:-na}"

ENC_DIR=${ENC_DIR:-outputs/dblp_encoder}
SEED=${SEED:-42}
IDX=${SLURM_ARRAY_TASK_ID:-0}

case "$IDX" in
    0) TAG=baseline   ; EMA_FLAGS=""                                            ; VIC_FLAGS="" ;;
    1) TAG=ema        ; EMA_FLAGS="--ema_target --train_encoder"                ; VIC_FLAGS="" ;;
    2) TAG=vicreg     ; EMA_FLAGS=""                                            ; VIC_FLAGS="--vicreg_var 0.5 --vicreg_cov 0.05" ;;
    3) TAG=ema_vicreg ; EMA_FLAGS="--ema_target --train_encoder"                ; VIC_FLAGS="--vicreg_var 0.5 --vicreg_cov 0.05" ;;
    *) echo "[error] unknown SLURM_ARRAY_TASK_ID=$IDX"; exit 2 ;;
esac

DELOS_DIR="outputs/dblp_cjepa_${TAG}"
DIAG_DIR="outputs/diagnostic_dblp_${TAG}"
echo "[config] idx=$IDX  TAG=$TAG  EMA=[$EMA_FLAGS]  VICReg=[$VIC_FLAGS]"
echo "[paths] delos_out=$DELOS_DIR  diagnostic_out=$DIAG_DIR  encoder_in=$ENC_DIR"

DATA_ARG="--synthetic --synthetic_nodes 500 --synthetic_years 30"
if [[ -f dataset/dblp/edges.csv ]]; then
    DATA_ARG="--data_path dataset/dblp/edges.csv"
fi

# Stage 1: DELOS training under this cell's anti-collapse configuration.
echo
echo "================================================================="
echo "  Stage 1/2: DELOS train ($TAG) -> $DELOS_DIR"
echo "================================================================="
srun --export=ALL python src/train/train_delos.py --no_attention_mask $DATA_ARG \
    --encoder_ckpt "$ENC_DIR/encoder.ckpt" \
    --history_size 3 --num_preds 1 \
    --encoder_kind tgcn --encoder_hidden 128 --node_dim 128 \
    --num_masked_nodes 30 --depth 4 --heads 4 --mlp_dim 512 \
    $EMA_FLAGS $VIC_FLAGS \
    --batch_size 8 --num_workers ${SLURM_CPUS_PER_TASK:-8} \
    --device cuda --epochs 50 --lr 5e-4 --probe_epochs 3 \
    --seed "$SEED" --out_dir "$DELOS_DIR"

# Stage 2: representation diagnostic on the resulting checkpoint.
echo
echo "================================================================="
echo "  Stage 2/2: diagnostic ($TAG) -> $DIAG_DIR"
echo "================================================================="
srun --export=ALL python src/train/diagnose_jepa_dblp.py $DATA_ARG \
    --encoder_ckpt "$ENC_DIR/encoder.ckpt" \
    --jepa_ckpt "$DELOS_DIR/best.ckpt" \
    --split test \
    --batch_size 4 --num_workers ${SLURM_CPUS_PER_TASK:-8} \
    --device cuda \
    --out_dir "$DIAG_DIR"

echo
echo "[done] $TAG complete."
echo "  link prediction:   cat $DELOS_DIR/test_metrics.json"
echo "  representation:    cat $DIAG_DIR/diagnostic.json"
