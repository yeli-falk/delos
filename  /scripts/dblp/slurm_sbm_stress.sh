#!/usr/bin/env bash
#SBATCH --job-name=cjepa-sbm
#SBATCH --output=logs/cjepa-sbm-%j.out
#SBATCH --error=logs/cjepa-sbm-%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --time=08:00:00
#SBATCH --partition=gpu
#
# Architecture-vs-data ceiling stress test.
#
# Trains the full C-JEPA pipeline on a synthetic stochastic block model
# (default: 8 blocks, 1000 nodes, 30 years) and reports effective rank +
# probe metrics. The SBM has structural rank ~ log2(num_blocks) by
# construction; if the model recovers an effective rank close to that, the
# architecture is capable of >1 rank and the DBLP rank-1 finding is a
# data ceiling. If the model still gives eff_rank ~ 1 here, the
# architecture itself is the bottleneck.
#
# Stages:
#   1) edge-masking encoder pretraining on SBM
#   2) C-JEPA training (baseline configuration)
#   3) diagnostic
#
# Submit:    sbatch scripts/dblp/slurm_sbm_stress.sh
# Variants:  BLOCKS=16 NODES=2000 sbatch scripts/dblp/slurm_sbm_stress.sh

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
COMMUNITY=${COMMUNITY:-0}
COMMUNITY_FLAG=""
if [[ "$COMMUNITY" == "1" ]]; then
    COMMUNITY_FLAG="--community_masking"
fi
NODES=${NODES:-1000}
YEARS=${YEARS:-30}
BLOCKS=${BLOCKS:-8}
if [[ "$COMMUNITY" == "1" ]]; then
    TAG=${TAG:-sbm${BLOCKS}_community}
else
    TAG=${TAG:-sbm${BLOCKS}}
fi
echo "[config] SEED=$SEED  NODES=$NODES  YEARS=$YEARS  BLOCKS=$BLOCKS  COMMUNITY=$COMMUNITY  TAG=$TAG"

DATA_ARG="--synthetic --synthetic_kind sbm --synthetic_nodes $NODES --synthetic_years $YEARS --synthetic_blocks $BLOCKS"

ENC_DIR="outputs/${TAG}_encoder"
CJEPA_DIR="outputs/${TAG}_cjepa"
DIAG_DIR="outputs/${TAG}_diagnostic"

# -----------------------------------------------------------------------------
# Stage 1: encoder pretraining via edge masking
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
        --device cuda --epochs 30 --lr 1e-3 \
        --seed "$SEED" --out_dir "$ENC_DIR"
    enc_rc=$?
    if [[ $enc_rc -ne 0 || ! -f "$ENC_DIR/encoder.ckpt" ]]; then
        echo "[FATAL] Stage 1 failed (rc=$enc_rc, ckpt exists=$( [[ -f "$ENC_DIR/encoder.ckpt" ]] && echo y || echo n ))" >&2
        exit 11
    fi
else
    echo "[stage 1] $ENC_DIR/encoder.ckpt already exists, skipping pretrain"
fi

# -----------------------------------------------------------------------------
# Stage 2: C-JEPA training (baseline configuration — no EMA/VICReg to keep
# the comparison clean against the baseline DBLP cell).
# -----------------------------------------------------------------------------
echo
echo "================================================================="
echo "  Stage 2/3: C-JEPA train ($TAG) -> $CJEPA_DIR"
echo "================================================================="
srun --export=ALL python src/train/train_dyg_jepa.py --no_attention_mask $DATA_ARG $COMMUNITY_FLAG \
    --encoder_ckpt "$ENC_DIR/encoder.ckpt" \
    --history_size 3 --num_preds 1 \
    --encoder_kind tgcn --encoder_hidden 128 --node_dim 128 \
    --num_masked_nodes 30 --depth 4 --heads 4 --mlp_dim 512 \
    --batch_size 8 --num_workers ${SLURM_CPUS_PER_TASK:-8} \
    --device cuda --epochs 50 --lr 5e-4 --probe_epochs 3 \
    --seed "$SEED" --out_dir "$CJEPA_DIR"
cjepa_rc=$?
if [[ $cjepa_rc -ne 0 || ! -f "$CJEPA_DIR/best.ckpt" ]]; then
    echo "[FATAL] Stage 2 failed (rc=$cjepa_rc, best.ckpt exists=$( [[ -f "$CJEPA_DIR/best.ckpt" ]] && echo y || echo n ))" >&2
    exit 12
fi

# -----------------------------------------------------------------------------
# Stage 3: representation diagnostic.
# -----------------------------------------------------------------------------
echo
echo "================================================================="
echo "  Stage 3/3: diagnostic ($TAG) -> $DIAG_DIR"
echo "================================================================="
srun --export=ALL python src/train/diagnose_jepa_dblp.py $DATA_ARG \
    --encoder_ckpt "$ENC_DIR/encoder.ckpt" \
    --jepa_ckpt "$CJEPA_DIR/best.ckpt" \
    --split test \
    --batch_size 4 --num_workers ${SLURM_CPUS_PER_TASK:-8} \
    --device cuda \
    --out_dir "$DIAG_DIR"

echo
echo "[done] SBM stress test ($TAG) complete."
echo "  link prediction (all / new):  cat $CJEPA_DIR/test_metrics.json"
echo "  representation:               cat $DIAG_DIR/diagnostic.json"
echo
echo "Expected if architecture is capable: eff_rank well above 1 (target"
echo "roughly log2($BLOCKS) = $(python3 -c "import math; print(round(math.log2($BLOCKS), 2))"))."
