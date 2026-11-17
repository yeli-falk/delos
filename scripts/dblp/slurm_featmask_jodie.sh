#!/usr/bin/env bash
#SBATCH --job-name=cjepa-featmask-jodie
#SBATCH --output=logs/cjepa-featmask-jodie-%j.out
#SBATCH --error=logs/cjepa-featmask-jodie-%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --time=10:00:00
#SBATCH --partition=gpu
#
# JODIE-Reddit / JODIE-Wikipedia + feature-mask pretext + LIWC features.
# Parametrised by SEED, VARIANT — drives:
#   B) multi-seed on Reddit (jepa-vs-encoder gap stability)
#   C) Wikipedia variant (cross-dataset replication)
#
# Usage examples:
#   # Reddit default (seed swept by caller):
#   SEED=43 sbatch scripts/dblp/slurm_featmask_jodie.sh
#   # Wikipedia, seed 42:
#   VARIANT=wikipedia sbatch scripts/dblp/slurm_featmask_jodie.sh

set -uo pipefail
mkdir -p logs outputs dataset/jodie

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"

if [[ -d "$HOME/.conda/envs/cjepa" ]]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate cjepa
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo "[slurm] repo_root=$REPO_ROOT  node=$(hostname)"

VARIANT=${VARIANT:-reddit}   # 'reddit' or 'wikipedia'
case "$VARIANT" in
    reddit|wikipedia) ;;
    *) echo "[error] VARIANT must be 'reddit' or 'wikipedia' (got $VARIANT)"; exit 2 ;;
esac
SEED=${SEED:-42}
BINS=${BINS:-30}
TAG=${TAG:-jodie_${VARIANT}_featmask_s${SEED}}
echo "[config] VARIANT=$VARIANT  SEED=$SEED  BINS=$BINS  TAG=$TAG"

CSV="dataset/jodie/${VARIANT}.csv"
if [[ ! -f "$CSV" ]]; then
    echo "[FATAL] $CSV missing — download on the login node first." >&2
    exit 10
fi

DATA_ARG="--data_source jodie_${VARIANT} --jodie_data_path $CSV --jodie_num_bins $BINS --with_edge_features"

ENC_DIR="outputs/${TAG}_encoder"
DELOS_DIR="outputs/${TAG}_cjepa"
DIAG_DIR="outputs/${TAG}_diagnostic"

# -----------------------------------------------------------------------------
# Stage 1: feature-mask encoder pretraining
# -----------------------------------------------------------------------------
if [[ ! -f "$ENC_DIR/encoder.ckpt" ]]; then
    echo
    echo "================================================================="
    echo "  Stage 1/4: encoder pretrain ($TAG) -> $ENC_DIR"
    echo "================================================================="
    srun --export=ALL python src/train/pretrain_encoder.py $DATA_ARG \
        --encoder_kind tgcn --encoder_hidden 128 --node_dim 128 \
        --decoder_hidden 64 --mask_rate 0.25 \
        --pretext feature_mask \
        --batch_size 8 --num_workers ${SLURM_CPUS_PER_TASK:-8} \
        --device cuda --epochs 30 --lr 1e-3 \
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
# Stage 2: DELOS training
# -----------------------------------------------------------------------------
echo
echo "================================================================="
echo "  Stage 2/4: DELOS train ($TAG) -> $DELOS_DIR"
echo "================================================================="
srun --export=ALL python src/train/train_delos.py --no_attention_mask $DATA_ARG \
    --encoder_ckpt "$ENC_DIR/encoder.ckpt" \
    --history_size 3 --num_preds 1 \
    --encoder_kind tgcn --encoder_hidden 128 --node_dim 128 \
    --num_masked_nodes 30 --depth 4 --heads 4 --mlp_dim 512 \
    --batch_size 8 --num_workers ${SLURM_CPUS_PER_TASK:-8} \
    --device cuda --epochs 50 --lr 5e-4 --probe_epochs 3 \
    --seed "$SEED" --out_dir "$DELOS_DIR"
delos_rc=$?
if [[ $delos_rc -ne 0 || ! -f "$DELOS_DIR/best.ckpt" ]]; then
    echo "[FATAL] Stage 2 failed (rc=$delos_rc)" >&2
    exit 12
fi

# -----------------------------------------------------------------------------
# Stage 3: representation diagnostic (Louvain block probe optional)
# -----------------------------------------------------------------------------
echo
echo "================================================================="
echo "  Stage 3/4: diagnostic ($TAG) -> $DIAG_DIR"
echo "================================================================="
srun --export=ALL python src/train/diagnose_jepa_dblp.py $DATA_ARG \
    --encoder_ckpt "$ENC_DIR/encoder.ckpt" \
    --jepa_ckpt "$DELOS_DIR/best.ckpt" \
    --split test \
    --batch_size 4 --num_workers ${SLURM_CPUS_PER_TASK:-8} \
    --device cuda --block_probe louvain \
    --out_dir "$DIAG_DIR"

# -----------------------------------------------------------------------------
# Stage 4: state-label probe (the headline external-label experiment)
# -----------------------------------------------------------------------------
echo
echo "================================================================="
echo "  Stage 4/4: state-label probe ($TAG)"
echo "================================================================="
srun --export=ALL python scripts/dblp/run_block_probe.py \
    --jepa_ckpt "$DELOS_DIR/best.ckpt" \
    --task binary \
    --device cuda \
    --batch_size 4 --num_workers ${SLURM_CPUS_PER_TASK:-8}

echo
echo "[done] $TAG complete."
echo "  link prediction:   cat $DELOS_DIR/test_metrics.json"
echo "  representation:    cat $DIAG_DIR/diagnostic.json"
echo "  state-label AUROC: cat $DELOS_DIR/state_probe.json"
