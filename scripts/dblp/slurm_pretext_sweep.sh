#!/usr/bin/env bash
#SBATCH --job-name=cjepa-pretext
#SBATCH --output=logs/cjepa-pretext-%A_%a.out
#SBATCH --error=logs/cjepa-pretext-%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --array=0-3
#
# Pretext-task ablation: does feature-mask pretraining (which has no
# degree shortcut) escape the rank-1 collapse that edge-mask pretraining
# falls into? 2x2 = {edge_mask, feature_mask} x {jodie_reddit, reddit_body}.
#
# All cells use --with_edge_features so the dataset carries the per-edge
# LIWC features aggregated to per-(bin, node) means (172 dims for JODIE,
# 86 dims for Reddit-Body). edge_mask cells just ignore the LIWC channels;
# feature_mask cells supervise the encoder to reconstruct them.
#
# Cells:
#   0: jodie_reddit  + edge_mask     (current baseline)
#   1: jodie_reddit  + feature_mask  (does the new pretext help?)
#   2: reddit_body   + edge_mask     (current baseline)
#   3: reddit_body   + feature_mask  (does the new pretext help?)
#
# Submit (all 4):
#     sbatch scripts/dblp/slurm_pretext_sweep.sh
# Submit (just the new feature_mask cells):
#     sbatch --array=1,3 scripts/dblp/slurm_pretext_sweep.sh

set -uo pipefail
mkdir -p logs outputs dataset/jodie dataset/reddit

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"

if [[ -d "$HOME/.conda/envs/cjepa" ]]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate cjepa
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo "[slurm] repo_root=$REPO_ROOT  node=$(hostname)  array_id=${SLURM_ARRAY_TASK_ID:-na}"

SEED=${SEED:-42}
IDX=${SLURM_ARRAY_TASK_ID:-0}

case "$IDX" in
    0) DATASET=jodie_reddit ; PRETEXT=edge_mask    ; ARGS="--jodie_num_bins 30" ; CSV=dataset/jodie/reddit.csv     ;;
    1) DATASET=jodie_reddit ; PRETEXT=feature_mask ; ARGS="--jodie_num_bins 30" ; CSV=dataset/jodie/reddit.csv     ;;
    2) DATASET=reddit_body  ; PRETEXT=edge_mask    ; ARGS="--reddit_top_k 1000 --reddit_num_bins 36"                            ; CSV=dataset/reddit/soc-redditHyperlinks-body.tsv ;;
    3) DATASET=reddit_body  ; PRETEXT=feature_mask ; ARGS="--reddit_top_k 1000 --reddit_num_bins 36"                            ; CSV=dataset/reddit/soc-redditHyperlinks-body.tsv ;;
    *) echo "[error] unknown array idx $IDX"; exit 2 ;;
esac

TAG="${DATASET}_${PRETEXT}"
ENC_DIR="outputs/${TAG}_encoder"
CJEPA_DIR="outputs/${TAG}_cjepa"
DIAG_DIR="outputs/${TAG}_diagnostic"

echo "[config] idx=$IDX  DATASET=$DATASET  PRETEXT=$PRETEXT  TAG=$TAG"
echo "[paths] CSV=$CSV  ENC=$ENC_DIR  JEPA=$CJEPA_DIR  DIAG=$DIAG_DIR"

if [[ ! -f "$CSV" ]]; then
    echo "[FATAL] data file not found at $CSV — download on a login node first" >&2
    exit 10
fi

# Data flags: every cell loads with edge features (the dataset is the same
# regardless of pretext; only the training objective differs).
case "$DATASET" in
    jodie_reddit) DATA_ARG="--data_source jodie_reddit --jodie_data_path $CSV $ARGS --with_edge_features" ;;
    reddit_body)  DATA_ARG="--data_source reddit_body  --reddit_data_path $CSV $ARGS --with_edge_features" ;;
esac

# -----------------------------------------------------------------------------
# Stage 1: encoder pretraining (pretext-dependent)
# -----------------------------------------------------------------------------
if [[ ! -f "$ENC_DIR/encoder.ckpt" ]]; then
    echo
    echo "================================================================="
    echo "  Stage 1/3: encoder pretrain ($TAG) -> $ENC_DIR"
    echo "================================================================="
    srun --export=ALL python src/train/pretrain_encoder.py $DATA_ARG \
        --encoder_kind tgcn --encoder_hidden 128 --node_dim 128 \
        --decoder_hidden 64 --mask_rate 0.25 \
        --pretext "$PRETEXT" \
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
    echo "[stage 1] $ENC_DIR/encoder.ckpt already exists, skipping pretrain"
fi

# -----------------------------------------------------------------------------
# Stage 2: C-JEPA training (same in both cells; only the encoder differs)
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
    --device cuda --epochs 50 --lr 5e-4 --probe_epochs 3 \
    --seed "$SEED" --out_dir "$CJEPA_DIR"
cjepa_rc=$?
if [[ $cjepa_rc -ne 0 || ! -f "$CJEPA_DIR/best.ckpt" ]]; then
    echo "[FATAL] Stage 2 failed (rc=$cjepa_rc)" >&2
    exit 12
fi

# -----------------------------------------------------------------------------
# Stage 3: representation diagnostic — eff_rank is the headline measurement
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
echo "[done] $TAG complete."
echo "  link prediction (all / new):  cat $CJEPA_DIR/test_metrics.json"
echo "  representation:               cat $DIAG_DIR/diagnostic.json"
