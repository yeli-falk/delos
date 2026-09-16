#!/usr/bin/env bash
#SBATCH --job-name=cjepa-featmask
#SBATCH --output=logs/cjepa-featmask-%j.out
#SBATCH --error=logs/cjepa-featmask-%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --time=10:00:00
#SBATCH --partition=gpu
#
# Reddit-Body + feature-mask pretext, parametrised by SEED and TOPK so we
# can use it for two sweeps in one script:
#
#   Multi-seed verification (TOPK fixed, SEED varied):
#       for S in 42 43 44 45 46; do
#         SEED=$S TOPK=1000 sbatch scripts/dblp/slurm_featmask_reddit.sh
#       done
#
#   K-sweep (SEED fixed, TOPK varied):
#       for K in 500 2000 3000 5000; do
#         SEED=42 TOPK=$K sbatch scripts/dblp/slurm_featmask_reddit.sh
#       done
#
# Background:
#   Cell 3 of slurm_pretext_sweep.sh found that feature-mask pretraining
#   on Reddit-Body (with LIWC PROPERTIES loaded as node features) gives
#   MRR_all 0.243, MRR_new 0.237, new/all ratio 0.97 — a 2x improvement
#   over the same dataset under edge-mask pretraining, and the first real
#   data showing JEPA escape the degree shortcut. This sweep checks (a)
#   whether the result holds across seeds, and (b) whether it scales as
#   we widen the kept-subreddit subset.

set -uo pipefail
mkdir -p logs outputs dataset/reddit

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
TOPK=${TOPK:-1000}
BINS=${BINS:-36}
VARIANT=${VARIANT:-body}   # 'body' or 'title'
case "$VARIANT" in
    body|title) ;;
    *) echo "[error] VARIANT must be 'body' or 'title' (got $VARIANT)"; exit 2 ;;
esac
# Tag distinguishes axes so multi-seed at the same K, K-sweep at the same
# seed, and the variant produce distinct output directories.
TAG=${TAG:-reddit_${VARIANT}_featmask_k${TOPK}_s${SEED}}
ENC_EPOCHS=${ENC_EPOCHS:-30}
JEPA_EPOCHS=${JEPA_EPOCHS:-50}
PROBE_EPOCHS=${PROBE_EPOCHS:-3}
ENC_LR=${ENC_LR:-1e-3}
JEPA_LR=${JEPA_LR:-5e-4}
PATIENCE=${PATIENCE:-0}
echo "[config] VARIANT=$VARIANT  SEED=$SEED  TOPK=$TOPK  BINS=$BINS  TAG=$TAG"
echo "[config] ENC_EPOCHS=$ENC_EPOCHS  JEPA_EPOCHS=$JEPA_EPOCHS  PROBE_EPOCHS=$PROBE_EPOCHS"

TSV="dataset/reddit/soc-redditHyperlinks-${VARIANT}.tsv"
if [[ ! -f "$TSV" ]]; then
    echo "[FATAL] $TSV missing — download on the login node first." >&2
    exit 10
fi

DATA_ARG="--data_source reddit_${VARIANT} --reddit_data_path $TSV --reddit_num_bins $BINS --reddit_top_k $TOPK --with_edge_features"

ENC_DIR="outputs/${TAG}_encoder"
CJEPA_DIR="outputs/${TAG}_cjepa"
DIAG_DIR="outputs/${TAG}_diagnostic"

# -----------------------------------------------------------------------------
# Stage 1: feature-mask encoder pretraining
# -----------------------------------------------------------------------------
if [[ ! -f "$ENC_DIR/encoder.ckpt" ]]; then
    echo
    echo "================================================================="
    echo "  Stage 1/3: encoder pretrain ($TAG) -> $ENC_DIR"
    echo "================================================================="
    srun --export=ALL python src/train/pretrain_encoder.py $DATA_ARG \
        --encoder_kind tgcn --encoder_hidden 128 --node_dim 128 \
        --decoder_hidden 64 --mask_rate 0.25 \
        --pretext feature_mask \
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
    echo "[FATAL] Stage 2 failed (rc=$cjepa_rc)" >&2
    exit 12
fi

# -----------------------------------------------------------------------------
# Stage 3: representation diagnostic
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
