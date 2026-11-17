#!/usr/bin/env bash
#SBATCH --job-name=cjepa-reddit
#SBATCH --output=logs/cjepa-reddit-%j.out
#SBATCH --error=logs/cjepa-reddit-%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --time=10:00:00
#SBATCH --partition=gpu
#
# Full DELOS pipeline on the SNAP Reddit subreddit-hyperlink graph
# (Kumar et al., 2018). This is the HOMOGENEOUS variant — each node is a
# subreddit, edges are post-hyperlinks between subreddits — distinct from
# the bipartite user x subreddit JODIE Reddit.
#
# Stages:
#   1) edge-masking encoder pretraining
#   2) DELOS training (with new-edge probe split in the evaluator)
#   3) representation diagnostic (eff_rank, CKA, structural probes)
#
# Submit (body, the larger variant):
#     sbatch scripts/dblp/slurm_reddit_pipeline.sh
# Submit (title variant):
#     VARIANT=title sbatch scripts/dblp/slurm_reddit_pipeline.sh
# Bigger node subset:
#     TOPK=5000 sbatch scripts/dblp/slurm_reddit_pipeline.sh
# Finer binning:
#     BINS=72 sbatch scripts/dblp/slurm_reddit_pipeline.sh

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

VARIANT=${VARIANT:-body}
SEED=${SEED:-42}
BINS=${BINS:-36}
TOPK=${TOPK:-1000}
COMMUNITY=${COMMUNITY:-0}
COMMUNITY_FLAG=""
if [[ "$COMMUNITY" == "1" ]]; then
    COMMUNITY_FLAG="--community_masking"
    TAG=${TAG:-reddit_${VARIANT}_community}
else
    TAG=${TAG:-reddit_${VARIANT}}
fi
ENC_EPOCHS=${ENC_EPOCHS:-30}
JEPA_EPOCHS=${JEPA_EPOCHS:-50}
PROBE_EPOCHS=${PROBE_EPOCHS:-3}
ENC_LR=${ENC_LR:-1e-3}
JEPA_LR=${JEPA_LR:-5e-4}
PATIENCE=${PATIENCE:-0}    # 0 = no early stopping (legacy default)
echo "[config] VARIANT=$VARIANT  SEED=$SEED  BINS=$BINS  TOPK=$TOPK  COMMUNITY=$COMMUNITY  TAG=$TAG"
echo "[config] ENC_EPOCHS=$ENC_EPOCHS  JEPA_EPOCHS=$JEPA_EPOCHS  PROBE_EPOCHS=$PROBE_EPOCHS  ENC_LR=$ENC_LR  JEPA_LR=$JEPA_LR"

case "$VARIANT" in
    body)  URL="https://snap.stanford.edu/data/soc-redditHyperlinks-body.tsv"  ;;
    title) URL="https://snap.stanford.edu/data/soc-redditHyperlinks-title.tsv" ;;
    *) echo "[error] VARIANT must be 'body' or 'title'"; exit 2 ;;
esac

TSV="dataset/reddit/soc-redditHyperlinks-${VARIANT}.tsv"
if [[ ! -f "$TSV" ]]; then
    echo "[stage 0] downloading Reddit-$VARIANT TSV from $URL"
    wget -O "$TSV" "$URL"
    wget_rc=$?
    sz=$(stat -c %s "$TSV" 2>/dev/null || echo 0)
    echo "[stage 0] wget rc=$wget_rc  size=$sz bytes"
    if [[ $wget_rc -ne 0 || $sz -lt 1000 ]]; then
        echo "[FATAL] download failed at $TSV (rc=$wget_rc, size=$sz)" >&2
        echo "  Compute nodes on most HPC systems lack outbound internet." >&2
        echo "  Workaround: download from the LOGIN node first:" >&2
        echo "    mkdir -p $REPO_ROOT/dataset/reddit && cd $REPO_ROOT" >&2
        echo "    wget $URL -O dataset/reddit/soc-redditHyperlinks-${VARIANT}.tsv" >&2
        echo "  then resubmit; the script will skip the download." >&2
        exit 10
    fi
fi
echo "[stage 0] $TSV size=$(du -h "$TSV" | cut -f1)"

DATA_ARG="--data_source reddit_${VARIANT} --reddit_data_path $TSV --reddit_num_bins $BINS --reddit_top_k $TOPK"

ENC_DIR="outputs/${TAG}_encoder"
DELOS_DIR="outputs/${TAG}_cjepa"
DIAG_DIR="outputs/${TAG}_diagnostic"

# Stage 1: encoder pretraining via edge masking
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
        echo "[FATAL] Stage 1 failed (rc=$enc_rc, ckpt exists=$( [[ -f "$ENC_DIR/encoder.ckpt" ]] && echo y || echo n ))" >&2
        exit 11
    fi
else
    echo "[stage 1] $ENC_DIR/encoder.ckpt already exists, skipping pretrain"
fi

# Stage 2: DELOS training
echo
echo "================================================================="
echo "  Stage 2/3: DELOS train ($TAG) -> $DELOS_DIR"
echo "================================================================="
srun --export=ALL python src/train/train_delos.py --no_attention_mask $DATA_ARG $COMMUNITY_FLAG \
    --encoder_ckpt "$ENC_DIR/encoder.ckpt" \
    --history_size 3 --num_preds 1 \
    --encoder_kind tgcn --encoder_hidden 128 --node_dim 128 \
    --num_masked_nodes 30 --depth 4 --heads 4 --mlp_dim 512 \
    --batch_size 8 --num_workers ${SLURM_CPUS_PER_TASK:-8} \
    --device cuda --epochs "$JEPA_EPOCHS" --lr "$JEPA_LR" --probe_epochs "$PROBE_EPOCHS" \
    --early_stop_patience "$PATIENCE" \
    --seed "$SEED" --out_dir "$DELOS_DIR"
delos_rc=$?
if [[ $delos_rc -ne 0 || ! -f "$DELOS_DIR/best.ckpt" ]]; then
    echo "[FATAL] Stage 2 failed (rc=$delos_rc, best.ckpt exists=$( [[ -f "$DELOS_DIR/best.ckpt" ]] && echo y || echo n ))" >&2
    exit 12
fi

# Stage 3: representation diagnostic
echo
echo "================================================================="
echo "  Stage 3/3: diagnostic ($TAG) -> $DIAG_DIR"
echo "================================================================="
srun --export=ALL python src/train/diagnose_jepa_dblp.py $DATA_ARG \
    --encoder_ckpt "$ENC_DIR/encoder.ckpt" \
    --jepa_ckpt "$DELOS_DIR/best.ckpt" \
    --split test \
    --batch_size 4 --num_workers ${SLURM_CPUS_PER_TASK:-8} \
    --device cuda \
    --out_dir "$DIAG_DIR"

echo
echo "[done] Reddit-$VARIANT pipeline complete."
echo "  link prediction (all / new):  cat $DELOS_DIR/test_metrics.json"
echo "  representation:               cat $DIAG_DIR/diagnostic.json"
