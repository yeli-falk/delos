#!/usr/bin/env bash
#SBATCH --job-name=cjepa-train
#SBATCH --output=logs/cjepa-train-%j.out
#SBATCH --error=logs/cjepa-train-%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --time=08:00:00
#SBATCH --partition=gpu
#
# DyG-JEPA training, parameterised over --data_source. Auto-detects a matching
# pretrained encoder under outputs/encoders/<dataset>_encoder/ (from
# slurm_pretrain_encoder.sh); trains from scratch if none is found.
#
# Pick the dataset with the DATASET env var (default: dblp), e.g.
#   sbatch --export=ALL,DATASET=metr_la scripts/dblp/slurm_train.sh
# Extra args are forwarded to train_dyg_jepa.py and override the defaults below.

set -euo pipefail

DATASET="${DATASET:-dblp}"

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"
echo "[slurm] repo_root=$REPO_ROOT dataset=$DATASET"

OUT_DIR="outputs/${DATASET}_cjepa"
mkdir -p logs "$OUT_DIR"

if [[ -d "$HOME/.conda/envs/cjepa" ]]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate cjepa
fi

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

# Per-dataset config: data source + flags, downstream task (fixed/dynamic
# topology graphs forecast a signal -> regression; evolving graphs -> link_pred),
# the from-scratch encoder kind, and history/horizon/batch. All overridable via "$@".
ENC_KIND=tgcn   # only used when no pretrained checkpoint is found (random init)
case "$DATASET" in
    dblp)
        if [[ ! -f dataset/dblp/edges.csv ]]; then
            echo "[error] dataset/dblp/edges.csv not found" >&2; exit 1
        fi
        DATA_ARG="--data_source dblp --data_path dataset/dblp/edges.csv"
        DOWNSTREAM="link_pred"; SHAPE="--history_size 3 --num_preds 1"; BS=8
        ;;
    jodie_reddit|jodie_wikipedia|reddit_body|reddit_title)
        DATA_ARG="--data_source $DATASET --with_edge_features"
        DOWNSTREAM="link_pred"; SHAPE="--history_size 3 --num_preds 1"; BS=8
        ;;
    ehrshot)
        DATA_ARG="--data_source ehrshot --ehrshot_data_path dataset/ehrshot/ehrshot_cohort.pkl"
        DOWNSTREAM="link_pred"; SHAPE="--history_size 3 --num_preds 1"; BS=8
        ;;
    metr_la)
        DATA_ARG="--data_source metr_la --metr_la_data_dir dataset/metr_la"
        DOWNSTREAM="regression"; SHAPE="--history_size 12 --num_preds 12"; BS=64
        ENC_KIND=gcn
        ;;
    england_covid)
        DATA_ARG="--data_source england_covid --england_covid_data_dir dataset/england_covid"
        DOWNSTREAM="regression"; SHAPE="--history_size 8 --num_preds 1"; BS=32
        ENC_KIND=gcn
        ;;
    *)
        echo "[error] unknown DATASET='$DATASET'. Supported: dblp, jodie_reddit," \
             "jodie_wikipedia, reddit_body, reddit_title, ehrshot, metr_la, england_covid" >&2
        exit 1
        ;;
esac

# Auto-detect a pretrained encoder in the new location. slurm_pretrain_encoder.sh
# writes outputs/encoders/<dataset>_encoder/encoder_<src>_<kind>_<pretext>.ckpt.
# The trainer reads the encoder kind/dims back from the ckpt, so match any kind
# (newest wins). 
ENC_DIR="outputs/encoders/${DATASET}_encoder"
ENC_ARG=""
ENC_CKPT=$(ls -t "$ENC_DIR"/encoder_*.ckpt 2>/dev/null | head -1)
if [[ -n "$ENC_CKPT" ]]; then
    ENC_ARG="--encoder_ckpt $ENC_CKPT"
    echo "[encoder] will load $ENC_CKPT"
else
    echo "[encoder] not found in $ENC_DIR; training encoder from scratch"
fi

srun --export=ALL python src/train/train_dyg_jepa.py --no_attention_mask $DATA_ARG $ENC_ARG \
    --downstream_task "$DOWNSTREAM" $SHAPE \
    --encoder_kind "$ENC_KIND" \
    --encoder_hidden 128 \
    --node_dim 128 \
    --num_masked_nodes 30 \
    --depth 4 \
    --heads 4 \
    --mlp_dim 512 \
    --batch_size "$BS" \
    --num_workers ${SLURM_CPUS_PER_TASK:-8} \
    --device cuda \
    --epochs 500 \
    --lr 5e-4 \
    --probe_epochs 3 \
    --out_dir "$OUT_DIR" \
    "$@"
