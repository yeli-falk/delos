#!/usr/bin/env bash
#SBATCH --job-name=pretrain-encoder
#SBATCH --output=logs/pretrain-encoder-%j.out
#SBATCH --error=logs/pretrain-encoder-%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#
# GraphMAE-style encoder pretraining, parameterised over --data_source.
# Produces outputs/encoders/<dataset>_encoder/encoder_<...>.ckpt for slurm_train.sh.
#
# Pick the dataset with the DATASET env var (default: dblp), e.g.
#   sbatch --export=ALL,DATASET=metr_la      scripts/dblp/slurm_pretrain_encoder.sh
#   sbatch --export=ALL,DATASET=ehrshot       scripts/dblp/slurm_pretrain_encoder.sh
#   sbatch --export=ALL,DATASET=jodie_reddit  scripts/dblp/slurm_pretrain_encoder.sh
# Any extra args are forwarded to pretrain_encoder.py and override the
# per-dataset defaults below, e.g.  sbatch ... -- --epochs 100 --lr 5e-4

set -euo pipefail

DATASET="${DATASET:-dblp}"

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"
echo "[slurm] repo_root=$REPO_ROOT dataset=$DATASET"

OUT_DIR="outputs/encoders/${DATASET}_encoder"
mkdir -p logs "$OUT_DIR"

if [[ -d "$HOME/.conda/envs/cjepa" ]]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate cjepa
fi

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

# Per-dataset defaults: the --data_source dispatch, the pretext task that suits
# its topology (fixed-topology graphs only support masked_signal), the encoder
# backbone, and reasonable history/horizon/batch sizes. All overridable via "$@".
case "$DATASET" in
    dblp)
        # Default: real DBLP edges if dataset/dblp/edges.csv exists, else synthetic.
        if [[ -f dataset/dblp/edges.csv ]]; then
            DATA_ARG="--data_source dblp --data_path dataset/dblp/edges.csv"
            echo "[data] using real DBLP edges from dataset/dblp/edges.csv"
        else
            DATA_ARG="--data_source dblp --synthetic --synthetic_nodes 500 --synthetic_years 30"
            echo "[data] no edges.csv found; using synthetic graph"
        fi
        DS_ARGS="--pretext edge_mask --encoder_kind tgcn --history_size 3 --num_preds 1 --batch_size 4"
        ;;
    jodie_reddit|jodie_wikipedia)
        DATA_ARG="--data_source $DATASET --with_edge_features"
        DS_ARGS="--pretext feature_mask --encoder_kind tgcn --history_size 3 --num_preds 1 --batch_size 4"
        ;;
    reddit_body|reddit_title)
        DATA_ARG="--data_source $DATASET --with_edge_features"
        DS_ARGS="--pretext feature_mask --encoder_kind tgcn --history_size 3 --num_preds 1 --batch_size 4"
        ;;
    ehrshot)
        DATA_ARG="--data_source ehrshot --ehrshot_data_path dataset/ehrshot/ehrshot_cohort.pkl"
        DS_ARGS="--pretext feature_mask --encoder_kind tgcn --history_size 3 --num_preds 1 --batch_size 4"
        ;;
    metr_la)
        DATA_ARG="--data_source metr_la --metr_la_data_dir dataset/metr_la"
        DS_ARGS="--pretext masked_signal --encoder_kind gcn --history_size 12 --num_preds 12 --batch_size 64"
        ;;
    england_covid)
        DATA_ARG="--data_source england_covid --england_covid_data_dir dataset/england_covid"
        DS_ARGS="--pretext masked_signal --encoder_kind gcn --history_size 8 --num_preds 1 --batch_size 32"
        ;;
    *)
        echo "[error] unknown DATASET='$DATASET'. Supported: dblp, jodie_reddit," \
             "jodie_wikipedia, reddit_body, reddit_title, ehrshot, metr_la, england_covid" >&2
        exit 1
        ;;
esac

srun --export=ALL python src/train/pretrain_encoder.py $DATA_ARG $DS_ARGS \
    --encoder_hidden 128 \
    --node_dim 128 \
    --decoder_hidden 64 \
    --mask_rate 0.25 \
    --num_workers ${SLURM_CPUS_PER_TASK:-8} \
    --device cuda \
    --epochs 30 \
    --lr 1e-3 \
    --out_dir "$OUT_DIR" \
    "$@"
