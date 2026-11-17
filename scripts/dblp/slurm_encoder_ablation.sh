#!/usr/bin/env bash
#SBATCH --job-name=enc-ablation
#SBATCH --output=logs/enc-ablation-%A_%a.out
#SBATCH --error=logs/enc-ablation-%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --time=06:00:00
#SBATCH --partition=gpu
#SBATCH --array=0-14
#
# Encoder ablation as a SLURM array job: sweeps ENCODERS x SEEDS, holding the
# dataset, pretext, predictor, probe and splits fixed. Each task (a) pretrains a
# GraphMAE encoder of one kind, then (b) trains DELOS on top of it and fits
# the downstream probe -- writing test_metrics.json the aggregator picks up.
#
# Default grid: 3 encoders x 5 seeds = 15 tasks  ->  --array=0-14.
# Pick the dataset with DATASET (default dblp):
#   sbatch --export=ALL,DATASET=metr_la scripts/dblp/slurm_encoder_ablation.sh
# If you change the ENCODERS/SEEDS arrays below, update --array to
# (n_encoders * n_seeds - 1).

set -uo pipefail

DATASET="${DATASET:-dblp}"

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"

if [[ -d "$HOME/.conda/envs/cjepa" ]]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate cjepa
fi

# -----------------------------------------------------------------------------
# Sweep grid -- edit and update --array=0-N (N = n_enc * n_seed - 1) to match.
# -----------------------------------------------------------------------------
ENCODERS=(linear gcn tgcn)   # 'linear' = no-message-passing floor
SEEDS=(42 43 44 45 46)

N_SEEDS=${#SEEDS[@]}
ENC_IDX=$((SLURM_ARRAY_TASK_ID / N_SEEDS))
SEED_IDX=$((SLURM_ARRAY_TASK_ID % N_SEEDS))
ENC=${ENCODERS[$ENC_IDX]}
SEED=${SEEDS[$SEED_IDX]}

ROOT="outputs/encoder_ablation/${DATASET}"
PRE_DIR="${ROOT}/${ENC}_s${SEED}/pretrain"
JEPA_DIR="${ROOT}/${ENC}_s${SEED}/jepa"
mkdir -p logs "$PRE_DIR" "$JEPA_DIR"

echo "[array] task=$SLURM_ARRAY_TASK_ID dataset=$DATASET encoder=$ENC seed=$SEED node=$(hostname)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

# -----------------------------------------------------------------------------
# Per-dataset config: data source + flags, pretext (fixed-topology graphs only
# support masked_signal), downstream task/metric, and history/horizon/batch.
# The encoder is the swept variable, so it is NOT set here.
# -----------------------------------------------------------------------------
case "$DATASET" in
    dblp)
        if [[ -f dataset/dblp/edges.csv ]]; then
            DATA_ARG="--data_source dblp --data_path dataset/dblp/edges.csv"
        else
            DATA_ARG="--data_source dblp --synthetic --synthetic_nodes 500 --synthetic_years 30"
        fi
        PRETEXT="edge_mask"; DOWNSTREAM="link_pred"
        SHAPE="--history_size 3 --num_preds 1"; PRE_BS=4; JEPA_BS=8
        ;;
    jodie_reddit|jodie_wikipedia|reddit_body|reddit_title)
        DATA_ARG="--data_source $DATASET --with_edge_features"
        PRETEXT="feature_mask"; DOWNSTREAM="link_pred"
        SHAPE="--history_size 3 --num_preds 1"; PRE_BS=4; JEPA_BS=8
        ;;
    ehrshot)
        DATA_ARG="--data_source ehrshot --ehrshot_data_path dataset/ehrshot/ehrshot_cohort.pkl"
        PRETEXT="feature_mask"; DOWNSTREAM="link_pred"
        SHAPE="--history_size 3 --num_preds 1"; PRE_BS=4; JEPA_BS=8
        ;;
    metr_la)
        DATA_ARG="--data_source metr_la --metr_la_data_dir dataset/metr_la"
        PRETEXT="masked_signal"; DOWNSTREAM="regression"
        SHAPE="--history_size 12 --num_preds 12"; PRE_BS=64; JEPA_BS=64
        ;;
    england_covid)
        DATA_ARG="--data_source england_covid --england_covid_data_dir dataset/england_covid"
        PRETEXT="masked_signal"; DOWNSTREAM="regression"
        SHAPE="--history_size 8 --num_preds 1"; PRE_BS=32; JEPA_BS=32
        ;;
    *)
        echo "[error] unknown DATASET='$DATASET'" >&2; exit 1 ;;
esac

# -----------------------------------------------------------------------------
# (a) Pretrain the encoder of this kind.
# -----------------------------------------------------------------------------
echo "[pretrain] encoder=$ENC pretext=$PRETEXT -> $PRE_DIR"
srun --export=ALL python src/train/pretrain_encoder.py $DATA_ARG \
    --encoder_kind "$ENC" --pretext "$PRETEXT" $SHAPE \
    --encoder_hidden 128 --node_dim 128 --decoder_hidden 64 --mask_rate 0.25 \
    --batch_size "$PRE_BS" --num_workers ${SLURM_CPUS_PER_TASK:-8} \
    --device cuda --epochs 500 --lr 1e-3 \
    --seed "$SEED" --out_dir "$PRE_DIR"

# pretrain_encoder.py names the ckpt encoder_<src>_<enc>_<pretext>.ckpt.
ENC_CKPT=$(ls -t "$PRE_DIR"/encoder_*_"${ENC}"_"${PRETEXT}".ckpt 2>/dev/null | head -1)
if [[ -z "${ENC_CKPT:-}" ]]; then
    echo "[error] no pretrained ckpt found in $PRE_DIR" >&2; exit 1
fi
echo "[pretrain] ckpt=$ENC_CKPT"

# -----------------------------------------------------------------------------
# (b) Train DELOS on top + fit the downstream probe (writes test_metrics.json).
#     Everything except the encoder is held constant across the grid.
# -----------------------------------------------------------------------------
echo "[jepa] encoder=$ENC task=$DOWNSTREAM -> $JEPA_DIR"
srun --export=ALL python src/train/train_delos.py --no_attention_mask $DATA_ARG \
    --encoder_ckpt "$ENC_CKPT" --encoder_kind "$ENC" \
    --downstream_task "$DOWNSTREAM" $SHAPE \
    --encoder_hidden 128 --node_dim 128 \
    --num_masked_nodes 30 --depth 4 --heads 4 --mlp_dim 512 \
    --batch_size "$JEPA_BS" --num_workers ${SLURM_CPUS_PER_TASK:-8} \
    --device cuda --epochs 500 --lr 5e-4 --probe_epochs 50 \
    --seed "$SEED" --out_dir "$JEPA_DIR"

echo "[done] task=$SLURM_ARRAY_TASK_ID encoder=$ENC seed=$SEED -> $JEPA_DIR/test_metrics.json"
