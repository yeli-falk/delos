#!/usr/bin/env bash
#SBATCH --job-name=dblp-dysubc
#SBATCH --output=logs/dblp-dysubc-%j.out
#SBATCH --error=logs/dblp-dysubc-%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --time=10:00:00
#SBATCH --partition=gpu
#
# DySubC baseline (Self-Supervised Dynamic Graph Representation Learning via
# Temporal Subgraph Contrast, Chen et al., ACM TKDD'23), two-stage so it is
# directly comparable to the DELOS pipeline:
#   1) train the DySubC encoder: temporal subgraph sampling (BFS ranked by
#      recency + degree), a time-weighted and an unweighted subgraph encoder,
#      margin losses against a shuffled and an unweighted subgraph summary
#   2) freeze it and fit the SAME link-prediction probe DELOS uses
#
# Outputs:
#   outputs/dysubc_<TAG>/encoder.ckpt
#   outputs/dysubc_<TAG>_probe/analysis.json
#
# Run:    DATASET=dblp sbatch scripts/dblp/slurm_dysubc.sh
# Knobs:  DATASET=reddit_body  SEED=43  TAG=<name>  K=20  ALPHA=5  BETA=1
#         MARGIN=0.75  LAM=0.5  LOOKBACK=0 (0 = history_size)
#         Link-pred datasets: dblp, jodie_reddit, jodie_wikipedia, reddit_body,
#         reddit_title (or synthetic). Stage 1 + stage 2 share these flags.
#
# Dataset / encoder / window / early-stop config comes from
# scripts/dblp/_dataset_config.sh so it is identical to every other method.

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

DATASET=${DATASET:-dblp}
# Seed is part of TAG so a multi-seed sweep does not reuse seed 42's checkpoints.
TAG=${TAG:-${DATASET}_s${SEED:-42}}
K=${K:-20}
# alpha weighs degree against recency (on a [0,10] scale) in the sampling
# score; the released code uses 5 on MOOC, 1 on Wikipedia, 10 on bitcoin-alpha.
ALPHA=${ALPHA:-5}
BETA=${BETA:-1}
MARGIN=${MARGIN:-0.75}
LAM=${LAM:-0.5}
LOOKBACK=${LOOKBACK:-0}

# Link-prediction datasets only. Sets DATA_ARG / WINDOW_ARG / ENCODER_ARG /
# EARLY_STOP_ARG / SEED from the shared config, so both stages and every other
# method construct an identical node set.
source "$REPO_ROOT/scripts/dblp/_linkpred_data_arg.sh"
echo "[config] TAG=$TAG  K=$K  ALPHA=$ALPHA  BETA=$BETA  MARGIN=$MARGIN  LAM=$LAM  LOOKBACK=$LOOKBACK"

# Stage 1: train the DySubC encoder (temporal subgraph contrast)
ENC_DIR="outputs/dysubc_${TAG}"
if [[ ! -f "$ENC_DIR/encoder.ckpt" ]]; then
    echo
    echo "================================================================="
    echo "  Stage 1/2: DySubC contrastive pretrain -> $ENC_DIR"
    echo "================================================================="
    srun --export=ALL python -m src.baselines.dysubc.train_dysubc $DATA_ARG \
        $ENCODER_ARG $WINDOW_ARG $EARLY_STOP_ARG \
        --subgraph_size "$K" --alpha "$ALPHA" --beta "$BETA" \
        --margin "$MARGIN" --lam "$LAM" --lookback "$LOOKBACK" \
        --node_batch_size 256 --steps_per_epoch 20 --val_steps 10 \
        --device cuda --epochs 100 --lr 1e-3 \
        --seed "$SEED" --out_dir "$ENC_DIR"
    enc_rc=$?
    if [[ $enc_rc -ne 0 || ! -f "$ENC_DIR/encoder.ckpt" ]]; then
        echo "[FATAL] Stage 1 failed (rc=$enc_rc)" >&2; exit 11
    fi
else
    echo "[stage 1] $ENC_DIR/encoder.ckpt exists, skipping DySubC pretrain"
fi

# Stage 2: freeze the DySubC encoder, fit the shared link-prediction probe
PROBE_DIR="outputs/dysubc_${TAG}_probe"
echo
echo "================================================================="
echo "  Stage 2/2: frozen-encoder + link-prediction probe -> $PROBE_DIR"
echo "================================================================="
srun --export=ALL python -m src.train.analyze_jepa --skip_diagnostic \
    --link_sources encoder $DATA_ARG $WINDOW_ARG \
    --encoder_ckpt "$ENC_DIR/encoder.ckpt" \
    --num_workers ${SLURM_CPUS_PER_TASK:-8} \
    --device cuda --seed "$SEED" --out_dir "$PROBE_DIR"

echo
echo "[done] DySubC baseline complete ($DATASET)."
echo "       encoder : $ENC_DIR/encoder.ckpt"
echo "       metrics : $PROBE_DIR/analysis.json  (link_pred.encoder.{all,new}; compare to DELOS)"
