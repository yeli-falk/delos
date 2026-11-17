#!/usr/bin/env bash
#SBATCH --job-name=dblp-dyg2vec
#SBATCH --output=logs/dblp-dyg2vec-%j.out
#SBATCH --error=logs/dblp-dyg2vec-%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --time=10:00:00
#SBATCH --partition=gpu
#
# DyG2Vec baseline (Efficient Representation Learning for Dynamic Graphs,
# Alomrani et al., TMLR'24), two-stage so it is directly comparable to the
# DELOS pipeline:
#   1) pre-train its window-based temporal attention encoder (--encoder_kind
#      dyg2vec) with VICReg on two distorted views (edge dropout +
#      structural-feature masking) of each history window
#   2) freeze it and fit the SAME link-prediction probe DELOS uses
#      (the paper's own "SSL-init" protocol)
#
# Outputs:
#   outputs/dyg2vec_<TAG>/encoder.ckpt
#   outputs/dyg2vec_<TAG>_probe/analysis.json
#
# Run:    DATASET=dblp sbatch scripts/dblp/slurm_dyg2vec.sh
# Knobs:  DATASET=reddit_body  SEED=43  TAG=<name>  PD=0.3 (both distortions)
#         LR=1e-4  WINDOW=0 (0 = history_size)
#         Link-pred datasets: dblp, jodie_reddit, jodie_wikipedia, reddit_body,
#         reddit_title (or synthetic). Stage 1 + stage 2 share these flags.
#
# Dataset / encoder / window / early-stop config comes from
# scripts/dblp/_dataset_config.sh so it is identical to every other method
# (the encoder kind is the one exception, by design: it is the method).

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
PD=${PD:-0.3}
LR=${LR:-1e-4}
WINDOW=${WINDOW:-0}

# Link-prediction datasets only. Sets DATA_ARG / WINDOW_ARG / ENCODER_ARG /
# EARLY_STOP_ARG / SEED from the shared config, so both stages and every other
# method construct an identical node set.
source "$REPO_ROOT/scripts/dblp/_linkpred_data_arg.sh"
echo "[config] TAG=$TAG  PD=$PD  LR=$LR  WINDOW=$WINDOW"

# Stage 1: pre-train the DyG2Vec encoder (VICReg on distorted window views)
ENC_DIR="outputs/dyg2vec_${TAG}"
if [[ ! -f "$ENC_DIR/encoder.ckpt" ]]; then
    echo
    echo "================================================================="
    echo "  Stage 1/2: DyG2Vec VICReg pretrain -> $ENC_DIR"
    echo "================================================================="
    srun --export=ALL python -m src.baselines.dyg2vec.train_dyg2vec $DATA_ARG \
        $ENCODER_ARG --encoder_kind dyg2vec $WINDOW_ARG $EARLY_STOP_ARG \
        --p_edge "$PD" --p_struct "$PD" --lam 25 --mu 25 --nu 1 \
        --window "$WINDOW" --steps_per_epoch 20 --val_steps 10 \
        --device cuda --epochs 100 --lr "$LR" \
        --seed "$SEED" --out_dir "$ENC_DIR"
    enc_rc=$?
    if [[ $enc_rc -ne 0 || ! -f "$ENC_DIR/encoder.ckpt" ]]; then
        echo "[FATAL] Stage 1 failed (rc=$enc_rc)" >&2; exit 11
    fi
else
    echo "[stage 1] $ENC_DIR/encoder.ckpt exists, skipping DyG2Vec pretrain"
fi

# Stage 2: freeze the DyG2Vec encoder, fit the shared link-prediction probe
PROBE_DIR="outputs/dyg2vec_${TAG}_probe"
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
echo "[done] DyG2Vec baseline complete ($DATASET)."
echo "       encoder : $ENC_DIR/encoder.ckpt"
echo "       metrics : $PROBE_DIR/analysis.json  (link_pred.encoder.{all,new}; compare to DELOS)"
