#!/usr/bin/env bash
#SBATCH --job-name=dblp-ddgcl
#SBATCH --output=logs/dblp-ddgcl-%j.out
#SBATCH --error=logs/dblp-ddgcl-%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --time=10:00:00
#SBATCH --partition=gpu
#
# DDGCL baseline (Debiased Dynamic Graph Contrastive Learning, Tian et al.,
# CIKM'21), two-stage so it is directly comparable to the DELOS pipeline:
#   1) train the DDGCL encoder with time-aware debiased contrastive learning
#   2) freeze it and fit the SAME link-prediction probe DELOS uses
#
# Outputs:
#   outputs/ddgcl_<TAG>/encoder.ckpt
#   outputs/ddgcl_<TAG>_probe/analysis.json
#
# Run:    DATASET=dblp sbatch scripts/dblp/slurm_ddgcl.sh
# Knobs:  DATASET=reddit_body  SEED=43  TAG=<name>  MAX_GAP=4  TAU=0.01
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
MAX_GAP=${MAX_GAP:-4}
# DDGCL's `--tau` is the debias class prior (the reference's tau_plus), not a
# softmax temperature. The reference default is 0.01, and the trainer has no
# `--tau_plus` flag.
TAU=${TAU:-0.01}
BETA=${BETA:-1.5}

# Link-prediction datasets only. Sets DATA_ARG / WINDOW_ARG / ENCODER_ARG /
# EARLY_STOP_ARG / SEED from the shared config, so both stages and every other
# method construct an identical node set.
source "$REPO_ROOT/scripts/dblp/_linkpred_data_arg.sh"
echo "[config] TAG=$TAG  MAX_GAP=$MAX_GAP  TAU=$TAU  BETA=$BETA"

# Stage 1: train the DDGCL encoder (time-aware debiased contrastive learning)
ENC_DIR="outputs/ddgcl_${TAG}"
if [[ ! -f "$ENC_DIR/encoder.ckpt" ]]; then
    echo
    echo "================================================================="
    echo "  Stage 1/2: DDGCL contrastive pretrain -> $ENC_DIR"
    echo "================================================================="
    srun --export=ALL python -m src.baselines.ddgcl.train_ddgcl $DATA_ARG \
        $ENCODER_ARG $WINDOW_ARG $EARLY_STOP_ARG \
        --max_gap "$MAX_GAP" --time_dim 32 --tau "$TAU" --beta "$BETA" \
        --node_batch_size 256 --steps_per_epoch 20 --val_steps 10 \
        --device cuda --epochs 100 --lr 4e-3 \
        --seed "$SEED" --out_dir "$ENC_DIR"
    enc_rc=$?
    if [[ $enc_rc -ne 0 || ! -f "$ENC_DIR/encoder.ckpt" ]]; then
        echo "[FATAL] Stage 1 failed (rc=$enc_rc)" >&2; exit 11
    fi
else
    echo "[stage 1] $ENC_DIR/encoder.ckpt exists, skipping DDGCL pretrain"
fi

# Stage 2: freeze the DDGCL encoder, fit the shared link-prediction probe
PROBE_DIR="outputs/ddgcl_${TAG}_probe"
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
echo "[done] DDGCL baseline complete ($DATASET)."
echo "       encoder : $ENC_DIR/encoder.ckpt"
echo "       metrics : $PROBE_DIR/analysis.json  (link_pred.encoder.{all,new}; compare to DELOS)"
