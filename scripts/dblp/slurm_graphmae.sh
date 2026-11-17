#!/usr/bin/env bash
#SBATCH --job-name=dblp-graphmae
#SBATCH --output=logs/dblp-graphmae-%j.out
#SBATCH --error=logs/dblp-graphmae-%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --time=10:00:00
#SBATCH --partition=gpu
#
# GraphMAE baseline (Self-Supervised Masked Graph Autoencoders, Hou et al.,
# KDD'22) -- the STATIC generative self-supervised control, two-stage so it is
# directly comparable to the DELOS pipeline:
#   1) train the GraphMAE encoder: each step draws ONE training snapshot,
#      masks a fraction of its nodes with a learnable token, re-masks their
#      latents and reconstructs their features with a GCN decoder under the
#      scaled cosine error. No history is aggregated and no two snapshots are
#      compared: the objective is static.
#   2) freeze it and fit the SAME link-prediction probe DELOS uses, which
#      reads the encoder at the last history snapshot ("static SSL, last
#      frame").
#
# Outputs:
#   outputs/graphmae_<TAG>/encoder.ckpt
#   outputs/graphmae_<TAG>_probe/analysis.json
#
# Run:    DATASET=dblp sbatch scripts/dblp/slurm_graphmae.sh
# Knobs:  DATASET=reddit_body  SEED=43  TAG=<name>  MASK_RATE=0.5
#         REPLACE_RATE=0.05  GAMMA=2  TARGET=all|semantic  (reference defaults)
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
MASK_RATE=${MASK_RATE:-0.5}
REPLACE_RATE=${REPLACE_RATE:-0.05}
GAMMA=${GAMMA:-2}
TARGET=${TARGET:-all}

# Link-prediction datasets only. Sets DATA_ARG / WINDOW_ARG / ENCODER_ARG /
# EARLY_STOP_ARG / SEED from the shared config, so both stages and every other
# method construct an identical node set.
source "$REPO_ROOT/scripts/dblp/_linkpred_data_arg.sh"
echo "[config] TAG=$TAG  MASK_RATE=$MASK_RATE  REPLACE_RATE=$REPLACE_RATE  GAMMA=$GAMMA  TARGET=$TARGET"

# Stage 1: train the GraphMAE encoder (static masked-feature AE, one snapshot/step)
ENC_DIR="outputs/graphmae_${TAG}"
if [[ ! -f "$ENC_DIR/encoder.ckpt" ]]; then
    echo
    echo "================================================================="
    echo "  Stage 1/2: GraphMAE masked-feature pretrain -> $ENC_DIR"
    echo "================================================================="
    srun --export=ALL python -m src.baselines.graphmae.train_graphmae $DATA_ARG \
        $ENCODER_ARG $WINDOW_ARG $EARLY_STOP_ARG \
        --mask_rate "$MASK_RATE" --replace_rate "$REPLACE_RATE" \
        --sce_gamma "$GAMMA" --target "$TARGET" \
        --steps_per_epoch 20 --val_steps 10 \
        --device cuda --epochs 100 --lr 1e-3 \
        --seed "$SEED" --out_dir "$ENC_DIR"
    enc_rc=$?
    if [[ $enc_rc -ne 0 || ! -f "$ENC_DIR/encoder.ckpt" ]]; then
        echo "[FATAL] Stage 1 failed (rc=$enc_rc)" >&2; exit 11
    fi
else
    echo "[stage 1] $ENC_DIR/encoder.ckpt exists, skipping GraphMAE pretrain"
fi

# Stage 2: freeze the GraphMAE encoder, fit the shared link-prediction probe
PROBE_DIR="outputs/graphmae_${TAG}_probe"
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
echo "[done] GraphMAE baseline complete ($DATASET)."
echo "       encoder : $ENC_DIR/encoder.ckpt"
echo "       metrics : $PROBE_DIR/analysis.json  (link_pred.encoder.{all,new}; compare to DELOS)"
