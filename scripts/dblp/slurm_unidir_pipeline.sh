#!/usr/bin/env bash
#SBATCH --job-name=dblp-unidir
#SBATCH --output=logs/dblp-unidir-%j.out
#SBATCH --error=logs/dblp-unidir-%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --time=10:00:00
#SBATCH --partition=gpu
#
# Three-stage unidirectional ablation pipeline:
#   1) pretrain a temporally-causal T-GCN encoder (bidirectional=False)
#   2) train DELOS with that encoder, optionally with the edge-masking
#      term enabled in the JEPA loss
#   3) re-run pretrained-encoder + probe ablation on the same encoder
#
# Outputs:
#   outputs/dblp_encoder_unidir/encoder.ckpt
#   outputs/dblp_cjepa_unidir/best.ckpt + test_metrics.json
#   outputs/dblp_ablation_probe_unidir/test_metrics.json
#
# Run:    sbatch scripts/dblp/slurm_unidir_pipeline.sh
# With edge masking in JEPA:
#         EDGE=0.15 sbatch scripts/dblp/slurm_unidir_pipeline.sh
# Different seed:
#         SEED=43 sbatch scripts/dblp/slurm_unidir_pipeline.sh

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

SEED=${SEED:-42}
EDGE=${EDGE:-0.0}            # JEPA edge-mask rate; 0 = pure latent JEPA
TAG=${TAG:-unidir}
if [[ "$EDGE" != "0.0" && "$EDGE" != "0" ]]; then
    TAG="${TAG}_edge${EDGE}"
fi
echo "[config] SEED=$SEED  EDGE_MASK_RATE=$EDGE  TAG=$TAG"

DATA_ARG="--synthetic --synthetic_nodes 500 --synthetic_years 30"
if [[ -f dataset/dblp/edges.csv ]]; then
    DATA_ARG="--data_path dataset/dblp/edges.csv"
fi

# -----------------------------------------------------------------------------
# Stage 1: pretrain a unidirectional T-GCN encoder
# -----------------------------------------------------------------------------
ENC_DIR="outputs/dblp_encoder_${TAG}"
if [[ ! -f "$ENC_DIR/encoder.ckpt" ]]; then
    echo
    echo "================================================================="
    echo "  Stage 1/3: pretrain unidirectional encoder -> $ENC_DIR"
    echo "================================================================="
    srun --export=ALL python src/train/pretrain_encoder.py $DATA_ARG \
        --encoder_kind tgcn --encoder_hidden 128 --node_dim 128 \
        --bidirectional false \
        --decoder_hidden 64 --mask_rate 0.25 \
        --batch_size 4 --num_workers ${SLURM_CPUS_PER_TASK:-8} \
        --device cuda --epochs 30 --lr 1e-3 \
        --seed "$SEED" --out_dir "$ENC_DIR"
else
    echo "[stage 1] encoder.ckpt already exists at $ENC_DIR, skipping pretrain"
fi

# -----------------------------------------------------------------------------
# Stage 2: train DELOS with the unidir encoder (optionally + edge masking)
# -----------------------------------------------------------------------------
DELOS_DIR="outputs/dblp_cjepa_${TAG}"
echo
echo "================================================================="
echo "  Stage 2/3: DELOS train -> $DELOS_DIR"
echo "================================================================="
srun --export=ALL python src/train/train_delos.py --no_attention_mask $DATA_ARG \
    --encoder_ckpt "$ENC_DIR/encoder.ckpt" \
    --history_size 3 --num_preds 1 \
    --encoder_kind tgcn --encoder_hidden 128 --node_dim 128 \
    --num_masked_nodes 30 --depth 4 --heads 4 --mlp_dim 512 \
    --edge_mask_rate "$EDGE" --edge_weight 1.0 --edge_neg_per_pos 1 \
    --batch_size 8 --num_workers ${SLURM_CPUS_PER_TASK:-8} \
    --device cuda --epochs 50 --lr 5e-4 --probe_epochs 3 \
    --seed "$SEED" --out_dir "$DELOS_DIR"

# -----------------------------------------------------------------------------
# Stage 3: pretrained-encoder + probe ablation with the same unidir encoder
# -----------------------------------------------------------------------------
PROBE_DIR="outputs/dblp_ablation_probe_${TAG}"
echo
echo "================================================================="
echo "  Stage 3/3: pretrained encoder + probe ablation -> $PROBE_DIR"
echo "================================================================="
srun --export=ALL python src/train/ablations_dblp.py --mode pretrained_probe \
    $DATA_ARG \
    --encoder_ckpt "$ENC_DIR/encoder.ckpt" \
    --history_size 3 --num_preds 1 \
    --encoder_kind tgcn --encoder_hidden 128 --node_dim 128 \
    --probe_hidden 64 \
    --batch_size 4 --num_workers ${SLURM_CPUS_PER_TASK:-8} \
    --device cuda --epochs 30 --lr 1e-3 \
    --seed "$SEED" --out_dir "$PROBE_DIR"

echo
echo "[done] all three stages finished. Compare:"
echo "  cat $DELOS_DIR/test_metrics.json"
echo "  cat $PROBE_DIR/test_metrics.json"
