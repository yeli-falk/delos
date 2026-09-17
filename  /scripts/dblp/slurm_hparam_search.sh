#!/usr/bin/env bash
#SBATCH --job-name=dyg-hparam
#SBATCH --output=logs/dyg-hparam-%A_%a.out
#SBATCH --error=logs/dyg-hparam-%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --time=08:00:00
#SBATCH --partition=gpu
#SBATCH --array=0-35
#
# Hyperparameter search for DyG-JEPA (src/train/train_dyg_jepa.py).
#
# A SLURM array job that sweeps a Cartesian grid over the four most impactful
# DyG-JEPA knobs. The array task id is decoded into one grid cell, each cell
# lands in its own outputs/ directory, and a companion aggregator ranks them.
#
#   LRS      : optimiser learning rate          (--lr)
#   MASKS    : number of masked target nodes    (--num_masked_nodes)
#   DEPTHS   : predictor transformer depth       (--depth)
#   DROPOUTS : predictor dropout                  (--dropout)
#
# Grid size = |LRS| * |MASKS| * |DEPTHS| * |DROPOUTS| = 3*3*2*2 = 36.
# IMPORTANT: if you edit the value arrays below, recompute the product and
# update `#SBATCH --array=0-<product-1>` above (or override with
# `sbatch --array=0-N`).
#
# Submit all:        sbatch scripts/dblp/slurm_hparam_search.sh
# Submit one cell:   sbatch --array=7 scripts/dblp/slurm_hparam_search.sh
# Throttle to 4 at a time (spare the queue):
#                    sbatch --array=0-35%4 scripts/dblp/slurm_hparam_search.sh
# Watch:             squeue -u $USER
# After it finishes, rank the runs:
#                    python scripts/dblp/aggregate_hparam_search.py outputs/dyg_hparam
#
# Override knobs from the command line, e.g. a 25-epoch run on real edges:
#   EPOCHS=25 sbatch scripts/dblp/slurm_hparam_search.sh

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

# -----------------------------------------------------------------------------
# Grid definition -- edit these, then update --array=0-N above accordingly.
# -----------------------------------------------------------------------------
LRS=(5e-4 1e-3 2e-3)
MASKS=(20 30 50)
DEPTHS=(2 4)
DROPOUTS=(0.1 0.2)

# Knobs held fixed across the sweep (override via environment if needed).
SEED=${SEED:-42}
EPOCHS=${EPOCHS:-50}
PROBE_EPOCHS=${PROBE_EPOCHS:-3}
HEADS=${HEADS:-4}
MLP_DIM=${MLP_DIM:-512}
ENCODER_HIDDEN=${ENCODER_HIDDEN:-128}
NODE_DIM=${NODE_DIM:-128}
BATCH_SIZE=${BATCH_SIZE:-8}
SWEEP_ROOT=${SWEEP_ROOT:-outputs/dyg_hparam}

# -----------------------------------------------------------------------------
# Decode the flat array task id into a (lr, mask, depth, dropout) grid cell.
# Mixed-radix: fastest-varying dimension is DROPOUTS, slowest is LRS.
# -----------------------------------------------------------------------------
N_DROP=${#DROPOUTS[@]}
N_DEPTH=${#DEPTHS[@]}
N_MASK=${#MASKS[@]}
N_LR=${#LRS[@]}
TOTAL=$((N_LR * N_MASK * N_DEPTH * N_DROP))

IDX=${SLURM_ARRAY_TASK_ID:-0}
if (( IDX >= TOTAL )); then
    echo "[error] task id $IDX >= grid size $TOTAL -- fix the --array range" >&2
    exit 2
fi

DROP_I=$(( IDX % N_DROP ));                     REST=$(( IDX / N_DROP ))
DEPTH_I=$(( REST % N_DEPTH ));                  REST=$(( REST / N_DEPTH ))
MASK_I=$(( REST % N_MASK ));                    REST=$(( REST / N_MASK ))
LR_I=$(( REST % N_LR ))

LR=${LRS[$LR_I]}
MASK=${MASKS[$MASK_I]}
DEPTH=${DEPTHS[$DEPTH_I]}
DROPOUT=${DROPOUTS[$DROP_I]}

TAG="lr${LR}_m${MASK}_d${DEPTH}_do${DROPOUT}"
OUT_DIR="${SWEEP_ROOT}/${TAG}"

echo "[array] task_id=$IDX/$((TOTAL-1))  ->  $TAG"
echo "[config] lr=$LR  num_masked_nodes=$MASK  depth=$DEPTH  dropout=$DROPOUT"
echo "[paths] out_dir=$OUT_DIR  seed=$SEED  epochs=$EPOCHS"
echo "[slurm] repo_root=$REPO_ROOT  node=$(hostname)"

# -----------------------------------------------------------------------------
# Data source: real edges if present, else the synthetic DBLP generator.
# -----------------------------------------------------------------------------
DATA_ARG="--synthetic --synthetic_nodes 500 --synthetic_years 30"
if [[ -f dataset/dblp/edges.csv ]]; then
    DATA_ARG="--data_path dataset/dblp/edges.csv"
    echo "[data] using dataset/dblp/edges.csv"
fi

# Reuse a pretrained encoder if one exists (otherwise the encoder is random/
# trained from scratch by the entry point's defaults).
ENC_ARG=""
if [[ -f outputs/dblp_encoder/encoder.ckpt ]]; then
    ENC_ARG="--encoder_ckpt outputs/dblp_encoder/encoder.ckpt"
    echo "[encoder] loading outputs/dblp_encoder/encoder.ckpt"
fi

# -----------------------------------------------------------------------------
# Train this grid cell. --no_attention_mask reproduces the historical DBLP
# attention-mask behaviour (see scripts/dblp/slurm_train.sh and the refactor
# note in memory).
# -----------------------------------------------------------------------------
srun --export=ALL python src/train/train_dyg_jepa.py --no_attention_mask $DATA_ARG $ENC_ARG \
    --history_size 3 --num_preds 1 \
    --encoder_kind tgcn --encoder_hidden "$ENCODER_HIDDEN" --node_dim "$NODE_DIM" \
    --num_masked_nodes "$MASK" --depth "$DEPTH" --heads "$HEADS" --mlp_dim "$MLP_DIM" \
    --dropout "$DROPOUT" \
    --batch_size "$BATCH_SIZE" --num_workers ${SLURM_CPUS_PER_TASK:-8} \
    --device cuda --epochs "$EPOCHS" --lr "$LR" --probe_epochs "$PROBE_EPOCHS" \
    --seed "$SEED" --out_dir "$OUT_DIR" \
    "$@"

echo "[done] task_id=$IDX  $TAG"
echo "  metrics:    cat $OUT_DIR/test_metrics.json"
echo "  checkpoint: $OUT_DIR/best.ckpt   (carries val_loss for model selection)"
