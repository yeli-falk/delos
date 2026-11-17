#!/usr/bin/env bash
#SBATCH --job-name=dblp-multiseed
#SBATCH --output=logs/dblp-multiseed-%A_%a.out
#SBATCH --error=logs/dblp-multiseed-%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --time=06:00:00
#SBATCH --partition=gpu
#SBATCH --array=0-9
#
# SLURM array job: 5 seeds x 2 modes = 10 independent tasks.
# Each task trains either DELOS or fits the pretrained-encoder + probe
# baseline with one seed. Each lands in its own outputs/ directory so the
# aggregator can pick them all up afterwards.
#
# Submit:  sbatch scripts/dblp/slurm_multiseed.sh
# Watch:   squeue -u $USER -t R,PD --array
# Cancel one task: scancel <jobid>_<task_id>
# Cancel all: scancel <jobid>
#
# To use different seeds or modes, override the arrays at the top of this
# file and update the --array range accordingly (range = n_seeds * n_modes - 1).

set -uo pipefail
mkdir -p logs outputs

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"

if [[ -d "$HOME/.conda/envs/cjepa" ]]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate cjepa
fi

# Sweep configuration -- edit and update --array=0-N at the top if you change.
SEEDS=(42 43 44 45 46)
MODES=(cjepa probe)
ENCODER_CKPT=outputs/dblp_encoder/encoder.ckpt

# Decode the array task ID into (seed, mode).
N_MODES=${#MODES[@]}
SEED_IDX=$((SLURM_ARRAY_TASK_ID / N_MODES))
MODE_IDX=$((SLURM_ARRAY_TASK_ID % N_MODES))
SEED=${SEEDS[$SEED_IDX]}
MODE=${MODES[$MODE_IDX]}

echo "[array] task_id=$SLURM_ARRAY_TASK_ID  seed=$SEED  mode=$MODE"
echo "[slurm] repo_root=$REPO_ROOT  node=$(hostname)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

if [[ ! -f "$ENCODER_CKPT" ]]; then
    echo "[error] encoder not found at $ENCODER_CKPT" >&2
    echo "[error] run scripts/dblp/slurm_pretrain_encoder.sh first" >&2
    exit 1
fi

DATA_ARG="--synthetic --synthetic_nodes 500 --synthetic_years 30"
if [[ -f dataset/dblp/edges.csv ]]; then
    DATA_ARG="--data_path dataset/dblp/edges.csv"
fi

# Dispatch
case $MODE in
    cjepa)
        OUT_DIR="outputs/dblp_cjepa_s${SEED}"
        srun --export=ALL python src/train/train_delos.py --no_attention_mask $DATA_ARG \
            --encoder_ckpt "$ENCODER_CKPT" \
            --history_size 3 --num_preds 1 \
            --encoder_kind tgcn --encoder_hidden 128 --node_dim 128 \
            --num_masked_nodes 30 --depth 4 --heads 4 --mlp_dim 512 \
            --batch_size 8 --num_workers ${SLURM_CPUS_PER_TASK:-8} \
            --device cuda --epochs 5 --lr 5e-4 --probe_epochs 3 \
            --seed "$SEED" --out_dir "$OUT_DIR"
        ;;
    probe)
        OUT_DIR="outputs/dblp_ablation_probe_s${SEED}"
        srun --export=ALL python src/train/ablations_dblp.py --mode pretrained_probe \
            $DATA_ARG \
            --encoder_ckpt "$ENCODER_CKPT" \
            --history_size 3 --num_preds 1 \
            --encoder_kind tgcn --encoder_hidden 128 --node_dim 128 \
            --probe_hidden 64 \
            --batch_size 4 --num_workers ${SLURM_CPUS_PER_TASK:-8} \
            --device cuda --epochs 30 --lr 1e-3 \
            --seed "$SEED" --out_dir "$OUT_DIR"
        ;;
    *)
        echo "[error] unknown mode '$MODE'" >&2
        exit 1
        ;;
esac

echo "[done] task_id=$SLURM_ARRAY_TASK_ID seed=$SEED mode=$MODE  -> $OUT_DIR"
