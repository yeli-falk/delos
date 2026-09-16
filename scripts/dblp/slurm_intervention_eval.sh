#!/usr/bin/env bash
#SBATCH --job-name=dblp-interv
#SBATCH --output=logs/dblp-interv-%j.out
#SBATCH --error=logs/dblp-interv-%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --time=03:00:00
#SBATCH --partition=gpu

set -uo pipefail
mkdir -p logs outputs

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"
echo "[slurm] repo_root=$REPO_ROOT  node=$(hostname)"

if [[ -d "$HOME/.conda/envs/cjepa" ]]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate cjepa
fi

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

DATA_ARG=""
if [[ -f dataset/dblp/edges.csv ]]; then
    DATA_ARG="--data_path dataset/dblp/edges.csv"
else
    DATA_ARG="--synthetic --synthetic_nodes 500 --synthetic_years 30"
fi

RESUME=${RESUME:-outputs/dblp_cjepa/best.ckpt}
ENC=${ENC:-outputs/dblp_encoder/encoder.ckpt}

srun --export=ALL python src/train/intervention_eval_dblp.py $DATA_ARG \
    --resume "$RESUME" \
    --encoder_ckpt "$ENC" \
    --strategies none random hub peripheral \
    --num_masked_nodes 30 100 300 \
    --batch_size 4 \
    --num_workers ${SLURM_CPUS_PER_TASK:-8} \
    --device cuda \
    --probe_epochs 3 \
    --out_dir outputs/dblp_intervention_eval \
    "$@"
