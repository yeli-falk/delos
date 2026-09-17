#!/usr/bin/env bash
#SBATCH --job-name=metrla-pretrain
#SBATCH --output=logs/metrla-pretrain-%j.out
#SBATCH --error=logs/metrla-pretrain-%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --time=06:00:00
#SBATCH --partition=gpu
#
# GraphMAE-style encoder pretraining. Produces outputs/metr_la_encoder/encoder.ckpt
# which you then pass to slurm_train.sh via --encoder_ckpt.

set -euo pipefail
mkdir -p logs outputs/metr_la_encoder

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"
echo "[slurm] repo_root=$REPO_ROOT"

if [[ -d "$HOME/.conda/envs/cjepa" ]]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate cjepa
fi

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

srun --export=ALL python src/train/pretrain_encoder.py --data_source metr_la --pretext masked_signal \
    --metr_la_data_dir dataset/metr_la \
    --history_size 12 \
    --num_preds 12 \
    --encoder_kind gcn \
    --encoder_hidden 128 \
    --node_dim 128 \
    --decoder_hidden 64 \
    --mask_rate 0.25 \
    --batch_size 64 \
    --num_workers ${SLURM_CPUS_PER_TASK:-8} \
    --device cuda \
    --epochs 15 \
    --lr 1e-3 \
    --out_dir outputs/metr_la_encoder \
    "$@"
