#!/usr/bin/env bash
#SBATCH --job-name=metrla-pretrain-tgcn
#SBATCH --output=logs/metrla-pretrain-tgcn-%j.out
#SBATCH --error=logs/metrla-pretrain-tgcn-%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --time=06:00:00
#SBATCH --partition=gpu
#
# T-GCN encoder pretraining (Zhao et al., 2019 style). Spatial GCN + bidirectional
# GRU across time. Saves to a dedicated dir so it doesn't clobber the simple GCN
# encoder. Pass the resulting outputs/metr_la_encoder_tgcn/encoder.ckpt to
# slurm_train.sh via --encoder_ckpt; the JEPA trainer auto-detects the kind
# from the checkpoint metadata.

set -euo pipefail
mkdir -p logs outputs/metr_la_encoder_tgcn

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
    --encoder_kind tgcn \
    --encoder_hidden 128 \
    --node_dim 128 \
    --decoder_hidden 64 \
    --mask_rate 0.25 \
    --batch_size 64 \
    --num_workers ${SLURM_CPUS_PER_TASK:-8} \
    --device cuda \
    --epochs 30 \
    --lr 1e-3 \
    --out_dir outputs/metr_la_encoder_tgcn \
    "$@"
