#!/usr/bin/env bash
#SBATCH --job-name=cjepa-rescore
#SBATCH --output=logs/cjepa-rescore-%j.out
#SBATCH --error=logs/cjepa-rescore-%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#
# Rescore one or more completed C-JEPA runs with the extended link-metric
# set (adds AUC and exposes Hits@1). Writes test_metrics_v2.json next to
# the original test_metrics.json; dump_runs.py auto-prefers v2 if present.
#
# Submit (single run):
#     TAG=reddit_body_long_s42 sbatch scripts/dblp/slurm_rescore_metrics.sh
#
# Submit (whole long-training sweep, one job per seed):
#     for S in 42 43 44 45 46; do
#       TAG=reddit_body_long_s$S sbatch scripts/dblp/slurm_rescore_metrics.sh
#     done

set -uo pipefail
mkdir -p logs

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"

if [[ -d "$HOME/.conda/envs/cjepa" ]]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate cjepa
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo "[slurm] repo_root=$REPO_ROOT  node=$(hostname)"

TAG=${TAG:?TAG env var required, e.g. TAG=reddit_body_long_s42}
CJEPA_DIR="outputs/${TAG}_cjepa"
ENC_DIR="outputs/${TAG}_encoder"

if [[ ! -f "$CJEPA_DIR/best.ckpt" ]]; then
    echo "[FATAL] $CJEPA_DIR/best.ckpt missing" >&2; exit 10
fi
if [[ ! -f "$ENC_DIR/encoder.ckpt" ]]; then
    echo "[FATAL] $ENC_DIR/encoder.ckpt missing" >&2; exit 11
fi

echo "[rescore] TAG=$TAG"
srun --export=ALL python scripts/dblp/rescore_link_metrics.py \
    --cjepa_dir "$CJEPA_DIR" --encoder_dir "$ENC_DIR" \
    --device cuda

echo
echo "[done] $TAG rescored."
echo "  cat $CJEPA_DIR/test_metrics_v2.json"
