#!/usr/bin/env bash
#SBATCH --job-name=delos-metrla
#SBATCH --output=logs/delos-metrla-%j.out
#SBATCH --error=logs/delos-metrla-%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
# --partition=gpu                  # uncomment + set to your cluster's partition
# --qos=gpu                        # uncomment if QoS is required
# --account=<your-account>         # uncomment if accounting is required
#
# Submit with:    sbatch scripts/metr_la/slurm_train.sh
# Inspect:        squeue -u $USER ; tail -f logs/delos-metrla-<jobid>.out
# Cancel:         scancel <jobid>

set -euo pipefail
mkdir -p logs outputs/metr_la_cjepa

# Environment activation. Pick the path that matches how your cluster works:
#   1) module-managed PyTorch:    module load pytorch/2.5
#   2) personal conda env:        conda activate cjepa
#   3) virtualenv:                source ~/envs/cjepa/bin/activate
if [[ -n "${MODULESHOME:-}" ]] && command -v module >/dev/null 2>&1; then
    : # module load python/3.10  pytorch/2.5  cuda/12.1   # customise here
fi

if [[ -d "$HOME/.conda/envs/cjepa" ]]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate cjepa
fi

# The repo root is the directory sbatch was invoked from: submit from the
# repo root.
REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"
echo "[slurm] repo_root=$REPO_ROOT"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

# We keep --no_attention_mask: the structural mask is (B, 1, TN, TN), which
# forces SDPA off the memory-efficient path, and the materialised
# (B, heads, TN, TN) attention does not fit in GPU memory at B=64, TN=4968.
#
# Use `srun --export=ALL` so PYTHONPATH propagates into the task on clusters
# that strip env vars by default.
srun --export=ALL python src/train/train_delos.py --data_source metr_la --downstream_task regression \
    --metr_la_data_dir dataset/metr_la \
    --history_size 12 \
    --num_preds 12 \
    --node_dim 128 \
    --encoder_kind gcn \
    --encoder_hidden 128 \
    --num_masked_nodes 30 \
    --depth 6 \
    --heads 8 \
    --mlp_dim 1024 \
    --batch_size 64 \
    --no_attention_mask \
    --num_workers ${SLURM_CPUS_PER_TASK:-8} \
    --device cuda \
    --epochs 30 \
    --lr 5e-4 \
    --weight_decay 1e-4 \
    --probe_epochs 5 \
    --out_dir outputs/metr_la_cjepa \
    "$@"
