#!/usr/bin/env bash
#SBATCH --job-name=cjepa-blockprobe
#SBATCH --output=logs/cjepa-blockprobe-%j.out
#SBATCH --error=logs/cjepa-blockprobe-%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#
# Run the Louvain-community node-classification probe across a glob of
# trained checkpoints. Outputs one block_probe.json per checkpoint dir.
#
# Submit (default = all reddit_body featmask runs):
#     sbatch scripts/dblp/slurm_block_probe.sh
# Different pattern:
#     PATTERN='outputs/reddit_*_cjepa' sbatch scripts/dblp/slurm_block_probe.sh

set -uo pipefail
mkdir -p logs

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"

if [[ -d "$HOME/.conda/envs/cjepa" ]]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate cjepa
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

PATTERN=${PATTERN:-'outputs/reddit_body_featmask_*_cjepa'}
echo "[config] PATTERN=$PATTERN"

srun --export=ALL python scripts/dblp/run_block_probe.py \
    --pattern "$PATTERN" \
    --device cuda \
    --batch_size 4 \
    --num_workers ${SLURM_CPUS_PER_TASK:-4}

echo
echo "[done] Summarise with:"
echo "  python scripts/dblp/dump_runs.py --pattern '$PATTERN'"
