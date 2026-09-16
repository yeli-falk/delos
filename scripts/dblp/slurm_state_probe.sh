#!/usr/bin/env bash
#SBATCH --job-name=cjepa-stateprobe
#SBATCH --output=logs/cjepa-stateprobe-%j.out
#SBATCH --error=logs/cjepa-stateprobe-%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#
# Run the JODIE state_label (ban prediction) probe across a glob of trained
# JODIE checkpoints. External-label node classification: a stronger test
# than the Louvain probe because the labels are NOT derived from the graph
# topology.
#
# Severely imbalanced (~3% ban rate). Headline metric is AUROC, not accuracy.
#
# Submit (default = all JODIE-Reddit checkpoints):
#     sbatch scripts/dblp/slurm_state_probe.sh
# Different pattern:
#     PATTERN='outputs/jodie_*_cjepa' sbatch scripts/dblp/slurm_state_probe.sh

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

PATTERN=${PATTERN:-'outputs/jodie_reddit_*_cjepa'}
echo "[config] PATTERN=$PATTERN"

srun --export=ALL python scripts/dblp/run_block_probe.py \
    --pattern "$PATTERN" \
    --task binary \
    --device cuda \
    --batch_size 4 \
    --num_workers ${SLURM_CPUS_PER_TASK:-4}

echo
echo "[done] Summarise with:"
echo "  python scripts/dblp/dump_runs.py --pattern '$PATTERN'"
echo "  # AUROC columns: enc_state_auroc, jepa_state_auroc, state_auroc_gap"
