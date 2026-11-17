#!/usr/bin/env bash
#SBATCH --job-name=dyn-audit
#SBATCH --output=logs/dyn-audit-%j.out
#SBATCH --error=logs/dyn-audit-%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=03:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#
# Dataset dynamics audit (scripts/analysis/dataset_dynamics.py) for every
# dynamic-label dataset in the suite: label persistence / switch structure,
# label-history and graph(t-1) probes (is a switch predictable, and from what),
# snapshot-to-snapshot structure, coverage. CPU-only work; the GPU partition
# is requested only because it is the partition every cluster here has.
# Override with PARTITION=cpu_med (ruche) if you have one. 120 GB covers full
# DBLP-S (17 GB dense adjacency + 15 GB features + the probe rows).
#
#   sbatch scripts/analysis/slurm_dataset_dynamics.sh
#   DATASETS="chase_dblp_s" sbatch scripts/analysis/slurm_dataset_dynamics.sh
#   EXTRA="--no_probe" sbatch ...                      # layers 1, 4, 5 only
#
# Writes outputs/dataset_dynamics/<tag>.json and prints one block per dataset
# plus a summary table at the end of the log.

set -uo pipefail
mkdir -p logs outputs/dataset_dynamics
REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"; cd "$REPO_ROOT"; export PYTHONPATH="$REPO_ROOT"
if [[ -d "$HOME/.conda/envs/cjepa" ]]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate cjepa
elif [[ -f "$REPO_ROOT/.venv/bin/activate" ]]; then
    source "$REPO_ROOT/.venv/bin/activate"
fi

DATASETS="${DATASETS:-mooc reddit wikipedia chase_music chase_gossipcop chase_dblp_s bts_flights nyc_taxi}"
TAG="${TAG:-$(date +%Y%m%d)_${SLURM_JOB_ID:-local}}"
# --feats probes the release / ops node features at t-1 as well (PCA 64).
EXTRA="${EXTRA:---feats}"

echo "[config] DATASETS='$DATASETS' EXTRA='$EXTRA' -> outputs/dataset_dynamics/$TAG.json"
python scripts/analysis/dataset_dynamics.py --datasets $DATASETS $EXTRA \
    --out "outputs/dataset_dynamics/$TAG.json"
