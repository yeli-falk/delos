#!/usr/bin/env bash
#SBATCH --job-name=cjepa-baselines
#SBATCH --output=logs/cjepa-baselines-%j.out
#SBATCH --error=logs/cjepa-baselines-%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --time=02:00:00
#SBATCH --partition=cpu
#
# Parameter-free link-prediction baselines. CPU-only (no GPU needed) since
# none of these involve training -- just adjacency lookups + degree counts.
#
# Submit:
#   sbatch scripts/dblp/slurm_baselines.sh                                # default = reddit_body
#   DATASET=jodie_reddit  sbatch scripts/dblp/slurm_baselines.sh
#   DATASET=reddit_title  sbatch scripts/dblp/slurm_baselines.sh
#
# Once complete, the new "baselines_<DATASET>_<name>_cjepa" directories
# slot into the existing dump_runs.py output alongside trained methods:
#   python scripts/dblp/dump_runs.py --pattern 'outputs/*_cjepa' --fmt md

set -uo pipefail
mkdir -p logs outputs

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"

if [[ -d "$HOME/.conda/envs/cjepa" ]]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate cjepa
fi
echo "[slurm] repo_root=$REPO_ROOT  node=$(hostname)"

DATASET=${DATASET:-reddit_body}
OUT=outputs/baselines_${DATASET}

# Same node subset / binning / window as every trained link-pred method — the
# heuristics are scored by the same evaluator against the same negatives, so
# their rows drop straight into the trained-method table.
TASK=link_pred
source "$REPO_ROOT/scripts/dblp/_dataset_config.sh"
echo "[config] OUT=$OUT"

srun --export=ALL python scripts/dblp/run_baselines.py $DATA_ARG $WINDOW_ARG \
    --batch_size ${BATCH:-4} --num_workers ${SLURM_CPUS_PER_TASK:-4} \
    --seed "$SEED" \
    --out_dir "$OUT"

echo
echo "[done] baselines for $DATASET written."
echo "  cat $OUT/baselines.json"
echo "  python scripts/dblp/dump_runs.py --pattern '${OUT}_*_cjepa'"
