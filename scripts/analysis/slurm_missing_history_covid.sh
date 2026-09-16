#!/usr/bin/env bash
#SBATCH --job-name=mh-covid
#SBATCH --output=logs/mh-covid-%j.out
#SBATCH --error=logs/mh-covid-%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#
# England COVID missing-history / partial-observability sweep, ONE seed per job.
# For each method's checkpoint, hides a fraction of nodes' history and scores the
# forecast on those hidden nodes vs relational-persistence (neighbour-mean). The
# claim: DyG-JEPA (trained with node masking) degrades least as the hidden
# fraction grows. Post-hoc — reuses the checkpoints from slurm_all_methods.sh,
# no retraining.
#
#   for SEED in 42 43 44; do SEED=$SEED sbatch scripts/analysis/slurm_missing_history_covid.sh; done
# Then aggregate:  python scripts/analysis/dump_missing_history.py --task covid

set -uo pipefail
mkdir -p logs
REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"; cd "$REPO_ROOT"; export PYTHONPATH="$REPO_ROOT"
if [[ -d "$HOME/.conda/envs/cjepa" ]]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate cjepa
fi

SEED="${SEED:-42}"; HS="${HS:-8}"; NP="${NP:-6}"
FRACS="${FRACS:-0 0.25 0.5 0.75 1.0}"
NW="${SLURM_CPUS_PER_TASK:-4}"
echo "[config] SEED=$SEED HS=$HS NP=$NP FRACS='$FRACS'"

# (name, checkpoint path) — DyG-JEPA predictor + encoder control + baselines.
DECL=(
  "dygjepa       outputs/ec_dygjepa_s${SEED}/best.ckpt"
  "dygjepa_enc   outputs/ec_dygjepa_encoder_s${SEED}/encoder_england_covid_tgcn_masked_signal.ckpt"
  "cldg          outputs/ec_cldg_s${SEED}/encoder.ckpt"
  "ddgcl         outputs/ec_ddgcl_s${SEED}/encoder.ckpt"
  "idol          outputs/ec_idol_s${SEED}/encoder.ckpt"
)
for row in "${DECL[@]}"; do
    name=$(echo "$row" | awk '{print $1}'); ckpt=$(echo "$row" | awk '{print $2}')
    if [[ ! -f "$ckpt" ]]; then echo "[skip] $name: no $ckpt"; continue; fi
    echo; echo "=== missing-history: $name ($ckpt) ==="
    srun --export=ALL python scripts/analysis/missing_history_covid.py \
        --ckpt "$ckpt" --history_size $HS --num_preds $NP --fracs $FRACS \
        --probe_epochs 10 --probe_hidden 64 --num_workers "$NW" \
        --seed "$SEED" --device cuda || echo "[warn] $name failed"
done
echo; echo "[done] seed $SEED. Aggregate: python scripts/analysis/dump_missing_history.py --task covid"
