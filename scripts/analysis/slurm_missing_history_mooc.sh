#!/usr/bin/env bash
#SBATCH --job-name=mh-mooc
#SBATCH --output=logs/mh-mooc-%j.out
#SBATCH --error=logs/mh-mooc-%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --partition=gpu
#
# MOOC node-state missing-history sweep, ONE seed per job. Hides a fraction of
# students' interaction history and scores state (dropout) AUROC on those hidden
# nodes for the DELOS forecast (jepa) vs the last-history embedding
# (enc-last) of the SAME frozen encoder. The claim: jepa (trained with node
# masking) degrades least as the hidden fraction grows. Post-hoc, no retraining.
#
# CKPT (required) is the best.ckpt of a trained MOOC DELOS run for that seed:
#   for SEED in 42 43 44; do
#     SEED=$SEED CKPT=outputs/cls_mooc/jepa_ema_s$SEED/best.ckpt \
#       sbatch scripts/analysis/slurm_missing_history_mooc.sh
#   done
# Then aggregate:  python scripts/analysis/dump_missing_history.py --task mooc

set -uo pipefail
mkdir -p logs
REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"; cd "$REPO_ROOT"; export PYTHONPATH="$REPO_ROOT"
if [[ -d "$HOME/.conda/envs/cjepa" ]]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate cjepa
fi

SEED="${SEED:-42}"
STATE_MODE="${STATE_MODE:-state}"
FRACS="${FRACS:-0 0.25 0.5 0.75}"
NW="${SLURM_CPUS_PER_TASK:-8}"
CKPT="${CKPT:-}"
if [[ -z "$CKPT" ]]; then
    echo "[FATAL] set CKPT=/path/to/best.ckpt of a trained MOOC DELOS run" >&2
    exit 2
fi
echo "[config] SEED=$SEED STATE_MODE=$STATE_MODE FRACS='$FRACS' CKPT=$CKPT"

if [[ ! -f "$CKPT" ]]; then
    echo "[FATAL] no checkpoint at $CKPT — train the MOOC DELOS run first" >&2
    exit 10
fi
srun --export=ALL python scripts/analysis/missing_history_mooc.py \
    --jepa_ckpt "$CKPT" --state_label_mode "$STATE_MODE" --fracs $FRACS \
    --num_workers "$NW" --seed "$SEED" --device cuda
echo; echo "[done] seed $SEED. Aggregate: python scripts/analysis/dump_missing_history.py --task mooc"
