#!/usr/bin/env bash
#SBATCH --job-name=cls-finetune
#SBATCH --output=logs/cls-finetune-%A_%a.out
#SBATCH --error=logs/cls-finetune-%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#
# FINE-TUNING / LABEL-EFFICIENCY for dynamic node classification.
#
# The DDGCL-style protocol: self-supervised weights are an INITIALISATION, the
# whole network (encoder + predictor + head) is then trained on labels. One
# SLURM array cell = one (init, aux, label fraction). Every cell shares the
# predictor architecture of the table's DyG-JEPA checkpoint and the same
# chronological split / schedule / head; only the starting weights differ.
#
# Inits (INITS, default "random encoder jepa ddgcl"):
#   random    encoder + predictor from scratch         (supervised floor)
#   encoder   the table's shared edge-mask encoder, predictor from scratch
#   jepa      encoder + predictor from $ROOT/$JEPA_ARM_s$SEED/best.ckpt
#   ddgcl|cldg|idol|maskdgnn|dysubc   that baseline's encoder.ckpt, predictor from scratch
#
# Regimes (AUXW, default "0 1.0"): 0 = pure fine-tuning (DDGCL fune_train),
# w>0 = BCE + w * JEPA latent loss (DDGCL's multi-task scheme).
#
# Reads the artefacts the final-table launcher wrote under ROOT, so run
# scripts/classification/slurm_all_methods.sh for the seed first.
#
# Usage:
#   DATASET=mooc ROOT=outputs/cls_mooc_full SEED=42 \
#       bash scripts/classification/slurm_finetune.sh --list
#   DATASET=mooc ROOT=outputs/cls_mooc_full SEED=42 \
#       sbatch --array=0-39%6 scripts/classification/slurm_finetune.sh
#   python scripts/classification/aggregate_finetune.py \
#       --root outputs/cls_mooc_full/finetune --seeds 42 43 44
#
# Knobs: FRACS="0.01 0.05 0.1 0.25 1.0"  INITS  AUXW  JEPA_ARM=jepa_frozen
#        FT_EPOCHS=100 FT_PATIENCE=15 LR=1e-4 ENC_LR=<LR> HEAD_LR=1e-2 CLIP=1.0 BATCH=4
#        LABEL_SEED=0 STATE_MODE=state FT_HP="<extra finetune_state flags>"

set -uo pipefail

DATASET="${DATASET:-mooc}"
SEED="${SEED:-42}"
JEPA_ARM="${JEPA_ARM:-jepa_frozen}"
INITS="${INITS:-random encoder jepa ddgcl}"
AUXW="${AUXW:-0 1.0}"
FRACS="${FRACS:-0.01 0.05 0.1 0.25 1.0}"
LABEL_SEED="${LABEL_SEED:-0}"
STATE_MODE="${STATE_MODE:-state}"

GRID=()
for init in $INITS; do
  for w in $AUXW; do
    for f in $FRACS; do
      GRID+=("$init|$w|$f")
    done
  done
done

if [[ "${1:-}" == "--list" ]]; then
    for i in "${!GRID[@]}"; do echo "$i: ${GRID[$i]}"; done
    echo "cells: ${#GRID[@]}  (sbatch --array=0-$((${#GRID[@]}-1))%6 ...)"
    exit 0
fi

IDX="${SLURM_ARRAY_TASK_ID:-${IDX:-}}"
[[ -z "$IDX" ]] && { echo "[FATAL] no SLURM_ARRAY_TASK_ID / IDX (use --list to see cells)" >&2; exit 2; }
[[ "$IDX" -ge "${#GRID[@]}" ]] && { echo "[FATAL] IDX $IDX >= ${#GRID[@]} cells" >&2; exit 2; }
CELL="${GRID[$IDX]}"
INIT="${CELL%%|*}"; REST="${CELL#*|}"
AUX="${REST%%|*}"; FRAC="${REST#*|}"

mkdir -p logs
REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"
# Full-N adjacency tensors are multi-GB; expandable segments stop the caching
# allocator from stranding reserved-but-unusable blocks between them.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# Interpreter. Three ways to point the job at the right python, in priority:
#   PY=/path/to/python          use this interpreter outright (no activation)
#   CONDA_ENV=<name|/path>      activate this conda env, then use its python
#   (neither)                   activate ~/.conda/envs/cjepa if it exists (ruche)
# On a cluster where none of these resolve, srun fails with
# "execve(): python: No such file or directory" -- hence the hard check.
CONDA_ENV="${CONDA_ENV:-}"
if [[ -z "${PY:-}" ]]; then
    [[ -z "$CONDA_ENV" && -d "$HOME/.conda/envs/cjepa" ]] && CONDA_ENV=cjepa
    if [[ -n "$CONDA_ENV" ]]; then
        CONDA_BASE="$(conda info --base 2>/dev/null || true)"
        [[ -n "$CONDA_BASE" ]] && source "$CONDA_BASE/etc/profile.d/conda.sh"
        conda activate "$CONDA_ENV" || { echo "[FATAL] conda activate $CONDA_ENV failed" >&2; exit 4; }
    fi
    PY=python
fi
command -v "$PY" >/dev/null 2>&1 \
    || { echo "[FATAL] interpreter '$PY' not found. Set PY=/path/to/python or CONDA_ENV=<name|path>" >&2; exit 4; }
echo "[env] python=$(command -v "$PY")  $("$PY" -c 'import torch; print("torch", torch.__version__)' 2>&1 | tail -1)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

TASK=state_probe
source "$REPO_ROOT/scripts/dblp/_dataset_config.sh"
SHORT="${DATASET#jodie_}"
ROOT="${ROOT:-outputs/cls_${SHORT}}"
BATCH="${BATCH:-4}"
NW="${SLURM_CPUS_PER_TASK:-8}"
RUN="${SLURM_JOB_ID:+srun --export=ALL}"

# --- artefacts from the final-table launcher -------------------------------
JEPA_CKPT="$ROOT/${JEPA_ARM}_s${SEED}/best.ckpt"
[[ -f "$JEPA_CKPT" ]] || { echo "[FATAL] $JEPA_CKPT missing: run slurm_all_methods.sh (ARMS=$JEPA_ARM) for seed $SEED first" >&2; exit 10; }
INIT_ARGS=""
case "$INIT" in
  random)  INIT_ARGS="--init random" ;;
  jepa)    INIT_ARGS="--init jepa" ;;
  encoder)
    ENC=$(ls "$ROOT/_shared_encoder_s${SEED}"/encoder_*_edge_mask.ckpt 2>/dev/null | head -1)
    [[ -f "${ENC:-}" ]] || { echo "[FATAL] shared encoder missing under $ROOT/_shared_encoder_s${SEED}" >&2; exit 11; }
    INIT_ARGS="--init encoder --encoder_ckpt $ENC" ;;
  ddgcl|cldg|idol|maskdgnn|dysubc)
    ENC="$ROOT/${INIT}_s${SEED}/encoder.ckpt"
    [[ -f "$ENC" ]] || { echo "[FATAL] $ENC missing: run slurm_all_methods.sh (ARMS=$INIT) first" >&2; exit 12; }
    INIT_ARGS="--init encoder --encoder_ckpt $ENC" ;;
  *) echo "[FATAL] unknown init '$INIT'" >&2; exit 3 ;;
esac

OUT="$ROOT/finetune/${INIT}_aux${AUX}_f${FRAC}_s${SEED}"
mkdir -p "$OUT"
echo "[cell $IDX] init=$INIT aux=$AUX frac=$FRAC seed=$SEED -> $OUT"
if [[ -f "$OUT/finetune.json" ]]; then
    echo "[skip] $OUT/finetune.json exists"; exit 0
fi

ENC_LR_ARG=""
[[ -n "${ENC_LR:-}" ]] && ENC_LR_ARG="--encoder_lr $ENC_LR"
# Head LR defaults to the frozen probe's 1e-2 so head-only cells reproduce it.
HEAD_LR_ARG="--head_lr ${HEAD_LR:-1e-2}"
CLIP_ARG="--clip_norm ${CLIP:-1.0}"

$RUN "$PY" src/train/finetune_state.py $DATA_ARG $WINDOW_ARG $ENCODER_ARG \
    --jepa_ckpt "$JEPA_CKPT" $INIT_ARGS \
    --label_fraction "$FRAC" --label_seed "$LABEL_SEED" \
    --aux_jepa_weight "$AUX" --state_label_mode "$STATE_MODE" \
    --epochs "${FT_EPOCHS:-100}" --early_stop_patience "${FT_PATIENCE:-15}" \
    --lr "${LR:-1e-4}" $ENC_LR_ARG $HEAD_LR_ARG $CLIP_ARG --batch_size "$BATCH" \
    --seed "$SEED" --num_workers "$NW" --device cuda \
    --out_dir "$OUT" ${FT_HP:-} \
    || { echo "[FATAL] finetune cell $IDX failed" >&2; exit 20; }
[[ -f "$OUT/finetune.json" ]] || { echo "[FATAL] cell $IDX wrote no finetune.json" >&2; exit 21; }
echo "[done] cell $IDX -> $OUT"
