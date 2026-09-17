#!/usr/bin/env bash
#SBATCH --job-name=mooc-jepavar
#SBATCH --output=logs/mooc-jepavar-%A_%a.out
#SBATCH --error=logs/mooc-jepavar-%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --time=08:00:00
#SBATCH --partition=gpu
#
# DyG-JEPA VARIANTS on MOOC state prediction: how the encoder is trained.
#
#   frozen  encoder frozen, predictor only.  The targets are fixed, so latent
#           collapse is impossible by construction. This is the reference arm.
#   ema     --train_encoder --ema_target.  I-JEPA: the online encoder builds the
#           context, an EMA copy builds the targets. The standard way to train
#           end-to-end in latent space without collapsing.
#   naive   --train_encoder alone.  Targets come from the SAME encoder being
#           optimised, so mapping everything to one point is a global optimum.
#           This arm exists to show whether EMA is load-bearing — without it a
#           reader cannot tell whether `ema` beat `frozen` because of EMA or
#           merely because the encoder was unfrozen.
#   vicreg  --train_encoder + VICReg variance/covariance on the predictor output.
#           Anti-collapse WITHOUT a target network: does any collapse penalty do
#           the job, or is the EMA target specifically what matters?
#
# All four start from the SAME pretrained encoder and the SAME tuned predictor
# hyperparameters (BASE_HP), so the only difference is the encoder-training rule.
#
# Two modes:
#   MODE=search  one seed, small per-variant grid, probes the VAL split.
#   MODE=final   3 seeds at the chosen config, probes the TEST split.
#
# Usage:
#   bash scripts/dblp/slurm_mooc_jepa_variants.sh --list          # grid + sizes
#   MODE=search sbatch --array=0-15%4 scripts/dblp/slurm_mooc_jepa_variants.sh
#   python scripts/dblp/aggregate_mooc_hparam.py --root outputs/mooc_jepa_variants
#   MODE=final VARIANT_HP_ema="--encoder_lr 1e-4 --ema_momentum 0.996" \
#       sbatch --array=0-11%4 scripts/dblp/slurm_mooc_jepa_variants.sh
#   python scripts/dblp/aggregate_mooc_state.py --layout variants --seeds 42 43 44

set -uo pipefail

MODE=${MODE:-search}
SEEDS_FINAL=${SEEDS_FINAL:-"42 43 44"}

# Predictor hyperparameters shared by every arm — paste the winner from
# aggregate_mooc_hparam.py so the variants differ ONLY in encoder training.
BASE_HP=${BASE_HP:-"--lr 1e-3 --change_weight 2.0 --num_masked_nodes 30"}

# Variant name -> the flags that DEFINE it. Names are lowercase letters only so
# aggregate_mooc_hparam.py's <method>_<slug>_s<seed> parser groups them.
arm_flags() {
    case "$1" in
        frozen) echo "" ;;
        ema)    echo "--train_encoder --ema_target" ;;
        naive)  echo "--train_encoder" ;;
        vicreg) echo "--train_encoder --vicreg_cov 1.0" ;;
        # NOTE: `exit` here would only leave the $( ) subshell, so callers
        # must check the result — see the guard after sourcing the config.
        *) echo "__UNKNOWN__" ;;
    esac
}

build_grid() {
    GRID=()
    if [[ "$MODE" == "search" ]]; then
        # frozen has no encoder-side knob left to tune (BASE_HP already covers
        # the predictor), so it contributes a single reference cell.
        GRID+=("frozen|")
        # Unfreezing makes the encoder LR the dominant knob: too high and the
        # targets move faster than the predictor can track.
        for elr in 1e-5 1e-4 5e-4; do
            for m in 0.996 0.999; do
                GRID+=("ema|--encoder_lr $elr --ema_momentum $m")
            done
            GRID+=("naive|--encoder_lr $elr")
            for vv in 5.0 25.0; do
                GRID+=("vicreg|--encoder_lr $elr --vicreg_var $vv")
            done
        done
    else
        for v in frozen ema naive vicreg; do
            for s in $SEEDS_FINAL; do GRID+=("$v|SEED=$s"); done
        done
    fi
}

build_grid

if [[ "${1:-}" == "--list" ]]; then
    echo "MODE=$MODE  grid size = ${#GRID[@]}  ->  sbatch --array=0-$(( ${#GRID[@]} - 1 ))%4 $0"
    for i in "${!GRID[@]}"; do printf "  [%2d] %s\n" "$i" "${GRID[$i]}"; done
    exit 0
fi

mkdir -p logs outputs
REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"
if [[ -d "$HOME/.conda/envs/cjepa" ]]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate cjepa
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

IDX=${SLURM_ARRAY_TASK_ID:-0}
if (( IDX >= ${#GRID[@]} )); then
    echo "[error] array index $IDX >= grid size ${#GRID[@]}" >&2; exit 2
fi
CELL="${GRID[$IDX]}"
ARM="${CELL%%|*}"
CELL_ARG="${CELL#*|}"

DATASET=${DATASET:-mooc}
TASK=state_probe
source "$REPO_ROOT/scripts/dblp/_dataset_config.sh"
STATE_MODE=${STATE_MODE:-state}
BATCH=${BATCH:-8}
NW=${SLURM_CPUS_PER_TASK:-8}
# 30 epochs was a hard cap, not convergence: the pretrain val loss was
# still falling at epoch 30 (best val landed ON the last epoch), and with
# PATIENCE=20 early stopping cannot fire inside 30 epochs. pretrain_encoder
# restores best-val weights, so a high cap is monotone-safe — it can only
# find an equal or better checkpoint. Let PATIENCE decide when to stop.
ENC_EPOCHS=${ENC_EPOCHS:-300}
JEPA_EPOCHS=${JEPA_EPOCHS:-50}

if [[ "$MODE" == "final" ]]; then
    SEED="${CELL_ARG#SEED=}"          # final cells carry their own seed
    # Per-variant winning flags from the search, e.g. VARIANT_HP_ema="..."
    eval "HP_ARGS=\${VARIANT_HP_${ARM}:-}"
    PROBE_SPLIT=test
    OUT="outputs/mooc_jepa_variants/${ARM}_final_s${SEED}"
else
    HP_ARGS="$CELL_ARG"
    PROBE_SPLIT=val
    SLUG=$(echo "$HP_ARGS" | sed 's/--//g; s/[ .]/_/g; s/__*/_/g; s/^_//; s/_$//')
    [[ -z "$SLUG" ]] && SLUG=base
    OUT="outputs/mooc_jepa_variants/${ARM}_${SLUG}_s${SEED}"
fi

# Sourcing _dataset_config.sh clobbers any variable it also uses (it sets
# VARIANT=${DATASET#jodie_}), so re-derive the arm flags AFTER it and verify.
VFLAGS=$(arm_flags "$ARM")
if [[ "$VFLAGS" == "__UNKNOWN__" ]]; then
    echo "[FATAL] arm '$ARM' is not one of frozen/ema/naive/vicreg — a sourced" >&2
    echo "        script probably clobbered it. Refusing to run a mislabelled cell." >&2
    exit 2
fi
mkdir -p "$OUT"
printf '%s\n' "$VFLAGS $HP_ARGS" > "$OUT/hp_args.txt"
echo "[cell $IDX] mode=$MODE variant=$ARM seed=$SEED probe=$PROBE_SPLIT"
echo "[cell $IDX] flags='$VFLAGS $HP_ARGS'  base='$BASE_HP'"
echo "[cell $IDX] out=$OUT"

# --- shared pretrained encoder (same file the method search uses) ------------
# Every arm starts from THIS encoder; `frozen` keeps it, the others fine-tune a
# copy. flock because concurrent array cells would otherwise all pretrain it.
ENC_DIR="outputs/mooc_hp/_shared_encoder_s${SEED}"
ENC_CKPT="$ENC_DIR/encoder_${DATASET}_${ENC_KIND}_edge_mask.ckpt"
mkdir -p "$ENC_DIR"
exec 9>"$ENC_DIR/.pretrain.lock"
flock -w 7200 9 || { echo "[FATAL] encoder lock timeout" >&2; exit 10; }
if [[ ! -f "$ENC_CKPT" ]]; then
    echo "=== shared encoder pretrain (lock held by cell $IDX) -> $ENC_DIR ==="
    srun --export=ALL python src/train/pretrain_encoder.py $DATA_ARG \
        $ENCODER_ARG $WINDOW_ARG $EARLY_STOP_ARG \
        --decoder_hidden 64 --mask_rate 0.25 --pretext edge_mask \
        --batch_size "$BATCH" --num_workers "$NW" --device cuda \
        --epochs "$ENC_EPOCHS" --lr 1e-3 --seed "$SEED" --out_dir "$ENC_DIR"
    if [[ ! -f "$ENC_CKPT" ]]; then
        flock -u 9; echo "[FATAL] encoder pretrain failed" >&2; exit 11
    fi
else
    echo "[skip] shared encoder exists"
fi
flock -u 9; exec 9>&-

# --- train the variant ------------------------------------------------------
# Order matters: BASE_HP then the variant flags then the cell's own knobs, so
# the most specific setting wins (argparse honours the last occurrence).
srun --export=ALL python src/train/train_dyg_jepa.py --no_attention_mask \
    $DATA_ARG $ENCODER_ARG $WINDOW_ARG $EARLY_STOP_ARG \
    --encoder_ckpt "$ENC_CKPT" --mask_scheme node_mask --residual_head \
    --depth 4 --heads 4 --mlp_dim 512 \
    --batch_size "$BATCH" --num_workers "$NW" --device cuda \
    --epochs "$JEPA_EPOCHS" --probe_epochs 3 \
    --seed "$SEED" --out_dir "$OUT" $BASE_HP $VFLAGS $HP_ARGS
[[ -f "$OUT/best.ckpt" ]] || { echo "[FATAL] $ARM train failed" >&2; exit 12; }

srun --export=ALL python scripts/dblp/run_block_probe.py --jepa_ckpt "$OUT/best.ckpt" \
    --probe_split "$PROBE_SPLIT" --task binary --state_label_mode "$STATE_MODE" \
    --device cuda --batch_size 4 --num_workers "$NW"

# Collapse check for the unfrozen arms. If the encoder collapsed, every node's
# latent is (nearly) the same point: the probe cannot separate anything and the
# forecast column sits at chance no matter how good the training loss looked.
if [[ "$ARM" != "frozen" ]]; then
    srun --export=ALL python -m src.train.analyze_jepa --skip_diagnostic \
        --link_sources jepa $DATA_ARG $WINDOW_ARG \
        --jepa_ckpt "$OUT/best.ckpt" --device cuda --seed "$SEED" \
        --out_dir "$OUT/collapse_check" || \
        echo "[warn] collapse check failed (non-fatal)"
fi

echo "[done] cell $IDX ($ARM) -> $OUT"
