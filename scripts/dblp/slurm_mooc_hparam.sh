#!/usr/bin/env bash
#SBATCH --job-name=mooc-hparam
#SBATCH --output=logs/mooc-hparam-%A_%a.out
#SBATCH --error=logs/mooc-hparam-%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --time=06:00:00
#SBATCH --partition=gpu
#
# Hyperparameter search for the MOOC state-prediction table: DELOS and every
# baseline on one flat SLURM array, so each method gets a comparable tuning
# budget.
#
# Selection is on val, never test: each cell probes with `--probe_split val`
# and writes state_probe_val.json. The test file is never read or written here.
#
# One seed (SEED, default 42): the search picks a configuration. Re-run the
# winners across several seeds afterwards, on test.
#
# Cost: 58 cells. DELOS cells reuse one pretrained encoder per seed, so only
# the predictor is retrained. The baselines tune encoder knobs, so each cell
# retrains its own.
#
# Submit everything, 6 concurrent:
#     sbatch --array=0-57%6 scripts/dblp/slurm_mooc_hparam.sh
# One method only (print the index ranges first):
#     bash scripts/dblp/slurm_mooc_hparam.sh --list
#     sbatch --array=0-17%6 scripts/dblp/slurm_mooc_hparam.sh      # jepa cells
# Rank the results:
#     python scripts/dblp/aggregate_mooc_hparam.py

set -uo pipefail

# Grid: one entry per cell, "<method>|<extra args for that method's trainer>".
# Keep it in sync with --array=0-N above: N = (number of cells) - 1.
build_grid() {
    GRID=()
    # --- DELOS predictor knobs (18) -------------------------------------
    # The encoder is fixed and shared; these are all predictor-side.
    for lr in 5e-4 1e-3; do
      for cw in 0 2.0 5.0; do
        for nm in 20 30 50; do
          GRID+=("jepa|--lr $lr --change_weight $cw --num_masked_nodes $nm")
        done
      done
    done
    # --- CLDG (12): temporal-view sampling + InfoNCE temperature -----------
    for tau in 0.05 0.07 0.5; do
      for snaps in 4 8; do
        for lr in 1e-3 4e-3; do
          GRID+=("cldg|--tau $tau --snapshots $snaps --lr $lr")
        done
      done
    done
    # --- DDGCL (18): debias prior, view gap, lr ----------------------------
    for tau in 0.005 0.01 0.05; do
      for gap in 2 4 8; do
        for lr in 1e-3 4e-3; do
          GRID+=("ddgcl|--tau $tau --max_gap $gap --lr $lr")
        done
      done
    done
    # --- IDOL (6): GGD projection depth + lr -------------------------------
    for lr in 1e-4 1e-3 4e-3; do
      for pl in 1 2; do
        GRID+=("idol|--lr $lr --proj_layers $pl")
      done
    done
    # --- JODIE (4): lr + embedding-drift regulariser -----------------------
    for lr in 1e-4 1e-3; do
      for reg in 0.1 1.0; do
        GRID+=("jodie|--lr $lr --reg $reg")
      done
    done
}

build_grid

if [[ "${1:-}" == "--list" ]]; then
    echo "grid size = ${#GRID[@]}  ->  sbatch --array=0-$(( ${#GRID[@]} - 1 ))%6 $0"
    prev=""; start=0
    for i in "${!GRID[@]}"; do
        m="${GRID[$i]%%|*}"
        if [[ "$m" != "$prev" ]]; then
            [[ -n "$prev" ]] && echo "  $prev: $start-$((i-1))"
            prev="$m"; start=$i
        fi
    done
    echo "  $prev: $start-$(( ${#GRID[@]} - 1 ))"
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
METHOD="${CELL%%|*}"
HP_ARGS="${CELL#*|}"

# --- shared config: identical data/encoder/window for every cell ------------
DATASET=${DATASET:-mooc}
TASK=state_probe
source "$REPO_ROOT/scripts/dblp/_dataset_config.sh"
STATE_MODE=${STATE_MODE:-state}
BATCH=${BATCH:-8}
NW=${SLURM_CPUS_PER_TASK:-8}
# The epoch cap is deliberately high: pretrain_encoder restores best-val
# weights, so PATIENCE decides when to stop.
ENC_EPOCHS=${ENC_EPOCHS:-300}
JEPA_EPOCHS=${JEPA_EPOCHS:-50}
BASE_EPOCHS=${BASE_EPOCHS:-100}
JODIE_EPOCHS=${JODIE_EPOCHS:-30}
# JODIE is by far the heaviest cell: its t-batch loop is sequential over the
# whole stream and the val pass is a second full pass. Scoring val every 5th
# epoch keeps the search affordable; the final run can use VAL_EVERY=1.
VAL_EVERY=${VAL_EVERY:-5}

# A slug of the HP args, so each cell owns its directory.
SLUG=$(echo "$HP_ARGS" | sed 's/--//g; s/[ .]/_/g; s/__*/_/g; s/^_//; s/_$//')
OUT="outputs/mooc_hp/${METHOD}_${SLUG}_s${SEED}"
mkdir -p "$OUT"
# Record the exact flag string: the slug cannot be turned back into flags, and
# the aggregator reads this file to print a pasteable config.
printf '%s\n' "$HP_ARGS" > "$OUT/hp_args.txt"
echo "[cell $IDX] method=$METHOD  hp='$HP_ARGS'  seed=$SEED"
echo "[cell $IDX] out=$OUT"

probe() {   # $1 = flag name (--jepa_ckpt|--encoder_ckpt|--jodie_ckpt), $2 = ckpt
    srun --export=ALL python scripts/dblp/run_block_probe.py "$1" "$2" \
        --probe_split val --task binary --state_label_mode "$STATE_MODE" \
        --device cuda --batch_size 4 --num_workers "$NW"
}

case "$METHOD" in
  jepa)
    # One shared encoder per seed across every JEPA cell: predictor knobs do
    # not touch it. The array runs cells concurrently, so flock serialises the
    # pretrain: the first holder trains, the rest block, then find the file
    # and skip. Without it several cells would write the same checkpoint.
    ENC_DIR="outputs/mooc_hp/_shared_encoder_s${SEED}"
    ENC_CKPT="$ENC_DIR/encoder_${DATASET}_${ENC_KIND}_edge_mask.ckpt"
    mkdir -p "$ENC_DIR"
    exec 9>"$ENC_DIR/.pretrain.lock"
    if ! flock -w 7200 9; then
        echo "[FATAL] timed out waiting for the shared-encoder lock" >&2; exit 10
    fi
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
        echo "[skip] shared encoder exists: $ENC_CKPT"
    fi
    flock -u 9
    exec 9>&-
    srun --export=ALL python src/train/train_delos.py --no_attention_mask \
        $DATA_ARG $ENCODER_ARG $WINDOW_ARG $EARLY_STOP_ARG \
        --encoder_ckpt "$ENC_CKPT" --mask_scheme node_mask --residual_head \
        --depth 4 --heads 4 --mlp_dim 512 \
        --batch_size "$BATCH" --num_workers "$NW" --device cuda \
        --epochs "$JEPA_EPOCHS" --probe_epochs 3 \
        --seed "$SEED" --out_dir "$OUT" $HP_ARGS
    [[ -f "$OUT/best.ckpt" ]] || { echo "[FATAL] JEPA train failed" >&2; exit 12; }
    probe --jepa_ckpt "$OUT/best.ckpt" ;;

  cldg)
    srun --export=ALL python -m src.baselines.cldg.train_cldg $DATA_ARG \
        $ENCODER_ARG $WINDOW_ARG $EARLY_STOP_ARG \
        --views 2 --strategy random --node_batch_size 256 \
        --steps_per_epoch 20 --val_steps 10 --device cuda \
        --epochs "$BASE_EPOCHS" --seed "$SEED" --out_dir "$OUT" $HP_ARGS
    [[ -f "$OUT/encoder.ckpt" ]] || { echo "[FATAL] CLDG failed" >&2; exit 13; }
    probe --encoder_ckpt "$OUT/encoder.ckpt" ;;

  ddgcl)
    srun --export=ALL python -m src.baselines.ddgcl.train_ddgcl $DATA_ARG \
        $ENCODER_ARG $WINDOW_ARG $EARLY_STOP_ARG \
        --time_dim 32 --beta 1.5 --node_batch_size 256 \
        --steps_per_epoch 20 --val_steps 10 --device cuda \
        --epochs "$BASE_EPOCHS" --seed "$SEED" --out_dir "$OUT" $HP_ARGS
    [[ -f "$OUT/encoder.ckpt" ]] || { echo "[FATAL] DDGCL failed" >&2; exit 14; }
    probe --encoder_ckpt "$OUT/encoder.ckpt" ;;

  idol)
    srun --export=ALL python -m src.baselines.idol.train_idol $DATA_ARG \
        $ENCODER_ARG $WINDOW_ARG $EARLY_STOP_ARG \
        --node_batch_size 256 --device cuda \
        --epochs "$BASE_EPOCHS" --seed "$SEED" --out_dir "$OUT" $HP_ARGS
    [[ -f "$OUT/encoder.ckpt" ]] || { echo "[FATAL] IDOL failed" >&2; exit 15; }
    probe --encoder_ckpt "$OUT/encoder.ckpt" ;;

  jodie)
    srun --export=ALL python -m src.baselines.jodie.train_jodie $DATA_ARG \
        $WINDOW_ARG $EARLY_STOP_ARG \
        --node_dim "$NODE_DIM" --device cuda --val_every "$VAL_EVERY" \
        --epochs "$JODIE_EPOCHS" --seed "$SEED" --out_dir "$OUT" $HP_ARGS
    [[ -f "$OUT/jodie.ckpt" ]] || { echo "[FATAL] JODIE failed" >&2; exit 16; }
    probe --jodie_ckpt "$OUT/jodie.ckpt" ;;

  *) echo "[error] unknown method '$METHOD'" >&2; exit 2 ;;
esac

echo "[done] cell $IDX ($METHOD) -> $OUT/state_probe_val.json"
