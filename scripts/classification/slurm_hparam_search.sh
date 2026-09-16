#!/usr/bin/env bash
#SBATCH --job-name=cls-hparam
#SBATCH --output=logs/cls-hparam-%A_%a.out
#SBATCH --error=logs/cls-hparam-%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=20:00:00
#SBATCH --partition=gpu
#
# HYPERPARAMETER SEARCH for dynamic node classification. The classification twin
# of scripts/regression/slurm_hparam_search.sh: a SLURM array job where each task
# id decodes to one grid cell, each cell writes its own output dir, and
# aggregate_hparam.py ranks them.
#
# EVERY cell probes the VALIDATION split (--probe_split val -> state_probe_val.json).
# The test split is never read during the search, so the configuration carried
# into the final 3-seed table was not chosen on the numbers that table reports.
#
# Three stages, because they tune different things:
#
#   STAGE=predictor  (do this first)
#       Sweeps the predictor knobs on the FROZEN arm -- lr x num_masked_nodes x
#       depth x change_weight. The encoder is fixed, so cells differ only in the
#       predictor. The winner becomes BASE_HP for everything downstream.
#
#   STAGE=baselines  (independent of the other two)
#       Tunes the COMPETING methods -- CLDG / DDGCL / IDOL -- so the final table
#       is not "our tuned method vs their defaults". Those defaults came from
#       each paper's own datasets and were never tuned for this subset.
#
#   STAGE=variants   (after predictor)
#       Holds BASE_HP fixed and sweeps only the encoder-training knob each
#       variant actually has. `frozen` has none left, so it contributes a single
#       reference cell; unfreezing makes the encoder LR dominant.
#
#       VARIANT_SET picks which arms are in the grid:
#         old  -> the 4 legacy arms (frozen / ema / naive / vicreg), 16 cells,
#                 the exact grid + indices the earlier sweeps used
#         new  -> the JEPA-recipe arms only (ctx / tnorm / ema_cos / sigreg /
#                 lejepa / ijepa / lejepa_full), 32 cells
#         all  -> old followed by new (default), 48 cells. Indices 0-15 are the
#                 legacy cells unchanged, so an --array=0-15 submission still
#                 reproduces the old sweep and 16-47 appends the new arms.
#       Arms: ctx = frozen + context-encoder masking, tnorm = frozen + target
#       LayerNorm, ema_cos = EMA with cosine momentum, sigreg = stop-grad target
#       + SIGReg, lejepa = no stop-grad + SIGReg, ijepa = ema_cos + ctx + tnorm,
#       lejepa_full = lejepa + ctx + tnorm. `--list` prints every cell.
#
# Usage:
#   DATASET=mooc STAGE=predictor bash scripts/classification/slurm_hparam_search.sh --list
#   DATASET=mooc STAGE=predictor sbatch --array=0-35%6 scripts/classification/slurm_hparam_search.sh
#   python scripts/classification/aggregate_hparam.py --root outputs/cls_hparam/mooc_predictor --emit_flags
#
#   DATASET=mooc STAGE=baselines sbatch --array=0-20%6 scripts/classification/slurm_hparam_search.sh
#   python scripts/classification/aggregate_hparam.py --root outputs/cls_hparam/mooc_baselines --by_arm --emit_flags
#
#   DATASET=mooc STAGE=variants BASE_HP="<from predictor>" \
#       sbatch --array=0-47%4 scripts/classification/slurm_hparam_search.sh     # VARIANT_SET=all
#   DATASET=mooc STAGE=variants VARIANT_SET=new BASE_HP="<from predictor>" \
#       sbatch --array=0-31%4 scripts/classification/slurm_hparam_search.sh     # new arms only
#   python scripts/classification/aggregate_hparam.py --root outputs/cls_hparam/mooc_variants --by_arm --emit_flags

set -uo pipefail

DATASET="${DATASET:-mooc}"
STAGE="${STAGE:-predictor}"
SEED="${SEED:-42}"
# One seed for the search: the ranking only needs to be consistent, and seeds
# are far better spent on the 3-seed final table than on the sweep.
BASE_HP="${BASE_HP:-}"
STATE_MODE="${STATE_MODE:-state}"
VARIANT_SET="${VARIANT_SET:-all}"

build_grid() {
    GRID=()
    if [[ "$STAGE" == "predictor" ]]; then
        for lr in 5e-4 1e-3 2e-3; do
            for nm in 20 30 50; do
                for depth in 4 6; do
                    for cw in 0 2.0; do
                        GRID+=("frozen|--lr $lr --num_masked_nodes $nm --depth $depth --change_weight $cw")
                    done
                done
            done
        done
    elif [[ "$STAGE" == "variants" ]]; then
        if [[ "$VARIANT_SET" == "old" || "$VARIANT_SET" == "all" ]]; then
            # Legacy grid -- do NOT reorder: cell indices are on disk / in logs.
            GRID+=("frozen|")
            for elr in 1e-5 1e-4 5e-4; do
                for m in 0.996 0.999; do
                    GRID+=("ema|--encoder_lr $elr --ema_momentum $m")
                done
                GRID+=("naive|--encoder_lr $elr")
                for vv in 5.0 25.0; do
                    GRID+=("vicreg|--encoder_lr $elr --vicreg_var $vv")
                done
            done
        fi
        if [[ "$VARIANT_SET" == "new" || "$VARIANT_SET" == "all" ]]; then
            # Frozen-encoder recipe arms: no encoder knob, one cell each.
            GRID+=("ctx|")
            GRID+=("tnorm|")
            for elr in 1e-5 1e-4 5e-4; do
                # cosine anneal: sweep the START momentum (end is 1.0)
                for m in 0.996 0.999; do
                    GRID+=("ema_cos|--encoder_lr $elr --ema_momentum $m")
                    GRID+=("ijepa|--encoder_lr $elr --ema_momentum $m")
                done
                # SIGReg weight replaces vicreg_var as the collapse knob. The raw
                # statistic sits ~0.25-0.3 at init, so 0.5 / 2.0 bracket "as strong
                # as the prediction term".
                for sw in 0.5 2.0; do
                    GRID+=("sigreg|--encoder_lr $elr --sigreg $sw")
                    GRID+=("lejepa|--encoder_lr $elr --sigreg $sw")
                    GRID+=("lejepa_full|--encoder_lr $elr --sigreg $sw")
                done
            done
        fi
        if [[ "$VARIANT_SET" != "old" && "$VARIANT_SET" != "new" && "$VARIANT_SET" != "all" ]]; then
            echo "[FATAL] VARIANT_SET must be old, new or all, got '$VARIANT_SET'" >&2
            exit 2
        fi
    elif [[ "$STAGE" == "baselines" ]]; then
        # LR plus each method's most load-bearing knob, matching the regression
        # sweep's shape. Cheap: none of these run the T*N-token predictor.
        for lr in 1e-3 4e-3 1e-2; do
            for v in 2 4; do GRID+=("cldg|--lr $lr --views $v"); done
        done
        for lr in 1e-3 4e-3 1e-2; do
            for g in 2 4 8; do GRID+=("ddgcl|--lr $lr --max_gap $g"); done
        done
        for lr in 3e-4 1e-3 3e-3; do
            for pl in 1 2; do GRID+=("idol|--lr $lr --proj_layers $pl"); done
        done
        # MaskDGNN: LR x mask ratio (the paper's most sensitive knob; best at
        # 10-30%, collapses past 40%).
        for lr in 1e-3 4e-3 1e-2; do
            for mr in 0.1 0.2 0.4; do GRID+=("maskdgnn|--lr $lr --mask_ratio $mr"); done
        done
        # DySubC: LR x alpha (the sampling trade-off between recency and
        # degree; the paper's most sensitive knob after subgraph size).
        for lr in 1e-3 4e-3 1e-2; do
            for a in 1 5 10; do GRID+=("dysubc|--lr $lr --alpha $a"); done
        done
    else
        echo "[FATAL] STAGE must be 'predictor', 'variants' or 'baselines', " \
             "got '$STAGE'" >&2
        exit 2
    fi
}

build_grid

if [[ "${1:-}" == "--list" ]]; then
    echo "DATASET=$DATASET STAGE=$STAGE VARIANT_SET=$VARIANT_SET  grid size=${#GRID[@]}"
    echo "  ->  sbatch --array=0-$(( ${#GRID[@]} - 1 ))%6 $0"
    for i in "${!GRID[@]}"; do printf "  [%2d] %s\n" "$i" "${GRID[$i]}"; done
    exit 0
fi

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
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

IDX="${SLURM_ARRAY_TASK_ID:-0}"
if (( IDX >= ${#GRID[@]} )); then
    echo "[error] array index $IDX >= grid size ${#GRID[@]}" >&2; exit 2
fi
CELL="${GRID[$IDX]}"
ARM="${CELL%%|*}"
HP_ARGS="${CELL#*|}"

CSV="dataset/jodie/${DATASET}.csv"
[[ "$DATASET" == "synthetic_dyn" || -f "$CSV" ]] || { echo "[FATAL] $CSV missing; fetch it from the login node" >&2; exit 10; }

TASK=state_probe
source "$REPO_ROOT/scripts/dblp/_dataset_config.sh"
BATCH="${BATCH:-8}"
NW="${SLURM_CPUS_PER_TASK:-8}"
SHORT="${DATASET#jodie_}"

# Sourcing _dataset_config.sh can clobber loop variables, so re-derive the arm
# flags AFTER it -- a clobbered $ARM would mislabel a whole cell.
arm_flags() {
    case "$1" in
        frozen)      echo "" ;;
        naive)       echo "--train_encoder" ;;
        ema)         echo "--train_encoder --ema_target" ;;
        vicreg)      echo "--train_encoder --vicreg_cov 1.0" ;;
        input)       echo "--predict_space input" ;;
        # --- JEPA-recipe variants (2026-09): each flag is opt-in in
        # train_dyg_jepa.py, so the five arms above are byte-for-byte the old ones.
        ctx)         echo "--mask_encoder_input" ;;
        tnorm)       echo "--target_norm" ;;
        ema_cos)     echo "--train_encoder --ema_target --ema_schedule cosine" ;;
        sigreg)      echo "--train_encoder --sigreg 1.0" ;;
        lejepa)      echo "--train_encoder --target_grad --sigreg 1.0" ;;
        ijepa)       echo "--train_encoder --ema_target --ema_schedule cosine --mask_encoder_input --target_norm" ;;
        lejepa_full) echo "--train_encoder --target_grad --sigreg 1.0 --mask_encoder_input --target_norm" ;;
        *)           echo "__UNKNOWN__" ;;
    esac
}

SLUG=$(echo "$HP_ARGS" | sed 's/--//g; s/[ .]/_/g; s/__*/_/g; s/^_//; s/_$//')
[[ -z "$SLUG" ]] && SLUG=base
ROOT="${ROOT:-outputs/cls_hparam/${SHORT}_${STAGE}}"
OUT="$ROOT/${ARM}_${SLUG}_s${SEED}"
mkdir -p "$OUT"
printf '%s\n' "arm=$ARM flags=$HP_ARGS base=$BASE_HP" > "$OUT/hp_args.txt"
echo "[cell $IDX] dataset=$DATASET stage=$STAGE arm=$ARM seed=$SEED -> $OUT"

# synthetic_dyn exposes a dynamic MULTICLASS state (see slurm_all_methods.sh);
# rank its cells with aggregate_hparam.py --metric acc_test (or
# subset_switch.acc_test for the anticipation-only view).
PROBE_TASK=binary
[[ "$DATASET" == "synthetic_dyn" && "${SYN_STATE_TARGET:-community}" != "switch" ]] && PROBE_TASK=multiclass
PROBE_COMMON="--task $PROBE_TASK --state_label_mode $STATE_MODE --probe_split val \
    --device cuda --batch_size ${PROBE_BATCH:-4} --num_workers $NW"

# --- STAGE=baselines: one competing method per cell, probed on VAL -----------
if [[ "$STAGE" == "baselines" ]]; then
    case "$ARM" in
      cldg)
        srun --export=ALL "$PY" -m src.baselines.cldg.train_cldg $DATA_ARG \
            $ENCODER_ARG $WINDOW_ARG $EARLY_STOP_ARG \
            --snapshots 4 --strategy random --tau 0.07 \
            --node_batch_size 256 --steps_per_epoch 20 --val_steps 10 \
            --epochs "${BASE_EPOCHS:-100}" --device cuda --seed "$SEED" \
            --out_dir "$OUT" $HP_ARGS \
            || { echo "[FATAL] CLDG cell failed" >&2; exit 21; } ;;
      ddgcl)
        srun --export=ALL "$PY" -m src.baselines.ddgcl.train_ddgcl $DATA_ARG \
            $ENCODER_ARG $WINDOW_ARG $EARLY_STOP_ARG \
            --time_dim 32 --tau 0.01 --beta 1.5 \
            --node_batch_size 256 --steps_per_epoch 20 --val_steps 10 \
            --epochs "${BASE_EPOCHS:-100}" --device cuda --seed "$SEED" \
            --out_dir "$OUT" $HP_ARGS \
            || { echo "[FATAL] DDGCL cell failed" >&2; exit 22; } ;;
      idol)
        srun --export=ALL "$PY" -m src.baselines.idol.train_idol $DATA_ARG \
            $ENCODER_ARG $WINDOW_ARG $EARLY_STOP_ARG \
            --node_batch_size 256 --steps_per_epoch 20 --val_steps 10 \
            --epochs "${BASE_EPOCHS:-100}" --device cuda --seed "$SEED" \
            --out_dir "$OUT" $HP_ARGS \
            || { echo "[FATAL] IDOL cell failed" >&2; exit 23; } ;;
      maskdgnn)
        srun --export=ALL "$PY" -m src.baselines.maskdgnn.train_maskdgnn $DATA_ARG \
            $ENCODER_ARG --encoder_kind maskdgnn $WINDOW_ARG $EARLY_STOP_ARG \
            --alpha 0.7 --target next --neg_per_pos 1 \
            --steps_per_epoch 20 --val_steps 10 --e2e_eval none \
            --epochs "${BASE_EPOCHS:-100}" --device cuda --seed "$SEED" \
            --out_dir "$OUT" $HP_ARGS \
            || { echo "[FATAL] MaskDGNN cell failed" >&2; exit 25; } ;;
      dysubc)
        srun --export=ALL "$PY" -m src.baselines.dysubc.train_dysubc $DATA_ARG \
            $ENCODER_ARG $WINDOW_ARG $EARLY_STOP_ARG \
            --subgraph_size 20 --beta 1 --margin 0.75 --lam 0.5 \
            --node_batch_size 256 --steps_per_epoch 20 --val_steps 10 \
            --epochs "${BASE_EPOCHS:-100}" --device cuda --seed "$SEED" \
            --out_dir "$OUT" $HP_ARGS \
            || { echo "[FATAL] DySubC cell failed" >&2; exit 26; } ;;
      *) echo "[FATAL] unknown baseline arm '$ARM'" >&2; exit 3 ;;
    esac
    srun --export=ALL "$PY" scripts/dblp/run_block_probe.py \
        --encoder_ckpt "$OUT/encoder.ckpt" $PROBE_COMMON \
        || { echo "[FATAL] probe failed" >&2; exit 24; }
    [[ -f "$OUT/state_probe_val.json" ]] || { echo "[FATAL] cell $IDX wrote no metrics" >&2; exit 12; }
    echo "[done] cell $IDX ($ARM) -> $OUT"
    exit 0
fi

VFLAGS="$(arm_flags "$ARM")"
[[ "$VFLAGS" == "__UNKNOWN__" ]] && { echo "[FATAL] bad arm '$ARM'" >&2; exit 3; }

# --- shared pretrained encoder ----------------------------------------------
# Every cell starts from THIS encoder, so the sweep measures the predictor /
# encoder-training knobs and not pretraining variance. flock because concurrent
# array cells would otherwise all pretrain it at once.
ENC_DIR="$ROOT/_shared_encoder_s${SEED}"
ENC_CKPT="$ENC_DIR/encoder_${DATASET}_${ENC_KIND}_edge_mask.ckpt"
mkdir -p "$ENC_DIR"
exec 9>"$ENC_DIR/.pretrain.lock"
flock -w 21600 9 || { echo "[FATAL] encoder lock timeout" >&2; exit 10; }
# Same completion rule as slurm_all_methods.sh: the ckpt is saved on every val
# improvement, so only the `.done` marker (touched after a clean exit) proves
# the pretrain finished. ENC_TRUST=1 accepts a marker-less ckpt as complete.
if [[ -f "$ENC_CKPT" && -f "$ENC_DIR/.done" ]]; then
    echo "[skip] shared encoder complete"
elif [[ -f "$ENC_CKPT" && "${ENC_TRUST:-0}" == "1" ]]; then
    echo "[skip] shared encoder exists without .done marker; ENC_TRUST=1 -> keeping it"
    touch "$ENC_DIR/.done"
else
    [[ -f "$ENC_CKPT" ]] && echo "[resume] $ENC_CKPT has no .done marker (interrupted run?) -> retraining"
    echo "=== shared encoder pretrain (lock held by cell $IDX) ==="
    rm -f "$ENC_CKPT"
    if srun --export=ALL "$PY" src/train/pretrain_encoder.py $DATA_ARG \
        $ENCODER_ARG $WINDOW_ARG $EARLY_STOP_ARG \
        --pretext edge_mask --decoder_hidden 64 --mask_rate 0.25 \
        --batch_size "$BATCH" --epochs "${ENC_EPOCHS:-300}" --lr 1e-3 \
        --seed "$SEED" --num_workers "$NW" --device cuda --out_dir "$ENC_DIR" \
       && [[ -f "$ENC_CKPT" ]]; then
        touch "$ENC_DIR/.done"
    else
        flock -u 9; echo "[FATAL] encoder pretrain failed" >&2; exit 11
    fi
fi
flock -u 9; exec 9>&-

# --- train + probe THE VAL SPLIT --------------------------------------------
# A cell is complete once its val probe wrote state_probe_val.json; resubmitting
# the array (e.g. after a walltime kill) then re-runs only the unfinished cells.
if [[ -f "$OUT/state_probe_val.json" ]]; then
    echo "[skip] cell $IDX complete: $OUT/state_probe_val.json exists"; exit 0
fi
if [[ -f "$OUT/best.ckpt" ]]; then
    echo "[resume] cell $IDX has best.ckpt but no state_probe_val.json (interrupted) -> restarting from scratch"
    rm -f "$OUT/best.ckpt" "$OUT/history.json" "$OUT/test_metrics.json"
fi
srun --export=ALL "$PY" src/train/train_dyg_jepa.py \
    $DATA_ARG $ENCODER_ARG $WINDOW_ARG $EARLY_STOP_ARG \
    --encoder_ckpt "$ENC_CKPT" --mask_scheme "${MASK_SCHEME:-node_mask}" \
    --residual_head \
    --heads "${HEADS:-4}" --mlp_dim "${MLP_DIM:-512}" \
    --batch_size "$BATCH" --epochs "${JEPA_EPOCHS:-300}" \
    --probe_epochs "${JEPA_PROBE_EPOCHS:-3}" \
    --seed "$SEED" --num_workers "$NW" --device cuda --out_dir "$OUT" \
    $BASE_HP $VFLAGS $HP_ARGS
[[ -f "$OUT/best.ckpt" ]] || { echo "[FATAL] cell $IDX produced no best.ckpt" >&2; exit 12; }

srun --export=ALL "$PY" scripts/dblp/run_block_probe.py \
    --jepa_ckpt "$OUT/best.ckpt" $PROBE_COMMON \
    || { echo "[FATAL] probe failed" >&2; exit 13; }
[[ -f "$OUT/state_probe_val.json" ]] || { echo "[FATAL] cell $IDX wrote no metrics" >&2; exit 12; }

echo "[done] cell $IDX -> $OUT"
