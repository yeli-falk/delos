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
# Hyperparameter search for dynamic node classification: a SLURM array job in
# which each task id is one grid cell with its own output directory;
# aggregate_hparam.py ranks the cells. Every cell probes the validation split
# (state_probe_val.json), so the test split is never read during the search.
# The full protocol is in scripts/classification/README.md.
#
# Stages (STAGE=):
#   predictor  Run first. Predictor knobs on the frozen arm (lr x
#              num_masked_nodes x depth x change_weight). The winner is BASE_HP.
#   baselines  Independent. Tunes each competing method (lr x its main knob).
#   variants   After predictor. BASE_HP fixed, sweeps the encoder-training knob
#              of each arm. VARIANT_SET=old (frozen / ema / naive / vicreg),
#              new (the JEPA-recipe arms) or all (old then new, default).
#   ema        After variants. A wider encoder_lr x ema_momentum grid for the
#              arms in EMA_ARMS (ema, ema_cos, ema_tnorm, ema_cos_tnorm; default
#              ema). EMA_FREEZE="0 10 30" appends the grid with
#              --encoder_freeze_epochs 10 and 30. Cells go to the variants root
#              so they share its encoder and are ranked with --by_arm; pass
#              ROOT= if that root holds a different data configuration.
#   ema_pred   Predictor knobs (lr x mask_frac x change_weight) re-tuned under a
#              trained EMA encoder, with the target knobs fixed at ENC_HP. Run
#              it on two seeds (aggregate_hparam.py averages over seeds); copy
#              the _shared_encoder_s<seed> directories into its root to skip
#              re-pretraining.
#
# Usage (`--list` prints every cell and the --array range of each arm):
#   DATASET=mooc STAGE=predictor bash scripts/classification/slurm_hparam_search.sh --list
#   DATASET=mooc STAGE=predictor sbatch --array=0-35%6 scripts/classification/slurm_hparam_search.sh
#   python scripts/classification/aggregate_hparam.py --root outputs/cls_hparam/mooc_predictor --emit_flags
#   DATASET=mooc STAGE=baselines sbatch --array=0-80%6 scripts/classification/slurm_hparam_search.sh
#   DATASET=mooc STAGE=variants BASE_HP="<from predictor>" \
#       sbatch --array=0-59%4 scripts/classification/slurm_hparam_search.sh
#   DATASET=mooc STAGE=ema EMA_ARMS="ema ema_cos" BASE_HP="<from predictor>" \
#       sbatch --array=0-23%4 scripts/classification/slurm_hparam_search.sh
#   for S in 42 43; do DATASET=mooc STAGE=ema_pred SEED=$S BATCH=2 BASE_HP="<from predictor>" \
#       sbatch --array=0-23%4 scripts/classification/slurm_hparam_search.sh; done
#   python scripts/classification/aggregate_hparam.py --root outputs/cls_hparam/mooc_variants --by_arm --emit_flags
#
# Other knobs: SEED (42), LRS, EMA_ANNEAL_EPOCHS (50), ENC_HP, ONLY_ARMS, ROOT,
# BATCH (8), PROBE_BATCH (4), DOWNSTREAM (none: the JEPA cells skip the
# trainer's end-of-run link probe, which nothing here reads).

set -uo pipefail

DATASET="${DATASET:-mooc}"
STAGE="${STAGE:-predictor}"
SEED="${SEED:-42}"
# The search uses one seed: the ranking only needs to be consistent.
BASE_HP="${BASE_HP:-}"
# One learning-rate grid shared by every arm that sweeps --lr (all baselines
# and the DELOS predictor / ema_pred stages), so that no method is tuned on a
# grid of its own. The decade ladder spans every reference default. Cell
# directories are named by flag values, so existing cells are reused. If you
# override LRS, use the same override for every stage.
LRS="${LRS:-1e-4 1e-3 1e-2}"
STATE_MODE="${STATE_MODE:-state}"
VARIANT_SET="${VARIANT_SET:-all}"
EMA_ARMS="${EMA_ARMS:-ema}"
# Encoder freeze epochs swept by STAGE=ema (outer loop, so the K=0 cells keep
# their indices).
EMA_FREEZE="${EMA_FREEZE:-0}"
# Cosine-anneal horizon shared by every EMA arm that anneals its momentum
# (DELOS ema_cos / ijepa / ema_cos_tnorm and the BGRL baseline). The trainers'
# default spans the whole --epochs budget, which early stopping rarely reaches,
# so the schedule would never complete. 0 = anneal over the whole run.
EMA_ANNEAL_EPOCHS="${EMA_ANNEAL_EPOCHS:-50}"
ANNEAL_ARG=""; [[ "$EMA_ANNEAL_EPOCHS" != "0" ]] && ANNEAL_ARG="--ema_anneal_epochs $EMA_ANNEAL_EPOCHS"
ENC_HP="${ENC_HP:---encoder_lr 5e-4 --ema_momentum 0.999}"

ema_arm_ok() {
    case "$1" in
        ema|ema_cos|ema_tnorm|ema_cos_tnorm) ;;
        *) echo "[FATAL] EMA_ARMS entries must be ema, ema_cos, ema_tnorm or ema_cos_tnorm, got '$1'" >&2
           exit 2 ;;
    esac
}

build_grid() {
    GRID=()
    if [[ "$STAGE" == "predictor" ]]; then
        for lr in $LRS; do
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
            # Do not reorder: cell indices are on disk and in logs.
            # encoder_lr uses the same ladder as --lr (LRS).
            GRID+=("frozen|")
            for elr in $LRS; do
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
            for elr in $LRS; do
                # Cosine anneal: sweep the start momentum (the end is 1.0).
                for m in 0.996 0.999; do
                    GRID+=("ema_cos|--encoder_lr $elr --ema_momentum $m")
                    GRID+=("ijepa|--encoder_lr $elr --ema_momentum $m")
                done
                # The SIGReg weight replaces vicreg_var as the collapse knob.
                for sw in 0.5 2.0; do
                    GRID+=("sigreg|--encoder_lr $elr --sigreg $sw")
                    GRID+=("lejepa|--encoder_lr $elr --sigreg $sw")
                    GRID+=("lejepa_full|--encoder_lr $elr --sigreg $sw")
                done
            done
            # EMA arms with target LayerNorm but without context masking, which
            # isolates tnorm from ctx (the ijepa arm bundles both).
            for elr in $LRS; do
                for m in 0.996 0.999; do
                    GRID+=("ema_tnorm|--encoder_lr $elr --ema_momentum $m")
                    GRID+=("ema_cos_tnorm|--encoder_lr $elr --ema_momentum $m")
                done
            done
        fi
        if [[ "$VARIANT_SET" != "old" && "$VARIANT_SET" != "new" && "$VARIANT_SET" != "all" ]]; then
            echo "[FATAL] VARIANT_SET must be old, new or all, got '$VARIANT_SET'" >&2
            exit 2
        fi
    elif [[ "$STAGE" == "ema" ]]; then
        # EMA target-network knobs. Do not reorder: indices are on disk.
        for fz in $EMA_FREEZE; do
            FZ_ARG=""; [[ "$fz" != "0" ]] && FZ_ARG=" --encoder_freeze_epochs $fz"
            for arm in $EMA_ARMS; do
                ema_arm_ok "$arm"
                # encoder_lr uses the shared ladder; the momentum axis is this
                # stage's own.
                for elr in $LRS; do
                    for m in 0.99 0.999 0.9995 0.9999; do
                        GRID+=("$arm|--encoder_lr $elr --ema_momentum $m$FZ_ARG")
                    done
                done
            done
        done
    elif [[ "$STAGE" == "ema_pred" ]]; then
        # Predictor knobs under a trained EMA encoder; the target knobs stay at
        # ENC_HP. Do not reorder: indices are on disk.
        for arm in $EMA_ARMS; do
            ema_arm_ok "$arm"
            for lr in $LRS; do
                for mf in 0 0.05 0.15 0.30; do
                    for cw in 0 2.0; do
                        GRID+=("$arm|--lr $lr --mask_frac $mf --change_weight $cw $ENC_HP")
                    done
                done
            done
        done
    elif [[ "$STAGE" == "baselines" ]]; then
        # LR plus each method's most load-bearing knob.
        for lr in $LRS; do
            for v in 2 4; do GRID+=("cldg|--lr $lr --views $v"); done
        done
        for lr in $LRS; do
            for g in 2 4 8; do GRID+=("ddgcl|--lr $lr --max_gap $g"); done
        done
        for lr in $LRS; do
            for pl in 1 2; do GRID+=("idol|--lr $lr --proj_layers $pl"); done
        done
        # MaskDGNN: LR x mask ratio.
        for lr in $LRS; do
            for mr in 0.1 0.2 0.4; do GRID+=("maskdgnn|--lr $lr --mask_ratio $mr"); done
        done
        # DySubC: LR x alpha (sampling trade-off between recency and degree).
        for lr in $LRS; do
            for a in 1 5 10; do GRID+=("dysubc|--lr $lr --alpha $a"); done
        done
        # DyG2Vec: LR x distortion strength; both dropouts move together.
        for lr in $LRS; do
            for pd in 0.15 0.3; do GRID+=("dyg2vec|--lr $lr --p_edge $pd --p_struct $pd"); done
        done
        # DyG-WM: LR x EMA momentum (the paper's node-task and link values).
        for lr in $LRS; do
            for m in 0.995 0.999; do GRID+=("dygwm|--lr $lr --ema_momentum $m"); done
        done
        # SG-JEPA: LR x InfoNCE temperature (the paper names no temperature).
        for lr in $LRS; do
            for t in 0.1 0.5; do GRID+=("sgjepa|--lr $lr --tau $t"); done
        done
        # GRACE: LR x temperature; the drop rates stay at the Cora defaults.
        for lr in $LRS; do
            for t in 0.2 0.4 0.7; do GRID+=("grace|--lr $lr --tau $t"); done
        done
        # GraphMAE: LR x mask rate.
        for lr in $LRS; do
            for mr in 0.3 0.5 0.75; do GRID+=("graphmae|--lr $lr --mask_rate $mr"); done
        done
        # BGRL: LR x initial EMA momentum.
        for lr in $LRS; do
            for m in 0.99 0.999; do GRID+=("bgrl|--lr $lr --ema_momentum $m"); done
        done
    else
        echo "[FATAL] STAGE must be 'predictor', 'variants', 'ema', 'ema_pred' or 'baselines', " \
             "got '$STAGE'" >&2
        exit 2
    fi
}

build_grid

if [[ "${1:-}" == "--list" ]]; then
    echo "DATASET=$DATASET STAGE=$STAGE VARIANT_SET=$VARIANT_SET EMA_ARMS='$EMA_ARMS'  grid size=${#GRID[@]}"
    echo "  ->  sbatch --array=0-$(( ${#GRID[@]} - 1 ))%6 $0"
    for i in "${!GRID[@]}"; do printf "  [%2d] %s\n" "$i" "${GRID[$i]}"; done
    # Exact --array= index list per arm (arms interleave in some stages, so a
    # first-last range would be wrong).
    declare -A _idx; _order=()
    for i in "${!GRID[@]}"; do
        a="${GRID[$i]%%|*}"
        [[ -v _idx[$a] ]] || _order+=("$a")
        _idx[$a]="${_idx[$a]:+${_idx[$a]} }$i"
    done
    echo "  per-arm --array= lists (or set ONLY_ARMS='<arm> <arm>' with the full array):"
    for a in "${_order[@]}"; do
        out=""; lo=""; prev=""
        for i in ${_idx[$a]}; do
            if [[ -z "$lo" ]]; then lo=$i; prev=$i
            elif (( i == prev + 1 )); then prev=$i
            else out+="${out:+,}$([[ $lo == $prev ]] && echo "$lo" || echo "$lo-$prev")"; lo=$i; prev=$i
            fi
        done
        out+="${out:+,}$([[ $lo == $prev ]] && echo "$lo" || echo "$lo-$prev")"
        printf "    %-14s %s\n" "$a" "$out"
    done
    exit 0
fi

mkdir -p logs
REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"
# Full-N adjacency tensors are multi-GB; expandable segments stop the caching
# allocator from stranding reserved-but-unusable blocks between them.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# Interpreter, in priority order: PY=/path/to/python (used as is),
# CONDA_ENV=<name|/path> (activated), otherwise ~/.conda/envs/cjepa if it exists.
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
# ONLY_ARMS="maskdgnn dyg2vec" runs just those arms' cells and skips the rest
# of the array as no-ops, so indices stay the same as in --list.
if [[ -n "${ONLY_ARMS:-}" && " $ONLY_ARMS " != *" $ARM "* ]]; then
    echo "[skip] cell $IDX is arm '$ARM', not in ONLY_ARMS='$ONLY_ARMS'"; exit 0
fi

CSV="dataset/jodie/${DATASET}.csv"
case "$DATASET" in
    mooc|reddit|wikipedia)
        [[ -f "$CSV" ]] || { echo "[FATAL] $CSV missing; fetch it from the login node" >&2; exit 10; } ;;
    *) ;;   # synthetic_dyn needs no file; cldg_* is checked by _dataset_config.sh
esac

TASK=state_probe
source "$REPO_ROOT/scripts/dblp/_dataset_config.sh"
BATCH="${BATCH:-8}"
NW="${SLURM_CPUS_PER_TASK:-8}"
SHORT="${DATASET#jodie_}"

# Sourcing _dataset_config.sh can clobber loop variables, so the arm flags are
# derived after it.
arm_flags() {
    case "$1" in
        frozen)      echo "" ;;
        naive)       echo "--train_encoder" ;;
        ema)         echo "--train_encoder --ema_target" ;;
        vicreg)      echo "--train_encoder --vicreg_cov 1.0" ;;
        input)       echo "--predict_space input" ;;
        # JEPA-recipe variants; every flag is opt-in in train_delos.py.
        ctx)         echo "--mask_encoder_input" ;;
        tnorm)       echo "--target_norm" ;;
        ema_cos)     echo "--train_encoder --ema_target --ema_schedule cosine $ANNEAL_ARG" ;;
        # EMA + target LayerNorm without context masking.
        ema_tnorm)     echo "--train_encoder --ema_target --target_norm" ;;
        ema_cos_tnorm) echo "--train_encoder --ema_target --ema_schedule cosine $ANNEAL_ARG --target_norm" ;;
        sigreg)      echo "--train_encoder --sigreg 1.0" ;;
        lejepa)      echo "--train_encoder --target_grad --sigreg 1.0" ;;
        ijepa)       echo "--train_encoder --ema_target --ema_schedule cosine $ANNEAL_ARG --mask_encoder_input --target_norm" ;;
        lejepa_full) echo "--train_encoder --target_grad --sigreg 1.0 --mask_encoder_input --target_norm" ;;
        *)           echo "__UNKNOWN__" ;;
    esac
}

SLUG=$(echo "$HP_ARGS" | sed 's/--//g; s/[ .]/_/g; s/__*/_/g; s/^_//; s/_$//')
[[ -z "$SLUG" ]] && SLUG=base
# STAGE=ema shares the variants root (encoder reuse + joint ranking, see header).
ROOT_STAGE="$STAGE"; [[ "$STAGE" == "ema" ]] && ROOT_STAGE=variants
ROOT="${ROOT:-outputs/cls_hparam/${SHORT}_${ROOT_STAGE}}"
OUT="$ROOT/${ARM}_${SLUG}_s${SEED}"
mkdir -p "$OUT"
printf '%s\n' "arm=$ARM flags=$HP_ARGS base=$BASE_HP" > "$OUT/hp_args.txt"
echo "[cell $IDX] dataset=$DATASET stage=$STAGE arm=$ARM seed=$SEED -> $OUT"

# Probe task per dataset, as in slurm_all_methods.sh. Rank non-binary cells
# with aggregate_hparam.py --metric acc_test (multiclass) or ndcg10_test (tgbn_*).
PROBE_TASK=binary; PROBE_LABELS=""
[[ "$DATASET" == "synthetic_dyn" && "${SYN_STATE_TARGET:-community}" != "switch" ]] && PROBE_TASK=multiclass
[[ "$DATASET" == cldg_* ]] && { PROBE_TASK=multiclass; PROBE_LABELS="--label_source dataset"; }
[[ "$DATASET" == "chase_music" ]] && PROBE_TASK=multiclass
[[ "$DATASET" == tgbn_* ]] && PROBE_TASK=ranking
PROBE_COMMON="--task $PROBE_TASK $PROBE_LABELS --state_label_mode $STATE_MODE --probe_split val \
    --device cuda --batch_size ${PROBE_BATCH:-4} --num_workers $NW"

# --- STAGE=baselines: one competing method per cell, probed on val -----------
if [[ "$STAGE" == "baselines" ]]; then
    # A cell is complete once state_probe_val.json exists (the last file it
    # writes), so resubmitting the array re-runs only the missing cells.
    if [[ -f "$OUT/state_probe_val.json" ]]; then
        echo "[skip] cell $IDX ($ARM) complete: $OUT/state_probe_val.json exists"; exit 0
    fi
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
      dyg2vec)
        srun --export=ALL "$PY" -m src.baselines.dyg2vec.train_dyg2vec $DATA_ARG \
            $ENCODER_ARG --encoder_kind dyg2vec $WINDOW_ARG $EARLY_STOP_ARG \
            --lam 25 --mu 25 --nu 1 --steps_per_epoch 20 --val_steps 10 \
            --epochs "${BASE_EPOCHS:-100}" --device cuda --seed "$SEED" \
            --out_dir "$OUT" $HP_ARGS \
            || { echo "[FATAL] DyG2Vec cell failed" >&2; exit 29; } ;;
      dygwm)
        srun --export=ALL "$PY" -m src.baselines.dygwm.train_dygwm $DATA_ARG \
            $ENCODER_ARG --encoder_kind dygwm $WINDOW_ARG $EARLY_STOP_ARG \
            --n_ctx 32 --decay 0.8 --cn_bonus 1.0 \
            --lam_node 1 --lam_rel 1 --lam_var 0.25 --lam_cov 0.025 \
            --node_batch_size 256 --pair_batch_size 256 \
            --steps_per_epoch 20 --val_steps 10 \
            --epochs "${BASE_EPOCHS:-100}" --device cuda --seed "$SEED" \
            --out_dir "$OUT" $HP_ARGS \
            || { echo "[FATAL] DyG-WM cell failed" >&2; exit 31; } ;;
      sgjepa)
        srun --export=ALL "$PY" -m src.baselines.sgjepa.train_sgjepa $DATA_ARG \
            $ENCODER_ARG $WINDOW_ARG $EARLY_STOP_ARG \
            --snn_layers 1 --init_tau 2.0 --v_th 1.0 \
            --node_batch_size 256 --steps_per_epoch 20 --val_steps 10 \
            --epochs "${BASE_EPOCHS:-100}" --device cuda --seed "$SEED" \
            --out_dir "$OUT" $HP_ARGS \
            || { echo "[FATAL] SG-JEPA cell failed" >&2; exit 32; } ;;
      bgrl)
        srun --export=ALL "$PY" -m src.baselines.bgrl.train_bgrl $DATA_ARG \
            $ENCODER_ARG $WINDOW_ARG $EARLY_STOP_ARG \
            --p_e1 0.2 --p_e2 0.3 --p_f1 0.2 --p_f2 0.1 \
            --steps_per_epoch 20 --val_steps 10 \
            --epochs "${BASE_EPOCHS:-100}" --device cuda --seed "$SEED" \
            --out_dir "$OUT" $ANNEAL_ARG $HP_ARGS \
            || { echo "[FATAL] BGRL cell failed" >&2; exit 30; } ;;
      graphmae)
        srun --export=ALL "$PY" -m src.baselines.graphmae.train_graphmae $DATA_ARG \
            $ENCODER_ARG $WINDOW_ARG $EARLY_STOP_ARG \
            --replace_rate 0.05 --sce_gamma 2 --target all \
            --steps_per_epoch 20 --val_steps 10 \
            --epochs "${BASE_EPOCHS:-100}" --device cuda --seed "$SEED" \
            --out_dir "$OUT" $HP_ARGS \
            || { echo "[FATAL] GraphMAE cell failed" >&2; exit 29; } ;;
      grace)
        srun --export=ALL "$PY" -m src.baselines.grace.train_grace $DATA_ARG \
            $ENCODER_ARG $WINDOW_ARG $EARLY_STOP_ARG \
            --p_e1 0.2 --p_e2 0.4 --p_f1 0.3 --p_f2 0.4 \
            --node_batch_size 256 --steps_per_epoch 20 --val_steps 10 \
            --epochs "${BASE_EPOCHS:-100}" --device cuda --seed "$SEED" \
            --out_dir "$OUT" $HP_ARGS \
            || { echo "[FATAL] GRACE cell failed" >&2; exit 28; } ;;
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
# Every cell starts from this encoder, so the sweep does not measure
# pretraining variance. flock stops concurrent cells from pretraining it at once.
ENC_DIR="$ROOT/_shared_encoder_s${SEED}"
ENC_CKPT="$ENC_DIR/encoder_${DATASET}_${ENC_KIND}_edge_mask.ckpt"
mkdir -p "$ENC_DIR"
exec 9>"$ENC_DIR/.pretrain.lock"
flock -w 21600 9 || { echo "[FATAL] encoder lock timeout" >&2; exit 10; }
# Only the `.done` marker (touched after a clean exit) proves the pretrain
# finished; ENC_TRUST=1 accepts a checkpoint without it.
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

# --- train, then probe the val split ----------------------------------------
if [[ -f "$OUT/state_probe_val.json" ]]; then
    echo "[skip] cell $IDX complete: $OUT/state_probe_val.json exists"; exit 0
fi
# best.ckpt with test_metrics.json means training finished and only the probe
# is missing (if it ran out of memory, rerun with a smaller PROBE_BATCH).
# best.ckpt alone means the job was killed mid-training: restart from scratch.
if [[ -f "$OUT/best.ckpt" && -f "$OUT/test_metrics.json" ]]; then
    echo "[skip] cell $IDX training complete (best.ckpt + test_metrics.json); running only the probe"
else
    if [[ -f "$OUT/best.ckpt" ]]; then
        echo "[resume] cell $IDX has best.ckpt but no test_metrics.json (interrupted) -> restarting from scratch"
        rm -f "$OUT/best.ckpt" "$OUT/history.json"
    fi
    srun --export=ALL "$PY" src/train/train_delos.py \
        $DATA_ARG $ENCODER_ARG $WINDOW_ARG $EARLY_STOP_ARG \
        --encoder_ckpt "$ENC_CKPT" --mask_scheme "${MASK_SCHEME:-node_mask}" \
        --residual_head \
        --heads "${HEADS:-4}" --mlp_dim "${MLP_DIM:-512}" \
        --batch_size "$BATCH" --epochs "${JEPA_EPOCHS:-300}" \
        --probe_epochs "${JEPA_PROBE_EPOCHS:-3}" \
        --downstream_task "${DOWNSTREAM:-none}" \
        --seed "$SEED" --num_workers "$NW" --device cuda --out_dir "$OUT" \
        $BASE_HP $VFLAGS $HP_ARGS
    [[ -f "$OUT/best.ckpt" ]] || { echo "[FATAL] cell $IDX produced no best.ckpt" >&2; exit 12; }
fi

srun --export=ALL "$PY" scripts/dblp/run_block_probe.py \
    --jepa_ckpt "$OUT/best.ckpt" $PROBE_COMMON \
    || { echo "[FATAL] probe failed" >&2; exit 13; }
[[ -f "$OUT/state_probe_val.json" ]] || { echo "[FATAL] cell $IDX wrote no metrics" >&2; exit 12; }

echo "[done] cell $IDX -> $OUT"
