#!/usr/bin/env bash
#SBATCH --job-name=reg-hparam
#SBATCH --output=logs/reg-hparam-%A_%a.out
#SBATCH --error=logs/reg-hparam-%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=20:00:00
#SBATCH --partition=gpu
#
# HYPERPARAMETER SEARCH for node-signal forecasting. A SLURM array job: each
# task id decodes to one grid cell, each cell writes its own output dir, and
# aggregate_hparam.py ranks them.
#
# PREREQUISITE on METR-LA: run scripts/regression/slurm_lr_probe.sh first and
# pass LRS="..." from its output (it replaces the shared default grid for EVERY
# arm, so use the same override for all stages). At lr=5e-4 the predictor's
# TRAIN loss rises from epoch 3 with a frozen
# encoder (static targets), i.e. the optimiser diverges. Searching a grid whose
# every cell diverges ranks "which LR diverges least" and poisons everything
# downstream.
#
# COST: at batch 64 a METR-LA step is ~5.2 s on a V100 and stride-1 gives 375
# steps/epoch, so ONE cell at 50 epochs is ~27 h. Budget deliberately:
#   STRIDE   windows are stride-1 and overlap by 23/24, so consecutive
#            minibatches are near-duplicates. STRIDE=12 gives non-overlapping
#            history: 12x fewer steps per epoch, and each step sees fresher
#            data. It is a smaller TRAINING BUDGET, not a free speedup -- at a
#            fixed epoch count you get 12x fewer gradient steps. Fine for
#            ranking, never for the final table (which must stay stride 1).
#   JEPA_EPOCHS  set it from the probe's val_argmin: no point training far past
#            where a stable run stops improving.
# Check the arithmetic before submitting 36 of these:
#   JEPA_EPOCHS=50 STRIDE=12 -> 32 steps/epoch -> ~2.8 min/epoch -> ~2.5 h/cell
#
# EVERY cell probes the VALIDATION split (--probe_split val). The test split is
# never read during the search, so the configuration you carry into the final
# 3-seed table was not chosen on the numbers that table reports.
#
# Two stages, because they tune different things:
#
#   STAGE=predictor  (do this first)
#       Sweeps the predictor knobs on the FROZEN arm -- lr x num_masked_nodes x
#       depth x change_weight. The encoder is fixed, so these cells differ only
#       in the predictor. The winner becomes BASE_HP for everything downstream.
#
#   STAGE=baselines  (independent of the other two)
#       Tunes the COMPETING methods -- supervised / GRACE / CLDG / DDGCL / IDOL /
#       STD-MAE -- so the final table is not "our tuned method vs their
#       defaults". Each cell trains one baseline and probes VAL through the
#       same probe and metrics as every other arm. Cheap: none of these run the
#       T*N-token predictor.
#
#   STAGE=variants   (after predictor)
#       Holds BASE_HP fixed and sweeps only the encoder-training knob that each
#       variant actually has. `frozen` has none left, so it contributes a single
#       reference cell; unfreezing makes the encoder LR dominant (too high and
#       the targets move faster than the predictor can track).
#
# Usage:
#   DATASET=metr_la STAGE=predictor bash scripts/regression/slurm_hparam_search.sh --list
#   DATASET=metr_la STAGE=predictor sbatch --array=0-35%6 scripts/regression/slurm_hparam_search.sh
#   python scripts/regression/aggregate_hparam.py --root outputs/reg_hparam/metr_la_predictor
#
#   DATASET=metr_la STAGE=variants BASE_HP="--lr 1e-3 --num_masked_nodes 30 --change_weight 2.0" \
#       sbatch --array=0-15%4 scripts/regression/slurm_hparam_search.sh
#   python scripts/regression/aggregate_hparam.py --root outputs/reg_hparam/metr_la_variants --by_arm
#
#   DATASET=metr_la STAGE=baselines bash scripts/regression/slurm_hparam_search.sh --list
#   DATASET=metr_la STAGE=baselines sbatch --array=0-89%8 scripts/regression/slurm_hparam_search.sh
#   python scripts/regression/aggregate_hparam.py --root outputs/reg_hparam/metr_la_baselines \
#       --by_arm --emit_flags        # -> BASELINE_HP_cldg="..." etc.

set -uo pipefail

DATASET="${DATASET:-england_covid}"
STAGE="${STAGE:-predictor}"
SEED="${SEED:-42}"
# Cosine-anneal horizon for every EMA-target arm that anneals its momentum
# (DyG-JEPA ema_cos / ijepa / ema_cos_tnorm and the BGRL baseline). The
# trainers' default spans the whole --epochs budget, which early stopping
# (patience 20) rarely lets a run reach: on MOOC the ema_cos cells stopped at
# 85-240 of 300 epochs with the momentum still at ~0.997, i.e. the cosine arm
# was the constant arm. 50 epochs is below every observed run length, so the
# schedule finishes while the predictor still trains. 0 = whole run (legacy).
# Same value for every arm and method so no row gets its own schedule.
EMA_ANNEAL_EPOCHS="${EMA_ANNEAL_EPOCHS:-50}"
ANNEAL_ARG=""; [[ "$EMA_ANNEAL_EPOCHS" != "0" ]] && ANNEAL_ARG="--ema_anneal_epochs $EMA_ANNEAL_EPOCHS"
# One seed for the search: the ranking only needs to be consistent, and seeds
# are far better spent on the 3-seed final table than on the sweep.
BASE_HP="${BASE_HP:-}"
# ONE learning-rate grid shared by every arm that sweeps --lr: all baselines and
# DyG-JEPA's own predictor / ema_pred stages. Per-method grids centred on each
# paper's default put several winners on a grid EDGE (CLDG lowest, DDGCL / IDOL
# highest on MOOC) and made "who was tuned harder" depend on the method. A decade
# ladder spans every reference default (1e-4 DyG2Vec ... 5e-3 STEP / STD-MAE).
# Cell dirs are named by flag values, not index, so cells already on disk at
# 1e-3 / 1e-2 are reused. Override: LRS="3e-5 1e-4 3e-4" (e.g. METR-LA, from
# slurm_lr_probe.sh) -- but then use the SAME override for every stage.
LRS="${LRS:-1e-4 1e-3 1e-2}"

# ---------------------------------------------------------------------------
# Grid definition. GRID entries are "arm|extra flags".
# ---------------------------------------------------------------------------
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
    elif [[ "$STAGE" == "baselines" ]]; then
        # LR plus each method's most load-bearing knob. The published defaults
        # were tuned on each paper's OWN datasets: CLDG ships lr=4e-3, which was
        # never tuned for METR-LA's 24k 5-minute snapshots, and we already know
        # an LR that is fine elsewhere DIVERGES on this data. Tuning DyG-JEPA
        # while the baselines sit at transferred defaults inflates our margin
        # for a reason unrelated to the method.
        # These are cheap: no T*N-token predictor, so cells are minutes.
        for lr in $LRS; do
            for elr in default 1e-4; do
                EXTRA=""; [[ "$elr" != "default" ]] && EXTRA="--encoder_lr $elr"
                GRID+=("supervised|--lr $lr $EXTRA")
            done
        done
        for lr in $LRS; do
            for v in 2 4; do GRID+=("cldg|--lr $lr --views $v"); done
        done
        for lr in $LRS; do
            for g in 2 4 8; do GRID+=("ddgcl|--lr $lr --max_gap $g"); done
        done
        for lr in $LRS; do
            for pl in 1 2; do GRID+=("idol|--lr $lr --proj_layers $pl"); done
        done
        # MaskDGNN: LR x mask ratio (the paper's most sensitive knob; best at
        # 10-30%, collapses past 40%).
        for lr in $LRS; do
            for mr in 0.1 0.2 0.4; do GRID+=("maskdgnn|--lr $lr --mask_ratio $mr"); done
        done
        # DySubC: LR x alpha (the sampling trade-off between recency and
        # degree; the paper's most sensitive knob after subgraph size).
        for lr in $LRS; do
            for a in 1 5 10; do GRID+=("dysubc|--lr $lr --alpha $a"); done
        done
        # DyG2Vec: LR x distortion strength (p_d, the only knob the paper
        # validates; 0.3 is its choice). Both dropouts move together.
        for lr in $LRS; do
            for pd in 0.15 0.3; do GRID+=("dyg2vec|--lr $lr --p_edge $pd --p_struct $pd"); done
        done
        # DyG-WM: LR x EMA momentum (the target encoder's own stability knob;
        # the paper uses 0.995 for node tasks and 0.999 for links).
        for lr in $LRS; do
            for m in 0.995 0.999; do GRID+=("dygwm|--lr $lr --ema_momentum $m"); done
        done
        # SG-JEPA: LR x InfoNCE temperature (the paper names no temperature).
        for lr in $LRS; do
            for t in 0.1 0.5; do GRID+=("sgjepa|--lr $lr --tau $t"); done
        done
        # GRACE (static control): LR x tau. The temperature is the paper's
        # load-bearing knob after the drop rates, which stay at the Cora defaults.
        for lr in $LRS; do
            for t in 0.2 0.4 0.7; do GRID+=("grace|--lr $lr --tau $t"); done
        done
        # GraphMAE (static control): LR x mask rate, the paper's headline knob
        # (robust 0.3-0.75 on citation graphs; lower on feature-poor inputs).
        for lr in $LRS; do
            for mr in 0.3 0.5 0.75; do GRID+=("graphmae|--lr $lr --mask_rate $mr"); done
        done
        # BGRL (static control): LR x initial EMA momentum, the bootstrap's own
        # stability knob.
        for lr in $LRS; do
            for m in 0.99 0.999; do GRID+=("bgrl|--lr $lr --ema_momentum $m"); done
        done
        # STEP: forecaster LR x --dgl_k, the discrete-graph-learning sparsity
        # that names the method. Both are forecaster-side, so all 9 cells share
        # ONE TSFormer pre-training.
        for lr in $LRS; do
            for k in 5 10 20; do GRID+=("step|--lr $lr --dgl_k $k"); done
        done
        # STD-MAE: forecaster LR x pre-training mask ratio. The two MAEs are
        # pre-trained ONCE per mask_ratio and shared across the LR cells, so
        # this costs 2 pre-trains, not 6.
        for lr in $LRS; do
            for mr in 0.25 0.5; do GRID+=("stdmae|--lr $lr --mask_ratio $mr"); done
        done
    else
        echo "[FATAL] STAGE must be 'predictor', 'variants' or 'baselines', " \
             "got '$STAGE'" >&2
        exit 2
    fi
}

build_grid

if [[ "${1:-}" == "--list" ]]; then
    echo "DATASET=$DATASET STAGE=$STAGE  grid size=${#GRID[@]}"
    echo "  ->  sbatch --array=0-$(( ${#GRID[@]} - 1 ))%6 $0"
    for i in "${!GRID[@]}"; do printf "  [%2d] %s\n" "$i" "${GRID[$i]}"; done
    exit 0
fi

mkdir -p logs
REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"
if [[ -d "$HOME/.conda/envs/cjepa" ]]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate cjepa
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

IDX="${SLURM_ARRAY_TASK_ID:-0}"
if (( IDX >= ${#GRID[@]} )); then
    echo "[error] array index $IDX >= grid size ${#GRID[@]}" >&2; exit 2
fi
CELL="${GRID[$IDX]}"
ARM="${CELL%%|*}"
HP_ARGS="${CELL#*|}"

TASK=regression
source "$REPO_ROOT/scripts/dblp/_dataset_config.sh"
BATCH="${BATCH:-$REG_BATCH}"
ENC_ARGS="$ENCODER_ARG --bidirectional false"
NW="${SLURM_CPUS_PER_TASK:-8}"

ATTN_FLAG=""
[[ "$DATASET" == "metr_la" ]] && ATTN_FLAG="--no_attention_mask"

# Search-only window stride. Deliberately NOT in _dataset_config.sh: that file
# is what the final table sources, and a stride leaking into it would silently
# change the evaluation windows for every arm.
STRIDE="${STRIDE:-1}"
STRIDE_ARG=""
[[ "$DATASET" == "metr_la" && "$STRIDE" != "1" ]] && STRIDE_ARG="--metr_la_stride $STRIDE"
echo "[config] STRIDE=$STRIDE JEPA_EPOCHS=${JEPA_EPOCHS:-300} BATCH=$BATCH"

# Sourcing _dataset_config.sh sets VARIANT/DATASET of its own, so re-derive the
# arm flags AFTER it and verify — a clobbered $ARM would mislabel a whole cell.
arm_flags() {
    case "$1" in
        frozen) echo "" ;;
        naive)  echo "--train_encoder" ;;
        ema)    echo "--train_encoder --ema_target" ;;
        vicreg) echo "--train_encoder --vicreg_cov 1.0" ;;
        *)      echo "__UNKNOWN__" ;;
    esac
}
if [[ "$STAGE" != "baselines" ]]; then
    VFLAGS="$(arm_flags "$ARM")"
    [[ "$VFLAGS" == "__UNKNOWN__" ]] && { echo "[FATAL] bad arm '$ARM'" >&2; exit 3; }
else
    VFLAGS=""
fi

SLUG=$(echo "$HP_ARGS" | sed 's/--//g; s/[ .]/_/g; s/__*/_/g; s/^_//; s/_$//')
[[ -z "$SLUG" ]] && SLUG=base
ROOT="outputs/reg_hparam/${DATASET}_${STAGE}"
OUT="$ROOT/${ARM}_${SLUG}_s${SEED}"
mkdir -p "$OUT"
printf '%s\n' "arm=$ARM flags=$VFLAGS $HP_ARGS base=$BASE_HP" > "$OUT/hp_args.txt"
echo "[cell $IDX] dataset=$DATASET stage=$STAGE arm=$ARM seed=$SEED"
echo "[cell $IDX] flags='$VFLAGS $HP_ARGS' base='$BASE_HP' -> $OUT"

# --- STAGE=baselines: one competing method per cell, probed on VAL -----------
# Each baseline gets the SAME probe, metrics and val split as every other arm,
# so aggregate_hparam.py ranks them on the identical basis it ranks DyG-JEPA.
if [[ "$STAGE" == "baselines" ]]; then
    # A cell is complete once its val metrics exist (test_metrics.json is the last
    # thing a cell writes), so resubmitting the array -- after a walltime kill or
    # a grid change -- re-runs only the missing cells. Delete the dir to force one.
    if [[ -f "$OUT/test_metrics.json" ]]; then
        echo "[skip] cell $IDX ($ARM) complete: $OUT/test_metrics.json exists"; exit 0
    fi
    PROBE_ARGS="$DATA_ARG $WINDOW_ARG --probe_epochs ${PROBE_EPOCHS:-10} \
        --probe_hidden 64 --persist_residual --probe_split val \
        --batch_size $BATCH --num_workers $NW --seed $SEED --device cuda \
        $EARLY_STOP_ARG"
    SSL_COMMON="$DATA_ARG $ENC_ARGS $WINDOW_ARG $EARLY_STOP_ARG $STRIDE_ARG \
        --node_batch_size 128 --epochs ${SSL_EPOCHS:-100} \
        --device cuda --seed $SEED --out_dir $OUT"

    case "$ARM" in
      supervised)
        # HP_ARGS carries --lr and optionally --encoder_lr; both are this
        # script's own flags on probe_encoder_regression.
        srun --export=ALL python scripts/regression/probe_encoder_regression.py \
            --random_init --train_encoder $ENC_ARGS $PROBE_ARGS $STRIDE_ARG \
            --probe_epochs "${SUP_EPOCHS:-100}" $HP_ARGS --out_dir "$OUT" \
            || { echo "[FATAL] supervised cell failed" >&2; exit 20; }
        ;;
      cldg)
        srun --export=ALL python -m src.baselines.cldg.train_cldg $SSL_COMMON \
            --view_length "$HIST" --strategy random --tau 0.07 \
            --steps_per_epoch 20 --val_steps 10 $HP_ARGS \
            || { echo "[FATAL] CLDG cell failed" >&2; exit 21; }
        srun --export=ALL python scripts/regression/probe_encoder_regression.py \
            --encoder_ckpt "$OUT/encoder.ckpt" $PROBE_ARGS $STRIDE_ARG \
            --out_dir "$OUT" || { echo "[FATAL] CLDG probe failed" >&2; exit 21; }
        ;;
      ddgcl)
        srun --export=ALL python -m src.baselines.ddgcl.train_ddgcl $SSL_COMMON \
            --time_dim 32 --tau 0.01 --beta 1.5 \
            --steps_per_epoch 20 --val_steps 10 $HP_ARGS \
            || { echo "[FATAL] DDGCL cell failed" >&2; exit 22; }
        srun --export=ALL python scripts/regression/probe_encoder_regression.py \
            --encoder_ckpt "$OUT/encoder.ckpt" $PROBE_ARGS $STRIDE_ARG \
            --out_dir "$OUT" || { echo "[FATAL] DDGCL probe failed" >&2; exit 22; }
        ;;
      idol)
        srun --export=ALL python -m src.baselines.idol.train_idol $SSL_COMMON \
            --steps_per_epoch 20 --val_steps 10 $HP_ARGS || { echo "[FATAL] IDOL cell failed" >&2; exit 23; }
        srun --export=ALL python scripts/regression/probe_encoder_regression.py \
            --encoder_ckpt "$OUT/encoder.ckpt" $PROBE_ARGS $STRIDE_ARG \
            --out_dir "$OUT" || { echo "[FATAL] IDOL probe failed" >&2; exit 23; }
        ;;
      maskdgnn)
        # Not $SSL_COMMON: that carries --node_batch_size, which this trainer
        # (window-based, not node-batched) does not define.
        srun --export=ALL python -m src.baselines.maskdgnn.train_maskdgnn $DATA_ARG \
            $ENC_ARGS --encoder_kind maskdgnn $WINDOW_ARG $EARLY_STOP_ARG $STRIDE_ARG \
            --alpha 0.7 --target next --neg_per_pos 1 \
            --steps_per_epoch 20 --val_steps 10 --e2e_eval none \
            --epochs "${SSL_EPOCHS:-100}" --device cuda --seed "$SEED" \
            --out_dir "$OUT" $HP_ARGS \
            || { echo "[FATAL] MaskDGNN cell failed" >&2; exit 26; }
        srun --export=ALL python scripts/regression/probe_encoder_regression.py \
            --encoder_ckpt "$OUT/encoder.ckpt" $PROBE_ARGS $STRIDE_ARG \
            --out_dir "$OUT" || { echo "[FATAL] MaskDGNN probe failed" >&2; exit 26; }
        ;;
      dysubc)
        srun --export=ALL python -m src.baselines.dysubc.train_dysubc $SSL_COMMON \
            --subgraph_size 20 --beta 1 --margin 0.75 --lam 0.5 \
            --steps_per_epoch 20 --val_steps 10 $HP_ARGS \
            || { echo "[FATAL] DySubC cell failed" >&2; exit 27; }
        srun --export=ALL python scripts/regression/probe_encoder_regression.py \
            --encoder_ckpt "$OUT/encoder.ckpt" $PROBE_ARGS $STRIDE_ARG \
            --out_dir "$OUT" || { echo "[FATAL] DySubC probe failed" >&2; exit 27; }
        ;;
      dyg2vec)
        # Not $SSL_COMMON: that carries --node_batch_size, which this
        # window-based trainer does not define.
        srun --export=ALL python -m src.baselines.dyg2vec.train_dyg2vec $DATA_ARG \
            $ENC_ARGS --encoder_kind dyg2vec $WINDOW_ARG $EARLY_STOP_ARG $STRIDE_ARG \
            --lam 25 --mu 25 --nu 1 --steps_per_epoch 20 --val_steps 10 \
            --epochs "${SSL_EPOCHS:-100}" --device cuda --seed "$SEED" \
            --out_dir "$OUT" $HP_ARGS \
            || { echo "[FATAL] DyG2Vec cell failed" >&2; exit 29; }
        srun --export=ALL python scripts/regression/probe_encoder_regression.py \
            --encoder_ckpt "$OUT/encoder.ckpt" $PROBE_ARGS $STRIDE_ARG \
            --out_dir "$OUT" || { echo "[FATAL] DyG2Vec probe failed" >&2; exit 29; }
        ;;
      dygwm)
        # Not $SSL_COMMON's --node_batch_size default: both this trainer and
        # SG-JEPA define it, so $SSL_COMMON is fine here.
        srun --export=ALL python -m src.baselines.dygwm.train_dygwm $SSL_COMMON \
            --encoder_kind dygwm --n_ctx 32 --decay 0.8 --cn_bonus 1.0 \
            --lam_node 1 --lam_rel 1 --lam_var 0.25 --lam_cov 0.025 \
            --pair_batch_size 128 --steps_per_epoch 20 --val_steps 10 $HP_ARGS \
            || { echo "[FATAL] DyG-WM cell failed" >&2; exit 31; }
        srun --export=ALL python scripts/regression/probe_encoder_regression.py \
            --encoder_ckpt "$OUT/encoder.ckpt" $PROBE_ARGS $STRIDE_ARG \
            --out_dir "$OUT" || { echo "[FATAL] DyG-WM probe failed" >&2; exit 31; }
        ;;
      sgjepa)
        srun --export=ALL python -m src.baselines.sgjepa.train_sgjepa $SSL_COMMON \
            --snn_layers 1 --init_tau 2.0 --v_th 1.0 \
            --steps_per_epoch 20 --val_steps 10 $HP_ARGS \
            || { echo "[FATAL] SG-JEPA cell failed" >&2; exit 32; }
        srun --export=ALL python scripts/regression/probe_encoder_regression.py \
            --encoder_ckpt "$OUT/encoder.ckpt" $PROBE_ARGS $STRIDE_ARG \
            --out_dir "$OUT" || { echo "[FATAL] SG-JEPA probe failed" >&2; exit 32; }
        ;;
      bgrl)
        # Not $SSL_COMMON: that carries --node_batch_size, which this trainer
        # (per-node loss over every active node, no negatives) does not define.
        srun --export=ALL python -m src.baselines.bgrl.train_bgrl $DATA_ARG \
            $ENC_ARGS $WINDOW_ARG $EARLY_STOP_ARG $STRIDE_ARG \
            --p_e1 0.2 --p_e2 0.3 --p_f1 0.2 --p_f2 0.1 \
            --steps_per_epoch 20 --val_steps 10 \
            --epochs "${SSL_EPOCHS:-100}" --device cuda --seed "$SEED" \
            --out_dir "$OUT" $ANNEAL_ARG $HP_ARGS \
            || { echo "[FATAL] BGRL cell failed" >&2; exit 30; }
        srun --export=ALL python scripts/regression/probe_encoder_regression.py \
            --encoder_ckpt "$OUT/encoder.ckpt" $PROBE_ARGS $STRIDE_ARG \
            --out_dir "$OUT" || { echo "[FATAL] BGRL probe failed" >&2; exit 30; }
        ;;
      graphmae)
        # Not $SSL_COMMON: that carries --node_batch_size, which this trainer
        # (whole-snapshot masking, not node-batched) does not define.
        srun --export=ALL python -m src.baselines.graphmae.train_graphmae $DATA_ARG \
            $ENC_ARGS $WINDOW_ARG $EARLY_STOP_ARG $STRIDE_ARG \
            --replace_rate 0.05 --sce_gamma 2 --target all \
            --steps_per_epoch 20 --val_steps 10 \
            --epochs "${SSL_EPOCHS:-100}" --device cuda --seed "$SEED" \
            --out_dir "$OUT" $HP_ARGS \
            || { echo "[FATAL] GraphMAE cell failed" >&2; exit 29; }
        srun --export=ALL python scripts/regression/probe_encoder_regression.py \
            --encoder_ckpt "$OUT/encoder.ckpt" $PROBE_ARGS $STRIDE_ARG \
            --out_dir "$OUT" || { echo "[FATAL] GraphMAE probe failed" >&2; exit 29; }
        ;;
      grace)
        srun --export=ALL python -m src.baselines.grace.train_grace $SSL_COMMON \
            --p_e1 0.2 --p_e2 0.4 --p_f1 0.3 --p_f2 0.4 \
            --steps_per_epoch 20 --val_steps 10 $HP_ARGS \
            || { echo "[FATAL] GRACE cell failed" >&2; exit 28; }
        srun --export=ALL python scripts/regression/probe_encoder_regression.py \
            --encoder_ckpt "$OUT/encoder.ckpt" $PROBE_ARGS $STRIDE_ARG \
            --out_dir "$OUT" || { echo "[FATAL] GRACE probe failed" >&2; exit 28; }
        ;;
      stdmae)
        source "$REPO_ROOT/scripts/regression/_stdmae_config.sh"
        # Split HP_ARGS: --mask_ratio belongs to the (shared) pre-training,
        # --lr to the forecaster. Cells with the same mask_ratio reuse ONE
        # pre-train, so this grid costs 2 pre-trains rather than 6.
        MR=$(echo "$HP_ARGS" | sed -n 's/.*--mask_ratio \([0-9.]*\).*/\1/p')
        FLR=$(echo "$HP_ARGS" | sed -n 's/.*--lr \([0-9eE.+-]*\).*/\1/p')
        SM_PRE="$ROOT/_stdmae_pretrain_mr${MR}_s${SEED}"
        mkdir -p "$SM_PRE"
        exec 8>"$SM_PRE/.pretrain.lock"
        flock -w 21600 8 || { echo "[FATAL] stdmae lock timeout" >&2; exit 24; }
        for V in temporal spatial; do
            case "$V" in temporal) CK=tmae ;; spatial) CK=smae ;; esac
            [[ -f "$SM_PRE/$CK.ckpt" ]] && { echo "[skip] $CK exists"; continue; }
            srun --export=ALL python -m src.baselines.stdmae.pretrain_stdmae \
                --variant "$V" $DATA_ARG $WINDOW_ARG $EARLY_STOP_ARG \
                --long_history "$SM_LONG" --patch_size "$SM_PATCH" \
                --mask_ratio "$MR" --window_stride "$SM_STRIDE" \
                --batch_size "$SM_BATCH" --epochs "${SM_PRE_EPOCHS:-200}" \
                --lr "${SM_PRE_LR:-5e-4}" --seed "$SEED" --num_workers "$NW" \
                --device cuda --out_dir "$SM_PRE"
            if [[ ! -f "$SM_PRE/$CK.ckpt" ]]; then
                flock -u 8; echo "[FATAL] stdmae $CK pretrain failed" >&2; exit 24
            fi
        done
        flock -u 8; exec 8>&-
        srun --export=ALL python -m src.baselines.stdmae.train_stdmae \
            $DATA_ARG $WINDOW_ARG $EARLY_STOP_ARG \
            --smae_ckpt "$SM_PRE/smae.ckpt" --tmae_ckpt "$SM_PRE/tmae.ckpt" \
            --batch_size "$SM_BATCH" --epochs "${SM_EPOCHS:-100}" --lr "$FLR" \
            --cl_epochs "$SM_CL" --probe_split val --seed "$SEED" \
            --num_workers "$NW" --device cuda --out_dir "$OUT" \
            || { echo "[FATAL] STD-MAE forecaster failed" >&2; exit 24; }
        ;;
      step)
        # Mirrors slurm_all_methods.sh's ST_* geometry. Kept in sync by hand;
        # if you change one, change the other.
        case "$DATASET" in
            metr_la) ST_LONG=${ST_LONG:-2016}; ST_PATCH=${ST_PATCH:-12}
                     ST_BATCH=${ST_BATCH:-32}; ST_CL=${ST_CL:-6}
                     ST_STRIDE=${ST_STRIDE:-12} ;;
            *)       ST_LONG=${ST_LONG:-$HIST}; ST_PATCH=${ST_PATCH:-2}
                     ST_BATCH=${ST_BATCH:-$BATCH}; ST_CL=${ST_CL:-0}
                     ST_STRIDE=${ST_STRIDE:-1} ;;
        esac
        # dgl_k is forecaster-side, so every cell reuses ONE pre-training.
        ST_PRE="$ROOT/_step_pretrain_s${SEED}"
        mkdir -p "$ST_PRE"
        exec 7>"$ST_PRE/.pretrain.lock"
        flock -w 21600 7 || { echo "[FATAL] step lock timeout" >&2; exit 25; }
        if [[ ! -f "$ST_PRE/tsformer.ckpt" ]]; then
            srun --export=ALL python -m src.baselines.step.pretrain_tsformer \
                $DATA_ARG $WINDOW_ARG $EARLY_STOP_ARG \
                --long_history "$ST_LONG" --patch_size "$ST_PATCH" \
                --window_stride "$ST_STRIDE" --batch_size "$ST_BATCH" \
                --epochs "${ST_PRE_EPOCHS:-100}" --lr "${ST_PRE_LR:-5e-4}" \
                --seed "$SEED" --num_workers "$NW" --device cuda --out_dir "$ST_PRE"
            if [[ ! -f "$ST_PRE/tsformer.ckpt" ]]; then
                flock -u 7; echo "[FATAL] TSFormer pre-train failed" >&2; exit 25
            fi
        else
            echo "[skip] tsformer.ckpt exists"
        fi
        flock -u 7; exec 7>&-
        srun --export=ALL python -m src.baselines.step.train_step \
            $DATA_ARG $WINDOW_ARG $EARLY_STOP_ARG \
            --tsformer_ckpt "$ST_PRE/tsformer.ckpt" \
            --batch_size "$ST_BATCH" --epochs "${ST_EPOCHS:-100}" \
            --cl_epochs "$ST_CL" --probe_split val $HP_ARGS \
            --seed "$SEED" --num_workers "$NW" --device cuda --out_dir "$OUT" \
            || { echo "[FATAL] STEP cell failed" >&2; exit 25; }
        ;;
      *) echo "[FATAL] unknown baseline arm '$ARM'" >&2; exit 3 ;;
    esac
    [[ -f "$OUT/test_metrics.json" ]] || { echo "[FATAL] cell $IDX wrote no metrics" >&2; exit 12; }
    echo "[done] cell $IDX ($ARM) -> $OUT"
    exit 0
fi

# --- shared pretrained encoder ----------------------------------------------
# Every cell starts from THIS encoder, so the sweep measures the predictor /
# encoder-training knobs and not pretraining variance. flock because concurrent
# array cells would otherwise all pretrain it at once.
ENC_DIR="$ROOT/_shared_encoder_s${SEED}"
ENC_CKPT="$ENC_DIR/encoder_${DATASET}_${ENC_KIND}_masked_signal.ckpt"
mkdir -p "$ENC_DIR"
exec 9>"$ENC_DIR/.pretrain.lock"
flock -w 7200 9 || { echo "[FATAL] encoder lock timeout" >&2; exit 10; }
if [[ ! -f "$ENC_CKPT" ]]; then
    echo "=== shared encoder pretrain (lock held by cell $IDX) ==="
    srun --export=ALL python src/train/pretrain_encoder.py $DATA_ARG $STRIDE_ARG \
        --pretext masked_signal $ENC_ARGS --mask_rate 0.25 --decoder_hidden 64 \
        $WINDOW_ARG $EARLY_STOP_ARG \
        --batch_size "$BATCH" --epochs "${ENC_EPOCHS:-300}" --lr 1e-3 \
        --seed "$SEED" --num_workers "$NW" --device cuda --out_dir "$ENC_DIR"
    if [[ ! -f "$ENC_CKPT" ]]; then
        flock -u 9; echo "[FATAL] encoder pretrain failed" >&2; exit 11
    fi
else
    echo "[skip] shared encoder exists"
fi
flock -u 9; exec 9>&-

# --- train + probe THE VAL SPLIT --------------------------------------------
srun --export=ALL python src/train/train_dyg_jepa.py $DATA_ARG $WINDOW_ARG \
    $EARLY_STOP_ARG $ENC_ARGS $ATTN_FLAG $STRIDE_ARG \
    --downstream_task regression --mask_scheme node_mask \
    --encoder_ckpt "$ENC_CKPT" \
    --residual_head --persist_residual \
    --heads "${HEADS:-8}" --mlp_dim "${MLP_DIM:-512}" \
    --batch_size "$BATCH" --epochs "${JEPA_EPOCHS:-300}" --weight_decay 1e-4 \
    --probe_epochs "${PROBE_EPOCHS:-10}" --probe_hidden 64 \
    --probe_split val \
    --seed "$SEED" --num_workers "$NW" --device cuda --out_dir "$OUT" \
    $BASE_HP $VFLAGS $HP_ARGS
[[ -f "$OUT/test_metrics.json" ]] || { echo "[FATAL] cell $IDX produced no metrics" >&2; exit 12; }

echo "[done] cell $IDX -> $OUT"
