#!/usr/bin/env bash
#SBATCH --job-name=lp-hparam
#SBATCH --output=logs/lp-hparam-%A_%a.out
#SBATCH --error=logs/lp-hparam-%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=20:00:00
#SBATCH --partition=gpu
#
# HYPERPARAMETER SEARCH for dynamic link prediction. Third of the three sweeps,
# identical in structure to scripts/regression/ and scripts/classification/.
#
# EVERY cell scores the VALIDATION split. The JEPA arms get --probe_split val
# (test_metrics.json from the val loader); the baselines get analyze_jepa
# --probe_split val, which writes analysis_val.json. The test split is never
# read during the search.
#
# Stages (same set, grids and cell indices as scripts/classification/):
#   STAGE=predictor  predictor knobs on the FROZEN arm -> BASE_HP
#   STAGE=baselines  GRACE / CLDG / DDGCL / IDOL / ..., so the table is not "our tuned method
#                    vs their defaults"
#   STAGE=variants   encoder-training knob per variant, BASE_HP fixed.
#                    VARIANT_SET picks the arms: old = frozen / ema / naive /
#                    vicreg (16 cells, indices 0-15 as before); new = the
#                    JEPA-recipe arms ctx / tnorm / ema_cos / sigreg / lejepa /
#                    ijepa / lejepa_full (32 cells); all = old then new (48,
#                    default). --list prints every cell.
#   STAGE=ema        the EMA arm's wider MOMENTUM grid (phase A; encoder_lr on $LRS):
#                    encoder_lr 5e-4/1e-3/2e-3 x momentum 0.99..0.9999, 12 cells
#                    per arm in EMA_ARMS (ema / ema_cos / ema_tnorm /
#                    ema_cos_tnorm), x EMA_FREEZE encoder-freeze values. Written into
#                    the VARIANTS root so --by_arm ranks old and new cells together.
#   STAGE=ema_pred   predictor knobs re-tuned UNDER a trained EMA encoder (phase
#                    B): lr x mask_frac x change_weight with the target knobs
#                    fixed at ENC_HP. Own root. Run on two seeds.
#
# Usage:
#   DATASET=mooc STAGE=predictor bash scripts/linkpred/slurm_hparam_search.sh --list
#   DATASET=mooc STAGE=predictor sbatch --array=0-35%6 scripts/linkpred/slurm_hparam_search.sh
#   python scripts/linkpred/aggregate_hparam.py --root outputs/lp_hparam/mooc_predictor --emit_flags
#
#   DATASET=mooc STAGE=baselines sbatch --array=0-68%6 scripts/linkpred/slurm_hparam_search.sh
#   python scripts/linkpred/aggregate_hparam.py --root outputs/lp_hparam/mooc_baselines --by_arm --emit_flags
#
#   DATASET=mooc STAGE=variants BASE_HP="<from predictor>" \
#       sbatch --array=0-47%4 scripts/linkpred/slurm_hparam_search.sh     # VARIANT_SET=all
#   DATASET=mooc STAGE=variants VARIANT_SET=old BASE_HP="<from predictor>" \
#       sbatch --array=0-15%4 scripts/linkpred/slurm_hparam_search.sh     # legacy four arms
#   python scripts/linkpred/aggregate_hparam.py --root outputs/lp_hparam/mooc_variants --by_arm --emit_flags
#
#   DATASET=mooc STAGE=ema EMA_ARMS="ema ema_cos" BASE_HP="<from predictor>" \
#       sbatch --array=0-23%4 scripts/linkpred/slurm_hparam_search.sh
#   for S in 42 43; do DATASET=mooc STAGE=ema_pred SEED=$S BASE_HP="<from predictor>" \
#       ENC_HP="<from variants/ema>" sbatch --array=0-23%6 scripts/linkpred/slurm_hparam_search.sh; done
#   python scripts/linkpred/aggregate_hparam.py --root outputs/lp_hparam/mooc_ema_pred --emit_flags

set -uo pipefail

DATASET="${DATASET:-mooc}"
STAGE="${STAGE:-predictor}"
SEED="${SEED:-42}"
BASE_HP="${BASE_HP:-}"
# ONE learning-rate grid shared by every arm that sweeps --lr: all baselines and
# DELOS's own predictor / ema_pred stages. Per-method grids centred on each
# paper's default put several winners on a grid EDGE (CLDG lowest, DDGCL / IDOL
# highest on MOOC) and made "who was tuned harder" depend on the method. A decade
# ladder spans every reference default (1e-4 DyG2Vec ... 5e-3 STEP / STD-MAE).
# Cell dirs are named by flag values, not index, so cells already on disk at
# 1e-3 / 1e-2 are reused. Override: LRS="3e-5 1e-4 3e-4" (e.g. METR-LA, from
# slurm_lr_probe.sh) -- but then use the SAME override for every stage.
LRS="${LRS:-1e-4 1e-3 1e-2}"
VARIANT_SET="${VARIANT_SET:-all}"
EMA_ARMS="${EMA_ARMS:-ema}"
ENC_HP="${ENC_HP:---encoder_lr 5e-4 --ema_momentum 0.999}"
# Encoder freeze epochs swept by STAGE=ema (outer loop, so the legacy 12 cells
# per arm keep their indices; "0 10 30" appends the two frozen-start variants).
EMA_FREEZE="${EMA_FREEZE:-0}"
# Cosine-anneal horizon for every EMA-target arm that anneals its momentum
# (DELOS ema_cos / ijepa / ema_cos_tnorm and the BGRL baseline). The
# trainers' default spans the whole --epochs budget, which early stopping
# (patience 20) rarely lets a run reach: on MOOC the ema_cos cells stopped at
# 85-240 of 300 epochs with the momentum still at ~0.997, i.e. the cosine arm
# was the constant arm. 50 epochs is below every observed run length, so the
# schedule finishes while the predictor still trains. 0 = whole run (legacy).
# Same value for every arm and method so no row gets its own schedule.
EMA_ANNEAL_EPOCHS="${EMA_ANNEAL_EPOCHS:-50}"
ANNEAL_ARG=""; [[ "$EMA_ANNEAL_EPOCHS" != "0" ]] && ANNEAL_ARG="--ema_anneal_epochs $EMA_ANNEAL_EPOCHS"

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
            # Legacy grid -- do NOT reorder: cell indices are on disk / in logs.
            # encoder_lr on the SAME ladder as --lr (LRS) since 2026-09-23; the
            # 1e-5/1e-4/5e-4 grid put the EMA winners on its top edge.
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
            # Appended 2026-09-23 (indices above unchanged): the EMA arms with
            # target LayerNorm but WITHOUT context masking -- isolates tnorm
            # from ctx, which the ijepa arm bundles. Same grid as ema / ema_cos.
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
        # Phase A: the EMA arm's target-network knobs, both edges of the
        # variants grid extended. Do NOT reorder: indices are on disk / in logs.
        for fz in $EMA_FREEZE; do
            FZ_ARG=""; [[ "$fz" != "0" ]] && FZ_ARG=" --encoder_freeze_epochs $fz"
            for arm in $EMA_ARMS; do
                ema_arm_ok "$arm"
                # encoder_lr on the shared ladder since 2026-09-23 (was
                # 5e-4/1e-3/2e-3, which did not contain the variants-stage
                # winner 1e-4); the momentum axis is this stage's own.
                for elr in $LRS; do
                    for m in 0.99 0.999 0.9995 0.9999; do
                        GRID+=("$arm|--encoder_lr $elr --ema_momentum $m$FZ_ARG")
                    done
                done
            done
        done
    elif [[ "$STAGE" == "ema_pred" ]]; then
        # Phase B: predictor knobs under a TRAINED EMA encoder; the target knobs
        # stay at ENC_HP. Do NOT reorder: indices are on disk / in logs.
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
    else
        echo "[FATAL] STAGE must be 'predictor', 'variants', 'ema', 'ema_pred' " \
             "or 'baselines', got '$STAGE'" >&2
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

case "$DATASET" in
    mooc|reddit|wikipedia) RAW="dataset/jodie/${DATASET}.csv" ;;
    reddit_body)           RAW="dataset/reddit/soc-redditHyperlinks-body.tsv" ;;
    dblp)                  RAW="dataset/dblp/edges.csv" ;;
    cldg_dblp|cldg_patent|patent)
        N="${DATASET#cldg_}"; RAW="dataset/cldg/${N}/${N}.txt" ;;
    tgbn_trade)            RAW="dataset/tgb/tgbn-trade/tgbn-trade_node_labels.csv" ;;
    *) echo "[FATAL] DATASET must be one of: mooc reddit wikipedia reddit_body " \
            "dblp cldg_dblp cldg_patent tgbn_trade -- got '$DATASET'" >&2; exit 2 ;;
esac
[[ -f "$RAW" ]] || { echo "[FATAL] $RAW missing; fetch it from the login node" >&2; exit 10; }

TASK=link_pred
source "$REPO_ROOT/scripts/dblp/_dataset_config.sh"
BATCH="${BATCH:-8}"
# Link-probe window batch. analyze_jepa materialises B x HIST float (N, N)
# adjacencies: at N=10984 (reddit) B=4 is 5.4 GiB and OOMs an 11 GB card, so
# use LINK_BATCH=1 there. Use the SAME value in slurm_all_methods.sh.
LINK_BATCH="${LINK_BATCH:-4}"
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
        # train_delos.py, so the five arms above are byte-for-byte the old ones.
        ctx)         echo "--mask_encoder_input" ;;
        tnorm)       echo "--target_norm" ;;
        ema_cos)     echo "--train_encoder --ema_target --ema_schedule cosine $ANNEAL_ARG" ;;
        # EMA + target LayerNorm, no context masking (2026-09-23): the
        # I-JEPA target normalisation on its own, on top of each EMA arm.
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
ROOT="${ROOT:-outputs/lp_hparam/${SHORT}_${ROOT_STAGE}}"
OUT="$ROOT/${ARM}_${SLUG}_s${SEED}"
mkdir -p "$OUT"
printf '%s\n' "arm=$ARM flags=$HP_ARGS base=$BASE_HP" > "$OUT/hp_args.txt"
echo "[cell $IDX] dataset=$DATASET stage=$STAGE arm=$ARM seed=$SEED -> $OUT"

# --- STAGE=baselines: one competing method per cell, scored on VAL ----------
if [[ "$STAGE" == "baselines" ]]; then
    # A cell is complete once its val metrics exist (analysis_val.json is the last
    # thing a cell writes), so resubmitting the array -- after a walltime kill or
    # a grid change -- re-runs only the missing cells. Delete the dir to force one.
    if [[ -f "$OUT/analysis_val.json" ]]; then
        echo "[skip] cell $IDX ($ARM) complete: $OUT/analysis_val.json exists"; exit 0
    fi
    # A trained encoder whose probe died (OOM) is re-probed, not re-trained.
    if [[ -f "$OUT/encoder.ckpt" ]]; then
        echo "[skip] $OUT/encoder.ckpt exists -> probe only"
    else
    case "$ARM" in
      cldg)
        srun --export=ALL python -m src.baselines.cldg.train_cldg $DATA_ARG \
            $ENCODER_ARG $WINDOW_ARG $EARLY_STOP_ARG \
            --snapshots 4 --strategy random --tau 0.07 \
            --node_batch_size 256 --steps_per_epoch 20 --val_steps 10 \
            --epochs "${BASE_EPOCHS:-100}" --device cuda --seed "$SEED" \
            --out_dir "$OUT" $HP_ARGS \
            || { echo "[FATAL] CLDG cell failed" >&2; exit 21; } ;;
      ddgcl)
        srun --export=ALL python -m src.baselines.ddgcl.train_ddgcl $DATA_ARG \
            $ENCODER_ARG $WINDOW_ARG $EARLY_STOP_ARG \
            --time_dim 32 --tau 0.01 --beta 1.5 \
            --node_batch_size 256 --steps_per_epoch 20 --val_steps 10 \
            --epochs "${BASE_EPOCHS:-100}" --device cuda --seed "$SEED" \
            --out_dir "$OUT" $HP_ARGS \
            || { echo "[FATAL] DDGCL cell failed" >&2; exit 22; } ;;
      idol)
        srun --export=ALL python -m src.baselines.idol.train_idol $DATA_ARG \
            $ENCODER_ARG $WINDOW_ARG $EARLY_STOP_ARG \
            --node_batch_size 256 --steps_per_epoch 20 --val_steps 10 \
            --epochs "${BASE_EPOCHS:-100}" --device cuda --seed "$SEED" \
            --out_dir "$OUT" $HP_ARGS \
            || { echo "[FATAL] IDOL cell failed" >&2; exit 23; } ;;
      maskdgnn)
        srun --export=ALL python -m src.baselines.maskdgnn.train_maskdgnn $DATA_ARG \
            $ENCODER_ARG --encoder_kind maskdgnn $WINDOW_ARG $EARLY_STOP_ARG \
            --alpha 0.7 --target next --neg_per_pos 1 \
            --steps_per_epoch 20 --val_steps 10 --e2e_eval val \
            --e2e_batch_size "$BATCH" --num_workers "$NW" \
            --epochs "${BASE_EPOCHS:-100}" --device cuda --seed "$SEED" \
            --out_dir "$OUT" $HP_ARGS \
            || { echo "[FATAL] MaskDGNN cell failed" >&2; exit 25; } ;;
      dysubc)
        srun --export=ALL python -m src.baselines.dysubc.train_dysubc $DATA_ARG \
            $ENCODER_ARG $WINDOW_ARG $EARLY_STOP_ARG \
            --subgraph_size 20 --beta 1 --margin 0.75 --lam 0.5 \
            --node_batch_size 256 --steps_per_epoch 20 --val_steps 10 \
            --epochs "${BASE_EPOCHS:-100}" --device cuda --seed "$SEED" \
            --out_dir "$OUT" $HP_ARGS \
            || { echo "[FATAL] DySubC cell failed" >&2; exit 26; } ;;
      dyg2vec)
        srun --export=ALL python -m src.baselines.dyg2vec.train_dyg2vec $DATA_ARG \
            $ENCODER_ARG --encoder_kind dyg2vec $WINDOW_ARG $EARLY_STOP_ARG \
            --lam 25 --mu 25 --nu 1 --steps_per_epoch 20 --val_steps 10 \
            --epochs "${BASE_EPOCHS:-100}" --device cuda --seed "$SEED" \
            --out_dir "$OUT" $HP_ARGS \
            || { echo "[FATAL] DyG2Vec cell failed" >&2; exit 29; } ;;
      dygwm)
        srun --export=ALL python -m src.baselines.dygwm.train_dygwm $DATA_ARG \
            $ENCODER_ARG --encoder_kind dygwm $WINDOW_ARG $EARLY_STOP_ARG \
            --n_ctx 32 --decay 0.8 --cn_bonus 1.0 \
            --lam_node 1 --lam_rel 1 --lam_var 0.25 --lam_cov 0.025 \
            --node_batch_size 256 --pair_batch_size 256 \
            --steps_per_epoch 20 --val_steps 10 \
            --epochs "${BASE_EPOCHS:-100}" --device cuda --seed "$SEED" \
            --out_dir "$OUT" $HP_ARGS \
            || { echo "[FATAL] DyG-WM cell failed" >&2; exit 31; } ;;
      sgjepa)
        srun --export=ALL python -m src.baselines.sgjepa.train_sgjepa $DATA_ARG \
            $ENCODER_ARG $WINDOW_ARG $EARLY_STOP_ARG \
            --snn_layers 1 --init_tau 2.0 --v_th 1.0 \
            --node_batch_size 256 --steps_per_epoch 20 --val_steps 10 \
            --epochs "${BASE_EPOCHS:-100}" --device cuda --seed "$SEED" \
            --out_dir "$OUT" $HP_ARGS \
            || { echo "[FATAL] SG-JEPA cell failed" >&2; exit 32; } ;;
      bgrl)
        srun --export=ALL python -m src.baselines.bgrl.train_bgrl $DATA_ARG \
            $ENCODER_ARG $WINDOW_ARG $EARLY_STOP_ARG \
            --p_e1 0.2 --p_e2 0.3 --p_f1 0.2 --p_f2 0.1 \
            --steps_per_epoch 20 --val_steps 10 \
            --epochs "${BASE_EPOCHS:-100}" --device cuda --seed "$SEED" \
            --out_dir "$OUT" $ANNEAL_ARG $HP_ARGS \
            || { echo "[FATAL] BGRL cell failed" >&2; exit 30; } ;;
      graphmae)
        srun --export=ALL python -m src.baselines.graphmae.train_graphmae $DATA_ARG \
            $ENCODER_ARG $WINDOW_ARG $EARLY_STOP_ARG \
            --replace_rate 0.05 --sce_gamma 2 --target all \
            --steps_per_epoch 20 --val_steps 10 \
            --epochs "${BASE_EPOCHS:-100}" --device cuda --seed "$SEED" \
            --out_dir "$OUT" $HP_ARGS \
            || { echo "[FATAL] GraphMAE cell failed" >&2; exit 29; } ;;
      grace)
        srun --export=ALL python -m src.baselines.grace.train_grace $DATA_ARG \
            $ENCODER_ARG $WINDOW_ARG $EARLY_STOP_ARG \
            --p_e1 0.2 --p_e2 0.4 --p_f1 0.3 --p_f2 0.4 \
            --node_batch_size 256 --steps_per_epoch 20 --val_steps 10 \
            --epochs "${BASE_EPOCHS:-100}" --device cuda --seed "$SEED" \
            --out_dir "$OUT" $HP_ARGS \
            || { echo "[FATAL] GRACE cell failed" >&2; exit 28; } ;;
      *) echo "[FATAL] unknown baseline arm '$ARM'" >&2; exit 3 ;;
    esac
    fi
    srun --export=ALL python -m src.train.analyze_jepa --link_sources encoder \
        $DATA_ARG $WINDOW_ARG --skip_diagnostic --probe_split val \
        --link_epochs "${LINK_EPOCHS:-30}" --link_lr "${LINK_LR:-1e-3}" \
        --link_probe_hidden 64 --encoder_ckpt "$OUT/encoder.ckpt" \
        --batch_size "$LINK_BATCH" \
        --seed "$SEED" --device cuda --out_dir "$OUT" \
        || { echo "[FATAL] link probe failed" >&2; exit 24; }
    [[ -f "$OUT/analysis_val.json" ]] || { echo "[FATAL] cell $IDX wrote no metrics" >&2; exit 12; }
    # analyze_jepa exits 0 even when the link source OOMs and writes an EMPTY
    # link_pred block. That must not count as a finished cell: drop the file so
    # a resubmit re-probes it (the encoder.ckpt guard above skips retraining).
    python - "$OUT/analysis_val.json" <<'PY' || { rm -f "$OUT/analysis_val.json"; echo "[FATAL] cell $IDX: probe produced no metrics (OOM? lower LINK_BATCH)" >&2; exit 24; }
import json, sys
lp = (json.load(open(sys.argv[1])).get("link_pred") or {}).get("encoder") or {}
sys.exit(0 if lp.get("all") else 1)
PY
    echo "[done] cell $IDX ($ARM) -> $OUT"
    exit 0
fi

VFLAGS="$(arm_flags "$ARM")"
[[ "$VFLAGS" == "__UNKNOWN__" ]] && { echo "[FATAL] bad arm '$ARM'" >&2; exit 3; }

# --- shared pretrained encoder ----------------------------------------------
ENC_DIR="$ROOT/_shared_encoder_s${SEED}"
ENC_CKPT="$ENC_DIR/encoder_${DATASET}_${ENC_KIND}_edge_mask.ckpt"
mkdir -p "$ENC_DIR"
exec 9>"$ENC_DIR/.pretrain.lock"
flock -w 21600 9 || { echo "[FATAL] encoder lock timeout" >&2; exit 10; }
if [[ ! -f "$ENC_CKPT" ]]; then
    echo "=== shared encoder pretrain (lock held by cell $IDX) ==="
    srun --export=ALL python src/train/pretrain_encoder.py $DATA_ARG \
        $ENCODER_ARG $WINDOW_ARG $EARLY_STOP_ARG \
        --pretext edge_mask --decoder_hidden 64 --mask_rate 0.25 \
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
srun --export=ALL python src/train/train_delos.py \
    $DATA_ARG $ENCODER_ARG $WINDOW_ARG $EARLY_STOP_ARG \
    --encoder_ckpt "$ENC_CKPT" --mask_scheme "${MASK_SCHEME:-node_mask}" \
    --downstream_task link_pred --residual_head \
    --heads "${HEADS:-4}" --mlp_dim "${MLP_DIM:-512}" \
    --batch_size "$BATCH" --epochs "${JEPA_EPOCHS:-300}" \
    --probe_epochs "${JEPA_PROBE_EPOCHS:-3}" --probe_split val \
    --seed "$SEED" --num_workers "$NW" --device cuda --out_dir "$OUT" \
    $BASE_HP $VFLAGS $HP_ARGS
[[ -f "$OUT/test_metrics.json" ]] || { echo "[FATAL] cell $IDX wrote no metrics" >&2; exit 12; }

echo "[done] cell $IDX -> $OUT"
