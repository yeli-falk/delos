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
# Three stages:
#   STAGE=predictor  predictor knobs on the FROZEN arm -> BASE_HP
#   STAGE=baselines  GRACE / CLDG / DDGCL / IDOL / ..., so the table is not "our tuned method
#                    vs their defaults"
#   STAGE=variants   encoder-training knob per variant, BASE_HP fixed
#
# Usage:
#   DATASET=mooc STAGE=predictor bash scripts/linkpred/slurm_hparam_search.sh --list
#   DATASET=mooc STAGE=predictor sbatch --array=0-35%6 scripts/linkpred/slurm_hparam_search.sh
#   python scripts/linkpred/aggregate_hparam.py --root outputs/lp_hparam/mooc_predictor --emit_flags
#
#   DATASET=mooc STAGE=baselines sbatch --array=0-20%6 scripts/linkpred/slurm_hparam_search.sh
#   python scripts/linkpred/aggregate_hparam.py --root outputs/lp_hparam/mooc_baselines --by_arm --emit_flags
#
#   DATASET=mooc STAGE=variants BASE_HP="<from predictor>" \
#       sbatch --array=0-15%4 scripts/linkpred/slurm_hparam_search.sh
#   python scripts/linkpred/aggregate_hparam.py --root outputs/lp_hparam/mooc_variants --by_arm --emit_flags

set -uo pipefail

DATASET="${DATASET:-mooc}"
STAGE="${STAGE:-predictor}"
SEED="${SEED:-42}"
BASE_HP="${BASE_HP:-}"

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
        # DyG2Vec: LR x distortion strength (p_d, the only knob the paper
        # validates; 0.3 is its choice). Both dropouts move together.
        for lr in 1e-4 1e-3 4e-3; do
            for pd in 0.15 0.3; do GRID+=("dyg2vec|--lr $lr --p_edge $pd --p_struct $pd"); done
        done
        # GRACE (static control): LR x tau. The temperature is the paper's
        # load-bearing knob after the drop rates, which stay at the Cora
        # defaults; lr spans the reference 5e-4 up to the temporal arms' 4e-3.
        for lr in 5e-4 1e-3 4e-3; do
            for t in 0.2 0.4 0.7; do GRID+=("grace|--lr $lr --tau $t"); done
        done
        # GraphMAE (static control): LR x mask rate, the paper's headline knob
        # (robust 0.3-0.75 on citation graphs; lower on feature-poor inputs).
        for lr in 5e-4 1e-3 4e-3; do
            for mr in 0.3 0.5 0.75; do GRID+=("graphmae|--lr $lr --mask_rate $mr"); done
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

case "$DATASET" in
    mooc|reddit|wikipedia) RAW="dataset/jodie/${DATASET}.csv" ;;
    reddit_body)           RAW="dataset/reddit/soc-redditHyperlinks-body.tsv" ;;
    dblp)                  RAW="dataset/dblp/edges.csv" ;;
    *) echo "[FATAL] DATASET must be one of: mooc reddit wikipedia reddit_body " \
            "dblp -- got '$DATASET'" >&2; exit 2 ;;
esac
[[ -f "$RAW" ]] || { echo "[FATAL] $RAW missing; fetch it from the login node" >&2; exit 10; }

TASK=link_pred
source "$REPO_ROOT/scripts/dblp/_dataset_config.sh"
BATCH="${BATCH:-8}"
NW="${SLURM_CPUS_PER_TASK:-8}"
SHORT="${DATASET#jodie_}"

arm_flags() {
    case "$1" in
        frozen) echo "" ;;
        naive)  echo "--train_encoder" ;;
        ema)    echo "--train_encoder --ema_target" ;;
        vicreg) echo "--train_encoder --vicreg_cov 1.0" ;;
        *)      echo "__UNKNOWN__" ;;
    esac
}

SLUG=$(echo "$HP_ARGS" | sed 's/--//g; s/[ .]/_/g; s/__*/_/g; s/^_//; s/_$//')
[[ -z "$SLUG" ]] && SLUG=base
ROOT="outputs/lp_hparam/${SHORT}_${STAGE}"
OUT="$ROOT/${ARM}_${SLUG}_s${SEED}"
mkdir -p "$OUT"
printf '%s\n' "arm=$ARM flags=$HP_ARGS base=$BASE_HP" > "$OUT/hp_args.txt"
echo "[cell $IDX] dataset=$DATASET stage=$STAGE arm=$ARM seed=$SEED -> $OUT"

# --- STAGE=baselines: one competing method per cell, scored on VAL ----------
if [[ "$STAGE" == "baselines" ]]; then
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
    srun --export=ALL python -m src.train.analyze_jepa --link_sources encoder \
        $DATA_ARG $WINDOW_ARG --skip_diagnostic --probe_split val \
        --link_epochs "${LINK_EPOCHS:-30}" --link_lr "${LINK_LR:-1e-3}" \
        --link_probe_hidden 64 --encoder_ckpt "$OUT/encoder.ckpt" \
        --seed "$SEED" --device cuda --out_dir "$OUT" \
        || { echo "[FATAL] link probe failed" >&2; exit 24; }
    [[ -f "$OUT/analysis_val.json" ]] || { echo "[FATAL] cell $IDX wrote no metrics" >&2; exit 12; }
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
srun --export=ALL python src/train/train_dyg_jepa.py \
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
