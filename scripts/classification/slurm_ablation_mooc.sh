#!/usr/bin/env bash
#SBATCH --job-name=cls-ablation
#SBATCH --output=logs/cls-ablation-%A_%a.out
#SBATCH --error=logs/cls-ablation-%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#
# DyG-JEPA ABLATION on the MOOC node-state task: one component removed per
# row, everything else held at the final recipe (cosine-EMA target, trained
# encoder, node masking, residual head, change-weighted loss, pretrained
# encoder, no structural attention mask).
#
# Arms (row label in the table -> what changes vs `full`):
#   full           the final model: --train_encoder --ema_target --ema_schedule cosine
#   const_ema      w/o cosine anneal          (--ema_schedule const)
#   no_target_net  w/o EMA target             (--train_encoder alone; same-encoder targets)
#   frozen_enc     w/o encoder training       (predictor only, pretrained encoder frozen)
#   no_node_mask   w/o node masking           (--mask_scheme future_forecast)
#   input_space    w/o latent-space target    (--predict_space input: raw-feature loss)
#   no_change_w    w/o change weighting       (--change_weight 0)
#   no_residual    w/o residual head          (no --residual_head)
#   no_pretrain    w/o encoder pretraining    (random-init encoder, no --encoder_ckpt)
#   mlp_pred       w/o graph predictor        (--mask_scheme mlp_forecast, per-node MLP)
#   gru_pred       w/o graph predictor        (--mask_scheme gru_forecast, per-node GRU)
#   no_attn_mask   w/o structural attention mask (--no_attention_mask: dense
#                  all-to-all attention; graph structure then enters only via
#                  the encoder)
#   ctx_mask       + context-input masking    (--mask_encoder_input)  [opt-in]
#
# NOTE: the main table (slurm_all_methods.sh) runs with the structural mask ON
# (trainer default, sparse edge-list attention). `no_attn_mask` is the honest
# ablation of it. Results produced by earlier versions of these scripts (mask
# off) are not comparable -- use a fresh ROOT.
#
# Reuse: four arms are byte-identical to arms of slurm_all_methods.sh
# (full=jepa_ema_cos, const_ema=jepa_ema, no_target_net=jepa_naive,
# frozen_enc=jepa_frozen). If that run's state_probe.json exists under $ROOT it
# is used as-is and nothing is retrained. Every other arm writes to
# $ROOT/abl_<arm>_s<seed>/. Every stage skips when its output exists, so the
# job is resumable and the table can be regenerated at any time.
#
# Usage (from the repo root):
#   bash scripts/classification/slurm_ablation_mooc.sh --list        # grid -> array size
#   sbatch --array=0-29%4 scripts/classification/slurm_ablation_mooc.sh
#   bash scripts/classification/slurm_ablation_mooc.sh table         # LaTeX table (no GPU)
#
# One arm, one seed:   ARMS=no_pretrain SEEDS=42 sbatch --array=0 scripts/classification/slurm_ablation_mooc.sh
# Without SLURM:       ARMS="full no_node_mask" SEEDS="42" bash scripts/classification/slurm_ablation_mooc.sh
#
# Tuned flags: pass the SAME ones the main table used, or the `full` row will
# not match it:
#   BASE_HP="--lr 1e-3 --num_masked_nodes 30" ENC_HP="--encoder_lr 1e-4" sbatch ...
# ENC_HP is appended to every arm that trains the encoder (all but frozen_enc).

set -uo pipefail
mkdir -p logs

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# --- grid -------------------------------------------------------------------
DEFAULT_ARMS="full const_ema no_target_net frozen_enc no_node_mask input_space no_change_w no_residual no_pretrain mlp_pred no_attn_mask"
ARMS="${ARMS:-$DEFAULT_ARMS}"
SEEDS="${SEEDS:-42 43 44}"
DATASET="${DATASET:-mooc}"
SHORT="${DATASET#jodie_}"
ROOT="${ROOT:-outputs/cls_${SHORT}}"

GRID=()
for S in $SEEDS; do for A in $ARMS; do GRID+=("$A|$S"); done; done

if [[ "${1:-}" == "--list" ]]; then
    echo "arms  = $ARMS"
    echo "seeds = $SEEDS"
    echo "grid size = ${#GRID[@]}  ->  sbatch --array=0-$(( ${#GRID[@]} - 1 ))%4 $0"
    for i in "${!GRID[@]}"; do echo "  $i: ${GRID[$i]}"; done
    exit 0
fi
if [[ "${1:-}" == "table" ]]; then
    exec python scripts/classification/aggregate_ablation.py \
        --dataset "$SHORT" --root "$ROOT" --seeds $SEEDS --arms $ARMS --latex "${@:2}"
fi

# --- interpreter (same resolution as slurm_all_methods.sh) --------------------
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
if command -v srun >/dev/null 2>&1 && [[ -n "${SLURM_JOB_ID:-}" ]]; then
    RUN="srun --export=ALL $PY"
else
    RUN="$PY"
fi
echo "[env] python=$(command -v "$PY")"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

# --- shared config (identical to the main classification table) -------------
case "$DATASET" in mooc|reddit|wikipedia|jodie_mooc|jodie_reddit|jodie_wikipedia) ;;
    *) echo "[FATAL] DATASET must be a JODIE dataset, got '$DATASET'" >&2; exit 2 ;;
esac
CSV="dataset/jodie/${SHORT}.csv"
[[ -f "$CSV" ]] || { echo "[FATAL] $CSV missing (wget http://snap.stanford.edu/jodie/${SHORT}.csv -O $CSV on the login node)" >&2; exit 10; }

NW="${SLURM_CPUS_PER_TASK:-8}"
PROBE_SPLIT="${PROBE_SPLIT:-test}"
STATE_MODE="${STATE_MODE:-state}"
PROBE_BATCH="${PROBE_BATCH:-4}"
TASK=state_probe
source "$REPO_ROOT/scripts/dblp/_dataset_config.sh"
BATCH="${BATCH:-8}"
BASE_HP="${BASE_HP:-}"
ENC_HP="${ENC_HP:-}"
mkdir -p "$ROOT"

JEPA_COMMON="$DATA_ARG $WINDOW_ARG $EARLY_STOP_ARG $ENCODER_ARG \
    --num_masked_nodes ${NMASK:-30} --depth ${DEPTH:-4} --heads ${HEADS:-4} \
    --mlp_dim ${MLP_DIM:-512} --batch_size $BATCH --epochs ${JEPA_EPOCHS:-300} \
    --lr ${LR:-5e-4} --probe_epochs ${JEPA_PROBE_EPOCHS:-3} --num_workers $NW --device cuda"
PROBE_COMMON="--task binary --state_label_mode $STATE_MODE \
    --probe_split $PROBE_SPLIT --device cuda --batch_size $PROBE_BATCH --num_workers $NW"
FULL="--train_encoder --ema_target --ema_schedule cosine"

# arm -> "<flags>|<residual 0/1>|<change_w>|<use_ckpt 0/1>|<no_attn_mask 0/1>|<legacy stem or ->"
arm_spec() {
    case "$1" in
        full)          echo "$FULL|1|2.0|1|0|jepa_ema_cos" ;;
        const_ema)     echo "--train_encoder --ema_target|1|2.0|1|0|jepa_ema" ;;
        no_target_net) echo "--train_encoder|1|2.0|1|0|jepa_naive" ;;
        frozen_enc)    echo "|1|2.0|1|0|jepa_frozen" ;;
        no_node_mask)  echo "$FULL --mask_scheme future_forecast|1|2.0|1|0|-" ;;
        input_space)   echo "--train_encoder --predict_space input|1|2.0|1|0|-" ;;
        no_change_w)   echo "$FULL|1|0|1|0|-" ;;
        no_residual)   echo "$FULL|0|2.0|1|0|-" ;;
        no_pretrain)   echo "$FULL|1|2.0|0|0|-" ;;
        mlp_pred)      echo "$FULL --mask_scheme mlp_forecast|1|2.0|1|0|-" ;;
        gru_pred)      echo "$FULL --mask_scheme gru_forecast|1|2.0|1|0|-" ;;
        ctx_mask)      echo "$FULL --mask_encoder_input|1|2.0|1|0|-" ;;
        no_attn_mask)  echo "$FULL|1|2.0|1|1|-" ;;
        *)             echo "__UNKNOWN__" ;;
    esac
}

fail() { echo "[FATAL] $*" >&2; exit "${2:-1}"; }

# --- shared pretrained encoder (same path and lock as slurm_all_methods.sh) --
ensure_encoder() {   # $1 = seed
    local seed="$1"
    ENC_DIR="$ROOT/_shared_encoder_s${seed}"
    ENC_CKPT="$ENC_DIR/encoder_${DATASET}_${ENC_KIND}_edge_mask.ckpt"
    mkdir -p "$ENC_DIR"
    exec 9>"$ENC_DIR/.pretrain.lock"
    flock -w 21600 9 || fail "encoder lock timeout" 10
    if [[ -f "$ENC_CKPT" && -f "$ENC_DIR/.done" ]]; then
        echo "[skip] shared encoder complete: $ENC_CKPT"
    elif [[ -f "$ENC_CKPT" && "${ENC_TRUST:-0}" == "1" ]]; then
        echo "[skip] shared encoder exists without .done; ENC_TRUST=1 -> keeping it"
        touch "$ENC_DIR/.done"
    else
        echo; echo "=== shared encoder pretrain (seed $seed) -> $ENC_DIR ==="
        rm -f "$ENC_CKPT"
        if $RUN src/train/pretrain_encoder.py $DATA_ARG $ENCODER_ARG $WINDOW_ARG $EARLY_STOP_ARG \
            --pretext edge_mask --decoder_hidden 64 --mask_rate 0.25 \
            --batch_size "$BATCH" --epochs "${ENC_EPOCHS:-300}" --lr 1e-3 \
            --seed "$seed" --num_workers "$NW" --device cuda --out_dir "$ENC_DIR" \
           && [[ -f "$ENC_CKPT" ]]; then
            touch "$ENC_DIR/.done"
        else
            flock -u 9; fail "encoder pretrain failed / produced no $ENC_CKPT" 11
        fi
    fi
    flock -u 9; exec 9>&-
}

# --- one (arm, seed) cell ----------------------------------------------------
run_cell() {         # $1 = arm, $2 = seed
    local arm="$1" seed="$2"
    local spec; spec="$(arm_spec "$arm")"
    [[ "$spec" == "__UNKNOWN__" ]] && fail "unknown arm '$arm'" 12
    IFS='|' read -r vflags residual change_w use_ckpt attn legacy <<< "$spec"
    local probe_file="state_probe$([[ "$PROBE_SPLIT" == "val" ]] && echo _val).json"

    # 1) reuse the main-table run when it is the same configuration.
    if [[ "$legacy" != "-" && -f "$ROOT/${legacy}_s${seed}/$probe_file" ]]; then
        echo "[reuse] $arm s$seed <- $ROOT/${legacy}_s${seed}/$probe_file"
        return 0
    fi
    local out="$ROOT/abl_${arm}_s${seed}"
    if [[ -f "$out/$probe_file" ]]; then
        echo "[skip] $arm s$seed complete: $out/$probe_file"
        return 0
    fi
    mkdir -p "$out"

    # 2) assemble flags for this cell.
    local base="--mask_scheme node_mask"
    [[ "$residual" == "1" ]] && base="$base --residual_head"
    [[ "$change_w" != "0" ]] && base="$base --change_weight $change_w"
    local ckpt_arg="" enc_hp=""
    if [[ "$use_ckpt" == "1" ]]; then ensure_encoder "$seed"; ckpt_arg="--encoder_ckpt $ENC_CKPT"; fi
    [[ "$vflags" == *"--train_encoder"* ]] && enc_hp="$ENC_HP"
    local attn_arg=""
    [[ "$attn" == "1" ]] && attn_arg="--no_attention_mask"

    echo; echo "=== ablation [$arm] seed $seed -> $out ==="
    echo "    flags='$base $BASE_HP $vflags $enc_hp' ckpt='${ckpt_arg:-none}' attn='${attn_arg:-structural (sparse)}'"

    # 3) train (complete = best.ckpt + test_metrics.json; a lone best.ckpt is an
    #    interrupted run and restarts from scratch, as in the main script).
    if [[ -f "$out/best.ckpt" && -f "$out/test_metrics.json" ]]; then
        echo "[skip] training complete in $out"
    else
        [[ -f "$out/best.ckpt" ]] && { echo "[resume] interrupted run -> restart"; rm -f "$out/best.ckpt" "$out/history.json"; }
        # Order: recipe, tuned HP, variant flags (argparse keeps the last occurrence).
        $RUN src/train/train_dyg_jepa.py $JEPA_COMMON $attn_arg $ckpt_arg \
            $base $BASE_HP $vflags $enc_hp --seed "$seed" --out_dir "$out" \
            || { echo "[warn] $arm s$seed training failed"; return 1; }
        [[ -f "$out/best.ckpt" && -f "$out/test_metrics.json" ]] \
            || { echo "[warn] $arm s$seed did not finish"; return 1; }
    fi

    # 4) state probe (jepa + encoder_last + oracle + degree columns).
    echo "=== state probe [$arm] seed $seed ==="
    $RUN scripts/dblp/run_block_probe.py --jepa_ckpt "$out/best.ckpt" $PROBE_COMMON \
        || { echo "[warn] state probe failed for $arm s$seed"; return 1; }

    # 5) collapse check for the arms that move the encoder.
    if [[ "$vflags" == *"--train_encoder"* ]]; then
        $RUN -m src.train.analyze_jepa --skip_diagnostic --link_sources jepa \
            $DATA_ARG $WINDOW_ARG --jepa_ckpt "$out/best.ckpt" --device cuda \
            --seed "$seed" --out_dir "$out/collapse_check" \
            || echo "[warn] collapse check failed for $arm s$seed (non-fatal)"
    fi
}

echo "[config] DATASET=$DATASET ROOT=$ROOT ARMS='$ARMS' SEEDS='$SEEDS' BASE_HP='$BASE_HP' ENC_HP='$ENC_HP'"
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    cell="${GRID[$SLURM_ARRAY_TASK_ID]:-}"
    [[ -n "$cell" ]] || fail "array index $SLURM_ARRAY_TASK_ID outside grid of ${#GRID[@]}" 2
    run_cell "${cell%|*}" "${cell#*|}"
else
    for cell in "${GRID[@]}"; do run_cell "${cell%|*}" "${cell#*|}" || true; done
fi

echo; echo "[done] regenerate the table with:"
echo "  bash scripts/classification/slurm_ablation_mooc.sh table"
