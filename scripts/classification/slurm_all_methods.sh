#!/usr/bin/env bash
#SBATCH --job-name=cls-allmethods
#SBATCH --output=logs/cls-allmethods-%j.out
#SBATCH --error=logs/cls-allmethods-%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#
# Dynamic node classification: every method, one seed per job. Each arm writes
# its own state_probe.json. The arm table, the probe columns and the
# per-dataset notes are in scripts/classification/README.md.
#
# Usage:
#   DATASET=mooc SEED=42 sbatch scripts/classification/slurm_all_methods.sh
#   for S in 42 43 44; do DATASET=mooc SEED=$S sbatch scripts/classification/slurm_all_methods.sh; done
#   python scripts/classification/aggregate_classification.py --dataset mooc --seeds 42 43 44 --latex
#   # One arm per job (the JEPA arms are independent once the shared encoder exists):
#   DATASET=mooc SEED=42 ARMS=jepa_ema sbatch scripts/classification/slurm_all_methods.sh
#
# Main environment knobs (the others are documented where they are defined):
#   DATASET      mooc | reddit | wikipedia | synthetic_dyn | cldg_dblp |
#                cldg_patent (alias patent) | tgbn_* | bts_flights | nyc_taxi |
#                chase_{dblp_s,gossipcop,music}. Default mooc.
#   SEED         Default 42.
#   ARMS         Stages to run, see "which arms this job runs". Default all.
#   STATE_MODE   state (dynamic cumulative, default) | ever (static) |
#                event (instantaneous, very sparse).
#   PROBE_SPLIT  Split the state probe is scored on. Default test.
#   ROOT         Results root. Default outputs/cls_<dataset>.
#   BASE_HP, VARIANT_HP_<arm>, BASELINE_HP_<method>
#                Tuned flags from slurm_hparam_search.sh, for example
#                BASE_HP="--lr 1e-3 --num_masked_nodes 30"
#                VARIANT_HP_ema="--encoder_lr 1e-4" BASELINE_HP_cldg="--lr 1e-3 --views 4".
#
# There is no jepa_enc arm: run_block_probe.py reports encoder_last (copy-last),
# encoder_oracle and degree_history_baseline as columns of every
# state_probe.json. The binary labels are heavily imbalanced, so the headline
# is AUROC/AUPRC, never accuracy.

set -uo pipefail
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

DATASET="${DATASET:-mooc}"
case "$DATASET" in
    mooc|reddit|wikipedia|synthetic_dyn|cldg_dblp|cldg_patent|patent|tgbn_trade|tgbn_genre|tgbn_reddit|tgbn_token|bts_flights|nyc_taxi|chase_dblp_s|chase_gossipcop|chase_music) ;;
    *) echo "[FATAL] DATASET must be mooc, reddit, wikipedia, synthetic_dyn, cldg_dblp, cldg_patent, tgbn_*, bts_flights, nyc_taxi, or chase_{dblp_s,gossipcop,music}, got '$DATASET'" >&2
       exit 2 ;;
esac
SEED="${SEED:-42}"
# Cosine-anneal horizon shared by every EMA arm that anneals its momentum
# (DELOS ema_cos / ijepa / ema_cos_tnorm and the BGRL baseline). The trainers'
# default spans the whole --epochs budget, which early stopping rarely reaches,
# so the schedule would never complete. 0 = anneal over the whole run.
EMA_ANNEAL_EPOCHS="${EMA_ANNEAL_EPOCHS:-50}"
ANNEAL_ARG=""; [[ "$EMA_ANNEAL_EPOCHS" != "0" ]] && ANNEAL_ARG="--ema_anneal_epochs $EMA_ANNEAL_EPOCHS"
NW="${SLURM_CPUS_PER_TASK:-8}"
PROBE_SPLIT="${PROBE_SPLIT:-test}"
# DOWNSTREAM is the JEPA trainer's end-of-run link probe. No classification
# aggregator reads it and it can run out of GPU memory on the larger graphs, so
# it is off by default; the trainer still writes test_metrics.json (the
# completion marker). COLLAPSE_CHECK=0 skips the unfrozen arms' collapse check.
DOWNSTREAM="${DOWNSTREAM:-none}"
COLLAPSE_CHECK="${COLLAPSE_CHECK:-1}"
STATE_MODE="${STATE_MODE:-state}"
PROBE_BATCH="${PROBE_BATCH:-4}"

# --- which arms this job runs ------------------------------------------------
# Stage names: grace graphmae bgrl cldg ddgcl idol maskdgnn dysubc dyg2vec
# dygwm sgjepa summary, plus jepa_<arm> for every arm in JEPA_ARMS.
# ARMS=all runs the baselines and JEPA_ARMS_OLD, ARMS=all_jepa adds
# JEPA_ARMS_NEW, ARMS=new_jepa runs only JEPA_ARMS_NEW. Stages can also be
# listed explicitly: ARMS="jepa_ijepa jepa_lejepa".
JEPA_ARMS_OLD="frozen naive ema vicreg input"
JEPA_ARMS_NEW="ctx tnorm ema_cos sigreg lejepa ijepa lejepa_full ema_tnorm ema_cos_tnorm"
JEPA_ARMS="$JEPA_ARMS_OLD $JEPA_ARMS_NEW"
ARMS="${ARMS:-all}"
want() {
    [[ " $ARMS " == *" all "* || " $ARMS " == *" $1 "* ]] && return 0
    case "$1" in
        jepa_*)
            local a="${1#jepa_}"
            [[ " $ARMS " == *" all_jepa "* ]] && return 0
            [[ " $ARMS " == *" new_jepa "* && " $JEPA_ARMS_NEW " == *" $a "* ]] && return 0
            ;;
    esac
    return 1
}
need_encoder() {
    local a
    for a in $JEPA_ARMS; do want "jepa_$a" && return 0; done
    return 1
}

# The JODIE CSV must exist. Compute nodes usually have no internet, so this
# fails with the login-node command rather than half-downloading.
CSV="dataset/jodie/${DATASET}.csv"
case "$DATASET" in
    mooc|reddit|wikipedia)
        if [[ ! -f "$CSV" ]]; then
            echo "[FATAL] $CSV missing. From the LOGIN node run:" >&2
            echo "  cd $REPO_ROOT && wget http://snap.stanford.edu/jodie/${DATASET}.csv -O $CSV" >&2
            exit 10
        fi ;;
    *) ;;   # synthetic_dyn needs no file; cldg_* is checked by _dataset_config.sh
esac

# Shared subset / binning / encoder / window / early-stop config, also sourced
# by the forecasting pipeline. Normalises mooc -> jodie_mooc.
TASK=state_probe
source "$REPO_ROOT/scripts/dblp/_dataset_config.sh"
BATCH="${BATCH:-8}"
SHORT="${DATASET#jodie_}"
# MaskDGNN = the table's shared spatial encoder + its frequency layer; follow
# ENC_KIND so a root-weight table (ENC_KIND=gcn_root) keeps that relation.
MASKDGNN_KIND=maskdgnn
[[ "$ENC_KIND" == "gcn_root" || "$ENC_KIND" == "maskdgnn_root" ]] && MASKDGNN_KIND=maskdgnn_root
# DyG-WM = the shared spatial encoder + its trajectory GRUs; same rule.
DYGWM_KIND=dygwm
[[ "$ENC_KIND" == "gcn_root" || "$ENC_KIND" == "maskdgnn_root" || "$ENC_KIND" == "dygwm_root" ]] && DYGWM_KIND=dygwm_root

BASE_HP="${BASE_HP:-}"
BASELINE_HP_cldg="${BASELINE_HP_cldg:-}"
BASELINE_HP_ddgcl="${BASELINE_HP_ddgcl:-}"
BASELINE_HP_idol="${BASELINE_HP_idol:-}"
BASELINE_HP_maskdgnn="${BASELINE_HP_maskdgnn:-}"
BASELINE_HP_dysubc="${BASELINE_HP_dysubc:-}"
BASELINE_HP_dyg2vec="${BASELINE_HP_dyg2vec:-}"
BASELINE_HP_dygwm="${BASELINE_HP_dygwm:-}"
BASELINE_HP_sgjepa="${BASELINE_HP_sgjepa:-}"
BASELINE_HP_grace="${BASELINE_HP_grace:-}"
BASELINE_HP_graphmae="${BASELINE_HP_graphmae:-}"
BASELINE_HP_bgrl="${BASELINE_HP_bgrl:-}"

# --- predictor recipe --------------------------------------------------------
# Residual head (copy-last is the training floor) and transition up-weighting
# are on by default.
RESIDUAL="${RESIDUAL:-1}"
CHANGE_W="${CHANGE_W:-2.0}"
# INPUT_DELTA=1: velocity channel (learned projection of z_t - z_{t-1} added to
# the history tokens; --input_delta). Off by default.
INPUT_DELTA="${INPUT_DELTA:-0}"
# HORIZON_NORM=1: per-horizon loss normalisation (--horizon_norm). Off by default.
HORIZON_NORM="${HORIZON_NORM:-0}"
# LINEAR_SKIP=1: linear-dynamics prior from the history latents (--linear_skip).
LINEAR_SKIP="${LINEAR_SKIP:-0}"
# LINEAR_SKIP_INIT=ridge: closed-form least-squares start for the skip (frozen
# encoder); LINEAR_SKIP_FREEZE=1 keeps it fixed so the transformer learns the residual.
LINEAR_SKIP_INIT="${LINEAR_SKIP_INIT:-zero}"
LINEAR_SKIP_FREEZE="${LINEAR_SKIP_FREEZE:-0}"
# JEPA_EXTRA: extra train_delos.py flags appended last to every JEPA arm (for
# example the two-level flags used by scripts/synthetic/hier_showcase.sh). Use
# it together with JEPA_TAG.
JEPA_EXTRA="${JEPA_EXTRA:-}"
# STATE_PROBE=0: train the JEPA arms only, skip their state probe. Needed for
# a COARSE-level run (--coarse_window), whose windows are not in fine bins, so
# the per-bin labels the probe reads would be misaligned.
STATE_PROBE="${STATE_PROBE:-1}"
JEPA_BASE="--mask_scheme ${MASK_SCHEME:-node_mask}"
[[ "$RESIDUAL" == "1" ]] && JEPA_BASE="$JEPA_BASE --residual_head"
[[ "$INPUT_DELTA" == "1" ]] && JEPA_BASE="$JEPA_BASE --input_delta"
[[ "$HORIZON_NORM" == "1" ]] && JEPA_BASE="$JEPA_BASE --horizon_norm"
[[ "$LINEAR_SKIP" == "1" ]] && JEPA_BASE="$JEPA_BASE --linear_skip --linear_skip_init $LINEAR_SKIP_INIT"
[[ "$LINEAR_SKIP" == "1" && "$LINEAR_SKIP_FREEZE" == "1" ]] && JEPA_BASE="$JEPA_BASE --linear_skip_freeze"
[[ "$CHANGE_W" != "0" ]] && JEPA_BASE="$JEPA_BASE --change_weight $CHANGE_W"

# One results root per data configuration: sweeping a generator knob (SYN_*)
# into the same ROOT overwrites cells and reuses the wrong shared encoder.
ROOT="${ROOT:-outputs/cls_${SHORT}}"
mkdir -p "$ROOT"
d() { echo "$ROOT/$1_s${SEED}"; }

# The structural attention mask stays on (trainer default): sparse edge-list
# attention keeps memory linear in the number of allowed pairs.
JEPA_COMMON="$DATA_ARG $WINDOW_ARG $EARLY_STOP_ARG $ENCODER_ARG \
    --num_masked_nodes ${NMASK:-30} --depth ${DEPTH:-4} --heads ${HEADS:-4} \
    --mlp_dim ${MLP_DIM:-512} --batch_size $BATCH --epochs ${JEPA_EPOCHS:-300} \
    --lr ${LR:-5e-4} --probe_epochs ${JEPA_PROBE_EPOCHS:-3} \
    --downstream_task $DOWNSTREAM \
    --seed $SEED --num_workers $NW --device cuda"

# Probe task per dataset. synthetic_dyn is dynamic multiclass unless
# SYN_STATE_TARGET=switch (binary switch event). The CLDG graphs carry one
# static class per node, scored on held-out nodes. Every dataset not listed
# below uses the binary state label.
PROBE_LABELS=""
if [[ "$DATASET" == "synthetic_dyn" && "${SYN_STATE_TARGET:-community}" != "switch" ]]; then
    PROBE_TASK="multiclass"
    AGG_METRICS="--metrics acc_test macro_f1_test"
elif [[ "$DATASET" == cldg_* ]]; then
    PROBE_TASK="multiclass"
    PROBE_LABELS="--label_source dataset"
    AGG_METRICS="--metrics acc_test macro_f1_test"
elif [[ "$DATASET" == "chase_music" ]]; then
    # Five genres per month: dynamic multiclass with a label-persistence
    # floor and switch / stay subsets, like synthetic_dyn.
    PROBE_TASK="multiclass"
    AGG_METRICS="--metrics acc_test macro_f1_test subset_switch.acc_test"
elif [[ "$DATASET" == tgbn_* ]]; then
    PROBE_TASK="ranking"
    AGG_METRICS="--metrics ndcg10_test"
else
    PROBE_TASK="binary"
    AGG_METRICS=""
fi
PROBE_COMMON="--task $PROBE_TASK $PROBE_LABELS --state_label_mode $STATE_MODE \
    --probe_split $PROBE_SPLIT --device cuda --batch_size $PROBE_BATCH \
    --num_workers $NW"

# --- checkpoint selection ------------------------------------------------------
# Default: best.ckpt = lowest val JEPA loss. SELECT_BY_PROBE=1 keeps a snapshot
# every SNAPSHOT_EVERY epochs and picks the one with the best VALIDATION probe
# score instead. Apply it to every method in a table or to none.
SELECT_BY_PROBE="${SELECT_BY_PROBE:-0}"
SNAPSHOT_EVERY="${SNAPSHOT_EVERY:-25}"
SNAP_ARG=""; SELECT_PRUNE=""
[[ "$SELECT_BY_PROBE" == "1" ]] && SNAP_ARG="--snapshot_every $SNAPSHOT_EVERY"
[[ "${SELECT_KEEP_SNAPSHOTS:-0}" != "1" ]] && SELECT_PRUNE="--prune"

echo "[config] DATASET=$DATASET SEED=$SEED STATE_MODE=$STATE_MODE BATCH=$BATCH"
echo "[config] SELECT_BY_PROBE=$SELECT_BY_PROBE snapshot_every=$SNAPSHOT_EVERY"
echo "[config] ARMS='$ARMS' JEPA_BASE='$JEPA_BASE' BASE_HP='$BASE_HP' probe_split=$PROBE_SPLIT"
echo "[config] DOWNSTREAM=$DOWNSTREAM COLLAPSE_CHECK=$COLLAPSE_CHECK JEPA_TAG='${JEPA_TAG:-}'"
echo "[config] JEPA_EXTRA='$JEPA_EXTRA' STATE_PROBE=$STATE_PROBE"

fail() { echo "[FATAL] $*" >&2; exit "${2:-1}"; }

# --- 1. shared SSL encoder (edge_mask), the start point of every JEPA arm ----
# pretrain_encoder.py writes a templated filename, not encoder.ckpt.
ENC_DIR="$ROOT/_shared_encoder_s${SEED}"
ENC_CKPT="$ENC_DIR/encoder_${DATASET}_${ENC_KIND}_edge_mask.ckpt"
mkdir -p "$ENC_DIR"
if need_encoder; then
    # flock: with one arm per job, several jobs reach this at once and would
    # otherwise pretrain into the same path.
    exec 9>"$ENC_DIR/.pretrain.lock"
    flock -w 21600 9 || fail "encoder lock timeout" 10
    # The checkpoint is saved on every val improvement, so only `.done`
    # (touched after a clean exit) marks a finished pretrain. A checkpoint
    # without the marker is retrained unless ENC_TRUST=1.
    if [[ -f "$ENC_CKPT" && -f "$ENC_DIR/.done" ]]; then
        echo "[skip] shared encoder complete: $ENC_CKPT"
    elif [[ -f "$ENC_CKPT" && "${ENC_TRUST:-0}" == "1" ]]; then
        echo "[skip] shared encoder exists without .done marker; ENC_TRUST=1 -> keeping it"
        touch "$ENC_DIR/.done"
    else
        [[ -f "$ENC_CKPT" ]] && echo "[resume] $ENC_CKPT has no .done marker (interrupted run?) -> retraining"
        echo; echo "=== shared encoder pretrain (lock held) -> $ENC_DIR ==="
        rm -f "$ENC_CKPT"
        if srun --export=ALL "$PY" src/train/pretrain_encoder.py $DATA_ARG \
            $ENCODER_ARG $WINDOW_ARG $EARLY_STOP_ARG \
            --pretext edge_mask --decoder_hidden 64 --mask_rate 0.25 \
            --batch_size "$BATCH" --epochs "${ENC_EPOCHS:-300}" --lr 1e-3 \
            --seed "$SEED" --num_workers "$NW" --device cuda --out_dir "$ENC_DIR" \
           && [[ -f "$ENC_CKPT" ]]; then
            touch "$ENC_DIR/.done"
        else
            flock -u 9; fail "encoder pretrain failed / produced no $ENC_CKPT" 11
        fi
    fi
    flock -u 9; exec 9>&-
fi

# --- 2. DELOS encoder-training variants --------------------------------------
arm_flags() {
    case "$1" in
        frozen)      echo "" ;;
        naive)       echo "--train_encoder" ;;
        ema)         echo "--train_encoder --ema_target" ;;
        vicreg)      echo "--train_encoder --vicreg_cov 1.0 --vicreg_var 5.0" ;;
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

for ARM in $JEPA_ARMS; do
    want "jepa_$ARM" || continue
    VFLAGS="$(arm_flags "$ARM")"
    # A sourced script clobbering $ARM would silently mislabel a whole column.
    [[ "$VFLAGS" == "__UNKNOWN__" ]] && fail "unknown arm '$ARM'" 12
    eval "HP_ARGS=\${VARIANT_HP_${ARM}:-}"
    # JEPA_TAG runs a predictor variant of an arm (e.g. MASK_SCHEME=future_forecast
    # JEPA_TAG=ff) into its own directory, sharing the root's baselines and encoder.
    OUT="$(d "jepa_${ARM}${JEPA_TAG:+_$JEPA_TAG}")"
    mkdir -p "$OUT"
    echo; echo "=== DELOS [$ARM] -> $OUT ==="
    echo "    flags='$VFLAGS $HP_ARGS'"
    # An arm is complete only once the trainer wrote test_metrics.json.
    # best.ckpt alone is written on every val improvement, so a job killed by
    # the walltime leaves one behind. There is no epoch-level resume: an
    # interrupted arm restarts from scratch.
    if [[ -f "$OUT/best.ckpt" && -f "$OUT/test_metrics.json" ]]; then
        echo "[skip] $OUT complete (best.ckpt + test_metrics.json)"
    else
        if [[ -f "$OUT/best.ckpt" ]]; then
            echo "[resume] $OUT has best.ckpt but no test_metrics.json (interrupted) -> restarting from scratch"
            rm -f "$OUT/best.ckpt" "$OUT/history.json"
        fi
        # Order matters: base recipe, tuned predictor HP, variant flags, then
        # the variant's own knobs -- argparse honours the last occurrence.
        srun --export=ALL "$PY" src/train/train_delos.py $JEPA_COMMON \
            --encoder_ckpt "$ENC_CKPT" $JEPA_BASE $BASE_HP $VFLAGS $HP_ARGS \
            $JEPA_EXTRA $SNAP_ARG --out_dir "$OUT" || { echo "[warn] arm $ARM training failed"; continue; }
        [[ -f "$OUT/best.ckpt" && -f "$OUT/test_metrics.json" ]] \
            || { echo "[warn] arm $ARM did not finish (no best.ckpt/test_metrics.json)"; continue; }
    fi

    # SELECT_BY_PROBE=1: every snapshot and the loss-selected checkpoint are
    # probed on val; the winner becomes best.ckpt and the original is kept as
    # best_loss.ckpt. Skipped once ckpt_selection.json exists.
    if [[ "$SELECT_BY_PROBE" == "1" ]]; then
        if [[ -f "$OUT/ckpt_selection.json" ]]; then
            echo "[skip] checkpoint already selected on the val probe"
        elif compgen -G "$OUT/snapshots/ep*/best.ckpt" >/dev/null; then
            echo "=== checkpoint selection on the VAL probe [$ARM] ==="
            "$PY" scripts/classification/select_ckpt_by_probe.py --run_dir "$OUT" --stage \
              && srun --export=ALL "$PY" scripts/dblp/run_block_probe.py \
                    --pattern "$OUT/snapshots/*" --task $PROBE_TASK $PROBE_LABELS \
                    --state_label_mode $STATE_MODE --probe_split val --device cuda \
                    --batch_size $PROBE_BATCH --num_workers $NW \
              && "$PY" scripts/classification/select_ckpt_by_probe.py --run_dir "$OUT" \
                    --apply $SELECT_PRUNE \
              || echo "[warn] probe-based selection failed for $ARM; keeping the loss-selected ckpt"
        else
            echo "[warn] SELECT_BY_PROBE=1 but $OUT has no snapshots (trained without" \
                 "SNAPSHOT_EVERY?) -> keeping the loss-selected ckpt"
        fi
    fi

    if [[ "$STATE_PROBE" != "1" ]]; then
        echo "[skip] STATE_PROBE=0 -> no state probe for $ARM"; continue
    fi
    echo "=== state probe [$ARM] (jepa + encoder_last + oracle + degree) ==="
    srun --export=ALL "$PY" scripts/dblp/run_block_probe.py \
        --jepa_ckpt "$OUT/best.ckpt" $PROBE_COMMON \
        || echo "[warn] state probe failed for $ARM"

    # Collapse check for the arms that train the encoder: a collapsed encoder
    # leaves the probe at chance however good the training loss looked.
    if [[ "$COLLAPSE_CHECK" == "1" && "$VFLAGS" == *"--train_encoder"* ]]; then
        srun --export=ALL "$PY" -m src.train.analyze_jepa --skip_diagnostic \
            --link_sources jepa $DATA_ARG $WINDOW_ARG \
            --jepa_ckpt "$OUT/best.ckpt" --device cuda --seed "$SEED" \
            --out_dir "$OUT/collapse_check" \
            || echo "[warn] collapse check failed for $ARM (non-fatal)"
    fi
done

# --- 3. baselines: pretrain the encoder, then the same frozen state probe ----
for M in grace graphmae bgrl cldg ddgcl idol maskdgnn dysubc dyg2vec dygwm sgjepa; do
    want "$M" || continue
    D="$(d "$M")"
    mkdir -p "$D"
    eval "HP=\${BASELINE_HP_${M}:-}"
    if [[ ! -f "$D/encoder.ckpt" ]]; then
        echo; echo "=== ${M^^} pretrain -> $D ==="
        case "$M" in
          graphmae)
            # Static generative control: one snapshot per step, reference
            # Cora defaults. Reconstructs every input channel, as published.
            srun --export=ALL "$PY" -m src.baselines.graphmae.train_graphmae $DATA_ARG \
                $ENCODER_ARG $WINDOW_ARG $EARLY_STOP_ARG \
                --mask_rate 0.5 --replace_rate 0.05 --sce_gamma 2 --target all \
                --steps_per_epoch 20 --val_steps 10 \
                --epochs "${BASE_EPOCHS:-100}" --lr 1e-3 \
                --device cuda --seed "$SEED" --out_dir "$D" $HP ;;
          bgrl)
            # Static negative-free control: one snapshot per step, reference
            # WikiCS config. Val-loss selection like every other arm; pass
            # --select last via BASELINE_HP_bgrl for the paper's fixed budget.
            srun --export=ALL "$PY" -m src.baselines.bgrl.train_bgrl $DATA_ARG \
                $ENCODER_ARG $WINDOW_ARG $EARLY_STOP_ARG \
                --p_e1 0.2 --p_e2 0.3 --p_f1 0.2 --p_f2 0.1 --ema_momentum 0.99 $ANNEAL_ARG \
                --steps_per_epoch 20 --val_steps 10 \
                --epochs "${BASE_EPOCHS:-100}" --lr 5e-4 \
                --device cuda --seed "$SEED" --out_dir "$D" $HP ;;
          grace)
            # Static contrastive control: two augmented views of one snapshot,
            # reference Cora defaults.
            srun --export=ALL "$PY" -m src.baselines.grace.train_grace $DATA_ARG \
                $ENCODER_ARG $WINDOW_ARG $EARLY_STOP_ARG \
                --tau 0.4 --p_e1 0.2 --p_e2 0.4 --p_f1 0.3 --p_f2 0.4 \
                --node_batch_size 256 --steps_per_epoch 20 --val_steps 10 \
                --epochs "${BASE_EPOCHS:-100}" --lr 5e-4 \
                --device cuda --seed "$SEED" --out_dir "$D" $HP ;;
          cldg)
            srun --export=ALL "$PY" -m src.baselines.cldg.train_cldg $DATA_ARG \
                $ENCODER_ARG $WINDOW_ARG $EARLY_STOP_ARG \
                --snapshots 4 --views 2 --strategy random --tau 0.07 \
                --node_batch_size 256 --steps_per_epoch 20 --val_steps 10 \
                --epochs "${BASE_EPOCHS:-100}" --lr 4e-3 \
                --device cuda --seed "$SEED" --out_dir "$D" $HP ;;
          ddgcl)
            # `--tau` is the debias class prior (the reference's tau_plus);
            # there is no --tau_plus flag.
            srun --export=ALL "$PY" -m src.baselines.ddgcl.train_ddgcl $DATA_ARG \
                $ENCODER_ARG $WINDOW_ARG $EARLY_STOP_ARG \
                --max_gap 4 --time_dim 32 --tau 0.01 --beta 1.5 \
                --node_batch_size 256 --steps_per_epoch 20 --val_steps 10 \
                --epochs "${BASE_EPOCHS:-100}" --lr 4e-3 \
                --device cuda --seed "$SEED" --out_dir "$D" $HP ;;
          idol)
            # --steps_per_epoch caps IDOL to the budget CLDG/DDGCL get; uncapped
            # it visits every training snapshot per epoch.
            srun --export=ALL "$PY" -m src.baselines.idol.train_idol $DATA_ARG \
                $ENCODER_ARG $WINDOW_ARG $EARLY_STOP_ARG \
                --proj_layers 1 --node_batch_size 256 \
                --steps_per_epoch 20 --val_steps 10 \
                --epochs "${BASE_EPOCHS:-100}" --lr 1e-3 \
                --device cuda --seed "$SEED" --out_dir "$D" $HP ;;
          maskdgnn)
            # Own encoder kind (the `--encoder_kind` after $ENCODER_ARG wins).
            # --window defaults to HIST so the frequency filter sees the
            # probe's window length. No end-to-end link row here.
            srun --export=ALL "$PY" -m src.baselines.maskdgnn.train_maskdgnn $DATA_ARG \
                $ENCODER_ARG --encoder_kind "$MASKDGNN_KIND" $WINDOW_ARG $EARLY_STOP_ARG \
                --mask_ratio 0.2 --alpha 0.7 --target next --neg_per_pos 1 \
                --steps_per_epoch 20 --val_steps 10 --e2e_eval none \
                --epochs "${BASE_EPOCHS:-100}" --lr 4e-3 \
                --device cuda --seed "$SEED" --out_dir "$D" $HP ;;
          dysubc)
            # --lookback defaults to HIST, so the time-weighted graph spans
            # exactly the window the probe hands the encoder.
            srun --export=ALL "$PY" -m src.baselines.dysubc.train_dysubc $DATA_ARG \
                $ENCODER_ARG $WINDOW_ARG $EARLY_STOP_ARG \
                --subgraph_size 20 --alpha 5 --beta 1 --margin 0.75 --lam 0.5 \
                --node_batch_size 256 --steps_per_epoch 20 --val_steps 10 \
                --epochs "${BASE_EPOCHS:-100}" --lr 1e-3 \
                --device cuda --seed "$SEED" --out_dir "$D" $HP ;;
          dyg2vec)
            # Own encoder kind (the `--encoder_kind` after $ENCODER_ARG wins).
            # --window defaults to HIST; the other values are paper defaults.
            srun --export=ALL "$PY" -m src.baselines.dyg2vec.train_dyg2vec $DATA_ARG \
                $ENCODER_ARG --encoder_kind dyg2vec $WINDOW_ARG $EARLY_STOP_ARG \
                --p_edge 0.3 --p_struct 0.3 --lam 25 --mu 25 --nu 1 \
                --steps_per_epoch 20 --val_steps 10 \
                --epochs "${BASE_EPOCHS:-100}" --lr 1e-4 \
                --device cuda --seed "$SEED" --out_dir "$D" $HP ;;
          dygwm)
            # Own encoder kind (the `--encoder_kind` after $ENCODER_ARG wins).
            # SSL-only (no link / rank loss), paper node-task defaults; var /
            # cov keep the paper's ratios to the JEPA terms.
            srun --export=ALL "$PY" -m src.baselines.dygwm.train_dygwm $DATA_ARG \
                $ENCODER_ARG --encoder_kind "$DYGWM_KIND" $WINDOW_ARG $EARLY_STOP_ARG \
                --ema_momentum 0.995 --n_ctx 32 --decay 0.8 --cn_bonus 1.0 \
                --lam_node 1 --lam_rel 1 --lam_var 0.25 --lam_cov 0.025 \
                --node_batch_size 256 --pair_batch_size 256 \
                --steps_per_epoch 20 --val_steps 10 \
                --epochs "${BASE_EPOCHS:-100}" --lr 1e-3 \
                --device cuda --seed "$SEED" --out_dir "$D" $HP ;;
          sgjepa)
            # Shared per-snapshot GCN as the stop-gradient target encoder,
            # spiking context encoder on its latents. The paper gives no
            # temperature or optimiser: tau 0.1 and AdamW 1e-3 are our choice.
            srun --export=ALL "$PY" -m src.baselines.sgjepa.train_sgjepa $DATA_ARG \
                $ENCODER_ARG $WINDOW_ARG $EARLY_STOP_ARG \
                --snn_layers 1 --tau 0.1 --init_tau 2.0 --v_th 1.0 \
                --node_batch_size 256 --steps_per_epoch 20 --val_steps 10 \
                --epochs "${BASE_EPOCHS:-100}" --lr 1e-3 \
                --device cuda --seed "$SEED" --out_dir "$D" $HP ;;
        esac
        # Hard-fail: a missing row reads as "not applicable", not "this crashed".
        [[ -f "$D/encoder.ckpt" ]] || fail "${M^^} pretrain failed" 13
    else
        echo "[skip] $D/encoder.ckpt exists"
    fi
    echo; echo "=== state probe: frozen $M encoder -> $D ==="
    srun --export=ALL "$PY" scripts/dblp/run_block_probe.py \
        --encoder_ckpt "$D/encoder.ckpt" $PROBE_COMMON \
        || echo "[warn] state probe failed for $D"
done

# --- 4. summary for this seed ------------------------------------------------
if want summary; then
    echo; echo "=== seed $SEED summary ==="
    srun --export=ALL "$PY" scripts/classification/aggregate_classification.py \
        --dataset "$SHORT" --seeds "$SEED" $AGG_METRICS || true
fi

echo; echo "[done] $DATASET seed $SEED -> $ROOT"
echo "Aggregate across seeds with:"
echo "  python scripts/classification/aggregate_classification.py \\"
echo "      --dataset $SHORT --seeds 42 43 44 --latex $AGG_METRICS"
