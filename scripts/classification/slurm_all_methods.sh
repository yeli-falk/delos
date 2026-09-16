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
# DYNAMIC NODE CLASSIFICATION: every method, one seed per job.
# The classification twin of scripts/regression/slurm_all_methods.sh -- same
# ARMS gating, same tuned-flag plumbing, same one-arm-per-job option.
#
#   DATASET=mooc|reddit|wikipedia   (JODIE state_label; binning/subset and the
#                                    encoder come from _dataset_config.sh)
#   DATASET=synthetic_dyn           (latent-driven generator, dense MULTICLASS
#                                    node state = nearest community anchor;
#                                    knobs SYN_* in _dataset_config.sh. Probe
#                                    is --task multiclass: accuracy + macro-F1
#                                    overall and on the switch / stay rows,
#                                    with label_persistence as the floor.
#                                    NPRED=3 scores horizons h0..h2 too.)
#
# Arms produced (each writes its own state_probe.json):
#   cldg / ddgcl / idol   contrastive SSL encoder, probed on FROZEN latents.
#   maskdgnn              MaskDGNN (He et al., IJCAI'25): activeness-aware edge
#                         masking + masked/next-edge BCE on a GCN with an
#                         adaptive frequency layer; no state label is ever used,
#                         so its encoder is probed on FROZEN latents like the
#                         three above.
#   dysubc                DySubC (Chen et al., TKDD'23): temporal subgraph
#                         contrast -- BFS subgraphs ranked by recency + degree,
#                         a time-weighted and an unweighted subgraph encoder,
#                         margin losses against a shuffled and an unweighted
#                         summary. Encoder probed on FROZEN latents like the
#                         three contrastive arms.
#   jepa_frozen           SSL encoder frozen, JEPA predictor on top. Targets are
#                         fixed, so latent collapse is impossible by
#                         construction. The reference arm.
#   jepa_naive            --train_encoder alone. Targets come from the SAME
#                         encoder being optimised, so collapsing to a point is a
#                         global optimum. Exists so a reader can tell whether
#                         `ema` won because of EMA or merely because the encoder
#                         was unfrozen.
#   jepa_ema              --train_encoder --ema_target (I-JEPA).
#   jepa_vicreg           --train_encoder + VICReg variance/covariance.
#   jepa_input            --predict_space input: the SAME frozen encoder,
#                         predictor and masking, but trained to reconstruct the
#                         raw features of the target step instead of matching
#                         the encoder's latent. The control that isolates the
#                         JEPA objective (prediction SPACE) from everything
#                         else; probed on its pre-decoder latent like the rest.
#
# NOT separate arms, because run_block_probe.py already reports them as COLUMNS
# inside every state_probe.json -- they cost nothing and are matched per seed:
#   encoder_last              the frozen encoder at the last history bin
#                             (copy-last / the persistence analogue)
#   encoder_oracle            an encoder that SAW the target bin (upper bound)
#   degree_history_baseline   degree-at-each-history-bin (activity-trend control)
# That is why there is no `jepa_enc` arm here: the DyG-JEPA run's own
# `encoder_last` column IS the encoder-without-predictor control, measured on
# the same rows in the same run -- a strictly tighter comparison than the
# separate arm the forecasting pipeline needs.
#
# Label semantics: STATE_MODE=state (dynamic cumulative, the practical default)
# | ever (static, legacy) | event (instantaneous, very sparse).
# ~3% positive rate, so the headline is AUROC/AUPRC, NEVER accuracy.
#
# Usage:
#   DATASET=mooc SEED=42 sbatch scripts/classification/slurm_all_methods.sh
#   for S in 42 43 44; do DATASET=mooc SEED=$S sbatch scripts/classification/slurm_all_methods.sh; done
#   python scripts/classification/aggregate_classification.py --dataset mooc --seeds 42 43 44 --latex
#
# One arm per job (the JEPA arms are independent once the shared encoder exists):
#   DATASET=mooc SEED=42 ARMS=jepa_ema sbatch scripts/classification/slurm_all_methods.sh
#
# Tuned flags from scripts/classification/slurm_hparam_search.sh:
#   BASE_HP="--lr 1e-3 --num_masked_nodes 30" VARIANT_HP_ema="--encoder_lr 1e-4" \
#   BASELINE_HP_cldg="--lr 1e-3 --views 4" ... sbatch ...

set -uo pipefail
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

DATASET="${DATASET:-mooc}"
case "$DATASET" in
    mooc|reddit|wikipedia|synthetic_dyn) ;;
    *) echo "[FATAL] DATASET must be mooc, reddit, wikipedia or synthetic_dyn, got '$DATASET'" >&2
       exit 2 ;;
esac
SEED="${SEED:-42}"
NW="${SLURM_CPUS_PER_TASK:-8}"
PROBE_SPLIT="${PROBE_SPLIT:-test}"
STATE_MODE="${STATE_MODE:-state}"
PROBE_BATCH="${PROBE_BATCH:-4}"

# --- which arms this job runs ------------------------------------------------
# Default "all" runs the whole table in one job. Split with ARMS= when a single
# JEPA arm is long enough that four of them will not fit the walltime.
# Stage names: cldg ddgcl idol maskdgnn dysubc summary, plus jepa_<arm> for every arm in
# JEPA_ARMS below. ARMS=all runs the legacy five JEPA arms + baselines (the old
# table); ARMS=all_jepa adds the JEPA-recipe arms too; ARMS=new_jepa runs only
# those. Or list stages explicitly: ARMS="jepa_ijepa jepa_lejepa".
JEPA_ARMS_OLD="frozen naive ema vicreg input"
JEPA_ARMS_NEW="ctx tnorm ema_cos sigreg lejepa ijepa lejepa_full"
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
if [[ "$DATASET" != "synthetic_dyn" && ! -f "$CSV" ]]; then
    echo "[FATAL] $CSV missing. From the LOGIN node run:" >&2
    echo "  cd $REPO_ROOT && wget http://snap.stanford.edu/jodie/${DATASET}.csv -O $CSV" >&2
    exit 10
fi

# Shared subset / binning / encoder / window / early-stop config -- the same
# file the forecasting pipeline and the existing state scripts source, so all
# three tables describe the same experiment. Normalises mooc -> jodie_mooc.
TASK=state_probe
source "$REPO_ROOT/scripts/dblp/_dataset_config.sh"
BATCH="${BATCH:-8}"
SHORT="${DATASET#jodie_}"

BASE_HP="${BASE_HP:-}"
BASELINE_HP_cldg="${BASELINE_HP_cldg:-}"
BASELINE_HP_ddgcl="${BASELINE_HP_ddgcl:-}"
BASELINE_HP_idol="${BASELINE_HP_idol:-}"
BASELINE_HP_maskdgnn="${BASELINE_HP_maskdgnn:-}"
BASELINE_HP_dysubc="${BASELINE_HP_dysubc:-}"

# --- predictor recipe --------------------------------------------------------
# The MOOC-validated fix: residual head (copy-last is the training floor) plus
# transition-upweighting. Defaults ON here because that is the configuration the
# state results were established with.
RESIDUAL="${RESIDUAL:-1}"
CHANGE_W="${CHANGE_W:-2.0}"
JEPA_BASE="--mask_scheme ${MASK_SCHEME:-node_mask}"
[[ "$RESIDUAL" == "1" ]] && JEPA_BASE="$JEPA_BASE --residual_head"
[[ "$CHANGE_W" != "0" ]] && JEPA_BASE="$JEPA_BASE --change_weight $CHANGE_W"

# One results root per DATA CONFIGURATION. Sweeping a generator knob (SYN_*)
# must go to its own ROOT, or cells overwrite each other and silently reuse the
# previous configuration's shared encoder: scripts/synthetic/sweep_state.sh
# sets ROOT per axis value for exactly this reason.
ROOT="${ROOT:-outputs/cls_${SHORT}}"
mkdir -p "$ROOT"
d() { echo "$ROOT/$1_s${SEED}"; }

# Structural attention mask ON (trainer default, sparse edge-list attention:
# memory is linear in #allowed pairs, so full-N MOOC fits). Earlier runs of
# this script passed --no_attention_mask; those results live in the old ROOT
# and are NOT comparable with runs from this version -- use a fresh ROOT.
JEPA_COMMON="$DATA_ARG $WINDOW_ARG $EARLY_STOP_ARG $ENCODER_ARG \
    --num_masked_nodes ${NMASK:-30} --depth ${DEPTH:-4} --heads ${HEADS:-4} \
    --mlp_dim ${MLP_DIM:-512} --batch_size $BATCH --epochs ${JEPA_EPOCHS:-300} \
    --lr ${LR:-5e-4} --probe_epochs ${JEPA_PROBE_EPOCHS:-3} \
    --seed $SEED --num_workers $NW --device cuda"

# synthetic_dyn exposes a dynamic MULTICLASS state (which community next) by
# default, or -- with SYN_STATE_TARGET=switch -- a BINARY switch event (does the
# node cross a basin boundary at the scored step), which uses the same AUROC /
# AUPRC probe as the JODIE state label and is directly comparable to the MOOC
# table. Everything else is the JODIE binary label. All write state_probe.json.
if [[ "$DATASET" == "synthetic_dyn" && "${SYN_STATE_TARGET:-community}" != "switch" ]]; then
    PROBE_TASK="multiclass"
    AGG_METRICS="--metrics acc_test macro_f1_test"
else
    PROBE_TASK="binary"
    AGG_METRICS=""
fi
PROBE_COMMON="--task $PROBE_TASK --state_label_mode $STATE_MODE \
    --probe_split $PROBE_SPLIT --device cuda --batch_size $PROBE_BATCH \
    --num_workers $NW"

echo "[config] DATASET=$DATASET SEED=$SEED STATE_MODE=$STATE_MODE BATCH=$BATCH"
echo "[config] ARMS='$ARMS' JEPA_BASE='$JEPA_BASE' BASE_HP='$BASE_HP' probe_split=$PROBE_SPLIT"

fail() { echo "[FATAL] $*" >&2; exit "${2:-1}"; }

# =============================================================================
# 1. shared SSL encoder (edge_mask) -- the start point for all four variants
# =============================================================================
# pretrain_encoder.py writes a TEMPLATED filename, not encoder.ckpt.
ENC_DIR="$ROOT/_shared_encoder_s${SEED}"
ENC_CKPT="$ENC_DIR/encoder_${DATASET}_${ENC_KIND}_edge_mask.ckpt"
mkdir -p "$ENC_DIR"
if need_encoder; then
    # flock: one-arm-per-job means several jobs reach this at once and would
    # otherwise all pretrain into the SAME path, racing on one file.
    exec 9>"$ENC_DIR/.pretrain.lock"
    flock -w 21600 9 || fail "encoder lock timeout" 10
    # pretrain_encoder.py saves the ckpt on EVERY val improvement, so a ckpt
    # alone cannot distinguish a finished pretrain from one killed by the
    # walltime. `.done` is touched only after the run returned 0. A ckpt with
    # no marker is an older or interrupted run: retrain it (the ckpt would be
    # overwritten from epoch 0 anyway), unless ENC_TRUST=1 says to keep it.
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

# =============================================================================
# 2. DyG-JEPA encoder-training variants
# =============================================================================
arm_flags() {
    case "$1" in
        frozen)      echo "" ;;
        naive)       echo "--train_encoder" ;;
        ema)         echo "--train_encoder --ema_target" ;;
        vicreg)      echo "--train_encoder --vicreg_cov 1.0 --vicreg_var 5.0" ;;
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

for ARM in $JEPA_ARMS; do
    want "jepa_$ARM" || continue
    VFLAGS="$(arm_flags "$ARM")"
    # A sourced script clobbering $ARM would silently mislabel a whole column.
    [[ "$VFLAGS" == "__UNKNOWN__" ]] && fail "unknown arm '$ARM'" 12
    eval "HP_ARGS=\${VARIANT_HP_${ARM}:-}"
    OUT="$(d "jepa_$ARM")"
    mkdir -p "$OUT"
    echo; echo "=== DyG-JEPA [$ARM] -> $OUT ==="
    echo "    flags='$VFLAGS $HP_ARGS'"
    # An arm is COMPLETE only when the trainer reached its end: it writes
    # test_metrics.json after the last epoch and the link probe. best.ckpt on
    # its own is written on every val improvement, so a job killed by the
    # walltime leaves one behind -- treating that as done would silently put an
    # under-trained arm in the table. There is no epoch-level resume (no
    # optimiser state is saved), so an interrupted arm restarts from scratch.
    if [[ -f "$OUT/best.ckpt" && -f "$OUT/test_metrics.json" ]]; then
        echo "[skip] $OUT complete (best.ckpt + test_metrics.json)"
    else
        if [[ -f "$OUT/best.ckpt" ]]; then
            echo "[resume] $OUT has best.ckpt but no test_metrics.json (interrupted) -> restarting from scratch"
            rm -f "$OUT/best.ckpt" "$OUT/history.json"
        fi
        # Order matters: base recipe, tuned predictor HP, variant flags, then
        # the variant's own knobs -- argparse honours the last occurrence.
        srun --export=ALL "$PY" src/train/train_dyg_jepa.py $JEPA_COMMON \
            --encoder_ckpt "$ENC_CKPT" $JEPA_BASE $BASE_HP $VFLAGS $HP_ARGS \
            --out_dir "$OUT" || { echo "[warn] arm $ARM training failed"; continue; }
        [[ -f "$OUT/best.ckpt" && -f "$OUT/test_metrics.json" ]] \
            || { echo "[warn] arm $ARM did not finish (no best.ckpt/test_metrics.json)"; continue; }
    fi

    echo "=== state probe [$ARM] (jepa + encoder_last + oracle + degree) ==="
    srun --export=ALL "$PY" scripts/dblp/run_block_probe.py \
        --jepa_ckpt "$OUT/best.ckpt" $PROBE_COMMON \
        || echo "[warn] state probe failed for $ARM"

    # Collapse check for the unfrozen arms. If the encoder collapsed, every
    # node's latent is (nearly) the same point: the probe cannot separate
    # anything and AUROC sits at chance no matter how good training looked.
    # (ctx / tnorm keep the encoder frozen, so they are skipped like frozen.)
    if [[ "$VFLAGS" == *"--train_encoder"* ]]; then
        srun --export=ALL "$PY" -m src.train.analyze_jepa --skip_diagnostic \
            --link_sources jepa $DATA_ARG $WINDOW_ARG \
            --jepa_ckpt "$OUT/best.ckpt" --device cuda --seed "$SEED" \
            --out_dir "$OUT/collapse_check" \
            || echo "[warn] collapse check failed for $ARM (non-fatal)"
    fi
done

# =============================================================================
# 3. contrastive baselines: pretrain encoder -> the SAME frozen state probe
# =============================================================================
for M in cldg ddgcl idol maskdgnn dysubc; do
    want "$M" || continue
    D="$(d "$M")"
    mkdir -p "$D"
    eval "HP=\${BASELINE_HP_${M}:-}"
    if [[ ! -f "$D/encoder.ckpt" ]]; then
        echo; echo "=== ${M^^} pretrain -> $D ==="
        case "$M" in
          cldg)
            srun --export=ALL "$PY" -m src.baselines.cldg.train_cldg $DATA_ARG \
                $ENCODER_ARG $WINDOW_ARG $EARLY_STOP_ARG \
                --snapshots 4 --views 2 --strategy random --tau 0.07 \
                --node_batch_size 256 --steps_per_epoch 20 --val_steps 10 \
                --epochs "${BASE_EPOCHS:-100}" --lr 4e-3 \
                --device cuda --seed "$SEED" --out_dir "$D" $HP ;;
          ddgcl)
            # `--tau` IS the debias class prior (the reference's tau_plus).
            # Passing --tau_plus aborts the stage with an argparse error.
            srun --export=ALL "$PY" -m src.baselines.ddgcl.train_ddgcl $DATA_ARG \
                $ENCODER_ARG $WINDOW_ARG $EARLY_STOP_ARG \
                --max_gap 4 --time_dim 32 --tau 0.01 --beta 1.5 \
                --node_batch_size 256 --steps_per_epoch 20 --val_steps 10 \
                --epochs "${BASE_EPOCHS:-100}" --lr 4e-3 \
                --device cuda --seed "$SEED" --out_dir "$D" $HP ;;
          idol)
            # --steps_per_epoch caps IDOL to the same budget CLDG/DDGCL get.
            # Uncapped it visits EVERY training snapshot per epoch, which on a
            # long timeline is orders of magnitude more gradient steps at the
            # same nominal --epochs -- a table comparing budgets, not methods.
            srun --export=ALL "$PY" -m src.baselines.idol.train_idol $DATA_ARG \
                $ENCODER_ARG $WINDOW_ARG $EARLY_STOP_ARG \
                --proj_layers 1 --node_batch_size 256 \
                --steps_per_epoch 20 --val_steps 10 \
                --epochs "${BASE_EPOCHS:-100}" --lr 1e-3 \
                --device cuda --seed "$SEED" --out_dir "$D" $HP ;;
          maskdgnn)
            # Own encoder kind (shared GCN + adaptive frequency layer; the
            # `--encoder_kind` after $ENCODER_ARG wins). --window defaults to
            # HIST so the frequency filter sees the probe's window length. No
            # end-to-end link row in this pipeline (--e2e_eval none).
            srun --export=ALL "$PY" -m src.baselines.maskdgnn.train_maskdgnn $DATA_ARG \
                $ENCODER_ARG --encoder_kind maskdgnn $WINDOW_ARG $EARLY_STOP_ARG \
                --mask_ratio 0.2 --alpha 0.7 --target next --neg_per_pos 1 \
                --steps_per_epoch 20 --val_steps 10 --e2e_eval none \
                --epochs "${BASE_EPOCHS:-100}" --lr 4e-3 \
                --device cuda --seed "$SEED" --out_dir "$D" $HP ;;
          dysubc)
            # Temporal subgraph contrast on the shared GCN. --lookback
            # defaults to HIST, so the time-weighted graph spans exactly the
            # window the probe hands the encoder. No state label is used.
            srun --export=ALL "$PY" -m src.baselines.dysubc.train_dysubc $DATA_ARG \
                $ENCODER_ARG $WINDOW_ARG $EARLY_STOP_ARG \
                --subgraph_size 20 --alpha 5 --beta 1 --margin 0.75 --lam 0.5 \
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

# =============================================================================
# 4. summary for this seed
# =============================================================================
if want summary; then
    echo; echo "=== seed $SEED summary ==="
    srun --export=ALL "$PY" scripts/classification/aggregate_classification.py \
        --dataset "$SHORT" --seeds "$SEED" $AGG_METRICS || true
fi

echo; echo "[done] $DATASET seed $SEED -> $ROOT"
echo "Aggregate across seeds with:"
echo "  python scripts/classification/aggregate_classification.py \\"
echo "      --dataset $SHORT --seeds 42 43 44 --latex $AGG_METRICS"
