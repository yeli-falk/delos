#!/usr/bin/env bash
#SBATCH --job-name=reg-allmethods
#SBATCH --output=logs/reg-allmethods-%j.out
#SBATCH --error=logs/reg-allmethods-%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#
# Node-signal forecasting: every method, one seed per job. Each arm writes its
# own test_metrics.json. The arm table, the cost estimate and the full
# protocol are in scripts/regression/README.md.
#
# Usage:
#   DATASET=england_covid SEED=42 sbatch scripts/regression/slurm_all_methods.sh
#   for S in 42 43 44; do DATASET=metr_la SEED=$S sbatch scripts/regression/slurm_all_methods.sh; done
#   python scripts/regression/aggregate_regression.py --dataset metr_la --seeds 42 43 44 --latex
#
# Main environment knobs (the others are documented where they are defined):
#   DATASET      metr_la | england_covid. Window geometry comes from
#                scripts/dblp/_dataset_config.sh.
#   SEED         Default 42.
#   ARMS         Stages to run, see "which arms this job runs". Default all.
#   BASE_HP, VARIANT_HP_<arm>, BASELINE_HP_<method>
#                Tuned flags from slurm_hparam_search.sh, for example
#                BASE_HP="--lr 1e-3 --num_masked_nodes 30 --change_weight 2.0"
#                VARIANT_HP_ema="--encoder_lr 1e-4 --ema_momentum 0.996".
#
# Walltime: on METR-LA the default ARMS=all does not fit the 48 h above. Run
# one JEPA arm per job (see below) or lower JEPA_EPOCHS. The probe runs after
# training, so a walltime kill leaves no test_metrics.json.
# The final table stays at stride 1 (the DCRNN convention); --metr_la_stride is
# a search knob and is deliberately not wired in here.

set -uo pipefail
mkdir -p logs

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"
if [[ -d "$HOME/.conda/envs/cjepa" ]]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate cjepa
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

DATASET="${DATASET:-england_covid}"
case "$DATASET" in
    metr_la|england_covid) ;;
    *) echo "[FATAL] DATASET must be metr_la or england_covid, got '$DATASET'" >&2
       exit 2 ;;
esac
SEED="${SEED:-42}"
# Cosine-anneal horizon of the EMA momentum, shared by every arm that anneals
# it. The trainers' default spans the whole --epochs budget, which early
# stopping rarely reaches. 0 = anneal over the whole run.
EMA_ANNEAL_EPOCHS="${EMA_ANNEAL_EPOCHS:-50}"
ANNEAL_ARG=""; [[ "$EMA_ANNEAL_EPOCHS" != "0" ]] && ANNEAL_ARG="--ema_anneal_epochs $EMA_ANNEAL_EPOCHS"
NW="${SLURM_CPUS_PER_TASK:-8}"
PROBE_SPLIT="${PROBE_SPLIT:-test}"

# --- which arms this job runs ------------------------------------------------
# The default "all" is right for England COVID. On METR-LA the JEPA arms are
# too long for one job; they are independent once the shared encoder exists,
# so give each its own job:
#
#   for A in jepa_frozen jepa_naive jepa_ema jepa_vicreg; do
#     DATASET=metr_la SEED=42 ARMS=$A sbatch scripts/regression/slurm_all_methods.sh
#   done
#   DATASET=metr_la SEED=42 ARMS="baselines supervised jepa_enc grace graphmae bgrl cldg ddgcl idol maskdgnn dysubc dyg2vec dygwm sgjepa" sbatch ...
#   DATASET=metr_la SEED=42 ARMS=summary sbatch ...        # once the rest land
#
# Stage names: baselines supervised jepa_frozen jepa_naive jepa_ema jepa_vicreg
#              jepa_enc grace graphmae bgrl cldg ddgcl idol maskdgnn dysubc dyg2vec dygwm sgjepa stdmae step summary
ARMS="${ARMS:-all}"
want() {
    [[ " $ARMS " == *" all "* || " $ARMS " == *" $1 "* ]]
}
# The shared encoder is a prerequisite of every DELOS arm, so any job that
# runs one must be prepared to build it (under flock, below).
need_encoder() {
    want jepa_frozen || want jepa_naive || want jepa_ema || want jepa_vicreg \
        || want jepa_enc
}

# Shared dataset / encoder / window / early-stop config. HIST, NPRED and
# REG_BATCH default to the dataset's published convention.
TASK=regression
source "$REPO_ROOT/scripts/dblp/_dataset_config.sh"
BATCH="${BATCH:-$REG_BATCH}"
# `--bidirectional false` keeps the encoder temporally causal (assumption A4);
# it is a no-op for the shape-agnostic `gcn` the shared config pins.
ENC_ARGS="$ENCODER_ARG --bidirectional false"

# --- predictor recipe --------------------------------------------------------
# PERSIST_RES makes the probe predict the delta over the last observed value,
# so persistence is the floor.
RESIDUAL="${RESIDUAL:-1}"
CHANGE_W="${CHANGE_W:-2.0}"
PERSIST_RES="${PERSIST_RES:-1}"
BASE_HP="${BASE_HP:-}"
# Tuned baseline hyperparameters from STAGE=baselines, appended last to each
# baseline's command so that they override the literature defaults.
BASELINE_HP_supervised="${BASELINE_HP_supervised:-}"
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
BASELINE_HP_stdmae="${BASELINE_HP_stdmae:-}"
# STD-MAE's tuned flags straddle two commands: --mask_ratio belongs to the
# pre-training, --lr to the forecaster. Split them rather than passing both to
# both (pretrain_stdmae has no --cl_epochs; train_stdmae has no --mask_ratio).
SM_MR_OVERRIDE=$(echo "$BASELINE_HP_stdmae" | sed -n 's/.*--mask_ratio \([0-9.]*\).*/\1/p')
SM_LR_OVERRIDE=$(echo "$BASELINE_HP_stdmae" | sed -n 's/.*--lr \([0-9eE.+-]*\).*/\1/p')
JEPA_BASE=""
[[ "$RESIDUAL" == "1" ]]    && JEPA_BASE="$JEPA_BASE --residual_head"
[[ "$CHANGE_W" != "0" ]]    && JEPA_BASE="$JEPA_BASE --change_weight $CHANGE_W"
[[ "$PERSIST_RES" == "1" ]] && JEPA_BASE="$JEPA_BASE --persist_residual"
PR_FLAG=""; [[ "$PERSIST_RES" == "1" ]] && PR_FLAG="--persist_residual"

# On METR-LA the structural attention mask does not fit in GPU memory
# (T*N tokens), so the predictor uses full attention there.
ATTN_FLAG=""
[[ "$DATASET" == "metr_la" ]] && ATTN_FLAG="--no_attention_mask"

# --- STD-MAE recipe (shared with the baseline HP search) --------------------
source "$REPO_ROOT/scripts/regression/_stdmae_config.sh"

# --- STEP recipe ------------------------------------------------------------
# STEP reads a long window that TSFormer reconstructs (one week on METR-LA, as
# in the paper). On England COVID the long window is pinned to the forecast
# history, the largest that needs no padding. ST_LONG and SM_LONG differ by
# default because each paper's own window length is reproduced.
case "$DATASET" in
    metr_la)
        ST_LONG=${ST_LONG:-2016}; ST_PATCH=${ST_PATCH:-12}
        ST_BATCH=${ST_BATCH:-32}                # the paper's
        ST_CL=${ST_CL:-6}                       # curriculum: +1 horizon / 6 epochs
        ST_STRIDE=${ST_STRIDE:-12}              # pre-train on every 12th window
        ;;
    *)
        ST_LONG=${ST_LONG:-$HIST}; ST_PATCH=${ST_PATCH:-2}
        ST_BATCH=${ST_BATCH:-$BATCH}
        ST_CL=${ST_CL:-0}                       # a 3-step horizon needs no curriculum
        ST_STRIDE=${ST_STRIDE:-1}
        ;;
esac

ROOT="outputs/reg_${DATASET}"
mkdir -p "$ROOT"
d() { echo "$ROOT/$1_s${SEED}"; }

PROBE_ARGS="$DATA_ARG $WINDOW_ARG --probe_epochs ${PROBE_EPOCHS:-10} \
    --probe_hidden 64 $PR_FLAG --probe_split $PROBE_SPLIT \
    --batch_size $BATCH --num_workers $NW --seed $SEED --device cuda \
    $EARLY_STOP_ARG"

JEPA_COMMON="$DATA_ARG $WINDOW_ARG $EARLY_STOP_ARG $ENC_ARGS $ATTN_FLAG \
    --downstream_task regression --mask_scheme node_mask \
    --depth ${DEPTH:-6} --heads ${HEADS:-8} --mlp_dim ${MLP_DIM:-512} \
    --num_masked_nodes ${NMASK:-20} \
    --batch_size $BATCH --epochs ${JEPA_EPOCHS:-300} --lr ${LR:-5e-4} \
    --weight_decay 1e-4 --probe_epochs ${PROBE_EPOCHS:-10} --probe_hidden 64 \
    --probe_split $PROBE_SPLIT --seed $SEED --num_workers $NW --device cuda"

echo "[config] DATASET=$DATASET SEED=$SEED HIST=$HIST NPRED=$NPRED BATCH=$BATCH"
echo "[config] JEPA_BASE='$JEPA_BASE' BASE_HP='$BASE_HP' probe_split=$PROBE_SPLIT"

fail() { echo "[FATAL] $*" >&2; exit "${2:-1}"; }

# --- 0. parameter-free baselines ---------------------------------------------
if want baselines; then
echo; echo "=== parameter-free baselines ==="
srun --export=ALL python scripts/regression/run_baselines.py $DATA_ARG \
    $WINDOW_ARG --batch_size "$BATCH" --num_workers "$NW" \
    --out_dir "$ROOT/baselines" || echo "[warn] baselines failed"
fi

# --- 1. supervised end-to-end reference (no SSL) -----------------------------
SUP_DIR="$(d supervised)"
if want supervised; then
echo; echo "=== supervised end-to-end -> $SUP_DIR ==="
srun --export=ALL python scripts/regression/probe_encoder_regression.py \
    --random_init --train_encoder $ENC_ARGS $PROBE_ARGS \
    --probe_epochs "${SUP_EPOCHS:-100}" --lr "${SUP_LR:-1e-3}" \
    --out_dir "$SUP_DIR" $BASELINE_HP_supervised || echo "[warn] supervised arm failed"
fi

# --- 2. shared SSL encoder (masked_signal), the start of every JEPA arm ------
ENC_DIR="$ROOT/_shared_encoder_s${SEED}"
ENC_CKPT="$ENC_DIR/encoder_${DATASET}_${ENC_KIND}_masked_signal.ckpt"
mkdir -p "$ENC_DIR"
if need_encoder; then
    # flock: with one arm per job, several jobs reach this at once and would
    # otherwise pretrain into the same path.
    exec 9>"$ENC_DIR/.pretrain.lock"
    flock -w 21600 9 || fail "encoder lock timeout" 10
    if [[ ! -f "$ENC_CKPT" ]]; then
        echo; echo "=== shared encoder pretrain (lock held) -> $ENC_DIR ==="
        srun --export=ALL python src/train/pretrain_encoder.py $DATA_ARG \
            --pretext masked_signal $ENC_ARGS --mask_rate 0.25 \
            $WINDOW_ARG $EARLY_STOP_ARG --decoder_hidden 64 \
            --batch_size "$BATCH" --epochs "${ENC_EPOCHS:-300}" --lr 1e-3 \
            --seed "$SEED" --num_workers "$NW" --device cuda --out_dir "$ENC_DIR"
        if [[ ! -f "$ENC_CKPT" ]]; then
            flock -u 9; fail "encoder pretrain produced no $ENC_CKPT" 11
        fi
    else
        echo "[skip] shared encoder exists: $ENC_CKPT"
    fi
    flock -u 9; exec 9>&-
fi

# --- 3. DELOS encoder-training variants --------------------------------------
arm_flags() {
    case "$1" in
        frozen) echo "" ;;
        naive)  echo "--train_encoder" ;;
        ema)    echo "--train_encoder --ema_target" ;;
        vicreg) echo "--train_encoder --vicreg_cov 1.0 --vicreg_var 5.0" ;;
        *)      echo "__UNKNOWN__" ;;
    esac
}

for ARM in frozen naive ema vicreg; do
    want "jepa_$ARM" || continue
    VFLAGS="$(arm_flags "$ARM")"
    # A sourced script clobbering $ARM would silently mislabel a whole column.
    [[ "$VFLAGS" == "__UNKNOWN__" ]] && fail "unknown arm '$ARM'" 12
    eval "HP_ARGS=\${VARIANT_HP_${ARM}:-}"
    OUT="$(d "jepa_$ARM")"
    mkdir -p "$OUT"
    echo; echo "=== DELOS [$ARM] -> $OUT ==="
    echo "    flags='$VFLAGS $HP_ARGS'"
    # Order matters: base recipe, tuned predictor HP, variant flags, then the
    # variant's own knobs. argparse honours the last occurrence.
    srun --export=ALL python src/train/train_delos.py $JEPA_COMMON \
        --encoder_ckpt "$ENC_CKPT" $JEPA_BASE $BASE_HP $VFLAGS $HP_ARGS \
        --out_dir "$OUT" || { echo "[warn] arm $ARM failed"; continue; }

    # Collapse check for the arms that train the encoder: a collapsed encoder
    # leaves the forecast at persistence however good the training loss looked.
    if [[ "$ARM" != "frozen" ]]; then
        srun --export=ALL python -m src.train.analyze_jepa --skip_diagnostic \
            --link_sources jepa $DATA_ARG $WINDOW_ARG \
            --jepa_ckpt "$OUT/best.ckpt" --device cuda --seed "$SEED" \
            --out_dir "$OUT/collapse_check" \
            || echo "[warn] collapse check failed for $ARM (non-fatal)"
    fi
done

# --- 4. jepa_enc control: the shared encoder + a head, no predictor ----------
JENC_DIR="$(d jepa_enc)"
if want jepa_enc; then
echo; echo "=== DELOS-enc control (no predictor) -> $JENC_DIR ==="
srun --export=ALL python scripts/regression/probe_encoder_regression.py \
    --encoder_ckpt "$ENC_CKPT" $PROBE_ARGS --out_dir "$JENC_DIR" \
    || echo "[warn] jepa_enc probe failed"
fi

# --- 5. SSL baselines: pretrain the encoder, then the same frozen head -------
CLDG_DIR="$(d cldg)"; DDGCL_DIR="$(d ddgcl)"; IDOL_DIR="$(d idol)"
MASKDGNN_DIR="$(d maskdgnn)"; DYSUBC_DIR="$(d dysubc)"; GRACE_DIR="$(d grace)"; GRAPHMAE_DIR="$(d graphmae)"
BGRL_DIR="$(d bgrl)"
DYG2VEC_DIR="$(d dyg2vec)"; DYGWM_DIR="$(d dygwm)"; SGJEPA_DIR="$(d sgjepa)"
mkdir -p "$CLDG_DIR" "$DDGCL_DIR" "$IDOL_DIR" "$MASKDGNN_DIR" "$DYSUBC_DIR" "$GRACE_DIR" "$GRAPHMAE_DIR" "$BGRL_DIR" "$DYG2VEC_DIR" "$DYGWM_DIR" "$SGJEPA_DIR"

if want bgrl && [[ ! -f "$BGRL_DIR/encoder.ckpt" ]]; then
    echo; echo "=== BGRL pretrain -> $BGRL_DIR ==="
    # Static negative-free control: one snapshot per step, reference WikiCS
    # config. Val-loss selection like every other arm.
    srun --export=ALL python -m src.baselines.bgrl.train_bgrl $DATA_ARG $ENC_ARGS \
        $WINDOW_ARG $EARLY_STOP_ARG \
        --p_e1 0.2 --p_e2 0.3 --p_f1 0.2 --p_f2 0.1 --ema_momentum 0.99 $ANNEAL_ARG \
        --steps_per_epoch 20 --val_steps 10 \
        --epochs "${SSL_EPOCHS:-100}" --lr 5e-4 \
        --device cuda --seed "$SEED" --out_dir "$BGRL_DIR" $BASELINE_HP_bgrl
    [[ -f "$BGRL_DIR/encoder.ckpt" ]] || fail "BGRL pretrain failed" 20
fi
if want graphmae && [[ ! -f "$GRAPHMAE_DIR/encoder.ckpt" ]]; then
    echo; echo "=== GraphMAE pretrain -> $GRAPHMAE_DIR ==="
    # Static generative control: one snapshot per step, reference Cora
    # defaults. Reconstructs every input channel, as published.
    srun --export=ALL python -m src.baselines.graphmae.train_graphmae $DATA_ARG $ENC_ARGS \
        $WINDOW_ARG $EARLY_STOP_ARG \
        --mask_rate 0.5 --replace_rate 0.05 --sce_gamma 2 --target all \
        --steps_per_epoch 20 --val_steps 10 \
        --epochs "${SSL_EPOCHS:-100}" --lr 1e-3 \
        --device cuda --seed "$SEED" --out_dir "$GRAPHMAE_DIR" $BASELINE_HP_graphmae
    [[ -f "$GRAPHMAE_DIR/encoder.ckpt" ]] || fail "GraphMAE pretrain failed" 19
fi
if want grace && [[ ! -f "$GRACE_DIR/encoder.ckpt" ]]; then
    echo; echo "=== GRACE pretrain -> $GRACE_DIR ==="
    # Static contrastive control, reference Cora defaults. On METR-LA the
    # graph never changes, so the edge view differs per step only through its
    # random mask; the feature view carries that step's readings.
    srun --export=ALL python -m src.baselines.grace.train_grace $DATA_ARG $ENC_ARGS \
        $WINDOW_ARG $EARLY_STOP_ARG \
        --tau 0.4 --p_e1 0.2 --p_e2 0.4 --p_f1 0.3 --p_f2 0.4 \
        --node_batch_size 128 --steps_per_epoch 20 --val_steps 10 \
        --epochs "${SSL_EPOCHS:-100}" --lr 5e-4 \
        --device cuda --seed "$SEED" --out_dir "$GRACE_DIR" $BASELINE_HP_grace
    [[ -f "$GRACE_DIR/encoder.ckpt" ]] || fail "GRACE pretrain failed" 18
fi

if want cldg && [[ ! -f "$CLDG_DIR/encoder.ckpt" ]]; then
    echo; echo "=== CLDG pretrain -> $CLDG_DIR ==="
    # --view_length, not --snapshots: CLDG's reference default is a quarter of
    # the whole timeline, which on METR-LA mean-pools weeks of traffic into one
    # view and erases the temporal variation CLDG contrasts. Pinning the view
    # to the forecast history keeps it comparable on both datasets.
    srun --export=ALL python -m src.baselines.cldg.train_cldg $DATA_ARG $ENC_ARGS \
        $WINDOW_ARG $EARLY_STOP_ARG \
        --view_length "$HIST" --views 2 --strategy random --tau 0.07 \
        --node_batch_size 128 --steps_per_epoch 20 --val_steps 10 \
        --epochs "${SSL_EPOCHS:-100}" --lr 4e-3 \
        --device cuda --seed "$SEED" --out_dir "$CLDG_DIR" $BASELINE_HP_cldg
    # Hard-fail: a missing row reads as "not applicable", not "this crashed".
    [[ -f "$CLDG_DIR/encoder.ckpt" ]] || fail "CLDG pretrain failed" 13
fi
if want ddgcl && [[ ! -f "$DDGCL_DIR/encoder.ckpt" ]]; then
    echo; echo "=== DDGCL pretrain -> $DDGCL_DIR ==="
    # `--tau` is the debias class prior (the reference's tau_plus); there is
    # no --tau_plus flag.
    srun --export=ALL python -m src.baselines.ddgcl.train_ddgcl $DATA_ARG $ENC_ARGS \
        $WINDOW_ARG $EARLY_STOP_ARG \
        --max_gap 4 --time_dim 32 --tau 0.01 --beta 1.5 \
        --node_batch_size 128 --steps_per_epoch 20 --val_steps 10 \
        --epochs "${SSL_EPOCHS:-100}" --lr 4e-3 \
        --device cuda --seed "$SEED" --out_dir "$DDGCL_DIR" $BASELINE_HP_ddgcl
    [[ -f "$DDGCL_DIR/encoder.ckpt" ]] || fail "DDGCL pretrain failed" 14
fi
if want idol && [[ ! -f "$IDOL_DIR/encoder.ckpt" ]]; then
    echo; echo "=== IDOL pretrain -> $IDOL_DIR ==="
    srun --export=ALL python -m src.baselines.idol.train_idol $DATA_ARG $ENC_ARGS \
        $WINDOW_ARG $EARLY_STOP_ARG \
        --proj_layers 1 --node_batch_size 128 --epochs "${SSL_EPOCHS:-100}" \
        --steps_per_epoch 20 --val_steps 10 \
        --lr 1e-3 --device cuda --seed "$SEED" --out_dir "$IDOL_DIR" $BASELINE_HP_idol
    [[ -f "$IDOL_DIR/encoder.ckpt" ]] || fail "IDOL pretrain failed" 15
fi
if want maskdgnn && [[ ! -f "$MASKDGNN_DIR/encoder.ckpt" ]]; then
    echo; echo "=== MaskDGNN pretrain -> $MASKDGNN_DIR ==="
    # Own encoder kind (the `--encoder_kind` after $ENC_ARGS wins). --window
    # defaults to HIST. The link-style objective is scored on the graph's own
    # edges, never on the signal.
    srun --export=ALL python -m src.baselines.maskdgnn.train_maskdgnn $DATA_ARG $ENC_ARGS \
        --encoder_kind maskdgnn $WINDOW_ARG $EARLY_STOP_ARG \
        --mask_ratio 0.2 --alpha 0.7 --target next --neg_per_pos 1 \
        --steps_per_epoch 20 --val_steps 10 --e2e_eval none \
        --epochs "${SSL_EPOCHS:-100}" --lr 4e-3 \
        --device cuda --seed "$SEED" --out_dir "$MASKDGNN_DIR" $BASELINE_HP_maskdgnn
    [[ -f "$MASKDGNN_DIR/encoder.ckpt" ]] || fail "MaskDGNN pretrain failed" 16
fi
if want dysubc && [[ ! -f "$DYSUBC_DIR/encoder.ckpt" ]]; then
    echo; echo "=== DySubC pretrain -> $DYSUBC_DIR ==="
    # --lookback defaults to HIST: the time-weighted temporal graph spans the
    # same history the forecast head probes. Paper defaults otherwise.
    srun --export=ALL python -m src.baselines.dysubc.train_dysubc $DATA_ARG $ENC_ARGS \
        $WINDOW_ARG $EARLY_STOP_ARG \
        --subgraph_size 20 --alpha 5 --beta 1 --margin 0.75 --lam 0.5 \
        --node_batch_size 128 --steps_per_epoch 20 --val_steps 10 \
        --epochs "${SSL_EPOCHS:-100}" --lr 1e-3 \
        --device cuda --seed "$SEED" --out_dir "$DYSUBC_DIR" $BASELINE_HP_dysubc
    [[ -f "$DYSUBC_DIR/encoder.ckpt" ]] || fail "DySubC pretrain failed" 17
fi
if want dyg2vec && [[ ! -f "$DYG2VEC_DIR/encoder.ckpt" ]]; then
    echo; echo "=== DyG2Vec pretrain -> $DYG2VEC_DIR ==="
    # Own encoder kind (the `--encoder_kind` after $ENC_ARGS wins). --window
    # defaults to HIST; the other values are paper defaults.
    srun --export=ALL python -m src.baselines.dyg2vec.train_dyg2vec $DATA_ARG $ENC_ARGS \
        --encoder_kind dyg2vec $WINDOW_ARG $EARLY_STOP_ARG \
        --p_edge 0.3 --p_struct 0.3 --lam 25 --mu 25 --nu 1 \
        --steps_per_epoch 20 --val_steps 10 \
        --epochs "${SSL_EPOCHS:-100}" --lr 1e-4 \
        --device cuda --seed "$SEED" --out_dir "$DYG2VEC_DIR" $BASELINE_HP_dyg2vec
    [[ -f "$DYG2VEC_DIR/encoder.ckpt" ]] || fail "DyG2Vec pretrain failed" 19
fi
if want dygwm && [[ ! -f "$DYGWM_DIR/encoder.ckpt" ]]; then
    echo; echo "=== DyG-WM pretrain -> $DYGWM_DIR ==="
    # Own encoder kind (the `--encoder_kind` after $ENC_ARGS wins). SSL-only
    # core (no link / rank loss), paper node-task defaults; var / cov keep the
    # paper's ratios to the JEPA terms.
    srun --export=ALL python -m src.baselines.dygwm.train_dygwm $DATA_ARG $ENC_ARGS \
        --encoder_kind dygwm $WINDOW_ARG $EARLY_STOP_ARG \
        --ema_momentum 0.995 --n_ctx 32 --decay 0.8 --cn_bonus 1.0 \
        --lam_node 1 --lam_rel 1 --lam_var 0.25 --lam_cov 0.025 \
        --node_batch_size 128 --pair_batch_size 128 \
        --steps_per_epoch 20 --val_steps 10 \
        --epochs "${SSL_EPOCHS:-100}" --lr 1e-3 \
        --device cuda --seed "$SEED" --out_dir "$DYGWM_DIR" $BASELINE_HP_dygwm
    [[ -f "$DYGWM_DIR/encoder.ckpt" ]] || fail "DyG-WM pretrain failed" 21
fi
if want sgjepa && [[ ! -f "$SGJEPA_DIR/encoder.ckpt" ]]; then
    echo; echo "=== SG-JEPA pretrain -> $SGJEPA_DIR ==="
    # Shared per-snapshot GCN as the stop-gradient target encoder, spiking
    # context encoder on its latents. The paper gives no temperature or
    # optimiser: tau 0.1 and AdamW 1e-3 are our choice.
    srun --export=ALL python -m src.baselines.sgjepa.train_sgjepa $DATA_ARG $ENC_ARGS \
        $WINDOW_ARG $EARLY_STOP_ARG \
        --snn_layers 1 --tau 0.1 --init_tau 2.0 --v_th 1.0 \
        --node_batch_size 128 --steps_per_epoch 20 --val_steps 10 \
        --epochs "${SSL_EPOCHS:-100}" --lr 1e-3 \
        --device cuda --seed "$SEED" --out_dir "$SGJEPA_DIR" $BASELINE_HP_sgjepa
    [[ -f "$SGJEPA_DIR/encoder.ckpt" ]] || fail "SG-JEPA pretrain failed" 22
fi

for M in grace graphmae bgrl cldg ddgcl idol maskdgnn dysubc dyg2vec dygwm sgjepa; do
    want "$M" || continue
    D="$(d "$M")"
    [[ -f "$D/encoder.ckpt" ]] || { echo "[skip] $M: no encoder.ckpt"; continue; }
    echo; echo "=== frozen forecast probe -> $D ==="
    srun --export=ALL python scripts/regression/probe_encoder_regression.py \
        --encoder_ckpt "$D/encoder.ckpt" $PROBE_ARGS \
        || echo "[warn] probe failed for $D"
done

# --- 5b. STD-MAE (Gao et al., IJCAI 2024) ------------------------------------
# S-MAE and T-MAE are pre-trained on the long window and frozen; a Graph
# WaveNet backend is then trained on the forecast loss with their
# representations injected. This arm does not use the shared frozen head: its
# published downstream stage is a full supervised forecaster, so read the row
# next to `supervised`. It is scored through the same probes.py accounting.
if want stdmae; then
SM_PRE="$ROOT/_stdmae_pretrain${SM_MR_OVERRIDE:+_mr$SM_MR_OVERRIDE}_s${SEED}"
SM_DIR="$(d stdmae)"
mkdir -p "$SM_PRE" "$SM_DIR"
SM_ARGS="$DATA_ARG $WINDOW_ARG --long_history $SM_LONG --patch_size $SM_PATCH"

for V in temporal spatial; do
    case "$V" in temporal) CK=tmae ;; spatial) CK=smae ;; esac
    if [[ -f "$SM_PRE/$CK.ckpt" ]]; then
        echo "[skip] STD-MAE $CK exists: $SM_PRE/$CK.ckpt"
        continue
    fi
    echo; echo "=== STD-MAE $CK pre-train -> $SM_PRE ==="
    srun --export=ALL python -m src.baselines.stdmae.pretrain_stdmae \
        --variant "$V" $SM_ARGS $EARLY_STOP_ARG \
        --window_stride "$SM_STRIDE" --batch_size "$SM_BATCH" \
        --epochs "${SM_PRE_EPOCHS:-200}" --lr "${SM_PRE_LR:-5e-4}" \
        ${SM_MR_OVERRIDE:+--mask_ratio $SM_MR_OVERRIDE} \
        --seed "$SEED" --num_workers "$NW" --device cuda --out_dir "$SM_PRE"
    # Hard-fail: a missing row reads as "not applicable", not "this crashed".
    [[ -f "$SM_PRE/$CK.ckpt" ]] || fail "STD-MAE $CK pre-train failed" 16
done

echo; echo "=== STD-MAE forecasting -> $SM_DIR ==="
srun --export=ALL python -m src.baselines.stdmae.train_stdmae \
    $DATA_ARG $WINDOW_ARG $EARLY_STOP_ARG \
    --smae_ckpt "$SM_PRE/smae.ckpt" --tmae_ckpt "$SM_PRE/tmae.ckpt" \
    --batch_size "$SM_BATCH" --epochs "${SM_EPOCHS:-100}" \
    --lr "${SM_LR_OVERRIDE:-${SM_LR:-5e-3}}" --cl_epochs "$SM_CL" \
    --probe_split "$PROBE_SPLIT" \
    --seed "$SEED" --num_workers "$NW" --device cuda --out_dir "$SM_DIR" \
    || echo "[warn] STD-MAE forecasting arm failed"
fi

# --- 5c. STEP (Shao et al., SIGKDD 2022) -------------------------------------
# TSFormer is pre-trained on a long per-node window and frozen; a discrete
# graph learner and a Graph WaveNet backend then forecast. Like STD-MAE it
# brings its own backend, and it ignores the dataset's graph (it learns one).
if want step; then
ST_PRE="$ROOT/_step_pretrain_s${SEED}"
ST_DIR="$(d step)"
mkdir -p "$ST_PRE" "$ST_DIR"

if [[ -f "$ST_PRE/tsformer.ckpt" ]]; then
    echo "[skip] TSFormer exists: $ST_PRE/tsformer.ckpt"
else
    echo; echo "=== TSFormer pre-train -> $ST_PRE ==="
    srun --export=ALL python -m src.baselines.step.pretrain_tsformer \
        $DATA_ARG $WINDOW_ARG $EARLY_STOP_ARG \
        --long_history "$ST_LONG" --patch_size "$ST_PATCH" \
        --window_stride "$ST_STRIDE" --batch_size "$ST_BATCH" \
        --epochs "${ST_PRE_EPOCHS:-100}" --lr "${ST_PRE_LR:-5e-4}" \
        --seed "$SEED" --num_workers "$NW" --device cuda --out_dir "$ST_PRE"
    # Hard-fail: a missing row reads as "not applicable", not "this crashed".
    [[ -f "$ST_PRE/tsformer.ckpt" ]] || fail "TSFormer pre-train failed" 17
fi

echo; echo "=== STEP forecasting -> $ST_DIR ==="
srun --export=ALL python -m src.baselines.step.train_step \
    $DATA_ARG $WINDOW_ARG $EARLY_STOP_ARG \
    --tsformer_ckpt "$ST_PRE/tsformer.ckpt" \
    --batch_size "$ST_BATCH" --epochs "${ST_EPOCHS:-100}" \
    --lr "${ST_LR:-5e-3}" --cl_epochs "$ST_CL" --dgl_k "${ST_K:-10}" \
    --probe_split "$PROBE_SPLIT" \
    --seed "$SEED" --num_workers "$NW" --device cuda --out_dir "$ST_DIR" \
    || echo "[warn] STEP forecasting arm failed"

# --- 6. summary for this seed ------------------------------------------------
fi

if want summary || [[ "$ARMS" == "all" ]]; then
echo; echo "=== seed $SEED summary ==="
srun --export=ALL python scripts/regression/aggregate_regression.py \
    --dataset "$DATASET" --seeds "$SEED" || true
fi

echo; echo "[done] $DATASET seed $SEED -> $ROOT"
echo "Aggregate across seeds with:"
echo "  python scripts/regression/aggregate_regression.py --dataset $DATASET \\"
echo "      --seeds 42 43 44 --latex"
