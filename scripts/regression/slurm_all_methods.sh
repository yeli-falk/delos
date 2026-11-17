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
# NODE-SIGNAL FORECASTING: every method, one seed per job.
#
#   DATASET=metr_la|england_covid   (window geometry comes from the shared
#                                    config; see scripts/dblp/_dataset_config.sh)
#
# Arms produced (each writes its own test_metrics.json):
#   persistence / window_mean / historical_mean   parameter-free
#   supervised      encoder + forecast head trained END-TO-END on the forecast
#                   loss from random init. No SSL. The reference every
#                   self-supervised arm is trying to reach without labels.
#   jepa_frozen     SSL-pretrained encoder FROZEN, JEPA predictor trained on top.
#                   Targets are fixed, so latent collapse is impossible by
#                   construction. The reference arm for the variants.
#   jepa_naive      --train_encoder alone. Targets come from the SAME encoder
#                   being optimised, so mapping everything to one point is a
#                   global optimum. This arm exists so a reader can tell whether
#                   `ema` won because of EMA or merely because the encoder was
#                   unfrozen.
#   jepa_ema        --train_encoder --ema_target. I-JEPA: online encoder builds
#                   the context, an EMA copy builds the targets.
#   jepa_vicreg     --train_encoder + VICReg variance/covariance on the predictor
#                   output. Anti-collapse WITHOUT a target network: does any
#                   collapse penalty do the job, or is EMA specifically what
#                   matters?
#   jepa_enc        the frozen arm's encoder + a forecast head, NO predictor.
#                   Isolates "does the predictor rollout help?"
#   graphmae        GraphMAE (Hou et al., KDD'22): STATIC masked-feature
#                   autoencoder -- mask a fraction of ONE snapshot's nodes
#                   with a learnable token, re-mask their latents, GCN
#                   decoder, scaled cosine error, + the SAME frozen forecast
#                   head. The generative half of the static control block
#                   (GRACE is the contrastive half).
#   bgrl            BGRL (Thakoor et al., ICLR'22): STATIC bootstrapped latents
#                   -- two augmented views of ONE snapshot, an online encoder +
#                   predictor regress an EMA target's embedding of the other
#                   view; no negatives, + the SAME frozen forecast head. The
#                   closest static relative of jepa_ema: same bootstrap, but
#                   the target is the same snapshot re-augmented, not a future
#                   one.
#   grace           GRACE (Zhu et al., 2020): STATIC graph contrast -- two
#                   augmented views (edge removal + feature-dimension masking)
#                   of ONE snapshot, inter- and intra-view negatives, + the
#                   SAME frozen forecast head. Never sees two snapshots at
#                   once: the control for "does the SSL objective need to be
#                   temporal at all?". On a fixed topology the edge view
#                   differs per step only through its random mask.
#   cldg / ddgcl / idol
#                   contrastive SSL encoders + the SAME frozen forecast head.
#   maskdgnn        MaskDGNN (IJCAI'25): activeness-aware edge masking +
#                   masked/next-edge BCE on a GCN with an adaptive frequency
#                   layer, + the SAME frozen forecast head. On a fixed topology
#                   the dynamics score is identically 0, so its masking is
#                   PageRank-weighted only (the trainer prints this).
#   dysubc          DySubC (TKDD'23): temporal subgraph contrast (BFS
#                   subgraphs ranked by recency + degree, time-weighted vs
#                   unweighted subgraph encoders, margin losses), + the SAME
#                   frozen forecast head. On a fixed topology every edge
#                   shares one time level, so its sampling score reduces to
#                   degree.
#   dyg2vec         DyG2Vec (TMLR'24): window temporal attention encoder
#                   (own encoder kind) pre-trained with VICReg on two
#                   distorted views of each window, + the SAME frozen
#                   forecast head. On a fixed topology every snapshot has
#                   the same interactions; the time encoding separates them.
#   dygwm           DyG-WM (Li et al., 2026): SSL-only core of the dynamic
#                   graph world model -- context encoder (own encoder kind:
#                   shared GCN + node / neighbourhood trajectory GRUs) + path
#                   signatures predict an EMA target's NEXT-step node and
#                   relation latents, + the SAME frozen forecast head. On a
#                   fixed topology the structural descriptors are constant,
#                   so it reduces to next-step latent prediction.
#   sgjepa          SG-JEPA (Zhang et al., 2026): spiking (PLIF) context
#                   encoder on the shared GCN's latents of the earlier
#                   snapshots, spike counts predict the LAST snapshot's
#                   stop-gradient latent (per-node InfoNCE), + the SAME
#                   frozen forecast head.
#   step            STEP (KDD'22): TSFormer masked pre-training over a LONG
#                   history window, then a LEARNED discrete graph + a Graph
#                   WaveNet forecaster on the frozen representations. Uses no
#                   given adjacency at all.
#   stdmae          STD-MAE (IJCAI'24): spatial- and temporal-decoupled masked
#                   pre-training over a LONG history window, then a Graph
#                   WaveNet forecaster on the frozen representations. The one
#                   arm that brings its own backend rather than the shared head
#                   -- that backend is the published method.
#
# COST (METR-LA): a step is ~5.2 s on a V100 and stride-1 gives 375 steps/epoch,
# so ONE JEPA arm at 50 epochs is ~27 h and this job runs FOUR of them plus the
# supervised and contrastive arms. The 48 h walltime above is NOT enough for the
# full default; either raise it, lower JEPA_EPOCHS to what the LR probe shows is
# needed, or split the arms across jobs. Do the arithmetic before submitting --
# the probe stage runs AFTER training, so a walltime kill loses the whole cell
# and leaves no test_metrics.json.
# The final table must stay at stride 1 (the DCRNN convention); --metr_la_stride
# is a search/diagnostic knob only and is deliberately not wired in here.
#
# Every arm sources the same dataset/encoder/window/early-stop config, so the
# columns differ only by method. All four JEPA variants start from the SAME
# pretrained encoder and the same predictor hyperparameters (BASE_HP), so their
# only difference is the encoder-training rule.
#
# Usage:
#   DATASET=england_covid SEED=42 sbatch scripts/regression/slurm_all_methods.sh
#   DATASET=metr_la       SEED=42 sbatch scripts/regression/slurm_all_methods.sh
#   # three seeds:
#   for S in 42 43 44; do DATASET=metr_la SEED=$S sbatch scripts/regression/slurm_all_methods.sh; done
#   # then:
#   python scripts/regression/aggregate_regression.py --dataset metr_la --seeds 42 43 44 --latex
#
# Paste the winning predictor config from the hyperparameter search:
#   BASE_HP="--lr 1e-3 --num_masked_nodes 30 --change_weight 2.0" DATASET=... sbatch ...
# Per-variant winners from the same search:
#   VARIANT_HP_ema="--encoder_lr 1e-4 --ema_momentum 0.996" ...

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
NW="${SLURM_CPUS_PER_TASK:-8}"
PROBE_SPLIT="${PROBE_SPLIT:-test}"

# --- which arms this job runs ------------------------------------------------
# Default "all" keeps the single-job behaviour, which is right for England
# COVID (the whole table is minutes). On METR-LA it is NOT: one JEPA arm at 50
# epochs is ~30 h, so running all four sequentially needs ~121 h and no sane
# walltime covers it. The four arms are INDEPENDENT once the shared encoder
# exists, so give each its own job and they run concurrently instead:
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
# REG_BATCH default to this dataset's published convention (DCRNN 12->12 for
# METR-LA; 8 daily lags for England COVID) — see _dataset_config.sh.
TASK=regression
source "$REPO_ROOT/scripts/dblp/_dataset_config.sh"
BATCH="${BATCH:-$REG_BATCH}"
# `--bidirectional false` keeps the encoder temporally causal (assumption A4);
# it is a no-op for the shape-agnostic `gcn` the shared config pins.
ENC_ARGS="$ENCODER_ARG --bidirectional false"

# --- predictor recipe (the MOOC-validated fix, ported to regression) ---------
# PERSIST_RES is load-bearing: it makes the probe predict the delta over the
# last observed value, so persistence is the FLOOR. Without it a latent->signal
# probe historically landed ~1.8x WORSE than persistence.
RESIDUAL="${RESIDUAL:-1}"
CHANGE_W="${CHANGE_W:-2.0}"
PERSIST_RES="${PERSIST_RES:-1}"
BASE_HP="${BASE_HP:-}"
# Tuned baseline hyperparameters from STAGE=baselines. Appended LAST to each
# baseline's command so they override the hardcoded literature defaults
# (argparse honours the last occurrence). Empty = keep the published default.
#   BASELINE_HP_cldg="--lr 1e-3 --views 4" ... sbatch ...
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

# METR-LA's structural mask is (B, 1, T*N, T*N); at N=207, T=24 the materialised
# (B, heads, TN, TN) attention is tens of GB and OOMs even a 32 GB card. Full
# attention is the right trade until factorised attention lands.
ATTN_FLAG=""
[[ "$DATASET" == "metr_la" ]] && ATTN_FLAG="--no_attention_mask"

# --- STD-MAE recipe (shared with the baseline HP search) --------------------
source "$REPO_ROOT/scripts/regression/_stdmae_config.sh"

# --- STEP recipe ------------------------------------------------------------
# Same shape of problem as STD-MAE's: STEP reads a long window (the paper's
# METR-LA setting is 288*7 = 2016 5-minute steps = one week) that TSFormer
# reconstructs. England COVID gets the same treatment as above -- the long
# window is pinned to the forecast history, the largest that needs no
# fabricated padding. Keeping ST_LONG == SM_LONG on a dataset would make the
# two arms' inputs identical; they are NOT equal by default, because each
# paper's own window length is the thing being reproduced.
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

# =============================================================================
# 0. parameter-free baselines
# =============================================================================
if want baselines; then
echo; echo "=== parameter-free baselines ==="
srun --export=ALL python scripts/regression/run_baselines.py $DATA_ARG \
    $WINDOW_ARG --batch_size "$BATCH" --num_workers "$NW" \
    --out_dir "$ROOT/baselines" || echo "[warn] baselines failed"
fi

# =============================================================================
# 1. supervised end-to-end reference (no SSL at all)
# =============================================================================
SUP_DIR="$(d supervised)"
if want supervised; then
echo; echo "=== supervised end-to-end -> $SUP_DIR ==="
srun --export=ALL python scripts/regression/probe_encoder_regression.py \
    --random_init --train_encoder $ENC_ARGS $PROBE_ARGS \
    --probe_epochs "${SUP_EPOCHS:-100}" --lr "${SUP_LR:-1e-3}" \
    --out_dir "$SUP_DIR" $BASELINE_HP_supervised || echo "[warn] supervised arm failed"
fi

# =============================================================================
# 2. shared SSL encoder (masked_signal) — the start point for all four variants
# =============================================================================
ENC_DIR="$ROOT/_shared_encoder_s${SEED}"
ENC_CKPT="$ENC_DIR/encoder_${DATASET}_${ENC_KIND}_masked_signal.ckpt"
mkdir -p "$ENC_DIR"
if need_encoder; then
    # flock, because one-arm-per-job means several jobs reach this at once and
    # would otherwise all pretrain into the SAME path concurrently -- racing on
    # one checkpoint file and burning four GPUs on identical work.
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

# =============================================================================
# 3. DELOS encoder-training variants
# =============================================================================
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
    # Order matters: base recipe, then tuned predictor HP, then the variant
    # flags, then the variant's own tuned knobs — argparse honours the last
    # occurrence, so the most specific setting wins.
    srun --export=ALL python src/train/train_delos.py $JEPA_COMMON \
        --encoder_ckpt "$ENC_CKPT" $JEPA_BASE $BASE_HP $VFLAGS $HP_ARGS \
        --out_dir "$OUT" || { echo "[warn] arm $ARM failed"; continue; }

    # Collapse check for the unfrozen arms. If the encoder collapsed, every
    # node's latent is (nearly) the same point: the probe cannot separate
    # anything and the forecast sits at persistence no matter how good the
    # training loss looked.
    if [[ "$ARM" != "frozen" ]]; then
        srun --export=ALL python -m src.train.analyze_jepa --skip_diagnostic \
            --link_sources jepa $DATA_ARG $WINDOW_ARG \
            --jepa_ckpt "$OUT/best.ckpt" --device cuda --seed "$SEED" \
            --out_dir "$OUT/collapse_check" \
            || echo "[warn] collapse check failed for $ARM (non-fatal)"
    fi
done

# =============================================================================
# 4. DELOS-enc control: the SAME shared encoder + a head, no predictor
# =============================================================================
JENC_DIR="$(d jepa_enc)"
if want jepa_enc; then
echo; echo "=== DELOS-enc control (no predictor) -> $JENC_DIR ==="
srun --export=ALL python scripts/regression/probe_encoder_regression.py \
    --encoder_ckpt "$ENC_CKPT" $PROBE_ARGS --out_dir "$JENC_DIR" \
    || echo "[warn] jepa_enc probe failed"
fi

# =============================================================================
# 5. contrastive baselines: pretrain encoder -> the same frozen forecast head
# =============================================================================
CLDG_DIR="$(d cldg)"; DDGCL_DIR="$(d ddgcl)"; IDOL_DIR="$(d idol)"
MASKDGNN_DIR="$(d maskdgnn)"; DYSUBC_DIR="$(d dysubc)"; GRACE_DIR="$(d grace)"; GRAPHMAE_DIR="$(d graphmae)"
BGRL_DIR="$(d bgrl)"
DYG2VEC_DIR="$(d dyg2vec)"; DYGWM_DIR="$(d dygwm)"; SGJEPA_DIR="$(d sgjepa)"
mkdir -p "$CLDG_DIR" "$DDGCL_DIR" "$IDOL_DIR" "$MASKDGNN_DIR" "$DYSUBC_DIR" "$GRACE_DIR" "$GRAPHMAE_DIR" "$BGRL_DIR" "$DYG2VEC_DIR" "$DYGWM_DIR" "$SGJEPA_DIR"

if want bgrl && [[ ! -f "$BGRL_DIR/encoder.ckpt" ]]; then
    echo; echo "=== BGRL pretrain -> $BGRL_DIR ==="
    # STATIC negative-free control: one snapshot per step, reference WikiCS
    # config (p_e 0.2/0.3, p_f 0.2/0.1, EMA 0.99 -> 1 cosine, AdamW 5e-4 with
    # warm-up + cosine decay). Val-loss selection like every other arm.
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
    # STATIC generative control: one snapshot per step, reference Cora
    # defaults (mask_rate 0.5, replace_rate 0.05, gamma 2, Adam 1e-3 / wd
    # 2e-4). Reconstructs every input channel, as published.
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
    # STATIC control: one snapshot per step, two augmented views of it,
    # reference Cora defaults (tau 0.4, p_e 0.2/0.4, p_f 0.3/0.4, Adam 5e-4).
    # On METR-LA the graph never changes, so the edge view differs per step
    # only through its random mask; the feature view carries that step's
    # readings. Same per-step node budget as the temporal arms.
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
    # --view_length, not --snapshots: CLDG's reference default means "a quarter
    # of the whole timeline", which is ~10 days on England COVID's 42 daily
    # snapshots but ~20 DAYS on METR-LA's ~24k 5-minute ones. Mean-pooling 20
    # days of traffic into one static view erases the temporal variation CLDG
    # contrasts and its objective sits at chance. Pinning the view to the
    # forecast history keeps it a comparable amount of real time on both.
    srun --export=ALL python -m src.baselines.cldg.train_cldg $DATA_ARG $ENC_ARGS \
        $WINDOW_ARG $EARLY_STOP_ARG \
        --view_length "$HIST" --views 2 --strategy random --tau 0.07 \
        --node_batch_size 128 --steps_per_epoch 20 --val_steps 10 \
        --epochs "${SSL_EPOCHS:-100}" --lr 4e-3 \
        --device cuda --seed "$SEED" --out_dir "$CLDG_DIR" $BASELINE_HP_cldg
    # Hard-fail: a warn leaves the method absent from the table, which reads as
    # "not applicable" rather than "this run crashed".
    [[ -f "$CLDG_DIR/encoder.ckpt" ]] || fail "CLDG pretrain failed" 13
fi
if want ddgcl && [[ ! -f "$DDGCL_DIR/encoder.ckpt" ]]; then
    echo; echo "=== DDGCL pretrain -> $DDGCL_DIR ==="
    # `--tau` IS the debias class prior (the reference's tau_plus, default
    # 0.01). Passing `--tau_plus` here aborts the stage with an argparse error.
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
    # Own encoder kind (the shared GCN + an adaptive frequency layer along the
    # window); the `--encoder_kind` after $ENC_ARGS wins. --window defaults to
    # HIST, so the frequency filter is trained on the length the forecast head
    # probes. The link-style objective is scored on the graph's own edges (on
    # a fixed topology, the same edges every step), never on the signal.
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
    # Own encoder kind (the `--encoder_kind` after $ENC_ARGS wins); --window
    # defaults to HIST so the encoder is pre-trained on the length the
    # forecast head probes. Paper defaults: p_d 0.3, VICReg 25/25/1, lr 1e-4.
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
    # Own encoder kind (the `--encoder_kind` after $ENC_ARGS wins); --window
    # defaults to HIST context steps + one target step. SSL-only core (no
    # link / rank loss). Paper node-task defaults: EMA 0.995, B=32, decay 0.8,
    # AdamW 1e-3 / wd 1e-5; var / cov keep the paper's ratios to the JEPA terms.
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
    # Shared per-snapshot GCN as the stop-gradient target encoder; spiking
    # context encoder on its latents. tau 0.1 / AdamW 1e-3 (paper gives neither).
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

# =============================================================================
# 5b. STD-MAE (Gao et al., IJCAI 2024) -- spatial/temporal-decoupled masked
#     pre-training. Two MAEs are pre-trained on the long window (S-MAE masks
#     whole NODES, T-MAE masks whole time PATCHES), then frozen, then a Graph
#     WaveNet backend is trained on the forecast loss with their representations
#     injected into its skip path.
#
#     Unlike CLDG / DDGCL / IDOL this arm does NOT use the shared frozen
#     forecast head: its published downstream stage IS a full supervised
#     forecaster, and replacing it would not be STD-MAE. Read the row next to
#     `supervised` -- same label budget, self-supervised representations. It is
#     scored through the same probes.py accounting on the same windows, so
#     `MAE_persist` still cross-checks against every other arm.
# =============================================================================
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

# =============================================================================
# 5c. STEP (Shao et al., SIGKDD 2022) -- pre-training-enhanced STGNN. TSFormer
#     is pre-trained by masking 75% of the time patches of a long per-node
#     window, then frozen. Downstream, a discrete graph learner produces an
#     adjacency (regularised towards a kNN graph over TSFormer's
#     representations) and a Graph WaveNet backend forecasts on it with
#     TSFormer's representation injected into its skip path.
#
#     Like STD-MAE this arm brings its own backend rather than the shared head.
#     Note also that STEP ignores the dataset's graph entirely -- it learns one.
#     On METR-LA the sensor adjacency every other arm is given is unused; on
#     England COVID so is the mobility graph. That is the method's claim.
# =============================================================================
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

# =============================================================================
# 6. summary for this seed
# =============================================================================
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
