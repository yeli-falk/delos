#!/usr/bin/env bash
#SBATCH --job-name=synth-noise
#SBATCH --output=logs/synth-noise-%A_%a.out
#SBATCH --error=logs/synth-noise-%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#
# Does latent-space prediction beat observation-space prediction as the
# observations get noisier? Controlled sweep on the synthetic latent-driven
# generator (src/custom_codes/synthetic_dyn_dataset.py).
#
# The generator draws every observable from one per-node latent z. Raising
# --synth_node_noise adds noise after the latent readout, so the observations
# degrade while the underlying cause stays as predictable as before.
#
# --synth_target_noise -1 applies the same relative noise to the observed
# history of the regression target, and evaluation stays against the clean
# future value. Without it the persistence baseline reads the clean target and
# is immune to the sweep, so raising noise only handicaps the learned arms and
# leaves no headroom at the noisy end.
#
# Arms (all share the same pretrained encoder within a noise x seed cell):
#   jepa_latent      --predict_space latent: match the frozen encoder's latents.
#   jepa_input       --predict_space input: same encoder, predictor and
#                    masking, plus a linear readout; match the raw features.
#                    The control: any difference to jepa_latent is due to the
#                    prediction space alone.
#   jepa_latent_ema  --train_encoder --ema_target + VICReg. With a frozen
#                    encoder the latent target encodes the observation noise;
#                    only a trained target space can discard it.
#   enc_only         The shared encoder with no predictor, probed with the
#                    same forecast head. Isolates what the predictor adds.
#   cldg / ddgcl / idol
#                    Encoder-only contrastive SSL with the same forecast head,
#                    as context.
#
# Prediction: the jepa_input - jepa_latent(_ema) margin should grow with
# --synth_node_noise. If both arms move together, the result is a null.
# Compare arms on the probe metric (MAE vs persistence), never on the training
# loss: the JEPA arms minimise losses in different spaces.
#
# Submit:   bash scripts/synthetic/slurm_synth_noise_sweep.sh --list
#           # --list prints the exact --array range for the current grid
#           sbatch --array=0-349%16 scripts/synthetic/slurm_synth_noise_sweep.sh
# Report:   python scripts/synthetic/aggregate_synth_sweep.py
# Pilot:    NOISES=0.5 ARMS="jepa_latent jepa_input" bash ...sweep.sh --list

set -uo pipefail

NOISES=${NOISES:-"0.0 0.1 0.25 0.5 1.0"}
ARMS=${ARMS:-"jepa_latent jepa_latent_ema jepa_input enc_only cldg ddgcl idol"}
# Generation is cheap and the margin is paired within a seed, so seeds are the
# cheapest variance reduction; the verdict rule needs a standard error.
SEEDS=${SEEDS:-"42 43 44 45 46 47 48 49 50 51"}

build_grid() {
    GRID=()
    for n in $NOISES; do for a in $ARMS; do for s in $SEEDS; do
        GRID+=("$n|$a|$s")
    done; done; done
}
build_grid

if [[ "${1:-}" == "--list" ]]; then
    echo "grid size = ${#GRID[@]}  ->  sbatch --array=0-$(( ${#GRID[@]} - 1 ))%8 $0"
    for i in "${!GRID[@]}"; do printf "  [%2d] noise=%s arm=%s seed=%s\n" "$i" \
        "$(cut -d'|' -f1 <<<"${GRID[$i]}")" "$(cut -d'|' -f2 <<<"${GRID[$i]}")" \
        "$(cut -d'|' -f3 <<<"${GRID[$i]}")"; done
    exit 0
fi

mkdir -p logs outputs
REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"
if [[ -d "$HOME/.conda/envs/cjepa" ]]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate cjepa
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

IDX=${SLURM_ARRAY_TASK_ID:-0}
if (( IDX >= ${#GRID[@]} )); then
    echo "[error] array index $IDX >= grid size ${#GRID[@]}" >&2; exit 2
fi
CELL="${GRID[$IDX]}"
NOISE=$(cut -d'|' -f1 <<<"$CELL")
ARM=$(cut -d'|' -f2 <<<"$CELL")
SEED=$(cut -d'|' -f3 <<<"$CELL")

# --- generator + geometry, identical for every arm --------------------------
# ou_sigma is held low on purpose: it is latent noise, which makes the cause
# itself unpredictable and hurts every method. Only node_noise should vary.
# MOMENTUM is pinned here rather than left to the CLI default because it is
# the knob that creates headroom over persistence.
NODES=${NODES:-200}
STEPS=${STEPS:-300}
OU_SIGMA=${OU_SIGMA:-0.15}
MOMENTUM=${MOMENTUM:-0.8}
TARGET_NOISE=${TARGET_NOISE:--1}      # -1 = match the feature SNR; see header
SWITCH_MODE=${SWITCH_MODE:-latent_driven}
HIST=${HIST:-3}
NPRED=${NPRED:-1}
NODE_DIM=${NODE_DIM:-128}
ENC_KIND=${ENC_KIND:-gcn}
# What the shared encoder is pretrained on. This decides what the latent arms
# can predict: --predict_space latent inherits whatever its target encoder
# preserved, while --predict_space input reconstructs raw features. Changing
# it changes the encoders, so give it its own ROOT.
PRETEXT=${PRETEXT:-edge_mask}
# Extra flags for the pretext head. feature_mask reconstructs the semantic
# channels from --graph_feat_dim onward, and its default (8) would leave half
# of this generator's 16 all-semantic dims unsupervised, so the mechanism test
# is:  PRETEXT=feature_mask PRETEXT_ARGS="--graph_feat_dim 0"
PRETEXT_ARGS=${PRETEXT_ARGS:-}
ENC_EPOCHS=${ENC_EPOCHS:-300}
JEPA_EPOCHS=${JEPA_EPOCHS:-50}
BASE_EPOCHS=${BASE_EPOCHS:-100}
PATIENCE=${PATIENCE:-20}
BATCH=${BATCH:-8}
NW=${SLURM_CPUS_PER_TASK:-8}

DATA_ARG="--data_source synthetic_dyn --synth_nodes $NODES --synth_steps $STEPS \
--synth_node_noise $NOISE --synth_target_noise $TARGET_NOISE \
--synth_ou_sigma $OU_SIGMA --synth_momentum $MOMENTUM \
--synth_switch_mode $SWITCH_MODE"
ENCODER_ARG="--encoder_kind $ENC_KIND --encoder_hidden 128 --node_dim $NODE_DIM"
WINDOW_ARG="--history_size $HIST --num_preds $NPRED"
ES_ARG="--early_stop_patience $PATIENCE --early_stop_min_delta 0.0"
PROBE_ARGS="$WINDOW_ARG --probe_epochs 10 --probe_hidden 64 --persist_residual \
--batch_size $BATCH --num_workers $NW --seed $SEED --device cuda"

# One results root per geometry. TAG carries only noise and seed (the
# aggregator's directory regex depends on that), so anything else that changes
# the data or the window (HIST, NPRED, NODES, STEPS) must go to its own ROOT:
#   ROOT=outputs/synth_sweep_h6p6 HIST=6 NPRED=6 sbatch ...
#   python scripts/synthetic/aggregate_synth_sweep.py --root outputs/synth_sweep_h6p6
ROOT=${ROOT:-outputs/synth_sweep}
TAG="n${NOISE}_s${SEED}"
OUT="$ROOT/${ARM}_${TAG}"
mkdir -p "$OUT"
echo "[cell $IDX] noise=$NOISE arm=$ARM seed=$SEED -> $OUT"

# --- shared pretrained encoder for the two JEPA arms ------------------------
# Both JEPA arms must start from the same encoder. flock stops concurrent
# array cells from pretraining it at once.
jepa_encoder() {
    ENC_DIR="$ROOT/_encoder_${TAG}"
    ENC_CKPT="$ENC_DIR/encoder_synthetic_dyn_${ENC_KIND}_${PRETEXT}.ckpt"
    mkdir -p "$ENC_DIR"
    exec 9>"$ENC_DIR/.lock"
    flock -w 7200 9 || { echo "[FATAL] encoder lock timeout" >&2; exit 10; }
    if [[ ! -f "$ENC_CKPT" ]]; then
        echo "=== shared encoder pretrain (lock held by cell $IDX) ==="
        srun --export=ALL python src/train/pretrain_encoder.py $DATA_ARG \
            $ENCODER_ARG $WINDOW_ARG $ES_ARG \
            --decoder_hidden 64 --mask_rate 0.25 --pretext "$PRETEXT" \
            $PRETEXT_ARGS \
            --batch_size "$BATCH" --num_workers "$NW" --device cuda \
            --epochs "$ENC_EPOCHS" --lr 1e-3 --seed "$SEED" --out_dir "$ENC_DIR"
        if [[ ! -f "$ENC_CKPT" ]]; then
            flock -u 9; echo "[FATAL] encoder pretrain failed" >&2; exit 11
        fi
    else
        echo "[skip] shared encoder exists"
    fi
    flock -u 9; exec 9>&-
}

case "$ARM" in
  jepa_latent|jepa_input|jepa_latent_ema)
    case "$ARM" in
      jepa_input)      SPACE=input ; TARGET_FLAGS="" ;;
      jepa_latent)     SPACE=latent; TARGET_FLAGS="" ;;
      # Learned target space: the online encoder trains against an EMA copy
      # under VICReg, which is what lets the latent space DISCARD the
      # unpredictable observation noise instead of encoding it.
      jepa_latent_ema) SPACE=latent
                       TARGET_FLAGS="--train_encoder --ema_target \
--vicreg_var 0.5 --vicreg_cov 0.05" ;;
    esac
    jepa_encoder
    srun --export=ALL python src/train/train_delos.py --no_attention_mask \
        $DATA_ARG $ENCODER_ARG $WINDOW_ARG $ES_ARG \
        --encoder_ckpt "$ENC_CKPT" \
        --downstream_task regression --mask_scheme node_mask --residual_head \
        --persist_residual --predict_space "$SPACE" $TARGET_FLAGS \
        --num_masked_nodes 20 --depth 4 --heads 4 --mlp_dim 512 \
        --batch_size "$BATCH" --num_workers "$NW" --device cuda \
        --epochs "$JEPA_EPOCHS" --lr 5e-4 --probe_epochs 10 --probe_hidden 64 \
        --seed "$SEED" --out_dir "$OUT"
    [[ -f "$OUT/best.ckpt" ]] || { echo "[FATAL] $ARM train failed" >&2; exit 12; }
    ;;

  enc_only)
    # Same shared encoder as the JEPA arms, no predictor: the difference
    # between this row and jepa_* is exactly what the predictor contributes.
    jepa_encoder
    # DATA_ARG is passed explicitly: the probe would otherwise rebuild the
    # dataset from the checkpoint's saved args, which omit any flag added
    # after the checkpoint was written.
    srun --export=ALL python scripts/regression/probe_encoder_regression.py \
        --encoder_ckpt "$ENC_CKPT" --out_dir "$OUT" $DATA_ARG $PROBE_ARGS
    [[ -f "$OUT/test_metrics.json" ]] || {
        echo "[FATAL] enc_only probe failed" >&2; exit 16; }
    ;;

  cldg)
    srun --export=ALL python -m src.baselines.cldg.train_cldg $DATA_ARG \
        $ENCODER_ARG $WINDOW_ARG $ES_ARG \
        --snapshots 4 --views 2 --strategy random --tau 0.07 \
        --node_batch_size 256 --steps_per_epoch 20 --val_steps 10 --device cuda \
        --epochs "$BASE_EPOCHS" --lr 4e-3 --seed "$SEED" --out_dir "$OUT"
    [[ -f "$OUT/encoder.ckpt" ]] || { echo "[FATAL] cldg failed" >&2; exit 13; }
    srun --export=ALL python scripts/regression/probe_encoder_regression.py \
        --encoder_ckpt "$OUT/encoder.ckpt" $DATA_ARG $PROBE_ARGS
    ;;

  ddgcl)
    srun --export=ALL python -m src.baselines.ddgcl.train_ddgcl $DATA_ARG \
        $ENCODER_ARG $WINDOW_ARG $ES_ARG \
        --max_gap 4 --time_dim 32 --tau 0.01 --beta 1.5 \
        --node_batch_size 256 --steps_per_epoch 20 --val_steps 10 --device cuda \
        --epochs "$BASE_EPOCHS" --lr 4e-3 --seed "$SEED" --out_dir "$OUT"
    [[ -f "$OUT/encoder.ckpt" ]] || { echo "[FATAL] ddgcl failed" >&2; exit 14; }
    srun --export=ALL python scripts/regression/probe_encoder_regression.py \
        --encoder_ckpt "$OUT/encoder.ckpt" $DATA_ARG $PROBE_ARGS
    ;;

  idol)
    srun --export=ALL python -m src.baselines.idol.train_idol $DATA_ARG \
        $ENCODER_ARG $WINDOW_ARG $ES_ARG \
        --proj_layers 1 --node_batch_size 256 --device cuda \
        --epochs "$BASE_EPOCHS" --lr 1e-3 --seed "$SEED" --out_dir "$OUT"
    [[ -f "$OUT/encoder.ckpt" ]] || { echo "[FATAL] idol failed" >&2; exit 15; }
    srun --export=ALL python scripts/regression/probe_encoder_regression.py \
        --encoder_ckpt "$OUT/encoder.ckpt" $DATA_ARG $PROBE_ARGS
    ;;

  *) echo "[error] unknown arm '$ARM'" >&2; exit 2 ;;
esac

echo "[done] cell $IDX ($ARM, noise=$NOISE, seed=$SEED) -> $OUT"
