#!/usr/bin/env bash
#SBATCH --job-name=dblp-bgrl
#SBATCH --output=logs/dblp-bgrl-%j.out
#SBATCH --error=logs/dblp-bgrl-%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --time=10:00:00
#SBATCH --partition=gpu
#
# BGRL baseline (Large-Scale Representation Learning on Graphs via
# Bootstrapping, Thakoor et al., ICLR'22) -- the STATIC negative-free control,
# two-stage so it is directly comparable to the DELOS pipeline:
#   1) train the BGRL encoder: each step draws ONE training snapshot, builds
#      two augmented views of it (edge removal + feature-dimension masking),
#      and trains an online encoder + predictor to match an EMA target
#      encoder's embedding of the other view. No negatives, no history, no two
#      snapshots compared: the objective is static.
#   2) freeze the ONLINE encoder and fit the SAME link-prediction probe
#      DELOS uses, which reads it at the last history snapshot ("static
#      SSL, last frame").
#
# Outputs:
#   outputs/bgrl_<TAG>/encoder.ckpt
#   outputs/bgrl_<TAG>_probe/analysis.json
#
# Run:    DATASET=dblp sbatch scripts/dblp/slurm_bgrl.sh
# Knobs:  DATASET=reddit_body  SEED=43  TAG=<name>  EMA=0.99  SELECT=val|last
#         PE1=0.2 PE2=0.3 PF1=0.2 PF2=0.1  (reference WikiCS config)
#         Link-pred datasets: dblp, jodie_reddit, jodie_wikipedia, reddit_body,
#         reddit_title (or synthetic). Stage 1 + stage 2 share these flags.
#
# Dataset / encoder / window / early-stop config comes from
# scripts/dblp/_dataset_config.sh so it is identical to every other method.

set -uo pipefail
mkdir -p logs outputs

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"

if [[ -d "$HOME/.conda/envs/cjepa" ]]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate cjepa
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo "[slurm] repo_root=$REPO_ROOT  node=$(hostname)"

DATASET=${DATASET:-dblp}
# Seed is part of TAG so a multi-seed sweep does not reuse seed 42's checkpoints.
TAG=${TAG:-${DATASET}_s${SEED:-42}}
EMA=${EMA:-0.99}
SELECT=${SELECT:-val}
# Momentum anneal horizon in epochs (0 = whole run). Under SELECT=val the run
# stops early and a whole-run anneal never finishes; 50 matches every other
# BGRL invocation and the DELOS cosine arms.
EMA_ANNEAL_EPOCHS="${EMA_ANNEAL_EPOCHS:-50}"
ANNEAL_ARG=""; [[ "$EMA_ANNEAL_EPOCHS" != "0" ]] && ANNEAL_ARG="--ema_anneal_epochs $EMA_ANNEAL_EPOCHS"
PE1=${PE1:-0.2}
PE2=${PE2:-0.3}
PF1=${PF1:-0.2}
PF2=${PF2:-0.1}

# Link-prediction datasets only. Sets DATA_ARG / WINDOW_ARG / ENCODER_ARG /
# EARLY_STOP_ARG / SEED from the shared config, so both stages and every other
# method construct an identical node set.
source "$REPO_ROOT/scripts/dblp/_linkpred_data_arg.sh"
echo "[config] TAG=$TAG  EMA=$EMA  SELECT=$SELECT  p_e=($PE1,$PE2)  p_f=($PF1,$PF2)"

# Stage 1: train the BGRL encoder (static two-view bootstrap, one snapshot/step)
ENC_DIR="outputs/bgrl_${TAG}"
if [[ ! -f "$ENC_DIR/encoder.ckpt" ]]; then
    echo
    echo "================================================================="
    echo "  Stage 1/2: BGRL bootstrap pretrain -> $ENC_DIR"
    echo "================================================================="
    srun --export=ALL python -m src.baselines.bgrl.train_bgrl $DATA_ARG \
        $ENCODER_ARG $WINDOW_ARG $EARLY_STOP_ARG \
        --p_e1 "$PE1" --p_e2 "$PE2" --p_f1 "$PF1" --p_f2 "$PF2" \
        --ema_momentum "$EMA" --select "$SELECT" $ANNEAL_ARG \
        --steps_per_epoch 20 --val_steps 10 \
        --device cuda --epochs 100 --lr 5e-4 \
        --seed "$SEED" --out_dir "$ENC_DIR"
    enc_rc=$?
    if [[ $enc_rc -ne 0 || ! -f "$ENC_DIR/encoder.ckpt" ]]; then
        echo "[FATAL] Stage 1 failed (rc=$enc_rc)" >&2; exit 11
    fi
else
    echo "[stage 1] $ENC_DIR/encoder.ckpt exists, skipping BGRL pretrain"
fi

# Stage 2: freeze the BGRL encoder, fit the shared link-prediction probe
PROBE_DIR="outputs/bgrl_${TAG}_probe"
echo
echo "================================================================="
echo "  Stage 2/2: frozen-encoder + link-prediction probe -> $PROBE_DIR"
echo "================================================================="
srun --export=ALL python -m src.train.analyze_jepa --skip_diagnostic \
    --link_sources encoder $DATA_ARG $WINDOW_ARG \
    --encoder_ckpt "$ENC_DIR/encoder.ckpt" \
    --num_workers ${SLURM_CPUS_PER_TASK:-8} \
    --device cuda --seed "$SEED" --out_dir "$PROBE_DIR"

echo
echo "[done] BGRL baseline complete ($DATASET)."
echo "       encoder : $ENC_DIR/encoder.ckpt"
echo "       metrics : $PROBE_DIR/analysis.json  (link_pred.encoder.{all,new}; compare to DELOS)"
