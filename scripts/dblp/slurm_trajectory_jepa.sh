#!/usr/bin/env bash
#SBATCH --job-name=cjepa-traj
#SBATCH --output=logs/cjepa-traj-%j.out
#SBATCH --error=logs/cjepa-traj-%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --time=08:00:00
#SBATCH --partition=gpu
#
# Trajectory-JEPA pipeline: re-tokenise time at the window level, V-JEPA
# style. Tokens are now per-(window, node) instead of per-(timestep, node);
# masking is bidirectional (random rectangles in K x N), no past/future
# split. This is the one remaining unexplored JEPA variant for dynamic
# graphs.
#
# Stages:
#   1) Encoder pretraining (reuses standard edge-mask pretext on the
#      long-sequence dataset).
#   2) Trajectory-JEPA training: window pooler + transformer predictor,
#      multi-block masking, latent MSE on masked positions.
#   3) Diagnostic adapted for trajectory tokens (last-window prediction).
#
# Falsification test (SBM-8): trajectory JEPA must show meaningfully
# higher jepa_eff_rank and probe_block_acc than node-level JEPA. If not,
# token granularity is not the bottleneck and the JEPA-on-DTDG hypothesis
# is fully falsified.
#
# Submit (SBM-8, single seed):
#     SEED=42 sbatch scripts/dblp/slurm_trajectory_jepa.sh
# Submit (Reddit-Body):
#     DATASET=reddit_body SEED=42 sbatch scripts/dblp/slurm_trajectory_jepa.sh
# Multi-seed SBM:
#     for S in 42 43 44 45 46; do SEED=$S sbatch scripts/dblp/slurm_trajectory_jepa.sh; done

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

# Defaults: SBM-8 synthetic stress test.
DATASET=${DATASET:-sbm8}
SEED=${SEED:-42}
WINDOW_SIZE=${WINDOW_SIZE:-4}
HIST=${HIST:-20}
NPRED=${NPRED:-4}
NODE_POS_MODE=${NODE_POS_MODE:-off}
case "$NODE_POS_MODE" in
    off|learned|from_input) ;;
    *) echo "[error] NODE_POS_MODE must be off|learned|from_input"; exit 2 ;;
esac
TAG=${TAG:-${DATASET}_traj_w${WINDOW_SIZE}_${NODE_POS_MODE}_s${SEED}}
echo "[config] DATASET=$DATASET  SEED=$SEED  W=$WINDOW_SIZE  HIST=$HIST  "\
"NPRED=$NPRED  NODE_POS_MODE=$NODE_POS_MODE  TAG=$TAG"

case "$DATASET" in
    sbm8)
        # 80 years gives ~56 train timesteps under the 70/10/20 split,
        # easily fitting a 24-timestep trajectory window (history+num_preds).
        DATA_ARG="--synthetic --synthetic_kind sbm --synthetic_nodes 1000 --synthetic_years 80 --synthetic_blocks 8"
        ;;
    sbm16)
        DATA_ARG="--synthetic --synthetic_kind sbm --synthetic_nodes 1000 --synthetic_years 80 --synthetic_blocks 16"
        ;;
    reddit_body)
        TSV="dataset/reddit/soc-redditHyperlinks-body.tsv"
        if [[ ! -f "$TSV" ]]; then
            echo "[FATAL] $TSV missing — download on login node first." >&2
            exit 10
        fi
        # 72 bins ≈ 2.5-week granularity over the SNAP dump's ~3 years.
        # Gives ~50 train bins, comfortable headroom for 24-timestep windows.
        DATA_ARG="--data_source reddit_body --reddit_data_path $TSV --reddit_num_bins 72 --reddit_top_k 1000"
        ;;
    *) echo "[error] unknown DATASET=$DATASET"; exit 2 ;;
esac

ENC_DIR="outputs/${TAG}_encoder"
DELOS_DIR="outputs/${TAG}_cjepa"
DIAG_DIR="outputs/${TAG}_diagnostic"

# -----------------------------------------------------------------------------
# Stage 1: encoder pretraining (edge-mask, default settings).
# Use the same encoder pretraining as the node-JEPA pipeline -- the
# trajectory-JEPA contribution is at the *prediction* stage, not the encoder.
# -----------------------------------------------------------------------------
if [[ ! -f "$ENC_DIR/encoder.ckpt" ]]; then
    echo
    echo "================================================================="
    echo "  Stage 1/3: encoder pretrain ($TAG) -> $ENC_DIR"
    echo "================================================================="
    srun --export=ALL python src/train/pretrain_encoder.py $DATA_ARG \
        --encoder_kind tgcn --encoder_hidden 128 --node_dim 128 \
        --decoder_hidden 64 --mask_rate 0.25 \
        --batch_size 8 --num_workers ${SLURM_CPUS_PER_TASK:-8} \
        --device cuda --epochs 30 --lr 1e-3 \
        --history_size 3 --num_preds 1 \
        --seed "$SEED" --out_dir "$ENC_DIR"
    enc_rc=$?
    if [[ $enc_rc -ne 0 || ! -f "$ENC_DIR/encoder.ckpt" ]]; then
        echo "[FATAL] Stage 1 failed (rc=$enc_rc)" >&2; exit 11
    fi
else
    echo "[stage 1] $ENC_DIR/encoder.ckpt exists, skipping pretrain"
fi

# -----------------------------------------------------------------------------
# Stage 2: trajectory-JEPA training. Uses the long-sequence window
# (HIST + NPRED = T_long timesteps, divided into K = T_long / W windows).
# -----------------------------------------------------------------------------
echo
echo "================================================================="
echo "  Stage 2/3: trajectory-JEPA train ($TAG) -> $DELOS_DIR"
echo "================================================================="
srun --export=ALL python src/train/train_delos.py --mask_scheme trajectory_block --downstream_task none $DATA_ARG \
    --encoder_ckpt "$ENC_DIR/encoder.ckpt" \
    --history_size "$HIST" --num_preds "$NPRED" \
    --window_size "$WINDOW_SIZE" --pool mean \
    --encoder_kind tgcn --encoder_hidden 128 --node_dim 128 \
    --depth 4 --heads 4 --mlp_dim 512 \
    --node_pos_mode "$NODE_POS_MODE" \
    --n_blocks 4 --block_k_range 1 3 --block_n_frac 0.1 0.3 \
    --batch_size 4 --num_workers ${SLURM_CPUS_PER_TASK:-8} \
    --device cuda --epochs 50 --lr 5e-4 \
    --seed "$SEED" --out_dir "$DELOS_DIR"
delos_rc=$?
if [[ $delos_rc -ne 0 || ! -f "$DELOS_DIR/best.ckpt" ]]; then
    echo "[FATAL] Stage 2 failed (rc=$delos_rc)" >&2; exit 12
fi

# -----------------------------------------------------------------------------
# Stage 3: trajectory diagnostic. Adds Louvain block probe automatically.
# -----------------------------------------------------------------------------
echo
echo "================================================================="
echo "  Stage 3/3: trajectory diagnostic ($TAG) -> $DIAG_DIR"
echo "================================================================="
srun --export=ALL python src/train/diagnose_trajectory_jepa.py $DATA_ARG \
    --jepa_ckpt "$DELOS_DIR/best.ckpt" \
    --history_size "$HIST" --num_preds "$NPRED" \
    --split test --device cuda \
    --batch_size 4 --num_workers ${SLURM_CPUS_PER_TASK:-8} \
    --block_probe louvain \
    --out_dir "$DIAG_DIR"

echo
echo "[done] $TAG complete."
echo "  diagnostic:   cat $DIAG_DIR/diagnostic.json"
echo "  best.ckpt:    $DELOS_DIR/best.ckpt"
