#!/usr/bin/env bash
#SBATCH --job-name=dblp-idol
#SBATCH --output=logs/dblp-idol-%j.out
#SBATCH --error=logs/dblp-idol-%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --time=10:00:00
#SBATCH --partition=gpu
#
# IDOL baseline (Topology-monitorable Contrastive Learning on Dynamic Graphs,
# Zhu et al., KDD'24), two-stage so it is directly comparable to the DELOS
# pipeline:
#   1) train the IDOL encoder with GGD group discrimination over PPR-propagated
#      features (the ppr_gcn encoder = dense analogue of InstantGNN's dynamic PPR)
#   2) freeze it and fit the SAME link-prediction probe DELOS uses
#
# Outputs:
#   outputs/idol_<TAG>/encoder.ckpt
#   outputs/idol_<TAG>_probe/analysis.json
#
# Run:    DATASET=dblp sbatch scripts/dblp/slurm_idol.sh
# Knobs:  DATASET=reddit_body  SEED=43  TAG=<name>  ENC_KIND=ppr_gcn
#         Link-pred datasets: dblp, jodie_reddit, jodie_wikipedia, reddit_body,
#         reddit_title (or synthetic). Stage 1 + stage 2 share these flags.

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

# Link-prediction datasets only. Sets DATA_ARG / WINDOW_ARG / ENCODER_ARG /
# EARLY_STOP_ARG / SEED from the shared config, so both stages and every other
# method construct an identical node set.
#
# The shared config sets ENC_KIND=gcn for every method. IDOL's native encoder
# is `ppr_gcn` (dynamic PPR propagation); run `ENC_KIND=ppr_gcn sbatch ...`
# for the faithful-IDOL ablation, which is then no longer encoder-matched to
# the other rows of the table.
source "$REPO_ROOT/scripts/dblp/_linkpred_data_arg.sh"
echo "[config] TAG=$TAG"

# Stage 1: train the IDOL encoder (GGD group discrimination on PPR features)
ENC_DIR="outputs/idol_${TAG}"
if [[ ! -f "$ENC_DIR/encoder.ckpt" ]]; then
    echo
    echo "================================================================="
    echo "  Stage 1/2: IDOL GGD pretrain -> $ENC_DIR"
    echo "================================================================="
    srun --export=ALL python -m src.baselines.idol.train_idol $DATA_ARG \
        $ENCODER_ARG $WINDOW_ARG $EARLY_STOP_ARG \
        --proj_layers 1 --node_batch_size 256 \
        --device cuda --epochs 100 --lr 1e-3 \
        --seed "$SEED" --out_dir "$ENC_DIR"
    enc_rc=$?
    if [[ $enc_rc -ne 0 || ! -f "$ENC_DIR/encoder.ckpt" ]]; then
        echo "[FATAL] Stage 1 failed (rc=$enc_rc)" >&2; exit 11
    fi
else
    echo "[stage 1] $ENC_DIR/encoder.ckpt exists, skipping IDOL pretrain"
fi

# Stage 2: freeze the IDOL encoder, fit the shared link-prediction probe
PROBE_DIR="outputs/idol_${TAG}_probe"
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
echo "[done] IDOL baseline complete ($DATASET)."
echo "       encoder : $ENC_DIR/encoder.ckpt"
echo "       metrics : $PROBE_DIR/analysis.json  (link_pred.encoder.{all,new}; compare to DELOS)"
