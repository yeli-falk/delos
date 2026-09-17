#!/usr/bin/env bash
#SBATCH --job-name=cjepa-jodie
#SBATCH --output=logs/cjepa-jodie-%j.out
#SBATCH --error=logs/cjepa-jodie-%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --time=10:00:00
#SBATCH --partition=gpu
#
# Full C-JEPA-DG pipeline on a JODIE dataset (Reddit or Wikipedia), binned
# into DTDG snapshots:
#   1) edge-masking encoder pretraining
#   2) C-JEPA training (with new-edge probe split in the evaluator)
#   3) representation diagnostic (eff_rank, CKA, structural probes)
#
# Submit (Reddit):     sbatch scripts/dblp/slurm_jodie_pipeline.sh
# Submit (Wikipedia):  DATASET=wikipedia sbatch scripts/dblp/slurm_jodie_pipeline.sh
# More bins:           BINS=60 sbatch scripts/dblp/slurm_jodie_pipeline.sh

set -uo pipefail
mkdir -p logs outputs dataset/jodie

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"

if [[ -d "$HOME/.conda/envs/cjepa" ]]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate cjepa
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo "[slurm] repo_root=$REPO_ROOT  node=$(hostname)"

DATASET=${DATASET:-reddit}
SEED=${SEED:-42}
COMMUNITY=${COMMUNITY:-0}
COMMUNITY_FLAG=""
if [[ "$COMMUNITY" == "1" ]]; then
    COMMUNITY_FLAG="--community_masking"
fi
if [[ "$COMMUNITY" == "1" ]]; then
    TAG=${TAG:-jodie_${DATASET}_community}
else
    TAG=${TAG:-jodie_${DATASET}}
fi
ENC_EPOCHS=${ENC_EPOCHS:-30}
JEPA_EPOCHS=${JEPA_EPOCHS:-50}
PROBE_EPOCHS=${PROBE_EPOCHS:-3}
ENC_LR=${ENC_LR:-1e-3}
JEPA_LR=${JEPA_LR:-5e-4}
# Subset / binning / encoder / window / seed / early-stop from the shared
# config (TASK=link_pred), so this JEPA row and the CLDG / DDGCL / IDOL /
# heuristic rows of the link-prediction table describe the same experiment.
TASK=link_pred
# Sourcing normalises DATASET to the --data_source spelling
# (reddit -> jodie_reddit); SHORT keeps the bare name for file paths.
source "$REPO_ROOT/scripts/dblp/_dataset_config.sh"
SHORT=${DATASET#jodie_}
echo "[config] COMMUNITY=$COMMUNITY  TAG=$TAG  ENC_EPOCHS=$ENC_EPOCHS  JEPA_EPOCHS=$JEPA_EPOCHS  PROBE_EPOCHS=$PROBE_EPOCHS"

case "$DATASET" in
    jodie_reddit)    URL="http://snap.stanford.edu/jodie/reddit.csv"    ;;
    jodie_wikipedia) URL="http://snap.stanford.edu/jodie/wikipedia.csv" ;;
    *) echo "[error] DATASET must be 'reddit' or 'wikipedia'"; exit 2 ;;
esac

CSV="dataset/jodie/${SHORT}.csv"
if [[ ! -f "$CSV" ]]; then
    echo "[stage 0] downloading $SHORT CSV from $URL"
    wget -O "$CSV" "$URL"
    wget_rc=$?
    sz=$(stat -c %s "$CSV" 2>/dev/null || echo 0)
    echo "[stage 0] wget rc=$wget_rc  size=$sz bytes"
    if [[ $wget_rc -ne 0 || $sz -lt 1000 ]]; then
        echo "[FATAL] download failed at $CSV (rc=$wget_rc, size=$sz)" >&2
        echo "  Compute nodes on most HPC systems lack outbound internet." >&2
        echo "  Workaround: download from the LOGIN node first," >&2
        echo "    mkdir -p $REPO_ROOT/dataset/jodie && cd $REPO_ROOT" >&2
        echo "    wget $URL -O dataset/jodie/${SHORT}.csv" >&2
        echo "  then resubmit this job; the script will skip the download." >&2
        exit 10
    fi
fi
echo "[stage 0] $CSV size=$(du -h "$CSV" | cut -f1)"

ENC_DIR="outputs/${TAG}_encoder"
CJEPA_DIR="outputs/${TAG}_cjepa"
DIAG_DIR="outputs/${TAG}_diagnostic"
# pretrain_encoder.py writes encoder_<src>_<kind>_<pretext>.ckpt, NOT
# encoder.ckpt — reference that exact templated name everywhere below.
# (encoder_kind comes from the shared config; pretext=edge_mask is pinned in
# the stage-1 command.)
ENC_CKPT="$ENC_DIR/encoder_${DATASET}_${ENC_KIND}_edge_mask.ckpt"

# -----------------------------------------------------------------------------
# Stage 1: encoder pretraining via edge masking
# -----------------------------------------------------------------------------
if [[ ! -f "$ENC_CKPT" ]]; then
    echo
    echo "================================================================="
    echo "  Stage 1/3: encoder pretrain ($TAG) -> $ENC_DIR"
    echo "================================================================="
    srun --export=ALL python src/train/pretrain_encoder.py $DATA_ARG \
        $ENCODER_ARG \
        --decoder_hidden 64 --mask_rate 0.25 --pretext edge_mask \
        --batch_size 8 --num_workers ${SLURM_CPUS_PER_TASK:-8} \
        --device cuda --epochs "$ENC_EPOCHS" --lr "$ENC_LR" \
        $EARLY_STOP_ARG $WINDOW_ARG \
        --seed "$SEED" --out_dir "$ENC_DIR"
    enc_rc=$?
    if [[ $enc_rc -ne 0 || ! -f "$ENC_CKPT" ]]; then
        echo "[FATAL] Stage 1 failed (rc=$enc_rc, ckpt exists=$( [[ -f "$ENC_CKPT" ]] && echo y || echo n ))" >&2
        exit 11
    fi
else
    echo "[stage 1] $ENC_CKPT already exists, skipping pretrain"
fi

# -----------------------------------------------------------------------------
# Stage 2: C-JEPA training (baseline configuration)
# -----------------------------------------------------------------------------
echo
echo "================================================================="
echo "  Stage 2/3: C-JEPA train ($TAG) -> $CJEPA_DIR"
echo "================================================================="
srun --export=ALL python src/train/train_dyg_jepa.py --no_attention_mask $DATA_ARG $COMMUNITY_FLAG \
    --encoder_ckpt "$ENC_CKPT" \
    $WINDOW_ARG $ENCODER_ARG \
    --num_masked_nodes 30 --depth 4 --heads 4 --mlp_dim 512 \
    --batch_size 8 --num_workers ${SLURM_CPUS_PER_TASK:-8} \
    --device cuda --epochs "$JEPA_EPOCHS" --lr "$JEPA_LR" --probe_epochs "$PROBE_EPOCHS" \
    $EARLY_STOP_ARG \
    --seed "$SEED" --out_dir "$CJEPA_DIR"
cjepa_rc=$?
if [[ $cjepa_rc -ne 0 || ! -f "$CJEPA_DIR/best.ckpt" ]]; then
    echo "[FATAL] Stage 2 failed (rc=$cjepa_rc, best.ckpt exists=$( [[ -f "$CJEPA_DIR/best.ckpt" ]] && echo y || echo n ))" >&2
    exit 12
fi

# -----------------------------------------------------------------------------
# Stage 3: representation diagnostic
# -----------------------------------------------------------------------------
echo
echo "================================================================="
echo "  Stage 3/3: diagnostic ($TAG) -> $DIAG_DIR"
echo "================================================================="
srun --export=ALL python src/train/diagnose_jepa_dblp.py $DATA_ARG \
    --encoder_ckpt "$ENC_CKPT" \
    --jepa_ckpt "$CJEPA_DIR/best.ckpt" \
    --split test \
    --batch_size 4 --num_workers ${SLURM_CPUS_PER_TASK:-8} \
    --device cuda \
    --out_dir "$DIAG_DIR"

echo
echo "[done] JODIE $DATASET pipeline complete."
echo "  link prediction (all / new):  cat $CJEPA_DIR/test_metrics.json"
echo "  representation:               cat $DIAG_DIR/diagnostic.json"
