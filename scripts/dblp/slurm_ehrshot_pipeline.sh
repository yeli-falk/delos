#!/usr/bin/env bash
#SBATCH --job-name=cjepa-ehrshot
#SBATCH --output=logs/cjepa-ehrshot-%j.out
#SBATCH --error=logs/cjepa-ehrshot-%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#
# Full DELOS pipeline on the EHRshot per-patient temporal event graphs:
#   0) preprocess the raw EHRshot CSV into a per-patient pickle if the pickle
#      does not exist (or PREPROCESS_FORCE=1 is set)
#   1) edge-masking encoder pretrain
#   2) DELOS training
#   3) representation diagnostic (eff_rank, CKA, structural probes)
#
# Submit (defaults):
#     sbatch scripts/dblp/slurm_ehrshot_pipeline.sh
# Bigger vocab:
#     TOP_K_CODES=1000 sbatch scripts/dblp/slurm_ehrshot_pipeline.sh
# Longer per-patient windows:
#     MAX_VISITS=120 sbatch scripts/dblp/slurm_ehrshot_pipeline.sh
# Force-re-preprocess (e.g. after changing TOP_K_CODES):
#     PREPROCESS_FORCE=1 sbatch scripts/dblp/slurm_ehrshot_pipeline.sh

set -uo pipefail
mkdir -p logs outputs dataset/ehrshot

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"

if [[ -d "$HOME/.conda/envs/cjepa" ]]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate cjepa
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo "[slurm] repo_root=$REPO_ROOT  node=$(hostname)"

# --- config -------------------------------------------------------------------
SEED=${SEED:-42}
TOP_K_CODES=${TOP_K_CODES:-500}
PMI_TOPK=${PMI_TOPK:-10}    # 0 = clique fallback (don't, makes graph degenerate)
MAX_VISITS=${MAX_VISITS:-60}
MAX_WINDOWS=${MAX_WINDOWS:-16}
TAG=${TAG:-ehrshot_k${TOP_K_CODES}_pmi${PMI_TOPK}_s${SEED}}
ENC_EPOCHS=${ENC_EPOCHS:-30}
JEPA_EPOCHS=${JEPA_EPOCHS:-50}
PROBE_EPOCHS=${PROBE_EPOCHS:-3}
ENC_LR=${ENC_LR:-1e-3}
JEPA_LR=${JEPA_LR:-5e-4}
PATIENCE=${PATIENCE:-0}

# Where the raw EHRshot CSV lives. Default is a symlink inside the repo's
# dataset/ dir; on the cluster create it once with:
#   ln -s /path/to/ehrshot/ehrshot.csv dataset/ehrshot/ehrshot.csv
#   ln -s /path/to/ehrshot/person_id_map.csv dataset/ehrshot/person_id_map.csv
# Or override per-submit: EHRSHOT_CSV=/path/to/ehrshot.csv sbatch ...
EHRSHOT_CSV=${EHRSHOT_CSV:-dataset/ehrshot/ehrshot.csv}
EHRSHOT_SPLITS=${EHRSHOT_SPLITS:-dataset/ehrshot/person_id_map.csv}

# The cache dir is overridable so that big pickles can live on a filesystem
# with more quota than the repo.
EHRSHOT_CACHE_DIR=${EHRSHOT_CACHE_DIR:-dataset/ehrshot}
mkdir -p "$EHRSHOT_CACHE_DIR"
# Vocab size and PMI-K are baked into the pickle (PMI is computed at preprocess
# time), so cache per (K, pmi).
PKL="$EHRSHOT_CACHE_DIR/cohort_k${TOP_K_CODES}_pmi${PMI_TOPK}.pkl"

echo "[config] SEED=$SEED  TOP_K_CODES=$TOP_K_CODES  MAX_VISITS=$MAX_VISITS"
echo "[config] MAX_WINDOWS=$MAX_WINDOWS  TAG=$TAG"
echo "[config] ENC_EPOCHS=$ENC_EPOCHS  JEPA_EPOCHS=$JEPA_EPOCHS  PROBE_EPOCHS=$PROBE_EPOCHS  PATIENCE=$PATIENCE"
echo "[config] EHRSHOT_CSV=$EHRSHOT_CSV"
echo "[config] EHRSHOT_SPLITS=$EHRSHOT_SPLITS"
echo "[config] PKL=$PKL"

# Stage 0: preprocess the EHRshot CSV (one-time per TOP_K_CODES / PMI_TOPK)
if [[ ! -f "$PKL" || "${PREPROCESS_FORCE:-0}" == "1" ]]; then
    if [[ ! -f "$EHRSHOT_CSV" ]]; then
        echo "[FATAL] EHRSHOT_CSV=$EHRSHOT_CSV not found." >&2
        echo "  Copy the EHRshot CSV to the cluster first, or override" >&2
        echo "  EHRSHOT_CSV=/cluster/path/to/ehrshot.csv when submitting." >&2
        exit 10
    fi
    echo
    echo "================================================================="
    echo "  Stage 0/3: preprocess EHRshot CSV -> $PKL"
    echo "================================================================="
    srun --export=ALL python scripts/dblp/preprocess_ehrshot.py \
        --csv "$EHRSHOT_CSV" --splits "$EHRSHOT_SPLITS" \
        --top_k_codes "$TOP_K_CODES" --min_visits 4 --min_year 2010 \
        --out "$PKL"
    pre_rc=$?
    if [[ $pre_rc -ne 0 || ! -f "$PKL" ]]; then
        echo "[FATAL] Stage 0 failed (rc=$pre_rc)" >&2
        exit 11
    fi
else
    echo "[stage 0] $PKL already exists, skipping preprocess"
fi

DATA_ARG="--data_source ehrshot --ehrshot_data_path $PKL \
--ehrshot_max_visits $MAX_VISITS --ehrshot_max_windows $MAX_WINDOWS"

ENC_DIR="outputs/${TAG}_encoder"
DELOS_DIR="outputs/${TAG}_cjepa"
DIAG_DIR="outputs/${TAG}_diagnostic"

# Stage 1: encoder pretrain via edge masking
if [[ ! -f "$ENC_DIR/encoder.ckpt" ]]; then
    echo
    echo "================================================================="
    echo "  Stage 1/3: encoder pretrain ($TAG) -> $ENC_DIR"
    echo "================================================================="
    srun --export=ALL python src/train/pretrain_encoder.py $DATA_ARG \
        --encoder_kind tgcn --encoder_hidden 128 --node_dim 128 \
        --decoder_hidden 64 --mask_rate 0.25 \
        --batch_size 8 --num_workers ${SLURM_CPUS_PER_TASK:-8} \
        --device cuda --epochs "$ENC_EPOCHS" --lr "$ENC_LR" \
        --early_stop_patience "$PATIENCE" \
        --history_size 3 --num_preds 1 \
        --seed "$SEED" --out_dir "$ENC_DIR"
    enc_rc=$?
    if [[ $enc_rc -ne 0 || ! -f "$ENC_DIR/encoder.ckpt" ]]; then
        echo "[FATAL] Stage 1 failed (rc=$enc_rc, ckpt exists=$( [[ -f "$ENC_DIR/encoder.ckpt" ]] && echo y || echo n ))" >&2
        exit 12
    fi
else
    echo "[stage 1] $ENC_DIR/encoder.ckpt already exists, skipping pretrain"
fi

# Stage 2: DELOS training
echo
echo "================================================================="
echo "  Stage 2/3: DELOS train ($TAG) -> $DELOS_DIR"
echo "================================================================="
srun --export=ALL python src/train/train_delos.py --no_attention_mask $DATA_ARG \
    --encoder_ckpt "$ENC_DIR/encoder.ckpt" \
    --history_size 3 --num_preds 1 \
    --encoder_kind tgcn --encoder_hidden 128 --node_dim 128 \
    --num_masked_nodes 30 --depth 4 --heads 4 --mlp_dim 512 \
    --batch_size 8 --num_workers ${SLURM_CPUS_PER_TASK:-8} \
    --device cuda --epochs "$JEPA_EPOCHS" --lr "$JEPA_LR" --probe_epochs "$PROBE_EPOCHS" \
    --early_stop_patience "$PATIENCE" \
    --seed "$SEED" --out_dir "$DELOS_DIR"
delos_rc=$?
if [[ $delos_rc -ne 0 || ! -f "$DELOS_DIR/best.ckpt" ]]; then
    echo "[FATAL] Stage 2 failed (rc=$delos_rc, best.ckpt exists=$( [[ -f "$DELOS_DIR/best.ckpt" ]] && echo y || echo n ))" >&2
    exit 13
fi

# Stage 3: representation diagnostic
echo
echo "================================================================="
echo "  Stage 3/3: diagnostic ($TAG) -> $DIAG_DIR"
echo "================================================================="
srun --export=ALL python src/train/diagnose_jepa_dblp.py $DATA_ARG \
    --encoder_ckpt "$ENC_DIR/encoder.ckpt" \
    --jepa_ckpt "$DELOS_DIR/best.ckpt" \
    --split test \
    --batch_size 4 --num_workers ${SLURM_CPUS_PER_TASK:-8} \
    --device cuda \
    --out_dir "$DIAG_DIR"

echo
echo "[done] EHRshot pipeline complete."
echo "  link prediction (all / new):  cat $DELOS_DIR/test_metrics.json"
echo "  representation:               cat $DIAG_DIR/diagnostic.json"
