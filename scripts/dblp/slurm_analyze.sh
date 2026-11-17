#!/usr/bin/env bash
#SBATCH --job-name=jepa-analyze
#SBATCH --output=logs/jepa-analyze-%j.out
#SBATCH --error=logs/jepa-analyze-%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --time=06:00:00
#SBATCH --partition=gpu
#
# Unified analysis for a trained encoder + DELOS predictor. Runs BOTH:
#   (1) the representation diagnostic (effective rank, CKA, *trained* structural
#       probes for degree / PageRank / k-core / is-hub on encoder vs JEPA), and
#   (2) the downstream link-prediction ablation across four representation
#       sources (random / encoder / jepa / supervised), MRR + Hits@k per horizon.
#
# Dataset-agnostic: defaults to the local DBLP edges if present, else a
# synthetic smoke graph. Point it at any dataset via env vars / extra flags.
#
# Run:           sbatch scripts/dblp/slurm_analyze.sh
# Diag only:     sbatch scripts/dblp/slurm_analyze.sh --skip_linkpred
# Link only:     sbatch scripts/dblp/slurm_analyze.sh --skip_diagnostic
# Variants:      ENC=outputs/dblp_encoder_unidir JEPA=outputs/dblp_cjepa_unidir \
#                    OUT=outputs/analyze_dblp_unidir sbatch scripts/dblp/slurm_analyze.sh

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

ENC=${ENC:-outputs/dblp_encoder}
JEPA=${JEPA:-outputs/dblp_cjepa}
OUT=${OUT:-outputs/analyze_dblp}
SPLIT=${SPLIT:-test}

DATA_ARG="--synthetic --synthetic_nodes 500 --synthetic_years 30"
if [[ -f dataset/dblp/edges.csv ]]; then
    DATA_ARG="--data_path dataset/dblp/edges.csv"
fi

echo "[config] ENC=$ENC  JEPA=$JEPA  OUT=$OUT  SPLIT=$SPLIT"

srun --export=ALL python src/train/analyze_jepa.py $DATA_ARG \
    --encoder_ckpt "$ENC/encoder.ckpt" \
    --jepa_ckpt "$JEPA/best.ckpt" \
    --split "$SPLIT" \
    --history_size 3 --num_preds 1 \
    --encoder_kind tgcn --encoder_hidden 128 --node_dim 128 \
    --link_probe_hidden 64 --link_epochs 30 --link_lr 1e-3 \
    --probe_epochs 200 --probe_lr 1e-2 \
    --batch_size 4 --num_workers ${SLURM_CPUS_PER_TASK:-8} \
    --device cuda \
    --out_dir "$OUT" "$@"

echo
echo "[done] cat $OUT/analysis.json for the full dump (diagnostic + link_pred)."
