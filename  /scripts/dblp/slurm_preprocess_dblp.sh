#!/usr/bin/env bash
#SBATCH --job-name=dblp-preprocess
#SBATCH --output=logs/dblp-preprocess-%j.out
#SBATCH --error=logs/dblp-preprocess-%j.err
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --time=01:30:00
# No GPU requested -- this is CPU + memory only.
# If your cluster has a "cpu" or similar non-GPU partition, set it here:
# #SBATCH --partition=cpu
#
# Submit with:    sbatch scripts/dblp/slurm_preprocess_dblp.sh
# Override args:  sbatch scripts/dblp/slurm_preprocess_dblp.sh --min_year 2010

set -euo pipefail
mkdir -p logs dataset/dblp

# SLURM copies this script to a spool dir, so ${BASH_SOURCE[0]} can't locate
# the repo. SLURM_SUBMIT_DIR is the directory you ran `sbatch` from -- always
# submit from the repo root.
REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"
echo "[slurm] repo_root=$REPO_ROOT"
echo "[slurm] node=$(hostname)"
echo "[slurm] cwd=$(pwd)"
if [[ ! -f dataset/dblp/dblp.xml.gz ]]; then
    echo "[error] dataset/dblp/dblp.xml.gz not found under $REPO_ROOT" >&2
    echo "[error] submit from the repo root: cd $REPO_ROOT && sbatch scripts/dblp/slurm_preprocess_dblp.sh" >&2
fi

if [[ -d "$HOME/.conda/envs/cjepa" ]]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate cjepa
fi

# Ensure lxml is present in the env.
python -c "import lxml" 2>/dev/null || pip install --quiet lxml

if [[ ! -f dataset/dblp/dblp.xml.gz ]]; then
    echo "[fetch] downloading dblp.xml.gz ..."
    wget -q --show-progress https://dblp.org/xml/dblp.xml.gz \
        -O dataset/dblp/dblp.xml.gz
fi

# Defaults that work for a first-pass DBLP DTDG. Override on the sbatch CLI.
python scripts/dblp/preprocess_dblp_xml.py \
    --xml_path dataset/dblp/dblp.xml.gz \
    --out_csv dataset/dblp/edges.csv \
    --names_csv dataset/dblp/authors.csv \
    --min_year 2000 --max_year 2023 \
    --min_author_papers 5 \
    "$@"

echo "[done] dataset/dblp/edges.csv:"
wc -l dataset/dblp/edges.csv
echo "[done] dataset/dblp/authors.csv:"
wc -l dataset/dblp/authors.csv
