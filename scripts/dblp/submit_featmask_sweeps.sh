#!/usr/bin/env bash
#
# Submit both verification sweeps for the Reddit-Body + feature-mask
# positive result:
#
#   1) Multi-seed at TOPK=1000  — 5 seeds, dispels single-seed-fluke doubts
#   2) K-sweep at SEED=42       — does the win scale with subset size?
#
# Total: 5 + 4 = 9 jobs. ~10 GPU-hours assuming ~1h per job. Each goes
# to its own outputs/reddit_body_featmask_k${TOPK}_s${SEED}_{...} dir, so
# nothing overwrites the original cell-3 result.
#
# Run from the repo root on the cluster:
#     bash scripts/dblp/submit_featmask_sweeps.sh
#
# To check progress later:
#     squeue -u $USER --name=cjepa-featmask
#     python scripts/dblp/dump_runs.py --pattern 'outputs/reddit_body_featmask_*_cjepa'

set -euo pipefail

if [[ ! -f scripts/dblp/slurm_featmask_reddit.sh ]]; then
    echo "[error] run from the repo root (slurm_featmask_reddit.sh not found)" >&2
    exit 1
fi

echo "=== Multi-seed @ TOPK=1000 (5 jobs) ==="
for S in 42 43 44 45 46; do
    SEED=$S TOPK=1000 sbatch scripts/dblp/slurm_featmask_reddit.sh
done

echo
echo "=== K-sweep @ SEED=42 (4 jobs, skipping K=1000 which is in the multi-seed) ==="
for K in 500 2000 3000 5000; do
    SEED=42 TOPK=$K sbatch scripts/dblp/slurm_featmask_reddit.sh
done

echo
echo "=== Reddit-Title control (1 job): does feature_mask help the weaker variant too? ==="
SEED=42 TOPK=1000 VARIANT=title sbatch scripts/dblp/slurm_featmask_reddit.sh

echo
echo "[done] 10 jobs submitted."
echo "       squeue -u \$USER --name=cjepa-featmask"
echo "       # then once complete:"
echo "       python scripts/dblp/dump_runs.py --pattern 'outputs/reddit_*_featmask_*_cjepa'"
echo "       python scripts/dblp/dump_runs.py --pattern 'outputs/reddit_body_featmask_*_cjepa' --aggregate"
