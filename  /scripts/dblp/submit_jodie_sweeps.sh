#!/usr/bin/env bash
#
# Submit the JODIE follow-up sweeps for the state-label result (full datasets):
#   B) Multi-seed JODIE-Reddit feature_mask (jepa-vs-encoder gap stability)
#   C) JODIE-Wikipedia feature_mask (cross-dataset replication)
#
# Total: 5 + 1 = 6 jobs.
#
# Submit:
#     bash scripts/dblp/submit_jodie_sweeps.sh
#
# Monitor:
#     squeue -u $USER --name=cjepa-featmask-jodie
# Summarise once complete:
#     python scripts/dblp/dump_runs.py --pattern 'outputs/jodie_*_featmask_*_cjepa'

set -euo pipefail

if [[ ! -f scripts/dblp/slurm_featmask_jodie.sh ]]; then
    echo "[error] run from repo root (slurm_featmask_jodie.sh not found)" >&2
    exit 1
fi

echo "=== B) JODIE-Reddit multi-seed (5 jobs) ==="
for S in 42 43 44 45 46; do
    SEED=$S VARIANT=reddit \
        sbatch scripts/dblp/slurm_featmask_jodie.sh
done

echo
echo "=== C) JODIE-Wikipedia (1 job, seed 42) ==="
SEED=42 VARIANT=wikipedia \
    sbatch scripts/dblp/slurm_featmask_jodie.sh

echo
echo "[done] 6 jobs submitted."
echo "       squeue -u \$USER --name=cjepa-featmask-jodie"
echo "       python scripts/dblp/dump_runs.py --pattern 'outputs/jodie_*_featmask_*_cjepa'"
echo "       python scripts/dblp/dump_runs.py --pattern 'outputs/jodie_reddit_featmask_s*_cjepa' --aggregate"
