#!/usr/bin/env bash
#SBATCH --job-name=collapse-analyze
#SBATCH --output=logs/collapse-analyze-%j.out
#SBATCH --error=logs/collapse-analyze-%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --partition=gpu
#
# REPRESENTATION-COLLAPSE STUDY: does DELOS collapse, and what prevents it?
#
# One command trains every arm, measures the latents and writes the paper's
# table and figures:
#
#   bash scripts/collapse/slurm_collapse.sh              # submit everything
#
# which submits, per seed (all through scripts/classification/slurm_all_methods.sh,
# so data, encoder, window and early stopping are the main table's):
#   1. the DELOS arms ($JEPA_ARMS) with checkpoint snapshots
#      (--snapshot_first $SNAP_FIRST --snapshot_every $SNAP_EVERY) for the
#      training-dynamics curves
#   2. the SSL baselines ($BASELINES)
#   3. a second collapse control: jepa_naive without the residual head
#      (JEPA_TAG=nores; CTRL_NORES=0 to skip)
# and then ONE analysis job (this file, `analyze` mode) that waits for all of
# them (--dependency=afterany) and runs
#   scripts/collapse/collapse_metrics.py   -> <run>/collapse.json, collapse_dynamics.json
#   scripts/collapse/collapse_report.py    -> $FIG/collapse_table.{txt,tex},
#                                             fig_{spectrum,dynamics,temporal,rank_vs_probe}.pdf
#
# What the arms are for:
#   jepa_ema_cos / jepa_ema   the model (EMA target encoder)            -> MAIN
#   jepa_frozen               encoder fixed: collapse impossible by construction
#   jepa_naive (+ _nores)     encoder trained against its OWN stop-gradient
#                             targets: a constant latent is a global optimum.
#                             The positive control -- without a row that does
#                             collapse, a rank number has no floor to read against.
#   jepa_vicreg / jepa_sigreg the alternative anti-collapse mechanisms
#   baselines                 where the SSL methods we compare against sit
# Reference rows (raw features, random-init encoder, edge-mask pretrain init)
# are computed by the analysis job, no training needed.
#
# Other modes:
#   bash  scripts/collapse/slurm_collapse.sh --list      # print the sbatch lines, submit nothing
#   sbatch scripts/collapse/slurm_collapse.sh analyze    # metrics + report on what is on disk
#   bash  scripts/collapse/slurm_collapse.sh report      # tables + figures only (no GPU, login node ok)
#
# Knobs (env):
#   DATASET     mooc (any dataset slurm_all_methods.sh accepts; SYN_* / NPRED /
#               HIST / ENC_KIND / BATCH ... pass straight through)
#   SEEDS       "42 43 44 45 46"
#   ROOT        outputs/collapse/<dataset>[_<TAG>]   FIG  figures/collapse/<dataset>[_<TAG>]
#   JEPA_ARMS   "jepa_frozen jepa_naive jepa_ema jepa_ema_cos jepa_vicreg jepa_sigreg"
#   BASELINES   "cldg ddgcl idol maskdgnn dysubc dyg2vec dygwm sgjepa bgrl"   ("" = none)
#   MAIN        stem of the reported arm (default: first of jepa_ema_cos /
#               jepa_ema / jepa_frozen with results);  COLLAPSED  control stems
#               (default: every jepa_naive*)
#   SNAP_FIRST 10, SNAP_EVERY 10   snapshot schedule (each snapshot is one
#               encoder + predictor checkpoint; PRUNE_SNAPSHOTS=1 deletes them
#               once collapse_dynamics.json is written)
#   BASE_HP, VARIANT_HP_<arm>, BASELINE_HP_<m>   tuned flags, as in the main table.
#               Pass the SAME ones the main table used or the rows differ from it.
#   PARTITION / GRES / CONSTRAINT   sbatch overrides (margaret: PARTITION=gpu-best;
#               GRES=gpu:<type>:1 picks ONE card type, CONSTRAINT="typeA|typeB" a set,
#               if the nodes carry those features)
#   SPLIT       test (val to keep the test split unread)
#
# Reusing an existing table root: finished arms are skipped, so point ROOT at
# it and only the missing pieces train. Arms trained without the snapshot flags
# have no dynamics curve; to get one without touching them, retrain into tagged
# directories next to them:
#   ROOT=outputs/cls_mooc JEPA_TAG=dyn BASELINES="" bash scripts/collapse/slurm_collapse.sh
#   (then MAIN=jepa_ema_cos_dyn COLLAPSED="jepa_naive_dyn jepa_naive_dyn_nores")
# Synthetic showcase root with its own main arm:
#   DATASET=synthetic_dyn ROOT=outputs/synth_horizon/NPRED_12_lin_rho095_root_T1000 \
#       MAIN=jepa_frozen_small sbatch scripts/collapse/slurm_collapse.sh analyze

set -uo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"
mkdir -p logs

MODE="${1:-submit}"
DATASET="${DATASET:-mooc}"
SEEDS="${SEEDS:-42 43 44 45 46}"
TAG="${TAG:-}"
ROOT="${ROOT:-outputs/collapse/${DATASET}${TAG:+_$TAG}}"
FIG="${FIG:-figures/collapse/${DATASET}${TAG:+_$TAG}}"
JEPA_ARMS="${JEPA_ARMS-jepa_frozen jepa_naive jepa_ema jepa_ema_cos jepa_vicreg jepa_sigreg}"
BASELINES="${BASELINES-cldg ddgcl idol maskdgnn dysubc dyg2vec dygwm sgjepa bgrl}"
CTRL_NORES="${CTRL_NORES:-1}"
SNAP_FIRST="${SNAP_FIRST:-10}"
SNAP_EVERY="${SNAP_EVERY:-10}"
PRUNE_SNAPSHOTS="${PRUNE_SNAPSHOTS:-0}"
SPLIT="${SPLIT:-test}"
MAIN="${MAIN:-}"
COLLAPSED="${COLLAPSED:-}"
PARTITION="${PARTITION:-}"
GRES="${GRES:-}"
# CONSTRAINT: sbatch --constraint, for node FEATURES (which accept alternation,
# e.g. CONSTRAINT="hopper|ampere"); --gres takes exactly one type. Check what the
# cluster defines with:  sinfo -p gpu-best -o "%N %G %f"
CONSTRAINT="${CONSTRAINT:-}"
JEPA_TAG="${JEPA_TAG:-}"

# --- report: tables + figures from the collapse*.json already on disk ---
report() {   # $1 = python launcher
    $1 scripts/collapse/collapse_report.py --root "$ROOT" --out "$FIG" --split "$SPLIT" \
        --seeds $SEEDS ${MAIN:+--main "$MAIN"} ${COLLAPSED:+--collapsed $COLLAPSED} --latex
}

if [[ "$MODE" == "report" ]]; then
    report "${PY:-python}"
    exit $?
fi

# --- analyze: metrics for every run under $ROOT, then the report ---
if [[ "$MODE" == "analyze" ]]; then
    # Interpreter: same resolution as slurm_all_methods.sh.
    CONDA_ENV="${CONDA_ENV:-}"
    if [[ -z "${PY:-}" ]]; then
        [[ -z "$CONDA_ENV" && -d "$HOME/.conda/envs/cjepa" ]] && CONDA_ENV=cjepa
        if [[ -n "$CONDA_ENV" ]]; then
            CONDA_BASE="$(conda info --base 2>/dev/null || true)"
            [[ -n "$CONDA_BASE" ]] && source "$CONDA_BASE/etc/profile.d/conda.sh"
            conda activate "$CONDA_ENV" || { echo "[FATAL] conda activate $CONDA_ENV failed" >&2; exit 4; }
        fi
        PY=python
    fi
    command -v "$PY" >/dev/null 2>&1 \
        || { echo "[FATAL] interpreter '$PY' not found. Set PY=/path/to/python or CONDA_ENV=<name|path>" >&2; exit 4; }
    if command -v srun >/dev/null 2>&1 && [[ -n "${SLURM_JOB_ID:-}" ]]; then
        RUN="srun --export=ALL $PY"
    else
        RUN="$PY"
    fi
    echo "[env] python=$(command -v "$PY")"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
    echo "[config] ROOT=$ROOT FIG=$FIG SEEDS='$SEEDS' SPLIT=$SPLIT MAIN='${MAIN:-auto}' COLLAPSED='${COLLAPSED:-auto}'"
    [[ -d "$ROOT" ]] || { echo "[FATAL] $ROOT does not exist" >&2; exit 2; }

    echo; echo "=== collapse metrics -> <run>/collapse.json ==="
    # A failed run is reported at the end with exit 1; the report still covers the rest.
    $RUN scripts/collapse/collapse_metrics.py --root "$ROOT" --seeds $SEEDS --split "$SPLIT" \
        --device cuda --batch_size "${PROBE_BATCH:-4}" --num_workers "${SLURM_CPUS_PER_TASK:-4}" \
        ${FORCE:+--force}
    METRICS_RC=$?

    if [[ "$PRUNE_SNAPSHOTS" == "1" ]]; then
        SUF=""; [[ "$SPLIT" == "val" ]] && SUF="_val"
        for d in "$ROOT"/*_s*/; do
            if [[ -f "$d/collapse_dynamics${SUF}.json" && -d "$d/snapshots" ]]; then
                echo "[prune] $d/snapshots"; rm -rf "$d/snapshots"
            fi
        done
    fi

    echo; echo "=== report -> $FIG ==="
    report "$RUN" || exit $?
    [[ "$METRICS_RC" == "0" ]] || echo "[warn] collapse_metrics.py failed on some runs (see above); the report covers the rest"
    exit "$METRICS_RC"
fi

# --- submit: training jobs per seed + the dependent analysis job ---
DRY=0
case "$MODE" in
    submit) ;;
    --list) DRY=1 ;;
    *) echo "usage: $0 [submit|--list|analyze|report]" >&2; exit 2 ;;
esac
command -v sbatch >/dev/null 2>&1 || [[ "$DRY" == "1" ]] \
    || { echo "[FATAL] sbatch not found; use '--list', or 'analyze' / 'report' on existing runs" >&2; exit 3; }

SB_OPTS=(${PARTITION:+--partition="$PARTITION"} ${GRES:+--gres="$GRES"} ${CONSTRAINT:+--constraint="$CONSTRAINT"})
SNAP_HP="--snapshot_first $SNAP_FIRST --snapshot_every $SNAP_EVERY"
JOBS=()

submit() {   # $1 = job name, $2.. = VAR=value env for slurm_all_methods.sh
    local name="$1"; shift
    if [[ "$DRY" == "1" ]]; then
        echo "env $* sbatch ${SB_OPTS[*]} --job-name=$name scripts/classification/slurm_all_methods.sh"
        return
    fi
    local id
    id=$(env "$@" sbatch --parsable ${SB_OPTS[@]+"${SB_OPTS[@]}"} --job-name="$name" \
            --output="logs/${name}-%j.out" --error="logs/${name}-%j.err" \
            scripts/classification/slurm_all_methods.sh) || { echo "[FATAL] sbatch failed for $name" >&2; exit 5; }
    id="${id%%;*}"
    JOBS+=("$id")
    echo "submitted $name -> job $id"
}

# DOWNSTREAM=none / COLLAPSE_CHECK=0: the trainer's link probe and the link-MRR
# collapse check are not part of this study (the state probe on best.ckpt and
# collapse_metrics.py are); the snapshots ride on BASE_HP, which only the
# DELOS arms read.
COMMON=(DATASET="$DATASET" ROOT="$ROOT" DOWNSTREAM="${DOWNSTREAM:-none}" COLLAPSE_CHECK=0)
for S in $SEEDS; do
    [[ -n "$JEPA_ARMS" ]] && submit "col-jepa-s$S" "${COMMON[@]}" SEED="$S" ARMS="$JEPA_ARMS" \
        BASE_HP="${BASE_HP:-} $SNAP_HP" JEPA_TAG="$JEPA_TAG"
    [[ "$CTRL_NORES" == "1" ]] && submit "col-nores-s$S" "${COMMON[@]}" SEED="$S" ARMS="jepa_naive" \
        BASE_HP="${BASE_HP:-} $SNAP_HP" RESIDUAL=0 JEPA_TAG="${JEPA_TAG:+${JEPA_TAG}_}nores"
    [[ -n "$BASELINES" ]] && submit "col-base-s$S" "${COMMON[@]}" SEED="$S" ARMS="$BASELINES"
done

DEP=""
(( ${#JOBS[@]} > 0 )) && DEP="--dependency=afterany:$(IFS=:; echo "${JOBS[*]}")"
export DATASET SEEDS ROOT FIG SPLIT MAIN COLLAPSED PRUNE_SNAPSHOTS
if [[ "$DRY" == "1" ]]; then
    echo "sbatch ${SB_OPTS[*]} --dependency=afterany:<the jobs above> scripts/collapse/slurm_collapse.sh analyze"
    exit 0
fi
AID=$(sbatch --parsable ${SB_OPTS[@]+"${SB_OPTS[@]}"} $DEP scripts/collapse/slurm_collapse.sh analyze) \
    || { echo "[FATAL] sbatch failed for the analysis job" >&2; exit 5; }
echo "submitted analysis -> job ${AID%%;*} (runs after: ${JOBS[*]:-nothing})"
echo "Outputs: $FIG/collapse_table.{txt,tex}, fig_*.pdf   (log: logs/collapse-analyze-${AID%%;*}.out)"
echo "Re-run by hand:  ROOT=$ROOT FIG=$FIG sbatch ${SB_OPTS[*]} scripts/collapse/slurm_collapse.sh analyze"
