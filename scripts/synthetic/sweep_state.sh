#!/usr/bin/env bash
# Sweep ONE generator knob on the synthetic node-STATE task and see which data
# property the JEPA forecast benefits from.
#
# Every axis value gets its own results root (its own shared encoder, its own
# arms), and each (value, seed) is one slurm_all_methods.sh job -- the same
# launcher, arms and probe as the MOOC table, so the columns are comparable.
#
# Axes are the SYN_* env vars _dataset_config.sh reads for DATASET=synthetic_dyn:
#
#   SYN_ANCHOR       anchor spread sigma_a     community separation / switch rate
#   SYN_OU_SIGMA     latent noise sigma        unpredictability of the CAUSE
#   SYN_MOMENTUM     velocity persistence rho  how much history matters
#   SYN_THETA        mean reversion theta      how fast a node returns to its anchor
#   SYN_NODE_NOISE   observation noise nu      SNR of the features (latent vs input space)
#   SYN_DIFFUSION    feature smoothing alpha   how much the GRAPH shapes features
#   SYN_BETA         edge sharpness beta       how informative topology is about z
#   SYN_PERSISTENCE  edge lifetime eta         temporal structure in the graph
#   SYN_DENSITY      edge density kappa        sparsity
#   SYN_K            communities K             number of states
#   SYN_DLAT         latent dim d              intrinsic dimensionality
#   SYN_HYST         hysteresis h              label stickiness
#   SYN_NODES / SYN_STEPS                      graph size / timeline length
# and the window geometry NPRED (horizon) / HIST (history), plus STATE_MODE etc.
#
# Binary switch-event task on the same generator (AUROC/AUPRC, MOOC-comparable):
#   SYN_STATE_TARGET=switch AXIS=NPRED VALUES="1 3 6" bash scripts/synthetic/sweep_state.sh
#   SYN_STATE_TARGET=switch AXIS=NPRED VALUES="1 3 6" bash scripts/synthetic/sweep_state.sh --aggregate
#
# Submit one axis:
#   AXIS=SYN_MOMENTUM VALUES="0.0 0.5 0.8 0.95" bash scripts/synthetic/sweep_state.sh
#   AXIS=SYN_NODE_NOISE VALUES="0.0 0.25 0.5 1.0" ARMS="jepa_frozen jepa_ema cldg" bash ...
#   AXIS=NPRED VALUES="1 3 6" bash ...                       # horizon axis
# (For the horizon CURVE prefer scripts/synthetic/horizon_showcase.sh: one run
#  at NPRED=12 whose probe fits a head per horizon; this axis retrains per
#  value and is the robustness check that h=1 does not depend on the window.)
# Any other SYN_* set in the environment is held fixed for the whole sweep.
#
# Report (after the jobs finish), one table per axis value plus a one-line
# summary per value of the paired JEPA-minus-copy-last gap on switch rows:
#   AXIS=SYN_MOMENTUM VALUES="0.0 0.5 0.8 0.95" bash scripts/synthetic/sweep_state.sh --aggregate
#
# Logs: logs/st-<AXIS>-<value>-s<seed>-<jobid>.out (and .err). Job state:
#   sacct -X --starttime today --format=JobID,JobName%30,State,Elapsed,ExitCode
#
# Read the loader's "[synthetic_dyn] state task" line in each job log before
# trusting a cell: it prints the switch rate and the oracle's error reduction
# over persistence. A value where that is below ~10% cannot separate methods,
# whatever the arms score.
set -uo pipefail

AXIS=${AXIS:?set AXIS to a SYN_* knob (or NPRED / HIST)}
VALUES=${VALUES:?set VALUES=\"v1 v2 ...\"}
SEEDS=${SEEDS:-"42 43 44"}
ARMS=${ARMS:-"jepa_frozen jepa_ema cldg summary"}
# SYN_STATE_TARGET=switch runs the BINARY switch-event task (AUROC/AUPRC, the
# MOOC metric) instead of the multiclass state; it gets its own root and metrics.
if [[ "${SYN_STATE_TARGET:-community}" == "switch" ]]; then
    SWEEP_ROOT=${SWEEP_ROOT:-outputs/synth_switch/${AXIS}}
    METRICS=${METRICS:-"auroc_test auprc_test subset_onset.auroc_test"}
else
    SWEEP_ROOT=${SWEEP_ROOT:-outputs/synth_state/${AXIS}}
    METRICS=${METRICS:-"acc_test subset_switch.acc_test subset_stay.acc_test"}
fi

root_for() { echo "$SWEEP_ROOT/${AXIS}_$1"; }

if [[ "${1:-}" == "--aggregate" ]]; then
    for v in $VALUES; do
        R="$(root_for "$v")"
        echo; echo "################ $AXIS = $v   ($R)"
        python scripts/classification/aggregate_classification.py \
            --dataset synthetic_dyn --root "$R" --seeds $SEEDS --metrics $METRICS \
            --json_out "$R/summary.json" 2>&1 | grep -v '^\[skip\]'
    done
    echo; echo "=== paired gap (jepa - copy-last) on SWITCH rows, per $AXIS value ==="
    python - "$AXIS" "$SWEEP_ROOT" $VALUES <<'EOF'
import json, sys, pathlib
axis, root, *values = sys.argv[1:]
print(f"{axis:>14} {'frozen gap':>16} {'ema gap':>16} {'cldg switch':>12} {'copy-last switch':>17}")
for v in values:
    p = pathlib.Path(root) / f"{axis}_{v}" / "summary.json"
    if not p.exists():
        print(f"{v:>14}  (no summary.json)"); continue
    d = json.load(open(p))
    def g(label, key):
        e = d.get(label, {}).get(key)
        return "--" if not e else f"{e['mean']:+.3f} ± {e['std']:.3f}"
    def val(label, key):
        e = d.get(label, {}).get(key)
        return "--" if not e else f"{e['mean']:.3f}"
    # copy-last for the frozen arm = its forecast value minus its paired gap
    fz = d.get("DyG-JEPA (frozen)", {})
    cl = (fz.get("subset_switch.acc_test") or {}).get("mean")
    gp = (fz.get("gap_subset_switch.acc_test") or {}).get("mean")
    last = "--" if cl is None or gp is None else f"{cl - gp:.3f}"
    print(f"{v:>14} {g('DyG-JEPA (frozen)','gap_subset_switch.acc_test'):>16} "
          f"{g('DyG-JEPA (EMA)','gap_subset_switch.acc_test'):>16} "
          f"{val('CLDG','subset_switch.acc_test'):>12} {last:>17}")
EOF
    exit 0
fi

# Slurm does NOT create the log directory: if it is missing at submission time
# the job dies before its first line with no output anywhere. Create it here,
# and name the logs after the cell so they can be found.
mkdir -p logs
for v in $VALUES; do
    for s in $SEEDS; do
        R="$(root_for "$v")"
        mkdir -p "$R"
        LOG="logs/st-${AXIS}-${v}-s${s}-%j"
        # The axis variable is exported by name; everything else in the
        # environment (other SYN_*, HIST, STATE_MODE, ...) passes through as-is.
        echo "sbatch: $AXIS=$v seed=$s -> $R   (log: $LOG.out)"
        env "$AXIS=$v" DATASET=synthetic_dyn SEED="$s" ROOT="$R" ARMS="$ARMS" \
            sbatch --job-name="st-${AXIS}-${v}-s${s}" \
                   --output="$LOG.out" --error="$LOG.err" \
            scripts/classification/slurm_all_methods.sh
    done
done
echo "When done:  AXIS=$AXIS VALUES=\"$VALUES\" bash $0 --aggregate"
