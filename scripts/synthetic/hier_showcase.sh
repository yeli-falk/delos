#!/usr/bin/env bash
# TWO-LEVEL DELOS on the synthetic horizon showcase: does a frozen COARSE-time
# DELOS, fed top-down into the fine predictor's future query tokens, improve
# the long-horizon forecast? (src/world_models/hier.py)
#
# Per seed, as a dependency chain of scripts/classification/slurm_all_methods.sh
# jobs in ONE root (shared encoder pretrained once, by the flat job):
#
#   jepa_ema                    flat control, HIST fine bins (matched context)
#   jepa_ema_c<W><mode>         COARSE level: one step per W fine bins,
#                               HIST/W + NPRED/W coarse steps; mode = last
#                               (keep the last snapshot of each block) or mean
#                               (average features, union of edges). Not probed.
#   jepa_ema_h<W><mode>_fc      fine DELOS + the coarse FORECAST as hint (the method)
#   jepa_ema_h<W><mode>_true    DIAGNOSTIC ceiling: hint = coarse encoding of the
#                               TRUE future (reads future snapshots)
#   jepa_ema_h<W><mode>_shuf    DIAGNOSTIC capacity control: the forecast hint
#                               with nodes permuted
#
# All fine arms use the main-table predictor (node_mask; the hint is switched
# off for hidden nodes). A hint arm starts from exactly the flat run of the same
# seed (zero-initialised projection, same init, same RNG stream), so the
# per-seed difference to `jepa_ema` is a paired comparison.
#
# Usage (from the repo root); generator knobs are the showcase's SYN_* / ENC_KIND
# / TAG and must be the SAME for submit and aggregate:
#   SYN_MOMENTUM=0.95 SYN_THETA=0.05 SYN_READOUT=linear SYN_DIFFUSION=0 \
#     ENC_KIND=gcn_root SYN_STEPS=1000 TAG=lin_rho095_root_T1000 \
#     bash scripts/synthetic/hier_showcase.sh
#   TAG=lin_rho095_root_T1000 bash scripts/synthetic/hier_showcase.sh --aggregate
#
# Knobs (env): NPRED (12), HIST (9), W (3), MODES ("last mean"),
# HINTS ("fc true shuf"), SEEDS ("42 43 44"), ROOT, TAG, PARTITION, GRES,
# HORIZONS ("1 3 6 9 12"), plus everything slurm_all_methods.sh reads.
# With 21-bin windows the default SYN_STEPS=300 leaves ~10 validation windows;
# prefer SYN_STEPS=1000.
#
# Check the coarse level BEFORE reading the fine arms (a collapsed or
# copy-last coarse model makes the hint uninformative, which says nothing
# about the hierarchy):
#   python scripts/synthetic/predictor_latent_error.py --pattern "$ROOT/jepa_ema_c*_s*/best.ckpt"
set -uo pipefail

NPRED=${NPRED:-12}
HIST=${HIST:-9}
W=${W:-3}
MODES=${MODES:-"last mean"}
HINTS=${HINTS:-"fc true shuf"}
SEEDS=${SEEDS:-"42 43 44"}
TAG=${TAG:-}
ROOT=${ROOT:-outputs/synth_hier/NPRED_${NPRED}_HIST_${HIST}${TAG:+_$TAG}}
HORIZONS=${HORIZONS:-"1 3 6 9 12"}
PARTITION=${PARTITION:-}
GRES=${GRES:-}
export DOWNSTREAM=${DOWNSTREAM:-none}
export COLLAPSE_CHECK=${COLLAPSE_CHECK:-0}

(( HIST % W == 0 && NPRED % W == 0 )) \
    || { echo "HIST=$HIST and NPRED=$NPRED must both be divisible by W=$W" >&2; exit 1; }

hint_flag() {
    case "$1" in
        fc) echo forecast ;; true) echo true ;; shuf) echo shuffle ;;
        *) echo "unknown hint '$1' (fc | true | shuf)" >&2; exit 1 ;;
    esac
}

TAGS=""
for m in $MODES; do for h in $HINTS; do TAGS="$TAGS h${W}${m}_${h}"; done; done

if [[ "${1:-}" == "--aggregate" ]]; then
    SW=""; OV=""
    for h in $HORIZONS; do
        k=$((h - 1))
        (( k < NPRED )) || continue
        SW="$SW subset_h${k}.switch_acc_test"; OV="$OV subset_h${k}.acc_test"
    done
    echo; echo "################ SWITCH-row accuracy per horizon (steps ahead: $HORIZONS)   ($ROOT)"
    python scripts/classification/aggregate_classification.py \
        --dataset synthetic_dyn --root "$ROOT" --seeds $SEEDS --metrics $SW \
        --jepa_tag $TAGS 2>&1 | grep -v '^\[skip\]'
    echo; echo "################ OVERALL accuracy per horizon"
    python scripts/classification/aggregate_classification.py \
        --dataset synthetic_dyn --root "$ROOT" --seeds $SEEDS --metrics $OV \
        --jepa_tag $TAGS 2>&1 | grep -v '^\[skip\]'
    echo; echo "################ PAIRED difference to the flat run of the same seed"
    python scripts/synthetic/hier_paired.py --root "$ROOT" --seeds $SEEDS \
        --tags $TAGS --horizons $HORIZONS
    exit 0
fi

mkdir -p logs "$ROOT"
submit() {   # submit <job-name> <dependency or ""> VAR=VALUE...
    local name=$1 dep=$2; shift 2
    env "$@" DATASET=synthetic_dyn ROOT="$ROOT" ARMS=jepa_ema \
        sbatch --parsable ${PARTITION:+--partition="$PARTITION"} ${GRES:+--gres="$GRES"} \
               ${dep:+--dependency="$dep"} --job-name="$name" \
               --output="logs/$name-%j.out" --error="logs/$name-%j.err" \
        scripts/classification/slurm_all_methods.sh
}

for s in $SEEDS; do
    # Flat control first: it also pretrains the root's shared encoder, so the
    # other jobs of this seed wait for it.
    j0=$(submit "hr-flat-s$s" "" SEED="$s" HIST="$HIST" NPRED="$NPRED") || exit 1
    j0=${j0%%;*}
    echo "seed $s: flat=$j0"
    for m in $MODES; do
        ctag="c${W}${m}"
        jc=$(submit "hr-$ctag-s$s" "afterany:$j0" SEED="$s" \
                HIST=$((HIST / W)) NPRED=$((NPRED / W)) JEPA_TAG="$ctag" STATE_PROBE=0 \
                JEPA_EXTRA="--coarse_window $W --coarse_mode $m") || exit 1
        jc=${jc%%;*}
        echo "  coarse $ctag=$jc"
        for h in $HINTS; do
            hf=$(hint_flag "$h") || exit 1
            jh=$(submit "hr-h${W}${m}_${h}-s$s" "afterok:$jc" SEED="$s" \
                    HIST="$HIST" NPRED="$NPRED" JEPA_TAG="h${W}${m}_${h}" \
                    JEPA_EXTRA="--hier_ckpt $ROOT/jepa_ema_${ctag}_s${s}/best.ckpt --hier_hint $hf") || exit 1
            echo "    hint h${W}${m}_${h}=${jh%%;*}"
        done
    done
done
echo "Root: $ROOT"
echo "When done:  TAG=$TAG bash $0 --aggregate"
