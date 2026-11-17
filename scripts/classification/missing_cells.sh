#!/usr/bin/env bash
# Lists the cells of a scripts/classification/slurm_hparam_search.sh grid that
# have no state_probe_val.json yet, as "array index <tab> directory". It takes
# the same env knobs as the sweep (DATASET, STAGE, VARIANT_SET, EMA_ARMS,
# EMA_FREEZE, SEED, ROOT). The directory name is rebuilt as the sweep builds
# it, so a finished cell trained under different flags is not reported.
#
#   DATASET=mooc STAGE=variants VARIANT_SET=all bash scripts/classification/missing_cells.sh
#   Submit only those: sbatch --array=$(... | cut -f1 | paste -sd,) ...
SEED="${SEED:-42}"; SHORT="${SHORT:-$DATASET}"
RS="$STAGE"; [[ "$STAGE" == "ema" ]] && RS=variants
ROOT="${ROOT:-outputs/cls_hparam/${SHORT}_${RS}}"
bash scripts/classification/slurm_hparam_search.sh --list 2>/dev/null \
| grep -E '^\s+\[ ?[0-9]+\] ' | while read -r line; do
    idx=$(echo "$line" | sed -E 's/^\s*\[ ?([0-9]+)\].*/\1/')
    spec=${line#*] }; arm=${spec%%|*}; flags=${spec#*|}
    slug=$(echo "$flags" | sed 's/--//g; s/[ .]/_/g; s/__*/_/g; s/^_//; s/_$//'); [[ -z "$slug" ]] && slug=base
    d="$ROOT/${arm}_${slug}_s${SEED}"
    [[ -f "$d/state_probe_val.json" ]] || printf "%s\t%s\n" "$idx" "$d"
done
