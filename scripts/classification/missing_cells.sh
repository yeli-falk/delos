#!/usr/bin/env bash
# Which cells of a scripts/classification/slurm_hparam_search.sh grid are NOT
# finished yet (no state_probe_val.json), as array indices + dirs. Same env
# knobs as the sweep (DATASET, STAGE, VARIANT_SET, EMA_ARMS, EMA_FREEZE, SEED,
# ROOT); the dir name is rebuilt exactly as the sweep builds it, so a cell
# trained under different flags (e.g. ema_cos before --ema_anneal_epochs) is
# NOT reported -- delete such dirs first if they must be re-run.
#
#   DATASET=mooc STAGE=variants VARIANT_SET=all bash scripts/classification/missing_cells.sh
#   -> "48  outputs/cls_hparam/mooc_variants/ema_tnorm_encoder_lr_1e-4_ema_momentum_0_996_s42"
#   Submit only those: sbatch --array=$(... | cut -f1 | paste -sd,) ...
# prints the array indices whose cell has no state_probe_val.json yet
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
