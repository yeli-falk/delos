# shellcheck shell=bash
# STD-MAE window geometry. Sourced by both slurm_all_methods.sh (final table)
# and slurm_hparam_search.sh (STAGE=baselines), so the search tunes the config
# the final run uses. Set DATASET, HIST and BATCH first.
#
# STD-MAE reads a long window (the paper's METR-LA setting is 288*3 = 864
# 5-minute steps = 3 days) that its two masked autoencoders reconstruct.
# England COVID's timeline is too short for that geometry, so there the long
# window is pinned to the forecast history (4 patches of 2), the largest
# window that needs no padding.
case "$DATASET" in
    metr_la)
        SM_LONG=${SM_LONG:-864}; SM_PATCH=${SM_PATCH:-12}
        SM_BATCH=${SM_BATCH:-8}                 # the paper's; 64 OOMs a 32 GB card
        SM_CL=${SM_CL:-6}                       # curriculum: +1 horizon / 6 epochs
        SM_STRIDE=${SM_STRIDE:-12}              # pre-train on every 12th window
        ;;
    *)
        SM_LONG=${SM_LONG:-$HIST}; SM_PATCH=${SM_PATCH:-2}
        SM_BATCH=${SM_BATCH:-$BATCH}
        SM_CL=${SM_CL:-0}                       # a 3-step horizon needs no curriculum
        SM_STRIDE=${SM_STRIDE:-1}
        ;;
esac
