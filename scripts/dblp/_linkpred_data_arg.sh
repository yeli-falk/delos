# shellcheck shell=bash
# Thin wrapper around scripts/dblp/_dataset_config.sh with TASK=link_pred.
#
# Sourced by the link-prediction baseline launchers (slurm_cldg.sh,
# slurm_idol.sh, ...) after they set DATASET. Sets DATA_ARG, WINDOW_ARG,
# ENCODER_ARG, EARLY_STOP_ARG (plus VARIANT for reddit / jodie).
TASK=link_pred
source "${REPO_ROOT:-$PWD}/scripts/dblp/_dataset_config.sh"
