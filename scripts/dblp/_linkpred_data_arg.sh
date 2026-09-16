# shellcheck shell=bash
# Compatibility shim: the link-prediction baselines (CLDG / IDOL / DDGCL) used
# to build DATA_ARG here. Configuration now lives in one place —
# scripts/dblp/_dataset_config.sh — so this just forwards with TASK=link_pred.
#
# Sourced by slurm_{cldg,idol,ddgcl}.sh after they set DATASET. Sets DATA_ARG,
# WINDOW_ARG, ENCODER_ARG, EARLY_STOP_ARG (plus VARIANT for reddit / jodie).
TASK=link_pred
source "${REPO_ROOT:-$PWD}/scripts/dblp/_dataset_config.sh"
