#!/usr/bin/env bash
# DELOS on England COVID (dynamic-topology node regression). Run from repo root.
#
# England COVID is a time-varying mobility graph (129 NUTS3 regions, a different
# edge set every day) with a per-node daily case-count signal. Unlike METR-LA
# (fixed adjacency), this uses the dynamic per-snapshot adjacency path.
#
# The england_covid.json is auto-downloaded + cached on first use.
# For a pretrained encoder, run pretrain_encoder.sh first and pass:
#   --encoder_ckpt outputs/england_covid_encoder/encoder_england_covid_tgcn_masked_signal.ckpt

set -e
export PYTHONPATH=$(pwd)

python src/train/train_delos.py --data_source england_covid \
    --downstream_task regression \
    --mask_scheme node_mask \
    --history_size 8 \
    --num_preds 3 \
    --node_dim 64 \
    --encoder_kind tgcn \
    --bidirectional false \
    --num_masked_nodes 20 \
    --depth 4 \
    --heads 4 \
    --mlp_dim 256 \
    --batch_size 8 \
    --epochs 50 \
    --lr 5e-4 \
    --probe_epochs 10 \
    --probe_hidden 64 \
    --device cuda \
    --out_dir outputs/england_covid_cjepa \
    "$@"
