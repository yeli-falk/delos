#!/usr/bin/env bash
# DELOS on METR-LA. Run from the repo root.
#
# Expects dataset/metr_la/{metr-la.h5,adj_mx.pkl} (see download_metr_la.sh).
# To smoke-test the pipeline without downloading anything, append --synthetic.

set -e
export PYTHONPATH=$(pwd)
# Synchronous CUDA launches give accurate stack traces if the GPU faults.
# Drop this once the run is stable; it costs ~10-20% throughput.
export CUDA_LAUNCH_BLOCKING=1

python src/train/train_delos.py --data_source metr_la --downstream_task regression \
    --metr_la_data_dir dataset/metr_la \
    --history_size 12 \
    --num_preds 12 \
    --node_dim 64 \
    --encoder_kind gcn \
    --num_masked_nodes 30 \
    --depth 4 \
    --heads 4 \
    --mlp_dim 256 \
    --batch_size 8 \
    --device cuda \
    --no_attention_mask \
    --epochs 20 \
    --lr 5e-4 \
    --out_dir outputs/metr_la_cjepa \
    "$@"
