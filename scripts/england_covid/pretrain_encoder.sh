#!/usr/bin/env bash
# Pretrain a DELOS encoder on England COVID (self-supervised masked-signal).
# Run from the repo root. The england_covid.json is auto-downloaded + cached
# under dataset/england_covid/ on first use.

set -e
export PYTHONPATH=$(pwd)

python src/train/pretrain_encoder.py --data_source england_covid \
    --pretext masked_signal \
    --encoder_kind tgcn \
    --bidirectional false \
    --history_size 8 \
    --num_preds 3 \
    --node_dim 64 \
    --encoder_hidden 64 \
    --mask_rate 0.25 \
    --batch_size 8 \
    --epochs 300 \
    --lr 1e-3 \
    --early_stop_patience 30 \
    --device cuda \
    --out_dir outputs/england_covid_encoder \
    --plot_losses \
    "$@"

# Produces outputs/england_covid_encoder/encoder_england_covid_tgcn_masked_signal.ckpt
# Pass that to train_delos.sh via --encoder_ckpt for a pretrained run.
