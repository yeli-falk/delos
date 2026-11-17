#!/usr/bin/env bash
# Full England COVID comparison: parameter-free baselines + DELOS
# (random-init encoder and pretrained encoder). Run from the repo root.
#
#   bash scripts/england_covid/run_all.sh            # uses cuda
#   bash scripts/england_covid/run_all.sh --device cpu
#
# All runs share history_size=8 / num_preds=3 so the test windows (and hence the
# MAE/RMSE/MAPE numbers) are directly comparable across the table.

set -e
export PYTHONPATH=$(pwd)

HS=8
NP=3
EXTRA="$@"   # e.g. --device cpu

echo "================ 1/4  parameter-free baselines ================"
python scripts/regression/run_baselines.py \
    --history_size $HS --num_preds $NP \
    --out_dir outputs/england_covid_baselines

echo "================ 2/4  DELOS (random-init encoder) ================"
python src/train/train_delos.py --data_source england_covid \
    --downstream_task regression --mask_scheme node_mask \
    --history_size $HS --num_preds $NP \
    --encoder_kind tgcn --bidirectional false --node_dim 64 \
    --num_masked_nodes 20 --depth 4 --heads 4 --mlp_dim 256 \
    --batch_size 8 --epochs 50 --lr 5e-4 \
    --probe_epochs 10 --probe_hidden 64 \
    --out_dir outputs/england_covid_cjepa_randenc $EXTRA

echo "================ 3/4  pretrain encoder (masked_signal) ================"
python src/train/pretrain_encoder.py --data_source england_covid \
    --pretext masked_signal --encoder_kind tgcn --bidirectional false \
    --history_size $HS --num_preds $NP \
    --node_dim 64 --encoder_hidden 64 --mask_rate 0.25 \
    --batch_size 8 --epochs 300 --lr 1e-3 --early_stop_patience 30 \
    --out_dir outputs/england_covid_encoder $EXTRA

echo "================ 4/4  DELOS (pretrained encoder) ================"
python src/train/train_delos.py --data_source england_covid \
    --downstream_task regression --mask_scheme node_mask \
    --history_size $HS --num_preds $NP \
    --encoder_kind tgcn --bidirectional false --node_dim 64 \
    --encoder_ckpt outputs/england_covid_encoder/encoder_england_covid_tgcn_masked_signal.ckpt \
    --num_masked_nodes 20 --depth 4 --heads 4 --mlp_dim 256 \
    --batch_size 8 --epochs 50 --lr 5e-4 \
    --probe_epochs 10 --probe_hidden 64 \
    --out_dir outputs/england_covid_cjepa_pretrained $EXTRA

echo "================ done ================"
echo "Baselines : outputs/england_covid_baselines/baseline_metrics.json"
echo "DELOS  : outputs/england_covid_cjepa_{randenc,pretrained}/test_metrics.json"
