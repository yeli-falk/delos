#!/usr/bin/env bash
# DDGCL hyperparameter sweep. Selects on the DOWNSTREAM probe metric (val MRR),
# NOT loss_cpc_debais (whose scale changes with tau/beta and is incomparable
# across configs). Trains each config, probes it on the VAL split, records MRR,
# then re-probes the winner on TEST.
#
#   bash scripts/ddgcl/sweep_ddgcl.sh
#
# IMPORTANT: DATA/FEATURE flags must be IDENTICAL for train and probe, or the
# encoder input dim / snapshot binning won't match the checkpoint.
set -euo pipefail

PY=${PY:-python}
DATA_FLAGS="--data_source jodie_mooc --with_edge_features"   # <- shared by BOTH steps
COMMON="--encoder_kind gcn --epochs 100 --device cuda"
ROOT="outputs/ddgcl_sweep"
RESULTS="$ROOT/results.csv"
mkdir -p "$ROOT"
echo "tau,beta,lr,node_dim,max_gap,momentum,queue_size,val_mrr,val_auc,ckpt" > "$RESULTS"

# --- grid (edit to taste) -------------------------------------------------
TAUS=(0.01 0.05 0.1)
BETAS=(0.0 1.5)
LRS=(1e-3 4e-3)
NODE_DIMS=(64 128)
MAX_GAPS=(2 4)
MOMENTUMS=(0.999)         # add 0 to also try in-batch negatives
QUEUES=(512)              # paired with momentum; 0 = in-batch

parse_metric() {          # $1 = probe log file, $2 = "MRR" or "AUC"
  grep -Eo "all  \+ 1:.*" "$1" | grep -Eo "$2=[0-9.]+" | head -1 | cut -d= -f2
}

for tau in "${TAUS[@]}"; do
 for beta in "${BETAS[@]}"; do
  for lr in "${LRS[@]}"; do
   for nd in "${NODE_DIMS[@]}"; do
    for mg in "${MAX_GAPS[@]}"; do
     for mom in "${MOMENTUMS[@]}"; do
      for q in "${QUEUES[@]}"; do
        tag="tau${tau}_b${beta}_lr${lr}_d${nd}_g${mg}_m${mom}_q${q}"
        odir="$ROOT/$tag"
        echo "=== $tag ==="
        $PY -m src.baselines.ddgcl.train_ddgcl $DATA_FLAGS $COMMON \
            --encoder_hidden "$nd" --node_dim "$nd" \
            --tau "$tau" --beta "$beta" --lr "$lr" --max_gap "$mg" \
            --momentum "$mom" --queue_size "$q" \
            --out_dir "$odir" > "$odir.train.log" 2>&1 || { echo "  train FAILED"; continue; }

        ckpt=$(find "$odir" -name '*encoder.ckpt' 2>/dev/null | head -1)
        [ -z "$ckpt" ] && { echo "  no ckpt"; continue; }

        $PY -m src.train.analyze_jepa --skip_diagnostic --link_sources encoder \
            $DATA_FLAGS --split val --encoder_ckpt "$ckpt" \
            > "$odir.probe.log" 2>&1 || { echo "  probe FAILED"; continue; }

        mrr=$(parse_metric "$odir.probe.log" MRR)
        auc=$(parse_metric "$odir.probe.log" AUC)
        echo "  val MRR=$mrr AUC=$auc"
        echo "$tau,$beta,$lr,$nd,$mg,$mom,$q,$mrr,$auc,$ckpt" >> "$RESULTS"
      done
     done
    done
   done
  done
 done
done

echo; echo "=== leaderboard (by val MRR) ==="
sort -t, -k8 -gr "$RESULTS" | head -10 | column -s, -t
best_ckpt=$(sort -t, -k8 -gr "$RESULTS" | head -1 | cut -d, -f10)
echo; echo "=== BEST on TEST: $best_ckpt ==="
$PY -m src.train.analyze_jepa --skip_diagnostic --link_sources encoder \
    $DATA_FLAGS --split test --encoder_ckpt "$best_ckpt"
