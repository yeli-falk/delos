#!/usr/bin/env bash
#SBATCH --job-name=delos-nodeclf-jodie
#SBATCH --output=logs/delos-nodeclf-jodie-%j.out
#SBATCH --error=logs/delos-nodeclf-jodie-%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --time=20:00:00
#SBATCH --partition=gpu
#
# Node classification (the JODIE `state_label` task, e.g. Reddit ban
# prediction) for DELOS and the CLDG / DDGCL / IDOL baselines, on a matched
# encoder architecture.
#
#   methods : delos, cldg, ddgcl, idol   (all --encoder_kind gcn)
#   task    : per-user binary state_label. The positive rate is low, so the
#             headline metric is AUROC, not accuracy.
#   probe   : frozen encoder latents -> linear probe. DELOS additionally
#             reports its JEPA-predictor latent column. A degree-only probe is
#             fit alongside as the trivial baseline.
#
# Each stage is skipped if its checkpoint already exists, so the job is
# resumable: resubmit if it hits the walltime.
#
# Submit:            sbatch scripts/dblp/slurm_state_probe_jodie_all.sh
# Wikipedia instead: DATASET=wikipedia sbatch scripts/dblp/slurm_state_probe_jodie_all.sh
# MOOC (dropout):    DATASET=mooc sbatch scripts/dblp/slurm_state_probe_jodie_all.sh
# Event (sparse):    STATE_MODE=event sbatch scripts/dblp/slurm_state_probe_jodie_all.sh
# Predictor fixes (use a distinct TAG so a fresh predictor is trained):
#   RESIDUAL=1 CHANGE_W=2.0 TAG=jodie_mooc_gcn_full_res_cw2_s42 ... sbatch ...

set -uo pipefail
mkdir -p logs outputs dataset/jodie

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"

if [[ -d "$HOME/.conda/envs/cjepa" ]]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate cjepa
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo "[slurm] repo_root=$REPO_ROOT  node=$(hostname)"

# --- config -----------------------------------------------------------------
DATASET=${DATASET:-reddit}          # reddit | wikipedia | mooc
# Binning, edge features, encoder, window geometry, seed and early stopping
# all come from the shared config. The dataset is used at full size and the
# snapshot adjacency is dense (T,N,N): lower BATCH via env if you OOM.
TASK=state_probe
BATCH=${BATCH:-8}
# The epoch cap is deliberately high: pretrain_encoder restores best-val
# weights, so PATIENCE decides when to stop.
ENC_EPOCHS=${ENC_EPOCHS:-300}
JEPA_EPOCHS=${JEPA_EPOCHS:-50}
JEPA_PROBE_EPOCHS=${JEPA_PROBE_EPOCHS:-3}
BASE_EPOCHS=${BASE_EPOCHS:-100}
NW=${SLURM_CPUS_PER_TASK:-8}
# Label semantics: 'ever' (static), 'state' (dynamic cumulative, the default),
# 'event' (dynamic instantaneous, very sparse).
STATE_MODE=${STATE_MODE:-state}
# RESIDUAL=1 turns on the residual head (copy-last as the training floor);
# CHANGE_W>0 weights the latent MSE toward transition-carrying nodes.
RESIDUAL=${RESIDUAL:-0}
CHANGE_W=${CHANGE_W:-0}
# MASK_SCHEME=future_forecast drops the masked-history objective (pure
# forecaster).
MASK_SCHEME=${MASK_SCHEME:-node_mask}
# TRAIN_ENC=1 unfreezes the encoder during JEPA training with an EMA target.
TRAIN_ENC=${TRAIN_ENC:-0}
# BASELINES=0 skips the CLDG/DDGCL/IDOL stages (for DELOS-only ablations).
BASELINES=${BASELINES:-1}
# Per-method hyperparameter overrides, appended verbatim to that method's
# trainer. Paste the winning strings printed by aggregate_mooc_hparam.py here:
#   JEPA_HP="--lr 1e-3 --change_weight 2.0 --num_masked_nodes 50" \
#   CLDG_HP="--tau 0.5 --snapshots 8 --lr 1e-3" ... sbatch ...
# argparse honours the last occurrence of a flag, so these beat the defaults
# in the stage commands below.
JEPA_HP=${JEPA_HP:-}
CLDG_HP=${CLDG_HP:-}
DDGCL_HP=${DDGCL_HP:-}
IDOL_HP=${IDOL_HP:-}
JEPA_EXTRA="--mask_scheme $MASK_SCHEME"
[[ "$RESIDUAL" == "1" ]] && JEPA_EXTRA="$JEPA_EXTRA --residual_head"
[[ "$CHANGE_W" != "0" ]] && JEPA_EXTRA="$JEPA_EXTRA --change_weight $CHANGE_W"
[[ "$TRAIN_ENC" == "1" ]] && JEPA_EXTRA="$JEPA_EXTRA --train_encoder --ema_target"

# --- stage 0: data ----------------------------------------------------------
case "$DATASET" in
    reddit)    URL="http://snap.stanford.edu/jodie/reddit.csv"    ;;
    wikipedia) URL="http://snap.stanford.edu/jodie/wikipedia.csv" ;;
    mooc)      URL="http://snap.stanford.edu/jodie/mooc.csv"      ;;
    *) echo "[error] DATASET must be 'reddit', 'wikipedia' or 'mooc'"; exit 2 ;;
esac
CSV="dataset/jodie/${DATASET}.csv"
if [[ ! -f "$CSV" ]]; then
    echo "[stage 0] downloading $DATASET CSV from $URL"
    wget -O "$CSV" "$URL"; wget_rc=$?
    sz=$(stat -c %s "$CSV" 2>/dev/null || echo 0)
    if [[ $wget_rc -ne 0 || $sz -lt 1000 ]]; then
        echo "[FATAL] download failed (rc=$wget_rc size=$sz)." >&2
        echo "  Compute nodes usually lack internet. From the LOGIN node run:" >&2
        echo "    cd $REPO_ROOT && wget $URL -O $CSV" >&2
        echo "  then resubmit; this script will skip the download." >&2
        exit 10
    fi
fi
echo "[stage 0] $CSV size=$(du -h "$CSV" | cut -f1)"

# Sourcing normalises DATASET to the --data_source spelling (mooc -> jodie_mooc).
source "$REPO_ROOT/scripts/dblp/_dataset_config.sh"
SHORT=${DATASET#jodie_}
# The seed is part of TAG so that a multi-seed sweep does not find another
# seed's checkpoints and skip every training stage.
TAG=${TAG:-jodie_${SHORT}_${ENC_KIND}_s${SEED}}
echo "[config] TAG=$TAG STATE_MODE=$STATE_MODE JEPA_EXTRA='$JEPA_EXTRA'"

# --- checkpoint locations ---------------------------------------------------
JEPA_ENC_DIR="outputs/${TAG}_encoder"
JEPA_DIR="outputs/${TAG}_cjepa"
CLDG_DIR="outputs/cldg_${TAG}"
DDGCL_DIR="outputs/ddgcl_${TAG}"
IDOL_DIR="outputs/idol_${TAG}"

# --- DELOS: edge-mask encoder pretrain -> DELOS train -------------------------
# pretrain_encoder.py writes encoder_<src>_<kind>_<pretext>.ckpt (not
# encoder.ckpt). --pretext is pinned to edge_mask so the filename is stable.
JEPA_ENC_CKPT="$JEPA_ENC_DIR/encoder_${DATASET}_${ENC_KIND}_edge_mask.ckpt"
if [[ ! -f "$JEPA_ENC_CKPT" ]]; then
    echo; echo "=== DELOS stage 1/2: encoder pretrain -> $JEPA_ENC_DIR ==="
    srun --export=ALL python src/train/pretrain_encoder.py $DATA_ARG \
        $ENCODER_ARG $WINDOW_ARG $EARLY_STOP_ARG \
        --decoder_hidden 64 --mask_rate 0.25 --pretext edge_mask \
        --batch_size "$BATCH" --num_workers "$NW" --device cuda \
        --epochs "$ENC_EPOCHS" --lr 1e-3 \
        --seed "$SEED" --out_dir "$JEPA_ENC_DIR"
    [[ -f "$JEPA_ENC_CKPT" ]] || { echo "[FATAL] DELOS pretrain failed" >&2; exit 11; }
else
    echo "[skip] $JEPA_ENC_CKPT exists"
fi

if [[ ! -f "$JEPA_DIR/best.ckpt" ]]; then
    echo; echo "=== DELOS stage 2/2: DELOS train -> $JEPA_DIR ==="
    srun --export=ALL python src/train/train_delos.py --no_attention_mask $DATA_ARG $JEPA_EXTRA \
        $ENCODER_ARG $WINDOW_ARG $EARLY_STOP_ARG \
        --encoder_ckpt "$JEPA_ENC_CKPT" \
        --num_masked_nodes 30 --depth 4 --heads 4 --mlp_dim 512 \
        --batch_size "$BATCH" --num_workers "$NW" --device cuda \
        --epochs "$JEPA_EPOCHS" --lr 5e-4 --probe_epochs "$JEPA_PROBE_EPOCHS" \
        --seed "$SEED" --out_dir "$JEPA_DIR" $JEPA_HP
    [[ -f "$JEPA_DIR/best.ckpt" ]] || { echo "[FATAL] DELOS train failed" >&2; exit 12; }
else
    echo "[skip] $JEPA_DIR/best.ckpt exists"
fi

# --- Baselines: contrastive encoder pretrain only (the frozen encoder is probed)
if [[ "$BASELINES" != "1" ]]; then
    echo "[skip] BASELINES=0 — skipping CLDG/DDGCL/IDOL stages"
elif [[ ! -f "$CLDG_DIR/encoder.ckpt" ]]; then
    echo; echo "=== CLDG pretrain -> $CLDG_DIR ==="
    srun --export=ALL python -m src.baselines.cldg.train_cldg $DATA_ARG \
        $ENCODER_ARG $WINDOW_ARG $EARLY_STOP_ARG \
        --snapshots 4 --views 2 --strategy random --tau 0.07 \
        --node_batch_size 256 --steps_per_epoch 20 --val_steps 10 --device cuda \
        --epochs "$BASE_EPOCHS" --lr 4e-3 --seed "$SEED" --out_dir "$CLDG_DIR" $CLDG_HP
    # Hard-fail: a missing baseline would otherwise render as an absent table
    # row, which reads as "not applicable" rather than "this run crashed".
    [[ -f "$CLDG_DIR/encoder.ckpt" ]] || { echo "[FATAL] CLDG pretrain failed" >&2; exit 13; }
else
    echo "[skip] $CLDG_DIR/encoder.ckpt exists"
fi

if [[ "$BASELINES" != "1" ]]; then
    :
elif [[ ! -f "$DDGCL_DIR/encoder.ckpt" ]]; then
    echo; echo "=== DDGCL pretrain -> $DDGCL_DIR ==="
    # `--tau` is the debias class prior (the reference's tau_plus, default
    # 0.01); the trainer has no `--tau_plus` flag.
    srun --export=ALL python -m src.baselines.ddgcl.train_ddgcl $DATA_ARG \
        $ENCODER_ARG $WINDOW_ARG $EARLY_STOP_ARG \
        --max_gap 4 --time_dim 32 --tau 0.01 --beta 1.5 \
        --node_batch_size 256 --steps_per_epoch 20 --val_steps 10 --device cuda \
        --epochs "$BASE_EPOCHS" --lr 4e-3 --seed "$SEED" --out_dir "$DDGCL_DIR" $DDGCL_HP
    [[ -f "$DDGCL_DIR/encoder.ckpt" ]] || { echo "[FATAL] DDGCL pretrain failed" >&2; exit 14; }
else
    echo "[skip] $DDGCL_DIR/encoder.ckpt exists"
fi

if [[ "$BASELINES" != "1" ]]; then
    :
elif [[ ! -f "$IDOL_DIR/encoder.ckpt" ]]; then
    echo; echo "=== IDOL pretrain -> $IDOL_DIR ==="
    srun --export=ALL python -m src.baselines.idol.train_idol $DATA_ARG \
        $ENCODER_ARG $WINDOW_ARG $EARLY_STOP_ARG \
        --proj_layers 1 --node_batch_size 256 --device cuda \
        --epochs "$BASE_EPOCHS" --lr 1e-3 --seed "$SEED" --out_dir "$IDOL_DIR" $IDOL_HP
    [[ -f "$IDOL_DIR/encoder.ckpt" ]] || { echo "[FATAL] IDOL pretrain failed" >&2; exit 15; }
else
    echo "[skip] $IDOL_DIR/encoder.ckpt exists"
fi

# --- State probe (node classification) on every checkpoint -------------------
echo; echo "=== state probe: DELOS (encoder + JEPA columns)  mode=$STATE_MODE ==="
srun --export=ALL python scripts/dblp/run_block_probe.py \
    --jepa_ckpt "$JEPA_DIR/best.ckpt" \
    --task binary --state_label_mode "$STATE_MODE" \
    --device cuda --batch_size 4 --num_workers "$NW"

if [[ "$BASELINES" == "1" ]]; then
    for D in "$CLDG_DIR" "$DDGCL_DIR" "$IDOL_DIR"; do
        if [[ -f "$D/encoder.ckpt" ]]; then
            echo; echo "=== state probe: bare encoder -> $D  mode=$STATE_MODE ==="
            srun --export=ALL python scripts/dblp/run_block_probe.py \
                --encoder_ckpt "$D/encoder.ckpt" \
                --task binary --state_label_mode "$STATE_MODE" \
                --device cuda --batch_size 4 --num_workers "$NW"
        fi
    done
fi

# --- Summary table ------------------------------------------------------------
echo; echo "================================================================="
echo "  NODE CLASSIFICATION (JODIE-$SHORT ban prediction, AUROC)  [mode=$STATE_MODE]"
echo "================================================================="
srun --export=ALL python - "$JEPA_DIR" "$CLDG_DIR" "$DDGCL_DIR" "$IDOL_DIR" <<'PY'
import json, sys
from pathlib import Path

rows = [("delos", sys.argv[1]),
        ("cldg",     sys.argv[2]),
        ("ddgcl",    sys.argv[3]),
        ("idol",     sys.argv[4])]

def g(d, *ks):
    for k in ks:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d

hdr = (f"{'method':<10} {'oracle_auc':>10} {'last_auc':>9} {'jepa_auc':>9} "
       f"{'deg_auc':>8} {'deghist_auc':>11} {'last_ap':>8} {'jepa_ap':>8} "
       f"{'last_on_ap':>10} {'jepa_on_ap':>10} {'n_pos':>6}")
print(hdr); print("-" * len(hdr))
for name, d in rows:
    p = Path(d) / "state_probe.json"
    if not p.exists():
        print(f"{name:<10} {'(no state_probe.json — training may have failed)':>10}")
        continue
    r = json.loads(p.read_text())
    f = lambda x, w: (f"{x:>{w}.4f}" if isinstance(x, float) else f"{'-':>{w}}")
    npos = g(r, "n_pos_test")
    print(f"{name:<10} "
          f"{f(g(r, 'encoder', 'auroc_test'), 10)} "
          f"{f(g(r, 'encoder_last', 'auroc_test'), 9)} "
          f"{f(g(r, 'jepa', 'auroc_test'), 9)} "
          f"{f(g(r, 'degree_baseline', 'auroc_test'), 8)} "
          f"{f(g(r, 'degree_history_baseline', 'auroc_test'), 11)} "
          f"{f(g(r, 'encoder_last', 'auprc_test'), 8)} "
          f"{f(g(r, 'jepa', 'auprc_test'), 8)} "
          f"{f(g(r, 'encoder_last', 'subset_onset', 'auprc_test'), 10)} "
          f"{f(g(r, 'jepa', 'subset_onset', 'auprc_test'), 10)} "
          f"{npos if npos is not None else '-':>6}")
print()
print("oracle_auc: encoder that SAW the target snapshot (upper bound).")
print("last_auc/ap: encoder last-history latent = fair copy-last baseline.")
print("jepa_*: the forecast — the claim is jepa > last, approaching oracle.")
print("*_on_ap: onset subset (prev label 0) — persistence carries no signal there.")
print("deg_auc: degree-only confound control.")
print("deghist_auc: degree-at-each-history-bin probe = activity-TREND control;")
print("  the forecast must beat this to show learned dynamics, not raw decline.")
PY

echo; echo "[done] node-classification comparison complete ($SHORT, $ENC_KIND)."
echo "  per-method JSON: outputs/{${TAG}_cjepa,cldg_${TAG},ddgcl_${TAG},idol_${TAG}}/state_probe.json"
