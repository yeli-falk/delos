#!/usr/bin/env bash
#SBATCH --job-name=ec-allmethods
#SBATCH --output=logs/ec-allmethods-%j.out
#SBATCH --error=logs/ec-allmethods-%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#
# England COVID final comparison table, ONE seed per job:
#   parameter-free (persistence / window_mean / historical_mean)
#   DELOS        (encoder pretrain -> JEPA predictor + recipe, forecast rollout)
#   DELOS-enc    (its encoder + a forecast head, NO predictor — "does the predictor help?")
#   CLDG / DDGCL / IDOL (contrastive encoder + the same forecast head)
#
# Every learned method uses the SAME encoder-only forecast head + the same
# persistence/inflection eval (scripts/regression/probe_encoder_regression.py),
# so the columns are directly comparable; DELOS additionally has its
# predictor-rollout row (the full method).
#
# Multi-seed sweep:
#   for SEED in 42 43 44; do SEED=$SEED sbatch scripts/england_covid/slurm_all_methods.sh; done
# Recipe / geometry knobs (defaults = the validated recipe):
#   RESIDUAL=1 CHANGE_W=2.0 PERSIST_RES=1 TRAIN_ENC=1 HS=8 NP=6 SEED=42 sbatch ...

set -uo pipefail
mkdir -p logs

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"
if [[ -d "$HOME/.conda/envs/cjepa" ]]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate cjepa
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

SEED="${SEED:-42}"
HS="${HS:-8}"
NP="${NP:-3}"
NW="${SLURM_CPUS_PER_TASK:-4}"
# TAG distinguishes variant runs (e.g. the TRAIN_ENC=0 frozen-encoder control)
# so they write to their OWN dirs instead of overwriting the default sweep.
#   TRAIN_ENC=0 TAG=frozen SEED=42 sbatch ...   ->  outputs/ec_dygjepa_frozen_s42
TAG="${TAG:-}"
SUF="${TAG:+_$TAG}_s${SEED}"

# --- DELOS predictor recipe (the validated fix, ported to regression) ---
RESIDUAL="${RESIDUAL:-1}"
CHANGE_W="${CHANGE_W:-2.0}"
PERSIST_RES="${PERSIST_RES:-1}"
# Frozen encoder by default: ec_dygjepa and ec_dygjepa_encoder then share the
# SAME frozen encoder, so their difference isolates the predictor rollout (the
# clean claim). TRAIN_ENC=1 is now a real ablation (joint encoder fine-tuning),
# not a silent no-op as it used to be in the regression path.
TRAIN_ENC="${TRAIN_ENC:-0}"
JEPA_EXTRA=""
[[ "$RESIDUAL" == "1" ]]    && JEPA_EXTRA="$JEPA_EXTRA --residual_head"
[[ "$CHANGE_W" != "0" ]]    && JEPA_EXTRA="$JEPA_EXTRA --change_weight $CHANGE_W"
[[ "$PERSIST_RES" == "1" ]] && JEPA_EXTRA="$JEPA_EXTRA --persist_residual"
[[ "$TRAIN_ENC" == "1" ]]   && JEPA_EXTRA="$JEPA_EXTRA --train_encoder --ema_target"
# The encoder-only probe (baselines + the JEPA-enc control) mirrors the JEPA
# persist_residual setting so every column shares the same floor.
PR_FLAG=""; [[ "$PERSIST_RES" == "1" ]] && PR_FLAG="--persist_residual"
echo "[config] SEED=$SEED HS=$HS NP=$NP JEPA_EXTRA='$JEPA_EXTRA'"

# Shared dataset / encoder / window / early-stop config for ALL methods — the
# same file every other comparison sources, so this table uses the identical
# encoder and model-selection rule. HIST/NPRED are this task's forecasting
# window (HS/NP), passed in before sourcing.
DATASET=england_covid
TASK=regression
HIST="$HS" NPRED="$NP"
source "$REPO_ROOT/scripts/dblp/_dataset_config.sh"
# `--bidirectional false` keeps the encoder temporally causal (assumption A4);
# it is a no-op for the shape-agnostic `gcn` the shared config pins.
ENC_ARGS="$ENCODER_ARG --bidirectional false"
PROBE_ARGS="$WINDOW_ARG --probe_epochs 10 --probe_hidden 64 \
    $PR_FLAG --batch_size 8 --num_workers $NW --seed $SEED --device cuda"

JEPA_DIR="outputs/ec_dygjepa${SUF}"
JEPA_ENC_DIR="outputs/ec_dygjepa_encoder${SUF}"
JEPA_ENC_CKPT="$JEPA_ENC_DIR/encoder_england_covid_tgcn_masked_signal.ckpt"
CLDG_DIR="outputs/ec_cldg${SUF}"
DDGCL_DIR="outputs/ec_ddgcl${SUF}"
IDOL_DIR="outputs/ec_idol${SUF}"
mkdir -p "$JEPA_DIR" "$JEPA_ENC_DIR" "$CLDG_DIR" "$DDGCL_DIR" "$IDOL_DIR" \
    outputs/england_covid_baselines

# =============================================================================
# 0. parameter-free baselines (persistence / window_mean / historical_mean)
# =============================================================================
echo; echo "=== parameter-free baselines ==="
srun --export=ALL python scripts/regression/run_baselines.py \
    --history_size $HS --num_preds $NP \
    --out_dir outputs/england_covid_baselines || echo "[warn] baselines failed"

# =============================================================================
# 1. DELOS: encoder pretrain (masked_signal) -> JEPA predictor (recipe)
# =============================================================================
if [[ ! -f "$JEPA_ENC_CKPT" ]]; then
    echo; echo "=== DELOS 1/2: encoder pretrain -> $JEPA_ENC_DIR ==="
    srun --export=ALL python src/train/pretrain_encoder.py $DATA_ARG \
        --pretext masked_signal $ENC_ARGS --mask_rate 0.25 \
        $WINDOW_ARG $EARLY_STOP_ARG \
        --batch_size 8 --epochs 300 --lr 1e-3 \
        --seed "$SEED" --num_workers "$NW" --device cuda --out_dir "$JEPA_ENC_DIR"
    [[ -f "$JEPA_ENC_CKPT" ]] || { echo "[FATAL] JEPA pretrain failed" >&2; exit 11; }
else
    echo "[skip] $JEPA_ENC_CKPT exists"
fi

echo; echo "=== DELOS 2/2: predictor train -> $JEPA_DIR ==="
srun --export=ALL python src/train/train_delos.py $DATA_ARG \
    --downstream_task regression --mask_scheme node_mask $JEPA_EXTRA \
    $WINDOW_ARG $EARLY_STOP_ARG $ENC_ARGS \
    --encoder_ckpt "$JEPA_ENC_CKPT" \
    --num_masked_nodes 20 --depth 6 --heads 8 --mlp_dim 512 \
    --batch_size 8 --epochs 50 --lr 5e-4 --weight_decay 1e-4 \
    --probe_epochs 10 --probe_hidden 64 \
    --seed "$SEED" --num_workers "$NW" --device cuda --out_dir "$JEPA_DIR"

# DELOS-enc control: its OWN encoder + a forecast head, no predictor.
echo; echo "=== DELOS-enc control (encoder-only, no predictor) ==="
srun --export=ALL python scripts/regression/probe_encoder_regression.py \
    --encoder_ckpt "$JEPA_ENC_CKPT" $PROBE_ARGS || echo "[warn] jepa-enc probe failed"

# =============================================================================
# 2. contrastive baselines: pretrain encoder -> encoder-only forecast head
# =============================================================================
if [[ ! -f "$CLDG_DIR/encoder.ckpt" ]]; then
    echo; echo "=== CLDG pretrain -> $CLDG_DIR ==="
    srun --export=ALL python -m src.baselines.cldg.train_cldg $DATA_ARG $ENC_ARGS \
        $WINDOW_ARG $EARLY_STOP_ARG \
        --snapshots 4 --views 2 --strategy random --tau 0.07 \
        --node_batch_size 128 --steps_per_epoch 20 --val_steps 10 \
        --epochs 100 --lr 4e-3 \
        --device cuda --seed "$SEED" --out_dir "$CLDG_DIR"
    # Hard-fail: a warn left the method absent from the table, which reads as
    # "not applicable" rather than "this run crashed".
    [[ -f "$CLDG_DIR/encoder.ckpt" ]] || { echo "[FATAL] CLDG pretrain failed" >&2; exit 13; }
fi
if [[ ! -f "$DDGCL_DIR/encoder.ckpt" ]]; then
    echo; echo "=== DDGCL pretrain -> $DDGCL_DIR ==="
    # `--tau` IS the debias class prior (the reference's tau_plus, default
    # 0.01). Passing `--tau_plus` here aborted the stage with an argparse error.
    srun --export=ALL python -m src.baselines.ddgcl.train_ddgcl $DATA_ARG $ENC_ARGS \
        $WINDOW_ARG $EARLY_STOP_ARG \
        --max_gap 4 --time_dim 32 --tau 0.01 --beta 1.5 \
        --node_batch_size 128 --steps_per_epoch 20 --val_steps 10 \
        --epochs 100 --lr 4e-3 \
        --device cuda --seed "$SEED" --out_dir "$DDGCL_DIR"
    [[ -f "$DDGCL_DIR/encoder.ckpt" ]] || { echo "[FATAL] DDGCL pretrain failed" >&2; exit 14; }
fi
if [[ ! -f "$IDOL_DIR/encoder.ckpt" ]]; then
    echo; echo "=== IDOL pretrain -> $IDOL_DIR ==="
    srun --export=ALL python -m src.baselines.idol.train_idol $DATA_ARG $ENC_ARGS \
        $WINDOW_ARG $EARLY_STOP_ARG \
        --proj_layers 1 --node_batch_size 128 --epochs 100 --lr 1e-3 \
        --device cuda --seed "$SEED" --out_dir "$IDOL_DIR"
    [[ -f "$IDOL_DIR/encoder.ckpt" ]] || { echo "[FATAL] IDOL pretrain failed" >&2; exit 15; }
fi

for D in "$CLDG_DIR" "$DDGCL_DIR" "$IDOL_DIR"; do
    if [[ -f "$D/encoder.ckpt" ]]; then
        echo; echo "=== encoder-only forecast probe -> $D ==="
        srun --export=ALL python scripts/regression/probe_encoder_regression.py \
            --encoder_ckpt "$D/encoder.ckpt" $PROBE_ARGS \
            || echo "[warn] probe failed for $D"
    fi
done

# =============================================================================
# 3. summary table (forecast MAE vs persistence, overall + inflection subset)
# =============================================================================
echo; echo "================================================================="
echo "  ENGLAND COVID forecast comparison (seed $SEED, HS=$HS NP=$NP)"
echo "================================================================="
srun --export=ALL python - "$JEPA_DIR" "$JEPA_ENC_DIR" "$CLDG_DIR" "$DDGCL_DIR" "$IDOL_DIR" <<'PY'
import json, sys
from pathlib import Path

methods = [("delos(full)", sys.argv[1]),
           ("delos(enc)",  sys.argv[2]),
           ("cldg",           sys.argv[3]),
           ("ddgcl",          sys.argv[4]),
           ("idol",           sys.argv[5])]

def load(d):
    p = Path(d) / "test_metrics.json"
    return json.loads(p.read_text()) if p.exists() else None

# Persistence is parameter-free & identical across methods; print it once.
ref = next((load(d) for _, d in methods if load(d)), None)
if ref:
    print(f"{'horizon':<8}{'persist_MAE':>12}{'persist_infl':>13}")
    for h in sorted(ref, key=int):
        m = ref[h]
        pi = m.get('MAE_persist_inflect', float('nan'))
        print(f"+{h:<7}{m['MAE_persist']:>12.3f}{pi:>13.3f}")
    print()

hdr = f"{'method':<16}" + "".join(f"h{h}_MAE  h{h}_infl   " for h in sorted(ref, key=int)) if ref else ""
print(hdr); print("-" * len(hdr))
for name, d in methods:
    m = load(d)
    if not m:
        print(f"{name:<16}(no test_metrics.json)"); continue
    cells = ""
    for h in sorted(m, key=int):
        e = m[h]
        infl = e.get('MAE_inflect', float('nan'))
        cells += f"{e['MAE']:>6.3f} {infl:>7.3f}   "
    print(f"{name:<16}{cells}")
print()
print("Lower is better. Compare each method's h*_MAE to persist_MAE (overall)")
print("and h*_infl to persist_infl (turning points — where forecasting must win).")
print("delos(full)=predictor rollout; (enc)=same encoder, forecast head only.")
PY

echo; echo "[done] seed $SEED. Aggregate later across seeds from the test_metrics.json files."
