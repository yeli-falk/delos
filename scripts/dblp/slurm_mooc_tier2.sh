#!/usr/bin/env bash
#SBATCH --job-name=mooc-tier2
#SBATCH --output=logs/mooc-tier2-%j.out
#SBATCH --error=logs/mooc-tier2-%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#
# Tier-2 (predictor-mechanism) ablation on JODIE-MOOC dropout (state task).
#
# Question: is the win the MASKED GRAPH predictor, or just having ANY forecaster
# on top of the frozen encoder? Four predictor heads, ONE shared frozen encoder
# per seed (so encoder_last / copy-last is identical across heads and the only
# thing that varies is the predictor):
#
#   node_mask        the model      (masked spacetime transformer)
#   future_forecast  same transformer, masking OFF   (isolates masking)
#   mlp_forecast     graph-free per-node MLP          (isolates the whole design)
#   gru_forecast     graph-free per-node GRU          (recurrent generic head)
#
# All use the locked recipe: tgcn + edge features + residual_head + change_weight 2,
# frozen encoder (TRAIN_ENC=0). Bars to beat live in every state_probe.json:
# encoder_last (copy-last / dygjepa_persistence) and degree_history (trend).
#
# Submit (compute):     sbatch scripts/dblp/slurm_mooc_tier2.sh
# Merge only (no GPU):  bash   scripts/dblp/slurm_mooc_tier2.sh merge
#
# Resumable: every stage skips if its output exists — just resubmit on walltime.

set -uo pipefail
mkdir -p logs outputs dataset/jodie

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"

PY=${PY:-python}
if [[ -d "$HOME/.conda/envs/cjepa" ]]; then
    source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null && conda activate cjepa
fi

# --- config (env-overridable) -----------------------------------------------
SCHEMES=${SCHEMES:-"node_mask future_forecast mlp_forecast gru_forecast"}
# Dataset subset, binning, edge features, encoder, window geometry, seeds and
# early-stopping come from the shared config — the SAME values the baseline
# comparison (slurm_state_probe_jodie_all.sh) and the JODIE baseline
# (slurm_jodie_baseline.sh) use, so all three tables can be merged.
DATASET=mooc
TASK=state_probe
BATCH=${BATCH:-8}
# 30 epochs was a hard cap, not convergence: the pretrain val loss was
# still falling at epoch 30 (best val landed ON the last epoch), and with
# PATIENCE=20 early stopping cannot fire inside 30 epochs. pretrain_encoder
# restores best-val weights, so a high cap is monotone-safe — it can only
# find an equal or better checkpoint. Let PATIENCE decide when to stop.
ENC_EPOCHS=${ENC_EPOCHS:-300}
JEPA_EPOCHS=${JEPA_EPOCHS:-50}
JEPA_PROBE_EPOCHS=${JEPA_PROBE_EPOCHS:-3}
STATE_MODE=${STATE_MODE:-state}
CHANGE_W=${CHANGE_W:-2.0}
NW=${SLURM_CPUS_PER_TASK:-8}
PREFIX=${PREFIX:-mooc_tier2}
CSV="dataset/jodie/mooc.csv"

source "$REPO_ROOT/scripts/dblp/_dataset_config.sh"

enc_dir() { echo "outputs/${PREFIX}_enc_s$1"; }
enc_ckpt() { echo "$(enc_dir "$1")/encoder_jodie_mooc_${ENC_KIND}_edge_mask.ckpt"; }
jepa_dir() { echo "outputs/${PREFIX}_$2_s$1"; }   # $1=seed $2=scheme

# ============================================================================
# MERGE-ONLY MODE
# ============================================================================
merge() {
    echo; echo "===== MERGE: MOOC Tier-2 (state=$STATE_MODE) ====="
    $PY - "$PREFIX" <<'PYEOF'
import glob, json, math, re, sys, statistics as st
from pathlib import Path

prefix = sys.argv[1]
pat = re.compile(rf"{re.escape(prefix)}_(?P<scheme>.+)_s(?P<seed>\d+)$")

def get(d, *ks):
    for k in ks:
        if not isinstance(d, dict) or k not in d: return None
        d = d[k]
    if isinstance(d, float) and math.isnan(d): return None   # empty-subset nan
    return d

# scheme -> list of per-seed dicts
rows = {}
for p in sorted(glob.glob(f"outputs/{prefix}_*_s*/state_probe.json")):
    m = pat.match(Path(p).parent.name)
    if not m: continue
    scheme = m.group("scheme")
    if scheme == "enc":  # the bare encoder-pretrain dirs, if any leaked a json
        continue
    r = json.loads(Path(p).read_text())
    rows.setdefault(scheme, []).append((int(m.group("seed")), r))

def agg(vals):
    vals = [v for v in vals if v is not None]
    if not vals: return None, None
    m = st.mean(vals)
    s = st.stdev(vals) if len(vals) > 1 else 0.0
    return m, s

def cell(vals):
    m, s = agg(vals)
    return f"{m:.4f}±{s:.4f}" if m is not None else "   -    "

order = ["node_mask", "future_forecast", "mlp_forecast", "gru_forecast"]
schemes = [s for s in order if s in rows] + [s for s in rows if s not in order]

hdr = (f"{'scheme':<16} {'jepa_auroc':>14} {'jepa_ap':>14} "
       f"{'jepa-last(gap)':>16} {'jepa_onset_ap':>14} "
       f"{'last_auroc':>14} {'deghist_auroc':>14} {'oracle_auroc':>14} {'seeds':>6}")
print(hdr); print("-"*len(hdr))
for sch in schemes:
    seeds = rows[sch]
    ja  = [get(r,'jepa','auroc_test') for _,r in seeds]
    jap = [get(r,'jepa','auprc_test') for _,r in seeds]
    la  = [get(r,'encoder_last','auroc_test') for _,r in seeds]
    lap = [get(r,'encoder_last','auprc_test') for _,r in seeds]
    dh  = [get(r,'degree_history_baseline','auroc_test') for _,r in seeds]
    orc = [get(r,'encoder_oracle','auroc_test') for _,r in seeds]
    jon = [get(r,'jepa','subset_onset','auprc_test') for _,r in seeds]
    # PAIRED gap per seed (the stat that matters — same encoder within a seed)
    gap = [a-b for a,b in zip(ja,la) if a is not None and b is not None]
    print(f"{sch:<16} {cell(ja):>14} {cell(jap):>14} {cell(gap):>16} "
          f"{cell(jon):>14} {cell(la):>14} {cell(dh):>14} {cell(orc):>14} "
          f"{len(seeds):>6}")

print()
# Sanity: WITHIN a seed, all heads share ONE frozen encoder, so encoder_last
# must be ~identical. (Across seeds it legitimately differs — different
# encoders — so an all-runs spread is meaningless; the within-seed spread is
# the real determinism check.)
by_seed = {}
for sch in schemes:
    for seed, r in rows[sch]:
        v = get(r,'encoder_last','auroc_test')
        if v is not None:
            by_seed.setdefault(seed, []).append(v)
spreads = {s: max(vs)-min(vs) for s, vs in by_seed.items() if len(vs) > 1}
if spreads:
    per = ', '.join(f"{s}:{d:.4f}" for s, d in sorted(spreads.items()))
    print(f"[sanity] within-seed encoder_last spread across heads: "
          f"max={max(spreads.values()):.4f}  per-seed={{{per}}}")
    print("         shared frozen encoder -> should be ~0.000; "
          "large = nondeterministic probe.")
print()
print("gap = jepa_auroc - encoder_last (paired within seed). >0 = predictor beats copy-last.")
print("The claim: node_mask gap > {future_forecast, mlp, gru} gaps -> the MASKING is the win,")
print("not merely having a forecaster. All must also clear deghist to be 'learned dynamics'.")

# --- per-horizon curve (present only when trained with num_preds > 1) --------
def gcell(vals):
    m, s = agg(vals)
    return f"{m:+.4f}±{s:.4f}" if m is not None else "    -     "

hs = sorted({int(k[8:]) for sch in schemes for _, r in rows[sch]
             for k in (get(r,'jepa') or {}) if k.startswith('subset_h')})
if hs:
    print()
    print("PREDICTOR HORIZON CURVE -- paired gap (jepa_auroc - encoder_last) per head, per step")
    print("(same axes as the JODIE projection curve; growing gap = predictor extrapolates,")
    print(" shrinking = copy-last catches up -- expected on MOOC's absorbing 'state' label).")
    hdr2 = f"{'step':>4} " + " ".join(f"{s[:14]:>16}" for s in schemes) + f" {'n_pos':>6}"
    print(hdr2); print("-" * len(hdr2))
    for tp in hs:
        cells, npos = [], None
        for sch in schemes:
            gseed = []
            for _, r in rows[sch]:
                a = get(r,'jepa',f'subset_h{tp}','auroc_test')
                b = get(r,'encoder_last',f'subset_h{tp}','auroc_test')
                if a is not None and b is not None:
                    gseed.append(a - b)
                if npos is None:
                    npos = get(r,'jepa',f'subset_h{tp}','n_pos')
            cells.append(gcell(gseed))
        line = f"{tp+1:>4} " + " ".join(f"{c:>16}" for c in cells)
        print(line + f" {(npos if npos is not None else '-'):>6}")
PYEOF
}

if [[ "${1:-}" == "merge" ]]; then merge; exit 0; fi

# ============================================================================
# COMPUTE
# ============================================================================
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
fi
echo "[cfg] SEEDS='$SEEDS' SCHEMES='$SCHEMES' ENC_KIND=$ENC_KIND BINS=$BINS "\
     "CHANGE_W=$CHANGE_W STATE_MODE=$STATE_MODE"

# --- data --------------------------------------------------------------------
if [[ ! -f "$CSV" ]]; then
    echo "[data] downloading MOOC csv..."
    wget -O "$CSV" "http://snap.stanford.edu/jodie/mooc.csv" || true
    if [[ ! -s "$CSV" || $(stat -c %s "$CSV" 2>/dev/null || echo 0) -lt 1000 ]]; then
        echo "[FATAL] MOOC download failed (compute nodes usually lack internet)." >&2
        echo "  From the LOGIN node: wget http://snap.stanford.edu/jodie/mooc.csv -O $CSV" >&2
        echo "  then resubmit." >&2
        exit 10
    fi
fi

for SEED in $SEEDS; do
    ENC_DIR=$(enc_dir "$SEED"); ENC_CKPT=$(enc_ckpt "$SEED")

    # --- stage 1: ONE shared encoder for this seed --------------------------
    if [[ ! -f "$ENC_CKPT" ]]; then
        echo; echo "=== [seed $SEED] encoder pretrain -> $ENC_DIR ==="
        srun --export=ALL $PY src/train/pretrain_encoder.py $DATA_ARG \
            $ENCODER_ARG $WINDOW_ARG $EARLY_STOP_ARG \
            --decoder_hidden 64 --mask_rate 0.25 --pretext edge_mask \
            --batch_size "$BATCH" --num_workers "$NW" --device cuda \
            --epochs "$ENC_EPOCHS" --lr 1e-3 --seed "$SEED" \
            --out_dir "$ENC_DIR"
        [[ -f "$ENC_CKPT" ]] || { echo "[FATAL] encoder pretrain failed (seed $SEED)" >&2; exit 11; }
    else
        echo "[skip] $ENC_CKPT exists"
    fi

    # --- stage 2+3: each predictor head on the SAME frozen encoder ----------
    for SCHEME in $SCHEMES; do
        JD=$(jepa_dir "$SEED" "$SCHEME")
        EXTRA="--mask_scheme $SCHEME --residual_head --change_weight $CHANGE_W"

        if [[ ! -f "$JD/best.ckpt" ]]; then
            echo; echo "=== [seed $SEED / $SCHEME] DELOS train -> $JD ==="
            srun --export=ALL $PY src/train/train_delos.py --no_attention_mask \
                $DATA_ARG $EXTRA --encoder_ckpt "$ENC_CKPT" \
                $ENCODER_ARG $WINDOW_ARG $EARLY_STOP_ARG \
                --num_masked_nodes 30 --depth 4 --heads 4 --mlp_dim 512 \
                --batch_size "$BATCH" --num_workers "$NW" --device cuda \
                --epochs "$JEPA_EPOCHS" --lr 5e-4 --probe_epochs "$JEPA_PROBE_EPOCHS" \
                --seed "$SEED" --out_dir "$JD"
            [[ -f "$JD/best.ckpt" ]] || { echo "[warn] train failed: seed $SEED $SCHEME" >&2; continue; }
        else
            echo "[skip] $JD/best.ckpt exists"
        fi

        if [[ ! -f "$JD/state_probe.json" ]]; then
            echo "=== [seed $SEED / $SCHEME] state probe ==="
            srun --export=ALL $PY scripts/dblp/run_block_probe.py \
                --jepa_ckpt "$JD/best.ckpt" \
                --task binary --state_label_mode "$STATE_MODE" \
                --device cuda --batch_size 4 --num_workers "$NW"
        else
            echo "[skip] $JD/state_probe.json exists"
        fi
    done
done

merge
echo; echo "[done] MOOC Tier-2 complete. Re-merge anytime: bash $0 merge"
