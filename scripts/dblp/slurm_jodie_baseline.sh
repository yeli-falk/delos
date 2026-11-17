#!/usr/bin/env bash
#SBATCH --job-name=jodie-mooc
#SBATCH --output=logs/jodie-mooc-%j.out
#SBATCH --error=logs/jodie-mooc-%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#
# JODIE baseline (Kumar et al. 2019) on JODIE-MOOC dropout, 3 seeds, then a
# mean+-std report of the standard state-probe columns.
#
# JODIE has its OWN forecaster (the temporal projection operator), so it fills
# the jepa column too -- comparable to DELOS head-to-head, not just at the
# persistence level like the contrastive encoders. All columns come from the
# SAME dataset + SAME deterministic CPU probe as every other method.
#
# Submit (compute):     sbatch scripts/dblp/slurm_jodie_baseline.sh
# Merge only (no GPU):  bash   scripts/dblp/slurm_jodie_baseline.sh merge
# Resumable: each seed's train/probe skips if its output already exists.

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
# JODIE is BIPARTITE (user x item), so only the bipartite CTDG datasets apply:
#   mooc | reddit | wikipedia   (any other bipartite stream in JODIE CSV format
#   works too: drop it at dataset/jodie/<name>.csv and set DATASET=<name>).
# Homogeneous graphs (dblp, reddit_body/title) have no user/item split -> N/A.
DATASET=${DATASET:-mooc}
case "$DATASET" in
    mooc)      URL="http://snap.stanford.edu/jodie/mooc.csv"      ;;
    reddit)    URL="http://snap.stanford.edu/jodie/reddit.csv"    ;;
    wikipedia) URL="http://snap.stanford.edu/jodie/wikipedia.csv" ;;
    *)         URL="" ;;   # custom bipartite stream at dataset/jodie/<name>.csv
esac
CSV="dataset/jodie/${DATASET}.csv"

# Subset, binning, edge features, window geometry, seeds and early-stopping come
# from the shared config — identical to the DELOS runs this table is compared
# against (slurm_mooc_tier2.sh / slurm_state_probe_jodie_all.sh).
TASK=state_probe
EPOCHS=${EPOCHS:-30}
LR=${LR:-1e-3}
# Hyperparameter overrides from the search, appended verbatim (argparse honours
# the last occurrence of a flag): JODIE_HP="--lr 1e-4 --reg 0.1" sbatch ...
JODIE_HP=${JODIE_HP:-}
# Score val every N epochs. Val replays the stream a second time, so on MOOC it
# more than doubles the epoch; raise this if you hit the walltime.
VAL_EVERY=${VAL_EVERY:-1}
# HORIZONS>1 re-probes existing checkpoints at steps 1..HORIZONS (no retraining;
# all per-bin embeddings are stored). Kept separate from NPRED, which the shared
# config pins to the trained window geometry — overloading one variable for both
# meant the "did the user ask for a curve?" test was always true.
HORIZONS=${HORIZONS:-}
STATE_MODE=${STATE_MODE:-state}
NW=${SLURM_CPUS_PER_TASK:-8}
PREFIX=${PREFIX:-jodie_${DATASET}}

# Sets DATA_ARG / WINDOW_ARG / EARLY_STOP_ARG / NODE_DIM / SEED / SEEDS, and
# normalises DATASET to the --data_source spelling (mooc -> jodie_mooc).
source "$REPO_ROOT/scripts/dblp/_dataset_config.sh"

out_dir() { echo "outputs/${PREFIX}_s$1"; }

# ============================================================================
# MERGE-ONLY MODE
# ============================================================================
merge() {
    echo; echo "===== JODIE-MOOC baseline (state=$STATE_MODE) ====="
    $PY - "$PREFIX" <<'PYEOF'
import glob, json, math, re, sys, statistics as st
from pathlib import Path

prefix = sys.argv[1]
pat = re.compile(rf"{re.escape(prefix)}_s(?P<seed>\d+)$")

def get(d, *ks):
    for k in ks:
        if not isinstance(d, dict) or k not in d: return None
        d = d[k]
    if isinstance(d, float) and math.isnan(d): return None   # empty-subset nan
    return d

runs = []
for p in sorted(glob.glob(f"outputs/{prefix}_s*/state_probe.json")):
    m = pat.match(Path(p).parent.name)
    if m:
        runs.append((int(m.group("seed")), json.loads(Path(p).read_text())))

if not runs:
    print(f"[merge] no runs found under outputs/{prefix}_s*/state_probe.json")
    raise SystemExit(0)

def col(*ks):
    vals = [get(r, *ks) for _, r in runs]
    vals = [v for v in vals if v is not None]
    if not vals: return "   -    "
    m = st.mean(vals); s = st.stdev(vals) if len(vals) > 1 else 0.0
    return f"{m:.4f}±{s:.4f}"

# paired gap jepa - copy-last, within each run
gaps = [get(r,'jepa','auroc_test') - get(r,'encoder_last','auroc_test')
        for _, r in runs
        if get(r,'jepa','auroc_test') is not None
        and get(r,'encoder_last','auroc_test') is not None]
gm = st.mean(gaps); gs = st.stdev(gaps) if len(gaps) > 1 else 0.0

seeds = ", ".join(str(s) for s, _ in runs)
npos = get(runs[0][1], 'n_pos_test')
print(f"method = JODIE   seeds = [{seeds}]   n_pos_test = {npos}\n")
print(f"  jepa_auroc            {col('jepa','auroc_test')}   (JODIE's projection forecast)")
print(f"  jepa_ap               {col('jepa','auprc_test')}")
print(f"  encoder_last_auroc    {col('encoder_last','auroc_test')}   (persistence / copy-last)")
print(f"  encoder_last_ap       {col('encoder_last','auprc_test')}")
print(f"  jepa - last (gap)     {gm:+.4f}±{gs:.4f}   (>0 => projection beats copy-last)")
print(f"  degree_history_auroc  {col('degree_history_baseline','auroc_test')}   (trend control -- must be beaten)")
print(f"  oracle_auroc          {col('encoder_oracle','auroc_test')}   (upper bound)")
print(f"  jepa_onset_ap         {col('jepa','subset_onset','auprc_test')}")

# --- per-horizon curve (present only when probed with --num_preds > 1) -------
hs = sorted({int(k[8:]) for _, r in runs for k in (get(r, 'jepa') or {})
             if k.startswith('subset_h')})
if hs:
    def hcell(tp, who, metric):
        vals = [get(r, who, f'subset_h{tp}', metric) for _, r in runs]
        vals = [v for v in vals if v is not None]
        if not vals: return "   -    "
        m = st.mean(vals); s = st.stdev(vals) if len(vals) > 1 else 0.0
        return f"{m:.4f}±{s:.4f}"
    print("\n  PROJECTION HORIZON CURVE (does the projected forecast keep "
          "beating copy-last as it reaches further?)")
    print(f"    {'step':>4} {'jepa_auroc':>15} {'last_auroc':>15} "
          f"{'gap':>16} {'jepa_ap':>15} {'n_pos':>6}")
    for tp in hs:
        gaps = [get(r,'jepa',f'subset_h{tp}','auroc_test')
                - get(r,'encoder_last',f'subset_h{tp}','auroc_test')
                for _, r in runs
                if get(r,'jepa',f'subset_h{tp}','auroc_test') is not None
                and get(r,'encoder_last',f'subset_h{tp}','auroc_test') is not None]
        gm = st.mean(gaps) if gaps else float('nan')
        gs = st.stdev(gaps) if len(gaps) > 1 else 0.0
        npos = [get(r,'jepa',f'subset_h{tp}','n_pos') for _, r in runs]
        npos = [n for n in npos if n is not None]
        npm = int(st.mean(npos)) if npos else 0
        print(f"    {tp+1:>4} {hcell(tp,'jepa','auroc_test'):>15} "
              f"{hcell(tp,'encoder_last','auroc_test'):>15} "
              f"{gm:+.4f}±{gs:.4f}   {hcell(tp,'jepa','auprc_test'):>15} {npm:>6}")
    print("    step = bins ahead of the last observed bin. Growing gap = the "
          "projection extrapolates; flat/shrinking = copy-last catches up")
    print("    (expected on MOOC's ABSORBING 'state' label -- persistence wins "
          "far out, since dropout never reverts).")
PYEOF
}

if [[ "${1:-}" == "merge" ]]; then merge; exit 0; fi

# ============================================================================
# COMPUTE
# ============================================================================
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
fi
# --- data --------------------------------------------------------------------
if [[ ! -f "$CSV" ]]; then
    if [[ -z "$URL" ]]; then
        echo "[FATAL] no CSV at $CSV and no known URL for DATASET=$DATASET." >&2
        echo "  Place a JODIE-format CSV there, or use a known DATASET." >&2
        exit 10
    fi
    echo "[data] downloading $DATASET csv from $URL..."
    wget -O "$CSV" "$URL" || true
    if [[ ! -s "$CSV" || $(stat -c %s "$CSV" 2>/dev/null || echo 0) -lt 1000 ]]; then
        echo "[FATAL] $DATASET download failed (compute nodes usually lack internet)." >&2
        echo "  From the LOGIN node: wget $URL -O $CSV" >&2
        exit 10
    fi
fi

for SEED in $SEEDS; do
    OD=$(out_dir "$SEED")
    if [[ ! -f "$OD/jodie.ckpt" ]]; then
        echo; echo "=== [seed $SEED] train JODIE -> $OD ==="
        srun --export=ALL $PY -m src.baselines.jodie.train_jodie $DATA_ARG \
            $WINDOW_ARG $EARLY_STOP_ARG \
            --node_dim "$NODE_DIM" --epochs "$EPOCHS" --lr "$LR" \
            --device cuda --val_every "$VAL_EVERY" \
            --seed "$SEED" --out_dir "$OD" $JODIE_HP
        [[ -f "$OD/jodie.ckpt" ]] || { echo "[FATAL] train failed: seed $SEED" >&2; exit 12; }
    else
        echo "[skip] $OD/jodie.ckpt exists"
    fi

    # With HORIZONS set we ALWAYS re-probe (to pick up the requested horizons);
    # otherwise skip if the single-horizon probe already ran.
    PROBE_EXTRA=""; [[ -n "$HORIZONS" ]] && PROBE_EXTRA="--num_preds $HORIZONS"
    if [[ -n "$HORIZONS" || ! -f "$OD/state_probe.json" ]]; then
        echo "=== [seed $SEED] state probe ${HORIZONS:+(horizons 1..$HORIZONS)} ==="
        srun --export=ALL $PY scripts/dblp/run_block_probe.py \
            --jodie_ckpt "$OD/jodie.ckpt" $PROBE_EXTRA \
            --task binary --state_label_mode "$STATE_MODE" \
            --device cuda --batch_size 4 --num_workers "$NW"
    else
        echo "[skip] $OD/state_probe.json exists"
    fi
done

merge
echo; echo "[done] JODIE baseline complete. Re-merge anytime: bash $0 merge"
