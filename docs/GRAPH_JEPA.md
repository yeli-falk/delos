# Graph C-JEPA

Extension of C-JEPA from object-centric video slots to **graph nodes**, for
self-supervised representation learning on spatiotemporal and dynamic graphs.

The same machinery runs in two settings:

| setting | dataset | topology | task | metric |
|---|---|---|---|---|
| spatiotemporal forecasting | METR-LA | fixed (207 sensors) | future speed | MAE / RMSE / MAPE |
| dynamic graph (DTDG) | DBLP coauthorship-by-year | evolving | future links | MRR / Hits@k |

## Idea

C-JEPA's three video ingredients map onto graphs:

- **object slots → graph nodes** (stable identity across time)
- **history frames → graph snapshots** at `t-3 … t`
- **masked-history + forward-prediction → same loss in latent space**, but over
  node tokens instead of slot tokens.

Prediction happens in a **frozen encoder's latent space**, never in input
(adjacency / feature) space. The frozen encoder makes representation collapse
impossible (targets are constants, not co-trained), so no EMA / VICReg tricks
are needed.

```
raw signals (B,T,N,F)
  │  frozen encoder (Linear / GCN / T-GCN), pretrained GraphMAE-style
  ▼
latent slots (B,T,N,D)
  │  MaskedNodePredictor: mask M nodes' history + all future, predict latents
  ▼
predicted latents (B,T,N,D)  ── JEPA loss (masked-history + forward) in latent space
  │  linear / pairwise probe (fit separately, frozen backbone)
  ▼
forecast: speed MAE   |   link prediction MRR / Hits@k
```

Labels enter **only at the probe stage** — encoder pretraining and JEPA
training are fully self-supervised. (On METR-LA, "labels" are speed values
used as masked-reconstruction targets within a window, not held-out future
targets; the future-time supervision only appears in the probe.)

## Files

**Core model**
- [`src/cjepa_graph_predictor.py`](../src/cjepa_graph_predictor.py) —
  `MaskedNodePredictor` (spacetime transformer with SDPA attention,
  per-timestep adjacency, optional edge head), `compute_graph_jepa_loss`,
  `build_spacetime_mask`.
- [`src/world_models/graph_jepa_encoder.py`](../src/world_models/graph_jepa_encoder.py) —
  encoders (`LinearNodeEncoder`, `GCNNodeEncoder`, `TGCNEncoder`),
  `EncoderPretrainer` (masked-feature reconstruction), probes
  (`LinearProbe`, `LinkPredictionProbe`), `link_ranking_metrics`,
  `normalize_adjacency`, `build_encoder`.

**Datasets**
- [`src/custom_codes/metr_la_dataset.py`](../src/custom_codes/metr_la_dataset.py) —
  METR-LA via the HuggingFace parquet mirror or the DCRNN `.h5`; synthetic fallback.
- [`src/custom_codes/dblp_dataset.py`](../src/custom_codes/dblp_dataset.py) —
  DBLP-style yearly co-authorship from an `(src,dst,year)` CSV; synthetic fallback.

**Training**
- `src/train/pretrain_encoder_{metr_la,dblp}.py` — GraphMAE-style encoder pretraining.
- `src/train/train_cjepa_{metr_la,dblp}.py` — JEPA training + probe + eval.

**Scripts**
- `scripts/metr_la/` — `download_metr_la.sh`, `train_cjepa.sh` (laptop),
  `slurm_pretrain_encoder.sh`, `slurm_pretrain_tgcn.sh`, `slurm_train.sh`,
  `requirements.txt`.
- `scripts/dblp/` — `preprocess_dblp_xml.py` (raw dump → CSV),
  `pickled_to_csv.py` / `tgb_to_edges_csv.py` (other source formats),
  `slurm_preprocess_dblp.sh`, `slurm_pretrain_encoder.sh`, `slurm_train.sh`.

## Setup

```bash
conda create -n cjepa python=3.10 -y && conda activate cjepa
pip install -r scripts/metr_la/requirements.txt   # torch<2.6 (sm_70-friendly), einops, pandas, pyarrow, h5py, huggingface_hub
pip install lxml                                   # only for DBLP XML preprocessing
```

On a V100 (CC 7.0) pin `torch==2.5.1+cu121` — PyPI wheels ≥2.6 drop sm_70.

## METR-LA workflow

```bash
# 1. Data (HuggingFace mirror, ~217 MB)
bash scripts/metr_la/download_metr_la.sh

# 2. Pretrain the encoder (GraphMAE-style masked-speed reconstruction)
sbatch scripts/metr_la/slurm_pretrain_tgcn.sh          # T-GCN encoder
#   or scripts/metr_la/slurm_pretrain_encoder.sh        # plain GCN encoder

# 3. JEPA train + linear probe + test eval
sbatch scripts/metr_la/slurm_train.sh \
    --encoder_ckpt outputs/metr_la_encoder_tgcn/encoder.ckpt --epochs 10

# Local smoke test (no data / no GPU needed)
PYTHONPATH=. python src/train/train_cjepa_metr_la.py --synthetic --smoke --device cpu
```

The JEPA trainer auto-detects the encoder kind / dims from the checkpoint, so
you never pass `--encoder_kind` when loading a pretrained encoder.

### Results (linear probe on frozen latents)

| encoder | MAE@15 | MAE@30 | MAE@60 |
|---|---|---|---|
| random GCN | 8.00 | 8.38 | 9.19 |
| pretrained GCN | 7.56 | 8.01 | 8.58 |
| **pretrained T-GCN** | **5.28** | **5.98** | **7.22** |
| DCRNN (supervised, end-to-end) | 2.77 | 3.15 | 3.60 |

The gap to DCRNN is the standard **linear-probe-vs-supervised** gap — DCRNN
trains everything on speed labels end-to-end; we train a tiny linear map on a
frozen self-supervised representation. Add end-to-end fine-tuning to close it.

## DBLP (dynamic graph) workflow

```bash
# 1. Get edges as `src,dst,year` CSV. Either:
#    (a) parse the official XML dump (run as a job; OOMs on login nodes):
wget https://dblp.org/xml/dblp.xml.gz -O dataset/dblp/dblp.xml.gz
sbatch scripts/dblp/slurm_preprocess_dblp.sh    # → dataset/dblp/edges.csv
#    (b) convert a pickled-snapshots dump (DySAT/EvolveGCN style):
python scripts/dblp/pickled_to_csv.py --input graphs.pkl --output dataset/dblp/edges.csv

# 2. Subsample to a tractable node count (dense NxN adjacency!)
python - <<'PY'
import pandas as pd
df = pd.read_csv('dataset/dblp/edges.csv', header=None, names=['src','dst','year'])
deg = pd.concat([df.src, df.dst]).value_counts()
keep = set(deg.head(2000).index)
sub = df[df.src.isin(keep) & df.dst.isin(keep)].copy()
ids = {o:n for n,o in enumerate(sorted(set(sub.src)|set(sub.dst)))}
sub.src, sub.dst = sub.src.map(ids), sub.dst.map(ids)
sub.to_csv('dataset/dblp/edges_top2k.csv', header=False, index=False)
PY

# 3. Pretrain encoder, then JEPA train + link-prediction probe
sbatch scripts/dblp/slurm_pretrain_encoder.sh
sbatch scripts/dblp/slurm_train.sh   # auto-detects encoder.ckpt + edges.csv

# Local smoke test
PYTHONPATH=. python src/train/train_cjepa_dblp.py --synthetic --smoke \
    --device cpu --synthetic_years 60
```

## Key design notes

- **Per-timestep adjacency.** `_spatial_aggregate` dispatches any adjacency
  shape — `(N,N)` static, `(T,N,N)` dynamic, `(B,N,N)`, `(B,T,N,N)` — through
  the right einsum, so METR-LA (broadcast static) and DBLP (per-year) share one
  code path.
- **SDPA attention.** `MaskedNodePredictor` uses
  `F.scaled_dot_product_attention`; the dense `(B,1,TN,TN)` structural mask is
  off by default (`--no_attention_mask`) on METR-LA because at 207 nodes it
  forces SDPA off its memory-efficient kernel.
- **Per-batch random node masking.** Masked node indices are resampled every
  forward pass during training (fixed seed only in eval) — fixing them collapses
  training on graphs where node identity is geographic.
- **Best-checkpoint reload before probe.** Training that bottoms early then
  diverges still evaluates the best-val checkpoint, not the final-epoch one.
- **Encoder ceiling.** Plain GCN pretraining saturates at ~13% explained
  variance (no temporal mixing); T-GCN (spatial GCN + bidirectional GRU) reaches
  ~24% and is the reason MAE drops from 7.56 → 5.28.

## Status / next steps

- [x] METR-LA spatiotemporal pipeline (pretrain → JEPA → probe → MAE)
- [x] T-GCN encoder closes most of the gap to supervised baselines
- [x] DTDG scaffolding: per-timestep adjacency, DBLP dataset, link-prediction probe
- [x] DBLP XML preprocessor
- [ ] First link-prediction numbers on real DBLP (top-K subsample)
- [ ] End-to-end fine-tune variant for apples-to-apples vs supervised baselines
- [ ] Edge-masking pretraining objective (more principled for DTDG)
- [ ] Sparse adjacency for full-scale graphs (>10k nodes)
