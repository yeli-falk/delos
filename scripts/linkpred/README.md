# Dynamic link-prediction pipeline

Datasets: `mooc`, `reddit`, `wikipedia` (JODIE), plus `reddit_body`
(Kumar 2018 subreddit hyperlinks) and `dblp`. The last two are
**link-prediction only** — they carry no `state_label`, so the
classification pipeline deliberately rejects them.

Third of the three task pipelines, structurally identical to
[`scripts/regression/`](../regression/) and
[`scripts/classification/`](../classification/): an `ARMS`-gated launcher, a
three-stage sweep that only reads **val**, and a mean ± std LaTeX aggregator.

## Task and metrics

Rank each positive edge against 99 sampled negatives per window. Metrics are
**MRR / Hits@1 / Hits@3 / Hits@10**, averaged over horizons, reported over two
edge subsets:

| subset | what it contains |
|---|---|
| `all` | every positive edge in the horizon — dominated by edges already present in the history window, which a model scores well by echoing the current graph |
| `new` | only edges absent from every earlier snapshot — **formation, not persistence** |

`new` is the analogue of the forecasting inflection subset and of the
classification `gap`: the subset where echoing the input does not help. Report
both; the sweep ranks on `new` by default for the same reason.

**`TASK=link_pred` uses the full dataset** (every node; JODIE datasets have no
subset knob). The probe scores 99 negatives per positive per window, so cost
grows ~N²; lower `BATCH` if you OOM.

## Arms

| row | what it is |
|---|---|
| `random` | U(0,1) edge scores, no training. The floor. |
| `supervised` | encoder + link head trained end-to-end from random init, no SSL |
| `grace` | GRACE (Zhu et al., 2020): **static** graph contrast — two augmented views (edge removal + feature-dimension masking) of one snapshot, inter- and intra-view negatives. Never sees two snapshots at once, so it is the control for "does the SSL objective need to be temporal at all?"; encoder scored by the same frozen link probe (static SSL, last frame) |
| `graphmae` | GraphMAE (Hou et al., KDD'22): **static** masked-feature autoencoder — mask a fraction of one snapshot's nodes with a learnable token, re-mask their latents, GCN decoder, scaled cosine error on the masked nodes. The generative half of the static control block (GRACE is the contrastive half); encoder scored by the same frozen link probe (static SSL, last frame) |
| `cldg` / `ddgcl` / `idol` | contrastive SSL encoder, scored by the same link probe on frozen latents |
| `maskdgnn` | MaskDGNN (He et al., IJCAI'25): activeness-aware edge masking + masked/next-edge BCE on a GCN with an adaptive frequency layer. **Two rows**: its encoder through the same frozen link probe (`MaskDGNN`), and its own encoder + edge head scored end-to-end by the same evaluator (`MaskDGNN (end-to-end)` — the paper's protocol, read next to `supervised`) |
| `dysubc` | DySubC (Chen et al., TKDD'23): temporal subgraph contrast — BFS subgraphs ranked by recency + degree, a time-weighted and an unweighted subgraph encoder, margin losses against a shuffled and an unweighted summary; encoder scored by the same frozen link probe |
| `dyg2vec` | DyG2Vec (Alomrani et al., TMLR'24): window-based temporal attention encoder (its own encoder kind, like `maskdgnn`) pre-trained with VICReg on two distorted views (edge dropout + structural-feature masking) of each history window; encoder probed on frozen latents, which is the paper's own SSL-init protocol |
| `jepa_frozen` | SSL encoder frozen, JEPA predictor on top. Collapse impossible by construction. |
| `jepa_naive` | `--train_encoder` alone — targets come from the encoder being optimised, so collapse is a global optimum. Separates "EMA helped" from "unfreezing helped". |
| `jepa_ema` | `--train_encoder --ema_target` (I-JEPA) |
| `jepa_vicreg` | `--train_encoder` + VICReg variance/covariance |

`random` and `supervised` are free — `analyze_jepa.py` computes them as link
sources, so one `ARMS=controls` pass produces both.

## Two writers, one schema

| producer | file | who |
|---|---|---|
| `train_dyg_jepa.py --downstream_task link_pred` | `test_metrics.json` | the four JEPA arms |
| `analyze_jepa.py --link_sources ...` | `analysis.json` (`["link_pred"][source]`) | baselines + controls |
| `train_maskdgnn.py --e2e_eval test` | `link_e2e.json` | the MaskDGNN end-to-end row |

Both are `{"all": {horizon: {...}}, "new": {...}}`, and both aggregators read
either. With `--probe_split val` the analyze path writes `analysis_val.json`,
which is what keeps the sweep off the test split.

## Running it

```bash
# 0. one-time, from the LOGIN node (each source keeps its raw file elsewhere;
#    the launcher checks the right one and prints the fetch command)
wget http://snap.stanford.edu/jodie/mooc.csv -O dataset/jodie/mooc.csv
wget https://snap.stanford.edu/data/soc-redditHyperlinks-body.tsv \
    -O dataset/reddit/soc-redditHyperlinks-body.tsv
# dblp: see scripts/dblp/preprocess_dblp_xml.py or fetch_preprocessed_dblp.sh

# 1. predictor knobs, on VAL
DATASET=mooc STAGE=predictor sbatch --array=0-35%6 scripts/linkpred/slurm_hparam_search.sh
python scripts/linkpred/aggregate_hparam.py \
    --root outputs/lp_hparam/mooc_predictor --emit_flags        # -> BASE_HP="..."

# 1b. baselines, so the table is not "our tuned method vs their defaults"
DATASET=mooc STAGE=baselines sbatch --array=0-53%6 scripts/linkpred/slurm_hparam_search.sh
python scripts/linkpred/aggregate_hparam.py \
    --root outputs/lp_hparam/mooc_baselines --by_arm --emit_flags

# 2. per-variant encoder knobs, BASE_HP fixed
DATASET=mooc STAGE=variants BASE_HP="<step 1>" \
    sbatch --array=0-15%4 scripts/linkpred/slurm_hparam_search.sh
python scripts/linkpred/aggregate_hparam.py \
    --root outputs/lp_hparam/mooc_variants --by_arm --emit_flags

# 3. final table: 3 seeds, probing TEST
for S in 42 43 44; do
  DATASET=mooc SEED=$S BASE_HP="<step 1>" \
    VARIANT_HP_ema="<step 2>" VARIANT_HP_naive="<step 2>" VARIANT_HP_vicreg="<step 2>" \
    BASELINE_HP_cldg="<step 1b>" BASELINE_HP_ddgcl="<step 1b>" BASELINE_HP_idol="<step 1b>" \
    BASELINE_HP_maskdgnn="<step 1b>" BASELINE_HP_dysubc="<step 1b>" BASELINE_HP_grace="<step 1b>" BASELINE_HP_graphmae="<step 1b>" BASELINE_HP_dyg2vec="<step 1b>" \
    sbatch scripts/linkpred/slurm_all_methods.sh
done

# 4. mean ± std LaTeX, both subsets
python scripts/linkpred/aggregate_linkpred.py --dataset mooc --seeds 42 43 44 --latex
```

Swap `DATASET=` for any of `mooc reddit wikipedia reddit_body dblp`.
Use `ARMS=jepa_ema` for one arm per job.

The combined table takes them all at once:

```bash
python scripts/linkpred/aggregate_linkpred.py \
    --datasets reddit_body reddit dblp --seeds 42 43 44 \
    --exclude Random Supervised
```

## Transfer table: link prediction from the classification checkpoints

The representation is selected **once** (recipe and hyper-parameters on the
node-state val split) and then probed frozen on every downstream task, with no
per-task retuning. That is the transferability claim, so the link table for
the state datasets comes from the classification trees, not from a second
sweep. Point the launcher at that ROOT and match its feature set
(`TASK=state_probe` trains **with** edge features; `TASK=link_pred` alone
would build a different input dim and the checkpoints would not load):

```bash
for DS in mooc reddit wikipedia; do
  for S in 42 43 44; do
    DATASET=$DS SEED=$S ROOT=outputs/cls_$DS EDGE_FEATS=1 \
        sbatch scripts/linkpred/slurm_all_methods.sh
  done
done
python scripts/linkpred/aggregate_linkpred.py \
    --datasets mooc reddit wikipedia --root_pattern outputs/cls_{ds} \
    --seeds 42 43 44 --exclude Random Supervised
```

Nothing is retrained. `train_dyg_jepa.py` defaults `--downstream_task` to
`link_pred`, so every JEPA arm's classification run already wrote the link
`test_metrics.json` (all/new, test split) and the launcher skips it as
complete. Every baseline's `encoder.ckpt` is found, so only the frozen link
probe runs and writes `analysis.json` next to it; the controls reuse the
shared encoder. Roughly a probe pass per arm per seed.

The one row this cannot produce is **MaskDGNN (end-to-end)**: its trainer
saves the encoder but not the edge head, and the classification run used
`--e2e_eval none`. Either drop the row or retrain that arm alone
(`rm outputs/cls_<ds>/maskdgnn_s<seed>/encoder.ckpt`, then
`ARMS=maskdgnn` with the same ROOT / EDGE_FEATS).

Caption for the paper: hyper-parameters selected on node-state validation,
identical checkpoints for both tasks. A baseline that transfers badly under
its state-selected config is a result, not something to retune.

## Checks that matter

- **Compare against `new`, not just `all`.** A method can look strong on `all`
  by reproducing the current graph. `new` is where the predictor has to earn it.
- **`random` is a floor, not a competitor** — the aggregator excludes it from
  bolding for that reason.
- **Collapse check** runs after every unfrozen arm.
- **Spread on the sweep.** `aggregate_hparam.py` prints the range across cells.
  If the whole grid spans less than the between-seed noise, the sweep found
  nothing and the "winner" is arbitrary.
