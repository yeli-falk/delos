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

**`TASK=link_pred` uses the full dataset** (TOPU=TOPI=0 keeps every node, the
default for all JODIE datasets). The probe scores 99 negatives per positive per
window, so cost grows ~N²; lower `BATCH` if you OOM. The node set comes from
`_dataset_config.sh` and is shared by every arm — do not change it for one method.

## Arms

| row | what it is |
|---|---|
| `random` | U(0,1) edge scores, no training. The floor. |
| `supervised` | encoder + link head trained end-to-end from random init, no SSL |
| `cldg` / `ddgcl` / `idol` | contrastive SSL encoder, scored by the same link probe on frozen latents |
| `maskdgnn` | MaskDGNN (He et al., IJCAI'25): activeness-aware edge masking + masked/next-edge BCE on a GCN with an adaptive frequency layer. **Two rows**: its encoder through the same frozen link probe (`MaskDGNN`), and its own encoder + edge head scored end-to-end by the same evaluator (`MaskDGNN (end-to-end)` — the paper's protocol, read next to `supervised`) |
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
DATASET=mooc STAGE=baselines sbatch --array=0-20%6 scripts/linkpred/slurm_hparam_search.sh
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
    BASELINE_HP_maskdgnn="<step 1b>" \
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

## Checks that matter

- **Compare against `new`, not just `all`.** A method can look strong on `all`
  by reproducing the current graph. `new` is where the predictor has to earn it.
- **`random` is a floor, not a competitor** — the aggregator excludes it from
  bolding for that reason.
- **Collapse check** runs after every unfrozen arm.
- **Spread on the sweep.** `aggregate_hparam.py` prints the range across cells.
  If the whole grid spans less than the between-seed noise, the sweep found
  nothing and the "winner" is arbitrary.
