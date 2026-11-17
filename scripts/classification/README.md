# Dynamic node-classification pipeline (JODIE MOOC / Reddit / Wikipedia / synthetic state)

The classification twin of [`scripts/regression/`](../regression/). Same
structure: an `ARMS`-gated launcher, a three-stage hyperparameter sweep that
only ever reads the **val** split, and a mean ± std LaTeX aggregator.

## Task

Per-user binary `state_label` (MOOC dropout / Reddit-Wikipedia ban). Roughly a
**3% positive rate**, so the headline metrics are AUROC and AUPRC — never
accuracy. `STATE_MODE` picks the label semantics: `state` (dynamic cumulative,
the default), `ever` (static, legacy), `event` (instantaneous, very sparse).

Subset, binning, edge features, encoder and window geometry all come from
[`_dataset_config.sh`](../dblp/_dataset_config.sh) with `TASK=state_probe` — the
same file the forecasting pipeline sources, so the two tables describe
consistent experiments.

## Arms

| row | what it is |
|---|---|
| `grace` | GRACE (Zhu et al., 2020): **static** graph contrast — two augmented views (edge removal + feature-dimension masking) of one snapshot, inter- and intra-view negatives. Never sees two snapshots at once, so it is the control for "does the SSL objective need to be temporal at all?"; probed on frozen latents at the last history bin (static SSL, last frame) |
| `graphmae` | GraphMAE (Hou et al., KDD'22): **static** masked-feature autoencoder — mask a fraction of one snapshot's nodes with a learnable token, re-mask their latents, GCN decoder, scaled cosine error on the masked nodes. The generative half of the static control block (GRACE is the contrastive half); probed on frozen latents at the last history bin (static SSL, last frame) |
| `bgrl` | BGRL (Thakoor et al., ICLR'22): **static** bootstrapped latents — two augmented views of one snapshot; an online encoder + predictor regress an EMA target encoder's embedding of the other view, no negatives. The negative-free member of the static control block and the closest static relative of `jepa_ema`: same bootstrap, but the target is the same snapshot re-augmented, not a future one; online encoder probed on frozen latents at the last history bin |
| `cldg` / `ddgcl` / `idol` | contrastive SSL encoder, probed on frozen latents |
| `maskdgnn` | MaskDGNN (He et al., IJCAI'25): activeness-aware edge masking + masked/next-edge BCE on a GCN with an adaptive frequency layer; uses no state label, probed on frozen latents like the three above |
| `dysubc` | DySubC (Chen et al., TKDD'23): temporal subgraph contrast — BFS subgraphs ranked by recency + degree, a time-weighted and an unweighted subgraph encoder, margin losses; uses no state label, probed on frozen latents |
| `dyg2vec` | DyG2Vec (Alomrani et al., TMLR'24): window-based temporal attention encoder (its own encoder kind, like `maskdgnn`) pre-trained with VICReg on two distorted views (edge dropout + structural-feature masking) of each history window; encoder probed on frozen latents, which is the paper's own SSL-init protocol |
| `dygwm` | DyG-WM (Li et al., 2026, arXiv 2610.19364): **SSL-only core** of the dynamic graph world model — its context encoder (own encoder kind: shared GCN + node / neighbourhood trajectory GRUs + gated fusion) and path signatures of hand-crafted node / pair descriptors predict an EMA target encoder's **next-bin** node and relation latents (normalised squared distance + var / cov regularisers). The paper's supervised link + ranking losses are not used; encoder probed on frozen latents, which is the paper's own node-classification protocol |
| `sgjepa` | SG-JEPA (Zhang et al., 2026, arXiv 2607.18412): the shared GCN's latents on the window's earlier snapshots drive spiking (PLIF) layers whose spike counts predict, from every prefix length (Matryoshka-style), the GCN's stop-gradient latent of the window's **last** snapshot under a per-node InfoNCE; encoder probed on frozen latents |
| `jepa_frozen` | SSL encoder frozen, JEPA predictor on top. Collapse impossible by construction; the reference variant. |
| `jepa_naive` | `--train_encoder` alone. Targets come from the encoder being optimised, so collapse is a global optimum. Exists so EMA's win is attributable to EMA and not merely to unfreezing. |
| `jepa_ema` | `--train_encoder --ema_target` (I-JEPA) |
| `jepa_vicreg` | `--train_encoder` + VICReg variance/covariance |
| `jepa_ctx` | frozen + `--mask_encoder_input`: the online encoder is blinded to the masked nodes (I-JEPA's context-encoder rule), so masked-history cannot be read off neighbours' latents |
| `jepa_tnorm` | frozen + `--target_norm`: LayerNorm on target latents (I-JEPA / V-JEPA) |
| `jepa_ema_cos` | `jepa_ema` with `--ema_schedule cosine` (momentum annealed to 1.0 over `EMA_ANNEAL_EPOCHS`, default 50 -- a whole-run horizon is never reached under early stopping) |
| `jepa_ema_tnorm` / `jepa_ema_cos_tnorm` | the two EMA arms + `--target_norm` (target LayerNorm) without context masking -- isolates tnorm from ctx, which `jepa_ijepa` bundles |
| `jepa_sigreg` | `--train_encoder --sigreg`: stop-grad online target + LeJEPA's SIGReg instead of VICReg |
| `jepa_lejepa` | `--train_encoder --target_grad --sigreg`: LeJEPA proper, no stop-grad, no EMA |
| `jepa_ijepa` | `jepa_ema_cos` + ctx-mask + target-LN: the full I-JEPA recipe |
| `jepa_lejepa_full` | `jepa_lejepa` + ctx-mask + target-LN |

The first five arms are unchanged (every new flag defaults off), so old
results reproduce; the seven below them are additive. `ARMS=all` still means
the legacy table; `ARMS=all_jepa` adds the new arms and `ARMS=new_jepa` runs
only them.

**There is no `jepa_enc` arm, and that is deliberate.** `run_block_probe.py`
reports several columns inside *every* `state_probe.json`:

| column | meaning |
|---|---|
| `jepa` | the JEPA predictor's rolled-out latent — the forecast |
| `encoder_last` | frozen encoder at the last history bin — copy-last |
| `encoder_oracle` | an encoder that SAW the target bin — upper bound |
| `degree_history_baseline` | degree at each history bin — activity-trend control |

So the encoder-without-predictor control is measured *inside the same run, on
the same rows, from the same encoder* — strictly tighter than the separate arm
the forecasting pipeline needs. Baselines are encoder-only and have no `jepa`
column: their `encoder_last` **is** their prediction.

## The paired statistic

`gap = jepa − encoder_last`, differenced **within each seed**. Both columns come
from one encoder per seed, so the difference cancels the encoder variance that
dominates either column alone. In practice the raw AUROC std is several times
the gap std, which is what lets `frozen` / `naive` / `EMA` / `VICReg` actually
separate — the forecasting table's four variants never did, sitting inside one
standard deviation of each other.

`aggregate_hparam.py` therefore ranks sweep cells on **gap** by default
(`--rank_by auroc` overrides): ranking on raw AUROC mostly ranks encoders, while
these stages are tuning predictors.

## Running it

```bash
# 0. one-time, from the LOGIN node (compute nodes have no internet)
wget http://snap.stanford.edu/jodie/mooc.csv -O dataset/jodie/mooc.csv

# 1. predictor knobs, on VAL
DATASET=mooc STAGE=predictor bash scripts/classification/slurm_hparam_search.sh --list
DATASET=mooc STAGE=predictor sbatch --array=0-35%6 scripts/classification/slurm_hparam_search.sh
python scripts/classification/aggregate_hparam.py \
    --root outputs/cls_hparam/mooc_predictor --emit_flags        # -> BASE_HP="..."

# 1b. tune the BASELINES too, or the table is "our tuned method vs their defaults"
DATASET=mooc STAGE=baselines sbatch --array=0-53%6 scripts/classification/slurm_hparam_search.sh
python scripts/classification/aggregate_hparam.py \
    --root outputs/cls_hparam/mooc_baselines --by_arm --emit_flags

# 2. per-variant encoder knobs, BASE_HP fixed. VARIANT_SET=all (default) is the
#    legacy 16 cells (indices 0-15, unchanged) followed by the 32 new-arm cells;
#    VARIANT_SET=old / new restrict to one half. `--list` prints the grid.
DATASET=mooc STAGE=variants BASE_HP="<step 1>" \
    sbatch --array=0-47%4 scripts/classification/slurm_hparam_search.sh
python scripts/classification/aggregate_hparam.py \
    --root outputs/cls_hparam/mooc_variants --by_arm --emit_flags   # -> VARIANT_HP_<arm> for every arm

# 3. final table: 3 seeds, probing TEST
for S in 42 43 44; do
  DATASET=mooc SEED=$S BASE_HP="<step 1>" \
    VARIANT_HP_ema="<step 2>" VARIANT_HP_naive="<step 2>" VARIANT_HP_vicreg="<step 2>" \
    VARIANT_HP_ema_cos="<step 2>" VARIANT_HP_sigreg="<step 2>" VARIANT_HP_lejepa="<step 2>" \
    VARIANT_HP_ijepa="<step 2>" VARIANT_HP_lejepa_full="<step 2>" ARMS=all_jepa \
    BASELINE_HP_cldg="<step 1b>" BASELINE_HP_ddgcl="<step 1b>" BASELINE_HP_idol="<step 1b>" \
    BASELINE_HP_maskdgnn="<step 1b>" BASELINE_HP_dysubc="<step 1b>" BASELINE_HP_grace="<step 1b>" BASELINE_HP_graphmae="<step 1b>" BASELINE_HP_bgrl="<step 1b>" BASELINE_HP_dyg2vec="<step 1b>" BASELINE_HP_dygwm="<step 1b>" BASELINE_HP_sgjepa="<step 1b>" \
    sbatch scripts/classification/slurm_all_methods.sh
done

# 4. mean ± std LaTeX
python scripts/classification/aggregate_classification.py \
    --dataset mooc --seeds 42 43 44 --latex
```

Swap `DATASET=reddit` or `wikipedia` for the others. Use `ARMS=jepa_ema` to put
one arm in its own job if four in one exceeds the walltime.

## Synthetic node-state task (`DATASET=synthetic_dyn`)

The controlled version of the same question, on the latent-driven generator
([`dynamic_graph_generator.py`](../../dynamic_graph_generator.py)). Every
observable is a readout of one per-node latent `z`, and with
`--synth_label_mode nearest_anchor` the **community label is too**: a node's
state is the anchor its latent is currently closest to, and a switch is a
boundary crossing. The state at `t+1` is then knowable from the trajectory
(through the velocity), which is exactly what a temporal-invariance objective
erases and a predictive latent keeps.

Differences from the JODIE runs:

* The label is **dense multiclass** (`--synth_state_target community`, K=4 at
  every `(step, node)`), so the probe is `--task multiclass` and the metrics are
  accuracy / macro-F1, not AUROC. The launcher switches this automatically.
* `state_probe.json` gains a `label_persistence` column (copy the previous
  label: the floor, ~95% overall) and two subsets on every column:
  `subset_switch` (rows whose label changes — persistence scores **0** there by
  construction, so this is the pure anticipation measure) and `subset_stay`
  (the false-alarm side). Report the switch rows next to the overall number;
  overall accuracy alone is persistence-dominated.
* The loader prints a `state task` line on the test split: switch rate, the
  persistence and oracle accuracies, and the oracle's accuracy on switch rows.
  The oracle forecasts the TRUE latent by ridge and decodes it the way the
  generator does, so it is the ceiling for any method. At the defaults in
  `_dataset_config.sh` (anchor spread 1.0, latent noise 0.35, momentum 0.8) it
  reads ~5% switches, ~16% error reduction over persistence, ~97% of switches
  anticipated. Below ~10% treat the configuration as unable to separate methods.
* Two more columns exist only here. `change_f1` (with precision and recall)
  scores "predicted state ≠ current state" as a change alarm against the true
  change, so anticipation and false alarms land in one number; note that a
  misread current state also fires the alarm, so precision is capped by how
  readable the state is (~86%) and only the comparison across methods is
  meaningful. `latent_r2_test` / `delta_r2_test` fit a ridge from each
  column's representation to the generator's TRUE latent at the target step,
  and to its change since the last observed step. The change is the sharper
  one: position alone predicts the mean-reversion part, the momentum part
  needs velocity, which only a forecast can carry. Both are dense over all
  rows, so they do not suffer the 5%-of-rows statistics problem.
* A `jepa_input` arm (`--predict_space input`) trains the same frozen encoder,
  predictor and masking to reconstruct the target step's raw features instead
  of matching the latent. It isolates the prediction space, the JEPA objective
  itself, from the predictor architecture.
* Knobs are `SYN_*` env vars (`SYN_ANCHOR`, `SYN_OU_SIGMA`, `SYN_MOMENTUM`,
  `SYN_HYST`, `SYN_K`, ...). Closer anchors and a noisier latent raise the switch
  rate; momentum and hysteresis do not raise the ceiling (hysteresis only makes
  labels stickier). `NPRED>1` adds a per-horizon block `subset_h0..h{NPRED-1}`
  to every column (k = 0 is one step past the last observed bin).
* **Multi-horizon runs (`NPRED>1`) are the horizon showcase.** Each column
  gets a head **fitted per horizon** (`run_block_probe.py::_fit_per_horizon`),
  not one pooled head scored per horizon: a pooled head is fed the same
  copy-last latent at every horizon and cannot tell which one it is
  answering, which handicaps the encoder-only baselines. Per horizon and
  column you get `acc_test`, `macro_f1_test`, `switch_acc_test`,
  `stay_acc_test`, the change alarm and `latent_r2_test` / `delta_r2_test`.
  Label persistence, the switch / stay subsets and the change alarm are all
  defined against the label at the **last observed** bin (`abs_bin - 1 - h`),
  so at every horizon the floor is what a forecaster could actually see. The
  aggregator reads them as `subset_h<k>.<metric>`. Run the whole thing with
  [`scripts/synthetic/horizon_showcase.sh`](../synthetic/horizon_showcase.sh)
  (5 seeds, NPRED=12, every baseline; `--aggregate` / `--plot` for the table
  and `plot_horizon.py` figures).
* The legacy hazard label (`--synth_label_mode hazard`, the regression sweep's
  setting) is unusable here: the destination of a switch is a uniform draw, so
  the oracle lands exactly on persistence and the loader warns.

```bash
for S in 42 43 44; do DATASET=synthetic_dyn SEED=$S sbatch scripts/classification/slurm_all_methods.sh; done
python scripts/classification/aggregate_classification.py --dataset synthetic_dyn \
    --seeds 42 43 44 --metrics acc_test subset_switch.acc_test subset_stay.acc_test --latex
```

## Static node classification (`DATASET=cldg_dblp` / `cldg_patent`)

The two CLDG-release graphs (Xu et al., ICDE'23) carry **one class per node
for the whole timeline**: 10 research areas on DBLP, 6 categories on the
patent citation graph. Files come from the Google-Drive zips in `dataset/`
(`bash scripts/dblp/extract_cldg_datasets.sh` -> `dataset/cldg/<name>/`);
the loader is [`src/custom_codes/cldg_format_dataset.py`](../../src/custom_codes/cldg_format_dataset.py).

| | nodes (kept) | edges | snapshots | classes |
|---|---|---|---|---|
| `cldg_dblp` | 28,085 (8,000) | 236,894 | 27 native years | 10 |
| `cldg_patent` | 12,214 (8,000) | 41,916 | 30 equal-frequency bins | 6 |

The release's third graph, arXivAI (69,854 nodes, 699k edges), was tried and
dropped: its timestamps are years and 99% of the edges fall in the last six
of 27, so the chronological train period holds 4,660 edges. Not a temporal
benchmark.

Choices, all in [`_dataset_config.sh`](../dblp/_dataset_config.sh) and
env-overridable (`CLDG_BINS`, `CLDG_BINNING`, `CLDG_TOPK`, `CLDG_FEATS`):

* **Snapshots.** DBLP's timestamps are 27 distinct values (the years), so
  `--cldg_num_bins 0` uses them as they are. Patent's axis is so skewed
  (83% of the edges in the last tenth) that equal-width bins leave most
  snapshots empty; 30 quantile bins hold ~1,400 edges each.
* **Top-K = 8,000.** The pipeline stores dense `(T, N, N)` snapshots, so
  28k-node DBLP does not fit (21 GB per timeline, 12 GB per window). The
  8,000 most active nodes keep 48% of DBLP's edges (min degree 16) and 90%
  of patent's (only degree-1 leaves go). Labels follow the subset; the
  snapshot grid is computed on the full stream, so it does not depend on K.
  `BATCH` defaults to 4 here (the normalised adjacency is `B x 4 x N^2`).
* **Features.** `feature.txt` is the sinusoidal encoding of the node index,
  the input CLDG itself trains on. It is an identity code, no content, and it
  is what lets a frozen GCN latent say *which* neighbours a node has; without
  it the 8 graph statistics carry little about a research area. On by
  default (136-dim input); `CLDG_FEATS=0` is the topology-only ablation.
* **Probe.** `--task multiclass --label_source dataset`: the same columns as
  the state probes (`jepa`, `encoder_last`, `encoder_oracle`,
  `degree_history_baseline`) written to `state_probe.json`, but the
  train/test split is over **nodes** (30% held out, seed 0). A static label
  repeats in every window a node appears in, so a window split would let the
  head memorise it. There is no persistence floor and no switch/stay subset
  (a static label never changes). Metrics: accuracy and macro-F1 on the
  held-out nodes; the paired `jepa - encoder_last` gap reads as before.
* **Not wired:** `slurm_finetune.sh` (needs a dynamic binary label) and the
  static-label equivalents of the horizon curves.

```bash
for S in 42 43 44; do DATASET=cldg_dblp SEED=$S sbatch scripts/classification/slurm_all_methods.sh; done
for S in 42 43 44; do DATASET=patent    SEED=$S sbatch scripts/classification/slurm_all_methods.sh; done
python scripts/classification/aggregate_classification.py --dataset cldg_dblp \
    --seeds 42 43 44 --metrics acc_test macro_f1_test --latex
# hyperparameter search ranks on the val probe as usual
DATASET=cldg_patent STAGE=predictor sbatch --array=0-35%6 scripts/classification/slurm_hparam_search.sh
python scripts/classification/aggregate_hparam.py --root outputs/cls_hparam/cldg_patent_predictor --metric acc_test
```

Both are also accepted by the link-prediction pipeline
(`DATASET=cldg_dblp sbatch scripts/linkpred/slurm_all_methods.sh`), where the
labels are simply unused.

## Dynamic node-label ranking (`DATASET=tgbn_trade`)

TGB's tgbn-trade (Huang et al., NeurIPS'23): 255 nations, 31 yearly
snapshots (1986-2016), and a label that is **genuinely time-varying**: for
nation *u* in year *t* the target is *u*'s vector of export shares to every
nation in year *t* (sums to one). Fetch on a login node with
`bash scripts/dblp/fetch_tgbn_trade.sh` (-> `dataset/tgb/tgbn-trade/`);
loader [`src/custom_codes/tgbn_dataset.py`](../../src/custom_codes/tgbn_dataset.py).
The whole graph fits, so there are no binning or subset knobs; `TGBN_FEATS=0`
drops the 5 per-(year, nation) trade-weight descriptors that carry the edge
weights the bool snapshot loses.

The probe is `--task ranking`: a linear head from the frozen latent to 255
scores, trained with the soft-target cross-entropy against the share vector
on the train windows' rows and scored with **NDCG@10** (TGB's metric,
`analyze_jepa.ndcg_at_k`, identical to `sklearn.metrics.ndcg_score`) on the
test windows' rows. Same columns and file as the state probes, plus
`label_persistence` = the last observed share vector used as the scores.
That is TGB's persistent-forecast baseline and it is **very strong here**
(0.855 on our test years, the same value the TGB paper reports); every
learned model in the TGB leaderboard sits far below it (~0.37-0.39). So read
this table the same way as MOOC: the claim is the paired `jepa - encoder_last`
gap and the ranking against the SSL baselines, not beating persistence.
`NPRED>1` gives per-horizon `subset_h<k>` entries for every column and for
the floor.

```bash
for S in 42 43 44; do DATASET=tgbn_trade SEED=$S sbatch scripts/classification/slurm_all_methods.sh; done
python scripts/classification/aggregate_classification.py --dataset tgbn_trade \
    --seeds 42 43 44 --metrics ndcg10_test --latex
DATASET=tgbn_trade STAGE=predictor sbatch --array=0-35%6 scripts/classification/slurm_hparam_search.sh
python scripts/classification/aggregate_hparam.py --root outputs/cls_hparam/tgbn_trade_predictor --metric ndcg10_test
```

Pass `--metric ndcg10_test` to `aggregate_hparam.py` and
`select_ckpt_by_probe.py` too (their default is AUROC). `slurm_finetune.sh`
is not wired for this task.

## External dynamic-label benchmarks (`DATASET=chase_dblp_s` / `chase_gossipcop` / `chase_music`)

The three snapshot graphs of CHASE (Kim et al., "Chasing Label Shifters",
NeurIPS'26, arXiv 2610.17582), the only public dynamic graphs whose node
labels **change per snapshot by construction**. Released on the Hugging Face
Hub (`jhkim611/CHASE-datasets`, CC BY-NC 4.0; `bash scripts/dblp/fetch_chase.sh`
-> `dataset/chase/`); loader
[`src/custom_codes/chase_dataset.py`](../../src/custom_codes/chase_dataset.py).

| | nodes (kept) | snapshots (train/val/test) | classes | label | persistence floor |
|---|---|---|---|---|---|
| `chase_dblp_s` | 34,717 (8,000) | 14 yearly (9/2/3) | 2 | main research area that year | 0.974 |
| `chase_gossipcop` | 6,091 | 37 weekly (26/5/6) | 2 | >50% of the articles shared that week are fake | 0.965 |
| `chase_music` | 878 | 24 monthly (15/4/5) | 5 | dominant genre listened to that month | 0.812 |

(persistence floor = share of nodes active at t-1 and t whose label did not
change; the loader prints it, overall and on the test snapshots, as its
label audit.) Node sets grow over time; a node has no label before its
first snapshot. Per-snapshot node features ship with the release (768-dim
BERT embeddings of that period's papers / articles, 779-dim listening
descriptors) and are genuinely time-varying.

Choices, all in [`_dataset_config.sh`](../dblp/_dataset_config.sh) and
env-overridable (`CHASE_TOPK`, `CHASE_FEATS`, `CHASE_NORM`, `CHASE_PCA`):

* **Splits are the paper's snapshot counts**, not the generic 70/10/20, so
  the test period is theirs. Windows are target-based as everywhere else.
* **Top-K = 8,000 on DBLP-S only.** 34,717 dense nodes do not fit the
  `(T, N, N)` pipeline; the K most active nodes by summed degree are kept,
  labels and features follow. GossipCop and Music run in full. `BATCH`
  defaults to 4 on DBLP-S.
* **Features.** 8 graph statistics + the release's vectors, z-scored on the
  train snapshots' active rows (`CHASE_NORM=0` skips it), optionally reduced
  with a train-fitted PCA (`CHASE_PCA=64`). `CHASE_FEATS=0` is the
  topology-only ablation.
* **Probe.** DBLP-S and GossipCop are binary: the MOOC probe unchanged
  (AUROC / AUPRC, onset subset). Music is a 5-class dynamic state: the
  launchers switch to `--task multiclass`, which takes the dynamic
  multiclass path of the synthetic task (accuracy / macro-F1, label
  persistence floor, `subset_switch` / `subset_stay`). The release's own
  numbers (supervised GNNs with per-snapshot training) are not comparable
  to a frozen probe; the comparison here is between SSL methods and
  against persistence.

```bash
bash scripts/dblp/fetch_chase.sh                     # login node, ~3.9 GB
for S in 42 43 44; do DATASET=chase_music     SEED=$S sbatch scripts/classification/slurm_all_methods.sh; done
for S in 42 43 44; do DATASET=chase_gossipcop SEED=$S sbatch scripts/classification/slurm_all_methods.sh; done
for S in 42 43 44; do DATASET=chase_dblp_s    SEED=$S sbatch scripts/classification/slurm_all_methods.sh; done
python scripts/classification/aggregate_classification.py --dataset chase_music \
    --seeds 42 43 44 --metrics acc_test macro_f1_test subset_switch.acc_test --latex
python scripts/classification/aggregate_classification.py --dataset chase_gossipcop --seeds 42 43 44 --latex
# hyperparameter search ranks on the val probe as usual (music: --metric acc_test)
DATASET=chase_music STAGE=baselines sbatch --array=0-80%6 scripts/classification/slurm_hparam_search.sh
```

## Checks that matter

- **`N/M cells have the forecast BEATING copy-last`** in the sweep output. If
  that is 0, the predictor adds nothing over the frozen encoder's last-bin
  latent and the ranking is noise.
- **The degree-history control.** A method that does not clear it is showing
  "this user is winding down", not learned dynamics. `aggregate_hparam.py`
  prints a `[WARN]` when the top cell fails this.
- **Collapse check** runs after every unfrozen arm. A collapsed encoder puts
  every node's latent at the same point, and AUROC sits at chance however good
  the training loss looked.
- **Never report accuracy.** At ~3% positives it is ~97% for a constant
  predictor.

## Not included

`JODIE` and a supervised end-to-end arm. JODIE needs no new code — its trainer
exists and it is already in `aggregate_mooc_state.py`'s layout — and it is the
one baseline with a forecast column of its own, so it would compare against the
JEPA predictor rather than encoder-only. A supervised arm would need new code,
since `run_block_probe.py` only probes frozen encoders.

## Fine-tuning / label efficiency (`slurm_finetune.sh`)

The DDGCL-style protocol (Tian et al., CIKM'21): the self-supervised weights
are only an **initialisation** and the whole network -- encoder, predictor and
an MLP head on the predictor's forecast latent -- is then trained end-to-end on
the state label. It answers the question the frozen probe cannot: does the
pretraining transfer into a *supervised* model? Its natural form is a
label-efficiency curve, so every cell is one (init, regime, label fraction).

| init | what starts trained |
|---|---|
| `random` | nothing (the supervised floor) |
| `encoder` | the table's shared edge-mask encoder; predictor from scratch |
| `jepa` | encoder **and** predictor from `$ROOT/$JEPA_ARM_s$SEED/best.ckpt` |
| `ddgcl` / `cldg` / `idol` / `maskdgnn` / `dysubc` / `grace` / `graphmae` / `bgrl` / `dyg2vec` / `dygwm` / `sgjepa` | that baseline's `encoder.ckpt`; predictor from scratch |

`encoder` vs `jepa` is the row pair that isolates the JEPA predictor from
encoder pretraining. Every row uses the predictor **architecture** of the JEPA
checkpoint (read from it under every init), the same MLP head, the dataset's
chronological split (fit on train windows, select on val AUROC, report test)
and the same schedule; only the starting weights differ.

Two regimes (`AUXW`): `0` is pure fine-tuning (DDGCL `fune_train`, BCE only,
the supervised loss may overwrite the forecast); `w>0` adds `w x` the JEPA
latent loss with an EMA target (DDGCL's multi-task scheme, keeps the
predictor honest as a forecaster). `--label_fraction f` keeps a stratified
`f` of the train-period positive users and `f` of the negative ones
(`--label_seed`, shared across inits so every row sees the same users); val
and test always use all rows. Rows are (target step, active user) with a
valid label, exactly the frozen probe's definition, so `onset` (previous
label 0) is reported too.

```bash
# needs the final-table artefacts for the seed (shared encoder, jepa_frozen, ddgcl)
DATASET=mooc ROOT=outputs/cls_mooc_full SEED=42 \
    bash scripts/classification/slurm_finetune.sh --list          # 40 cells
for S in 42 43 44; do
  DATASET=mooc ROOT=outputs/cls_mooc_full SEED=$S \
    sbatch --array=0-39%6 scripts/classification/slurm_finetune.sh
done
python scripts/classification/aggregate_finetune.py \
    --root outputs/cls_mooc_full/finetune --seeds 42 43 44 [--metric onset_auroc] [--latex]
```

Read the table column-wise at the **low** fractions: that is where
pretraining shows (DyG2Vec Fig. 4). At 100% the rows usually merge, which is
fine -- the aggregator also prints the smallest fraction at which each row
matches random-init at 100%, the headline number if pretraining helps. This
table does not replace the frozen one: fine-tuned numbers mix representation
quality with supervised capacity, and the paired `jepa - encoder_last` gap
only exists with the encoder frozen.
