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
| `cldg` / `ddgcl` / `idol` | contrastive SSL encoder, probed on frozen latents |
| `maskdgnn` | MaskDGNN (He et al., IJCAI'25): activeness-aware edge masking + masked/next-edge BCE on a GCN with an adaptive frequency layer; uses no state label, probed on frozen latents like the three above |
| `dysubc` | DySubC (Chen et al., TKDD'23): temporal subgraph contrast — BFS subgraphs ranked by recency + degree, a time-weighted and an unweighted subgraph encoder, margin losses; uses no state label, probed on frozen latents |
| `jepa_frozen` | SSL encoder frozen, JEPA predictor on top. Collapse impossible by construction; the reference variant. |
| `jepa_naive` | `--train_encoder` alone. Targets come from the encoder being optimised, so collapse is a global optimum. Exists so EMA's win is attributable to EMA and not merely to unfreezing. |
| `jepa_ema` | `--train_encoder --ema_target` (I-JEPA) |
| `jepa_vicreg` | `--train_encoder` + VICReg variance/covariance |
| `jepa_ctx` | frozen + `--mask_encoder_input`: the online encoder is blinded to the masked nodes (I-JEPA's context-encoder rule), so masked-history cannot be read off neighbours' latents |
| `jepa_tnorm` | frozen + `--target_norm`: LayerNorm on target latents (I-JEPA / V-JEPA) |
| `jepa_ema_cos` | `jepa_ema` with `--ema_schedule cosine` (momentum annealed to 1.0) |
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
DATASET=mooc STAGE=baselines sbatch --array=0-38%6 scripts/classification/slurm_hparam_search.sh
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
    BASELINE_HP_maskdgnn="<step 1b>" BASELINE_HP_dysubc="<step 1b>" \
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
  labels stickier). `NPRED=3` adds per-horizon subsets `h0..h2`.
* The legacy hazard label (`--synth_label_mode hazard`, the regression sweep's
  setting) is unusable here: the destination of a switch is a uniform draw, so
  the oracle lands exactly on persistence and the loader warns.

```bash
for S in 42 43 44; do DATASET=synthetic_dyn SEED=$S sbatch scripts/classification/slurm_all_methods.sh; done
python scripts/classification/aggregate_classification.py --dataset synthetic_dyn \
    --seeds 42 43 44 --metrics acc_test subset_switch.acc_test subset_stay.acc_test --latex
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
| `ddgcl` / `cldg` / `idol` / `maskdgnn` / `dysubc` | that baseline's `encoder.ckpt`; predictor from scratch |

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
