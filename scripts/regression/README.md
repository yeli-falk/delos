# Node-signal forecasting pipeline (METR-LA + England COVID)

One harness, two datasets, every method. Each arm writes `test_metrics.json` in
the same schema, so `aggregate_regression.py` merges them into a single
mean ± std LaTeX table.

## Window geometry

Set in [`scripts/dblp/_dataset_config.sh`](../dblp/_dataset_config.sh), following
each dataset's published convention so the numbers sit on the literature's axis:

| dataset | history → horizon | resolution | reported at |
|---|---|---|---|
| `metr_la` | 12 → 12 | 5 min | +3 / +6 / +12 steps = 15 / 30 / 60 min (DCRNN protocol) |
| `england_covid` | 8 → 3 | daily | +1 / +2 / +3 days (PyG-Temporal setting) |

**Every method sees the same snapshots at the same resolution** — the
contrastive baselines included. Override with `HIST=… NPRED=…` if you need to,
but override it for all arms or the rows stop being comparable.

## Arms

| row | what it is |
|---|---|
| `persistence` | predict the last observed value. The reference to beat. |
| `window_mean` | mean over the node's history window |
| `historical_mean` | per-node climatology over the train split |
| `supervised` | encoder + head trained **end-to-end** on the forecast loss from random init. No SSL. The label-hungry reference. |
| `cldg` / `ddgcl` / `idol` | contrastive SSL encoder + the same frozen forecast head |
| `maskdgnn` | MaskDGNN (He et al., IJCAI'25): activeness-aware edge masking + masked/next-edge BCE on a GCN with an adaptive frequency layer, + the same frozen forecast head. On a fixed topology the dynamics score is 0, so masking is PageRank-weighted only |
| `dysubc` | DySubC (Chen et al., TKDD'23): temporal subgraph contrast (BFS subgraphs ranked by recency + degree, time-weighted vs unweighted subgraph encoders, margin losses), + the same frozen forecast head. On a fixed topology every edge shares one time level, so the sampling score reduces to degree |
| `stdmae` | STD-MAE: S-MAE + T-MAE pre-training, then its own Graph WaveNet forecaster |
| `step` | STEP (KDD'22): TSFormer masked pre-training (mask 75% of the time patches) over a **long** history window, then a **learned** discrete graph + a Graph WaveNet forecaster on the frozen representations |
| `stdmae` | STD-MAE (IJCAI'24): masked pre-training decoupled into S-MAE (mask whole nodes) and T-MAE (mask whole time patches) over a **long** history window, then a Graph WaveNet forecaster on the frozen representations |
| `jepa_enc` | the SSL encoder + a forecast head, **no predictor** — isolates what the predictor rollout adds |
| `jepa_frozen` | SSL encoder frozen, JEPA predictor on top. Collapse impossible by construction; the reference variant. |
| `jepa_naive` | `--train_encoder` alone. Targets come from the encoder being optimised, so collapse is a global optimum. Exists so EMA's win is attributable to EMA and not merely to unfreezing. |
| `jepa_ema` | `--train_encoder --ema_target` (I-JEPA) |
| `jepa_vicreg` | `--train_encoder` + VICReg variance/covariance. Anti-collapse *without* a target network. |

## Running it

### Cost, before you submit anything

On METR-LA a training step is **~5.2 s** on a V100 (the predictor attends over
`T*N = 24*207 = 4968` tokens, so attention is 24.7M pairs per head). Stride-1
gives 375 steps/epoch:

| `JEPA_EPOCHS` | per arm | 4 arms in ONE job | 4 arms, one job each |
|---|---|---|---|
| 10 | 7.5 h | 30 h | 7.5 h |
| 15 | 10.3 h | 41 h | 10.3 h |
| 30 | 18.8 h | 75 h | 18.8 h |
| 50 (default) | 30.1 h | **121 h** | 30.1 h |

Every stage's `--time` has to be checked against that, and the probe runs
*after* training, so a walltime kill loses the entire cell and writes no
`test_metrics.json`. Use `ARMS=` to put one JEPA arm per job (see step 3);
that is what makes the wall time the per-arm column, not the sum.
Only the DyG-JEPA arms are expensive — they are the only ones running the
`T*N`-token predictor. The supervised, contrastive and encoder-only arms have
no predictor and are far cheaper.

`--metr_la_stride` cuts steps/epoch (the stride-1 windows overlap by 23/24, so
consecutive minibatches are near-duplicates). It is a **search and diagnostic**
knob: it shrinks the training budget rather than giving a free speedup, and the
final table must stay at stride 1.

```bash
# 0. one-time: fetch METR-LA (England COVID auto-downloads)
bash scripts/metr_la/download_metr_la.sh          # needs pyarrow

# 0b. METR-LA ONLY -- find the stable LR range BEFORE searching.
#     At lr=5e-4 (the lowest value the grid shipped with) the predictor's TRAIN
#     loss rises from epoch 3 with a frozen encoder, i.e. it diverges. Searching
#     a grid where every cell diverges just ranks "which diverges least".
DATASET=metr_la sbatch scripts/regression/slurm_lr_probe.sh        # array of 6
python scripts/regression/aggregate_lr_probe.py --root outputs/reg_lrprobe/metr_la
#     -> edit LRS in slurm_hparam_search.sh, and set JEPA_EPOCHS from val_argmin

# 1. hyperparameter search — predictor knobs first, on the VAL split
DATASET=metr_la STAGE=predictor bash scripts/regression/slurm_hparam_search.sh --list
DATASET=metr_la STAGE=predictor STRIDE=12 JEPA_EPOCHS=50 \
    sbatch --array=0-35%6 scripts/regression/slurm_hparam_search.sh   # ~2.5 h/cell
python scripts/regression/aggregate_hparam.py \
    --root outputs/reg_hparam/metr_la_predictor --emit_flags

# 1b. tune the BASELINES too -- otherwise the table is "our tuned method vs
#     their defaults", which is the first thing a reviewer attacks. Those
#     defaults came from each paper's own datasets; CLDG ships lr=4e-3, never
#     tuned for METR-LA. Cheap: no predictor, so cells are minutes not hours.
DATASET=metr_la STAGE=baselines bash scripts/regression/slurm_hparam_search.sh --list
DATASET=metr_la STAGE=baselines sbatch --array=0-59%8 scripts/regression/slurm_hparam_search.sh
python scripts/regression/aggregate_hparam.py \
    --root outputs/reg_hparam/metr_la_baselines --by_arm --emit_flags
#     -> BASELINE_HP_cldg="..." etc; pass them to slurm_all_methods.sh in step 3

# 2. then the per-variant encoder knobs, holding the winner fixed
DATASET=metr_la STAGE=variants BASE_HP="<paste from step 1>" \
    sbatch --array=0-15%4 scripts/regression/slurm_hparam_search.sh
python scripts/regression/aggregate_hparam.py \
    --root outputs/reg_hparam/metr_la_variants --by_arm --emit_flags

# 3. final table: 3 seeds at the chosen config, probing TEST.
#    ONE ARM PER JOB on METR-LA -- a single JEPA arm is ~30 h at 50 epochs, so
#    all four in one job needs ~121 h and no walltime covers it. They are
#    independent once the shared encoder exists (built once, under flock).
for S in 42 43 44; do
  # the cheap arms fit in one job
  DATASET=metr_la SEED=$S ARMS="baselines supervised jepa_enc cldg ddgcl idol maskdgnn dysubc stdmae" \
    BASELINE_HP_cldg="<step 1b>" BASELINE_HP_ddgcl="<step 1b>" \
    BASELINE_HP_idol="<step 1b>" BASELINE_HP_maskdgnn="<step 1b>" BASELINE_HP_dysubc="<step 1b>" \
    BASELINE_HP_supervised="<step 1b>" \
    BASELINE_HP_stdmae="<step 1b>" \
    sbatch scripts/regression/slurm_all_methods.sh
  # each JEPA arm gets its own
  for A in jepa_frozen jepa_naive jepa_ema jepa_vicreg; do
    DATASET=metr_la SEED=$S ARMS=$A BASE_HP="<step 1>" VARIANT_HP_ema="<step 2>" \
      sbatch scripts/regression/slurm_all_methods.sh
  done
done
# England COVID is minutes end-to-end; just omit ARMS and run one job per seed.

# 4. mean ± std LaTeX
python scripts/regression/aggregate_regression.py \
    --dataset metr_la --seeds 42 43 44 --latex
```

Swap `DATASET=england_covid` for the other dataset; nothing else changes.

## STEP and STD-MAE: the two arms with their own backend

Every other learned arm is *encoder + the shared forecast head*, which is what
makes their columns attributable to the representation alone. These two are
not: their published downstream stage **is** a full Graph WaveNet trained on the
forecast loss, and cutting that out to bolt on the shared head would not be the
method. So read their rows next to `supervised` — same label budget, same
end-to-end forecaster, but starting from self-supervised representations rather
than random init. Both are still scored by the same
`probes.py::_accum_reg`/`_finalize_reg` on the same windows with the same valid
mask, so `MAE_persist` cross-checks them against every other row.

STEP came first and STD-MAE forked its codebase, so the port shares whatever is
byte-identical between the two references — the patchifier, transformer stack,
mask generator and `masked_mae` (`src/baselines/masked_pretrain.py`), the Graph
WaveNet backend (`src/baselines/graphwavenet.py`) and the long-window plumbing
(`src/baselines/long_history.py`). A gap between their rows is then the
pre-training objective, not the patchifier or the forecaster.

What actually differs, and is preserved:

| | STEP | STD-MAE |
|---|---|---|
| masks | time patches only, 75% | decoupled: whole nodes (S-MAE) **and** time patches (T-MAE), 25% each |
| positional encoding | learned table | 2-D sin/cos over the node × patch grid |
| graph | **learned** discrete adjacency, regularised towards a kNN graph over TSFormer's representations | the dataset's, as `doubletransition` supports |
| injected into backend | one stream | two, concatenated |
| paper's METR-LA window | 1 week (2016 steps) | 3 days (864) |

Note the third row: **STEP does not use the given graph at all.** On METR-LA the
sensor adjacency every other arm receives is unused; on England COVID so is the
mobility graph. That is the method's claim, not an omission in the port. It also
means STEP's graph learner reads every node's whole *train* signal as a feature
— it is handed exactly the train region of the timeline, never val or test,
since the learned adjacency is a parameter that persists into evaluation.

**Long history.** The MAEs read a much longer window than the forecast history.
This repo's datasets are per-split sliding-window views and carry no such
window, so `src/baselines/long_history.py` reconstructs the global signal
timeline once (stitching the pre-windowed HF parquet splits back into one
contiguous series) and slices the long window off it. A val/test window's long
history reaching back into earlier data is *past input*, not leakage — it is
exactly what both references do, and the forecast target always comes from the
window itself. Where the timeline runs out, the missing steps are left-padded
with the raw `0.0` that already means "no reading", so the `null_val=0.0` MAE
loss drops them.

England COVID's whole timeline is 61 daily snapshots, so neither paper's
geometry exists there. Both long windows are pinned to the forecast history
(8 days, 4 patches of 2) — the largest that needs no fabricated padding.
Override with `SM_LONG=… SM_PATCH=…` / `ST_LONG=… ST_PATCH=…`; each must be a
multiple of its patch size. They are deliberately *not* forced equal by default:
each paper's own window length is the thing being reproduced.

**Graph supports (STD-MAE only).** Graph WaveNet wants DCRNN `doubletransition`
supports. METR-LA has one static graph, so they are precomputed. England
COVID's mobility graph changes every snapshot, so the supports are built *per
batch* from that window's own history adjacency (Graph WaveNet's `nconv` einsum
already accepts a batched `(B,N,N)` adjacency) rather than flattening the
dataset to a single static graph.

**Pre-training cost.** Both run at `SM_STRIDE` / `ST_STRIDE` `=12` on METR-LA —
stride-1 windows overlap in all but one of their 864/2016 steps, so every 12th
window is nearly the same information for a twelfth of the cost. It affects
pre-training only; the forecasting stage and every reported metric still use all
windows. STEP's graph learner is separately expensive: it convolves every node's
entire training series on every training step (a ~38M-parameter `Linear` at
METR-LA's train length — that is the reference's `dim_fc`, not a porting
artefact). Under `no_grad` the port caches that embedding for the pass, which is
exactly equivalent and makes evaluation cheap.

## The check that matters

`aggregate_regression.py` verifies every arm reports the **same**
`MAE_persist`. Persistence is parameter-free, so a disagreement means the arms
did not score identical windows — different history length, split, or a stale
checkpoint — and the table is comparing different experiments. It prints a loud
`[WARN]` if they diverge. Do not merge rows past that warning: the numbers look
perfectly plausible either way.

The `persistence` row from `run_baselines.py` is computed independently and
should match that column too — a free second opinion.

## Notes

- `--probe_split val` during search, `test` for the final table. No
  configuration is ever ranked on the split it is reported on.
- METR-LA passes `--no_attention_mask`: the structural mask is
  `(B, 1, T·N, T·N)`, and at N=207, T=24 the materialised attention OOMs even a
  32 GB card.
- `history.json` in every run dir holds the per-epoch train/val loss. Check
  it before concluding a run was undertrained: a rising TRAIN loss with a
  frozen encoder means the optimiser diverged, not that it needs longer.
- The unfrozen arms run a collapse check after training. If the encoder
  collapsed, every node's latent is the same point and the forecast sits at
  persistence no matter how good the training loss looked.
