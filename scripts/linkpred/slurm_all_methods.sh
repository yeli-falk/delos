#!/usr/bin/env bash
#SBATCH --job-name=lp-allmethods
#SBATCH --output=logs/lp-allmethods-%j.out
#SBATCH --error=logs/lp-allmethods-%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#
# DYNAMIC LINK PREDICTION: every method, one seed per job.
# Third of the three task pipelines, structurally identical to
# scripts/regression/ and scripts/classification/.
#
#   DATASET=mooc|reddit|wikipedia|reddit_body|dblp
#       JODIE state graphs, Kumar-2018 subreddit hyperlinks, or DBLP.
#       reddit_body and dblp are LINK-PREDICTION ONLY -- they carry no
#       state_label, so the classification pipeline rejects them. (JODIE; subset/binning/encoder from
#                                    _dataset_config.sh with TASK=link_pred)
#
# Arms (each writes its own metrics JSON):
#   graphmae              GraphMAE (Hou et al., KDD'22): STATIC masked-feature
#                         autoencoder -- mask a fraction of ONE snapshot's
#                         nodes with a learnable token, re-mask their latents,
#                         GCN decoder, scaled cosine error. The generative
#                         half of the static control block (GRACE is the
#                         contrastive half); probed on FROZEN latents at the
#                         last history bin like the rows below.
#   grace                 GRACE (Zhu et al., 2020): STATIC graph contrast --
#                         two augmented views (edge removal + feature-
#                         dimension masking) of ONE snapshot, inter- and
#                         intra-view negatives. Never sees two snapshots at
#                         once: the control for "does the SSL objective need
#                         to be temporal at all?". Scored by the same frozen
#                         link probe (static SSL, last frame).
#   cldg / ddgcl / idol   contrastive SSL encoder, scored by the link probe on
#                         FROZEN latents (via analyze_jepa --link_sources encoder).
#   maskdgnn              MaskDGNN (He et al., IJCAI'25): activeness-aware edge
#                         masking + masked/next-edge BCE on a GCN with an
#                         adaptive frequency layer. Scored TWICE: its encoder
#                         through the same frozen link probe (row "MaskDGNN"),
#                         and its own encoder + edge head end-to-end through the
#                         same 99-negative evaluator (row "MaskDGNN (end-to-end)",
#                         the paper's protocol; link_e2e.json).
#   dysubc                DySubC (Chen et al., TKDD'23): temporal subgraph
#                         contrast -- BFS subgraphs ranked by recency + degree,
#                         a time-weighted and an unweighted subgraph encoder,
#                         margin losses against a shuffled and an unweighted
#                         summary. Encoder probed on FROZEN latents like the
#                         three contrastive arms.
#   dyg2vec               DyG2Vec (Alomrani et al., TMLR'24): window-based
#                         temporal attention encoder (its own encoder kind,
#                         like maskdgnn) pre-trained with VICReg on two
#                         distorted views (edge dropout + structural-feature
#                         masking) of each history window. Encoder probed on
#                         FROZEN latents -- the paper's own SSL-init protocol.
#   jepa_frozen           SSL encoder frozen, JEPA predictor on top. Targets are
#                         fixed, so latent collapse is impossible by construction.
#   jepa_naive            --train_encoder alone: targets come from the SAME
#                         encoder being optimised, so collapsing to a point is a
#                         global optimum. Separates "EMA helped" from "unfreezing
#                         helped".
#   jepa_ema              --train_encoder --ema_target (I-JEPA).
#   jepa_vicreg           --train_encoder + VICReg variance/covariance.
#   random                U(0,1) edge scores, no training. Free: analyze_jepa
#                         computes it as a link source.
#   supervised            encoder + link head trained end-to-end from random
#                         init, no SSL. Also a link source, also free.
#
# Metrics are MRR / Hits@1 / Hits@3 / Hits@10, reported over TWO edge subsets:
#   all   every positive edge in the horizon
#   new   edges NOT present in the history window -- the hard subset, where a
#         model has to predict formation rather than echo an existing edge.
# `new` is the link-prediction analogue of the forecasting inflection subset.
#
# Usage:
#   DATASET=mooc SEED=42 sbatch scripts/linkpred/slurm_all_methods.sh
#   for S in 42 43 44; do DATASET=mooc SEED=$S sbatch scripts/linkpred/slurm_all_methods.sh; done
#   python scripts/linkpred/aggregate_linkpred.py --dataset mooc --seeds 42 43 44 --latex
#
# One arm per job:
#   DATASET=mooc SEED=42 ARMS=jepa_ema sbatch scripts/linkpred/slurm_all_methods.sh
#
# Tuned flags from scripts/linkpred/slurm_hparam_search.sh:
#   BASE_HP="..." VARIANT_HP_ema="..." BASELINE_HP_cldg="..." ... sbatch ...
#
# TRANSFER TABLE -- link prediction from the CLASSIFICATION checkpoints.
# The representation is selected ONCE (recipe + hyper-parameters on the
# node-state val split) and then probed frozen on every downstream task; no
# per-task retuning. Point ROOT at the classification tree and match its
# feature set (TASK=state_probe trains WITH edge features):
#
#   for S in 42 43 44; do
#     DATASET=mooc SEED=$S ROOT=outputs/cls_mooc EDGE_FEATS=1 \
#         sbatch scripts/linkpred/slurm_all_methods.sh
#   done
#   python scripts/linkpred/aggregate_linkpred.py --dataset mooc \
#       --root outputs/cls_mooc --seeds 42 43 44 --latex
#
# Nothing is retrained: the JEPA arms are skipped as complete (their
# classification run already wrote the link test_metrics.json, because
# train_dyg_jepa's downstream task defaults to link_pred), every baseline's
# encoder.ckpt is found and only the frozen link probe runs, and the controls
# reuse the shared encoder. The one row this cannot produce is
# "MaskDGNN (end-to-end)": its trainer saves the encoder but not the edge
# head, and the classification run used --e2e_eval none, so that row needs a
# retrain (rm outputs/cls_<ds>/maskdgnn_s<seed>/encoder.ckpt first).

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

DATASET="${DATASET:-mooc}"
case "$DATASET" in
    mooc|reddit|wikipedia|reddit_body|dblp) ;;
    *) echo "[FATAL] DATASET must be one of: mooc reddit wikipedia reddit_body " \
            "dblp -- got '$DATASET'" >&2
       exit 2 ;;
esac
SEED="${SEED:-42}"
NW="${SLURM_CPUS_PER_TASK:-8}"
PROBE_SPLIT="${PROBE_SPLIT:-test}"

# Stage names: grace graphmae cldg ddgcl idol maskdgnn dysubc dyg2vec jepa_frozen jepa_naive jepa_ema jepa_vicreg
#              controls summary
# `controls` produces the random + supervised link sources in one pass.
ARMS="${ARMS:-all}"
want() { [[ " $ARMS " == *" all "* || " $ARMS " == *" $1 "* ]]; }
need_encoder() {
    want jepa_frozen || want jepa_naive || want jepa_ema || want jepa_vicreg
}

# Each source keeps its raw file somewhere different; check the right one and
# print the command that fetches it. Compute nodes usually have no internet, so
# this fails early with a login-node instruction rather than mid-run.
case "$DATASET" in
    mooc|reddit|wikipedia)
        RAW="dataset/jodie/${DATASET}.csv"
        FETCH="wget http://snap.stanford.edu/jodie/${DATASET}.csv -O $RAW" ;;
    reddit_body)
        RAW="dataset/reddit/soc-redditHyperlinks-body.tsv"
        FETCH="wget https://snap.stanford.edu/data/soc-redditHyperlinks-body.tsv -O $RAW" ;;
    dblp)
        RAW="dataset/dblp/edges.csv"
        FETCH="see scripts/dblp/preprocess_dblp_xml.py (or fetch_preprocessed_dblp.sh)" ;;
esac
if [[ ! -f "$RAW" ]]; then
    echo "[FATAL] $RAW missing. From the LOGIN node:" >&2
    echo "  cd $REPO_ROOT && $FETCH" >&2
    exit 10
fi

# TASK=link_pred uses the FULL dataset (every node), like the other tasks.
# The probe scores 99 negatives per positive edge per window, so cost grows
# ~N^2 -- lower BATCH if you OOM.
TASK=link_pred
source "$REPO_ROOT/scripts/dblp/_dataset_config.sh"
BATCH="${BATCH:-8}"
SHORT="${DATASET#jodie_}"

BASE_HP="${BASE_HP:-}"
BASELINE_HP_cldg="${BASELINE_HP_cldg:-}"
BASELINE_HP_ddgcl="${BASELINE_HP_ddgcl:-}"
BASELINE_HP_idol="${BASELINE_HP_idol:-}"
BASELINE_HP_maskdgnn="${BASELINE_HP_maskdgnn:-}"
BASELINE_HP_dysubc="${BASELINE_HP_dysubc:-}"
BASELINE_HP_dyg2vec="${BASELINE_HP_dyg2vec:-}"
BASELINE_HP_grace="${BASELINE_HP_grace:-}"
BASELINE_HP_graphmae="${BASELINE_HP_graphmae:-}"

RESIDUAL="${RESIDUAL:-1}"
CHANGE_W="${CHANGE_W:-0}"
JEPA_BASE="--mask_scheme ${MASK_SCHEME:-node_mask} --downstream_task link_pred"
[[ "$RESIDUAL" == "1" ]] && JEPA_BASE="$JEPA_BASE --residual_head"
[[ "$CHANGE_W" != "0" ]] && JEPA_BASE="$JEPA_BASE --change_weight $CHANGE_W"

ROOT="${ROOT:-outputs/lp_${SHORT}}"
mkdir -p "$ROOT"
d() { echo "$ROOT/$1_s${SEED}"; }

# Structural attention mask ON (trainer default, sparse edge-list attention).
# Earlier runs passed --no_attention_mask; results from the old ROOT are not
# comparable with this version -- use a fresh ROOT.
JEPA_COMMON="$DATA_ARG $WINDOW_ARG $EARLY_STOP_ARG $ENCODER_ARG \
    --num_masked_nodes ${NMASK:-30} --depth ${DEPTH:-4} --heads ${HEADS:-4} \
    --mlp_dim ${MLP_DIM:-512} --batch_size $BATCH --epochs ${JEPA_EPOCHS:-300} \
    --lr ${LR:-5e-4} --probe_epochs ${JEPA_PROBE_EPOCHS:-3} \
    --probe_split $PROBE_SPLIT --seed $SEED --num_workers $NW --device cuda"

# analyze_jepa scores a frozen encoder with the SAME link probe the JEPA arms
# get, so the baseline columns are attributable to the representation.
LINK_COMMON="$DATA_ARG $WINDOW_ARG --skip_diagnostic --probe_split $PROBE_SPLIT \
    --link_epochs ${LINK_EPOCHS:-30} --link_lr ${LINK_LR:-1e-3} \
    --link_probe_hidden 64 --seed $SEED --device cuda"

echo "[config] DATASET=$DATASET SEED=$SEED BATCH=$BATCH"
echo "[config] ARMS='$ARMS' JEPA_BASE='$JEPA_BASE' BASE_HP='$BASE_HP' probe_split=$PROBE_SPLIT"

fail() { echo "[FATAL] $*" >&2; exit "${2:-1}"; }

# =============================================================================
# 1. shared SSL encoder (edge_mask) -- the start point for all four variants
# =============================================================================
ENC_DIR="$ROOT/_shared_encoder_s${SEED}"
ENC_CKPT="$ENC_DIR/encoder_${DATASET}_${ENC_KIND}_edge_mask.ckpt"
mkdir -p "$ENC_DIR"
if need_encoder || want controls; then
    exec 9>"$ENC_DIR/.pretrain.lock"
    flock -w 21600 9 || fail "encoder lock timeout" 10
    if [[ ! -f "$ENC_CKPT" ]]; then
        echo; echo "=== shared encoder pretrain (lock held) -> $ENC_DIR ==="
        srun --export=ALL python src/train/pretrain_encoder.py $DATA_ARG \
            $ENCODER_ARG $WINDOW_ARG $EARLY_STOP_ARG \
            --pretext edge_mask --decoder_hidden 64 --mask_rate 0.25 \
            --batch_size "$BATCH" --epochs "${ENC_EPOCHS:-300}" --lr 1e-3 \
            --seed "$SEED" --num_workers "$NW" --device cuda --out_dir "$ENC_DIR"
        if [[ ! -f "$ENC_CKPT" ]]; then
            flock -u 9; fail "encoder pretrain produced no $ENC_CKPT" 11
        fi
    else
        echo "[skip] shared encoder exists: $ENC_CKPT"
    fi
    flock -u 9; exec 9>&-
fi

# =============================================================================
# 2. DyG-JEPA encoder-training variants
# =============================================================================
arm_flags() {
    case "$1" in
        frozen) echo "" ;;
        naive)  echo "--train_encoder" ;;
        ema)    echo "--train_encoder --ema_target" ;;
        vicreg) echo "--train_encoder --vicreg_cov 1.0 --vicreg_var 5.0" ;;
        *)      echo "__UNKNOWN__" ;;
    esac
}

for ARM in frozen naive ema vicreg; do
    want "jepa_$ARM" || continue
    VFLAGS="$(arm_flags "$ARM")"
    [[ "$VFLAGS" == "__UNKNOWN__" ]] && fail "unknown arm '$ARM'" 12
    eval "HP_ARGS=\${VARIANT_HP_${ARM}:-}"
    OUT="$(d "jepa_$ARM")"
    mkdir -p "$OUT"
    echo; echo "=== DyG-JEPA [$ARM] -> $OUT ==="
    echo "    flags='$VFLAGS $HP_ARGS'"
    # train_dyg_jepa's own link probe writes test_metrics.json with the
    # all/new subsets -- that IS this arm's result, no second scoring pass.
    # An arm is COMPLETE only with both files: best.ckpt alone is written on
    # every val improvement, so a walltime kill leaves one behind. Same guard
    # as the classification launcher, and what makes ROOT=outputs/cls_<ds>
    # (transfer table, see header) a probe-only pass for these arms.
    if [[ -f "$OUT/best.ckpt" && -f "$OUT/test_metrics.json" ]]; then
        echo "[skip] $OUT complete (best.ckpt + test_metrics.json)"
    else
        if [[ -f "$OUT/best.ckpt" ]]; then
            echo "[resume] $OUT has best.ckpt but no test_metrics.json (interrupted) -> restarting from scratch"
            rm -f "$OUT/best.ckpt" "$OUT/history.json"
        fi
        srun --export=ALL python src/train/train_dyg_jepa.py $JEPA_COMMON \
            --encoder_ckpt "$ENC_CKPT" $JEPA_BASE $BASE_HP $VFLAGS $HP_ARGS \
            --out_dir "$OUT" || { echo "[warn] arm $ARM failed"; continue; }
        [[ -f "$OUT/best.ckpt" && -f "$OUT/test_metrics.json" ]] \
            || { echo "[warn] arm $ARM did not finish (no best.ckpt/test_metrics.json)"; continue; }
    fi

    if [[ "$ARM" != "frozen" ]]; then
        srun --export=ALL python -m src.train.analyze_jepa --skip_diagnostic \
            --link_sources jepa $DATA_ARG $WINDOW_ARG \
            --jepa_ckpt "$OUT/best.ckpt" --device cuda --seed "$SEED" \
            --out_dir "$OUT/collapse_check" \
            || echo "[warn] collapse check failed for $ARM (non-fatal)"
    fi
done

# =============================================================================
# 3. contrastive baselines: pretrain encoder -> the SAME frozen link probe
# =============================================================================
for M in grace graphmae cldg ddgcl idol maskdgnn dysubc dyg2vec; do
    want "$M" || continue
    D="$(d "$M")"
    mkdir -p "$D"
    eval "HP=\${BASELINE_HP_${M}:-}"
    if [[ ! -f "$D/encoder.ckpt" ]]; then
        echo; echo "=== ${M^^} pretrain -> $D ==="
        case "$M" in
          graphmae)
            # STATIC generative control: one snapshot per step, reference
            # Cora defaults (mask_rate 0.5, replace_rate 0.05, gamma 2, Adam
            # 1e-3 / wd 2e-4). Reconstructs every input channel, as published.
            srun --export=ALL python -m src.baselines.graphmae.train_graphmae $DATA_ARG \
                $ENCODER_ARG $WINDOW_ARG $EARLY_STOP_ARG \
                --mask_rate 0.5 --replace_rate 0.05 --sce_gamma 2 --target all \
                --steps_per_epoch 20 --val_steps 10 \
                --epochs "${BASE_EPOCHS:-100}" --lr 1e-3 \
                --device cuda --seed "$SEED" --out_dir "$D" $HP ;;
          grace)
            # STATIC control: one snapshot per step, two augmented views of
            # it, reference Cora defaults (tau 0.4, p_e 0.2/0.4, p_f 0.3/0.4,
            # Adam 5e-4). Same per-step node budget as the temporal arms.
            srun --export=ALL python -m src.baselines.grace.train_grace $DATA_ARG \
                $ENCODER_ARG $WINDOW_ARG $EARLY_STOP_ARG \
                --tau 0.4 --p_e1 0.2 --p_e2 0.4 --p_f1 0.3 --p_f2 0.4 \
                --node_batch_size 256 --steps_per_epoch 20 --val_steps 10 \
                --epochs "${BASE_EPOCHS:-100}" --lr 5e-4 \
                --device cuda --seed "$SEED" --out_dir "$D" $HP ;;
          cldg)
            srun --export=ALL python -m src.baselines.cldg.train_cldg $DATA_ARG \
                $ENCODER_ARG $WINDOW_ARG $EARLY_STOP_ARG \
                --snapshots 4 --views 2 --strategy random --tau 0.07 \
                --node_batch_size 256 --steps_per_epoch 20 --val_steps 10 \
                --epochs "${BASE_EPOCHS:-100}" --lr 4e-3 \
                --device cuda --seed "$SEED" --out_dir "$D" $HP ;;
          ddgcl)
            srun --export=ALL python -m src.baselines.ddgcl.train_ddgcl $DATA_ARG \
                $ENCODER_ARG $WINDOW_ARG $EARLY_STOP_ARG \
                --max_gap 4 --time_dim 32 --tau 0.01 --beta 1.5 \
                --node_batch_size 256 --steps_per_epoch 20 --val_steps 10 \
                --epochs "${BASE_EPOCHS:-100}" --lr 4e-3 \
                --device cuda --seed "$SEED" --out_dir "$D" $HP ;;
          idol)
            # --steps_per_epoch caps IDOL to the same budget CLDG/DDGCL get;
            # uncapped it visits every training snapshot per epoch.
            srun --export=ALL python -m src.baselines.idol.train_idol $DATA_ARG \
                $ENCODER_ARG $WINDOW_ARG $EARLY_STOP_ARG \
                --proj_layers 1 --node_batch_size 256 \
                --steps_per_epoch 20 --val_steps 10 \
                --epochs "${BASE_EPOCHS:-100}" --lr 1e-3 \
                --device cuda --seed "$SEED" --out_dir "$D" $HP ;;
          maskdgnn)
            # Its own encoder kind (the shared GCN + an adaptive frequency
            # layer): that layer IS the method's representation learner, so it
            # is part of what the probe sees. `--encoder_kind` after
            # $ENCODER_ARG wins (argparse keeps the last occurrence). --window
            # defaults to HIST so the frequency filter sees the same length at
            # pretrain and probe time. --e2e_eval writes link_e2e[_val].json
            # for the end-to-end row.
            srun --export=ALL python -m src.baselines.maskdgnn.train_maskdgnn $DATA_ARG \
                $ENCODER_ARG --encoder_kind maskdgnn $WINDOW_ARG $EARLY_STOP_ARG \
                --mask_ratio 0.2 --alpha 0.7 --target next --neg_per_pos 1 \
                --steps_per_epoch 20 --val_steps 10 \
                --epochs "${BASE_EPOCHS:-100}" --lr 4e-3 \
                --e2e_eval "$PROBE_SPLIT" --e2e_batch_size "$BATCH" --num_workers "$NW" \
                --device cuda --seed "$SEED" --out_dir "$D" $HP ;;
          dysubc)
            # Temporal subgraph contrast on the shared GCN. --lookback
            # defaults to HIST, so the time-weighted graph spans exactly the
            # window the probe hands the encoder. Paper defaults (k=20,
            # margin 0.75, lambda 0.5, Adam 1e-3); alpha=5 is the released
            # code's MOOC value.
            srun --export=ALL python -m src.baselines.dysubc.train_dysubc $DATA_ARG \
                $ENCODER_ARG $WINDOW_ARG $EARLY_STOP_ARG \
                --subgraph_size 20 --alpha 5 --beta 1 --margin 0.75 --lam 0.5 \
                --node_batch_size 256 --steps_per_epoch 20 --val_steps 10 \
                --epochs "${BASE_EPOCHS:-100}" --lr 1e-3 \
                --device cuda --seed "$SEED" --out_dir "$D" $HP ;;
          dyg2vec)
            # Own encoder kind (window temporal attention; the `--encoder_kind`
            # after $ENCODER_ARG wins). --window defaults to HIST so the
            # encoder is pre-trained on the length it is probed on. Paper
            # defaults: p_d 0.3, VICReg 25/25/1, Adam 1e-4.
            srun --export=ALL python -m src.baselines.dyg2vec.train_dyg2vec $DATA_ARG \
                $ENCODER_ARG --encoder_kind dyg2vec $WINDOW_ARG $EARLY_STOP_ARG \
                --p_edge 0.3 --p_struct 0.3 --lam 25 --mu 25 --nu 1 \
                --steps_per_epoch 20 --val_steps 10 \
                --epochs "${BASE_EPOCHS:-100}" --lr 1e-4 \
                --device cuda --seed "$SEED" --out_dir "$D" $HP ;;
        esac
        [[ -f "$D/encoder.ckpt" ]] || fail "${M^^} pretrain failed" 13
    else
        echo "[skip] $D/encoder.ckpt exists"
        if [[ "$M" == "maskdgnn" && ! -f "$D/link_e2e${PROBE_SPLIT/test/}.json" ]]; then
            echo "[note] $D has no link_e2e json: the end-to-end row needs a retrain" \
                 "(the trainer does not save the edge head). Frozen-probe row still runs."
        fi
    fi
    echo; echo "=== link probe: frozen $M encoder -> $D ==="
    srun --export=ALL python -m src.train.analyze_jepa --link_sources encoder \
        $LINK_COMMON --encoder_ckpt "$D/encoder.ckpt" --out_dir "$D" \
        || echo "[warn] link probe failed for $D"
done

# =============================================================================
# 4. controls: random + supervised, both free link sources
# =============================================================================
if want controls; then
    CTRL_DIR="$(d controls)"
    mkdir -p "$CTRL_DIR"
    echo; echo "=== link controls (random + supervised) -> $CTRL_DIR ==="
    srun --export=ALL python -m src.train.analyze_jepa \
        --link_sources random supervised $LINK_COMMON \
        --encoder_ckpt "$ENC_CKPT" --out_dir "$CTRL_DIR" \
        || echo "[warn] link controls failed"
fi

# =============================================================================
# 5. summary for this seed
# =============================================================================
if want summary; then
    echo; echo "=== seed $SEED summary ==="
    srun --export=ALL python scripts/linkpred/aggregate_linkpred.py \
        --dataset "$SHORT" --root "$ROOT" --seeds "$SEED" || true
fi

echo; echo "[done] $DATASET seed $SEED -> $ROOT"
echo "Aggregate across seeds with:"
echo "  python scripts/linkpred/aggregate_linkpred.py --dataset $SHORT --root $ROOT --seeds 42 43 44 --latex"
