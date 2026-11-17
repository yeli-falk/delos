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
# Dynamic link prediction: every method, one seed per job. Each arm writes its
# own metrics JSON (MRR / Hits@k over the `all` and `new` edge subsets). The
# arm table, the metrics and the transfer-table protocol are in
# scripts/linkpred/README.md.
#
# Usage:
#   DATASET=mooc SEED=42 sbatch scripts/linkpred/slurm_all_methods.sh
#   for S in 42 43 44; do DATASET=mooc SEED=$S sbatch scripts/linkpred/slurm_all_methods.sh; done
#   python scripts/linkpred/aggregate_linkpred.py --dataset mooc --seeds 42 43 44 --latex
#   # One arm per job:
#   DATASET=mooc SEED=42 ARMS=jepa_ema sbatch scripts/linkpred/slurm_all_methods.sh
#   # Transfer table: probe the classification checkpoints without training.
#   DATASET=mooc SEED=42 ROOT=outputs/cls_mooc EDGE_FEATS=1 PROBE_ONLY=1 \
#     ARMS="grace graphmae cldg ddgcl idol maskdgnn dysubc dyg2vec dygwm sgjepa jepa_ema" \
#     sbatch scripts/linkpred/slurm_all_methods.sh
#
# Main environment knobs (the others are documented where they are defined):
#   DATASET      mooc | reddit | wikipedia | reddit_body | dblp | cldg_dblp |
#                cldg_patent (alias patent) | tgbn_trade |
#                chase_{dblp_s,gossipcop,music}. Default mooc.
#   SEED         Default 42.
#   ARMS         Stages to run, see "which arms this job runs". Default all.
#   ROOT         Results root. Default outputs/lp_<dataset>.
#   PROBE_ONLY   1 = score existing checkpoints and never train: an arm whose
#                checkpoint is missing is skipped, the JEPA arms get the same
#                link probe as the baselines, and every probe writes to
#                <run>/link_probe/ so the run's own args.json is not overwritten.
#   LINK_BATCH   Window batch of the frozen link probe. Use 1 on reddit.
#   BASE_HP, VARIANT_HP_<arm>, BASELINE_HP_<method>
#                Tuned flags from scripts/linkpred/slurm_hparam_search.sh.

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
    mooc|reddit|wikipedia|reddit_body|dblp|cldg_dblp|cldg_patent|patent|tgbn_trade|chase_dblp_s|chase_gossipcop|chase_music) ;;
    *) echo "[FATAL] DATASET must be one of: mooc reddit wikipedia reddit_body " \
            "dblp cldg_dblp cldg_patent tgbn_trade chase_{dblp_s,gossipcop,music} -- got '$DATASET'" >&2
       exit 2 ;;
esac
SEED="${SEED:-42}"
# Cosine-anneal horizon shared by every EMA arm that anneals its momentum
# (DELOS ema_cos / ijepa / ema_cos_tnorm and the BGRL baseline). The trainers'
# default spans the whole --epochs budget, which early stopping rarely reaches,
# so the schedule would never complete. 0 = anneal over the whole run.
EMA_ANNEAL_EPOCHS="${EMA_ANNEAL_EPOCHS:-50}"
ANNEAL_ARG=""; [[ "$EMA_ANNEAL_EPOCHS" != "0" ]] && ANNEAL_ARG="--ema_anneal_epochs $EMA_ANNEAL_EPOCHS"
NW="${SLURM_CPUS_PER_TASK:-8}"
PROBE_SPLIT="${PROBE_SPLIT:-test}"

# Stage names: grace graphmae bgrl cldg ddgcl idol maskdgnn dysubc dyg2vec
# dygwm sgjepa controls summary, plus jepa_<arm> for every arm in JEPA_ARMS.
# ARMS=all runs the baselines and JEPA_ARMS_OLD, ARMS=all_jepa adds
# JEPA_ARMS_NEW, ARMS=new_jepa runs only JEPA_ARMS_NEW. Stages can also be
# listed explicitly. `controls` produces the random and supervised link
# sources in one pass.
JEPA_ARMS_OLD="frozen naive ema vicreg input"
JEPA_ARMS_NEW="ctx tnorm ema_cos sigreg lejepa ijepa lejepa_full ema_tnorm ema_cos_tnorm"
JEPA_ARMS="$JEPA_ARMS_OLD $JEPA_ARMS_NEW"
ARMS="${ARMS:-all}"
# PROBE_ONLY=1: score existing checkpoints, never train (see header).
PROBE_ONLY="${PROBE_ONLY:-0}"
# Where the analyze_jepa link probes land inside a run dir.
lp_dir() { [[ "$PROBE_ONLY" == "1" ]] && echo "$1/link_probe" || echo "$1"; }
want() {
    [[ " $ARMS " == *" all "* || " $ARMS " == *" $1 "* ]] && return 0
    case "$1" in
        jepa_*)
            local a="${1#jepa_}"
            [[ " $ARMS " == *" all_jepa "* ]] && return 0
            [[ " $ARMS " == *" new_jepa "* && " $JEPA_ARMS_NEW " == *" $a "* ]] && return 0
            ;;
    esac
    return 1
}
need_encoder() {
    local a
    for a in $JEPA_ARMS; do want "jepa_$a" && return 0; done
    return 1
}

# Each source keeps its raw file somewhere different. Compute nodes usually
# have no internet, so fail early with the login-node command.
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
    cldg_dblp|cldg_patent|patent)
        # CLDG-release graphs, used here for links only.
        N="${DATASET#cldg_}"
        RAW="dataset/cldg/${N}/${N}.txt"
        FETCH="bash scripts/dblp/extract_cldg_datasets.sh" ;;
    tgbn_trade)
        RAW="dataset/tgb/tgbn-trade/tgbn-trade_node_labels.csv"
        FETCH="bash scripts/dblp/fetch_tgbn_trade.sh" ;;
    chase_dblp_s)   RAW="dataset/chase/DBLP-S.pkl";    FETCH="bash scripts/dblp/fetch_chase.sh" ;;
    chase_gossipcop) RAW="dataset/chase/GossipCop.pkl"; FETCH="bash scripts/dblp/fetch_chase.sh" ;;
    chase_music)    RAW="dataset/chase/Music.pkl";     FETCH="bash scripts/dblp/fetch_chase.sh" ;;
esac
if [[ ! -f "$RAW" ]]; then
    echo "[FATAL] $RAW missing. From the LOGIN node:" >&2
    echo "  cd $REPO_ROOT && $FETCH" >&2
    exit 10
fi

# The probe scores 99 negatives per positive edge per window on the full
# dataset, so cost grows as N^2. Lower BATCH if you run out of memory.
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
BASELINE_HP_dygwm="${BASELINE_HP_dygwm:-}"
BASELINE_HP_sgjepa="${BASELINE_HP_sgjepa:-}"
BASELINE_HP_grace="${BASELINE_HP_grace:-}"
BASELINE_HP_graphmae="${BASELINE_HP_graphmae:-}"
BASELINE_HP_bgrl="${BASELINE_HP_bgrl:-}"

RESIDUAL="${RESIDUAL:-1}"
CHANGE_W="${CHANGE_W:-0}"
JEPA_BASE="--mask_scheme ${MASK_SCHEME:-node_mask} --downstream_task link_pred"
[[ "$RESIDUAL" == "1" ]] && JEPA_BASE="$JEPA_BASE --residual_head"
[[ "$CHANGE_W" != "0" ]] && JEPA_BASE="$JEPA_BASE --change_weight $CHANGE_W"

ROOT="${ROOT:-outputs/lp_${SHORT}}"
mkdir -p "$ROOT"
d() { echo "$ROOT/$1_s${SEED}"; }

# The structural attention mask stays on (trainer default).
JEPA_COMMON="$DATA_ARG $WINDOW_ARG $EARLY_STOP_ARG $ENCODER_ARG \
    --num_masked_nodes ${NMASK:-30} --depth ${DEPTH:-4} --heads ${HEADS:-4} \
    --mlp_dim ${MLP_DIM:-512} --batch_size $BATCH --epochs ${JEPA_EPOCHS:-300} \
    --lr ${LR:-5e-4} --probe_epochs ${JEPA_PROBE_EPOCHS:-3} \
    --probe_split $PROBE_SPLIT --seed $SEED --num_workers $NW --device cuda"

# analyze_jepa scores a frozen encoder with the same link probe the JEPA arms
# get. LINK_BATCH x HIST float (N, N) adjacencies live on the GPU, so large
# graphs (reddit) need LINK_BATCH=1; keep it equal to the value used in
# slurm_hparam_search.sh.
LINK_COMMON="$DATA_ARG $WINDOW_ARG --skip_diagnostic --probe_split $PROBE_SPLIT \
    --link_epochs ${LINK_EPOCHS:-30} --link_lr ${LINK_LR:-1e-3} \
    --link_probe_hidden 64 --batch_size ${LINK_BATCH:-4} --seed $SEED --device cuda"

echo "[config] DATASET=$DATASET SEED=$SEED BATCH=$BATCH"
echo "[config] ARMS='$ARMS' JEPA_BASE='$JEPA_BASE' BASE_HP='$BASE_HP' probe_split=$PROBE_SPLIT"

fail() { echo "[FATAL] $*" >&2; exit "${2:-1}"; }

# --- 1. shared SSL encoder (edge_mask), the start point of every JEPA arm ----
ENC_DIR="$ROOT/_shared_encoder_s${SEED}"
ENC_CKPT="$ENC_DIR/encoder_${DATASET}_${ENC_KIND}_edge_mask.ckpt"
mkdir -p "$ENC_DIR"
if need_encoder || want controls; then
    exec 9>"$ENC_DIR/.pretrain.lock"
    flock -w 21600 9 || fail "encoder lock timeout" 10
    if [[ ! -f "$ENC_CKPT" && "$PROBE_ONLY" == "1" ]]; then
        echo "[warn] PROBE_ONLY: no shared encoder at $ENC_CKPT (controls will be skipped)"
    elif [[ ! -f "$ENC_CKPT" ]]; then
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

# --- 2. DELOS encoder-training variants --------------------------------------
arm_flags() {
    case "$1" in
        frozen)      echo "" ;;
        naive)       echo "--train_encoder" ;;
        ema)         echo "--train_encoder --ema_target" ;;
        vicreg)      echo "--train_encoder --vicreg_cov 1.0 --vicreg_var 5.0" ;;
        input)       echo "--predict_space input" ;;
        # JEPA-recipe variants; every flag is opt-in in train_delos.py.
        ctx)         echo "--mask_encoder_input" ;;
        tnorm)       echo "--target_norm" ;;
        ema_cos)     echo "--train_encoder --ema_target --ema_schedule cosine $ANNEAL_ARG" ;;
        # EMA + target LayerNorm without context masking.
        ema_tnorm)     echo "--train_encoder --ema_target --target_norm" ;;
        ema_cos_tnorm) echo "--train_encoder --ema_target --ema_schedule cosine $ANNEAL_ARG --target_norm" ;;
        sigreg)      echo "--train_encoder --sigreg 1.0" ;;
        lejepa)      echo "--train_encoder --target_grad --sigreg 1.0" ;;
        ijepa)       echo "--train_encoder --ema_target --ema_schedule cosine $ANNEAL_ARG --mask_encoder_input --target_norm" ;;
        lejepa_full) echo "--train_encoder --target_grad --sigreg 1.0 --mask_encoder_input --target_norm" ;;
        *)           echo "__UNKNOWN__" ;;
    esac
}

for ARM in $JEPA_ARMS; do
    want "jepa_$ARM" || continue
    VFLAGS="$(arm_flags "$ARM")"
    [[ "$VFLAGS" == "__UNKNOWN__" ]] && fail "unknown arm '$ARM'" 12
    eval "HP_ARGS=\${VARIANT_HP_${ARM}:-}"
    OUT="$(d "jepa_$ARM")"
    [[ "$PROBE_ONLY" == "1" ]] || mkdir -p "$OUT"
    if [[ "$PROBE_ONLY" == "1" ]]; then
        # Complete run only (see the guard below): a lone best.ckpt is an
        # interrupted, under-trained arm and must not reach the table.
        if [[ ! ( -f "$OUT/best.ckpt" && -f "$OUT/test_metrics.json" ) ]]; then
            echo "[warn] PROBE_ONLY: $OUT is not a complete run -> skipped"; continue
        fi
        echo; echo "=== link probe: DELOS [$ARM] best.ckpt -> $(lp_dir "$OUT") ==="
        srun --export=ALL python -m src.train.analyze_jepa --link_sources jepa \
            $LINK_COMMON --jepa_ckpt "$OUT/best.ckpt" --out_dir "$(lp_dir "$OUT")" \
            || echo "[warn] link probe failed for $OUT"
        continue
    fi
    echo; echo "=== DELOS [$ARM] -> $OUT ==="
    echo "    flags='$VFLAGS $HP_ARGS'"
    # The trainer's own link probe writes test_metrics.json with the all / new
    # subsets, which is this arm's result. An arm is complete only with both
    # files: best.ckpt alone is written on every val improvement.
    if [[ -f "$OUT/best.ckpt" && -f "$OUT/test_metrics.json" ]]; then
        echo "[skip] $OUT complete (best.ckpt + test_metrics.json)"
    else
        if [[ -f "$OUT/best.ckpt" ]]; then
            echo "[resume] $OUT has best.ckpt but no test_metrics.json (interrupted) -> restarting from scratch"
            rm -f "$OUT/best.ckpt" "$OUT/history.json"
        fi
        srun --export=ALL python src/train/train_delos.py $JEPA_COMMON \
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

# --- 3. baselines: pretrain the encoder, then the same frozen link probe -----
for M in grace graphmae bgrl cldg ddgcl idol maskdgnn dysubc dyg2vec dygwm sgjepa; do
    want "$M" || continue
    D="$(d "$M")"
    if [[ "$PROBE_ONLY" == "1" && ! -f "$D/encoder.ckpt" ]]; then
        echo "[warn] PROBE_ONLY: no $D/encoder.ckpt -> $M skipped"; continue
    fi
    mkdir -p "$D"
    eval "HP=\${BASELINE_HP_${M}:-}"
    if [[ ! -f "$D/encoder.ckpt" ]]; then
        echo; echo "=== ${M^^} pretrain -> $D ==="
        case "$M" in
          graphmae)
            # Static generative control: one snapshot per step, reference
            # Cora defaults. Reconstructs every input channel, as published.
            srun --export=ALL python -m src.baselines.graphmae.train_graphmae $DATA_ARG \
                $ENCODER_ARG $WINDOW_ARG $EARLY_STOP_ARG \
                --mask_rate 0.5 --replace_rate 0.05 --sce_gamma 2 --target all \
                --steps_per_epoch 20 --val_steps 10 \
                --epochs "${BASE_EPOCHS:-100}" --lr 1e-3 \
                --device cuda --seed "$SEED" --out_dir "$D" $HP ;;
          bgrl)
            # Static negative-free control: one snapshot per step, reference
            # WikiCS config. Val-loss selection like every other arm; pass
            # --select last via BASELINE_HP_bgrl for the paper's fixed budget.
            srun --export=ALL python -m src.baselines.bgrl.train_bgrl $DATA_ARG \
                $ENCODER_ARG $WINDOW_ARG $EARLY_STOP_ARG \
                --p_e1 0.2 --p_e2 0.3 --p_f1 0.2 --p_f2 0.1 --ema_momentum 0.99 $ANNEAL_ARG \
                --steps_per_epoch 20 --val_steps 10 \
                --epochs "${BASE_EPOCHS:-100}" --lr 5e-4 \
                --device cuda --seed "$SEED" --out_dir "$D" $HP ;;
          grace)
            # Static contrastive control: two augmented views of one snapshot,
            # reference Cora defaults.
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
            # Own encoder kind (the `--encoder_kind` after $ENCODER_ARG wins).
            # --window defaults to HIST so the frequency filter sees the same
            # length at pretrain and probe time. --e2e_eval writes
            # link_e2e[_val].json for the end-to-end row.
            srun --export=ALL python -m src.baselines.maskdgnn.train_maskdgnn $DATA_ARG \
                $ENCODER_ARG --encoder_kind maskdgnn $WINDOW_ARG $EARLY_STOP_ARG \
                --mask_ratio 0.2 --alpha 0.7 --target next --neg_per_pos 1 \
                --steps_per_epoch 20 --val_steps 10 \
                --epochs "${BASE_EPOCHS:-100}" --lr 4e-3 \
                --e2e_eval "$PROBE_SPLIT" --e2e_batch_size "$BATCH" --num_workers "$NW" \
                --device cuda --seed "$SEED" --out_dir "$D" $HP ;;
          dysubc)
            # --lookback defaults to HIST, so the time-weighted graph spans
            # exactly the window the probe hands the encoder. Paper defaults;
            # alpha=5 is the released code's MOOC value.
            srun --export=ALL python -m src.baselines.dysubc.train_dysubc $DATA_ARG \
                $ENCODER_ARG $WINDOW_ARG $EARLY_STOP_ARG \
                --subgraph_size 20 --alpha 5 --beta 1 --margin 0.75 --lam 0.5 \
                --node_batch_size 256 --steps_per_epoch 20 --val_steps 10 \
                --epochs "${BASE_EPOCHS:-100}" --lr 1e-3 \
                --device cuda --seed "$SEED" --out_dir "$D" $HP ;;
          dyg2vec)
            # Own encoder kind (the `--encoder_kind` after $ENCODER_ARG wins).
            # --window defaults to HIST; the other values are paper defaults.
            srun --export=ALL python -m src.baselines.dyg2vec.train_dyg2vec $DATA_ARG \
                $ENCODER_ARG --encoder_kind dyg2vec $WINDOW_ARG $EARLY_STOP_ARG \
                --p_edge 0.3 --p_struct 0.3 --lam 25 --mu 25 --nu 1 \
                --steps_per_epoch 20 --val_steps 10 \
                --epochs "${BASE_EPOCHS:-100}" --lr 1e-4 \
                --device cuda --seed "$SEED" --out_dir "$D" $HP ;;
          dygwm)
            # Own encoder kind (the `--encoder_kind` after $ENCODER_ARG wins).
            # SSL-only core of DyG-WM (no link / rank loss), so this row is its
            # frozen-encoder protocol, not the paper's supervised link numbers.
            srun --export=ALL python -m src.baselines.dygwm.train_dygwm $DATA_ARG \
                $ENCODER_ARG --encoder_kind dygwm $WINDOW_ARG $EARLY_STOP_ARG \
                --ema_momentum 0.999 --n_ctx 32 --decay 0.8 --cn_bonus 1.0 \
                --lam_node 1 --lam_rel 1 --lam_var 0.25 --lam_cov 0.025 \
                --node_batch_size 256 --pair_batch_size 256 \
                --steps_per_epoch 20 --val_steps 10 \
                --epochs "${BASE_EPOCHS:-100}" --lr 3e-4 \
                --device cuda --seed "$SEED" --out_dir "$D" $HP ;;
          sgjepa)
            # Shared per-snapshot GCN as the stop-gradient target encoder,
            # spiking context encoder on its latents. The paper gives no
            # temperature or optimiser: tau 0.1 and AdamW 1e-3 are our choice.
            srun --export=ALL python -m src.baselines.sgjepa.train_sgjepa $DATA_ARG \
                $ENCODER_ARG $WINDOW_ARG $EARLY_STOP_ARG \
                --snn_layers 1 --tau 0.1 --init_tau 2.0 --v_th 1.0 \
                --node_batch_size 256 --steps_per_epoch 20 --val_steps 10 \
                --epochs "${BASE_EPOCHS:-100}" --lr 1e-3 \
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
    echo; echo "=== link probe: frozen $M encoder -> $(lp_dir "$D") ==="
    srun --export=ALL python -m src.train.analyze_jepa --link_sources encoder \
        $LINK_COMMON --encoder_ckpt "$D/encoder.ckpt" --out_dir "$(lp_dir "$D")" \
        || echo "[warn] link probe failed for $D"
done

# --- 4. controls: random and supervised link sources -------------------------
if want controls && [[ "$PROBE_ONLY" == "1" && ! -f "$ENC_CKPT" ]]; then
    echo "[warn] PROBE_ONLY: controls skipped (no shared encoder)"
elif want controls; then
    CTRL_DIR="$(d controls)"
    mkdir -p "$CTRL_DIR"
    echo; echo "=== link controls (random + supervised) -> $CTRL_DIR ==="
    srun --export=ALL python -m src.train.analyze_jepa \
        --link_sources random supervised $LINK_COMMON \
        --encoder_ckpt "$ENC_CKPT" --out_dir "$CTRL_DIR" \
        || echo "[warn] link controls failed"
fi

# --- 5. summary for this seed ------------------------------------------------
if want summary; then
    echo; echo "=== seed $SEED summary ==="
    srun --export=ALL python scripts/linkpred/aggregate_linkpred.py \
        --dataset "$SHORT" --root "$ROOT" --seeds "$SEED" || true
fi

echo; echo "[done] $DATASET seed $SEED -> $ROOT"
echo "Aggregate across seeds with:"
echo "  python scripts/linkpred/aggregate_linkpred.py --dataset $SHORT --root $ROOT --seeds 42 43 44 --latex"
