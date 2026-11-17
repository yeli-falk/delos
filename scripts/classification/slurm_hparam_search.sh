#!/usr/bin/env bash
#SBATCH --job-name=cls-hparam
#SBATCH --output=logs/cls-hparam-%A_%a.out
#SBATCH --error=logs/cls-hparam-%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=20:00:00
#SBATCH --partition=gpu
#
# HYPERPARAMETER SEARCH for dynamic node classification. The classification twin
# of scripts/regression/slurm_hparam_search.sh: a SLURM array job where each task
# id decodes to one grid cell, each cell writes its own output dir, and
# aggregate_hparam.py ranks them.
#
# EVERY cell probes the VALIDATION split (--probe_split val -> state_probe_val.json).
# The test split is never read during the search, so the configuration carried
# into the final 3-seed table was not chosen on the numbers that table reports.
#
# Three stages, because they tune different things:
#
#   STAGE=predictor  (do this first)
#       Sweeps the predictor knobs on the FROZEN arm -- lr x num_masked_nodes x
#       depth x change_weight. The encoder is fixed, so cells differ only in the
#       predictor. The winner becomes BASE_HP for everything downstream.
#
#   STAGE=baselines  (independent of the other two)
#       Tunes the COMPETING methods -- GRACE / CLDG / DDGCL / IDOL / ... -- so the final table
#       is not "our tuned method vs their defaults". Those defaults came from
#       each paper's own datasets and were never tuned for this subset.
#
#   STAGE=variants   (after predictor)
#       Holds BASE_HP fixed and sweeps only the encoder-training knob each
#       variant actually has. `frozen` has none left, so it contributes a single
#       reference cell; unfreezing makes the encoder LR dominant.
#
#       VARIANT_SET picks which arms are in the grid:
#         old  -> the 4 legacy arms (frozen / ema / naive / vicreg), 16 cells,
#                 the exact grid + indices the earlier sweeps used
#         new  -> the JEPA-recipe arms only (ctx / tnorm / ema_cos / sigreg /
#                 lejepa / ijepa / lejepa_full), 32 cells
#         all  -> old followed by new (default), 48 cells. Indices 0-15 are the
#                 legacy cells unchanged, so an --array=0-15 submission still
#                 reproduces the old sweep and 16-47 appends the new arms.
#       Arms: ctx = frozen + context-encoder masking, tnorm = frozen + target
#       LayerNorm, ema_cos = EMA with cosine momentum, ema_tnorm / ema_cos_tnorm
#       = those two EMA arms + target LayerNorm (appended 2026-09-23, 12 cells,
#       so "new" is 44 cells and "all" 60), sigreg = stop-grad target
#       + SIGReg, lejepa = no stop-grad + SIGReg, ijepa = ema_cos + ctx + tnorm,
#       lejepa_full = lejepa + ctx + tnorm. `--list` prints every cell.
#
#   STAGE=ema        (after variants; the EMA arm's own, wider grid -- "phase A")
#       The variants stage gives EMA only 6 cells (encoder_lr x momentum over
#       3 x 2 values) and its winners sat on the grid EDGES, so the optimum may
#       lie outside it. This stage extends the momentum axis, 12 cells per arm:
#           encoder_lr   $LRS (the shared ladder, default 1e-4 1e-3 1e-2)
#           ema_momentum 0.99 0.999 0.9995 0.9999
#       EMA_ARMS picks the arm(s): "ema" (constant momentum, default), "ema_cos"
#       (the value is then the cosine START momentum), "ema_tnorm",
#       "ema_cos_tnorm", or any list of them. EMA_FREEZE="0 10 30" additionally
#       appends the same grid with --encoder_freeze_epochs 10 and 30 (encoder
#       held fixed for the first K epochs, EMA clock paused), 12 cells per
#       (arm, K); the K=0 cells keep their legacy indices.
#       Cells are written into the VARIANTS root by default, on purpose: they
#       reuse its shared pretrained encoder, the one overlapping cell
#       (5e-4 x 0.999) is skipped as already complete, and
#       `aggregate_hparam.py --by_arm` then ranks old and new EMA cells together.
#       Pass ROOT=... if that root predates a config change (attention mask,
#       node subset) -- cells from different configurations must never be mixed.
#
#   STAGE=ema_pred   (the predictor knobs, re-tuned UNDER a trained EMA encoder -- "phase B")
#       STAGE=predictor tunes lr / masking / change_weight on the FROZEN arm and
#       everything downstream inherits them, so they were never tuned with the
#       encoder moving. This stage holds the EMA target knobs fixed at ENC_HP
#       (default "--encoder_lr 5e-4 --ema_momentum 0.999"; phase A showed they
#       are flat) and sweeps, 24 cells per arm:
#           lr            $LRS (the shared grid, default 1e-4 1e-3 1e-2)
#           mask_frac     0 (= the --num_masked_nodes count in BASE_HP) 0.05 0.15 0.30
#           change_weight 0 2.0
#       mask_frac masks a FRACTION of the maskable nodes; the count-based grid
#       (20-50 nodes) is under 1% of full-N MOOC. Own root (<dataset>_ema_pred).
#       RUN IT ON TWO SEEDS: phase A showed single-seed val noise of ~0.01-0.02,
#       larger than the effects being ranked. Submit once per SEED;
#       aggregate_hparam.py averages cells that differ only in the seed. To skip
#       re-pretraining, copy the shared encoders in first:
#           cp -r outputs/cls_mooc/_shared_encoder_s4{2,3} outputs/cls_hparam/mooc_ema_pred/
#
# Usage:
#   DATASET=mooc STAGE=predictor bash scripts/classification/slurm_hparam_search.sh --list
#   DATASET=mooc STAGE=predictor sbatch --array=0-35%6 scripts/classification/slurm_hparam_search.sh
#   python scripts/classification/aggregate_hparam.py --root outputs/cls_hparam/mooc_predictor --emit_flags
#
#   DATASET=mooc STAGE=baselines sbatch --array=0-80%6 scripts/classification/slurm_hparam_search.sh
#
# The JEPA cells skip the trainer's end-of-run link probe (DOWNSTREAM=none,
# 2026-11-12): the cell's result is the val state probe below, nobody reads the
# link numbers, and on GossipCop that probe OOMed a 32 GB card. DOWNSTREAM=link_pred
# restores it.
#   python scripts/classification/aggregate_hparam.py --root outputs/cls_hparam/mooc_baselines --by_arm --emit_flags
#
#   DATASET=mooc STAGE=variants BASE_HP="<from predictor>" \
#       sbatch --array=0-59%4 scripts/classification/slurm_hparam_search.sh     # VARIANT_SET=all
#   DATASET=mooc STAGE=variants VARIANT_SET=new BASE_HP="<from predictor>" \
#       sbatch --array=0-43%4 scripts/classification/slurm_hparam_search.sh     # new arms only
#   python scripts/classification/aggregate_hparam.py --root outputs/cls_hparam/mooc_variants --by_arm --emit_flags
#
#   DATASET=mooc STAGE=ema BASE_HP="<from predictor>" \
#       sbatch --array=0-11%4 scripts/classification/slurm_hparam_search.sh     # EMA_ARMS=ema
#   DATASET=mooc STAGE=ema EMA_ARMS="ema ema_cos" BASE_HP="<from predictor>" \
#       sbatch --array=0-23%4 scripts/classification/slurm_hparam_search.sh
#   DATASET=mooc STAGE=ema EMA_ARMS="ema_cos ema_cos_tnorm" EMA_FREEZE="0 10 30" \
#       BASE_HP="<from predictor>" sbatch --array=0-71%4 scripts/classification/slurm_hparam_search.sh
#   Every cosine arm and BGRL anneal over EMA_ANNEAL_EPOCHS (default 50) epochs,
#   see below; cells trained before 2026-09-23 annealed over the whole (cut-off)
#   run and must be deleted to be re-run, they are NOT picked up as stale.
#   python scripts/classification/aggregate_hparam.py --root outputs/cls_hparam/mooc_variants --by_arm --emit_flags
#
#   for S in 42 43; do DATASET=mooc STAGE=ema_pred SEED=$S BATCH=2 BASE_HP="<from predictor>" \
#       sbatch --array=0-23%4 scripts/classification/slurm_hparam_search.sh; done
#   python scripts/classification/aggregate_hparam.py --root outputs/cls_hparam/mooc_ema_pred --emit_flags

set -uo pipefail

DATASET="${DATASET:-mooc}"
STAGE="${STAGE:-predictor}"
SEED="${SEED:-42}"
# One seed for the search: the ranking only needs to be consistent, and seeds
# are far better spent on the 3-seed final table than on the sweep.
BASE_HP="${BASE_HP:-}"
# ONE learning-rate grid shared by every arm that sweeps --lr: all baselines and
# DyG-JEPA's own predictor / ema_pred stages. Per-method grids centred on each
# paper's default put several winners on a grid EDGE (CLDG lowest, DDGCL / IDOL
# highest on MOOC) and made "who was tuned harder" depend on the method. A decade
# ladder spans every reference default (1e-4 DyG2Vec ... 5e-3 STEP / STD-MAE).
# Cell dirs are named by flag values, not index, so cells already on disk at
# 1e-3 / 1e-2 are reused. Override: LRS="3e-5 1e-4 3e-4" (e.g. METR-LA, from
# slurm_lr_probe.sh) -- but then use the SAME override for every stage.
LRS="${LRS:-1e-4 1e-3 1e-2}"
STATE_MODE="${STATE_MODE:-state}"
VARIANT_SET="${VARIANT_SET:-all}"
EMA_ARMS="${EMA_ARMS:-ema}"
# Encoder freeze epochs swept by STAGE=ema (outer loop, so the legacy 12 cells
# per arm keep their indices; "0 10 30" appends the two frozen-start variants).
EMA_FREEZE="${EMA_FREEZE:-0}"
# Cosine-anneal horizon for every EMA-target arm that anneals its momentum
# (DyG-JEPA ema_cos / ijepa / ema_cos_tnorm and the BGRL baseline). The
# trainers' default spans the whole --epochs budget, which early stopping
# (patience 20) rarely lets a run reach: on MOOC the ema_cos cells stopped at
# 85-240 of 300 epochs with the momentum still at ~0.997, i.e. the cosine arm
# was the constant arm. 50 epochs is below every observed run length, so the
# schedule finishes while the predictor still trains. 0 = whole run (legacy).
# Same value for every arm and method so no row gets its own schedule.
EMA_ANNEAL_EPOCHS="${EMA_ANNEAL_EPOCHS:-50}"
ANNEAL_ARG=""; [[ "$EMA_ANNEAL_EPOCHS" != "0" ]] && ANNEAL_ARG="--ema_anneal_epochs $EMA_ANNEAL_EPOCHS"
ENC_HP="${ENC_HP:---encoder_lr 5e-4 --ema_momentum 0.999}"

ema_arm_ok() {
    case "$1" in
        ema|ema_cos|ema_tnorm|ema_cos_tnorm) ;;
        *) echo "[FATAL] EMA_ARMS entries must be ema, ema_cos, ema_tnorm or ema_cos_tnorm, got '$1'" >&2
           exit 2 ;;
    esac
}

build_grid() {
    GRID=()
    if [[ "$STAGE" == "predictor" ]]; then
        for lr in $LRS; do
            for nm in 20 30 50; do
                for depth in 4 6; do
                    for cw in 0 2.0; do
                        GRID+=("frozen|--lr $lr --num_masked_nodes $nm --depth $depth --change_weight $cw")
                    done
                done
            done
        done
    elif [[ "$STAGE" == "variants" ]]; then
        if [[ "$VARIANT_SET" == "old" || "$VARIANT_SET" == "all" ]]; then
            # Legacy grid -- do NOT reorder: cell indices are on disk / in logs.
            # encoder_lr on the SAME ladder as --lr (LRS) since 2026-09-23; the
            # 1e-5/1e-4/5e-4 grid put the EMA winners on its top edge.
            GRID+=("frozen|")
            for elr in $LRS; do
                for m in 0.996 0.999; do
                    GRID+=("ema|--encoder_lr $elr --ema_momentum $m")
                done
                GRID+=("naive|--encoder_lr $elr")
                for vv in 5.0 25.0; do
                    GRID+=("vicreg|--encoder_lr $elr --vicreg_var $vv")
                done
            done
        fi
        if [[ "$VARIANT_SET" == "new" || "$VARIANT_SET" == "all" ]]; then
            # Frozen-encoder recipe arms: no encoder knob, one cell each.
            GRID+=("ctx|")
            GRID+=("tnorm|")
            for elr in $LRS; do
                # cosine anneal: sweep the START momentum (end is 1.0)
                for m in 0.996 0.999; do
                    GRID+=("ema_cos|--encoder_lr $elr --ema_momentum $m")
                    GRID+=("ijepa|--encoder_lr $elr --ema_momentum $m")
                done
                # SIGReg weight replaces vicreg_var as the collapse knob. The raw
                # statistic sits ~0.25-0.3 at init, so 0.5 / 2.0 bracket "as strong
                # as the prediction term".
                for sw in 0.5 2.0; do
                    GRID+=("sigreg|--encoder_lr $elr --sigreg $sw")
                    GRID+=("lejepa|--encoder_lr $elr --sigreg $sw")
                    GRID+=("lejepa_full|--encoder_lr $elr --sigreg $sw")
                done
            done
            # Appended 2026-09-23 (indices above unchanged): the EMA arms with
            # target LayerNorm but WITHOUT context masking -- isolates tnorm
            # from ctx, which the ijepa arm bundles. Same encoder_lr x momentum
            # grid as ema / ema_cos.
            for elr in $LRS; do
                for m in 0.996 0.999; do
                    GRID+=("ema_tnorm|--encoder_lr $elr --ema_momentum $m")
                    GRID+=("ema_cos_tnorm|--encoder_lr $elr --ema_momentum $m")
                done
            done
        fi
        if [[ "$VARIANT_SET" != "old" && "$VARIANT_SET" != "new" && "$VARIANT_SET" != "all" ]]; then
            echo "[FATAL] VARIANT_SET must be old, new or all, got '$VARIANT_SET'" >&2
            exit 2
        fi
    elif [[ "$STAGE" == "ema" ]]; then
        # Phase A: the EMA arm's target-network knobs, both edges of the
        # variants grid extended. Do NOT reorder: indices are on disk / in logs.
        for fz in $EMA_FREEZE; do
            FZ_ARG=""; [[ "$fz" != "0" ]] && FZ_ARG=" --encoder_freeze_epochs $fz"
            for arm in $EMA_ARMS; do
                ema_arm_ok "$arm"
                # encoder_lr on the shared ladder since 2026-09-23 (was
                # 5e-4/1e-3/2e-3, which did not contain the variants-stage
                # winner 1e-4); the momentum axis is this stage's own.
                for elr in $LRS; do
                    for m in 0.99 0.999 0.9995 0.9999; do
                        GRID+=("$arm|--encoder_lr $elr --ema_momentum $m$FZ_ARG")
                    done
                done
            done
        done
    elif [[ "$STAGE" == "ema_pred" ]]; then
        # Phase B: predictor knobs under a TRAINED EMA encoder; the target knobs
        # stay at ENC_HP. Do NOT reorder: indices are on disk / in logs.
        for arm in $EMA_ARMS; do
            ema_arm_ok "$arm"
            for lr in $LRS; do
                for mf in 0 0.05 0.15 0.30; do
                    for cw in 0 2.0; do
                        GRID+=("$arm|--lr $lr --mask_frac $mf --change_weight $cw $ENC_HP")
                    done
                done
            done
        done
    elif [[ "$STAGE" == "baselines" ]]; then
        # LR plus each method's most load-bearing knob, matching the regression
        # sweep's shape. Cheap: none of these run the T*N-token predictor.
        for lr in $LRS; do
            for v in 2 4; do GRID+=("cldg|--lr $lr --views $v"); done
        done
        for lr in $LRS; do
            for g in 2 4 8; do GRID+=("ddgcl|--lr $lr --max_gap $g"); done
        done
        for lr in $LRS; do
            for pl in 1 2; do GRID+=("idol|--lr $lr --proj_layers $pl"); done
        done
        # MaskDGNN: LR x mask ratio (the paper's most sensitive knob; best at
        # 10-30%, collapses past 40%).
        for lr in $LRS; do
            for mr in 0.1 0.2 0.4; do GRID+=("maskdgnn|--lr $lr --mask_ratio $mr"); done
        done
        # DySubC: LR x alpha (the sampling trade-off between recency and
        # degree; the paper's most sensitive knob after subgraph size).
        for lr in $LRS; do
            for a in 1 5 10; do GRID+=("dysubc|--lr $lr --alpha $a"); done
        done
        # DyG2Vec: LR x distortion strength (p_d, the only knob the paper
        # validates; 0.3 is its choice). Both dropouts move together.
        for lr in $LRS; do
            for pd in 0.15 0.3; do GRID+=("dyg2vec|--lr $lr --p_edge $pd --p_struct $pd"); done
        done
        # DyG-WM: LR x EMA momentum (the target encoder's own stability knob;
        # the paper uses 0.995 for node tasks and 0.999 for links).
        for lr in $LRS; do
            for m in 0.995 0.999; do GRID+=("dygwm|--lr $lr --ema_momentum $m"); done
        done
        # SG-JEPA: LR x InfoNCE temperature (the paper names no temperature).
        for lr in $LRS; do
            for t in 0.1 0.5; do GRID+=("sgjepa|--lr $lr --tau $t"); done
        done
        # GRACE (static control): LR x tau. The temperature is the paper's
        # load-bearing knob after the drop rates, which stay at the Cora defaults.
        for lr in $LRS; do
            for t in 0.2 0.4 0.7; do GRID+=("grace|--lr $lr --tau $t"); done
        done
        # GraphMAE (static control): LR x mask rate, the paper's headline knob
        # (robust 0.3-0.75 on citation graphs; lower on feature-poor inputs).
        for lr in $LRS; do
            for mr in 0.3 0.5 0.75; do GRID+=("graphmae|--lr $lr --mask_rate $mr"); done
        done
        # BGRL (static control): LR x initial EMA momentum, the bootstrap's own
        # stability knob.
        for lr in $LRS; do
            for m in 0.99 0.999; do GRID+=("bgrl|--lr $lr --ema_momentum $m"); done
        done
    else
        echo "[FATAL] STAGE must be 'predictor', 'variants', 'ema', 'ema_pred' or 'baselines', " \
             "got '$STAGE'" >&2
        exit 2
    fi
}

build_grid

if [[ "${1:-}" == "--list" ]]; then
    echo "DATASET=$DATASET STAGE=$STAGE VARIANT_SET=$VARIANT_SET EMA_ARMS='$EMA_ARMS'  grid size=${#GRID[@]}"
    echo "  ->  sbatch --array=0-$(( ${#GRID[@]} - 1 ))%6 $0"
    for i in "${!GRID[@]}"; do printf "  [%2d] %s\n" "$i" "${GRID[$i]}"; done
    # exact --array= index list per arm (contiguous runs collapsed to a-b),
    # for a partial submission; arms interleave in some stages, so a plain
    # first-last range would be wrong there.
    declare -A _idx; _order=()
    for i in "${!GRID[@]}"; do
        a="${GRID[$i]%%|*}"
        [[ -v _idx[$a] ]] || _order+=("$a")
        _idx[$a]="${_idx[$a]:+${_idx[$a]} }$i"
    done
    echo "  per-arm --array= lists (or set ONLY_ARMS='<arm> <arm>' with the full array):"
    for a in "${_order[@]}"; do
        out=""; lo=""; prev=""
        for i in ${_idx[$a]}; do
            if [[ -z "$lo" ]]; then lo=$i; prev=$i
            elif (( i == prev + 1 )); then prev=$i
            else out+="${out:+,}$([[ $lo == $prev ]] && echo "$lo" || echo "$lo-$prev")"; lo=$i; prev=$i
            fi
        done
        out+="${out:+,}$([[ $lo == $prev ]] && echo "$lo" || echo "$lo-$prev")"
        printf "    %-14s %s\n" "$a" "$out"
    done
    exit 0
fi

mkdir -p logs
REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"
# Full-N adjacency tensors are multi-GB; expandable segments stop the caching
# allocator from stranding reserved-but-unusable blocks between them.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# Interpreter. Three ways to point the job at the right python, in priority:
#   PY=/path/to/python          use this interpreter outright (no activation)
#   CONDA_ENV=<name|/path>      activate this conda env, then use its python
#   (neither)                   activate ~/.conda/envs/cjepa if it exists (ruche)
# On a cluster where none of these resolve, srun fails with
# "execve(): python: No such file or directory" -- hence the hard check.
CONDA_ENV="${CONDA_ENV:-}"
if [[ -z "${PY:-}" ]]; then
    [[ -z "$CONDA_ENV" && -d "$HOME/.conda/envs/cjepa" ]] && CONDA_ENV=cjepa
    if [[ -n "$CONDA_ENV" ]]; then
        CONDA_BASE="$(conda info --base 2>/dev/null || true)"
        [[ -n "$CONDA_BASE" ]] && source "$CONDA_BASE/etc/profile.d/conda.sh"
        conda activate "$CONDA_ENV" || { echo "[FATAL] conda activate $CONDA_ENV failed" >&2; exit 4; }
    fi
    PY=python
fi
command -v "$PY" >/dev/null 2>&1 \
    || { echo "[FATAL] interpreter '$PY' not found. Set PY=/path/to/python or CONDA_ENV=<name|path>" >&2; exit 4; }
echo "[env] python=$(command -v "$PY")  $("$PY" -c 'import torch; print("torch", torch.__version__)' 2>&1 | tail -1)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

IDX="${SLURM_ARRAY_TASK_ID:-0}"
if (( IDX >= ${#GRID[@]} )); then
    echo "[error] array index $IDX >= grid size ${#GRID[@]}" >&2; exit 2
fi
CELL="${GRID[$IDX]}"
ARM="${CELL%%|*}"
HP_ARGS="${CELL#*|}"
# ONLY_ARMS="maskdgnn dyg2vec": run just those arms' cells, skipping the rest
# of the array as no-ops. Indices stay the same as in --list, so a later full
# submission reuses the cells already on disk. Alternative without the
# variable: submit the index ranges --list prints, e.g. --array=21-29,39-44.
if [[ -n "${ONLY_ARMS:-}" && " $ONLY_ARMS " != *" $ARM "* ]]; then
    echo "[skip] cell $IDX is arm '$ARM', not in ONLY_ARMS='$ONLY_ARMS'"; exit 0
fi

CSV="dataset/jodie/${DATASET}.csv"
case "$DATASET" in
    mooc|reddit|wikipedia)
        [[ -f "$CSV" ]] || { echo "[FATAL] $CSV missing; fetch it from the login node" >&2; exit 10; } ;;
    *) ;;   # synthetic_dyn needs no file; cldg_* is checked by _dataset_config.sh
esac

TASK=state_probe
source "$REPO_ROOT/scripts/dblp/_dataset_config.sh"
BATCH="${BATCH:-8}"
NW="${SLURM_CPUS_PER_TASK:-8}"
SHORT="${DATASET#jodie_}"

# Sourcing _dataset_config.sh can clobber loop variables, so re-derive the arm
# flags AFTER it -- a clobbered $ARM would mislabel a whole cell.
arm_flags() {
    case "$1" in
        frozen)      echo "" ;;
        naive)       echo "--train_encoder" ;;
        ema)         echo "--train_encoder --ema_target" ;;
        vicreg)      echo "--train_encoder --vicreg_cov 1.0" ;;
        input)       echo "--predict_space input" ;;
        # --- JEPA-recipe variants (2026-09): each flag is opt-in in
        # train_dyg_jepa.py, so the five arms above are byte-for-byte the old ones.
        ctx)         echo "--mask_encoder_input" ;;
        tnorm)       echo "--target_norm" ;;
        ema_cos)     echo "--train_encoder --ema_target --ema_schedule cosine $ANNEAL_ARG" ;;
        # EMA + target LayerNorm, no context masking (2026-09-23): the
        # I-JEPA target normalisation on its own, on top of each EMA arm.
        ema_tnorm)     echo "--train_encoder --ema_target --target_norm" ;;
        ema_cos_tnorm) echo "--train_encoder --ema_target --ema_schedule cosine $ANNEAL_ARG --target_norm" ;;
        sigreg)      echo "--train_encoder --sigreg 1.0" ;;
        lejepa)      echo "--train_encoder --target_grad --sigreg 1.0" ;;
        ijepa)       echo "--train_encoder --ema_target --ema_schedule cosine $ANNEAL_ARG --mask_encoder_input --target_norm" ;;
        lejepa_full) echo "--train_encoder --target_grad --sigreg 1.0 --mask_encoder_input --target_norm" ;;
        *)           echo "__UNKNOWN__" ;;
    esac
}

SLUG=$(echo "$HP_ARGS" | sed 's/--//g; s/[ .]/_/g; s/__*/_/g; s/^_//; s/_$//')
[[ -z "$SLUG" ]] && SLUG=base
# STAGE=ema shares the variants root (encoder reuse + joint ranking, see header).
ROOT_STAGE="$STAGE"; [[ "$STAGE" == "ema" ]] && ROOT_STAGE=variants
ROOT="${ROOT:-outputs/cls_hparam/${SHORT}_${ROOT_STAGE}}"
OUT="$ROOT/${ARM}_${SLUG}_s${SEED}"
mkdir -p "$OUT"
printf '%s\n' "arm=$ARM flags=$HP_ARGS base=$BASE_HP" > "$OUT/hp_args.txt"
echo "[cell $IDX] dataset=$DATASET stage=$STAGE arm=$ARM seed=$SEED -> $OUT"

# synthetic_dyn exposes a dynamic MULTICLASS state (see slurm_all_methods.sh);
# rank its cells with aggregate_hparam.py --metric acc_test (or
# subset_switch.acc_test for the anticipation-only view).
# cldg_dblp / cldg_patent: static per-node classes, --label_source dataset
# (node-level split); rank with --metric acc_test or macro_f1_test.
PROBE_TASK=binary; PROBE_LABELS=""
[[ "$DATASET" == "synthetic_dyn" && "${SYN_STATE_TARGET:-community}" != "switch" ]] && PROBE_TASK=multiclass
[[ "$DATASET" == cldg_* ]] && { PROBE_TASK=multiclass; PROBE_LABELS="--label_source dataset"; }
# chase_music: 5-class dynamic state (rank with --metric acc_test); the other
# two CHASE graphs are binary.
[[ "$DATASET" == "chase_music" ]] && PROBE_TASK=multiclass
# tgbn_trade: per-step ranking, rank cells with aggregate_hparam.py --metric ndcg10_test
[[ "$DATASET" == tgbn_* ]] && PROBE_TASK=ranking
PROBE_COMMON="--task $PROBE_TASK $PROBE_LABELS --state_label_mode $STATE_MODE --probe_split val \
    --device cuda --batch_size ${PROBE_BATCH:-4} --num_workers $NW"

# --- STAGE=baselines: one competing method per cell, probed on VAL -----------
if [[ "$STAGE" == "baselines" ]]; then
    # A cell is complete once its val metrics exist (state_probe_val.json is the last
    # thing a cell writes), so resubmitting the array -- after a walltime kill or
    # a grid change -- re-runs only the missing cells. Delete the dir to force one.
    if [[ -f "$OUT/state_probe_val.json" ]]; then
        echo "[skip] cell $IDX ($ARM) complete: $OUT/state_probe_val.json exists"; exit 0
    fi
    case "$ARM" in
      cldg)
        srun --export=ALL "$PY" -m src.baselines.cldg.train_cldg $DATA_ARG \
            $ENCODER_ARG $WINDOW_ARG $EARLY_STOP_ARG \
            --snapshots 4 --strategy random --tau 0.07 \
            --node_batch_size 256 --steps_per_epoch 20 --val_steps 10 \
            --epochs "${BASE_EPOCHS:-100}" --device cuda --seed "$SEED" \
            --out_dir "$OUT" $HP_ARGS \
            || { echo "[FATAL] CLDG cell failed" >&2; exit 21; } ;;
      ddgcl)
        srun --export=ALL "$PY" -m src.baselines.ddgcl.train_ddgcl $DATA_ARG \
            $ENCODER_ARG $WINDOW_ARG $EARLY_STOP_ARG \
            --time_dim 32 --tau 0.01 --beta 1.5 \
            --node_batch_size 256 --steps_per_epoch 20 --val_steps 10 \
            --epochs "${BASE_EPOCHS:-100}" --device cuda --seed "$SEED" \
            --out_dir "$OUT" $HP_ARGS \
            || { echo "[FATAL] DDGCL cell failed" >&2; exit 22; } ;;
      idol)
        srun --export=ALL "$PY" -m src.baselines.idol.train_idol $DATA_ARG \
            $ENCODER_ARG $WINDOW_ARG $EARLY_STOP_ARG \
            --node_batch_size 256 --steps_per_epoch 20 --val_steps 10 \
            --epochs "${BASE_EPOCHS:-100}" --device cuda --seed "$SEED" \
            --out_dir "$OUT" $HP_ARGS \
            || { echo "[FATAL] IDOL cell failed" >&2; exit 23; } ;;
      maskdgnn)
        srun --export=ALL "$PY" -m src.baselines.maskdgnn.train_maskdgnn $DATA_ARG \
            $ENCODER_ARG --encoder_kind maskdgnn $WINDOW_ARG $EARLY_STOP_ARG \
            --alpha 0.7 --target next --neg_per_pos 1 \
            --steps_per_epoch 20 --val_steps 10 --e2e_eval none \
            --epochs "${BASE_EPOCHS:-100}" --device cuda --seed "$SEED" \
            --out_dir "$OUT" $HP_ARGS \
            || { echo "[FATAL] MaskDGNN cell failed" >&2; exit 25; } ;;
      dysubc)
        srun --export=ALL "$PY" -m src.baselines.dysubc.train_dysubc $DATA_ARG \
            $ENCODER_ARG $WINDOW_ARG $EARLY_STOP_ARG \
            --subgraph_size 20 --beta 1 --margin 0.75 --lam 0.5 \
            --node_batch_size 256 --steps_per_epoch 20 --val_steps 10 \
            --epochs "${BASE_EPOCHS:-100}" --device cuda --seed "$SEED" \
            --out_dir "$OUT" $HP_ARGS \
            || { echo "[FATAL] DySubC cell failed" >&2; exit 26; } ;;
      dyg2vec)
        srun --export=ALL "$PY" -m src.baselines.dyg2vec.train_dyg2vec $DATA_ARG \
            $ENCODER_ARG --encoder_kind dyg2vec $WINDOW_ARG $EARLY_STOP_ARG \
            --lam 25 --mu 25 --nu 1 --steps_per_epoch 20 --val_steps 10 \
            --epochs "${BASE_EPOCHS:-100}" --device cuda --seed "$SEED" \
            --out_dir "$OUT" $HP_ARGS \
            || { echo "[FATAL] DyG2Vec cell failed" >&2; exit 29; } ;;
      dygwm)
        srun --export=ALL "$PY" -m src.baselines.dygwm.train_dygwm $DATA_ARG \
            $ENCODER_ARG --encoder_kind dygwm $WINDOW_ARG $EARLY_STOP_ARG \
            --n_ctx 32 --decay 0.8 --cn_bonus 1.0 \
            --lam_node 1 --lam_rel 1 --lam_var 0.25 --lam_cov 0.025 \
            --node_batch_size 256 --pair_batch_size 256 \
            --steps_per_epoch 20 --val_steps 10 \
            --epochs "${BASE_EPOCHS:-100}" --device cuda --seed "$SEED" \
            --out_dir "$OUT" $HP_ARGS \
            || { echo "[FATAL] DyG-WM cell failed" >&2; exit 31; } ;;
      sgjepa)
        srun --export=ALL "$PY" -m src.baselines.sgjepa.train_sgjepa $DATA_ARG \
            $ENCODER_ARG $WINDOW_ARG $EARLY_STOP_ARG \
            --snn_layers 1 --init_tau 2.0 --v_th 1.0 \
            --node_batch_size 256 --steps_per_epoch 20 --val_steps 10 \
            --epochs "${BASE_EPOCHS:-100}" --device cuda --seed "$SEED" \
            --out_dir "$OUT" $HP_ARGS \
            || { echo "[FATAL] SG-JEPA cell failed" >&2; exit 32; } ;;
      bgrl)
        srun --export=ALL "$PY" -m src.baselines.bgrl.train_bgrl $DATA_ARG \
            $ENCODER_ARG $WINDOW_ARG $EARLY_STOP_ARG \
            --p_e1 0.2 --p_e2 0.3 --p_f1 0.2 --p_f2 0.1 \
            --steps_per_epoch 20 --val_steps 10 \
            --epochs "${BASE_EPOCHS:-100}" --device cuda --seed "$SEED" \
            --out_dir "$OUT" $ANNEAL_ARG $HP_ARGS \
            || { echo "[FATAL] BGRL cell failed" >&2; exit 30; } ;;
      graphmae)
        srun --export=ALL "$PY" -m src.baselines.graphmae.train_graphmae $DATA_ARG \
            $ENCODER_ARG $WINDOW_ARG $EARLY_STOP_ARG \
            --replace_rate 0.05 --sce_gamma 2 --target all \
            --steps_per_epoch 20 --val_steps 10 \
            --epochs "${BASE_EPOCHS:-100}" --device cuda --seed "$SEED" \
            --out_dir "$OUT" $HP_ARGS \
            || { echo "[FATAL] GraphMAE cell failed" >&2; exit 29; } ;;
      grace)
        srun --export=ALL "$PY" -m src.baselines.grace.train_grace $DATA_ARG \
            $ENCODER_ARG $WINDOW_ARG $EARLY_STOP_ARG \
            --p_e1 0.2 --p_e2 0.4 --p_f1 0.3 --p_f2 0.4 \
            --node_batch_size 256 --steps_per_epoch 20 --val_steps 10 \
            --epochs "${BASE_EPOCHS:-100}" --device cuda --seed "$SEED" \
            --out_dir "$OUT" $HP_ARGS \
            || { echo "[FATAL] GRACE cell failed" >&2; exit 28; } ;;
      *) echo "[FATAL] unknown baseline arm '$ARM'" >&2; exit 3 ;;
    esac
    srun --export=ALL "$PY" scripts/dblp/run_block_probe.py \
        --encoder_ckpt "$OUT/encoder.ckpt" $PROBE_COMMON \
        || { echo "[FATAL] probe failed" >&2; exit 24; }
    [[ -f "$OUT/state_probe_val.json" ]] || { echo "[FATAL] cell $IDX wrote no metrics" >&2; exit 12; }
    echo "[done] cell $IDX ($ARM) -> $OUT"
    exit 0
fi

VFLAGS="$(arm_flags "$ARM")"
[[ "$VFLAGS" == "__UNKNOWN__" ]] && { echo "[FATAL] bad arm '$ARM'" >&2; exit 3; }

# --- shared pretrained encoder ----------------------------------------------
# Every cell starts from THIS encoder, so the sweep measures the predictor /
# encoder-training knobs and not pretraining variance. flock because concurrent
# array cells would otherwise all pretrain it at once.
ENC_DIR="$ROOT/_shared_encoder_s${SEED}"
ENC_CKPT="$ENC_DIR/encoder_${DATASET}_${ENC_KIND}_edge_mask.ckpt"
mkdir -p "$ENC_DIR"
exec 9>"$ENC_DIR/.pretrain.lock"
flock -w 21600 9 || { echo "[FATAL] encoder lock timeout" >&2; exit 10; }
# Same completion rule as slurm_all_methods.sh: the ckpt is saved on every val
# improvement, so only the `.done` marker (touched after a clean exit) proves
# the pretrain finished. ENC_TRUST=1 accepts a marker-less ckpt as complete.
if [[ -f "$ENC_CKPT" && -f "$ENC_DIR/.done" ]]; then
    echo "[skip] shared encoder complete"
elif [[ -f "$ENC_CKPT" && "${ENC_TRUST:-0}" == "1" ]]; then
    echo "[skip] shared encoder exists without .done marker; ENC_TRUST=1 -> keeping it"
    touch "$ENC_DIR/.done"
else
    [[ -f "$ENC_CKPT" ]] && echo "[resume] $ENC_CKPT has no .done marker (interrupted run?) -> retraining"
    echo "=== shared encoder pretrain (lock held by cell $IDX) ==="
    rm -f "$ENC_CKPT"
    if srun --export=ALL "$PY" src/train/pretrain_encoder.py $DATA_ARG \
        $ENCODER_ARG $WINDOW_ARG $EARLY_STOP_ARG \
        --pretext edge_mask --decoder_hidden 64 --mask_rate 0.25 \
        --batch_size "$BATCH" --epochs "${ENC_EPOCHS:-300}" --lr 1e-3 \
        --seed "$SEED" --num_workers "$NW" --device cuda --out_dir "$ENC_DIR" \
       && [[ -f "$ENC_CKPT" ]]; then
        touch "$ENC_DIR/.done"
    else
        flock -u 9; echo "[FATAL] encoder pretrain failed" >&2; exit 11
    fi
fi
flock -u 9; exec 9>&-

# --- train + probe THE VAL SPLIT --------------------------------------------
# A cell is complete once its val probe wrote state_probe_val.json; resubmitting
# the array (e.g. after a walltime kill) then re-runs only the unfinished cells.
if [[ -f "$OUT/state_probe_val.json" ]]; then
    echo "[skip] cell $IDX complete: $OUT/state_probe_val.json exists"; exit 0
fi
# Same completion rule as slurm_all_methods.sh: the trainer writes
# test_metrics.json only after its last epoch + link probe, so best.ckpt AND
# test_metrics.json = training finished cleanly (only the probe is missing,
# e.g. it OOMed -- rerun with a smaller PROBE_BATCH); best.ckpt alone = killed
# mid-training (saved on every val improvement) -> restart from scratch.
if [[ -f "$OUT/best.ckpt" && -f "$OUT/test_metrics.json" ]]; then
    echo "[skip] cell $IDX training complete (best.ckpt + test_metrics.json); running only the probe"
else
    if [[ -f "$OUT/best.ckpt" ]]; then
        echo "[resume] cell $IDX has best.ckpt but no test_metrics.json (interrupted) -> restarting from scratch"
        rm -f "$OUT/best.ckpt" "$OUT/history.json"
    fi
    srun --export=ALL "$PY" src/train/train_dyg_jepa.py \
        $DATA_ARG $ENCODER_ARG $WINDOW_ARG $EARLY_STOP_ARG \
        --encoder_ckpt "$ENC_CKPT" --mask_scheme "${MASK_SCHEME:-node_mask}" \
        --residual_head \
        --heads "${HEADS:-4}" --mlp_dim "${MLP_DIM:-512}" \
        --batch_size "$BATCH" --epochs "${JEPA_EPOCHS:-300}" \
        --probe_epochs "${JEPA_PROBE_EPOCHS:-3}" \
        --downstream_task "${DOWNSTREAM:-none}" \
        --seed "$SEED" --num_workers "$NW" --device cuda --out_dir "$OUT" \
        $BASE_HP $VFLAGS $HP_ARGS
    [[ -f "$OUT/best.ckpt" ]] || { echo "[FATAL] cell $IDX produced no best.ckpt" >&2; exit 12; }
fi

srun --export=ALL "$PY" scripts/dblp/run_block_probe.py \
    --jepa_ckpt "$OUT/best.ckpt" $PROBE_COMMON \
    || { echo "[FATAL] probe failed" >&2; exit 13; }
[[ -f "$OUT/state_probe_val.json" ]] || { echo "[FATAL] cell $IDX wrote no metrics" >&2; exit 12; }

echo "[done] cell $IDX -> $OUT"
