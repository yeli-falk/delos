#!/usr/bin/env bash
#SBATCH --job-name=cjepa-covid
#SBATCH --output=logs/cjepa-covid-%j.out
#SBATCH --error=logs/cjepa-covid-%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:30:00
# --partition=gpu                  # uncomment + set to your cluster's partition
# --qos=gpu                        # uncomment if QoS is required
# --account=<your-account>         # uncomment if accounting is required
#
# Full England COVID comparison in one job: parameter-free baselines ->
# pretrain encoder (masked_signal) -> DyG-JEPA (random-init encoder) ->
# DyG-JEPA (pretrained encoder). All stages share history_size=8 / num_preds=3
# so the test windows (and the MAE/RMSE/MAPE numbers) line up across the table.
#
# England COVID is small (129 nodes, 61 daily snapshots, time-varying mobility
# graph), so this needs far less compute/time than METR-LA. The dataset JSON is
# auto-downloaded + cached under dataset/england_covid/ on first use; if the
# compute nodes have no internet, rsync dataset/england_covid/england_covid.json
# to the cluster beforehand.
#
# Submit with:    sbatch scripts/england_covid/slurm_train.sh
# Inspect:        squeue -u $USER ; tail -f logs/cjepa-covid-<jobid>.out
# Cancel:         scancel <jobid>

set -euo pipefail

# Seed-aware output dirs so a multi-seed sweep doesn't overwrite itself:
#   for SEED in 42 43 44; do SEED=$SEED ... sbatch scripts/england_covid/slurm_train.sh; done
SEED="${SEED:-0}"
SUF="_s${SEED}"
ENC_DIR="outputs/england_covid_encoder${SUF}"
RAND_DIR="outputs/england_covid_cjepa_randenc${SUF}"
PRE_DIR="outputs/england_covid_cjepa_pretrained${SUF}"
mkdir -p logs outputs/england_covid_baselines "$ENC_DIR" "$RAND_DIR" "$PRE_DIR"

# ---------------------------------------------------------------------------
# Environment activation. Pick the path that matches how your cluster works:
#   1) module-managed PyTorch:    module load pytorch/2.5
#   2) personal conda env:        conda activate cjepa
#   3) virtualenv:                source ~/envs/cjepa/bin/activate
# ---------------------------------------------------------------------------
if [[ -n "${MODULESHOME:-}" ]] && command -v module >/dev/null 2>&1; then
    : # module load python/3.10  pytorch/2.5  cuda/12.1   # customise here
fi

if [[ -d "$HOME/.conda/envs/cjepa" ]]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate cjepa
fi

# Resolve the repo root from where sbatch was invoked, so it doesn't matter
# whether you submit from inside cjepa/ or from its parent.
REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"
echo "[slurm] repo_root=$REPO_ROOT"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

HS="${HS:-8}"
NP="${NP:-3}"
NW="${SLURM_CPUS_PER_TASK:-4}"
ENC_CKPT="$ENC_DIR/encoder_england_covid_tgcn_masked_signal.ckpt"

# --- predictor-objective recipe (see train_dyg_jepa.py --help) ---------------
# The MOOC-validated fix, ported to regression. PERSIST_RES is the load-bearing
# one here: it makes the probe predict the delta over the last observed value,
# so persistence is the FLOOR (without it a latent->signal probe historically
# landed ~1.8x WORSE than persistence). RESIDUAL + CHANGE_W add the latent
# residual head and transition-upweighting; TRAIN_ENC unfreezes the encoder.
#   RESIDUAL=1 CHANGE_W=2.0 PERSIST_RES=1 sbatch scripts/england_covid/slurm_train.sh
RESIDUAL="${RESIDUAL:-0}"
CHANGE_W="${CHANGE_W:-0}"
PERSIST_RES="${PERSIST_RES:-0}"
TRAIN_ENC="${TRAIN_ENC:-0}"
JEPA_EXTRA=""
[[ "$RESIDUAL" == "1" ]]    && JEPA_EXTRA="$JEPA_EXTRA --residual_head"
[[ "$CHANGE_W" != "0" ]]    && JEPA_EXTRA="$JEPA_EXTRA --change_weight $CHANGE_W"
[[ "$PERSIST_RES" == "1" ]] && JEPA_EXTRA="$JEPA_EXTRA --persist_residual"
[[ "$TRAIN_ENC" == "1" ]]   && JEPA_EXTRA="$JEPA_EXTRA --train_encoder --ema_target"
echo "[config] SEED=$SEED HS=$HS NP=$NP JEPA_EXTRA='$JEPA_EXTRA'"

# Use `srun --export=ALL` so PYTHONPATH propagates into the task on clusters
# that strip env vars by default. 129 nodes x (8+3) steps keeps the structural
# attention mask small, so we leave it ON (no --no_attention_mask needed).

echo "================ 1/4  parameter-free baselines ================"
srun --export=ALL python scripts/regression/run_baselines.py \
    --history_size $HS --num_preds $NP \
    --out_dir outputs/england_covid_baselines

echo "================ 2/4  pretrain encoder (masked_signal) ================"
srun --export=ALL python src/train/pretrain_encoder.py --data_source england_covid \
    --pretext masked_signal --encoder_kind tgcn --bidirectional false \
    --history_size $HS --num_preds $NP \
    --node_dim 128 --encoder_hidden 128 --mask_rate 0.25 \
    --batch_size 8 --epochs 300 --lr 1e-3 --early_stop_patience 30 \
    --seed "$SEED" --num_workers "$NW" --device cuda \
    --out_dir "$ENC_DIR" --plot_losses

# The pretrained stage below hard-depends on this checkpoint. If pretraining
# didn't write it, say so loudly but keep the job's baseline + random-init
# results rather than dying on an opaque FileNotFoundError.
if [[ ! -f "$ENC_CKPT" ]]; then
    echo "[error] pretrain produced no checkpoint at $ENC_CKPT" >&2
    ls -la "$ENC_DIR/" >&2 || true
fi

echo "================ 3/4  DyG-JEPA (random-init encoder) ================"
srun --export=ALL python src/train/train_dyg_jepa.py --data_source england_covid \
    --downstream_task regression --mask_scheme node_mask $JEPA_EXTRA \
    --history_size $HS --num_preds $NP \
    --encoder_kind tgcn --bidirectional false --node_dim 128 --encoder_hidden 128 \
    --num_masked_nodes 20 --depth 6 --heads 8 --mlp_dim 512 \
    --batch_size 8 --epochs 50 --lr 5e-4 --weight_decay 1e-4 \
    --probe_epochs 10 --probe_hidden 64 \
    --seed "$SEED" --num_workers "$NW" --device cuda \
    --out_dir "$RAND_DIR"

echo "================ 4/4  DyG-JEPA (pretrained encoder) ================"
if [[ -f "$ENC_CKPT" ]]; then
    srun --export=ALL python src/train/train_dyg_jepa.py --data_source england_covid \
        --downstream_task regression --mask_scheme node_mask $JEPA_EXTRA \
        --history_size $HS --num_preds $NP \
        --encoder_kind tgcn --bidirectional false --node_dim 128 --encoder_hidden 128 \
        --encoder_ckpt "$ENC_CKPT" \
        --num_masked_nodes 20 --depth 6 --heads 8 --mlp_dim 512 \
        --batch_size 8 --epochs 50 --lr 5e-4 --weight_decay 1e-4 \
        --probe_epochs 10 --probe_hidden 64 \
        --seed "$SEED" --num_workers "$NW" --device cuda \
        --out_dir "$PRE_DIR"
else
    echo "[skip] no encoder checkpoint at $ENC_CKPT; skipping pretrained stage." >&2
    echo "       Re-run the pretrain stage (2/4) or drop --encoder_ckpt to use random init." >&2
fi

echo "================ done ================"
echo "Baselines : outputs/england_covid_baselines/baseline_metrics.json"
echo "DyG-JEPA  : outputs/england_covid_cjepa_{randenc,pretrained}/test_metrics.json"
