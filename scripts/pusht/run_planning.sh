#!/bin/bash
#SBATCH --job-name=planning
#SBATCH --time=5-00:00:00
#SBATCH --partition=gpus
#SBATCH --ntasks=4
#SBATCH --gres=gpu:nvidia_titan_rtx:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

echo "SLURM job started on: $(date)"
echo "Node list: $SLURM_NODELIST"

export PYTHONPATH=$(pwd)

SEEDS=(0 1 2)
EXPNUM=170

for SEED in "${SEEDS[@]}"; do
    echo "Running experiment with seed=${SEED}"
# policy=${EXPNUM}p_epoch_30 
# output.filename=cjepa_pusht_from_${EXPNUM}_seed_${SEED}_defaultset.txt \


# CAUTION!!!!!!!!!! Currently 10 epochs 
    HYDRA_FULL_ERROR=1 
    STABLEWM_HOME=/cs/data/people/hnam16/.stable_worldmodel/ 
    python plan/run.py \
        seed=${SEED} \
        policy=${EXPNUM}_epoch_30 \
        world.history_size=1 \
        world.frame_skip=1 \
        plan_config.horizon=3 \
        plan_config.receding_horizon=3 \
        plan_config.action_block=5 \
        eval.eval_budget=50 \
        output.filename=cjepa_pusht_from_${EXPNUM}_seed_${SEED}_horizon3.txt \
        eval.dataset_name=pusht_expert_train \
        eval.goal_offset_steps=25
    # python -m pdb plan/run.py \
    #     seed=${SEED} \
    #     policy=${EXPNUM}_epoch_30 \
    #     world.history_size=1 \
    #     world.frame_skip=1 \
    #     plan_config.horizon=5 \
    #     plan_config.receding_horizon=5 \
    #     plan_config.action_block=5 \
    #     eval.eval_budget=50 \
    #     output.filename=cjepa_pusht_from_${EXPNUM}_seed_${SEED}.txt \
    #     eval.dataset_name=pusht_expert_val
done

echo "All experiments finished on: $(date)"

