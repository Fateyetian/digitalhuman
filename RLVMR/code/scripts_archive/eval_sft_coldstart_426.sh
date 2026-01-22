#!/bin/bash
# Evaluation Script for SFT Cold Start Model (426 trajectories)
# Usage: bash eval_sft_coldstart_426.sh [checkpoint_path] [num_tasks]

set -x

# 参数配置
CHECKPOINT_PATH=${1:-./checkpoints/cold_start/alfworld/qwen1.5b_rebel_426/global_step_75}
NUM_TASKS=${2:-134}  # ALFWorld 评测任务数量（默认134个测试任务）
ENGINE=${3:-vllm}
GPUS=${4:-2}

echo "==================== Evaluation Configuration ===================="
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Num Tasks: $NUM_TASKS"
echo "Engine: $ENGINE"
echo "GPUs: $GPUS"
echo "================================================================="

# 检查checkpoint是否存在
if [ ! -d "$CHECKPOINT_PATH" ]; then
    echo "Error: Checkpoint not found at $CHECKPOINT_PATH"
    echo "Available checkpoints:"
    ls -la ./checkpoints/cold_start/alfworld/
    exit 1
fi

echo ""
echo "Starting evaluation..."

python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=gae \
  actor_rollout_ref.model.path=$CHECKPOINT_PATH \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.rollout.n=1 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.name=$ENGINE \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
  actor_rollout_ref.rollout.enable_chunked_prefill=False \
  actor_rollout_ref.rollout.enforce_eager=False \
  actor_rollout_ref.rollout.free_cache_engine=False \
  actor_rollout_ref.rollout.val_kwargs.n=1 \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.0 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=False \
  env.env_name=alfworld/AlfredTWEnv \
  env.seed=0 \
  env.max_steps=30 \
  env.rollout.n=$NUM_TASKS \
  env.alfworld.generalization_level=0 \
  env.alfworld.meta_think=True \
  +env.alfworld.action_only=False \
  trainer.logger=[console] \
  trainer.project_name=RLVMR_Evaluation \
  trainer.experiment_name=eval_rebel_cold_start_426 \
  trainer.n_gpus_per_node=$GPUS \
  trainer.nnodes=1 \
  trainer.total_epochs=0 \
  trainer.val_before_train=True \
  trainer.save_freq=-1 \
  trainer.test_freq=-1

echo ""
echo "==================== Evaluation Completed ===================="
