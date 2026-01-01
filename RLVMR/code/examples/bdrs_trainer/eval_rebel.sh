#!/bin/bash
# ReBel Evaluation Script for ALFWorld
# This script evaluates a trained model using the ReBel (Reward Belief) framework

# Usage: bash eval_rebel.sh <MODEL_PATH> <CHECKPOINT_PATH> <NUM_TASKS> <EXPERIMENT_NAME>

MODEL_PATH=${1:-Qwen/Qwen2.5-1.5B-Instruct}
CHECKPOINT_PATH=${2:-checkpoints/qwen1.5b_rebel}
NUM_TASKS=${3:-134}
EXPERIMENT_NAME=${4:-eval_rebel_qwen1.5b}

echo "==================== ReBel Evaluation ===================="
echo "Model: $MODEL_PATH"
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Num Tasks: $NUM_TASKS"
echo "Experiment: $EXPERIMENT_NAME"
echo "=========================================================="

python3 -m verl.trainer.main_ppo \
      algorithm.adv_estimator=gae \
      algorithm.use_kl_in_reward=False \
      +algorithm.rebel.enable=True \
      +algorithm.rebel.alpha=0.3 \
      +algorithm.rebel.beta=0.5 \
      +algorithm.rebel.gamma=0.2 \
      actor_rollout_ref.model.path=$MODEL_PATH \
      actor_rollout_ref.model.use_remove_padding=True \
      actor_rollout_ref.rollout.n=1 \
      actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
      actor_rollout_ref.rollout.log_prob_micro_batch_size=10 \
      actor_rollout_ref.rollout.micro_batch_size=10 \
      actor_rollout_ref.rollout.name=vllm \
      actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
      data.train_batch_size=1 \
      data.val_batch_size=$NUM_TASKS \
      data.max_prompt_length=2048 \
      data.max_response_length=512 \
      env.env_name='alfworld/AlfredTWEnv' \
      env.seed=42 \
      env.max_steps=30 \
      env.rollout.n=$NUM_TASKS \
      env.alfworld.generalization_level=0 \
      +env.alfworld.action_only=False \
      trainer.critic_warmup=0 \
      trainer.logger=[console,wandb] \
      trainer.default_hdfs_dir=null \
      trainer.project_name='rlvmr-rebel-eval' \
      trainer.experiment_name=$EXPERIMENT_NAME \
      trainer.n_gpus_per_node=2 \
      trainer.nnodes=1 \
      trainer.total_epochs=0 \
      +trainer.load_checkpoint=True \
      +trainer.checkpoint_path=$CHECKPOINT_PATH
