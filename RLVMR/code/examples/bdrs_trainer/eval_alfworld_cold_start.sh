#!/bin/bash
# ALFWorld 冷启动模型评测脚本
# 用法: bash eval_alfworld_cold_start.sh [模型路径] [环境数量]

set -x

# 参数配置
MODEL_PATH=${1:-/root/digitalhuman/RLVMR/code/checkpoints/cold_start/alfworld/qwen1.5b_plan_a/global_step_75}
NUM_ENVS=${2:-64}  # 评测的环境数量（任务数量）
ENGINE=${3:-vllm}

# 运行评测
python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=gae \
  actor_rollout_ref.model.path=$MODEL_PATH \
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
  env.rollout.n=$NUM_ENVS \
  env.alfworld.generalization_level=0 \
  env.alfworld.meta_think=True \
  +env.alfworld.action_only=False \
  trainer.logger=[console] \
  trainer.project_name=RLVMR_Evaluation \
  trainer.experiment_name=eval_cold_start_alfworld \
  trainer.n_gpus_per_node=2 \
  trainer.nnodes=1 \
  trainer.total_epochs=0 \
  trainer.val_before_train=True \
  trainer.save_freq=-1 \
  trainer.test_freq=-1
