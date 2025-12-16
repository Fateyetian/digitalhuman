#!/bin/bash
# =============================================================================
# ALFWorld 冷启动模型评测脚本
# =============================================================================
#
# 说明：本脚本支持两种评测模式
#   模式1（环境交互模式）：在真实ALFWorld环境中评测，模型与环境多轮交互
#   模式2（数据集模式）：使用预先准备的parquet数据集评测（仅生成，无环境交互）
#
# 当前启用：模式1（环境交互模式）- 推荐用于真实性能评估
# =============================================================================

set -x

# ============= 配置参数 =============
MODEL_PATH=${1:-/root/digitalhuman/RLVMR/code/checkpoints/cold_start/alfworld/qwen1.5b_plan_a/global_step_75}
NUM_TASKS=${2:-64}        # 评测的任务数量
ENGINE=${3:-vllm}         # 推理引擎: vllm 或 sglang
EXPERIMENT_NAME=${4:-eval_cold_start_qwen1.5b_step75}

# ============= 模式1：环境交互模式（当前启用）=============
# 优点：
#   - 真实环境评测，结果最可靠
#   - 可以看到每一步的动作和观察
#   - 自动记录成功率、步数等指标到wandb
# 缺点：
#   - 速度相对较慢（需要环境交互）
#
# 评测流程：
#   1. 启动 $NUM_TASKS 个ALFWorld环境实例
#   2. 每个实例运行一个随机任务
#   3. 模型与环境交互，最多30步
#   4. 记录成功率、平均步数、奖励等指标

python3 -m verl.trainer.main_ppo \
      algorithm.adv_estimator=gae \
      algorithm.use_kl_in_reward=False \
      +algorithm.bdrs.enable=True \
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
      env.rollout.n=$NUM_TASKS \
      env.alfworld.generalization_level=0 \
      env.alfworld.meta_think=False \
      +env.alfworld.action_only=False \
      trainer.critic_warmup=0 \
      trainer.logger=[console,wandb] \
      trainer.project_name=RLVMR_Evaluation \
      trainer.experiment_name=$EXPERIMENT_NAME \
      trainer.n_gpus_per_node=2 \
      trainer.nnodes=1 \
      trainer.resume_mode=disable \
      trainer.save_freq=-1 \
      trainer.test_freq=-1 \
      trainer.total_epochs=0 \
      trainer.val_before_train=True

# ============= 模式2：数据集模式（已注释）=============
# 优点：
#   - 速度快（无需环境交互）
#   - 可以精确控制评测样本
# 缺点：
#   - 不是真实环境，结果可能不准确
#   - 需要预先准备parquet数据文件
#   - 无法评估多轮交互能力
#
# 使用方法：
#   1. 取消下面代码的注释
#   2. 注释掉上面的"模式1"代码
#   3. 确保数据文件存在：/root/data/verl-agent/text/train.parquet

# python3 -m verl.trainer.main_ppo \
#       algorithm.adv_estimator=bdrs \
#       algorithm.bdrs.enable=True \
#       algorithm.bdrs.step_advantage_w=1.0 \
#       data.train_files=/root/data/verl-agent/text/train.parquet \
#       data.val_files=/root/data/verl-agent/text/train.parquet \
#       data.train_batch_size=4 \
#       data.val_batch_size=64 \
#       data.max_prompt_length=6000 \
#       data.max_response_length=1024 \
#       data.filter_overlong_prompts=True \
#       data.truncation=error \
#       data.return_raw_chat=True \
#       actor_rollout_ref.model.path=$MODEL_PATH \
#       actor_rollout_ref.actor.optim.lr=1e-6 \
#       actor_rollout_ref.model.use_remove_padding=True \
#       actor_rollout_ref.actor.ppo_mini_batch_size=256 \
#       actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
#       actor_rollout_ref.actor.use_kl_loss=True \
#       actor_rollout_ref.actor.kl_loss_coef=0.01 \
#       actor_rollout_ref.actor.kl_loss_type=low_var_kl \
#       actor_rollout_ref.model.enable_gradient_checkpointing=True \
#       actor_rollout_ref.actor.fsdp_config.param_offload=False \
#       actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
#       actor_rollout_ref.rollout.n=1 \
#       actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
#       actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
#       actor_rollout_ref.rollout.name=$ENGINE \
#       actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
#       actor_rollout_ref.rollout.enable_chunked_prefill=False \
#       actor_rollout_ref.rollout.enforce_eager=False \
#       actor_rollout_ref.rollout.free_cache_engine=False \
#       actor_rollout_ref.rollout.val_kwargs.n=1 \
#       actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
#       actor_rollout_ref.rollout.val_kwargs.do_sample=True \
#       actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
#       actor_rollout_ref.ref.fsdp_config.param_offload=True \
#       actor_rollout_ref.actor.use_invalid_action_penalty=True \
#       actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
#       algorithm.use_kl_in_reward=False \
#       env.env_name=alfworld/AlfredTWEnv \
#       env.seed=0 \
#       env.max_steps=30 \
#       env.rollout.n=0 \
#       env.alfworld.generalization_level=0 \
#       env.alfworld.meta_think=True \
#       +env.alfworld.action_only=False \
#       trainer.critic_warmup=0 \
#       trainer.logger=[console,wandb] \
#       trainer.project_name=RLVMR_Evaluation \
#       trainer.experiment_name=$EXPERIMENT_NAME \
#       trainer.n_gpus_per_node=2 \
#       trainer.nnodes=1 \
#       trainer.resume_mode=disable \
#       trainer.save_freq=-1 \
#       trainer.test_freq=-1 \
#       trainer.total_epochs=1 \
#       trainer.val_before_train=True

# =============================================================================
# 使用说明
# =============================================================================
#
# 基本用法（使用默认参数）：
#   bash eval_cold_start.sh
#
# 自定义参数：
#   bash eval_cold_start.sh [模型路径] [任务数量] [推理引擎] [实验名称]
#
# 示例1：评测64个任务
#   bash eval_cold_start.sh \
#       /root/digitalhuman/RLVMR/code/checkpoints/cold_start/alfworld/qwen1.5b/global_step_100 \
#       64 \
#       vllm \
#       eval_qwen1.5b_64tasks
#
# 示例2：评测128个任务（更全面）
#   bash eval_cold_start.sh \
#       /root/digitalhuman/RLVMR/code/checkpoints/cold_start/alfworld/qwen1.5b/global_step_100 \
#       128 \
#       vllm \
#       eval_qwen1.5b_128tasks
#
# 查看结果：
#   1. 控制台输出：查看实时进度和最终成功率
#   2. WandB: https://wandb.ai/YOUR_ENTITY/RLVMR_Evaluation
#      关键指标：
#        - val/success_rate: 任务成功率
#        - val/episode_length: 平均步数
#        - val/episode_reward: 平均奖励
#
# =============================================================================
