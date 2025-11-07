#!/bin/bash
# =============================================================================
# ALFWorld 冷启动模型评测 - 最小化配置（用于诊断问题）
# =============================================================================
#
# 用途：当完整评测脚本卡住时，使用此最小化配置定位问题
#
# 主要简化：
#   - 只评测 4 个任务（而不是64个）
#   - 只用 1 个 GPU
#   - 降低显存占用（0.4 而不是 0.6）
#   - 启用 eager 模式（更稳定但慢一些）
#   - 禁用 WandB（减少依赖）
#   - 减少每个任务的最大步数（10步而不是30步）
#
# 使用方法：
#   bash eval_cold_start_minimal.sh [模型路径]
#
# 诊断流程：
#   1. 如果此脚本能成功运行 -> 说明是资源配置问题，需要调整参数
#   2. 如果仍然卡住 -> 说明是 vLLM 或模型文件问题，需要独立测试
# =============================================================================

set -x

MODEL_PATH=${1:-/root/digitalhuman/RLVMR/code/checkpoints/cold_start/alfworld/qwen1.5b_plan_a/global_step_75}
NUM_TASKS=4  # 最小测试：只评测4个任务

echo "=========================================="
echo "开始最小化配置评测"
echo "模型路径: $MODEL_PATH"
echo "任务数量: $NUM_TASKS"
echo "=========================================="

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gae \
    algorithm.use_kl_in_reward=False \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.0 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=False \
    env.env_name=alfworld/AlfredTWEnv \
    env.seed=0 \
    env.max_steps=10 \
    env.rollout.n=$NUM_TASKS \
    env.alfworld.generalization_level=0 \
    env.alfworld.meta_think=True \
    +env.alfworld.action_only=False \
    trainer.critic_warmup=0 \
    trainer.logger=[console] \
    trainer.project_name=RLVMR_Evaluation \
    trainer.experiment_name=eval_minimal_test \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.resume_mode=disable \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_epochs=0 \
    trainer.val_before_train=True

echo "=========================================="
echo "最小化配置评测完成"
echo "=========================================="
