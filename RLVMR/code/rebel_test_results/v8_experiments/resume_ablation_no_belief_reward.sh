#!/bin/bash
# =============================================================================
# ReBel V8 消融实验: 移除信念奖励 - 断点续训脚本
# =============================================================================
# 目的: 从中断的 checkpoint 继续训练
# 继续自: global_step_50 (已完成 epoch 50)
# =============================================================================

set -e

NUM_GPUS=${NUM_GPUS:-4}
EPOCHS=${EPOCHS:-100}
SEED=${SEED:-42}

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_ATTENTION_BACKEND=XFORMERS

# 清理残留的 ray 进程
echo "正在清理残留进程..."
ray stop --force 2>/dev/null || true
pkill -f "ray::" 2>/dev/null || true
sleep 3

# 中断的实验目录 (已有 checkpoint)
RESUME_DIR="/fs-computility-new/UPDZ03_chengjun/huangsijie.p/rebel_results/v8_experiments/ablation_no_belief_reward_seed42_20260122_150758"
CHECKPOINT_DIR="${RESUME_DIR}/checkpoints"

# 验证 checkpoint 存在
if [ ! -d "${CHECKPOINT_DIR}/global_step_50" ]; then
    echo "错误: Checkpoint 不存在: ${CHECKPOINT_DIR}/global_step_50"
    exit 1
fi

# 动态查找SFT模型路径
find_sft_model() {
    local path=$(ls -d /root/testttt/RLVMR/code/checkpoints/cold_start/alfworld/rebel_full_*/global_step_* 2>/dev/null | sort -V | tail -1)
    [ -n "$path" ] && { echo "$path"; return; }
    path=$(ls -d /root/testttt/RLVMR/code/checkpoints/cold_start/alfworld/*/global_step_* 2>/dev/null | sort -V | tail -1)
    [ -n "$path" ] && { echo "$path"; return; }
    echo "/root/testttt/RLVMR/code/checkpoints/cold_start/alfworld/bdrs_qwen1.5b_2gpu_20251110/global_step_75"
}

SFT_MODEL_PATH=$(find_sft_model)
RESULTS_DIR="${RESUME_DIR}"
EXP_NAME="ablation_no_belief_reward_seed${SEED}"

# V8配置
CLIP_RATIO_LOW=0.2
CLIP_RATIO_HIGH=0.28
ENTROPY_COEFF=0.001
USE_KL_LOSS=True
KL_LOSS_COEF=0.01
USE_KL_IN_REWARD=False
MIN_SAMPLES_RATIO=0.15
ENTROPY_PROTECTION_ENABLE=True
ENTROPY_PROTECTION_METHOD=clip_cov
CLIP_COV_LB=0.0
CLIP_COV_UB=0.3

# 任务权重配置
USE_TASK_WEIGHTING="true"
WEIGHT_ALPHA=2.0
WEIGHT_MIN=0.3
WEIGHT_MAX=3.0
WARMUP_EPOCHS=20

echo "═══════════════════════════════════════════════════════════════════"
echo "  ReBel V8 消融实验: 断点续训 - 移除信念奖励"
echo "═══════════════════════════════════════════════════════════════════"
echo ""
echo "配置:"
echo "  - 实验名称: ${EXP_NAME}"
echo "  - 随机种子: ${SEED}"
echo "  - GPU数量: ${NUM_GPUS}"
echo "  - Epochs: ${EPOCHS}"
echo "  - 从 checkpoint 继续: ${CHECKPOINT_DIR}/global_step_50"
echo "  - 结果奖励: 启用"
echo "  - 信念奖励: 禁用"
echo "  - 结果目录: ${RESULTS_DIR}"
echo ""
echo "───────────────────────────────────────────────────────────────────"

cd /root/testttt/RLVMR/code

# 准备数据
if [ ! -f "$HOME/data/verl-agent/text/train.parquet" ]; then
    python3 -m examples.data_preprocess.prepare \
        --mode 'text' \
        --train_data_size 16 \
        --val_data_size 128 2>/dev/null || true
fi

# 运行训练 - 从 checkpoint 恢复
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=rebel \
    algorithm.rebel.enable=True \
    algorithm.rebel.belief_granularity="adaptive" \
    algorithm.rebel.step_advantage_w=0.5 \
    algorithm.rebel.mode="mean_norm" \
    algorithm.rebel.task_aware_grouping=true \
    algorithm.rebel.per_task_normalization=true \
    algorithm.rebel.conditional_norm=true \
    algorithm.rebel.min_samples_for_norm=10 \
    algorithm.rebel.min_std_for_norm=0.2 \
    algorithm.rebel.min_samples_ratio=${MIN_SAMPLES_RATIO} \
    algorithm.rebel.entropy_protection.enable=${ENTROPY_PROTECTION_ENABLE} \
    algorithm.rebel.entropy_protection.method=${ENTROPY_PROTECTION_METHOD} \
    algorithm.rebel.entropy_protection.clip_cov_lb=${CLIP_COV_LB} \
    algorithm.rebel.entropy_protection.clip_cov_ub=${CLIP_COV_UB} \
    +algorithm.rebel.use_task_weighting=${USE_TASK_WEIGHTING} \
    +algorithm.rebel.weight_alpha=${WEIGHT_ALPHA} \
    +algorithm.rebel.weight_min=${WEIGHT_MIN} \
    +algorithm.rebel.weight_max=${WEIGHT_MAX} \
    +algorithm.rebel.weight_baseline_sr=0.85 \
    +algorithm.rebel.task_weighting_warmup_epochs=${WARMUP_EPOCHS} \
    +algorithm.rebel.use_result_reward=true \
    +algorithm.rebel.use_belief_reward=false \
    data.train_files=$HOME/data/verl-agent/text/train.parquet \
    data.val_files=$HOME/data/verl-agent/text/test.parquet \
    data.train_batch_size=16 \
    data.val_batch_size=128 \
    data.max_prompt_length=6000 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path="${SFT_MODEL_PATH}" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.clip_ratio=0.2 \
    actor_rollout_ref.actor.clip_ratio_low=${CLIP_RATIO_LOW} \
    actor_rollout_ref.actor.clip_ratio_high=${CLIP_RATIO_HIGH} \
    actor_rollout_ref.actor.entropy_coeff=${ENTROPY_COEFF} \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.use_kl_loss=${USE_KL_LOSS} \
    actor_rollout_ref.actor.kl_loss_coef=${KL_LOSS_COEF} \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=${USE_KL_IN_REWARD} \
    algorithm.kl_penalty=kl \
    algorithm.kl_ctrl.type=fixed \
    algorithm.kl_ctrl.kl_coef=0.001 \
    env.env_name=alfworld/AlfredTWEnv \
    env.seed=${SEED} \
    env.max_steps=30 \
    env.rollout.n=16 \
    env.alfworld.generalization_level=0 \
    env.alfworld.meta_think=True \
    env.alfworld.use_rebel=True \
    env.alfworld.prompt_template_type="explicit_task_type" \
    env.use_teacher_planner=True \
    trainer.critic_warmup=0 \
    trainer.logger="['console','swanlab']" \
    trainer.project_name='ReBel_V8_Ablation' \
    trainer.experiment_name="${EXP_NAME}_resumed" \
    trainer.n_gpus_per_node=${NUM_GPUS} \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=5 \
    trainer.total_epochs=${EPOCHS} \
    trainer.default_local_dir="${CHECKPOINT_DIR}" \
    trainer.resume_mode=auto \
    trainer.val_before_train=False \
    2>&1 | tee -a "${RESULTS_DIR}/training_resumed.log"

echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "  断点续训完成: 移除信念奖励"
echo "═══════════════════════════════════════════════════════════════════"
echo ""
echo "结果保存在: ${RESULTS_DIR}"
echo ""
