#!/bin/bash
# =============================================================================
# V7 Exp1 恢复训练: 从 epoch 60 继续到 epoch 100
# =============================================================================
#
# 基于已完成的 V7-Exp1 (Clip-Cov + 放宽clip上限) 继续训练
# 原实验: rebel_v7_exp1_clip_cov_high_clip_20260113_064557
#
# =============================================================================

set -e

NUM_GPUS=${NUM_GPUS:-4}
TOTAL_EPOCHS=100
SAVE_FREQ=${SAVE_FREQ:-100}
TEST_FREQ=${TEST_FREQ:-5}

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_ATTENTION_BACKEND=XFORMERS

# 原实验checkpoint路径 (用于恢复训练状态)
CHECKPOINT_DIR="/fs-computility-new/UPDZ03_chengjun/huangsijie.p/rebel_results/v7_experiments/rebel_v7_exp1_clip_cov_high_clip_20260113_064557/checkpoints"
RESUME_FROM="${CHECKPOINT_DIR}/global_step_60"

# 原始SFT模型路径 (包含tokenizer，用于初始化)
find_sft_model() {
    local path=$(ls -d /root/testttt/RLVMR/code/checkpoints/cold_start/alfworld/rebel_full_*/global_step_* 2>/dev/null | sort -V | tail -1)
    [ -n "$path" ] && { echo "$path"; return; }
    path=$(ls -d /root/testttt/RLVMR/code/checkpoints/cold_start/alfworld/*/global_step_* 2>/dev/null | sort -V | tail -1)
    [ -n "$path" ] && { echo "$path"; return; }
    echo "/root/testttt/RLVMR/code/checkpoints/cold_start/alfworld/bdrs_qwen1.5b_2gpu_20251110/global_step_75"
}
SFT_MODEL_PATH=$(find_sft_model)

# 验证checkpoint存在 (检查latest_checkpointed_iteration.txt)
if [ ! -f "${CHECKPOINT_DIR}/latest_checkpointed_iteration.txt" ]; then
    echo "错误: 找不到checkpoint文件 ${CHECKPOINT_DIR}/latest_checkpointed_iteration.txt"
    exit 1
fi

CURRENT_EPOCH=$(cat "${CHECKPOINT_DIR}/latest_checkpointed_iteration.txt")

# 验证SFT模型存在
if [ ! -d "$SFT_MODEL_PATH" ]; then
    echo "错误: SFT模型不存在: $SFT_MODEL_PATH"
    exit 1
fi

EXPERIMENT_NAME="rebel_v7_exp1_clip_cov_high_clip_20260113_064557"
EXP_DESC="V7-Exp1 恢复训练: epoch ${CURRENT_EPOCH} → ${TOTAL_EPOCHS} (Clip-Cov + 放宽clip上限)"

# 原实验结果目录
ORIGINAL_RESULTS_DIR="/fs-computility-new/UPDZ03_chengjun/huangsijie.p/rebel_results/v7_experiments/${EXPERIMENT_NAME}"

# V7-Exp1 配置 (与原实验保持一致)
MIN_SAMPLES_RATIO=0.15
USE_KL_IN_REWARD=False
USE_KL_LOSS=True
KL_LOSS_COEF=0.01

# Clip-Cov熵保护
ENTROPY_PROTECTION_ENABLE=True
ENTROPY_PROTECTION_METHOD="clip_cov"
CLIP_COV_LB=0.0
CLIP_COV_UB=0.5

# V7-Exp1特有: 放宽clip上限
CLIP_RATIO_LOW=0.2
CLIP_RATIO_HIGH=0.28
ENTROPY_COEFF=0.001

echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "  ${EXP_DESC}"
echo "═══════════════════════════════════════════════════════════════════"
echo ""
echo "恢复训练配置:"
echo "  - 当前checkpoint: epoch ${CURRENT_EPOCH}"
echo "  - 目标epoch: ${TOTAL_EPOCHS}"
echo "  - 剩余训练: $((TOTAL_EPOCHS - CURRENT_EPOCH)) epochs"
echo "  - SFT模型(tokenizer): ${SFT_MODEL_PATH}"
echo "  - Checkpoint目录: ${CHECKPOINT_DIR}"
echo ""
echo "V7-Exp1 关键配置 (与原实验一致):"
echo "  - KL方式: KL in Loss"
echo "  - Clip-Cov: lb=${CLIP_COV_LB}, ub=${CLIP_COV_UB}"
echo "  - clip_ratio: [${CLIP_RATIO_LOW}, ${CLIP_RATIO_HIGH}]"
echo "  - entropy_coeff: ${ENTROPY_COEFF}"
echo ""
echo "配置: GPU=${NUM_GPUS}, Total Epochs=${TOTAL_EPOCHS}"
echo ""

# 检查是否已达到目标epoch
if [ "${CURRENT_EPOCH}" -ge "${TOTAL_EPOCHS}" ]; then
    echo "警告: 当前epoch (${CURRENT_EPOCH}) 已经达到或超过目标epoch (${TOTAL_EPOCHS})"
    echo "请增加 TOTAL_EPOCHS 值来继续训练"
    exit 1
fi

read -p "开始恢复训练? [Y/n] " -n 1 -r
echo
[[ $REPLY =~ ^[Nn]$ ]] && exit 0

cd /root/testttt/RLVMR/code
python3 -m examples.data_preprocess.prepare \
    --mode 'text' \
    --train_data_size 16 \
    --val_data_size 128 2>/dev/null || true

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
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
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
    env.seed=0 \
    env.max_steps=30 \
    env.rollout.n=16 \
    env.alfworld.generalization_level=0 \
    env.alfworld.meta_think=True \
    env.alfworld.use_rebel=True \
    env.alfworld.prompt_template_type="explicit_task_type" \
    env.use_teacher_planner=True \
    trainer.critic_warmup=0 \
    trainer.logger="['console','swanlab']" \
    trainer.project_name='ReBel_V7_Experiments' \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.n_gpus_per_node=${NUM_GPUS} \
    trainer.nnodes=1 \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.test_freq=${TEST_FREQ} \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    trainer.resume_mode=auto \
    trainer.default_local_dir="${CHECKPOINT_DIR}" \
    trainer.val_before_train=True \
    2>&1 | tee "${ORIGINAL_RESULTS_DIR}/training_continued.log"

echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "  恢复训练完成: ${EXP_DESC}"
echo "═══════════════════════════════════════════════════════════════════"
echo ""
echo "结果目录: ${ORIGINAL_RESULTS_DIR}"
echo "Checkpoint: ${CHECKPOINT_DIR}/global_step_${TOTAL_EPOCHS}/"
echo ""
echo "关键指标检查:"
echo "  grep 'val/success_rate' ${ORIGINAL_RESULTS_DIR}/training_continued.log | tail -10"
echo "  grep 'val/look_at' ${ORIGINAL_RESULTS_DIR}/training_continued.log | tail -10"
echo "  grep 'actor/entropy_loss' ${ORIGINAL_RESULTS_DIR}/training_continued.log | tail -10"
echo ""
