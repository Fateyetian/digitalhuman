#!/bin/bash
# =============================================================================
# V7 实验: 基于问题本质的改进
# =============================================================================
#
# 问题本质分析:
#   1. 熵坍塌: entropy从0.107下降到0.065 (下降39%)
#   2. clip上限限制: clip_ratio=0.2限制策略更新幅度
#   3. 样本不平衡: look_at仅占~5%样本，梯度信号弱
#
# 解决方案:
#   - 熵坍塌 → Clip-Cov (已实现)
#   - clip限制 → 提高clip_ratio_high
#   - 样本不平衡 → 高熵系数补偿少数任务探索
#
# 实验设计:
#   - Exp1: Clip-Cov + 放宽clip上限 (解决因素1+2)
#   - Exp2: Clip-Cov + 高熵系数 (解决因素1+3)
#
# 预期目标: 90%整体成功率
#   | 实验    | 整体成功率 | look_at | pick_two_obj |
#   |---------|-----------|---------|--------------|
#   | V6-Exp2 | 78.1%     | 83.3%   | 56.5%        |
#   | V6-Exp3 | 84.4%     | 16.7%   | 87.0%        |
#   | V7-Exp1 | 88-92%    | 75-85%  | 80-90%       |
#   | V7-Exp2 | 85-90%    | 80-88%  | 70-85%       |
#
# =============================================================================

set -e

NUM_GPUS=${NUM_GPUS:-4}
EPOCHS=${EPOCHS:-60}
SAVE_FREQ=${SAVE_FREQ:-60}
TEST_FREQ=${TEST_FREQ:-5}
EXPERIMENT=${EXPERIMENT:-1}

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_ATTENTION_BACKEND=XFORMERS

find_sft_model() {
    local path=$(ls -d /root/testttt/RLVMR/code/checkpoints/cold_start/alfworld/rebel_full_*/global_step_* 2>/dev/null | sort -V | tail -1)
    [ -n "$path" ] && { echo "$path"; return; }
    path=$(ls -d /root/testttt/RLVMR/code/checkpoints/cold_start/alfworld/*/global_step_* 2>/dev/null | sort -V | tail -1)
    [ -n "$path" ] && { echo "$path"; return; }
    echo "/root/testttt/RLVMR/code/checkpoints/cold_start/alfworld/bdrs_qwen1.5b_2gpu_20251110/global_step_75"
}

SFT_MODEL_PATH=$(find_sft_model)

# V7共同配置 (基于V6-Exp2，保持KL in Loss)
MIN_SAMPLES_RATIO=0.15
USE_KL_IN_REWARD=False
USE_KL_LOSS=True
KL_LOSS_COEF=0.01

# Clip-Cov熵保护 (两个实验都启用)
ENTROPY_PROTECTION_ENABLE=True
ENTROPY_PROTECTION_METHOD="clip_cov"
CLIP_COV_LB=0.0
CLIP_COV_UB=0.5

case $EXPERIMENT in
    1)
        EXPERIMENT_NAME="rebel_v7_exp1_clip_cov_high_clip_$(date +%Y%m%d_%H%M%S)"
        EXP_DESC="V7-Exp1: Clip-Cov + 放宽clip上限 (解决熵坍塌+clip限制)"

        # 放宽clip上限，允许更大策略更新
        CLIP_RATIO_LOW=0.2
        CLIP_RATIO_HIGH=0.28      # 提高上限 (原0.2)

        # 保持原熵系数
        ENTROPY_COEFF=0.001
        ;;
    2)
        EXPERIMENT_NAME="rebel_v7_exp2_clip_cov_high_entropy_$(date +%Y%m%d_%H%M%S)"
        EXP_DESC="V7-Exp2: Clip-Cov + 高熵系数 (解决熵坍塌+样本不平衡)"

        # 保持原clip
        CLIP_RATIO_LOW=0.2
        CLIP_RATIO_HIGH=0.2

        # 提高熵系数，补偿少数任务探索
        ENTROPY_COEFF=0.01        # 提高10倍 (原0.001)
        ;;
    *)
        echo "无效的实验编号: $EXPERIMENT (可选: 1, 2)"
        exit 1
        ;;
esac

RESULTS_DIR="/fs-computility-new/UPDZ03_chengjun/huangsijie.p/rebel_results/v7_experiments/${EXPERIMENT_NAME}"

echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "  ${EXP_DESC}"
echo "═══════════════════════════════════════════════════════════════════"
echo ""
echo "问题本质与解决方案:"
echo "  1. 熵坍塌 → Clip-Cov: ${ENTROPY_PROTECTION_ENABLE}"
echo "  2. clip限制 → clip_ratio_high: ${CLIP_RATIO_HIGH}"
echo "  3. 样本不平衡 → entropy_coeff: ${ENTROPY_COEFF}"
echo ""
echo "关键配置:"
echo "  - KL方式: KL in Loss (保护少数任务)"
echo "  - Clip-Cov: lb=${CLIP_COV_LB}, ub=${CLIP_COV_UB}"
echo "  - clip_ratio: [${CLIP_RATIO_LOW}, ${CLIP_RATIO_HIGH}]"
echo "  - entropy_coeff: ${ENTROPY_COEFF}"
echo ""
echo "配置: GPU=${NUM_GPUS}, Epochs=${EPOCHS}"
echo "结果目录: ${RESULTS_DIR}"
echo ""

read -p "开始训练? [Y/n] " -n 1 -r
echo
[[ $REPLY =~ ^[Nn]$ ]] && exit 0

mkdir -p "${RESULTS_DIR}/checkpoints"

cat > "${RESULTS_DIR}/experiment_config.txt" << EOF
实验名称: ${EXPERIMENT_NAME}
实验描述: ${EXP_DESC}
开始时间: $(date)

问题本质分析:
  1. 熵坍塌: entropy从0.107下降到0.065 (下降39%)
  2. clip上限限制: clip_ratio=0.2限制策略更新幅度
  3. 样本不平衡: look_at仅占~5%样本

解决方案:
  - Clip-Cov熵保护: ${ENTROPY_PROTECTION_ENABLE}
  - clip_cov_ub: ${CLIP_COV_UB}
  - clip_ratio_high: ${CLIP_RATIO_HIGH}
  - entropy_coeff: ${ENTROPY_COEFF}

基线对比:
  - V6-Exp2: 78.1% overall, 83.3% look_at, 56.5% pick_two_obj
  - V6-Exp3: 84.4% overall, 16.7% look_at, 87.0% pick_two_obj

目标: 90%整体成功率，同时保持look_at > 75%
EOF

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
    trainer.total_epochs=${EPOCHS} \
    trainer.default_local_dir="${RESULTS_DIR}/checkpoints" \
    trainer.val_before_train=True \
    2>&1 | tee "${RESULTS_DIR}/training.log"

echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "  实验完成: ${EXP_DESC}"
echo "═══════════════════════════════════════════════════════════════════"
echo ""
echo "关键指标检查:"
echo "  # 整体成功率"
echo "  grep 'val/success_rate' ${RESULTS_DIR}/training.log | tail -10"
echo ""
echo "  # look_at成功率 (目标: >75%)"
echo "  grep 'val/look_at' ${RESULTS_DIR}/training.log | tail -10"
echo ""
echo "  # pick_two_obj成功率 (目标: >80%)"
echo "  grep 'val/pick_two_obj' ${RESULTS_DIR}/training.log | tail -10"
echo ""
echo "  # 熵变化 (监控是否坍塌)"
echo "  grep 'actor/entropy_loss' ${RESULTS_DIR}/training.log | tail -10"
echo ""
echo "  # Clip-Cov比例 (监控熵保护效果)"
echo "  grep 'actor/clip_cov_ratio' ${RESULTS_DIR}/training.log | tail -10"
echo ""
echo "  # 信念偏移程度 (V7新增，用于论文)"
echo "  grep 'rebel/belief_deviation_total_mean' ${RESULTS_DIR}/training.log | tail -10"
echo ""
echo "  # 信念有效率"
echo "  grep 'rebel/belief_valid_ratio' ${RESULTS_DIR}/training.log | tail -10"
echo ""
echo "V7 目标: 90%整体成功率，look_at > 75%"
echo ""
