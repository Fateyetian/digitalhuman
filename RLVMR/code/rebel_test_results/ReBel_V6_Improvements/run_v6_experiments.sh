#!/bin/bash
# =============================================================================
# V6 实验: 目标 90% 整体成功率
# =============================================================================
#
# 核心改进 (已在代码中实现):
#   P0: Ground Truth 宽松验证 + look_at 专用进度检测
#   P1: Stage Type 优先级修复 ("find" 优先于 "lamp")
#   P2: 相对阈值归一化 + KL in Reward
#
# 子实验设计:
#   - exp1: 基础配置 (验证 P0+P1 代码修改效果)
#   - exp2: P0+P1 + 相对阈值归一化 (min_samples_ratio=0.15)
#   - exp3: 全部改进 + KL in Reward
#
# 预期改进:
#   | 实验   | 整体成功率 | look_at | 主要贡献 |
#   |--------|-----------|---------|----------|
#   | V5     | 71.1%     | 8.3%    | baseline |
#   | V6-exp1| 78-82%    | 35-45%  | 代码修复 |
#   | V6-exp2| 82-87%    | 50-60%  | 相对阈值 |
#   | V6-exp3| 88-92%    | 65-75%  | KL优化   |
#
# =============================================================================

set -e

# 默认配置
NUM_GPUS=${NUM_GPUS:-4}
EPOCHS=${EPOCHS:-60}
SAVE_FREQ=${SAVE_FREQ:-60}   # 只保存最后一次模型权重 (设为与EPOCHS相同)
TEST_FREQ=${TEST_FREQ:-5}
EXPERIMENT=${EXPERIMENT:-1}  # 1, 2, or 3

# Offline mode
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_ATTENTION_BACKEND=XFORMERS

# Find SFT Model
find_sft_model() {
    local path=$(ls -d /root/testttt/RLVMR/code/checkpoints/cold_start/alfworld/rebel_full_*/global_step_* 2>/dev/null | sort -V | tail -1)
    [ -n "$path" ] && { echo "$path"; return; }
    path=$(ls -d /root/testttt/RLVMR/code/checkpoints/cold_start/alfworld/*/global_step_* 2>/dev/null | sort -V | tail -1)
    [ -n "$path" ] && { echo "$path"; return; }
    echo "/root/testttt/RLVMR/code/checkpoints/cold_start/alfworld/bdrs_qwen1.5b_2gpu_20251110/global_step_75"
}

SFT_MODEL_PATH=$(find_sft_model)

# 根据实验编号设置配置
case $EXPERIMENT in
    1)
        EXPERIMENT_NAME="rebel_v6_exp1_baseline_$(date +%Y%m%d_%H%M%S)"
        EXP_DESC="V6-Exp1: 基础配置 (验证代码修复效果)"

        # 使用默认配置，验证 P0+P1 代码修改的效果
        MIN_SAMPLES_RATIO=0.0
        USE_KL_IN_REWARD=False
        KL_PENALTY="kl"
        KL_COEF=0.001
        USE_KL_LOSS=True
        KL_LOSS_COEF=0.01
        ;;
    2)
        EXPERIMENT_NAME="rebel_v6_exp2_relative_norm_$(date +%Y%m%d_%H%M%S)"
        EXP_DESC="V6-Exp2: 相对阈值归一化"

        # 启用相对阈值归一化
        MIN_SAMPLES_RATIO=0.15  # V6关键: 样本占比 < 15% 使用保护归一化
        USE_KL_IN_REWARD=False
        KL_PENALTY="kl"
        KL_COEF=0.001
        USE_KL_LOSS=True
        KL_LOSS_COEF=0.01
        ;;
    3)
        EXPERIMENT_NAME="rebel_v6_exp3_full_improvements_$(date +%Y%m%d_%H%M%S)"
        EXP_DESC="V6-Exp3: 全部改进 + KL in Reward"

        # 全部改进
        MIN_SAMPLES_RATIO=0.15
        USE_KL_IN_REWARD=True   # V6关键: KL in Reward
        KL_PENALTY="kl"          # K1 估计器
        KL_COEF=0.001            # 较小系数
        USE_KL_LOSS=False        # 关闭 Loss 中的 KL
        KL_LOSS_COEF=0
        ;;
    *)
        echo "无效的实验编号: $EXPERIMENT (可选: 1, 2, 3)"
        exit 1
        ;;
esac

RESULTS_DIR="/fs-computility-new/UPDZ03_chengjun/huangsijie.p/rebel_results/v6_experiments/${EXPERIMENT_NAME}"

echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "  ${EXP_DESC}"
echo "═══════════════════════════════════════════════════════════════════"
echo ""
echo "V6 核心代码改进 (已实现):"
echo "  P0: Ground Truth 宽松验证 (早期阶段避免错误惩罚)"
echo "  P0: look_at 任务专用进度检测 (4阶段奖励)"
echo "  P1: Stage Type 优先级修复 ('find' 优先于 'lamp')"
echo ""
echo "本实验配置:"
echo "  - min_samples_ratio: ${MIN_SAMPLES_RATIO}"
echo "  - use_kl_in_reward: ${USE_KL_IN_REWARD}"
echo "  - use_kl_loss: ${USE_KL_LOSS}"
echo "  - kl_loss_coef: ${KL_LOSS_COEF}"
echo ""
echo "配置:"
echo "  - GPU: ${NUM_GPUS} × A800-80GB"
echo "  - Epochs: ${EPOCHS}"
echo "  - SFT模型: ${SFT_MODEL_PATH}"
echo "  - 结果目录: ${RESULTS_DIR}"
echo ""

read -p "开始训练? [Y/n] " -n 1 -r
echo
[[ $REPLY =~ ^[Nn]$ ]] && exit 0

mkdir -p "${RESULTS_DIR}/checkpoints"

# 保存实验配置
cat > "${RESULTS_DIR}/experiment_config.txt" << EOF
实验名称: ${EXPERIMENT_NAME}
实验描述: ${EXP_DESC}
开始时间: $(date)

V6 核心代码改进:
  - Stage Type 优先级修复: "find" 优先于 "lamp"
  - Ground Truth 宽松验证: 早期阶段(step<3)或GT不完整时使用结构化奖励
  - look_at 专用进度检测: 4阶段奖励(拾取+找灯+使用+完成)

实验配置:
  - belief_granularity: adaptive
  - task_aware_grouping: true
  - per_task_normalization: true
  - conditional_norm: true
  - min_samples_for_norm: 10
  - min_std_for_norm: 0.2
  - min_samples_ratio: ${MIN_SAMPLES_RATIO}
  - use_kl_in_reward: ${USE_KL_IN_REWARD}
  - kl_penalty: ${KL_PENALTY}
  - kl_coef: ${KL_COEF}
  - use_kl_loss: ${USE_KL_LOSS}
  - kl_loss_coef: ${KL_LOSS_COEF}

硬件配置:
  - GPU数量: ${NUM_GPUS}
  - 训练轮数: ${EPOCHS}
  - SFT模型: ${SFT_MODEL_PATH}
EOF

# 准备数据
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
    actor_rollout_ref.actor.entropy_coeff=0.001 \
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
    algorithm.kl_penalty=${KL_PENALTY} \
    algorithm.kl_ctrl.type=fixed \
    algorithm.kl_ctrl.kl_coef=${KL_COEF} \
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
    trainer.project_name='ReBel_V6_Experiments' \
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
echo "结果目录: ${RESULTS_DIR}"
echo ""
echo "检查关键指标:"
echo "  grep 'val/success_rate' ${RESULTS_DIR}/training.log | tail -10"
echo "  grep 'val/look_at_obj_in_light' ${RESULTS_DIR}/training.log | tail -10"
echo "  grep 'rebel/num_groups' ${RESULTS_DIR}/training.log | tail -5"
echo "  grep 'rebel/single_sample_ratio' ${RESULTS_DIR}/training.log | tail -5"
echo ""
echo "V6 目标:"
echo "  - 整体成功率: 90%"
echo "  - look_at_obj_in_light: 75%"
echo ""
