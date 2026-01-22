#!/bin/bash
# =============================================================================
# ReBel V8 实验脚本: 任务自适应权重
# =============================================================================
# 基于V7配置，添加任务自适应权重功能
# 核心改进: 根据任务成功率动态调整优势缩放权重
# =============================================================================

set -e

# 配置参数
EXPERIMENT=${EXPERIMENT:-1}
NUM_GPUS=${NUM_GPUS:-8}
EPOCHS=${EPOCHS:-100}
SAVE_FREQ=${SAVE_FREQ:-50}
TEST_FREQ=${TEST_FREQ:-5}

# 环境变量
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_ATTENTION_BACKEND=XFORMERS

# 动态查找SFT模型路径 (与V7保持一致)
find_sft_model() {
    local path=$(ls -d /root/testttt/RLVMR/code/checkpoints/cold_start/alfworld/rebel_full_*/global_step_* 2>/dev/null | sort -V | tail -1)
    [ -n "$path" ] && { echo "$path"; return; }
    path=$(ls -d /root/testttt/RLVMR/code/checkpoints/cold_start/alfworld/*/global_step_* 2>/dev/null | sort -V | tail -1)
    [ -n "$path" ] && { echo "$path"; return; }
    echo "/root/testttt/RLVMR/code/checkpoints/cold_start/alfworld/bdrs_qwen1.5b_2gpu_20251110/global_step_75"
}

# 基础路径
SFT_MODEL_PATH=$(find_sft_model)
RESULTS_BASE="/fs-computility-new/UPDZ03_chengjun/huangsijie.p/rebel_results/v8_experiments"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# V7基础配置 (保持不变)
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

# =============================================================================
# V8实验配置
# =============================================================================

run_v8_experiment() {
    local EXP_NAME=$1
    local USE_TASK_WEIGHTING=$2
    local WEIGHT_ALPHA=$3
    local WEIGHT_MIN=$4
    local WEIGHT_MAX=$5
    local WARMUP_EPOCHS=$6
    local EXP_DESC=$7

    local RESULTS_DIR="${RESULTS_BASE}/${EXP_NAME}_${TIMESTAMP}"
    mkdir -p "${RESULTS_DIR}/checkpoints"

    echo "═══════════════════════════════════════════════════════════════════"
    echo "  V8实验: ${EXP_NAME}"
    echo "  ${EXP_DESC}"
    echo "═══════════════════════════════════════════════════════════════════"
    echo ""
    echo "V8配置:"
    echo "  - use_task_weighting: ${USE_TASK_WEIGHTING}"
    echo "  - weight_alpha: ${WEIGHT_ALPHA}"
    echo "  - weight_min: ${WEIGHT_MIN}"
    echo "  - weight_max: ${WEIGHT_MAX}"
    echo "  - warmup_epochs: ${WARMUP_EPOCHS}"
    echo ""
    echo "训练配置:"
    echo "  - GPU数量: ${NUM_GPUS}"
    echo "  - 总Epochs: ${EPOCHS}"
    echo "  - 保存频率: 每${SAVE_FREQ}个epoch"
    echo ""

    cd /root/testttt/RLVMR/code

    # 准备数据 (如果数据已存在则跳过)
    if [ ! -f "$HOME/data/verl-agent/text/train.parquet" ]; then
        echo "准备数据..."
        python3 -m examples.data_preprocess.prepare \
            --mode 'text' \
            --train_data_size 16 \
            --val_data_size 128 2>/dev/null || true
    else
        echo "数据已存在，跳过预处理"
    fi

    # 运行训练
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
        actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
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
        trainer.project_name='ReBel_V8_Experiments' \
        trainer.experiment_name="${EXP_NAME}" \
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
    echo "  实验完成: ${EXP_NAME}"
    echo "═══════════════════════════════════════════════════════════════════"
    echo ""
    echo "结果目录: ${RESULTS_DIR}"
    echo ""
    echo "关键指标检查:"
    echo "  grep 'val/success_rate' ${RESULTS_DIR}/training.log | tail -10"
    echo "  grep 'val/look_at' ${RESULTS_DIR}/training.log | tail -10"
    echo "  grep 'rebel/task_weight' ${RESULTS_DIR}/training.log | tail -10"
}

# =============================================================================
# 实验选择
# =============================================================================

case $EXPERIMENT in
    1)
        # 实验1: V8核心实验 - 启用任务自适应权重 (推荐)
        run_v8_experiment \
            "rebel_v8_exp1_task_weighting" \
            "true" \
            "2.0" \
            "0.3" \
            "3.0" \
            "20" \
            "V8核心实验: 启用任务自适应权重 (alpha=2.0, warmup=20)"
        ;;
    2)
        # 实验2: 更激进的权重配置
        run_v8_experiment \
            "rebel_v8_exp2_aggressive" \
            "true" \
            "3.0" \
            "0.2" \
            "4.0" \
            "15" \
            "激进配置: alpha=3.0, max_weight=4.0"
        ;;
    3)
        # 实验3: 对照组 - V7基线 (不启用任务权重)
        run_v8_experiment \
            "rebel_v8_exp3_baseline" \
            "false" \
            "2.0" \
            "0.3" \
            "3.0" \
            "20" \
            "对照组: V7基线 (不启用任务权重)"
        ;;
    4)
        # 实验4: 温和配置
        run_v8_experiment \
            "rebel_v8_exp4_mild" \
            "true" \
            "1.5" \
            "0.5" \
            "2.0" \
            "25" \
            "温和配置: alpha=1.5, max_weight=2.0"
        ;;
    *)
        echo "═══════════════════════════════════════════════════════════════════"
        echo "  ReBel V8 实验脚本"
        echo "═══════════════════════════════════════════════════════════════════"
        echo ""
        echo "用法: EXPERIMENT=N NUM_GPUS=8 EPOCHS=100 $0"
        echo ""
        echo "可用实验:"
        echo "  1: V8核心实验 - 任务自适应权重 (推荐)"
        echo "  2: 激进配置 (alpha=3.0)"
        echo "  3: 对照组 - V7基线"
        echo "  4: 温和配置 (alpha=1.5)"
        echo ""
        echo "示例:"
        echo "  EXPERIMENT=1 NUM_GPUS=8 EPOCHS=100 SAVE_FREQ=50 $0"
        exit 1
        ;;
esac
