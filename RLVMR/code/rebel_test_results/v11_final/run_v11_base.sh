#!/bin/bash
# =============================================================================
# ReBel V11 通用基础训练脚本
# =============================================================================
# ReBel (Reinforcement Learning with Belief-State Enhancement) 正式实验的统一入口。
# 支持多种子批量运行，通过环境变量控制实验配置。
#
# 必需环境变量:
#   EXP_ID          - 实验编号 (M1-M5 主实验, A1-A6 消融)
#   EXP_NAME        - 实验名称 (如 grpo_baseline, rebel_full)
#   ADV_ESTIMATOR   - advantage 估计器 (grpo/gigpo/rebel/rebel_hibo)
#   USE_REBEL_PROMPT - 是否使用 <belief> 提示格式 (true/false)
#
# 可选环境变量 (有默认值):
#   USE_TRAINING_TRICKS  - 是否启用训练稳定化 (true/false, 默认 false)
#   USE_ADV_TRICKS       - 是否启用 advantage tricks (true/false, 默认 false)
#   USE_BELIEF_REWARD    - 是否使用内在信念奖励 (true/false, 默认 false)
#   USE_RESULT_REWARD    - 是否使用结果奖励 (true/false, 默认 true)
#   USE_BELIEF_DECAY     - 是否启用信念奖励衰减 (true/false, 默认 false)
#   DECAY_METHOD         - 衰减方法 (cosine/adaptive, 默认 cosine)
#   USE_ADAPTIVE_DECAY   - V11: 是否启用自适应衰减 (true/false, 默认 false)
#   USE_DIFFERENTIAL_DECAY - V11: 是否启用差异化组件衰减 (true/false, 默认 true)
#   NUM_GPUS             - GPU 数量 (默认 8)
#   EPOCHS               - 训练轮次 (默认 100)
#   SEED                 - 随机种子 (默认 42)
#   MODEL_PATH           - 起始模型路径 (默认自动查找 SFT checkpoint)
# =============================================================================

set -e

# ======================== 基本参数 ========================
NUM_GPUS=${NUM_GPUS:-8}
EPOCHS=${EPOCHS:-100}
SEED=${SEED:-42}

# ======================== 实验标识 ========================
EXP_ID=${EXP_ID:?"ERROR: EXP_ID is required (M1-M5/A1-A6)"}
EXP_NAME=${EXP_NAME:?"ERROR: EXP_NAME is required"}
ADV_ESTIMATOR=${ADV_ESTIMATOR:?"ERROR: ADV_ESTIMATOR is required (grpo/gigpo/rebel/rebel_hibo)"}
USE_REBEL_PROMPT=${USE_REBEL_PROMPT:?"ERROR: USE_REBEL_PROMPT is required (true/false)"}

# ======================== 可选配置 ========================
USE_TRAINING_TRICKS=${USE_TRAINING_TRICKS:-false}
USE_ADV_TRICKS=${USE_ADV_TRICKS:-false}
USE_BELIEF_REWARD=${USE_BELIEF_REWARD:-false}
USE_RESULT_REWARD=${USE_RESULT_REWARD:-true}
USE_BELIEF_DECAY=${USE_BELIEF_DECAY:-false}
DECAY_METHOD=${DECAY_METHOD:-cosine}
# V11: Adaptive decay control
USE_ADAPTIVE_DECAY=${USE_ADAPTIVE_DECAY:-false}
USE_DIFFERENTIAL_DECAY=${USE_DIFFERENTIAL_DECAY:-true}

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_ATTENTION_BACKEND=XFORMERS

# ======================== 模型路径 ========================
find_sft_model() {
    local path=$(find /root/testttt/RLVMR/code/checkpoints/cold_start/alfworld/rebel_full_*/global_step_* -maxdepth 0 -type d 2>/dev/null | sort -V | tail -1)
    [ -n "$path" ] && { echo "$path"; return; }
    path=$(find /root/testttt/RLVMR/code/checkpoints/cold_start/alfworld/*/global_step_* -maxdepth 0 -type d 2>/dev/null | sort -V | tail -1)
    [ -n "$path" ] && { echo "$path"; return; }
    echo "Qwen/Qwen2.5-1.5B-Instruct"
}

MODEL_PATH=${MODEL_PATH:-$(find_sft_model)}

# ======================== 输出路径 ========================
RESULTS_BASE="/fs-computility-new/UPDZ03_chengjun/huangsijie.p/rebel_results/v11_final"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
FULL_EXP_NAME="${EXP_ID}_${EXP_NAME}_seed${SEED}"
RESULTS_DIR="${RESULTS_BASE}/${FULL_EXP_NAME}_${TIMESTAMP}"
mkdir -p "${RESULTS_DIR}/checkpoints"

# ======================== 训练 Tricks 参数 ========================
if [ "$USE_TRAINING_TRICKS" = "true" ]; then
    CLIP_RATIO_LOW=0.2
    CLIP_RATIO_HIGH=0.28
    ENTROPY_COEFF=0.001
    ENTROPY_PROTECTION_ENABLE=True
    ENTROPY_PROTECTION_METHOD=clip_cov
    CLIP_COV_LB=0.0
    CLIP_COV_UB=0.3
    INVALID_ACTION_PENALTY=True
    INVALID_ACTION_PENALTY_COEF=0.1
else
    CLIP_RATIO_LOW=0.2
    CLIP_RATIO_HIGH=0.2    # 对称裁剪
    ENTROPY_COEFF=0.001
    ENTROPY_PROTECTION_ENABLE=False
    ENTROPY_PROTECTION_METHOD=clip_cov
    CLIP_COV_LB=0.0
    CLIP_COV_UB=0.3
    INVALID_ACTION_PENALTY=False
    INVALID_ACTION_PENALTY_COEF=0.0
fi

# ======================== Advantage Tricks 参数 ========================
if [ "$USE_ADV_TRICKS" = "true" ]; then
    USE_TASK_WEIGHTING="true"
    WEIGHT_ALPHA=2.0
    WEIGHT_MIN=0.3
    WEIGHT_MAX=3.0
    TASK_WEIGHTING_WARMUP=20
    MIN_SAMPLES_RATIO=0.15
else
    USE_TASK_WEIGHTING="false"
    WEIGHT_ALPHA=2.0
    WEIGHT_MIN=0.3
    WEIGHT_MAX=3.0
    TASK_WEIGHTING_WARMUP=20
    MIN_SAMPLES_RATIO=0.0
fi

# ======================== Belief Decay 参数 ========================
DECAY_WARMUP_EPOCHS=3
DECAY_START_EPOCH=5
DECAY_END_EPOCH=40
DECAY_MIN_WEIGHT=0.05
# V11: Adaptive decay parameters
DECAY_TARGET_SR=0.90
DECAY_ALPHA=2.0
# V11: Differential component decay rates
PROGRESS_DECAY_RATE=0.7
CONSISTENCY_DECAY_RATE=1.0
EXPLORATION_DECAY_RATE=2.0

# V11: Override decay rates if differential decay is disabled
if [ "$USE_DIFFERENTIAL_DECAY" = "false" ]; then
    PROGRESS_DECAY_RATE=1.0
    CONSISTENCY_DECAY_RATE=1.0
    EXPLORATION_DECAY_RATE=1.0
fi

# ======================== 打印配置 ========================
echo "═══════════════════════════════════════════════════════════════════"
echo "  ReBel V11 正式实验: ${EXP_ID} - ${EXP_NAME}"
echo "═══════════════════════════════════════════════════════════════════"
echo ""
echo "实验配置:"
echo "  - ID:               ${EXP_ID}"
echo "  - 名称:             ${EXP_NAME}"
echo "  - Advantage:        ${ADV_ESTIMATOR}"
echo "  - 提示格式:         $([ "$USE_REBEL_PROMPT" = "true" ] && echo '<belief>' || echo '<think>')"
echo "  - 训练稳定化:       ${USE_TRAINING_TRICKS}"
echo "  - Advantage Tricks: ${USE_ADV_TRICKS}"
echo "  - 信念奖励:         ${USE_BELIEF_REWARD}"
echo "  - 结果奖励:         ${USE_RESULT_REWARD}"
echo "  - 信念衰减:         ${USE_BELIEF_DECAY} (${DECAY_METHOD})"
echo "  - 自适应衰减:       ${USE_ADAPTIVE_DECAY}"
echo "  - 差异化衰减:       ${USE_DIFFERENTIAL_DECAY}"
echo ""
echo "基本参数:"
echo "  - 种子:             ${SEED}"
echo "  - GPU:              ${NUM_GPUS}"
echo "  - Epochs:           ${EPOCHS}"
echo "  - 模型路径:         ${MODEL_PATH}"
echo "  - 结果目录:         ${RESULTS_DIR}"
echo ""
echo "───────────────────────────────────────────────────────────────────"

cd /root/testttt/RLVMR/code

# ======================== 准备数据 ========================
if [ ! -f "$HOME/data/verl-agent/text/train.parquet" ]; then
    python3 -m examples.data_preprocess.prepare \
        --mode 'text' \
        --train_data_size 16 \
        --val_data_size 128 2>/dev/null || true
fi

# ======================== 构建训练命令 ========================
# 基础参数 (所有实验共享)
BASE_ARGS=(
    "algorithm.adv_estimator=${ADV_ESTIMATOR}"
    "data.train_files=$HOME/data/verl-agent/text/train.parquet"
    "data.val_files=$HOME/data/verl-agent/text/test.parquet"
    "data.train_batch_size=16"
    "data.val_batch_size=128"
    "data.max_prompt_length=6000"
    "data.max_response_length=1024"
    "data.filter_overlong_prompts=True"
    "data.truncation=error"
    "data.return_raw_chat=True"
    "actor_rollout_ref.model.path=${MODEL_PATH}"
    "actor_rollout_ref.actor.optim.lr=1e-6"
    "actor_rollout_ref.actor.clip_ratio=0.2"
    "actor_rollout_ref.actor.clip_ratio_low=${CLIP_RATIO_LOW}"
    "actor_rollout_ref.actor.clip_ratio_high=${CLIP_RATIO_HIGH}"
    "actor_rollout_ref.actor.entropy_coeff=${ENTROPY_COEFF}"
    "actor_rollout_ref.actor.ppo_epochs=1"
    "actor_rollout_ref.actor.use_kl_loss=True"
    "actor_rollout_ref.actor.kl_loss_coef=0.01"
    "actor_rollout_ref.actor.kl_loss_type=low_var_kl"
    "actor_rollout_ref.model.use_remove_padding=True"
    "actor_rollout_ref.actor.ppo_mini_batch_size=256"
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8"
    "actor_rollout_ref.model.enable_gradient_checkpointing=True"
    "actor_rollout_ref.actor.fsdp_config.param_offload=False"
    "actor_rollout_ref.actor.fsdp_config.optimizer_offload=False"
    "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16"
    "actor_rollout_ref.rollout.tensor_model_parallel_size=1"
    "actor_rollout_ref.rollout.name=vllm"
    "actor_rollout_ref.rollout.gpu_memory_utilization=0.65"
    "actor_rollout_ref.rollout.enable_chunked_prefill=False"
    "actor_rollout_ref.rollout.enforce_eager=False"
    "actor_rollout_ref.rollout.free_cache_engine=False"
    "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16"
    "actor_rollout_ref.ref.fsdp_config.param_offload=True"
    "algorithm.use_kl_in_reward=False"
    "algorithm.kl_penalty=kl"
    "algorithm.kl_ctrl.type=fixed"
    "algorithm.kl_ctrl.kl_coef=0.001"
    "env.env_name=alfworld/AlfredTWEnv"
    "env.seed=${SEED}"
    "env.max_steps=30"
    "env.rollout.n=16"
    "env.alfworld.generalization_level=0"
    "env.alfworld.meta_think=True"
    "env.use_teacher_planner=True"
    "trainer.critic_warmup=0"
    "trainer.logger=['console','swanlab']"
    "trainer.project_name=ReBel_V11"
    "trainer.experiment_name=${FULL_EXP_NAME}"
    "trainer.n_gpus_per_node=${NUM_GPUS}"
    "trainer.nnodes=1"
    "trainer.save_freq=25"
    "trainer.test_freq=5"
    "trainer.total_epochs=${EPOCHS}"
    "trainer.default_local_dir=${RESULTS_DIR}/checkpoints"
    "trainer.val_before_train=True"
)

# Invalid action penalty
if [ "$INVALID_ACTION_PENALTY" = "True" ]; then
    BASE_ARGS+=(
        "actor_rollout_ref.actor.use_invalid_action_penalty=True"
        "actor_rollout_ref.actor.invalid_action_penalty_coef=${INVALID_ACTION_PENALTY_COEF}"
    )
fi

# ReBel 提示格式相关参数
if [ "$USE_REBEL_PROMPT" = "true" ]; then
    BASE_ARGS+=(
        "algorithm.rebel.enable=True"
        "env.alfworld.use_rebel=True"
        "env.alfworld.prompt_template_type=explicit_task_type"
    )

    # ReBel 核心参数 (当 rebel.enable=True 时需要提供)
    BASE_ARGS+=(
        "algorithm.rebel.belief_granularity=adaptive"
        "algorithm.rebel.step_advantage_w=0.5"
        "algorithm.rebel.mode=mean_norm"
        "algorithm.rebel.task_aware_grouping=true"
        "algorithm.rebel.per_task_normalization=true"
        "algorithm.rebel.conditional_norm=true"
        "algorithm.rebel.min_samples_for_norm=10"
        "algorithm.rebel.min_std_for_norm=0.2"
        "algorithm.rebel.min_samples_ratio=${MIN_SAMPLES_RATIO}"
        "algorithm.rebel.entropy_protection.enable=${ENTROPY_PROTECTION_ENABLE}"
        "algorithm.rebel.entropy_protection.method=${ENTROPY_PROTECTION_METHOD}"
        "algorithm.rebel.entropy_protection.clip_cov_lb=${CLIP_COV_LB}"
        "algorithm.rebel.entropy_protection.clip_cov_ub=${CLIP_COV_UB}"
    )

    # V11: HiBO-specific parameter (min obs group size for fallback)
    BASE_ARGS+=(
        "+algorithm.rebel.min_obs_group_size=2"
    )

    # 信念奖励和结果奖励开关
    BASE_ARGS+=(
        "+algorithm.rebel.use_belief_reward=${USE_BELIEF_REWARD}"
        "+algorithm.rebel.use_result_reward=${USE_RESULT_REWARD}"
    )

    # Advantage tricks (task weighting)
    BASE_ARGS+=(
        "+algorithm.rebel.use_task_weighting=${USE_TASK_WEIGHTING}"
        "+algorithm.rebel.weight_alpha=${WEIGHT_ALPHA}"
        "+algorithm.rebel.weight_min=${WEIGHT_MIN}"
        "+algorithm.rebel.weight_max=${WEIGHT_MAX}"
        "+algorithm.rebel.weight_baseline_sr=0.85"
        "+algorithm.rebel.task_weighting_warmup_epochs=${TASK_WEIGHTING_WARMUP}"
    )

    # Belief reward decay
    BASE_ARGS+=(
        "+algorithm.rebel.belief_reward_decay.enable=${USE_BELIEF_DECAY}"
        "+algorithm.rebel.belief_reward_decay.method=${DECAY_METHOD}"
        "+algorithm.rebel.belief_reward_decay.warmup_epochs=${DECAY_WARMUP_EPOCHS}"
        "+algorithm.rebel.belief_reward_decay.decay_start_epoch=${DECAY_START_EPOCH}"
        "+algorithm.rebel.belief_reward_decay.decay_end_epoch=${DECAY_END_EPOCH}"
        "+algorithm.rebel.belief_reward_decay.min_weight=${DECAY_MIN_WEIGHT}"
        # V11: Adaptive decay parameters
        "+algorithm.rebel.belief_reward_decay.adaptive=${USE_ADAPTIVE_DECAY}"
        "+algorithm.rebel.belief_reward_decay.target_sr=${DECAY_TARGET_SR}"
        "+algorithm.rebel.belief_reward_decay.alpha=${DECAY_ALPHA}"
        # V11: Differential component decay rates
        "+algorithm.rebel.belief_reward_decay.progress_decay_rate=${PROGRESS_DECAY_RATE}"
        "+algorithm.rebel.belief_reward_decay.consistency_decay_rate=${CONSISTENCY_DECAY_RATE}"
        "+algorithm.rebel.belief_reward_decay.exploration_decay_rate=${EXPLORATION_DECAY_RATE}"
    )
else
    BASE_ARGS+=(
        "algorithm.rebel.enable=False"
        "env.alfworld.use_rebel=False"
    )
fi

# GiGPO / HiBO 特定参数 (both need gamma and step params)
if [ "$ADV_ESTIMATOR" = "gigpo" ] || [ "$ADV_ESTIMATOR" = "rebel_hibo" ]; then
    BASE_ARGS+=(
        "algorithm.gamma=0.95"
        "algorithm.gigpo.step_advantage_w=0.5"
        "algorithm.gigpo.mode=mean_norm"
    )
fi

# ======================== 执行训练 ========================
echo ""
echo "开始训练..."
echo ""

python3 -m verl.trainer.main_ppo \
    "${BASE_ARGS[@]}" \
    2>&1 | tee "${RESULTS_DIR}/training.log"

echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "  实验完成: ${EXP_ID} - ${EXP_NAME} (seed=${SEED})"
echo "═══════════════════════════════════════════════════════════════════"
echo ""
echo "结果保存在: ${RESULTS_DIR}"
echo ""
