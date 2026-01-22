#!/bin/bash
# =============================================================================
# V3 Training with 8 GPUs - 150 Epochs
# =============================================================================
#
# 配置:
#   - 8 GPUs (A800-80GB)
#   - 150 epochs
#   - 每50 epochs保存checkpoint (共3次)
#
# 存储估算:
#   - 每个checkpoint: ~19GB
#   - 保存次数: 3 (epoch 50, 100, 150)
#   - 总计: ~57GB + 日志 ≈ 60GB
#
# 时间估算:
#   - 每epoch约 8-10 分钟 (8 GPU)
#   - 150 epochs ≈ 20-25 小时
#
# Usage:
#   bash run_v3_8gpu_150epochs.sh
#
# =============================================================================

set -e

NUM_GPUS=8
EPOCHS=150
SAVE_FREQ=50  # 每50 epochs保存一次

SEARCH_NAME="rebel_v3_8gpu_150ep_$(date +%Y%m%d_%H%M%S)"
RESULTS_DIR="/fs-computility-new/UPDZ03_chengjun/huangsijie.p/rebel_results/v3_experiments/${SEARCH_NAME}"

# Offline mode
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_ATTENTION_BACKEND=XFORMERS

python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

# Find SFT Model
find_sft_model() {
    local path=$(ls -d ./checkpoints/cold_start/alfworld/rebel_full_*/global_step_* 2>/dev/null | sort -V | tail -1)
    [ -n "$path" ] && { echo "$path"; return; }
    path=$(ls -d ./checkpoints/cold_start/alfworld/*/global_step_* 2>/dev/null | sort -V | tail -1)
    [ -n "$path" ] && { echo "$path"; return; }
    echo "./checkpoints/cold_start/alfworld/bdrs_qwen1.5b_2gpu_20251110/global_step_75"
}

SFT_MODEL_PATH=$(find_sft_model)

echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "  V3 Training: 8 GPUs × 150 Epochs"
echo "═══════════════════════════════════════════════════════════════════"
echo ""
echo "GPU:             ${NUM_GPUS} × A800-80GB"
echo "Epochs:          ${EPOCHS}"
echo "保存频率:        每 ${SAVE_FREQ} epochs (共3次: epoch 50, 100, 150)"
echo "存储:            ~57GB (3 × 19GB)"
echo "预计时间:        ~20-25小时"
echo ""
echo "SFT模型:         ${SFT_MODEL_PATH}"
echo "结果目录:        ${RESULTS_DIR}"
echo ""

read -p "开始训练? [Y/n] " -n 1 -r
echo
[[ $REPLY =~ ^[Nn]$ ]] && exit 0

mkdir -p "${RESULTS_DIR}/v3_explicit_task_type/checkpoints"

python3 -m examples.data_preprocess.prepare \
    --mode 'text' \
    --train_data_size 16 \
    --val_data_size 128 2>/dev/null || true

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=rebel \
    algorithm.rebel.enable=True \
    algorithm.rebel.belief_granularity="subgoal" \
    algorithm.rebel.step_advantage_w=0.5 \
    algorithm.rebel.mode="mean_norm" \
    algorithm.rebel.task_aware_grouping=false \
    algorithm.rebel.per_task_normalization=false \
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
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
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
    algorithm.use_kl_in_reward=False \
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
    trainer.project_name='ReBel_V3_8GPU' \
    trainer.experiment_name="${SEARCH_NAME}" \
    trainer.n_gpus_per_node=${NUM_GPUS} \
    trainer.nnodes=1 \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.test_freq=5 \
    trainer.total_epochs=${EPOCHS} \
    trainer.default_local_dir="${RESULTS_DIR}/v3_explicit_task_type/checkpoints" \
    trainer.val_before_train=True \
    2>&1 | tee "${RESULTS_DIR}/training.log"

echo ""
echo "训练完成! 结果: ${RESULTS_DIR}"
