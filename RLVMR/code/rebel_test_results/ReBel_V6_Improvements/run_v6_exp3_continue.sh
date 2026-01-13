#!/bin/bash
# =============================================================================
# V6 Exp3 继续训练脚本
# 从 epoch 60 继续训练到 epoch 100
# =============================================================================
#
# 使用方法:
#   ./run_v6_exp3_continue.sh
#
# 特性:
#   - 从现有checkpoint (epoch 60) 自动恢复
#   - 记录每个epoch的训练/测试任务完整信息 (任务名、是否成功)
#   - 任务分布保存到 task_distribution.json
#   - 只保存最后一个epoch (epoch 100) 的模型
#
# =============================================================================

set -e

# 配置
NUM_GPUS=${NUM_GPUS:-4}
EPOCHS=100                      # 目标epoch总数
SAVE_FREQ=100                   # 只保存最后一个epoch
TEST_FREQ=${TEST_FREQ:-5}       # 每5个epoch验证一次

# Offline mode
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_ATTENTION_BACKEND=XFORMERS

# 原实验路径
ORIGINAL_EXPERIMENT="rebel_v6_exp3_full_improvements_20260108_160959"
ORIGINAL_RESULTS_DIR="/fs-computility-new/UPDZ03_chengjun/huangsijie.p/rebel_results/v6_experiments/${ORIGINAL_EXPERIMENT}"

# 验证checkpoint存在
CHECKPOINT_DIR="${ORIGINAL_RESULTS_DIR}/checkpoints"
if [ ! -f "${CHECKPOINT_DIR}/latest_checkpointed_iteration.txt" ]; then
    echo "错误: 找不到checkpoint文件 ${CHECKPOINT_DIR}/latest_checkpointed_iteration.txt"
    exit 1
fi

CURRENT_EPOCH=$(cat "${CHECKPOINT_DIR}/latest_checkpointed_iteration.txt")
echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "  V6 Exp3 继续训练: Epoch ${CURRENT_EPOCH} -> ${EPOCHS}"
echo "═══════════════════════════════════════════════════════════════════"
echo ""
echo "当前checkpoint: epoch ${CURRENT_EPOCH}"
echo "目标epoch: ${EPOCHS}"
echo "剩余训练: $((EPOCHS - CURRENT_EPOCH)) epochs"
echo ""
echo "配置:"
echo "  - GPU: ${NUM_GPUS}"
echo "  - 只保存最后一个epoch (epoch ${EPOCHS})"
echo "  - 验证频率: 每${TEST_FREQ}个epoch"
echo "  - Checkpoint目录: ${CHECKPOINT_DIR}"
echo ""
echo "V6 Exp3 配置:"
echo "  - min_samples_ratio: 0.15"
echo "  - use_kl_in_reward: True"
echo "  - use_kl_loss: False"
echo ""
echo "任务分布记录:"
echo "  - 每个epoch记录训练/测试任务的完整信息"
echo "  - 包含: 任务名(gamefile)、任务类型、是否成功"
echo "  - 保存到: ${CHECKPOINT_DIR}/task_distribution.json"
echo ""

if [ "${CURRENT_EPOCH}" -ge "${EPOCHS}" ]; then
    echo "警告: 当前epoch (${CURRENT_EPOCH}) 已经达到或超过目标epoch (${EPOCHS})"
    echo "请增加 EPOCHS 值来继续训练"
    exit 1
fi

read -p "开始继续训练? [Y/n] " -n 1 -r
echo
[[ $REPLY =~ ^[Nn]$ ]] && exit 0

# SFT模型路径 (与原始实验相同)
SFT_MODEL_PATH="/root/testttt/RLVMR/code/checkpoints/cold_start/alfworld/rebel_full_20251225_024950/global_step_90"

# 准备数据
cd /root/testttt/RLVMR/code
python3 -m examples.data_preprocess.prepare \
    --mode 'text' \
    --train_data_size 16 \
    --val_data_size 128 2>/dev/null || true

# 继续训练
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
    algorithm.rebel.min_samples_ratio=0.15 \
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
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0 \
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
    algorithm.use_kl_in_reward=True \
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
    trainer.project_name='ReBel_V6_Experiments' \
    trainer.experiment_name="${ORIGINAL_EXPERIMENT}_continued" \
    trainer.n_gpus_per_node=${NUM_GPUS} \
    trainer.nnodes=1 \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.test_freq=${TEST_FREQ} \
    trainer.total_epochs=${EPOCHS} \
    trainer.resume_mode=auto \
    trainer.default_local_dir="${CHECKPOINT_DIR}" \
    trainer.val_before_train=True \
    2>&1 | tee "${ORIGINAL_RESULTS_DIR}/training_continued.log"

echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "  训练完成! (Epoch 60 -> ${EPOCHS})"
echo "═══════════════════════════════════════════════════════════════════"
echo ""
echo "结果目录: ${ORIGINAL_RESULTS_DIR}"
echo "Checkpoint: ${CHECKPOINT_DIR}/global_step_${EPOCHS}/"
echo ""
echo "任务分布日志:"
echo "  ${CHECKPOINT_DIR}/task_distribution.json"
echo ""
echo "查看任务分布 (包含完整任务名和成功状态):"
echo "  python3 -c \"import json; data=json.load(open('${CHECKPOINT_DIR}/task_distribution.json')); print(json.dumps(data, indent=2))\" | head -100"
echo ""
echo "查看训练/测试任务类型差异:"
echo "  python3 << 'EOF'
import json
data = json.load(open('${CHECKPOINT_DIR}/task_distribution.json'))
train_epochs = [e for e in data['epochs'] if e['phase'] == 'train']
val_epochs = [e for e in data['epochs'] if e['phase'] == 'val']
print('=== 训练任务类型分布 (最后一个epoch) ===')
if train_epochs:
    for task, info in train_epochs[-1]['task_type_distribution'].items():
        print(f\"  {task}: {info['count']} ({info['ratio']*100:.1f}%)\")
print()
print('=== 验证任务类型分布 (最后一个epoch) ===')
if val_epochs:
    for task, info in val_epochs[-1]['task_type_distribution'].items():
        print(f\"  {task}: {info['count']} ({info['ratio']*100:.1f}%)\")
    if val_epochs[-1].get('task_success_rates'):
        print()
        print('=== 验证任务成功率 ===')
        for task, sr in val_epochs[-1]['task_success_rates'].items():
            print(f\"  {task}: {sr['success_count']}/{sr['total_count']} = {sr['success_rate']*100:.1f}%\")
EOF"
echo ""
echo "检查关键指标:"
echo "  grep 'val/success_rate' ${ORIGINAL_RESULTS_DIR}/training_continued.log | tail -10"
echo "  grep 'val/look_at_obj_in_light' ${ORIGINAL_RESULTS_DIR}/training_continued.log | tail -10"
echo ""
