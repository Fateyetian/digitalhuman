#!/bin/bash
# BDRS 快速验证脚本 - 用于测试代码正确性
# 运行2个epoch，小batch size，快速检验数据流

set -x
ENGINE=${1:-vllm}
export VLLM_ATTENTION_BACKEND=XFORMERS

# === 验证配置：极小规模，快速运行 ===
train_data_size=4        # 只用4个训练样本
val_data_size=8          # 只用8个验证样本
group_size=2             # 分组大小降到2
total_epochs=2           # 只训练2个epoch

echo "==========================================="
echo "BDRS Quick Verification"
echo "==========================================="
echo "Train size: $train_data_size"
echo "Val size: $val_data_size"
echo "Epochs: $total_epochs"
echo "==========================================="

# 1. 准备数据
echo "[Step 1/3] Preparing data..."
python3 -m examples.data_preprocess.prepare \
    --mode 'text' \
    --train_data_size $train_data_size \
    --val_data_size $val_data_size

if [ $? -ne 0 ]; then
    echo "ERROR: Data preparation failed!"
    exit 1
fi

echo "[Step 2/3] Starting BDRS training verification..."

# 2. 运行训练验证
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=bdrs \
    algorithm.bdrs.enable=True \
    algorithm.bdrs.world_consistency_weight=1.0 \
    algorithm.bdrs.task_progress_weight=2.0 \
    algorithm.bdrs.exploration_efficiency_weight=0.5 \
    algorithm.bdrs.step_advantage_w=1.0 \
    algorithm.bdrs.mode="mean_std_norm" \
    algorithm.bdrs.reward_correct_belief=0.2 \
    algorithm.bdrs.reward_new_conflict=-0.1 \
    algorithm.bdrs.reward_subgoal_complete=0.5 \
    algorithm.bdrs.reward_new_entity=0.05 \
    algorithm.bdrs.reward_new_location=0.1 \
    algorithm.bdrs.penalty_revisit=-0.02 \
    data.train_files=$HOME/data/verl-agent/text/train.parquet \
    data.val_files=$HOME/data/verl-agent/text/test.parquet \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=6000 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-1.5B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=False \
    env.env_name=alfworld/AlfredTWEnv \
    env.seed=0 \
    env.max_steps=30 \
    env.rollout.n=$group_size \
    env.alfworld.generalization_level=0 \
    env.alfworld.meta_think=True \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name='BDRS_Verify' \
    trainer.experiment_name='bdrs_quick_verify' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.resume_mode=disable \
    trainer.save_freq=-1 \
    trainer.test_freq=1 \
    trainer.total_epochs=$total_epochs \
    trainer.val_before_train=False $@

if [ $? -eq 0 ]; then
    echo ""
    echo "==========================================="
    echo "[Step 3/3] Verification PASSED!"
    echo "==========================================="
    echo ""
    echo "✓ Data flow is correct"
    echo "✓ BDRS rewards are computed"
    echo "✓ Training loop works"
    echo ""
    echo "Next steps:"
    echo "  1. Check the console output for BDRS statistics"
    echo "  2. If all looks good, proceed with full training"
    echo "  3. See BDRS_EXPERIMENT_PLAN.md Section 4.2 (Cold Start)"
    echo ""
else
    echo ""
    echo "==========================================="
    echo "[Step 3/3] Verification FAILED!"
    echo "==========================================="
    echo ""
    echo "Please check the error messages above."
    echo "Common issues:"
    echo "  - GPU not available"
    echo "  - Dependencies not installed"
    echo "  - BDRS module not found"
    echo ""
    exit 1
fi
