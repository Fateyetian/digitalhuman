#!/bin/bash
# ReBel测试脚本 - 使用Qwen2.5-1.5B-Instruct基础模型
# 用途：在原始预训练模型上评测ReBel，无需已训练的checkpoint

set -x
set -e

ENGINE=${1:-vllm}
export VLLM_ATTENTION_BACKEND=XFORMERS

# 基础模型路径
BASE_MODEL_PATH="/root/testttt/RLVMR/code/base_models/Qwen2.5-1.5B-Instruct"

# 验证模型存在
if [ ! -d "$BASE_MODEL_PATH" ]; then
    echo "❌ 错误: 基础模型不存在: $BASE_MODEL_PATH"
    echo "请先运行: python3 verify_base_model.py"
    exit 1
fi

echo "✅ 使用基础模型: $BASE_MODEL_PATH"

# 小规模测试参数
train_data_size=2
val_data_size=4
group_size=8
max_steps=15

# 输出目录
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="rebel_test_results/base_model_${TIMESTAMP}"
mkdir -p ${OUTPUT_DIR}

# 保存配置
cat > ${OUTPUT_DIR}/test_config.yaml <<EOF
test_name: "ReBel Base Model Evaluation"
timestamp: "${TIMESTAMP}"
model: "Qwen2.5-1.5B-Instruct (Base)"
model_path: "${BASE_MODEL_PATH}"
purpose: "Evaluate ReBel on pre-trained model without fine-tuning"
configuration:
  train_data_size: ${train_data_size}
  val_data_size: ${val_data_size}
  group_size: ${group_size}
  max_steps: ${max_steps}
  engine: ${ENGINE}
  belief_granularity: "subgoal"
  step_advantage_w: 1.0
  mode: "mean_norm"
EOF

echo "=========================================="
echo "ReBel 基础模型评测"
echo "模型: Qwen2.5-1.5B-Instruct (原始预训练)"
echo "输出目录: ${OUTPUT_DIR}"
echo "=========================================="

# 数据准备
echo "[1/3] 准备测试数据..."
python3 -m examples.data_preprocess.prepare \
    --mode 'text' \
    --train_data_size $train_data_size \
    --val_data_size $val_data_size \
    2>&1 | tee ${OUTPUT_DIR}/data_prep.log

echo "[2/3] 运行ReBel评测（基础模型）..."

# 运行ReBel评测
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=rebel \
    algorithm.rebel.enable=True \
    algorithm.rebel.belief_granularity='subgoal' \
    algorithm.rebel.step_advantage_w=1.0 \
    algorithm.rebel.mode='mean_norm' \
    data.train_files=$HOME/data/verl-agent/text/train.parquet \
    data.val_files=$HOME/data/verl-agent/text/test.parquet \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=6000 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=${BASE_MODEL_PATH} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=False \
    env.env_name=alfworld/AlfredTWEnv \
    env.seed=0 \
    env.max_steps=$max_steps \
    env.rollout.n=$group_size \
    env.alfworld.generalization_level=0 \
    env.alfworld.meta_think=True \
    env.alfworld.use_rebel=True \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name='ReBel_BaseModel_Test' \
    trainer.experiment_name="rebel_base_${TIMESTAMP}" \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_epochs=1 \
    trainer.val_before_train=True \
    trainer.default_local_dir=${OUTPUT_DIR}/checkpoints \
    2>&1 | tee ${OUTPUT_DIR}/training.log

echo "[3/3] 分析结果..."

# 使用分析脚本
python3 analyze_rebel_results.py ${OUTPUT_DIR}

echo ""
echo "=========================================="
echo "✅ 基础模型评测完成！"
echo "=========================================="
echo "结果目录: ${OUTPUT_DIR}"
echo ""
echo "关键文件:"
echo "  - test_config.yaml          : 测试配置"
echo "  - analysis_results.json     : 结构化结果"
echo "  - ANALYSIS_REPORT.md        : 分析报告"
echo "  - paper_tables.tex          : LaTeX表格"
echo "  - training.log              : 完整日志"
echo "=========================================="
echo ""
echo "查看报告:"
echo "  cat ${OUTPUT_DIR}/ANALYSIS_REPORT.md"
echo ""
echo "查看LaTeX表格:"
echo "  cat ${OUTPUT_DIR}/paper_tables.tex"
echo "=========================================="
