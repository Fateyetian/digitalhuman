#!/bin/bash
# =============================================================================
# AlfWorld本地模型评测脚本 - 使用vLLM后端
# =============================================================================

set -x

# 配置参数
MODEL=${1:-"/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306"}  # 使用本地模型路径
NUM_ENVS=${2:-4}                           # 评测环境数量
BATCH_SIZE=${3:-4}                         # 批处理大小
MAX_STEPS=${4:-30}                         # 每个任务最大步数
OUTPUT_DIR=${5:-"results/local_eval_$(date +%Y%m%d_%H%M%S)"}

echo "========================================="
echo "本地模型评测 - AlfWorld"
echo "========================================="
echo "模型: $MODEL"
echo "环境数: $NUM_ENVS"
echo "最大步数: $MAX_STEPS"
echo "输出目录: $OUTPUT_DIR"
echo "========================================="

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 注意：这个脚本假设你已经启动了vLLM服务器
# 如果还没有启动，请先运行：
# python -m vllm.entrypoints.openai.api_server \
#     --model Qwen/Qwen2.5-1.5B-Instruct \
#     --host 0.0.0.0 \
#     --port 8000

# 运行评测（使用vLLM作为后端）
python3 examples/bdrs_trainer/rollout/run_alfworld_rollout.py \
    --env_name alfworld \
    --batch_size $BATCH_SIZE \
    --total_envs $NUM_ENVS \
    --max_steps $MAX_STEPS \
    --model "$MODEL" \
    --base_url "http://localhost:8000/v1" \
    --temperature 0.4 \
    --concurrency 4 \
    --dump_path "$OUTPUT_DIR/trajectory.jsonl" \
    --chat_root "$OUTPUT_DIR/chats" \
    --unique_envs

echo ""
echo "========================================="
echo "评测完成！"
echo "========================================="
echo "轨迹文件: $OUTPUT_DIR/trajectory.jsonl"
echo "对话历史: $OUTPUT_DIR/chats/"
echo ""
