#!/bin/bash
# 启动vLLM服务（使用Qwen2.5-1.5B-Instruct基础模型）

set -e

MODEL_PATH="/root/testttt/RLVMR/code/base_models/Qwen2.5-1.5B-Instruct"
PORT=8000
GPU_MEMORY_UTILIZATION=0.5

echo "=========================================="
echo "启动vLLM服务器"
echo "=========================================="
echo "模型: $MODEL_PATH"
echo "端口: $PORT"
echo "GPU内存: ${GPU_MEMORY_UTILIZATION}"
echo "=========================================="

# 检查模型路径
if [ ! -d "$MODEL_PATH" ]; then
    echo "❌ 错误: 模型路径不存在: $MODEL_PATH"
    exit 1
fi

# 检查端口是否被占用
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "⚠️  端口 $PORT 已被占用"
    echo "尝试停止已有服务..."
    kill $(lsof -t -i:$PORT) 2>/dev/null || true
    sleep 2
fi

echo "✅ 启动vLLM服务器..."

# 设置环境变量避免CUDA fork问题和网络访问
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME=/root/.cache/huggingface

# 启动vLLM (使用稳定参数配置)
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --tokenizer "$MODEL_PATH" \
    --host 0.0.0.0 \
    --port $PORT \
    --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
    --max-model-len 4096 \
    --tensor-parallel-size 1 \
    --disable-log-requests \
    2>&1 | tee vllm_server.log

echo "vLLM服务器已停止"
