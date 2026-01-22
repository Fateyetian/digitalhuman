#!/bin/bash
# 启动vLLM服务器

MODEL_PATH="/root/testttt/RLVMR/code/base_models/Qwen2.5-1.5B-Instruct"

echo "======================================"
echo "启动vLLM服务器"
echo "======================================"
echo "模型: ${MODEL_PATH}"
echo "端口: 8000"
echo "GPU利用率: 0.85"
echo "======================================"

# 清理GPU - 杀掉所有可能占用GPU的Python进程
echo "清理GPU内存..."

# 1. 杀掉vLLM进程
OLD_VLLM_PID=$(ps aux | grep "vllm.entrypoints.openai.api_server" | grep -v grep | awk '{print $2}')
if [ ! -z "$OLD_VLLM_PID" ]; then
    echo "  - 停止旧的vLLM进程 (PID: $OLD_VLLM_PID)..."
    kill -9 $OLD_VLLM_PID 2>/dev/null
fi

# 2. 杀掉所有multiprocessing Python进程（通常是训练或推理残留）
MULTI_PIDS=$(ps aux | grep "multiprocessing" | grep python | grep -v grep | awk '{print $2}')
if [ ! -z "$MULTI_PIDS" ]; then
    echo "  - 清理残留的multiprocessing进程..."
    echo "$MULTI_PIDS" | xargs -r kill -9 2>/dev/null
fi

# 3. 等待GPU内存释放
echo "  - 等待GPU内存释放..."
sleep 3

# 4. 检查GPU状态
GPU_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
echo "  - GPU 0 内存使用: ${GPU_MEM}MB"
if [ "$GPU_MEM" -gt 1000 ]; then
    echo "⚠️  警告: GPU内存仍然占用较高，可能影响vLLM启动"
fi

echo "✅ GPU清理完成"
echo ""

# 设置环境变量避免CUDA fork问题
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# 启动vLLM (优化参数以减少内存占用)
python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_PATH}" \
    --host 0.0.0.0 \
    --port 8000 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 4096 \
    --tensor-parallel-size 1 \
    --disable-log-requests
