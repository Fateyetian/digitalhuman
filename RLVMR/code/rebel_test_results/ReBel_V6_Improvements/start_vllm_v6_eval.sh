#!/bin/bash
# =============================================================================
# Start vLLM Server for V6 Evaluation
# 启动vLLM服务用于评测模型
# =============================================================================

# 模型选择 (与run_v6_detailed_eval.sh保持一致)
# 1 = SFT模型 (HuggingFace格式)
# 2 = V6 Exp1 60epoch模型 (需先转换)
MODEL_CHOICE="${MODEL_CHOICE:-1}"

# SFT模型路径
SFT_MODEL_PATH="/root/testttt/RLVMR/code/checkpoints/cold_start/alfworld/rebel_full_20251225_024950/global_step_90"

# V6 Exp1 60epoch 转换后的模型路径
V6_HF_PATH="/root/testttt/RLVMR/code/rebel_test_results/ReBel_V6_Improvements/v6_exp1_epoch60_hf"

# 根据选择设置模型路径
if [ "$MODEL_CHOICE" == "1" ]; then
    MODEL_PATH="$SFT_MODEL_PATH"
    MODEL_NAME="SFT_epoch90"
elif [ "$MODEL_CHOICE" == "2" ]; then
    MODEL_PATH="$V6_HF_PATH"
    MODEL_NAME="V6_exp1_epoch60"

    if [ ! -d "$V6_HF_PATH" ] || [ ! -f "$V6_HF_PATH/config.json" ]; then
        echo "Error: V6 HuggingFace model not found at $V6_HF_PATH"
        echo ""
        echo "Please run conversion first:"
        echo "  MODEL_CHOICE=2 bash run_v6_detailed_eval.sh"
        echo ""
        echo "Or manually convert with:"
        echo "  python scripts/model_merger.py --backend fsdp \\"
        echo "    --hf_model_path /root/testttt/RLVMR/code/base_models/Qwen2.5-1.5B-Instruct \\"
        echo "    --local_dir /fs-computility-new/.../global_step_60/actor \\"
        echo "    --target_dir $V6_HF_PATH"
        exit 1
    fi
else
    echo "Invalid MODEL_CHOICE: $MODEL_CHOICE (use 1 for SFT, 2 for V6_epoch60)"
    exit 1
fi

# vLLM配置
PORT=${PORT:-8000}
TP_SIZE=${TP_SIZE:-1}
GPU_MEM=${GPU_MEM:-0.8}

# 检查模型是否存在
if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model not found at $MODEL_PATH"
    exit 1
fi

echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "  Starting vLLM Server for V6 Evaluation"
echo "═══════════════════════════════════════════════════════════════════"
echo ""
echo "Model Name: $MODEL_NAME"
echo "Model Path: $MODEL_PATH"
echo "Port: $PORT"
echo "Tensor Parallel: $TP_SIZE"
echo "GPU Memory: $GPU_MEM"
echo ""

# 设置环境变量
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_ATTENTION_BACKEND=XFORMERS

# 启动vLLM服务
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --tensor-parallel-size $TP_SIZE \
    --gpu-memory-utilization $GPU_MEM \
    --port $PORT \
    --trust-remote-code \
    --max-model-len 8192
