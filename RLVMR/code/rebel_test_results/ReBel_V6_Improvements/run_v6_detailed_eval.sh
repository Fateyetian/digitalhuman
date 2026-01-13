#!/bin/bash
# =============================================================================
# V6 Detailed Evaluation Script
# 详细评测脚本：评测60epoch模型，记录任务分布、失败轨迹、step奖励
# =============================================================================

set -e

# 配置参数
ENV_NUM=${ENV_NUM:-128}          # 评测环境数量
MAX_STEPS=${MAX_STEPS:-30}       # 每个episode最大步数
SEED=${SEED:-42}                 # 随机种子
VLLM_PORT=${VLLM_PORT:-8000}     # vLLM服务端口
BASE_URL="http://127.0.0.1:${VLLM_PORT}/v1"
TEMPERATURE=0.0                  # 使用温度0进行确定性评测

# 模型选择
# 1 = SFT模型 (HuggingFace格式，可直接使用)
# 2 = V6 Exp1 60epoch模型 (FSDP格式，需先转换)
MODEL_CHOICE="${MODEL_CHOICE:-1}"

# SFT模型路径 (HuggingFace格式)
SFT_MODEL_PATH="/root/testttt/RLVMR/code/checkpoints/cold_start/alfworld/rebel_full_20251225_024950/global_step_90"

# V6 Exp1 60epoch模型路径 (FSDP格式)
V6_FSDP_PATH="/fs-computility-new/UPDZ03_chengjun/huangsijie.p/rebel_results/v6_experiments/rebel_v6_exp1_baseline_20260107_120300/checkpoints/global_step_60/actor"
V6_HF_PATH="/root/testttt/RLVMR/code/rebel_test_results/ReBel_V6_Improvements/v6_exp1_epoch60_hf"
BASE_HF_MODEL="/root/testttt/RLVMR/code/base_models/Qwen2.5-1.5B-Instruct"

# 输出目录
OUTPUT_DIR="/root/testttt/RLVMR/code/rebel_test_results/ReBel_V6_Improvements/detailed_eval_results"

# 根据选择设置模型路径
if [ "$MODEL_CHOICE" == "1" ]; then
    MODEL_PATH="$SFT_MODEL_PATH"
    MODEL_NAME="SFT_epoch90"
elif [ "$MODEL_CHOICE" == "2" ]; then
    # 检查是否需要转换FSDP模型
    if [ ! -d "$V6_HF_PATH" ] || [ ! -f "$V6_HF_PATH/config.json" ]; then
        echo "V6 60epoch model needs to be converted from FSDP format..."
        echo ""

        if [ ! -d "$V6_FSDP_PATH" ]; then
            echo "Error: FSDP checkpoint not found at $V6_FSDP_PATH"
            echo "Please check if the training completed and saved checkpoint."
            exit 1
        fi

        echo "Converting FSDP checkpoint to HuggingFace format..."
        cd /root/testttt/RLVMR/code
        python scripts/model_merger.py \
            --backend fsdp \
            --hf_model_path "$BASE_HF_MODEL" \
            --local_dir "$V6_FSDP_PATH" \
            --target_dir "$V6_HF_PATH"

        # 复制tokenizer文件
        if [ -d "$BASE_HF_MODEL" ]; then
            echo "Copying tokenizer files..."
            cp -n "$BASE_HF_MODEL"/*.json "$V6_HF_PATH/" 2>/dev/null || true
            cp -n "$BASE_HF_MODEL"/tokenizer* "$V6_HF_PATH/" 2>/dev/null || true
            cp -n "$BASE_HF_MODEL"/merges.txt "$V6_HF_PATH/" 2>/dev/null || true
            cp -n "$BASE_HF_MODEL"/vocab.json "$V6_HF_PATH/" 2>/dev/null || true
            cp -n "$BASE_HF_MODEL"/special_tokens_map.json "$V6_HF_PATH/" 2>/dev/null || true
        fi

        echo "Model conversion completed: $V6_HF_PATH"
    else
        echo "V6 60epoch HuggingFace model already exists at: $V6_HF_PATH"
    fi
    MODEL_PATH="$V6_HF_PATH"
    MODEL_NAME="V6_exp1_epoch60"
else
    echo "Invalid MODEL_CHOICE: $MODEL_CHOICE (use 1 for SFT, 2 for V6_epoch60)"
    exit 1
fi

# 检查模型是否存在
if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model not found at $MODEL_PATH"
    exit 1
fi

echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "  ReBel V6 Detailed Evaluation"
echo "═══════════════════════════════════════════════════════════════════"
echo ""
echo "Configuration:"
echo "  - Model Name: $MODEL_NAME"
echo "  - Model Path: $MODEL_PATH"
echo "  - Environments: $ENV_NUM"
echo "  - Max Steps: $MAX_STEPS"
echo "  - Seed: $SEED"
echo "  - Temperature: $TEMPERATURE"
echo "  - Output: $OUTPUT_DIR"
echo ""

# 步骤1: 检查或启动vLLM服务
echo "[1/3] Checking vLLM service..."
if curl -s "$BASE_URL/models" > /dev/null 2>&1; then
    echo "vLLM service is running"

    # 检查当前加载的模型
    CURRENT_MODEL=$(curl -s "$BASE_URL/models" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['data'][0]['id'] if d['data'] else 'None')" 2>/dev/null)
    echo "Current model: $CURRENT_MODEL"

    if [ "$CURRENT_MODEL" != "$MODEL_PATH" ]; then
        echo ""
        echo "WARNING: Different model is loaded!"
        echo "Need to restart vLLM with the correct model."
        echo ""
        echo "Please run in another terminal:"
        echo "  pkill -f vllm.entrypoints"
        echo "  bash $(dirname $0)/start_vllm_v6_eval.sh"
        echo ""
        read -p "Press Enter after vLLM is restarted with correct model, or Ctrl+C to exit..."
    fi
else
    echo "vLLM service not running!"
    echo ""
    echo "Please start vLLM in another terminal with:"
    echo "  bash $(dirname $0)/start_vllm_v6_eval.sh"
    echo ""
    echo "Or run directly:"
    echo "  python -m vllm.entrypoints.openai.api_server \\"
    echo "    --model $MODEL_PATH \\"
    echo "    --tensor-parallel-size 1 \\"
    echo "    --gpu-memory-utilization 0.8 \\"
    echo "    --port $VLLM_PORT"
    echo ""
    exit 1
fi

# 步骤2: 运行评测
echo ""
echo "[2/3] Running detailed evaluation..."
echo ""

cd /root/testttt/RLVMR/code

python3 rebel_test_results/ReBel_V6_Improvements/run_detailed_evaluation.py \
    --env_num $ENV_NUM \
    --max_steps $MAX_STEPS \
    --seed $SEED \
    --base_url "$BASE_URL" \
    --model "$MODEL_PATH" \
    --temperature $TEMPERATURE \
    --output_dir "$OUTPUT_DIR"

# 步骤3: 显示结果摘要
echo ""
echo "[3/3] Evaluation complete!"
echo "═══════════════════════════════════════════════════════════════════"

# 查找最新结果目录
LATEST_DIR=$(ls -td "$OUTPUT_DIR"/eval_* 2>/dev/null | head -1)

if [ -n "$LATEST_DIR" ]; then
    echo ""
    echo "Results directory: $LATEST_DIR"
    echo ""
    echo "Output files:"
    ls -la "$LATEST_DIR/"
    echo ""

    # 显示结果摘要
    if [ -f "$LATEST_DIR/results.json" ]; then
        echo "Quick Summary:"
        echo "─────────────────────────────────────────────────────────────────────"
        python3 << PYEOF
import json
with open('$LATEST_DIR/results.json') as f:
    r = json.load(f)
    print(f"Success Rate: {r['basic_metrics']['success_rate']*100:.1f}%")
    print(f"Average Steps: {r['basic_metrics']['avg_episode_length']:.1f}")
    print(f"Average Reward: {r['basic_metrics']['avg_episode_reward']:.2f}")
    print()
    print("Per-Task Success Rate:")
    for task, metrics in sorted(r['per_task_metrics'].items()):
        sr = metrics['success_rate'] * 100
        cnt = metrics['count']
        succ = metrics['successes']
        print(f"  {task}: {sr:.1f}% ({succ}/{cnt})")
    print()
    print("Average Step Rewards:")
    sr = r['avg_step_rewards']
    print(f"  r_consistency: {sr['r_consistency']:.6f}")
    print(f"  r_progress:    {sr['r_progress']:.6f}")
    print(f"  r_exploration: {sr['r_exploration']:.6f}")
    print(f"  r_format:      {sr['r_format']:.6f}")
    print(f"  r_intrinsic:   {sr['r_intrinsic']:.6f}")
PYEOF
        echo "─────────────────────────────────────────────────────────────────────"
    fi

    echo ""
    echo "To view failed trajectories:"
    echo "  head -1 $LATEST_DIR/failed_trajectories.jsonl | python3 -m json.tool"
    echo ""
    echo "To analyze step rewards:"
    echo "  python3 -c \"import json; print(json.load(open('$LATEST_DIR/step_rewards.json'))['0'])\""
fi

echo ""
echo "═══════════════════════════════════════════════════════════════════"
