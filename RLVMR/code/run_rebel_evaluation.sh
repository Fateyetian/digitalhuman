#!/bin/bash
# ReBel Rollout 一键评测脚本

set -e

echo "=========================================="
echo "ReBel Rollout 快速评测"
echo "=========================================="

# 配置参数
ENV_NUM=4
MAX_STEPS=30
SEED=1
VLLM_PORT=8000
BASE_URL="http://127.0.0.1:${VLLM_PORT}/v1"
MODEL_PATH="/root/testttt/RLVMR/code/base_models/Qwen2.5-1.5B-Instruct"

# 步骤1: 检查vLLM服务是否运行
echo "[1/3] 检查vLLM服务..."
if ! curl -s "$BASE_URL/models" > /dev/null 2>&1; then
    echo "❌ vLLM服务未运行！"
    echo ""
    echo "请在另一个终端运行:"
    echo "  bash start_vllm_server.sh"
    echo ""
    echo "或后台运行:"
    echo "  nohup bash start_vllm_server.sh > vllm.log 2>&1 &"
    echo ""
    echo "等待服务启动后再运行本脚本"
    exit 1
fi

echo "✅ vLLM服务正在运行"
echo ""

# 步骤2: 运行ReBel rollout
echo "[2/3] 运行ReBel rollout评测..."
echo "   - 环境数量: $ENV_NUM"
echo "   - 最大步数: $MAX_STEPS"
echo "   - 随机种子: $SEED"
echo ""

python3 run_rebel_rollout.py \
    --env_num $ENV_NUM \
    --max_steps $MAX_STEPS \
    --seed $SEED \
    --base_url "$BASE_URL" \
    --model "/root/testttt/RLVMR/code/base_models/Qwen2.5-1.5B-Instruct" \
    --temperature 0.4 \
    --output_dir rebel_rollout_results

# 步骤3: 显示结果
echo ""
echo "[3/3] 评测完成！"
echo "=========================================="

# 查找最新结果目录
LATEST_DIR=$(ls -td rebel_rollout_results/run_* 2>/dev/null | head -1)

if [ -n "$LATEST_DIR" ]; then
    echo "📁 结果目录: $LATEST_DIR"
    echo ""
    echo "关键文件:"
    echo "  - results.json        : 统计结果"
    echo "  - trajectories.jsonl  : 完整轨迹"
    echo "  - rollout.log         : 运行日志"
    echo ""

    # 显示简要结果
    if [ -f "$LATEST_DIR/results.json" ]; then
        echo "📊 快速查看结果:"
        echo "----------------------------------------"
        python3 -c "
import json
with open('$LATEST_DIR/results.json') as f:
    r = json.load(f)
    print(f\"✅ 成功率: {r['basic_metrics']['success_rate']*100:.1f}%\")
    print(f\"📏 平均步数: {r['basic_metrics']['avg_episode_length']:.1f}\")
    print(f\"🏆 平均奖励: {r['basic_metrics']['avg_episode_reward']:.2f}\")
    print(f\"\")
    print(f\"🧠 ReBel指标:\")
    print(f\"  - 一致性奖励: {r['rebel_metrics']['avg_r_consistency']:.4f}\")
    print(f\"  - 进度奖励:   {r['rebel_metrics']['avg_r_progress']:.4f}\")
    print(f\"  - 探索奖励:   {r['rebel_metrics']['avg_r_exploration']:.4f}\")
    print(f\"  - 格式奖励:   {r['rebel_metrics']['avg_r_format']:.4f}\")
    print(f\"  - 信念解析率: {r['rebel_metrics']['avg_belief_parse_rate']*100:.1f}%\")
"
        echo "----------------------------------------"
    fi

    echo ""
    echo "查看完整结果:"
    echo "  cat $LATEST_DIR/results.json"
    echo ""
    echo "查看详细日志:"
    echo "  cat $LATEST_DIR/rollout.log"
fi

echo "=========================================="
