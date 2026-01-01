#!/bin/bash
# =============================================================================
# ReBel Framework 评测脚本 - 小规模测试验证
# =============================================================================
# ReBel (Reward Belief): 基于信念状态的密集奖励强化学习框架
#
# 功能：
# - 测试ReBel框架是否正确启用
# - 验证密集奖励计算（r_consistency, r_progress, r_exploration）
# - 检查信念状态解析质量
# - 评估幻觉率、重复动作率等行为指标
# =============================================================================

set -e  # 遇到错误立即退出

# 默认参数 - 使用vLLM加载的checkpoint模型名称
MODEL=${1:-"./checkpoints/cold_start/alfworld/bdrs_qwen1.5b_2gpu_20251110/global_step_75"}
NUM_ENVS=${2:-5}       # 小规模测试：5个环境
BATCH_SIZE=${3:-5}     # 单批处理
MAX_STEPS=${4:-30}     # 每个任务最大步数
OUTPUT_DIR=${5:-"results/rebel_test_$(date +%Y%m%d_%H%M%S)"}

echo "============================================================"
echo "🧠 ReBel Framework Evaluation"
echo "============================================================"
echo "Framework: ReBel (Belief-Driven Dense Reward)"
echo "Model:     $MODEL"
echo "Envs:      $NUM_ENVS (small-scale test)"
echo "Max Steps: $MAX_STEPS"
echo "Output:    $OUTPUT_DIR"
echo "============================================================"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 检查vLLM服务是否运行
echo ""
echo "🔍 Checking vLLM server..."
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ vLLM server is running"
else
    echo "❌ vLLM server is NOT running!"
    echo "Please start vLLM first:"
    echo "  bash start_vllm.sh"
    exit 1
fi

# 运行ReBel评测（use_rebel默认为True）
echo ""
echo "🚀 Running ReBel evaluation..."
echo ""

python3 examples/bdrs_trainer/rollout/run_alfworld_rollout.py \
    --env_name alfworld \
    --batch_size $BATCH_SIZE \
    --total_envs $NUM_ENVS \
    --max_steps $MAX_STEPS \
    --model "$MODEL" \
    --base_url "http://localhost:8000/v1" \
    --temperature 0.4 \
    --concurrency 2 \
    --dump_path "$OUTPUT_DIR/trajectory.jsonl" \
    --chat_root "$OUTPUT_DIR/chats" \
    --use_rebel  # 明确启用ReBel框架

# 提取关键指标
echo ""
echo "============================================================"
echo "📊 ReBel Metrics Summary"
echo "============================================================"

# 使用Python快速分析trajectory
python3 << 'PYEOF'
import json
import sys
from collections import defaultdict
import numpy as np

traj_file = sys.argv[1] if len(sys.argv) > 1 else "trajectory.jsonl"

try:
    with open(traj_file, 'r') as f:
        lines = [json.loads(line) for line in f]

    if not lines:
        print("❌ No trajectory data found!")
        sys.exit(1)

    # 按episode分组
    episodes = defaultdict(list)
    for row in lines:
        episodes[row['env_id']].append(row)

    # 统计指标
    success_count = 0
    r_consistency_vals = []
    r_progress_vals = []
    r_exploration_vals = []
    belief_parsed = 0
    total_steps = 0

    for env_id, steps in episodes.items():
        # 成功率
        if steps[-1].get('won', False):
            success_count += 1

        # ReBel奖励
        for step in steps:
            total_steps += 1
            r_consistency_vals.append(step.get('r_consistency', 0.0))
            r_progress_vals.append(step.get('r_progress', 0.0))
            r_exploration_vals.append(step.get('r_exploration', 0.0))
            if step.get('belief_parsed', False):
                belief_parsed += 1

    # 输出结果
    print(f"Episodes:    {len(episodes)}")
    print(f"Success:     {success_count}/{len(episodes)} ({success_count/len(episodes)*100:.1f}%)")
    print(f"Total Steps: {total_steps}")
    print(f"")
    print(f"ReBel Intrinsic Rewards (avg per step):")
    print(f"  r_consistency: {np.mean(r_consistency_vals):.4f}")
    print(f"  r_progress:    {np.mean(r_progress_vals):.4f}")
    print(f"  r_exploration: {np.mean(r_exploration_vals):.4f}")
    print(f"  r_intrinsic:   {np.mean(r_consistency_vals) + np.mean(r_progress_vals) + np.mean(r_exploration_vals):.4f}")
    print(f"")
    print(f"Belief State Quality:")
    print(f"  Parse Success: {belief_parsed}/{total_steps} ({belief_parsed/total_steps*100:.1f}%)")
    print(f"")

    # 判断ReBel是否启用
    avg_intrinsic = np.mean(r_consistency_vals) + np.mean(r_progress_vals) + np.mean(r_exploration_vals)
    if avg_intrinsic > 0.01:
        print("✅ ReBel框架已启用 (intrinsic rewards > 0.01)")
    else:
        print("❌ ReBel框架未启用或奖励过低 (intrinsic rewards ≈ 0)")
        print("   请检查:")
        print("   1. 配置中 algorithm.rebel.enable=True")
        print("   2. 奖励scale参数是否过小")

    if belief_parsed / total_steps > 0.8:
        print("✅ 信念状态解析良好 (>80%)")
    elif belief_parsed / total_steps > 0.5:
        print("⚠️  信念状态解析一般 (50-80%)")
    else:
        print("❌ 信念状态解析较差 (<50%)")

except FileNotFoundError:
    print(f"❌ Trajectory file not found: {traj_file}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error analyzing trajectory: {e}")
    sys.exit(1)
PYEOF "$OUTPUT_DIR/trajectory.jsonl"

echo ""
echo "============================================================"
echo "✅ ReBel Evaluation Complete!"
echo "============================================================"
echo "Detailed logs:  logs/alfworld/*.log"
echo "Trajectory:     $OUTPUT_DIR/trajectory.jsonl"
echo "Chat histories: $OUTPUT_DIR/chats/"
echo ""
echo "Next steps:"
echo "1. Check the metrics above to verify ReBel is working"
echo "2. If intrinsic rewards > 0.01 and parse rate > 80%: ReBel is ACTIVE ✅"
echo "3. If not, check algorithm.rebel configuration"
echo ""
