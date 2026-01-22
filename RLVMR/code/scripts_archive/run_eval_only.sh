#!/bin/bash
# =============================================================================
# 评测脚本 - 需要先启动vLLM
# =============================================================================

set -e
cd "$(dirname "$0")"

# 自动查找最新的模型
MODEL_DIR="./checkpoints/cold_start/alfworld/bdrs_qwen1.5b_2gpu_20251110"
LATEST_CHECKPOINT=$(ls -d ${MODEL_DIR}/global_step_* 2>/dev/null | sort -V | tail -1)

if [ -z "${LATEST_CHECKPOINT}" ]; then
    echo "错误: 找不到 checkpoint"
    exit 1
fi

# 配置参数
NUM_ENVS=64
BATCH_SIZE=8
MAX_STEPS=30
EXPERIMENT_NAME="eval_$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="results/${EXPERIMENT_NAME}"

echo "========================================="
echo "评测配置"
echo "========================================="
echo "模型: ${LATEST_CHECKPOINT}"
echo "评测环境数: ${NUM_ENVS}"
echo "========================================="
echo ""

# 检查vLLM是否在运行
if ! curl -s http://localhost:8000/v1/models >/dev/null 2>&1; then
    echo "⚠️  vLLM服务器未运行！"
    echo ""
    echo "请先在另一个终端启动vLLM："
    echo "  bash start_vllm.sh"
    echo ""
    echo "或者按照以下步骤："
    echo "  1. 打开新终端"
    echo "  2. cd /root/digitalhuman/RLVMR/code"
    echo "  3. bash start_vllm.sh"
    echo "  4. 等待看到 'Uvicorn running on http://0.0.0.0:8000'"
    echo "  5. 回到本终端重新运行: bash run_eval_only.sh"
    echo ""
    exit 1
fi

echo "✅ vLLM服务器已在运行"
mkdir -p "$OUTPUT_DIR"

# 运行评测
echo ""
echo ">>> 开始评测..."
bash examples/bdrs_trainer/rollout/run_local_eval.sh \
    "${LATEST_CHECKPOINT}" \
    ${NUM_ENVS} \
    ${BATCH_SIZE} \
    ${MAX_STEPS} \
    "${OUTPUT_DIR}"

# 解析结果并记录到WandB
echo ""
echo ">>> 记录到WandB..."
python3 << EOF
import json, wandb
from collections import defaultdict

with open("${OUTPUT_DIR}/trajectory.jsonl", 'r') as f:
    episodes = defaultdict(lambda: {'steps': 0, 'won': False})
    for line in f:
        d = json.loads(line)
        env_id = d['env_id']
        episodes[env_id]['steps'] += 1
        if d.get('done'): episodes[env_id]['won'] = d.get('won', False)

    eps = list(episodes.values())
    success_rate = sum(1 for e in eps if e['won']) / len(eps)
    avg_steps = sum(e['steps'] for e in eps) / len(eps)

    print(f"\n========================================")
    print(f"评测结果")
    print(f"========================================")
    print(f"成功率: {success_rate:.2%}")
    print(f"平均步数: {avg_steps:.1f}")
    print(f"========================================\n")

    wandb.init(project="RLVMR_Evaluation", name="${EXPERIMENT_NAME}")
    wandb.log({"val/success_rate": success_rate, "val/episode_length": avg_steps})
    wandb.finish()
    print("✅ 已记录到WandB")
EOF

echo ""
echo "========================================="
echo "✅ 评测完成！"
echo "========================================="
echo "结果目录: ${OUTPUT_DIR}"
echo "WandB: https://wandb.ai"
echo "========================================="

