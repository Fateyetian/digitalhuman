#!/bin/bash
# =============================================================================
# V6 快速启动脚本
# =============================================================================
#
# 用法:
#   ./run_v6_quick.sh 1    # 运行 exp1 (基础配置)
#   ./run_v6_quick.sh 2    # 运行 exp2 (相对阈值)
#   ./run_v6_quick.sh 3    # 运行 exp3 (全部改进)
#   ./run_v6_quick.sh all  # 依次运行所有实验
#
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

run_experiment() {
    local exp_num=$1
    echo "═══════════════════════════════════════════════════════════════════"
    echo "  启动 V6 实验 ${exp_num}"
    echo "═══════════════════════════════════════════════════════════════════"
    EXPERIMENT=$exp_num bash run_v6_experiments.sh
}

case "${1:-}" in
    1|2|3)
        run_experiment $1
        ;;
    all)
        echo "将依次运行所有 V6 实验..."
        for exp in 1 2 3; do
            run_experiment $exp
        done
        ;;
    *)
        echo "V6 实验快速启动"
        echo ""
        echo "用法: $0 <实验编号|all>"
        echo ""
        echo "实验说明:"
        echo "  1 - V6-Exp1: 基础配置 (验证代码修复效果)"
        echo "  2 - V6-Exp2: 相对阈值归一化 (min_samples_ratio=0.15)"
        echo "  3 - V6-Exp3: 全部改进 + KL in Reward"
        echo "  all - 依次运行所有实验"
        echo ""
        echo "示例:"
        echo "  $0 1      # 运行实验1"
        echo "  $0 all    # 运行所有实验"
        echo ""
        ;;
esac
