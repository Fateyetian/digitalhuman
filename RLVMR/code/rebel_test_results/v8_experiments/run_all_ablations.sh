#!/bin/bash
# =============================================================================
# ReBel V8 消融实验 - 主控脚本
# =============================================================================
# 运行所有消融实验：
# 1. 移除结果奖励 (No Result Reward)
# 2. 移除信念奖励 (No Belief Reward)
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NUM_GPUS=${NUM_GPUS:-8}
EPOCHS=${EPOCHS:-100}
SEED=${SEED:-42}

echo "═══════════════════════════════════════════════════════════════════"
echo "  ReBel V8 消融实验套件"
echo "═══════════════════════════════════════════════════════════════════"
echo ""
echo "配置:"
echo "  - GPU数量: ${NUM_GPUS}"
echo "  - Epochs: ${EPOCHS}"
echo "  - 随机种子: ${SEED}"
echo ""
echo "实验列表:"
echo "  1. 移除结果奖励 (只使用信念奖励)"
echo "  2. 移除信念奖励 (只使用结果奖励)"
echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo ""

# 询问用户要运行哪个实验
if [ -z "$ABLATION_TYPE" ]; then
    echo "请选择要运行的消融实验:"
    echo "  1) 移除结果奖励"
    echo "  2) 移除信念奖励"
    echo "  3) 运行所有实验 (串行)"
    echo "  q) 退出"
    echo ""
    read -p "请输入选项 [1/2/3/q]: " choice
else
    choice=$ABLATION_TYPE
fi

case $choice in
    1)
        echo ""
        echo "───────────────────────────────────────────────────────────────────"
        echo "  开始实验 1: 移除结果奖励"
        echo "───────────────────────────────────────────────────────────────────"
        echo ""

        NUM_GPUS=$NUM_GPUS EPOCHS=$EPOCHS SEED=$SEED \
            bash "${SCRIPT_DIR}/run_ablation_no_result_reward.sh"

        echo ""
        echo "✓ 实验 1 完成"
        ;;

    2)
        echo ""
        echo "───────────────────────────────────────────────────────────────────"
        echo "  开始实验 2: 移除信念奖励"
        echo "───────────────────────────────────────────────────────────────────"
        echo ""

        NUM_GPUS=$NUM_GPUS EPOCHS=$EPOCHS SEED=$SEED \
            bash "${SCRIPT_DIR}/run_ablation_no_belief_reward.sh"

        echo ""
        echo "✓ 实验 2 完成"
        ;;

    3)
        echo ""
        echo "───────────────────────────────────────────────────────────────────"
        echo "  串行运行所有消融实验"
        echo "───────────────────────────────────────────────────────────────────"
        echo ""

        # 实验 1: 移除结果奖励
        echo ""
        echo "【1/2】开始实验: 移除结果奖励"
        echo ""
        NUM_GPUS=$NUM_GPUS EPOCHS=$EPOCHS SEED=$SEED \
            bash "${SCRIPT_DIR}/run_ablation_no_result_reward.sh"
        echo ""
        echo "✓ 实验 1/2 完成"
        echo ""

        # 实验 2: 移除信念奖励
        echo ""
        echo "【2/2】开始实验: 移除信念奖励"
        echo ""
        NUM_GPUS=$NUM_GPUS EPOCHS=$EPOCHS SEED=$SEED \
            bash "${SCRIPT_DIR}/run_ablation_no_belief_reward.sh"
        echo ""
        echo "✓ 实验 2/2 完成"
        echo ""

        echo ""
        echo "═══════════════════════════════════════════════════════════════════"
        echo "  所有消融实验完成！"
        echo "═══════════════════════════════════════════════════════════════════"
        ;;

    q|Q)
        echo "退出"
        exit 0
        ;;

    *)
        echo "无效选项: $choice"
        exit 1
        ;;
esac

echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "  完成！"
echo "═══════════════════════════════════════════════════════════════════"
echo ""
echo "结果保存在: /fs-computility-new/UPDZ03_chengjun/huangsijie.p/rebel_results/v8_experiments/"
echo ""
