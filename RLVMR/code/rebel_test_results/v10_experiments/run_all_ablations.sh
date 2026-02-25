#!/bin/bash
# =============================================================================
# ReBel V10 消融实验编排脚本
# =============================================================================
# 按优先级分批运行所有消融实验
#
# 使用方法:
#   bash run_all_ablations.sh                # 运行所有实验 (顺序执行)
#   bash run_all_ablations.sh batch1         # 只运行第一批 (A1+A2+B1)
#   bash run_all_ablations.sh batch2         # 只运行第二批 (C1+C2)
#   bash run_all_ablations.sh batch3         # 只运行第三批 (D1+E1)
#   bash run_all_ablations.sh A1             # 只运行单个实验
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="${SCRIPT_DIR}/experiments"

BATCH=${1:-all}

run_experiment() {
    local exp_file=$1
    local exp_name=$(basename "$exp_file" .sh)
    echo ""
    echo "╔═══════════════════════════════════════════════════════════════════╗"
    echo "║  启动实验: ${exp_name}"
    echo "╚═══════════════════════════════════════════════════════════════════╝"
    echo ""
    bash "${exp_file}"
    echo ""
    echo ">>> 实验完成: ${exp_name}"
    echo ""
}

echo "═══════════════════════════════════════════════════════════════════"
echo "  ReBel V10 系统消融实验"
echo "═══════════════════════════════════════════════════════════════════"
echo ""
echo "  批次: ${BATCH}"
echo ""

case "$BATCH" in
    batch1|all)
        echo ">>> 第一批: A1 (GRPO基线) + A2 (GRPO+tricks) + B1 (GRPO+信念提示)"
        echo ">>> 目标: 确定 tricks 和信念提示格式的各自贡献"
        echo ""
        run_experiment "${EXP_DIR}/A1_grpo_baseline.sh"
        run_experiment "${EXP_DIR}/A2_grpo_tricks.sh"
        run_experiment "${EXP_DIR}/B1_grpo_belief_prompt.sh"
        echo ""
        echo "======================================================="
        echo "  第一批完成！请检查结果后决定是否继续。"
        echo "  关键对比: B1 vs A2 → 信念提示格式的贡献"
        echo "  如果 B1 >> A2: 继续第二批"
        echo "  如果 B1 ≈ A2: 需要重新评估方向"
        echo "======================================================="
        ;;&

    batch2|all)
        echo ">>> 第二批: C1 (GiGPO+信念提示) + C2 (ReBel信念分组)"
        echo ">>> 目标: 确定信念分组 vs 观测分组的贡献"
        echo ""
        run_experiment "${EXP_DIR}/C1_gigpo_belief_prompt.sh"
        run_experiment "${EXP_DIR}/C2_rebel_group_only.sh"
        echo ""
        echo "======================================================="
        echo "  第二批完成！请检查结果后决定是否继续。"
        echo "  关键对比: C2 vs C1 → 信念分组的贡献"
        echo "======================================================="
        ;;&

    batch3|all)
        echo ">>> 第三批: D1 (完整ReBel) + E1 (ReBel+课程衰减)"
        echo ">>> 目标: 确定内在奖励和课程学习的贡献"
        echo ""
        run_experiment "${EXP_DIR}/D1_rebel_full.sh"
        run_experiment "${EXP_DIR}/E1_rebel_curriculum.sh"
        echo ""
        echo "======================================================="
        echo "  第三批完成！"
        echo "  关键对比: D1 vs C2 → 内在奖励贡献"
        echo "  关键对比: E1 vs D1 → 课程衰减贡献"
        echo "  关键对比: E1 vs C2 → 课程化内在奖励整体贡献"
        echo "======================================================="
        ;;

    A1) run_experiment "${EXP_DIR}/A1_grpo_baseline.sh" ;;
    A2) run_experiment "${EXP_DIR}/A2_grpo_tricks.sh" ;;
    B1) run_experiment "${EXP_DIR}/B1_grpo_belief_prompt.sh" ;;
    C1) run_experiment "${EXP_DIR}/C1_gigpo_belief_prompt.sh" ;;
    C2) run_experiment "${EXP_DIR}/C2_rebel_group_only.sh" ;;
    D1) run_experiment "${EXP_DIR}/D1_rebel_full.sh" ;;
    E1) run_experiment "${EXP_DIR}/E1_rebel_curriculum.sh" ;;

    *)
        echo "未知的批次或实验: ${BATCH}"
        echo ""
        echo "用法:"
        echo "  bash run_all_ablations.sh          # 运行所有"
        echo "  bash run_all_ablations.sh batch1    # 第一批 (A1+A2+B1)"
        echo "  bash run_all_ablations.sh batch2    # 第二批 (C1+C2)"
        echo "  bash run_all_ablations.sh batch3    # 第三批 (D1+E1)"
        echo "  bash run_all_ablations.sh A1        # 单个实验"
        exit 1
        ;;
esac

echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "  所有指定实验已完成"
echo "═══════════════════════════════════════════════════════════════════"
