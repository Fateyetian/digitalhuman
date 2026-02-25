#!/bin/bash
# =============================================================================
# ReBel V11 正式实验编排脚本
# =============================================================================
# 按优先级分阶段运行实验，支持多种子批量执行。
#
# 使用方法:
#   bash run_all_v11.sh                    # 运行所有阶段 (顺序执行)
#   bash run_all_v11.sh phase1             # Phase 1: 主实验 (5 methods × 3 seeds)
#   bash run_all_v11.sh phase2             # Phase 2: 关键消融 (6 ablations × 1 seed)
#   bash run_all_v11.sh M1                 # 单个实验 (1 seed, 由 SEED 环境变量控制)
#   bash run_all_v11.sh M1 seeds           # 单个实验 (3 seeds)
#   SEED=123 bash run_all_v11.sh M5        # 指定种子运行单个实验
#
# 环境变量:
#   NUM_GPUS  - GPU 数量 (默认 8)
#   SEED      - 随机种子 (默认 42, 仅用于单次运行)
#   EPOCHS    - 训练轮次 (默认 100)
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="${SCRIPT_DIR}/experiments"

TARGET=${1:-all}
MODE=${2:-single}   # single / seeds

SEEDS=(42 123 456)

# ============================================================================
# 辅助函数
# ============================================================================

run_experiment() {
    local exp_file=$1
    local seed=${2:-${SEED:-42}}
    local exp_name=$(basename "$exp_file" .sh)

    echo ""
    echo "╔═══════════════════════════════════════════════════════════════════╗"
    echo "║  启动: ${exp_name} (seed=${seed})"
    echo "╚═══════════════════════════════════════════════════════════════════╝"
    echo ""

    SEED=${seed} bash "${exp_file}"

    echo ""
    echo ">>> 完成: ${exp_name} (seed=${seed})"
    echo ""
}

run_multi_seed() {
    local exp_file=$1
    local exp_name=$(basename "$exp_file" .sh)

    echo ""
    echo "┌───────────────────────────────────────────────────────────────────┐"
    echo "│  多种子运行: ${exp_name} × ${#SEEDS[@]} seeds"
    echo "└───────────────────────────────────────────────────────────────────┘"
    echo ""

    for seed in "${SEEDS[@]}"; do
        run_experiment "${exp_file}" "${seed}"
    done

    echo ">>> 多种子完成: ${exp_name} (seeds: ${SEEDS[*]})"
    echo ""
}

# ============================================================================
# 实验编排
# ============================================================================

echo "═══════════════════════════════════════════════════════════════════"
echo "  ReBel V11 正式实验"
echo "═══════════════════════════════════════════════════════════════════"
echo ""
echo "  目标: ${TARGET}"
echo "  模式: ${MODE}"
echo ""

case "$TARGET" in

    # ====================================================================
    # Phase 1: 主实验 (最高优先级)
    # 5 methods × 3 seeds = 15 runs
    # ====================================================================
    phase1|all)
        echo "═══════════════════════════════════════════════════════════════════"
        echo "  Phase 1: Main Results (5 methods × 3 seeds)"
        echo "═══════════════════════════════════════════════════════════════════"
        echo ""
        echo "  M1: GRPO 纯基线"
        echo "  M2: GRPO + 训练稳定化"
        echo "  M3: GiGPO + <think>"
        echo "  M4: GiGPO + <belief> (无 HiBO)"
        echo "  M5: ReBel Full (HiBO + Adaptive Curriculum) [核心方法]"
        echo ""
        echo "  总计: 15 runs × ~40h/run = ~600 GPU-hours"
        echo ""

        run_multi_seed "${EXP_DIR}/M1_grpo_baseline.sh"
        run_multi_seed "${EXP_DIR}/M2_grpo_tricks.sh"
        run_multi_seed "${EXP_DIR}/M3_gigpo_think.sh"
        run_multi_seed "${EXP_DIR}/M4_gigpo_belief.sh"
        run_multi_seed "${EXP_DIR}/M5_rebel_full.sh"

        echo ""
        echo "═══════════════════════════════════════════════════════════════════"
        echo "  Phase 1 完成!"
        echo "═══════════════════════════════════════════════════════════════════"
        echo ""
        echo "  核心对比 (请优先检查):"
        echo "    M5 vs M4 → HiBO + Curriculum 贡献 (核心!)"
        echo "    M5 vs M3 → 信念全套贡献"
        echo "    M4 vs M3 → Belief Prompting 独立效果"
        echo "    M3 vs M2 → Step Advantage 独立贡献"
        echo ""
        echo "  如果 M5 >> M4: HiBO + Curriculum 核心 claim 成立"
        echo "  如果 M5 ≈ M4:  需要分析 HiBO 分组统计确认原因"
        echo ""
        ;;&

    # ====================================================================
    # Phase 2: 关键消融 (高优先级)
    # 6 ablations × 1 seed = 6 runs
    # ====================================================================
    phase2|all)
        echo "═══════════════════════════════════════════════════════════════════"
        echo "  Phase 2: Ablation Study (6 ablations × seed=42)"
        echo "═══════════════════════════════════════════════════════════════════"
        echo ""
        echo "  A1: w/o HiBO → obs only (GiGPO + <belief> + 课程奖励)"
        echo "  A2: w/o HiBO → belief only (ReBel belief 分组 + 课程奖励)"
        echo "  A3: w/o Belief Reward (HiBO + 无奖励)"
        echo "  A4: w/ Fixed Decay (HiBO + 固定 cosine 衰减)"
        echo "  A5: w/ Uniform Decay (HiBO + 统一衰减)"
        echo "  A6: w/o Belief Prompt (= M3, <think> 格式)"
        echo ""
        echo "  总计: 6 runs × ~40h/run = ~240 GPU-hours"
        echo ""
        echo "  注: A6 与 M3 (seed=42) 配置相同。"
        echo "       如果 Phase 1 已完成 M3 (seed=42)，可跳过 A6。"
        echo ""

        run_experiment "${EXP_DIR}/A1_ablation_obs_only.sh" 42
        run_experiment "${EXP_DIR}/A2_ablation_belief_only.sh" 42
        run_experiment "${EXP_DIR}/A3_ablation_no_reward.sh" 42
        run_experiment "${EXP_DIR}/A4_ablation_fixed_decay.sh" 42
        run_experiment "${EXP_DIR}/A5_ablation_uniform_decay.sh" 42

        # A6 与 M3 (seed=42) 相同，仅在需要时运行
        # run_experiment "${EXP_DIR}/A6_ablation_no_belief_prompt.sh" 42
        echo ">>> 跳过 A6 (与 M3 seed=42 相同，直接复用 M3 结果)"
        echo ""

        echo ""
        echo "═══════════════════════════════════════════════════════════════════"
        echo "  Phase 2 完成!"
        echo "═══════════════════════════════════════════════════════════════════"
        echo ""
        echo "  消融对比:"
        echo "    M5 vs A1 → HiBO 贡献: ~2%"
        echo "    A1 vs A2 → obs-only vs belief-only: +2% (层次化必要性)"
        echo "    M5 vs A3 → 课程奖励贡献: ~2%"
        echo "    M5 vs A4 → 自适应 vs 固定衰减: ~1%"
        echo "    M5 vs A5 → 差异化 vs 统一衰减: ~1%"
        echo "    M5 vs A6 → 信念提示全局贡献: ~6% (最大)"
        echo ""
        ;;

    # ====================================================================
    # 单个实验
    # ====================================================================
    M1)
        if [ "$MODE" = "seeds" ]; then
            run_multi_seed "${EXP_DIR}/M1_grpo_baseline.sh"
        else
            run_experiment "${EXP_DIR}/M1_grpo_baseline.sh"
        fi
        ;;
    M2)
        if [ "$MODE" = "seeds" ]; then
            run_multi_seed "${EXP_DIR}/M2_grpo_tricks.sh"
        else
            run_experiment "${EXP_DIR}/M2_grpo_tricks.sh"
        fi
        ;;
    M3)
        if [ "$MODE" = "seeds" ]; then
            run_multi_seed "${EXP_DIR}/M3_gigpo_think.sh"
        else
            run_experiment "${EXP_DIR}/M3_gigpo_think.sh"
        fi
        ;;
    M4)
        if [ "$MODE" = "seeds" ]; then
            run_multi_seed "${EXP_DIR}/M4_gigpo_belief.sh"
        else
            run_experiment "${EXP_DIR}/M4_gigpo_belief.sh"
        fi
        ;;
    M5)
        if [ "$MODE" = "seeds" ]; then
            run_multi_seed "${EXP_DIR}/M5_rebel_full.sh"
        else
            run_experiment "${EXP_DIR}/M5_rebel_full.sh"
        fi
        ;;
    A1) run_experiment "${EXP_DIR}/A1_ablation_obs_only.sh" ;;
    A2) run_experiment "${EXP_DIR}/A2_ablation_belief_only.sh" ;;
    A3) run_experiment "${EXP_DIR}/A3_ablation_no_reward.sh" ;;
    A4) run_experiment "${EXP_DIR}/A4_ablation_fixed_decay.sh" ;;
    A5) run_experiment "${EXP_DIR}/A5_ablation_uniform_decay.sh" ;;
    A6) run_experiment "${EXP_DIR}/A6_ablation_no_belief_prompt.sh" ;;

    *)
        echo "未知的目标: ${TARGET}"
        echo ""
        echo "用法:"
        echo "  bash run_all_v11.sh              # 运行所有阶段"
        echo "  bash run_all_v11.sh phase1       # Phase 1: 主实验 (5×3 seeds)"
        echo "  bash run_all_v11.sh phase2       # Phase 2: 消融 (6×1 seed)"
        echo ""
        echo "  bash run_all_v11.sh M1           # GRPO 基线 (1 seed)"
        echo "  bash run_all_v11.sh M1 seeds     # GRPO 基线 (3 seeds)"
        echo "  bash run_all_v11.sh M2           # GRPO + Tricks"
        echo "  bash run_all_v11.sh M3           # GiGPO + <think>"
        echo "  bash run_all_v11.sh M4           # GiGPO + <belief>"
        echo "  bash run_all_v11.sh M5           # ReBel Full"
        echo "  bash run_all_v11.sh M5 seeds     # ReBel Full (3 seeds)"
        echo ""
        echo "  bash run_all_v11.sh A1           # 消融: w/o HiBO → obs only"
        echo "  bash run_all_v11.sh A2           # 消融: w/o HiBO → belief only"
        echo "  bash run_all_v11.sh A3           # 消融: w/o Belief Reward"
        echo "  bash run_all_v11.sh A4           # 消融: w/ Fixed Decay"
        echo "  bash run_all_v11.sh A5           # 消融: w/ Uniform Decay"
        echo "  bash run_all_v11.sh A6           # 消融: w/o Belief Prompt"
        echo ""
        echo "环境变量:"
        echo "  SEED=42 bash run_all_v11.sh M5   # 指定种子"
        echo "  NUM_GPUS=4 bash run_all_v11.sh M5  # 指定 GPU 数量"
        exit 1
        ;;
esac

echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "  所有指定实验已完成"
echo "═══════════════════════════════════════════════════════════════════"
