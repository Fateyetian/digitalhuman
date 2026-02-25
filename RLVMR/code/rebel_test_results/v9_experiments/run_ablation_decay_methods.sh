#!/bin/bash
# =============================================================================
# ReBel V9 消融实验: 比较不同的信念奖励衰减方法
# =============================================================================
# 目的: 对比 cosine / linear / exponential 衰减策略
# 用于论文消融实验 Table
# =============================================================================

set -e

# 基本参数
NUM_GPUS=${NUM_GPUS:-4}
EPOCHS=${EPOCHS:-100}
BASE_SEED=${BASE_SEED:-42}

# 实验参数
WARMUP_EPOCHS=5
DECAY_START_EPOCH=10
DECAY_END_EPOCH=60
MIN_WEIGHT=0.1

RESULTS_BASE="/fs-computility-new/UPDZ03_chengjun/huangsijie.p/rebel_results/v9_experiments/ablation_decay_methods"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "═══════════════════════════════════════════════════════════════════"
echo "  ReBel V9 消融实验: 衰减方法对比"
echo "═══════════════════════════════════════════════════════════════════"
echo ""
echo "实验配置:"
echo "  - 对比方法: cosine, linear, exponential, none (baseline)"
echo "  - Warmup: ${WARMUP_EPOCHS} epochs"
echo "  - Decay: ${DECAY_START_EPOCH} -> ${DECAY_END_EPOCH} epochs"
echo "  - Min weight: ${MIN_WEIGHT}"
echo "  - Total epochs: ${EPOCHS}"
echo ""
echo "───────────────────────────────────────────────────────────────────"

# 定义要测试的衰减方法
DECAY_METHODS=("cosine" "linear" "exponential")

# 运行每种衰减方法的实验
for METHOD in "${DECAY_METHODS[@]}"; do
    echo ""
    echo "═══════════════════════════════════════════════════════════════════"
    echo "  Running: ${METHOD} decay"
    echo "═══════════════════════════════════════════════════════════════════"

    DECAY_ENABLE=true \
    DECAY_METHOD=${METHOD} \
    WARMUP_EPOCHS=${WARMUP_EPOCHS} \
    DECAY_START_EPOCH=${DECAY_START_EPOCH} \
    DECAY_END_EPOCH=${DECAY_END_EPOCH} \
    MIN_WEIGHT=${MIN_WEIGHT} \
    NUM_GPUS=${NUM_GPUS} \
    EPOCHS=${EPOCHS} \
    SEED=${BASE_SEED} \
    bash /root/testttt/RLVMR/code/rebel_test_results/v9_experiments/run_belief_reward_decay.sh

    echo "Completed: ${METHOD} decay"
done

# 运行 baseline (无衰减，始终使用信念奖励)
echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "  Running: baseline (no decay, always use belief reward)"
echo "═══════════════════════════════════════════════════════════════════"

DECAY_ENABLE=false \
NUM_GPUS=${NUM_GPUS} \
EPOCHS=${EPOCHS} \
SEED=${BASE_SEED} \
bash /root/testttt/RLVMR/code/rebel_test_results/v9_experiments/run_belief_reward_decay.sh

echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "  所有消融实验完成"
echo "═══════════════════════════════════════════════════════════════════"
echo ""
echo "结果保存在: ${RESULTS_BASE}"
echo ""
echo "实验组:"
echo "  1. cosine decay (推荐)"
echo "  2. linear decay"
echo "  3. exponential decay"
echo "  4. baseline (no decay)"
echo ""
