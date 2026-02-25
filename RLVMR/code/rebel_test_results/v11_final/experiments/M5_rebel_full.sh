#!/bin/bash
# =============================================================================
# M5: ReBel Full — 核心方法 (HiBO + Adaptive Curriculum + Belief Prompting)
# =============================================================================
# 目的: ReBel 完整方法，包含所有三大创新
# 配置:
#   - <belief> 提示格式 (Innovation 3: Structured Belief Prompting)
#   - rebel_hibo advantage (Innovation 1: HiBO 层次化分组)
#   - 自适应差异化信念课程奖励 (Innovation 2: Adaptive Belief Curriculum)
#   - 训练稳定化
#
# 关键对比:
#   M5 vs M4 → HiBO + 课程奖励 的增量贡献 (核心 claim!)
#   M5 vs M3 → 信念全套贡献 (prompting + HiBO + curriculum)
#   M5 vs M1 → ReBel 完整方法 vs 基线
#
# 这是论文的核心实验！
#
# 支持多种子: SEED=42 bash M5_rebel_full.sh
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

EXP_ID=M5 \
EXP_NAME=rebel_full \
ADV_ESTIMATOR=rebel_hibo \
USE_REBEL_PROMPT=true \
USE_TRAINING_TRICKS=true \
USE_ADV_TRICKS=false \
USE_BELIEF_REWARD=true \
USE_RESULT_REWARD=true \
USE_BELIEF_DECAY=true \
DECAY_METHOD=cosine \
USE_ADAPTIVE_DECAY=true \
USE_DIFFERENTIAL_DECAY=true \
bash "${SCRIPT_DIR}/run_v11_base.sh"
