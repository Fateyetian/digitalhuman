#!/bin/bash
# =============================================================================
# E1: ReBel + 课程衰减 (ReBel with Belief Reward Curriculum)
# =============================================================================
# 目的: 测试课程化过程奖励策略 (密集→稀疏)
# 配置: <belief> 提示 + ReBel advantage + 内在奖励 cosine 衰减 + 全部 tricks
#
# 衰减曲线:
#   权重
#   1.0 |  /--\
#       | /    \
#       |/      \
#   0.1 |        \__________________________
#       |_________________________________
#         0  3  5       30            100  epoch
#         warmup  decay     maintain
#
# 对比: E1 vs D1 → 课程衰减的贡献
#        E1 vs C2 → 课程化内在奖励是否优于完全不用内在奖励
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

EXP_ID=E1 \
EXP_NAME=rebel_curriculum \
ADV_ESTIMATOR=rebel \
USE_REBEL_PROMPT=true \
USE_TRAINING_TRICKS=true \
USE_ADV_TRICKS=true \
USE_BELIEF_REWARD=true \
USE_RESULT_REWARD=true \
USE_BELIEF_DECAY=true \
DECAY_METHOD=cosine \
bash "${SCRIPT_DIR}/run_v10_base.sh"
