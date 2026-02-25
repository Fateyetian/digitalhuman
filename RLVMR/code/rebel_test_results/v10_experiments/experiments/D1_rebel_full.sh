#!/bin/bash
# =============================================================================
# D1: 完整 ReBel (Full ReBel with Intrinsic Rewards)
# =============================================================================
# 目的: 测试完整 ReBel 配置 (信念分组 + 全部内在奖励)
# 配置: <belief> 提示 + ReBel advantage + 全部内在奖励 + 全部 tricks
# 对比: D1 vs C2 → 密集内在奖励的贡献 (验证 V8 发现: 可能为负)
#        E1 vs D1 → 课程衰减是否能缓解内在奖励的负面影响
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

EXP_ID=D1 \
EXP_NAME=rebel_full \
ADV_ESTIMATOR=rebel \
USE_REBEL_PROMPT=true \
USE_TRAINING_TRICKS=true \
USE_ADV_TRICKS=true \
USE_BELIEF_REWARD=true \
USE_RESULT_REWARD=true \
USE_BELIEF_DECAY=false \
bash "${SCRIPT_DIR}/run_v10_base.sh"
