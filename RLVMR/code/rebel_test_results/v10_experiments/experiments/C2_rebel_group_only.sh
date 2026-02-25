#!/bin/bash
# =============================================================================
# C2: ReBel 信念分组 (Belief Grouping Only, No Intrinsic Reward) [关键实验]
# =============================================================================
# 目的: 隔离信念分组对 advantage 估计的贡献
# 配置: <belief> 提示格式 + ReBel advantage (信念 hash 分组)
#        + 全部 tricks + 无内在奖励
# 对比: C2 vs C1 → 信念分组 vs 观测分组 (核心假设验证)
#        D1 vs C2 → 内在奖励的贡献 (预期可能为负)
#
# 注: 此配置等价于 V8 No Belief 消融，但在统一框架下重新运行
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

EXP_ID=C2 \
EXP_NAME=rebel_group_only \
ADV_ESTIMATOR=rebel \
USE_REBEL_PROMPT=true \
USE_TRAINING_TRICKS=true \
USE_ADV_TRICKS=true \
USE_BELIEF_REWARD=false \
USE_RESULT_REWARD=true \
USE_BELIEF_DECAY=false \
bash "${SCRIPT_DIR}/run_v10_base.sh"
