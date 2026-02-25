#!/bin/bash
# =============================================================================
# A3: 消融 — 关闭信念奖励 (w/o Belief Reward)
# =============================================================================
# 目的: 验证课程奖励的贡献
# 配置: <belief> 提示格式 + HiBO advantage + 无信念奖励
#        (仅使用环境奖励进行训练)
#
# 对比: M5 vs A3 → 课程奖励的贡献 (~+2%)
#
# 预期: A3 < M5, 差距 ~2%
#        课程奖励在早期加速环境认知建立
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

EXP_ID=A3 \
EXP_NAME=ablation_no_reward \
ADV_ESTIMATOR=rebel_hibo \
USE_REBEL_PROMPT=true \
USE_TRAINING_TRICKS=true \
USE_ADV_TRICKS=false \
USE_BELIEF_REWARD=false \
USE_RESULT_REWARD=true \
USE_BELIEF_DECAY=false \
bash "${SCRIPT_DIR}/run_v11_base.sh"
