#!/bin/bash
# =============================================================================
# M4: GiGPO + <belief> 格式 (GiGPO with Belief Prompting)
# =============================================================================
# 目的: 隔离 Belief Prompting 的独立贡献 (认知脚手架作用)
# 配置: <belief> 提示格式 + GiGPO advantage (obs hash 分组) + 训练稳定化
#        无 HiBO, 无信念奖励
#
# 关键对比:
#   M4 vs M3 → Belief Prompting 独立效果 (认知脚手架)
#   M5 vs M4 → HiBO + 课程奖励 的增量贡献 (核心!)
#
# 等同 V10-C1 配置，需 3 seeds 验证。
#
# 支持多种子: SEED=42 bash M4_gigpo_belief.sh
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

EXP_ID=M4 \
EXP_NAME=gigpo_belief \
ADV_ESTIMATOR=gigpo \
USE_REBEL_PROMPT=true \
USE_TRAINING_TRICKS=true \
USE_ADV_TRICKS=false \
USE_BELIEF_REWARD=false \
USE_RESULT_REWARD=true \
USE_BELIEF_DECAY=false \
bash "${SCRIPT_DIR}/run_v11_base.sh"
