#!/bin/bash
# =============================================================================
# C1: GiGPO + 信念提示格式 (GiGPO with Belief Prompt)
# =============================================================================
# 目的: 在使用 <belief> 提示格式下，测试观测 hash 分组 (GiGPO) 的效果
# 配置: <belief> 提示格式 + GiGPO advantage (观测 hash 分组) + 训练 tricks
# 对比: C1 vs B1 → step 分组 (观测级别) 的贡献
#        C2 vs C1 → 信念分组 vs 观测分组
#
# 技术说明:
#   - algorithm.adv_estimator=gigpo → 使用 anchor_obs 进行 step 分组
#   - algorithm.rebel.enable=true → 激活 <belief> 提示格式
#   - GiGPO 使用环境观测 hash 分组，不使用信念状态
#
# 注意: 需要验证 rebel 模式下 anchor_obs 是否正确填充
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

EXP_ID=C1 \
EXP_NAME=gigpo_belief_prompt \
ADV_ESTIMATOR=gigpo \
USE_REBEL_PROMPT=true \
USE_TRAINING_TRICKS=true \
USE_ADV_TRICKS=false \
USE_BELIEF_REWARD=false \
USE_RESULT_REWARD=true \
USE_BELIEF_DECAY=false \
bash "${SCRIPT_DIR}/run_v10_base.sh"
