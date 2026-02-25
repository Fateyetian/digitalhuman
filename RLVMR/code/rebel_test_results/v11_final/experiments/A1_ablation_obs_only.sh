#!/bin/bash
# =============================================================================
# A1: 消融 — 移除 HiBO, 仅用 obs 分组 (w/o HiBO → obs only)
# =============================================================================
# 目的: 验证 HiBO 层次化分组的贡献
# 配置: <belief> 提示格式 + GiGPO advantage (纯 obs hash 分组)
#        + 自适应差异化信念课程奖励
#
# 与 M4 的区别: A1 有信念奖励, M4 没有
# 对比: M5 vs A1 → HiBO 层次化分组的贡献 (~+2%)
#
# 预期: A1 < M5, 差距 ~2% (HiBO 挽救 70% 单样本组)
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

EXP_ID=A1 \
EXP_NAME=ablation_obs_only \
ADV_ESTIMATOR=gigpo \
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
