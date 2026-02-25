#!/bin/bash
# =============================================================================
# A2: 消融 — 移除 HiBO, 仅用 belief 分组 (w/o HiBO → belief only)
# =============================================================================
# 目的: 验证为什么需要层次化分组 (而非纯 belief 分组)
# 配置: <belief> 提示格式 + ReBel advantage (纯 belief hash 分组)
#        + 自适应差异化信念课程奖励
#
# 对比: M5 vs A2 → 层次化必要性 (~-4%)
#        A1 vs A2 → obs-only vs belief-only (验证 obs 质量 > belief 质量)
#
# 预期: A2 < A1 < M5 (纯 belief 分组因 bootstrap 不稳定性最差)
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

EXP_ID=A2 \
EXP_NAME=ablation_belief_only \
ADV_ESTIMATOR=rebel \
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
