#!/bin/bash
# =============================================================================
# A4: 消融 — 固定 cosine 衰减替代自适应衰减 (w/ Fixed Decay)
# =============================================================================
# 目的: 验证自适应衰减 (基于 success rate) vs 固定 cosine 衰减的差异
# 配置: <belief> 提示格式 + HiBO advantage + 信念奖励 (固定 cosine 衰减)
#        + 差异化组件衰减
#
# 对比: M5 vs A4 → 自适应衰减 vs 固定衰减 (~+1%)
#
# 预期: A4 < M5, 差距 ~1%
#        固定衰减不能根据模型能力调整，可能衰减过早或过晚
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

EXP_ID=A4 \
EXP_NAME=ablation_fixed_decay \
ADV_ESTIMATOR=rebel_hibo \
USE_REBEL_PROMPT=true \
USE_TRAINING_TRICKS=true \
USE_ADV_TRICKS=false \
USE_BELIEF_REWARD=true \
USE_RESULT_REWARD=true \
USE_BELIEF_DECAY=true \
DECAY_METHOD=cosine \
USE_ADAPTIVE_DECAY=false \
USE_DIFFERENTIAL_DECAY=true \
bash "${SCRIPT_DIR}/run_v11_base.sh"
