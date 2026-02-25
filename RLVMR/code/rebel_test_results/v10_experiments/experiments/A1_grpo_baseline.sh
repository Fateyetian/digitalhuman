#!/bin/bash
# =============================================================================
# A1: GRPO 纯基线 (Pure GRPO Baseline)
# =============================================================================
# 目的: 建立 GRPO 基准线，无任何额外 tricks
# 配置: <think> 提示格式 + GRPO advantage + 对称裁剪
# 对比: A2 vs A1 → 量化训练 tricks 的贡献
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

EXP_ID=A1 \
EXP_NAME=grpo_baseline \
ADV_ESTIMATOR=grpo \
USE_REBEL_PROMPT=false \
USE_TRAINING_TRICKS=false \
USE_ADV_TRICKS=false \
USE_BELIEF_REWARD=false \
USE_RESULT_REWARD=true \
USE_BELIEF_DECAY=false \
bash "${SCRIPT_DIR}/run_v10_base.sh"
