#!/bin/bash
# =============================================================================
# A2: GRPO + 训练 Tricks (GRPO with Training Tricks)
# =============================================================================
# 目的: 量化训练 tricks (非对称裁剪、熵保护等) 的贡献
# 配置: <think> 提示格式 + GRPO advantage + 非对称裁剪 + 熵保护
# 对比: A2 vs A1 → tricks 贡献; B1 vs A2 → 信念提示格式贡献
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

EXP_ID=A2 \
EXP_NAME=grpo_tricks \
ADV_ESTIMATOR=grpo \
USE_REBEL_PROMPT=false \
USE_TRAINING_TRICKS=true \
USE_ADV_TRICKS=false \
USE_BELIEF_REWARD=false \
USE_RESULT_REWARD=true \
USE_BELIEF_DECAY=false \
bash "${SCRIPT_DIR}/run_v10_base.sh"
