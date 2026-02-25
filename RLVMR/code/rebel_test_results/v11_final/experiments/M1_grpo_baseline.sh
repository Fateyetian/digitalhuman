#!/bin/bash
# =============================================================================
# M1: GRPO 纯基线 (Pure GRPO Baseline)
# =============================================================================
# 目的: 建立 GRPO 基准线，无任何额外 tricks
# 配置: <think> 提示格式 + GRPO advantage + 对称裁剪
# 对比: M2 vs M1 → 量化训练稳定化技术的贡献
#        M6 vs M1 → BeST 完整方法 vs 基线
#
# 支持多种子: SEED=42 bash M1_grpo_baseline.sh
#             SEED=123 bash M1_grpo_baseline.sh
#             SEED=456 bash M1_grpo_baseline.sh
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

EXP_ID=M1 \
EXP_NAME=grpo_baseline \
ADV_ESTIMATOR=grpo \
USE_REBEL_PROMPT=false \
USE_TRAINING_TRICKS=false \
USE_ADV_TRICKS=false \
USE_BELIEF_REWARD=false \
USE_RESULT_REWARD=true \
USE_BELIEF_DECAY=false \
bash "${SCRIPT_DIR}/run_v11_base.sh"
