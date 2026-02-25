#!/bin/bash
# =============================================================================
# M2: GRPO + 训练稳定化 (GRPO with Training Stabilization)
# =============================================================================
# 目的: 量化训练稳定化技术的贡献 (非对称裁剪、Clip-Cov 熵保护等)
# 配置: <think> 提示格式 + GRPO advantage + 训练稳定化
# 对比: M2 vs M1 → 训练稳定化贡献
#        M3 vs M2 → Step-level advantage 独立贡献
#        M6 vs M2 → BeST 方法贡献 (方法 vs 工程优化)
#
# 支持多种子: SEED=42 bash M2_grpo_tricks.sh
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

EXP_ID=M2 \
EXP_NAME=grpo_tricks \
ADV_ESTIMATOR=grpo \
USE_REBEL_PROMPT=false \
USE_TRAINING_TRICKS=true \
USE_ADV_TRICKS=false \
USE_BELIEF_REWARD=false \
USE_RESULT_REWARD=true \
USE_BELIEF_DECAY=false \
bash "${SCRIPT_DIR}/run_v11_base.sh"
