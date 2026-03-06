#!/bin/bash
# =============================================================================
# M1: WebShop GRPO Baseline
# =============================================================================
# Pure GRPO baseline for WebShop environment.
# Mirrors: experiments/M1_grpo_baseline.sh (ALFWorld version)
#
# Config: <think> prompt format + GRPO advantage + symmetric clipping
#
# Usage: SEED=42 bash M1_webshop_grpo.sh
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

EXP_ID=M1 \
EXP_NAME=webshop_grpo_baseline \
ADV_ESTIMATOR=grpo \
USE_REBEL_PROMPT=false \
USE_TRAINING_TRICKS=false \
USE_ADV_TRICKS=false \
USE_BELIEF_REWARD=false \
USE_RESULT_REWARD=true \
USE_BELIEF_DECAY=false \
bash "${SCRIPT_DIR}/run_v11_webshop_base.sh"
