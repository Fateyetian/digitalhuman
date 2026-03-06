#!/bin/bash
# =============================================================================
# M5: WebShop ReBel Full (HiBO + Adaptive Curriculum + Belief Prompting)
# =============================================================================
# Full ReBel method applied to WebShop e-commerce environment.
# Mirrors: experiments/M5_rebel_full.sh (ALFWorld version)
#
# Config:
#   - <belief> prompt format (Structured Belief Prompting)
#   - rebel_hibo advantage (HiBO hierarchical grouping)
#   - Adaptive differential belief curriculum reward
#   - Training stabilization
#
# Usage: SEED=42 bash M5_webshop_rebel_full.sh
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

EXP_ID=M5 \
EXP_NAME=webshop_rebel_full \
ADV_ESTIMATOR=rebel_hibo \
USE_REBEL_PROMPT=true \
USE_TRAINING_TRICKS=true \
USE_ADV_TRICKS=false \
USE_BELIEF_REWARD=true \
USE_RESULT_REWARD=true \
USE_BELIEF_DECAY=true \
DECAY_METHOD=cosine \
USE_ADAPTIVE_DECAY=true \
USE_DIFFERENTIAL_DECAY=true \
bash "${SCRIPT_DIR}/run_v11_webshop_base.sh"
