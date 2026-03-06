#!/bin/bash
# =============================================================================
# M3: WebShop GiGPO + <think> Baseline
# =============================================================================
# GiGPO baseline with <think> prompting for WebShop environment.
# Mirrors: experiments/M3_gigpo_think.sh (ALFWorld version)
#
# Config: <think> prompt format + GiGPO advantage (obs hash grouping)
#
# Key comparison:
#   M5 vs M3 -> Full belief contribution (prompting + HiBO + curriculum)
#   M3 vs M1 -> Step-level advantage contribution
#
# Usage: SEED=42 bash M3_webshop_gigpo.sh
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

EXP_ID=M3 \
EXP_NAME=webshop_gigpo_think \
ADV_ESTIMATOR=gigpo \
USE_REBEL_PROMPT=false \
USE_TRAINING_TRICKS=false \
USE_ADV_TRICKS=false \
USE_BELIEF_REWARD=false \
USE_RESULT_REWARD=true \
USE_BELIEF_DECAY=false \
bash "${SCRIPT_DIR}/run_v11_webshop_base.sh"
