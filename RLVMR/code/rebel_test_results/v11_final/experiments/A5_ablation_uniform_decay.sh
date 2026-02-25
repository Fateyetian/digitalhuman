#!/bin/bash
# =============================================================================
# A5: 消融 — 统一衰减替代差异化衰减 (w/ Uniform Decay)
# =============================================================================
# 目的: 验证差异化组件衰减的贡献
# 配置: <belief> 提示格式 + HiBO advantage + 信念奖励
#        + 自适应衰减 + 统一组件衰减 (所有组件以相同速率衰减)
#
# 与 M5 的区别: M5 用差异化衰减 (progress=0.7, consistency=1.0, exploration=2.0)
#              A5 用统一衰减 (所有组件 rate=1.0)
#
# 对比: M5 vs A5 → 差异化衰减 vs 统一衰减 (~+1%)
#
# 预期: A5 < M5, 差距 ~1%
#        统一衰减让有害的 exploration 奖励衰减太慢
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

EXP_ID=A5 \
EXP_NAME=ablation_uniform_decay \
ADV_ESTIMATOR=rebel_hibo \
USE_REBEL_PROMPT=true \
USE_TRAINING_TRICKS=true \
USE_ADV_TRICKS=false \
USE_BELIEF_REWARD=true \
USE_RESULT_REWARD=true \
USE_BELIEF_DECAY=true \
DECAY_METHOD=cosine \
USE_ADAPTIVE_DECAY=true \
USE_DIFFERENTIAL_DECAY=false \
bash "${SCRIPT_DIR}/run_v11_base.sh"
