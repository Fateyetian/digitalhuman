#!/bin/bash
# =============================================================================
# B1: GRPO + 信念提示格式 (GRPO with Belief Prompt) [关键实验]
# =============================================================================
# 目的: 隔离信念提示格式 (<belief>) 的贡献
# 配置: <belief> 提示格式 + GRPO advantage + 训练 tricks
#        模型被要求输出结构化 belief JSON，但不用于奖励/分组
# 对比: B1 vs A2 → 信念提示格式的贡献 (核心假设验证)
#
# 技术说明:
#   - algorithm.adv_estimator=grpo → GRPO 只用 token_level_rewards
#   - algorithm.rebel.enable=true → 激活 <belief> 提示和解析
#   - use_belief_reward=false → 内在奖励不参与 advantage 计算
#   - GRPO 不读取 rebel_intrinsic_reward，因此不受内在奖励影响
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

EXP_ID=B1 \
EXP_NAME=grpo_belief_prompt \
ADV_ESTIMATOR=grpo \
USE_REBEL_PROMPT=true \
USE_TRAINING_TRICKS=true \
USE_ADV_TRICKS=false \
USE_BELIEF_REWARD=false \
USE_RESULT_REWARD=true \
USE_BELIEF_DECAY=false \
bash "${SCRIPT_DIR}/run_v10_base.sh"
