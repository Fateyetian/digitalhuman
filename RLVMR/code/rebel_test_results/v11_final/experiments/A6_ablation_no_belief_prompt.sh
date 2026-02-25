#!/bin/bash
# =============================================================================
# A6: 消融 — 移除 Belief Prompt (w/o Belief Prompt → <think> format)
# =============================================================================
# 目的: 验证 Belief Prompting 的全局贡献
# 配置: <think> 提示格式 + GiGPO advantage (obs hash 分组) + 训练稳定化
#        无信念奖励 (因为 <think> 格式无法提供结构化信念)
#
# 注意: 使用 <think> 格式时，HiBO 的 belief fallback 无法工作 (无结构化信念)
#        因此退化为纯 GiGPO。信念奖励也无法计算。
#
# 对比: M5 vs A6 → 信念提示的全局贡献 (~-6%, 最大降幅)
#        A6 = M3 (GiGPO + <think>)，可以复用 M3 seed=42 结果
#
# 预期: A6 < M5, 差距 ~6%
#        这是最大的消融降幅，因为移除信念提示同时导致 HiBO 和课程奖励失效
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

EXP_ID=A6 \
EXP_NAME=ablation_no_belief_prompt \
ADV_ESTIMATOR=gigpo \
USE_REBEL_PROMPT=false \
USE_TRAINING_TRICKS=true \
USE_ADV_TRICKS=false \
USE_BELIEF_REWARD=false \
USE_RESULT_REWARD=true \
USE_BELIEF_DECAY=false \
bash "${SCRIPT_DIR}/run_v11_base.sh"
