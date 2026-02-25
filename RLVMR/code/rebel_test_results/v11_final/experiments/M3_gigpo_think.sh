#!/bin/bash
# =============================================================================
# M3: GiGPO + <think> 格式 (GiGPO Baseline) [关键新增实验]
# =============================================================================
# 目的: 建立 GiGPO + <think> 基线，隔离 Belief Prompting 在 GiGPO 框架下
#        的贡献。这是论文最需要的对比组！
#
# 配置: <think> 提示格式 + GiGPO advantage (观测 hash 分组) + 训练稳定化
#
# 关键对比:
#   M6 vs M3 → Belief Prompting 在 GiGPO 下的协同效应 (核心 Claim!)
#   M3 vs M2 → Step-level advantage 的独立贡献
#
# 预期:
#   如果 M6 >> M3: Belief Prompting 与 Step Advantage 有显著协同效应
#                  → 论文核心 claim 成立
#   如果 M6 ≈ M3: Belief Prompting 在 GiGPO 下无显著贡献
#                  → 需要调整论文叙事
#
# 注意:
#   - GiGPO 使用 anchor_obs (环境原始观测) 进行 step 分组
#   - 非 rebel 模式下，需要确认 anchor_obs 是否正确填充
#   - 训练稳定化启用 (与 M6 保持公平对比)
#
# 支持多种子: SEED=42 bash M3_gigpo_think.sh
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

EXP_ID=M3 \
EXP_NAME=gigpo_think \
ADV_ESTIMATOR=gigpo \
USE_REBEL_PROMPT=false \
USE_TRAINING_TRICKS=false \
USE_ADV_TRICKS=false \
USE_BELIEF_REWARD=false \
USE_RESULT_REWARD=true \
USE_BELIEF_DECAY=false \
bash "${SCRIPT_DIR}/run_v11_base.sh"
