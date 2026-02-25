# ReBel V11 实验进度记录

**最后更新**: 2026-02-23
**项目目标**: 完成 ReBel 算法实验，撰写顶会论文

---

## 一、算法概述

**ReBel** (Reinforcement Learning with Belief-State Enhancement) 三大创新：
1. **HiBO** — 层次化信念-观测分组：obs hash 主层 + belief abstract 回退层，挽救 GiGPO 70-85% 的单样本组浪费
2. **能力自适应信念课程奖励** — SR-based 自适应衰减 + 差异化组件衰减（progress 慢衰, exploration 快衰）
3. **结构化信念提示** — `<belief>` JSON 同时服务分组信号、奖励计算、认知脚手架

---

## 二、实验方案

### 起始模型
- **SFT 模型**: `/root/testttt/RLVMR/code/checkpoints/cold_start/alfworld/rebel_full_20251225_024950/global_step_90`
- **Base 模型**: `/root/testttt/RLVMR/code/base_models/Qwen2.5-1.5B-Instruct`

### 主实验（所有方法不开 tricks，M1/M2 用 `<think>`）

| 新编号 | 方法 | 脚本 | Advantage | 提示 | HiBO | 信念奖励 |
|--------|------|------|-----------|------|------|---------|
| M1 | GRPO | `M1_grpo_baseline.sh` | grpo | `<think>` | ✗ | ✗ |
| M2 | GiGPO | `M3_gigpo_think.sh` | gigpo | `<think>` | ✗ | ✗ |
| M3 | GiGPO + belief | `M4_gigpo_belief.sh` | gigpo | `<belief>` | ✗ | ✗ |
| M4 | **ReBel Full** | `M5_rebel_full.sh` | rebel_hibo | `<belief>` | ✓ | ✓ |

### 消融实验（精简版，3 个，针对 M4 分解）

| 编号 | 消融 | 脚本 | 与 M4 的差异 | 验证目标 |
|------|------|------|-------------|---------|
| A1 | w/o HiBO (obs only) | `A1_ablation_obs_only.sh` | gigpo 分组替代 rebel_hibo | HiBO 贡献 |
| A2 | w/o Belief Reward | `A3_ablation_no_reward.sh` | 关闭信念奖励 | 课程奖励贡献 |
| A3 | w/o Adaptive Decay | `A4_ablation_fixed_decay.sh` | 固定 cosine 替代自适应 | 自适应衰减贡献 |

---

## 三、已完成实验

### 1. M5 (ReBel Full) on SFT 模型 ✅
- **目录**: `M5_rebel_full_seed42_20260220_161334`
- **起始模型**: SFT (global_step_90)
- **最终 val SR**: **0.891** (peak: 0.953 @ epoch 80)
- **各任务 SR**: pick_and_place=0.923, pick_heat=1.0, look_at_obj=0.875
- **状态**: 100 epochs 完成

### 2. M5 (ReBel Full) on Base 模型 ✅
- **目录**: `M5_rebel_full_seed42_20260221_084129`
- **起始模型**: Base (Qwen2.5-1.5B-Instruct)
- **最终 val SR**: **0.078** (基本没学会)
- **状态**: 100 epochs 完成，性能极低
- **结论**: Base 模型需要 SFT 预训练才能进行 RL，后续正式实验全部使用 SFT 模型

### 3. M1 (GRPO 基线) on SFT 模型 ✅
- **目录**: `M1_grpo_baseline_seed42_20260222_152316`
- **起始模型**: SFT (global_step_90)
- **最终 val SR**: **0.820**
- **各任务 SR**: pick_two_obj=0.938, pick_clean=0.931, pick_and_place=0.808, look_at_obj=0.875, pick_heat=0.786, pick_cool=0.630
- **状态**: 100 epochs 完成

### 4. M2 (GiGPO) on SFT 模型 ✅
- **目录**: `M3_gigpo_think_seed42_20260222_224131`
- **起始模型**: SFT (global_step_90)
- **最终 val SR**: **0.820** (peak: 0.914 @ epoch 85)
- **状态**: 100 epochs 完成
- **注意**: 脚本 `M3_gigpo_think.sh` 已改为 tricks=false

---

## 四、正在运行的实验

### A1 (w/o HiBO → obs only) 🔄
- **目录**: `A1_ablation_obs_only_seed42_20260223_093208`
- **起始模型**: SFT (global_step_90)
- **启动时间**: 2026-02-23 09:32
- **当前进度**: epoch 1/100
- **后续队列**: A3 (w/o Belief Reward) → A4 (w/o Adaptive Decay)

---

## 五、待运行实验

| 优先级 | 实验 | 脚本 | 命令 |
|--------|------|------|------|
| 1 | A2 (w/o Belief Reward) | `A3_ablation_no_reward.sh` | `EPOCHS=100 SEED=42 bash run_all_v11.sh A3` |
| 2 | A3 (w/o Adaptive Decay) | `A4_ablation_fixed_decay.sh` | `EPOCHS=100 SEED=42 bash run_all_v11.sh A4` |
| 3 | M3 (GiGPO + belief) | `M4_gigpo_belief.sh` | `EPOCHS=100 SEED=42 bash run_all_v11.sh M4` |
| 4 | 多 seed 复跑 | M1-M4 × seed 123,456 | 见下方命令 |

### 运行消融实验（串行，A1 完成后执行）
```bash
cd /root/testttt/RLVMR/code/rebel_test_results/v11_final && \
EPOCHS=100 SEED=42 bash run_all_v11.sh A3 && \
EPOCHS=100 SEED=42 bash run_all_v11.sh A4
```

### 运行 M3 补充实验
```bash
cd /root/testttt/RLVMR/code/rebel_test_results/v11_final && \
EPOCHS=100 SEED=42 bash run_all_v11.sh M4
```

### 多 seed 实验（正式实验后）
```bash
cd /root/testttt/RLVMR/code/rebel_test_results/v11_final && \
SEED=123 bash run_all_v11.sh M1 && SEED=456 bash run_all_v11.sh M1 && \
SEED=123 bash run_all_v11.sh M3 && SEED=456 bash run_all_v11.sh M3 && \
SEED=123 bash run_all_v11.sh M4 && SEED=456 bash run_all_v11.sh M4 && \
SEED=123 bash run_all_v11.sh M5 && SEED=456 bash run_all_v11.sh M5
```

---

## 六、当前结果汇总

| 方法 | val SR (seed=42) | 备注 |
|------|-----------------|------|
| M1 GRPO | 0.820 | 基线 |
| M2 GiGPO | 0.820 (peak 0.914) | step advantage 收敛值相近但 peak 更高 |
| M4 ReBel Full | **0.891** (peak **0.953**) | 🏆 最优 |
| M5 ReBel on Base | 0.078 | Base 模型无法直接 RL |

---

## 七、关键文件位置

| 文件 | 路径 |
|------|------|
| 算法设计文档 | `code/rebel_test_results/v11_final/V11_FINAL_ALGORITHM_AND_EXPERIMENT_PLAN.md` |
| HiBO 核心模块 | `code/rebel/hibo_grouping.py` |
| 信念奖励计算 | `code/agent_system/environments/env_package/alfworld/belief_tracker.py` |
| 训练主循环 | `code/verl/trainer/ppo/ray_trainer.py` |
| Rollout 循环 | `code/agent_system/multi_turn_rollout/rollout_loop.py` |
| 实验脚本目录 | `code/rebel_test_results/v11_final/experiments/` |
| 基础训练脚本 | `code/rebel_test_results/v11_final/run_v11_base.sh` |
| 编排脚本 | `code/rebel_test_results/v11_final/run_all_v11.sh` |
| 实验结果目录 | `/fs-computility-new/UPDZ03_chengjun/huangsijie.p/rebel_results/v11_final/` |
| SFT checkpoint | `code/checkpoints/cold_start/alfworld/rebel_full_20251225_024950/global_step_90` |
| Base 模型 | `code/base_models/Qwen2.5-1.5B-Instruct` |
| V10 日志 (已删权重) | `/fs-computility-new/UPDZ03_chengjun/huangsijie.p/rebel_results/v10_experiments/` |

---

## 八、脚本编号映射（新方案 → 现有脚本）

| 新方案编号 | 现有脚本名 | run_all 命令参数 |
|-----------|-----------|----------------|
| M1 GRPO | M1_grpo_baseline.sh | `M1` |
| M2 GiGPO | M3_gigpo_think.sh (tricks=false) | `M3` |
| M3 GiGPO+belief | M4_gigpo_belief.sh | `M4` |
| M4 ReBel Full | M5_rebel_full.sh | `M5` |
| A1 w/o HiBO | A1_ablation_obs_only.sh | `A1` |
| A2 w/o Reward | A3_ablation_no_reward.sh | `A3` |
| A3 w/o Adaptive | A4_ablation_fixed_decay.sh | `A4` |

---

## 九、注意事项

1. **M3 脚本已修改**: `M3_gigpo_think.sh` 的 `USE_TRAINING_TRICKS` 已从 true 改为 false
2. **存储已清理**: V10 实验 checkpoints 已全部删除（释放 ~629G），仅保留 training.log
3. **SFT checkpoint 可清理中间步**: `global_step_{18,36,54,72}` 和 `.tar.gz` 可删除释放 ~31G
4. **所有正式实验统一使用 SFT 模型**，Base 模型已验证不可行
