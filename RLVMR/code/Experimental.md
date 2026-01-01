# ReBel方法实验报告

**项目**: ReBel (Reward-guided Belief State Learning)
**环境**: ALFWorld Embodied Environment
**实验周期**: 2025-12-24 开始
**实验人员**: [Your Name]

---

## 目录

1. [实验概述](#1-实验概述)
2. [实验设置](#2-实验设置)
3. [主要实验](#3-主要实验)
4. [消融实验](#4-消融实验)
5. [分析与讨论](#5-分析与讨论)
6. [附录](#6-附录)

---

## 1. 实验概述

### 1.1 研究目标

验证ReBel方法在Embodied AI任务中的有效性，主要研究问题：

**RQ1**: ReBel的结构化belief state能否提升RL训练效果？
**RQ2**: Cold-start SFT策略（无Available Actions）是否能增强推理能力？
**RQ3**: Hindsight标注方法的数据质量如何？
**RQ4**: ReBel与现有方法（RLVMR、BDRS）相比性能如何？

### 1.2 核心创新点

1. **结构化Belief State**: 三组件设计（World Model, Task Progress, Exploration Map）
2. **Hindsight Annotation**: 从专家轨迹反向推理belief state
3. **渐进式课程学习**: Cold-start SFT → RL with Available Actions
4. **Belief-guided Reward**: 基于belief质量的密集奖励信号

---

## 2. 实验设置

### 2.1 数据集

#### 2.1.1 专家轨迹数据
```
来源: ALFWorld官方数据集
训练集: alfworld_expert_traj/ (3553个任务)
测试集: 134个任务 (6种任务类型)
```

#### 2.1.2 ReBel标注数据
```
标注方法: Hindsight Annotation
Teacher LLM: Claude 3.5 Sonnet
原始标注: 426个轨迹
过滤后(成功率≥1.0): 390个轨迹 (4640步)
数据位置: data/alfworld_rebel_merged_final/rebel_coldstart_clean.json
```

**数据质量验证**:
- ✅ Belief State格式正确率: 100%
- ✅ Available Actions删除率: 100%
- ✅ 平均轨迹长度: 11.9步/任务

### 2.2 评估指标

#### 2.2.1 主要指标
| 指标 | 定义 | 计算方式 |
|------|------|----------|
| **Success Rate (SR)** | 任务成功完成率 | SR = 成功任务数 / 总任务数 |
| **Average Steps (AS)** | 平均完成步数 | AS = Σ(成功任务步数) / 成功任务数 |
| **Efficiency** | 任务效率 | Efficiency = SR / AS |

#### 2.2.2 Belief State质量指标
| 指标 | 定义 | 计算方式 |
|------|------|----------|
| **Belief Consistency (BC)** | Belief与实际状态的一致性 | BC = 正确belief / 总belief数 |
| **Belief Completeness** | Belief包含关键信息的完整度 | 必需字段完整率 |
| **Belief Progression** | Belief随任务进展的合理性 | Subgoal进度合理性 |

#### 2.2.3 Reward信号指标
| 指标 | 定义 |
|------|------|
| **Task Reward** | ALFWorld环境奖励 (+1成功, 0失败) |
| **Belief Reward** | ReBel密集奖励 (α·一致性 + β·质量 + γ·完整性 + δ·进度) |
| **Total Reward** | Task Reward + Belief Reward |

### 2.3 基线方法

| 方法 | 描述 | Prompt类型 | Available Actions | 数据来源 |
|------|------|-----------|------------------|----------|
| **RLVMR** | Meta-cognitive prompting | 4种标签 (plan/explore/reflect/monitor) | ✅ 有 | Cold-start SFT |
| **BDRS** | BDRS模式分类 | 4种模式 (PLAN/EXECUTE/EXPLORE/VERIFY) | ✅ 有 | Cold-start SFT |
| **PPO-Baseline** | 标准PPO | 简单think标签 | ✅ 有 | Expert轨迹 |
| **ReBel (Ours)** | Belief-guided RL | 结构化Belief State | Cold-start: ❌ / RL: ✅ | Hindsight标注 |

### 2.4 实现细节

#### 2.4.1 模型配置
```yaml
Base Model: Qwen/Qwen2.5-1.5B-Instruct
Tokenizer: Qwen2.5 tokenizer
Max Sequence Length: 4096
Context Window: 8192
```

#### 2.4.2 训练超参数

**SFT阶段 (Cold-start)**:
```yaml
Optimizer: AdamW
Learning Rate: 1e-5
Batch Size: 16
Gradient Accumulation: 2
Epochs: 5
Warmup Ratio: 0.1
Weight Decay: 0.01
Training Data: 390 trajectories (4640 steps)
No Available Actions: True  # 关键设置
```

**RL阶段 (ReBel)**:
```yaml
Algorithm: PPO + ReBel
Learning Rate: 1e-6
Group Size: 64  # ReBel需要更大group size用于belief分组
Rollout Batch: 16
PPO Epochs: 4
Clip Range: 0.2
Value Loss Coef: 0.5
Entropy Coef: 0.01
Belief Reward Weights:
  alpha: 0.3  # Belief一致性
  beta: 0.5   # Belief质量
  gamma: 0.2  # Belief完整性
  delta: 0.1  # Belief进度
Available Actions: True  # RL阶段提供
```

#### 2.4.3 计算资源
```
GPU: 4 × NVIDIA A800 (80GB)
Training Time (估计):
  - SFT: 4-8小时
  - RL: 12-24小时
Total: ~32小时
```

### 2.5 实验执行和结果记录

#### 2.5.1 核心脚本

| 脚本名称 | 功能 | 输出位置 |
|---------|------|---------|
| `regenerate_coldstart_clean.py` | 生成符合规范的Cold-start数据 | `data/alfworld_rebel_merged_final/rebel_coldstart_clean.json` |
| `run_sft_coldstart_426.sh` | SFT训练 | `checkpoints/cold_start/alfworld/qwen1.5b_rebel_426/` |
| `run_rebel_rl_training.sh` | RL训练 | `checkpoints/rl_training/rebel/` |
| `run_rebel_evaluation.sh` | 评测脚本 | `rebel_rollout_results/run_*/` |
| `run_rebel_full_pipeline.sh` | 完整流程（数据生成→SFT→评测→RL→评测） | 各阶段输出目录 |

#### 2.5.2 执行命令

**生成Clean Cold-start数据**:
```bash
python3 regenerate_coldstart_clean.py
```

**SFT训练**:
```bash
# ⚠️ 执行前需要修复数据路径（见2.5.5）
bash run_sft_coldstart_426.sh
```

**SFT评测**:
```bash
# 先启动vLLM服务器（使用SFT checkpoint）
# 修改start_vllm_server.sh中的MODEL_PATH
bash start_vllm_server.sh

# 在另一个终端运行评测
bash run_rebel_evaluation.sh
```

**RL训练**:
```bash
# ⚠️ 执行前需要替换COLD_START_MODEL_PATH（见2.5.5）
bash run_rebel_rl_training.sh vllm
```

**RL评测**:
```bash
# 修改start_vllm_server.sh中的MODEL_PATH为RL checkpoint
bash start_vllm_server.sh
bash run_rebel_evaluation.sh
```

#### 2.5.3 指标记录系统

**训练阶段（SFT + RL）**:
```yaml
Logger: wandb + console
Wandb Projects:
  - SFT: "RLVMR_ReBel"
  - RL: "ReBel"
Experiment Names:
  - SFT: "qwen1.5b_rebel_cold-start_426"
  - RL: "rebel_qwen2.5-1.5b_L0"

Alternative Loggers (支持但未使用):
  - swanlab (已安装 v0.7.1)
  - mlflow
  - tensorboard
```

**评测阶段**:
```yaml
Logger: 本地JSON文件（不使用wandb）
Output Directory: rebel_rollout_results/run_<timestamp>/
Output Files:
  - results.json: 聚合指标
  - trajectories.jsonl: 完整轨迹数据
  - rollout.log: 执行日志
```

#### 2.5.4 记录的指标

**训练阶段（Wandb）**:
```yaml
SFT阶段:
  - train/loss
  - train/learning_rate
  - val/loss
  - train/grad_norm
  - train/perplexity

RL阶段:
  - rollout/episode_reward
  - rollout/episode_length
  - rollout/success_rate
  - train/policy_loss
  - train/value_loss
  - train/entropy
  - train/kl_divergence
  - train/clip_fraction
  - rebel/belief_consistency_reward
  - rebel/belief_quality_reward
  - rebel/belief_completeness_reward
  - rebel/belief_progression_reward
```

**评测阶段（results.json）**:
```json
{
  "basic_metrics": {
    "success_rate": 0.0,
    "avg_episode_length": 0.0,
    "avg_episode_reward": 0.0
  },
  "rebel_metrics": {
    "avg_r_consistency": 0.0,
    "avg_r_progress": 0.0,
    "avg_r_exploration": 0.0,
    "avg_r_format": 0.0,
    "avg_belief_parse_rate": 0.0
  }
}
```

#### 2.5.5 执行前必要修复

**问题1: SFT脚本数据路径错误**
```bash
# 当前（错误）:
# DATA_SOURCE=/root/testttt/RLVMR/code/data/alfworld_rebel_merged_final/rebel_coldstart.json

# 修复方法:
sed -i 's|rebel_coldstart.json|rebel_coldstart_clean.json|g' run_sft_coldstart_426.sh
```

**问题2: RL脚本模型路径占位符**
```bash
# 当前（错误）:
# actor_rollout_ref.model.path=COLD_START_MODEL_PATH

# 修复方法（手动编辑run_rebel_rl_training.sh，替换为实际SFT checkpoint路径）:
# actor_rollout_ref.model.path=./checkpoints/cold_start/alfworld/qwen1.5b_rebel_426/global_step_XXX
# 其中XXX需要根据实际训练结果确定（通常是最后一个checkpoint，如global_step_75）
```

#### 2.5.6 结果查看方法

**查看Wandb训练曲线**:
```bash
# 访问: https://wandb.ai/
# 查找项目: RLVMR_ReBel (SFT) 或 ReBel (RL)
# 或在终端查看输出的wandb链接
```

**查看本地评测结果**:
```bash
# 找到最新结果目录
LATEST=$(ls -td rebel_rollout_results/run_* | head -1)

# 查看汇总结果
cat $LATEST/results.json | python3 -m json.tool

# 查看详细轨迹
head -5 $LATEST/trajectories.jsonl

# 查看执行日志
cat $LATEST/rollout.log
```

**提取关键指标**:
```bash
# 成功率
jq '.basic_metrics.success_rate' $LATEST/results.json

# Belief解析率
jq '.rebel_metrics.avg_belief_parse_rate' $LATEST/results.json

# 所有ReBel指标
jq '.rebel_metrics' $LATEST/results.json
```

### 2.6 各阶段记录的完整指标清单

#### 2.6.1 SFT训练阶段记录的指标

**通过Wandb记录** (`trainer.logger=['console','wandb']`):

| 指标名称 | 说明 | 记录频率 |
|---------|------|---------|
| `train/loss` | 训练损失 | 每个step |
| `train/lr(1e-3)` | 学习率（×1e-3） | 每个step |
| `val/loss` | 验证损失 | 每个epoch |

**训练输出位置**:
- Wandb项目: `RLVMR_ReBel`
- Experiment名称: `qwen1.5b_rebel_cold-start_426`
- Checkpoint: `checkpoints/cold_start/alfworld/qwen1.5b_rebel_426/global_step_*`

**关键观察点**:
- `train/loss` 应该从初始值（通常>1.5）下降到 <0.5
- `val/loss` 与 `train/loss` 差距不应过大（避免过拟合）
- Training完成后最后一个checkpoint通常是global_step_75（5 epochs × 15 steps/epoch）

#### 2.6.2 RL训练阶段记录的指标

**通过Wandb记录** (`trainer.logger=['console','wandb']`):

**Rollout阶段指标**:
| 指标名称 | 说明 |
|---------|------|
| `rollout/episode_reward` | Episode累积奖励 |
| `rollout/episode_length` | Episode长度 |
| `rollout/success_rate` | 批次成功率 |
| `rollout/valid_action_ratio` | 有效动作比例 |

**ReBel密集奖励指标**:
| 指标名称 | 说明 |
|---------|------|
| `rebel/belief_consistency_reward` | Belief与环境状态一致性 |
| `rebel/belief_quality_reward` | Belief质量分数 |
| `rebel/belief_completeness_reward` | Belief完整性 |
| `rebel/belief_progression_reward` | Belief进度合理性 |

**训练阶段指标**:
| 指标名称 | 说明 |
|---------|------|
| `train/policy_loss` | 策略网络损失 |
| `train/value_loss` | 价值网络损失 |
| `train/entropy` | 策略熵（探索性） |
| `train/kl_divergence` | 与参考模型的KL散度 |
| `train/clip_fraction` | PPO clip比例 |
| `actor/reward_kl_penalty` | KL penalty奖励 |
| `actor/reward_kl_penalty_coeff` | KL penalty系数 |

**训练输出位置**:
- Wandb项目: `ReBel`
- Experiment名称: `rebel_qwen2.5-1.5b_L0`
- Checkpoint: RL脚本配置的输出目录

**关键观察点**:
- `rollout/success_rate` 应该逐步上升
- `rebel/*_reward` 应该为正值且稳定增长
- `train/kl_divergence` 不应过大（<0.1），避免偏离SFT模型过多

#### 2.6.3 评测阶段记录的指标

**本地JSON文件** (`rebel_rollout_results/run_*/results.json`):

**基础指标** (`basic_metrics`):
| 指标名称 | 说明 | 计算方式 |
|---------|------|---------|
| `success_rate` | 任务成功率 | 成功任务数 / 总任务数 |
| `avg_episode_length` | 平均步数 | Σ(steps) / 任务数 |
| `avg_episode_reward` | 平均奖励 | Σ(rewards) / 任务数 |

**ReBel专用指标** (`rebel_metrics`):
| 指标名称 | 说明 | 计算方式 |
|---------|------|---------|
| `avg_r_consistency` | 平均一致性奖励 | Σ(r_consistency) / 总步数 |
| `avg_r_progress` | 平均进度奖励 | Σ(r_progress) / 总步数 |
| `avg_r_exploration` | 平均探索奖励 | Σ(r_exploration) / 总步数 |
| `avg_r_format` | 平均格式奖励 | Σ(r_format) / 总步数 |
| `avg_belief_parse_rate` | Belief解析成功率 | 成功解析步数 / 总步数 |

**轨迹数据** (`trajectories.jsonl`，每行一个step):
| 字段名称 | 说明 |
|---------|------|
| `env_id` | 环境ID |
| `step` | 步数 |
| `prompt` | 发送给模型的prompt |
| `action` | 模型原始输出 |
| `action_exec` | 实际执行的动作 |
| `reward` | 环境奖励 |
| `r_consistency` | 一致性奖励 |
| `r_progress` | 进度奖励 |
| `r_exploration` | 探索奖励 |
| `r_intrinsic_total` | 内在奖励总和 |
| `belief_parsed` | Belief是否成功解析（bool） |
| `belief_state` | 解析后的Belief状态（JSON） |
| `done` | Episode是否结束 |
| `won` | 是否成功完成任务 |
| `is_action_valid` | 动作是否有效 |

**评测输出位置**:
- 结果目录: `rebel_rollout_results/run_<timestamp>/`
- 汇总结果: `results.json`
- 完整轨迹: `trajectories.jsonl`
- 执行日志: `rollout.log`

### 2.7 自动化实验报告生成

#### 2.7.1 快速生成报告

**一键生成**（自动查找最近的2次评测结果）:
```bash
python3 generate_experiment_report.py --auto_find
```

输出: `EXPERIMENT_REPORT.md`

**手动指定评测目录**:
```bash
# 指定SFT和RL的评测结果目录
python3 generate_experiment_report.py \
    --sft_eval_dir rebel_rollout_results/run_20251224_100000 \
    --rl_eval_dir rebel_rollout_results/run_20251224_150000 \
    --output MY_EXPERIMENT_REPORT.md
```

#### 2.7.2 报告内容

生成的报告包含以下部分：

**1. Cold-start SFT 评测结果**
- 基础指标表格（成功率、平均步数、平均奖励）
- ReBel密集奖励指标
- Belief State质量指标
- SFT阶段自动评估（✅/⚠️/❌）

**2. RL训练后评测结果**
- 基础指标表格
- ReBel密集奖励指标
- Belief State质量指标
- RL阶段自动评估

**3. SFT vs RL 对比分析**
- 关键指标对比表
- 提升百分比计算
- 自动化提升分析

**4. 数据来源**
- 记录结果文件路径
- Episode数量

**5. 结论与建议**
- 根据成功率和Belief解析率自动生成建议
- 提供下一步改进方向

#### 2.7.3 报告生成逻辑

脚本会自动：
1. 从 `results.json` 加载指标（如果存在）
2. 如果没有 `results.json`，从 `trajectories.jsonl` 实时计算所有指标
3. 对比SFT和RL阶段的性能提升
4. 根据预设阈值自动生成评估和建议：
   - 成功率 ≥ 30% → 🎉 实验成功
   - 20% ≤ 成功率 < 30% → ✅ 部分成功
   - 成功率 < 20% → ⚠️ 需要改进
   - Belief解析率 ≥ 80% → ✅ 格式学习充分
   - Belief解析率 < 50% → ❌ 未掌握格式

#### 2.7.4 完整实验流程示例

```bash
# ========== 步骤1: 修复数据路径 ==========
sed -i 's|rebel_coldstart.json|rebel_coldstart_clean.json|g' run_sft_coldstart_426.sh

# ========== 步骤2: SFT训练 ==========
bash run_sft_coldstart_426.sh
# 训练完成后，checkpoint保存在: checkpoints/cold_start/alfworld/qwen1.5b_rebel_426/

# ========== 步骤3: SFT评测 ==========
# 修改start_vllm_server.sh中的MODEL_PATH为SFT checkpoint
# 启动vLLM服务器
bash start_vllm_server.sh &
sleep 60  # 等待服务器启动

# 运行评测
bash run_rebel_evaluation.sh
# 结果保存在: rebel_rollout_results/run_<timestamp_1>/

# ========== 步骤4: RL训练 ==========
# 修复RL脚本的模型路径（用SFT checkpoint替换COLD_START_MODEL_PATH）
# 然后运行RL训练
bash run_rebel_rl_training.sh vllm
# 训练完成后，checkpoint保存在RL配置的目录

# ========== 步骤5: RL评测 ==========
# 修改start_vllm_server.sh中的MODEL_PATH为RL checkpoint
pkill -f vllm  # 停止之前的vLLM服务
bash start_vllm_server.sh &
sleep 60

bash run_rebel_evaluation.sh
# 结果保存在: rebel_rollout_results/run_<timestamp_2>/

# ========== 步骤6: 生成实验报告 ==========
python3 generate_experiment_report.py --auto_find

# 查看报告
cat EXPERIMENT_REPORT.md
```

#### 2.7.5 实验结果检查清单

在生成报告后，检查以下关键点：

**SFT阶段** (必须达标才能进入RL):
- [ ] `train/loss` 下降到 <0.5
- [ ] `success_rate` ≥ 10%
- [ ] `avg_belief_parse_rate` ≥ 80%
- [ ] `avg_r_consistency` > 0

**RL阶段** (目标):
- [ ] `success_rate` ≥ 30% (目标)
- [ ] `success_rate` 相比SFT提升 > 20%
- [ ] `avg_belief_parse_rate` ≥ 90%
- [ ] `rollout/success_rate` 曲线呈上升趋势

**如果未达标**:
1. SFT success_rate < 10% → 检查数据质量，重新生成clean数据
2. Belief parse rate < 80% → 增加SFT训练轮数或检查prompt格式
3. RL无提升 → 检查reward权重配置，查看wandb曲线诊断问题

---

## 3. 主要实验

### 3.1 实验1: ReBel完整流程验证

#### 实验目标
验证ReBel方法从数据标注到RL训练的完整流程是否有效

#### 实验配置
```bash
# 数据: data/alfworld_rebel_merged_final/rebel_coldstart_clean.json
# 脚本: run_sft_coldstart_426.sh + run_rebel_rl_training.sh

# Step 1: SFT训练
bash run_sft_coldstart_426.sh

# Step 2: SFT评测
bash run_rebel_evaluation.sh

# Step 3: RL训练
bash run_rebel_rl_training.sh vllm

# Step 4: RL评测
bash run_rebel_evaluation.sh
```

#### 预期结果
| 阶段 | Success Rate | Average Steps | Belief Consistency |
|------|--------------|---------------|-------------------|
| **Base Model** (零样本) | 5-10% | N/A | N/A |
| **After SFT** (Cold-start) | 40-50% | 15-20步 | 60-70% |
| **After RL** (ReBel) | **70-85%** | 10-15步 | **80-90%** |

#### 结果记录模板

**Date**: 2025-12-__
**Experiment ID**: EXP-001-ReBel-Full-Pipeline

**SFT Results**:
```
Success Rate: ___%
Average Steps: ___
Belief Consistency: ___%
Training Time: ___ hours
Checkpoint: checkpoints/cold_start/alfworld/qwen1.5b_rebel_426/global_step___
```

**RL Results**:
```
Success Rate: ___%
Average Steps: ___
Belief Consistency: ___%
Training Time: ___ hours
Final Checkpoint: checkpoints/rl_training/rebel/epoch___
```

**Performance Gain**:
```
SR Improvement: +___% (from SFT to RL)
Steps Reduction: -___ steps
BC Improvement: +___%
```

---

### 3.2 实验2: 与Baseline方法对比

#### 实验目标
对比ReBel与现有方法（RLVMR, BDRS, PPO-Baseline）的性能

#### 实验配置

**方法A: RLVMR**
```bash
# 使用RLVMR提示词和训练流程
bash examples/rlvmr_trainer/run_alfworld.sh
```

**方法B: BDRS**
```bash
# 使用BDRS提示词和训练流程
bash examples/bdrs_trainer/run_alfworld.sh
```

**方法C: PPO-Baseline**
```bash
# 使用标准PPO和简单提示词
bash examples/ppo_trainer/run_alfworld.sh
```

**方法D: ReBel (Ours)**
```bash
# 已完成（实验1）
```

#### 结果对比表

| 方法 | Success Rate | Avg Steps | Belief Quality | 数据需求 | 训练时间 |
|------|--------------|-----------|---------------|---------|----------|
| PPO-Baseline | ___% | ___ | N/A | Expert轨迹 | ___ hrs |
| RLVMR | ___% | ___ | N/A | Cold-start | ___ hrs |
| BDRS | ___% | ___ | Low | Cold-start | ___ hrs |
| **ReBel** | ___% | ___ | **High** | Hindsight | ___ hrs |

#### 任务类型细分结果

| 任务类型 | PPO | RLVMR | BDRS | ReBel | 任务数 |
|---------|-----|-------|------|-------|--------|
| pick_and_place | ___% | ___% | ___% | ___% | 22 |
| pick_clean_then_place | ___% | ___% | ___% | ___% | 22 |
| pick_heat_then_place | ___% | ___% | ___% | ___% | 22 |
| pick_cool_then_place | ___% | ___% | ___% | ___% | 22 |
| look_at_obj | ___% | ___% | ___% | ___% | 22 |
| pick_two_obj | ___% | ___% | ___% | ___% | 24 |
| **Overall** | ___% | ___% | ___% | ___% | 134 |

---

### 3.3 实验3: 数据规模影响分析

#### 实验目标
研究训练数据规模对ReBel性能的影响

#### 实验配置

从390个样本中采样不同数量进行训练：

| 数据规模 | 样本数 | 步数 | 采样方式 |
|---------|--------|------|----------|
| 10% | 39 | ~464 | 随机采样 |
| 25% | 98 | ~1160 | 随机采样 |
| 50% | 195 | ~2320 | 随机采样 |
| 75% | 293 | ~3480 | 随机采样 |
| 100% | 390 | 4640 | 全部数据 |

#### 结果记录

| 数据规模 | Success Rate | Avg Steps | Belief Consistency | 训练时间 |
|---------|--------------|-----------|-------------------|----------|
| 10% | ___% | ___ | ___% | ___ hrs |
| 25% | ___% | ___ | ___% | ___ hrs |
| 50% | ___% | ___ | ___% | ___ hrs |
| 75% | ___% | ___ | ___% | ___ hrs |
| 100% | ___% | ___ | ___% | ___ hrs |

**分析**：
- [ ] 绘制数据规模-性能曲线
- [ ] 确定最小有效数据量
- [ ] 分析边际收益递减点

---

## 4. 消融实验

### 4.1 实验4: Belief State组件消融

#### 实验目标
验证Belief State各组件的贡献

#### 实验配置

| 变体 | World Model | Task Progress | Exploration Map | 说明 |
|------|-------------|---------------|-----------------|------|
| **Full ReBel** | ✅ | ✅ | ✅ | 完整三组件 |
| **w/o World Model** | ❌ | ✅ | ✅ | 移除物体追踪 |
| **w/o Task Progress** | ✅ | ❌ | ✅ | 移除任务进度 |
| **w/o Exploration Map** | ✅ | ✅ | ❌ | 移除探索地图 |
| **Only Reasoning** | ❌ | ❌ | ❌ | 仅保留reasoning（类似RLVMR） |

#### 结果记录

| 变体 | Success Rate | Avg Steps | Belief Consistency | SR下降 |
|------|--------------|-----------|-------------------|--------|
| **Full ReBel** | ___% | ___ | ___% | - |
| w/o World Model | ___% | ___ | ___% | -___% |
| w/o Task Progress | ___% | ___ | ___% | -___% |
| w/o Exploration Map | ___% | ___ | ___% | -___% |
| Only Reasoning | ___% | ___ | N/A | -___% |

**关键发现**：
- [ ] 哪个组件最重要？
- [ ] 组件之间是否有协同效应？
- [ ] 最小可行配置是什么？

---

### 4.2 实验5: Belief Reward权重消融

#### 实验目标
研究Belief Reward各权重参数的最优配置

#### 实验配置

测试不同的α, β, γ, δ组合：

| 配置 | α (一致性) | β (质量) | γ (完整性) | δ (进度) | 说明 |
|------|-----------|---------|-----------|---------|------|
| **Default** | 0.3 | 0.5 | 0.2 | 0.1 | 默认配置 |
| Consistency-focused | 0.6 | 0.2 | 0.1 | 0.1 | 强调一致性 |
| Quality-focused | 0.2 | 0.6 | 0.1 | 0.1 | 强调质量 |
| Balanced | 0.25 | 0.25 | 0.25 | 0.25 | 均衡权重 |
| Task-only | 0 | 0 | 0 | 0 | 仅任务奖励 |

#### 结果记录

| 配置 | Success Rate | Belief Consistency | Total Reward | 备注 |
|------|--------------|-------------------|--------------|------|
| Default | ___% | ___% | ___ | |
| Consistency-focused | ___% | ___% | ___ | |
| Quality-focused | ___% | ___% | ___ | |
| Balanced | ___% | ___% | ___ | |
| Task-only | ___% | ___% | ___ | 对比基线 |

---

### 4.3 实验6: Cold-start策略消融

#### 实验目标
验证Cold-start SFT（无Available Actions）的必要性

#### 实验配置

| 变体 | SFT数据 | Available Actions (SFT) | Available Actions (RL) |
|------|---------|------------------------|------------------------|
| **ReBel (Cold-start)** | Hindsight | ❌ 删除 | ✅ 提供 |
| ReBel (No Cold-start) | Hindsight | ✅ 保留 | ✅ 提供 |
| RLVMR (Baseline) | Cold-start | ❌ 删除 | ✅ 提供 |

#### 结果记录

| 变体 | SFT SR | RL SR | SR提升 | 推理能力 |
|------|--------|-------|--------|---------|
| ReBel (Cold-start) | ___% | ___% | +___% | 强 |
| ReBel (No Cold-start) | ___% | ___% | +___% | 中 |
| RLVMR | ___% | ___% | +___% | 强 |

**假设验证**：
- [ ] Cold-start确实提升推理能力？
- [ ] 对RL阶段有显著帮助？
- [ ] 与RLVMR的差异主要来自Belief State？

---

## 5. 分析与讨论

### 5.1 定量分析

#### 5.1.1 学习曲线分析

**数据收集**：
- [ ] 每个epoch的success rate
- [ ] 每个epoch的average steps
- [ ] 每个epoch的belief consistency
- [ ] 每个epoch的total reward

**绘制图表**：
```
图1: Success Rate vs Training Epochs (SFT阶段)
图2: Success Rate vs Training Epochs (RL阶段)
图3: Belief Consistency vs Training Epochs
图4: Average Steps vs Training Epochs
图5: Total Reward vs Training Steps
```

#### 5.1.2 收敛性分析

| 阶段 | 收敛Epoch | 最终SR | 收敛稳定性 |
|------|----------|--------|-----------|
| SFT | ___ | ___% | ±___% |
| RL | ___ | ___% | ±___% |

#### 5.1.3 样本效率分析

| 方法 | 训练样本数 | 最终SR | 样本效率 (SR/样本数) |
|------|----------|--------|---------------------|
| PPO-Baseline | ___ | ___% | ___ |
| RLVMR | ___ | ___% | ___ |
| BDRS | ___ | ___% | ___ |
| **ReBel** | 390 (4640步) | ___% | ___ |

---

### 5.2 定性分析

#### 5.2.1 案例研究1: 成功案例

**任务**: pick_and_place_simple (cellphone → sidetable)

**轨迹对比**：

| 步骤 | ReBel Belief State | RLVMR Meta-cognitive | 动作 |
|------|-------------------|---------------------|------|
| 1 | World: {}, Subgoal: "找到cellphone" | \<planning>需要找cellphone\</planning> | go to bed 1 |
| 2 | World: {bed_1: [book, laptop]}, Subgoal: "搜索bed 2" | \<monitor>bed 1无cellphone\</monitor> | go to bed 2 |
| ... | ... | ... | ... |

**关键观察**：
- [ ] ReBel的belief state是否更结构化？
- [ ] 是否帮助模型做出更合理的决策？
- [ ] 与RLVMR相比有何优势？

#### 5.2.2 案例研究2: 失败案例

**任务**: pick_two_obj_and_place (cup + plate)

**失败原因分析**：
- [ ] Belief state错误？
- [ ] 探索策略不当？
- [ ] 模型能力不足？

**改进建议**：
- [ ] 增强multi-object tracking
- [ ] 改进exploration map
- [ ] 调整reward权重

#### 5.2.3 Belief State质量分析

**统计分析**：
- [ ] 平均belief state完整度: ___%
- [ ] World Model准确率: ___%
- [ ] Task Progress合理性: ___%
- [ ] Exploration Map有效性: ___%

**典型错误**：
1. 错误1: 提前预测inventory变化（违反时间一致性）
2. 错误2: 未更新cleared_receptacles
3. 错误3: Subgoal不合理

---

### 5.3 错误分析

#### 5.3.1 错误类型分类

| 错误类型 | 数量 | 占比 | 典型案例 |
|---------|------|------|----------|
| 探索失败 | ___ | ___% | 未找到目标物体 |
| 动作非法 | ___ | ___% | 动作不在admissible中 |
| Belief错误 | ___ | ___% | Belief与实际状态不符 |
| 规划失败 | ___ | ___% | Subgoal设置不当 |
| 其他 | ___ | ___% | - |

#### 5.3.2 各任务类型错误分布

| 任务类型 | 主要错误类型 | 失败率 | 改进方向 |
|---------|------------|--------|---------|
| pick_and_place | ___ | ___% | ___ |
| pick_clean_then_place | ___ | ___% | ___ |
| pick_heat_then_place | ___ | ___% | ___ |
| pick_cool_then_place | ___ | ___% | ___ |
| look_at_obj | ___ | ___% | ___ |
| pick_two_obj | ___ | ___% | ___ |

---

### 5.4 讨论

#### 5.4.1 ReBel的优势

**优势1: 结构化Belief State**
- 提供明确的world model, task progress, exploration map
- 便于分组和奖励计算
- 可解释性强

**优势2: Hindsight标注**
- 保证数据质量（belief → action逻辑链）
- 自动保持一致性
- 标注效率高

**优势3: 渐进式课程学习**
- Cold-start SFT强化推理能力
- RL阶段有Available Actions降低探索难度
- 性能提升明显

#### 5.4.2 ReBel的局限性

**局限1: 数据需求**
- 需要高质量的专家轨迹
- Hindsight标注成本（Teacher LLM调用）
- 数据规模影响最终性能

**局限2: Belief State复杂度**
- JSON格式解析可能失败
- 三组件维护增加模型负担
- 训练时间较长

**局限3: 泛化能力**
- 当前仅在ALFWorld验证
- 其他环境需要重新设计belief state
- Prompt工程依赖性强

#### 5.4.3 未来改进方向

1. **自动化Belief State设计**
   - 自动从环境中提取belief state结构
   - 减少人工prompt工程

2. **多环境泛化**
   - 在SciWorld, WebShop等环境验证
   - 设计通用belief state框架

3. **模型规模扩展**
   - 测试7B, 13B等更大模型
   - 分析scaling law

4. **在线学习**
   - 支持在线belief state更新
   - 动态调整reward权重

---

## 6. 附录

### 6.1 实验Checklist

#### 数据准备
- [x] Clean cold-start数据生成完成
- [x] 数据质量验证通过
- [ ] 数据备份完成

#### 实验执行
- [ ] 实验1: ReBel完整流程
- [ ] 实验2: 与Baseline对比
- [ ] 实验3: 数据规模影响
- [ ] 实验4: Belief组件消融
- [ ] 实验5: Reward权重消融
- [ ] 实验6: Cold-start策略消融

#### 结果分析
- [ ] 定量结果收集完成
- [ ] 定性分析完成
- [ ] 图表绘制完成
- [ ] 案例研究完成
- [ ] 错误分析完成

#### 论文撰写
- [ ] 实验设置章节
- [ ] 主要结果章节
- [ ] 消融实验章节
- [ ] 分析与讨论章节
- [ ] 结论章节

---

### 6.2 实验日志模板

```
============================================================
实验日志
============================================================

日期: 2025-12-__
实验ID: EXP-___
实验名称: ___

---------------- 实验目标 ----------------
[描述本次实验的具体目标]

---------------- 实验配置 ----------------
数据集: ___
模型: ___
脚本: ___
超参数:
  - Learning Rate: ___
  - Batch Size: ___
  - Epochs: ___
  - 其他: ___

---------------- 执行命令 ----------------
```bash
[粘贴实际执行的命令]
```

---------------- 执行时间 ----------------
开始时间: ___
结束时间: ___
总耗时: ___ hours

---------------- 实验结果 ----------------
Success Rate: ___%
Average Steps: ___
Belief Consistency: ___%
Total Reward: ___

---------------- 定量结果 ----------------
[粘贴详细数值结果表格]

---------------- 定性观察 ----------------
[记录观察到的现象、异常、有趣的case等]

---------------- 问题与改进 ----------------
遇到的问题:
1. ___
2. ___

改进措施:
1. ___
2. ___

---------------- 下一步计划 ----------------
[ ] ___
[ ] ___

============================================================
```

---

### 6.3 快速执行指南

#### 完整实验流程（推荐）

```bash
# 步骤1: 确认数据路径
echo "数据位置: data/alfworld_rebel_merged_final/rebel_coldstart_clean.json"
ls -lh data/alfworld_rebel_merged_final/rebel_coldstart_clean.json

# 步骤2: 修改SFT脚本数据路径
sed -i 's|rebel_coldstart.json|rebel_coldstart_clean.json|g' run_sft_coldstart_426.sh

# 步骤3: SFT训练
echo "=== 开始SFT训练 ==="
bash run_sft_coldstart_426.sh 2>&1 | tee logs/sft_training_$(date +%Y%m%d_%H%M%S).log

# 步骤4: SFT评测
echo "=== SFT评测 ==="
bash run_rebel_evaluation.sh 2>&1 | tee logs/sft_evaluation_$(date +%Y%m%d_%H%M%S).log

# 步骤5: RL训练
echo "=== 开始RL训练 ==="
bash run_rebel_rl_training.sh vllm 2>&1 | tee logs/rl_training_$(date +%Y%m%d_%H%M%S).log

# 步骤6: RL评测
echo "=== RL评测 ==="
bash run_rebel_evaluation.sh 2>&1 | tee logs/rl_evaluation_$(date +%Y%m%d_%H%M%S).log

echo "=== 实验完成！ ==="
```

---

### 6.4 结果可视化脚本（待开发）

```python
# scripts/visualize_results.py
# 用于绘制实验结果图表

import matplotlib.pyplot as plt
import json

# 图1: Success Rate对比
def plot_success_rate_comparison():
    methods = ['PPO', 'RLVMR', 'BDRS', 'ReBel']
    success_rates = [___, ___, ___, ___]  # 填入实际数据
    plt.bar(methods, success_rates)
    plt.ylabel('Success Rate (%)')
    plt.title('Success Rate Comparison')
    plt.savefig('figures/success_rate_comparison.png')

# 图2: 学习曲线
def plot_learning_curve():
    # 从日志文件读取training metrics
    # 绘制SR vs Epochs曲线
    pass

# 图3: 任务类型细分
def plot_task_type_breakdown():
    # 绘制各任务类型的成功率对比
    pass

if __name__ == '__main__':
    plot_success_rate_comparison()
    plot_learning_curve()
    plot_task_type_breakdown()
```

---

## 总结

本实验报告提供了完整的ReBel方法验证方案，包括：

✅ **6个主要实验**：完整流程、baseline对比、数据规模、3个消融实验
✅ **全面的评估指标**：成功率、步数、效率、belief质量
✅ **详细的分析框架**：定量分析、定性分析、错误分析
✅ **可执行的实验流程**：数据、脚本、命令全部就绪

**下一步**：开始执行实验，填写结果，完成论文！

---

**实验准备完成日期**: 2025-12-24
**准备人员**: Claude Code
**状态**: ✅ Ready to Execute
