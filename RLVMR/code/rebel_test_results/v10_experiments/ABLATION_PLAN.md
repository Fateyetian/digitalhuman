# ReBel V10: 系统消融研究方案

> 目标: 隔离 ReBel 相对于 GRPO/GiGPO 的核心贡献因素，排除工程trick干扰，为论文提供干净的因果证据。

---

## 一、问题陈述

### 1.1 核心矛盾

现有实验数据中存在一个对论文叙事极其不利的事实:

```
V8 No Belief (移除信念奖励) = 90.6%  >  V7 Full ReBel = 85.9%  >  V8 Full ReBel = 82~83.6%
```

移除了所谓的"核心贡献"(信念内在奖励)后，性能反而**提升了5个百分点**。如果论文声称信念奖励是核心贡献，审稿人会立刻质疑。

### 1.2 因素纠缠

ReBel 相对于 GRPO 存在 **5个正交维度** 的差异，但当前实验从未干净地隔离过它们:

| 维度 | 具体内容 | 是否已隔离 |
|------|---------|:----------:|
| **F1: 信念提示格式** | 模型输出结构化 `<belief>` JSON，而非简单 `<think>` | 否 |
| **F2: 信念分组** | 按 `(uid, belief_hash)` 分组计算 step advantage | 否 |
| **F3: 密集内在奖励** | 4组件内在奖励 (consistency+progress+exploration+format) | 部分 |
| **F4: 课程衰减** | 信念奖励 cosine/linear decay | 否 |
| **F5: 工程稳定化** | entropy protection, task weighting, min_samples_ratio 等 | 否 |

**关键问题**: V8 No Belief 消融 (90.6%) 只移除了 F3，但保留了 F1+F2+F5。我们无法判断 90.6% 中有多少归因于:
- 信念提示格式强迫模型结构化思考? (F1)
- 信念分组提供更好的 advantage 估计? (F2)
- 还是工程 tricks 本身就够了? (F5)

---

## 二、因素分解

### 2.1 训练级别因素 (适用于所有 advantage estimator)

| 因素 | 参数 | 说明 |
|------|------|------|
| 非对称裁剪 | `clip_ratio_high=0.28` | 允许更大的正向更新 |
| Clip-Cov 熵保护 | `entropy_protection.method=clip_cov` | 防止策略熵坍塌 |
| KL 散度正则 | `use_kl_loss=True, kl_loss_coef=0.01` | 防止策略偏离过大 |
| 无效动作惩罚 | `invalid_action_penalty_coef=0.1` | 惩罚格式错误的输出 |

### 2.2 Advantage 级别因素 (特定于 ReBel)

| 因素 | 参数 | 说明 |
|------|------|------|
| 最小样本比例 | `min_samples_ratio=0.15` | 过滤过小的归一化组 |
| 任务自适应权重 | `use_task_weighting=true` | 低成功率任务获得更高权重 |
| 条件归一化 | `conditional_norm=true` | 保护小样本任务 |
| 逐任务归一化 | `per_task_normalization=true` | 按任务类型分别归一化 |

### 2.3 算法级别因素 (核心区分)

| 因素 | GRPO | GiGPO | ReBel |
|------|------|-------|-------|
| 提示格式 | `<think>` | `<think>` | `<belief>` |
| Episode advantage | uid 分组 | uid 分组 | uid 分组 |
| Step advantage | 无 | 观测 hash 分组 | 信念 hash 分组 |
| 步骤奖励 | 无 | 折扣环境奖励 | 内在信念奖励 |

---

## 三、消融实验设计

### 3.1 实验矩阵 (7组实验)

```
实验链路:

A1 (GRPO 纯基线)
 |
 +-- A2 (+ 训练tricks)              --> 量化 tricks 的贡献
 |    |
 |    +-- B1 (+ 信念提示格式)        --> 量化 structured belief prompting 的贡献  [关键]
 |         |
 |         +-- C1 (+ 观测分组/GiGPO)  --> 量化 step grouping 的贡献
 |         |    |
 |         |    +-- C2 (改用信念分组)   --> 量化 belief grouping vs observation grouping  [关键]
 |         |         |
 |         |         +-- D1 (+ 内在奖励) --> 量化密集奖励的贡献
 |         |         |
 |         |         +-- E1 (+ 课程衰减) --> 量化课程学习的贡献
```

### 3.2 详细实验配置

| ID | 名称 | adv_estimator | 提示格式 | Step 分组 | 内在奖励 | 训练 Tricks | Adv Tricks |
|----|------|--------------|---------|----------|---------|-----------|-----------|
| **A1** | `grpo_baseline` | grpo | `<think>` | 无 | 无 | 无 | 无 |
| **A2** | `grpo_tricks` | grpo | `<think>` | 无 | 无 | 全部 | 无 |
| **B1** | `grpo_belief_prompt` | grpo | `<belief>` | 无 | 无 | 全部 | 无 |
| **C1** | `gigpo_belief_prompt` | gigpo | `<belief>` | 观测 hash | 无 | 全部 | 无 |
| **C2** | `rebel_group_only` | rebel | `<belief>` | 信念 hash | 无 | 全部 | 全部 |
| **D1** | `rebel_full` | rebel | `<belief>` | 信念 hash | 全部 | 全部 | 全部 |
| **E1** | `rebel_curriculum` | rebel | `<belief>` | 信念 hash | cosine 衰减 | 全部 | 全部 |

### 3.3 配置参数明细

#### A1: GRPO 纯基线
```yaml
algorithm.adv_estimator: grpo
algorithm.rebel.enable: false          # <think> 格式
actor_rollout_ref.actor.clip_ratio: 0.2
actor_rollout_ref.actor.clip_ratio_low: 0.2
actor_rollout_ref.actor.clip_ratio_high: 0.2   # 对称裁剪
actor_rollout_ref.actor.entropy_coeff: 0.001
actor_rollout_ref.actor.use_kl_loss: true       # KL 是标准配置，保留
actor_rollout_ref.actor.kl_loss_coef: 0.01
# 无 entropy protection, 无 task weighting
```

#### A2: GRPO + 训练 Tricks
```yaml
algorithm.adv_estimator: grpo
algorithm.rebel.enable: false          # <think> 格式
actor_rollout_ref.actor.clip_ratio_low: 0.2
actor_rollout_ref.actor.clip_ratio_high: 0.28   # 非对称裁剪
# + entropy protection (clip_cov)
# + invalid action penalty
```

#### B1: GRPO + 信念提示格式 (关键实验)
```yaml
algorithm.adv_estimator: grpo          # GRPO advantage
algorithm.rebel.enable: true           # <belief> 格式
+algorithm.rebel.use_belief_reward: false
+algorithm.rebel.use_result_reward: true
# + 训练 tricks (同 A2)
```

#### C1: GiGPO + 信念提示格式
```yaml
algorithm.adv_estimator: gigpo         # GiGPO advantage (观测 hash 分组)
algorithm.rebel.enable: true           # <belief> 格式
algorithm.gigpo.step_advantage_w: 0.5
algorithm.gigpo.mode: "mean_norm"
algorithm.gamma: 0.95
# + 训练 tricks
```

#### C2: ReBel 信念分组 (无内在奖励)
```yaml
algorithm.adv_estimator: rebel         # ReBel advantage (信念 hash 分组)
algorithm.rebel.enable: true
+algorithm.rebel.use_belief_reward: false   # 无内在奖励
+algorithm.rebel.use_result_reward: true
# + 训练 tricks + advantage tricks (task weighting 等)
```

#### D1: 完整 ReBel
```yaml
algorithm.adv_estimator: rebel
algorithm.rebel.enable: true
+algorithm.rebel.use_belief_reward: true    # 完整内在奖励
+algorithm.rebel.use_result_reward: true
# + 全部 tricks
```

#### E1: ReBel + 课程衰减
```yaml
algorithm.adv_estimator: rebel
algorithm.rebel.enable: true
+algorithm.rebel.use_belief_reward: true
+algorithm.rebel.use_result_reward: true
+algorithm.rebel.belief_reward_decay.enable: true
+algorithm.rebel.belief_reward_decay.method: cosine
+algorithm.rebel.belief_reward_decay.warmup_epochs: 3
+algorithm.rebel.belief_reward_decay.decay_start_epoch: 5
+algorithm.rebel.belief_reward_decay.decay_end_epoch: 30
+algorithm.rebel.belief_reward_decay.min_weight: 0.1
# + 全部 tricks
```

---

## 四、关键对比与预期结果

### 4.1 对比矩阵

| 对比 | 隔离因素 | 预期方向 | 如果显著 → 论文贡献 |
|------|---------|---------|-------------------|
| **A2 vs A1** | 训练 tricks | A2 > A1 (小幅) | 排除，tricks 是工程优化 |
| **B1 vs A2** | **信念提示格式** | **B1 >> A2** | **核心贡献 1: Structured Belief Prompting** |
| **C1 vs B1** | Step 分组 (观测) | C1 >= B1 | Step-level advantage 有价值 |
| **C2 vs C1** | **信念分组 vs 观测分组** | **C2 > C1** | **核心贡献 2: Belief-aware Grouping** |
| **D1 vs C2** | 密集内在奖励 | D1 <= C2 ? | 内在奖励后期有害 (验证 V8 发现) |
| **E1 vs D1** | 课程衰减 | E1 > D1 | 过程奖励需要衰减 |
| **E1 vs C2** | 课程化内在奖励 | E1 >= C2 | 课程化过程奖励有正面贡献 |

### 4.2 假设与论文叙事

**假设 1 (最可能成立)**: B1 >> A2

信念提示格式强迫模型在每一步输出结构化的环境认知 (物体位置、任务进度、探索状态)，本质上是一种 **structured chain-of-thought**。即使不使用这些信念状态计算奖励，这种格式本身就教会模型系统性地追踪环境。

**假设 2**: C2 > C1

按信念状态分组比按原始观测分组更合理，因为:
- 不同的观测可能对应相同的认知状态 (信念 hash 更粗粒度，更好聚合)
- 相同的观测在不同认知阶段有不同含义 (信念 hash 能区分)

**假设 3**: D1 <= C2

密集内在奖励在训练后期引入梯度噪声:
- 内在奖励的 4 个组件可能与最终任务目标不完全一致
- 训练后期模型已建立环境认知，内在奖励成为干扰信号
- 这解释了 V8 No Belief (90.6%) > Full ReBel (85.9%)

**假设 4**: E1 > D1，且 E1 >= C2

课程化过程奖励 (早期密集、后期衰减) 可以兼得:
- 早期: 密集奖励帮助快速建立环境认知
- 后期: 衰减避免梯度噪声，让结果奖励主导
- 如果 E1 > C2 成立，说明过程奖励在早期确实有引导价值

### 4.3 论文故事线 (依假设成立)

```
1. 问题: 长程交互决策任务中，LLM 需要可靠的环境认知能力

2. 发现: 标准 GRPO 缺乏结构化环境追踪
   → A1/A2 结果较差

3. 贡献 1 - Structured Belief Prompting:
   强迫模型在每步输出结构化信念状态，显著提升性能
   → B1 >> A2

4. 贡献 2 - Belief-aware Advantage Estimation:
   利用信念状态进行更合理的样本分组，优于观测分组
   → C2 > C1

5. 发现: 密集内在奖励是双刃剑 — 早期有益、后期有害
   → D1 <= C2，但 E1 >= C2

6. 贡献 3 (可选) - Process Reward Curriculum:
   课程化衰减过程奖励，兼得早期引导和后期优化
   → E1 > D1
```

---

## 五、实现注意事项

### 5.1 起始模型选择

**推荐方案**: 所有实验从 `Qwen2.5-1.5B-Instruct` 基座模型开始 (无 SFT)

原因:
- 现有 SFT 模型是用 `<belief>` 格式数据训练的
- 如果从 rebel SFT 开始，`<think>` 格式实验 (A1/A2) 会因格式不匹配而处于劣势
- 从基座模型开始，所有实验在同一起跑线

替代方案: 如果从基座模型训练不收敛，可以:
1. 训练一个通用 SFT (不含特定格式偏好)
2. 或从 rebel SFT 开始，但在论文中注明，并额外补充从基座模型的对比

### 5.2 需要验证的代码兼容性

| 实验 | 配置组合 | 潜在问题 | 验证方法 |
|------|---------|---------|---------|
| **B1** | `adv_estimator=grpo` + `rebel.enable=true` | GRPO 是否能正确忽略 rebel 数据字段 | 短跑 5 epochs 验证无报错 |
| **C1** | `adv_estimator=gigpo` + `rebel.enable=true` | GiGPO 需要 `anchor_obs`，rebel 模式下是否填充 | 检查 rollout_loop 数据流 |
| **E1** | belief_reward_decay | V9 decay 代码是否已合入当前分支 | 检查 core_rebel.py 是否有 decay 逻辑 |

### 5.3 统一实验参数

为保证公平对比，以下参数在所有实验中保持一致:

```yaml
seed: 42
total_epochs: 100
train_batch_size: 16
val_batch_size: 128
rollout.n: 16
max_steps: 30
lr: 1e-6
ppo_epochs: 1
test_freq: 5
save_freq: 50
val_before_train: true
env.use_teacher_planner: true
env.alfworld.meta_think: true
env.alfworld.generalization_level: 0
```

### 5.4 实验执行优先级

```
第一批 (优先, 可并行):
  A1 + A2 + B1  →  判断 tricks 和提示格式的各自贡献
  ↓
  如果 B1 >> A2:  论文方向确认，信念提示是核心
  如果 B1 ≈ A2:  需要重新审视方法，可能 tricks 才是关键

第二批 (依赖第一批):
  C1 + C2  →  判断分组方法的贡献
  ↓
  如果 C2 > C1:  信念分组优于观测分组，第二个贡献点成立
  如果 C2 ≈ C1:  分组方法差异不大，聚焦提示格式

第三批 (依赖第二批):
  D1 + E1  →  判断内在奖励和课程的贡献
  ↓
  如果 E1 > C2 > D1:  课程化过程奖励有正面贡献
  如果 E1 ≈ C2 > D1:  过程奖励无需使用，聚焦前两个贡献
```

---

## 六、预期论文表格

### Table 1: 主消融结果 (Main Ablation Results)

| Method | Prompt | Step Grouping | Intrinsic Reward | Overall SR | look_at | Avg Steps |
|--------|--------|--------------|-----------------|-----------|---------|-----------|
| GRPO | `<think>` | - | - | ? | ? | ? |
| GRPO + Tricks | `<think>` | - | - | ? | ? | ? |
| GRPO + Belief Prompt | `<belief>` | - | - | ? | ? | ? |
| GiGPO + Belief Prompt | `<belief>` | Observation | - | ? | ? | ? |
| ReBel (Group Only) | `<belief>` | Belief | - | ? | ? | ? |
| ReBel (Full) | `<belief>` | Belief | Full | ? | ? | ? |
| ReBel (Curriculum) | `<belief>` | Belief | Cosine Decay | ? | ? | ? |

### Table 2: 因素贡献分解 (Factor Contribution Breakdown)

| Factor | Delta SR | Comparison | Significance |
|--------|----------|------------|-------------|
| Training Tricks | +?% | A2 - A1 | 工程优化 |
| **Belief Prompting** | +?% | B1 - A2 | **核心贡献** |
| Step Grouping (Obs) | +?% | C1 - B1 | 辅助 |
| **Belief Grouping** | +?% | C2 - C1 | **核心贡献** |
| Dense Intrinsic Reward | +?% | D1 - C2 | 可能为负 |
| Reward Curriculum | +?% | E1 - D1 | 缓解密集奖励问题 |

---

## 七、后续工作: 密集过程奖励课程 (Paper 2 方向)

### 7.1 核心思路

V10 消融如果验证了"密集内在奖励后期有害"(D1 < C2) 和"课程衰减可以缓解"(E1 > D1)，这就为一篇独立的论文奠定了基础:

> **Process Reward Curriculum**: 密集过程奖励在 RL 训练早期提供引导信号帮助探索和认知建立，但在后期引入偏差和噪声。最优策略是先密后疏的课程化过程奖励。

### 7.2 论文定位

| 维度 | Paper 1 (ReBel) | Paper 2 (Process Reward Curriculum) |
|------|-----------------|-------------------------------------|
| 核心贡献 | 信念提示 + 信念分组 | 密集→稀疏奖励课程 |
| 方法论 | 架构设计 (怎么组织信息) | 训练策略 (怎么使用奖励) |
| 适用范围 | 交互式决策 (ALFWorld) | 通用 (数学推理、代码、对话) |
| 关键实验 | B1 vs A2, C2 vs C1 | E1 vs D1 vs C2 + 多域验证 |

### 7.3 Paper 2 实验设计 (初步)

**Phase 1: ALFWorld 上验证核心假设**
- 不同衰减函数 (cosine/linear/exponential)
- 不同切换时间点 (early/mid/late)
- 不同最小权重 (0.0/0.1/0.3)

**Phase 2: 推广到其他域**
- 数学推理: PRM (Process Reward Model) 作为过程奖励，ORM 作为结果奖励
- 代码生成: 编译通过/测试通过作为过程奖励，最终正确性作为结果奖励
- 多轮对话: 对话质量评分作为过程奖励，任务完成作为结果奖励

**Phase 3: 理论分析**
- 连接到 reward shaping theory (Ng et al., 1999)
- 分析过程奖励偏差 (potential-based vs non-potential-based)
- 什么样的过程奖励适合保持、什么样的需要衰减

### 7.4 可行性评估

**优势**:
- V10 消融数据直接支撑核心假设
- 问题通用性强，不限于 ALFWorld
- 与 Process Reward Model (PRM) 研究热点接轨

**风险**:
- 如果 E1 ≈ C2 (课程化无效)，则论文假设不成立
- 需要在多个域上验证，实验成本较高
- 理论分析可能不够深入

---

## 八、文件结构

```
v10_experiments/
├── ABLATION_PLAN.md                     # 本文档
├── run_v10_base.sh                      # 通用基础训练脚本
├── run_all_ablations.sh                 # 实验编排脚本
└── experiments/
    ├── A1_grpo_baseline.sh              # GRPO 纯基线
    ├── A2_grpo_tricks.sh                # GRPO + 训练 tricks
    ├── B1_grpo_belief_prompt.sh         # GRPO + 信念提示 [关键]
    ├── C1_gigpo_belief_prompt.sh        # GiGPO + 信念提示
    ├── C2_rebel_group_only.sh           # ReBel 信念分组 (无内在奖励) [关键]
    ├── D1_rebel_full.sh                 # 完整 ReBel
    └── E1_rebel_curriculum.sh           # ReBel + 课程衰减
```

---

*创建时间: 2026-02-11*
