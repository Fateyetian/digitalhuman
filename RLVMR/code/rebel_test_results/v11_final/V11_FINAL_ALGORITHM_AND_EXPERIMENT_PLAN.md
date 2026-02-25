# ReBel: Belief-Enhanced Policy Optimization for Interactive LLM Agents

## V11 最终算法设计与顶会实验方案 (Revised)

> **版本**: V11 Rev.2
> **日期**: 2026-02-20
> **目标**: 以信念为核心切入点，提出具有原创性的 SOTA 算法
> **目标会议**: NeurIPS 2026 / ICML 2026 / ICLR 2027

---

## 目录

- [第一部分: GiGPO 的本质缺陷分析](#第一部分-gigpo-的本质缺陷分析)
- [第二部分: ReBel 算法设计 — 三重信念增强](#第二部分-rebel-算法设计--三重信念增强)
- [第三部分: 理论分析](#第三部分-理论分析)
- [第四部分: 完整实验方案](#第四部分-完整实验方案)
- [第五部分: 论文结构规划](#第五部分-论文结构规划)
- [第六部分: 代码实现方案](#第六部分-代码实现方案)

---

# 第一部分: GiGPO 的本质缺陷分析

## 1.1 GiGPO 的核心机制与致命弱点

### GiGPO 的 Step-Level Advantage Estimation

GiGPO 通过两级优势估计进行信用分配:

```
A_total(i) = A_episode(i) + λ × A_step(i)
```

Step advantage 的计算依赖 **观测分组 (observation grouping)**:
1. 对同一 prompt (uid) 的多条轨迹，按 **精确观测文本 hash** 进行分组
2. 组内样本共享相同环境状态，通过比较不同动作的后续回报来估计优势
3. Step advantage = 当前样本回报 - 组内平均回报

**反事实语义**: "在完全相同的环境状态下，我的动作比平均水平好多少？"

### 致命弱点: 70-85% 的学习信号被浪费

**实测数据** (来自 V3-V6 实验日志):

| 指标 | 典型值 | 说明 |
|------|--------|------|
| 单样本组比例 | **63-85%** | 超过 2/3 的步骤组只有 1 个样本 |
| 中位数组大小 | **1.0** | 训练全程中位数始终为 1 |
| 平均组大小 | 1.7-3.5 | 被少数大组拉高 (max 可达 443) |

**根本原因**: ALFWorld 的观测文本高度多样。即使两条轨迹到达"相同的逻辑状态"，观测文本的细微差异 (物体枚举顺序、描述措辞) 导致 hash 不匹配。16 条轨迹 × 30 步 = 480 步中，大量步骤的观测是独一无二的。

### 单样本组的灾难性后果

**GiGPO 对单样本组的处理 (core_gigpo.py, line 266-268):**

```python
if len(id2score[idx]) == 1:
    id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))  # mean = 自身值
    id2std[idx] = torch.tensor(1.0)
```

归一化: `A_step = score - mean = score - score = 0`

**结论: 单样本组的 step advantage 恒等于 0。**

这意味着:
- **70-85% 的步骤完全没有 step-level 梯度信号**
- 仅 15-30% 的步骤贡献了 step-level 学习
- GiGPO 的 step advantage 机制在大部分情况下失效
- **然而，仅靠 15-30% 的有效信号，GiGPO 仍然比 GRPO 好 +6.2%**

> **关键洞察**: 如果能"挽救"这被浪费的 70-85% 信号，理论上可以获得远超当前水平的性能提升。

## 1.2 为什么之前的尝试失败了

### V10 消融数据回顾

| 实验 | 分组方式 | Peak SR | 说明 |
|------|---------|:-------:|------|
| C1 (GiGPO + belief prompt) | obs hash | **94.5%** | 最佳 |
| C2 (ReBel group only) | belief hash (adaptive) | 91.4% | 差 3.1% |
| D1 (ReBel full) | belief hash + 内在奖励 | 87.5% | 更差 |
| E1 (ReBel + curriculum) | belief hash + 衰减奖励 | 89.8% | 部分恢复 |

### 信念分组的三个失败原因

**原因 1: 粒度困境 — 没有甜点区间**

| 粒度 | 组数 | 平均大小 | 单样本率 | 问题 |
|------|:----:|:-------:|:-------:|------|
| `fine` (全 JSON hash) | ~1800 | 2.1 | **72.7%** | 与 obs hash 同样稀疏 |
| `subgoal` (子目标文本) | ~1600 | 2.5 | **70.5%** | 自由文本变异太大 |
| `adaptive` (4 特征) | ~130 | **28-32** | 中等 | **V10 C2 使用此级别** |
| `task_status` (3 布尔) | ~44 | 81.5 | **27.3%** | 混合不相关状态 |

`adaptive` 粒度的组大小 (~30) 实际是合理的，但问题在于:

**原因 2: 模型生成信念的 Bootstrap 不稳定性**

观测分组的 key (环境观测文本) 是 **外生的、确定性的** — 不随策略更新改变。
信念分组的 key (模型生成的信念) 是 **内生的、随机的** — 随着 θ 更新，信念内容变化 → 组边界移动 → 训练不稳定。

训练早期 (epoch 0-30)，模型信念极不可靠:
- `updated_subgoal` 是自由文本，措辞不一致
- `found_objects` 可能包含幻觉
- `stage_type` 提取依赖关键词匹配，对不规范输出失效

C1 (obs grouping) 在 epoch 30 达到 80% SR，C2 (belief grouping) 需要到 epoch 40 — 晚了 10 个 epoch。

**原因 3: V10 C2 的配置混淆**

C2 启用了 `USE_ADV_TRICKS=true` (包含 task_weighting, min_samples_ratio 等)，而 C1 没有。C2 用了更多工程手段反而更差，说明这些 tricks 可能引入了额外偏差。

### 密集内在奖励的失败原因

4 个奖励组件分析 (`belief_tracker.py`):

| 组件 | 权重 | 与任务成功的对齐度 | 潜在偏差 |
|------|:----:|:----------------:|---------|
| **Progress** | 0.5 | **高** — 直接检测子目标完成 | 轻微: 奖励冗长但无关的 evidence |
| **Consistency** | 0.3 | 中 — 准确的世界模型 ≠ 好的决策 | 中等: 早期宽松模式鼓励幻觉 |
| **Exploration** | 0.2 | **低** — 鼓励广度而非深度 | **严重: 与任务目标冲突** |
| **Format** | 0.1 | 结构必要性 | 无 |

**Exploration 奖励的严重偏差**: 奖励访问新位置 (70% 权重) + 维持长待访列表 (30%)。但 heat/cool/clean 任务需要 **深度交互** 而非广度探索。这个奖励积极推动模型偏离最优策略。

**V9 衰减的局限**: 固定 cosine 衰减 (epoch 5→30) 对所有组件统一衰减，无法区分有益 (progress) 和有害 (exploration) 的奖励组件。

## 1.3 核心洞察: 信念可以从三个层面解决 GiGPO 的缺陷

| GiGPO 的缺陷 | 信念增强方案 | 机制 |
|-------------|-----------|------|
| 70-85% 单样本组浪费 | **层次化分组 (HiBO)** | obs 分组为主 + 信念分组兜底 |
| 无密集训练信号 | **自适应信念课程奖励** | 早期密集引导，自适应衰减 |
| 无结构化推理 | **信念提示** | 认知脚手架 + HiBO 分组信号源 |

**三个创新形成闭环**:
- 信念提示产生结构化输出 → 为 HiBO 提供语义分组信号
- 信念课程奖励在早期引导信念质量提升 → 提高 HiBO 分组可靠性
- HiBO 挽救单样本组 → 放大 step advantage 的学习效率
- 信念质量提升 → 课程奖励自适应衰减 → 避免后期偏差

---

# 第二部分: ReBel 算法设计 — 三重信念增强

## 2.1 算法名称与核心思想

### ReBel: Reinforcement Learning with Belief-State Enhancement

**一句话描述**: ReBel 通过在 LLM 智能体的 RL 训练中系统性地利用结构化信念状态 — 作为分组依据挽救被浪费的学习信号、作为密集课程奖励加速早期学习、作为认知脚手架改善推理质量 — 在交互式决策任务中实现显著超越 GiGPO 的性能。

**核心创新 (Key Contributions)**:

1. **Contribution 1 — Hierarchical Belief-Observation Grouping (HiBO)**: 首次提出层次化分组策略，将观测分组 (高质量但稀疏) 与信念分组 (较低质量但密集) 结合。对 obs hash 匹配的步骤使用精确分组；对单样本组回退到粗粒度信念分组，将 step advantage 的有效覆盖率从 15-30% 提升至 70%+。

2. **Contribution 2 — Competence-Adaptive Belief Reward Curriculum**: 密集信念奖励加速早期环境认知建立，随智能体能力增长自适应衰减。关键改进: (1) 基于任务成功率的自适应衰减而非固定 cosine；(2) 差异化组件衰减 — 有害的 exploration 奖励快速衰减，有益的 progress 奖励慢速衰减。

3. **Contribution 3 — Structured Belief Prompting**: 强制模型在每步输出结构化信念状态 (世界模型、任务进度、探索地图)，同时服务于三个目的: 认知脚手架改善推理质量、为 HiBO 提供语义分组信号、为课程奖励提供评估基础。

## 2.2 算法整体架构

```
┌────────────────────────────────────────────────────────────────────────┐
│                       ReBel Training Framework                         │
│                                                                        │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  Innovation 1: Structured Belief Prompting                       │  │
│  │  输出格式: <belief> JSON </belief> <reasoning> </reasoning>       │  │
│  │            <action> action </action>                              │  │
│  │  三重作用: ① 认知脚手架  ② HiBO 分组信号  ③ 课程奖励基础         │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                    ↓                           ↓                       │
│  ┌────────────────────────────┐  ┌────────────────────────────────┐   │
│  │  Innovation 2:              │  │  Innovation 3:                 │   │
│  │  HiBO Step Advantage        │  │  Adaptive Belief Curriculum    │   │
│  │                             │  │                                │   │
│  │  A_total = A_ep + λ×A_step  │  │  R_step = R_env + w(t)×R_bel  │   │
│  │                             │  │                                │   │
│  │  Step Grouping:             │  │  w(t) = f(epoch, SR):          │   │
│  │  ┌─ obs hash 匹配? ─┐      │  │    早期: w ↑ (密集引导)        │   │
│  │  │Yes            │No │      │  │    后期: w ↓ (自适应衰减)      │   │
│  │  │精确 obs 组    │回退│      │  │                                │   │
│  │  │(高质量)      │信念组│     │  │  差异化衰减:                   │   │
│  │  │             │(密集)│      │  │    Progress: 慢衰减 (有益)     │   │
│  │  └───────────────┘    │      │  │    Exploration: 快衰减 (有害)  │   │
│  └────────────────────────────┘  └────────────────────────────────┘   │
│                    ↓                           ↓                       │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  Training Stabilization                                          │  │
│  │  非对称裁剪 + Clip-Cov 熵保护 + KL 正则化 + 无效动作惩罚         │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────────┘
```

## 2.3 Innovation 1: Hierarchical Belief-Observation Grouping (HiBO)

### 设计动机

GiGPO 的 obs hash 分组产生 70-85% 单样本组 → step advantage = 0。这些被浪费的样本蕴含有价值的学习信号 — 如果我们能找到一种合理的方式将它们分组。

信念状态提供了这个机会: 虽然精确信念 hash 同样稀疏 (70%+ 单样本)，但 **粗粒度语义信念抽象** 可以产生合理大小的组 (~30 样本/组)。

### 层次化分组策略

```
对每个 (uid, step) 组合:

第一层 — 精确观测分组 (GiGPO style):
  group_key = (uid, hash(anchor_obs))

  如果 group_size >= 2:
    → 使用精确观测组 (高质量: 相同环境状态 → 相同决策上下文)

  如果 group_size == 1 (单样本):
    → 进入第二层

第二层 — 语义信念分组 (Belief Fallback):
  belief_abstract = extract_semantic_features(belief_json)
  group_key = (uid, belief_abstract)

  → 使用信念组 (中等质量: 相似认知状态 → 相似决策上下文)
```

### 语义信念抽象 (Semantic Belief Abstraction)

从结构化信念 JSON 中提取 **离散语义特征**:

```python
def semantic_belief_abstract(belief_json):
    """
    从信念状态提取粗粒度语义特征。

    设计原则:
    1. 使用分类特征而非连续/文本特征 → 可靠 hash
    2. 捕捉决策相关的关键状态维度
    3. 粒度适中: ~50-100 种组合 → 组大小 ~5-15
    """
    task_progress = belief_json.get('task_progress_update', {})
    world_model = belief_json.get('world_model_update', {})
    exploration = belief_json.get('exploration_map_update', {})

    features = {
        # 维度 1: 当前阶段 (~10 类)
        'stage': classify_stage(task_progress.get('updated_subgoal', '')),
        # → find / navigate / pickup / place / heat / cool / clean / use / complete / other

        # 维度 2: 目标物体是否已找到 (2 类)
        'target_found': bool(world_model.get('found_objects', {})),

        # 维度 3: 手中是否持有物体 (2 类)
        'holding': bool(world_model.get('inventory', [])),

        # 维度 4: 探索进度 (3 类)
        'explore_level': bucket_exploration(
            len(exploration.get('cleared_receptacles', []))
        ),  # → 'low' (0-2) / 'mid' (3-5) / 'high' (6+)
    }

    return hash(tuple(sorted(features.items())))

# 理论组合数: 10 × 2 × 2 × 3 = 120
# 实际活跃组合: ~40-60 (并非所有组合在实践中出现)
# 每 uid ~320 步 / ~50 组 ≈ ~6 步/组 (合理范围)
```

### 组合优势估计

```python
def compute_hibo_step_advantage(batch):
    """
    HiBO: Hierarchical Belief-Observation Step Advantage
    """
    # 第一层: obs hash 分组
    obs_groups = build_obs_groups(batch.anchor_obs, batch.uid)

    # 第二层: 信念语义分组
    belief_groups = build_belief_groups(batch.belief_abstract, batch.uid)

    # 分配最终组
    final_groups = {}
    obs_group_used = 0
    belief_group_used = 0

    for step_idx in range(len(batch)):
        obs_group = obs_groups[step_idx]

        if len(obs_group.members) >= 2:
            # 多样本 obs 组 → 使用精确分组
            final_groups[step_idx] = obs_group
            obs_group_used += 1
        else:
            # 单样本 obs 组 → 回退到信念分组
            final_groups[step_idx] = belief_groups[step_idx]
            belief_group_used += 1

    # 日志记录 (用于论文分析)
    log({
        'hibo/obs_group_ratio': obs_group_used / len(batch),
        'hibo/belief_fallback_ratio': belief_group_used / len(batch),
    })

    # 计算组内归一化的 step advantage
    step_advantages = normalize_within_groups(
        batch.step_returns, final_groups, mode='mean_norm'
    )

    return step_advantages
```

### HiBO 相比 GiGPO 和纯信念分组的优势

| 特性 | GiGPO (obs only) | ReBel-old (belief only) | **ReBel-HiBO** |
|------|:----------------:|:----------------------:|:---------------:|
| 多样本组的质量 | **高** (精确匹配) | 中 (语义近似) | **高** (继承 obs) |
| 有效覆盖率 | 15-30% | 40-70% | **70-85%** |
| 单样本处理 | A_step = 0 | A_step = raw_reward | A_step = 信念组内归一化 |
| 训练稳定性 | **高** (外生分组) | 低 (内生分组) | **高** (主 obs, 辅 belief) |
| 需要信念 | 否 | 是 | 是 (仅对单样本) |

**关键优势**: HiBO 不是简单地选择 obs 或 belief，而是 **层次化利用**:
- 高质量信号来自 obs 分组 (保留 GiGPO 的优势)
- 额外信号来自信念分组 (挽救被浪费的 70-85%)

## 2.4 Innovation 2: Competence-Adaptive Belief Reward Curriculum

### 设计动机

V10 数据表明:
- 密集信念奖励在后期有害 (D1=87.5% < C2=91.4%)
- 固定 cosine 衰减部分缓解 (E1=89.8%)
- 问题: 衰减应该基于模型能力，而非固定时间表

### 自适应衰减机制

```python
def compute_belief_reward_weight(epoch, success_rate, config):
    """
    基于模型能力的自适应衰减。

    核心思想: 信念奖励的目的是帮助建立环境认知。
    当模型已经展现出足够的任务完成能力时，信念奖励应当减弱。

    参数:
        epoch: 当前训练轮次
        success_rate: 最近的验证集成功率 (0-1)
        config: 衰减配置
    """
    # 阶段 1: Warmup (epoch 0 → warmup_epochs)
    if epoch < config.warmup_epochs:
        base_weight = epoch / config.warmup_epochs
    else:
        base_weight = 1.0

    # 阶段 2: 自适应衰减 (基于成功率)
    if success_rate is not None and epoch >= config.warmup_epochs:
        # 成功率越高 → 权重越低
        # 当 SR 达到 target_sr 时, 权重降至 min_weight
        decay_factor = max(
            config.min_weight,
            1.0 - (success_rate / config.target_sr) ** config.alpha
        )
        adaptive_weight = base_weight * decay_factor
    else:
        adaptive_weight = base_weight

    # 阶段 3: 保底 cosine 衰减 (防止自适应失效)
    if epoch > config.decay_start_epoch:
        progress = (epoch - config.decay_start_epoch) / (config.decay_end_epoch - config.decay_start_epoch)
        progress = min(1.0, max(0.0, progress))
        cosine_floor = config.min_weight + (1 - config.min_weight) * 0.5 * (1 + math.cos(math.pi * progress))
        # 取两者中较小的 (更积极的衰减)
        adaptive_weight = min(adaptive_weight, cosine_floor)

    return max(config.min_weight, adaptive_weight)
```

### 差异化组件衰减 (Differential Component Decay)

**关键创新**: 不同奖励组件有不同的对齐度，应以不同速率衰减。

```python
def compute_component_weights(base_weight, config):
    """
    各组件以不同速率衰减。

    Progress (最对齐): 衰减最慢 → 保持最久
    Consistency (中等对齐): 正常衰减
    Exploration (最有害): 衰减最快 → 最早关闭
    Format: 不衰减 (结构必要性)
    """
    return {
        'progress':    base_weight ** config.progress_decay_rate,    # rate=0.7 → 慢衰减
        'consistency': base_weight ** config.consistency_decay_rate, # rate=1.0 → 正常
        'exploration': base_weight ** config.exploration_decay_rate, # rate=2.0 → 快衰减
        'format':      1.0,  # 不衰减
    }

# 示例: base_weight = 0.5 时
# progress:    0.5^0.7 = 0.62  (保留 62%)
# consistency: 0.5^1.0 = 0.50  (保留 50%)
# exploration: 0.5^2.0 = 0.25  (仅保留 25%)
```

### 总信念奖励计算

```python
R_belief = (
    α × R_progress    × w_progress +
    β × R_consistency  × w_consistency +
    γ × R_exploration  × w_exploration +
    δ × R_format       × 1.0
)

# 默认权重: α=0.5, β=0.3, γ=0.2, δ=0.1
```

### 自适应衰减的配置

```yaml
# Competence-Adaptive Belief Reward Curriculum
rebel.belief_reward_decay:
  enable: true
  method: adaptive              # adaptive / cosine / linear
  warmup_epochs: 3
  decay_start_epoch: 5          # 保底 cosine 开始
  decay_end_epoch: 40           # 保底 cosine 结束
  min_weight: 0.05              # 最终最小权重
  target_sr: 0.90               # 当 SR 达到此值时衰减至 min
  alpha: 2.0                    # 衰减曲线指数

  # 差异化衰减率
  progress_decay_rate: 0.7      # 慢衰减 (保留最久)
  consistency_decay_rate: 1.0   # 正常衰减
  exploration_decay_rate: 2.0   # 快衰减 (最早关闭)
```

### 与 V9 固定衰减的对比

```
权重
1.0 |  ╭──╮
    | ╱    ╲_____ V9: 固定 cosine (对所有组件统一)
    |╱
0.1 |________________
    0  3  5    30        100  epoch

vs ReBel 自适应:

权重
1.0 |  ╭──╮
    | ╱    ╲  ← progress (慢衰减, 随 SR 自适应)
0.5 |╱      ╲______
    |    ╲─── consistency (正常)
0.2 |      ╲___ exploration (快衰减)
0.05|________________
    0  3  5   20   40        100  epoch
         ↑ SR 达到 80% → 加速衰减
```

## 2.5 Innovation 3: Structured Belief Prompting (三重作用)

### 输出格式

```xml
<belief>
{
  "world_model_update": {
    "found_objects": {"object_name": "location"},
    "state_changes": {"object_name": "new_state"},
    "cleared_receptacles": ["receptacle1", "receptacle2"],
    "inventory": ["object_name"]
  },
  "task_progress_update": {
    "current_subgoal": "description",
    "subgoal_status": "in_progress/completed",
    "evidence": "observation-based reasoning"
  },
  "exploration_map_update": {
    "newly_visited": ["location1"],
    "next_priority": ["location2"]
  }
}
</belief>

<reasoning>
Based on my belief state, I should [strategy]...
</reasoning>

<action>
[action from admissible actions]
</action>
```

### 三重作用

| 作用 | 机制 | 对应创新 |
|------|------|---------|
| **认知脚手架** | 强制模型在每步系统性追踪环境状态 | 改善推理质量 |
| **HiBO 分组信号** | 结构化 JSON 提供语义特征用于分组 | Innovation 1 |
| **课程奖励基础** | 信念内容与 ground truth 比较计算奖励 | Innovation 2 |

**与 BeST 方案的区别**: BeST 将信念仅作为认知脚手架 (单一作用)，放弃了信念在分组和奖励中的价值。ReBel 让信念在三个层面都发挥作用，形成协同闭环。

## 2.6 完整算法伪代码

```python
Algorithm: ReBel (Reinforcement Learning with Belief-State Enhancement)
Input: LLM policy π_θ, environment E, prompts D
Output: Optimized policy π_θ*

1. Initialize π_θ from SFT checkpoint (with <belief> format)
2. Initialize running_success_rate = 0.0

3. For epoch = 1, ..., T:

   # === Phase 1: Compute Belief Reward Weight ===
   4. w_belief = compute_belief_reward_weight(
        epoch, running_success_rate, decay_config)
   5. component_weights = compute_component_weights(w_belief, decay_config)

   # === Phase 2: Rollout Collection ===
   6. For each prompt p_i in D:
      7. Reset environment: o_0 = E.reset()
      8. [Optional] Generate task plan via teacher model
      9. For t = 0, ..., max_steps:
         10. Build prompt with belief format:
             prompt_t = FORMAT_BELIEF(o_t, history, belief_cumulative)
         11. Generate response:
             y_t = π_θ(prompt_t)
         12. Parse: belief_t, reasoning_t, action_t = PARSE(y_t)
         13. Execute in environment:
             o_{t+1}, r_env_t, done_t = E.step(action_t)
         14. Store: anchor_obs_t = RAW_OBS(o_t)  # For HiBO obs grouping
         15. Extract: belief_abstract_t = SEMANTIC_ABSTRACT(belief_t)  # For HiBO fallback
         16. Compute intrinsic reward:
             r_belief_t = BELIEF_REWARD(belief_t, ground_truth_t, component_weights)
         17. Combined step reward:
             r_t = r_env_t + w_belief × r_belief_t

   # === Phase 3: HiBO Advantage Computation ===
   18. Compute episode rewards: R_ep = Σ_t r_env_t  # Episode level 只用环境奖励
   19. Compute discounted step returns: G_t = Σ_k γ^k r_t  # Step level 用混合奖励

   20. Episode Advantage (GRPO-style):
       For each uid group G_uid:
         A_episode(i) = R_ep(i) - mean(R_ep in G_uid)

   21. Step Advantage (HiBO):
       a) Build obs groups: G_obs by (uid, hash(anchor_obs))
       b) Build belief groups: G_bel by (uid, belief_abstract)
       c) For each step i:
            if |G_obs(i)| >= 2:
              A_step(i) = G_t(i) - mean(G_t in G_obs(i))  # 精确 obs 组
            else:
              A_step(i) = G_t(i) - mean(G_t in G_bel(i))  # 信念回退

   22. Combined Advantage:
       A_total(i) = A_episode(i) + λ × A_step(i)

   # === Phase 4: Policy Update ===
   23. PPO update with:
       - Asymmetric clipping: clip(ratio, 1-ε_low, 1+ε_high)
       - Clip-Cov entropy protection
       - KL regularization to reference policy
       - Invalid action penalty

   # === Phase 5: Update Running Stats ===
   24. If validation epoch:
       running_success_rate = EMA(running_success_rate, val_success_rate, β=0.3)

Return π_θ
```

## 2.7 Training Stabilization (继承并优化)

| 技术 | 参数 | 作用 |
|------|------|------|
| 非对称裁剪 | clip_low=0.2, clip_high=0.28 | 允许更大正向更新 |
| Clip-Cov 熵保护 | clip_cov_lb=0.0, clip_cov_ub=0.3 | 防止策略熵坍塌 |
| KL 正则化 | kl_loss_coef=0.01, type=low_var_kl | 防止过度偏离参考策略 |
| 无效动作惩罚 | penalty_coef=0.1 | 惩罚格式错误的输出 |

---

# 第三部分: 理论分析

## 3.1 HiBO 的信号恢复率分析

设总步数为 N，其中 obs hash 匹配率为 p (典型值 0.15-0.30)。

**GiGPO**: 有效学习样本 = p × N (70-85% 被浪费)

**ReBel-HiBO**:
- obs 匹配步骤: p × N (与 GiGPO 相同质量)
- 信念回退步骤: 设信念分组的非单样本率为 q (~0.70 for adaptive granularity)
- 有效学习样本 = p × N + (1-p) × q × N = [p + q - pq] × N

以 p=0.20, q=0.70 为例:
- GiGPO: 0.20N (20% 有效)
- ReBel-HiBO: 0.20 + 0.70 - 0.14 = 0.76N (**76% 有效, 提升 3.8×**)

**理论上限**: 如果信念分组质量足够高，HiBO 可以将 step advantage 的有效覆盖率从 ~20% 提升到 ~76%。即使信念分组引入一定噪声 (信号质量打折 70%)，总学习信号仍然是:

```
Signal_GiGPO = 1.0 × 0.20N = 0.20N
Signal_HiBO  = 1.0 × 0.20N + 0.70 × 0.56N = 0.20N + 0.39N = 0.59N  (2.95× 提升)
```

## 3.2 自适应衰减的理论动机

### Reward Shaping Theory (Ng et al., 1999)

Potential-based reward shaping 保持最优策略不变:
$$R'(s, a, s') = R(s, a, s') + \gamma \Phi(s') - \Phi(s)$$

信念奖励不是 potential-based (依赖模型输出，不是纯状态函数)，因此引入策略偏差:
$$\pi^*_{R+\alpha R_{belief}} \neq \pi^*_R$$

偏差大小与 α (权重) 成正比。自适应衰减策略:
- 训练早期: α 大 → 偏差大但探索效率高 (acceptable trade-off)
- 训练后期: α→0 → 偏差消失，策略收敛到 $\pi^*_R$

### 差异化衰减的理论依据

设每个组件的偏差为 $b_k$，与最优策略的对齐度为 $a_k$。总偏差:
$$B_{total} = \sum_k w_k \cdot b_k \cdot \alpha_k$$

最小化总偏差的策略: 对 $a_k$ 低 (偏差大) 的组件使用更快的衰减:
- Exploration ($a_k$ 低, $b_k$ 大): 快衰减 → 早期有用的探索引导，后期迅速消除
- Progress ($a_k$ 高, $b_k$ 小): 慢衰减 → 持续提供对齐的学习信号

## 3.3 信念提示改善 HiBO 分组质量

### 信念提示 vs 自由形式推理对 HiBO 的影响

使用 `<think>` 格式时，模型的推理无结构约束 → 无法提取可靠的语义特征 → HiBO 的信念回退层无法使用 → 退化为纯 GiGPO。

使用 `<belief>` 格式时，模型输出结构化 JSON → 可靠提取 (stage, target_found, holding, explore_level) → HiBO 信念回退有效 → 步骤覆盖率大幅提升。

**因此，信念提示不仅是认知脚手架，更是 HiBO 的必要前提条件。三个创新之间存在本质的依赖关系。**

---

# 第四部分: 完整实验方案

## 4.1 实验设计总览

```
Part 1: Main Results (主实验) — 证明 ReBel 达到 SOTA
Part 2: Ablation Study (消融研究) — 隔离三个创新的贡献
Part 3: Deep Analysis (深度分析) — HiBO 分组统计、信念质量、错误分析
Part 4: Generalization (泛化验证) — 模型规模、泛化级别
```

## 4.2 Part 1: Main Results — 方法对比 (5 methods × 3 seeds)

| ID | Method | Adv Estimator | Prompt | Step Grouping | Belief Reward | 对应 V10 |
|----|--------|:-------------:|:------:|:------------:|:------------:|:--------:|
| M1 | GRPO | GRPO | `<think>` | 无 | 无 | V10-A1 |
| M2 | GRPO + Tricks | GRPO | `<think>` | 无 | 无 | V10-A2 |
| M3 | GiGPO + `<think>` | GiGPO | `<think>` | Obs Hash | 无 | **新增** |
| M4 | GiGPO + `<belief>` | GiGPO | `<belief>` | Obs Hash | 无 | V10-C1 |
| **M5** | **ReBel (Ours)** | **HiBO** | **`<belief>`** | **Obs+Belief** | **自适应课程** | **新增** |

### 关键对比

| 对比 | 隔离因素 | 验证内容 |
|------|---------|---------|
| **M5 vs M4** | HiBO + 课程奖励 | **ReBel 完整贡献 (核心!)** |
| **M5 vs M3** | 信念提示 + HiBO + 课程奖励 | 信念全套贡献 |
| M4 vs M3 | 信念提示独立效果 | Belief Prompting as scaffolding |
| M3 vs M2 | Step Advantage 独立效果 | Step-level advantage value |
| M5 vs M1 | ReBel vs 基线 | 完整方法差距 |

### 预期论文表格

**Table 1: Main Results on ALFWorld (mean ± std, 3 seeds)**

| Method | Overall SR (%) | look_at SR (%) | Avg Steps |
|--------|:--------------:|:--------------:|:---------:|
| GRPO | ~81 ± 2 | ~70 ± 5 | ~16 |
| GRPO + Tricks | ~87 ± 2 | ~68 ± 5 | ~15 |
| GiGPO (`<think>`) | ~90 ± 2 | ~75 ± 5 | ~14 |
| GiGPO (`<belief>`) | ~94 ± 1 | ~90 ± 3 | ~13 |
| **ReBel (Ours)** | **~96 ± 1** | **~94 ± 2** | **~12** |

> 注: M5 预期 ~96% 基于: M4 (94.5%) + HiBO 信号恢复 (~+1-2%) + 早期课程加速 (~+0.5-1%)

## 4.3 Part 2: Ablation Study (6 消融 × 1 seed)

从 ReBel (M5) 逐步移除/替换组件:

| ID | 消融 | 改动 | 验证 |
|----|------|------|------|
| A1 | w/o HiBO (obs only) | HiBO → 纯 obs 分组 | HiBO 贡献 |
| A2 | w/o HiBO (belief only) | HiBO → 纯信念分组 | 为什么需要层次化 |
| A3 | w/o Belief Reward | 关闭所有信念奖励 | 课程奖励贡献 |
| A4 | w/ Fixed Decay | 自适应衰减 → 固定 cosine | 自适应衰减 vs 固定衰减 |
| A5 | w/ Uniform Decay | 差异化衰减 → 统一衰减 | 差异化组件衰减的贡献 |
| A6 | w/o Belief Prompt | `<belief>` → `<think>` | 信念提示 (含 HiBO 退化) |

### 预期论文表格

**Table 2: Ablation Study (seed=42)**

| Variant | SR (%) | Δ vs ReBel | 说明 |
|---------|:------:|:----------:|------|
| **ReBel (Full)** | **~96** | — | 全部创新 |
| w/o HiBO → obs only (A1) | ~94 | ~-2 | 回退到 GiGPO 分组 |
| w/o HiBO → belief only (A2) | ~92 | ~-4 | 验证层次化的必要性 |
| w/o Belief Reward (A3) | ~94 | ~-2 | 课程奖励贡献 |
| w/ Fixed Cosine Decay (A4) | ~95 | ~-1 | 自适应优于固定 |
| w/ Uniform Decay (A5) | ~95 | ~-1 | 差异化优于统一 |
| w/o Belief Prompt (A6) | ~90 | ~-6 | 最大降幅 (HiBO 也失效) |

**Table 3: Factor Contribution**

| Innovation | Delta SR | Source | Type |
|-----------|:--------:|:------:|:----:|
| Belief Prompting (全局) | ~+6% | A6 vs M5 | Core |
| HiBO vs Obs-only | ~+2% | A1 vs M5 | Core |
| HiBO vs Belief-only | +4% | A2 vs A1 | Core |
| Belief Curriculum | ~+2% | A3 vs M5 | Core |
| Adaptive Decay | ~+1% | A4 vs M5 | Novel |
| Differential Decay | ~+1% | A5 vs M5 | Novel |

## 4.4 Part 3: Deep Analysis

### 3a. HiBO 分组统计分析 (核心论文图表)

**Figure 1**: Step Advantage Coverage — 有效 step advantage (A_step ≠ 0) 的样本比例

| Method | 有效 Step Adv 比例 |
|--------|:------------------:|
| GiGPO (obs only) | ~20% |
| ReBel-old (belief only, adaptive) | ~65% |
| **ReBel-HiBO** | **~80%** |

**Figure 2**: Group Size Distribution
- 三列直方图对比: GiGPO / Belief-only / HiBO
- 显示 HiBO 如何通过信念回退消除大量单样本组

**Figure 3**: Obs Group vs Belief Fallback 使用比例随训练的变化
- 早期: 更多 belief fallback (obs 匹配少)
- 后期: obs 匹配增加 (轨迹更集中)

### 3b. 信念奖励衰减曲线

**Figure 4**: 实际衰减曲线 (基于训练日志)
- 各组件权重随 epoch 的变化
- SR 曲线叠加
- 标注自适应衰减触发时间点

### 3c. 信念质量分析

**Table 4**: Belief Accuracy Comparison

| Method | Object Loc Acc | Subgoal Acc | Exploration Recall |
|--------|:-------------:|:-----------:|:-----------------:|
| GiGPO (`<think>`) | N/A | N/A | N/A |
| GiGPO (`<belief>`, 无奖励) | ~75% | ~70% | ~65% |
| **ReBel (有课程奖励)** | **~85%** | **~80%** | **~75%** |

> 预期: 课程奖励在早期有效提升信念准确度，即使后期衰减，认知已内化。

### 3d. 错误类型分析

| Method | 认知错误 | 执行错误 | 探索错误 |
|--------|:-------:|:-------:|:-------:|
| GRPO | ~45% | ~30% | ~25% |
| GiGPO + belief | ~15% | ~55% | ~30% |
| **ReBel** | **~10%** | **~60%** | **~30%** |

### 3e. Training Dynamics

**Figure 5**: Learning Curves (all methods, 3-seed mean + shade)
**Figure 6**: Per-task SR evolution
**Figure 7**: Policy entropy evolution (BeST vs ReBel vs GRPO)

## 4.5 Part 4: Generalization

### 4a. ALFWorld 泛化级别

| Level | 说明 | Priority |
|:-----:|------|:--------:|
| 0 | Seen (训练环境) | **必需** |
| 1 | Unseen rooms | 高 |

### 4b. 模型规模 (if resources allow)

| Model | Params | Seeds |
|-------|:------:|:-----:|
| Qwen2.5-1.5B | 1.5B | 3 |
| Qwen2.5-3B | 3B | 1 |

## 4.6 实验优先级

```
Phase 1 [最高] — 核心对比 (5 methods × 3 seeds = 15 runs)
├── M1: GRPO baseline
├── M2: GRPO + Tricks
├── M3: GiGPO + <think>  ← 关键新增
├── M4: GiGPO + <belief> ← 等同 V10-C1，但需 3 seeds
└── M5: ReBel (Full)     ← 核心

Phase 2 [高] — 关键消融 (6 ablations × 1 seed = 6 runs)
├── A1: w/o HiBO → obs only
├── A2: w/o HiBO → belief only
├── A3: w/o Belief Reward
├── A4: w/ Fixed Decay
├── A5: w/ Uniform Decay
└── A6: w/o Belief Prompt

Phase 3 [中] — 分析
├── HiBO 分组统计
├── 信念质量对比
├── 错误类型分析
└── 训练动态图表

Phase 4 [低] — 泛化
├── ALFWorld gen=1
└── 3B 模型验证
```

## 4.7 计算资源估算

| Phase | Runs | GPU-hours |
|-------|:----:|:---------:|
| Phase 1: 5 × 3 seeds | 15 | ~600h |
| Phase 2: 6 × 1 seed | 6 | ~240h |
| Phase 3: Analysis | - | ~30h |
| Phase 4: Gen | ~2 | ~80h |
| **Total** | | **~950h** |

---

# 第五部分: 论文结构规划

## 5.1 推荐标题

1. **ReBel: Rescuing Step-Level Credit Assignment via Belief-Enhanced Grouping for LLM Agents**
2. **Belief-Enhanced Policy Optimization: Hierarchical Grouping and Adaptive Rewards for Interactive LLM Agents**
3. **ReBel: Three Ways Belief States Improve Reinforcement Learning for Interactive Decision-Making**

推荐: **选项 1** — "Rescuing" 精确对应核心故事 (挽救被浪费的学习信号)。

## 5.2 论文结构

```
1. Introduction (1 page)
   - 问题: 多步交互式决策中的 step-level 信用分配
   - 观察: GiGPO 的 obs grouping 产生 70-85% 单样本组 → 学习信号浪费
   - 关键洞察: 结构化信念状态可以从三个层面增强 step-level RL
   - 贡献列表: HiBO + Adaptive Belief Curriculum + Structured Prompting

2. Related Work (0.5-1 page)
   - RL for LLM Agents (GRPO, GiGPO, ArCHer, AgentQ)
   - Process Reward Models & Dense Rewards
   - Belief State Tracking in POMDPs
   - Structured Reasoning (ReAct, Reflexion, Inner Monologue)

3. Preliminary (0.5 page)
   - GRPO, GiGPO formulation
   - The singleton group problem (quantitative analysis)

4. Method: ReBel (2.5 pages)
   4.1 Overview: Three-Level Belief Enhancement
   4.2 Hierarchical Belief-Observation Grouping (HiBO)
       - Semantic Belief Abstraction
       - Hierarchical fallback mechanism
       - Signal recovery analysis
   4.3 Competence-Adaptive Belief Reward Curriculum
       - Performance-gated decay
       - Differential component decay
   4.4 Structured Belief Prompting as Unified Foundation

5. Experiments (3-4 pages)
   5.1 Setup (ALFWorld, model, baselines)
   5.2 Main Results (Table 1)
   5.3 Ablation Study (Table 2, 3)
   5.4 Analysis
       - HiBO group statistics (Figure 1-3) ← 核心图
       - Belief quality comparison (Table 4)
       - Reward curriculum visualization (Figure 4)
       - Error type analysis
       - Training dynamics (Figure 5-7)

6. Discussion
   - Why hierarchical grouping works
   - Connections to reward shaping theory
   - Limitations

7. Conclusion
```

## 5.3 核心贡献点

1. **方法贡献**: 提出 ReBel 框架，首次将结构化信念状态系统性地集成到 LLM 智能体的 RL 训练中 — 作为分组信号 (HiBO)、训练奖励 (课程)、推理结构 (提示)
2. **发现贡献**: 揭示 GiGPO 的严重效率问题 (70-85% 学习信号浪费)，并通过 HiBO 将有效覆盖率提升至 ~80%
3. **技术贡献**: 提出自适应差异化信念奖励衰减，解决密集内在奖励后期有害的问题
4. **分析贡献**: 完整的分组统计、信念质量、错误类型分析，为社区提供深入理解

## 5.4 Rebuttal 准备

| 可能审稿意见 | 回应 |
|------------|------|
| "HiBO 的改进幅度小 (+2%)" | HiBO 的价值在于将有效覆盖率从 20% → 80%，这是一个质变。+2% SR 是在已经很高的基线 (94%) 上的进一步提升 |
| "信念奖励在之前实验中有害" | 这正是我们论文的关键发现: 永久密集奖励有害 (D1=87.5%)，但自适应课程化奖励有益。差异化衰减解决了 exploration 偏差问题 |
| "只在 ALFWorld 验证" | Phase 4 补充; 讨论方法的通用性; HiBO 的思想可推广到任何需要 step-level grouping 的场景 |
| "信念抽象的特征选择很人工" | 讨论: 4 个特征 (stage, found, holding, explore_level) 是自动从 JSON 提取的分类特征，不需要领域专家设计 |
| "与 GiGPO 相比创新不够大" | ReBel 在三个层面创新: (1) 分组 (2) 奖励 (3) 提示，形成闭环系统。任何单一组件都不足以实现最终性能 |

---

# 第六部分: 代码实现方案

## 6.1 需要修改的核心代码

### 6.1.1 新增: HiBO 分组模块

**文件**: `code/rebel/hibo_grouping.py` (新建)

```python
# 核心功能:
# 1. semantic_belief_abstract(belief_json) → hash
# 2. build_hibo_groups(anchor_obs, belief_abstracts, uids) → group_assignments
# 3. compute_hibo_step_advantage(step_returns, groups, mode)
```

### 6.1.2 修改: 自适应衰减

**文件**: `code/agent_system/environments/env_package/alfworld/belief_tracker.py`

```python
# 修改 get_belief_reward_weight():
# - 接受 success_rate 参数
# - 实现自适应衰减逻辑
# - 实现差异化组件衰减

# 修改 calculate_total_intrinsic_reward():
# - 使用差异化组件权重
```

### 6.1.3 修改: 训练循环集成

**文件**: `code/verl/trainer/ppo/ray_trainer.py`

```python
# 新增 AdvantageEstimator.ReBelHiBO
# 在 advantage 计算分支中添加 HiBO 路径
# 传递 success_rate 给 env_manager
```

### 6.1.4 修改: 数据流

**文件**: `code/agent_system/multi_turn_rollout/rollout_loop.py`

```python
# 在 rollout 数据中添加 belief_abstract 字段
# 在每步调用 semantic_belief_abstract() 提取特征
```

## 6.2 完整配置

### ReBel (V11 Final) 完整配置

```yaml
# ========================================
# ReBel: Belief-Enhanced Policy Optimization
# V11 Rev.2 Final Configuration
# ========================================

# --- Advantage Estimation (HiBO) ---
algorithm.adv_estimator: rebel_hibo        # 新的 HiBO 估计器
algorithm.rebel_hibo.step_advantage_w: 0.5
algorithm.rebel_hibo.mode: "mean_norm"
algorithm.rebel_hibo.gamma: 0.95
algorithm.rebel_hibo.min_obs_group_size: 2  # obs 组最小大小 (低于此回退)
algorithm.rebel_hibo.belief_granularity: semantic  # 新的语义抽象粒度

# --- Belief Prompting ---
algorithm.rebel.enable: true
env.alfworld.use_rebel: true
env.alfworld.prompt_template_type: "explicit_task_type"

# --- Belief Curriculum Reward ---
algorithm.rebel.use_belief_reward: true     # 启用!
algorithm.rebel.use_result_reward: true
algorithm.rebel.belief_reward_decay:
  enable: true
  method: adaptive
  warmup_epochs: 3
  decay_start_epoch: 5
  decay_end_epoch: 40
  min_weight: 0.05
  target_sr: 0.90
  alpha: 2.0
  progress_decay_rate: 0.7
  consistency_decay_rate: 1.0
  exploration_decay_rate: 2.0

# --- Training Stabilization ---
actor_rollout_ref.actor.clip_ratio_low: 0.2
actor_rollout_ref.actor.clip_ratio_high: 0.28
actor_rollout_ref.actor.entropy_coeff: 0.001
actor_rollout_ref.actor.use_kl_loss: true
actor_rollout_ref.actor.kl_loss_coef: 0.01
actor_rollout_ref.actor.kl_loss_type: low_var_kl
actor_rollout_ref.actor.use_invalid_action_penalty: true
actor_rollout_ref.actor.invalid_action_penalty_coef: 0.1
algorithm.rebel.entropy_protection.enable: true
algorithm.rebel.entropy_protection.method: clip_cov
algorithm.rebel.entropy_protection.clip_cov_lb: 0.0
algorithm.rebel.entropy_protection.clip_cov_ub: 0.3

# --- Training Parameters ---
data.train_batch_size: 16
data.val_batch_size: 128
data.max_prompt_length: 6000
data.max_response_length: 1024
actor_rollout_ref.actor.optim.lr: 1e-6
actor_rollout_ref.actor.ppo_epochs: 1
env.max_steps: 30
env.rollout.n: 16
env.alfworld.generalization_level: 0
env.alfworld.meta_think: true
env.use_teacher_planner: true
trainer.total_epochs: 100
trainer.test_freq: 5
trainer.save_freq: 25
trainer.val_before_train: true
```

## 6.3 实验脚本清单

```
v11_final/
├── V11_FINAL_ALGORITHM_AND_EXPERIMENT_PLAN.md   # 本文档
├── run_v11_base.sh                              # 通用训练脚本
├── run_all_v11.sh                               # 实验编排脚本
├── experiments/
│   ├── M1_grpo_baseline.sh                      # GRPO 基线
│   ├── M2_grpo_tricks.sh                        # GRPO + Tricks
│   ├── M3_gigpo_think.sh                        # GiGPO + <think>
│   ├── M4_gigpo_belief.sh                       # GiGPO + <belief> (无 HiBO)
│   ├── M5_rebel_full.sh                         # ReBel Full (核心!)
│   ├── A1_ablation_obs_only.sh                  # 消融: HiBO → obs only
│   ├── A2_ablation_belief_only.sh               # 消融: HiBO → belief only
│   ├── A3_ablation_no_reward.sh                 # 消融: 关闭信念奖励
│   ├── A4_ablation_fixed_decay.sh               # 消融: 固定 cosine 衰减
│   ├── A5_ablation_uniform_decay.sh             # 消融: 统一衰减
│   └── A6_ablation_no_belief_prompt.sh          # 消融: <think> 格式
└── analysis/
    ├── hibo_group_statistics.py                 # HiBO 分组统计分析
    ├── belief_accuracy_eval.py                  # 信念准确度评测
    ├── error_type_classifier.py                 # 错误类型分类
    └── generate_paper_figures.py                # 论文图表生成
```

---

## 附录 A: ReBel vs BeST (前版方案) 对比

| 维度 | BeST (前版) | ReBel (本版) |
|------|-----------|------------|
| 信念的角色 | 仅认知脚手架 (1 个作用) | 三重作用 (分组+奖励+脚手架) |
| 分组策略 | GiGPO obs hash (70-85% 浪费) | HiBO (obs 为主, belief 兜底) |
| 信念奖励 | **不使用** (基于 V10 结论) | **使用** (自适应课程衰减) |
| 核心创新 | Prompt 格式 + 协同效应假设 | 层次化分组 + 自适应衰减 + 闭环 |
| 新颖性 | 低 (本质是 GiGPO + prompt) | **高** (三层创新, 理论支撑) |
| 审稿人可能质疑 | "与 GiGPO 差别只是 prompt" | 多维度创新, 定量分析支撑 |

## 附录 B: V10 实验的复用策略

| V11 实验 | V10 对应 | 复用策略 |
|---------|---------|---------|
| M1 (GRPO) | V10-A1 | 需重新运行 (3 seeds) |
| M2 (GRPO + Tricks) | V10-A2 | 需重新运行 (3 seeds) |
| M3 (GiGPO + think) | **缺失** | **必须新增** (3 seeds) |
| M4 (GiGPO + belief) | V10-C1 | 可复用 seed=42, 补充 2 seeds |
| M5 (ReBel) | **缺失** | **必须新增** (3 seeds, 需要代码修改) |
| A1 (obs only) | = M4 | 复用 M4 seed=42 |
| A2 (belief only) | ≈ V10-C2 | 需重新运行 (新配置) |
| A3 (no reward) | ≈ M4 | 复用或重新运行 |

## 附录 C: 实现优先级

```
Step 1: 实现 HiBO 分组模块
  → 新建 hibo_grouping.py
  → 修改 ray_trainer.py 添加 HiBO advantage 分支
  → 修改 rollout_loop.py 添加 belief_abstract 数据字段

Step 2: 实现自适应衰减
  → 修改 belief_tracker.py 的衰减逻辑
  → 修改 env_manager.py 传递 success_rate
  → 修改 ray_trainer.py 传递 validation metrics

Step 3: 验证实验
  → 短跑 10 epochs 验证 HiBO 分组正确性
  → 检查 belief_abstract 分组统计
  → 确认自适应衰减触发正常

Step 4: 正式实验
  → Phase 1: M3 + M4 + M5 (最关键)
  → Phase 2: 消融实验
  → Phase 3: 分析
```

---

*文档生成时间: 2026-02-20*
*基于 V10 消融数据 + 代码深度分析 + GiGPO 单样本组统计 + 信念奖励组件分析*
