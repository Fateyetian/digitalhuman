# ReBel RL算法实现指南

> 本文档用于指导 `core_rebel.py` 的实现与改进

---

## 目录
1. [算法概述](#1-算法概述)
2. [核心公式](#2-核心公式)
3. [数学推导与理论分析](#3-数学推导与理论分析)
4. [关键组件实现](#4-关键组件实现)
5. [与GiGPO/RLVMR共享组件](#5-与gigporlvmr共享组件)
6. [奖励系统详解](#6-奖励系统详解)
7. [分组机制](#7-分组机制)
8. [改进方向](#8-改进方向)
9. [实现Checklist](#9-实现checklist)

---

## 1. 算法概述

### 1.1 ReBel定位

```
GRPO ──> GiGPO ──> RLVMR ──> ReBel
 │         │         │         │
 │         │         │         └─ 语义Belief分组 + 4组件过程奖励
 │         │         └─ 思维标签分组 + 规则奖励
 │         └─ 环境状态分组 + Discounted Return
 └─ Episode级别分组
```

### 1.2 核心创新点

| 创新 | 描述 | 优势 |
|------|------|------|
| **Belief-based Grouping** | 按语义相似的belief state分组 | 粒度适中(20-100组) |
| **4组件内在奖励** | Consistency + Progress + Exploration + Format | 真正的过程奖励 |
| **Hindsight Annotation** | 给定专家action，反推belief | 数据质量高 |

### 1.3 与其他算法对比

| 维度 | GiGPO | RLVMR | ReBel |
|------|-------|-------|-------|
| 分组依据 | anchor_obs | tag_type | belief_hash |
| 分组数量 | 500-1000 | 3-4 | 20-100 |
| 组大小 | ~1.2 | ~350 | ~24 |
| 过程奖励 | ❌ | ⚠️部分 | ✅完全 |

---

## 2. 核心公式

### 2.1 总优势函数

```
A_total = A_episode + λ × A_step

其中:
- A_episode: Episode级别优势 (按uid分组归一化)
- A_step: Step级别优势 (按belief分组归一化)
- λ: step_advantage_w (默认1.0)
```

### 2.2 Episode优势

```python
# 按uid分组，计算组内归一化
A_episode[i] = (R_episode[i] - mean(R_episode[uid])) / (std(R_episode[uid]) + ε)

# 如果使用mean_norm模式:
A_episode[i] = R_episode[i] - mean(R_episode[uid])
```

### 2.3 Step优势

```python
# 按(uid, belief_hash)分组，计算组内归一化
group_key = (uid[i], canonicalize_belief(belief[i], granularity))
A_step[i] = (R_intrinsic[i] - mean(R_intrinsic[group_key])) / (std + ε)
```

### 2.4 内在奖励公式

```python
# 内在奖励只包含3个语义组件
R_intrinsic = (
    α × R_consistency +    # 默认 α=0.4
    β × R_progress +       # 默认 β=0.4
    γ × R_exploration      # 默认 γ=0.2
)
# 注意: Format不是内在奖励，而是独立的格式惩罚
```

### 2.5 格式惩罚 (独立于内在奖励)

```python
# 格式惩罚类似RLVMR，作为validity gate而非reward component
if not is_format_valid:
    R_step = -1.0  # 格式错误直接惩罚，不参与归一化
    skip_intrinsic = True  # 跳过内在奖励计算
else:
    R_step = R_intrinsic  # 格式正确才计算内在奖励
```

**与RLVMR的区别**:
| 维度 | RLVMR | ReBel |
|------|-------|-------|
| 格式惩罚 | valid=0 → 不给step奖励 | valid=0 → 给负惩罚 |
| 有效时 | 给固定step奖励 | 计算内在奖励 |

---

## 3. 数学推导与理论分析

### 3.1 PPO目标函数回顾

标准PPO的目标函数为：

```
L^{CLIP}(θ) = E_t[ min(r_t(θ) · A_t, clip(r_t(θ), 1-ε, 1+ε) · A_t) ]

其中:
- r_t(θ) = π_θ(a_t|s_t) / π_{θ_old}(a_t|s_t)  (importance ratio)
- A_t: 优势估计
```

### 3.2 GRPO的分组归一化

GRPO按episode分组，优势估计为：

```
A^{GRPO}_i = (R_i - μ_G) / (σ_G + ε)

其中 G = {j : prompt_j = prompt_i}  (同一prompt的所有轨迹)
```

**问题**: Episode级别归一化粒度太粗，无法区分同一轨迹内不同step的贡献。

### 3.3 GiGPO/RLVMR/ReBel的双层优势

三者共享相同的双层优势框架：

```
A^{total}_i = A^{episode}_i + λ · A^{step}_i
```

**Episode优势** (三者完全相同，复用GiGPO实现):

```
A^{episode}_i = R^{episode}_i - μ_{G_{uid}}

其中 G_{uid} = {j : uid_j = uid_i}
```

**Step优势** (分组策略不同):

| 算法 | 分组策略 | 分组数量 | 组大小 |
|------|----------|----------|--------|
| GiGPO | G = {j : obs_j = obs_i, uid_j = uid_i} | 500-1000 | ~1.2 |
| RLVMR | G = {j : tag_j = tag_i, uid_j = uid_i} | 3-4 | ~350 |
| ReBel | G = {j : belief_hash_j = belief_hash_i, uid_j = uid_i} | 20-100 | ~24 |

### 3.4 方差分析与最优分组粒度

**定理**: 给定固定的样本总数N和分组数K，归一化后优势的方差为：

```
Var(A^{norm}) = Var(A^{raw}) · (1 - 1/n_G)

其中 n_G = N/K 是平均组大小
```

**推导**:

设组G内有n个样本，原始优势为{A_1, ..., A_n}，均值为μ_G。

归一化后: A^{norm}_i = A_i - μ_G

```
Var(A^{norm}) = E[(A_i - μ_G)²] - E[A_i - μ_G]²
              = Var(A) - Var(μ_G)
              = Var(A) - Var(A)/n
              = Var(A) · (1 - 1/n)
```

**结论**:
- n=1 (GiGPO极端情况): Var(A^{norm}) = 0，无梯度信号
- n→∞ (GRPO): Var(A^{norm}) = Var(A)，无归一化效果
- **最优粒度**: 需要平衡信号强度与归一化效果

### 3.5 ReBel的理论优势

**命题1**: ReBel的Belief分组实现了状态-动作空间的语义压缩。

**证明**:
设状态空间S，动作空间A，Belief空间B。

GiGPO的分组: G = {(s,a) : s = s_i}，分组数 |G| ≈ |S|

ReBel的分组: G = {(s,a) : φ(s) = φ(s_i)}，其中φ: S → B是Belief映射

由于Belief是对任务进度的抽象，有 |B| << |S|，因此：

```
|G^{ReBel}| << |G^{GiGPO}|
n^{ReBel}_G >> n^{GiGPO}_G
```

**命题2**: ReBel的内在奖励提供了密集的学习信号。

**证明**:
定义信号密度: ρ = #{i : R_i ≠ 0} / N

- GiGPO: ρ = 1/T (只有最后一步有奖励)
- RLVMR: ρ = #{valid_tags} / T ≈ 0.7 (取决于格式正确率)
- ReBel: ρ = 1 (每步都有4组件奖励)

### 3.6 收敛性分析

**定理**: 在满足以下条件时，ReBel收敛到局部最优：

1. **组大小下界**: n_G ≥ 2 (保证可计算方差)
2. **奖励有界**: |R_intrinsic| ≤ R_max
3. **步长衰减**: α_t = O(1/t)

**梯度估计方差**:

```
Var(∇L^{ReBel}) = Var(∇L^{episode}) + λ² · Var(∇L^{step})
                ≤ Var(∇L^{GRPO}) + λ² · σ²_{step} / n_G
```

当n_G ≈ 24 (ReBel典型值) 时，step梯度方差被有效控制。

### 3.7 与GiGPO/RLVMR的理论对比

| 指标 | GiGPO | RLVMR | ReBel | 分析 |
|------|-------|-------|-------|------|
| 组大小方差 | 高 | 低 | 中 | ReBel平衡 |
| 信号密度 | 低(1/T) | 中(~0.7) | 高(1.0) | ReBel最优 |
| 语义一致性 | 低 | 中 | 高 | ReBel按语义分组 |
| 归一化效果 | 弱(组太小) | 强(组太大) | 适中 | ReBel平衡 |

**理论预测**: ReBel应该在以下场景表现最佳：
1. 长轨迹任务 (T > 10)
2. 需要规划的任务
3. 状态空间大但语义空间小的任务

---

## 4. 关键组件实现

### 3.1 Belief Canonicalization

**功能**: 将belief state转换为可哈希的规范表示

**文件位置**: `core_rebel.py:19-87`

```python
def canonicalize_belief(belief_state: Dict, granularity: str = 'subgoal') -> str:
    """
    Args:
        belief_state: 解析后的belief state字典
        granularity: 粒度级别
            - 'subgoal': 只用subgoal + status (推荐)
            - 'medium': + found_objects统计
            - 'fine': 完整belief state

    Returns:
        belief_hash: 16位MD5哈希字符串
    """
    if granularity == 'subgoal':
        canonical = {
            'subgoal': belief_state.get('task_progress_update', {})
                                   .get('updated_subgoal', '').lower().strip(),
            'status': belief_state.get('task_progress_update', {})
                                  .get('subgoal_status', '').lower().strip()
        }
    elif granularity == 'medium':
        # 加入found_objects统计
        found_objects = belief_state.get('world_model_update', {}).get('found_objects', {})
        canonical = {
            'subgoal': ...,
            'status': ...,
            'num_found_objects': len(found_objects),
            'found_object_types': sorted([obj.split()[0] for obj in found_objects.keys()])
        }
    elif granularity == 'fine':
        canonical = belief_state  # 完整state

    # 生成哈希
    canonical_str = json.dumps(canonical, sort_keys=True)
    return hashlib.md5(canonical_str.encode()).hexdigest()[:16]
```

**粒度选择指南**:

| 粒度 | 分组数 | 组大小 | 稳定性 | 推荐场景 |
|------|--------|--------|--------|----------|
| subgoal | 20-50 | 20-40 | 高 ✅ | **默认使用** |
| medium | 50-150 | 10-25 | 中 | 任务阶段差异大 |
| fine | 100-500 | 2-10 | 低 | 调试分析 |

---

### 3.2 Belief Group构建

**功能**: 按belief相似性将steps分组

**文件位置**: `core_rebel.py:89-186`

```python
def build_belief_group(
    belief_states: np.ndarray,
    index: np.ndarray,
    granularity: str = 'subgoal',
    summarize: bool = False
) -> Tuple[np.ndarray, Dict]:
    """
    Args:
        belief_states: shape (batch_size,), 每个元素是belief字典
        index: shape (batch_size,), 每个step的prompt uid
        granularity: canonicalization粒度
        summarize: 是否打印分组统计

    Returns:
        belief_group_uids: shape (batch_size,), 每个step的group uid
        group_stats: 分组统计信息
    """
    belief_group_uids = np.empty(len(belief_states), dtype=object)
    unique_indices = np.unique(index)
    group_sizes = []

    for idx in unique_indices:
        # 1. 获取该uid的所有steps
        step_indices = np.where(index == idx)[0]
        beliefs = belief_states[step_indices]

        # 2. 按belief hash聚类
        clusters = defaultdict(list)
        for i, belief in enumerate(beliefs):
            belief_hash = canonicalize_belief(belief, granularity)
            clusters[belief_hash].append(step_indices[i])

        # 3. 分配group uid
        for belief_hash, indices in clusters.items():
            group_uid = f"belief_{idx}_{belief_hash}"
            group_sizes.append(len(indices))
            for i in indices:
                belief_group_uids[i] = group_uid

    group_stats = {
        'num_groups': len(set(belief_group_uids)),
        'mean_group_size': np.mean(group_sizes),
        'median_group_size': np.median(group_sizes),
    }

    return belief_group_uids, group_stats
```

---

### 3.3 Step奖励归一化

**功能**: 在belief group内归一化step奖励

**文件位置**: `core_rebel.py:242-297`

```python
def step_norm_reward_by_belief(
    step_rewards: torch.Tensor,      # shape (batch_size,)
    eos_mask: torch.Tensor,          # shape (batch_size, seq_len)
    belief_group_uids: np.ndarray,   # shape (batch_size,)
    epsilon: float = 1e-6,
    remove_std: bool = True          # True=mean_norm, False=mean_std_norm
) -> torch.Tensor:
    """
    Returns:
        step_advantages: shape (batch_size, seq_len)
    """
    response_length = eos_mask.shape[-1]
    scores = step_rewards.clone()

    # 1. 收集每个group的rewards
    group2rewards = defaultdict(list)
    for i in range(len(scores)):
        group2rewards[belief_group_uids[i]].append(scores[i])

    # 2. 计算每个group的mean/std
    group2stats = {}
    for group_uid, rewards in group2rewards.items():
        t = torch.tensor(rewards)
        group2stats[group_uid] = (torch.mean(t), torch.std(t))

    # 3. 归一化
    for i in range(len(scores)):
        mean, std = group2stats[belief_group_uids[i]]
        if remove_std:
            scores[i] = scores[i] - mean
        else:
            scores[i] = (scores[i] - mean) / (std + epsilon)

    # 4. 广播到所有token
    step_advantages = scores.unsqueeze(-1).expand(-1, response_length) * eos_mask

    return step_advantages
```

---

### 3.4 主函数: compute_rebel_advantage

**文件位置**: `core_rebel.py:299-376`

```python
def compute_rebel_advantage(
    token_level_rewards: torch.Tensor,    # (batch, seq_len) episode reward
    rebel_intrinsic_rewards: torch.Tensor, # (batch,) intrinsic reward
    eos_mask: torch.Tensor,               # (batch, seq_len)
    belief_states: np.ndarray,            # (batch,) belief dicts
    index: np.ndarray,                    # (batch,) prompt uids
    epsilon: float = 1e-6,
    step_advantage_w: float = 1.0,
    mode: str = "mean_norm",
    belief_granularity: str = 'subgoal',
    summarize: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """
    Returns:
        advantages: (batch, seq_len) 总优势
        returns: (batch, seq_len) 同上
        adv_details: 包含episode_adv, step_adv, group_stats
    """
    remove_std = (mode == "mean_norm")

    # 1. Episode优势
    episode_advantages = episode_norm_reward(
        token_level_rewards, eos_mask, index, epsilon, remove_std
    )

    # 2. 构建belief groups
    belief_group_uids, group_stats = build_belief_group(
        belief_states, index, belief_granularity, summarize
    )

    # 3. Step优势
    step_advantages = step_norm_reward_by_belief(
        rebel_intrinsic_rewards, eos_mask, belief_group_uids, epsilon, remove_std
    )

    # 4. 组合
    total_advantages = episode_advantages + step_advantage_w * step_advantages

    return total_advantages, total_advantages, {
        'episode_advantages': episode_advantages,
        'step_advantages': step_advantages,
        'belief_group_stats': group_stats
    }
```

---

## 4. 奖励系统详解

### 4.1 四组件内在奖励

```python
def compute_rebel_intrinsic_reward(
    belief: Dict,
    prev_belief: Dict,
    ground_truth: GroundTruth,
    output: str,
    is_format_valid: bool,
    is_action_available: bool,
    weights: Dict = None
) -> float:
    """计算ReBel内在奖励"""
    weights = weights or {
        'consistency': 0.3,
        'progress': 0.5,
        'exploration': 0.2,
        'format': 0.1
    }

    reward = 0.0

    # 1. Consistency Reward
    reward += weights['consistency'] * consistency_reward(belief, ground_truth)

    # 2. Progress Reward
    reward += weights['progress'] * progress_reward(belief, prev_belief)

    # 3. Exploration Reward
    reward += weights['exploration'] * exploration_reward(belief, prev_belief)

    # 4. Format Reward
    reward += weights['format'] * format_reward(is_format_valid, is_action_available)

    return reward
```

### 4.2 各组件实现

#### Consistency Reward (一致性奖励)

```python
def consistency_reward(belief: Dict, ground_truth: GroundTruth) -> float:
    """评估belief与真实环境的一致性"""
    reward = 0.0
    world_model = belief.get('world_model_update', {})

    # 正确的物体位置
    for obj_id, believed_loc in world_model.get('found_objects', {}).items():
        if ground_truth.is_object_at(obj_id, believed_loc):
            reward += 0.2  # 正确belief
        else:
            reward -= 0.1  # 错误belief

    # 正确的状态变化
    for obj_id, believed_state in world_model.get('state_changes', {}).items():
        if ground_truth.check_state(obj_id, believed_state):
            reward += 0.1

    # Inventory一致性
    inventory = world_model.get('inventory')
    if ground_truth.check_inventory(inventory):
        reward += 0.1

    return np.clip(reward, -0.5, 1.0)
```

#### Progress Reward (进度奖励)

```python
def progress_reward(belief: Dict, prev_belief: Dict) -> float:
    """评估任务进度"""
    reward = 0.0
    task_progress = belief.get('task_progress_update', {})

    # 子目标完成
    if task_progress.get('subgoal_status') == 'completed':
        prev_status = prev_belief.get('task_progress_update', {}).get('subgoal_status', '')
        if prev_status != 'completed':
            reward += 0.5  # 新完成子目标

    # 有意义的证据
    evidence = task_progress.get('evidence', '')
    if evidence and len(evidence) > 10:
        reward += 0.1

    # 子目标更新
    curr_subgoal = task_progress.get('updated_subgoal', '')
    prev_subgoal = prev_belief.get('task_progress_update', {}).get('updated_subgoal', '')
    if curr_subgoal and curr_subgoal != prev_subgoal:
        reward += 0.1  # 子目标推进

    return np.clip(reward, 0, 1.0)
```

#### Exploration Reward (探索奖励)

```python
def exploration_reward(belief: Dict, prev_belief: Dict) -> float:
    """评估探索效率"""
    reward = 0.0
    exploration = belief.get('exploration_map_update', {})

    # 新访问的位置
    newly_visited = set(exploration.get('newly_visited', []))
    prev_visited = set(prev_belief.get('exploration_map_update', {})
                                  .get('newly_visited', []))
    new_locations = newly_visited - prev_visited
    reward += 0.1 * len(new_locations)

    # 避免重复探索
    cleared = belief.get('world_model_update', {}).get('cleared_receptacles', [])
    for loc in newly_visited:
        if loc in cleared:
            reward -= 0.02  # 重复访问已清理的位置

    return np.clip(reward, -0.1, 0.5)
```

#### Format Reward (格式奖励) - 改进版

**设计思路**: 参考RLVMR的二元惩罚机制，对格式错误给予严格惩罚

**ReBel格式验证规则**:

| 检查项 | 规则 | 惩罚 |
|--------|------|------|
| 1. 中文字符 | 输出不能包含中文 | valid=0 |
| 2. action标签数量 | `<action>...</action>` 必须恰好1个 | valid=0 |
| 3. belief标签数量 | `<belief>...</belief>` 必须恰好1个 | valid=0 |
| 4. reasoning标签数量 | `<reasoning>...</reasoning>` 必须恰好1个 | valid=0 |
| 5. belief内容非空 | belief标签内容不能为空 | valid=0 |
| 6. reasoning内容非空 | reasoning标签内容不能为空 | valid=0 |
| 7. 标签顺序 | 必须是 belief → reasoning → action | valid=0 |
| 8. belief JSON有效 | belief内必须是合法JSON | valid=0 |

**合法格式示例**:
```
<belief>
{
  "world_model_update": {"found_objects": {"apple 1": "fridge 1"}},
  "task_progress_update": {"subgoal_status": "in_progress", "updated_subgoal": "Find apple"}
}
</belief>

<reasoning>
I found the apple in fridge 1. Now I need to pick it up.
</reasoning>

<action>
take apple 1 from fridge 1
</action>
```

**非法格式示例**:
```python
# 错误1：缺少belief标签
<reasoning>I will go to desk</reasoning>
<action>go to desk 1</action>

# 错误2：多个action标签
<belief>{"world_model_update": {}}</belief>
<reasoning>xxx</reasoning>
<action>go to desk 1</action>
<action>take pen</action>

# 错误3：标签顺序错误 (action在reasoning之前)
<belief>{"world_model_update": {}}</belief>
<action>go to desk 1</action>
<reasoning>xxx</reasoning>

# 错误4：belief内容为空
<belief></belief>
<reasoning>xxx</reasoning>
<action>go to desk 1</action>

# 错误5：belief不是有效JSON
<belief>this is not json</belief>
<reasoning>xxx</reasoning>
<action>go to desk 1</action>

# 错误6：包含中文
<belief>{"world_model_update": {}}</belief>
<reasoning>去桌子那里</reasoning>
<action>go to desk 1</action>
```

```python
def format_reward(output: str, is_action_available: bool) -> float:
    """
    评估输出格式 - RLVMR风格惩罚机制

    Returns:
        reward: 格式正确返回小奖励，格式错误返回惩罚
    """
    import re
    import json

    # 默认格式无效
    is_format_valid = False

    # ========== 格式验证 (任一失败则valid=0) ==========

    # 1. 检查中文字符
    if re.search(r'[\u4e00-\u9fff]', output):
        return -1.0  # 严重惩罚：包含中文

    # 2. 检查action标签数量 (必须恰好1个)
    action_matches = re.findall(r"<action>([\s\S]*?)</action>", output, re.IGNORECASE)
    if len(action_matches) != 1:
        return -1.0  # 严重惩罚：action标签数量错误

    # 3. 检查belief标签数量 (必须恰好1个)
    belief_matches = re.findall(r"<belief>([\s\S]*?)</belief>", output, re.IGNORECASE)
    if len(belief_matches) != 1:
        return -1.0  # 严重惩罚：belief标签数量错误

    # 4. 检查reasoning标签数量 (必须恰好1个)
    reasoning_matches = re.findall(r"<reasoning>([\s\S]*?)</reasoning>", output, re.IGNORECASE)
    if len(reasoning_matches) != 1:
        return -1.0  # 严重惩罚：reasoning标签数量错误

    # 5. 检查belief内容非空
    belief_content = belief_matches[0].strip()
    if not belief_content:
        return -1.0  # 严重惩罚：belief内容为空

    # 6. 检查reasoning内容非空
    reasoning_content = reasoning_matches[0].strip()
    if not reasoning_content:
        return -1.0  # 严重惩罚：reasoning内容为空

    # 7. 检查标签顺序: belief → reasoning → action
    belief_pos = output.lower().find("<belief>")
    reasoning_pos = output.lower().find("<reasoning>")
    action_pos = output.lower().find("<action>")

    if not (belief_pos < reasoning_pos < action_pos):
        return -1.0  # 严重惩罚：标签顺序错误

    # 8. 检查belief是否为有效JSON
    try:
        belief_json = json.loads(belief_content)
        # 可选：检查必需字段
        valid_keys = ["world_model_update", "task_progress_update", "exploration_map_update"]
        if not any(key in belief_json for key in valid_keys):
            return -0.5  # 中等惩罚：缺少必需字段
    except json.JSONDecodeError:
        return -1.0  # 严重惩罚：JSON解析失败

    # ========== 格式验证通过 ==========
    is_format_valid = True
    reward = 0.1  # 基础格式正确奖励

    # 额外检查：action是否在可用列表中
    if not is_action_available:
        reward -= 0.2  # 惩罚：action不在可用列表

    return reward
```

**与RLVMR对比**:

| 维度 | RLVMR | ReBel |
|------|-------|-------|
| 思维标签 | `<planning>/<reflection>/<explore>/<monitor>` | `<belief>` + `<reasoning>` |
| 标签数量 | 技能标签1个 + action 1个 | belief 1个 + reasoning 1个 + action 1个 |
| 内容验证 | 标签内容非空 | 标签内容非空 + belief必须是有效JSON |
| 顺序要求 | 技能标签 → action | belief → reasoning → action |
| 惩罚机制 | 二元(valid=0/1) | 二元惩罚(-1.0) |

---

## 5. 分组机制

### 5.1 分组流程图

```
┌──────────────────────────────────────────────────────────┐
│                    输入: batch of steps                   │
│  (belief_states, uids, intrinsic_rewards)                │
└──────────────────────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────┐
│  Step 1: 按uid分组 (同一prompt的所有轨迹steps)            │
│                                                          │
│  uid_0: [step_0, step_1, step_2, ...]                   │
│  uid_1: [step_5, step_6, step_7, ...]                   │
└──────────────────────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────┐
│  Step 2: 每个uid内，按belief_hash聚类                     │
│                                                          │
│  uid_0:                                                  │
│    ├─ belief_hash_a: [step_0, step_2]  (找番茄)          │
│    └─ belief_hash_b: [step_1]          (拿番茄)          │
│  uid_1:                                                  │
│    ├─ belief_hash_a: [step_5, step_7]                   │
│    └─ belief_hash_c: [step_6]                           │
└──────────────────────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────┐
│  Step 3: 每个cluster分配group_uid                        │
│                                                          │
│  group_0: [step_0, step_2]  → 组内归一化                 │
│  group_1: [step_1]          → 单样本跳过                 │
│  group_2: [step_5, step_7]  → 组内归一化                 │
│  group_3: [step_6]          → 单样本跳过                 │
└──────────────────────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────┐
│  Step 4: 组内归一化 intrinsic reward                     │
│                                                          │
│  A_step[i] = (R[i] - mean(R[group])) / (std + ε)        │
└──────────────────────────────────────────────────────────┘
```

### 5.2 单样本处理

当组内只有1个样本时：
```python
if len(group_rewards) == 1:
    mean = 0.0  # 不减均值
    std = 1.0   # 不除标准差
    # 结果: A_step = R - 0 = R (保持原值)
```

---

## 6. 改进方向

### 6.1 规划奖励增强

**问题**: 当前ReBel没有显式的规划奖励

**方案**: 第一步强制规划 + 执行对齐检查

```python
def compute_planning_reward(
    step_idx: int,
    belief: Dict,
    plan: List[str],
    current_action: str
) -> float:
    """
    Args:
        step_idx: 当前步骤索引
        belief: 当前belief state
        plan: 第一步生成的规划列表
        current_action: 当前执行的action
    """
    if step_idx == 0:
        # 第一步: 评估规划质量
        return planning_quality_reward(plan, belief)
    else:
        # 后续步骤: 评估执行与规划的对齐
        return plan_alignment_reward(plan, step_idx, current_action)


def planning_quality_reward(plan: List[str], belief: Dict) -> float:
    """评估规划质量"""
    reward = 0.0

    # 1. 规划完整性
    if len(plan) >= 2:
        reward += 0.1
    if len(plan) >= 4:
        reward += 0.1

    # 2. 规划具体性 (每步是否可执行)
    for step in plan:
        if is_executable_step(step):
            reward += 0.05

    # 3. 规划一致性 (与当前belief不冲突)
    if not has_conflict(plan, belief):
        reward += 0.1

    return np.clip(reward, 0, 0.5)


def plan_alignment_reward(plan: List[str], step_idx: int, action: str) -> float:
    """评估执行与规划的对齐"""
    if step_idx > len(plan):
        return 0.0

    expected_step = plan[step_idx - 1]  # 规划的第step_idx步

    # 语义匹配
    similarity = semantic_similarity(expected_step, action)

    return 0.1 * similarity
```

### 6.2 延迟规划评估

```python
def delayed_planning_evaluation(plan: List[str], trajectory: List[Dict]) -> float:
    """轨迹结束时回溯评估规划质量"""

    # 1. 规划执行率
    executed_count = sum(1 for i, step in enumerate(plan)
                        if was_executed(step, trajectory))
    executed_ratio = executed_count / len(plan)

    # 2. 规划有效率
    effective_count = sum(1 for i, step in enumerate(plan)
                         if was_effective(step, trajectory))
    effective_ratio = effective_count / max(executed_count, 1)

    # 3. 组合评分
    planning_score = 0.4 * executed_ratio + 0.6 * effective_ratio

    return planning_score
```

### 6.3 动态粒度调整

```python
def adaptive_granularity(group_stats: Dict) -> str:
    """根据当前分组情况动态调整粒度"""
    mean_size = group_stats['mean_group_size']

    if mean_size < 5:
        return 'subgoal'  # 组太小，用更粗的粒度
    elif mean_size > 50:
        return 'medium'   # 组太大，用更细的粒度
    else:
        return 'subgoal'  # 保持当前
```

### 6.4 改进后的完整奖励公式

```python
R_intrinsic_v2 = (
    α × R_consistency +      # 0.25
    β × R_progress +         # 0.30
    γ × R_exploration +      # 0.15
    δ × R_format +           # 0.10
    ε × R_planning +         # 0.10 (新增)
    ζ × R_plan_alignment     # 0.10 (新增)
)
```

---

## 7. 实现Checklist

### 7.1 核心功能

- [ ] `canonicalize_belief()` - Belief规范化
  - [ ] subgoal粒度
  - [ ] medium粒度
  - [ ] fine粒度

- [ ] `build_belief_group()` - Belief分组
  - [ ] 按uid预分组
  - [ ] 按belief_hash聚类
  - [ ] 分组统计

- [ ] `step_norm_reward_by_belief()` - Step奖励归一化
  - [ ] 组内mean计算
  - [ ] 组内std计算
  - [ ] 单样本处理

- [ ] `compute_rebel_advantage()` - 主函数
  - [ ] Episode优势
  - [ ] Step优势
  - [ ] 加权组合

### 7.2 奖励组件

- [ ] `consistency_reward()` - 一致性奖励
  - [ ] found_objects检查
  - [ ] state_changes检查
  - [ ] inventory检查

- [ ] `progress_reward()` - 进度奖励
  - [ ] subgoal完成检测
  - [ ] evidence评估
  - [ ] subgoal更新检测

- [ ] `exploration_reward()` - 探索奖励
  - [ ] 新位置发现
  - [ ] 重复惩罚

- [ ] `format_reward()` - 格式奖励 (RLVMR风格惩罚)
  - [ ] 中文字符检查
  - [ ] action标签数量检查 (恰好1个)
  - [ ] belief标签数量检查 (恰好1个)
  - [ ] reasoning标签数量检查 (恰好1个)
  - [ ] 标签内容非空检查
  - [ ] 标签顺序检查 (belief → reasoning → action)
  - [ ] belief JSON有效性检查
  - [ ] action可用性检查

### 7.3 改进功能 (可选)

- [ ] `planning_quality_reward()` - 规划质量奖励
- [ ] `plan_alignment_reward()` - 规划对齐奖励
- [ ] `delayed_planning_evaluation()` - 延迟规划评估
- [ ] `adaptive_granularity()` - 动态粒度调整

### 7.4 集成测试

- [ ] 单元测试: 各组件独立测试
- [ ] 集成测试: 完整流程测试
- [ ] 性能测试: 分组效率测试
- [ ] 对比测试: vs GRPO/GiGPO/RLVMR

---

## 8. 配置参数

```yaml
algorithm:
  adv_estimator: rebel

  rebel:
    enable: true
    belief_granularity: 'subgoal'     # subgoal | medium | fine
    step_advantage_w: 1.0             # Step优势权重
    mode: 'mean_norm'                 # mean_norm | mean_std_norm

    # 奖励权重
    reward_weights:
      consistency: 0.3
      progress: 0.5
      exploration: 0.2
      format: 0.1
      # 改进版
      planning: 0.0          # 设为0.1启用规划奖励
      plan_alignment: 0.0    # 设为0.1启用对齐奖励
```

---

## 9. 调试与监控

### 9.1 关键指标

```python
# 训练时打印
print(f"""
ReBel Statistics:
├─ Num Groups: {group_stats['num_groups']}
├─ Mean Group Size: {group_stats['mean_group_size']:.2f}
├─ Median Group Size: {group_stats['median_group_size']:.2f}
├─ Episode Adv Mean: {episode_advantages.mean():.4f}
├─ Step Adv Mean: {step_advantages.mean():.4f}
└─ Total Adv Mean: {total_advantages.mean():.4f}
""")
```

### 9.2 健康检查

| 指标 | 正常范围 | 异常处理 |
|------|----------|----------|
| num_groups | 20-100 | 太少调细粒度，太多调粗粒度 |
| mean_group_size | 10-50 | 太小归一化无效，太大方差太大 |
| step_adv / episode_adv | 0.5-2.0 | 调整step_advantage_w |

---

**文档版本**: v1.0
**最后更新**: 2025-12-26
**适用代码**: `code/rebel/core_rebel.py`
