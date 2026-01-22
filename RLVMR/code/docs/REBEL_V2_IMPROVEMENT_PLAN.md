# ReBel V2 改进方案

**分支:** ReBel_v2_improvements
**基于分析:** EXPERIMENT_6_ANALYSIS_CN.md
**目标:** 解决三个核心问题，使模型在整个ALFWorld六类任务上持续改善

---

## 问题回顾

| 问题 | 现象 | 根因 |
|------|------|------|
| **P1** | Epoch 30后奖励平台期 | 策略熵坍缩、优势饱和、梯度干扰 |
| **P2** | Epoch 70后成功率下降19.4% | 过拟合、灾难性遗忘、任务干扰 |
| **P3** | 六类任务相互冲突 | 单策略容量不足、跨任务信念混合 |

**核心目标:** 让模型在整个训练过程中持续改善，而非依赖早停

---

## 改进方案总览

### 方案优先级

| 优先级 | 改进项 | 解决问题 | 实现难度 | 预期收益 |
|--------|--------|----------|----------|----------|
| **P0** | 任务感知信念分组 | P3 | 中 | **高** (核心改进) |
| **P1** | 按任务归一化优势 | P3 | 中 | 高 |
| **P2** | 增大KL惩罚 | P1, P2 | 低 | 中 |
| **P3** | 增加熵正则 | P1 | 低 | 中 |
| **P4** | 自适应步级权重 | P1 | 中 | 中 |
| **P5** | 信念粒度调优 | P1, P3 | 低 | 中 |

---

## 详细改进方案

### 1. 任务感知信念分组 (P0) ⭐核心改进

**问题:** 当前信念分组不区分任务类型，导致不同任务的状态被错误分组

**目标:** 只在相同任务类型内进行信念分组，防止跨任务干扰

**实现位置:** `rebel/core_rebel.py` - `build_belief_group()`

**当前逻辑:**
```python
group_key = f"belief_{uid}_{belief_hash}"  # 只按uid和belief分组
```

**改进逻辑:**
```python
group_key = f"belief_{task_type}_{uid}_{belief_hash}"  # 加入task_type
```

**配置参数:**
```yaml
algorithm.rebel.task_aware_grouping: True  # 新增参数
```

**预期效果:**
- 每个任务类型独立学习，减少梯度干扰
- 相同任务的相似状态才会被分到一组
- 预期成功率提升5-10%

---

### 2. 按任务归一化优势 (P1)

**问题:** 某些任务(如pick_and_place)学习快，主导梯度更新，抑制其他任务

**目标:** 分别对每个任务类型的优势进行归一化，平衡各任务的学习

**实现位置:** `rebel/core_rebel.py` - `compute_rebel_advantage()`

**改进逻辑:**
```python
def normalize_advantages_per_task(advantages, task_types, eos_mask):
    """对每个任务类型独立归一化，防止主导任务压倒其他"""
    result = torch.zeros_like(advantages)
    for task in set(task_types):
        mask = np.array([t == task for t in task_types])
        if mask.sum() > 1:
            task_adv = advantages[mask]
            # 只对有效token归一化
            valid_adv = task_adv[eos_mask[mask] > 0]
            if len(valid_adv) > 0:
                mean = valid_adv.mean()
                std = valid_adv.std() + 1e-8
                result[mask] = (task_adv - mean) / std
        else:
            result[mask] = advantages[mask]
    return result
```

**配置参数:**
```yaml
algorithm.rebel.per_task_normalization: True  # 新增参数
```

**预期效果:**
- 各任务获得均衡的梯度信号
- 慢学习任务不会被快学习任务压制

---

### 3. KL惩罚调整 (P2)

**问题:** 当前KL系数(0.01)可能不足以防止后期策略漂移

**目标:** 增大KL惩罚，保持策略稳定性

**当前值:** `actor.kl_loss_coef = 0.01`

**实验范围:**
```
kl_loss_coef ∈ [0.01, 0.02, 0.05]
```

**配置:**
```yaml
actor_rollout_ref.actor.kl_loss_coef: 0.02  # 温和增加
```

**预期效果:**
- 减少后期性能波动
- 防止策略从已学技能上"遗忘"

---

### 4. 熵正则化 (P3)

**问题:** 策略熵坍缩导致探索不足，陷入局部最优

**目标:** 维持策略多样性，延迟奖励平台期

**当前值:** `actor.entropy_coeff = 0.001`

**实验范围:**
```
entropy_coeff ∈ [0.001, 0.005, 0.01]
```

**配置:**
```yaml
actor_rollout_ref.actor.entropy_coeff: 0.005  # 增加5倍
```

**预期效果:**
- 保持探索能力
- 防止过早收敛到局部最优

---

### 5. 自适应步级权重 (P4)

**问题:** 固定的step_advantage_w可能不适合所有训练阶段

**目标:** 根据训练进度动态调整权重

**实现逻辑:**
```python
def get_adaptive_step_weight(epoch, total_epochs, base_weight=0.5):
    """
    训练进度适应:
    - 早期(0-30%): 更依赖episode信号 → 权重较低
    - 中期(30-70%): 平衡
    - 后期(70-100%): 更依赖step信号 → 权重较高
    """
    progress = epoch / total_epochs
    if progress < 0.3:
        return base_weight * 0.6  # 0.3
    elif progress < 0.7:
        return base_weight  # 0.5
    else:
        return base_weight * 1.4  # 0.7
```

**配置参数:**
```yaml
algorithm.rebel.adaptive_step_weight: True
algorithm.rebel.step_advantage_w: 0.5  # base weight
```

---

### 6. 信念粒度调优 (P5)

**问题:** 当前subgoal粒度可能过粗，无法区分细节差异

**目标:** 找到最优粒度

**实验范围:**
```
belief_granularity ∈ ['subgoal', 'medium', 'fine']
```

**预期:**
- `subgoal`: 20-50组，稳定但可能过粗
- `medium`: 50-150组，平衡选择
- `fine`: 100-500组，细粒度但可能过于稀疏

---

## 实验计划

### 第一轮: 单因素实验 (5个实验)

验证每个改进的单独效果

| 实验ID | 改进项 | 关键配置 | 对比基线 |
|--------|--------|----------|----------|
| **exp1_task_aware** | 任务感知分组 | `task_aware_grouping=True` | Exp6基线 |
| **exp2_per_task_norm** | 按任务归一化 | `per_task_normalization=True` | Exp6基线 |
| **exp3_kl_0.02** | KL=0.02 | `kl_loss_coef=0.02` | Exp6基线 |
| **exp4_entropy_0.005** | 熵=0.005 | `entropy_coeff=0.005` | Exp6基线 |
| **exp5_granularity_medium** | 中等粒度 | `belief_granularity='medium'` | Exp6基线 |

### 第二轮: 组合实验 (3-4个实验)

组合第一轮中有效的改进

| 实验ID | 组合 | 配置 |
|--------|------|------|
| **exp6_combo_a** | 任务感知 + 按任务归一化 | 核心改进组合 |
| **exp7_combo_b** | exp6 + KL调整 | 加入稳定性 |
| **exp8_combo_c** | exp7 + 熵正则 | 加入探索 |
| **exp9_full** | 全部有效改进 | 最优配置 |

### 第三轮: 精调 (2-3个实验)

基于最优组合进行参数精调

---

## 实验配置模板

### 基础配置 (保持不变)
```yaml
# 数据
data.train_batch_size: 16
data.val_batch_size: 128

# 模型
actor_rollout_ref.model.path: ${SFT_MODEL_PATH}
actor_rollout_ref.actor.optim.lr: 1e-6

# 环境
env.max_steps: 30
env.rollout.n: 16

# 训练
trainer.total_epochs: 100
trainer.test_freq: 5
trainer.save_freq: 10
```

### 实验变量定义

```bash
# =============================================================
# 第一轮实验配置
# =============================================================

# Exp1: 任务感知分组 (核心)
EXP1_CONFIG=(
    "algorithm.rebel.task_aware_grouping=True"
)

# Exp2: 按任务归一化优势
EXP2_CONFIG=(
    "algorithm.rebel.per_task_normalization=True"
)

# Exp3: KL惩罚调整
EXP3_CONFIG=(
    "actor_rollout_ref.actor.kl_loss_coef=0.02"
)

# Exp4: 熵正则化
EXP4_CONFIG=(
    "actor_rollout_ref.actor.entropy_coeff=0.005"
)

# Exp5: 信念粒度
EXP5_CONFIG=(
    "algorithm.rebel.belief_granularity='medium'"
)

# =============================================================
# 第二轮组合实验配置
# =============================================================

# Exp6: 核心组合 (任务感知 + 按任务归一化)
EXP6_CONFIG=(
    "algorithm.rebel.task_aware_grouping=True"
    "algorithm.rebel.per_task_normalization=True"
)

# Exp7: 核心 + KL
EXP7_CONFIG=(
    "algorithm.rebel.task_aware_grouping=True"
    "algorithm.rebel.per_task_normalization=True"
    "actor_rollout_ref.actor.kl_loss_coef=0.02"
)

# Exp8: 核心 + KL + 熵
EXP8_CONFIG=(
    "algorithm.rebel.task_aware_grouping=True"
    "algorithm.rebel.per_task_normalization=True"
    "actor_rollout_ref.actor.kl_loss_coef=0.02"
    "actor_rollout_ref.actor.entropy_coeff=0.005"
)
```

---

## 代码修改清单

### 需要修改的文件

| 文件 | 修改内容 | 优先级 |
|------|----------|--------|
| `rebel/core_rebel.py` | 任务感知分组、按任务归一化 | **P0** |
| `verl/trainer/config/ppo_trainer.yaml` | 新增配置参数 | P0 |
| `verl/trainer/ppo/ray_trainer.py` | 传递task_types、调用新函数 | P0 |
| `run_rebel_hyperparam_search.sh` | V2实验配置 | P1 |

### 核心代码修改

#### 1. `rebel/core_rebel.py` 修改

```python
# build_belief_group() 新增参数
def build_belief_group(
    belief_states: np.ndarray,
    index: np.ndarray,
    granularity: str = 'subgoal',
    summarize: bool = False,
    task_types: Optional[np.ndarray] = None,  # 新增
    task_aware: bool = False  # 新增
) -> Tuple[np.ndarray, Dict[str, Any]]:
    ...
    # 如果启用任务感知
    if task_aware and task_types is not None:
        cluster_key = (task_type, belief_hash)
    else:
        cluster_key = belief_hash
    ...

# compute_rebel_advantage() 新增参数
def compute_rebel_advantage(
    ...
    task_types: Optional[np.ndarray] = None,  # 新增
    task_aware: bool = False,  # 新增
    per_task_normalization: bool = False  # 新增
) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    ...
```

#### 2. `ppo_trainer.yaml` 新增配置

```yaml
algorithm:
  rebel:
    enable: False
    belief_granularity: "subgoal"
    step_advantage_w: 1.0
    mode: "mean_norm"
    summarize_groups: False
    # V2新增
    task_aware_grouping: False      # 任务感知分组
    per_task_normalization: False   # 按任务归一化
    adaptive_step_weight: False     # 自适应权重
```

---

## 成功标准

| 指标 | 基线(Exp6) | 目标 | 说明 |
|------|-----------|------|------|
| 峰值验证成功率 | 72.7% | **78%+** | 提升5%+ |
| **最终验证成功率** | 58.6% | **75%+** | **关键指标** |
| 训练成功率持续性 | 下降19.4% | **<5%下降** | 防止后期退化 |
| 任务成功率方差 | 高 | 降低50% | 各任务均衡 |
| 最差任务成功率 | ~40% | **60%+** | 提升短板 |

---

## 实验监控指标

### 核心指标
- `val/success_rate`: 总体验证成功率
- `val/{task_type}_success_rate`: 各任务验证成功率
- `episode/success_rate`: 训练成功率

### 诊断指标
- `rebel_num_groups`: 信念组数量
- `rebel_mean_group_size`: 平均组大小
- `actor/entropy`: 策略熵（监控探索）
- `actor/kl_loss`: KL散度（监控稳定性）

### 任务平衡指标
- 各任务成功率标准差（应降低）
- 最高/最低任务成功率比值（应接近1）

---

## 预期结果

### 最优配置预测

基于分析，预期最优配置:

```yaml
algorithm.rebel:
  enable: True
  belief_granularity: 'subgoal'  # 或 'medium'
  step_advantage_w: 0.5
  mode: 'mean_norm'
  task_aware_grouping: True       # ⭐核心
  per_task_normalization: True    # ⭐核心

actor_rollout_ref.actor:
  kl_loss_coef: 0.02             # 温和增加
  entropy_coeff: 0.005           # 增加探索
```

### 改进效果预期

| 改进项 | 预期提升 | 作用机制 |
|--------|----------|----------|
| 任务感知分组 | +5-8% | 消除跨任务干扰 |
| 按任务归一化 | +3-5% | 平衡任务学习 |
| KL调整 | +2-3% | 稳定后期训练 |
| 熵正则 | +1-2% | 延迟平台期 |
| **组合效果** | **+10-15%** | 协同增强 |

---

## 附录: 关键代码位置

```
rebel/core_rebel.py
├── canonicalize_belief()        # L29-96:  信念规范化
├── build_belief_group()         # L102-169: 信念分组 ← 添加task_aware
├── episode_norm_reward()        # L176-230: Episode优势
├── step_norm_reward_by_belief() # L237-297: Step优势
├── compute_rebel_advantage()    # L304-387: 总优势计算 ← 添加per_task_norm
└── [新增] normalize_per_task()  # 按任务归一化函数

verl/trainer/ppo/ray_trainer.py
├── compute_advantage()          # L308-350: 优势计算集成 ← 传递task_types
└── [修改] 从meta_info提取task_types

agent_system/environments/env_manager.py
└── step()                       # 确保task_type在info中返回
```

---

*文档更新: 2026-01-01*
*分支: ReBel_v2_improvements*
*核心理念: 解决根本问题，让整个ALFWorld持续改善*
