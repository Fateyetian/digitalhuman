# WebShop ReBel 奖励设计方案

> 文件来源：`agent_system/environments/env_package/webshop/envs.py`、
> `agent_system/environments/env_manager.py`、
> `agent_system/environments/env_package/webshop/belief_tracker.py`、
> `rebel/core_rebel.py`

---

## 一、整体架构

```
总奖励（每步）= 环境外在奖励（Result Reward）+ 内在奖励（Intrinsic / Belief Reward）
```

两部分可通过训练脚本开关独立控制：

| 开关 | 作用 |
|------|------|
| `USE_RESULT_REWARD=true` | 启用环境外在奖励（默认开） |
| `USE_BELIEF_REWARD=true` | 启用信念内在奖励（默认开） |
| `USE_BELIEF_DECAY=true` | 启用内在奖励权重衰减（M5 实验开） |

---

## 二、环境外在奖励（Result Reward）

**来源**：`envs.py`，由 WebShop 环境的 `get_reward()` 直接计算。

### 计算逻辑

```python
WIN_THRESHOLD = 1.0 - 1e-6          # GiGPO 标准：完美匹配才算成功

if done:                              # 仅终止步（agent 点击 Buy Now）给出奖励
    info['won'] = bool(reward >= WIN_THRESHOLD)
    reward = reward * 10.0            # 缩放 [0,1] → [0,10]
else:
    info['won'] = False
    reward = 0.0                      # 非终止步奖励为 0（稀疏）
```

### 奖励构成

`get_reward()` 返回 `[0,1]` 的连续分数，计算规则：

```
total_reward = (num_attr_matches + num_option_matches + r_price)
               / (len(attributes) + len(goal_options) + 1)
             × r_type
```

| 子项 | 说明 |
|------|------|
| `r_type` | 商品类型匹配系数（0 或 1） |
| `num_attr_matches` | 匹配的属性数量（颜色、材质等） |
| `num_option_matches` | 匹配的选项数量（尺码、颜色按钮等） |
| `r_price` | 价格在上限以内（0 或 1） |

### 尺度

| 情形 | 原始 score | 缩放后 reward |
|------|-----------|-------------|
| 完美匹配（所有属性+选项+价格全对） | 1.0 | **10.0** |
| 部分匹配（典型情况） | 0.3~0.8 | 3.0~8.0 |
| 完全不匹配 | 0.0 | 0.0 |
| 非终止步 | — | **0.0** |

> **设计说明**：原 GiGPO 使用二值奖励（成功=10，失败=0），但 score==1.0 极难触发，
> 导致训练信号为 0。改为连续奖励（score×10）保留部分信用，
> 同时 `won`（success_rate）仍用 score==1.0 的严格标准与 GiGPO 公平对比。

---

## 三、内在奖励（Intrinsic / Belief Reward）

**来源**：`belief_tracker.py`，`WebShopRebelRewardCalculator`，**每步**计算。

### 合成公式

```
r_intrinsic = α × r_consistency  +  β × r_progress  +  γ × r_exploration  +  δ × r_format
            = 0.3 × r_c          +  0.5 × r_p        +  0.2 × r_e          +  0.1 × r_f
```

---

### 3.1 一致性奖励（Consistency）

评估 `product_understanding` 字段的准确性。

| 子项 | 权重 | 原始分值范围 |
|------|------|------------|
| `target_attributes` 合理性 | 35% | [0.3, 1.0] |
| `current_product_match` 准确性 | 25% | 0.4 / 0.6 / 0.8 |
| `price_constraint` 感知 | 25% | [0.3, 0.9] |
| `attribute_verification` 质量 | 15% | [0.0, 1.0] |

```
r_c_raw = 0.35×c1 + 0.25×c2 + 0.25×c3 + 0.15×c4   ∈ [0, 1]
r_c     = r_c_raw × scale(0.1)                       ∈ [0, 0.1]
```

**加入合成公式后**：`0.3 × r_c ∈ [0, 0.03]`

---

### 3.2 进度奖励（Progress）

评估 `search_progress` 字段的准确性。

| 子项 | 权重 | 原始分值范围 |
|------|------|------------|
| `search_status` 准确性 | 40% | 0.3 / 0.7 / 1.0 |
| `evidence` 质量（长度+内容） | 30% | [0.0, 1.0] |
| `updated_subgoal` 逻辑 | 30% | [0.5, 1.0] |

```
r_p_raw = 0.4×s1 + 0.3×s2 + 0.3×s3   ∈ [0, 1]
r_p     = r_p_raw × scale(0.1)         ∈ [0, 0.1]
```

**加入合成公式后**：`0.5 × r_p ∈ [0, 0.05]`（权重最高，进度信号最重要）

---

### 3.3 探索奖励（Exploration）

评估 `exploration_state` 字段的准确性。

| 子项 | 权重 | 原始分值范围 |
|------|------|------------|
| `queries_tried` 准确性 | 40% | [0.0, 1.0] |
| `products_viewed` 追踪 | 30% | [0.0, 1.0] |
| `options_selected` 追踪 | 30% | [0.0, 1.0] |

```
r_e_raw = 0.4×q + 0.3×p + 0.3×o   ∈ [0, 1]
r_e     = r_e_raw × scale(0.05)    ∈ [0, 0.05]
```

**加入合成公式后**：`0.2 × r_e ∈ [0, 0.01]`（尺度最小）

---

### 3.4 格式奖励（Format）

| 情形 | `r_f` |
|------|-------|
| 格式有效，动作合法 | `+0.01` |
| 格式有效，但动作不可用 | `+0.01 − 0.02 = −0.01` |
| 格式无效（标签缺失/JSON 损坏） | `−0.05` |

**加入合成公式后**：`0.1 × r_f ∈ [−0.005, +0.001]`

---

### 3.5 附加惩罚：状态转移违规

若 `search_status = ready_to_buy` 但 `inferred_only` 非空（尚未充分核实属性就准备下单），
每个未核实属性额外扣分：

```
transition_penalty = −0.02 × len(inferred_only_attrs)
```

---

### 3.6 总内在奖励范围汇总

| 情形 | 近似值 |
|------|--------|
| 最优（格式完美，信念完全准确） | ≈ `0.03 + 0.05 + 0.01 + 0.001` ≈ **+0.091** |
| 典型（部分准确） | ≈ **+0.03 ~ +0.06** |
| 格式无效（信念解析失败） | `0.1 × (−0.05)` = **−0.005** |

---

## 四、奖励量级对比（关键）

```
环境外在奖励（done 时）:  [0,    10.0]    ← 绝对主导，约为内在奖励 100~300 倍
内在奖励（每步典型值）:   [−0.01, +0.09]  ← 每步均有，稀疏信号的补充
格式惩罚（解析失败时）:   −0.005          ← 极小，几乎可忽略
```

**每条轨迹的奖励积累（以 15 步为例）**：

| 场景 | 外在贡献 | 内在贡献（15步累计） | 合计 |
|------|---------|-------------------|------|
| 完美购买（score=1.0） | 10.0 | ~0.75 | ~10.75 |
| 部分匹配（score=0.5） | 5.0 | ~0.75 | ~5.75 |
| 失败但信念好（score=0） | 0.0 | ~0.75 | ~0.75 |
| 失败且信念差（score=0） | 0.0 | ~0.15 | ~0.15 |

> 内在奖励的作用是区分"失败轨迹的质量"（0.15 vs 0.75），
> 而非主导成功/失败判断（差距仍为 10:0.75 ≈ 13:1）。

---

## 五、信念奖励权重衰减（Belief Reward Decay）

启用条件：`USE_BELIEF_DECAY=true`（M5 实验配置）

权重 `w ∈ [min_weight, 1.0]` 随训练进度调整：

| 阶段 | epoch 范围 | 权重 `w` |
|------|-----------|---------|
| Warmup（线性升） | `[0, 5)` | `0 → 1.0` |
| 稳定 | `[5, 10)` | `1.0` |
| 衰减（cosine） | `[10, 60]` | `1.0 → 0.05` |
| 自适应（V11） | 随 success_rate 变化 | `max(0.05, 1 − (sr/0.90)²)` |

**差分衰减速率**（不同组件衰减快慢不同）：

| 组件 | decay_rate | 含义 |
|------|-----------|------|
| 探索（Exploration） | 2.0 | 最快衰减，训练后期不鼓励探索追踪 |
| 一致性（Consistency） | 1.0 | 标准速率 |
| 进度（Progress） | 0.7 | 最慢衰减，全程保持进度信号 |

---

## 六、ReBel 双层优势估计

内在奖励通过 Step 级优势影响梯度，与 Episode 级优势独立归一化后合并：

```
A_total = A_episode  +  λ × A_step
            ↑                ↑
   按 uid 分组归一化        按 (uid, belief_hash) 分组归一化
   基于累计 episode_reward   基于单步 rebel_intrinsic_reward
   λ = step_advantage_w = 0.5
```

### 为何绝对量级差异不影响优势计算

| 项目 | 原始范围 | 归一化后 |
|------|---------|--------|
| `A_episode` 输入（episode_reward） | [0.5, 11.0] | 均值≈0，组内相对差异 |
| `A_step` 输入（intrinsic_reward） | [0, 0.09] | 均值≈0，组内相对差异 |

两路各自独立归一化后量级相当，`λ=0.5` 是真实的相对权重。

### episode_reward 的轻微污染

由于内在奖励被**叠加**进 `episode_reward`（用于 A_episode 的输入）：

```python
rewards[i] = env_reward[i] + intrinsic_reward[i]   # 每步叠加
episode_reward = Σ rewards[i]
```

失败轨迹之间因信念质量差异导致 episode_reward 相差约 **5 倍**（0.15 vs 0.75），
但成功/失败之间的差距约 **13 倍**（10.75 vs 0.75），信念奖励不会颠覆方向。

---

## 七、Success Rate 指标说明（重要）

| 指标名 | 定义 | 用途 |
|--------|------|------|
| `val/success_rate` | score ≥ 1.0（严格，**GiGPO 标准**） | 论文对比指标 |
| `val/webshop_score_ge_0.5` | score ≥ 0.5（宽松） | 辅助诊断 |
| `val/webshop_score_ge_0.8` | score ≥ 0.8 | 辅助诊断 |
| `val/webshop_task_score` | 连续分均值（0~1） | 学习曲线参考 |
| `val/webshop_bought` | 点击了 Buy Now 的比例 | 行为分析 |

> **历史问题**：旧版代码使用 score ≥ 0.5 作为 success_rate，
> 已于本次修复中改为 score == 1.0（对齐 GiGPO）。
> 旧实验报告的 success_rate ≈ 0.6 对应新标准下约 0.1~0.2。
