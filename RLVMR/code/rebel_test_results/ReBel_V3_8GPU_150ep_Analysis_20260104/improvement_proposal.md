# ReBel V4 改进方案 - 目标: 验证成功率 90%

## 已完成的代码修改

### 1. 新增 `granularity='task_status'` 模式 (推荐)

**文件**: `rebel/core_rebel.py` (lines 77-101)

```python
elif granularity == 'task_status':
    # V4推荐: 基于结构化字段的稳健分组（100%覆盖，无脆弱性）
    task_progress = belief_state.get('task_progress_update', {}) or {}
    world_model = belief_state.get('world_model_update', {}) or {}

    # 1. subgoal_status - 结构化字段
    is_complete = 'complete' in str(task_progress.get('subgoal_status', '')).lower()

    # 2. has_state_change - 物体状态是否已改变
    state_changes = world_model.get('state_changes', {}) or {}
    has_state_change = len(state_changes) > 0

    # 3. has_inventory - 是否持有物体
    inventory = world_model.get('inventory', []) or []
    has_inventory = len(inventory) > 0

    canonical = {
        'is_complete': is_complete,
        'has_state_change': has_state_change,
        'has_inventory': has_inventory
    }
```

**预期效果**:
- Groups/Batch: 40-50% → <15%
- 单样本组: 68.5% → <30%
- 100% 覆盖率，无脆弱性

### 2. 新增 `granularity='state_aware'` 模式 (备选)

**文件**: `rebel/core_rebel.py` (lines 103-127)

- 在 subgoal 基础上加入物体状态类型
- 适用于状态改变任务的细粒度分析

### 3. 增强 `consistency_reward()` (lines 582-659)

- 新增 `task_type` 参数
- 对状态改变任务检查是否记录了正确的状态变化
- 验证物体状态一致性

### 4. 增强 `progress_reward()` (lines 662-738)

- 新增 `task_type` 参数
- 检测新的状态改变 (heated/cooled/cleaned)
- 根据任务类型给予额外奖励

### 5. 更新 `compute_intrinsic_reward()` (lines 847-921)

- 新增 `task_type` 参数传递

---

## 实验脚本

位置: `/root/testttt/RLVMR/code/rebel_test_results/ReBel_V3_8GPU_150ep_Analysis_20260104/`

| 脚本 | 配置 | 目标 |
|------|------|------|
| `run_v4_exp1_task_status.sh` | task_status + task_aware | Groups/Batch <15%, 单样本组 <30% |
| `run_v4_exp2_state_aware.sh` | state_aware + task_aware | 状态改变任务验证 >50% |
| `run_v4_exp3_subgoal_taskaware.sh` | subgoal + task_aware | 对照实验 |

---

## 实验对比设计

| 实验 | granularity | task_aware | 预期改进 |
|------|-------------|------------|----------|
| V3 Baseline | subgoal | false | - |
| Exp 1 (推荐) | **task_status** | **true** | Groups↓, 单样本组↓ |
| Exp 2 | state_aware | true | 状态任务↑ |
| Exp 3 | subgoal | true | 任务均衡性↑ |

---

## 评估指标

每个实验完成后检查:

```bash
# 1. 分组统计
grep "rebel/num_groups" training.log
grep "rebel/single_sample_ratio" training.log

# 2. 验证成功率
grep "val/success_rate" training.log

# 3. 各任务类型性能
grep "pick_heat\|pick_cool\|pick_clean" training.log

# 4. Train-Val Gap
# 计算: episode/success_rate - val/success_rate
```

---

## 运行顺序建议

```
1. 先运行 Exp 1 (task_status + task_aware)
   - 这是最稳健的方案
   - 验证 Groups/Batch 是否降到 <15%

2. 根据 Exp 1 结果决定:
   - 如果 Groups 改善但状态任务仍差 → 运行 Exp 2
   - 如果需要对照 → 运行 Exp 3

3. 综合分析选择最佳配置用于长期训练 (150 epochs)
```

---

## 目标指标

| 指标 | V3 当前值 | V4 目标值 |
|------|-----------|-----------|
| 验证成功率 | 50-70% | **>90%** |
| Groups/Batch | 40-50% | <15% |
| 单样本组比例 | 68.5% | <30% |
| 状态改变任务验证 | 29.7% | >70% |
| Train-Val Gap | 43.8% | <15% |

---

**请确认是否可以运行实验。**
