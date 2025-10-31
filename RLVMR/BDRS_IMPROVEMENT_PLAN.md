# BDRS (Belief-Driven Reward Shaping) 改进方案

## 一、方案概述

基于已实现的基础框架，完善BDRS系统，使其能够通过维护内部信念状态提供**密集的、可解释的奖励信号**，从而显著提升在ALFWorld等长horizon任务中的表现。

## 二、核心改进思路

通过维护内部信念状态（Belief State），从而获得密集的奖励信号：

### 内部信念状态组成
- **M_t (世界模型)**: 对环境事实的信念（对象位置、状态、房间信息）
- **P_t (任务进展)**: 目标/子目标状态追踪
- **E_t (探索地图)**: 访问过的房间、见过的对象

### 三种密集奖励信号

#### 1. 世界一致性奖励 (R_consistency)
**对应问题**: "我的认知与现实世界一致吗？"

**目标**: 激励智能体保持准确的世界模型，与物理观察同步的世界理解

**计算方式**:
```
R_consistency(t) = w_correct * N_corrected_beliefs(t) - w_wrong * N_new_conflicts(t)

其中:
- N_corrected_beliefs: t步修正了多少之前的错误信念
- N_new_conflicts: t步产生了多少新的信念-观察冲突
- w_correct: 修正信念的奖励权重 (建议值: 0.2)
- w_wrong: 产生冲突的惩罚权重 (建议值: 0.1)
```

**示例**:
- M_t 中有 {Microwave_1, HasState, Closed}, 但执行 open microwave 后观测到它已开，信念被修正为 {Microwave_1, HasState, Open}
- 这次修正为获得一次 R_consistency 奖励，这接续两次都应该被视为"纠错"行为

#### 2. 任务进展追踪奖励 (R_progress)
**对应问题**: "我的任务进行到第几步了？"

**目标**: 激励智能体保持护一个清晰的任务规划 P_t，并持续向着完成或完或完成目标的方向，有序逐段的正反馈

**计算方式**:
```
R_progress(t) = w_subgoal * ΔN_completed_subgoals(t)

其中:
- ΔN_completed_subgoals: t步新完成的子目标数量
- w_subgoal: 完成子目标的奖励权重 (建议值: 0.5)
```

**示例**:
- P_t 为 [{'goal': 'find key', 'status': 'completed'}, {'goal': 'unlock door', 'status': 'pending'}]
- 当智能体执行 unlock door 成功后，$P_t[1]$ 中第二个目标的状态变为 completed，此时获得一次性的 R_goal 奖励
- 这次奖励比仅从最终成败获得的反馈要更丰富得多

#### 3. 探索效率奖励 (R_explore)
**对应问题**: "基于我的认知，我下一步该做什么以收集高效地获取新信息？"

**目标**: 激励智能体探索未知同空间对象，收集新信息，而不是在已探索区域重复徘徊

**计算方式**:
```
R_explore(t) = w_entity * ΔN_entities(t) + w_location * ΔN_locations(t)
               - w_revisit * I_revisit(t)

其中:
- ΔN_entities: t步发现的新对象数
- ΔN_locations: t步访问的新房间数
- I_revisit: 是否重复访问已探索区域 (布尔值)
- w_entity: 发现新对象的奖励权重 (建议值: 0.05)
- w_location: 访问新房间的奖励权重 (建议值: 0.1)
- w_revisit: 重复访问的惩罚权重 (建议值: 0.02)
```

**示例**:
- 智能体体验身入一个新房间 living_room，E_t3 得到更新，获得 w_location 奖励
- 它在这房间发现一个之前未见过的 Sofa_1，E_t3 得到更新，获得 w_entity 奖励
- 这鼓励智能体探索"和"组织"的奖励

## 三、现状分析

### ✅ 已完成部分
- `belief_state.py`: BeliefState数据结构和BeliefStateManager基础实现
- `bdrs_rewards.py`: BDRSRewardCalculator基础框架
- `core_bdrs.py`: Advantage计算函数(类似RLVMR)
- `env_manager.py`: 已集成belief state更新和快照传递
- `rollout_loop.py`: 已在rollout后计算BDRS奖励并存入trajectory
- Prompt模板: 已添加BDRS专用模板(PLAN/EXECUTE/EXPLORE/VERIFY)
- 配置文件: 已添加BDRS相关参数

### ⚠️ 需要改进的部分
1. **Belief State更新逻辑**: 当前启发式规则过于简单，需要更准确的状态跟踪
2. **奖励计算逻辑**: 当前是静态快照奖励，需要改为**差分奖励**(t和t-1的对比)
3. **Advantage集成**: 需要完善BDRS rewards如何与PPO的advantage计算融合
4. **数据流**: 需要确保belief快照→奖励→advantage的完整数据流
5. **统计与日志**: 添加BDRS奖励的详细统计和可视化

## 四、详细改进计划

### 阶段1: 增强Belief State跟踪精度

**目标**: 提升世界模型、任务进展、探索地图的准确性

**改进点**:

1. **世界模型 (M_t)**
   - 改进对象位置解析(支持"X in/on Y"模式)
   - 添加对象状态跟踪(clean/dirty, hot/cold, open/closed)
   - 记录历史信念和观察的差异

2. **任务进展 (P_t)**
   - 从任务描述自动解析子目标(如"heat apple and put in fridge"→["heat apple", "put apple in fridge"])
   - 基于动作和观察自动标记子目标完成
   - 支持子目标的依赖关系

3. **探索地图 (E_t)**
   - 差分跟踪: 记录上一步到当前步新发现的房间/对象
   - 添加"revisit"计数，惩罚重复访问
   - 记录每个房间的访问时间戳

**涉及文件**: `code/bdrs/belief_state.py`

---

### 阶段2: 改进奖励计算为差分形式

**目标**: 从静态快照奖励改为基于belief变化的差分奖励

**三种奖励的差分计算** (详见上文)

**关键变化**:
- 需要在计算奖励时访问`prev_belief`和`curr_belief`
- 奖励基于两个belief state的**差异**而非快照本身

**涉及文件**: `code/bdrs/bdrs_rewards.py`

---

### 阶段3: 完善Advantage计算集成

**目标**: 确保BDRS step rewards正确融入PPO的advantage估计

**计算流程**:
```
1. Episode-level reward: 环境最终奖励 (success=1, failure=0)
2. Step-level BDRS reward: 每步的内在奖励 (R_consistency + R_progress + R_explore)
3. Combined advantage:
   A(s,a) = A_episode(s,a) + λ_bdrs * A_step_bdrs(s,a)

   其中 λ_bdrs 由 config.algorithm.bdrs.step_advantage_w 控制
```

**归一化策略**:
- Episode advantages: 按prompt_id分组归一化
- Step BDRS advantages: 按(prompt_id)分组归一化

**涉及文件**: `code/bdrs/core_bdrs.py`

---

### 阶段4: 数据流完整性

**确保以下数据流畅通**:

```
Environment Step (env_manager.py)
  ↓ belief_mgr.step_update(...)
  ↓ belief_mgr.snapshot(...)
  ↓ info['belief'] = {...}
  ↓
Rollout Loop (rollout_loop.py)
  ↓ total_infos[i].append(info)
  ↓ [after rollout完成]
  ↓ BDRSRewardCalculator.step_reward(prev_belief, curr_belief, ...)
  ↓ step['bdrs_step_reward'] = ...
  ↓ step['bdrs_components'] = ...
  ↓
Advantage Calculation (core_bdrs.py)
  ↓ compute_bdrs_outcome_advantage(...)
  ↓ A_episode + λ * A_step_bdrs
  ↓
PPO Update
```

**需要添加**:
- 在trajectory中记录`prev_belief`和`curr_belief`以支持差分计算
- 确保第一步的prev_belief正确处理（使用初始状态）

**涉及文件**: `code/agent_system/multi_turn_rollout/rollout_loop.py`, `code/agent_system/environments/env_manager.py`

---

### 阶段5: 日志、统计与调试

**添加以下内容**:

1. **WandB统计**:
   - `bdrs/world_consistency_{mean,min,max,std}`
   - `bdrs/task_progress_{mean,min,max,std}`
   - `bdrs/exploration_efficiency_{mean,min,max,std}`
   - `bdrs/total_reward_{mean,min,max,std}`
   - `bdrs/belief_update_count`
   - `bdrs/subgoal_completion_rate`
   - `bdrs/new_entities_discovered`
   - `bdrs/new_locations_discovered`

2. **详细轨迹日志** (可选，用于调试):
   - 每个step的belief状态变化
   - 奖励分解(三个分量的具体值)
   - 冲突和修正事件

**涉及文件**: `code/agent_system/multi_turn_rollout/rollout_loop.py`

---

## 五、配置参数说明

```yaml
algorithm:
  adv_estimator: bdrs  # 使用BDRS advantage估计器
  bdrs:
    enable: True

    # 三种奖励的基础权重（用于加权求和）
    world_consistency_weight: 1.0      # R_consistency的权重
    task_progress_weight: 2.0          # R_progress的权重(建议更高)
    exploration_efficiency_weight: 0.5 # R_explore的权重

    # Step advantage在总advantage中的权重
    step_advantage_w: 1.0

    # 归一化模式
    mode: "mean_std_norm"  # 或 "mean_norm"

    # 差分奖励的细粒度系数
    reward_correct_belief: 0.2         # 修正一个错误信念的奖励
    reward_new_conflict: -0.1          # 产生一个新冲突的惩罚
    reward_subgoal_complete: 0.5       # 完成一个子目标的奖励
    reward_new_entity: 0.05            # 发现新对象的奖励
    reward_new_location: 0.1           # 访问新房间的奖励
    penalty_revisit: -0.02             # 重复访问的惩罚
```

## 六、实现优先级

### P0 (必须完成，核心功能)
1. ✅ 改进`BDRSRewardCalculator`为差分奖励计算
2. ✅ 完善`BeliefStateManager`的更新逻辑（对象状态、探索差分）
3. ✅ 确保数据流完整（prev_belief传递）
4. ✅ 集成到`core_bdrs.py`的advantage计算

### P1 (重要，显著提升效果)
5. ✅ 从任务描述自动解析子目标
6. ✅ 添加对象状态跟踪
7. ✅ 探索地图差分跟踪
8. ✅ 添加WandB统计

### P2 (可选，方便调试)
9. 详细轨迹日志
10. Belief state可视化
11. 单元测试

## 七、预期效果

### 对比RLVMR的优势

| 维度 | RLVMR | BDRS (改进后) |
|------|-------|---------------|
| **奖励密度** | 依赖手动标签(`<planning>`, `<explore>`) | 自动从belief state变化计算，更密集 |
| **可解释性** | 标签可解释，但粗粒度 | 三种奖励分量清晰，可追溯到belief变化 |
| **泛化性** | 需要针对不同任务调整标签规则 | Belief state是通用抽象，易迁移 |
| **信号质量** | 标签可能不准确(如错误的planning) | 基于实际状态变化，更可靠 |

### 预期提升
- **ALFWorld成功率**: +5-10%
- **样本效率**: 更快收敛(减少20-30%训练步数)
- **模型行为**: 更系统化的规划和探索，减少无效动作

## 八、测试与验证计划

1. **单元测试**:
   - 测试BeliefStateManager的更新逻辑
   - 测试差分奖励计算的正确性

2. **消融实验**:
   - 对比 episode-only vs episode+BDRS
   - 对比三种奖励权重的影响
   - 对比不同`step_advantage_w`的效果

3. **可视化验证**:
   - 查看成功/失败轨迹的belief state演化
   - 检查奖励分量是否符合预期

## 九、参考文献

- RLVMR论文: 提供了meta-reasoning reward的基础思路
- veRL框架: 提供了RL训练的基础设施
- ALFWorld环境: 具身智能体任务的标准测试环境
