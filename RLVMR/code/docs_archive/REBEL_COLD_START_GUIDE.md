# ReBel 冷启动数据生成指南

## 概述

本指南说明如何从专家轨迹生成带有黄金信念状态的 ReBel 冷启动数据。

## 已完成的代码修改

### 1. `env_manager.py` - 信念状态管理改进

**改进 1: 从环境初始观测初始化信念状态**
```python
# 旧代码：空白初始化
self.cumulative_beliefs[i] = {
    'world_model': {},  # 空字典
    'task_progress': {'subgoal': None, 'status': 'in_progress'},
    'exploration_map': {'visited': set(), 'unexplored': set()}
}

# 新代码：从真实环境状态初始化
gt_state = tracker.get_ground_truth_state()
self.cumulative_beliefs[i] = {
    'world_model': gt_state['object_locations'].copy(),  # 包含初始可见物体
    'task_progress': {
        'subgoal': None,
        'status': 'in_progress',
        'task_goal': gt_state['task_goal']  # 包含任务目标
    },
    'exploration_map': {
        'visited': set(gt_state['visited']),  # 包含初始可见位置
        'unexplored': set()
    }
}
```

**改进 2: 记录完整的信念状态历史**
```python
# 新增数据结构
self.belief_history = {}  # env_id -> list of predicted belief states
self.gt_history = {}      # env_id -> list of ground truth states

# 在每一步记录
self.belief_history[i].append(belief_state)  # 模型预测的信念状态
self.gt_history[i].append(ground_truth)      # 真实的环境状态
```

### 2. `run_rebel_rollout.py` - 轨迹记录增强

**新增字段**:
```json
{
  "step": 0,
  "env_id": 0,
  "observation": "...",              // NEW: 原始环境观测
  "action": "...",
  "reward": 0.0,
  "belief_state_pred": {...},        // NEW: 模型预测的信念状态
  "belief_state_gt": {...},          // NEW: 真实的环境状态
  "belief_parsed": true,
  "rebel_rewards": {...}
}
```

### 3. `generate_rebel_cold_start_data.py` - 数据增强脚本

从专家轨迹生成带黄金信念状态的训练数据。

## 使用方法

### 步骤 1: 生成冷启动数据

```bash
# 生成所有数据（2224 条轨迹）
python3 generate_rebel_cold_start_data.py

# 或者先测试少量数据
python3 generate_rebel_cold_start_data.py --limit 100

# 指定输入输出目录
python3 generate_rebel_cold_start_data.py \
    --input_dir data/alfworld_expert_traj \
    --output_dir data/alfworld_rebel_cold_start
```

**输出**:
- `data/alfworld_rebel_cold_start/` - HuggingFace Dataset 格式
- `data/alfworld_rebel_cold_start/rebel_cold_start.jsonl` - JSONL 格式（便于检查）

### 步骤 2: 验证生成的数据

```python
from datasets import load_from_disk

# 加载数据集
ds = load_from_disk("data/alfworld_rebel_cold_start")
print(f"Dataset size: {len(ds)}")

# 查看示例
example = ds[0]
for turn in example['conversations']:
    if turn['from'] == 'gpt' and '<belief>' in turn['value']:
        print(turn['value'])
        break
```

### 步骤 3: 使用冷启动数据进行训练

将生成的数据用于训练配置中（替换原有的专家轨迹）。

## 数据格式说明

### 输入格式（专家轨迹）

```json
{
  "conversations": [
    {"from": "human", "value": "环境观测..."},
    {"from": "gpt", "value": "Thought: 思考...\nAction: 动作"}
  ],
  "item_id": "task_123"
}
```

### 输出格式（ReBel 冷启动数据）

```json
{
  "conversations": [
    {"from": "human", "value": "环境观测..."},
    {
      "from": "gpt",
      "loss": true,
      "value": "<belief>\n{...黄金信念状态...}\n</belief>\n\n<reasoning>\n思考过程\n</reasoning>\n\n<action>\n动作\n</action>"
    }
  ],
  "item_id": "task_123_rebel"
}
```

### 黄金信念状态结构

```json
{
  "world_model_update": {
    "found_objects": {"apple 1": "on countertop 1"},
    "state_changes": {"microwave 1": "open"},
    "cleared_receptacles": ["drawer 1", "cabinet 2"]
  },
  "task_progress_update": {
    "subgoal_status": "in_progress",
    "evidence": "当前观测的证据",
    "updated_subgoal": "下一个子目标"
  },
  "exploration_map_update": {
    "newly_visited": ["countertop 1", "microwave 1"],
    "next_priority": ["fridge 1", "cabinet 1"]
  }
}
```

## 数据特点

### 黄金标准保证

1. **信念状态与环境真实状态一致**
   - 通过 `GroundTruthTracker` 跟踪真实环境状态
   - 确保信念状态反映实际的环境变化

2. **完整的状态演化链**
   - 记录每一步的信念状态更新
   - 可追溯整个任务执行过程

3. **专家级的动作序列**
   - 保留原始专家轨迹的高质量动作
   - 添加符合环境真实状态的信念更新

## 预期改进

使用冷启动数据训练后，模型应该学会：

1. **正确的输出格式**
   - 遵循 `<belief>...</belief><reasoning>...</reasoning><action>...</action>` 格式
   - 格式解析率应该 > 95%

2. **准确的信念状态跟踪**
   - 信念状态与环境真实状态高度一致
   - 一致性奖励显著提高

3. **有效的任务规划**
   - 基于信念状态做出合理的决策
   - 成功率应该显著提升

## 下一步

1. **生成完整的冷启动数据集**
   ```bash
   python3 generate_rebel_cold_start_data.py
   ```

2. **使用数据进行微调**
   - 配置训练脚本使用生成的数据
   - 进行冷启动训练（few-shot 或 full training）

3. **评估改进效果**
   ```bash
   bash start_vllm.sh  # 使用微调后的模型
   bash run_rebel_evaluation.sh
   ```

4. **分析对比**
   - 对比 base 模型 vs 冷启动模型
   - 分析信念状态质量和成功率提升

## 常见问题

**Q: 为什么要生成冷启动数据？**
A: Base 模型不知道 ReBel 的格式，直接使用会导致输出混乱。冷启动数据提供正确格式的示例，让模型快速学会 ReBel 的输出方式。

**Q: 黄金信念状态是如何生成的？**
A: 通过 `GroundTruthTracker` 模拟环境执行，跟踪每一步的真实状态变化。这确保了信念状态与环境完全一致。

**Q: 可以用于其他环境吗？**
A: 脚本目前针对 ALFWorld，但可以轻松扩展到其他环境。只需修改观测解析和状态跟踪逻辑。

**Q: 生成的数据量够吗？**
A: 2224 条专家轨迹应该足够冷启动。如果需要更多，可以使用数据增强（如添加噪声、变换任务等）。

## 总结

通过这套工具链，你可以：
1. ✅ 从环境初始观测初始化信念状态
2. ✅ 记录完整的信念状态演化历史
3. ✅ 生成带黄金信念状态的训练数据
4. ✅ 为冷启动训练提供高质量的监督信号

这将显著提升 ReBel 框架的实际效果！
