# ReBel 评测结果分析报告

## 评测结果概览

- **成功率**: 0% (0/4 环境)
- **平均步数**: 30 (全部达到最大步数限制)
- **平均奖励**: -0.15
- **ReBel 指标**:
  - 一致性奖励: 0.0
  - 进度奖励: 0.0
  - 探索奖励: 0.0
  - 格式奖励: -0.05
  - 信念解析率: 0% (所有信念状态都是 null)

## 核心问题分析

### 问题1: 初始信念状态未从环境初始观测初始化 ⚠️

**现状**:
```python
# env_manager.py:92-97
self.cumulative_beliefs[i] = {
    'world_model': {},  # 空字典
    'task_progress': {'subgoal': None, 'status': 'in_progress'},
    'exploration_map': {'visited': set(), 'unexplored': set()}
}
```

**问题**:
- 模型在第一步收到的信念状态是空的 ("None", "None", "None")
- 但环境的初始观测包含了大量有用信息（房间布局、可见物体等）
- 模型应该从初始观测中提取信息来初始化信念状态

**影响**:
- 模型从空白状态开始推理，无法利用初始观测信息
- 导致第一步的输出质量差，影响后续所有步骤

### 问题2: 缺少完整的信念状态记录链 ⚠️

**当前实现**:
- 只保存了累积的信念状态 (`cumulative_beliefs`)
- 只保存了真实状态 (`ground_truth_trackers`)
- 但没有保存每一步的完整序列

**缺失内容**:
1. **真实信念状态更新链**: 每一步环境的真实状态变化
   - `step_0_gt`, `step_1_gt`, ..., `step_n_gt`

2. **模型预测的信念状态更新链**: 每一步模型输出的信念状态
   - `step_0_pred`, `step_1_pred`, ..., `step_n_pred`

3. **信念状态对比**: 真实 vs 预测的差异分析
   - 用于训练时的监督信号
   - 用于评估时的可解释性分析

### 问题3: 模型输出格式不一致 ⚠️

**从轨迹数据观察到**:
```json
// 有些输出有 ```json 标记
{"step": 0, "env_id": 0, "action": "```json\n{\n  \"world_model_update\": ...

// 有些输出没有标记
{"step": 0, "env_id": 1, "action": "{\n  \"belief\": {\n  \"world_model_update\": ...

// 有些输出格式混乱
{"step": 0, "env_id": 2, "action": "{\n  \"belief\": {...}\n  \"reasoning\": ...
```

**问题**:
- Base 模型（Qwen2.5-1.5B-Instruct）没有针对 ReBel 格式进行微调
- 输出格式五花八门，解析成功率为 0%
- 格式奖励全是负值 (-0.05)

### 问题4: 观测信息传递问题 ⚠️

**从轨迹看**:
```
step 0: "I see a hot plate on the countertop."  // 还有一些观测
step 1: "I see no objects or containers."       // 开始出现空观测
step 2: "I see no containers or objects."
...
```

**可能原因**:
1. 环境的 `text_obs[i]` 在后续步骤变为空或无效
2. 或者模型在生成输出时没有正确引用观测内容

### 问题5: 动作格式不统一 ⚠️

**观察到的动作格式**:
```
"action": "go to drawer 1"        // 小写，空格
"action": "GoTo|drawer 1"         // 驼峰，管道符
"action": "GoToCountertop 1"      // 驼峰，无分隔符
"action": "None"                  // 无效动作
"action": ""                      // 空动作
```

**问题**:
- ALFWorld 期望的格式是小写 + 空格，如 "go to drawer 1"
- 但模型输出的格式不一致
- 导致很多动作无法被环境执行

## 改进方案

### 方案1: 初始化信念状态从环境初始观测

**修改位置**: `env_manager.py:reset()` 方法

**改进思路**:
```python
# 在 reset() 中，使用初始观测来初始化信念状态
for i in range(len(text_obs)):
    tracker = GroundTruthTracker()
    tracker.update_from_observation(text_obs[i])
    self.ground_truth_trackers[i] = tracker

    # 从真实状态初始化模型的信念状态
    gt_state = tracker.get_ground_truth_state()
    self.cumulative_beliefs[i] = {
        'world_model': gt_state['object_locations'].copy(),
        'task_progress': {
            'subgoal': None,  # 首个子目标待模型生成
            'status': 'in_progress'
        },
        'exploration_map': {
            'visited': set(gt_state['visited']),
            'unexplored': set()
        }
    }
```

**或者**: 在 prompt 中明确要求模型从初始观测中提取信息来初始化信念状态。

### 方案2: 记录完整的信念状态更新链

**修改位置**: `env_manager.py` 和 `run_rebel_rollout.py`

**新增数据结构**:
```python
# 在 env_manager 中添加
self.belief_history = {}  # env_id -> list of belief states
self.gt_history = {}      # env_id -> list of ground truth states

# 在每一步更新
def step(self, text_actions):
    ...
    for i in range(env_num):
        # 保存模型预测的信念状态
        if belief_state:
            if i not in self.belief_history:
                self.belief_history[i] = []
            self.belief_history[i].append(belief_state)

        # 保存真实状态
        gt_state = self.ground_truth_trackers[i].get_ground_truth_state()
        if i not in self.gt_history:
            self.gt_history[i] = []
        self.gt_history[i].append(gt_state)
```

**在轨迹中记录**:
```python
row = {
    "step": step,
    "env_id": i,
    "observation": text_obs[i],  # 新增：原始观测
    "action": raw_actions[i],
    "reward": float(rewards[i]),
    "done": bool(dones[i]),
    "belief_state_pred": belief_state,      # 模型预测的信念状态
    "belief_state_gt": gt_state,            # 真实的信念状态
    "belief_parsed": belief_state is not None,
    "rebel_rewards": {...}
}
```

### 方案3: 增强 Prompt 来引导模型输出正确格式

**修改位置**: `alfworld_rebel_prompt.py`

**改进思路**:
1. 在初始 prompt 中明确说明如何从观测中提取信息
2. 提供更多示例
3. 强调格式的重要性（与奖励挂钩）

### 方案4: 使用微调后的模型

**问题根源**: Base 模型没有针对 ReBel 格式训练

**解决方案**:
1. 使用已经训练过的 checkpoint (`checkpoints/cold_start/alfworld/bdrs_qwen1.5b_2gpu_20251110/global_step_75`)
2. 或者先进行少量步骤的监督微调，让模型学会输出正确格式

### 方案5: 增加格式检查和重试机制

**修改位置**: `run_rebel_rollout.py:get_action()`

**改进思路**:
```python
def get_action(self, obs, max_retries=3):
    for attempt in range(max_retries):
        response = self.client.chat.completions.create(...)
        action = response.choices[0].message.content.strip()

        # 检查格式
        if self._check_format(action):
            return action
        else:
            # 在 prompt 中添加错误提示
            obs = f"{obs}\n\n[Previous output had incorrect format. Please follow the exact format with <belief>...</belief> and <action>...</action> tags.]"

    return action  # 最后一次尝试的结果
```

## 优先级建议

### 高优先级 (立即修复)
1. ✅ **修复初始信念状态初始化** - 确保模型从有意义的状态开始
2. ✅ **记录完整的信念状态更新链** - 用于后续分析和训练

### 中优先级 (短期改进)
3. **使用微调后的模型** - 或者至少用少量样本微调 base 模型
4. **增强 prompt 引导** - 提供更清晰的格式说明和示例

### 低优先级 (长期优化)
5. **添加格式重试机制** - 提高鲁棒性
6. **改进奖励函数** - 根据实际表现调整权重

## 实验建议

### 实验1: 使用训练后的 checkpoint
```bash
# 修改 start_vllm.sh 使用训练后的模型
MODEL_PATH="./checkpoints/cold_start/alfworld/bdrs_qwen1.5b_2gpu_20251110/global_step_75"
```

预期结果:
- 格式解析率应该显著提高 (>80%)
- 成功率应该 >0%

### 实验2: 初始化信念状态
修改代码后重新评测，观察：
- 第一步的信念状态质量
- 一致性奖励是否变为正值

### 实验3: 记录完整状态链
添加状态链记录后，分析：
- 模型预测的信念状态 vs 真实状态的差异
- 哪些类型的错误最常见
- 信念状态的演化是否合理

## 总结

成功率为 0 的主要原因是：
1. **Base 模型未针对 ReBel 格式训练** - 输出格式混乱
2. **初始信念状态为空** - 模型从无信息状态开始
3. **缺少完整的状态记录链** - 无法追踪和分析信念演化

建议优先：
1. 使用微调后的 checkpoint 进行评测
2. 修复初始信念状态初始化
3. 添加完整的状态记录链

这样可以快速验证 ReBel 框架的有效性。
