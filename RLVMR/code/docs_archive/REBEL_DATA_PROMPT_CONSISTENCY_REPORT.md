# ReBel数据格式与Prompt一致性分析报告

## 📋 执行摘要

**检查日期**: 2025-12-24
**数据来源**: `/root/testttt/RLVMR/code/data/alfworld_rebel_250_new/rebel_coldstart.json`
**核心发现**: ⚠️ **Cold-start数据格式与SFT Prompt存在不一致**

---

## 🔍 一、数据格式一致性问题

### 1.1 问题概述

Cold-start数据应该**删除Available Actions**以提高训练难度，但实际数据：
- ✅ **第一步（Step 1）**：包含完整的Available Actions列表
- ✅ **后续步骤（Steps 2+）**：显示"Available Actions: N/A"

### 1.2 实际数据检查结果

**第一步数据（Line 9）**:
```
Available Actions: go to bed 1, go to bed 2, go to diningtable 1, go to drawer 1, go to drawer 2, go to garbagecan 1, go to shelf 1, go to shelf 10, go to shelf 11, go to shelf 12, go to shelf 13, go to shelf 14, go to shelf 2, go to shelf 3, go to shelf 4, go to shelf 5, go to shelf 6, go to shelf 7, go to shelf 8, go to shelf 9, go to sidetable 1, inventory, look
```

**后续步骤数据（Line 15, 21, 27...）**:
```
Available Actions: N/A
```

### 1.3 预期的数据格式

根据`generate_rebel_hindsight.py`的代码逻辑，Cold-start数据应该：

**第一步（Line 497-499）**:
```python
if step_num == 1:
    prompt = ALFWORLD_REBEL_TEMPLATE_NO_HIS_CS.format(
        current_observation=obs
    )
```
- 使用`ALFWORLD_REBEL_TEMPLATE_NO_HIS_CS`模板
- 该模板**不包含**`{admissible_actions}`字段
- 应该完全没有Available Actions信息

**后续步骤（Line 507-511）**:
```python
# CRITICAL: Remove "Available Actions" line for cold-start
prompt_lines = human_value.split('\n')
filtered_lines = [line for line in prompt_lines if 'Available Actions:' not in line]
prompt = '\n'.join(filtered_lines)
```
- 应该从原始prompt中**删除**"Available Actions:"行
- 不应该有"Available Actions: N/A"

---

## 📊 二、与各阶段Prompt的一致性对比

| 阶段 | Prompt模板 | Available Actions（期望） | Available Actions（实际数据） | 一致性 |
|------|-----------|-------------------------|--------------------------|--------|
| **数据标注** | `ALFWORLD_REBEL_TAGGING_TEMPLATE` | ✅ 有（辅助Teacher LLM） | N/A（仅用于标注） | N/A |
| **Cold-start SFT（第一步）** | `ALFWORLD_REBEL_TEMPLATE_NO_HIS_CS` | ❌ **应该无** | ⚠️ **实际有完整列表** | ❌ **不一致** |
| **Cold-start SFT（后续步）** | `ALFWORLD_REBEL_TEMPLATE_CS` | ❌ **应该无** | ⚠️ **实际有"N/A"** | ⚠️ **部分一致** |
| **RL训练（第一步）** | `ALFWORLD_REBEL_TEMPLATE_NO_HIS_RL` | ✅ 应该有 | N/A（RL阶段动态生成） | ✅ 一致 |
| **RL训练（后续步）** | `ALFWORLD_REBEL_TEMPLATE_RL` | ✅ 应该有 | N/A（RL阶段动态生成） | ✅ 一致 |

### 2.1 Cold-start Prompt模板内容

**ALFWORLD_REBEL_TEMPLATE_NO_HIS_CS（Line 77-151）**:
```python
"""
You are an expert agent operating in the ALFRED Embodied Environment.
Your current observation is: {current_observation}

Now it's your turn to take an action, following these steps:
...
"""
```
- ❌ **没有**`{admissible_actions}`参数
- ❌ **没有**任何Available Actions字段

**ALFWORLD_REBEL_TEMPLATE_CS（Line 153-235）**:
```python
"""
You are an expert agent operating in the ALFRED Embodied Environment.
Your task is to: {task_description}

Prior to this step, you have already taken {step_count} step(s).
Below are the most recent {history_length} observations and actions: {action_history}

You are now at step {current_step} and your current observation is: {current_observation}
...
"""
```
- ❌ **没有**`{admissible_actions}`参数
- ❌ **没有**任何Available Actions字段

### 2.2 RL训练Prompt模板内容

**ALFWORLD_REBEL_TEMPLATE_NO_HIS_RL（Line 242-312）**:
```python
"""
You are an expert agent operating in the ALFRED Embodied Environment.
Your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].
...
"""
```
- ✅ **包含**`{admissible_actions}`参数
- ✅ **明确显示**Available Actions列表

---

## 🐛 三、问题根源分析

### 3.1 代码逻辑分析

**数据生成代码**: `generate_rebel_hindsight.py:Line 493-523`

```python
# IMPORTANT: Reconstruct prompt WITHOUT admissible actions
# This is the key difference from RL training
if step_num == 1:
    # First step: use NO_HIS_CS template (no history, no admissible actions)
    prompt = ALFWORLD_REBEL_TEMPLATE_NO_HIS_CS.format(
        current_observation=obs
    )
else:
    # Subsequent steps: extract task, history, planning from original prompt
    # Then use CS template (no admissible actions)
    # Extract necessary info from human_value
    task_match = re.search(r'Your task is to: (.+)', human_value)
    task_desc = task_match.group(1) if task_match else task

    # CRITICAL: Remove "Available Actions" line for cold-start
    # Must use strip() because the line might have leading whitespace
    prompt_lines = human_value.split('\n')
    filtered_lines = [line for line in prompt_lines if 'Available Actions:' not in line]
    prompt = '\n'.join(filtered_lines)

response = conversations[i + 1]['value']  # Complete gpt turn as response

# Add to coldstart data
coldstart_data["data"].append({
    "step": step_num,
    "obs": obs,
    "prompt": prompt,  # NO admissible actions!
    "response": response
})
```

### 3.2 问题推测

可能的原因：
1. ❌ **第一步逻辑错误**：
   - 代码中第一步应该使用`ALFWORLD_REBEL_TEMPLATE_NO_HIS_CS.format(current_observation=obs)`
   - 但实际数据显示有Available Actions，说明可能是从hindsight数据直接拷贝的

2. ❌ **后续步骤逻辑不完善**：
   - 代码中删除"Available Actions:"行的逻辑应该工作
   - 但"Available Actions: N/A"可能是**没有完全删除**，而是被替换成了"N/A"

3. ⚠️ **数据版本问题**：
   - 当前数据可能是旧版本，没有应用最新的删除逻辑
   - 需要重新生成Cold-start数据

---

## ✅ 四、解决方案

### 4.1 短期解决方案（使用现有数据）

**选项A**: 接受当前数据格式，修改SFT Prompt
- 修改`ALFWORLD_REBEL_TEMPLATE_NO_HIS_CS`和`ALFWORLD_REBEL_TEMPLATE_CS`
- 添加`{admissible_actions}`字段（但设置为空或"N/A"）
- ⚠️ **不推荐**：违反渐进式课程学习设计

**选项B**: 数据预处理，删除Available Actions
- 写一个脚本，扫描所有cold-start数据
- 删除所有"Available Actions:"行
- ✅ **推荐**：保持设计一致性

### 4.2 长期解决方案（重新生成数据）

**完整流程**:
```bash
# 1. 使用最新的generate_rebel_hindsight.py重新生成
python generate_rebel_hindsight.py \
    --expert_data data/alfworld_expert_traj \
    --output_dir data/alfworld_rebel_regenerated \
    --num_samples 250

# 2. 验证生成的cold-start数据
grep "Available Actions" data/alfworld_rebel_regenerated/rebel_coldstart.json

# 3. 应该没有任何输出（即没有Available Actions）
```

### 4.3 验证脚本

创建验证脚本`verify_coldstart_consistency.py`:
```python
#!/usr/bin/env python3
"""验证Cold-start数据格式是否符合Prompt要求"""

import json

def verify_coldstart_data(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)

    issues = []
    for item_idx, item in enumerate(data):
        for step_idx, step in enumerate(item['data']):
            prompt = step['prompt']

            # 检查是否有Available Actions
            if 'Available Actions:' in prompt:
                issues.append({
                    'item': item_idx,
                    'step': step_idx + 1,
                    'issue': 'Contains "Available Actions:" in prompt'
                })

    if issues:
        print(f"❌ Found {len(issues)} inconsistencies:")
        for issue in issues[:10]:  # Show first 10
            print(f"  - Item {issue['item']}, Step {issue['step']}: {issue['issue']}")
    else:
        print("✅ All prompts are consistent (no Available Actions found)")

    return len(issues) == 0

if __name__ == '__main__':
    verify_coldstart_data('data/alfworld_rebel_250_new/rebel_coldstart.json')
```

---

## 📝 五、建议的最佳实践

### 5.1 数据生成流程

```
1. Hindsight标注（WITH Available Actions）
   ↓
   generate_rebel_hindsight.py
   ├─ 生成rebel_hindsight.jsonl
   └─ 同时生成rebel_coldstart.json（删除Available Actions）
   ↓
2. 验证Cold-start数据
   ↓
   verify_coldstart_consistency.py
   ↓
3. SFT训练（使用cold-start数据）
   ↓
4. RL训练（动态添加Available Actions）
```

### 5.2 Prompt使用规范

| 阶段 | 数据格式 | Prompt模板 | Available Actions |
|------|---------|-----------|-------------------|
| **Hindsight标注** | hindsight.jsonl | `ALFWORLD_REBEL_TAGGING_TEMPLATE` | ✅ 有（辅助标注） |
| **Cold-start SFT** | coldstart.json | `ALFWORLD_REBEL_TEMPLATE_*_CS` | ❌ **必须无** |
| **RL训练** | 动态生成 | `ALFWORLD_REBEL_TEMPLATE_*_RL` | ✅ **必须有** |
| **RL评测** | 动态生成 | `ALFWORLD_REBEL_TEMPLATE_*_RL` | ✅ **必须有** |

---

## 🎯 六、总结

### 6.1 关键发现

1. ⚠️ **数据不一致**：Current cold-start数据包含Available Actions，与设计不符
2. ⚠️ **第一步最严重**：第一步包含完整的动作列表，完全违反课程学习设计
3. ⚠️ **后续步骤部分不一致**：显示"N/A"而不是完全删除

### 6.2 影响评估

**对SFT训练的影响**:
- 🔴 **高风险**：模型可能学会依赖Available Actions
- 🔴 **降低难度**：削弱了课程学习的"困难→简单"设计
- 🔴 **能力退化**：模型可能无法学会从纯observation推理动作

**对RL训练的影响**:
- 🟡 **中等影响**：SFT模型可能过度依赖动作列表
- 🟡 **探索困难**：RL阶段可能需要更长时间学习

### 6.3 行动建议

1. ✅ **立即行动**：使用验证脚本检查所有cold-start数据
2. ✅ **重新生成**：使用最新代码重新生成符合规范的cold-start数据
3. ✅ **重新训练**：使用正确数据重新进行SFT训练
4. ✅ **流程规范化**：建立数据生成和验证的标准流程

---

## 📂 附录：相关文件位置

**数据文件**:
- Hindsight数据: `data/alfworld_rebel_250_new/rebel_hindsight.jsonl`
- Cold-start数据: `data/alfworld_rebel_250_new/rebel_coldstart.json`

**Prompt文件**:
- ReBel Prompts: `agent_system/environments/prompts/rebel_prompts.py`

**代码文件**:
- 数据生成: `generate_rebel_hindsight.py`
- Cold-start转换: `convert_hindsight_to_coldstart.py`

**验证脚本**:
- 注释质量验证: `verify_annotation_quality.py`
- （待创建）数据一致性验证: `verify_coldstart_consistency.py`
