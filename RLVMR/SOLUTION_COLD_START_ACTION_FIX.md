# 冷启动数据动作格式问题修复方案

## 问题诊断

### 根本原因
冷启动数据中的动作格式与ALFWorld环境的admissible actions不匹配：

**训练数据格式**：
```
put remotecontrol 1 in/on sofa 1
```

**环境期望格式**：
```
move remotecontrol 1 to sofa 1
```

### 影响
- 模型在评测时生成的动作被环境拒绝（"Nothing happens"）
- 导致任务失败，成功率为0%
- 模型误判任务完成状态

## 解决方案

### 方案 1：修复原始数据源 + 重新生成（推荐）

#### 步骤1：检查专家轨迹数据源

```bash
cd /root/digitalhuman/RLVMR/code

# 检查原始数据集的动作格式
python -c "
from datasets import load_from_disk
import json

dataset = load_from_disk('data/alfworld_expert_traj')
print(f'数据集大小: {len(dataset)}')

# 查看前3个轨迹
for i in range(min(3, len(dataset))):
    traj = dataset[i]
    convs = traj['conversations']
    print(f'\n轨迹 {i+1}:')
    for j in range(2, min(6, len(convs)), 2):
        action = convs[j+1]['value']
        print(f'  {action[:100]}...')
"
```

#### 步骤2：修复数据生成脚本

在 `scripts/alfworld_prepare.py` 中添加动作格式转换：

```python
# 在 line 101 之后添加
action = llm_action.split("Action:")[-1].strip()

# 添加动作格式转换
action = convert_action_format(action)

def convert_action_format(action):
    """
    Convert action from expert trajectory format to admissible action format
    """
    import re

    # put [obj] in/on [recep] -> move [obj] to [recep]
    match = re.match(r'put\s+(.+?)\s+in/on\s+(.+)', action, re.IGNORECASE)
    if match:
        obj = match.group(1).strip()
        recep = match.group(2).strip()
        return f'move {obj} to {recep}'

    return action
```

#### 步骤3：重新生成冷启动数据

```bash
cd /root/digitalhuman/RLVMR/code

# 备份旧数据
cp data/alfworld_cold-start.json data/alfworld_cold-start_backup.json

# 重新生成300条数据
python scripts/alfworld_prepare.py \
    --api_key YOUR_API_KEY \
    --model gpt-4o \
    --num_trajs 300 \
    --output_path data/alfworld_cold-start_fixed.json
```

#### 步骤4：转换为parquet格式

```bash
python -m examples.data_preprocess.cold_start_data \
    --data_source data/alfworld_cold-start_fixed.json \
    --local_dir $HOME/data/alfworld
```

#### 步骤5：重新训练SFT模型

```bash
bash examples/sft/cold_start/run_alfworld_qwen2.5-1.5b.sh
```

### 方案 2：后处理修复现有数据（快速但不完美）

如果不想重新生成数据，可以直接修复现有的300条数据：

```bash
cd /root/digitalhuman/RLVMR/code

python -c "
import json
import re

# 读取现有数据
with open('data/alfworld_cold-start.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

def fix_action(response):
    '''修复response中的动作格式'''
    # 提取<action>标签内容
    action_match = re.search(r'<action>(.*?)</action>', response, re.IGNORECASE)
    if not action_match:
        return response

    old_action = action_match.group(1)

    # put [obj] in/on [recep] -> move [obj] to [recep]
    new_action = re.sub(
        r'put\s+(.+?)\s+in/on\s+(.+)',
        r'move \1 to \2',
        old_action,
        flags=re.IGNORECASE
    )

    # 替换response中的动作
    if new_action != old_action:
        new_response = response.replace(
            f'<action>{old_action}</action>',
            f'<action>{new_action}</action>'
        )
        return new_response
    return response

# 修复所有数据
fixed_count = 0
for task in data:
    for step in task['data']:
        old_response = step['response']
        new_response = fix_action(old_response)
        if old_response != new_response:
            step['response'] = new_response
            fixed_count += 1

print(f'修复了 {fixed_count} 个动作')

# 保存修复后的数据
with open('data/alfworld_cold-start_fixed.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print('已保存到 data/alfworld_cold-start_fixed.json')
"

# 转换为parquet
python -m examples.data_preprocess.cold_start_data \
    --data_source data/alfworld_cold-start_fixed.json \
    --local_dir $HOME/data/alfworld_fixed
```

### 方案 3：修改运行时Prompt模板（最简单，但不治本）

修改 `agent_system/environments/prompts/alfworld.py` 中的prompt模板，让模型直接从admissible actions中选择：

在BDRS runtime prompt中强调：
```
IMPORTANT: Your action MUST be EXACTLY copied from one of the admissible actions list below.
Do NOT paraphrase or modify the action format.
```

然后在评测时设置更严格的动作验证。

## 推荐步骤

**立即执行（服务器）**：

```bash
# 1. 快速修复现有数据（方案2）
cd /root/digitalhuman/RLVMR/code
python fix_action_format.py  # 见上面的脚本

# 2. 转换为parquet
python -m examples.data_preprocess.cold_start_data \
    --data_source data/alfworld_cold-start_fixed.json \
    --local_dir $HOME/data/alfworld_fixed

# 3. 重新训练SFT（使用修复后的数据）
# 修改run_alfworld_qwen2.5-1.5b.sh中的数据路径
# data.train_files=$HOME/data/alfworld_fixed/train.parquet

# 4. 评测新模型
bash examples/bdrs_trainer/eval_cold_start.sh
```

**长期优化**：
- 检查并修复 `alfworld_expert_traj` 数据源
- 使用方案1重新生成高质量的300条数据
- 确保数据生成prompt与运行时prompt对齐

## 验证

修复后，检查动作格式：

```bash
python -c "
import json
data = json.load(open('data/alfworld_cold-start_fixed.json', encoding='utf-8'))

# 统计动作类型
actions = {}
for task in data:
    for step in task['data']:
        import re
        match = re.search(r'<action>(.*?)</action>', step['response'], re.IGNORECASE)
        if match:
            action = match.group(1)
            action_type = action.split()[0]
            actions[action_type] = actions.get(action_type, 0) + 1

print('动作类型分布:')
for act, count in sorted(actions.items()):
    print(f'  {act}: {count}')

# 检查是否还有 put ... in/on 格式
has_old_format = False
for task in data:
    for step in task['data']:
        if 'in/on' in step['response'].lower():
            print(f'警告: 仍有旧格式动作: {step[\"response\"][:100]}')
            has_old_format = True
            break
    if has_old_format:
        break

if not has_old_format:
    print('✓ 所有动作格式已修复')
"
```
