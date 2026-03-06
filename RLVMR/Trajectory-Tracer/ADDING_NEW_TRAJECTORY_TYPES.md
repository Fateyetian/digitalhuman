# 如何添加新的轨迹类型

本文档说明如何在 Trajectory-Tracer 中添加对新轨迹格式的支持。

## 架构概述

Trajectory-Tracer 使用适配器模式（Adapter Pattern）来支持多种轨迹数据格式。系统包含以下核心组件：

1. **TrajectoryAdapter（基类）** - 定义了所有适配器必须实现的接口
2. **具体适配器** - 针对特定格式的实现（如 HuggingFaceDatasetAdapter、REBELJSONAdapter）
3. **TrajectoryLoader** - 自动检测格式并选择合适的适配器
4. **统一数据模型** - `Trajectory` 和 `Message` 类定义了标准化的内部表示

## 支持的格式

目前支持以下格式：

### 1. HuggingFace Dataset 格式

```python
{
  'conversations': [
    {'from': 'human', 'value': '...'},
    {'from': 'gpt', 'value': '...'}
  ],
  'item_id': '...'
}
```

### 2. REBEL JSON 格式

```json
{
  "task": "put some cellphone on sidetable.",
  "done": "True",
  "data": [
    {
      "step": 1,
      "obs": "...",
      "prompt": "...",
      "response": "<belief>...</belief><reasoning>...</reasoning><action>...</action>"
    }
  ]
}
```

## 添加新格式的步骤

### 步骤 1: 分析原始数据格式

首先，了解你的轨迹数据格式：

- 数据的存储方式（文件格式：JSON、CSV、数据库等）
- 数据的结构（字段、嵌套关系等）
- 如何表示对话轮次
- 如何表示任务、状态、步骤等信息

### 步骤 2: 创建适配器类

在 `backend/trajectory_adapters.py` 中创建新的适配器类，继承 `TrajectoryAdapter`：

```python
class YourFormatAdapter(TrajectoryAdapter):
    """你的格式适配器描述"""

    def load(self, path: Path) -> List[Dict[str, Any]]:
        """
        加载原始数据文件

        Args:
            path: 数据文件或目录路径

        Returns:
            原始数据项列表
        """
        # 实现你的加载逻辑
        # 例如：JSON 文件
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

        # 或者：CSV 文件
        # import pandas as pd
        # df = pd.read_csv(path)
        # return df.to_dict('records')

    def parse(self, raw_item: Dict[str, Any], idx: int) -> Trajectory:
        """
        将原始数据项转换为统一的 Trajectory 对象

        Args:
            raw_item: 单条原始数据
            idx: 索引号

        Returns:
            Trajectory 对象
        """
        # 1. 提取基本信息
        task = raw_item.get('task_field', '')
        status = 'success'  # 或根据数据判断

        # 2. 解析消息序列
        messages = []
        for step in raw_item.get('steps', []):
            # 添加人类消息（观察/环境反馈）
            messages.append(Message(
                role='human',
                content=step.get('observation', ''),
                metadata={'step': step.get('step_num')}
            ))

            # 添加 agent 消息
            messages.append(Message(
                role='agent',
                content=step.get('response', ''),
                thought=step.get('thinking', None),
                action=step.get('action', None),
                metadata={'step': step.get('step_num')}
            ))

        # 3. 提取任务类型
        task_type = self._extract_task_type(task)

        # 4. 计算步数
        steps = len([m for m in messages if m.role == 'agent' and m.action])

        # 5. 返回 Trajectory 对象
        return Trajectory(
            id=f"your_format_{idx:05d}",
            task=task,
            status=status,
            steps=steps,
            task_type=task_type,
            messages=messages,
            environment=raw_item.get('env_description', ''),
            metadata={'source': 'your_format', 'original_id': raw_item.get('id')}
        )

    def _extract_task_type(self, task: str) -> str:
        """辅助方法：从任务描述中提取任务类型"""
        if not task:
            return 'unknown'

        first_word = task.split()[0].lower()
        known_types = ['put', 'clean', 'heat', 'cool', 'find', 'examine', 'use']
        return first_word if first_word in known_types else 'other'
```

### 步骤 3: 注册适配器

在 `TrajectoryLoader.__init__()` 中注册你的适配器：

```python
class TrajectoryLoader:
    def __init__(self):
        self.adapters = {
            'huggingface': HuggingFaceDatasetAdapter(),
            'rebel_json': REBELJSONAdapter(),
            'your_format': YourFormatAdapter(),  # 添加这一行
        }
```

### 步骤 4: 实现格式检测

在 `TrajectoryLoader.detect_format()` 中添加检测逻辑：

```python
def detect_format(self, path: Path) -> Optional[str]:
    """自动检测轨迹格式"""
    if path.is_file() and path.suffix == '.json':
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list) and len(data) > 0:
                    first_item = data[0]

                    # 检查你的格式特征
                    if 'your_special_field' in first_item:
                        return 'your_format'

                    # 其他格式的检测...
        except Exception:
            pass

    return None
```

### 步骤 5: 配置数据源

在 `backend/data_sources.json` 中添加数据源配置：

```json
{
  "data_sources": [
    {
      "name": "Your Dataset Name",
      "path": "path/to/your/data.json",
      "type": "your_format",
      "enabled": true,
      "description": "Description of your dataset"
    }
  ]
}
```

### 步骤 6: 测试

创建测试脚本验证你的适配器：

```python
from pathlib import Path
from trajectory_adapters import TrajectoryLoader

# 测试加载
loader = TrajectoryLoader()
trajectories = loader.load(Path("path/to/your/data"))

# 验证结果
print(f"Loaded {len(trajectories)} trajectories")
for i, traj in enumerate(trajectories[:3]):
    print(f"\nTrajectory {i}:")
    print(f"  ID: {traj.id}")
    print(f"  Task: {traj.task}")
    print(f"  Status: {traj.status}")
    print(f"  Steps: {traj.steps}")
    print(f"  Messages: {len(traj.messages)}")
```

## 统一数据模型

所有适配器必须将原始数据转换为以下统一模型：

### Trajectory 类

```python
class Trajectory:
    id: str              # 唯一标识符
    task: str            # 任务描述
    status: str          # 'success', 'failed', 'unknown'
    steps: int           # 步骤数（agent 执行的动作数）
    task_type: str       # 任务类型（put, clean, heat, cool 等）
    messages: List[Message]  # 消息序列
    environment: str     # 环境描述
    metadata: Dict       # 额外的元数据
```

### Message 类

```python
class Message:
    role: str            # 'human' 或 'agent'
    content: str         # 完整的消息内容
    thought: Optional[str]   # agent 的思考过程（如果有）
    action: Optional[str]    # agent 的动作（如果有）
    metadata: Dict       # 额外的元数据（如 step, belief 等）
```

## 最佳实践

1. **错误处理**：在 `load()` 和 `parse()` 中使用 try-except 捕获异常
2. **元数据**：将格式特有的信息存储在 `metadata` 中
3. **命名规范**：ID 使用 `{format}_traj_{idx:05d}` 格式
4. **日志输出**：使用 print 输出加载进度和警告信息
5. **向后兼容**：确保新格式不影响现有功能

## 示例：添加 CSV 格式支持

```python
import csv
from pathlib import Path
from typing import List, Dict, Any

class CSVTrajectoryAdapter(TrajectoryAdapter):
    """CSV 格式轨迹适配器"""

    def load(self, path: Path) -> List[Dict[str, Any]]:
        """加载 CSV 文件"""
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            return list(reader)

    def parse(self, raw_item: Dict[str, Any], idx: int) -> Trajectory:
        """解析 CSV 行"""
        messages = []

        # 假设 CSV 有 obs 和 action 列
        messages.append(Message(
            role='human',
            content=raw_item.get('obs', '')
        ))

        messages.append(Message(
            role='agent',
            content=raw_item.get('action', ''),
            action=raw_item.get('action', '')
        ))

        return Trajectory(
            id=f"csv_traj_{idx:05d}",
            task=raw_item.get('task', ''),
            status=raw_item.get('status', 'unknown'),
            steps=1,
            task_type=raw_item.get('task_type', 'unknown'),
            messages=messages,
            environment='',
            metadata={'source': 'csv'}
        )
```

## 常见问题

### Q: 我的数据格式非常复杂，如何处理？

A: 将复杂的解析逻辑拆分成多个辅助方法，保持 `parse()` 方法简洁清晰。

### Q: 如何处理不同版本的同一格式？

A: 在 `parse()` 中检测版本字段，使用条件分支处理不同版本。

### Q: 原始数据中缺少某些必需字段怎么办？

A: 为缺失字段提供合理的默认值，并在 metadata 中标记数据质量问题。

## 相关文件

- `backend/trajectory_adapters.py` - 适配器实现
- `backend/main.py` - FastAPI 后端主文件
- `backend/data_sources.json` - 数据源配置
- `TESTING.md` - 测试指南

## 贡献

如果你实现了新的适配器，欢迎提交 Pull Request 分享给社区！
