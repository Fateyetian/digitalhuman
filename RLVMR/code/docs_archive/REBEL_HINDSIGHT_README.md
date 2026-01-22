# ReBel Hindsight Annotation System

基于Teacher LLM的后见之明（Hindsight）标注系统，用于生成高质量的ReBel格式训练数据。

## 核心特性

✅ **后见之明标注**: 利用专家动作反向推导合理的信念状态
✅ **完整状态追踪**: Inventory、Cleared receptacles、Task progress
✅ **自动修正**: LLM失败时的Fallback机制
✅ **全量/增量平衡**: 输入完整上下文，输出聚焦更新
✅ **Admissible Actions**: 包含可行动作约束

## 快速开始

### 1. 准备数据

确保有专家轨迹数据：
```bash
# 检查数据是否存在
ls data/alfworld_expert_traj.json
# 或
ls data/alfworld_expert_traj/
```

### 2. 启动Teacher LLM服务器

#### 选项A: 本地vLLM（推荐）

```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --port 8000 \
    --tensor-parallel-size 1
```

#### 选项B: 使用OpenAI API

```bash
export OPENAI_API_KEY="your-key-here"
```

### 3. 运行标注

```bash
# 快速测试（2个样本）
bash test_rebel_hindsight.sh

# 完整标注（使用本地模型）
python generate_rebel_hindsight.py \
    --expert_data data/alfworld_expert_traj.json \
    --output_dir data/alfworld_rebel_hindsight \
    --num_samples 100 \
    --teacher_model_url http://127.0.0.1:8000/v1 \
    --teacher_model_name "Qwen2.5-7B-Instruct"

# 高质量标注（使用GPT-4）
python generate_rebel_hindsight.py \
    --expert_data data/alfworld_expert_traj.json \
    --output_dir data/alfworld_rebel_hindsight_gpt4 \
    --num_samples 100 \
    --teacher_model_url https://api.openai.com/v1 \
    --teacher_model_name "gpt-4" \
    --temperature 0.2
```

## 输出格式

生成的数据包含：

```
data/alfworld_rebel_hindsight/
├── dataset_info.json
├── data-00000-of-00001.arrow
└── rebel_hindsight.jsonl
```

### 训练样本格式

**Human Turn (输入)**:
```
Task: put alarmclock on desk

Observation:
You are in the middle of a room...

Current Belief State:
{
  "world_model": {
    "found_objects": {"alarmclock 1": "sidetable 1"},
    "inventory": null,
    "cleared_receptacles": ["drawer 1", "drawer 2"]
  },
  ...
}

Available Actions: go to shelf 1, take alarmclock 1, ...
```

**GPT Turn (输出)**:
```xml
<belief>
{
  "world_model_update": {
    "inventory": "alarmclock 1",
    ...
  },
  ...
}
</belief>

<reasoning>
I found alarmclock 1 on sidetable 1. Since my task is to put an alarmclock on the desk, I should take it first...
</reasoning>

<action>
take alarmclock 1 from sidetable 1
</action>
```

## 质量评估

标注成功率：
- **>90%**: 优秀（GPT-4, Qwen2.5-72B）
- **70-90%**: 良好（Qwen2.5-7B/14B）
- **<70%**: 需要更强模型

## 主要改进点

相比原有方法 (`generate_rebel_golden_v2.py`):

| 特性 | 原方法 | 后见之明方法 |
|------|--------|------------|
| 环境依赖 | ✅ 需要 | ❌ 不需要 |
| Inventory追踪 | ❌ | ✅ |
| Cleared逻辑 | ❌ | ✅ 自动推断 |
| Admissible Actions | ❌ | ✅ |
| Task上下文 | ⚠️ | ✅ 每步显式 |
| 状态一致性 | ❌ | ✅ 累积传递 |

## 文档

- **改进总结**: `REBEL_HINDSIGHT_IMPROVEMENTS.md` - 详细技术文档
- **代码结构**: `generate_rebel_hindsight.py` - 700+行，包含详细注释

## 问题排查

### Teacher LLM连接失败

```bash
# 检查服务器状态
curl http://127.0.0.1:8000/v1/models
```

### 标注成功率低

- 使用更强的Teacher模型（GPT-4）
- 降低temperature（0.1-0.2）
- 检查prompt是否适合你的Teacher模型

### JSON解析错误

- 模型可能输出格式不规范
- 尝试添加 `--temperature 0.1`
- 查看日志中的原始输出

## 下一步

1. 查看生成的数据质量
2. 使用数据进行SFT训练
3. 在ALFWorld环境中评估模型性能

详细信息请参考 `REBEL_HINDSIGHT_IMPROVEMENTS.md`。
