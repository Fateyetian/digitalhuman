# AlfWorld 评测指南

## 迁移完成 ✅

已将评测脚本从 `openmanus_rl` 项目迁移到当前 RLVMR 项目。

### 修改内容

1. **Import路径**: `openmanus_rl` → `agent_system`
2. **Config路径**: 更新为当前项目的配置文件路径

## 使用方法

### 方案A：使用本地vLLM后端（推荐）

#### 步骤1：启动vLLM服务器

在一个终端窗口中：

```bash
conda activate rlvmr-alfworld

python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --gpu-memory-utilization 0.7
```

#### 步骤2：运行评测

在另一个终端窗口中：

```bash
cd /root/digitalhuman/RLVMR/code

# 基础用法（4个环境）
bash examples/bdrs_trainer/rollout/run_local_eval.sh

# 自定义参数
bash examples/bdrs_trainer/rollout/run_local_eval.sh \
    Qwen/Qwen2.5-1.5B-Instruct \  # 模型路径
    10 \                           # 环境数量
    5 \                            # 批处理大小
    30 \                           # 最大步数
    results/my_eval                # 输出目录
```

### 方案B：使用OpenAI API

如果你有OpenAI API密钥：

```bash
export OPENAI_API_KEY="your-api-key"

python3 examples/bdrs_trainer/rollout/run_alfworld_rollout.py \
    --env_name alfworld \
    --batch_size 4 \
    --total_envs 8 \
    --max_steps 30 \
    --model gpt-4o-mini \
    --temperature 0.4 \
    --concurrency 4 \
    --dump_path results/gpt4o_eval.jsonl \
    --chat_root results/gpt4o_chats \
    --unique_envs
```

### 方案C：使用远程vLLM服务器

如果vLLM已部署在其他服务器：

```bash
python3 examples/bdrs_trainer/rollout/run_alfworld_rollout.py \
    --env_name alfworld \
    --batch_size 4 \
    --total_envs 8 \
    --model "Qwen/Qwen2.5-7B-Instruct" \
    --base_url "http://remote-server-ip:8000/v1" \
    --dump_path results/remote_eval.jsonl \
    --unique_envs
```

## 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--env_name` | 环境名称 | alfworld |
| `--batch_size` | 每批处理的环境数 | 10 |
| `--total_envs` | 总评测环境数 | 1000 |
| `--max_steps` | 每个任务最大步数 | 50 |
| `--model` | 模型名称 | gpt-4o-mini |
| `--base_url` | vLLM服务器URL | None (使用OpenAI) |
| `--temperature` | 采样温度 | 0.4 |
| `--concurrency` | 并发请求数 | 4 |
| `--dump_path` | 轨迹保存路径 | None |
| `--chat_root` | 对话历史保存目录 | None |
| `--unique_envs` | 使用唯一任务（无重复） | False |

## 输出文件

### 1. 轨迹文件 (trajectory.jsonl)

每行是一个JSON对象：

```json
{
  "batch_idx": 0,
  "env_id": 5,
  "step": 10,
  "prompt": "观察文本...",
  "action": "模型生成的动作",
  "action_exec": "执行的动作",
  "reward": 0.0,
  "done": false,
  "won": false,
  "gamefile": "pick_and_place_simple-Apple-None-DiningTable-315/trial_T20190908_180405_686265",
  "is_action_valid": true
}
```

### 2. 对话历史 (chats/)

```
results/
└── chats/
    └── trajectories/
        └── 20250109_153045/
            └── alfworld/
                └── Qwen2.5-1.5B-Instruct/
                    ├── pick_and_place/
                    │   ├── chat_pick_and_place-task1-b000_t00_e00.json
                    │   └── ...
                    └── look_at_obj_in_light/
                        └── ...
```

每个JSON文件包含：
- `messages`: 完整的对话历史
- `metadata`: 元数据（任务、成功率等）

## 评测结果

脚本会在日志中输出：

```
=============== Final Summary ===============
Total batches: 3 | Batch size: 4 | Total envs processed: 10
Overall success avg ± std: 0.4500 ± 0.1234

pick_and_place              : 0.5000 ± 0.2000
pick_two_obj_and_place      : 0.3333 ± 0.1500
look_at_obj_in_light        : 0.6667 ± 0.1000
...
```

## 常见问题

### Q1: vLLM内存不足怎么办？

降低GPU内存占用：
```bash
--gpu-memory-utilization 0.5
```

或使用更小的模型：
```bash
Qwen/Qwen2.5-1.5B-Instruct  # 1.5B参数
```

### Q2: 评测速度太慢？

增加并发数和批处理大小：
```bash
--concurrency 10 \
--batch_size 10
```

### Q3: 如何只评测特定任务类型？

修改代码中的任务过滤逻辑，或者查看保存的轨迹文件，按gamefile过滤。

### Q4: 报错 "Creating 4096 workers"？

已修复！确保使用最新的 `agent_system/environments/env_manager.py`。

## 下一步

- 分析轨迹数据：`python analyze_trajectories.py results/trajectory.jsonl`
- 可视化成功率：使用保存的chat历史
- 对比不同模型：运行多次评测并比较结果
