# 🎯 ReBel Rollout 评测指南

## 📋 什么是Rollout评测？

**Rollout评测** = 使用ReBel的Prompt让模型与ALFWorld环境实际交互，记录并分析交互过程

**与完整训练的区别:**
- ✅ **Rollout评测**: 直接交互 → 记录结果 → 分析（10-30分钟）
- ⏳ **完整训练**: 收集数据 → RL训练 → 更新模型 → 评测（数小时）

**适用场景:**
- ✅ 快速验证ReBel的Prompt设计
- ✅ 分析模型的信念状态质量
- ✅ 评估ReBel内在奖励的有效性
- ✅ 对比不同模型的表现

---

## 🚀 快速开始（2步）

### 方法1: 一键运行（最简单）⭐

```bash
cd /root/testttt/RLVMR/code

# 终端1: 启动vLLM服务器
bash start_vllm_server.sh

# 终端2: 运行ReBel评测（等vLLM启动完成后）
bash run_rebel_evaluation.sh
```

**预计时间:** 10-20分钟（4个环境，最多30步）

---

## 📖 详细步骤

### 第1步: 启动vLLM服务器

**在终端1运行:**
```bash
bash start_vllm_server.sh
```

**输出示例:**
```
==========================================
启动vLLM服务器
==========================================
模型: /root/testttt/RLVMR/code/base_models/Qwen2.5-1.5B-Instruct
端口: 8000
GPU内存: 0.5
==========================================
✅ 启动vLLM服务器...
INFO: Started server process
INFO: Application startup complete
INFO: Uvicorn running on http://0.0.0.0:8000
```

**等待看到 "Application startup complete" 表示服务启动成功**

**后台运行（推荐）:**
```bash
nohup bash start_vllm_server.sh > vllm.log 2>&1 &

# 查看日志
tail -f vllm.log
```

---

### 第2步: 运行ReBel评测

**在终端2运行:**
```bash
bash run_rebel_evaluation.sh
```

**运行过程:**
```
==========================================
ReBel Rollout 快速评测
==========================================
[1/3] 检查vLLM服务...
✅ vLLM服务正在运行

[2/3] 运行ReBel rollout评测...
   - 环境数量: 4
   - 最大步数: 30
   - 随机种子: 1

2025-12-20 12:34:56 - ==========================================
2025-12-20 12:34:56 - ReBel Rollout Evaluation
2025-12-20 12:34:56 - ==========================================
✅ ReBel环境创建成功
   - 环境数量: 4
   - ReBel启用: True
   - 奖励权重: α=0.3, β=0.5, γ=0.2, δ=0.1
✅ Agent初始化成功
   - 模型: Qwen/Qwen2.5-1.5B-Instruct
   - URL: http://127.0.0.1:8000/v1

开始Rollout...
Step  0 | Done: 0/4 | Success: 0/4 (0.0%)
Step  1 | Done: 0/4 | Success: 0/4 (0.0%)
...
```

---

### 第3步: 查看结果

**评测完成后自动显示:**
```
📊 快速查看结果:
----------------------------------------
✅ 成功率: 25.0%
📏 平均步数: 18.5
🏆 平均奖励: 0.75

🧠 ReBel指标:
  - 一致性奖励: 0.0234
  - 进度奖励:   0.0456
  - 探索奖励:   0.0123
  - 格式奖励:   0.0089
  - 信念解析率: 85.2%
----------------------------------------
```

---

## 📁 输出文件

评测完成后生成目录: `rebel_rollout_results/run_YYYYMMDD_HHMMSS/`

```
rebel_rollout_results/run_20251220_123456/
├── results.json          # ⭐ 统计结果（JSON格式）
├── trajectories.jsonl    # ⭐ 完整交互轨迹
└── rollout.log           # 运行日志
```

### results.json 结构

```json
{
  "timestamp": "20251220_123456",
  "config": {
    "env_num": 4,
    "max_steps": 30,
    "seed": 1,
    "model": "Qwen/Qwen2.5-1.5B-Instruct"
  },
  "basic_metrics": {
    "success_rate": 0.25,
    "avg_episode_length": 18.5,
    "avg_episode_reward": 0.75,
    "elapsed_time": 245.6
  },
  "rebel_metrics": {
    "avg_r_consistency": 0.0234,
    "avg_r_progress": 0.0456,
    "avg_r_exploration": 0.0123,
    "avg_r_format": 0.0089,
    "avg_belief_parse_rate": 0.852
  },
  "per_env": {
    "success": [true, false, false, true],
    "lengths": [15, 30, 30, 20],
    "rewards": [1.0, 0.0, 0.0, 1.5]
  }
}
```

### trajectories.jsonl 格式

每行一个JSON对象，记录每一步的详细信息：

```json
{
  "step": 0,
  "env_id": 0,
  "action": "**Belief State Update:** {...} **Action:** go to fridge 1",
  "reward": 0.05,
  "done": false,
  "rebel_rewards": {
    "r_consistency": 0.02,
    "r_progress": 0.05,
    "r_exploration": 0.01,
    "r_format": 0.01,
    "r_intrinsic_total": 0.09
  },
  "belief_state": {
    "world_model_update": {...},
    "task_progress_update": {...},
    "exploration_map_update": {...}
  },
  "belief_parsed": true
}
```

---

## ⚙️ 自定义配置

### 修改评测参数

编辑 `run_rebel_evaluation.sh`:

```bash
# 增加环境数量（更多样本）
ENV_NUM=10

# 增加最大步数（允许更长轨迹）
MAX_STEPS=50

# 改变随机种子（不同任务）
SEED=42
```

### 直接使用Python脚本

```bash
python3 run_rebel_rollout.py \
    --env_num 10 \
    --max_steps 50 \
    --seed 42 \
    --base_url "http://127.0.0.1:8000/v1" \
    --model "Qwen/Qwen2.5-1.5B-Instruct" \
    --temperature 0.4 \
    --output_dir rebel_rollout_results
```

### 使用不同的模型

```bash
# 启动不同的vLLM服务器（修改start_vllm_server.sh中的MODEL_PATH）
# 然后运行评测
python3 run_rebel_rollout.py --model "您的模型名称"
```

---

## 📊 结果分析

### 查看JSON结果

```bash
# 查看最新结果
cat rebel_rollout_results/run_*/results.json | python3 -m json.tool

# 提取成功率
python3 -c "
import json
import sys
with open('rebel_rollout_results/run_TIMESTAMP/results.json') as f:
    data = json.load(f)
    print(f\"Success Rate: {data['basic_metrics']['success_rate']*100:.1f}%\")
"
```

### 分析轨迹数据

```bash
# 统计信念解析成功的步数
grep '"belief_parsed": true' rebel_rollout_results/run_*/trajectories.jsonl | wc -l

# 查看所有奖励
cat rebel_rollout_results/run_*/trajectories.jsonl | \
    python3 -c "
import json
import sys
rewards = []
for line in sys.stdin:
    data = json.loads(line)
    rewards.append(data['rebel_rewards']['r_intrinsic_total'])
print(f'Avg intrinsic reward: {sum(rewards)/len(rewards):.4f}')
"
```

### 生成论文用表格

```python
import json
import pandas as pd

# 加载结果
with open('rebel_rollout_results/run_TIMESTAMP/results.json') as f:
    results = json.load(f)

# 创建表格
data = {
    'Metric': [
        'Success Rate (%)',
        'Avg Episode Length',
        'Avg Episode Reward',
        'r_consistency',
        'r_progress',
        'r_exploration',
        'r_format',
        'Belief Parse Rate (%)',
    ],
    'Value': [
        f"{results['basic_metrics']['success_rate']*100:.1f}",
        f"{results['basic_metrics']['avg_episode_length']:.1f}",
        f"{results['basic_metrics']['avg_episode_reward']:.2f}",
        f"{results['rebel_metrics']['avg_r_consistency']:.4f}",
        f"{results['rebel_metrics']['avg_r_progress']:.4f}",
        f"{results['rebel_metrics']['avg_r_exploration']:.4f}",
        f"{results['rebel_metrics']['avg_r_format']:.4f}",
        f"{results['rebel_metrics']['avg_belief_parse_rate']*100:.1f}",
    ]
}

df = pd.DataFrame(data)
print(df.to_latex(index=False))
```

---

## 🐛 常见问题

### Q1: vLLM服务启动失败

**错误:** `CUDA out of memory`

**解决:**
```bash
# 减少GPU内存使用
# 编辑 start_vllm_server.sh
GPU_MEMORY_UTILIZATION=0.3  # 改为0.3
```

### Q2: 端口被占用

**错误:** `Address already in use`

**解决:**
```bash
# 查找占用进程
lsof -i :8000

# 杀死进程
kill -9 $(lsof -t -i:8000)

# 重新启动
bash start_vllm_server.sh
```

### Q3: 评测运行很慢

**原因:** 模型推理慢或环境步数太多

**解决:**
```bash
# 减少环境数量和最大步数
python3 run_rebel_rollout.py --env_num 2 --max_steps 20
```

### Q4: 信念解析率很低

**可能原因:**
1. 模型输出格式不规范
2. ReBel prompt未正确加载
3. 模型能力不足

**检查:**
```bash
# 查看实际的model输出
head -5 rebel_rollout_results/run_*/trajectories.jsonl | python3 -m json.tool
```

### Q5: 所有环境都失败

**检查:**
1. vLLM服务是否正常响应
2. 环境是否正确创建
3. 查看详细日志

```bash
# 测试vLLM
curl http://127.0.0.1:8000/v1/models

# 查看完整日志
cat rebel_rollout_results/run_*/rollout.log
```

---

## 📈 对比实验

### ReBel vs 基线（不使用信念状态）

```bash
# 1. 运行ReBel评测
bash run_rebel_evaluation.sh
# 记录结果: success_rate_rebel

# 2. 修改代码禁用ReBel（修改run_rebel_rollout.py中的use_rebel=False）
# 运行基线评测
bash run_rebel_evaluation.sh
# 记录结果: success_rate_baseline

# 3. 对比
echo "ReBel提升: $((success_rate_rebel - success_rate_baseline))%"
```

### 不同模型对比

```bash
# Qwen-1.5B
python3 run_rebel_rollout.py --model "Qwen/Qwen2.5-1.5B-Instruct"

# Qwen-7B (需要更多GPU内存)
python3 run_rebel_rollout.py --model "Qwen/Qwen2.5-7B-Instruct"
```

---

## 📝 总结

**ReBel Rollout评测流程:**
1. ✅ 启动vLLM服务器（基础模型）
2. ✅ 运行rollout评测（ReBel Prompt + 环境交互）
3. ✅ 记录轨迹和信念状态
4. ✅ 分析ReBel各项指标

**关键文件:**
- `run_rebel_rollout.py` - 主评测脚本
- `start_vllm_server.sh` - vLLM服务启动
- `run_rebel_evaluation.sh` - 一键运行脚本

**输出:**
- `results.json` - 统计结果
- `trajectories.jsonl` - 完整轨迹
- `rollout.log` - 运行日志

**下一步:**
- 分析结果写入论文
- 对比不同模型/配置
- 优化ReBel Prompt

---

**创建时间:** 2025-12-20
**状态:** ✅ Ready to Run
