# ReBel 快速开始指南

## 🎯 目标

从 0% 成功率提升到 >30% 成功率

## 📦 已完成的改进

✅ 初始信念状态从环境观测初始化
✅ 记录完整的信念状态历史（预测 + 真实）
✅ 轨迹数据增强（包含观测、预测状态、真实状态）
✅ 冷启动数据生成工具

## 🚀 快速开始

### 选项 1: 使用已训练的 Checkpoint（最快）

```bash
# 1. 修改 vLLM 配置使用训练后的模型
# 编辑 start_vllm.sh，改为：
MODEL_PATH="./checkpoints/cold_start/alfworld/bdrs_qwen1.5b_2gpu_20251110/global_step_75"

# 2. 启动 vLLM（会自动清理 GPU）
bash start_vllm.sh

# 3. 运行评估
bash run_rebel_evaluation.sh

# 4. 查看结果
cat rebel_rollout_results/run_*/results.json
```

**预期结果**:
- 格式解析率: >80%
- 一致性奖励: >0
- 成功率: >10%

---

### 选项 2: 生成冷启动数据并训练（最佳效果）

```bash
# 1. 生成完整的冷启动数据集（~5分钟）
python3 generate_rebel_cold_start_data.py

# 2. 验证生成的数据
python3 << 'EOF'
from datasets import load_from_disk
ds = load_from_disk("data/alfworld_rebel_cold_start")
print(f"✅ 生成了 {len(ds)} 条训练数据")

# 查看第一个样本
example = ds[0]
for turn in example['conversations']:
    if '<belief>' in turn.get('value', ''):
        print("\n样本格式:")
        print(turn['value'][:500])
        break
EOF

# 3. 配置训练使用新数据
# TODO: 更新训练配置文件，指向 data/alfworld_rebel_cold_start

# 4. 进行冷启动训练
# TODO: 运行训练脚本

# 5. 评估训练后的模型
bash start_vllm.sh  # 使用新训练的模型
bash run_rebel_evaluation.sh
```

**预期结果**:
- 格式解析率: >95%
- 一致性奖励: >0.05
- 成功率: >30%

---

## 📊 验证改进效果

### 对比评估结果

**Base 模型（之前）**:
```json
{
  "success_rate": 0.0,
  "avg_r_consistency": 0.0,
  "avg_r_progress": 0.0,
  "avg_r_exploration": 0.0,
  "avg_r_format": -0.05,
  "avg_belief_parse_rate": 0.0
}
```

**训练后的模型（预期）**:
```json
{
  "success_rate": 0.3,
  "avg_r_consistency": 0.05,
  "avg_r_progress": 0.08,
  "avg_r_exploration": 0.03,
  "avg_r_format": 0.01,
  "avg_belief_parse_rate": 0.95
}
```

### 分析新的轨迹数据

```bash
# 查看最新评估结果
LATEST=$(ls -td rebel_rollout_results/run_* | head -1)
echo "最新结果: $LATEST"

# 分析信念状态质量
python3 << EOF
import json

# 读取轨迹
with open('$LATEST/trajectories.jsonl', 'r') as f:
    lines = f.readlines()

# 统计信念状态解析情况
total = 0
parsed = 0
for line in lines[:10]:  # 前10步
    data = json.loads(line)
    total += 1
    if data.get('belief_parsed'):
        parsed += 1
        # 显示一个示例
        if parsed == 1:
            print("示例信念状态（预测）:")
            print(json.dumps(data['belief_state_pred'], indent=2)[:500])
            print("\n真实环境状态:")
            print(json.dumps(data['belief_state_gt'], indent=2)[:500])

print(f"\n解析率: {parsed}/{total} = {parsed/total*100:.1f}%")
EOF
```

---

## 📁 生成的文件说明

### 代码修改
- `agent_system/environments/env_manager.py` - 信念状态管理
- `run_rebel_rollout.py` - 轨迹记录

### 工具脚本
- `generate_rebel_cold_start_data.py` - 生成冷启动数据
- `test_rebel_improvements.sh` - 快速测试

### 数据文件
- `data/alfworld_rebel_cold_start/` - 冷启动数据集
- `rebel_rollout_results/run_*/trajectories.jsonl` - 评估轨迹（增强版）

### 文档
- `REBEL_EVALUATION_ANALYSIS.md` - 问题分析
- `REBEL_COLD_START_GUIDE.md` - 冷启动数据指南
- `REBEL_IMPROVEMENTS_SUMMARY.md` - 改进总结
- `REBEL_QUICK_START.md` (本文件)

---

## 🐛 故障排除

### GPU 内存不足
```bash
# start_vllm.sh 已经自动清理 GPU
# 如果仍然报错，手动清理：
ps aux | grep python | grep -v grep | awk '{print $2}' | xargs kill -9
nvidia-smi
```

### 格式解析失败
- 确认使用的是训练后的模型，不是 base 模型
- 检查 `belief_state_pred` 字段是否为 `null`
- 查看模型输出是否包含 `<belief>` 标签

### 数据生成失败
```bash
# 测试少量数据
python3 generate_rebel_cold_start_data.py --limit 5

# 查看错误日志
# 如果某些轨迹失败，会显示具体错误
```

---

## 💡 关键洞察

### 为什么需要冷启动数据？

Base 模型不知道 ReBel 格式：
```
# Base 模型输出（混乱）
```json
{
  "belief": {
    "world_model_update": {...}
  },
  "reasoning": "...",
  "action": "go to drawer 1"
}
```

冷启动数据提供正确格式：
```
<belief>
{
  "world_model_update": {...}
}
</belief>

<reasoning>
...
</reasoning>

<action>
go to drawer 1
</action>
```

### 黄金信念状态的价值

通过跟踪环境真实状态，我们确保：
1. ✅ 信念状态与环境完全一致
2. ✅ 状态更新准确无误
3. ✅ 提供高质量的监督信号

这是提升成功率的关键！

---

## 🎯 下一步

1. **立即行动**: 运行选项 1，验证代码改进
2. **短期**: 生成完整冷启动数据
3. **中期**: 进行冷启动训练
4. **长期**: 分析结果，迭代改进

**现在就开始**:
```bash
# 快速测试所有改进
bash test_rebel_improvements.sh
```

---

## 📞 需要帮助？

查看文档：
- 问题分析: `REBEL_EVALUATION_ANALYSIS.md`
- 冷启动指南: `REBEL_COLD_START_GUIDE.md`
- 改进总结: `REBEL_IMPROVEMENTS_SUMMARY.md`

祝你好运！🚀
