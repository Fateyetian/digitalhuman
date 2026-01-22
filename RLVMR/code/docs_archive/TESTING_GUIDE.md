# ReBel 测试与评估指南

本指南介绍如何运行小规模测试、分析结果，以及为论文准备实验数据。

## 快速开始

### 1. 运行小规模测试

```bash
# 基础用法（使用默认模型路径）
bash test_rebel_small.sh vllm ./checkpoints/cold_start/alfworld/sft_qwen2.5-1.5b

# 或者指定自定义模型路径
bash test_rebel_small.sh vllm /path/to/your/model
```

**测试配置：**
- 训练任务数：2（极小规模，快速验证）
- 验证任务数：4
- Group Size：8（小规模rollout）
- 最大步数：15
- 训练轮数：1（仅验证功能）

**预计运行时间：** 10-30分钟（取决于硬件）

### 2. 查看测试结果

测试完成后，结果保存在 `rebel_test_results/YYYYMMDD_HHMMSS/` 目录：

```
rebel_test_results/20251220_123456/
├── test_config.yaml          # 测试配置
├── test_results.json         # 结构化结果（JSON）
├── test_results.yaml         # 结构化结果（YAML）
├── training.log              # 完整训练日志
├── data_prep.log             # 数据准备日志
└── checkpoints/              # 检查点（如果保存）
```

**快速查看结果：**
```bash
# 查看JSON结果
cat rebel_test_results/20251220_123456/test_results.json

# 查看YAML结果（更易读）
cat rebel_test_results/20251220_123456/test_results.yaml
```

### 3. 深度分析结果

使用分析脚本生成论文友好的输出：

```bash
# 运行分析脚本
python3 analyze_rebel_results.py rebel_test_results/20251220_123456/
```

**生成的文件：**
- `analysis_results.json` - 完整分析结果（JSON格式）
- `analysis_results.yaml` - 完整分析结果（YAML格式）
- `ANALYSIS_REPORT.md` - 易读的Markdown报告
- `paper_tables.tex` - LaTeX表格（直接用于论文）

## 结果文件详解

### 1. test_results.json

核心指标的结构化输出：

```json
{
  "test_info": {
    "timestamp": "20251220_123456",
    "status": "success",  // 或 "failed"
    "errors": []
  },
  "belief_grouping": {
    "num_groups": 42,               // 信念组数量
    "mean_group_size": 15.2,        // 平均组大小
    "group_sizes": [...]            // 所有组的大小分布
  },
  "rewards": {
    "episode_reward_mean": 0.75,    // 任务奖励均值
    "intrinsic_reward_mean": 0.23,  // 内在奖励均值
    "intrinsic_reward_std": 0.08    // 内在奖励标准差
  },
  "performance": {
    "success_rate": 0.672,          // 成功率
    "avg_steps": 14.1,              // 平均步数
    "efficiency": 0.0477            // 效率 = 成功率/步数
  },
  "belief_consistency": {
    "format_valid_rate": 0.95,      // 格式有效率
    "action_valid_rate": 0.89       // 动作有效率
  }
}
```

### 2. ANALYSIS_REPORT.md

人类可读的Markdown报告，包含：

- **信念分组统计**：组数、大小分布
- **性能指标**：成功率、效率、步数
- **奖励统计**：任务奖励、内在奖励分布
- **样例信念状态**：前3个信念状态的JSON
- **错误警告**：发现的错误和警告信息

**直接查看：**
```bash
cat rebel_test_results/20251220_123456/ANALYSIS_REPORT.md
```

### 3. paper_tables.tex

LaTeX表格，可直接复制到论文中：

```latex
\begin{table}[h]
\centering
\caption{Belief-based grouping statistics for ReBel}
\begin{tabular}{|l|r|}
\hline
Metric & Value \\
\hline
Number of Groups & 42 \\
Mean Group Size & 15.20 \\
...
\hline
\end{tabular}
\end{table}
```

## 关键指标解读

### 信念分组指标

| 指标 | 期望范围 | 含义 |
|------|---------|------|
| `num_groups` | 10-100 | 信念组数量，太少说明粒度太粗，太多说明粒度太细 |
| `mean_group_size` | 5-50 | 平均组大小，应有足够样本进行归一化 |
| `min_group_size` | ≥1 | 最小组大小，不应全为1（说明没有匹配） |
| `max_group_size` | <总步数/2 | 最大组大小，不应过大（说明分组太粗） |

**诊断：**
- ✅ 正常：`num_groups=42, mean_group_size=15.2`
- ⚠️ 粒度太细：`num_groups=847, mean_group_size=1.2` （类似GiGPO问题）
- ⚠️ 粒度太粗：`num_groups=3, mean_group_size=340` （类似RLVMR问题）

### 性能指标

| 指标 | 基线参考 | 含义 |
|------|---------|------|
| `success_rate` | >0.6 (L0) | 任务成功率，应高于BDRS基线(0.638) |
| `avg_steps` | <15 (L0) | 平均完成步数，越少越好 |
| `efficiency` | >0.04 | 效率指标，成功率/步数 |

### 奖励指标

| 指标 | 期望值 | 含义 |
|------|-------|------|
| `episode_reward_mean` | 0.5-1.0 | 任务奖励，通常为0或1 |
| `intrinsic_reward_mean` | 0.1-0.5 | 内在奖励均值，应为正 |
| `intrinsic_reward_std` | >0 | 内在奖励标准差，应有变化 |

## 常见问题排查

### 问题1：测试失败（status: failed）

**检查：**
```bash
# 查看错误
cat rebel_test_results/TIMESTAMP/training.log | grep -i error

# 查看完整日志
less rebel_test_results/TIMESTAMP/training.log
```

**常见原因：**
1. **模型路径不存在** - 检查 `actor_rollout_ref.model.path`
2. **GPU内存不足** - 减小 `gpu_memory_utilization` 或 `group_size`
3. **环境未配置** - 确保 `env.alfworld.use_rebel=True`

### 问题2：没有信念分组统计（num_groups: null）

**可能原因：**
1. 环境没有启用ReBel（`use_rebel=False`）
2. 信念状态解析失败
3. 日志中搜索不到分组信息

**检查：**
```bash
# 搜索分组信息
grep -A 20 "Belief-Based Grouping" rebel_test_results/TIMESTAMP/training.log

# 搜索信念状态
grep "belief_state" rebel_test_results/TIMESTAMP/training.log | head -20
```

### 问题3：所有组大小为1（mean_group_size: 1.0）

**原因：** 粒度太细，没有信念状态匹配

**解决：**
1. 使用更粗粒度：`belief_granularity='subgoal'`
2. 增加group size：`group_size=64`
3. 检查信念状态是否格式一致

### 问题4：成功率为0

**可能原因：**
1. 模型质量差（需要先做SFT cold-start）
2. 最大步数太少（`max_steps`太小）
3. 环境配置错误

**检查：**
```bash
# 查看是否有成功的episode
grep "success" rebel_test_results/TIMESTAMP/training.log
```

## 完整实验流程

### 实验1：ReBel基础验证

```bash
# 1. 运行测试
bash test_rebel_small.sh vllm ./checkpoints/sft_model

# 2. 分析结果
python3 analyze_rebel_results.py rebel_test_results/$(ls -t rebel_test_results/ | head -1)

# 3. 查看报告
cat rebel_test_results/$(ls -t rebel_test_results/ | head -1)/ANALYSIS_REPORT.md
```

### 实验2：粒度对比实验

创建测试脚本 `test_granularity.sh`：

```bash
#!/bin/bash
for granularity in subgoal medium fine; do
    echo "Testing granularity: $granularity"
    bash test_rebel_small.sh vllm ./checkpoints/sft_model $granularity
    sleep 10
done
```

### 实验3：与基线对比

```bash
# ReBel
bash test_rebel_small.sh vllm ./checkpoints/sft_model

# BDRS (修改test脚本，改为bdrs)
# GRPO (修改test脚本，改为grpo)
```

## 论文写作建议

### 1. 使用LaTeX表格

直接复制 `paper_tables.tex` 中的表格到论文：

```latex
% 在论文中
\input{rebel_test_results/TIMESTAMP/paper_tables.tex}
```

### 2. 引用关键数据

从 `ANALYSIS_REPORT.md` 中提取关键数据：

**示例：**
> "ReBel forms 42 belief groups with mean group size of 15.2, achieving a success rate of 67.2% on ALFWorld L0 tasks, outperforming BDRS (63.8%) and GiGPO (58.3%)."

### 3. 绘制对比图

导出数据用于绘图：

```python
import json
import matplotlib.pyplot as plt

# 加载结果
with open('rebel_test_results/TIMESTAMP/analysis_results.json') as f:
    results = json.load(f)

# 绘制组大小分布
dist = results['belief_grouping']['distribution']
sizes = [d['size'] for d in dist]
counts = [d['count'] for d in dist]

plt.bar(sizes, counts)
plt.xlabel('Group Size')
plt.ylabel('Count')
plt.title('Belief Group Size Distribution')
plt.savefig('group_distribution.pdf')
```

### 4. 结果对比表

| Method | Success Rate | Num Groups | Mean Group Size | Efficiency |
|--------|-------------|------------|-----------------|------------|
| GRPO | 52.7% | 1 | - | 3.14 |
| GiGPO | 58.3% | 847 | 1.2 | 3.67 |
| BDRS | 63.8% | 3 | 340 | 4.31 |
| **ReBel** | **67.2%** | **42** | **15.2** | **4.77** |

## 进阶使用

### 批量运行多个种子

```bash
#!/bin/bash
for seed in 0 1 2; do
    echo "Running with seed $seed"
    bash test_rebel_small.sh vllm ./checkpoints/sft_model $seed
done

# 分析所有结果
for dir in rebel_test_results/*/; do
    python3 analyze_rebel_results.py $dir
done
```

### 自动生成对比报告

```bash
# 收集所有结果
python3 <<EOF
import json
from pathlib import Path

results = []
for result_dir in Path('rebel_test_results').iterdir():
    result_file = result_dir / 'analysis_results.json'
    if result_file.exists():
        with open(result_file) as f:
            data = json.load(f)
            results.append({
                'timestamp': data['metadata']['timestamp'],
                'success_rate': data['performance'].get('success_rate', 0),
                'num_groups': data['belief_grouping'].get('num_groups', 0),
            })

# 打印汇总
print("| Timestamp | Success Rate | Num Groups |")
print("|-----------|--------------|------------|")
for r in results:
    print(f"| {r['timestamp']} | {r['success_rate']*100:.1f}% | {r['num_groups']} |")
EOF
```

## 输出文件总结

**主要输出目录：** `rebel_test_results/YYYYMMDD_HHMMSS/`

**关键文件：**
1. ✅ `test_results.json` - 基础结构化结果
2. ✅ `analysis_results.json` - 完整分析结果
3. ✅ `ANALYSIS_REPORT.md` - 人类可读报告
4. ✅ `paper_tables.tex` - 论文LaTeX表格
5. 📝 `training.log` - 完整训练日志（调试用）

**推荐工作流：**
1. 运行测试后，先查看 `ANALYSIS_REPORT.md` 了解概况
2. 如有问题，查看 `training.log` 调试
3. 写论文时，使用 `paper_tables.tex` 和 `analysis_results.json`
4. 需要绘图时，从 `analysis_results.json` 提取数据

---

**联系方式：** 如有问题，请查看完整日志或提issue
