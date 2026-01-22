# ReBel标注完整运行指南

## 快速开始（3步）

### 1. 运行标注向导

```bash
bash run_rebel_annotation.sh
```

这个脚本会引导你完成：
- ✅ 环境检查
- ✅ 选择Teacher模型
- ✅ 设置参数
- ✅ 运行标注
- ✅ 自动质量检验

### 2. 查看质量报告

```bash
cat data/alfworld_rebel_hindsight/quality_report.txt
```

### 3. 检查示例数据

```bash
head -100 data/alfworld_rebel_hindsight/rebel_hindsight.jsonl | less
```

---

## 推荐Teacher模型配置

### 方案1: GPT-4 (最高质量) ⭐⭐⭐⭐⭐

**适用场景**: 生成最终黄金数据集，预算充足

```bash
export OPENAI_API_KEY="sk-..."

python generate_rebel_hindsight.py \
    --expert_data data/alfworld_expert_traj.json \
    --output_dir data/rebel_gpt4 \
    --num_samples 100 \
    --teacher_model_url https://api.openai.com/v1 \
    --teacher_model_name gpt-4 \
    --temperature 0.2
```

**预期质量**:
- 标注成功率: 95-98%
- 最终得分: 85-92%
- 成本: ~$30-50/100样本 (~300-500 RMB)

---

### 方案2: Qwen2.5-72B (高质量本地) ⭐⭐⭐⭐

**适用场景**: 大规模数据集，有GPU资源

**启动vLLM服务器**:
```bash
# 需要 2x A100 (80GB) 或 4x A6000 (48GB)
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-72B-Instruct \
    --port 8000 \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.95
```

**运行标注**:
```bash
python generate_rebel_hindsight.py \
    --expert_data data/alfworld_expert_traj.json \
    --output_dir data/rebel_qwen72b \
    --num_samples 100 \
    --teacher_model_url http://127.0.0.1:8000/v1 \
    --teacher_model_name Qwen2.5-72B-Instruct \
    --temperature 0.3
```

**预期质量**:
- 标注成功率: 85-92%
- 最终得分: 78-86%
- 速度: ~2-3秒/样本
- 成本: 仅GPU时间

---

### 方案3: Qwen2.5-32B (性价比最佳) ⭐⭐⭐⭐

**适用场景**: 中等规模数据集，GPU资源有限

**启动vLLM服务器**:
```bash
# 需要 1x A100 (80GB) 或 2x A6000 (48GB)
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-32B-Instruct \
    --port 8000 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.95
```

**运行标注**:
```bash
python generate_rebel_hindsight.py \
    --expert_data data/alfworld_expert_traj.json \
    --output_dir data/rebel_qwen32b \
    --num_samples 100 \
    --teacher_model_url http://127.0.0.1:8000/v1 \
    --teacher_model_name Qwen2.5-32B-Instruct \
    --temperature 0.3
```

**预期质量**:
- 标注成功率: 80-88%
- 最终得分: 73-82%
- 速度: ~1-2秒/样本

---

### 方案4: Qwen2.5-14B (快速测试) ⭐⭐⭐

**适用场景**: 小规模测试，单卡GPU

**启动vLLM服务器**:
```bash
# 需要 1x A6000 (48GB) 或 RTX 6000 Ada
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-14B-Instruct \
    --port 8000
```

**运行标注**:
```bash
python generate_rebel_hindsight.py \
    --expert_data data/alfworld_expert_traj.json \
    --output_dir data/rebel_qwen14b \
    --num_samples 50 \
    --teacher_model_url http://127.0.0.1:8000/v1 \
    --teacher_model_name Qwen2.5-14B-Instruct \
    --temperature 0.3
```

**预期质量**:
- 标注成功率: 75-85%
- 最终得分: 68-78%
- 速度: ~0.5-1秒/样本

---

## 质量检验标准

### 自动检验

脚本会自动检查7个方面：

1. **标注成功率** (权重30%)
   - ✅ 优秀: >90%
   - ⚠️ 良好: 70-90%
   - ❌ 需改进: <70%

2. **Belief完整性** (权重20%)
   - 检查是否包含所有必需字段
   - world_model, task_progress, exploration_map

3. **Inventory追踪** (权重15%)
   - 检查take动作后inventory是否更新
   - 检查put动作后inventory是否清空
   - 错误率应 <5%

4. **Cleared逻辑** (权重15%)
   - 检测"open X -> go to Y"模式
   - 覆盖率应 >80%

5. **Reasoning质量** (权重10%)
   - 平均长度: 100-300字符
   - 包含证据/观测
   - 提及任务目标
   - 说明策略/原因

6. **格式正确性** (权重10%)
   - 检查XML标签完整性
   - 检查JSON有效性
   - 检查交替模式

7. **综合得分**
   - ✅ ≥85%: 可直接用于训练
   - ⚠️ 70-85%: 建议检查低分项
   - ⚠️ 55-70%: 建议用更强模型
   - ❌ <55%: 必须重新标注

### 手动检验

**查看单个样本**:
```python
import json
with open('data/rebel_hindsight/rebel_hindsight.jsonl') as f:
    sample = json.loads(f.readline())

# 查看对话
for turn in sample['conversations'][2:6]:  # 跳过系统消息
    print(f"\n{turn['from']}:")
    print(turn['value'][:500])
```

**检查要点**:
1. ✅ Human Turn包含: Task + Observation + Belief + Actions
2. ✅ GPT Turn包含: <belief> + <reasoning> + <action>
3. ✅ Belief JSON格式正确，包含inventory字段
4. ✅ Reasoning解释合理，长度适中
5. ✅ Action与Available Actions匹配

---

## 常见问题排查

### Q1: 标注成功率低 (<70%)

**可能原因**:
- Teacher模型太弱
- Temperature太高导致输出不稳定
- Prompt与模型不匹配

**解决方案**:
```bash
# 方案1: 使用更强的模型
# Qwen2.5-7B → Qwen2.5-32B 或 GPT-4

# 方案2: 降低temperature
python generate_rebel_hindsight.py ... --temperature 0.1

# 方案3: 查看失败案例
grep "⚠️" logs.txt | head -10
```

### Q2: Inventory错误率高 (>10%)

**可能原因**:
- Teacher LLM不理解inventory逻辑
- Prompt引导不够强

**解决方案**:
1. 检查是否是自动fallback修正的
2. 如果自动修正率高(>15%)，说明LLM能力不足
3. 使用更强模型或手动调整prompt

### Q3: Cleared覆盖率低 (<50%)

**可能原因**:
- 数据集不涉及搜索任务
- LLM未能识别模式

**检查**:
```bash
# 查看是否有open->go模式
grep -A1 "open" data/rebel_hindsight/rebel_hindsight.jsonl | grep "go to" | wc -l
```

### Q4: JSON解析错误频繁

**可能原因**:
- LLM输出格式不规范
- Temperature太高

**解决方案**:
```bash
# 降低temperature到0.1
python generate_rebel_hindsight.py ... --temperature 0.1

# 或使用更强的模型
```

### Q5: 速度太慢

**优化方案**:
```bash
# 1. 使用更小的模型测试
# Qwen2.5-72B → Qwen2.5-32B

# 2. 增加vLLM批处理大小
python -m vllm.entrypoints.openai.api_server \
    --max-model-len 4096 \
    --max-num-seqs 8  # 增加并发

# 3. 启用量化
python -m vllm.entrypoints.openai.api_server \
    --quantization awq  # 或 gptq
```

---

## 质量达标判断流程图

```
开始
  ↓
运行标注 + 质量检验
  ↓
最终得分 ≥ 85%? ──YES──> ✅ 可直接用于训练 ──> 结束
  ↓ NO
  ↓
最终得分 ≥ 70%? ──YES──> 检查低分项
  ↓                         ↓
  NO                    可修复? ──YES──> 手动修正 ──> 结束
  ↓                         ↓ NO
  ↓                         ↓
最终得分 ≥ 55%? ──YES──> 使用更强Teacher模型 ──> 重新标注
  ↓ NO
  ↓
❌ 检查数据和配置
  ↓
修正后重新标注
```

---

## 示例：完整工作流

```bash
# 1. 启动vLLM (使用Qwen2.5-32B)
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-32B-Instruct \
    --port 8000 \
    --tensor-parallel-size 1 &

# 等待启动完成
sleep 30

# 2. 运行标注
python generate_rebel_hindsight.py \
    --expert_data data/alfworld_expert_traj.json \
    --output_dir data/rebel_hindsight \
    --num_samples 100 \
    --teacher_model_url http://127.0.0.1:8000/v1 \
    --teacher_model_name Qwen2.5-32B-Instruct \
    --temperature 0.3

# 3. 质量检验
python verify_annotation_quality.py \
    --annotated_data data/rebel_hindsight \
    --output_report data/rebel_hindsight/quality_report.txt

# 4. 查看报告
cat data/rebel_hindsight/quality_report.txt

# 5. 如果得分≥70%，可用于训练
# 如果得分<70%，使用更强模型重新标注
```

---

## 进阶：批量处理大规模数据

```bash
#!/bin/bash
# 处理1000个样本，分批进行

for batch in {0..9}; do
    start=$((batch * 100))

    python generate_rebel_hindsight.py \
        --expert_data data/alfworld_expert_traj.json \
        --output_dir data/rebel_batch_${batch} \
        --num_samples 100 \
        --offset $start \
        --teacher_model_url http://127.0.0.1:8000/v1 \
        --teacher_model_name Qwen2.5-32B-Instruct

    # 检验质量
    python verify_annotation_quality.py \
        --annotated_data data/rebel_batch_${batch} \
        --output_report data/rebel_batch_${batch}/report.txt

    # 检查得分
    score=$(grep "最终质量得分" data/rebel_batch_${batch}/report.txt | awk '{print $NF}')
    echo "Batch $batch: $score"

    sleep 5
done

# 合并所有批次
python merge_datasets.py --input_dirs data/rebel_batch_* --output data/rebel_full
```

---

## 总结

### 推荐配置

| 场景 | Teacher模型 | 预期质量 | 成本 |
|------|------------|---------|------|
| 最终生产数据 | GPT-4 | 90%+ | 高 |
| 大规模本地 | Qwen2.5-72B | 85%+ | 中 |
| 性价比选择 | Qwen2.5-32B | 80%+ | 低 |
| 快速测试 | Qwen2.5-14B | 75%+ | 极低 |

### 质量检验清单

- [ ] 标注成功率 ≥ 80%
- [ ] Belief完整性 ≥ 90%
- [ ] Inventory准确性 ≥ 90%
- [ ] Cleared覆盖率 ≥ 70%
- [ ] Reasoning质量 ≥ 75%
- [ ] 格式正确率 = 100%
- [ ] **最终得分 ≥ 75%**

全部达标即可用于训练！
