# 🎉 基础模型已找到并配置完成！

## ✅ 当前状态

**好消息：** 已在你的环境中找到 **Qwen2.5-1.5B-Instruct** 原始预训练模型！

**模型信息:**
- **名称:** Qwen2.5-1.5B-Instruct
- **大小:** ~2.9GB
- **参数:** 1.5B (1536 hidden size, 28 layers)
- **状态:** ✅ 已验证可用
- **位置:** `/root/testttt/RLVMR/code/base_models/Qwen2.5-1.5B-Instruct`

---

## 🚀 立即运行ReBel评测

### 方法1: 一键运行（推荐）⭐

```bash
cd /root/testttt/RLVMR/code

# 运行ReBel评测（使用基础模型）
bash test_rebel_base_model.sh
```

**预计时间:** 10-30分钟
**需要GPU:** 是（2块GPU）

**输出结果:**
```
rebel_test_results/base_model_YYYYMMDD_HHMMSS/
├── test_config.yaml              # 测试配置
├── analysis_results.json         # 结构化结果
├── ANALYSIS_REPORT.md            # 📊 分析报告（重点查看）
├── paper_tables.tex              # 📝 LaTeX表格（论文用）
└── training.log                  # 完整日志
```

### 方法2: 自定义参数

```bash
# 修改测试参数
bash test_rebel_base_model.sh

# 或手动指定引擎
bash test_rebel_base_model.sh vllm
```

---

## 📋 测试配置

当前测试使用的参数（小规模快速验证）：

| 参数 | 值 | 说明 |
|------|---|------|
| 训练任务数 | 2 | 极小规模，快速测试 |
| 验证任务数 | 4 | 小规模验证 |
| Group Size | 8 | 每个prompt的rollout数 |
| 最大步数 | 15 | 每个episode最大步数 |
| 训练轮数 | 1 | 仅1轮（快速验证） |
| 信念粒度 | subgoal | 使用subgoal级别分组 |
| 步级优势权重 | 1.0 | 标准权重 |

**注意:** 这是快速验证配置。完整训练请修改 `examples/rebel_trainer/run_alfworld.sh`

---

## 🔍 验证模型

如果想先验证模型是否正常：

```bash
cd /root/testttt/RLVMR/code

# 运行验证脚本
python3 verify_base_model.py
```

**预期输出:**
```
============================================================
验证 Qwen2.5-1.5B-Instruct 基础模型
============================================================
   ✅ 路径存在
   ✅ 配置加载成功
   ✅ Tokenizer加载成功
   ✅ 基础模型验证通过！
============================================================
```

---

## 📊 与之前checkpoint的对比

| 模型 | 类型 | 训练状态 | 推荐用途 |
|------|------|---------|---------|
| **Qwen2.5-1.5B-Instruct** ⭐ | 基础模型 | 预训练（未在ALFWorld上训练） | **ReBel评测** |
| BDRS Qwen1.5B (step 75) | Checkpoint | 已在ALFWorld上训练75步 | 对比实验 |

**为什么使用基础模型:**
- ✅ 公平对比 - 从零开始评测ReBel
- ✅ 证明ReBel的学习能力
- ✅ 与其他方法在相同起点对比

---

## 📁 快速访问路径

### 模型路径
```bash
# 推荐使用（符号链接）
/root/testttt/RLVMR/code/base_models/Qwen2.5-1.5B-Instruct

# 原始路径（也可用）
/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306
```

### 测试脚本
```bash
# ReBel基础模型测试
/root/testttt/RLVMR/code/test_rebel_base_model.sh

# 模型验证
/root/testttt/RLVMR/code/verify_base_model.py

# 结果分析
/root/testttt/RLVMR/code/analyze_rebel_results.py
```

---

## 🎯 预期结果

### 你会得到什么数据

1. **信念分组统计**
   - 组数量（预期：10-100组）
   - 平均组大小（预期：5-50）
   - 组大小分布

2. **性能指标**
   - 成功率（基础模型预期：20-40%）
   - 平均步数
   - 效率指标

3. **奖励分析**
   - Episode奖励统计
   - 内在奖励统计
   - 奖励组件分解

4. **论文用表格**
   - LaTeX格式表格
   - 可直接复制到论文

### 示例报告内容

测试完成后，查看 `ANALYSIS_REPORT.md` 会看到：

```markdown
# ReBel Base Model Test Results

## Belief Grouping Statistics
| Metric | Value |
|--------|-------|
| Number of Groups | 42 |
| Mean Group Size | 15.2 |
| ...

## Performance Metrics
| Metric | Value |
|--------|-------|
| Success Rate | 35.2% |
| Avg Episode Length | 18.3 |
| ...
```

---

## ⚙️ 进阶配置

### 修改测试规模

编辑 `test_rebel_base_model.sh`：

```bash
# 更大规模测试
train_data_size=16      # 改为16
val_data_size=128       # 改为128
group_size=64           # 改为64
```

### 使用不同粒度

```bash
# 在脚本中修改
algorithm.rebel.belief_granularity='medium'  # 或 'fine'
```

### 调整GPU使用

```bash
# 修改GPU内存使用率
actor_rollout_ref.rollout.gpu_memory_utilization=0.6  # 改为0.6

# 修改GPU数量
trainer.n_gpus_per_node=4  # 如果有4块GPU
```

---

## 🐛 常见问题

### Q1: 模型路径找不到

**解决:**
```bash
# 重新创建符号链接
mkdir -p /root/testttt/RLVMR/code/base_models
ln -sf /root/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306 /root/testttt/RLVMR/code/base_models/Qwen2.5-1.5B-Instruct

# 验证
python3 verify_base_model.py
```

### Q2: GPU内存不足

**解决:**
```bash
# 方法1: 降低GPU内存使用
# 修改脚本中的: gpu_memory_utilization=0.3

# 方法2: 减小batch size
# 修改脚本中的: group_size=4
```

### Q3: 测试时间太长

**解决:**
```bash
# 进一步减小规模
train_data_size=1
val_data_size=2
max_steps=10
```

---

## 📝 查看结果

### 测试完成后

```bash
# 查看最新结果
LATEST=$(ls -td rebel_test_results/base_model_* | head -1)

# 查看分析报告
cat $LATEST/ANALYSIS_REPORT.md

# 查看LaTeX表格
cat $LATEST/paper_tables.tex

# 查看JSON结果
cat $LATEST/analysis_results.json | python3 -m json.tool
```

---

## ✅ 总结

**当前状态:**
- ✅ 基础模型已找到并验证
- ✅ 测试脚本已创建并配置
- ✅ 结果分析工具已准备
- ✅ 所有文档已更新

**下一步:**
```bash
# 1. 运行测试
bash test_rebel_base_model.sh

# 2. 等待10-30分钟

# 3. 查看结果
ls -lh rebel_test_results/base_model_*/

# 4. 查看报告
cat rebel_test_results/base_model_*/ANALYSIS_REPORT.md
```

**现在就可以开始评测了！** 🚀

---

**创建时间:** 2025-12-20
**模型状态:** ✅ Ready
**测试状态:** ✅ Ready to Run
