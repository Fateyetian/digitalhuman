# BDRS冷启动数据生成快速指南（方案A）

## 🎯 这是什么？

这是BDRS冷启动数据生成的快速启动指南。使用**方案A（快速验证）**，您可以在30分钟内生成高质量的冷启动数据。

## ✅ 已完成的优化

1. ✅ **Prompt优化**：已使用增强的BDRS prompt（包含结构化信念示例）
2. ✅ **防止hindsight bias**：添加了明确的标注指导
3. ✅ **信念上下文提示**：要求在推理中包含信念状态
4. ✅ **数据质量检查**：自动生成质量报告

## 📋 前置要求

### 1. 检查环境

```bash
# 确保在code目录
cd F:\1.Git-Repository\RLVR\digitalhuman\RLVMR\code

# 检查Python环境
python -c "import openai; print('OpenAI library OK')"

# 检查数据是否存在
python -c "
from datasets import load_from_disk
dataset = load_from_disk('data/alfworld_expert_traj')
print(f'✓ Found {len(dataset)} expert trajectories')
"
```

### 2. 设置OpenAI API Key

```bash
# 方法1：临时设置（推荐）
export OPENAI_API_KEY="sk-your-api-key-here"

# 方法2：修改脚本
# 编辑 scripts/alfworld_prepare.py 第13行
# openai.api_key = "YOUR_OPENAI_API_KEY"  # 改为你的API key
```

### 3. 选择模型（可选）

**默认**：GPT-4o（最高质量，但较贵）

**替代**：GPT-4o-mini（便宜10倍，质量略低）

```python
# 编辑 scripts/alfworld_prepare.py 第14行
MODEL = "gpt-4o-mini"  # 改为gpt-4o-mini节省成本
```

**成本对比**：
- GPT-4o：300条轨迹约 $15-30
- GPT-4o-mini：300条轨迹约 $1.5-3

## 🚀 快速开始（3步走）

### Step 1: 生成冷启动数据（1小时）

```bash
cd F:\1.Git-Repository\RLVR\digitalhuman\RLVMR\code

# 确保API key已设置
echo $OPENAI_API_KEY

# 运行数据生成脚本
python scripts/alfworld_prepare.py

# 输出：
# data/alfworld_cold-start.json
# 约300条轨迹，每条10-25步
```

**预计时间**：
- GPT-4o：60-90分钟（取决于API速度）
- GPT-4o-mini：45-60分钟

**实时监控**：
```bash
# 在另一个终端查看进度
tail -f nohup.out  # 如果后台运行

# 或直接看屏幕输出
# [1/300] Processing trajectory... ✓ Success (15 steps)
# [2/300] Processing trajectory... ✓ Success (12 steps)
# ...
```

### Step 2: 转换为训练格式（1分钟）

```bash
cd F:\1.Git-Repository\RLVR\digitalhuman\RLVMR\code

# 转换为parquet格式
python -m examples.data_preprocess.cold_start_data \
    --local_dir=$HOME/data/alfworld \
    --data_source=data/alfworld_cold-start.json

# 输出：
# $HOME/data/alfworld/train.parquet
# $HOME/data/alfworld/val.parquet
```

### Step 3: 训练冷启动模型（1-2小时）

```bash
cd F:\1.Git-Repository\RLVR\digitalhuman\RLVMR\code

# 8卡训练（推荐Qwen2.5-7B）
torchrun --standalone --nnodes=1 --nproc_per_node=8 \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$HOME/data/alfworld/train.parquet \
    data.val_files=$HOME/data/alfworld/val.parquet \
    data.prompt_key=extra_info \
    data.response_key=extra_info \
    data.max_length=3000 \
    +data.prompt_dict_keys=['question'] \
    +data.response_dict_keys=['answer'] \
    optim.lr=1e-5 \
    data.micro_batch_size_per_gpu=16 \
    model.partial_pretrain=Qwen/Qwen2.5-7B-Instruct \
    trainer.default_hdfs_dir=null \
    trainer.project_name=BDRS \
    trainer.experiment_name=qwen7b_cold_start_bdrs_plan_a \
    trainer.total_epochs=5 \
    trainer.default_local_dir=./checkpoints/cold_start/alfworld/qwen7b_plan_a \
    trainer.logger=['console','wandb'] \
    ulysses_sequence_parallel_size=4 \
    use_remove_padding=true

# 模型保存在：
# ./checkpoints/cold_start/alfworld/qwen7b_plan_a/default/epoch_5
```

**时间估算**：
- Qwen2.5-7B + 8×A100：1-1.5小时
- Qwen2.5-1.5B + 8×A100：30-45分钟

## 📊 数据质量检查

生成完数据后，脚本会自动输出质量报告：

```
============================================================
Data Quality Report
============================================================

Total steps: 4523

Mode Distribution:
  EXECUTE : 2461 ( 54.4%)
  EXPLORE : 1131 ( 25.0%)
  PLAN    :  678 ( 15.0%)
  VERIFY  :  253 (  5.6%)

Trajectory Length Statistics:
  Average: 15.1 steps
  Min: 8 steps
  Max: 28 steps

Task Type Coverage: 6 types
  - look_at_obj
  - pick_and_place
  - pick_clean_then_place
  - pick_cool_then_place
  - pick_heat_then_place
  - pick_two_obj_and_place

Format Errors: 0 / 4523 (0.00%)

============================================================
Quality Assessment
============================================================

✅ Data quality looks good!
   - Mode distribution is balanced
   - No format errors detected
   - Ready for cold start training
```

### 如何解读？

**✅ 好的信号**：
- EXECUTE: 50-60%
- EXPLORE: 20-30%
- PLAN: 10-20%
- VERIFY: 5-10%
- Format errors: <5%
- 覆盖6种任务类型

**⚠️ 警告信号**：
- EXECUTE > 65%：分布不平衡
- EXPLORE < 15%：探索样本不足
- VERIFY < 5%：验证样本不足
- Format errors > 5%：LLM输出有问题

## 🎯 评估冷启动模型

```bash
cd F:\1.Git-Repository\RLVR\digitalhuman\RLVMR\code

# 在验证集上测试
bash examples/bdrs_trainer/eval_alfworld.sh \
    actor_rollout_ref.model.path=./checkpoints/cold_start/alfworld/qwen7b_plan_a/default/epoch_5 \
    env.alfworld.generalization_level=0 \
    data.val_batch_size=64

# 预期结果（方案A）：
# Success Rate: 15-25%
# Valid Action Ratio: 88-92%
```

### 决策点：是否需要方案C？

**如果成功率 ≥ 20%**：✅ 数据质量足够好，继续RL训练
**如果成功率 15-20%**：⚠️ 可用，但建议尝试方案C
**如果成功率 < 15%**：❌ 需要实施方案C（见BDRS_COLD_START_PLAN_C.md）

## 🐛 常见问题

### Q1: API调用失败

```
Error: RateLimitError: You exceeded your current quota
```

**解决**：
- 检查API key是否有效
- 检查账户余额
- 降低并发（脚本默认串行调用）

### Q2: 生成的数据格式错误

```
Format Errors: 450 / 4523 (9.95%)
```

**解决**：
1. 检查是否使用了正确的模型（GPT-4o或GPT-4o-mini）
2. 降低temperature（脚本中llm_json的temperature参数）
3. 增加重试次数（脚本中retries=5改为retries=10）

### Q3: 模式分布不平衡

```
EXECUTE: 3200 (70.8%)  # 太高
EXPLORE: 600 (13.3%)   # 太低
```

**解决**：
- 方案1：手动调整prompt中的模式选择指导
- 方案2：实施方案C的模式平衡采样

### Q4: 训练OOM（显存不足）

```
RuntimeError: CUDA out of memory
```

**解决**：
```bash
# 减小batch size
data.micro_batch_size_per_gpu=8  # 从16改为8

# 或使用小模型
model.partial_pretrain=Qwen/Qwen2.5-1.5B-Instruct

# 或启用gradient checkpointing
model.enable_gradient_checkpointing=True
```

## 📈 后续步骤

### 1. 使用冷启动模型进行RL训练

```bash
cd F:\1.Git-Repository\RLVR\digitalhuman\RLVMR\code

# BDRS训练
bash examples/bdrs_trainer/run_alfworld.sh \
    actor_rollout_ref.model.path=./checkpoints/cold_start/alfworld/qwen7b_plan_a/default/epoch_5 \
    trainer.experiment_name=bdrs_qwen7b_L0_from_plan_a

# 预期：
# - 收敛epoch：60-70（比无冷启动的100+快30-40%）
# - 最终成功率：82-85%
```

### 2. 对比实验

建议运行对比实验：

| 实验 | 冷启动模型 | 预期成功率 | 收敛epoch |
|------|-----------|-----------|-----------|
| **Baseline** | 无 | 75% | 100+ |
| **方案A冷启动** | plan_a | 82-85% | 60-70 |
| **方案C冷启动** | plan_c | 85-88% | 45-55 |

### 3. 如果效果不理想

**如果冷启动成功率 < 15%**：
1. 检查数据质量报告
2. 随机抽查10条样本，人工评估
3. 考虑实施方案C（见BDRS_COLD_START_PLAN_C.md）

**如果RL训练收敛慢或成功率低**：
1. 检查BDRS奖励是否正常计算
2. 调整BDRS奖励权重
3. 增加训练epoch数

## 📚 相关文档

- **BDRS实验计划**：BDRS_EXPERIMENT_PLAN.md
- **完整优化方案**：BDRS_COLD_START_PLAN_C.md
- **项目文档**：CLAUDE.md
- **快速问题修复**：QUICK_FIXES.md

## 🎉 总结

方案A快速优化已完成！主要改进：

1. ✅ 使用优化后的BDRS prompt（结构化信念示例）
2. ✅ 添加防止hindsight bias的明确指导
3. ✅ 要求在推理中包含信念上下文
4. ✅ 自动数据质量检查和报告

**预期效果**：
- 冷启动模型成功率：15-25%（基线10-15%）
- 数据生成时间：1小时
- 训练时间：1-2小时
- 总成本：$15-30（GPT-4o）或$1.5-3（GPT-4o-mini）

**下一步**：立即运行 `python scripts/alfworld_prepare.py` 开始生成数据！

---

**问题反馈**：如有任何问题，请查看本文档的"常见问题"章节或参考BDRS_EXPERIMENT_PLAN.md第9节"故障排除"。
