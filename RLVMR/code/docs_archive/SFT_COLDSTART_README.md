# ReBel SFT 冷启动训练和评测指南

本指南介绍如何使用合并后的426条高质量ReBel轨迹进行SFT冷启动训练和评测。

## 📊 数据集信息

**位置**: `/root/testttt/RLVMR/code/data/alfworld_rebel_merged_final/`

**统计信息**:
- 总样本数: 426 条
- 成功率: 100% (所有轨迹都是成功的)
- 平均步数: 12.7 步
- 数据来源:
  - alfworld_rebel_250_new: 250条 (新生成的高质量轨迹)
  - alfworld_rebel_full_improved: 176条 (从2224条中筛选的成功轨迹)

**任务类型分布**:
- put: 337条 (79.1%)
- cool: 32条 (7.5%)
- clean: 25条 (5.9%)
- find: 18条 (4.2%)
- heat: 14条 (3.3%)

**文件格式**:
- `rebel_coldstart.json` - 用于SFT训练的原始轨迹格式
- `rebel_hindsight.jsonl` - 对话格式（也可用于SFT）
- `data-00000-of-00001.arrow` - HuggingFace Datasets格式

## 🚀 快速开始

### 方法1: 一键运行（推荐）

```bash
# 完整流程：数据预处理 -> SFT训练 -> 评测
bash quick_start_sft_426.sh
```

### 方法2: 分步运行

#### 步骤1: SFT 训练

```bash
# 训练SFT冷启动模型
bash run_sft_coldstart_426.sh
```

**训练配置**:
- 基础模型: Qwen2.5-1.5B-Instruct
- 训练轮数: 5 epochs
- 学习率: 1e-5
- GPU数量: 8
- Batch size: 16 per GPU
- 预计时间: ~2-3小时

**输出**:
- Checkpoint保存在: `./checkpoints/cold_start/alfworld/qwen1.5b_rebel_426/`
- 日志: 终端输出 + WandB

#### 步骤2: 评测

```bash
# 评测训练好的模型
bash eval_sft_coldstart_426.sh <checkpoint_path> <num_tasks>

# 示例：评测134个任务
bash eval_sft_coldstart_426.sh ./checkpoints/cold_start/alfworld/qwen1.5b_rebel_426/global_step_75 134
```

**评测配置**:
- 任务数量: 134 (ALFWorld测试集)
- Temperature: 0.0 (确定性推理)
- GPU数量: 2
- 预计时间: ~30分钟

## 📁 脚本说明

### 1. `run_sft_coldstart_426.sh`
SFT训练脚本，包含：
- 数据预处理（将coldstart格式转为parquet）
- SFT训练（使用FSDP）

### 2. `eval_sft_coldstart_426.sh`
评测脚本，支持参数：
- `$1`: checkpoint路径
- `$2`: 评测任务数量（默认134）
- `$3`: 推理引擎（默认vllm）
- `$4`: GPU数量（默认2）

### 3. `quick_start_sft_426.sh`
一键运行脚本，自动执行完整流程

## 📝 训练流程详解

### 1. 数据预处理

脚本 `examples/data_preprocess/cold_start_data.py` 会：
- 读取 `rebel_coldstart.json`
- 提取每个轨迹的 `data` 字段（包含所有步骤）
- 转换为训练格式：
  ```python
  {
    "prompt": [{"role": "user", "content": "..."}],
    "extra_info": {
      "question": "<prompt>",
      "answer": "<response>"
    }
  }
  ```
- 保存为 `train.parquet`

### 2. SFT训练

使用 `verl.trainer.fsdp_sft_trainer`：
- 加载 Qwen2.5-1.5B-Instruct
- 在426条轨迹的所有步骤上训练
- 使用FSDP进行分布式训练
- 每个epoch保存checkpoint

### 3. 评测

使用 `verl.trainer.main_ppo` 的评测模式：
- 在134个ALFWorld测试任务上评测
- 记录成功率、平均步数等指标
- 输出详细的任务完成情况

## 🔧 自定义配置

### 修改训练参数

编辑 `run_sft_coldstart_426.sh`:

```bash
# 修改学习率
optim.lr=1e-5  # 改为 2e-5 或其他值

# 修改训练轮数
trainer.total_epochs=5  # 改为 3 或 10

# 修改batch size
data.micro_batch_size_per_gpu=16  # 改为 8 或 32

# 修改GPU数量
torchrun --nproc_per_node=8  # 改为 4 或其他值
```

### 修改评测参数

编辑 `eval_sft_coldstart_426.sh`:

```bash
# 修改评测任务数量
NUM_TASKS=${2:-134}  # 改为更少的任务（如10）用于快速测试

# 修改温度
actor_rollout_ref.rollout.val_kwargs.temperature=0.0  # 改为 0.3 增加多样性

# 修改采样
actor_rollout_ref.rollout.val_kwargs.do_sample=False  # 改为 True
```

## 📊 预期结果

### SFT训练后的预期性能

基于426条高质量轨迹训练，预期在ALFWorld测试集上：
- 成功率: 40-50% (冷启动基线)
- 平均步数: 15-20步
- 这是RL训练的良好起点

### 对比基线

- 基础Qwen2.5-1.5B: ~5-10% 成功率
- 250条轨迹训练: ~35-45% 成功率
- **426条轨迹训练（当前）**: 预期 ~40-50% 成功率
- RL训练后: 目标 60-70%+ 成功率

## 🐛 故障排查

### 1. 数据预处理失败

```bash
# 检查数据文件是否存在
ls -lh /root/testttt/RLVMR/code/data/alfworld_rebel_merged_final/rebel_coldstart.json

# 验证JSON格式
python3 -c "import json; json.load(open('/root/testttt/RLVMR/code/data/alfworld_rebel_merged_final/rebel_coldstart.json'))"
```

### 2. GPU内存不足

减少batch size或GPU数量：
```bash
# 在 run_sft_coldstart_426.sh 中修改
data.micro_batch_size_per_gpu=8  # 从16减少到8
torchrun --nproc_per_node=4      # 从8减少到4
```

### 3. 找不到checkpoint

```bash
# 列出所有checkpoints
find ./checkpoints/cold_start/alfworld/qwen1.5b_rebel_426 -name "global_step_*"

# 使用具体路径
bash eval_sft_coldstart_426.sh ./checkpoints/cold_start/alfworld/qwen1.5b_rebel_426/global_step_150
```

### 4. 评测速度慢

```bash
# 先用少量任务测试
bash eval_sft_coldstart_426.sh <checkpoint> 10  # 只评测10个任务

# 确认正常后再运行完整评测
bash eval_sft_coldstart_426.sh <checkpoint> 134
```

## 📈 后续步骤

### 1. RL训练（可选）

如果SFT效果满意，可以进行ReBel RL训练：

```bash
cd examples/rebel_trainer
# 修改 run_alfworld.sh 中的 COLD_START_MODEL_PATH
bash run_alfworld.sh
```

### 2. 更大模型

训练7B模型获得更好性能：

```bash
# 复制并修改脚本
cp run_sft_coldstart_426.sh run_sft_coldstart_426_7b.sh

# 修改模型路径
model.partial_pretrain=Qwen/Qwen2.5-7B-Instruct

# 调整GPU和batch size
torchrun --nproc_per_node=8
data.micro_batch_size_per_gpu=4  # 7B模型需要更小的batch
```

### 3. 数据增强

可以继续生成更多高质量轨迹并合并。

## 📚 相关文档

- `REBEL_COLD_START_GUIDE.md` - ReBel冷启动详细指南
- `REBEL_IMPLEMENTATION_SUMMARY.md` - ReBel实现总结
- `data/alfworld_rebel_merged_final/FINAL_REPORT.txt` - 数据集合并报告

## 💡 提示

1. **首次运行建议**: 先用少量数据/任务测试流程
2. **监控训练**: 关注WandB中的loss下降曲线
3. **Checkpoint选择**: 通常最后一个checkpoint效果最好
4. **评测多样性**: 可以用不同温度评测多次
5. **保存结果**: 评测输出建议重定向到文件保存

---

**创建时间**: 2025-12-23
**数据版本**: alfworld_rebel_merged_final (426条)
**脚本版本**: v1.0
