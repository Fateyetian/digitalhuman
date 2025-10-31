# BDRS (Belief-Driven Reward Shaping) 完整实验方案

## 目录
- [1. 实验概述](#1-实验概述)
- [2. 关于冷启动](#2-关于冷启动)
- [3. 实验环境准备](#3-实验环境准备)
- [4. 实验阶段规划](#4-实验阶段规划)
- [5. 服务器部署指南](#5-服务器部署指南)
- [6. 实验执行手册](#6-实验执行手册)
- [7. 监控与调试](#7-监控与调试)
- [8. 结果分析](#8-结果分析)
- [9. 故障排除](#9-故障排除)

---

## 1. 实验概述

### 1.1 实验目标
验证 BDRS（信念驱动奖励塑造）相比 RLVMR 的改进效果：
- **核心改进**：引入内部信念状态（世界模型、任务进展、探索地图）
- **密集奖励**：通过三种奖励分量提供逐步反馈
  - R_consistency：世界模型一致性奖励
  - R_progress：任务进展奖励
  - R_explore：探索效率奖励

### 1.2 对比实验
| 方法 | 奖励类型 | 特点 | 实验组 |
|------|----------|------|--------|
| **Vanilla PPO** | 环境稀疏奖励 | 基线方法 | 控制组1 |
| **RLVMR** | 稀疏奖励 + meta-reasoning标签 | 原方法 | 控制组2 |
| **BDRS** | 稀疏奖励 + 密集信念奖励 | 新方法 | 实验组 |

### 1.3 评估指标
- **主要指标**：
  - 成功率 (Success Rate)
  - 平均episode长度
  - 有效动作比例 (Valid Action Ratio)
  - 训练收敛速度（达到目标成功率的epoch数）

- **BDRS特有指标**：
  - 世界一致性奖励均值/方差
  - 任务进展奖励均值/方差
  - 探索效率奖励均值/方差
  - 信念修正频率
  - 子目标完成率

---

## 2. 关于冷启动

### 2.1 什么是冷启动？

**冷启动 (Cold Start)** 是指在强化学习训练之前，使用**监督微调 (SFT)** 让模型先学习基本的任务执行能力。这是一个**可选但强烈推荐**的步骤。

**为什么需要冷启动？**
1. **加速收敛**：从零开始的RL训练会有大量无效探索，冷启动可以让模型快速学会基本动作
2. **提升稳定性**：避免训练初期的崩溃和振荡
3. **提高成功率**：冷启动后的模型初始性能更好，后续RL训练更容易提升

**RLVMR vs BDRS 的冷启动区别**：
- **RLVMR冷启动**：使用 `<planning>/<explore>/<reflection>/<monitor>` 标签教模型meta-reasoning
- **BDRS冷启动**：使用 `<PLAN>/<EXECUTE>/<EXPLORE>/<VERIFY>` 标签教模型信念驱动推理

### 2.2 是否需要冷启动？

| 场景 | 是否需要 | 说明 |
|------|----------|------|
| **首次训练** | **强烈推荐** | 可以大幅减少训练时间（从100+ epochs降到30-50 epochs） |
| **已有RLVMR模型** | 可选 | 可以直接用RLVMR检查点继续训练BDRS |
| **快速验证** | 不需要 | 如果只是测试代码和数据流，可以跳过 |
| **对比实验** | **必须一致** | 如果RLVMR用了冷启动，BDRS也必须用；反之亦然 |

### 2.3 冷启动数据准备

**BDRS需要的冷启动数据格式**：
```json
{
  "data": [
    {
      "prompt": "You are an expert agent...",
      "response": "<PLAN>First find the apple, then heat it, finally put in fridge.</PLAN>\n<action>go to counter 1</action>"
    },
    ...
  ]
}
```

**两种获取方式**：

#### 方式1：从成功轨迹生成（推荐）
```bash
# 需要OpenAI API Key用于自动标注
cd code
export OPENAI_API_KEY="sk-..."

# 使用BDRS模板生成冷启动数据
python scripts/alfworld_prepare.py
# 输出: data/alfworld_cold-start.json (约300条标注轨迹)

# 转换为parquet格式
python -m examples.data_preprocess.cold_start_data \
    --local_dir=$HOME/data/alfworld \
    --data_source=data/alfworld_cold-start.json
```

**注意**：`alfworld_prepare.py` 中默认使用RLVMR的标签模板，你需要修改为BDRS模板：
```python
# 第6行，修改导入
from agent_system.environments.prompts.cold_start import ALFWORLD_TAGGING_TEMPLATE_BDRS

# 找到调用llm_json的地方，改用BDRS模板
prompt = ALFWORLD_TAGGING_TEMPLATE_BDRS.format(traj=json.dumps(traj["traj"]))
```

#### 方式2：复用RLVMR的冷启动模型（快速但不完美）
如果你已经有RLVMR的冷启动模型，可以直接使用它作为BDRS的起点。虽然prompt模板不同，但模型已经学会了基本的任务执行能力。

```bash
# 在BDRS训练脚本中指定RLVMR冷启动模型
actor_rollout_ref.model.path=/path/to/rlvmr_cold_start_model
```

### 2.4 冷启动训练

```bash
# 假设已有冷启动数据: $HOME/data/alfworld/train.parquet

cd code

# 使用8卡训练（推荐Qwen2.5-7B，效果更好）
torchrun --standalone --nnodes=1 --nproc_per_node=8 \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$HOME/data/alfworld/train.parquet \
    data.val_files=$HOME/data/alfworld/train.parquet \
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
    trainer.experiment_name=qwen7b_cold_start_bdrs \
    trainer.total_epochs=5 \
    trainer.default_local_dir=./checkpoints/cold_start/alfworld/qwen7b_bdrs \
    trainer.logger=['console','wandb'] \
    ulysses_sequence_parallel_size=4 \
    use_remove_padding=true

# 训练完成后，模型保存在:
# ./checkpoints/cold_start/alfworld/qwen7b_bdrs/default/epoch_5
```

**冷启动训练时间估算**：
- Qwen2.5-7B + 8卡A100 + 300条数据：约 30-60 分钟
- Qwen2.5-1.5B + 8卡A100 + 300条数据：约 15-30 分钟

---

## 3. 实验环境准备

### 3.1 硬件要求

| 组件 | 最低配置 | 推荐配置 | 说明 |
|------|----------|----------|------|
| **GPU** | 4×A100 (40GB) | 8×A100 (80GB) | 用于FSDP分布式训练 |
| **CPU** | 32核 | 64核 | 用于环境并行和数据预处理 |
| **内存** | 256GB | 512GB | 大batch size需要更多内存 |
| **存储** | 1TB SSD | 2TB NVMe SSD | 用于检查点和日志 |

### 3.2 软件依赖

```bash
# 系统环境
Ubuntu 20.04 / 22.04
CUDA 12.1+
Python 3.10+

# 核心依赖
PyTorch 2.1+
transformers 4.36+
vllm 0.2.7+
ray 2.9+
wandb (可选，用于可视化)
```

### 3.3 环境检查清单

```bash
# 1. CUDA可用性
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"

# 2. 分布式通信
python -c "import torch.distributed as dist; print('NCCL backend available')"

# 3. BDRS模块导入
cd code
python -c "from bdrs import BeliefStateManager, BDRSRewardCalculator; print('BDRS modules OK')"

# 4. 环境仿真器
python -c "from agent_system.environments.env_package.alfworld import build_alfworld_envs; print('ALFWorld OK')"

# 5. vLLM后端
python -c "import vllm; print(f'vLLM version: {vllm.__version__}')"
```

---

## 4. 实验阶段规划

### 阶段1：代码验证（1天）

**目标**：确保所有代码正常运行，数据流正确

**任务**：
1. 运行单元测试
   ```bash
   cd /path/to/RLVMR
   python test_bdrs.py
   # 预期：所有测试通过，总奖励约0.95
   ```

2. 小规模试运行（不冷启动）
   ```bash
   cd code
   # 修改 examples/bdrs_trainer/run_alfworld.sh:
   # train_data_size=4  # 改为4
   # val_data_size=8    # 改为8
   # trainer.total_epochs=2  # 改为2

   bash examples/bdrs_trainer/run_alfworld.sh
   ```

3. 检查关键输出
   - WandB日志中有 `bdrs_world_consistency_mean` 等指标
   - Console日志中有 `belief state updated` 信息
   - 训练2个epoch无报错

**验收标准**：
- 单元测试全部通过
- 试运行完成2个epoch
- BDRS奖励统计正常记录

---

### 阶段2：冷启动训练（1-2天）

**目标**：获得具备基本能力的初始化模型

**任务**：
1. 生成冷启动数据（如果没有）
   ```bash
   cd code
   export OPENAI_API_KEY="sk-..."

   # 修改 scripts/alfworld_prepare.py 使用BDRS模板
   # 然后运行
   python scripts/alfworld_prepare.py
   ```

2. 训练冷启动模型
   ```bash
   # 见 2.4 节的命令
   # 训练 Qwen2.5-7B 5个epoch
   ```

3. 验证冷启动模型
   ```bash
   # 使用冷启动模型进行推理测试
   bash examples/bdrs_trainer/eval_alfworld.sh \
       actor_rollout_ref.model.path=/path/to/cold_start_checkpoint
   ```

**验收标准**：
- 冷启动数据生成约300条
- SFT训练5个epoch无报错
- 冷启动模型在验证集上成功率 > 10%

**可选：跳过此阶段**，直接使用预训练模型开始RL训练（收敛会慢很多）

---

### 阶段3：基线实验（3-5天）

**目标**：建立对比基线（Vanilla PPO、RLVMR）

**任务1：Vanilla PPO基线**
```bash
cd code

# 修改 run_alfworld.sh，禁用RLVMR和BDRS
bash examples/bdrs_trainer/run_alfworld.sh \
    algorithm.rlvmr.enable=False \
    algorithm.bdrs.enable=False \
    algorithm.adv_estimator=gae \
    env.alfworld.meta_think=False \
    trainer.experiment_name='vanilla_ppo_qwen7b_L0' \
    actor_rollout_ref.model.path=/path/to/cold_start_checkpoint
```

**任务2：RLVMR基线**
```bash
cd code
bash examples/rlvmr_trainer/run_alfworld.sh \
    actor_rollout_ref.model.path=/path/to/cold_start_checkpoint
```

**训练配置**：
- 训练100个epoch（每5个epoch保存检查点）
- batch_size=16, group_size=8
- 每5个epoch在验证集上测试

**验收标准**：
- 两个基线训练完成100个epoch
- WandB日志记录完整
- 保存所有检查点

---

### 阶段4：BDRS实验（3-5天）

**目标**：训练BDRS模型，验证改进效果

**任务1：BDRS训练（默认超参数）**
```bash
cd code
bash examples/bdrs_trainer/run_alfworld.sh \
    actor_rollout_ref.model.path=/path/to/cold_start_checkpoint \
    trainer.experiment_name='bdrs_qwen7b_L0_default'
```

**任务2：BDRS超参数调优（可选）**

尝试不同的奖励权重配置：
```bash
# 配置1：更重视任务进展
bash examples/bdrs_trainer/run_alfworld.sh \
    algorithm.bdrs.world_consistency_weight=0.5 \
    algorithm.bdrs.task_progress_weight=3.0 \
    algorithm.bdrs.exploration_efficiency_weight=0.5 \
    trainer.experiment_name='bdrs_qwen7b_L0_progress_heavy'

# 配置2：更重视探索
bash examples/bdrs_trainer/run_alfworld.sh \
    algorithm.bdrs.world_consistency_weight=1.0 \
    algorithm.bdrs.task_progress_weight=1.0 \
    algorithm.bdrs.exploration_efficiency_weight=2.0 \
    trainer.experiment_name='bdrs_qwen7b_L0_explore_heavy'

# 配置3：平衡配置
bash examples/bdrs_trainer/run_alfworld.sh \
    algorithm.bdrs.world_consistency_weight=1.0 \
    algorithm.bdrs.task_progress_weight=2.0 \
    algorithm.bdrs.exploration_efficiency_weight=1.0 \
    trainer.experiment_name='bdrs_qwen7b_L0_balanced'
```

**验收标准**：
- 至少完成1个完整BDRS训练（100 epochs）
- BDRS统计指标正常记录
- 如有时间，完成2-3组超参数对比

---

### 阶段5：泛化性测试（2天）

**目标**：测试模型在不同泛化级别上的表现

**任务**：
```bash
# L0: 训练集任务（seen tasks, seen instances）
# 已在阶段3-4完成

# L1: 新实例（seen tasks, unseen instances）
cd code
bash examples/bdrs_trainer/run_alfworld.sh \
    env.alfworld.generalization_level=1 \
    trainer.experiment_name='bdrs_qwen7b_L1' \
    actor_rollout_ref.model.path=/path/to/cold_start_checkpoint

# L2: 新任务（unseen tasks, unseen instances）
bash examples/bdrs_trainer/run_alfworld.sh \
    env.alfworld.generalization_level=2 \
    trainer.experiment_name='bdrs_qwen7b_L2' \
    actor_rollout_ref.model.path=/path/to/cold_start_checkpoint
```

**验收标准**：
- 完成L1和L2的训练
- 对比三个级别的成功率

---

### 阶段6：结果分析与论文撰写（3-5天）

**任务**：
1. 数据收集与整理
2. 绘制学习曲线对比图
3. 统计显著性检验
4. 案例分析（信念状态可视化）
5. 撰写实验报告/论文

---

## 5. 服务器部署指南

### 5.1 初次部署

```bash
# 1. 连接服务器
ssh your_username@your_server

# 2. 创建工作目录
mkdir -p ~/projects
cd ~/projects

# 3. 克隆代码
git clone <your_repo_url> RLVMR
cd RLVMR

# 4. 创建conda环境
conda create -n bdrs python=3.10 -y
conda activate bdrs

# 5. 安装依赖
cd code
pip install -r requirements.txt
pip install -e .

# 6. 验证安装
python -c "from bdrs import BeliefStateManager; print('OK')"

# 7. 配置WandB（可选）
wandb login
# 输入你的API key
```

### 5.2 数据准备

```bash
# 创建数据目录
mkdir -p ~/data/verl-agent
mkdir -p ~/data/alfworld

# 准备环境数据（在首次运行时自动下载）
cd ~/projects/RLVMR/code
python -m examples.data_preprocess.prepare \
    --mode 'text' \
    --train_data_size 16 \
    --val_data_size 128

# 如果有冷启动数据
cp /path/to/alfworld_cold-start.json ~/data/alfworld/
python -m examples.data_preprocess.cold_start_data \
    --local_dir=$HOME/data/alfworld \
    --data_source=$HOME/data/alfworld/alfworld_cold-start.json
```

### 5.3 模型准备

```bash
# 下载预训练模型（自动下载到 ~/.cache/huggingface）
# 或者手动下载到指定目录
mkdir -p ~/models

# 使用 huggingface-cli
huggingface-cli download Qwen/Qwen2.5-7B-Instruct \
    --local-dir ~/models/Qwen2.5-7B-Instruct

# 或者在Python中自动下载（首次运行时）
# 修改脚本中的 model.path=~/models/Qwen2.5-7B-Instruct
```

### 5.4 配置运行脚本

```bash
cd ~/projects/RLVMR/code

# 复制并修改运行脚本
cp examples/bdrs_trainer/run_alfworld.sh my_run_bdrs.sh

# 编辑脚本，修改以下关键参数：
vim my_run_bdrs.sh
```

**需要修改的参数**：
```bash
# 1. 模型路径（修改为你的冷启动模型或预训练模型路径）
actor_rollout_ref.model.path=$HOME/models/Qwen2.5-7B-Instruct
# 或使用冷启动模型
actor_rollout_ref.model.path=$HOME/projects/RLVMR/code/checkpoints/cold_start/alfworld/qwen7b_bdrs/default/epoch_5

# 2. 数据路径（确保与你的数据目录一致）
data.train_files=$HOME/data/verl-agent/text/train.parquet \
data.val_files=$HOME/data/verl-agent/text/test.parquet \

# 3. GPU数量（根据你的服务器调整）
trainer.n_gpus_per_node=8  # 如果只有4卡，改为4

# 4. Batch size（根据GPU显存调整）
data.train_batch_size=16  # 显存不足时减半
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16  # 显存不足时减半

# 5. 实验名称
trainer.experiment_name='bdrs_qwen7b_L0_exp1'
```

---

## 6. 实验执行手册

### 6.1 训练启动

```bash
# 激活环境
conda activate bdrs
cd ~/projects/RLVMR/code

# 启动训练（前台运行，用于调试）
bash my_run_bdrs.sh

# 或后台运行（推荐）
nohup bash my_run_bdrs.sh > logs/bdrs_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# 查看进程
ps aux | grep main_ppo

# 实时查看日志
tail -f logs/bdrs_*.log
```

### 6.2 训练命令详解

完整的训练命令结构：
```bash
python3 -m verl.trainer.main_ppo \
    # === 算法配置 ===
    algorithm.adv_estimator=bdrs \              # 使用BDRS advantage估计
    algorithm.bdrs.enable=True \                # 启用BDRS
    algorithm.bdrs.world_consistency_weight=1.0 \
    algorithm.bdrs.task_progress_weight=2.0 \
    algorithm.bdrs.exploration_efficiency_weight=0.5 \
    algorithm.bdrs.step_advantage_w=1.0 \       # step-level advantage权重

    # === 数据配置 ===
    data.train_batch_size=16 \                  # 训练批次大小
    data.val_batch_size=128 \                   # 验证批次大小
    data.max_prompt_length=6000 \               # 最大prompt长度
    data.max_response_length=1024 \             # 最大回复长度

    # === 模型配置 ===
    actor_rollout_ref.model.path=/path/to/model \  # 模型路径
    actor_rollout_ref.actor.optim.lr=1e-6 \        # 学习率
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \

    # === 环境配置 ===
    env.env_name=alfworld/AlfredTWEnv \         # 环境名称
    env.max_steps=30 \                          # 最大步数
    env.rollout.n=8 \                           # 分组大小
    env.alfworld.generalization_level=0 \       # 泛化级别

    # === 训练配置 ===
    trainer.total_epochs=100 \                  # 总epoch数
    trainer.save_freq=5 \                       # 保存频率
    trainer.test_freq=5 \                       # 测试频率
    trainer.n_gpus_per_node=8 \                 # GPU数量
    trainer.project_name='BDRS' \               # WandB项目名
    trainer.experiment_name='bdrs_exp1'         # 实验名称
```

### 6.3 中断与恢复

```bash
# 训练会自动保存检查点到：
# checkpoints/BDRS/bdrs_exp1/

# 恢复训练（自动从最新检查点继续）
bash my_run_bdrs.sh  # resume_mode=auto 会自动检测检查点

# 或手动指定检查点
bash my_run_bdrs.sh \
    trainer.resume_mode=resume_path \
    trainer.resume_from_path=checkpoints/BDRS/bdrs_exp1/default/epoch_50
```

### 6.4 多实验并行

如果服务器资源充足，可以同时运行多个实验：

```bash
# 实验1：默认配置（使用GPU 0-3）
CUDA_VISIBLE_DEVICES=0,1,2,3 bash my_run_bdrs_default.sh &

# 实验2：高探索权重（使用GPU 4-7）
CUDA_VISIBLE_DEVICES=4,5,6,7 bash my_run_bdrs_explore.sh &

# 查看GPU占用
nvidia-smi
```

**注意**：
- 每个实验要修改 `trainer.experiment_name` 避免冲突
- 确保每个实验使用不同的GPU
- 监控GPU显存和系统内存

---

## 7. 监控与调试

### 7.1 WandB监控

登录 https://wandb.ai 查看实验：

**关键指标图表**：

1. **训练进度**
   - `rollout/episode_rewards_mean`：平均episode奖励（应逐渐上升）
   - `rollout/success_rate`：成功率（核心指标，目标>80%）
   - `rollout/episode_lengths_mean`：平均步数（成功后应逐渐减少）

2. **BDRS奖励分量**
   - `rollout/bdrs_world_consistency_mean`：世界一致性奖励
   - `rollout/bdrs_task_progress_mean`：任务进展奖励
   - `rollout/bdrs_exploration_efficiency_mean`：探索效率奖励
   - 查看这三个分量的比例，确保没有某个分量过度主导

3. **训练稳定性**
   - `train/actor_loss`：Actor损失（应稳定下降）
   - `train/critic_loss`：Critic损失
   - `rollout/valid_action_ratio`：有效动作比例（应>90%）
   - `train/kl_divergence`：KL散度（不应过大，<0.1为佳）

4. **子任务指标**（ALFWorld特有）
   - `rollout/pick_and_place_success_rate`
   - `rollout/pick_heat_then_place_in_recep_success_rate`
   - 等6个子任务的成功率

### 7.2 日志分析

```bash
# 实时查看训练日志
tail -f logs/bdrs_*.log

# 查找错误
grep -i "error\|exception" logs/bdrs_*.log

# 查看BDRS奖励统计
grep "bdrs_world_consistency" logs/bdrs_*.log

# 查看成功率变化
grep "success_rate" logs/bdrs_*.log | tail -20
```

**正常日志示例**：
```
Epoch 10/100, Step 160/1600
[Rollout] episode_rewards_mean: 0.45, success_rate: 0.35
[BDRS] world_consistency: 0.12±0.05, task_progress: 0.28±0.10, exploration: 0.05±0.02
[Train] actor_loss: 0.023, critic_loss: 0.045, kl: 0.008
Valid action ratio: 0.94
```

### 7.3 异常检测

**警告信号**：
1. **成功率不上升**：
   - 检查冷启动模型是否加载正确
   - 检查学习率是否过大/过小
   - 检查BDRS奖励是否计算正常

2. **BDRS奖励全为0**：
   - 检查 `algorithm.bdrs.enable=True`
   - 检查 belief state是否正确更新
   - 查看 `info['belief']` 和 `info['prev_belief']`

3. **GPU OOM**：
   - 减小 `data.train_batch_size`
   - 减小 `ppo_micro_batch_size_per_gpu`
   - 启用梯度检查点：`enable_gradient_checkpointing=True`

4. **训练速度慢**：
   - 检查环境并行是否正常（应有16个环境同时运行）
   - 检查vLLM是否启用（推理速度快很多）
   - 检查是否卡在某个step（可能是死锁）

### 7.4 调试模式

开启详细日志：
```bash
# 修改脚本，添加调试参数
bash my_run_bdrs.sh \
    trainer.logger=['console','wandb'] \
    +trainer.log_level='DEBUG'
```

手动检查belief state：
```python
# 在代码中添加调试打印
# code/agent_system/environments/env_manager.py, 第95行后
if i == 0 and step_idx % 5 == 0:  # 只打印env 0，每5步一次
    print(f"\n=== Env {i}, Step {step_idx} ===")
    print(f"Current belief: {curr_belief_snap.world_model}")
    print(f"Task progress: {curr_belief_snap.task_progress}")
    print(f"Exploration: {curr_belief_snap.exploration_map}")
```

---

## 8. 结果分析

### 8.1 数据导出

从WandB导出数据：
```python
import wandb
import pandas as pd

# 连接到项目
api = wandb.Api()
runs = api.runs("your_username/BDRS")

# 提取数据
data = []
for run in runs:
    if run.name.startswith('bdrs_qwen7b'):
        history = run.history()
        data.append({
            'name': run.name,
            'config': run.config,
            'history': history
        })

# 保存到CSV
df = pd.DataFrame(data)
df.to_csv('bdrs_results.csv')
```

### 8.2 对比分析

**成功率对比**：
```python
import matplotlib.pyplot as plt

methods = ['Vanilla PPO', 'RLVMR', 'BDRS']
success_rates = [0.45, 0.72, 0.85]  # 替换为实际数据

plt.bar(methods, success_rates)
plt.ylabel('Success Rate')
plt.title('ALFWorld L0 Success Rate Comparison')
plt.ylim(0, 1)
plt.savefig('success_rate_comparison.png')
```

**学习曲线对比**：
```python
# 绘制三条学习曲线
plt.figure(figsize=(10, 6))
plt.plot(vanilla_epochs, vanilla_success, label='Vanilla PPO')
plt.plot(rlvmr_epochs, rlvmr_success, label='RLVMR')
plt.plot(bdrs_epochs, bdrs_success, label='BDRS')
plt.xlabel('Epoch')
plt.ylabel('Success Rate')
plt.legend()
plt.grid(True)
plt.savefig('learning_curves.png')
```

### 8.3 统计显著性检验

```python
from scipy import stats

# t-test检验BDRS vs RLVMR
bdrs_scores = [0.85, 0.87, 0.84, 0.86, 0.88]  # 5次运行的成功率
rlvmr_scores = [0.72, 0.75, 0.70, 0.73, 0.74]

t_stat, p_value = stats.ttest_ind(bdrs_scores, rlvmr_scores)
print(f"t-statistic: {t_stat:.4f}, p-value: {p_value:.4f}")

if p_value < 0.05:
    print("差异显著 (p < 0.05)")
else:
    print("差异不显著 (p >= 0.05)")
```

### 8.4 案例分析

选择几个典型轨迹，可视化belief state的变化：

```python
# 提取一条轨迹的belief states
traj_beliefs = []
for step in trajectory:
    belief = step['info']['belief']
    traj_beliefs.append({
        'step': step['step_idx'],
        'world_consistency': step['bdrs_components']['world_consistency'],
        'task_progress': step['bdrs_components']['task_progress'],
        'exploration': step['bdrs_components']['exploration_efficiency'],
        'completed_subgoals': belief['task_progress']['completed_count']
    })

# 绘制奖励分量随时间变化
df = pd.DataFrame(traj_beliefs)
df.plot(x='step', y=['world_consistency', 'task_progress', 'exploration'])
plt.title('BDRS Reward Components over Time')
plt.savefig('reward_components_trajectory.png')
```

### 8.5 实验报告模板

```markdown
# BDRS实验报告

## 实验设置
- 环境：ALFWorld (Generalization Level 0)
- 模型：Qwen2.5-7B-Instruct
- 冷启动：5 epochs SFT on 300 trajectories
- RL训练：100 epochs, batch_size=16, group_size=8
- 硬件：8×A100 (80GB)

## 主要结果

| 方法 | 成功率 | 平均步数 | 收敛epoch | 有效动作率 |
|------|--------|----------|-----------|------------|
| Vanilla PPO | 45.2% | 28.5 | >100 | 87.3% |
| RLVMR | 72.4% | 22.1 | 65 | 91.2% |
| **BDRS** | **85.6%** | **19.8** | **45** | **93.5%** |

## BDRS奖励分析
- 世界一致性：0.12±0.05
- 任务进展：0.28±0.10
- 探索效率：0.05±0.02

## 子任务表现
（各子任务成功率对比图）

## 消融实验
（不同权重配置的结果）

## 结论
BDRS相比RLVMR提升了 **13.2个百分点** 的成功率，收敛速度提升 **30%**。
密集的信念奖励有效引导了探索和任务执行。
```

---

## 9. 故障排除

### 9.1 常见错误及解决方案

#### 错误1：ModuleNotFoundError: No module named 'bdrs'
**原因**：Python路径未包含code目录
**解决**：
```bash
cd ~/projects/RLVMR/code
export PYTHONPATH=$PYTHONPATH:$(pwd)
pip install -e .
```

#### 错误2：CUDA out of memory
**原因**：GPU显存不足
**解决**：
```bash
# 方案1：减小batch size
data.train_batch_size=8 \
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8

# 方案2：启用offload
actor_rollout_ref.actor.fsdp_config.param_offload=True \
actor_rollout_ref.actor.fsdp_config.optimizer_offload=True

# 方案3：减小模型（使用1.5B）
actor_rollout_ref.model.path=Qwen/Qwen2.5-1.5B-Instruct
```

#### 错误3：KeyError: 'prev_belief'
**原因**：env_manager未正确传递prev_belief
**检查**：
```python
# 在 env_manager.py 的 step 方法中确认
print(f"info keys: {info.keys()}")
# 应该有 'belief' 和 'prev_belief'
```

#### 错误4：BDRS rewards all zeros
**原因**：算法配置未启用或belief state未更新
**检查**：
```bash
# 1. 确认配置
grep "bdrs.enable" my_run_bdrs.sh
# 应该是 True

# 2. 确认advantage estimator
grep "adv_estimator" my_run_bdrs.sh
# 应该是 bdrs

# 3. 查看日志中的belief state
grep "belief state" logs/bdrs_*.log
```

#### 错误5：Training stuck at validation
**原因**：验证集太大或环境卡住
**解决**：
```bash
# 减小验证集
data.val_batch_size=64  # 从128减到64

# 或跳过初始验证
trainer.val_before_train=False
```

### 9.2 性能优化技巧

**技巧1：使用vLLM加速推理**
```bash
# 确保使用vLLM而不是HuggingFace rollout
actor_rollout_ref.rollout.name=vllm
```

**技巧2：调整环境并行数**
```bash
# 增加环境分组可以提高数据效率
env.rollout.n=16  # 从8增加到16
```

**技巧3：启用混合精度**
```bash
# 默认已启用，使用bfloat16
actor_rollout_ref.rollout.dtype=bfloat16
```

**技巧4：减少日志开销**
```bash
# 减少WandB日志频率
+trainer.log_freq=10  # 每10步记录一次
```

### 9.3 数据问题

**问题1：冷启动数据质量差**
- 症状：冷启动模型成功率<5%
- 解决：检查数据生成过程，确保使用了正确的prompt模板

**问题2：训练数据不够多样**
- 症状：模型在验证集上表现差
- 解决：增加数据预处理时的 `train_data_size`

### 9.4 紧急恢复

**场景1：训练意外中断**
```bash
# 检查最新的检查点
ls -lht checkpoints/BDRS/bdrs_exp1/default/

# 从最新检查点恢复
bash my_run_bdrs.sh  # resume_mode=auto 会自动恢复
```

**场景2：检查点损坏**
```bash
# 使用前一个epoch的检查点
bash my_run_bdrs.sh \
    trainer.resume_mode=resume_path \
    trainer.resume_from_path=checkpoints/BDRS/bdrs_exp1/default/epoch_45
```

**场景3：完全重新开始**
```bash
# 删除旧的检查点和日志
rm -rf checkpoints/BDRS/bdrs_exp1
rm logs/bdrs_*.log

# 重新开始训练
bash my_run_bdrs.sh
```

---

## 附录A：完整配置参考

### BDRS默认配置（推荐）
```yaml
algorithm:
  adv_estimator: bdrs
  bdrs:
    enable: True
    world_consistency_weight: 1.0
    task_progress_weight: 2.0
    exploration_efficiency_weight: 0.5
    step_advantage_w: 1.0
    mode: "mean_std_norm"
    reward_correct_belief: 0.2
    reward_new_conflict: -0.1
    reward_subgoal_complete: 0.5
    reward_new_entity: 0.05
    reward_new_location: 0.1
    penalty_revisit: -0.02

data:
  train_batch_size: 16
  val_batch_size: 128
  max_prompt_length: 6000
  max_response_length: 1024

actor_rollout_ref:
  model:
    path: Qwen/Qwen2.5-7B-Instruct
  actor:
    optim:
      lr: 1e-6
    ppo_mini_batch_size: 256
    ppo_micro_batch_size_per_gpu: 16
  rollout:
    name: vllm
    gpu_memory_utilization: 0.5

env:
  env_name: alfworld/AlfredTWEnv
  max_steps: 30
  rollout:
    n: 8
  alfworld:
    generalization_level: 0
    meta_think: True

trainer:
  total_epochs: 100
  save_freq: 5
  test_freq: 5
  n_gpus_per_node: 8
  project_name: 'BDRS'
  experiment_name: 'bdrs_qwen7b_L0'
```

---

## 附录B：实验检查清单

### 开始训练前
- [ ] 代码已同步到最新版本
- [ ] 单元测试全部通过 (`python test_bdrs.py`)
- [ ] 环境依赖已安装 (`pip install -e .`)
- [ ] GPU可用且数量正确 (`nvidia-smi`)
- [ ] 数据已准备完成
- [ ] 冷启动模型已训练（或决定跳过）
- [ ] WandB已配置（或决定使用console日志）
- [ ] 运行脚本已修改并检查
- [ ] 磁盘空间充足（>500GB）

### 训练过程中
- [ ] 成功率在前10个epoch内开始上升
- [ ] BDRS奖励统计正常记录到WandB
- [ ] 有效动作比例 > 85%
- [ ] 无OOM错误
- [ ] 检查点正常保存
- [ ] 日志无异常错误

### 训练结束后
- [ ] 保存所有检查点和日志
- [ ] 导出WandB数据
- [ ] 在验证集上评估最佳模型
- [ ] 记录超参数配置
- [ ] 准备对比分析数据

---

## 附录C：联系方式

如有问题，请：
1. 检查本手册的故障排除章节
2. 查看 CLAUDE.md 中的技术文档
3. 查看 BDRS_IMPROVEMENT_PLAN.md 中的设计细节
4. 查看 GitHub Issues（如有公开仓库）

---

**祝实验顺利！** 🚀
