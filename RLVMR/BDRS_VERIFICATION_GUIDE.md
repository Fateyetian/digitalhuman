# BDRS 代码验证指南

## 验证目标

快速确认以下关键功能正常：
1. ✅ BDRS模块正确导入和初始化
2. ✅ Belief state正确更新和传递
3. ✅ 差分奖励正确计算
4. ✅ 训练循环正常运行
5. ✅ 日志和统计信息正确输出

**预计时间**：30-60分钟（取决于GPU性能）

---

## 验证步骤

### 步骤1：单元测试（5分钟）

```bash
cd /path/to/RLVMR

# 运行BDRS单元测试
python test_bdrs.py
```

**预期输出**：
```
================================================================================
BDRS 功能测试
================================================================================
测试1: BeliefStateManager 基础功能
[OK] 子目标解析成功
[OK] 探索追踪成功
[OK] 任务进展追踪成功

测试2: BDRSRewardCalculator 差分奖励计算
[OK] 探索奖励计算正确
[OK] 任务进展奖励计算正确
[OK] 世界一致性奖励计算正确

测试3: 完整数据流集成测试
[OK] 集成测试完成！总累计奖励: 0.9500

*** 所有测试通过！***
```

**✅ 验收标准**：所有测试通过，总奖励约0.95

**❌ 如果失败**：
- 检查代码是否最新 (`git pull`)
- 检查依赖是否安装 (`pip install -e .`)
- 查看错误信息，根据提示修复

---

### 步骤2：快速训练验证（30-50分钟）

```bash
cd code

# 赋予执行权限
chmod +x examples/bdrs_trainer/quick_verify.sh

# 运行验证脚本
bash examples/bdrs_trainer/quick_verify.sh
```

**脚本做什么**：
- 生成4个训练样本 + 8个验证样本
- 使用Qwen2.5-1.5B模型（小模型，快速）
- 训练2个epoch
- 不保存检查点（节省空间）
- 只输出到console（不上传WandB）

**关键配置**：
```bash
train_data_size=4          # 极小批次
val_data_size=8
total_epochs=2             # 只训2轮
model=Qwen2.5-1.5B         # 小模型
gpu_memory_utilization=0.4 # 低显存占用
```

---

### 步骤3：检查输出（5分钟）

训练过程中，重点关注以下输出：

#### 3.1 数据准备阶段
```
[Step 1/3] Preparing data...
Processing train: 100%|████████| 4/4
Processing test: 100%|████████| 8/8
Train data saved to: ~/data/verl-agent/text/train.parquet
```
✅ **检查点**：数据文件生成成功

#### 3.2 初始化阶段
```
Loading model: Qwen/Qwen2.5-1.5B-Instruct
Initializing BDRS...
BeliefStateManager initialized for alfworld
BDRSRewardCalculator initialized with:
  - world_consistency_weight: 1.0
  - task_progress_weight: 2.0
  - exploration_efficiency_weight: 0.5
```
✅ **检查点**：BDRS模块正确初始化

#### 3.3 训练阶段（Epoch 1）
```
Epoch 1/2
[Rollout] Collecting trajectories...
  Environment: alfworld/AlfredTWEnv
  Batch size: 4, Group size: 2
  Max steps: 30

[Rollout] Episode statistics:
  episode_rewards_mean: 0.15 (±0.08)
  episode_lengths_mean: 25.3
  success_rate: 0.00
  valid_action_ratio: 0.85

[BDRS] Reward statistics:
  world_consistency: mean=0.05, std=0.03, min=0.00, max=0.15
  task_progress: mean=0.12, std=0.08, min=0.00, max=0.35
  exploration_efficiency: mean=0.08, std=0.05, min=0.00, max=0.20
  total_reward: mean=0.32, std=0.15, min=0.05, max=0.65
  n_steps: 101

[Train] Training actor...
  actor_loss: 0.245
  critic_loss: 0.187
  kl_divergence: 0.008
  entropy: 2.456
```

✅ **关键检查点**：
1. **BDRS统计存在**：`[BDRS] Reward statistics` 出现
2. **三个奖励分量都有值**：world_consistency, task_progress, exploration_efficiency
3. **奖励不全为0**：至少有一些正值
4. **有效动作比例合理**：valid_action_ratio > 0.8
5. **KL散度不过大**：kl_divergence < 0.1

#### 3.4 训练阶段（Epoch 2）
```
Epoch 2/2
[Rollout] Episode statistics:
  episode_rewards_mean: 0.18 (±0.10)  # 略有上升
  success_rate: 0.00 (可能仍为0，样本太少)

[BDRS] Reward statistics:
  world_consistency: mean=0.06, std=0.03
  task_progress: mean=0.15, std=0.09      # 可能略有上升
  exploration_efficiency: mean=0.09, std=0.06
```

✅ **检查点**：奖励有轻微波动（正常现象）

#### 3.5 完成阶段
```
[Step 3/3] Verification PASSED!

✓ Data flow is correct
✓ BDRS rewards are computed
✓ Training loop works

Next steps:
  1. Check the console output for BDRS statistics
  2. If all looks good, proceed with full training
  3. See BDRS_EXPERIMENT_PLAN.md Section 4.2 (Cold Start)
```

---

## 验收标准（Checklist）

### ✅ 必须满足（Critical）
- [ ] 单元测试全部通过
- [ ] 训练脚本运行完2个epoch无报错
- [ ] `[BDRS] Reward statistics` 在日志中出现
- [ ] 三个奖励分量（world/task/explore）都有非零值
- [ ] 没有CUDA OOM错误
- [ ] 没有"KeyError: prev_belief"错误

### 🟡 应该满足（Important）
- [ ] valid_action_ratio > 0.8
- [ ] BDRS total_reward > 0
- [ ] kl_divergence < 0.1
- [ ] 日志中有 "BeliefStateManager initialized"

### 🟢 可选满足（Nice to have）
- [ ] episode_rewards 在epoch 2略高于epoch 1
- [ ] 某些子任务的success_rate > 0
- [ ] 没有警告信息

---

## 常见问题及解决

### 问题1：找不到BDRS模块

**症状**：
```
ModuleNotFoundError: No module named 'bdrs'
```

**解决**：
```bash
cd code
pip install -e .
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

---

### 问题2：BDRS奖励全为0

**症状**：
```
[BDRS] Reward statistics:
  world_consistency: mean=0.00, std=0.00
  task_progress: mean=0.00, std=0.00
  exploration_efficiency: mean=0.00, std=0.00
```

**诊断**：
```bash
# 检查配置
grep "bdrs.enable" examples/bdrs_trainer/quick_verify.sh
# 应该输出: algorithm.bdrs.enable=True

# 检查advantage estimator
grep "adv_estimator" examples/bdrs_trainer/quick_verify.sh
# 应该输出: algorithm.adv_estimator=bdrs
```

**可能原因**：
1. belief state未正确更新
2. prev_belief未正确传递
3. 环境管理器未初始化BeliefStateManager

**解决**：检查代码是否是最新版本

---

### 问题3：GPU OOM

**症状**：
```
CUDA out of memory. Tried to allocate XX MB
```

**解决**：
```bash
# 方案1：使用更小的模型（已经在用1.5B了）
# 方案2：减少batch size
bash examples/bdrs_trainer/quick_verify.sh \
    data.train_batch_size=2 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4

# 方案3：启用offload
bash examples/bdrs_trainer/quick_verify.sh \
    actor_rollout_ref.actor.fsdp_config.param_offload=True
```

---

### 问题4：验证通过但成功率为0

**这是正常的！**

原因：
- 只有4个训练样本，太少了
- 只训练2个epoch，不足以学会
- 没有冷启动，模型从零开始
- Qwen2.5-1.5B是小模型

**不用担心**：
- 目的是验证**代码流程**，不是验证**性能**
- 只要BDRS奖励正确计算即可
- 正式训练会使用更大模型、更多数据、冷启动

---

## 验证通过后的下一步

### 如果验证全部通过 ✅

**恭喜！可以开始方案A（完整流程）**

1. **准备冷启动数据**（见 BDRS_EXPERIMENT_PLAN.md 第2.3节）
   ```bash
   cd code
   export OPENAI_API_KEY="sk-..."

   # 修改 scripts/alfworld_prepare.py 使用BDRS模板
   # 然后运行
   python scripts/alfworld_prepare.py
   ```

2. **训练冷启动模型**（见 BDRS_EXPERIMENT_PLAN.md 第2.4节）
   - 预计时间：1-2天
   - 使用Qwen2.5-7B（效果更好）
   - 训练5个epoch

3. **开始正式RL训练**
   - 使用冷启动模型
   - 训练100个epoch
   - 完整的batch size和数据

---

### 如果验证失败 ❌

1. **收集错误信息**
   - 完整的错误日志
   - 运行环境信息（GPU型号、CUDA版本、PyTorch版本）
   - Python依赖版本 (`pip list | grep -E "torch|vllm|transformers"`)

2. **检查环境**
   ```bash
   # GPU可用性
   python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"

   # BDRS模块
   cd code
   python -c "from bdrs import BeliefStateManager, BDRSRewardCalculator; print('OK')"

   # ALFWorld环境
   python -c "from agent_system.environments.env_package.alfworld import build_alfworld_envs; print('OK')"
   ```

3. **查看故障排除**
   - 参考 BDRS_EXPERIMENT_PLAN.md 第9节（故障排除）
   - 常见错误都有对应的解决方案

---

## 验证日志示例（完整）

为了帮助你对比，这里是一个完整的成功验证日志示例：

```
===========================================
BDRS Quick Verification
===========================================
Train size: 4
Val size: 8
Epochs: 2
===========================================
[Step 1/3] Preparing data...
Processing train dataset: 100%|████████████| 4/4 [00:01<00:00]
Processing test dataset: 100%|████████████| 8/8 [00:01<00:00]
Data saved to: ~/data/verl-agent/text/

[Step 2/3] Starting BDRS training verification...
Loading Qwen/Qwen2.5-1.5B-Instruct...
Model loaded on 8 GPUs with FSDP
Initializing environments...
  - 4 training environments (2 groups)
  - 8 validation environments
Initializing BDRS...
  - BeliefStateManager for alfworld
  - BDRSRewardCalculator with weights [1.0, 2.0, 0.5]

======== Epoch 1/2 ========
[Rollout] Collecting 4 trajectories...
  Progress: 100%|████████████| 4/4 [00:35<00:00]

[Rollout] Episode statistics:
  episode_rewards_mean: 0.15 ± 0.08
  episode_lengths_mean: 25.3
  success_rate: 0.00
  valid_action_ratio: 0.85

[BDRS] Reward statistics:
  world_consistency: mean=0.05, std=0.03, min=0.00, max=0.15
  task_progress: mean=0.12, std=0.08, min=0.00, max=0.35
  exploration_efficiency: mean=0.08, std=0.05, min=0.00, max=0.20
  total_reward: mean=0.32, std=0.15, min=0.05, max=0.65
  n_steps: 101

[Train] Training actor (PPO)...
  actor_loss: 0.245
  critic_loss: 0.187
  kl_divergence: 0.008
  Training completed in 42s

[Test] Testing on validation set...
  val_success_rate: 0.00
  val_episode_rewards: 0.08 ± 0.05

======== Epoch 2/2 ========
[Rollout] Collecting 4 trajectories...
  Progress: 100%|████████████| 4/4 [00:33<00:00]

[Rollout] Episode statistics:
  episode_rewards_mean: 0.18 ± 0.10
  episode_lengths_mean: 24.8
  success_rate: 0.00
  valid_action_ratio: 0.87

[BDRS] Reward statistics:
  world_consistency: mean=0.06, std=0.03, min=0.00, max=0.18
  task_progress: mean=0.15, std=0.09, min=0.00, max=0.40
  exploration_efficiency: mean=0.09, std=0.06, min=0.00, max=0.22
  total_reward: mean=0.38, std=0.17, min=0.08, max=0.72
  n_steps: 99

[Train] Training actor (PPO)...
  actor_loss: 0.232
  critic_loss: 0.175
  kl_divergence: 0.009

[Test] Testing on validation set...
  val_success_rate: 0.00
  val_episode_rewards: 0.10 ± 0.06

Training completed successfully!

===========================================
[Step 3/3] Verification PASSED!
===========================================

✓ Data flow is correct
✓ BDRS rewards are computed
✓ Training loop works

Next steps:
  1. Check the console output for BDRS statistics
  2. If all looks good, proceed with full training
  3. See BDRS_EXPERIMENT_PLAN.md Section 4.2 (Cold Start)
```

---

## 时间估算

| 步骤 | 预计时间 | 说明 |
|------|----------|------|
| 单元测试 | 5分钟 | 纯Python测试，无需GPU |
| 数据准备 | 2-5分钟 | 下载geometry3k数据集 |
| Epoch 1 | 15-25分钟 | Rollout + Train + Test |
| Epoch 2 | 15-25分钟 | 同上 |
| **总计** | **30-60分钟** | 取决于GPU性能 |

**GPU性能参考**：
- A100 (80GB): ~30分钟
- A100 (40GB): ~40分钟
- V100 (32GB): ~50分钟
- 更老的GPU: 可能需要1小时+

---

## 总结

验证流程很简单：

1. ✅ `python test_bdrs.py` - 5分钟
2. ✅ `bash examples/bdrs_trainer/quick_verify.sh` - 30-60分钟
3. ✅ 检查BDRS统计是否正确输出

**验证成功 → 开始方案A（冷启动+完整训练）**

**验证失败 → 查看故障排除章节**

祝验证顺利！🚀
