# BDRS 快速修复记录

本文档记录了在部署BDRS到服务器时遇到的问题和修复方案。

---

## 修复1：ImportError - alfworld projection函数不存在

### 错误信息
```
ImportError: cannot import name 'alfworld_projection_nothink' from
'agent_system.environments.env_package.alfworld.projection'
```

### 根本原因
`code/agent_system/environments/env_package/alfworld/__init__.py` 尝试导入两个不存在的函数：
- `alfworld_projection_nothink`
- `alfworld_projection_mcrl`

但是 `projection.py` 中只定义了：
- `alfworld_projection`
- `alfworld_projection_rlvmr`

### 修复方案
修改 `__init__.py` 的导入语句：

**修改前**：
```python
from .projection import alfworld_projection, alfworld_projection_nothink, alfworld_projection_mcrl
```

**修改后**：
```python
from .projection import alfworld_projection, alfworld_projection_rlvmr
```

### 文件位置
`code/agent_system/environments/env_package/alfworld/__init__.py`

### Commit
`15a8e35` - Fix: Remove non-existent function imports in alfworld __init__.py

---

## 修复2：NotImplementedError - BDRS不在advantage estimator列表中

### 错误信息
```
ray.exceptions.RayTaskError(NotImplementedError):
  File "verl/trainer/ppo/ray_trainer.py", line 376, in __init__
    raise NotImplementedError
NotImplementedError
```

### 根本原因
在 `ray_trainer.py` 的第368-376行，代码检查 `adv_estimator` 类型：
- `GAE` → 使用critic
- `GRPO/REINFORCE++/REMAX/RLOO/GiGPO/RLVMR` → 不使用critic
- 其他 → 抛出 `NotImplementedError`

虽然 `BDRS` 已经在 `AdvantageEstimator` 枚举中定义，但没有包含在第二个列表中。

### 修复方案
将 `AdvantageEstimator.BDRS` 添加到不使用critic的列表中。

**修改前（第370-373行）**：
```python
elif self.config.algorithm.adv_estimator in [
        AdvantageEstimator.GRPO, AdvantageEstimator.REINFORCE_PLUS_PLUS, AdvantageEstimator.REMAX,
        AdvantageEstimator.RLOO, AdvantageEstimator.GiGPO, AdvantageEstimator.RLVMR
]:
    self.use_critic = False
```

**修改后**：
```python
elif self.config.algorithm.adv_estimator in [
        AdvantageEstimator.GRPO, AdvantageEstimator.REINFORCE_PLUS_PLUS, AdvantageEstimator.REMAX,
        AdvantageEstimator.RLOO, AdvantageEstimator.GiGPO, AdvantageEstimator.RLVMR, AdvantageEstimator.BDRS
]:
    self.use_critic = False
```

### 文件位置
`code/verl/trainer/ppo/ray_trainer.py` (第372行)

### Commit
`643a818` - Fix: Add BDRS to advantage estimator list in ray_trainer.py

---

## 修复3：AssertionError - Batch size不能被GPU数整除

### 错误信息
```
AssertionError: real_train_batch_size (4) must be divisible by total n_gpus (8).
```

### 根本原因
在 `ray_trainer.py` 第387行，验证代码计算：
```python
real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
```

在 verl+env 模式下，`actor_rollout_ref.rollout.n` 必须为1（见 `main_ppo.py:164` 的断言）。
因此：
- `real_train_batch_size = train_batch_size * 1 = train_batch_size`
- 原配置 `train_batch_size = 4`，但 `4 % 8 != 0`

### 修复方案
将 `train_data_size` 从4改为8，使 `train_batch_size` 能被GPU数整除。

**修改位置**：`code/examples/bdrs_trainer/quick_verify.sh` (第10行)

**修改前**：
```bash
train_data_size=4        # 只用4个训练样本
```

**修改后**：
```bash
train_data_size=8        # 只用8个训练样本（必须能被GPU数整除）
```

这样设置后：
- `train_batch_size = 8`
- `rollout.n = 1` (verl+env模式固定值)
- `real_train_batch_size = 8 * 1 = 8`
- `8 % 8 == 0` ✓

**重要说明**：
- 在 verl+env 模式下，`actor_rollout_ref.rollout.n` 必须为1
- 分组通过 `env.rollout.n` 实现，而不是 `actor_rollout_ref.rollout.n`
- 这是 `main_ppo.py:164` 的约束要求

### 文件位置
`code/examples/bdrs_trainer/quick_verify.sh` (第10行)

### Commit
`d34aa19` → `(待重新提交)` - Fix: Increase train_data_size to 8 for GPU divisibility

---

## 修复4：AssertionError - actor_rollout_ref.rollout.n 必须为1

### 错误信息
```
AssertionError: In verl, actor_rollout_ref.rollout.n>1 is for GRPO.
In verl+env, we keep n=1, and achieve GRPO by env.rollout.n
```

### 根本原因
`main_ppo.py:164` 有一个断言，在 verl+env 模式下：
```python
assert config.actor_rollout_ref.rollout.n == 1
```

修复3中错误地设置了 `actor_rollout_ref.rollout.n=$group_size`，违反了这个约束。

### 修复方案
移除 `actor_rollout_ref.rollout.n=$group_size` 配置，保持默认值1。

**说明**：
- 在 verl+env 模式下，分组通过 `env.rollout.n` 实现
- `actor_rollout_ref.rollout.n` 必须保持为1
- 这与修复3合并，一起修正

### 文件位置
`code/examples/bdrs_trainer/quick_verify.sh` (移除第86行)

### Commit
合并到修复3 - Fix: Increase train_data_size to 8 and keep rollout.n=1

---

## 应用修复到服务器

如果你在服务器上遇到这些错误，有两种方式修复：

### 方式1：拉取最新代码（推荐）
```bash
cd ~/projects/RLVMR
git fetch origin improvement_1
git pull origin improvement_1
```

### 方式2：手动修复
如果无法拉取，可以手动编辑这两个文件：

1. **修复1**：编辑 `code/agent_system/environments/env_package/alfworld/__init__.py`
   ```bash
   vim code/agent_system/environments/env_package/alfworld/__init__.py
   # 将第1行改为：
   # from .projection import alfworld_projection, alfworld_projection_rlvmr
   ```

2. **修复2**：编辑 `code/verl/trainer/ppo/ray_trainer.py`
   ```bash
   vim code/verl/trainer/ppo/ray_trainer.py
   # 在第372行添加 AdvantageEstimator.BDRS
   # 具体见上面的代码示例
   ```

---

## 验证修复

修复后，重新运行验证脚本：

```bash
cd code
bash examples/bdrs_trainer/quick_verify.sh
```

**预期结果**：
- 不再出现 `ImportError`
- 不再出现 `NotImplementedError`
- 训练正常开始，能看到 `[BDRS] Reward statistics` 输出

---

## 其他可能的问题

### 问题3：ModuleNotFoundError: No module named 'bdrs'

**症状**：
```
ModuleNotFoundError: No module named 'bdrs'
```

**解决**：
```bash
cd code
pip install -e .
```

### 问题4：CUDA out of memory

**症状**：
```
CUDA out of memory. Tried to allocate XX MB
```

**解决**：
```bash
# 减小batch size
bash examples/bdrs_trainer/quick_verify.sh \
    data.train_batch_size=2 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4
```

### 问题5：数据文件不存在

**症状**：
```
FileNotFoundError: ~/data/verl-agent/text/train.parquet
```

**解决**：
```bash
cd code
python3 -m examples.data_preprocess.prepare \
    --mode 'text' \
    --train_data_size 4 \
    --val_data_size 8
```

---

## 修复历史

| 日期 | Commit | 描述 |
|------|--------|------|
| 2025-10-31 | (待提交) | 修复3+4: 修正batch size和rollout.n配置 |
| 2025-10-31 | d34aa19 | (已废弃) 错误的修复尝试 |
| 2025-10-31 | 643a818 | 修复2: 添加BDRS到advantage estimator列表 |
| 2025-10-31 | 15a8e35 | 修复1: 移除不存在的函数导入 |
| 2025-10-31 | e52d874 | 初始BDRS实现 |

---

## 联系与支持

如果遇到其他问题：
1. 查看 `BDRS_VERIFICATION_GUIDE.md` 的故障排除部分
2. 查看 `BDRS_EXPERIMENT_PLAN.md` 第9节（故障排除）
3. 检查错误日志的完整堆栈跟踪

**重要提示**：每次修复后都要重新运行 `git pull` 确保代码是最新的！
