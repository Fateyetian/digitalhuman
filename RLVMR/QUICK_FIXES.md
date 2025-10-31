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
`15a32a7` - Fix: Correct batch size configuration for verl+env mode

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

## 修复5：ValueError - GPU数量不匹配

### 错误信息
```
ValueError: Total available GPUs 2.0 is less than total desired GPUs 8
```

### 根本原因
`quick_verify.sh` 配置了 `trainer.n_gpus_per_node=8`，但服务器实际只有2个GPU可用。

在 `ray_trainer.py:121` 的资源检查中：
```python
if total_available_gpus < total_required_gpus:
    raise ValueError(f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}")
```

### 修复方案
将 `trainer.n_gpus_per_node` 从8改为2，匹配服务器实际GPU数量。

**修改位置**：`code/examples/bdrs_trainer/quick_verify.sh` (第96行)

**修改前**：
```bash
trainer.n_gpus_per_node=8 \
```

**修改后**：
```bash
trainer.n_gpus_per_node=2 \
```

**验证batch size兼容性**：
- `train_batch_size = 8`
- `n_gpus = 2`
- `8 % 2 == 0` ✓

### 文件位置
`code/examples/bdrs_trainer/quick_verify.sh` (第96行)

### Commit
`37c1504` - Fix: Adjust n_gpus_per_node to 2 for actual server configuration

---

## 修复6：ImportError - flash_attn兼容性问题

### 错误信息
```
ImportError: /root/miniconda3/lib/python3.10/site-packages/flash_attn_2_cuda.cpython-310-x86_64-linux-gnu.so:
undefined symbol: _ZN3c105ErrorC2ENS_14SourceLocationENSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE
```

### 根本原因
flash_attn是预编译的CUDA扩展，存在ABI兼容性问题：
- flash_attn编译时使用的PyTorch版本与当前环境不匹配
- C++ ABI符号无法解析
- 这是常见的预编译扩展版本不匹配问题

### 修复方案

#### 方案1：禁用flash_attn（推荐，最快）

修改 `quick_verify.sh`，禁用flash_attn和remove_padding：

**修改位置**：`code/examples/bdrs_trainer/quick_verify.sh` (第63-64行)

**修改前**：
```bash
actor_rollout_ref.model.use_remove_padding=True \
```

**修改后**：
```bash
actor_rollout_ref.model.use_remove_padding=False \
```

**说明**：
- 只需设置 `use_remove_padding=False` 即可
- `enable_flash_attn` 配置项不存在于配置schema中
- 禁用remove_padding会自动避免使用flash_attn依赖的功能
- 使用PyTorch原生SDPA（Scaled Dot-Product Attention）
- 对于1.5B小模型和验证任务，性能差异可忽略

#### 方案2：重新安装flash_attn（耗时）

如果需要flash_attn的性能优势：

```bash
# 卸载旧版本
pip uninstall flash-attn -y

# 从源码重新编译（需要30分钟-1小时）
pip install flash-attn --no-build-isolation

# 或者安装预编译版本（可能仍有兼容性问题）
pip install flash-attn==2.5.8
```

#### 方案3：升级PyTorch（不推荐）

可能引入其他兼容性问题，不建议在验证阶段尝试。

### 文件位置
`code/examples/bdrs_trainer/quick_verify.sh` (第63行)

### Commit
`abb02f7` - Fix: Disable flash_attn by setting use_remove_padding=False

---

## 修复7：ConfigAttributeError - 缺失action_only配置

### 错误信息
```
omegaconf.errors.ConfigAttributeError: Key 'action_only' is not in struct
    full_key: env.alfworld.action_only
    object_type=dict
```

### 根本原因
在 `env_manager.py:149`，代码尝试访问 `self.config.env.alfworld.action_only` 配置项：
```python
elif self.config is not None and self.config.env.alfworld.action_only and not use_bdrs_template:
    _ALFWORLD_TEMPLATE_NO_HIS = ALFWORLD_TEMPLATE_NO_HIS_NOTHINK
    _ALFWORLD_TEMPLATE = ALFWORLD_TEMPLATE_NOTHINK
```

但 `quick_verify.sh` 中没有提供这个配置项。

### 修复方案
在 `quick_verify.sh` 中使用 `+` 前缀添加 `action_only` 配置（绕过struct mode限制）。

**修改位置**：`code/examples/bdrs_trainer/quick_verify.sh` (第92行插入)

**添加**：
```bash
env.alfworld.meta_think=True \
+env.alfworld.action_only=False \
```

**重要说明**：
- 使用 `+` 前缀是Hydra的语法，用于添加未在schema中定义的配置项
- `action_only=False` 表示使用完整的observation模板（包含思考过程）
- 与 `meta_think=True` 配合，使用带有元认知的完整模板
- 这是BDRS推荐的设置，提供更丰富的上下文信息

### 文件位置
`code/examples/bdrs_trainer/quick_verify.sh` (第92行)

### Commit
`7582134` - Fix: Add missing action_only config with + prefix for struct mode

---

## 修复8：NameError - total_infos未定义

### 错误信息
```
NameError: name 'total_infos' is not defined
  File "agent_system/multi_turn_rollout/rollout_loop.py", line 523, in multi_turn_loop
    infos_seq = total_infos[env_idx]
```

### 根本原因
BDRS奖励计算代码（第523行）需要访问 `total_infos[env_idx]` 来获取环境返回的belief状态信息。

但是：
- `total_infos` 在 `vanilla_multi_turn_loop` 中定义（第312行）并填充（第373行）
- `vanilla_multi_turn_loop` 没有返回这个变量
- `dynamic_multi_turn_loop` 也没有收集和返回这个变量
- 导致 `multi_turn_loop` 中的BDRS代码无法访问

### 修复方案
修改 `rollout_loop.py` 的函数签名，在整个调用链中传递 `total_infos`：

**修改位置1**：`vanilla_multi_turn_loop` 返回值（第392行）
```python
# 修改前：
return total_batch_list, episode_rewards, episode_lengths, success, traj_uid

# 修改后：
return total_batch_list, episode_rewards, episode_lengths, success, traj_uid, total_infos
```

**修改位置2**：`dynamic_multi_turn_loop` 初始化（第422行）
```python
# 添加：
total_infos_list = []
```

**修改位置3**：`dynamic_multi_turn_loop` 收集（第451行）
```python
# 添加：
total_infos_list += infos_list
```

**修改位置4**：`dynamic_multi_turn_loop` 返回值（第458行）
```python
# 修改前：
return total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid

# 修改后：
return total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid, total_infos_list
```

**修改位置5**：`multi_turn_loop` 接收返回值（第482、490行）
```python
# 修改前：
total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid = \
    self.dynamic_multi_turn_loop(...)

# 修改后：
total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid, total_infos = \
    self.dynamic_multi_turn_loop(...)
```

### 文件位置
`code/agent_system/multi_turn_rollout/rollout_loop.py` (多处修改)

### Commit
`c68099b` - Fix: Return total_infos from rollout functions for BDRS belief tracking

---

## 修复9：ConfigAttributeError - 无法添加config属性（提前修复）

### 潜在错误
```
omegaconf.errors.ConfigAttributeError: Key 'meta_info_bdrs' is not in struct
    full_key: meta_info_bdrs
```

### 根本原因
在 `rollout_loop.py:562`，代码尝试给config对象添加新属性：
```python
self.config.meta_info_bdrs = {...}
```

但是 `ray_trainer.py:559` 启用了OmegaConf struct模式：
```python
OmegaConf.set_struct(self.config, True)
```

struct模式下不允许添加未在schema中定义的新属性。

### 修复方案
使用局部变量而不是config属性来存储BDRS统计信息。

**修改位置1**：定义统计信息（第562-568行）
```python
# 修改前：
self.config.meta_info_bdrs = {
    "world_consistency": _safe_stat(bdrs_world),
    ...
}

# 修改后：
meta_info_bdrs = {
    "world_consistency": _safe_stat(bdrs_world),
    ...
}
```

**修改位置2**：传递统计信息（第586-588行）
```python
# 修改前：
if hasattr(self.config, 'meta_info_bdrs'):
    gen_batch_output.meta_info['bdrs_stats'] = self.config.meta_info_bdrs

# 修改后：
if 'meta_info_bdrs' in locals():
    gen_batch_output.meta_info['bdrs_stats'] = meta_info_bdrs
```

### 文件位置
`code/agent_system/multi_turn_rollout/rollout_loop.py` (第562-568行、第586-588行)

### Commit
`f8e110d` - Fix: Avoid ConfigAttributeError by using local variable instead of config attribute

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
| 2025-10-31 | f8e110d | 修复9: 避免ConfigAttributeError（使用局部变量代替config属性）|
| 2025-10-31 | c68099b | 修复8: 返回total_infos以支持BDRS belief追踪 |
| 2025-10-31 | 7582134 | 修复7: 添加action_only配置（使用+前缀）|
| 2025-10-31 | abb02f7 | 修复6: 禁用flash_attn（只设置use_remove_padding=False）|
| 2025-10-31 | 8ed084e | 优化: 降低batch size为4适配2-GPU服务器 |
| 2025-10-31 | 37c1504 | 修复5: 调整GPU数量为2匹配服务器配置 |
| 2025-10-31 | 15a32a7 | 修复3+4: 修正batch size和rollout.n配置 |
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
