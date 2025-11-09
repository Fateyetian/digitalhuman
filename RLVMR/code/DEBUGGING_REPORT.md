# ALFWorld 评测卡住问题调试报告

## 问题描述

在执行 ALFWorld 环境的评测代码时，程序在加载完 ALFWorld 环境后卡住，无法继续执行。

## 调试过程

### 1. 添加调试日志

在关键位置添加了详细的调试打印信息：

- **main_ppo.py** (`verl/trainer/main_ppo.py`)
  - TaskRunner.run() 的各个步骤
  - 环境初始化
  - Worker 初始化
  - 训练开始

- **env_manager.py** (`agent_system/environments/env_manager.py`)
  - make_envs() 函数的各个阶段
  - ALFWorld 环境构建过程

### 2. 问题定位

通过调试日志发现，程序卡在 **Step 10: Initializing workers...** 阶段。

具体错误信息：
```
OSError: Can't load the configuration of '~/models/deepseek-llm-7b-chat'
HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name': '~/models/deepseek-llm-7b-chat'
```

### 3. 根本原因

**Critic 模型路径配置错误**：

- 配置文件中 `critic.model.path` 默认值为 `~/models/deepseek-llm-7b-chat`
- 该路径不存在或无法正确展开 `~` 符号
- **评估模式** (`trainer.total_epochs=0`) 实际上不需要 critic 模型，但初始化 worker 时仍然尝试加载

## 解决方案

### 方案 1：明确设置 Critic 模型路径（推荐）

在评估脚本中添加 critic 模型路径配置，使用与 actor 相同的模型：

```bash
python3 -m verl.trainer.main_ppo \
    # ... 其他配置 ...
    critic.model.path=$MODEL_PATH \
    critic.model.tokenizer_path=$MODEL_PATH \
    # ... 其他配置 ...
```

### 方案 2：修改默认配置文件

修改 `verl/trainer/config/ppo_trainer.yaml` 中的 critic 默认路径：

```yaml
critic:
  model:
    path: Qwen/Qwen2.5-1.5B-Instruct  # 改为有效路径
    tokenizer_path: null
```

### 方案 3：在评估时禁用 Critic（需要代码修改）

修改训练器代码，在评估模式下跳过 critic 初始化（需要更深入的代码修改）。

## 已修复的文件

1. **`examples/bdrs_trainer/eval_cold_start.sh`**
   - 添加了 `critic.model.path` 和 `critic.model.tokenizer_path` 参数

2. **`verl/trainer/main_ppo.py`**
   - 添加了详细的调试日志（可选保留或删除）

3. **`agent_system/environments/env_manager.py`**
   - 添加了详细的调试日志（可选保留或删除）

## 使用方法

### 运行修复后的评测脚本

```bash
cd /root/digitalhuman/RLVMR/code

# 使用默认参数
bash examples/bdrs_trainer/eval_cold_start.sh

# 或指定参数
bash examples/bdrs_trainer/eval_cold_start.sh \
    /path/to/model \
    64 \
    vllm \
    experiment_name
```

### 直接运行 Python 命令

```bash
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gae \
    algorithm.use_kl_in_reward=False \
    actor_rollout_ref.model.path=/path/to/model \
    critic.model.path=/path/to/model \
    critic.model.tokenizer_path=/path/to/model \
    env.env_name=alfworld/AlfredTWEnv \
    env.rollout.n=4 \
    trainer.total_epochs=0 \
    trainer.val_before_train=True \
    # ... 其他必要参数 ...
```

## 关键发现

1. **评估模式配置问题**：即使 `total_epochs=0`（纯评估模式），系统仍会初始化所有 worker 包括 critic
2. **路径展开问题**：`~` 符号在某些上下文中可能无法正确展开
3. **配置验证不足**：系统在初始化阶段才发现配置错误，而不是在配置解析阶段

## 建议

1. **评估脚本应明确所有模型路径**，不依赖默认配置
2. **添加配置验证**：在初始化之前验证所有路径的有效性
3. **优化评估模式**：在纯评估模式下跳过不必要的 worker 初始化
4. **文档更新**：在评估脚本的注释中说明 critic 路径的必要性

## 调试日志（可选择删除）

如果希望移除调试日志，可以还原以下文件的修改：
- `verl/trainer/main_ppo.py`
- `agent_system/environments/env_manager.py`

或者保留这些日志用于future debugging。
