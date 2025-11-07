# 评测脚本卡死问题诊断与解决

## 问题现象

运行 `eval_cold_start.sh` 环境交互模式后，卡在环境初始化后的模型加载阶段：

```
(TaskRunner pid=115800) Initializing AlfredTWEnv...
[100% 8810/8810 完成]
(TaskRunner pid=115800) use_expert = False
[然后卡住，GPU 利用率 0%，持续数小时]
```

## 根本原因分析

**正常流程**：
1. ✅ 环境初始化（已完成）
2. ❌ vLLM 加载模型（卡在这里）
3. ⏸ 开始评测

**GPU 0% 说明**：模型根本没有开始加载，而不是加载慢。

**可能原因**：
1. **Ray 多进程通信死锁**（最可能）
2. vLLM 配置参数冲突
3. 模型文件路径或格式问题
4. GPU 内存不足但没有报错
5. CUDA 初始化失败

## 立即解决步骤

### 步骤1：停止卡死的进程

```bash
# SSH 到服务器
ssh your_server

# 查找并杀死所有相关进程
pkill -f "main_ppo"
pkill -f "ray"
pkill -f "vllm"

# 确认进程已清理
ps aux | grep "main_ppo\|ray\|vllm"

# 清理 Ray 临时文件
ray stop
rm -rf /tmp/ray/*
```

### 步骤2：检查模型文件是否正常

```bash
cd /root/digitalhuman/RLVMR/code

# 检查模型路径是否存在
ls -lh checkpoints/cold_start/alfworld/qwen1.5b_plan_a/global_step_75/

# 应该看到这些文件
# - config.json
# - model.safetensors 或 pytorch_model.bin
# - tokenizer.json
# - tokenizer_config.json

# 如果文件不完整，说明训练checkpoint有问题
```

### 步骤3：使用最小化配置测试

创建简化的测试脚本 `eval_cold_start_minimal.sh`：

```bash
#!/bin/bash
# 最小化配置，用于诊断问题

set -x

MODEL_PATH=/root/digitalhuman/RLVMR/code/checkpoints/cold_start/alfworld/qwen1.5b_plan_a/global_step_75
NUM_TASKS=4  # 从64降到4，减少资源需求

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gae \
    algorithm.use_kl_in_reward=False \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.0 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=False \
    env.env_name=alfworld/AlfredTWEnv \
    env.seed=0 \
    env.max_steps=10 \
    env.rollout.n=$NUM_TASKS \
    env.alfworld.generalization_level=0 \
    env.alfworld.meta_think=True \
    +env.alfworld.action_only=False \
    trainer.critic_warmup=0 \
    trainer.logger=[console] \
    trainer.project_name=RLVMR_Evaluation \
    trainer.experiment_name=eval_minimal_test \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.resume_mode=disable \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_epochs=0 \
    trainer.val_before_train=True
```

**关键修改**：
- `NUM_TASKS=4`：减少到4个任务
- `gpu_memory_utilization=0.4`：降低显存占用
- `enforce_eager=True`：禁用CUDA graph，更稳定
- `free_cache_engine=True`：及时释放缓存
- `n_gpus_per_node=1`：只用1个GPU
- `trainer.logger=[console]`：禁用WandB，减少依赖
- `env.max_steps=10`：减少单任务步数

### 步骤4：运行最小化测试

```bash
cd /root/digitalhuman/RLVMR/code

# 给脚本执行权限
chmod +x examples/bdrs_trainer/eval_cold_start_minimal.sh

# 运行测试（设置30分钟超时）
timeout 1800 bash examples/bdrs_trainer/eval_cold_start_minimal.sh 2>&1 | tee eval_minimal.log
```

**观察点**：
- ✅ 如果成功运行，说明是资源配置问题
- ❌ 如果仍然卡住，查看下一步

### 步骤5：检查 Ray 和 vLLM 日志

```bash
# Ray 日志
ls -lt /tmp/ray/session_latest/logs/

# 查看最近的 worker 日志
tail -100 /tmp/ray/session_latest/logs/worker-*.out
tail -100 /tmp/ray/session_latest/logs/worker-*.err

# 查找错误关键词
grep -i "error\|failed\|timeout\|deadlock" /tmp/ray/session_latest/logs/*.log
```

### 步骤6：独立测试 vLLM 是否能加载模型

创建测试脚本 `test_vllm_load.py`：

```python
#!/usr/bin/env python3
"""
测试 vLLM 是否能独立加载模型
"""

import torch
from vllm import LLM, SamplingParams

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")

MODEL_PATH = "/root/digitalhuman/RLVMR/code/checkpoints/cold_start/alfworld/qwen1.5b_plan_a/global_step_75"

print(f"\n正在加载模型: {MODEL_PATH}")

try:
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.4,
        enforce_eager=True,
        trust_remote_code=True,
    )
    print("✓ vLLM 模型加载成功！")

    # 简单推理测试
    prompts = ["Hello, my name is"]
    sampling_params = SamplingParams(temperature=0.0, max_tokens=10)
    outputs = llm.generate(prompts, sampling_params)

    print(f"✓ 推理测试成功: {outputs[0].outputs[0].text}")

except Exception as e:
    print(f"✗ vLLM 加载失败: {e}")
    import traceback
    traceback.print_exc()
```

运行测试：

```bash
cd /root/digitalhuman/RLVMR/code
python test_vllm_load.py
```

**如果失败**，说明问题在 vLLM 或模型文件本身。

## 常见问题与解决方案

### 问题1：vLLM 加载模型时内存不足

**症状**：日志中有 `CUDA out of memory` 或进程被 killed

**解决方案**：
```bash
# 降低 GPU 内存占用
actor_rollout_ref.rollout.gpu_memory_utilization=0.3

# 或使用模型并行（如果有多张GPU）
actor_rollout_ref.rollout.tensor_model_parallel_size=2
```

### 问题2：Ray 多进程死锁

**症状**：卡住但无错误信息，CPU/GPU 都是0%

**解决方案**：
```bash
# 方案A：减少并行度
env.rollout.n=4  # 从64改为4
trainer.n_gpus_per_node=1

# 方案B：使用单进程模式（调试用）
# 在脚本开头添加
export RAY_DEBUG=1
export PYTHONUNBUFFERED=1
```

### 问题3：模型文件损坏或不兼容

**症状**：vLLM 独立测试失败

**解决方案**：
```bash
# 检查是否是相对路径问题
# 改用绝对路径
actor_rollout_ref.model.path=/root/digitalhuman/RLVMR/code/checkpoints/cold_start/alfworld/qwen1.5b_plan_a/global_step_75

# 检查模型是否完整
python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
model_path = '/root/digitalhuman/RLVMR/code/checkpoints/cold_start/alfworld/qwen1.5b_plan_a/global_step_75'
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
print('✓ Tokenizer 加载成功')
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map='cpu')
print('✓ Model 加载成功')
"
```

### 问题4：CUDA 版本不兼容

**症状**：vLLM 初始化时卡住或报错

**解决方案**：
```bash
# 检查 CUDA 版本
nvcc --version
python -c "import torch; print(torch.version.cuda)"

# 如果不一致，重新安装匹配的 vLLM
pip uninstall vllm -y
pip install vllm==0.4.2  # 或其他稳定版本
```

## 推荐的评测流程

**第一步：验证基础组件**
```bash
# 1. 测试模型加载（上面的 test_vllm_load.py）
python test_vllm_load.py

# 2. 测试单个环境
python -c "
from agent_system.environments.env_package.alfworld import AlfworldEnv
env = AlfworldEnv()
obs = env.reset()
print(f'✓ 环境初始化成功，观察: {obs[0][:100]}...')
"
```

**第二步：最小化配置测试**
```bash
# 使用 4 个任务测试
bash examples/bdrs_trainer/eval_cold_start_minimal.sh
```

**第三步：逐步增加规模**
```bash
# 如果4个任务成功，尝试16个
bash eval_cold_start.sh /path/to/model 16 vllm eval_test_16

# 再尝试64个
bash eval_cold_start.sh /path/to/model 64 vllm eval_test_64
```

## 替代方案：使用 SGLang 引擎

如果 vLLM 持续有问题，可以尝试 SGLang：

```bash
# 安装 SGLang
pip install "sglang[all]"

# 修改脚本，将 ENGINE 改为 sglang
ENGINE=sglang bash examples/bdrs_trainer/eval_cold_start.sh
```

SGLang 在某些情况下比 vLLM 更稳定。

## 紧急调试命令

如果以上都无法解决，运行以下命令收集诊断信息：

```bash
cd /root/digitalhuman/RLVMR/code

# 系统信息
nvidia-smi
free -h
df -h

# Python 环境
pip list | grep -E "vllm|ray|torch"

# Ray 状态
ray status

# 模型文件
ls -lh checkpoints/cold_start/alfworld/qwen1.5b_plan_a/global_step_75/

# 最近的错误日志
find /tmp/ray -name "*.log" -o -name "*.err" | xargs tail -50

# 保存完整诊断
bash examples/bdrs_trainer/eval_cold_start_minimal.sh > full_debug.log 2>&1
```

将 `full_debug.log` 的最后 200 行发给我分析。
