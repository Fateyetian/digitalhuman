# Ray 调试与常见问题解决经验

## 🔍 Ray 调试的核心思路

### 1. **Ray 的三层架构**
```
应用层 (Your Python Script)
    ↓
Worker 层 (ray.remote 装饰的函数/类)
    ↓
调度层 (Raylet + GCS Server)
```

**问题定位原则**：
- 卡住无输出 → 调度层问题（raylet/GCS）
- 有错误信息 → Worker 层问题
- 启动就失败 → 应用层配置问题

---

## 🚨 常见 Bug 类型与解决方案

### Bug 1: Worker 无法连接到 Raylet（最常见）

**症状**：
```
Connection refused: ipv4:xxx.xxx.xxx.xxx:xxxxx
Shutting down the core worker because the local raylet failed
Got negative used memory for cgroup -1
```

**根本原因**：
1. Ray 的 cgroup 内存监控与系统不兼容（Docker、某些 Linux 发行版）
2. Raylet 进程崩溃或启动失败
3. GPU/CPU 资源配置与实际不符

**解决方案（按优先级）**：

```bash
# A. 检查实际 GPU 数量
nvidia-smi --list-gpus
# 确保脚本中的 trainer.n_gpus_per_node 不超过实际数量

# B. 手动启动 Ray（明确指定资源）
ray stop --force
ray start --head --num-gpus=1 --num-cpus=8 --memory=50000000000
ray status  # 确认启动成功

# C. 禁用 cgroup 监控（如果 A/B 不行）
export RAY_memory_monitor_refresh_ms=0
export RAY_DISABLE_MEMORY_MONITOR=1

# D. 使用本地模式（不启动集群，单进程）
export RAY_local_mode=1
```

---

### Bug 2: Ray 启动慢或卡住

**症状**：
- `ray.init()` 或脚本启动后长时间无反应
- GPU 利用率 0%

**原因**：
1. Ray 自动检测资源时卡住
2. 网络配置问题（无法解析 localhost/127.0.0.1）
3. 端口被占用

**诊断命令**：
```bash
# 1. 查看 Ray 日志（最重要！）
tail -f /tmp/ray/session_latest/logs/raylet.out
tail -f /tmp/ray/session_latest/logs/gcs_server.out

# 2. 检查端口占用
netstat -tulnp | grep "6379\|8265"  # Ray 默认端口

# 3. 检查 Ray 进程状态
ps aux | grep "raylet\|gcs_server"

# 4. 查看完整的 Ray 日志路径
ls -lt /tmp/ray/session_*/logs/
```

**解决方案**：
```bash
# 方案1：明确指定 Ray 地址
export RAY_ADDRESS="127.0.0.1:6379"
ray start --head --port=6379

# 方案2：使用不同端口
ray start --head --port=6380 --redis-password="YOUR_PASSWORD"

# 方案3：完全清理后重启
pkill -9 -f ray
rm -rf /tmp/ray/*
rm -rf /tmp/ray_tmp_*
rm -rf ~/.cache/ray/*
# 等待 5 秒
ray start --head
```

---

### Bug 3: 多 GPU 资源分配冲突

**症状**：
- 脚本要求 2 个 GPU，但只有 1 个
- `CUDA out of memory` 但实际显存充足

**原因**：
Ray 的资源声明（`num_gpus`）与实际分配不符

**诊断**：
```bash
# 1. 查看 Ray 认为的资源
ray status

# 输出示例：
# Resources
#  0/1.0 CPU
#  0/1.0 GPU
#
# 如果 GPU 显示为 0，说明 Ray 没检测到 GPU

# 2. 查看实际 GPU
nvidia-smi

# 3. 检查 CUDA_VISIBLE_DEVICES
echo $CUDA_VISIBLE_DEVICES
```

**解决方案**：
```bash
# A. 明确指定 GPU
export CUDA_VISIBLE_DEVICES=0  # 只用第一个 GPU
ray stop --force
ray start --head --num-gpus=1

# B. 在脚本中降低 GPU 需求
trainer.n_gpus_per_node=1  # 而不是 2

# C. 如果有多个 GPU 但 Ray 只看到 1 个
# 检查是否被其他进程占用
fuser -v /dev/nvidia*
```

---

### Bug 4: Ray 进程僵尸/残留

**症状**：
- 之前的训练/评测异常退出
- 新任务无法启动，提示端口占用
- `ray stop` 无效

**诊断**：
```bash
# 1. 查找所有 Ray 进程
ps aux | grep ray

# 2. 查找占用端口的进程
lsof -i:6379  # Ray GCS 默认端口
lsof -i:8265  # Ray Dashboard 默认端口

# 3. 查看 Ray 临时文件
ls -lh /tmp/ray/
```

**解决方案**：
```bash
# 方案1：强制清理（最彻底）
pkill -9 -f ray
pkill -9 -f "python.*verl"
pkill -9 -f vllm
rm -rf /tmp/ray/*
rm -rf /tmp/ray_tmp_*

# 方案2：找到并杀死特定进程
ps aux | grep raylet | awk '{print $2}' | xargs kill -9
ps aux | grep gcs_server | awk '{print $2}' | xargs kill -9

# 方案3：重启机器（最简单，适合服务器环境）
sudo reboot
```

---

### Bug 5: Ray + vLLM 初始化冲突

**症状**：
- 环境初始化完成，但 vLLM 不加载模型
- GPU 0% 持续很久

**原因**：
Ray Worker 和 vLLM 都要抢占 GPU，导致死锁

**解决方案**：
```bash
# A. 调整 GPU 内存分配
actor_rollout_ref.rollout.gpu_memory_utilization=0.3  # 从 0.6 降到 0.3

# B. 启用 eager 模式（禁用 CUDA graph）
actor_rollout_ref.rollout.enforce_eager=True

# C. 减少并行度
env.rollout.n=16  # 从 64 降到 16
trainer.n_gpus_per_node=1

# D. 使用 Ray 的 GPU 隔离
在启动 Ray 时：
ray start --head --num-gpus=1 --resources='{"gpu_memory": 1}'
```

---

## 🛠️ Ray 调试工具箱

### 1. **日志文件位置**
```bash
# Ray 日志根目录
/tmp/ray/session_latest/logs/

# 关键日志（按重要性排序）
raylet.out          # Raylet 进程日志（资源调度）
gcs_server.out      # GCS 服务器日志（集群状态）
worker-*.out        # Worker 进程输出
worker-*.err        # Worker 错误输出
python-core-*.log   # Python 核心 Worker 日志
```

### 2. **Ray 状态检查命令**
```bash
# 集群状态
ray status

# 详细资源信息
ray status --verbose

# 查看所有 actor
ray list actors

# 查看所有任务
ray list tasks

# Ray Dashboard（浏览器访问）
http://localhost:8265
```

### 3. **调试环境变量**
```bash
# 启用详细日志
export RAY_BACKEND_LOG_LEVEL=debug
export VLLM_LOGGING_LEVEL=DEBUG

# 禁用某些功能（排查问题）
export RAY_DISABLE_MEMORY_MONITOR=1        # 禁用内存监控
export RAY_memory_monitor_refresh_ms=0     # 禁用 cgroup
export RAY_DEDUP_LOGS=0                    # 不合并日志
export PYTHONUNBUFFERED=1                  # 实时输出

# 本地模式（不启动集群，便于调试）
export RAY_local_mode=1
```

---

## 📋 标准 Ray 问题排查流程

### 步骤1：确认问题类型
```bash
# 能启动但卡住？
tail -f /tmp/ray/session_latest/logs/raylet.out

# 启动就报错？
查看控制台输出

# Worker 失败？
cat /tmp/ray/session_latest/logs/worker-*.err
```

### 步骤2：检查资源配置
```bash
# GPU 数量是否匹配？
nvidia-smi --list-gpus
ray status | grep GPU

# CPU/内存是否充足？
free -h
nproc
```

### 步骤3：隔离问题
```bash
# 测试1：Ray 能否正常启动？
ray stop --force
ray start --head --num-gpus=1
ray status

# 测试2：vLLM 能否单独加载模型？
python -c "from vllm import LLM; llm = LLM('YOUR_MODEL_PATH')"

# 测试3：环境能否正常初始化？
python -c "from agent_system.environments.env_package.alfworld import AlfworldEnv; env = AlfworldEnv()"

# 测试4：最小化配置能否运行？
# 只用 1 个 GPU、4 个任务
```

### 步骤4：清理并重试
```bash
# 完全清理
pkill -9 -f ray && pkill -9 -f python && pkill -9 -f vllm
rm -rf /tmp/ray/*
sleep 5

# 手动启动 Ray
ray start --head --num-gpus=1 --num-cpus=8

# 运行简化版脚本
```

---

## 💡 经验总结

### 黄金法则：
1. **永远先看日志** - `raylet.out` 和 `gcs_server.out` 是真相
2. **资源要对齐** - GPU 数量、内存、CPU 要与实际硬件匹配
3. **从小到大测试** - 先 4 个任务、1 个 GPU，再逐步增加
4. **清理要彻底** - 残留进程和临时文件会导致奇怪的问题
5. **手动启动 Ray** - 自动初始化经常出问题，手动更可控

### 常见场景的快速解决方案：

#### 场景1：评测脚本卡住，GPU 0%
```bash
# 1. 检查 GPU 数量
nvidia-smi --list-gpus

# 2. 如果只有 1 个 GPU，运行：
cd /root/digitalhuman/RLVMR/code
pkill -9 -f ray && ray stop --force && sleep 5
ray start --head --num-gpus=1 --num-cpus=8
ray status  # 确认成功

# 3. 运行评测（明确指定 1 个 GPU，减少任务数）
bash examples/bdrs_trainer/eval_cold_start.sh \
    /path/to/model \
    16 \
    vllm \
    eval_test

# 4. 在另一个终端监控日志
tail -f /tmp/ray/session_latest/logs/raylet.out
```

#### 场景2：之前能跑，现在不行
```bash
# 最可能的原因：残留进程或资源冲突
pkill -9 -f ray
pkill -9 -f python
rm -rf /tmp/ray/*
sleep 5

# 重新运行
```

#### 场景3：SFT 训练后评测失败
```bash
# 检查模型 checkpoint 是否完整
ls -lh /path/to/checkpoint/

# 测试模型能否加载
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained('/path/to/checkpoint', device_map='cpu')
print('✓ Model OK')
"
```

---

## 🆘 紧急救援命令

当一切都不工作时：

```bash
# 终极清理
pkill -9 -f ray
pkill -9 -f python
pkill -9 -f vllm
pkill -9 -f verl
rm -rf /tmp/ray/*
rm -rf /tmp/ray_tmp_*
rm -rf ~/.cache/ray/*
sleep 10

# 重启 Ray（最小配置）
ray start --head --num-gpus=1 --num-cpus=4

# 检查状态
ray status
nvidia-smi

# 如果还不行，重启服务器
sudo reboot
```

---

## 📞 获取帮助的正确方式

提问时请提供：
1. **完整的错误日志**（至少最后 100 行）
   ```bash
   tail -100 /tmp/ray/session_latest/logs/raylet.out
   tail -100 /tmp/ray/session_latest/logs/gcs_server.out
   ```

2. **系统资源信息**
   ```bash
   nvidia-smi
   free -h
   ray status
   ```

3. **完整的启动命令和配置**

4. **问题复现步骤**

这样别人才能快速帮您定位问题！
