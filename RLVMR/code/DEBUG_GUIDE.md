# Ray 调试工具使用指南

本指南帮助你快速定位和解决评测脚本卡住的问题。

## 🚀 快速开始

### 1. 安装必要的调试工具

```bash
# 安装 py-spy（强烈推荐）
pip install py-spy

# 安装 strace（系统级调试）
sudo apt-get install strace  # Ubuntu/Debian
# 或
sudo yum install strace      # CentOS/RHEL
```

### 2. 运行调试指南

```bash
cd /root/digitalhuman/RLVMR/code

# 查看完整的调试指南
bash debug_with_strace.sh
```

## 📊 方法 A: 使用 Ray Dashboard（最直观）

Ray Dashboard 提供了图形化界面来监控分布式任务。

### 步骤：

1. **找到 Dashboard URL**
   ```bash
   # Ray 默认运行在 http://localhost:8265
   # 如果在远程服务器，设置端口转发
   ssh -L 8265:localhost:8265 user@remote-host
   ```

2. **在浏览器中打开**
   ```
   http://localhost:8265
   ```

3. **查看关键信息**
   - **Jobs** 标签页：查看任务执行状态
   - **Actors** 标签页：查看 Actor 状态和日志
   - **Logs** 标签页：查看实时日志输出
   - **Cluster** 标签页：查看资源使用情况

## 🔧 方法 B: 使用自定义 Ray 调试工具

我们提供了专门的 Python 脚本来调试 Ray 任务。

### 基本用法：

```bash
cd /root/digitalhuman/RLVMR/code

# 1. 检查 Ray 集群状态
python debug_ray_toolkit.py status

# 2. 列出所有任务
python debug_ray_toolkit.py list-tasks

# 3. 列出运行中的任务
python debug_ray_toolkit.py list-tasks --state RUNNING

# 4. 实时监控（推荐！）
python debug_ray_toolkit.py monitor --interval 5

# 5. 查看任务详情
python debug_ray_toolkit.py task-details <task_id>

# 6. 列出所有 Actor
python debug_ray_toolkit.py list-actors

# 7. 查看 Actor 详情
python debug_ray_toolkit.py actor-details <actor_id>

# 8. 查看日志位置
python debug_ray_toolkit.py logs
```

### 典型调试流程：

```bash
# 在一个终端运行评测脚本
bash examples/bdrs_trainer/eval_cold_start.sh

# 在另一个终端运行监控
python debug_ray_toolkit.py monitor --interval 5
```

监控会显示：
- 总任务数和各状态任务数
- 长时间运行的任务（>30秒）
- 实时更新

## 🐍 方法 C: 使用 py-spy 获取堆栈跟踪

py-spy 可以在不中断进程的情况下查看 Python 代码执行位置。

### 步骤：

1. **找到卡住的进程 PID**
   ```bash
   # 查找所有 Python 进程
   ps aux | grep python | grep main_ppo

   # 或者查找 TaskRunner 进程
   ps aux | grep TaskRunner
   ```

2. **使用 py-spy 查看堆栈**
   ```bash
   # 一次性输出当前堆栈
   py-spy dump --pid <PID>

   # 实时监控（显示最热的函数）
   py-spy top --pid <PID>

   # 记录30秒的性能数据到文件
   py-spy record -o profile.svg --pid <PID> --duration 30
   # 然后用浏览器打开 profile.svg
   ```

3. **分析输出**
   ```
   如果看到堆栈卡在某个特定函数，比如：
   - alfworld.agents.environment.AlfredTWEnv.__init__
   - env_manager.make_envs
   - 某个文件 I/O 操作

   那就找到了问题所在！
   ```

## 🔍 方法 D: 使用环境挂起调试脚本

我们提供了专门测试环境创建的脚本。

```bash
cd /root/digitalhuman/RLVMR/code

# 运行所有测试（30秒超时）
python debug_env_hang.py --timeout 30

# 只测试特定部分
python debug_env_hang.py --test import      # 测试导入
python debug_env_hang.py --test config      # 测试配置加载
python debug_env_hang.py --test env         # 测试环境创建
python debug_env_hang.py --test make_envs   # 测试 make_envs 函数
```

这个脚本会：
- 检查环境配置
- 逐步测试环境初始化的各个阶段
- 在超时时给出诊断建议

## 📝 方法 E: 使用 strace 追踪系统调用

strace 可以看到进程在做什么系统级操作（文件读写、网络等）。

```bash
# 找到进程 PID
ps aux | grep python | grep main_ppo

# 追踪进程的系统调用
sudo strace -p <PID> -f -s 1000 -o strace.log

# 在另一个终端查看输出
tail -f strace.log

# 只看特定系统调用
sudo strace -p <PID> -f -e trace=open,read,write,stat
```

如果看到进程不断重复某个操作，那可能就是卡住的原因。

## 🎯 实战示例：定位你的问题

根据你的描述，进程卡在了：
```
(TaskRunner pid=184752) [DEBUG make_envs] generalization_level=0
(TaskRunner pid=184752) [DEBUG make_envs] Building training envs (level 0)...
```

### 立即尝试：

```bash
# 终端 1: 启动评测
cd /root/digitalhuman/RLVMR/code
bash examples/bdrs_trainer/eval_cold_start.sh

# 等待卡住后，在终端 2 执行：

# 方法 1: 使用我们的监控工具
python debug_ray_toolkit.py monitor

# 方法 2: 找到 TaskRunner 进程并查看堆栈
ps aux | grep TaskRunner
# 假设 PID 是 184752
py-spy dump --pid 184752

# 方法 3: 测试环境创建
python debug_env_hang.py --timeout 30 --test make_envs
```

### 预期结果：

1. **py-spy** 会告诉你代码卡在哪一行
2. **监控工具** 会显示哪个任务长时间运行
3. **环境测试** 会告诉你是否是环境初始化的问题

## 📋 常见问题诊断

### 问题 1: 环境数据未下载

```bash
# 检查 ALFWORLD_DATA
echo $ALFWORLD_DATA
ls -la $ALFWORLD_DATA

# 如果为空，下载数据
conda activate rlvmr-alfworld
alfworld-download
```

### 问题 2: Ray 资源不足

```bash
# 检查资源
python debug_ray_toolkit.py status

# 如果 GPU/CPU 不足，减少并行数
# 编辑 eval_cold_start.sh，修改：
env.rollout.n=4  # 改小这个值，比如改成 2
```

### 问题 3: 进程真的卡住了

```bash
# 使用 py-spy 确认
py-spy top --pid <PID>

# 如果发现卡在某个函数，在代码中添加更多日志
# 或者使用调试器
```

## 🆘 如果还是无法解决

1. **收集诊断信息**
   ```bash
   # 运行完整诊断
   python debug_ray_toolkit.py status > debug_status.txt
   python debug_ray_toolkit.py list-tasks > debug_tasks.txt
   python debug_env_hang.py > debug_env.txt 2>&1

   # 如果进程卡住
   py-spy dump --pid <PID> > debug_stack.txt
   ```

2. **查看 Ray 日志**
   ```bash
   python debug_ray_toolkit.py logs
   # 根据输出的路径查看日志文件
   ```

3. **添加更详细的日志**
   打开 `agent_system/environments/env_manager.py`，
   在 `make_envs` 函数中添加更多 print 语句。

## 📚 工具对比

| 工具 | 优点 | 缺点 | 推荐场景 |
|------|------|------|----------|
| **Ray Dashboard** | 图形化、直观 | 需要浏览器 | 整体监控 |
| **debug_ray_toolkit.py** | 灵活、详细 | 命令行 | 快速诊断 |
| **py-spy** | 精确定位代码位置 | 需要进程 PID | 定位卡住位置 |
| **debug_env_hang.py** | 针对性强 | 仅测试环境 | 环境初始化问题 |
| **strace** | 系统级视角 | 输出庞大 | 底层问题 |

## ✅ 下一步

1. **先试试最简单的**：
   ```bash
   python debug_ray_toolkit.py monitor
   ```

2. **如果还在卡**：
   ```bash
   # 找到 PID，使用 py-spy
   ps aux | grep python | grep main_ppo
   py-spy dump --pid <PID>
   ```

3. **根据堆栈输出**，我们就能精确定位问题所在了！

需要我陪你一步步执行吗？
