# Ray 调试工具 - 完整使用教程

## 🎯 你的问题

你的评测脚本卡在了环境初始化阶段：
```
(TaskRunner pid=184752) [DEBUG make_envs] generalization_level=0
(TaskRunner pid=184752) [DEBUG make_envs] Building training envs (level 0)...
```

## 📦 我为你准备的工具

我创建了以下调试工具来帮你定位问题：

```
code/
├── debug_ray_toolkit.py       # Ray 任务监控和调试工具
├── debug_env_hang.py          # 环境挂起专项调试工具
├── debug_with_strace.sh       # 系统级调试指南
├── start_debugging.sh         # 一键启动调试（推荐！）
└── DEBUG_GUIDE.md             # 完整使用指南
```

## 🚀 快速开始（3 步搞定）

### 第 1 步：安装调试工具

```bash
cd /root/digitalhuman/RLVMR/code

# 安装 py-spy（强烈推荐）
pip install py-spy
```

### 第 2 步：启动评测脚本

在**第一个终端**运行：

```bash
cd /root/digitalhuman/RLVMR/code
bash examples/bdrs_trainer/eval_cold_start.sh
```

等待它卡住...

### 第 3 步：运行调试工具

在**第二个终端**运行：

```bash
cd /root/digitalhuman/RLVMR/code

# 方法 A: 一键启动菜单（最简单）
bash start_debugging.sh

# 或者方法 B: 直接监控（推荐）
python debug_ray_toolkit.py monitor --interval 5

# 或者方法 C: 测试环境（针对你的问题）
python debug_env_hang.py --timeout 30
```

## 🔍 详细使用方法

### 方法 1: 一键启动菜单（推荐新手）

```bash
bash start_debugging.sh
```

这会显示一个交互式菜单：

```
请选择调试方法:

1) Ray Dashboard (图形化界面)
2) 实时监控任务 (推荐)
3) 查看集群状态
4) 列出所有任务
5) 测试环境创建
6) 查找并分析卡住的进程 (py-spy)
7) 查看完整调试指南
8) 退出
```

选择 **2** 或 **6** 最有用。

### 方法 2: 实时监控（最常用）

```bash
python debug_ray_toolkit.py monitor --interval 5
```

**输出示例：**
```
[2025-11-09 10:30:45] 迭代 #1
--------------------------------------------------------------------------------
总任务数: 8
  RUNNING: 3
  PENDING_NODE_ASSIGNMENT: 2
  FINISHED: 3

⚠️  长时间运行的任务 (>30秒):
  - make_envs (abc123...): 45.2秒
  - TaskRunner.run (def456...): 52.7秒
```

这会告诉你：
- ✅ 哪些任务在运行
- ⚠️ 哪些任务卡住了（超过30秒）
- 📊 任务的详细信息

### 方法 3: 精确定位卡住位置（最强大）

**步骤 1：找到卡住的进程**

```bash
# 查找 TaskRunner 进程
ps aux | grep TaskRunner

# 或者查找 main_ppo 进程
ps aux | grep python | grep main_ppo
```

**输出示例：**
```
root     184752  98.5  2.3  ... python -m verl.trainer.main_ppo ...
root     184753   0.0  1.5  ... (TaskRunner pid=184753)
```

记住 PID（例如 184752）

**步骤 2：使用 py-spy 查看堆栈**

```bash
# 一次性查看当前堆栈（快速诊断）
py-spy dump --pid 184752

# 实时监控（看哪个函数占用最多时间）
py-spy top --pid 184752

# 录制30秒的性能分析（生成可视化图表）
py-spy record -o profile.svg --pid 184752 --duration 30
# 然后用浏览器打开 profile.svg
```

**py-spy 输出示例：**
```python
Thread 184752 (active): "MainThread"
    make_envs (env_manager.py:156)
    AlfredTWEnv.__init__ (environment.py:89)
    load_json_config (config.py:45)  # <-- 卡在这里！
    open (/usr/lib/python3.10/pathlib.py:1234)
```

这样你就精确知道卡在哪一行代码了！

### 方法 4: 测试环境创建（针对你的问题）

```bash
python debug_env_hang.py --timeout 30
```

这个脚本会：
1. 检查环境配置（ALFWORLD_DATA 等）
2. 逐步测试环境初始化的每个阶段
3. 在超时时告诉你卡在哪里

**输出示例：**
```
================================================================================
环境配置检查
================================================================================

环境变量:
  ALFWORLD_DATA: /root/.alfworld
  CONDA_DEFAULT_ENV: rlvmr-alfworld

ALFWorld 数据:
  ✓ 数据目录存在: /root/.alfworld

================================================================================
测试 1: ALFWorld 导入
================================================================================
正在导入 alfworld...
✓ 导入成功: alfworld 0.3.3

================================================================================
测试 3: 环境创建
================================================================================
正在创建环境...
✗ 操作超时 (30秒)

⚠️  环境创建超时!

可能的原因:
  1. 数据加载慢 - 检查 ALFWORLD_DATA 路径
  2. 环境初始化卡住 - 检查 alfworld 版本
  3. 资源不足 - 检查内存和磁盘空间
```

### 方法 5: Ray Dashboard（图形化）

1. **在浏览器中打开**：`http://localhost:8265`

2. **如果在远程服务器**：
   ```bash
   # 在本地电脑运行
   ssh -L 8265:localhost:8265 root@your-server
   ```

3. **查看关键标签页**：
   - **Jobs**: 任务状态
   - **Actors**: Actor 列表和日志
   - **Logs**: 实时日志
   - **Metrics**: 性能指标

## 🎬 完整调试演示

让我演示一个完整的调试流程：

**场景**：评测脚本卡住了

### 第 1 步：启动监控

```bash
# 终端 1
cd /root/digitalhuman/RLVMR/code
bash examples/bdrs_trainer/eval_cold_start.sh

# 终端 2
python debug_ray_toolkit.py monitor --interval 5
```

**观察到**：`make_envs` 任务运行超过 60 秒

### 第 2 步：获取堆栈

```bash
# 找到进程 PID
ps aux | grep TaskRunner
# 假设 PID = 184752

# 查看堆栈
py-spy dump --pid 184752
```

**输出**：
```python
Thread 184752:
    make_envs (env_manager.py:156)
    AlfredTWEnv.__init__ (alfworld/.../environment.py:89)
    load_tasks (alfworld/.../tasks.py:123)
    # 卡在加载任务数据
```

### 第 3 步：诊断问题

根据堆栈，卡在了加载任务数据。可能原因：

1. **数据目录权限问题**
   ```bash
   ls -la $ALFWORLD_DATA
   ```

2. **数据文件损坏**
   ```bash
   python debug_env_hang.py --test config
   ```

3. **磁盘 I/O 慢**
   ```bash
   # 使用 iotop 查看磁盘 I/O
   sudo iotop -p 184752
   ```

### 第 4 步：解决问题

假设发现是数据目录问题：

```bash
# 重新下载数据
conda activate rlvmr-alfworld
alfworld-download -f

# 或者指定更快的存储位置
export ALFWORLD_DATA=/path/to/faster/storage
```

## 📊 各工具对比

| 工具 | 何时使用 | 优点 | 输出内容 |
|------|----------|------|----------|
| `start_debugging.sh` | 不确定用哪个工具 | 交互式菜单 | 引导式选择 |
| `debug_ray_toolkit.py monitor` | 想实时监控所有任务 | 自动刷新、检测长时间任务 | 任务状态汇总 |
| `py-spy dump` | 想知道代码卡在哪一行 | 精确定位 | Python 堆栈跟踪 |
| `debug_env_hang.py` | 怀疑是环境初始化问题 | 针对性测试 | 环境测试结果 |
| Ray Dashboard | 想要图形化界面 | 直观、功能全面 | 任务、日志、指标 |

## 🔧 常见问题及解决方案

### Q1: py-spy 显示 "Permission denied"

```bash
# 方法 1: 使用 sudo
sudo py-spy dump --pid 184752

# 方法 2: 调整权限
sudo sysctl kernel.yama.ptrace_scope=0
```

### Q2: 找不到卡住的进程

```bash
# 查看所有 Python 进程
ps aux | grep python

# 查看进程树
pstree -p | grep python

# 使用 Ray 工具查看
python debug_ray_toolkit.py list-tasks
python debug_ray_toolkit.py list-actors
```

### Q3: Ray Dashboard 打不开

```bash
# 检查 Ray 是否运行
python debug_ray_toolkit.py status

# 查看 Dashboard 地址
grep "dashboard" ~/.ray_logs/session_latest/logs/*

# 手动指定端口
ray start --head --dashboard-port=8265
```

### Q4: 监控没有显示长时间运行的任务

可能任务已经完成或失败了。查看所有任务：

```bash
python debug_ray_toolkit.py list-tasks --state FAILED
python debug_ray_toolkit.py list-tasks --state FINISHED
```

## 💡 针对你的具体问题

根据你的日志：
```
(TaskRunner pid=184752) [DEBUG make_envs] generalization_level=0
(TaskRunner pid=184752) [DEBUG make_envs] Building training envs (level 0)...
```

**最可能的原因**：

1. ✅ **ALFWorld 环境创建慢**
   - 解决：使用 `debug_env_hang.py` 测试
   - 如果超时，检查 ALFWORLD_DATA

2. ✅ **并行创建多个环境导致资源竞争**
   - 解决：减少 `env.rollout.n` 的值
   - 从 64 改成 4 或 8

3. ✅ **文件 I/O 瓶颈**
   - 解决：使用 SSD 存储数据
   - 或者减少并发数

**立即尝试**：

```bash
# 方案 1: 测试单个环境创建
python debug_env_hang.py --test env --timeout 60

# 方案 2: 使用 py-spy 定位
ps aux | grep TaskRunner
py-spy dump --pid <PID>

# 方案 3: 减少并发数重新运行
# 编辑 eval_cold_start.sh，修改：
env.rollout.n=4  # 从 64 改成 4
```

## 🎓 学习更多

### Ray 调试官方文档
- https://docs.ray.io/en/latest/ray-observability/user-guides/debug-apps.html

### py-spy 文档
- https://github.com/benfred/py-spy

### 查看我创建的完整指南
```bash
cat DEBUG_GUIDE.md
```

## ✅ 下一步行动

1. **安装工具**（如果还没安装）
   ```bash
   pip install py-spy
   ```

2. **重现问题**
   ```bash
   bash examples/bdrs_trainer/eval_cold_start.sh
   ```

3. **选择一个调试方法**
   ```bash
   # 推荐：一键启动
   bash start_debugging.sh
   # 选择选项 6: 查找并分析卡住的进程
   ```

4. **根据输出定位问题，然后告诉我结果！**

需要我陪你一起执行调试吗？把输出发给我，我会帮你分析！
