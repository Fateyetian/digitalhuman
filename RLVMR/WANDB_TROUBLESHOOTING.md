# WandB监控问题诊断与解决方案

## 问题现象

运行评测脚本后，WandB平台看不到指标数据。

## 根本原因

查看代码 `ray_trainer.py:896-901`：

```python
if self.val_reward_fn is not None and self.config.trainer.get('val_before_train', True):
    val_metrics = self._validate()
    pprint(f'Initial validation metrics: {val_metrics}')
    logger.log(data=val_metrics, step=self.global_steps)  # ← WandB日志在这里
    if self.config.trainer.get('val_only', False):
        return
```

**关键点**：
1. WandB日志记录**确实会执行**（line 899）
2. 指标通过 `logger.log()` 上传到WandB
3. 如果看不到数据，可能是以下原因：

## 可能原因与解决方案

### 原因1：WandB初始化失败 ⭐ 最常见

**症状**：
- 控制台有报错但被忽略了
- WandB显示 "run finished" 但无数据

**解决方案**：

#### 检查WandB登录状态

```bash
# 在服务器上运行
wandb login

# 或者设置API key
export WANDB_API_KEY=your_api_key_here
```

#### 检查WandB配置

```bash
# 查看WandB状态
wandb status

# 如果提示未登录，运行
wandb login --relogin
```

### 原因2：评测过程中断或失败

**症状**：
- 评测运行一半就报错退出
- WandB run创建了但没有上传指标

**解决方案**：

确保评测完整运行完毕，检查控制台输出：
```bash
# 应该看到类似输出
Initial validation metrics: {'val/success_rate': 0.125, 'val/test_score/agent': 0.25}
```

### 原因3：`total_epochs=0` 导致指标未记录

**问题分析**：

原始脚本 `total_epochs=1` 会在训练结束后记录一次指标，但 `total_epochs=0` 只在 `val_before_train` 时记录。

**解决方案**：

我的新脚本已经处理了这个问题：
```bash
trainer.total_epochs=0        # 不训练
trainer.val_before_train=True # 在训练前验证（实际上就是评测）
```

这样会在 line 899 记录指标。

### 原因4：环境评测没有 `success_rate` 指标

**问题**：

环境评测的指标来自 `info['success_rate']`（ray_trainer.py:661-667），但可能某些环境没有返回这个字段。

**诊断**：

查看控制台输出，找 `Initial validation metrics: {...}`，看看有哪些指标。

**解决方案**：

检查环境是否正确返回 success_rate：

```python
# 在 env_manager.py 中应该有类似代码
info['success_rate'] = 1.0 if done and reward > 0 else 0.0
```

### 原因5：多个Run同名导致覆盖

**症状**：
- WandB上有多个同名run
- 最新的run没有数据

**解决方案**：

每次评测使用不同的实验名称：
```bash
bash eval_cold_start.sh \
    /path/to/model \
    64 \
    vllm \
    eval_step75_run1  # ← 改这里，每次不同
```

## 完整诊断步骤

### 步骤1：检查WandB连接

```bash
# SSH到服务器
ssh your_server

# 检查WandB登录
wandb status

# 如果未登录
wandb login
# 输入你的API key（从 https://wandb.ai/authorize 获取）
```

### 步骤2：运行测试脚本

```bash
cd /root/digitalhuman/RLVMR/code

# 运行评测
bash examples/bdrs_trainer/eval_cold_start.sh
```

### 步骤3：检查控制台输出

**成功的输出应该包含**：
```
wandb: 🚀 View run eval_cold_start_qwen1.5b_step75 at: https://wandb.ai/...
...
(大量评测日志)
...
Initial validation metrics: {'val/success_rate': 0.125, 'val/test_score/agent': 0.25}
wandb: Synced 5 W&B file(s), 0 media file(s), 0 artifact file(s) and 0 other file(s)
```

**如果失败**，会看到：
```
wandb: ERROR Unable to initialize
```
或
```
AssertionError: gen_batch size 560 does not match obs size 64
```

### 步骤4：检查WandB平台

访问 WandB 项目页面：
```
https://wandb.ai/YOUR_USERNAME/RLVMR_Evaluation
```

**应该看到**：
- Run名称：`eval_cold_start_qwen1.5b_step75`
- Charts标签页有指标曲线
- 主要指标：
  - `val/success_rate`：成功率
  - `val/test_score/agent`：平均奖励
  - `val/episode_length`：平均步数（如果有）

### 步骤5：如果仍无数据，开启调试模式

在脚本开头添加：
```bash
export WANDB_MODE=online
export WANDB_CONSOLE=wrap
```

重新运行，观察详细输出。

## 预期的WandB指标

使用环境交互模式，应该看到以下指标：

| 指标名称 | 说明 | 示例值 |
|----------|------|--------|
| `val/success_rate` | 任务成功率 | 0.125 (12.5%) |
| `val/test_score/agent` | 平均奖励 | 0.25 |
| `val/episode_length` | 平均步数 | 12.3 |
| `val/episode_reward` | 平均总奖励 | 1.5 |

## 临时方案：使用本地日志

如果WandB始终有问题，可以只用console日志：

```bash
# 修改脚本，只用console
trainer.logger=[console] \  # 移除wandb
```

然后从控制台输出中提取成功率：
```bash
bash eval_cold_start.sh 2>&1 | tee eval_output.log
grep "success_rate" eval_output.log
```

## 测试WandB连接的简单脚本

创建测试文件 `test_wandb.py`：

```python
import wandb

# 初始化
run = wandb.init(
    project="RLVMR_Evaluation",
    name="test_connection",
)

# 记录测试指标
wandb.log({"test_metric": 0.5})

print("✓ WandB测试成功！")
print(f"查看: {run.url}")

wandb.finish()
```

运行测试：
```bash
python test_wandb.py
```

如果成功，会输出WandB链接。

## 联系我排查

如果以上方案都无效，请提供：
1. 完整的控制台输出（前100行和后100行）
2. WandB项目链接
3. `wandb status` 的输出
4. 是否能在WandB上看到run（即使没有数据）
