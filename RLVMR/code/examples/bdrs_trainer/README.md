# BDRS (Belief-Driven Reward Shaping) 训练指南

## 概述
BDRS 是基于信念驱动的内在奖励塑造框架，通过维护智能体的内部状态（世界模型、任务进展、探索地图）来提供密集的训练信号。

## 快速开始

### 1. 环境准备
```bash
# 安装依赖
pip install -r requirements.txt
pip install -e .

# 设置环境变量
export OPENAI_API_KEY="your_api_key_here"
export WANDB_API_KEY="your_wandb_key_here"  # 可选
```

### 2. 生成冷启动数据
```bash
# 使用 BDRS 模板生成冷启动数据
python scripts/alfworld_prepare.py
# 输出: data/alfworld_cold-start.json
```

### 3. 训练 BDRS 模型
```bash
# ALFWorld 环境
bash examples/bdrs_trainer/run_alfworld.sh

# ScienceWorld 环境  
bash examples/bdrs_trainer/run_sciworld.sh
```

### 4. 快速验证
```bash
# 生成测试（无训练）
bash examples/bdrs_trainer/eval_alfworld.sh
bash examples/bdrs_trainer/eval_sciworld.sh
```

## 配置说明

### BDRS 关键参数
```yaml
algorithm:
  adv_estimator: bdrs
  bdrs:
    enable: True
    world_consistency_weight: 1.0    # 世界模型一致性权重
    task_progress_weight: 1.0        # 任务进展权重
    exploration_efficiency_weight: 1.0 # 探索效率权重
    step_advantage_w: 1.0            # step级别优势权重
    mode: "mean_std_norm"            # 归一化模式
```

### 环境配置
```yaml
env:
  env_name: alfworld/AlfredTWEnv
  max_steps: 30
  rollout:
    n: 8  # 环境分组大小
```

## 监控指标

### WandB 指标
- `bdrs_world_consistency_mean/min/max`: 世界模型一致性
- `bdrs_task_progress_mean/min/max`: 任务进展奖励
- `bdrs_exploration_efficiency_mean/min/max`: 探索效率奖励
- `valid_action_ratio`: 有效动作比例
- `episode_rewards_mean`: 平均episode奖励

### 日志检查点
训练过程中关注：
1. **信念状态更新**: 每步的 `info['belief']` 是否正常更新
2. **奖励分量**: `bdrs_components` 是否随探索和一致性变化
3. **模板切换**: 是否使用了 BDRS 风格的提示模板

## 故障排除

### 常见问题
1. **导入错误**: 确保 `code/bdrs/` 模块在 Python 路径中
2. **配置错误**: 检查 `algorithm.bdrs.enable=True` 和 `algorithm.adv_estimator=bdrs`
3. **信念状态为空**: 检查环境管理器是否正确初始化 `BeliefStateManager`

### 调试技巧
```python
# 在 env_manager.py 的 step 方法中添加调试打印
if step_idx % 5 == 0:
    print(f"[BDRS] env={i} step={snap.step_idx}")
    print(f"  world_model: {snap.world_model}")
    print(f"  bdrs_components: {step.get('bdrs_components', {})}")
```

## 对比实验

### RLVMR vs BDRS
```bash
# RLVMR 基线
bash examples/rlvmr_trainer/run_alfworld.sh

# BDRS 改进
bash examples/bdrs_trainer/run_alfworld.sh
```

### 关键对比指标
- 成功率 (Success Rate)
- 平均episode长度
- 有效动作比例
- 训练收敛速度
- 信念状态质量

## 扩展开发

### 自定义奖励分量
修改 `code/bdrs/bdrs_rewards.py` 中的 `step_reward` 方法：
```python
def step_reward(self, belief, info):
    # 添加自定义奖励逻辑
    custom_reward = self._compute_custom_reward(belief, info)
    return {
        'world_consistency': ...,
        'task_progress': ...,
        'exploration_efficiency': ...,
        'custom': custom_reward,
        'total': ...
    }
```

### 新增环境支持
1. 在 `env_manager.py` 中添加新环境管理器
2. 在 `prompts/` 中添加对应的 BDRS 模板
3. 更新 `belief_state.py` 中的启发式规则

## 性能优化

### 内存优化
- 限制信念状态历史长度
- 定期清理探索地图
- 使用轻量级信念快照

### 计算优化
- 缓存一致性检查结果
- 批量处理信念更新
- 异步奖励计算

## 联系与支持
如有问题，请检查：
1. 日志中的错误信息
2. 配置文件的正确性
3. 依赖包的版本兼容性
