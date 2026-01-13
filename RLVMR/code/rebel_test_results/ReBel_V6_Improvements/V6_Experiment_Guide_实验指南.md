# ReBel V6 改进实验

## 目标
整体成功率从 71.1% (V5) 提升到 **90%**

## 核心代码改进 (已实现)

### P0 级别: 必须修复

#### 1. Stage Type 优先级修复
**文件**: `rebel/core_rebel.py`
**问题**: "find lamp" 被错误分类为 'use' 而非 'find'
**修复**: 添加 `_get_stage_type_v6()` 函数，确保 'find' 优先于 'lamp'/'light'

```python
# 优先级顺序:
# 1. complete -> 2. find -> 3. navigate -> 4. pickup -> 5. place
# 6. heat/cool/clean -> 7. use -> 8. interact
```

#### 2. Ground Truth 宽松验证
**文件**: `agent_system/environments/env_package/alfworld/belief_tracker.py`
**问题**: 早期阶段 ground_truth 不完整，正确预测被错误惩罚
**修复**: 在 `calculate_consistency_reward()` 中添加宽松模式
- `step < 3` 或 `gt_objects < 3` 时使用结构化奖励
- 奖励合理的 belief 输出结构，而非严格验证

#### 3. look_at 任务专用进度检测
**文件**: `agent_system/environments/env_package/alfworld/belief_tracker.py`
**问题**: look_at 任务无法获得进度奖励（缺乏状态变化检测）
**修复**: 添加 `_calculate_look_at_progress()` 方法
```
阶段1: 拾取目标物体 (+0.15)
阶段2: 找到灯光源 (+0.20)
阶段3: 使用灯检查 (+0.25)
阶段4: 任务完成 (+0.40)
```

### P2 级别: 优化改进

#### 4. 相对阈值归一化
**文件**: `rebel/core_rebel.py`
**改进**: `normalize_advantages_per_task()` 支持 `min_samples_ratio` 参数
- 样本占比 < 15% 的任务使用混合归一化
- 避免小样本任务的噪声放大

#### 5. KL in Reward
**配置**: `algorithm.use_kl_in_reward=True`
- 将 KL 惩罚加入 Reward 而非 Loss
- 与 per_task_normalization 协同

## 实验设计

| 实验 | 配置 | 预期成功率 | 预期 look_at |
|------|------|-----------|--------------|
| V6-Exp1 | 基础配置 (验证代码修复) | 78-82% | 35-45% |
| V6-Exp2 | + 相对阈值归一化 | 82-87% | 50-60% |
| V6-Exp3 | + KL in Reward | 88-92% | 65-75% |

## 运行方式

```bash
# 进入实验目录
cd /root/testttt/RLVMR/code/rebel_test_results/ReBel_V6_Improvements

# 运行单个实验
./run_v6_quick.sh 1    # 实验1
./run_v6_quick.sh 2    # 实验2
./run_v6_quick.sh 3    # 实验3

# 或者使用环境变量
EXPERIMENT=1 bash run_v6_experiments.sh
EXPERIMENT=2 bash run_v6_experiments.sh
EXPERIMENT=3 bash run_v6_experiments.sh
```

## 关键指标监控

```bash
# 查看成功率
grep 'val/success_rate' training.log | tail -10

# 查看 look_at 成功率
grep 'val/look_at_obj_in_light' training.log | tail -10

# 查看分组统计
grep 'rebel/num_groups' training.log | tail -5
grep 'rebel/single_sample_ratio' training.log | tail -5
```

## V5 → V6 对比

| 指标 | V5 | V6 目标 | 主要改进点 |
|------|-----|---------|-----------|
| 整体成功率 | 71.1% | 90% | 全部改进协同 |
| look_at | 8.3% | 75% | 专用进度检测 |
| 分组区分度 | 低 | 高 | Stage Type 修复 |
| 早期奖励 | 负值 | 正值 | 宽松验证 |
