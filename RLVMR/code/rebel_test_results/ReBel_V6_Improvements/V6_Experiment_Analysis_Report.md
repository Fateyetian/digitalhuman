# ReBel V6 实验分析报告

> 实验时间: 2026年1月7日 - 2026年1月9日
> 实验路径: `/fs-computility-new/UPDZ03_chengjun/huangsijie.p/rebel_results/v6_experiments/`
> 分析日期: 2026年1月13日

---

## 1. 实验概述

V6版本在V5的基础上进行了三个核心代码改进：
1. **Stage Type 优先级修复**: "find" 优先于 "lamp"
2. **Ground Truth 宽松验证**: 早期阶段(step<3)或GT不完整时使用结构化奖励
3. **look_at 专用进度检测**: 4阶段奖励(拾取+找灯+使用+完成)

### 1.1 实验矩阵

| 实验 | 名称 | min_samples_ratio | KL方式 | KL系数 |
|------|------|-------------------|--------|--------|
| V6-Exp1 | Baseline | 0.0 | KL in Loss | 0.01 |
| V6-Exp2 | Relative Norm | 0.15 | KL in Loss | 0.01 |
| V6-Exp3 | Full Improvements | 0.15 | KL in Reward | 0.001 |

### 1.2 共同配置
- belief_granularity: adaptive
- task_aware_grouping: true
- per_task_normalization: true
- conditional_norm: true
- min_samples_for_norm: 10
- min_std_for_norm: 0.2
- GPU数量: 4
- 训练轮数: 60

---

## 2. 实验结果对比

### 2.1 最终成功率 (Step 60)

| 任务类型 | V6-Exp1 | V6-Exp2 | V6-Exp3 | 最佳实验 |
|----------|---------|---------|---------|----------|
| **Overall** | 56.2% | 78.1% | **84.4%** | Exp3 |
| pick_and_place | 78.4% | 94.6% | **97.3%** | Exp3 |
| pick_two_obj | 60.9% | 56.5% | **87.0%** | Exp3 |
| pick_heat | 66.7% | 83.3% | **88.9%** | Exp3 |
| pick_cool | 36.0% | **72.0%** | 68.0% | Exp2 |
| pick_clean | 26.3% | 73.7% | **94.7%** | Exp3 |
| **look_at** | 50.0% | **83.3%** | 16.7% | **Exp2** |

### 2.2 关键发现

```
                    V6-Exp1    V6-Exp2    V6-Exp3
Overall Success:    56.2%      78.1%      84.4%  ← Exp3最高
look_at Success:    50.0%      83.3%      16.7%  ← Exp2最高，Exp3崩溃
Training Stability: 不稳定     稳定        部分稳定
```

### 2.3 训练曲线分析

**V6-Exp1 (Baseline)**:
- 峰值成功率约72.7%，但后期回落至56.2%
- look_at任务波动较大，最终50%
- 特征：训练不稳定，存在过拟合或遗忘现象

**V6-Exp2 (Relative Norm)**:
- 持续稳定提升，最终78.1%
- look_at任务表现最佳(83.3%)
- 特征：训练稳定，各任务均衡发展

**V6-Exp3 (Full Improvements + KL in Reward)**:
- 整体成功率最高(84.4%)
- 但look_at任务严重退化(16.7%)
- 特征：主流任务优化过度，少数任务被牺牲

---

## 3. 奖励归一化方法分析

### 3.1 相对阈值归一化 (min_samples_ratio)

| 参数设置 | 效果分析 |
|----------|----------|
| `min_samples_ratio: 0.0` (Exp1) | 无相对阈值，少数任务样本可能被过度归一化 |
| `min_samples_ratio: 0.15` (Exp2/3) | 相对阈值生效，少数任务样本保护 |

**观察到的adv_std值**:
```
V6-Exp1 (step 60):
  adv_std_heat: 1.000
  adv_std_clean: 1.000
  adv_std_and_place: 0.144  ← 条件归一化生效

V6-Exp2 (step 60):
  adv_std_two_and_place: 1.000
  adv_std_and_place: 0.097  ← 条件归一化生效
  adv_std_cool: 1.000

V6-Exp3 (step 60):
  adv_std_heat: 1.000
  adv_std_and_place: 0.091  ← 条件归一化生效
```

**结论**: 当任务样本数过少或标准差过低时，条件归一化正确跳过了归一化，避免了人工放大噪声。`min_samples_ratio: 0.15`的设置有助于保护少数任务。

### 3.2 任务级归一化 (per_task_normalization)

所有实验均启用任务级归一化。观察到的任务分组数：
```
V6-Exp1: num_groups=140, mean_group_size=29.3
V6-Exp2: num_groups=134, mean_group_size=32.5
V6-Exp3: num_groups=129, mean_group_size=27.8
```

任务级分组数量稳定，表明任务感知分组策略正常工作。

---

## 4. KL散度控制分析

### 4.1 两种KL控制方式对比

| 方式 | 实现 | 实验 |
|------|------|------|
| **KL in Loss** | `loss = pg_loss + kl_coef * kl_loss` | Exp1, Exp2 |
| **KL in Reward** | `reward = base_reward - kl_coef * kl` | Exp3 |

### 4.2 KL指标对比

```
V6-Exp1 (KL in Loss):
  actor/kl_loss: 0.048
  actor/kl_coef: 0.010

V6-Exp2 (KL in Loss):
  actor/kl_loss: 0.043
  actor/kl_coef: 0.010

V6-Exp3 (KL in Reward):
  actor/reward_kl_penalty: 0.085
  actor/reward_kl_penalty_coeff: 0.001
```

### 4.3 关键分析

**KL in Reward的问题**:
1. KL惩罚直接作用于奖励，导致高KL的探索行为被惩罚
2. 少数任务(如look_at)本身需要更多探索，但KL惩罚抑制了探索
3. 系数虽小(0.001)，但累积效应明显

**KL in Loss的优势**:
1. KL约束作用于整体策略，不直接影响单个样本的奖励
2. 允许局部高KL探索，同时保持整体策略稳定
3. Exp2证明这种方式对少数任务更友好

**结论**: 对于任务不平衡的场景，**KL in Loss优于KL in Reward**。

---

## 5. 熵坍塌分析

### 5.1 熵指标

由于训练日志未直接记录entropy指标，我们通过以下间接指标分析：

| 指标 | V6-Exp1 | V6-Exp2 | V6-Exp3 |
|------|---------|---------|---------|
| response_length/mean | 390 | 341 | 344 |
| pg_clipfrac | 0.003 | 0.004 | 0.004 |
| ppo_kl | 0.000 | 0.000 | 0.001 |
| grad_norm | 0.358 | 0.481 | 0.638 |

### 5.2 分析

- **response_length**: Exp1生成更长的响应，可能表示更多探索
- **pg_clipfrac**: 所有实验clip比例很低，策略更新保守
- **grad_norm**: Exp3梯度最大，策略变化更激进
- **ppo_kl**: 所有实验PPO KL约束正常工作

**潜在熵坍塌风险**:
- V6-Exp3的look_at从中期的较高成功率(如step 25约58.3%)急剧下降至16.7%
- 这可能是局部熵坍塌：模型对look_at任务的策略过度确定化
- KL in Reward机制可能加速了这种坍塌

---

## 6. look_at任务低成功率深度分析

### 6.1 问题现象

| 实验 | look_at最佳成功率 | look_at最终成功率 | 变化 |
|------|-------------------|-------------------|------|
| V6-Exp1 | ~58% | 50% | -8% |
| V6-Exp2 | 83.3% | 83.3% | 0% |
| V6-Exp3 | ~58% | 16.7% | **-41%** |

### 6.2 根因分析

**1. 样本数量不平衡**
```
任务分组数 (step 60):
  pick_heat: 58-65组
  pick_two: 24-25组
  pick_cool: 15-20组
  pick_clean: 14-24组
  pick_and_place: 11-14组
  look_at: 未单独统计，估计<10组
```
look_at任务样本最少，在优化中信号较弱。

**2. KL in Reward的负面影响**
- V6-Exp3使用KL in Reward
- look_at任务需要特定的探索策略(找灯→使用)
- KL惩罚抑制了这种探索

**3. 任务间优化竞争**
- 当主流任务(pick_*)持续优化时，梯度主要来自这些任务
- look_at任务的梯度信号被淹没

### 6.3 V6代码改进对look_at的影响

V6增加了look_at专用进度检测(4阶段奖励)，但：
- 在Exp1/Exp2中效果明显
- 在Exp3中被KL in Reward抵消

---

## 7. 训练稳定性分析

### 7.1 高性能任务的稳定化

当某些任务达到高性能时(如pick_and_place >95%)，需要考虑：

| 问题 | 解决方案 |
|------|----------|
| 梯度信号过小 | 使用task-aware分组，避免跨任务干扰 |
| 过拟合风险 | conditional_norm跳过低方差任务的归一化 |
| 探索不足 | 保持适当的KL约束 |

### 7.2 实验中观察到的稳定化机制

**V6-Exp2表现最稳定**:
1. `min_samples_ratio: 0.15` 保护少数任务
2. KL in Loss 不直接干预奖励
3. 条件归一化(`adv_std_and_place: 0.097`)正确跳过高性能任务

**V6-Exp3的不稳定来源**:
1. KL in Reward 过度惩罚探索
2. 高性能任务的KL惩罚较小，低性能任务的KL惩罚较大
3. 导致"富者愈富"的马太效应

---

## 8. 信念奖励与世界一致性奖励分析

### 8.1 信念奖励组成

根据ReBel算法设计，信念奖励包括：
1. **一致性奖励(consistency)**: 信念与观察的匹配程度
2. **进度奖励(progress)**: 任务阶段推进奖励
3. **探索奖励(exploration)**: 新状态发现奖励
4. **格式奖励(format)**: 输出格式正确性

### 8.2 V6改进对信念奖励的影响

V6的"Ground Truth 宽松验证"改进：
- 早期阶段使用结构化奖励而非严格GT匹配
- 提高了信念奖励的稳定性
- 减少了因GT不完整导致的假阴性惩罚

### 8.3 世界一致性奖励效果

从训练指标看：
```
critic/rewards/mean:
  V6-Exp1: 7.178
  V6-Exp2: 7.091
  V6-Exp3: 9.110  ← 最高奖励
```

V6-Exp3的平均奖励最高，但这主要来自主流任务的高成功率，而非世界一致性奖励的提升。

---

## 9. 轨迹保存情况

检查实验目录结构：
```
rebel_v6_exp1_baseline_20260107_120300/
├── checkpoints/
│   └── global_step_60/
│       └── actor/
├── experiment_config.txt
└── training.log
```

**结论**: 当前实验**未保存训练/评估轨迹**。仅保存了：
- 模型检查点 (checkpoints)
- 实验配置 (experiment_config.txt)
- 训练日志 (training.log)

**建议**: 后续实验应启用轨迹保存，用于：
- 错误案例分析
- 信念追踪可视化
- 奖励组成分析

---

## 10. 结论与建议

### 10.1 最佳配置推荐

基于V6实验结果，推荐配置：

```yaml
# 推荐配置 (基于V6-Exp2)
belief_granularity: adaptive
task_aware_grouping: true
per_task_normalization: true
conditional_norm: true
min_samples_for_norm: 10
min_std_for_norm: 0.2
min_samples_ratio: 0.15  # 关键：相对阈值归一化
use_kl_in_reward: false   # 关键：使用KL in Loss
use_kl_loss: true
kl_loss_coef: 0.01
```

### 10.2 各配置因素评估

| 因素 | 正面影响 | 负面影响 |
|------|----------|----------|
| task_aware_grouping | 避免跨任务干扰 | - |
| per_task_normalization | 任务间公平性 | 可能放大噪声 |
| conditional_norm | 保护高性能任务 | - |
| min_samples_ratio: 0.15 | 保护少数任务 | - |
| **KL in Loss** | 允许局部探索 | - |
| **KL in Reward** | - | 抑制少数任务探索 |

### 10.3 后续改进方向

1. **任务自适应KL系数**
   - 根据任务样本数动态调整KL系数
   - 少数任务使用较小KL约束

2. **熵正则化**
   - 添加显式熵奖励项
   - 防止少数任务策略坍塌

3. **样本重加权**
   - 提升少数任务样本权重
   - 平衡梯度贡献

4. **早停策略**
   - 监控各任务成功率
   - 当某任务退化超过阈值时调整学习率

5. **轨迹保存**
   - 启用训练/评估轨迹保存
   - 便于深度分析和调试

---

## 附录A: 详细指标表

### A.1 V6-Exp1 最终指标 (Step 60)
```
val/success_rate: 0.5625
val/pick_heat_then_place_in_recep_success_rate: 0.6667
val/pick_two_obj_and_place_success_rate: 0.6087
val/pick_cool_then_place_in_recep_success_rate: 0.3600
val/pick_and_place_success_rate: 0.7838
val/pick_clean_then_place_in_recep_success_rate: 0.2632
val/look_at_obj_in_light_success_rate: 0.5000
actor/kl_loss: 0.048
rebel/num_groups: 140
rebel/mean_group_size: 29.3
```

### A.2 V6-Exp2 最终指标 (Step 60)
```
val/success_rate: 0.7813
val/pick_heat_then_place_in_recep_success_rate: 0.8333
val/pick_two_obj_and_place_success_rate: 0.5652
val/pick_cool_then_place_in_recep_success_rate: 0.7200
val/pick_and_place_success_rate: 0.9459
val/pick_clean_then_place_in_recep_success_rate: 0.7368
val/look_at_obj_in_light_success_rate: 0.8333
actor/kl_loss: 0.043
rebel/num_groups: 134
rebel/mean_group_size: 32.5
```

### A.3 V6-Exp3 最终指标 (Step 60)
```
val/success_rate: 0.8438
val/pick_heat_then_place_in_recep_success_rate: 0.8889
val/pick_two_obj_and_place_success_rate: 0.8696
val/pick_cool_then_place_in_recep_success_rate: 0.6800
val/pick_and_place_success_rate: 0.9730
val/pick_clean_then_place_in_recep_success_rate: 0.9474
val/look_at_obj_in_light_success_rate: 0.1667
actor/reward_kl_penalty: 0.085
rebel/num_groups: 129
rebel/mean_group_size: 27.8
```

---

## 附录B: 实验配置详情

### B.1 V6-Exp1 配置
```
实验名称: rebel_v6_exp1_baseline_20260107_120300
实验描述: V6-Exp1: Baseline (KL in Loss)
min_samples_ratio: 0.0
use_kl_in_reward: False
use_kl_loss: True
kl_loss_coef: 0.01
```

### B.2 V6-Exp2 配置
```
实验名称: rebel_v6_exp2_relative_norm_20260108_015035
实验描述: V6-Exp2: 相对阈值归一化
min_samples_ratio: 0.15
use_kl_in_reward: False
use_kl_loss: True
kl_loss_coef: 0.01
```

### B.3 V6-Exp3 配置
```
实验名称: rebel_v6_exp3_full_improvements_20260108_160959
实验描述: V6-Exp3: 全部改进 + KL in Reward
min_samples_ratio: 0.15
use_kl_in_reward: True
kl_penalty: kl
kl_coef: 0.001
use_kl_loss: False
kl_loss_coef: 0
```

---

*报告生成时间: 2026-01-13*
*分析工具: Claude Code*
