# ReBel 超参数搜索实验报告

**生成时间**: 2025-12-31
**实验目标**: 探索 step_advantage_w 超参数对 ReBel 算法性能的影响

---

## 1. 实验概述

### 1.1 实验配置

| 配置项 | 子实验1-3 | 子实验4 |
|--------|-----------|---------|
| **基础模型** | Qwen2.5-1.5B-Instruct (SFT) | 同左 |
| **训练任务数** | 16 | 16 |
| **验证任务数** | 16 | **128** |
| **训练轮数** | 30 | **100** (实际79) |
| **Group Size** | 16 | 16 |
| **学习率** | 1e-6 | 1e-6 |
| **GPU数量** | 4 | 4 |

### 1.2 超参数变量

| 实验名称 | step_advantage_w | belief_granularity | mode |
|----------|------------------|-------------------|------|
| baseline | **1.0** | subgoal | mean_norm |
| step_adv_0.5 | **0.5** | subgoal | mean_norm |
| step_adv_2.0 | **2.0** | subgoal | mean_norm |
| step_adv_0.5_val128_ep100 | **0.5** | subgoal | mean_norm |

---

## 2. 实验结果汇总

### 2.1 总体成功率对比

| 实验 | 验证成功率 | 训练成功率 | 平均奖励 | 平均步数 |
|------|-----------|-----------|---------|---------|
| baseline (w=1.0) | 43.8% | 58.2% | 7.06 | 19.8 |
| step_adv_0.5 (w=0.5) | **62.5%** | 52.7% | 6.62 | 21.8 |
| step_adv_2.0 (w=2.0) | 43.8% | 61.3% | 7.37 | 19.6 |
| step_adv_0.5_val128 (w=0.5) | **70.3%** | 93.4% | 10.13 | 12.0 |

### 2.2 各任务类型成功率 (ALFWorld 验证集)

| 任务类型 | baseline | step_adv_0.5 | step_adv_2.0 | step_adv_0.5_val128 |
|----------|----------|--------------|--------------|---------------------|
| **Pick and Place** | 33.3% | 66.7% | 33.3% | **92.1%** |
| **Pick Heat then Place** | 0.0% | **100.0%** | **100.0%** | 64.3% |
| **Pick Clean then Place** | 25.0% | **75.0%** | 50.0% | 65.2% |
| **Pick Two Objects and Place** | 50.0% | 25.0% | 25.0% | **75.0%** |
| **Pick Cool then Place** | **100.0%** | **100.0%** | 50.0% | 36.4% |
| **Look at Object in Light** | **100.0%** | 0.0% | 0.0% | 71.4% |
| **Overall** | 43.8% | 62.5% | 43.8% | **70.3%** |

### 2.3 其他关键指标

| 指标 | baseline | step_adv_0.5 | step_adv_2.0 | step_adv_0.5_val128 |
|------|----------|--------------|--------------|---------------------|
| **有效动作比例** | 94.1% | 92.7% | 92.5% | **97.7%** |
| **KL散度** | 0.060 | 0.061 | 0.052 | 0.083 |
| **最大内存(GB)** | 98.5 | 98.4 | 73.2 | 73.8 |

---

## 3. 关键发现

### 3.1 step_advantage_w 的影响

```
                    验证成功率对比

    70% ┤                              ████ 70.3%
    65% ┤              ████ 62.5%
    60% ┤
    55% ┤
    50% ┤
    45% ┤ ████ 43.8%              ████ 43.8%
    40% ┤
        └──────────────────────────────────────
          baseline    step_adv_0.5  step_adv_2.0  val128
          (w=1.0)      (w=0.5)       (w=2.0)     (w=0.5)
```

**观察结论**:
1. **step_adv_w=0.5 显著优于 baseline(w=1.0) 和 w=2.0**
   - 62.5% vs 43.8%，相对提升 **42.9%**

2. **step_adv_w=2.0 与 baseline 持平**
   - 过高的 step-level advantage 权重可能导致训练不稳定

3. **扩大验证集 + 更多训练轮数进一步提升性能**
   - 70.3% (128任务/79轮) vs 62.5% (16任务/30轮)

### 3.2 训练 vs 验证的差异

| 实验 | 训练成功率 | 验证成功率 | 差异 |
|------|-----------|-----------|------|
| baseline | 58.2% | 43.8% | -14.4% |
| step_adv_0.5 | 52.7% | 62.5% | **+9.8%** |
| step_adv_2.0 | 61.3% | 43.8% | -17.5% |
| step_adv_0.5_val128 | 93.4% | 70.3% | -23.1% |

**分析**:
- step_adv_w=0.5 是唯一**验证成功率高于训练成功率**的配置
- 说明 w=0.5 有更好的**泛化能力**
- step_adv_0.5_val128 虽然训练-验证差距大，但绝对验证成功率最高

### 3.3 各任务类型分析

**强势任务** (成功率 > 70%):
- Pick and Place: 在 step_adv_0.5_val128 达到 **92.1%**
- Pick Two Objects: 在 step_adv_0.5_val128 达到 **75.0%**

**弱势任务** (成功率 < 50%):
- Pick Cool then Place: 在 step_adv_0.5_val128 仅 **36.4%**
- Look at Object in Light: 波动较大，0%-100%

---

## 4. 结论与建议

### 4.1 最优超参数配置

```yaml
algorithm.rebel:
  enable: True
  step_advantage_w: 0.5        # 最优值
  belief_granularity: subgoal
  mode: mean_norm

actor_rollout_ref.actor.optim:
  lr: 1e-6
```

### 4.2 主要结论

1. **step_advantage_w=0.5 是最优配置**
   - 相比 baseline 提升 42.9%
   - 具有更好的泛化能力（验证 > 训练）

2. **扩大验证集规模有助于更准确评估**
   - 16 任务可能存在较大方差
   - 建议使用 64-128 任务进行验证

3. **更长的训练轮数继续提升性能**
   - 79 epochs 时仍在上升趋势
   - 建议完整实验使用 100 epochs

### 4.3 下一步建议

1. **补充运行第四个实验至100 epochs**
   - 当前因 API 额度用尽在 79 epochs 停止
   - 需要重新运行完成

2. **正式论文实验配置**
   ```bash
   TRAIN_SIZE=64
   VAL_SIZE=128
   EPOCHS=100
   step_advantage_w=0.5
   ```

3. **消融实验**
   - belief_granularity: subgoal vs task
   - mode: mean_norm vs mean_std_norm

---

## 5. 实验存储位置

```
/fs-computility-new/UPDZ03_chengjun/huangsijie.p/rebel_results/hyperparam_search/
├── rebel_search_20251227_114948/baseline/
├── rebel_search_20251228_013943/step_adv_0.5/
├── rebel_search_20251229_084222/step_adv_2.0/
└── rebel_search_20251230_041227/step_adv_0.5_val128_ep100/
```

---

## 6. SwanLab 监控链接

- 项目主页: https://swanlab.cn/@yetian/ReBel_HyperSearch
- baseline: rebel_search_20251227_114948_baseline
- step_adv_0.5: rebel_search_20251228_013943_step_adv_0.5
- step_adv_2.0: rebel_search_20251229_084222_step_adv_2.0
- step_adv_0.5_val128: rebel_search_20251230_041227_step_adv_0.5_val128_ep100

---

*报告生成于 2025-12-31*
