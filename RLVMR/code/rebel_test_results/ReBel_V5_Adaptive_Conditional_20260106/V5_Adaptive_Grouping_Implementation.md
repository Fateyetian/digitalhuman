# ReBel V5 实验：Adaptive分组 + 条件归一化

**创建日期**: 2026-01-06
**目标**: 解决V4实验中look_at_obj_in_light任务成功率异常低的问题，同时保持其他任务性能

---

## 1. V4问题总结

| 问题 | V4 Exp1 (task_status) | V4 Exp2 (state_aware) |
|------|----------------------|----------------------|
| 分组粒度 | 过粗 (44组, 81样本/组) | 过细 (1800组, 72.7%单样本) |
| look_at成功率 | **8.3%** | **16.7%** |
| 根因 | per_task_norm强制std=1，放大噪声 | 同左 |

---

## 2. V5核心改进

### 2.1 Adaptive分组策略 (belief_granularity="adaptive")

结合task_status的稳定性与subgoal的区分度：

```
分组维度:
├── 结构化状态 (来自task_status, 2³=8种组合)
│   ├── is_complete: 子目标是否完成
│   ├── has_state_change: 是否有物体状态变化
│   └── has_inventory: 是否持有物体
│
└── 阶段类型 (9种，避免subgoal字符串的过度碎片化)
    ├── find: 查找物体
    ├── navigate: 导航移动
    ├── pickup: 拾取物体
    ├── place: 放置物体
    ├── heat: 加热
    ├── cool: 冷却
    ├── clean: 清洁
    ├── use: 使用设备
    └── interact: 交互(开/关)

总计: 9 × 8 = 72种基础分组
```

**预期效果**:
- 分组数: 100-300 (vs V4 Exp1的44 或 Exp2的1800)
- 样本/组: 5-20 (vs V4 Exp1的81 或 Exp2的2.1)
- 单样本组: <50% (vs V4 Exp2的72.7%)

### 2.2 条件归一化 (conditional_norm=true)

保护小样本任务免受噪声放大：

```python
if sample_count < min_samples_for_norm (10):
    # 使用全局统计量归一化，避免噪声放大
    normalized = (adv - global_mean) / global_std

elif std < min_std_for_norm (0.1):
    # 保守归一化，仅去均值，保留自然方差
    normalized = adv - task_mean

else:
    # 标准归一化
    normalized = (adv - task_mean) / task_std
```

---

## 3. 代码修改

修改文件: `/root/testttt/RLVMR/code/rebel/core_rebel.py`

1. **新增 `adaptive` 粒度** (line 129-177)
   - 9种阶段类型映射
   - 结合结构化状态

2. **改进 `normalize_advantages_per_task`** (line 577-669)
   - 新增 `min_samples_for_norm` 参数
   - 新增 `min_std_for_norm` 参数
   - 新增 `use_conditional_norm` 开关

3. **更新 `compute_rebel_advantage`** (line 458-565)
   - 新增 `conditional_norm` 参数
   - 透传条件归一化参数

---

## 4. 实验配置

```yaml
algorithm:
  rebel:
    enable: true
    belief_granularity: "adaptive"      # V5新增
    task_aware_grouping: true
    per_task_normalization: true
    conditional_norm: true              # V5新增
    min_samples_for_norm: 10            # V5新增
    min_std_for_norm: 0.1               # V5新增
```

---

## 5. 目标指标

| 指标 | V4最佳 | V5目标 |
|------|-------|-------|
| 整体成功率 | 74.2% | **≥80%** |
| look_at_obj_in_light | 16.7% | **≥50%** |
| pick_and_place | 93.5% | ≥90% |
| 单样本组比例 | 27.3% | **<50%** |

---

## 6. 运行命令

```bash
cd /root/testttt/RLVMR/code/rebel_test_results/ReBel_V5_Adaptive_Conditional_20260106
bash run_v5_exp1_adaptive_conditional.sh
```

---

## 7. 文件结构

```
ReBel_V5_Adaptive_Conditional_20260106/
├── V5_Adaptive_Grouping_Implementation.md  # 本文件
├── run_v5_exp1_adaptive_conditional.sh     # 实验运行脚本
└── (实验完成后)
    ├── training.log
    └── analysis/
```

---

*Created by Claude Code on 2026-01-06*
