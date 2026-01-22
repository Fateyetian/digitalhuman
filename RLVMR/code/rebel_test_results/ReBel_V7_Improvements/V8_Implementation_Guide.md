# ReBel V8 改进实施方案

> **基于V7实验分析的具体改进措施**
> **目标**: 成功率从82.4%提升至90%+，look_at稳定性显著改善

---

## 1. 改进方案总览

| 优先级 | 改进 | 修改文件 | 预期提升 | 实现难度 |
|--------|------|----------|---------|----------|
| **P0** | 任务自适应采样权重 | `ray_trainer.py` | +5-8% | 低 |
| **P1** | 任务级优势缩放 | `core_rebel.py` | +3-5% | 低 |
| **P2** | 分阶段训练策略 | `run_v8_experiments.sh` | +5-10% | 中 |
| P3 | 验证集扩充 | 数据配置 | 稳定性↑ | 低 |

---

## 2. P0: 任务自适应采样权重

### 2.1 原理

当前问题: look_at任务样本仅占13%，梯度信号弱
解决方案: 根据任务成功率动态调整采样权重，低成功率任务获得更多样本

```
采样权重公式:
w(task) = (1 - success_rate(task))^alpha / Σ(1 - success_rate(task_i))^alpha

示例 (alpha=2.0):
  pick_and_place: sr=97.8% → w=(1-0.978)^2 = 0.0005 → 极低权重
  look_at:        sr=67.6% → w=(1-0.676)^2 = 0.105 → 高权重
```

### 2.2 代码修改

**文件**: `verl/trainer/ppo/ray_trainer.py`

**位置**: 在 `_record_task_distribution` 方法附近添加

```python
# ============================================================================ #
# V8新增: 任务自适应采样权重
# ============================================================================ #

def compute_task_sampling_weights(
    self,
    task_success_rates: Dict[str, float],
    alpha: float = 2.0,
    min_weight: float = 0.05,
    max_weight: float = 0.4
) -> Dict[str, float]:
    """
    V8: 根据任务成功率计算采样权重

    低成功率任务获得更高权重，增加其训练样本

    Args:
        task_success_rates: 各任务当前成功率
        alpha: 权重调节指数，越大差异越明显
        min_weight: 最小权重（防止完全忽略高性能任务）
        max_weight: 最大权重（防止过度偏向单一任务）

    Returns:
        task_weights: 各任务的采样权重
    """
    if not task_success_rates:
        return {}

    raw_weights = {}
    for task, sr in task_success_rates.items():
        # 成功率越低，权重越高
        raw_weights[task] = (1.0 - sr) ** alpha

    # 归一化
    total = sum(raw_weights.values())
    if total == 0:
        return {t: 1.0/len(raw_weights) for t in raw_weights}

    weights = {}
    for task, w in raw_weights.items():
        normalized_w = w / total
        # 应用最小/最大限制
        weights[task] = max(min_weight, min(max_weight, normalized_w))

    # 重新归一化
    total = sum(weights.values())
    weights = {t: w/total for t, w in weights.items()}

    return weights


def apply_task_weights_to_batch(
    self,
    batch,
    task_weights: Dict[str, float],
    oversample_factor: float = 2.0
) -> Any:
    """
    V8: 对batch应用任务权重（通过重采样实现）

    Args:
        batch: 原始batch
        task_weights: 任务采样权重
        oversample_factor: 过采样因子

    Returns:
        weighted_batch: 加权后的batch
    """
    if not task_weights or 'task_type' not in batch.non_tensor_batch:
        return batch

    task_types = batch.non_tensor_batch['task_type']
    n_samples = len(task_types)

    # 计算每个样本的采样概率
    sample_weights = np.array([
        task_weights.get(t, 1.0/len(task_weights))
        for t in task_types
    ])
    sample_weights = sample_weights / sample_weights.sum()

    # 重采样（带放回）
    target_size = int(n_samples * oversample_factor)
    indices = np.random.choice(
        n_samples,
        size=target_size,
        replace=True,
        p=sample_weights
    )

    # 应用重采样到batch
    # ... 具体实现根据batch结构

    return weighted_batch
```

### 2.3 配置参数

```yaml
# 在config中添加
algorithm.rebel.task_adaptive_sampling:
  enable: true
  alpha: 2.0                    # 权重指数
  min_weight: 0.05              # 最小任务权重
  max_weight: 0.4               # 最大任务权重
  update_freq: 10               # 每10个epoch更新权重
  warmup_epochs: 20             # 前20个epoch不启用（让模型先学习基础）
```

---

## 3. P1: 任务级优势缩放

### 3.1 原理

当前问题: 高性能任务(pick_and_place)梯度接近0，但仍占用计算资源
解决方案: 根据任务成功率缩放优势值，低成功率任务获得更大梯度信号

```
优势缩放公式:
A'(task) = A(task) * scale(task)
scale(task) = min(max_scale, (1 + (1-sr)/sr_threshold))

示例 (sr_threshold=0.3, max_scale=3.0):
  pick_and_place: sr=97.8% → scale=1.0 (超过阈值，不缩放)
  look_at:        sr=67.6% → scale=1+(1-0.676)/0.3=2.08
```

### 3.2 代码修改

**文件**: `rebel/core_rebel.py`

**位置**: 在 `normalize_advantages_per_task` 函数中添加

```python
def normalize_advantages_per_task(
    advantages: torch.Tensor,
    task_types: np.ndarray,
    eos_mask: torch.Tensor,
    epsilon: float = 1e-8,
    min_samples_for_norm: int = 10,
    min_std_for_norm: float = 0.1,
    use_conditional_norm: bool = True,
    min_samples_ratio: float = 0.0,
    # V8新增参数
    task_success_rates: Optional[Dict[str, float]] = None,
    use_advantage_scaling: bool = False,
    scale_threshold: float = 0.85,
    max_scale: float = 3.0
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    V8改进: 在归一化基础上增加任务级优势缩放

    新增参数:
        task_success_rates: 各任务当前成功率（来自上一次验证）
        use_advantage_scaling: 是否启用优势缩放
        scale_threshold: 成功率低于此阈值的任务将被缩放
        max_scale: 最大缩放倍数

    Returns:
        normalized: 归一化+缩放后的优势
        task_scales: 每个任务的缩放系数（用于logging）
    """
    result = advantages.clone()
    unique_tasks = np.unique(task_types)
    total_samples = len(task_types)
    task_scales = {}

    # ... 原有归一化逻辑 ...

    # V8新增: 任务级优势缩放
    if use_advantage_scaling and task_success_rates:
        for task in unique_tasks:
            sr = task_success_rates.get(task, 0.5)

            if sr < scale_threshold:
                # 计算缩放系数: 成功率越低，缩放越大
                scale = 1.0 + (scale_threshold - sr) / scale_threshold
                scale = min(scale, max_scale)
            else:
                # 高成功率任务不缩放（或轻微降低）
                scale = max(0.5, 1.0 - (sr - scale_threshold) * 0.5)

            task_scales[task] = scale

            # 应用缩放
            mask = np.array([t == task for t in task_types])
            result[mask] = result[mask] * scale

    return result, task_scales
```

### 3.3 调用位置修改

**文件**: `verl/trainer/ppo/ray_trainer.py`

在 `compute_rebel_advantage` 调用处传入 `task_success_rates`:

```python
# 在训练循环中维护任务成功率
self.task_success_rates = {}  # epoch级别更新

# 调用时传入
advantages, returns, adv_details = compute_rebel_advantage(
    token_level_rewards=token_level_rewards,
    rebel_intrinsic_rewards=rebel_intrinsic_rewards,
    # ... 原有参数 ...
    # V8新增
    task_success_rates=self.task_success_rates,
    use_advantage_scaling=self.config.algorithm.rebel.get('use_advantage_scaling', False),
    scale_threshold=self.config.algorithm.rebel.get('scale_threshold', 0.85),
    max_scale=self.config.algorithm.rebel.get('max_scale', 3.0),
)
```

---

## 4. P2: 分阶段训练策略

### 4.1 三阶段训练设计

```
Phase 1: 基础训练 (0-30 epoch)
├─ 目标: 所有任务达到基础水平 (>50%)
├─ 配置: 均匀采样，标准学习率
└─ 重点: 快速学习基本策略

Phase 2: 难任务强化 (30-60 epoch)
├─ 目标: 提升look_at、pick_cool瓶颈任务
├─ 配置:
│   ├─ task_adaptive_sampling.enable: true
│   ├─ task_adaptive_sampling.alpha: 2.0
│   ├─ use_advantage_scaling: true
│   └─ entropy_coeff: 0.005 (提高探索)
└─ 重点: 针对性强化少数任务

Phase 3: 稳定微调 (60-100 epoch)
├─ 目标: 稳定所有任务性能
├─ 配置:
│   ├─ 降低学习率至 5e-7
│   ├─ task_adaptive_sampling.alpha: 1.0 (减少差异)
│   └─ clip_ratio: [0.1, 0.15] (减小更新步长)
└─ 重点: 防止过拟合，稳定收敛
```

### 4.2 实验脚本

**文件**: `run_v8_phased_training.sh`

```bash
#!/bin/bash
# =============================================================================
# V8 分阶段训练实验
# =============================================================================

set -e

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_ATTENTION_BACKEND=XFORMERS

NUM_GPUS=${NUM_GPUS:-4}

# 实验配置
EXPERIMENT_NAME="rebel_v8_phased_$(date +%Y%m%d_%H%M%S)"
RESULTS_DIR="/fs-computility-new/UPDZ03_chengjun/huangsijie.p/rebel_results/v8_experiments/${EXPERIMENT_NAME}"

mkdir -p "${RESULTS_DIR}/checkpoints"

# 共同配置
COMMON_CONFIG="
    algorithm.adv_estimator=rebel
    algorithm.rebel.enable=True
    algorithm.rebel.belief_granularity=adaptive
    algorithm.rebel.step_advantage_w=0.5
    algorithm.rebel.mode=mean_norm
    algorithm.rebel.task_aware_grouping=true
    algorithm.rebel.per_task_normalization=true
    algorithm.rebel.conditional_norm=true
    algorithm.rebel.min_samples_for_norm=10
    algorithm.rebel.min_std_for_norm=0.2
    algorithm.rebel.min_samples_ratio=0.15
    algorithm.rebel.entropy_protection.enable=True
    algorithm.rebel.entropy_protection.method=clip_cov
    algorithm.rebel.entropy_protection.clip_cov_lb=0.0
    algorithm.rebel.entropy_protection.clip_cov_ub=0.3
    algorithm.use_kl_in_reward=False
    algorithm.kl_penalty=kl
    algorithm.kl_ctrl.type=fixed
    algorithm.kl_ctrl.kl_coef=0.001
"

echo "═══════════════════════════════════════════════════════════════════"
echo "  V8分阶段训练: ${EXPERIMENT_NAME}"
echo "═══════════════════════════════════════════════════════════════════"

# ============================================================================
# Phase 1: 基础训练 (0-30 epoch)
# ============================================================================
echo ""
echo "Phase 1: 基础训练 (0-30 epoch)"
echo "目标: 所有任务达到基础水平 (>50%)"

python3 -m verl.trainer.main_ppo \
    ${COMMON_CONFIG} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    algorithm.rebel.task_adaptive_sampling.enable=False \
    algorithm.rebel.use_advantage_scaling=False \
    trainer.total_epochs=30 \
    trainer.save_freq=30 \
    trainer.test_freq=5 \
    trainer.default_local_dir="${RESULTS_DIR}/checkpoints" \
    trainer.experiment_name="${EXPERIMENT_NAME}_phase1" \
    2>&1 | tee "${RESULTS_DIR}/phase1_training.log"

# ============================================================================
# Phase 2: 难任务强化 (30-60 epoch)
# ============================================================================
echo ""
echo "Phase 2: 难任务强化 (30-60 epoch)"
echo "目标: 提升look_at、pick_cool瓶颈任务"

python3 -m verl.trainer.main_ppo \
    ${COMMON_CONFIG} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.entropy_coeff=0.005 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    algorithm.rebel.task_adaptive_sampling.enable=True \
    algorithm.rebel.task_adaptive_sampling.alpha=2.0 \
    algorithm.rebel.task_adaptive_sampling.min_weight=0.05 \
    algorithm.rebel.task_adaptive_sampling.max_weight=0.4 \
    algorithm.rebel.use_advantage_scaling=True \
    algorithm.rebel.scale_threshold=0.85 \
    algorithm.rebel.max_scale=3.0 \
    trainer.resume_mode=auto \
    trainer.total_epochs=60 \
    trainer.save_freq=30 \
    trainer.test_freq=5 \
    trainer.default_local_dir="${RESULTS_DIR}/checkpoints" \
    trainer.experiment_name="${EXPERIMENT_NAME}_phase2" \
    2>&1 | tee "${RESULTS_DIR}/phase2_training.log"

# ============================================================================
# Phase 3: 稳定微调 (60-100 epoch)
# ============================================================================
echo ""
echo "Phase 3: 稳定微调 (60-100 epoch)"
echo "目标: 稳定所有任务性能"

python3 -m verl.trainer.main_ppo \
    ${COMMON_CONFIG} \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.actor.clip_ratio_low=0.1 \
    actor_rollout_ref.actor.clip_ratio_high=0.15 \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.02 \
    algorithm.rebel.task_adaptive_sampling.enable=True \
    algorithm.rebel.task_adaptive_sampling.alpha=1.0 \
    algorithm.rebel.use_advantage_scaling=True \
    algorithm.rebel.scale_threshold=0.90 \
    algorithm.rebel.max_scale=2.0 \
    trainer.resume_mode=auto \
    trainer.total_epochs=100 \
    trainer.save_freq=100 \
    trainer.test_freq=5 \
    trainer.default_local_dir="${RESULTS_DIR}/checkpoints" \
    trainer.experiment_name="${EXPERIMENT_NAME}_phase3" \
    2>&1 | tee "${RESULTS_DIR}/phase3_training.log"

echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "  V8分阶段训练完成"
echo "═══════════════════════════════════════════════════════════════════"
echo ""
echo "结果目录: ${RESULTS_DIR}"
echo ""
echo "检查各阶段结果:"
echo "  grep 'val/success_rate' ${RESULTS_DIR}/phase1_training.log | tail -5"
echo "  grep 'val/success_rate' ${RESULTS_DIR}/phase2_training.log | tail -5"
echo "  grep 'val/success_rate' ${RESULTS_DIR}/phase3_training.log | tail -5"
```

---

## 5. P3: 验证集扩充（稳定性改善）

### 5.1 问题

当前验证集128个样本，look_at约12-18个
单样本变化导致±5-10%成功率波动

### 5.2 建议

```yaml
# 扩充验证集
data.val_batch_size: 256  # 从128增加到256

# 或按任务平衡
data.val_stratified_sampling: true
data.val_min_samples_per_task: 30
```

---

## 6. 快速验证实验

### 6.1 最小改动实验 (P0 only)

先单独验证任务自适应采样的效果:

```bash
# 只启用任务自适应采样
python3 -m verl.trainer.main_ppo \
    # ... 基础配置 ...
    algorithm.rebel.task_adaptive_sampling.enable=True \
    algorithm.rebel.task_adaptive_sampling.alpha=2.0 \
    # 其他保持与V7相同
```

**预期结果**:
- look_at成功率从67.6%提升至75-80%
- 整体成功率从82.4%提升至85-88%
- look_at波动从±12.5%降至±8%

### 6.2 成功指标

| 指标 | V7基线 | V8目标 | 达成标准 |
|------|--------|--------|----------|
| 整体成功率 | 82.4% | 90%+ | ≥88% |
| look_at成功率 | 67.6% | 80%+ | ≥75% |
| look_at波动 | ±12.5% | ±6% | ≤±8% |
| pick_cool成功率 | 72.3% | 82%+ | ≥78% |

---

## 7. 实施优先级建议

### 7.1 第一步: 快速验证 P0

```bash
# 修改 run_v7_experiments.sh 添加以下配置
algorithm.rebel.task_adaptive_sampling.enable=True
algorithm.rebel.task_adaptive_sampling.alpha=2.0
```

工作量: 配置修改 + 1次实验运行 (~8小时)

### 7.2 第二步: P0 + P1 组合

如果P0效果明显，再添加P1:
- 修改 `core_rebel.py` 添加优势缩放逻辑
- 工作量: ~2小时代码修改 + 1次实验运行

### 7.3 第三步: 完整分阶段训练

如果P0+P1达到85%+，进行完整的分阶段训练:
- 使用 `run_v8_phased_training.sh`
- 工作量: 3次阶段训练 (~24小时)

---

## 8. 监控指标

训练过程中重点关注:

```bash
# 关键指标
grep 'val/success_rate' training.log | tail -10
grep 'val/look_at_obj_in_light' training.log | tail -10
grep 'rebel/task_sample_weights' training.log | tail -5  # V8新增
grep 'rebel/task_advantage_scales' training.log | tail -5  # V8新增

# 稳定性监控 (计算连续epoch的标准差)
# 如果look_at标准差 > 10%，考虑调低alpha
```

---

*文档生成时间: 2026-01-14*
*基于V7实验分析报告*
