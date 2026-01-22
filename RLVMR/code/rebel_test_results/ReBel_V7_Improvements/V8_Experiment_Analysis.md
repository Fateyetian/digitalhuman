# ReBel V8 实验分析报告

> **实验目标**: 通过任务自适应权重机制解决样本不平衡问题
> **基线**: V7 (82.4% 整体成功率, 67.6% look_at)
> **目标**: 90%+ 整体成功率, 80%+ look_at

---

## 1. 实验设计评估

### 1.1 核心改进点

**V8 提出的任务自适应权重机制**:
```python
# 配置参数 (run_v8_experiments.sh)
+algorithm.rebel.use_task_weighting=true
+algorithm.rebel.weight_alpha=2.0
+algorithm.rebel.weight_min=0.3
+algorithm.rebel.weight_max=3.0
+algorithm.rebel.weight_baseline_sr=0.85
+algorithm.rebel.task_weighting_warmup_epochs=20
```

**问题分析**:
1. ❌ **代码未实现**: 脚本中使用了 `+algorithm.rebel.use_task_weighting` 等参数，但 `ray_trainer.py` 和 `core_rebel.py` 中**没有实现对应逻辑**
2. ❌ **参数无效**: 这些参数会被 Hydra 配置系统接受，但不会产生任何效果
3. ⚠️ **实验无意义**: 当前 V8 实验实际上等同于 V7 基线

### 1.2 实验配置对比

| 配置项 | V7 | V8-Exp1 | V8-Exp3 (对照) |
|--------|----|---------| --------------|
| use_task_weighting | N/A | true | false |
| weight_alpha | N/A | 2.0 | 2.0 |
| 其他参数 | 相同 | 相同 | 相同 |
| **实际效果** | 基线 | **等同V7** | **等同V7** |

**结论**: V8-Exp1 和 V8-Exp3 实际上都是 V7 的重复实验，因为任务权重代码未实现。

---

## 2. 是否开展正式实验？

### 2.1 当前状态

**❌ 不建议开展正式实验**

**原因**:
1. **核心功能缺失**: V8 的任务自适应权重机制完全未实现
2. **无效对照**: Exp1 vs Exp3 对比无意义（两者实际相同）
3. **资源浪费**: 每个实验需要 8 GPU × 100 epochs，但不会产生新结果

### 2.2 需要完成的工作

**必须先实现以下代码** (参考 V8_Implementation_Guide.md):

#### Step 1: 实现任务采样权重 (ray_trainer.py)
```python
def compute_task_sampling_weights(self, task_success_rates, alpha=2.0):
    """根据任务成功率计算采样权重"""
    raw_weights = {task: (1.0 - sr) ** alpha
                   for task, sr in task_success_rates.items()}
    # 归一化...
    return weights

def apply_task_weights_to_batch(self, batch, task_weights):
    """对batch应用任务权重（重采样）"""
    # 实现重采样逻辑...
    return weighted_batch
```

#### Step 2: 实现任务级优势缩放 (core_rebel.py)
```python
def normalize_advantages_per_task(
    advantages, task_types,
    task_success_rates=None,  # V8新增
    use_advantage_scaling=False,  # V8新增
    scale_threshold=0.85,
    max_scale=3.0
):
    # 原有归一化逻辑...

    # V8新增: 任务级优势缩放
    if use_advantage_scaling and task_success_rates:
        for task in unique_tasks:
            sr = task_success_rates.get(task, 0.5)
            if sr < scale_threshold:
                scale = 1.0 + (scale_threshold - sr) / scale_threshold
                scale = min(scale, max_scale)
                result[task_mask] *= scale

    return result
```

#### Step 3: 集成到训练循环
```python
# 在 ray_trainer.py 的训练循环中
self.task_success_rates = {}  # 维护任务成功率

# 每个 epoch 后更新
if self.config.algorithm.rebel.get('use_task_weighting', False):
    task_weights = self.compute_task_sampling_weights(
        self.task_success_rates,
        alpha=self.config.algorithm.rebel.weight_alpha
    )
    # 应用到下一个 epoch 的采样...
```

### 2.3 实施建议

**推荐路径**:

1. **先实现 P0 (任务自适应采样)**
   - 工作量: ~4小时代码 + 1次实验验证
   - 预期提升: +5-8% 整体成功率

2. **快速验证实验** (30 epochs, 单随机种子)
   ```bash
   EXPERIMENT=1 NUM_GPUS=8 EPOCHS=30 bash run_v8_experiments.sh
   ```
   - 如果 30 epoch 时 look_at > 75%，继续完整实验
   - 如果无明显改善，分析原因后调整

3. **完整实验** (100 epochs, 3 随机种子)
   - 仅在快速验证成功后进行
   - 使用下面的多种子脚本

---

## 3. 多随机种子实验脚本

### 3.1 设计原则

**论文要求**: "Results are averaged over 3 random seeds"

**实现方式**:
- 环境种子: `env.seed` (控制 AlfWorld 任务采样)
- 数据种子: `data.seed` (控制 dataloader 采样顺序)
- 每个种子独立运行完整训练

### 3.2 多种子脚本

创建 `run_v8_multi_seed.sh`:

```bash
#!/bin/bash
# =============================================================================
# ReBel V8 多随机种子实验
# =============================================================================

set -e

EXPERIMENT=${EXPERIMENT:-1}
NUM_GPUS=${NUM_GPUS:-8}
EPOCHS=${EPOCHS:-100}
SEEDS=${SEEDS:-"42 123 456"}  # 3个随机种子

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_ATTENTION_BACKEND=XFORMERS

# 动态查找SFT模型路径
find_sft_model() {
    local path=$(ls -d /root/testttt/RLVMR/code/checkpoints/cold_start/alfworld/rebel_full_*/global_step_* 2>/dev/null | sort -V | tail -1)
    [ -n "$path" ] && { echo "$path"; return; }
    path=$(ls -d /root/testttt/RLVMR/code/checkpoints/cold_start/alfworld/*/global_step_* 2>/dev/null | sort -V | tail -1)
    [ -n "$path" ] && { echo "$path"; return; }
    echo "/root/testttt/RLVMR/code/checkpoints/cold_start/alfworld/bdrs_qwen1.5b_2gpu_20251110/global_step_75"
}

SFT_MODEL_PATH=$(find_sft_model)
RESULTS_BASE="/fs-computility-new/UPDZ03_chengjun/huangsijie.p/rebel_results/v8_experiments"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# V8配置
CLIP_RATIO_LOW=0.2
CLIP_RATIO_HIGH=0.28
ENTROPY_COEFF=0.001
USE_KL_LOSS=True
KL_LOSS_COEF=0.01
USE_KL_IN_REWARD=False
MIN_SAMPLES_RATIO=0.15
ENTROPY_PROTECTION_ENABLE=True
ENTROPY_PROTECTION_METHOD=clip_cov
CLIP_COV_LB=0.0
CLIP_COV_UB=0.3

# 实验配置
case $EXPERIMENT in
    1)
        EXP_BASE_NAME="rebel_v8_exp1_task_weighting"
        USE_TASK_WEIGHTING="true"
        WEIGHT_ALPHA=2.0
        WEIGHT_MIN=0.3
        WEIGHT_MAX=3.0
        WARMUP_EPOCHS=20
        EXP_DESC="V8核心实验: 任务自适应权重"
        ;;
    3)
        EXP_BASE_NAME="rebel_v8_exp3_baseline"
        USE_TASK_WEIGHTING="false"
        WEIGHT_ALPHA=2.0
        WEIGHT_MIN=0.3
        WEIGHT_MAX=3.0
        WARMUP_EPOCHS=20
        EXP_DESC="V8对照组: V7基线"
        ;;
    *)
        echo "无效实验编号: $EXPERIMENT (可选: 1, 3)"
        exit 1
        ;;
esac

echo "═══════════════════════════════════════════════════════════════════"
echo "  V8多随机种子实验: ${EXP_DESC}"
echo "═══════════════════════════════════════════════════════════════════"
echo ""
echo "配置:"
echo "  - 实验: ${EXP_BASE_NAME}"
echo "  - 随机种子: ${SEEDS}"
echo "  - GPU数量: ${NUM_GPUS}"
echo "  - Epochs: ${EPOCHS}"
echo ""

# 运行多个种子
for SEED in $SEEDS; do
    EXP_NAME="${EXP_BASE_NAME}_seed${SEED}"
    RESULTS_DIR="${RESULTS_BASE}/${EXP_NAME}_${TIMESTAMP}"
    mkdir -p "${RESULTS_DIR}/checkpoints"

    echo ""
    echo "───────────────────────────────────────────────────────────────────"
    echo "  开始训练: Seed=${SEED}"
    echo "───────────────────────────────────────────────────────────────────"

    cd /root/testttt/RLVMR/code

    # 准备数据
    if [ ! -f "$HOME/data/verl-agent/text/train.parquet" ]; then
        python3 -m examples.data_preprocess.prepare \
            --mode 'text' \
            --train_data_size 16 \
            --val_data_size 128 2>/dev/null || true
    fi

    # 运行训练
    python3 -m verl.trainer.main_ppo \
        algorithm.adv_estimator=rebel \
        algorithm.rebel.enable=True \
        algorithm.rebel.belief_granularity="adaptive" \
        algorithm.rebel.step_advantage_w=0.5 \
        algorithm.rebel.mode="mean_norm" \
        algorithm.rebel.task_aware_grouping=true \
        algorithm.rebel.per_task_normalization=true \
        algorithm.rebel.conditional_norm=true \
        algorithm.rebel.min_samples_for_norm=10 \
        algorithm.rebel.min_std_for_norm=0.2 \
        algorithm.rebel.min_samples_ratio=${MIN_SAMPLES_RATIO} \
        algorithm.rebel.entropy_protection.enable=${ENTROPY_PROTECTION_ENABLE} \
        algorithm.rebel.entropy_protection.method=${ENTROPY_PROTECTION_METHOD} \
        algorithm.rebel.entropy_protection.clip_cov_lb=${CLIP_COV_LB} \
        algorithm.rebel.entropy_protection.clip_cov_ub=${CLIP_COV_UB} \
        +algorithm.rebel.use_task_weighting=${USE_TASK_WEIGHTING} \
        +algorithm.rebel.weight_alpha=${WEIGHT_ALPHA} \
        +algorithm.rebel.weight_min=${WEIGHT_MIN} \
        +algorithm.rebel.weight_max=${WEIGHT_MAX} \
        +algorithm.rebel.weight_baseline_sr=0.85 \
        +algorithm.rebel.task_weighting_warmup_epochs=${WARMUP_EPOCHS} \
        data.train_files=$HOME/data/verl-agent/text/train.parquet \
        data.val_files=$HOME/data/verl-agent/text/test.parquet \
        data.train_batch_size=16 \
        data.val_batch_size=128 \
        data.max_prompt_length=6000 \
        data.max_response_length=1024 \
        data.filter_overlong_prompts=True \
        data.truncation='error' \
        data.return_raw_chat=True \
        data.seed=${SEED} \
        actor_rollout_ref.model.path="${SFT_MODEL_PATH}" \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.actor.clip_ratio=0.2 \
        actor_rollout_ref.actor.clip_ratio_low=${CLIP_RATIO_LOW} \
        actor_rollout_ref.actor.clip_ratio_high=${CLIP_RATIO_HIGH} \
        actor_rollout_ref.actor.entropy_coeff=${ENTROPY_COEFF} \
        actor_rollout_ref.actor.ppo_epochs=1 \
        actor_rollout_ref.actor.use_kl_loss=${USE_KL_LOSS} \
        actor_rollout_ref.actor.kl_loss_coef=${KL_LOSS_COEF} \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.ppo_mini_batch_size=256 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.fsdp_config.param_offload=False \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
        actor_rollout_ref.rollout.enable_chunked_prefill=False \
        actor_rollout_ref.rollout.enforce_eager=False \
        actor_rollout_ref.rollout.free_cache_engine=False \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        actor_rollout_ref.actor.use_invalid_action_penalty=True \
        actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
        algorithm.use_kl_in_reward=${USE_KL_IN_REWARD} \
        algorithm.kl_penalty=kl \
        algorithm.kl_ctrl.type=fixed \
        algorithm.kl_ctrl.kl_coef=0.001 \
        env.env_name=alfworld/AlfredTWEnv \
        env.seed=${SEED} \
        env.max_steps=30 \
        env.rollout.n=16 \
        env.alfworld.generalization_level=0 \
        env.alfworld.meta_think=True \
        env.alfworld.use_rebel=True \
        env.alfworld.prompt_template_type="explicit_task_type" \
        env.use_teacher_planner=True \
        trainer.critic_warmup=0 \
        trainer.logger="['console','swanlab']" \
        trainer.project_name='ReBel_V8_MultiSeed' \
        trainer.experiment_name="${EXP_NAME}" \
        trainer.n_gpus_per_node=${NUM_GPUS} \
        trainer.nnodes=1 \
        trainer.save_freq=50 \
        trainer.test_freq=5 \
        trainer.total_epochs=${EPOCHS} \
        trainer.default_local_dir="${RESULTS_DIR}/checkpoints" \
        trainer.val_before_train=True \
        2>&1 | tee "${RESULTS_DIR}/training.log"

    echo ""
    echo "✓ Seed ${SEED} 完成"
    echo "  结果: ${RESULTS_DIR}"
done

echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "  所有种子训练完成"
echo "═══════════════════════════════════════════════════════════════════"
echo ""
echo "结果汇总脚本:"
cat > "${RESULTS_BASE}/aggregate_results_${TIMESTAMP}.py" << 'EOF'
#!/usr/bin/env python3
"""汇总多随机种子实验结果"""
import re
import glob
import numpy as np
from pathlib import Path

def extract_final_metrics(log_file):
    """从训练日志提取最终指标"""
    metrics = {}
    with open(log_file) as f:
        lines = f.readlines()

    # 提取最后10个epoch的验证指标
    success_rates = []
    look_at_rates = []

    for line in lines:
        if 'val/success_rate' in line:
            match = re.search(r'val/success_rate[\'"]:\s*([\d.]+)', line)
            if match:
                success_rates.append(float(match.group(1)))
        if 'val/look_at_obj_in_light' in line:
            match = re.search(r'val/look_at_obj_in_light[\'"]:\s*([\d.]+)', line)
            if match:
                look_at_rates.append(float(match.group(1)))

    if success_rates:
        # 取最后5个epoch的平均
        metrics['success_rate'] = np.mean(success_rates[-5:])
        metrics['success_rate_std'] = np.std(success_rates[-5:])

    if look_at_rates:
        metrics['look_at_rate'] = np.mean(look_at_rates[-5:])
        metrics['look_at_rate_std'] = np.std(look_at_rates[-5:])

    return metrics

# 查找所有实验目录
base_dir = Path(__file__).parent
exp_dirs = sorted(glob.glob(str(base_dir / "rebel_v8_*_seed*")))

results_by_exp = {}
for exp_dir in exp_dirs:
    exp_dir = Path(exp_dir)
    log_file = exp_dir / "training.log"

    if not log_file.exists():
        continue

    # 提取实验名称（去掉seed后缀）
    exp_name = exp_dir.name
    base_name = re.sub(r'_seed\d+_\d+', '', exp_name)

    if base_name not in results_by_exp:
        results_by_exp[base_name] = []

    metrics = extract_final_metrics(log_file)
    if metrics:
        results_by_exp[base_name].append(metrics)

# 打印汇总结果
print("=" * 80)
print("ReBel V8 多随机种子实验结果汇总")
print("=" * 80)
print()

for exp_name, results in sorted(results_by_exp.items()):
    print(f"实验: {exp_name}")
    print(f"  种子数量: {len(results)}")

    if results:
        success_rates = [r['success_rate'] for r in results if 'success_rate' in r]
        look_at_rates = [r['look_at_rate'] for r in results if 'look_at_rate' in r]

        if success_rates:
            mean_sr = np.mean(success_rates)
            std_sr = np.std(success_rates)
            print(f"  整体成功率: {mean_sr:.1%} ± {std_sr:.1%}")
            print(f"    各种子: {[f'{sr:.1%}' for sr in success_rates]}")

        if look_at_rates:
            mean_la = np.mean(look_at_rates)
            std_la = np.std(look_at_rates)
            print(f"  look_at成功率: {mean_la:.1%} ± {std_la:.1%}")
            print(f"    各种子: {[f'{la:.1%}' for la in look_at_rates]}")

    print()

print("=" * 80)
EOF

chmod +x "${RESULTS_BASE}/aggregate_results_${TIMESTAMP}.py"

echo ""
echo "运行汇总脚本:"
echo "  python3 ${RESULTS_BASE}/aggregate_results_${TIMESTAMP}.py"
```

### 3.3 使用方法

```bash
# 快速验证 (30 epochs, 3 seeds)
EXPERIMENT=1 NUM_GPUS=8 EPOCHS=30 bash run_v8_multi_seed.sh

# 完整实验 (100 epochs, 3 seeds)
EXPERIMENT=1 NUM_GPUS=8 EPOCHS=100 bash run_v8_multi_seed.sh

# 对照组
EXPERIMENT=3 NUM_GPUS=8 EPOCHS=100 bash run_v8_multi_seed.sh

# 自定义种子
EXPERIMENT=1 SEEDS="42 123 456 789 999" bash run_v8_multi_seed.sh
```

### 3.4 结果汇总

训练完成后运行自动生成的汇总脚本:
```bash
python3 /fs-computility-new/.../aggregate_results_*.py
```

输出示例:
```
实验: rebel_v8_exp1_task_weighting
  种子数量: 3
  整体成功率: 88.5% ± 1.2%
    各种子: ['87.3%', '89.1%', '89.1%']
  look_at成功率: 78.2% ± 3.5%
    各种子: ['75.0%', '79.2%', '80.4%']
```

---

## 4. 最终建议

### 4.1 立即行动

**❌ 不要运行当前 V8 脚本** - 代码未实现，浪费资源

**✅ 推荐路径**:

1. **实现 P0 代码** (任务自适应采样权重)
   - 参考 V8_Implementation_Guide.md 第2节
   - 工作量: ~4小时

2. **快速验证** (30 epochs, 单种子)
   ```bash
   EXPERIMENT=1 NUM_GPUS=8 EPOCHS=30 bash run_v8_experiments.sh
   ```

3. **如果验证成功** (look_at > 75% @ 30 epochs)
   - 运行完整多种子实验
   ```bash
   EXPERIMENT=1 NUM_GPUS=8 EPOCHS=100 bash run_v8_multi_seed.sh
   ```

4. **如果验证失败**
   - 分析原因，调整超参数
   - 考虑实现 P1 (任务级优势缩放)

### 4.2 论文写作准备

**当前可用数据**:
- ✅ V7 实验结果 (82.4% 整体, 67.6% look_at)
- ✅ V7 分析报告 (问题诊断)

**需要补充**:
- ❌ V8 实验结果 (代码未实现)
- ❌ 消融实验 (V8 vs V7 对比)
- ❌ 多随机种子统计

**时间估算**:
- 代码实现: 4-8 小时
- 快速验证: 8-12 小时 (30 epochs)
- 完整实验: 24-36 小时 (100 epochs × 3 seeds)
- **总计**: 2-3 天

---

*报告生成时间: 2026-01-16*
*基于 V8_Implementation_Guide.md 和 run_v8_experiments.sh 分析*
