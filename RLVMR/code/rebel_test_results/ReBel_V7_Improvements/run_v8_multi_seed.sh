#!/bin/bash
# =============================================================================
# ReBel V8 多随机种子实验
# =============================================================================
# 用于论文: Results are averaged over 3 random seeds
# =============================================================================

set -e

EXPERIMENT=${EXPERIMENT:-1}
NUM_GPUS=${NUM_GPUS:-8}
EPOCHS=${EPOCHS:-100}
SEEDS=${SEEDS:-"123 456"}  # 剩余2个随机种子（已完成seed 0）
EXISTING_EXP="/fs-computility-new/UPDZ03_chengjun/huangsijie.p/rebel_results/v8_experiments/rebel_v8_exp1_task_weighting_20260115_011942"

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

# 检查是否存在未完成的实验
find_incomplete_experiment() {
    local seed=$1
    local exp_pattern="${EXP_BASE_NAME}_seed${seed}_*"
    local exp_dir=$(ls -dt ${RESULTS_BASE}/${exp_pattern} 2>/dev/null | head -1)

    if [ -z "$exp_dir" ]; then
        echo ""
        return
    fi

    # 检查是否有checkpoint
    local checkpoint_dir="${exp_dir}/checkpoints"
    if [ ! -d "$checkpoint_dir" ]; then
        echo ""
        return
    fi

    # 检查是否有latest_checkpointed_iteration.txt
    local latest_file="${checkpoint_dir}/latest_checkpointed_iteration.txt"
    if [ ! -f "$latest_file" ]; then
        echo ""
        return
    fi

    # 读取最新的checkpoint epoch
    local latest_epoch=$(cat "$latest_file")

    # 检查训练是否已完成（epoch >= EPOCHS）
    if [ "$latest_epoch" -ge "$EPOCHS" ]; then
        echo ""
        return
    fi

    # 返回未完成的实验目录
    echo "$exp_dir"
}

# 查找最新的checkpoint路径
find_latest_checkpoint() {
    local checkpoint_dir=$1
    local latest_file="${checkpoint_dir}/latest_checkpointed_iteration.txt"

    if [ ! -f "$latest_file" ]; then
        echo ""
        return
    fi

    local latest_epoch=$(cat "$latest_file")
    local checkpoint_path="${checkpoint_dir}/global_step_${latest_epoch}"

    if [ -d "$checkpoint_path" ]; then
        echo "$checkpoint_path"
    else
        echo ""
    fi
}

# 运行多个种子
for SEED in $SEEDS; do
    EXP_NAME="${EXP_BASE_NAME}_seed${SEED}"

    # 检查是否存在未完成的实验
    INCOMPLETE_EXP=$(find_incomplete_experiment $SEED)

    if [ -n "$INCOMPLETE_EXP" ]; then
        # 从未完成的实验恢复
        RESULTS_DIR="$INCOMPLETE_EXP"
        CHECKPOINT_DIR="${RESULTS_DIR}/checkpoints"
        RESUME_CHECKPOINT=$(find_latest_checkpoint "$CHECKPOINT_DIR")
        RESUME_EPOCH=$(cat "${CHECKPOINT_DIR}/latest_checkpointed_iteration.txt")

        echo ""
        echo "───────────────────────────────────────────────────────────────────"
        echo "  恢复训练: Seed=${SEED} (从 Epoch ${RESUME_EPOCH} 继续)"
        echo "───────────────────────────────────────────────────────────────────"
        echo "  实验目录: ${RESULTS_DIR}"
        echo "  Checkpoint: ${RESUME_CHECKPOINT}"
        echo "───────────────────────────────────────────────────────────────────"
    else
        # 从头开始新实验
        RESULTS_DIR="${RESULTS_BASE}/${EXP_NAME}_${TIMESTAMP}"
        mkdir -p "${RESULTS_DIR}/checkpoints"
        RESUME_CHECKPOINT=""
        RESUME_EPOCH=0

        echo ""
        echo "───────────────────────────────────────────────────────────────────"
        echo "  开始训练: Seed=${SEED} (从头开始)"
        echo "───────────────────────────────────────────────────────────────────"
    fi

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
        $([ -n "$RESUME_CHECKPOINT" ] && echo "trainer.load_checkpoint=${RESUME_CHECKPOINT}") \
        2>&1 | tee -a "${RESULTS_DIR}/training.log"

    echo ""
    echo "✓ Seed ${SEED} 完成"
    echo "  结果: ${RESULTS_DIR}"
done

echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "  所有种子训练完成"
echo "═══════════════════════════════════════════════════════════════════"
echo ""
echo "开始汇总三个实验结果（包括已完成的seed 0实验）..."
echo ""

# 创建汇总脚本
cat > "${RESULTS_BASE}/aggregate_results_${TIMESTAMP}.py" << 'EOF'
#!/usr/bin/env python3
"""汇总多随机种子实验结果（包括已完成的seed 0实验）"""
import re
import glob
import numpy as np
from pathlib import Path

def extract_final_metrics(log_file):
    """从训练日志提取最终指标"""
    metrics = {}
    with open(log_file) as f:
        lines = f.readlines()

    # 提取所有验证指标
    success_rates = []
    look_at_rates = []
    pick_two_rates = []
    pick_clean_rates = []
    pick_and_place_rates = []
    pick_cool_rates = []
    pick_heat_rates = []

    for line in lines:
        if 'val/success_rate:' in line:
            match = re.search(r'val/success_rate:(\d+\.\d+)', line)
            if match:
                success_rates.append(float(match.group(1)))
        if 'val/look_at_obj_in_light_success_rate:' in line:
            match = re.search(r'val/look_at_obj_in_light_success_rate:(\d+\.\d+)', line)
            if match:
                look_at_rates.append(float(match.group(1)))
        if 'val/pick_two_obj_and_place_success_rate:' in line:
            match = re.search(r'val/pick_two_obj_and_place_success_rate:(\d+\.\d+)', line)
            if match:
                pick_two_rates.append(float(match.group(1)))
        if 'val/pick_clean_then_place_in_recep_success_rate:' in line:
            match = re.search(r'val/pick_clean_then_place_in_recep_success_rate:(\d+\.\d+)', line)
            if match:
                pick_clean_rates.append(float(match.group(1)))
        if 'val/pick_and_place_success_rate:' in line:
            match = re.search(r'val/pick_and_place_success_rate:(\d+\.\d+)', line)
            if match:
                pick_and_place_rates.append(float(match.group(1)))
        if 'val/pick_cool_then_place_in_recep_success_rate:' in line:
            match = re.search(r'val/pick_cool_then_place_in_recep_success_rate:(\d+\.\d+)', line)
            if match:
                pick_cool_rates.append(float(match.group(1)))
        if 'val/pick_heat_then_place_in_recep_success_rate:' in line:
            match = re.search(r'val/pick_heat_then_place_in_recep_success_rate:(\d+\.\d+)', line)
            if match:
                pick_heat_rates.append(float(match.group(1)))

    # 取最后5个epoch的平均
    if success_rates:
        metrics['success_rate'] = np.mean(success_rates[-5:])
        metrics['success_rate_std'] = np.std(success_rates[-5:])
        metrics['success_rate_final'] = success_rates[-1]

    if look_at_rates:
        metrics['look_at_rate'] = np.mean(look_at_rates[-5:])
        metrics['look_at_rate_std'] = np.std(look_at_rates[-5:])
        metrics['look_at_rate_final'] = look_at_rates[-1]

    if pick_two_rates:
        metrics['pick_two_rate'] = np.mean(pick_two_rates[-5:])
        metrics['pick_two_rate_final'] = pick_two_rates[-1]

    if pick_clean_rates:
        metrics['pick_clean_rate'] = np.mean(pick_clean_rates[-5:])
        metrics['pick_clean_rate_final'] = pick_clean_rates[-1]

    if pick_and_place_rates:
        metrics['pick_and_place_rate'] = np.mean(pick_and_place_rates[-5:])
        metrics['pick_and_place_rate_final'] = pick_and_place_rates[-1]

    if pick_cool_rates:
        metrics['pick_cool_rate'] = np.mean(pick_cool_rates[-5:])
        metrics['pick_cool_rate_final'] = pick_cool_rates[-1]

    if pick_heat_rates:
        metrics['pick_heat_rate'] = np.mean(pick_heat_rates[-5:])
        metrics['pick_heat_rate_final'] = pick_heat_rates[-1]

    return metrics

# 查找所有实验目录（包括已完成的实验）
base_dir = Path("/fs-computility-new/UPDZ03_chengjun/huangsijie.p/rebel_results/v8_experiments")
exp_dirs = sorted(glob.glob(str(base_dir / "rebel_v8_exp1_task_weighting_*")))

print("=" * 80)
print("ReBel V8 多随机种子实验结果汇总")
print("=" * 80)
print()
print(f"找到 {len(exp_dirs)} 个实验目录")
print()

all_results = []
for exp_dir in exp_dirs:
    exp_dir = Path(exp_dir)
    log_file = exp_dir / "training.log"

    if not log_file.exists():
        print(f"⚠️  跳过（无日志）: {exp_dir.name}")
        continue

    # 提取种子信息
    if "seed" in exp_dir.name:
        seed_match = re.search(r'seed(\d+)', exp_dir.name)
        seed = seed_match.group(1) if seed_match else "unknown"
    else:
        seed = "0"  # 已完成的实验使用seed 0

    metrics = extract_final_metrics(log_file)
    if metrics:
        metrics['seed'] = seed
        metrics['exp_dir'] = exp_dir.name
        all_results.append(metrics)
        print(f"✓ 已处理: {exp_dir.name} (seed={seed})")

print()
print("=" * 80)
print("实验结果汇总")
print("=" * 80)
print()

if len(all_results) >= 3:
    print(f"✓ 成功汇总 {len(all_results)} 个实验")
    print()

    # 整体成功率
    success_rates = [r['success_rate'] for r in all_results if 'success_rate' in r]
    if success_rates:
        mean_sr = np.mean(success_rates)
        std_sr = np.std(success_rates)
        print(f"【整体成功率】")
        print(f"  平均: {mean_sr:.1%} ± {std_sr:.1%}")
        print(f"  各种子: {[f'{sr:.1%}' for sr in success_rates]}")
        print(f"  论文格式: {mean_sr*100:.1f}% ± {std_sr*100:.1f}%")
        print()

    # look_at任务
    look_at_rates = [r['look_at_rate'] for r in all_results if 'look_at_rate' in r]
    if look_at_rates:
        mean_la = np.mean(look_at_rates)
        std_la = np.std(look_at_rates)
        print(f"【look_at_obj_in_light任务】")
        print(f"  平均: {mean_la:.1%} ± {std_la:.1%}")
        print(f"  各种子: {[f'{la:.1%}' for la in look_at_rates]}")
        print(f"  论文格式: {mean_la*100:.1f}% ± {std_la*100:.1f}%")
        print()

    # 其他任务
    print(f"【各任务成功率（最后5轮平均）】")

    task_names = [
        ('pick_clean_rate', 'pick_clean_then_place_in_recep'),
        ('pick_and_place_rate', 'pick_and_place'),
        ('pick_heat_rate', 'pick_heat_then_place_in_recep'),
        ('pick_two_rate', 'pick_two_obj_and_place'),
        ('pick_cool_rate', 'pick_cool_then_place_in_recep'),
    ]

    for key, name in task_names:
        rates = [r[key] for r in all_results if key in r]
        if rates:
            mean_rate = np.mean(rates)
            std_rate = np.std(rates)
            print(f"  {name:40s}: {mean_rate*100:.1f}% ± {std_rate*100:.1f}%")

    print()
    print("=" * 80)
    print("论文表格格式")
    print("=" * 80)
    print()
    print("| Task | Success Rate |")
    print("|------|--------------|")

    if success_rates:
        print(f"| Overall | {mean_sr*100:.1f} ± {std_sr*100:.1f} |")

    for key, name in task_names:
        rates = [r[key] for r in all_results if key in r]
        if rates:
            mean_rate = np.mean(rates)
            std_rate = np.std(rates)
            print(f"| {name} | {mean_rate*100:.1f} ± {std_rate*100:.1f} |")

    if look_at_rates:
        print(f"| look_at_obj_in_light | {mean_la*100:.1f} ± {std_la*100:.1f} |")

    print()

else:
    print(f"⚠️  警告: 只找到 {len(all_results)} 个有效实验，需要至少3个")
    print()
    for r in all_results:
        print(f"  - Seed {r['seed']}: 整体成功率 {r.get('success_rate', 0):.1%}")

print()
print("=" * 80)
EOF

chmod +x "${RESULTS_BASE}/aggregate_results_${TIMESTAMP}.py"

echo ""
echo "自动运行汇总脚本..."
echo ""

# 自动运行汇总脚本
python3 "${RESULTS_BASE}/aggregate_results_${TIMESTAMP}.py"

echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "  完成！"
echo "═══════════════════════════════════════════════════════════════════"
echo ""
echo "汇总脚本已保存: ${RESULTS_BASE}/aggregate_results_${TIMESTAMP}.py"
echo "可以随时重新运行: python3 ${RESULTS_BASE}/aggregate_results_${TIMESTAMP}.py"
