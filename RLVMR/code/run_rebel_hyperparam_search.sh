#!/bin/bash
# =============================================================================
# ReBel Hyperparameter Search - 小规模 RL 实验 + 自动报告生成
# =============================================================================
#
# 第四个子实验 (重新运行): step_adv_w=0.5 + 验证集128 + 100 epochs
# 注意: 每次运行会自动生成新目录，不会覆盖之前的结果
#
# Usage:
#   bash run_rebel_hyperparam_search.sh [OPTIONS]
#
# =============================================================================

set -e

# =============================================================================
# Configuration
# =============================================================================

SEARCH_NAME="rebel_search_$(date +%Y%m%d_%H%M%S)"
NUM_GPUS=${NUM_GPUS:-4}

# Offline mode
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_ATTENTION_BACKEND=XFORMERS

# 清理 GPU 缓存（解决内存碎片问题）
python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

# 实验设置
EPOCHS_PER_EXP=${EPOCHS_PER_EXP:-100}
TRAIN_SIZE=${TRAIN_SIZE:-16}
VAL_SIZE=${VAL_SIZE:-128}
GROUP_SIZE=${GROUP_SIZE:-16}
MAX_STEPS=${MAX_STEPS:-30}
EVAL_TASKS=${EVAL_TASKS:-100}

# SFT 模型路径
SFT_MODEL_PATH=${SFT_MODEL_PATH:-""}

# 结果目录 - 使用大容量存储
RESULTS_BASE="/fs-computility-new/UPDZ03_chengjun/huangsijie.p/rebel_results"
RESULTS_DIR="${RESULTS_BASE}/hyperparam_search/${SEARCH_NAME}"

# =============================================================================
# ReBel 核心参数实验配置
# 格式: "name|step_advantage_w|belief_granularity|lr|mode"
# =============================================================================

# 完整实验列表（备份）:
# EXPERIMENTS=(
#     "baseline|1.0|subgoal|1e-6|mean_norm"
#     "step_adv_0.5|0.5|subgoal|1e-6|mean_norm"
#     "step_adv_2.0|2.0|subgoal|1e-6|mean_norm"
#     "task_level|1.0|task|1e-6|mean_norm"
# )

# 第四个子实验: 使用最优配置 step_adv_w=0.5，扩大验证集到128，训练100 epochs
# 实验名加上 _rerun 标识，便于区分
EXPERIMENTS=(
    "step_adv_0.5_val128_ep100_rerun|0.5|subgoal|1e-6|mean_norm"
)

# =============================================================================
# Parse arguments
# =============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --sft-model) SFT_MODEL_PATH="$2"; shift 2 ;;
        --epochs) EPOCHS_PER_EXP="$2"; shift 2 ;;
        --experiments)
            case $2 in
                minimal) EXPERIMENTS=("baseline|1.0|subgoal|1e-6|mean_norm") ;;
                *) ;;
            esac
            shift 2 ;;
        --help|-h)
            echo "Usage: bash run_rebel_hyperparam_search.sh [OPTIONS]"
            echo "  --sft-model PATH    SFT checkpoint path"
            echo "  --epochs N          Epochs per experiment (default: 50)"
            exit 0 ;;
        *) shift ;;
    esac
done

# =============================================================================
# Utility
# =============================================================================

print_header() {
    echo ""
    echo "═══════════════════════════════════════════════════════════════════"
    echo "  $1"
    echo "═══════════════════════════════════════════════════════════════════"
}

print_info() { echo "[INFO] $(date '+%H:%M:%S') $1"; }
print_success() { echo "[SUCCESS] $(date '+%H:%M:%S') $1"; }
print_error() { echo "[ERROR] $(date '+%H:%M:%S') $1"; }

# =============================================================================
# Find SFT Model
# =============================================================================

find_sft_model() {
    if [ -n "${SFT_MODEL_PATH}" ] && [ -d "${SFT_MODEL_PATH}" ]; then
        echo "${SFT_MODEL_PATH}"; return
    fi

    local path=$(ls -d ./checkpoints/cold_start/alfworld/rebel_full_*/global_step_* 2>/dev/null | sort -V | tail -1)
    [ -n "$path" ] && { echo "$path"; return; }

    path=$(ls -d ./checkpoints/cold_start/alfworld/*/global_step_* 2>/dev/null | sort -V | tail -1)
    [ -n "$path" ] && { echo "$path"; return; }

    echo "./checkpoints/cold_start/alfworld/bdrs_qwen1.5b_2gpu_20251110/global_step_75"
}

# =============================================================================
# Run Single Experiment
# =============================================================================

run_experiment() {
    local config="$1"
    local index="$2"
    local total="$3"

    IFS='|' read -r exp_name step_adv_w granularity lr mode <<< "${config}"

    local exp_dir="${RESULTS_DIR}/${exp_name}"
    local checkpoint_dir="${exp_dir}/checkpoints"
    mkdir -p "${exp_dir}" "${checkpoint_dir}"

    print_header "Experiment ${index}/${total}: ${exp_name}"
    echo "  step_advantage_w:    ${step_adv_w}"
    echo "  belief_granularity:  ${granularity}"
    echo "  learning_rate:       ${lr}"
    echo "  mode:                ${mode}"
    echo ""

    local start_time=$(date +%s)

    # Prepare data
    python3 -m examples.data_preprocess.prepare \
        --mode 'text' \
        --train_data_size ${TRAIN_SIZE} \
        --val_data_size ${VAL_SIZE} 2>/dev/null || true

    # RL Training
    python3 -m verl.trainer.main_ppo \
        algorithm.adv_estimator=rebel \
        algorithm.rebel.enable=True \
        algorithm.rebel.belief_granularity="${granularity}" \
        algorithm.rebel.step_advantage_w=${step_adv_w} \
        algorithm.rebel.mode="${mode}" \
        data.train_files=$HOME/data/verl-agent/text/train.parquet \
        data.val_files=$HOME/data/verl-agent/text/test.parquet \
        data.train_batch_size=${TRAIN_SIZE} \
        data.val_batch_size=${VAL_SIZE} \
        data.max_prompt_length=6000 \
        data.max_response_length=1024 \
        data.filter_overlong_prompts=True \
        data.truncation='error' \
        data.return_raw_chat=True \
        actor_rollout_ref.model.path="${SFT_MODEL_PATH}" \
        actor_rollout_ref.actor.optim.lr=${lr} \
        actor_rollout_ref.actor.clip_ratio=0.2 \
        actor_rollout_ref.actor.entropy_coeff=0.001 \
        actor_rollout_ref.actor.ppo_epochs=1 \
        actor_rollout_ref.actor.use_kl_loss=True \
        actor_rollout_ref.actor.kl_loss_coef=0.01 \
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
        actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
        actor_rollout_ref.rollout.enable_chunked_prefill=False \
        actor_rollout_ref.rollout.enforce_eager=False \
        actor_rollout_ref.rollout.free_cache_engine=False \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        actor_rollout_ref.actor.use_invalid_action_penalty=True \
        actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
        algorithm.use_kl_in_reward=False \
        env.env_name=alfworld/AlfredTWEnv \
        env.seed=0 \
        env.max_steps=${MAX_STEPS} \
        env.rollout.n=${GROUP_SIZE} \
        env.alfworld.generalization_level=0 \
        env.alfworld.meta_think=True \
        env.alfworld.use_rebel=True \
        env.use_teacher_planner=True \
        trainer.critic_warmup=0 \
        trainer.logger="['console','swanlab']" \
        trainer.project_name='ReBel_HyperSearch' \
        trainer.experiment_name="${SEARCH_NAME}_${exp_name}" \
        trainer.n_gpus_per_node=${NUM_GPUS} \
        trainer.nnodes=1 \
        trainer.save_freq=10 \
        trainer.test_freq=5 \
        trainer.total_epochs=${EPOCHS_PER_EXP} \
        trainer.default_local_dir="${checkpoint_dir}" \
        trainer.val_before_train=True \
        2>&1 | tee "${exp_dir}/training.log"

    local end_time=$(date +%s)
    local duration=$((end_time - start_time))

    # Find best checkpoint
    local model_path=$(ls -d ${checkpoint_dir}/global_step_* 2>/dev/null | sort -V | tail -1)

    # Evaluation
    if [ -n "${model_path}" ]; then
        print_info "Evaluating ${exp_name}..."

        python3 scripts/simple_eval.py \
            --model_path "${model_path}" \
            --num_tasks ${EVAL_TASKS} \
            --max_steps ${MAX_STEPS} \
            --temperature 0.0 \
            --project_name "ReBel_HyperSearch" \
            --experiment_name "${SEARCH_NAME}_${exp_name}_eval" \
            --gpu_memory_utilization 0.7 \
            --tensor_parallel_size 1 \
            --generalization_level 0 \
            --output_dir "${exp_dir}/eval" \
            2>&1 | tee "${exp_dir}/eval.log"

        local success_rate=$(grep -oP 'Success Rate:\s+\K[\d.]+' "${exp_dir}/eval.log" | tail -1 || echo "N/A")
        local avg_steps=$(grep -oP 'Average Steps:\s+\K[\d.]+' "${exp_dir}/eval.log" | tail -1 || echo "N/A")
    else
        local success_rate="FAILED"
        local avg_steps="N/A"
    fi

    # Save result
    echo "${exp_name}|${step_adv_w}|${granularity}|${lr}|${mode}|${duration}|${success_rate}|${avg_steps}" >> "${RESULTS_DIR}/results.csv"

    # Generate individual report
    generate_experiment_report "${exp_name}" "${step_adv_w}" "${granularity}" "${lr}" "${mode}" "${duration}" "${success_rate}" "${avg_steps}" "${exp_dir}"

    print_success "${exp_name} completed: Success Rate = ${success_rate}%, Duration = ${duration}s"
}

# =============================================================================
# Generate Experiment Report (with charts)
# =============================================================================

generate_experiment_report() {
    local exp_name="$1"
    local step_adv_w="$2"
    local granularity="$3"
    local lr="$4"
    local mode="$5"
    local duration="$6"
    local success_rate="$7"
    local avg_steps="$8"
    local exp_dir="$9"

    print_info "Generating report for ${exp_name}..."

    # Generate report using Python script
    python3 << EOF
import os
import json
import re
from datetime import datetime

exp_name = "${exp_name}"
exp_dir = "${exp_dir}"
results_dir = "${RESULTS_DIR}"

# Config
config = {
    "experiment_name": exp_name,
    "step_advantage_w": ${step_adv_w},
    "belief_granularity": "${granularity}",
    "learning_rate": "${lr}",
    "mode": "${mode}",
    "epochs": ${EPOCHS_PER_EXP},
    "duration_seconds": ${duration},
    "success_rate": "${success_rate}",
    "avg_steps": "${avg_steps}",
}

# Parse training log for metrics
metrics = {
    "epochs": [],
    "success_rates": [],
    "rewards": [],
    "losses": [],
}

log_file = os.path.join(exp_dir, "training.log")
if os.path.exists(log_file):
    with open(log_file, 'r') as f:
        content = f.read()

    # Extract success rates from log
    for match in re.finditer(r'epoch[:\s]+(\d+).*?success[_\s]?rate[:\s]+([\d.]+)', content, re.IGNORECASE):
        try:
            metrics["epochs"].append(int(match.group(1)))
            metrics["success_rates"].append(float(match.group(2)))
        except:
            pass

# Generate Markdown report
report = f"""# Experiment Report: {exp_name}

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Configuration

| Parameter | Value |
|-----------|-------|
| step_advantage_w | {config['step_advantage_w']} |
| belief_granularity | {config['belief_granularity']} |
| learning_rate | {config['learning_rate']} |
| mode | {config['mode']} |
| epochs | {config['epochs']} |

## Results

| Metric | Value |
|--------|-------|
| Success Rate | {config['success_rate']}% |
| Average Steps | {config['avg_steps']} |
| Training Duration | {int(config['duration_seconds'])}s ({config['duration_seconds']/3600:.2f}h) |

## Training Curve

"""

# Add ASCII chart if we have data
if len(metrics["success_rates"]) > 0:
    max_rate = max(metrics["success_rates"]) if metrics["success_rates"] else 100
    min_rate = min(metrics["success_rates"]) if metrics["success_rates"] else 0
    chart_height = 10
    chart_width = min(50, len(metrics["success_rates"]))

    report += "```\n"
    report += f"Success Rate over Epochs (max: {max_rate:.1f}%)\n"
    report += "|\n"

    # Sample data points if too many
    step = max(1, len(metrics["success_rates"]) // chart_width)
    sampled = metrics["success_rates"][::step][:chart_width]

    for row in range(chart_height, 0, -1):
        threshold = min_rate + (max_rate - min_rate) * row / chart_height
        line = "|"
        for val in sampled:
            if val >= threshold:
                line += "█"
            else:
                line += " "
        report += line + "\n"

    report += "+" + "-" * len(sampled) + "> epochs\n"
    report += "```\n"

report += f"""
## SwanLab Dashboard

View detailed metrics and charts at: [SwanLab Project](https://swanlab.cn/@your-username/ReBel_HyperSearch)

## Files

- Training Log: `{exp_dir}/training.log`
- Evaluation Log: `{exp_dir}/eval.log`
- Checkpoints: `{exp_dir}/checkpoints/`

---
*Report generated by run_rebel_hyperparam_search.sh*
"""

# Save report
report_path = os.path.join(exp_dir, "report.md")
with open(report_path, 'w') as f:
    f.write(report)

# Save config as JSON
config_path = os.path.join(exp_dir, "config.json")
with open(config_path, 'w') as f:
    json.dump(config, f, indent=2)

print(f"Report saved to: {report_path}")
EOF
}

# =============================================================================
# Generate Final Comparison Report
# =============================================================================

generate_final_report() {
    print_header "Generating Final Comparison Report"

    python3 << 'PYTHON_SCRIPT'
import os
import csv
from datetime import datetime

results_dir = os.environ.get('RESULTS_DIR', './results')
search_name = os.environ.get('SEARCH_NAME', 'unknown')

csv_file = os.path.join(results_dir, "results.csv")
report_file = os.path.join(results_dir, "comparison_report.md")

# Read results
results = []
if os.path.exists(csv_file):
    with open(csv_file, 'r') as f:
        for line in f:
            parts = line.strip().split('|')
            if len(parts) >= 8:
                results.append({
                    'name': parts[0],
                    'step_adv_w': parts[1],
                    'granularity': parts[2],
                    'lr': parts[3],
                    'mode': parts[4],
                    'duration': parts[5],
                    'success_rate': parts[6],
                    'avg_steps': parts[7],
                })

# Sort by success rate
def get_rate(r):
    try:
        return float(r['success_rate'])
    except:
        return -1

results.sort(key=get_rate, reverse=True)

# Generate report
report = f"""# ReBel Hyperparameter Search Report

**Search Name:** {search_name}
**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Total Experiments:** {len(results)}

## Results Summary

| Rank | Experiment | step_adv_w | granularity | Success Rate | Avg Steps | Duration |
|------|------------|------------|-------------|--------------|-----------|----------|
"""

for i, r in enumerate(results):
    duration_h = float(r['duration']) / 3600 if r['duration'].replace('.','').isdigit() else 0
    report += f"| {i+1} | {r['name']} | {r['step_adv_w']} | {r['granularity']} | {r['success_rate']}% | {r['avg_steps']} | {duration_h:.1f}h |\n"

# Best config
if results and results[0]['success_rate'] not in ['N/A', 'FAILED']:
    best = results[0]
    report += f"""
## Best Configuration

**Winner: {best['name']}** with Success Rate: {best['success_rate']}%

```yaml
algorithm.rebel:
  enable: True
  step_advantage_w: {best['step_adv_w']}
  belief_granularity: {best['granularity']}
  mode: {best['mode']}

actor_rollout_ref.actor.optim:
  lr: {best['lr']}
```

## Recommended Full-Scale Training

```bash
bash run_rebel_full_pipeline.sh --skip-sft \\
    STEP_ADV_W={best['step_adv_w']} \\
    BELIEF_GRANULARITY={best['granularity']} \\
    ACTOR_LR={best['lr']}
```
"""

report += """
## Analysis

### Key Findings

1. **step_advantage_w impact**: Compare baseline vs step_adv_0.5/2.0
2. **Granularity impact**: Compare subgoal vs task level belief grouping
3. **Stability**: Check training curves in individual reports

### SwanLab Dashboard

View all experiments: [SwanLab Project](https://swanlab.cn/@your-username/ReBel_HyperSearch)

## Individual Reports

"""

for r in results:
    report += f"- [{r['name']}](./{r['name']}/report.md)\n"

report += """
---
*Generated by run_rebel_hyperparam_search.sh*
"""

with open(report_file, 'w') as f:
    f.write(report)

print(f"Final report saved to: {report_file}")

# Also print summary to console
print("\n" + "="*60)
print("EXPERIMENT RANKING")
print("="*60)
for i, r in enumerate(results[:5]):
    print(f"{i+1}. {r['name']}: {r['success_rate']}%")
print("="*60)
PYTHON_SCRIPT
}

# =============================================================================
# Main
# =============================================================================

print_header "ReBel Hyperparameter Search"

# Find SFT model
SFT_MODEL_PATH=$(find_sft_model)
if [ ! -d "${SFT_MODEL_PATH}" ]; then
    print_error "SFT model not found: ${SFT_MODEL_PATH}"
    echo "Please specify --sft-model PATH"
    exit 1
fi

# Setup
mkdir -p "${RESULTS_DIR}"
export RESULTS_DIR SEARCH_NAME

echo "Search Name:     ${SEARCH_NAME}"
echo "SFT Model:       ${SFT_MODEL_PATH}"
echo "Epochs/Exp:      ${EPOCHS_PER_EXP}"
echo "Results Dir:     ${RESULTS_DIR}"
echo "Experiments:     ${#EXPERIMENTS[@]}"
echo ""

# Initialize CSV
echo "name|step_adv_w|granularity|lr|mode|duration|success_rate|avg_steps" > "${RESULTS_DIR}/results.csv"

# List experiments
echo "Experiments to run:"
for i in "${!EXPERIMENTS[@]}"; do
    IFS='|' read -r name _ <<< "${EXPERIMENTS[$i]}"
    echo "  $((i+1)). ${name}"
done
echo ""

# Confirm
read -p "Start search? [Y/n] " -n 1 -r
echo
[[ $REPLY =~ ^[Nn]$ ]] && exit 0

# Run experiments
SEARCH_START=$(date +%s)

for i in "${!EXPERIMENTS[@]}"; do
    run_experiment "${EXPERIMENTS[$i]}" "$((i+1))" "${#EXPERIMENTS[@]}"
done

SEARCH_END=$(date +%s)
TOTAL_DURATION=$((SEARCH_END - SEARCH_START))

# Generate final report
generate_final_report

# Summary
print_header "Search Complete!"
echo "Total Duration:  $(echo "scale=2; ${TOTAL_DURATION}/3600" | bc) hours"
echo "Results:         ${RESULTS_DIR}"
echo "Report:          ${RESULTS_DIR}/comparison_report.md"
echo ""

cat "${RESULTS_DIR}/comparison_report.md"
