#!/bin/bash
# ReBel Small-Scale Test Script
# Purpose: Quick validation of ReBel implementation with structured output
# Usage: bash test_rebel_small.sh [model_path]

set -x
set -e  # Exit on error

# Configuration
ENGINE=${1:-vllm}
export VLLM_ATTENTION_BACKEND=XFORMERS

# Small-scale test parameters
train_data_size=2      # Very small for quick testing
val_data_size=4        # Small validation set
group_size=8           # Small group size for quick rollout
max_steps=15           # Fewer steps per episode

# Output directory with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="rebel_test_results/${TIMESTAMP}"
mkdir -p ${OUTPUT_DIR}

# Save test configuration
cat > ${OUTPUT_DIR}/test_config.yaml <<EOF
test_name: "ReBel Small-Scale Validation"
timestamp: "${TIMESTAMP}"
purpose: "Validate ReBel implementation and belief-based grouping"
configuration:
  train_data_size: ${train_data_size}
  val_data_size: ${val_data_size}
  group_size: ${group_size}
  max_steps: ${max_steps}
  engine: ${ENGINE}
  belief_granularity: "subgoal"
  step_advantage_w: 1.0
  mode: "mean_norm"
EOF

echo "=========================================="
echo "ReBel Small-Scale Test"
echo "Output Directory: ${OUTPUT_DIR}"
echo "=========================================="

# Data preparation
echo "[1/3] Preparing test data..."
python3 -m examples.data_preprocess.prepare \
    --mode 'text' \
    --train_data_size $train_data_size \
    --val_data_size $val_data_size \
    2>&1 | tee ${OUTPUT_DIR}/data_prep.log

# Check if model path is provided
MODEL_PATH=${2:-"./checkpoints/cold_start/alfworld/sft_qwen2.5-1.5b"}
if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: Model path not found: ${MODEL_PATH}"
    echo "Please provide a valid model path as the second argument"
    echo "Usage: bash test_rebel_small.sh [engine] [model_path]"
    exit 1
fi

echo "[2/3] Running ReBel training (1 epoch for testing)..."

# Run ReBel with test parameters
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=rebel \
    algorithm.rebel.enable=True \
    algorithm.rebel.belief_granularity='subgoal' \
    algorithm.rebel.step_advantage_w=1.0 \
    algorithm.rebel.mode='mean_norm' \
    data.train_files=$HOME/data/verl-agent/text/train.parquet \
    data.val_files=$HOME/data/verl-agent/text/test.parquet \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=6000 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=False \
    env.env_name=alfworld/AlfredTWEnv \
    env.seed=0 \
    env.max_steps=$max_steps \
    env.rollout.n=$group_size \
    env.alfworld.generalization_level=0 \
    env.alfworld.meta_think=True \
    env.alfworld.use_rebel=True \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name='ReBel_Test' \
    trainer.experiment_name="rebel_test_${TIMESTAMP}" \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_epochs=1 \
    trainer.val_before_train=True \
    trainer.default_local_dir=${OUTPUT_DIR}/checkpoints \
    2>&1 | tee ${OUTPUT_DIR}/training.log

echo "[3/3] Extracting and analyzing results..."

# Extract key metrics from training log
python3 <<EOF
import re
import json
import yaml
from pathlib import Path

output_dir = Path("${OUTPUT_DIR}")
log_file = output_dir / "training.log"

results = {
    "test_info": {
        "timestamp": "${TIMESTAMP}",
        "status": "unknown",
        "errors": []
    },
    "belief_grouping": {
        "num_groups": None,
        "mean_group_size": None,
        "group_sizes": []
    },
    "rewards": {
        "episode_reward_mean": None,
        "intrinsic_reward_mean": None,
        "intrinsic_reward_std": None
    },
    "performance": {
        "success_rate": None,
        "avg_steps": None,
        "efficiency": None
    },
    "belief_consistency": {
        "format_valid_rate": None,
        "action_valid_rate": None
    },
    "sample_trajectories": []
}

if log_file.exists():
    with open(log_file, 'r') as f:
        log_content = f.read()

    # Extract belief grouping stats
    num_groups_match = re.search(r"Number of groups:\s*(\d+)", log_content)
    if num_groups_match:
        results["belief_grouping"]["num_groups"] = int(num_groups_match.group(1))

    mean_size_match = re.search(r"Mean group size:\s*([\d.]+)", log_content)
    if mean_size_match:
        results["belief_grouping"]["mean_group_size"] = float(mean_size_match.group(1))

    # Extract reward stats
    ep_reward_match = re.search(r"episode_rewards_mean.*?([\d.]+)", log_content)
    if ep_reward_match:
        results["rewards"]["episode_reward_mean"] = float(ep_reward_match.group(1))

    # Extract success rate
    success_match = re.search(r"success_rate.*?([\d.]+)", log_content)
    if success_match:
        results["performance"]["success_rate"] = float(success_match.group(1))

    # Check for errors
    if "Error" in log_content or "ERROR" in log_content or "Traceback" in log_content:
        results["test_info"]["status"] = "failed"
        error_lines = [line for line in log_content.split('\n') if 'error' in line.lower() or 'traceback' in line.lower()]
        results["test_info"]["errors"] = error_lines[:10]  # First 10 error lines
    else:
        results["test_info"]["status"] = "success"
else:
    results["test_info"]["status"] = "failed"
    results["test_info"]["errors"] = ["Training log file not found"]

# Save results
with open(output_dir / "test_results.json", 'w') as f:
    json.dump(results, f, indent=2)

with open(output_dir / "test_results.yaml", 'w') as f:
    yaml.dump(results, f, default_flow_style=False)

print("\n" + "="*60)
print("TEST RESULTS SUMMARY")
print("="*60)
print(f"Status: {results['test_info']['status'].upper()}")
if results['belief_grouping']['num_groups']:
    print(f"Belief Groups: {results['belief_grouping']['num_groups']}")
    print(f"Mean Group Size: {results['belief_grouping']['mean_group_size']:.2f}")
if results['performance']['success_rate']:
    print(f"Success Rate: {results['performance']['success_rate']*100:.1f}%")
if results['rewards']['episode_reward_mean']:
    print(f"Episode Reward: {results['rewards']['episode_reward_mean']:.3f}")
print("="*60)
print(f"\nDetailed results saved to: {output_dir}/test_results.json")
print("="*60)
EOF

echo ""
echo "=========================================="
echo "Test Complete!"
echo "Results Directory: ${OUTPUT_DIR}"
echo "Key Files:"
echo "  - test_config.yaml      : Test configuration"
echo "  - test_results.json     : Structured results (JSON)"
echo "  - test_results.yaml     : Structured results (YAML)"
echo "  - training.log          : Full training log"
echo "  - data_prep.log         : Data preparation log"
echo "=========================================="
