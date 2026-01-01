#!/bin/bash
# =============================================================================
# ReBel Full Pipeline - SFT -> Eval -> RL -> Eval -> Report
# =============================================================================
#
# One-click script for complete ReBel experiment pipeline
#
# Usage:
#   bash run_rebel_full_pipeline.sh [OPTIONS]
#
# Options:
#   --skip-sft          Skip SFT training (use existing checkpoint)
#   --skip-sft-eval     Skip SFT evaluation
#   --skip-rl           Skip RL training
#   --skip-rl-eval      Skip RL evaluation
#   --quick             Quick test mode (reduced epochs/tasks)
#
# =============================================================================

set -e  # Exit on error
set -x  # Print commands

# =============================================================================
# Configuration
# =============================================================================

# Experiment settings
EXPERIMENT_NAME=${EXPERIMENT_NAME:-"rebel_full_$(date +%Y%m%d_%H%M%S)"}
NUM_GPUS=${NUM_GPUS:-4}

# Enable offline mode for HuggingFace
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_ATTENTION_BACKEND=XFORMERS

# Model paths
BASE_MODEL=${BASE_MODEL:-"$HOME/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306"}

# SFT Configuration
SFT_DATA_SOURCE=${SFT_DATA_SOURCE:-"/root/testttt/RLVMR/code/data/alfworld_rebel_merged_final/rebel_coldstart_clean.json"}
SFT_LOCAL_DIR=${SFT_LOCAL_DIR:-"$HOME/data/alfworld_rebel_pipeline"}
SFT_CHECKPOINT_DIR=${SFT_CHECKPOINT_DIR:-"./checkpoints/cold_start/alfworld/${EXPERIMENT_NAME}"}
SFT_EPOCHS=${SFT_EPOCHS:-5}

# RL Configuration (Full Training Defaults)
RL_TRAIN_SIZE=${RL_TRAIN_SIZE:-16}
RL_VAL_SIZE=${RL_VAL_SIZE:-16}
RL_GROUP_SIZE=${RL_GROUP_SIZE:-16}
RL_EPOCHS=${RL_EPOCHS:-200}
RL_CHECKPOINT_DIR=${RL_CHECKPOINT_DIR:-"./checkpoints/rl/alfworld/${EXPERIMENT_NAME}"}
RL_SAVE_FREQ=${RL_SAVE_FREQ:-10}
RL_TEST_FREQ=${RL_TEST_FREQ:-10}

# Evaluation Configuration
EVAL_NUM_TASKS=${EVAL_NUM_TASKS:-134}
EVAL_MAX_STEPS=${EVAL_MAX_STEPS:-30}

# Logger Configuration (wandb/swanlab/console)
LOGGER=${LOGGER:-"'console','wandb'"}

# Pipeline control
SKIP_SFT=${SKIP_SFT:-false}
SKIP_SFT_EVAL=${SKIP_SFT_EVAL:-false}
SKIP_RL=${SKIP_RL:-false}
SKIP_RL_EVAL=${SKIP_RL_EVAL:-false}
QUICK_MODE=${QUICK_MODE:-false}

# Results directory
RESULTS_DIR="./results/${EXPERIMENT_NAME}"
REPORT_FILE="${RESULTS_DIR}/experiment_report.md"

# =============================================================================
# Parse arguments
# =============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-sft) SKIP_SFT=true; shift ;;
        --skip-sft-eval) SKIP_SFT_EVAL=true; shift ;;
        --skip-rl) SKIP_RL=true; shift ;;
        --skip-rl-eval) SKIP_RL_EVAL=true; shift ;;
        --quick)
            QUICK_MODE=true
            SFT_EPOCHS=2
            RL_EPOCHS=5
            RL_TRAIN_SIZE=4
            RL_VAL_SIZE=4
            RL_GROUP_SIZE=4
            EVAL_NUM_TASKS=10
            RL_SAVE_FREQ=2
            RL_TEST_FREQ=2
            shift ;;
        --full)
            # Full training mode (default, explicitly set)
            RL_EPOCHS=200
            RL_TRAIN_SIZE=16
            RL_VAL_SIZE=16
            RL_GROUP_SIZE=16
            EVAL_NUM_TASKS=134
            RL_SAVE_FREQ=10
            RL_TEST_FREQ=10
            shift ;;
        --swanlab)
            LOGGER="'console','swanlab'"
            shift ;;
        --help|-h)
            echo "Usage: bash run_rebel_full_pipeline.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --skip-sft        Skip SFT training"
            echo "  --skip-sft-eval   Skip SFT evaluation"
            echo "  --skip-rl         Skip RL training"
            echo "  --skip-rl-eval    Skip RL evaluation"
            echo "  --quick           Quick test mode (5 epochs, 10 tasks)"
            echo "  --full            Full training mode (200 epochs, 134 tasks)"
            echo "  --swanlab         Use SwanLab instead of WandB"
            echo ""
            echo "Environment Variables:"
            echo "  NUM_GPUS=4        Number of GPUs to use"
            echo "  RL_EPOCHS=200     Number of RL training epochs"
            echo "  LOGGER=console,wandb  Logger backend"
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# =============================================================================
# Utility functions
# =============================================================================

print_header() {
    echo ""
    echo "============================================================================="
    echo "$1"
    echo "============================================================================="
    echo ""
}

print_info() {
    echo "[INFO] $1"
}

print_success() {
    echo "[SUCCESS] $1"
}

print_error() {
    echo "[ERROR] $1"
}

get_timestamp() {
    date "+%Y-%m-%d %H:%M:%S"
}

# =============================================================================
# Initialize
# =============================================================================

print_header "ReBel Full Pipeline"

echo "Experiment: ${EXPERIMENT_NAME}"
echo "GPUs: ${NUM_GPUS}"
echo "Quick Mode: ${QUICK_MODE}"
echo ""
echo "Pipeline Steps:"
echo "  [1] SFT Training:    $([ "$SKIP_SFT" = "true" ] && echo "SKIP" || echo "RUN")"
echo "  [2] SFT Evaluation:  $([ "$SKIP_SFT_EVAL" = "true" ] && echo "SKIP" || echo "RUN")"
echo "  [3] RL Training:     $([ "$SKIP_RL" = "true" ] && echo "SKIP" || echo "RUN")"
echo "  [4] RL Evaluation:   $([ "$SKIP_RL_EVAL" = "true" ] && echo "SKIP" || echo "RUN")"
echo ""

# Create results directory
mkdir -p "${RESULTS_DIR}"

# Initialize report
cat > "${REPORT_FILE}" << EOF
# ReBel Experiment Report

**Experiment Name:** ${EXPERIMENT_NAME}
**Date:** $(get_timestamp)
**GPUs:** ${NUM_GPUS}

## Configuration

| Parameter | Value |
|-----------|-------|
| Base Model | Qwen2.5-1.5B-Instruct |
| SFT Epochs | ${SFT_EPOCHS} |
| RL Epochs | ${RL_EPOCHS} |
| RL Group Size | ${RL_GROUP_SIZE} |
| Eval Tasks | ${EVAL_NUM_TASKS} |

---

EOF

# =============================================================================
# Step 1: SFT Training
# =============================================================================

if [ "${SKIP_SFT}" = "false" ]; then
    print_header "Step 1: SFT Cold Start Training"
    SFT_START_TIME=$(date +%s)

    # Data preprocessing
    print_info "Preprocessing SFT data..."
    python3 -m examples.data_preprocess.cold_start_data \
        --local_dir="${SFT_LOCAL_DIR}" \
        --data_source="${SFT_DATA_SOURCE}"

    if [ ! -f "${SFT_LOCAL_DIR}/train.parquet" ]; then
        print_error "Data preprocessing failed!"
        exit 1
    fi

    # SFT training
    print_info "Starting SFT training (${SFT_EPOCHS} epochs)..."
    torchrun --standalone --nnodes=1 --nproc_per_node=${NUM_GPUS} \
        -m verl.trainer.fsdp_sft_trainer \
        data.train_files="${SFT_LOCAL_DIR}/train.parquet" \
        data.val_files="${SFT_LOCAL_DIR}/train.parquet" \
        data.prompt_key=extra_info \
        data.response_key=extra_info \
        data.max_length=4096 \
        +data.prompt_dict_keys=['question'] \
        +data.response_dict_keys=['answer'] \
        optim.lr=1e-5 \
        data.micro_batch_size_per_gpu=16 \
        model.partial_pretrain="${BASE_MODEL}" \
        trainer.default_hdfs_dir=null \
        trainer.project_name=ReBel \
        trainer.experiment_name="${EXPERIMENT_NAME}_sft" \
        trainer.total_epochs=${SFT_EPOCHS} \
        trainer.default_local_dir="${SFT_CHECKPOINT_DIR}" \
        trainer.logger=['console','wandb'] \
        ulysses_sequence_parallel_size=2 \
        use_remove_padding=true

    SFT_END_TIME=$(date +%s)
    SFT_DURATION=$((SFT_END_TIME - SFT_START_TIME))

    # Find the latest checkpoint
    SFT_MODEL_PATH=$(ls -d ${SFT_CHECKPOINT_DIR}/global_step_* 2>/dev/null | sort -V | tail -1)
    if [ -z "${SFT_MODEL_PATH}" ]; then
        SFT_MODEL_PATH="${SFT_CHECKPOINT_DIR}/default/epoch_${SFT_EPOCHS}"
    fi

    print_success "SFT training completed in ${SFT_DURATION}s"
    print_info "Model saved to: ${SFT_MODEL_PATH}"

    # Update report
    cat >> "${REPORT_FILE}" << EOF
## Step 1: SFT Training

- **Duration:** ${SFT_DURATION} seconds
- **Checkpoint:** ${SFT_MODEL_PATH}
- **Status:** Completed

EOF

else
    print_header "Step 1: SFT Training (SKIPPED)"
    # Use existing checkpoint
    SFT_MODEL_PATH=$(ls -d ${SFT_CHECKPOINT_DIR}/global_step_* 2>/dev/null | sort -V | tail -1)
    if [ -z "${SFT_MODEL_PATH}" ]; then
        # Fallback to latest ReBel SFT checkpoint
        SFT_MODEL_PATH=$(ls -d ./checkpoints/cold_start/alfworld/rebel_full_*/global_step_* 2>/dev/null | sort -V | tail -1)
    fi
    if [ -z "${SFT_MODEL_PATH}" ]; then
        # Fallback to old BDRS checkpoint
        SFT_MODEL_PATH="./checkpoints/cold_start/alfworld/bdrs_qwen1.5b_2gpu_20251110/global_step_75"
    fi
    print_info "Using existing SFT model: ${SFT_MODEL_PATH}"
fi

# =============================================================================
# Step 2: SFT Evaluation
# =============================================================================

if [ "${SKIP_SFT_EVAL}" = "false" ]; then
    print_header "Step 2: SFT Model Evaluation"
    SFT_EVAL_START_TIME=$(date +%s)

    SFT_EVAL_DIR="${RESULTS_DIR}/sft_eval"
    mkdir -p "${SFT_EVAL_DIR}"

    print_info "Evaluating SFT model on ${EVAL_NUM_TASKS} tasks..."

    python3 scripts/simple_eval.py \
        --model_path "${SFT_MODEL_PATH}" \
        --num_tasks ${EVAL_NUM_TASKS} \
        --max_steps ${EVAL_MAX_STEPS} \
        --temperature 0.0 \
        --project_name "ReBel" \
        --experiment_name "${EXPERIMENT_NAME}_sft_eval" \
        --gpu_memory_utilization 0.7 \
        --tensor_parallel_size 1 \
        --generalization_level 0 \
        --seed 0 \
        --output_dir "${SFT_EVAL_DIR}" \
        2>&1 | tee "${SFT_EVAL_DIR}/eval.log"

    SFT_EVAL_END_TIME=$(date +%s)
    SFT_EVAL_DURATION=$((SFT_EVAL_END_TIME - SFT_EVAL_START_TIME))

    # Extract results from log
    SFT_SUCCESS_RATE=$(grep -oP 'Success Rate:\s+\K[\d.]+' "${SFT_EVAL_DIR}/eval.log" | tail -1 || echo "N/A")
    SFT_AVG_STEPS=$(grep -oP 'Average Steps:\s+\K[\d.]+' "${SFT_EVAL_DIR}/eval.log" | tail -1 || echo "N/A")

    print_success "SFT evaluation completed"
    print_info "Success Rate: ${SFT_SUCCESS_RATE}%"

    # Update report
    cat >> "${REPORT_FILE}" << EOF
## Step 2: SFT Evaluation

- **Duration:** ${SFT_EVAL_DURATION} seconds
- **Tasks:** ${EVAL_NUM_TASKS}
- **Success Rate:** ${SFT_SUCCESS_RATE}%
- **Average Steps:** ${SFT_AVG_STEPS}

EOF

else
    print_header "Step 2: SFT Evaluation (SKIPPED)"
    SFT_SUCCESS_RATE="N/A"
fi

# =============================================================================
# Step 3: RL Training
# =============================================================================

if [ "${SKIP_RL}" = "false" ]; then
    print_header "Step 3: ReBel RL Training"
    RL_START_TIME=$(date +%s)

    # Prepare RL data
    print_info "Preparing RL training data..."
    python3 -m examples.data_preprocess.prepare \
        --mode 'text' \
        --train_data_size ${RL_TRAIN_SIZE} \
        --val_data_size ${RL_VAL_SIZE}

    # RL training
    print_info "Starting ReBel RL training (${RL_EPOCHS} epochs)..."
    python3 -m verl.trainer.main_ppo \
        algorithm.adv_estimator=rebel \
        algorithm.rebel.enable=True \
        algorithm.rebel.belief_granularity='subgoal' \
        algorithm.rebel.step_advantage_w=1.0 \
        algorithm.rebel.mode='mean_norm' \
        data.train_files=$HOME/data/verl-agent/text/train.parquet \
        data.val_files=$HOME/data/verl-agent/text/test.parquet \
        data.train_batch_size=${RL_TRAIN_SIZE} \
        data.val_batch_size=${RL_VAL_SIZE} \
        data.max_prompt_length=6000 \
        data.max_response_length=1024 \
        data.filter_overlong_prompts=True \
        data.truncation='error' \
        data.return_raw_chat=True \
        actor_rollout_ref.model.path="${SFT_MODEL_PATH}" \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.ppo_mini_batch_size=256 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
        actor_rollout_ref.actor.use_kl_loss=True \
        actor_rollout_ref.actor.kl_loss_coef=0.01 \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
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
        actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
        actor_rollout_ref.rollout.val_kwargs.do_sample=True \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        actor_rollout_ref.actor.use_invalid_action_penalty=True \
        actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
        algorithm.use_kl_in_reward=False \
        env.env_name=alfworld/AlfredTWEnv \
        env.seed=0 \
        env.max_steps=30 \
        env.rollout.n=${RL_GROUP_SIZE} \
        env.alfworld.generalization_level=0 \
        env.alfworld.meta_think=True \
        env.alfworld.use_rebel=True \
        trainer.critic_warmup=0 \
        trainer.logger="[${LOGGER}]" \
        trainer.project_name='ReBel' \
        trainer.experiment_name="${EXPERIMENT_NAME}_rl" \
        trainer.n_gpus_per_node=${NUM_GPUS} \
        trainer.nnodes=1 \
        trainer.resume_mode=auto \
        trainer.save_freq=${RL_SAVE_FREQ} \
        trainer.test_freq=${RL_TEST_FREQ} \
        trainer.total_epochs=${RL_EPOCHS} \
        trainer.default_local_dir="${RL_CHECKPOINT_DIR}" \
        trainer.val_before_train=True

    RL_END_TIME=$(date +%s)
    RL_DURATION=$((RL_END_TIME - RL_START_TIME))

    # Find the latest RL checkpoint
    RL_MODEL_PATH=$(ls -d ${RL_CHECKPOINT_DIR}/global_step_* 2>/dev/null | sort -V | tail -1)
    if [ -z "${RL_MODEL_PATH}" ]; then
        RL_MODEL_PATH="${RL_CHECKPOINT_DIR}/default/epoch_${RL_EPOCHS}"
    fi

    print_success "RL training completed in ${RL_DURATION}s"
    print_info "Model saved to: ${RL_MODEL_PATH}"

    # Update report
    cat >> "${REPORT_FILE}" << EOF
## Step 3: RL Training

- **Duration:** ${RL_DURATION} seconds ($(echo "scale=2; ${RL_DURATION}/3600" | bc) hours)
- **Epochs:** ${RL_EPOCHS}
- **Checkpoint:** ${RL_MODEL_PATH}
- **Status:** Completed

EOF

else
    print_header "Step 3: RL Training (SKIPPED)"
    RL_MODEL_PATH=$(ls -d ${RL_CHECKPOINT_DIR}/global_step_* 2>/dev/null | sort -V | tail -1)
    if [ -z "${RL_MODEL_PATH}" ]; then
        print_error "No RL checkpoint found!"
        exit 1
    fi
    print_info "Using existing RL model: ${RL_MODEL_PATH}"
fi

# =============================================================================
# Step 4: RL Evaluation
# =============================================================================

if [ "${SKIP_RL_EVAL}" = "false" ]; then
    print_header "Step 4: RL Model Evaluation"
    RL_EVAL_START_TIME=$(date +%s)

    RL_EVAL_DIR="${RESULTS_DIR}/rl_eval"
    mkdir -p "${RL_EVAL_DIR}"

    print_info "Evaluating RL model on ${EVAL_NUM_TASKS} tasks..."

    python3 scripts/simple_eval.py \
        --model_path "${RL_MODEL_PATH}" \
        --num_tasks ${EVAL_NUM_TASKS} \
        --max_steps ${EVAL_MAX_STEPS} \
        --temperature 0.0 \
        --project_name "ReBel" \
        --experiment_name "${EXPERIMENT_NAME}_rl_eval" \
        --gpu_memory_utilization 0.7 \
        --tensor_parallel_size 1 \
        --generalization_level 0 \
        --seed 0 \
        --output_dir "${RL_EVAL_DIR}" \
        2>&1 | tee "${RL_EVAL_DIR}/eval.log"

    RL_EVAL_END_TIME=$(date +%s)
    RL_EVAL_DURATION=$((RL_EVAL_END_TIME - RL_EVAL_START_TIME))

    # Extract results from log
    RL_SUCCESS_RATE=$(grep -oP 'Success Rate:\s+\K[\d.]+' "${RL_EVAL_DIR}/eval.log" | tail -1 || echo "N/A")
    RL_AVG_STEPS=$(grep -oP 'Average Steps:\s+\K[\d.]+' "${RL_EVAL_DIR}/eval.log" | tail -1 || echo "N/A")

    print_success "RL evaluation completed"
    print_info "Success Rate: ${RL_SUCCESS_RATE}%"

    # Update report
    cat >> "${REPORT_FILE}" << EOF
## Step 4: RL Evaluation

- **Duration:** ${RL_EVAL_DURATION} seconds
- **Tasks:** ${EVAL_NUM_TASKS}
- **Success Rate:** ${RL_SUCCESS_RATE}%
- **Average Steps:** ${RL_AVG_STEPS}

EOF

else
    print_header "Step 4: RL Evaluation (SKIPPED)"
    RL_SUCCESS_RATE="N/A"
fi

# =============================================================================
# Generate Final Report
# =============================================================================

print_header "Generating Experiment Report"

# Calculate improvement
if [ "${SFT_SUCCESS_RATE}" != "N/A" ] && [ "${RL_SUCCESS_RATE}" != "N/A" ]; then
    IMPROVEMENT=$(echo "scale=2; ${RL_SUCCESS_RATE} - ${SFT_SUCCESS_RATE}" | bc)
else
    IMPROVEMENT="N/A"
fi

# Finalize report
cat >> "${REPORT_FILE}" << EOF
---

## Summary

| Metric | SFT Model | RL Model | Improvement |
|--------|-----------|----------|-------------|
| Success Rate | ${SFT_SUCCESS_RATE}% | ${RL_SUCCESS_RATE}% | ${IMPROVEMENT}% |
| Avg Steps | ${SFT_AVG_STEPS:-N/A} | ${RL_AVG_STEPS:-N/A} | - |

## Artifacts

- **SFT Checkpoint:** ${SFT_MODEL_PATH}
- **RL Checkpoint:** ${RL_MODEL_PATH}
- **Results Directory:** ${RESULTS_DIR}
- **WandB Project:** [ReBel](https://wandb.ai)

## Notes

- Framework: ReBel (Reward from Belief)
- Base Model: Qwen2.5-1.5B-Instruct
- Environment: ALFWorld
- Algorithm: PPO with belief-based dense rewards

---
*Report generated at $(get_timestamp)*
EOF

print_success "Report saved to: ${REPORT_FILE}"

# Print summary
print_header "Pipeline Complete!"

echo "============================================"
echo "           EXPERIMENT SUMMARY"
echo "============================================"
echo ""
echo "Experiment:     ${EXPERIMENT_NAME}"
echo ""
echo "SFT Model:      ${SFT_MODEL_PATH}"
echo "SFT Success:    ${SFT_SUCCESS_RATE}%"
echo ""
echo "RL Model:       ${RL_MODEL_PATH}"
echo "RL Success:     ${RL_SUCCESS_RATE}%"
echo ""
echo "Improvement:    ${IMPROVEMENT}%"
echo ""
echo "============================================"
echo "Report:         ${REPORT_FILE}"
echo "WandB:          https://wandb.ai"
echo "============================================"
echo ""

# Display report
echo ""
echo "--- Experiment Report ---"
cat "${REPORT_FILE}"
