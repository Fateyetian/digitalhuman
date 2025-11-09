#!/bin/bash
# =============================================================================
# Simple Evaluation Script (No Ray Required)
# =============================================================================
#
# Features:
#   - No Ray distributed system
#   - Fast startup
#   - SwanLab logging
#   - Easy to debug
#
# Usage:
#   bash scripts/run_simple_eval.sh [MODEL_PATH] [NUM_TASKS]
#
# =============================================================================

set -x

# ============= Configuration =============
MODEL_PATH=${1:-/root/digitalhuman/RLVMR/code/checkpoints/cold_start/alfworld/qwen1.5b_plan_a/global_step_75}
NUM_TASKS=${2:-10}
MAX_STEPS=${3:-30}
TEMPERATURE=${4:-0.0}
ENV_NAME=${5:-alfworld}

# Extract experiment name from model path
MODEL_NAME=$(basename $(dirname $(dirname $MODEL_PATH)))
STEP_NAME=$(basename $MODEL_PATH)
EXPERIMENT_NAME="eval_${MODEL_NAME}_${STEP_NAME}_${NUM_TASKS}tasks"

# ============= Check dependencies =============
echo "Checking dependencies..."

# Check if swanlab is installed
if ! python -c "import swanlab" 2>/dev/null; then
    echo "SwanLab not installed. Installing..."
    pip install swanlab
fi

# Check if model exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model path does not exist: $MODEL_PATH"
    exit 1
fi

# ============= Run Evaluation =============
echo "Starting evaluation..."
echo "  Model: $MODEL_PATH"
echo "  Tasks: $NUM_TASKS"
echo "  Max Steps: $MAX_STEPS"
echo "  Temperature: $TEMPERATURE"
echo "  Environment: $ENV_NAME"
echo "  Experiment: $EXPERIMENT_NAME"
echo ""

python scripts/simple_eval.py \
    --model_path $MODEL_PATH \
    --num_tasks $NUM_TASKS \
    --max_steps $MAX_STEPS \
    --temperature $TEMPERATURE \
    --project_name "RLVMR-Eval" \
    --experiment_name $EXPERIMENT_NAME \
    --gpu_memory_utilization 0.6 \
    --tensor_parallel_size 1 \
    --generalization_level 0 \
    --seed 0 \
    --verbose

echo ""
echo "Evaluation completed!"
echo "View results at: https://swanlab.cn"
