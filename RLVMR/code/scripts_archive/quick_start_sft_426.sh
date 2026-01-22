#!/bin/bash
# Quick Start Script - SFT Training and Evaluation for 426 ReBel Trajectories
# This script runs the complete pipeline: data prep -> training -> evaluation

set -e  # Exit on error

echo "======================================================================="
echo "ReBel SFT Cold Start - Complete Pipeline"
echo "======================================================================="
echo ""

# 配置
CHECKPOINT_DIR=./checkpoints/cold_start/alfworld/qwen1.5b_rebel_426
EXPERIMENT_NAME=qwen1.5b_rebel_426

# 步骤1: 训练
echo "Step 1/2: Starting SFT Training..."
echo "This will train on 426 high-quality ReBel trajectories"
echo "Expected time: ~2-3 hours on 8 GPUs"
echo ""

bash run_sft_coldstart_426.sh

# 检查训练是否完成
if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "Error: Training failed. Checkpoint directory not found."
    exit 1
fi

# 找到最新的checkpoint
LATEST_CHECKPOINT=$(ls -d $CHECKPOINT_DIR/global_step_* 2>/dev/null | sort -V | tail -1)

if [ -z "$LATEST_CHECKPOINT" ]; then
    echo "Error: No checkpoint found in $CHECKPOINT_DIR"
    exit 1
fi

echo ""
echo "✓ Training completed successfully!"
echo "  Latest checkpoint: $LATEST_CHECKPOINT"
echo ""

# 步骤2: 评测
echo "Step 2/2: Starting Evaluation..."
echo "Evaluating on 134 ALFWorld test tasks"
echo "Expected time: ~30 minutes on 2 GPUs"
echo ""

bash eval_sft_coldstart_426.sh $LATEST_CHECKPOINT 134 vllm 2

echo ""
echo "======================================================================="
echo "✓ Complete! Training and evaluation finished successfully."
echo "======================================================================="
echo ""
echo "Results Summary:"
echo "  - Training checkpoint: $LATEST_CHECKPOINT"
echo "  - Evaluation results: Check the console output above"
echo ""
echo "Next steps:"
echo "  1. Check wandb for training metrics"
echo "  2. If evaluation looks good, you can proceed with RL training"
echo "  3. Use examples/rebel_trainer/run_alfworld.sh for ReBel RL training"
echo ""
