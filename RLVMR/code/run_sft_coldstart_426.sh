#!/bin/bash
# SFT Cold Start Training Script for 426 ReBel Trajectories
# Usage: bash run_sft_coldstart_426.sh

set -x

# Enable offline mode for HuggingFace (model already cached locally)
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# 配置参数
ENV=alfworld
DATA_SOURCE=/root/testttt/RLVMR/code/data/alfworld_rebel_merged_final/rebel_coldstart_clean.json
LOCAL_DIR=$HOME/data/alfworld_rebel_426
MODEL_NAME=$HOME/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306
CHECKPOINT_DIR=./checkpoints/cold_start/alfworld/qwen1.5b_rebel_426
EXPERIMENT_NAME=qwen1.5b_rebel_cold-start_426

echo "==================== SFT Cold Start Training ===================="
echo "Data Source: $DATA_SOURCE"
echo "Local Dir: $LOCAL_DIR"
echo "Model: $MODEL_NAME"
echo "Checkpoint: $CHECKPOINT_DIR"
echo "Experiment: $EXPERIMENT_NAME"
echo "================================================================"

# 步骤1: 数据预处理
echo ""
echo "Step 1: Preprocessing data..."
python3 -m examples.data_preprocess.cold_start_data \
    --local_dir=$LOCAL_DIR \
    --data_source=$DATA_SOURCE

# 检查预处理是否成功
if [ ! -f "$LOCAL_DIR/train.parquet" ]; then
    echo "Error: Data preprocessing failed. train.parquet not found."
    exit 1
fi

echo "✓ Data preprocessing completed"
echo "  Train data: $LOCAL_DIR/train.parquet"

# 步骤2: SFT训练
echo ""
echo "Step 2: Starting SFT training..."
torchrun --standalone --nnodes=1 --nproc_per_node=4 \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$LOCAL_DIR/train.parquet \
    data.val_files=$LOCAL_DIR/train.parquet \
    data.prompt_key=extra_info \
    data.response_key=extra_info \
    data.max_length=2048 \
    +data.prompt_dict_keys=['question'] \
    +data.response_dict_keys=['answer'] \
    optim.lr=1e-5 \
    data.micro_batch_size_per_gpu=16 \
    model.partial_pretrain=$MODEL_NAME \
    trainer.default_hdfs_dir=null \
    trainer.project_name=RLVMR_ReBel \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.total_epochs=5 \
    trainer.default_local_dir=$CHECKPOINT_DIR \
    trainer.logger=['console','wandb'] \
    ulysses_sequence_parallel_size=2 \
    use_remove_padding=true

echo ""
echo "==================== Training Completed ===================="
echo "Checkpoint saved to: $CHECKPOINT_DIR"
echo "==========================================================="
