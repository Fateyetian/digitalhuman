#!/bin/bash
# =============================================================================
# WebShop ReBel Cold-Start SFT Training
# =============================================================================
# Train Qwen2.5-1.5B on hindsight-annotated WebShop ReBel data.
# Mirrors: examples/sft/cold_start/run_alfworld_qwen2.5-1.5b.sh
#
# Prerequisites:
#   1. Generate hindsight data:
#      python3 scripts/generate_webshop_rebel_hindsight.py \
#        --expert_data data/webshop_expert_traj.json \
#        --output_dir data/webshop_rebel_hindsight \
#        --generate_coldstart
#
#   2. Ensure train.parquet exists at $HOME/data/webshop/train.parquet
#
# Usage: bash run_webshop_sft.sh
# =============================================================================

set -e

ENV=webshop
BASE_MODEL=/root/testttt/RLVMR/code/base_models/Qwen2.5-1.5B-Instruct

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Prepare data if needed
if [ ! -f "$HOME/data/$ENV/train.parquet" ]; then
    echo "ERROR: Training data not found at \$HOME/data/$ENV/train.parquet"
    echo "Please run generate_webshop_rebel_hindsight.py first with --generate_coldstart"
    exit 1
fi

cd /root/testttt/RLVMR/code

torchrun --standalone --nnodes=1 --nproc_per_node=4 \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$HOME/data/$ENV/train.parquet \
    data.val_files=$HOME/data/$ENV/train.parquet \
    data.prompt_key=extra_info \
    data.response_key=extra_info \
    data.max_length=4096 \
    +data.prompt_dict_keys=['question'] \
    +data.response_dict_keys=['answer'] \
    optim.lr=1e-5 \
    data.micro_batch_size_per_gpu=8 \
    model.partial_pretrain=${BASE_MODEL} \
    trainer.default_hdfs_dir=null \
    trainer.project_name=ReBel_WebShop \
    trainer.experiment_name=qwen1.5b_webshop_cold-start \
    trainer.total_epochs=20 \
    trainer.default_local_dir=./checkpoints/cold_start/webshop/rebel_full_qwen1.5b \
    trainer.logger=['console','swanlab'] \
    ulysses_sequence_parallel_size=2 \
    use_remove_padding=true
