#!/bin/bash
# ALFWorld 冷启动模型评测脚本 - 使用固定的测试集数据
#
# 功能：
# 1. 首先将300条数据分割为训练集(240条)和测试集(60条)
# 2. 将测试集转换为parquet格式
# 3. 在ALFWorld环境中评测这60个固定任务的成功率
#
# 用法: bash eval_cold_start_with_testset.sh [模型路径]

set -x

# ============== 配置参数 ==============
MODEL_PATH=${1:-/root/digitalhuman/RLVMR/code/checkpoints/cold_start/alfworld/qwen1.5b_plan_a/global_step_75}
ENGINE=${2:-vllm}
DATA_DIR=/root/data/alfworld
RAW_DATA=/root/digitalhuman/RLVMR/code/data/alfworld_cold-start.json

# ============== 步骤1: 分割数据集 ==============
echo "========== 步骤1: 分割训练集和测试集 =========="
python scripts/split_cold_start_data.py \
    --input $RAW_DATA \
    --output $DATA_DIR \
    --train_ratio 0.8 \
    --seed 42

# ============== 步骤2: 转换测试集为parquet格式 ==============
echo "========== 步骤2: 转换测试集为parquet格式 =========="
python -m examples.data_preprocess.cold_start_data \
    --data_source $DATA_DIR/alfworld_cold-start_test.json \
    --local_dir $DATA_DIR/test

# ============== 步骤3: 评测 ==============
echo "========== 步骤3: 在ALFWorld环境中评测 =========="
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gae \
    data.train_files=$DATA_DIR/test/train.parquet \
    data.val_files=$DATA_DIR/test/train.parquet \
    data.train_batch_size=4 \
    data.val_batch_size=60 \
    data.max_prompt_length=6000 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.0 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=False \
    env.env_name=alfworld/AlfredTWEnv \
    env.seed=0 \
    env.max_steps=30 \
    env.rollout.n=60 \
    env.alfworld.generalization_level=0 \
    env.alfworld.meta_think=True \
    +env.alfworld.action_only=False \
    trainer.logger=[console] \
    trainer.project_name=RLVMR_Evaluation \
    trainer.experiment_name=eval_cold_start_testset \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.total_epochs=0 \
    trainer.val_before_train=True \
    trainer.save_freq=-1 \
    trainer.test_freq=-1

echo "========== 评测完成 =========="
echo "查看日志了解成功率指标"
