#!/bin/bash
# =============================================================================
# BDRS Pipeline 配置文件示例
# =============================================================================
#
# 使用方法：
#   1. 复制此文件: cp pipeline_config.example.sh pipeline_config.sh
#   2. 根据你的环境修改配置
#   3. 加载配置: source pipeline_config.sh
#   4. 运行: bash scripts/run_full_pipeline.sh
#
# =============================================================================

# =============================================================================
# 快速配置模板
# =============================================================================

# 模板1: Qwen2.5-7B 完整训练 (推荐)
export_qwen7b_full() {
    export BASE_MODEL="Qwen/Qwen2.5-7B-Instruct"
    export NUM_GPUS=8
    export EXPERIMENT_NAME="bdrs_qwen7b_full_$(date +%Y%m%d)"

    # API配置
    export OPENAI_API_KEY="sk-your-api-key-here"
    export LLM_MODEL="aws:claude-3-5-sonnet-20241022"
    export NUM_TRAJS=300

    # SFT配置
    export SFT_EPOCHS=5
    export SFT_LR=1e-5
    export SFT_BATCH_SIZE=16

    # RL配置
    export RL_EPOCHS=100
    export RL_LR=1e-6
    export RL_GROUP_SIZE=8
    export RL_SAVE_FREQ=5
    export RL_TEST_FREQ=5

    # 评测配置
    export EVAL_NUM_TASKS=128
    export EVAL_ENGINE="vllm"

    # 流程控制
    export SKIP_DATA_GENERATION=false
    export SKIP_SFT=false
    export SKIP_SFT_EVAL=false
    export SKIP_RL=false
    export SKIP_RL_EVAL=false
}

# 模板2: Qwen2.5-1.5B 快速实验
export_qwen1_5b_fast() {
    export BASE_MODEL="Qwen/Qwen2.5-1.5B-Instruct"
    export NUM_GPUS=8
    export EXPERIMENT_NAME="bdrs_qwen1.5b_fast_$(date +%Y%m%d)"

    export OPENAI_API_KEY="sk-your-api-key-here"
    export LLM_MODEL="gpt-4o-mini"  # 使用更便宜的模型
    export NUM_TRAJS=100  # 减少轨迹数量

    export SFT_EPOCHS=3
    export SFT_LR=1e-5
    export SFT_BATCH_SIZE=16
    export SFT_ULYSSES_SIZE=4

    export RL_EPOCHS=50  # 减少epoch
    export RL_LR=1e-6
    export RL_GROUP_SIZE=8
    export RL_SAVE_FREQ=5
    export RL_TEST_FREQ=5

    export EVAL_NUM_TASKS=64  # 减少评测任务
    export EVAL_ENGINE="vllm"

    export SKIP_DATA_GENERATION=false
    export SKIP_SFT=false
    export SKIP_SFT_EVAL=false
    export SKIP_RL=false
    export SKIP_RL_EVAL=false
}

# 模板2B: Qwen2.5-1.5B 2GPU配置（新增）
export_qwen1_5b_2gpu() {
    # 使用本地缓存的模型路径
    export BASE_MODEL="/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
    export NUM_GPUS=2
    export EXPERIMENT_NAME="bdrs_qwen1.5b_2gpu_$(date +%Y%m%d)"

    # SFT配置（2GPU优化，降低显存占用）
    export SFT_EPOCHS=5
    export SFT_LR=1e-5
    export SFT_BATCH_SIZE=8  # 降低batch size从32到8
    export SFT_ULYSSES_SIZE=1  # 2GPU时必须设为1
    export SFT_GRADIENT_CHECKPOINTING=True  # 启用gradient checkpointing节省显存

    # RL配置
    export RL_EPOCHS=50
    export RL_LR=1e-6
    export RL_GROUP_SIZE=4  # 2GPU适合4个并行环境
    export RL_SAVE_FREQ=5
    export RL_TEST_FREQ=5

    # 评测配置
    export EVAL_NUM_TASKS=64
    export EVAL_ENGINE="vllm"

    # 流程控制（仅SFT训练和评测）
    export SKIP_DATA_GENERATION=true
    export SKIP_SFT=false
    export SKIP_SFT_EVAL=false
    export SKIP_RL=true
    export SKIP_RL_EVAL=true
}

# 模板3: 仅RL训练（跳过数据生成和SFT）
export_rl_only() {
    export BASE_MODEL="Qwen/Qwen2.5-7B-Instruct"
    export NUM_GPUS=8
    export EXPERIMENT_NAME="bdrs_rl_only_$(date +%Y%m%d)"

    # 指定已有的SFT模型路径
    export SFT_CHECKPOINT_DIR="./checkpoints/cold_start/alfworld/existing_model"
    export SFT_EPOCHS=5  # 使用的checkpoint epoch

    export RL_EPOCHS=100
    export RL_LR=1e-6
    export RL_GROUP_SIZE=8
    export RL_SAVE_FREQ=5
    export RL_TEST_FREQ=5

    export EVAL_NUM_TASKS=128
    export EVAL_ENGINE="vllm"

    # 跳过前面的步骤
    export SKIP_DATA_GENERATION=true
    export SKIP_SFT=true
    export SKIP_SFT_EVAL=true
    export SKIP_RL=false
    export SKIP_RL_EVAL=false
}

# 模板4: 仅评测（跳过所有训练）
export_eval_only() {
    export BASE_MODEL="Qwen/Qwen2.5-7B-Instruct"
    export NUM_GPUS=2  # 评测只需要少量GPU
    export EXPERIMENT_NAME="bdrs_eval_$(date +%Y%m%d)"

    # 指定模型路径
    export SFT_CHECKPOINT_DIR="./checkpoints/cold_start/alfworld/existing_sft"
    export SFT_EPOCHS=5
    export RL_CHECKPOINT_DIR="./checkpoints/rl_training/alfworld/existing_rl"
    export RL_EPOCHS=100

    export EVAL_NUM_TASKS=256  # 完整评测
    export EVAL_ENGINE="vllm"

    # 跳过所有训练
    export SKIP_DATA_GENERATION=true
    export SKIP_SFT=true
    export SKIP_SFT_EVAL=false
    export SKIP_RL=true
    export SKIP_RL_EVAL=false
}

# 模板5: 最小化测试（快速验证流程）
export_minimal_test() {
    export BASE_MODEL="Qwen/Qwen2.5-1.5B-Instruct"
    export NUM_GPUS=4
    export EXPERIMENT_NAME="bdrs_minimal_test_$(date +%Y%m%d)"

    export OPENAI_API_KEY="sk-your-api-key-here"
    export LLM_MODEL="gpt-4o-mini"
    export NUM_TRAJS=50  # 最少轨迹

    export SFT_EPOCHS=1  # 最少epoch
    export SFT_LR=1e-5
    export SFT_BATCH_SIZE=8

    export RL_EPOCHS=10  # 最少epoch
    export RL_LR=1e-6
    export RL_GROUP_SIZE=4
    export RL_SAVE_FREQ=5
    export RL_TEST_FREQ=5

    export EVAL_NUM_TASKS=4  # 最少任务
    export EVAL_ENGINE="vllm"

    export SKIP_DATA_GENERATION=false
    export SKIP_SFT=false
    export SKIP_SFT_EVAL=true  # 跳过SFT评测以节省时间
    export SKIP_RL=false
    export SKIP_RL_EVAL=true  # 跳过RL评测以节省时间
}

# =============================================================================
# 使用说明
# =============================================================================

show_templates() {
    cat << EOF
可用配置模板:

1. qwen7b_full      - Qwen2.5-7B完整训练（推荐，需8×A100，约14小时）
2. qwen1_5b_fast    - Qwen2.5-1.5B快速实验（8×A100，约4小时）
2b. qwen1_5b_2gpu   - Qwen2.5-1.5B 2GPU配置（2×GPU，仅SFT+评测，约1小时）⭐新增
3. rl_only          - 仅RL训练（跳过数据生成和SFT）
4. eval_only        - 仅评测（跳过所有训练）
5. minimal_test     - 最小化测试（快速验证流程，约1小时）

使用方法:
    source pipeline_config.example.sh
    load_template <template_name>
    bash scripts/run_full_pipeline.sh

示例:
    source pipeline_config.example.sh
    load_template qwen1_5b_2gpu
    bash scripts/run_full_pipeline.sh

EOF
}

# 加载指定模板
load_template() {
    case $1 in
        qwen7b_full)
            export_qwen7b_full
            echo "Loaded: Qwen2.5-7B Full Training Template"
            ;;
        qwen1_5b_fast)
            export_qwen1_5b_fast
            echo "Loaded: Qwen2.5-1.5B Fast Experiment Template"
            ;;
        qwen1_5b_2gpu)
            export_qwen1_5b_2gpu
            echo "Loaded: Qwen2.5-1.5B 2GPU Template"
            ;;
        rl_only)
            export_rl_only
            echo "Loaded: RL Training Only Template"
            ;;
        eval_only)
            export_eval_only
            echo "Loaded: Evaluation Only Template"
            ;;
        minimal_test)
            export_minimal_test
            echo "Loaded: Minimal Test Template"
            ;;
        help|--help|-h)
            show_templates
            ;;
        *)
            echo "Unknown template: $1"
            show_templates
            return 1
            ;;
    esac

    echo ""
    echo "Current configuration:"
    echo "  - Experiment: ${EXPERIMENT_NAME}"
    echo "  - Base Model: ${BASE_MODEL}"
    echo "  - GPUs: ${NUM_GPUS}"
    echo "  - SFT Epochs: ${SFT_EPOCHS}"
    echo "  - SFT Ulysses Size: ${SFT_ULYSSES_SIZE:-1}"
    echo "  - RL Epochs: ${RL_EPOCHS}"
    echo ""
}

# 显示当前配置
show_config() {
    cat << EOF
Current Pipeline Configuration:
=================================

Experiment:
  Name: ${EXPERIMENT_NAME:-"Not set"}
  Base Model: ${BASE_MODEL:-"Not set"}
  GPUs: ${NUM_GPUS:-"Not set"}

Data Generation:
  API Key: ${OPENAI_API_KEY:0:10}... (hidden)
  LLM Model: ${LLM_MODEL:-"Not set"}
  Trajectories: ${NUM_TRAJS:-"Not set"}

SFT Training:
  Epochs: ${SFT_EPOCHS:-"Not set"}
  Learning Rate: ${SFT_LR:-"Not set"}
  Batch Size: ${SFT_BATCH_SIZE:-"Not set"}

RL Training:
  Epochs: ${RL_EPOCHS:-"Not set"}
  Learning Rate: ${RL_LR:-"Not set"}
  Group Size: ${RL_GROUP_SIZE:-"Not set"}

Evaluation:
  Num Tasks: ${EVAL_NUM_TASKS:-"Not set"}
  Engine: ${EVAL_ENGINE:-"Not set"}

Pipeline Control:
  Skip Data Generation: ${SKIP_DATA_GENERATION:-"false"}
  Skip SFT: ${SKIP_SFT:-"false"}
  Skip SFT Eval: ${SKIP_SFT_EVAL:-"false"}
  Skip RL: ${SKIP_RL:-"false"}
  Skip RL Eval: ${SKIP_RL_EVAL:-"false"}

=================================
EOF
}

# 默认显示帮助
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    show_templates
fi
