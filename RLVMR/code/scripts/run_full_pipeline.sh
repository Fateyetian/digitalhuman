#!/bin/bash
# =============================================================================
# BDRS 全流程自动化脚本
# =============================================================================
#
# 功能：一键运行从冷启动数据生成到RL训练和评测的完整流程
#
# 使用方法：
#   bash scripts/run_full_pipeline.sh [config_file]
#
# 示例：
#   bash scripts/run_full_pipeline.sh configs/bdrs_qwen7b.yaml
#   bash scripts/run_full_pipeline.sh  # 使用默认配置
#
# =============================================================================

set -e  # 遇到错误立即退出
set -u  # 使用未定义变量时报错

# =============================================================================
# 配置参数
# =============================================================================

# 基础配置
BASE_MODEL=${BASE_MODEL:-"Qwen/Qwen2.5-7B-Instruct"}  # 基础模型
NUM_GPUS=${NUM_GPUS:-8}                                # GPU数量
EXPERIMENT_NAME=${EXPERIMENT_NAME:-"bdrs_full_pipeline_$(date +%Y%m%d_%H%M%S)"}

# API配置（用于冷启动数据生成）
OPENAI_API_KEY=${OPENAI_API_KEY:-""}                   # OpenAI API Key
LLM_MODEL=${LLM_MODEL:-"aws:claude-3-5-sonnet-20241022"}  # 标注模型
NUM_TRAJS=${NUM_TRAJS:-300}                            # 生成轨迹数量

# 冷启动数据配置
COLD_START_DATA_DIR=${COLD_START_DATA_DIR:-"data"}
COLD_START_OUTPUT=${COLD_START_OUTPUT:-"${COLD_START_DATA_DIR}/alfworld_cold-start.json"}
EXPERT_TRAJ_PATH=${EXPERT_TRAJ_PATH:-"${COLD_START_DATA_DIR}/alfworld_expert_traj"}

# SFT训练配置
SFT_EPOCHS=${SFT_EPOCHS:-5}
SFT_LR=${SFT_LR:-1e-5}
SFT_BATCH_SIZE=${SFT_BATCH_SIZE:-16}
SFT_ULYSSES_SIZE=${SFT_ULYSSES_SIZE:-1}  # Ulysses序列并行大小，2GPU建议设为1
SFT_GRADIENT_CHECKPOINTING=${SFT_GRADIENT_CHECKPOINTING:-False}  # 是否启用gradient checkpointing
SFT_CHECKPOINT_DIR=${SFT_CHECKPOINT_DIR:-"./checkpoints/cold_start/alfworld/${EXPERIMENT_NAME}"}

# RL训练配置
RL_EPOCHS=${RL_EPOCHS:-100}
RL_LR=${RL_LR:-1e-6}
RL_GROUP_SIZE=${RL_GROUP_SIZE:-8}
RL_SAVE_FREQ=${RL_SAVE_FREQ:-5}
RL_TEST_FREQ=${RL_TEST_FREQ:-5}
RL_CHECKPOINT_DIR=${RL_CHECKPOINT_DIR:-"./checkpoints/rl_training/alfworld/${EXPERIMENT_NAME}"}

# 评测配置
EVAL_NUM_TASKS=${EVAL_NUM_TASKS:-128}
EVAL_ENGINE=${EVAL_ENGINE:-"vllm"}

# 流程控制开关（可通过环境变量控制跳过某些步骤）
SKIP_DATA_GENERATION=${SKIP_DATA_GENERATION:-false}   # 跳过冷启动数据生成
SKIP_SFT=${SKIP_SFT:-false}                          # 跳过SFT训练
SKIP_SFT_EVAL=${SKIP_SFT_EVAL:-false}                # 跳过SFT评测
SKIP_RL=${SKIP_RL:-false}                            # 跳过RL训练
SKIP_RL_EVAL=${SKIP_RL_EVAL:-false}                  # 跳过RL评测

# =============================================================================
# 工具函数
# =============================================================================

# 打印带颜色的信息
print_info() {
    echo -e "\n\033[1;34m[INFO]\033[0m $1\n"
}

print_success() {
    echo -e "\n\033[1;32m[SUCCESS]\033[0m $1\n"
}

print_error() {
    echo -e "\n\033[1;31m[ERROR]\033[0m $1\n"
}

print_warning() {
    echo -e "\n\033[1;33m[WARNING]\033[0m $1\n"
}

# 打印分隔线
print_separator() {
    echo "============================================================================="
}

# 检查命令是否存在
check_command() {
    if ! command -v $1 &> /dev/null; then
        print_error "Command '$1' not found. Please install it first."
        exit 1
    fi
}

# 检查文件是否存在
check_file() {
    if [ ! -f "$1" ]; then
        print_error "File not found: $1"
        exit 1
    fi
}

# 检查目录是否存在
check_dir() {
    if [ ! -d "$1" ]; then
        print_error "Directory not found: $1"
        exit 1
    fi
}

# 等待用户确认
wait_for_confirmation() {
    read -p "Press Enter to continue or Ctrl+C to abort..."
}

# =============================================================================
# 主流程
# =============================================================================

main() {
    print_separator
    echo "BDRS Full Pipeline - Automated Training Script"
    print_separator

    print_info "Experiment Name: ${EXPERIMENT_NAME}"
    print_info "Base Model: ${BASE_MODEL}"
    print_info "Number of GPUs: ${NUM_GPUS}"
    print_separator

    # 检查环境
    print_info "Step 0: Checking environment..."
    check_command python3
    check_command torchrun

    # 切换到代码目录
    cd "$(dirname "$0")/.." || exit 1
    WORK_DIR=$(pwd)
    print_info "Working directory: ${WORK_DIR}"

    # =============================================================================
    # Step 1: 生成冷启动数据
    # =============================================================================
    if [ "${SKIP_DATA_GENERATION}" = "false" ]; then
        print_separator
        print_info "Step 1: Generating cold start data..."
        print_separator

        # 检查API Key
        if [ -z "${OPENAI_API_KEY}" ]; then
            print_warning "OPENAI_API_KEY is not set. Please set it before running this script."
            echo "Example: export OPENAI_API_KEY='sk-...'"
            exit 1
        fi

        # 检查专家轨迹数据
        if [ ! -d "${EXPERT_TRAJ_PATH}" ]; then
            print_error "Expert trajectory data not found at: ${EXPERT_TRAJ_PATH}"
            print_info "Please download or generate expert trajectories first."
            exit 1
        fi

        # 生成冷启动数据
        print_info "Annotating ${NUM_TRAJS} trajectories using ${LLM_MODEL}..."
        python3 scripts/alfworld_prepare.py \
            --api_key="${OPENAI_API_KEY}" \
            --model="${LLM_MODEL}" \
            --num_trajs=${NUM_TRAJS} \
            --output_path="${COLD_START_OUTPUT}" \
            --dataset_path="${EXPERT_TRAJ_PATH}" \
            || { print_error "Data generation failed!"; exit 1; }

        check_file "${COLD_START_OUTPUT}"
        print_success "Cold start data generated: ${COLD_START_OUTPUT}"

        # 转换为parquet格式
        print_info "Converting to parquet format..."
        python3 -m examples.data_preprocess.cold_start_data \
            --local_dir=$HOME/data/alfworld \
            --data_source="${COLD_START_OUTPUT}" \
            || { print_error "Data conversion failed!"; exit 1; }

        print_success "Data converted successfully!"

    else
        print_warning "Skipping data generation (SKIP_DATA_GENERATION=true)"
        check_file "${COLD_START_OUTPUT}"
    fi

    # =============================================================================
    # Step 2: SFT冷启动训练
    # =============================================================================
    if [ "${SKIP_SFT}" = "false" ]; then
        print_separator
        print_info "Step 2: Training SFT cold start model..."
        print_separator

        print_info "Training configuration:"
        echo "  - Base model: ${BASE_MODEL}"
        echo "  - Epochs: ${SFT_EPOCHS}"
        echo "  - Learning rate: ${SFT_LR}"
        echo "  - Batch size: ${SFT_BATCH_SIZE}"
        echo "  - GPUs: ${NUM_GPUS}"
        echo "  - Output: ${SFT_CHECKPOINT_DIR}"

        # 创建输出目录
        mkdir -p "${SFT_CHECKPOINT_DIR}"

        # 启动SFT训练
        torchrun --standalone --nnodes=1 --nproc_per_node=${NUM_GPUS} \
            -m verl.trainer.fsdp_sft_trainer \
            data.train_files=$HOME/data/alfworld/train.parquet \
            data.val_files=$HOME/data/alfworld/train.parquet \
            data.prompt_key=extra_info \
            data.response_key=extra_info \
            data.max_length=3000 \
            +data.prompt_dict_keys=['question'] \
            +data.response_dict_keys=['answer'] \
            optim.lr=${SFT_LR} \
            data.micro_batch_size_per_gpu=${SFT_BATCH_SIZE} \
            model.partial_pretrain=${BASE_MODEL} \
            model.enable_gradient_checkpointing=${SFT_GRADIENT_CHECKPOINTING} \
            trainer.default_hdfs_dir=null \
            trainer.project_name=BDRS \
            trainer.experiment_name=${EXPERIMENT_NAME}_sft \
            trainer.total_epochs=${SFT_EPOCHS} \
            trainer.default_local_dir=${SFT_CHECKPOINT_DIR} \
            trainer.logger=['console','wandb'] \
            ulysses_sequence_parallel_size=${SFT_ULYSSES_SIZE} \
            use_remove_padding=true \
            || { print_error "SFT training failed!"; exit 1; }

        # 获取最终模型路径（自动查找最新checkpoint）
        # 尝试查找 epoch_X 或 global_step_X 格式的最新checkpoint
        if [ -d "${SFT_CHECKPOINT_DIR}/default/epoch_${SFT_EPOCHS}" ]; then
            SFT_MODEL_PATH="${SFT_CHECKPOINT_DIR}/default/epoch_${SFT_EPOCHS}"
        else
            # 查找最新的 global_step checkpoint
            LATEST_CHECKPOINT=$(ls -d ${SFT_CHECKPOINT_DIR}/global_step_* 2>/dev/null | sort -V | tail -1)
            if [ -z "${LATEST_CHECKPOINT}" ]; then
                print_error "No checkpoint found in ${SFT_CHECKPOINT_DIR}"
                exit 1
            fi
            SFT_MODEL_PATH="${LATEST_CHECKPOINT}"
        fi
        check_dir "${SFT_MODEL_PATH}"
        print_success "SFT training completed: ${SFT_MODEL_PATH}"

    else
        print_warning "Skipping SFT training (SKIP_SFT=true)"
        # 自动查找最新checkpoint
        if [ -d "${SFT_CHECKPOINT_DIR}/default/epoch_${SFT_EPOCHS}" ]; then
            SFT_MODEL_PATH="${SFT_CHECKPOINT_DIR}/default/epoch_${SFT_EPOCHS}"
        else
            LATEST_CHECKPOINT=$(ls -d ${SFT_CHECKPOINT_DIR}/global_step_* 2>/dev/null | sort -V | tail -1)
            if [ -z "${LATEST_CHECKPOINT}" ]; then
                print_error "No checkpoint found in ${SFT_CHECKPOINT_DIR}"
                exit 1
            fi
            SFT_MODEL_PATH="${LATEST_CHECKPOINT}"
        fi
        check_dir "${SFT_MODEL_PATH}"
    fi

    # =============================================================================
    # Step 3: 评测冷启动模型
    # =============================================================================
    if [ "${SKIP_SFT_EVAL}" = "false" ]; then
        print_separator
        print_info "Step 3: Evaluating SFT cold start model..."
        print_separator

        print_info "Evaluating ${EVAL_NUM_TASKS} tasks using rollout method..."

        EVAL_OUTPUT_DIR="results/eval_sft_${EXPERIMENT_NAME}"
        mkdir -p "${EVAL_OUTPUT_DIR}"

        # 启动vLLM服务器
        print_info "Starting vLLM server..."
        python -m vllm.entrypoints.openai.api_server \
            --model "${SFT_MODEL_PATH}" \
            --host 0.0.0.0 \
            --port 8000 \
            --gpu-memory-utilization 0.6 \
            > "${EVAL_OUTPUT_DIR}/vllm.log" 2>&1 &
        VLLM_PID=$!
        print_info "Waiting for vLLM initialization (60s)..."
        sleep 60

        # 运行rollout评测
        bash examples/bdrs_trainer/rollout/run_local_eval.sh \
            "${SFT_MODEL_PATH}" \
            ${EVAL_NUM_TASKS} \
            8 \
            30 \
            "${EVAL_OUTPUT_DIR}" \
            || { print_error "SFT evaluation failed!"; kill ${VLLM_PID} 2>/dev/null || true; exit 1; }

        # 关闭vLLM
        kill ${VLLM_PID} 2>/dev/null || true

        # 解析结果并记录到WandB
        print_info "Logging results to WandB..."
        python3 << EOF
import json, wandb
from collections import defaultdict

try:
    with open("${EVAL_OUTPUT_DIR}/trajectory.jsonl", 'r') as f:
        episodes = defaultdict(lambda: {'steps': 0, 'won': False})
        for line in f:
            d = json.loads(line)
            env_id = d['env_id']
            episodes[env_id]['steps'] += 1
            if d.get('done'): episodes[env_id]['won'] = d.get('won', False)

        eps = list(episodes.values())
        success_rate = sum(1 for e in eps if e['won']) / len(eps)
        avg_steps = sum(e['steps'] for e in eps) / len(eps)

        print(f"\n=== Evaluation Results ===")
        print(f"Success Rate: {success_rate:.2%}")
        print(f"Average Steps: {avg_steps:.1f}")

        wandb.init(project="BDRS", name="eval_sft_${EXPERIMENT_NAME}")
        wandb.log({"val/success_rate": success_rate, "val/episode_length": avg_steps})
        wandb.finish()
except Exception as e:
    print(f"Warning: Failed to log to WandB: {e}")
EOF

        print_success "SFT evaluation completed!"
        print_info "Results saved to: ${EVAL_OUTPUT_DIR}"
        print_info "WandB: https://wandb.ai"

    else
        print_warning "Skipping SFT evaluation (SKIP_SFT_EVAL=true)"
    fi

    # =============================================================================
    # Step 4: RL训练
    # =============================================================================
    if [ "${SKIP_RL}" = "false" ]; then
        print_separator
        print_info "Step 4: Training RL model with BDRS..."
        print_separator

        print_info "RL training configuration:"
        echo "  - Cold start model: ${SFT_MODEL_PATH}"
        echo "  - Epochs: ${RL_EPOCHS}"
        echo "  - Learning rate: ${RL_LR}"
        echo "  - Group size: ${RL_GROUP_SIZE}"
        echo "  - Save frequency: ${RL_SAVE_FREQ}"
        echo "  - Test frequency: ${RL_TEST_FREQ}"
        echo "  - Output: ${RL_CHECKPOINT_DIR}"

        # 创建输出目录
        mkdir -p "${RL_CHECKPOINT_DIR}"

        # 准备RL训练数据
        print_info "Preparing RL training data..."
        train_data_size=16
        val_data_size=128

        python3 -m examples.data_preprocess.prepare \
            --mode 'text' \
            --train_data_size ${train_data_size} \
            --val_data_size ${val_data_size} \
            || { print_error "RL data preparation failed!"; exit 1; }

        # 启动RL训练
        print_info "Starting RL training (this may take 8-12 hours)..."
        python3 -m verl.trainer.main_ppo \
            algorithm.adv_estimator=bdrs \
            algorithm.bdrs.enable=True \
            algorithm.bdrs.step_advantage_w=1.0 \
            data.train_files=$HOME/data/verl-agent/text/train.parquet \
            data.val_files=$HOME/data/verl-agent/text/test.parquet \
            data.train_batch_size=${train_data_size} \
            data.val_batch_size=${val_data_size} \
            data.max_prompt_length=6000 \
            data.max_response_length=1024 \
            data.filter_overlong_prompts=True \
            data.truncation='error' \
            data.return_raw_chat=True \
            actor_rollout_ref.model.path=${SFT_MODEL_PATH} \
            actor_rollout_ref.actor.optim.lr=${RL_LR} \
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
            actor_rollout_ref.rollout.name=${EVAL_ENGINE} \
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
            trainer.critic_warmup=0 \
            trainer.logger=['console','wandb'] \
            trainer.project_name='BDRS' \
            trainer.experiment_name=${EXPERIMENT_NAME}_rl \
            trainer.n_gpus_per_node=${NUM_GPUS} \
            trainer.nnodes=1 \
            trainer.resume_mode=auto \
            trainer.save_freq=${RL_SAVE_FREQ} \
            trainer.test_freq=${RL_TEST_FREQ} \
            trainer.total_epochs=${RL_EPOCHS} \
            trainer.default_local_dir=${RL_CHECKPOINT_DIR} \
            trainer.val_before_train=True \
            || { print_error "RL training failed!"; exit 1; }

        # 获取最终模型路径
        RL_MODEL_PATH="${RL_CHECKPOINT_DIR}/epoch_${RL_EPOCHS}"
        check_dir "${RL_MODEL_PATH}"
        print_success "RL training completed: ${RL_MODEL_PATH}"

    else
        print_warning "Skipping RL training (SKIP_RL=true)"
        RL_MODEL_PATH="${RL_CHECKPOINT_DIR}/epoch_${RL_EPOCHS}"
        check_dir "${RL_MODEL_PATH}"
    fi

    # =============================================================================
    # Step 5: 评测RL模型
    # =============================================================================
    if [ "${SKIP_RL_EVAL}" = "false" ]; then
        print_separator
        print_info "Step 5: Evaluating final RL model..."
        print_separator

        # 评测多个checkpoint
        print_info "Evaluating multiple checkpoints to find the best model..."
        best_epoch=0
        best_model_path=""

        # 评测最后5个checkpoint
        for epoch in $(seq $((RL_EPOCHS - 20)) ${RL_SAVE_FREQ} ${RL_EPOCHS}); do
            if [ ${epoch} -gt 0 ]; then
                model_path="${RL_CHECKPOINT_DIR}/epoch_${epoch}"
                if [ -d "${model_path}" ]; then
                    print_info "Evaluating epoch ${epoch}..."

                    # 提示启动vLLM
                    print_info "Please start vLLM server for this checkpoint:"
                    echo ""
                    echo "  pkill -f 'vllm.entrypoints.openai.api_server'  # Kill previous vLLM"
                    echo "  python -m vllm.entrypoints.openai.api_server \\"
                    echo "      --model ${model_path} \\"
                    echo "      --host 0.0.0.0 \\"
                    echo "      --port 8000 \\"
                    echo "      --gpu-memory-utilization 0.7"
                    echo ""
                    read -p "Press Enter after starting vLLM..."

                    # 运行评测
                    bash examples/bdrs_trainer/rollout/run_local_eval.sh \
                        "${BASE_MODEL}" \
                        ${EVAL_NUM_TASKS} \
                        8 \
                        30 \
                        results/eval_rl_epoch${epoch}_${EXPERIMENT_NAME} \
                        || print_warning "Evaluation failed for epoch ${epoch}"

                    best_epoch=${epoch}
                    best_model_path="${model_path}"
                fi
            fi
        done

        print_success "RL evaluation completed!"
        print_info "Best model: ${best_model_path}"
        print_info "Results saved in: results/eval_rl_epoch*_${EXPERIMENT_NAME}.jsonl"

    else
        print_warning "Skipping RL evaluation (SKIP_RL_EVAL=true)"
        RL_MODEL_PATH="${RL_CHECKPOINT_DIR}/epoch_${RL_EPOCHS}"
    fi

    # =============================================================================
    # 完成
    # =============================================================================
    print_separator
    print_success "BDRS Full Pipeline Completed Successfully!"
    print_separator

    echo "Summary:"
    echo "  - Experiment Name: ${EXPERIMENT_NAME}"
    echo "  - SFT Model: ${SFT_MODEL_PATH}"
    echo "  - RL Model: ${RL_MODEL_PATH}"
    echo ""
    echo "Next steps:"
    echo "  1. Check WandB dashboard for training curves and evaluation metrics"
    echo "  2. Run additional evaluations if needed"
    echo "  3. Compare with baseline methods (RLVMR, Vanilla PPO)"
    echo ""
    print_separator
}

# =============================================================================
# 脚本入口
# =============================================================================

# 显示帮助信息
show_help() {
    cat << EOF
Usage: bash scripts/run_full_pipeline.sh [OPTIONS]

Options:
  -h, --help                  Show this help message
  --experiment-name NAME      Set experiment name
  --base-model MODEL          Set base model (default: Qwen/Qwen2.5-7B-Instruct)
  --num-gpus N                Number of GPUs (default: 8)
  --skip-data                 Skip cold start data generation
  --skip-sft                  Skip SFT training
  --skip-sft-eval            Skip SFT evaluation
  --skip-rl                   Skip RL training
  --skip-rl-eval             Skip RL evaluation

Environment Variables:
  OPENAI_API_KEY             OpenAI API key for data annotation
  BASE_MODEL                 Base model to use
  NUM_GPUS                   Number of GPUs
  EXPERIMENT_NAME            Experiment name

Examples:
  # Run full pipeline
  bash scripts/run_full_pipeline.sh

  # Run with custom experiment name
  bash scripts/run_full_pipeline.sh --experiment-name my_experiment

  # Skip data generation (use existing data)
  bash scripts/run_full_pipeline.sh --skip-data

  # Skip SFT and only run RL training
  bash scripts/run_full_pipeline.sh --skip-data --skip-sft --skip-sft-eval

EOF
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        --experiment-name)
            EXPERIMENT_NAME="$2"
            shift 2
            ;;
        --base-model)
            BASE_MODEL="$2"
            shift 2
            ;;
        --num-gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --skip-data)
            SKIP_DATA_GENERATION=true
            shift
            ;;
        --skip-sft)
            SKIP_SFT=true
            shift
            ;;
        --skip-sft-eval)
            SKIP_SFT_EVAL=true
            shift
            ;;
        --skip-rl)
            SKIP_RL=true
            shift
            ;;
        --skip-rl-eval)
            SKIP_RL_EVAL=true
            shift
            ;;
        *)
            print_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# 运行主流程
main
