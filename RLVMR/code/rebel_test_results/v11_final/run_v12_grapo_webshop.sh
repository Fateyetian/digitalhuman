#!/bin/bash
# =============================================================================
# GraPO V12 WebShop Experiment Script
# =============================================================================
# Graph-based Path-Level Causal Credit Assignment for LLM Agent RL Training.
#
# Key differences from ReBel V11 (rebel_hibo):
#   - ADV_ESTIMATOR = grapo
#   - GRAPO_ENV_TYPE = webshop  → uses raw cumulative return (no discount)
#   - Three-layer grouping: obs anchor → belief anchor → path propagation
#
# Experiment matrix (run with different EXP_ID/EXP_NAME):
#   E1: GraPO only (no belief reward)     → validate path propagation alone
#   E2: GraPO + belief reward curriculum  → full GraPO system
#   E3: HiBO (rebel_hibo) baseline        → direct comparison
#   E4: Soft anchor ablation              → obs-only vs belief soft anchor
# =============================================================================

set -e

# ======================== Basic Parameters ========================
NUM_GPUS=${NUM_GPUS:-4}
EPOCHS=${EPOCHS:-100}
SEED=${SEED:-42}

# ======================== Experiment ID ========================
EXP_ID=${EXP_ID:?"ERROR: EXP_ID required (e.g. G1-G4)"}
EXP_NAME=${EXP_NAME:?"ERROR: EXP_NAME required"}

# ======================== GraPO Mode ========================
# ADV_ESTIMATOR: 'grapo'        → full GraPO (3-layer)
#                'rebel_hibo'   → HiBO baseline (comparison)
#                'gigpo'        → GiGPO baseline
ADV_ESTIMATOR=${ADV_ESTIMATOR:-grapo}

# For grapo: env_type MUST be 'webshop' (uses raw cumulative return)
GRAPO_ENV_TYPE=${GRAPO_ENV_TYPE:-webshop}
GRAPO_STEP_ADVANTAGE_W=${GRAPO_STEP_ADVANTAGE_W:-0.5}
GRAPO_MIN_ANCHOR_SIZE=${GRAPO_MIN_ANCHOR_SIZE:-2}
GRAPO_SUMMARIZE=${GRAPO_SUMMARIZE:-false}

# ======================== Optional Config ========================
USE_REBEL_PROMPT=${USE_REBEL_PROMPT:-true}
USE_TRAINING_TRICKS=${USE_TRAINING_TRICKS:-true}
USE_BELIEF_REWARD=${USE_BELIEF_REWARD:-false}    # E1: false, E2: true
USE_RESULT_REWARD=${USE_RESULT_REWARD:-true}
USE_BELIEF_DECAY=${USE_BELIEF_DECAY:-false}
USE_ADAPTIVE_DECAY=${USE_ADAPTIVE_DECAY:-false}
USE_DIFFERENTIAL_DECAY=${USE_DIFFERENTIAL_DECAY:-true}
DECAY_METHOD=${DECAY_METHOD:-adaptive}
ROLLOUT_N=${ROLLOUT_N:-16}
SAVE_TRAJECTORIES=${SAVE_TRAJECTORIES:-true}
SAVE_FREQ=${SAVE_FREQ:-50}        # set to 9999 to effectively disable checkpoint saving

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_ATTENTION_BACKEND=XFORMERS
export SWANLAB_MODE=cloud

if ss -tlnp 2>/dev/null | grep -q ":7890"; then
    export http_proxy=http://127.0.0.1:7890
    export https_proxy=http://127.0.0.1:7890
    export HTTP_PROXY=http://127.0.0.1:7890
    export HTTPS_PROXY=http://127.0.0.1:7890
    export no_proxy=127.0.0.1,localhost
    export NO_PROXY=127.0.0.1,localhost
    echo "[Proxy] Clash detected on :7890, proxy enabled"
else
    echo "[Proxy] WARNING: Clash not running on :7890. SwanLab logging may fail."
fi

# ======================== Model Path ========================
FALLBACK_BASE_MODEL="/root/testttt/RLVMR/code/base_models/Qwen2.5-1.5B-Instruct"
find_sft_model() {
    local path
    path=$(find /root/testttt/RLVMR/code/checkpoints/cold_start/webshop/rebel_full_*/global_step_* \
           -maxdepth 0 -type d 2>/dev/null | sort -V | tail -1)
    [ -n "$path" ] && { echo "$path"; return; }
    path=$(find /root/testttt/RLVMR/code/checkpoints/cold_start/webshop/*/global_step_* \
           -maxdepth 0 -type d 2>/dev/null | sort -V | tail -1)
    [ -n "$path" ] && { echo "$path"; return; }
    echo "${FALLBACK_BASE_MODEL}"
}
MODEL_PATH=${MODEL_PATH:-$(find_sft_model)}

if [ "$MODEL_PATH" = "$FALLBACK_BASE_MODEL" ]; then
    echo "WARNING: No WebShop SFT checkpoint found! Using base model (cold-start SFT recommended)."
    sleep 3
fi

# ======================== Output Path ========================
RESULTS_BASE="/fs-computility-new/UPDZ03_chengjun/huangsijie.p/rebel_results/v12_grapo_webshop"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
FULL_EXP_NAME="${EXP_ID}_${EXP_NAME}_seed${SEED}"
RESULTS_DIR="${RESULTS_BASE}/${FULL_EXP_NAME}_${TIMESTAMP}"
mkdir -p "${RESULTS_DIR}/checkpoints"

# ======================== Training Tricks ========================
if [ "$USE_TRAINING_TRICKS" = "true" ]; then
    CLIP_RATIO_LOW=0.2;  CLIP_RATIO_HIGH=0.28
    ENTROPY_COEFF=0.001
    ENTROPY_PROTECTION_ENABLE=True;  ENTROPY_PROTECTION_METHOD=clip_cov
    CLIP_COV_LB=0.0;   CLIP_COV_UB=0.3
    INVALID_ACTION_PENALTY=True;     INVALID_ACTION_PENALTY_COEF=0.1
else
    CLIP_RATIO_LOW=0.2;  CLIP_RATIO_HIGH=0.2
    ENTROPY_COEFF=0.001
    ENTROPY_PROTECTION_ENABLE=False; ENTROPY_PROTECTION_METHOD=clip_cov
    CLIP_COV_LB=0.0;   CLIP_COV_UB=0.3
    INVALID_ACTION_PENALTY=False;    INVALID_ACTION_PENALTY_COEF=0.0
fi

# ======================== Belief Decay ========================
DECAY_WARMUP_EPOCHS=3;    DECAY_START_EPOCH=5
DECAY_END_EPOCH=40;       DECAY_MIN_WEIGHT=0.05
DECAY_TARGET_SR=0.90;     DECAY_ALPHA=2.0
PROGRESS_DECAY_RATE=0.7;  CONSISTENCY_DECAY_RATE=1.0
EXPLORATION_DECAY_RATE=2.0

if [ "$USE_DIFFERENTIAL_DECAY" = "false" ]; then
    PROGRESS_DECAY_RATE=1.0; CONSISTENCY_DECAY_RATE=1.0; EXPLORATION_DECAY_RATE=1.0
fi

# ======================== Print Config ========================
echo "==================================================================="
echo "  GraPO V12 WebShop: ${EXP_ID} - ${EXP_NAME}"
echo "==================================================================="
echo "Config:"
echo "  ADV estimator:   ${ADV_ESTIMATOR}"
echo "  GraPO env type:  ${GRAPO_ENV_TYPE}  (raw cumulative return)"
echo "  Step adv weight: ${GRAPO_STEP_ADVANTAGE_W}"
echo "  Min anchor size: ${GRAPO_MIN_ANCHOR_SIZE}"
echo "  Belief reward:   ${USE_BELIEF_REWARD}"
echo "  Adaptive decay:  ${USE_ADAPTIVE_DECAY}  (${DECAY_METHOD})"
echo "  Seed:            ${SEED}"
echo "  GPUs:            ${NUM_GPUS}"
echo "  Model:           ${MODEL_PATH}"
echo "  Results:         ${RESULTS_DIR}"
echo "==================================================================="

cd /root/testttt/RLVMR/code

# ======================== Prepare Data ========================
_EXPECTED_VAL_ROWS=500
_TEST_PARQUET="$HOME/data/verl-agent/text/test.parquet"
_NEED_REGEN=false
if [ ! -f "$HOME/data/verl-agent/text/train.parquet" ]; then
    _NEED_REGEN=true
elif [ -f "$_TEST_PARQUET" ]; then
    _ACTUAL_ROWS=$(python3 -c "import pandas as pd; print(len(pd.read_parquet('$_TEST_PARQUET')))" 2>/dev/null || echo 0)
    [ "$_ACTUAL_ROWS" != "$_EXPECTED_VAL_ROWS" ] && _NEED_REGEN=true
else
    _NEED_REGEN=true
fi
if [ "$_NEED_REGEN" = "true" ]; then
    python3 -m examples.data_preprocess.prepare \
        --mode 'text' --train_data_size 16 \
        --val_data_size ${_EXPECTED_VAL_ROWS} 2>/dev/null || true
fi

# ======================== Build Training Command ========================
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-2048}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-6144}

BASE_ARGS=(
    "algorithm.adv_estimator=${ADV_ESTIMATOR}"
    "data.train_files=$HOME/data/verl-agent/text/train.parquet"
    "data.val_files=$HOME/data/verl-agent/text/test.parquet"
    "data.train_batch_size=16"
    "data.val_batch_size=500"
    "data.max_prompt_length=${MAX_PROMPT_LENGTH}"
    "data.max_response_length=${MAX_RESPONSE_LENGTH}"
    "data.filter_overlong_prompts=True"
    "data.truncation=error"
    "data.return_raw_chat=True"
    "actor_rollout_ref.model.path=${MODEL_PATH}"
    "actor_rollout_ref.actor.optim.lr=1e-6"
    "actor_rollout_ref.actor.clip_ratio=0.2"
    "actor_rollout_ref.actor.clip_ratio_low=${CLIP_RATIO_LOW}"
    "actor_rollout_ref.actor.clip_ratio_high=${CLIP_RATIO_HIGH}"
    "actor_rollout_ref.actor.entropy_coeff=${ENTROPY_COEFF}"
    "actor_rollout_ref.actor.ppo_epochs=1"
    "actor_rollout_ref.actor.use_kl_loss=True"
    "actor_rollout_ref.actor.kl_loss_coef=0.01"
    "actor_rollout_ref.actor.kl_loss_type=low_var_kl"
    "actor_rollout_ref.model.use_remove_padding=True"
    "actor_rollout_ref.actor.ppo_mini_batch_size=128"
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4"
    "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=20480"
    "actor_rollout_ref.model.enable_gradient_checkpointing=True"
    "actor_rollout_ref.actor.fsdp_config.param_offload=False"
    "actor_rollout_ref.actor.fsdp_config.optimizer_offload=False"
    "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8"
    "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=20480"
    "actor_rollout_ref.rollout.tensor_model_parallel_size=1"
    "actor_rollout_ref.rollout.name=vllm"
    "actor_rollout_ref.rollout.gpu_memory_utilization=0.8"
    "actor_rollout_ref.rollout.enable_chunked_prefill=False"
    "actor_rollout_ref.rollout.enforce_eager=False"
    "actor_rollout_ref.rollout.free_cache_engine=False"
    "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8"
    "actor_rollout_ref.ref.fsdp_config.param_offload=True"
    "algorithm.use_kl_in_reward=False"
    "algorithm.kl_penalty=kl"
    "algorithm.kl_ctrl.type=fixed"
    "algorithm.kl_ctrl.kl_coef=0.001"
    "algorithm.gamma=1.0"
    "env.env_name=Webshop"
    "env.seed=${SEED}"
    "env.max_steps=15"
    "env.rollout.n=${ROLLOUT_N}"
    "trainer.critic_warmup=0"
    "trainer.logger=['console','swanlab']"
    "trainer.project_name=GraPO_V12_WebShop"
    "trainer.experiment_name=${FULL_EXP_NAME}"
    "trainer.n_gpus_per_node=${NUM_GPUS}"
    "trainer.nnodes=1"
    "trainer.save_freq=${SAVE_FREQ}"
    "trainer.test_freq=5"
    "trainer.total_epochs=${EPOCHS}"
    "trainer.default_local_dir=${RESULTS_DIR}/checkpoints"
    "trainer.val_before_train=True"
    "+trainer.save_trajectories=${SAVE_TRAJECTORIES}"
    "+trainer.trajectory_save_dir=/root/testttt/RLVMR/Trajectory-Tracer/trajectories/${TIMESTAMP}_${FULL_EXP_NAME}"
)

# Invalid action penalty
if [ "$INVALID_ACTION_PENALTY" = "True" ]; then
    BASE_ARGS+=(
        "actor_rollout_ref.actor.use_invalid_action_penalty=True"
        "actor_rollout_ref.actor.invalid_action_penalty_coef=${INVALID_ACTION_PENALTY_COEF}"
    )
fi

# ======================== GraPO-specific parameters ========================
if [ "$ADV_ESTIMATOR" = "grapo" ]; then
    BASE_ARGS+=(
        # Override gamma=1.0 from BASE_ARGS: discounting gives buy-step ~1.3x
        # more credit than search steps, matching GiGPO temporal structure.
        # Without this, CumR[t]=r_T for all t → zero temporal gradient.
        "algorithm.gamma=0.95"
        "+algorithm.grapo.env_type=${GRAPO_ENV_TYPE}"
        "+algorithm.grapo.step_advantage_w=${GRAPO_STEP_ADVANTAGE_W}"
        "+algorithm.grapo.min_anchor_group_size=${GRAPO_MIN_ANCHOR_SIZE}"
        "+algorithm.grapo.mode=mean_norm"
        "+algorithm.grapo.summarize=${GRAPO_SUMMARIZE}"
    )
fi

# ======================== Belief prompt + reward (shared with GraPO and HiBO) ========================
if [ "$USE_REBEL_PROMPT" = "true" ]; then
    BASE_ARGS+=(
        "algorithm.rebel.enable=True"
        "algorithm.rebel.belief_granularity=adaptive"
        "algorithm.rebel.step_advantage_w=${GRAPO_STEP_ADVANTAGE_W}"
        "algorithm.rebel.mode=mean_norm"
        "algorithm.rebel.task_aware_grouping=false"
        "algorithm.rebel.per_task_normalization=false"
        "algorithm.rebel.conditional_norm=true"
        "algorithm.rebel.min_samples_for_norm=10"
        "algorithm.rebel.min_std_for_norm=0.2"
        "algorithm.rebel.min_samples_ratio=0.0"
        "algorithm.rebel.entropy_protection.enable=${ENTROPY_PROTECTION_ENABLE}"
        "algorithm.rebel.entropy_protection.method=${ENTROPY_PROTECTION_METHOD}"
        "algorithm.rebel.entropy_protection.clip_cov_lb=${CLIP_COV_LB}"
        "algorithm.rebel.entropy_protection.clip_cov_ub=${CLIP_COV_UB}"
        "+algorithm.rebel.min_obs_group_size=2"
        "+algorithm.rebel.use_belief_reward=${USE_BELIEF_REWARD}"
        "+algorithm.rebel.use_result_reward=${USE_RESULT_REWARD}"
        "+algorithm.rebel.use_task_weighting=false"
        "+algorithm.rebel.belief_reward_decay.enable=${USE_BELIEF_DECAY}"
        "+algorithm.rebel.belief_reward_decay.method=${DECAY_METHOD}"
        "+algorithm.rebel.belief_reward_decay.warmup_epochs=${DECAY_WARMUP_EPOCHS}"
        "+algorithm.rebel.belief_reward_decay.decay_start_epoch=${DECAY_START_EPOCH}"
        "+algorithm.rebel.belief_reward_decay.decay_end_epoch=${DECAY_END_EPOCH}"
        "+algorithm.rebel.belief_reward_decay.min_weight=${DECAY_MIN_WEIGHT}"
        "+algorithm.rebel.belief_reward_decay.adaptive=${USE_ADAPTIVE_DECAY}"
        "+algorithm.rebel.belief_reward_decay.target_sr=${DECAY_TARGET_SR}"
        "+algorithm.rebel.belief_reward_decay.alpha=${DECAY_ALPHA}"
        "+algorithm.rebel.belief_reward_decay.progress_decay_rate=${PROGRESS_DECAY_RATE}"
        "+algorithm.rebel.belief_reward_decay.consistency_decay_rate=${CONSISTENCY_DECAY_RATE}"
        "+algorithm.rebel.belief_reward_decay.exploration_decay_rate=${EXPLORATION_DECAY_RATE}"
    )
else
    BASE_ARGS+=("algorithm.rebel.enable=False")
fi

# GiGPO / HiBO specific parameters (for comparison baselines)
if [ "$ADV_ESTIMATOR" = "gigpo" ] || [ "$ADV_ESTIMATOR" = "rebel_hibo" ]; then
    BASE_ARGS+=(
        "algorithm.gamma=0.95"
        "algorithm.gigpo.step_advantage_w=0.5"
        "algorithm.gigpo.mode=mean_norm"
    )
fi

# ======================== Execute Training ========================
echo ""
echo "Starting GraPO V12 WebShop training..."
echo ""

python3 -m verl.trainer.main_ppo \
    "${BASE_ARGS[@]}" \
    2>&1 | tee "${RESULTS_DIR}/training.log"

echo ""
echo "==================================================================="
echo "  GraPO V12 WebShop complete: ${EXP_ID} - ${EXP_NAME} (seed=${SEED})"
echo "==================================================================="
echo "Results: ${RESULTS_DIR}"
