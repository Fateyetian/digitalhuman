# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RLVMR (Reinforcement Learning with Verifiable Meta-Reasoning Rewards) is a framework for training robust long-horizon LLM agents. It provides fine-grained meta-reasoning rewards that encourage agents to learn "how to think" rather than relying solely on outcome feedback. The framework is built on top of veRL (Volcano Engine Reinforcement Learning) and supports multiple agent environments including ALFWorld and ScienceWorld.

## Core Architecture

### Three-Phase Training Pipeline

1. **Cold Start Data Preparation**: Generate expert demonstration trajectories using strong LLMs
2. **Cold Start SFT**: Supervised fine-tuning on demonstration data
3. **RL Training**: PPO-based training with meta-reasoning rewards (RLVMR or BDRS)

### Key Components

- **`code/rlvmr/core_rlvmr.py`**: Core RLVMR reward computation logic
  - `process_trajectory_rlvmr_rewards()`: Assigns meta-reasoning rewards based on tags (`<planning>`, `<explore>`, `<reflection>`)
  - `compute_rlvmr_outcome_advantage()`: Combines episode-level and step-level rewards

- **`code/bdrs/`**: BDRS (Belief-Driven Reward Shaping) framework implementation
  - `belief_state.py`: Belief state tracking
  - `core_bdrs.py`: BDRS reward computation (similar interface to RLVMR)

- **`code/agent_system/environments/`**: Environment management
  - `env_manager.py`: Multi-threaded environment coordination
  - `prompts/`: Environment-specific prompt templates (alfworld.py, sciworld.py, webshop.py)
  - `env_package/`: Wrapped gym environments

- **`code/verl/`**: veRL infrastructure
  - `trainer/main_ppo.py`: Main PPO training entry point
  - `trainer/fsdp_sft_trainer.py`: FSDP-based SFT trainer
  - `workers/`: Distributed training workers (Actor, Critic, Rollout, Ref)

## Development Commands

### Environment Setup

Each environment requires a separate conda environment:

```bash
# ALFWorld
conda create -n rlvmr-alfworld -y
conda activate rlvmr-alfworld
pip install gymnasium==0.29.1 stable-baselines3==2.6.0 alfworld
alfworld-download -f

# ScienceWorld
conda create -n rlvmr-sciworld -y
conda activate rlvmr-sciworld
pip install scienceworld

# Install veRL and dependencies (in either environment)
cd code
pip install -e .
```

### Training Workflow

**1. Prepare Cold Start Data**

Replace `YOUR_API_KEY` and `YOUR_DATA_SOURCE` in the scripts:

```bash
python scripts/alfworld_prepare.py
python scripts/sciworld_prepare.py
```

**2. Cold Start SFT**

```bash
bash examples/sft/cold_start/run_alfworld_qwen2.5-7b.sh
bash examples/sft/cold_start/run_sciworld_qwen2.5-7b.sh
```

**3. RL Training**

Before running, set `actor_rollout_ref.model.path` in the script to your cold start checkpoint:

```bash
bash examples/rlvmr_trainer/run_alfworld.sh
bash examples/rlvmr_trainer/run_sciworld.sh
```

For BDRS training:
```bash
bash examples/bdrs_trainer/run_alfworld.sh
bash examples/bdrs_trainer/run_sciworld.sh
```

### Evaluation

```bash
bash examples/bdrs_trainer/eval_alfworld.sh
bash examples/bdrs_trainer/eval_sciworld.sh
```

### Data Preprocessing

```bash
# Prepare training data from various sources
python -m examples.data_preprocess.prepare --mode text --train_data_size 16 --val_data_size 128
```

## Configuration System

The project uses Hydra for configuration management. Config files are in `code/verl/trainer/config/`:
- `ppo_trainer.yaml`: PPO/RLVMR training config
- `sft_trainer.yaml`: Supervised fine-tuning config
- `generation.yaml`: Generation/inference config

Override parameters via command line:
```bash
python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=rlvmr \
    algorithm.rlvmr.enable=True \
    actor_rollout_ref.model.path=/path/to/model \
    env.env_name=alfworld/AlfredTWEnv
```

## Customization

### Adding Meta-Reasoning Reward Rules

Modify `process_trajectory_rlvmr_rewards()` in `code/rlvmr/core_rlvmr.py`:
- Current tags: `<planning>`, `<explore>`, `<reflection>`
- Rewards are assigned based on success and validity criteria
- Exploration checks for repeated actions, reflection checks for post-failure correction

### Adding New Environments

1. Create environment package in `code/agent_system/environments/env_package/` following gym interface
2. Add prompt templates in `code/agent_system/environments/prompts/`
3. Add environment manager in `code/agent_system/environments/env_manager.py` for multi-threading support
4. Implement `projection_f()` to map text actions to environment actions

### Model Support

The codebase supports various LLMs via veRL:
- Qwen 2.5 (1.5B, 7B)
- DeepSeek models
- Custom models via `actor_rollout_ref.model.path`

Inference backends:
- vLLM (default, set `ENGINE=vllm`)
- SGLang (set `ENGINE=sglang`, requires `pip install -r requirements_sglang.txt`)

## Code Organization

```
code/
├── agent_system/          # Agent environment orchestration
│   ├── environments/      # Gym environments and managers
│   ├── multi_turn_rollout/ # Trajectory collection
│   └── reward_manager/    # Reward computation
├── bdrs/                  # BDRS framework
├── examples/              # Training scripts and examples
│   ├── sft/cold_start/    # Cold start training scripts
│   ├── rlvmr_trainer/     # RLVMR training scripts
│   ├── bdrs_trainer/      # BDRS training scripts
│   └── data_preprocess/   # Data preparation scripts
├── rlvmr/                 # RLVMR core implementation
├── scripts/               # Utility scripts (data prep, formatting)
└── verl/                  # veRL RL infrastructure
    ├── trainer/           # Training loops and configs
    ├── workers/           # Distributed workers (Actor, Critic, etc.)
    └── utils/             # Utilities (tokenizers, datasets, etc.)
```

## Key Parameters

**RLVMR-specific:**
- `algorithm.adv_estimator=rlvmr`: Enable RLVMR advantage computation
- `algorithm.rlvmr.enable=True`: Enable RLVMR rewards
- `algorithm.rlvmr.step_advantage_w`: Weight for step-level rewards (default: 1.0)
- `algorithm.rlvmr.planning_reward`: Reward for planning tags
- `algorithm.rlvmr.exploration_reward`: Reward for exploration tags
- `algorithm.rlvmr.reflection_reward`: Reward for reflection tags

**Environment:**
- `env.env_name`: Environment identifier (e.g., `alfworld/AlfredTWEnv`)
- `env.max_steps`: Max steps per episode
- `env.rollout.n`: Number of parallel rollouts
- `env.alfworld.meta_think=True`: Enable meta-reasoning tags

**Training:**
- `actor_rollout_ref.actor.optim.lr`: Learning rate
- `actor_rollout_ref.actor.use_invalid_action_penalty`: Penalize invalid actions
- `actor_rollout_ref.rollout.name`: Inference engine (vllm/sglang)
- `trainer.total_epochs`: Number of training epochs
- `trainer.logger`: Logging backends (console, wandb)

## Important Notes

- **Cold Start Model Path**: Always update `actor_rollout_ref.model.path` in RL training scripts to point to your cold start checkpoint
- **API Keys**: Set API keys for data generation (used in prepare scripts)
- **GPU Memory**: Adjust `actor_rollout_ref.rollout.gpu_memory_utilization` based on available VRAM
- **Multi-Node**: Training scripts default to single-node; adjust `trainer.nnodes` and Ray cluster setup for multi-node
- **Data Paths**: Scripts assume data in `$HOME/data/verl-agent/` or `$HOME/data/[ENV_NAME]/`
