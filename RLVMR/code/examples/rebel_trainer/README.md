# ReBel Trainer Examples

This directory contains example training scripts for the ReBel (Reward Belief) algorithm.

## Quick Start

### 1. Train ReBel on ALFWorld

```bash
# Basic usage with default settings
bash examples/rebel_trainer/run_alfworld.sh

# Specify vLLM engine explicitly
bash examples/rebel_trainer/run_alfworld.sh vllm

# Override specific parameters
bash examples/rebel_trainer/run_alfworld.sh vllm \
    algorithm.rebel.belief_granularity='medium' \
    env.rollout.n=128
```

### 2. Prerequisites

Before running training, you need:

1. **Cold-start model**: Train or download a cold-start model first
   ```bash
   # Example using SFT
   bash examples/sft/cold_start/run_alfworld_qwen2.5-1.5b.sh
   ```

2. **Update model path**: In `run_alfworld.sh`, replace:
   ```bash
   actor_rollout_ref.model.path=COLD_START_MODEL_PATH
   ```
   with your actual model path, e.g.:
   ```bash
   actor_rollout_ref.model.path=./checkpoints/cold_start/alfworld/sft_qwen2.5-1.5b
   ```

3. **Data preparation**: The script automatically runs data preprocessing with:
   - Training samples: 16
   - Validation samples: 128

## Configuration

### Key ReBel Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `algorithm.rebel.enable` | True | Enable ReBel algorithm |
| `algorithm.rebel.belief_granularity` | 'subgoal' | Grouping granularity: 'subgoal', 'medium', or 'fine' |
| `algorithm.rebel.step_advantage_w` | 1.0 | Weight for step-level advantages |
| `algorithm.rebel.mode` | 'mean_norm' | Normalization: 'mean_norm' or 'mean_std_norm' |
| `env.rollout.n` | 64 | Group size (ReBel benefits from larger sizes) |
| `env.alfworld.use_rebel` | True | Enable ReBel prompt and belief tracking |

### Recommended Settings

**For stable training:**
```bash
algorithm.rebel.belief_granularity='subgoal'
algorithm.rebel.mode='mean_norm'
env.rollout.n=64
```

**For finer-grained grouping (requires larger batches):**
```bash
algorithm.rebel.belief_granularity='medium'
algorithm.rebel.mode='mean_norm'
env.rollout.n=128
```

**For maximum granularity (experimental):**
```bash
algorithm.rebel.belief_granularity='fine'
algorithm.rebel.mode='mean_std_norm'
env.rollout.n=256
```

## Monitoring

Training metrics are logged to WandB under project `ReBel`:

**Key metrics to watch:**
- `rebel_num_groups`: Number of belief groups (should be 10-100)
- `rebel_mean_group_size`: Average group size (should be 5-50)
- `rebel_stats/intrinsic_reward/mean`: Intrinsic reward statistics
- `success_rate`: Task success rate

**Good training indicators:**
- Belief groups are well-distributed (not too many, not too few)
- Success rate steadily increases
- Intrinsic rewards show meaningful variance across groups

## Comparison with Other Methods

### ReBel vs BDRS
```bash
# ReBel (belief-based grouping)
bash examples/rebel_trainer/run_alfworld.sh

# BDRS (tag-based grouping)
bash examples/bdrs_trainer/run_alfworld.sh
```

**Expected differences:**
- ReBel should show better generalization with similar success rates
- ReBel groups are semantic (belief-based) vs manual (tag-based)
- ReBel intrinsic rewards are pre-computed in env, BDRS computes differentially

### ReBel vs GiGPO
```bash
# ReBel (belief-based grouping)
bash examples/rebel_trainer/run_alfworld.sh

# GiGPO (observation-based grouping)
bash examples/gigpo_trainer/run_alfworld.sh
```

**Expected differences:**
- ReBel has more stable groups (beliefs are more generalizable than raw observations)
- ReBel should converge faster due to better similarity matching
- GiGPO may struggle with group sizes (observations rarely match exactly)

## Troubleshooting

### Issue: No belief groups formed
**Check:**
1. `env.alfworld.use_rebel=True` is set
2. Model is generating belief states in the expected format
3. Check logs for belief parsing errors

### Issue: All groups have size 1
**Solution:**
- Increase `env.rollout.n` to at least 64
- Use coarser granularity: `algorithm.rebel.belief_granularity='subgoal'`
- Check if belief states are too diverse (may indicate parsing issues)

### Issue: Training is unstable
**Solution:**
- Use `algorithm.rebel.mode='mean_norm'` instead of `mean_std_norm`
- Reduce `algorithm.rebel.step_advantage_w` to 0.5
- Increase group size: `env.rollout.n=128`

### Issue: Low success rate
**Check:**
1. Cold-start model quality (should have >30% success rate)
2. Intrinsic rewards are being computed correctly
3. Belief states are being parsed correctly
4. Compare with baseline methods (GRPO, BDRS) to isolate issue

## Advanced Usage

### Multi-node Training

```bash
# 2 nodes, 8 GPUs each
bash examples/rebel_trainer/run_alfworld.sh vllm \
    trainer.nnodes=2 \
    trainer.n_gpus_per_node=8
```

### Custom Intrinsic Rewards

Modify the reward calculator in:
```
agent_system/environments/env_package/alfworld/belief_tracker.py
```

Look for `RebelRewardCalculator` class and adjust:
- `alpha`: Consistency reward weight (default: 0.3)
- `beta`: Progress reward weight (default: 0.5)
- `gamma`: Exploration reward weight (default: 0.2)
- `delta`: Format reward weight (default: 0.1)

### Different Generalization Levels

ALFWorld supports different generalization levels:

```bash
# Level 0: Train tasks (easiest)
bash examples/rebel_trainer/run_alfworld.sh vllm \
    env.alfworld.generalization_level=0

# Level 1: Unseen object combinations
bash examples/rebel_trainer/run_alfworld.sh vllm \
    env.alfworld.generalization_level=1

# Level 2: Unseen task types (hardest)
bash examples/rebel_trainer/run_alfworld.sh vllm \
    env.alfworld.generalization_level=2
```

## Files

- `run_alfworld.sh`: Main training script for ReBel on ALFWorld
- `README.md`: This file

## Related Documentation

- [ReBel Configuration Guide](../../REBEL_CONFIG_GUIDE.md): Comprehensive config documentation
- [Core Algorithm](../../rebel/core_rebel.py): ReBel implementation
- [Belief Tracking](../../agent_system/environments/env_package/alfworld/belief_tracker.py): Belief state parsing and reward calculation

## Citation

If you use ReBel in your research, please cite:
```
[Citation will be added after publication]
```
