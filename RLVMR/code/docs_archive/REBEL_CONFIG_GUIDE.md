# ReBel (Reward Belief) Configuration Guide

This guide explains how to configure and use the ReBel algorithm in your training runs.

## Overview

ReBel is an improvement over GRPO and GiGPO that uses **belief-based grouping** instead of observation-based or tag-based grouping for advantage estimation. This enables better generalization by grouping trajectories based on their semantic belief similarity.

## Key Innovation

**Standard Methods:**
- **GRPO**: Groups all steps together (too coarse)
- **GiGPO**: Groups by observation history (too fine, hard to match)
- **RLVMR/BDRS**: Groups by manual tags like `<planning>`, `<explore>` (limited, binary)

**ReBel:** Groups by semantic belief states - trajectories with similar beliefs are grouped together for more meaningful step-level advantage normalization.

## Configuration

Add the following to your training config YAML file under the `algorithm` section:

```yaml
algorithm:
  # Set advantage estimator to ReBel
  adv_estimator: rebel

  # ReBel-specific configuration
  rebel:
    enable: true

    # Belief canonicalization granularity
    # - 'subgoal': Group by subgoal only (recommended for stability)
    # - 'medium': Group by subgoal + world knowledge
    # - 'fine': Group by entire belief state (most fine-grained)
    belief_granularity: 'subgoal'

    # Step-level advantage weight (how much to weight intrinsic rewards)
    step_advantage_w: 1.0

    # Normalization mode
    # - 'mean_norm': Only subtract mean (recommended)
    # - 'mean_std_norm': Subtract mean and divide by std
    mode: 'mean_norm'
```

## Environment Configuration

Enable ReBel in the environment settings:

```yaml
env:
  env_name: alfworld/AlfredTWEnv
  max_steps: 30
  rollout:
    n: 64
  alfworld:
    use_rebel: true  # Enable ReBel prompt and belief tracking
    meta_think: true
    action_only: false
```

## Full Example Configuration

Here's a complete example for training with ReBel on ALFWorld:

```yaml
data:
  train_files: ~/data/alfworld/train.parquet
  val_files: ~/data/alfworld/test.parquet
  prompt_key: prompt
  max_prompt_length: 2048
  max_response_length: 512
  train_batch_size: 1024

actor_rollout_ref:
  model:
    path: ~/models/Qwen2.5-1.5B-Instruct
  actor:
    ppo_mini_batch_size: 256
    ppo_epochs: 1
    grad_clip: 1.0
    clip_ratio: 0.2
    entropy_coeff: 0.001
  rollout:
    temperature: 1.0
    response_length: 512
    n: 64

algorithm:
  adv_estimator: rebel
  gamma: 1.0
  lam: 1.0

  rebel:
    enable: true
    belief_granularity: 'subgoal'
    step_advantage_w: 1.0
    mode: 'mean_norm'

env:
  env_name: alfworld/AlfredTWEnv
  max_steps: 30
  rollout:
    n: 64
  alfworld:
    use_rebel: true
    meta_think: true
    action_only: false

trainer:
  project_name: ReBel_Training
  experiment_name: rebel_alfworld_qwen1.5b
  total_epochs: 100
  save_freq: 10
  logger:
    - console
    - wandb
```

## Understanding Belief Granularity

### Subgoal (Recommended)
Groups trajectories by their current subgoal and status. This is the most stable approach.

**Example:** All steps working on "find tomato" are grouped together, regardless of which objects were found or locations visited.

**Use when:** You want stable training with meaningful grouping.

### Medium
Groups by subgoal + world knowledge (e.g., number of found objects).

**Example:** Steps working on "find tomato" with 2 objects found vs 3 objects found are in different groups.

**Use when:** You want finer-grained grouping but still maintain reasonable group sizes.

### Fine
Groups by the entire belief state including world model, task progress, and exploration map.

**Example:** Two steps are only grouped if they have identical beliefs across all components.

**Use when:** You have large batch sizes and want maximum granularity. May lead to small groups and high variance.

## Monitoring Training

When training with ReBel, the following metrics are logged to WandB:

- `rebel_num_groups`: Number of belief groups formed
- `rebel_mean_group_size`: Average size of each belief group
- `rebel_stats/intrinsic_reward/mean`: Mean intrinsic reward
- `rebel_stats/intrinsic_reward/std`: Std of intrinsic reward
- `rebel_stats/n_steps`: Total number of steps processed

**Good indicators:**
- `rebel_num_groups` should be 10-100 (not too many, not too few)
- `rebel_mean_group_size` should be 5-50 (enough samples per group)
- If you see too many groups (>200), consider using coarser granularity
- If you see too few groups (<5), consider using finer granularity

## Comparison with BDRS

Both ReBel and BDRS use belief states, but differently:

| Aspect | BDRS | ReBel |
|--------|------|-------|
| Grouping | Manual tags (`<planning>`, etc.) | Semantic belief similarity |
| Intrinsic Reward | Differential belief consistency | Already computed in env |
| Advantage | Episode + Tag-based step | Episode + Belief-based step |
| Generalization | Limited by tag categories | Better - semantic grouping |

You can enable both simultaneously, but it's recommended to use one at a time for clearer attribution.

## Troubleshooting

### Issue: No belief_states in batch
**Cause:** Environment not properly configured to output belief states.
**Solution:** Set `env.alfworld.use_rebel: true` in config.

### Issue: All groups have size 1
**Cause:** Granularity is too fine or batch size is too small.
**Solution:** Use `belief_granularity: 'subgoal'` or increase `rollout.n`.

### Issue: Only one group
**Cause:** All belief states are identical or granularity is too coarse.
**Solution:** Check that your prompt is generating diverse belief states. Verify belief tracking in env_manager.

### Issue: High variance in advantages
**Cause:** Group sizes are too small or normalization mode is too aggressive.
**Solution:** Use `mode: 'mean_norm'` instead of `mean_std_norm`, or use coarser granularity.

## Advanced Usage

### Custom Intrinsic Rewards

If you want to customize the intrinsic reward calculation, modify the `RebelRewardCalculator` in:
```
agent_system/environments/env_package/alfworld/belief_tracker.py
```

The intrinsic reward is computed as:
```python
intrinsic_reward = (
    alpha * consistency_reward +
    beta * progress_reward +
    gamma * exploration_reward +
    delta * format_reward
)
```

Default weights: `alpha=0.3, beta=0.5, gamma=0.2, delta=0.1`

### Combining with RLVMR

ReBel and RLVMR can be used together:
```yaml
algorithm:
  adv_estimator: rebel  # Use ReBel for advantage
  rlvmr:
    enable: true  # Still compute RLVMR tags for analysis
```

This allows you to compare tag-based vs belief-based grouping in the same run.

## Citation

If you use ReBel in your research, please cite:
```
[Citation will be added after publication]
```

## Related Files

- Core algorithm: `rebel/core_rebel.py`
- Belief tracking: `agent_system/environments/env_package/alfworld/belief_tracker.py`
- Prompt template: `agent_system/environments/env_package/alfworld/alfworld_rebel_prompt.py`
- Environment manager: `agent_system/environments/env_manager.py`
- Trainer integration: `verl/trainer/ppo/ray_trainer.py`
