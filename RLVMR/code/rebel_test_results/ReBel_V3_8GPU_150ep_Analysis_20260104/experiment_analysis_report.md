# ReBel V3 Experiment Analysis Report
## 8 GPU × 150 Epochs Training on AlfWorld

**Date**: 2026-01-04
**Experiment**: rebel_v3_8gpu_150ep_20260103_142940
**Model**: Qwen-1.5B with ReBel RL Training

---

## 1. Executive Summary

### Key Findings

1. **Severe Overfitting After Epoch 30**: Training success rate continues to climb (reaching 93.8%), while validation success rate plateaus around 55-70% and even declines in later epochs.

2. **Task Heterogeneity Problem**: Clear performance gap between "Pure Transport" tasks and "State-Change" tasks on validation set:
   - Pure Transport (pick_and_place, pick_two): Val ~73% average
   - State-Change (heat, cool, clean): Val ~30% average
   - **Gap: 43 percentage points**

3. **Belief Canonicalization Blind Spot**: The current `canonicalize_belief` implementation using `granularity='subgoal'` completely ignores object state attributes, causing hash collisions for semantically different states.

4. **High Single-Sample Ratio**: ~63-70% of belief groups contain only one sample, limiting normalization effectiveness.

---

## 2. Experimental Configuration

```yaml
Algorithm: ReBel
  belief_granularity: "subgoal"
  step_advantage_w: 0.5
  mode: "mean_norm"
  task_aware_grouping: false
  per_task_normalization: false

Training:
  GPUs: 8 × A800-80GB
  Epochs: 150
  Batch Size: 16
  Learning Rate: 1e-6
  KL Loss Coef: 0.01

Environment: AlfWorld (generalization_level=0)
  max_steps: 30
  rollout.n: 16
  use_teacher_planner: true
  prompt_template_type: "explicit_task_type"
```

---

## 3. Training Dynamics Analysis

### 3.1 Train-Val Gap Evolution

| Epoch | Train SR | Val SR | Gap |
|-------|----------|--------|-----|
| 0     | 0.000    | 0.000  | +0.000 |
| 15    | 0.480    | 0.484  | -0.004 |
| **30**    | 0.637    | 0.438  | **+0.199** |
| 60    | 0.707    | **0.695**  | +0.012 |
| 90    | 0.883    | 0.562  | +0.321 |
| 120   | 0.859    | 0.695  | +0.164 |
| **140**   | **0.938**    | 0.500  | **+0.438** |

**Observation**:
- Best validation performance: **69.5%** at epoch 60/120
- Final epoch shows **43.8%** overfitting gap
- Validation performance oscillates significantly (43.8% - 69.5%)

### 3.2 Critical Overfitting Pattern

```
Training Phase 1 (Epoch 0-15): Learning Phase
  - Both train and val improve together
  - Minimal gap (~0%)

Training Phase 2 (Epoch 15-30): Divergence Begins
  - Training continues improving
  - Validation starts to plateau
  - Gap reaches ~20%

Training Phase 3 (Epoch 30-150): Overfitting
  - Training reaches 90%+
  - Validation oscillates 45-70%
  - Gap can exceed 40%
```

---

## 4. Task Heterogeneity Analysis

### 4.1 Task Category Breakdown

#### Pure Transport Tasks (No State Modification)
| Task | Peak Val | Final Val | Final Train | Gap |
|------|----------|-----------|-------------|-----|
| pick_and_place | 95.3% | 86.7% | 100% | +13.3% |
| pick_two_and_place | 93.8% | 60.0% | 100% | +40.0% |

#### State-Change Tasks (Require Object Modification)
| Task | Peak Val | Final Val | Final Train | Gap |
|------|----------|-----------|-------------|-----|
| pick_heat_then_place | 64.3% | 36.4% | ~0% | **-36.4%** |
| pick_cool_then_place | 42.3% | 4.5% | 82.8% | **+78.3%** |
| pick_clean_then_place | 81.8% | 48.3% | 98.8% | **+50.5%** |

### 4.2 Category Comparison (Final Epoch)

```
                         Train Avg    Val Avg    Gap
State-Change Tasks:       60.5%       29.7%     +30.8%
Pure Transport Tasks:    100.0%       73.4%     +26.6%
```

**Key Insight**: State-change tasks show more severe overfitting and lower validation performance.

---

## 5. Root Cause Analysis

### 5.1 Belief Canonicalization Blind Spot

**Current Implementation** (`core_rebel.py:29-53`):

```python
def canonicalize_belief(belief_state, granularity='subgoal'):
    if granularity == 'subgoal':
        # ONLY uses subgoal + status - IGNORES state_changes!
        canonical = {
            'subgoal': task_progress.get('updated_subgoal', ''),
            'status': task_progress.get('subgoal_status', '')
        }
```

**Problem Scenario**:

```
State 1: Agent holding RAW potato in front of microwave
  - subgoal: "heat the potato"
  - status: "in_progress"
  - state_changes: {"potato": "raw"}

State 2: Agent holding COOKED potato in front of microwave
  - subgoal: "heat the potato"
  - status: "in_progress"
  - state_changes: {"potato": "heated"}

RESULT: Both states hash to IDENTICAL belief_hash!
```

**Impact**:
- State-change tasks require distinguishing object states (raw vs cooked, dirty vs clean, warm vs cool)
- Current canonicalization treats these as identical, causing:
  - Wrong advantage normalization (grouping incompatible states)
  - Model cannot learn state-dependent strategies
  - Training memorizes specific trajectories rather than learning generalizable policies

### 5.2 Intrinsic Reward Weight Imbalance

**Current Weights** (`core_rebel.py:758`):
```python
weights = {
    'consistency': 0.3,
    'progress': 0.5,
    'exploration': 0.2,
}
```

**Issues**:
1. `progress_reward` dominates (50%) but doesn't capture state changes
2. `consistency_reward` (30%) only checks location correctness, not state correctness
3. No explicit reward for achieving correct object states

### 5.3 ReBel Grouping Statistics

| Metric | Value |
|--------|-------|
| Single-Sample Ratio | 63-70% |
| Mean Group Size | 2.1-3.2 |
| Number of Groups | 877-3570 |

**Issue**: High single-sample ratio means most groups can't benefit from normalization - they default to 0 advantage.

---

## 6. Visualization Summary

### Figure 1: Train vs Validation Gap
![Train Val Gap](fig1_train_val_gap.png)

Shows the severe overfitting after epoch 30 with gap reaching 40%+.

### Figure 2: Task-Specific Validation
![Task Specific](fig2_task_specific_val.png)

Shows that state-change tasks (heat, cool, clean) consistently underperform.

### Figure 3: Task Category Comparison
![Task Category](fig3_task_category_comparison.png)

Compares state-change vs pure transport tasks - clear performance gap.

### Figure 4: ReBel Grouping Stats
![ReBel Stats](fig4_rebel_grouping_stats.png)

Shows high single-sample ratio limiting normalization effectiveness.

### Figure 5: Overfitting Gap by Task
![Overfitting Gap](fig5_overfitting_gap_by_task.png)

pick_cool and pick_clean show the largest overfitting gaps (>70%).

---

## 7. Improvement Proposals

### 7.1 Fix Belief Canonicalization (CRITICAL)

**Proposal**: Create `granularity='state_aware'` mode that includes object states:

```python
elif granularity == 'state_aware':
    task_progress = belief_state.get('task_progress_update', {}) or {}
    world_model = belief_state.get('world_model_update', {}) or {}

    # CRITICAL: Include state_changes
    state_changes = world_model.get('state_changes', {})
    if not isinstance(state_changes, dict):
        state_changes = {}

    # Normalize state descriptors
    normalized_states = {}
    for obj, state in state_changes.items():
        state_lower = str(state).lower()
        # Map to canonical state categories
        if any(x in state_lower for x in ['heated', 'hot', 'warm', 'cooked']):
            normalized_states[obj] = 'heated'
        elif any(x in state_lower for x in ['cooled', 'cold', 'cool', 'chilled']):
            normalized_states[obj] = 'cooled'
        elif any(x in state_lower for x in ['cleaned', 'clean', 'washed']):
            normalized_states[obj] = 'cleaned'
        else:
            normalized_states[obj] = state_lower

    canonical = {
        'subgoal': str(task_progress.get('updated_subgoal', '')).lower().strip(),
        'status': str(task_progress.get('subgoal_status', '')).lower().strip(),
        'object_states': sorted(normalized_states.items())
    }
```

**Config Change**:
```yaml
algorithm.rebel.belief_granularity: "state_aware"
```

### 7.2 Add State-Change Reward Component

**Proposal**: Add explicit `state_change_reward` to intrinsic rewards:

```python
def state_change_reward(belief, prev_belief, task_type):
    """Reward for achieving required state changes"""
    if task_type not in ['pick_heat', 'pick_cool', 'pick_clean']:
        return 0.0

    world_model = belief.get('world_model_update', {}) or {}
    state_changes = world_model.get('state_changes', {}) or {}

    reward = 0.0
    for obj, state in state_changes.items():
        state_lower = str(state).lower()

        # Check if state matches task requirement
        if task_type == 'pick_heat' and 'heated' in state_lower:
            reward += 0.3
        elif task_type == 'pick_cool' and 'cooled' in state_lower:
            reward += 0.3
        elif task_type == 'pick_clean' and 'cleaned' in state_lower:
            reward += 0.3

    return min(reward, 0.5)
```

**Updated Weights**:
```python
weights = {
    'consistency': 0.2,
    'progress': 0.3,
    'exploration': 0.2,
    'state_change': 0.3,  # NEW
}
```

### 7.3 Enable Task-Aware Grouping

**Proposal**: Use task-aware grouping to prevent cross-task interference:

```yaml
algorithm.rebel.task_aware_grouping: true
algorithm.rebel.per_task_normalization: true
```

This ensures:
- pick_heat trajectories are normalized only within pick_heat
- pick_cool trajectories are normalized only within pick_cool
- Prevents dominant tasks (pick_and_place) from overwhelming others

### 7.4 Early Stopping / Validation-Based Checkpointing

**Current**: Save every 50 epochs
**Proposal**: Implement validation-based checkpointing:

```python
if val_success_rate > best_val_success_rate:
    best_val_success_rate = val_success_rate
    save_checkpoint("best_val_model")
    patience_counter = 0
else:
    patience_counter += 1
    if patience_counter >= patience_limit:  # e.g., 20 epochs
        # Consider early stopping or LR reduction
```

### 7.5 Reduce Overfitting Measures

1. **Increase KL penalty**: `kl_loss_coef: 0.01 → 0.05`
2. **Reduce learning rate after epoch 30**: Implement LR scheduler
3. **Add dropout in policy network** (if applicable)
4. **Increase validation frequency**: `test_freq: 5 → 3`

---

## 8. Recommended Next Experiment Configuration

```yaml
algorithm:
  adv_estimator: rebel
  rebel:
    enable: True
    belief_granularity: "state_aware"  # CHANGED
    step_advantage_w: 0.3               # REDUCED
    mode: "mean_norm"
    task_aware_grouping: true           # ENABLED
    per_task_normalization: true        # ENABLED

# Intrinsic reward weights (in code)
intrinsic_weights:
  consistency: 0.2
  progress: 0.3
  exploration: 0.2
  state_change: 0.3   # NEW

actor_rollout_ref:
  actor:
    kl_loss_coef: 0.03  # INCREASED
    lr: 5e-7            # REDUCED

trainer:
  test_freq: 3          # MORE FREQUENT
  save_freq: 20         # SAVE BEST BASED ON VAL
```

---

## 9. Conclusion

The ReBel V3 experiment reveals a critical flaw in the belief canonicalization strategy that causes semantic hash collisions for state-change tasks. Combined with the lack of task-aware normalization, this leads to:

1. **Severe overfitting** (43% train-val gap)
2. **Poor generalization on state-change tasks** (30% val vs 73% for transport tasks)
3. **Ineffective belief-based grouping** (70% single-sample groups)

The proposed improvements focus on:
1. **State-aware belief hashing** (most critical)
2. **Task-aware advantage normalization**
3. **Explicit state-change rewards**
4. **Overfitting mitigation measures**

Implementing these changes should significantly improve validation performance, especially on state-change tasks.

---

*Report generated by automated analysis pipeline*
