# ReBel V5 Experiment Analysis Report

**Experiment Date**: 2026-01-06
**Analysis Date**: 2026-01-07
**Status**: Completed (50 epochs)

---

## 1. Executive Summary

### 1.1 Key Findings

| Metric | V4 Exp1 (task_status) | V4 Exp2 (state_aware) | V5 (adaptive) | Change |
|--------|----------------------|----------------------|---------------|--------|
| Final Success Rate | **73.4%** | 71.9% | 71.1% | -2.3% |
| Best Success Rate | 74.2% (step 45) | **75.8%** (step 45) | 71.1% (step 45) | -4.7% |
| look_at_obj_in_light | 8.3% | 16.7% | **8.3%** | 0% |
| Convergence Speed | Fast | Fast | **Slower** | - |

### 1.2 Critical Issue: V5 Performance Regression

**V5 did NOT improve look_at_obj_in_light performance and showed overall regression:**
- Final success rate dropped from 73.4% (V4 Exp1) to 71.1%
- look_at_obj_in_light remains at 8.3% (same as V4 Exp1)
- Convergence is slower: V4 reached 60% at step 25, V5 reached 52.3%

---

## 2. Detailed Performance Analysis

### 2.1 Validation Success Rate Over Training

| Step | V4 Exp1 | V4 Exp2 | V5 | V5 look_at |
|------|---------|---------|-----|------------|
| 0 | 0.0% | 0.0% | 0.0% | 0.0% |
| 5 | 0.0% | 0.0% | 0.0% | 0.0% |
| 10 | 7.8% | 16.4% | 1.6% | 12.5% |
| 15 | 37.5% | 38.3% | 13.3% | 0.0% |
| 20 | 46.9% | 42.2% | 25.0% | 15.8% |
| 25 | 60.9% | 58.6% | 52.3% | 16.7% |
| 30 | 62.5% | 65.6% | 53.9% | 14.3% |
| 35 | 53.9% | 64.8% | 56.2% | 16.7% |
| 40 | 72.7% | 69.5% | 59.4% | 15.4% |
| 45 | 74.2% | 75.8% | 71.1% | 18.2% |
| 50 | 73.4% | 71.9% | 71.1% | **8.3%** |

### 2.2 look_at_obj_in_light Trajectory Analysis

```
Training Step:  0   5   10  15  20  25  30  35  40  45  50
V4 Exp1:       0%  0%  0%  27% 0%  33% 29% 50% 39% 27% 8%
V4 Exp2:       0%  0%  25% 20% 5%  25% 14% 33% 15% 27% 17%
V5:            0%  0%  13% 0%  16% 17% 14% 17% 15% 18% 8%
```

**Observation**: All experiments show a performance drop at step 50 for look_at task.
This suggests **catastrophic forgetting** in late training stages.

---

## 3. Grouping Strategy Analysis

### 3.1 V5 Grouping Statistics

| Step Range | Num Groups | Mean Size | Single Sample % |
|------------|------------|-----------|-----------------|
| 1-10 | 124-192 | 32-62 | 15-32% |
| 41-50 | 102-125 | 32-47 | 7-15% |

### 3.2 Groups per Task Type (Average)

| Task Type | Avg Groups | Notes |
|-----------|------------|-------|
| pick_two_obj_and_place | 31.2 | - |
| pick_clean_then_place | 30.6 | - |
| pick_and_place | 29.0 | - |
| pick_cool_then_place | 25.0 | - |
| pick_heat_then_place | 22.2 | - |
| **look_at_obj_in_light** | **18.3** | **Fewest groups** |

### 3.3 Root Cause Analysis: Why V5 Failed

**Problem 1: Adaptive Granularity Still Not Optimal**

The `adaptive` granularity creates groups based on:
```
{is_complete, has_state_change, has_inventory, stage_type}
```

For look_at_obj_in_light:
- `has_state_change` is always `false` (no state changes in this task)
- `stage_type` mapping for "examine X in light" may be incorrect
- Results in fewer, less discriminative groups

**Problem 2: Conditional Normalization Not Triggering**

From logs, `adv_std_look_at_in_light = 1.000` consistently, indicating:
- per_task_normalization is still forcing std=1
- conditional_norm logic may not be working as expected
- The task has >10 samples, so it doesn't trigger global fallback

**Problem 3: Sample Imbalance**

| Task | Training Occurrences | % of Total |
|------|---------------------|------------|
| pick_two_obj_and_place | 49 | 18% |
| pick_clean_then_place | 49 | 18% |
| pick_and_place | 49 | 18% |
| pick_cool_then_place | 46 | 17% |
| pick_heat_then_place | 43 | 16% |
| **look_at_obj_in_light** | **35** | **13%** |

look_at has the fewest samples AND fewest groups, creating a double disadvantage.

---

## 4. Mathematical Analysis of Grouping Strategy

### 4.1 Current Grouping Formulation

Let $G_k = \{i : h(b_i) = k, \text{uid}_i = u, \tau_i = t\}$ be the belief group where:
- $h(b)$ = belief hash function
- $u$ = prompt uid
- $t$ = task type

The advantage normalization is:
$$\hat{A}_i = \frac{A_i - \mu_{G_k}}{\sigma_{G_k} + \epsilon}$$

### 4.2 Problem: Variance Collapse in Small Groups

When $|G_k|$ is small and samples are homogeneous:
- $\sigma_{G_k} \to 0$
- $\hat{A}_i \to \infty$ (numerically unstable)
- Forcing $\sigma = 1$ creates artificial noise

### 4.3 Optimal Grouping Criteria (Theoretical)

For effective contrastive learning, groups should satisfy:

1. **Sufficient Size**: $|G_k| \geq N_{min}$ (recommended: 5-20)
2. **Internal Variance**: $\sigma_{G_k} \geq \sigma_{min}$ (natural variance)
3. **Semantic Coherence**: Samples in same group should be at similar decision points
4. **Balanced Coverage**: Each task should have adequate group representation

### 4.4 Proposed Solution: Hierarchical Adaptive Grouping

```
Level 1 (Coarse): task_type
Level 2 (Medium): task_type × progress_stage (find/act/complete)
Level 3 (Fine): task_type × progress_stage × belief_hash

Selection Rule:
- If Level 3 group size >= 5: use Level 3
- Else if Level 2 group size >= 5: use Level 2
- Else: use Level 1 with conservative normalization
```

---

## 5. Detailed Diagnosis: Why look_at Fails

### 5.1 Task-Specific Challenges

The look_at_obj_in_light task requires:
1. Find the target object
2. Find a light source (lamp/desklamp)
3. Navigate to light source
4. Use light source (turn on)
5. Examine object under light

**Unique Challenge**: This task has NO state changes (`cleaned`, `heated`, `cooled`), so `has_state_change` is always `false`, reducing grouping granularity.

### 5.2 Belief State Analysis for look_at

Typical belief progression:
```
Step 1: {subgoal: "find X", has_inventory: false, has_state_change: false}
Step 5: {subgoal: "find lamp", has_inventory: true, has_state_change: false}
Step 8: {subgoal: "use lamp", has_inventory: true, has_state_change: false}
```

With adaptive granularity, steps 5 and 8 may be grouped together despite being at different stages because `has_state_change` is always false.

### 5.3 Hypothesis: Incorrect Stage Mapping

The `stage_type` mapping in V5:
```python
if any(x in subgoal for x in ['turn on', 'use', 'toggle', 'lamp', 'light']):
    stage_type = 'use'
```

Problem: "find lamp" contains "lamp" but should be `find`, not `use`.

---

## 6. Recommendations

### 6.1 Immediate Fixes (V6)

1. **Fix Stage Type Detection**:
```python
# Priority order matters - check 'find' before 'use'
if any(x in subgoal for x in ['find', 'look for', 'search', 'locate']):
    stage_type = 'find'
elif any(x in subgoal for x in ['turn on', 'use', 'toggle']):
    stage_type = 'use'
# ... etc
```

2. **Task-Specific Grouping for look_at**:
```python
if task_type == 'look_at_obj_in_light':
    # Use finer granularity for this task
    canonical = {
        'has_target_object': bool(inventory),
        'has_light_source': 'lamp' in str(found_objects).lower(),
        'subgoal_keyword': extract_keyword(subgoal)  # find/go/use/examine
    }
```

3. **Disable per_task_normalization for Small Tasks**:
```python
if task_sample_count < 50:  # Increased threshold
    use_global_normalization = True
```

### 6.2 Medium-Term Improvements

1. **Hierarchical Grouping with Fallback**
2. **Learned Belief Embeddings** instead of heuristic hashing
3. **Task-Specific Reward Weighting**

### 6.3 Data Augmentation

- Increase look_at_obj_in_light samples in training
- Current ratio: 13% → Target: 16-18%

---

## 7. Next Experiment Plan (V6)

### Configuration

```yaml
algorithm:
  rebel:
    belief_granularity: "task_specific"  # New: different granularity per task
    task_aware_grouping: true
    per_task_normalization: true
    conditional_norm: true
    min_samples_for_norm: 50  # Increased from 10
    min_std_for_norm: 0.2     # Increased from 0.1

    # Task-specific overrides
    task_configs:
      look_at_obj_in_light:
        use_global_norm: true
        belief_granularity: "detailed_subgoal"
```

### Expected Improvements

| Metric | V5 | V6 Target |
|--------|-----|-----------|
| Overall SR | 71.1% | 75%+ |
| look_at SR | 8.3% | 40%+ |
| Convergence (to 60%) | step 40 | step 25 |

---

## 8. Conclusion

V5's adaptive grouping and conditional normalization **failed to solve the look_at_obj_in_light problem** due to:

1. **Incorrect stage type detection** (keyword matching order)
2. **Insufficient granularity** for tasks without state changes
3. **Conditional normalization thresholds too low** (10 samples, 0.1 std)
4. **Sample imbalance** not addressed

The next iteration (V6) should implement **task-specific grouping strategies** with higher thresholds for conditional normalization.

---

*Report generated by Claude Code on 2026-01-07*
