# ReBel Algorithm Analysis: Challenges and Insights from ALFWorld Experiments

**Experiment ID:** rebel_search_20251231_024630_step_adv_0.5_val128_ep100_rerun
**Date:** 2026-01-01
**Configuration:** step_advantage_w=0.5, belief_granularity=subgoal, epochs=100, val_size=128

---

## Abstract

This report presents a comprehensive analysis of the ReBel (Reasoning with Belief Learning) algorithm's performance on the ALFWorld benchmark. Through detailed examination of Experiment 6, we identify three critical challenges limiting the algorithm's effectiveness: (1) reward plateau after epoch 30, (2) performance degradation after epoch 70, and (3) multi-task interference between the six ALFWorld task types. Our analysis reveals that these challenges stem from shared policy representation limitations, belief grouping granularity issues, and gradient interference in multi-task learning. We propose several solutions including task-aware belief grouping, adaptive regularization, and multi-head architectures.

---

## 1. Introduction

### 1.1 Background

The ReBel algorithm extends traditional policy gradient methods by incorporating belief-based state grouping for advantage estimation. In ALFWorld, an embodied AI benchmark with six distinct task types, this approach aims to leverage semantic similarity between states to improve credit assignment.

### 1.2 Experiment Summary

| Parameter | Value |
|-----------|-------|
| Model | Qwen2.5-1.5B-Instruct |
| Training Epochs | 100 |
| Validation Set Size | 128 |
| Step Advantage Weight | 0.5 |
| Belief Granularity | subgoal |
| Learning Rate | 1e-6 |
| KL Coefficient | 0.01 |

### 1.3 Key Results

| Metric | Value |
|--------|-------|
| **Peak Validation Success Rate** | **72.7%** (epoch 70) |
| Final Validation Success Rate | 58.6% (epoch 100) |
| Performance Decline from Peak | 19.4% |
| Reward Plateau Onset | ~epoch 30 |

---

## 2. Problem 1: Reward Plateau After Epoch 30

### 2.1 Phenomenon Description

As shown in **Figure 1(b)**, the mean episode reward exhibits rapid growth during the first 30 epochs, followed by a plateau phase where rewards fluctuate around a stable value without meaningful improvement.

![Overall Performance](fig1_overall_performance.png)
*Figure 1: (a) Training and validation success rates over epochs, with peak and decline markers. (b) Mean episode reward showing plateau after epoch 30.*

### 2.2 Quantitative Analysis

| Training Phase | Epoch Range | Mean Reward | Growth Rate |
|----------------|-------------|-------------|-------------|
| Rapid Learning | 0-30 | 4.82 | +0.15/epoch |
| Plateau | 30-100 | 8.21 | +0.002/epoch |

The growth rate decreases by approximately **98.7%** after epoch 30, indicating a near-complete stagnation of the reward signal.

### 2.3 Potential Causes

1. **Policy Entropy Collapse**
   - The policy converges to a local optimum with reduced action diversity
   - Limited exploration prevents discovery of higher-reward trajectories

2. **Advantage Estimation Saturation**
   - ReBel's belief-based grouping may create overly homogeneous groups after initial learning
   - Within-group variance decreases, reducing the discriminative power of step-level advantages

3. **KL Constraint Dominance**
   - The KL penalty (coef=0.01) increasingly constrains policy updates
   - As the policy diverges from the reference, the KL term dominates the gradient

4. **Task Interference** (detailed in Section 4)
   - Gradient conflicts between tasks create oscillatory updates that cancel out

### 2.4 Evidence from Training Dynamics

**Figure 5** shows additional training metrics that support this analysis:

![Training Dynamics](fig5_training_dynamics.png)
*Figure 5: Training dynamics showing (a) KL loss, (b) gradient norm, (c) valid action ratio, and (d) mean episode length over training.*

Key observations:
- KL loss shows oscillation rather than monotonic behavior
- Gradient norm exhibits high variance, suggesting conflicting gradient directions
- Valid action ratio plateaus, indicating the policy has learned basic action validity

---

## 3. Problem 2: Performance Decline After Epoch 70

### 3.1 Phenomenon Description

Despite achieving a peak validation success rate of **72.7%** at epoch 70, continued training results in performance degradation, with the final success rate dropping to **58.6%** (a 19.4% relative decline).

### 3.2 Analysis

| Metric | Value |
|--------|-------|
| Peak Success Rate | 72.7% |
| Peak Epoch | 70 |
| Final Success Rate | 58.6% |
| Absolute Decline | 14.1% |
| Relative Decline | 19.4% |

### 3.3 Potential Causes

1. **Overfitting to Training Distribution**
   - The model memorizes task-specific patterns that don't generalize
   - Gap between training (~90%) and validation (~70%) success rates supports this

2. **Catastrophic Forgetting**
   - Learning to solve new task instances interferes with previously learned solutions
   - The multi-task nature of ALFWorld exacerbates this problem

3. **Distribution Shift**
   - Policy updates push the model too far from the reference distribution
   - The KL penalty is insufficient to prevent harmful divergence

4. **Task-Specific Overfitting**
   - Some tasks reach near-perfect training performance while others lag
   - Continued optimization for lagging tasks degrades performance on mastered tasks

### 3.4 Implications

This finding suggests that **early stopping** at the validation peak would significantly improve final model performance. An automatic early stopping mechanism monitoring per-task validation metrics is recommended.

---

## 4. Problem 3: Multi-Task Interference

### 4.1 ALFWorld Task Types

ALFWorld comprises six distinct task types with different objectives and required skills:

| Task Type | Description | Complexity |
|-----------|-------------|------------|
| pick_and_place | Pick object, place at location | Low |
| pick_two_obj_and_place | Pick two objects, place both | Medium |
| pick_clean_then_place_in_recep | Clean object before placing | High |
| pick_heat_then_place_in_recep | Heat object before placing | High |
| pick_cool_then_place_in_recep | Cool object before placing | High |
| look_at_obj_in_light | Examine object under light | Different |

### 4.2 Per-Task Training Performance

**Figure 2** shows the training success rate for each task type over the 100 epochs:

![Per-Task Training](fig2_per_task_training.png)
*Figure 2: Per-task training success rates over epochs. Note the divergent trajectories and inverse relationships between certain task pairs.*

Key observations:
- **pick_and_place** achieves highest performance (~95%) but with high variance
- **look_at_obj_in_light** shows erratic behavior, often inversely correlated with other tasks
- Multi-step tasks (clean/heat/cool) show similar patterns but different peak timings
- **pick_two_obj_and_place** struggles throughout, rarely exceeding 60%

### 4.3 Per-Task Validation Performance

**Figure 3** shows validation success rates, revealing more pronounced task conflicts:

![Per-Task Validation](fig3_per_task_validation.png)
*Figure 3: Per-task validation success rates over epochs. The vertical line marks the overall performance peak.*

Critical observations:
- Tasks reach their peaks at different epochs (30-80 range)
- When one task improves, others often decline
- No epoch achieves uniformly high performance across all tasks

### 4.4 Task Conflict Correlation Analysis

**Figure 4** presents a correlation matrix of task improvement directions:

![Task Correlation](fig4_task_correlation.png)
*Figure 4: Correlation matrix of per-task success rate changes. Negative values (red) indicate conflicting tasks; positive values (blue) indicate cooperative tasks.*

#### Most Conflicting Task Pairs (Negative Correlation):

| Task Pair | Correlation | Interpretation |
|-----------|-------------|----------------|
| Pick-Clean-Place vs Look at Light | -0.31 | Strongly conflicting |
| Pick & Place vs Look at Light | -0.28 | Conflicting |
| Pick-Heat-Place vs Pick Two & Place | -0.22 | Moderate conflict |

#### Most Cooperative Task Pairs (Positive Correlation):

| Task Pair | Correlation | Interpretation |
|-----------|-------------|----------------|
| Pick-Heat-Place vs Pick-Cool-Place | +0.45 | Strong cooperation |
| Pick-Clean-Place vs Pick-Cool-Place | +0.38 | Moderate cooperation |
| Pick & Place vs Pick-Clean-Place | +0.32 | Moderate cooperation |

### 4.5 Conflict Analysis

The correlation analysis reveals a clear pattern:

1. **Similar tasks cooperate**: Tasks requiring heating, cooling, and cleaning share common sub-skills (object manipulation, appliance interaction) and improve together.

2. **Dissimilar tasks conflict**: "Look at Light" is fundamentally different (observation vs. manipulation) and competes with manipulation-focused tasks.

3. **No Pareto improvement**: After epoch ~50, improving any task tends to degrade another, suggesting the single-policy architecture has reached its representational capacity.

### 4.6 Per-Task Final Statistics

| Task | Initial | Final | Peak | Peak Epoch | Net Change |
|------|---------|-------|------|------------|------------|
| Pick & Place | 3.1% | 93.8% | 100.0% | 85 | +90.7% |
| Pick Two & Place | 0.0% | 43.8% | 68.8% | 65 | +43.8% |
| Pick-Clean-Place | 4.2% | 87.5% | 100.0% | 75 | +83.3% |
| Pick-Heat-Place | 0.0% | 81.2% | 93.8% | 70 | +81.2% |
| Pick-Cool-Place | 1.6% | 75.0% | 81.2% | 55 | +73.4% |
| Look at Light | 3.1% | 56.2% | 81.2% | 35 | +53.1% |

**Key insight**: While most tasks show substantial improvement, they peak at different times, and no single checkpoint optimizes all tasks simultaneously.

---

## 5. Root Cause Analysis

### 5.1 Architectural Limitations

**Shared Policy Representation**

The current architecture uses a single language model backbone with a shared output head for all task types. This creates:

1. **Gradient Interference**: Updates optimizing one task may degrade performance on others
2. **Representation Bottleneck**: Limited capacity must encode diverse task-specific behaviors
3. **Conflicting Action Distributions**: Different tasks require different action preferences in similar states

### 5.2 ReBel-Specific Issues

1. **Cross-Task Belief Mixing**
   - The current belief grouping does not consider task type
   - States from different tasks may be incorrectly grouped if they share similar subgoals
   - This corrupts the advantage estimation signal

2. **Granularity Limitations**
   - "Subgoal" level grouping may be too coarse for fine-grained credit assignment
   - States within the same subgoal may require different actions based on context

3. **Step Advantage Weight**
   - Fixed weight (0.5) may not be optimal across different training phases
   - Early training may benefit from higher episode-level signal
   - Late training may need finer step-level discrimination

### 5.3 Training Dynamics Issues

| Issue | Evidence | Impact |
|-------|----------|--------|
| Early Overfitting | Train/Val gap after epoch 30 | Reduced generalization |
| Gradient Conflict | Task correlation matrix | Optimization instability |
| Representation Saturation | Reward plateau | Limited capacity |
| Catastrophic Forgetting | Post-peak decline | Performance regression |

---

## 6. Proposed Solutions

### 6.1 Architecture Improvements

#### Option A: Task-Specific Output Heads
```
Shared Backbone → Task Classification → Task-Specific Head → Action
```
- Maintain shared representation learning
- Allow task-specific action distributions
- Modest parameter increase (~10%)

#### Option B: Mixture of Experts (MoE)
```
Input → Router → Expert 1 (manipulation tasks)
              → Expert 2 (observation tasks)
              → Expert 3 (multi-step tasks)
       → Combine → Output
```
- Dynamic routing based on task/state
- Better capacity utilization
- More complex training

#### Option C: Multi-Task Attention
```
Shared Encoder → Task Embedding → Cross-Attention → Task-Weighted Features → Output
```
- Soft task-specific modulation
- End-to-end trainable
- Interpretable attention weights

### 6.2 ReBel Algorithm Improvements

#### 6.2.1 Task-Aware Belief Grouping

```python
def group_beliefs_task_aware(beliefs, task_types):
    """Group beliefs only within the same task type"""
    groups = defaultdict(list)
    for belief, task in zip(beliefs, task_types):
        key = (task, canonicalize(belief))  # Include task in grouping key
        groups[key].append(belief)
    return groups
```

**Expected benefit**: Prevents cross-task contamination of advantage estimates.

#### 6.2.2 Adaptive Step Advantage Weight

```python
def adaptive_step_weight(epoch, val_success_rate):
    """Adjust weight based on training progress"""
    if epoch < 30:
        return 0.3  # More episode-level signal early
    elif val_success_rate > 0.6:
        return 0.7  # More step-level when performing well
    else:
        return 0.5  # Balanced default
```

**Expected benefit**: Better matches training signal to learning phase.

#### 6.2.3 Per-Task Advantage Normalization

```python
def normalize_advantages_per_task(advantages, task_types):
    """Normalize advantages separately for each task type"""
    normalized = torch.zeros_like(advantages)
    for task in unique(task_types):
        mask = (task_types == task)
        task_adv = advantages[mask]
        normalized[mask] = (task_adv - task_adv.mean()) / (task_adv.std() + 1e-8)
    return normalized
```

**Expected benefit**: Prevents dominant tasks from overwhelming gradient updates.

### 6.3 Training Strategy Improvements

#### 6.3.1 Gradient Surgery (PCGrad)

When task gradients conflict, project them to remove conflicting components:

```python
def pcgrad(task_gradients):
    """Project Conflicting Gradients"""
    for i, g_i in enumerate(task_gradients):
        for j, g_j in enumerate(task_gradients):
            if i != j:
                cos_sim = dot(g_i, g_j) / (norm(g_i) * norm(g_j))
                if cos_sim < 0:  # Conflicting
                    g_i = g_i - (dot(g_i, g_j) / (norm(g_j)**2)) * g_j
    return task_gradients
```

#### 6.3.2 Task-Balanced Sampling

Ensure each batch contains equal representation of all task types:

```python
def balanced_batch_sampler(dataset, batch_size, n_tasks=6):
    per_task = batch_size // n_tasks
    batch = []
    for task in task_types:
        task_samples = dataset.filter(task=task).sample(per_task)
        batch.extend(task_samples)
    return batch
```

#### 6.3.3 Per-Task Early Stopping

Monitor and save best checkpoints for each task independently:

```python
class PerTaskEarlyStopping:
    def __init__(self, tasks, patience=10):
        self.best_scores = {task: 0 for task in tasks}
        self.best_epochs = {task: 0 for task in tasks}

    def update(self, epoch, task_scores):
        for task, score in task_scores.items():
            if score > self.best_scores[task]:
                self.best_scores[task] = score
                self.best_epochs[task] = epoch
                save_checkpoint(f"best_{task}.pt")
```

### 6.4 Regularization Improvements

| Technique | Current | Proposed | Rationale |
|-----------|---------|----------|-----------|
| KL Coefficient | 0.01 | 0.05 | Prevent large policy shifts |
| Entropy Bonus | None | 0.01 | Maintain exploration |
| Weight Decay | None | 0.01 | Regularize representations |
| Gradient Clipping | None | 1.0 | Stabilize training |

---

## 7. Experimental Recommendations

### 7.1 Immediate Actions (High Priority)

| Action | Expected Impact | Effort |
|--------|-----------------|--------|
| Add early stopping at best validation | Prevent 19% decline | Low |
| Implement task-aware belief grouping | Reduce interference | Medium |
| Increase KL penalty to 0.05 | Stabilize training | Low |

### 7.2 Ablation Studies (Medium Priority)

1. **Belief Granularity**: Compare subgoal vs. medium vs. fine
2. **Step Advantage Weight**: Test {0.25, 0.5, 0.75, 1.0}
3. **Task-Specific Models**: Train separate model per task type
4. **Multi-Head Architecture**: Shared backbone + task heads

### 7.3 Key Metrics to Monitor

- Per-task success rate (not just aggregate)
- Policy entropy over training
- Gradient cosine similarity between tasks
- Train/validation success rate gap
- Per-task checkpoint timestamps

---

## 8. Conclusion

Our analysis of ReBel Experiment 6 reveals three interconnected challenges:

1. **Reward Plateau (Epoch 30+)**: The training signal diminishes due to policy entropy collapse, advantage saturation, and multi-task gradient interference.

2. **Performance Decline (Epoch 70+)**: Overfitting and catastrophic forgetting cause a 19.4% relative decline from peak performance.

3. **Multi-Task Conflict**: The six ALFWorld task types exhibit competitive dynamics, with a correlation analysis showing that "Look at Light" conflicts with manipulation tasks, while heating/cooling/cleaning tasks cooperate.

The fundamental issue is that a single policy network lacks sufficient capacity to simultaneously optimize all task types. We propose task-aware belief grouping, multi-head architectures, and gradient surgery as potential solutions.

**Recommended next step**: Implement task-aware belief grouping and early stopping, then re-run the experiment to validate improvement.

---

## Appendix A: Figure Reference

| Figure | Description | File |
|--------|-------------|------|
| Figure 1 | Overall success rate and reward curves | fig1_overall_performance.png |
| Figure 2 | Per-task training success rates | fig2_per_task_training.png |
| Figure 3 | Per-task validation success rates | fig3_per_task_validation.png |
| Figure 4 | Task correlation heatmap | fig4_task_correlation.png |
| Figure 5 | Training dynamics (KL, gradient, etc.) | fig5_training_dynamics.png |

## Appendix B: Data Source

- **SwanLab Local**: `/root/testttt/RLVMR/code/swanlog/run-20251231_024818-v025g0u2vlcma0375psrd/`
- **SwanLab Cloud**: https://swanlab.cn/@yetian/ReBel_HyperSearch/runs/v025g0u2vlcma0375psrd
- **Checkpoint**: `/fs-computility-new/UPDZ03_chengjun/huangsijie.p/rebel_results/hyperparam_search/rebel_search_20251231_024630/step_adv_0.5_val128_ep100_rerun/checkpoints/global_step_100/`

---

*Report generated: 2026-01-01*
*Analysis pipeline: ReBel Experiment Analysis v1.0*
