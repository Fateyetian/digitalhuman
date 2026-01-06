
# V4 Experiments Results Summary Table

## Table 1: Overall Performance Comparison

| Metric | Exp1 (task_status) | Exp2 (state_aware) | Exp3 (subgoal+task_aware) |
|--------|-------------------|-------------------|---------------------------|
| Final Success Rate | 73.4% (step 50) | 71.9% (step 50) | 53.1% (step 25, partial) |
| Best Success Rate | 74.2% (step 45) | 75.8% (step 45) | 53.1% (step 25) |
| Single Sample Ratio | 27.3% | 72.7% | 78.4% |
| Belief Granularity | task_status | state_aware | subgoal |
| Task-Aware Grouping | Yes | Yes | Yes |

## Table 2: Per-Task Success Rate at Final Checkpoint

| Task | Exp1 (step 50) | Exp2 (step 50) | Exp3 (step 25) |
|------|---------------|---------------|----------------|
| pick_and_place | **90.3%** | **93.5%** | 73.3% |
| pick_two_obj_and_place | **78.6%** | 67.9% | 50.0% |
| pick_cool_then_place_in_recep | **66.7%** | 61.9% | 44.4% |
| pick_clean_then_place_in_recep | 81.0% | **85.7%** | 60.9% |
| pick_heat_then_place_in_recep | **80.0%** | 73.3% | 35.7% |
| **look_at_obj_in_light** | <span style="color:red">**8.3%**</span> | <span style="color:red">**16.7%**</span> | 33.3% |

## Key Observations

### 1. State-Aware vs Task-Status Granularity
- Both achieve similar overall success rates (~72-74%)
- state_aware has higher single sample ratio (72.7% vs 27.3%), indicating finer-grained grouping
- task_status groups more samples together, potentially more efficient for RL training

### 2. look_at_obj_in_light Performance Issue
- **Critical Finding**: This task shows abnormally low success rates in later training
- Exp1 (step 50): Only 8.3% success rate
- Exp2 (step 50): Only 16.7% success rate
- Exp3 (step 25): 33.3% success rate (best, but incomplete training)

### 3. Task Conflict Improvement
- With state_aware grouping, task conflicts appear reduced
- However, look_at_obj_in_light has become a new bottleneck
