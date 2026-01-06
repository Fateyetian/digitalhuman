#!/usr/bin/env python3
"""
V4 Experiments Analysis Script
Generate charts and tables for ReBel V4 experiments comparison
"""

import matplotlib.pyplot as plt
import numpy as np
import os

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['figure.figsize'] = (10, 6)

output_dir = os.path.dirname(os.path.abspath(__file__))

# ============================================================================
# Data from experiments
# ============================================================================

# Validation success rates over training steps
steps = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
exp3_steps = [0, 5, 10, 15, 20, 25]  # Exp3 only ran to step 25

# Exp1: task_status (belief_granularity="task_status")
exp1_success_rate = [0.000, 0.000, 0.078, 0.375, 0.469, 0.609, 0.625, 0.539, 0.727, 0.742, 0.734]

# Exp2: state_aware (belief_granularity="state_aware")
exp2_success_rate = [0.000, 0.000, 0.164, 0.383, 0.422, 0.586, 0.656, 0.648, 0.695, 0.758, 0.719]

# Exp3: subgoal + task_aware (belief_granularity="subgoal", task_aware=True)
exp3_success_rate = [0.000, 0.000, 0.141, 0.422, 0.492, 0.531]

# Per-task success rates at final checkpoint
tasks = ['pick_and\n_place', 'pick_two_obj\n_and_place', 'pick_cool\n_then_place',
         'pick_clean\n_then_place', 'pick_heat\n_then_place', 'look_at_obj\n_in_light']

# Exp1 final (step 50)
exp1_task_rates = [0.903, 0.786, 0.667, 0.810, 0.800, 0.083]

# Exp2 final (step 50)
exp2_task_rates = [0.935, 0.679, 0.619, 0.857, 0.733, 0.167]

# Exp3 at step 25
exp3_task_rates = [0.733, 0.500, 0.444, 0.609, 0.357, 0.333]

# Single sample ratio over training
exp1_single_ratio = [0.182, 0.118, 0.172, 0.119, 0.212, 0.210, 0.182, 0.184, 0.229, 0.200, 0.273]
exp2_single_ratio = [0.777, 0.763, 0.800, 0.779, 0.737, 0.743, 0.755, 0.724, 0.749, 0.727, 0.727]
exp3_single_ratio = [0.789, 0.786, 0.804, 0.816, 0.784, 0.784]  # Only to step 25

# ============================================================================
# Figure 1: Overall Success Rate Comparison
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 7))

ax.plot(steps, [r*100 for r in exp1_success_rate], 'o-', linewidth=2.5, markersize=8,
        label='Exp1: task_status', color='#2ecc71')
ax.plot(steps, [r*100 for r in exp2_success_rate], 's-', linewidth=2.5, markersize=8,
        label='Exp2: state_aware', color='#3498db')
ax.plot(exp3_steps, [r*100 for r in exp3_success_rate], '^--', linewidth=2.5, markersize=8,
        label='Exp3: subgoal+task_aware (partial)', color='#e74c3c')

ax.set_xlabel('Training Steps (Epochs)')
ax.set_ylabel('Validation Success Rate (%)')
ax.set_title('V4 Experiments: Overall Success Rate Comparison')
ax.legend(loc='lower right', fontsize=12)
ax.set_xlim(-2, 52)
ax.set_ylim(-5, 85)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'fig1_success_rate_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()

# ============================================================================
# Figure 2: Per-Task Success Rate Comparison (Bar Chart)
# ============================================================================
fig, ax = plt.subplots(figsize=(14, 8))

x = np.arange(len(tasks))
width = 0.25

bars1 = ax.bar(x - width, [r*100 for r in exp1_task_rates], width, label='Exp1: task_status (step 50)', color='#2ecc71', alpha=0.8)
bars2 = ax.bar(x, [r*100 for r in exp2_task_rates], width, label='Exp2: state_aware (step 50)', color='#3498db', alpha=0.8)
bars3 = ax.bar(x + width, [r*100 for r in exp3_task_rates], width, label='Exp3: subgoal+task_aware (step 25)', color='#e74c3c', alpha=0.8)

ax.set_xlabel('Task Type')
ax.set_ylabel('Success Rate (%)')
ax.set_title('V4 Experiments: Per-Task Success Rate Comparison')
ax.set_xticks(x)
ax.set_xticklabels(tasks, fontsize=11)
ax.legend(loc='upper right', fontsize=11)
ax.set_ylim(0, 105)

# Add value labels
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

# Highlight the problematic task
ax.axvspan(5.5, 6.5, alpha=0.2, color='red')
ax.annotate('Low Success Rate\nIssue', xy=(5.7, 50), fontsize=11, color='red', weight='bold')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'fig2_per_task_success_rate.png'), dpi=150, bbox_inches='tight')
plt.close()

# ============================================================================
# Figure 3: Single Sample Ratio Comparison
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 7))

ax.plot(steps, [r*100 for r in exp1_single_ratio], 'o-', linewidth=2.5, markersize=8,
        label='Exp1: task_status', color='#2ecc71')
ax.plot(steps, [r*100 for r in exp2_single_ratio], 's-', linewidth=2.5, markersize=8,
        label='Exp2: state_aware', color='#3498db')
ax.plot(exp3_steps, [r*100 for r in exp3_single_ratio], '^--', linewidth=2.5, markersize=8,
        label='Exp3: subgoal+task_aware (partial)', color='#e74c3c')

ax.set_xlabel('Training Steps (Epochs)')
ax.set_ylabel('Single Sample Group Ratio (%)')
ax.set_title('V4 Experiments: Single Sample Group Ratio Comparison\n(Lower is better - indicates more effective grouping)')
ax.legend(loc='upper right', fontsize=12)
ax.set_xlim(-2, 52)
ax.set_ylim(0, 90)
ax.grid(True, alpha=0.3)

# Add annotation
ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
ax.annotate('50% threshold', xy=(45, 52), fontsize=10, color='gray')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'fig3_single_sample_ratio.png'), dpi=150, bbox_inches='tight')
plt.close()

# ============================================================================
# Figure 4: look_at_obj_in_light Success Rate Over Time
# ============================================================================
# Extract look_at_obj_in_light success rates over training
exp1_look_at = [0.000, 0.000, 0.000, 0.267, 0.000, 0.333, 0.286, 0.500, 0.385, 0.273, 0.083]
exp2_look_at = [0.000, 0.000, 0.250, 0.200, 0.053, 0.250, 0.143, 0.333, 0.154, 0.273, 0.167]
exp3_look_at = [0.000, 0.000, 0.125, 0.200, 0.421, 0.333]

fig, ax = plt.subplots(figsize=(12, 7))

ax.plot(steps, [r*100 for r in exp1_look_at], 'o-', linewidth=2.5, markersize=8,
        label='Exp1: task_status', color='#2ecc71')
ax.plot(steps, [r*100 for r in exp2_look_at], 's-', linewidth=2.5, markersize=8,
        label='Exp2: state_aware', color='#3498db')
ax.plot(exp3_steps, [r*100 for r in exp3_look_at], '^--', linewidth=2.5, markersize=8,
        label='Exp3: subgoal+task_aware (partial)', color='#e74c3c')

ax.set_xlabel('Training Steps (Epochs)')
ax.set_ylabel('Success Rate (%)')
ax.set_title('look_at_obj_in_light Task Success Rate Over Training\n(Problematic Task Analysis)')
ax.legend(loc='upper right', fontsize=12)
ax.set_xlim(-2, 52)
ax.set_ylim(-5, 60)
ax.grid(True, alpha=0.3)

# Add problem zone annotation
ax.fill_between([35, 52], [0, 0], [60, 60], alpha=0.1, color='red')
ax.annotate('Performance Drop\nZone', xy=(43, 45), fontsize=11, color='red', weight='bold', ha='center')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'fig4_look_at_obj_analysis.png'), dpi=150, bbox_inches='tight')
plt.close()

# ============================================================================
# Generate Summary Table (Markdown)
# ============================================================================
table_md = """
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
"""

with open(os.path.join(output_dir, 'summary_table.md'), 'w') as f:
    f.write(table_md)

print("Analysis charts generated successfully!")
print(f"Output directory: {output_dir}")
print("Generated files:")
print("  - fig1_success_rate_comparison.png")
print("  - fig2_per_task_success_rate.png")
print("  - fig3_single_sample_ratio.png")
print("  - fig4_look_at_obj_analysis.png")
print("  - summary_table.md")
