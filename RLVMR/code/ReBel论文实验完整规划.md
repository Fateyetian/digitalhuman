# ReBel 论文实验完整规划

> 本文档分为两部分：完整实验规划 + 预期结果与排版

---

# 第一部分：完整实验规划

## 1. 实验目标

验证ReBel的三个核心贡献：
1. **密集奖励**：信念质量奖励加速收敛、提高成功率
2. **O(1)上下文**：信念替代历史降低token消耗
3. **信念漂移缓解**：显式信念维护提高有效动作率

---

## 2. 实验环境

### 2.1 Benchmarks

| Benchmark | 类型 | 任务数 | 最大步数 | 特点 |
|-----------|------|--------|---------|------|
| **ALFWorld** | 具身环境 | 6类134个 | 50 | 长horizon、稀疏奖励 |
| **WebShop** | 网页购物 | 12k指令 | 15 | 复杂状态空间 |
| **SciWorld** | 科学实验 | 30类 | 40 | 多步推理 |

### 2.2 ALFWorld任务类型

| 任务类型 | 难度 | 平均步数 | 关键挑战 |
|----------|------|---------|----------|
| Pick | 简单 | 8-12 | 单目标定位 |
| Clean | 中等 | 12-18 | 状态变化追踪 |
| Heat | 中等 | 12-18 | 设备使用 |
| Cool | 困难 | 15-22 | 多步操作 |
| Pick2 | 困难 | 18-25 | 双目标协调 |
| PnP (Pick and Place) | 中等 | 10-15 | 位置记忆 |

### 2.3 泛化级别

| 级别 | 定义 | 示例 |
|------|------|------|
| L0 (Seen) | 训练中见过的任务 | put tomato in fridge |
| L1 (Novel Combo) | 新物体+见过的任务类型 | put apple in microwave |
| L2 (Novel Task) | 完全新的任务描述 | heat the egg and put on counter |

---

## 3. Baselines

### 3.1 Prompting Methods

| Method | 类型 | 特点 | 参考文献 |
|--------|------|------|----------|
| ReAct | Few-shot | 思考+行动交替 | Yao et al., 2023 |
| Reflexion | Self-reflection | 失败后反思 | Shinn et al., 2024 |

### 3.2 RL Methods

| Method | 优势估计 | 奖励类型 | 上下文 | 参考文献 |
|--------|---------|---------|--------|----------|
| PPO | GAE (Value Network) | 稀疏 | O(T) | Schulman et al., 2017 |
| RLOO | Leave-one-out | 稀疏 | O(T) | Ahmadian et al., 2024 |
| GRPO | Episode-level norm | 稀疏 | O(T) | Shao et al., 2024 |
| GiGPO | Anchor state grouping | 稀疏 | O(T) | [Our prior work] |
| RAGEN | Trajectory GRPO | 稀疏 | O(T) | Wang et al., 2025 |

### 3.3 对比设置

所有RL方法使用相同的：
- Base model: Qwen2.5-{1.5B, 3B, 7B}-Instruct
- Max steps: 50 (ALFWorld) / 15 (WebShop) / 40 (SciWorld)
- Training iterations: 200
- Group size: 8

---

## 4. 评估指标

### 4.1 主指标

| 指标 | 公式 | 说明 |
|------|------|------|
| **Success Rate (SR)** | $\frac{\text{成功轨迹数}}{\text{总轨迹数}}$ | 主要性能指标 |
| **Avg Steps** | $\frac{\sum \text{步数}}{\text{成功轨迹数}}$ | 效率指标 |
| **Avg Trajectory Tokens** | $\frac{\sum \text{tokens}}{\text{轨迹数}}$ | 上下文效率 |

### 4.2 信念漂移指标

| 指标 | 定义 | ReBel预期 |
|------|------|----------|
| **Valid Action Rate** | 有效动作数/总动作数 | ↑ 提高 |
| **Repeat Action Rate** | 重复动作数/总动作数 | ↓ 降低 |
| **Belief Accuracy** | 信念与真实状态的匹配度 | ↑ 提高 |

### 4.3 训练效率指标

| 指标 | 说明 | ReBel预期 |
|------|------|----------|
| **Convergence Steps** | 达到目标SR的迭代数 | ↓ 减少 |
| **Training Loss Variance** | 损失函数波动 | ↓ 减少 |
| **Sample Efficiency** | SR提升/样本数 | ↑ 提高 |

---

## 5. 实验设置

### 5.1 超参数配置

```yaml
# Common Settings
base_model: Qwen2.5-{1.5B, 3B, 7B}-Instruct
max_prompt_len: 2048 (ALFWorld) / 4096 (WebShop, SciWorld)
max_response_len: 512
learning_rate: 1e-6
batch_size: 128
group_size: 8
kl_coef: 0.01

# ReBel Specific
algorithm:
  adv_estimator: rebel
  rebel:
    enable: true
    belief_granularity: subgoal  # {subgoal, medium, fine}
    step_advantage_w: 1.0       # ω in paper
    mode: mean_norm             # {mean_norm, mean_std_norm}

# Reward Weights
rebel_reward_weights:
  consistency: 0.3   # w_1
  progress: 0.5      # w_2
  exploration: 0.2   # w_3
```

### 5.2 训练流程

```
Phase 1: Cold-Start SFT (10 epochs)
    ├─ 使用hindsight标注的专家轨迹
    ├─ 无admissible actions (强推理训练)
    └─ 输出: SFT checkpoint

Phase 2: ReBel RL Training (200 iterations)
    ├─ 从SFT checkpoint初始化
    ├─ 有admissible actions (降低探索难度)
    ├─ 密集信念奖励 + 稀疏轨迹奖励
    └─ 输出: Final checkpoint
```

### 5.3 统计配置

| 配置项 | 值 | 说明 |
|--------|---|------|
| 随机种子数 | 3 | seeds: 42, 123, 456 |
| 评估间隔 | 每10 iterations | 记录SR曲线 |
| 统计检验 | t-test / Wilcoxon | p < 0.05 |
| 置信区间 | 95% | 报告mean ± std |

---

## 6. 消融实验设计

### 6.1 核心组件消融

| Configuration | Belief Reward | Belief Context | 预期SR变化 |
|--------------|---------------|----------------|-----------|
| Full ReBel | ✓ | ✓ | Baseline |
| w/o Belief Reward | ✗ | ✓ | ↓ 8-12% |
| w/o Belief Context | ✓ | ✗ | ↓ 5-8% |
| w/o Both (=GRPO) | ✗ | ✗ | ↓ 15-20% |

### 6.2 奖励组件消融

| Configuration | Consistency | Progress | Exploration | 预期SR变化 |
|--------------|-------------|----------|-------------|-----------|
| Full | ✓ | ✓ | ✓ | Baseline |
| w/o Consistency | ✗ | ✓ | ✓ | ↓ 3-5% |
| w/o Progress | ✓ | ✗ | ✓ | ↓ 6-10% |
| w/o Exploration | ✓ | ✓ | ✗ | ↓ 2-4% |

### 6.3 超参数敏感性

#### ω (Step Advantage Weight)
| ω值 | 0.0 | 0.5 | 1.0 | 1.5 | 2.0 |
|-----|-----|-----|-----|-----|-----|
| 预期SR | 低 | 中 | 最优 | 略降 | 降低 |

#### Belief Granularity
| Granularity | 分组数 | 组大小 | 预期SR |
|-------------|-------|-------|--------|
| subgoal | 20-50 | 20-40 | 最优 |
| medium | 50-150 | 8-20 | 略低 |
| fine | 100-500 | 2-10 | 较低 |

---

## 7. 分析性实验

### 7.1 收敛速度分析

**实验设计**：
- 记录每10 iterations的SR
- 绘制训练曲线 (SR vs Iterations)
- 计算达到X% SR的步数

**预期结果**：
- ReBel达到目标SR速度比GRPO快2-3倍
- 训练曲线更平滑，方差更小

### 7.2 信念漂移分析

**实验设计**：
- 按episode步数分组统计指标
- 绘制指标随步数变化曲线

**指标**：
- Valid Action Rate vs Step Number
- Repeat Action Rate vs Step Number
- Belief Accuracy vs Step Number

**预期结果**：
- GRPO/GiGPO的指标随步数下降
- ReBel保持稳定

### 7.3 上下文效率分析

**实验设计**：
- 统计不同episode长度的token数
- 对比O(T) vs O(1)

**预期结果**：
- Baseline: tokens ≈ c₁ × T + c₀ (线性增长)
- ReBel: tokens ≈ c (恒定)

### 7.4 分组统计分析

**实验设计**：
- 统计belief group的数量和大小分布
- 与GiGPO和RLVMR对比

**预期结果**：
| Method | Num Groups | Mean Size | Median Size |
|--------|------------|-----------|-------------|
| GiGPO | 500-1000 | 1-2 | 1 |
| RLVMR | 3-4 | 300-400 | 350 |
| ReBel | 30-50 | 20-40 | 25 |

---

## 8. Case Study

### 8.1 成功案例分析

选择3-5个代表性成功轨迹：
- 展示完整的belief state演变
- 标注关键决策点
- 说明信念如何帮助决策

### 8.2 失败案例分析

分析失败模式：
- **类型1**: 信念维护错误 → 错误决策
- **类型2**: 探索不足 → 超时
- **类型3**: 动作解析失败 → 无效动作

### 8.3 信念质量可视化

- 信念状态与ground truth对比
- 信念漂移程度的热力图

---

## 9. 计算资源

### 9.1 硬件配置

| 资源 | 配置 |
|------|------|
| GPU | 8 × A100 80GB |
| CPU | 64 cores |
| Memory | 512GB |
| Storage | 2TB NVMe |

### 9.2 预估时间

| 实验 | GPU时间 | 说明 |
|------|---------|------|
| SFT Cold-start | ~4h | 每个模型 |
| RL Training (1 seed) | ~24h | 200 iterations |
| Full Experiments | ~500 GPU-hours | 所有baselines + 消融 |

### 9.3 开销分析

| 组件 | 时间占比 | 说明 |
|------|---------|------|
| Environment Step | 35% | 环境交互 |
| Model Forward | 40% | 生成动作 |
| Belief Parsing | 5% | ReBel额外开销 |
| Reward Computation | 5% | ReBel额外开销 |
| Policy Update | 15% | 梯度更新 |

---

## 10. 日志与监控

### 10.1 SwanLab/WandB配置

```python
# 训练指标
- train/loss
- train/policy_loss
- train/kl_divergence
- train/entropy

# 性能指标
- eval/success_rate
- eval/avg_steps
- eval/avg_tokens

# ReBel专属指标
- rebel/intrinsic_reward_mean
- rebel/intrinsic_reward_std
- rebel/consistency_reward
- rebel/progress_reward
- rebel/exploration_reward
- rebel/num_belief_groups
- rebel/mean_group_size
- rebel/valid_action_rate
- rebel/repeat_action_rate
```

### 10.2 Checkpoint保存

```
checkpoints/
├── sft/
│   └── qwen2.5-{size}-sft-epoch{N}/
├── rl/
│   ├── rebel-{size}-iter{N}/
│   ├── grpo-{size}-iter{N}/
│   └── gigpo-{size}-iter{N}/
└── best/
    └── rebel-{size}-best/
```

---

# 第二部分：预期结果与排版

## 1. 主实验结果

### Table 1: Main Results on ALFWorld and WebShop

**位置**: Section 5.2, 正文第一个表格

**LaTeX代码**: `figures_tables/table1_main_results.tex`

```latex
\begin{table*}[t]
\centering
\caption{Main results on ALFWorld and WebShop benchmarks. We report success rate (\%) for ALFWorld (6 subtasks) and WebShop. Best results are in \textbf{bold}, second best are \underline{underlined}. All RL methods use Qwen2.5-3B-Instruct as the base model.}
\label{tab:main_results}
\resizebox{\textwidth}{!}{
\begin{tabular}{llccccccc|c}
\toprule
\textbf{Type} & \textbf{Method} & \textbf{Pick} & \textbf{Clean} & \textbf{Heat} & \textbf{Cool} & \textbf{Pick2} & \textbf{PnP} & \textbf{Avg (ALF)} & \textbf{WebShop} \\
\midrule
\multirow{2}{*}{Prompting}
& ReAct & 45.2 & 38.7 & 41.3 & 32.5 & 28.4 & 42.1 & 38.0 & 42.3 \\
& Reflexion & 52.8 & 45.3 & 48.6 & 38.2 & 33.7 & 49.5 & 44.7 & 48.6 \\
\midrule
\multirow{5}{*}{RL Training}
& PPO & 58.3 & 52.1 & 55.4 & 44.7 & 38.2 & 56.8 & 50.9 & 51.2 \\
& RLOO & 61.2 & 54.8 & 57.9 & 47.3 & 41.5 & 59.2 & 53.7 & 53.8 \\
& GRPO & 64.5 & 58.2 & 61.3 & 51.8 & 45.7 & 62.4 & 57.3 & 56.4 \\
& GiGPO & \underline{68.7} & \underline{62.5} & \underline{65.8} & \underline{55.2} & \underline{49.3} & \underline{66.1} & \underline{61.3} & \underline{59.7} \\
& \textbf{ReBel (Ours)} & \textbf{74.2} & \textbf{68.9} & \textbf{71.5} & \textbf{62.8} & \textbf{56.4} & \textbf{72.3} & \textbf{67.7} & \textbf{65.2} \\
\midrule
\multicolumn{2}{l}{\textit{ReBel vs. GiGPO}} & \textcolor{ForestGreen}{+5.5} & \textcolor{ForestGreen}{+6.4} & \textcolor{ForestGreen}{+5.7} & \textcolor{ForestGreen}{+7.6} & \textcolor{ForestGreen}{+7.1} & \textcolor{ForestGreen}{+6.2} & \textcolor{ForestGreen}{+6.4} & \textcolor{ForestGreen}{+5.5} \\
\bottomrule
\end{tabular}
}
\end{table*}
```

**预期数据**（占位，待实验填充）:

| Method | Pick | Clean | Heat | Cool | Pick2 | PnP | Avg | WebShop |
|--------|------|-------|------|------|-------|-----|-----|---------|
| ReAct | 45.2 | 38.7 | 41.3 | 32.5 | 28.4 | 42.1 | 38.0 | 42.3 |
| Reflexion | 52.8 | 45.3 | 48.6 | 38.2 | 33.7 | 49.5 | 44.7 | 48.6 |
| PPO | 58.3 | 52.1 | 55.4 | 44.7 | 38.2 | 56.8 | 50.9 | 51.2 |
| RLOO | 61.2 | 54.8 | 57.9 | 47.3 | 41.5 | 59.2 | 53.7 | 53.8 |
| GRPO | 64.5 | 58.2 | 61.3 | 51.8 | 45.7 | 62.4 | 57.3 | 56.4 |
| GiGPO | 68.7 | 62.5 | 65.8 | 55.2 | 49.3 | 66.1 | 61.3 | 59.7 |
| **ReBel** | **74.2** | **68.9** | **71.5** | **62.8** | **56.4** | **72.3** | **67.7** | **65.2** |

---

### Table 2: Model Scale Analysis

**位置**: Section 5.2 或 Appendix

```latex
\begin{table}[t]
\centering
\caption{Performance across different model scales on ALFWorld.}
\label{tab:model_scale}
\begin{tabular}{lccc}
\toprule
\textbf{Method} & \textbf{1.5B} & \textbf{3B} & \textbf{7B} \\
\midrule
GRPO & 48.3 ± 1.2 & 57.3 ± 0.8 & 63.5 ± 0.9 \\
GiGPO & 52.7 ± 1.1 & 61.3 ± 0.7 & 68.2 ± 0.8 \\
\textbf{ReBel} & \textbf{58.4 ± 0.9} & \textbf{67.7 ± 0.6} & \textbf{74.8 ± 0.7} \\
\midrule
\textit{Δ vs. GiGPO} & +5.7 & +6.4 & +6.6 \\
\bottomrule
\end{tabular}
\end{table}
```

---

## 2. 训练曲线

### Figure 2: Training Curves

**位置**: Section 5.3

**文件**: `figures_tables/fig2_training_curves.pdf`

**图表描述**:
```
布局: 1行2列

左图: Success Rate vs Training Iterations
- X轴: Training Iterations (0-200)
- Y轴: Success Rate (%) (0-80)
- 曲线: GRPO (蓝), GiGPO (橙), ReBel (绿)
- 阴影: 95%置信区间
- 关键点标注: ReBel在iter=80达到GRPO在iter=200的性能

右图: Training Loss vs Iterations
- X轴: Training Iterations (0-200)
- Y轴: Policy Loss
- 曲线: 同上
- 观察: ReBel的loss曲线更平滑
```

**预期观察**:
- ReBel收敛速度是GRPO的2.5倍
- ReBel训练曲线方差更小
- GiGPO介于两者之间

**Matplotlib代码框架**:
```python
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Left: Success Rate
iterations = np.arange(0, 201, 10)
grpo_sr = [...]  # 实验数据
gigpo_sr = [...]
rebel_sr = [...]

axes[0].plot(iterations, grpo_sr, 'b-', label='GRPO')
axes[0].plot(iterations, gigpo_sr, 'orange', label='GiGPO')
axes[0].plot(iterations, rebel_sr, 'g-', label='ReBel')
axes[0].fill_between(iterations, rebel_sr_low, rebel_sr_high, alpha=0.2, color='green')
axes[0].set_xlabel('Training Iterations')
axes[0].set_ylabel('Success Rate (%)')
axes[0].legend()
axes[0].set_title('(a) Convergence Speed')

# Right: Training Loss
# Similar structure...

plt.tight_layout()
plt.savefig('figures_tables/fig2_training_curves.pdf', dpi=300)
```

---

## 3. 信念漂移分析

### Figure 3: Belief Drift Analysis

**位置**: Section 5.4

**文件**: `figures_tables/fig3_belief_drift.pdf`

**图表描述**:
```
布局: 1行3列

左图: Valid Action Rate vs Episode Step
- X轴: Episode Step (1-50)
- Y轴: Valid Action Rate (%) (50-100)
- 曲线: GRPO, GiGPO, ReBel
- 观察: GRPO/GiGPO下降，ReBel稳定

中图: Repeat Action Rate vs Episode Step
- X轴: Episode Step (1-50)
- Y轴: Repeat Action Rate (%) (0-30)
- 曲线: 同上
- 观察: GRPO/GiGPO上升，ReBel保持低位

右图: Belief Accuracy vs Episode Step (ReBel only)
- X轴: Episode Step
- Y轴: Belief Accuracy (%)
- 展示ReBel的信念准确度如何保持
```

**预期数据**:

| Step | GRPO Valid% | GiGPO Valid% | ReBel Valid% |
|------|-------------|--------------|--------------|
| 1-10 | 92.3 | 93.5 | 94.8 |
| 11-20 | 85.7 | 88.2 | 93.5 |
| 21-30 | 78.4 | 82.6 | 92.1 |
| 31-40 | 71.2 | 76.8 | 91.4 |
| 41-50 | 65.3 | 72.1 | 90.8 |

---

### Table 3: Belief Drift Indicators

**位置**: Section 5.4

```latex
\begin{table}[t]
\centering
\caption{Belief drift indicators on ALFWorld. Lower repeat rate and higher valid action rate indicate less drift.}
\label{tab:belief_drift}
\begin{tabular}{lccc}
\toprule
\textbf{Method} & \textbf{Valid Action ↑} & \textbf{Repeat Rate ↓} & \textbf{Drift Score ↓} \\
\midrule
GRPO & 76.4\% & 18.7\% & 0.42 \\
GiGPO & 82.1\% & 14.3\% & 0.32 \\
\textbf{ReBel} & \textbf{92.5\%} & \textbf{5.8\%} & \textbf{0.13} \\
\bottomrule
\end{tabular}
\end{table}
```

---

## 4. 上下文效率

### Figure 4: Context Efficiency

**位置**: Section 5.5

**文件**: `figures_tables/fig4_context_efficiency.pdf`

**图表描述**:
```
布局: 单图

Token Usage vs Episode Length
- X轴: Episode Length (steps)
- Y轴: Total Tokens
- 曲线1: Baseline methods (线性增长) y = 150*x + 500
- 曲线2: ReBel (恒定) y ≈ 1200
- 标注: "O(T)" 和 "O(1)" 复杂度
- 阴影区域显示token节省量
```

**预期数据**:

| Episode Length | Baseline Tokens | ReBel Tokens | Savings |
|---------------|-----------------|--------------|---------|
| 10 | 2,000 | 1,200 | 40% |
| 20 | 3,500 | 1,200 | 66% |
| 30 | 5,000 | 1,200 | 76% |
| 40 | 6,500 | 1,200 | 82% |
| 50 | 8,000 | 1,200 | 85% |

### Table 4: Context Efficiency Statistics

```latex
\begin{table}[t]
\centering
\caption{Context efficiency comparison. ReBel achieves O(1) context complexity.}
\label{tab:context_efficiency}
\begin{tabular}{lccc}
\toprule
\textbf{Method} & \textbf{Avg Tokens/Step} & \textbf{Avg Tokens/Traj} & \textbf{Complexity} \\
\midrule
GRPO & 180 ± 25 & 4,320 ± 580 & O(T) \\
GiGPO & 175 ± 22 & 4,200 ± 510 & O(T) \\
\textbf{ReBel} & \textbf{85 ± 8} & \textbf{1,445 ± 120} & \textbf{O(1)} \\
\midrule
\textit{Reduction} & 52.8\% & 66.5\% & - \\
\bottomrule
\end{tabular}
\end{table}
```

---

## 5. 消融实验

### Table 5: Ablation Study

**位置**: Section 5.6

```latex
\begin{table}[t]
\centering
\caption{Ablation study on ReBel components. All experiments use Qwen2.5-3B on ALFWorld.}
\label{tab:ablation}
\begin{tabular}{lcc}
\toprule
\textbf{Configuration} & \textbf{Success Rate} & \textbf{Δ} \\
\midrule
\textbf{Full ReBel} & \textbf{67.7\%} & - \\
\midrule
\multicolumn{3}{l}{\textit{Core Components}} \\
\quad w/o Belief Reward & 58.3\% & -9.4\% \\
\quad w/o Belief Context & 61.5\% & -6.2\% \\
\quad w/o Both (= GRPO) & 57.3\% & -10.4\% \\
\midrule
\multicolumn{3}{l}{\textit{Reward Components}} \\
\quad w/o Consistency ($r^{cons}$) & 64.2\% & -3.5\% \\
\quad w/o Progress ($r^{prog}$) & 59.8\% & -7.9\% \\
\quad w/o Exploration ($r^{exp}$) & 65.8\% & -1.9\% \\
\bottomrule
\end{tabular}
\end{table}
```

### Figure 5: Hyperparameter Sensitivity

**位置**: Section 5.6 或 Appendix

**文件**: `figures_tables/fig5_sensitivity.pdf`

**图表描述**:
```
布局: 1行2列

左图: ω (Step Advantage Weight) Sensitivity
- X轴: ω values (0, 0.5, 1.0, 1.5, 2.0)
- Y轴: Success Rate (%)
- 柱状图 + 误差线
- 标注最优值: ω = 1.0

右图: Belief Granularity Comparison
- X轴: Granularity (subgoal, medium, fine)
- Y轴: Success Rate (%)
- 柱状图
- 附加轴: Number of Groups (右Y轴)
```

**预期数据**:

| ω | Success Rate |
|---|--------------|
| 0.0 | 57.3% |
| 0.5 | 64.2% |
| 1.0 | 67.7% |
| 1.5 | 65.8% |
| 2.0 | 62.4% |

| Granularity | SR | Num Groups | Mean Size |
|-------------|-----|------------|-----------|
| subgoal | 67.7% | 38 | 27.2 |
| medium | 65.2% | 95 | 10.8 |
| fine | 61.8% | 312 | 3.3 |

---

## 6. 泛化实验

### Table 6: Generalization Results

**位置**: Appendix E.4

```latex
\begin{table}[t]
\centering
\caption{Generalization performance across different task novelty levels.}
\label{tab:generalization}
\begin{tabular}{lcccc}
\toprule
\textbf{Method} & \textbf{L0 (Seen)} & \textbf{L1 (Novel Combo)} & \textbf{L2 (Novel Task)} & \textbf{Gap (L0→L2)} \\
\midrule
GRPO & 57.3\% & 47.8\% & 38.2\% & -19.1\% \\
GiGPO & 61.3\% & 51.5\% & 41.7\% & -19.6\% \\
\textbf{ReBel} & \textbf{67.7\%} & \textbf{58.2\%} & \textbf{49.5\%} & \textbf{-18.2\%} \\
\bottomrule
\end{tabular}
\end{table}
```

---

## 7. 计算开销

### Figure 6: Computational Overhead

**位置**: Section 5.7

**文件**: `figures_tables/fig6_overhead.pdf`

**图表描述**:
```
布局: 堆叠柱状图

Per-Iteration Time Breakdown
- X轴: Methods (GRPO, GiGPO, ReBel)
- Y轴: Time (seconds)
- 堆叠组件:
  - Environment Step (绿)
  - Model Forward (蓝)
  - Belief Parsing (黄) - ReBel only
  - Reward Computation (红) - ReBel additional
  - Policy Update (紫)
- 标注总时间和额外开销百分比
```

**预期数据**:

| Component | GRPO | GiGPO | ReBel |
|-----------|------|-------|-------|
| Env Step | 35s | 35s | 35s |
| Model Forward | 42s | 42s | 42s |
| Belief Parsing | - | - | 5s |
| Reward Comp | 3s | 5s | 8s |
| Policy Update | 18s | 18s | 18s |
| **Total** | **98s** | **100s** | **108s** |
| Overhead vs GRPO | - | +2% | +10% |

---

## 8. Case Study

### Figure 7: Successful Trajectory Example

**位置**: Appendix F.1

**文件**: `figures_tables/fig7_case_study.pdf`

**图表描述**:
```
布局: 时间线 + 信念状态面板

展示一个成功轨迹:
Task: "put some alarmclock on desk"

Step 1: Observation → Belief Update → Action
- Observation: "You are in the middle of a room..."
- Belief: {subgoal: "Find alarm clock", found_objects: {}}
- Action: "go to drawer 1"

Step 2-5: ...

Step 6: 找到目标
- Belief: {subgoal: "Pick up alarm clock", found_objects: {alarmclock1: sidetable1}}
- Action: "take alarmclock 1 from sidetable 1"

Step 7: 完成任务
- Belief: {subgoal: "Place on desk", inventory: alarmclock1}
- Action: "put alarmclock 1 in/on desk 1"
```

---

## 9. 附录表格

### Table A1: Full Hyperparameter Configuration

**位置**: Appendix D.1

```latex
\begin{table}[h]
\centering
\caption{Complete hyperparameter configuration for all experiments.}
\label{tab:hyperparams}
\begin{tabular}{lcccc}
\toprule
\textbf{Parameter} & \textbf{ALFWorld} & \textbf{WebShop} & \textbf{SciWorld} \\
\midrule
\multicolumn{4}{l}{\textit{Model Configuration}} \\
Base Model & \multicolumn{3}{c}{Qwen2.5-\{1.5B, 3B, 7B\}-Instruct} \\
Max Prompt Length & 2048 & 4096 & 4096 \\
Max Response Length & 512 & 512 & 512 \\
\midrule
\multicolumn{4}{l}{\textit{Training Configuration}} \\
Learning Rate & 1e-6 & 1e-6 & 1e-6 \\
Batch Size & 128 & 128 & 128 \\
Group Size $N$ & 8 & 8 & 8 \\
KL Coefficient & 0.01 & 0.01 & 0.01 \\
Training Iterations & 200 & 200 & 200 \\
\midrule
\multicolumn{4}{l}{\textit{ReBel Configuration}} \\
$\omega$ (Step Advantage Weight) & 1.0 & 1.0 & 1.0 \\
$w_1$ (Consistency) & 0.3 & 0.3 & 0.3 \\
$w_2$ (Progress) & 0.5 & 0.5 & 0.5 \\
$w_3$ (Exploration) & 0.2 & 0.2 & 0.2 \\
Belief Granularity & subgoal & subgoal & subgoal \\
Normalization Mode & mean\_norm & mean\_norm & mean\_norm \\
\midrule
\multicolumn{4}{l}{\textit{Environment Configuration}} \\
Max Steps & 50 & 15 & 40 \\
Invalid Action Penalty & -0.1 & -0.1 & -0.1 \\
\bottomrule
\end{tabular}
\end{table}
```

### Table A2: Belief State Schema for ALFWorld

**位置**: Appendix D.2

```latex
\begin{table}[h]
\centering
\caption{Belief state schema for ALFWorld environment.}
\label{tab:belief_schema}
\begin{tabular}{lll}
\toprule
\textbf{Component} & \textbf{Field} & \textbf{Description} \\
\midrule
\multirow{4}{*}{world\_model}
& found\_objects & Dict: object\_id → location\_id \\
& inventory & String or null (single item) \\
& state\_changes & Dict: object\_id → state \\
& cleared\_receptacles & List of explored containers \\
\midrule
\multirow{3}{*}{task\_state}
& status & in\_progress | completed | failed \\
& current\_subgoal & Current objective string \\
& evidence & Observations supporting status \\
\midrule
\multirow{2}{*}{exploration\_map}
& visited\_locations & List of visited locations \\
& priority\_targets & List of next exploration targets \\
\bottomrule
\end{tabular}
\end{table}
```

---

## 10. 图表文件清单

| 文件名 | 类型 | 位置 | 描述 |
|--------|------|------|------|
| `table1_main_results.tex` | LaTeX | 正文5.2 | 主实验结果表 |
| `table2_model_scale.tex` | LaTeX | 正文5.2/附录 | 模型规模分析 |
| `table3_belief_drift.tex` | LaTeX | 正文5.4 | 信念漂移指标 |
| `table4_context_efficiency.tex` | LaTeX | 正文5.5 | 上下文效率 |
| `table5_ablation.tex` | LaTeX | 正文5.6 | 消融实验 |
| `table6_generalization.tex` | LaTeX | 附录E.4 | 泛化实验 |
| `tableA1_hyperparams.tex` | LaTeX | 附录D.1 | 完整超参数 |
| `tableA2_belief_schema.tex` | LaTeX | 附录D.2 | 信念状态结构 |
| `fig2_training_curves.pdf` | Figure | 正文5.3 | 训练曲线 |
| `fig3_belief_drift.pdf` | Figure | 正文5.4 | 信念漂移分析 |
| `fig4_context_efficiency.pdf` | Figure | 正文5.5 | 上下文效率 |
| `fig5_sensitivity.pdf` | Figure | 正文5.6 | 超参数敏感性 |
| `fig6_overhead.pdf` | Figure | 正文5.7 | 计算开销 |
| `fig7_case_study.pdf` | Figure | 附录F.1 | Case Study |

---

## 11. 实验执行检查清单

### 11.1 实验前准备

- [ ] 确认所有baselines代码可运行
- [ ] 准备好3个benchmark的数据
- [ ] 配置SwanLab/WandB项目
- [ ] 分配GPU资源

### 11.2 主实验

- [ ] ALFWorld: GRPO × 3 seeds
- [ ] ALFWorld: GiGPO × 3 seeds
- [ ] ALFWorld: ReBel × 3 seeds
- [ ] WebShop: 同上
- [ ] SciWorld: 同上 (如果时间允许)

### 11.3 消融实验

- [ ] w/o Belief Reward × 3 seeds
- [ ] w/o Belief Context × 3 seeds
- [ ] w/o Consistency Reward × 3 seeds
- [ ] w/o Progress Reward × 3 seeds
- [ ] w/o Exploration Reward × 3 seeds

### 11.4 分析实验

- [ ] ω 敏感性: 5个值 × 3 seeds
- [ ] Granularity 敏感性: 3个值 × 3 seeds
- [ ] 模型规模: 3个规模 × 3 seeds

### 11.5 结果整理

- [ ] 统计显著性检验
- [ ] 生成所有表格
- [ ] 生成所有图表
- [ ] Case study选择与标注
- [ ] 撰写实验分析

---

**文档版本**: v2.0
**更新日期**: 2026-01-03
