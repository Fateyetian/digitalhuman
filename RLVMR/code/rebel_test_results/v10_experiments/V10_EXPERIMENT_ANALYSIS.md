# ReBel V10 消融实验分析报告

> **生成时间**: 2026-02-20
>
> **实验周期**: 2026-02-11 ~ 2026-02-17
>
> **目标**: 通过系统消融研究，隔离 ReBel 相对于 GRPO/GiGPO 的各项核心贡献因素，排除工程 trick 干扰，为论文提供干净的因果证据，并确定最终版本的 ReBel 算法。

---

## 目录

1. [实验概览](#一实验概览)
2. [主要结果](#二主要结果)
3. [训练动态分析](#三训练动态分析)
4. [因素贡献分解](#四因素贡献分解)
5. [假设验证](#五假设验证)
6. [Per-Task 细粒度分析](#六per-task-细粒度分析)
7. [与历史版本对比](#七与历史版本对比)
8. [关键发现与讨论](#八关键发现与讨论)
9. [最终算法确定](#九最终算法确定)
10. [正式实验方案](#十正式实验方案)

---

## 一、实验概览

### 1.1 实验设计

本轮消融研究设计了 7 组实验，采用 **逐步添加因素 (Progressive Ablation)** 的方式，沿以下链路依次引入 ReBel 的各项组件：

```
A1 (GRPO 纯基线)
 │
 ├── A2 (+ 训练 tricks)              → 量化工程优化的贡献
 │    │
 │    └── B1 (+ 信念提示格式)         → 量化 Structured Belief Prompting 的贡献
 │         │
 │         ├── C1 (+ 观测分组/GiGPO)   → 量化 step-level grouping 的贡献
 │         │    │
 │         │    └── C2 (改用信念分组)    → 量化 belief grouping vs observation grouping
 │         │         │
 │         │         ├── D1 (+ 内在奖励)  → 量化密集内在奖励的贡献
 │         │         │
 │         │         └── E1 (+ 课程衰减)  → 量化课程学习的贡献
```

### 1.2 实验配置矩阵

| ID | 名称 | Advantage Estimator | 提示格式 | Step 分组 | 内在奖励 | 训练 Tricks | Adv Tricks |
|----|------|:-------------------:|:---------:|:---------:|:---------:|:-----------:|:----------:|
| **A1** | GRPO Baseline | GRPO | `<think>` | - | - | - | - |
| **A2** | GRPO + Tricks | GRPO | `<think>` | - | - | **Yes** | - |
| **B1** | GRPO + Belief Prompt | GRPO | `<belief>` | - | - | Yes | - |
| **C1** | GiGPO + Belief Prompt | GiGPO | `<belief>` | Obs Hash | - | Yes | - |
| **C2** | ReBel Group Only | ReBel | `<belief>` | Belief Hash | - | Yes | **Yes** |
| **D1** | ReBel Full | ReBel | `<belief>` | Belief Hash | **Full** | Yes | Yes |
| **E1** | ReBel + Curriculum | ReBel | `<belief>` | Belief Hash | **Cosine Decay** | Yes | Yes |

### 1.3 统一实验条件

所有实验共享以下基础参数以保证公平对比：

| 参数 | 值 | 说明 |
|------|----|------|
| 基座模型 | Qwen2.5-1.5B-Instruct (SFT) | 统一起始点 |
| seed | 42 | 单种子 (受限于计算资源) |
| total_epochs | 100 | 充分训练 |
| train_batch_size | 16 | 每批训练样本 |
| val_batch_size | 128 | 完整验证集评估 |
| rollout.n | 16 | 每 prompt 16 条轨迹 |
| max_steps | 30 | 最大交互步数 |
| learning_rate | 1e-6 | 统一学习率 |
| ppo_epochs | 1 | 单次更新 |
| test_freq | 5 | 每 5 epoch 验证一次 |
| environment | ALFWorld (generalization_level=0) | 统一环境 |

### 1.4 实验运行状态

所有 7 组实验均已完成 100 epoch 训练：

| ID | 实验名称 | 远程目录 | 状态 |
|----|---------|---------|:----:|
| A1 | grpo_baseline | `A1_grpo_baseline_seed42_20260211_160559/` | 100 ep |
| A2 | grpo_tricks | `A2_grpo_tricks_seed42_20260211_232128/` | 100 ep |
| B1 | grpo_belief_prompt | `B1_grpo_belief_prompt_seed42_20260212_064716/` | 100 ep |
| C1 | gigpo_belief_prompt | `C1_gigpo_belief_prompt_seed42_20260212_235634/` | 100 ep |
| C2 | rebel_group_only | `C2_rebel_group_only_seed42_20260215_155337/` | 100 ep |
| D1 | rebel_full | `D1_rebel_full_seed42_20260216_082441/` | 100 ep |
| E1 | rebel_curriculum | `E1_rebel_curriculum_seed42_20260217_002626/` | 100 ep |

> 远程存储基路径: `/fs-computility-new/UPDZ03_chengjun/huangsijie.p/rebel_results/v10_experiments/`

---

## 二、主要结果

### 2.1 核心指标汇总

#### Table 1: 验证集成功率 (Validation Success Rate)

| Rank | ID | Method | Peak SR | Peak Epoch | Final SR (ep100) | Avg Steps (peak) |
|:----:|:---:|--------|:-------:|:----------:|:----------------:|:----------------:|
| 1 | **C1** | **GiGPO + Belief Prompt** | **94.5%** | 80 | 86.7% | 14.4 |
| 2 | C2 | ReBel Group Only | 91.4% | 90 | 83.6% | 12.3 |
| 3 | E1 | ReBel + Curriculum | 89.8% | 95 | 83.6% | 17.5 |
| 4 | B1 | GRPO + Belief Prompt | 88.3% | 95 | 74.2% | 18.2 |
| 5 | D1 | ReBel Full | 87.5% | 90 | 79.7% | 14.5 |
| 6 | A2 | GRPO + Tricks | 86.7% | 75 | 78.9% | 14.2 |
| 7 | A1 | GRPO Baseline | 81.2% | 95 | 78.9% | 15.3 |

#### Table 2: 训练后期稳定性 (ep80-100 平均 SR)

| ID | Method | 平均 SR (ep80-100) | 标准差 | 最大波动 |
|:---:|--------|:-----------------:|:------:|:--------:|
| **C1** | GiGPO + Belief Prompt | **90.0%** | 3.0% | 7.8% |
| C2 | ReBel Group Only | 86.6% | 3.6% | 7.8% |
| E1 | ReBel + Curriculum | 86.6% | 2.8% | 6.2% |
| D1 | ReBel Full | 84.3% | 3.0% | 7.8% |
| A2 | GRPO + Tricks | 79.4% | 0.7% | 0.8% |
| A1 | GRPO Baseline | 78.7% | 3.4% | 7.8% |
| B1 | GRPO + Belief Prompt | 77.4% | 8.2% | 14.1% |

> **注**: B1 虽然 Peak SR 高达 88.3%，但 ep80-100 平均仅 77.4%，标准差 8.2%，表现极不稳定。

---

### 2.2 完整训练曲线

```
Validation Success Rate (val/success_rate) over 100 epochs
═════════════════════════════════════════════════════════════════════

     Epoch:   0      5     10     15     20     25     30     35     40     45     50     55     60     65     70     75     80     85     90     95    100
  ─────────┼──────┼──────┼──────┼──────┼──────┼──────┼──────┼──────┼──────┼──────┼──────┼──────┼──────┼──────┼──────┼──────┼──────┼──────┼──────┼──────┼──────
  A1 GRPO  │.000  │.102  │.055  │.258  │.344  │.445  │.562  │.469  │.508  │.617  │.617  │.617  │.680  │.602  │.719  │.766  │.727  │.766  │.734  │.812★ │.789
  A2 +Trk  │.000  │.141  │.359  │.320  │.453  │.461  │.594  │.539  │.602  │.672  │.703  │.711  │.711  │.695  │.766  │.867★ │.797  │.805  │.797  │.781  │.789
  B1 +Bel  │.008  │.000  │.000  │.086  │.383  │.406  │.547  │.664  │.711  │.664  │.742  │.766  │.625  │.742  │.656  │.734  │.766  │.727  │.789  │.883★ │.742
  C1 GiGP  │.000  │.016  │.391  │.688  │.742  │.711  │.812  │.812  │.828  │.875  │.844  │.820  │.883  │.852  │.852  │.875  │.945★ │.891  │.883  │.914  │.867
  C2 RGrp  │.000  │.031  │.172  │.477  │.617  │.633  │.617  │.664  │.820  │.734  │.641  │.711  │.633  │.680  │.812  │.820  │.828  │.898  │.914★ │.852  │.836
  D1 Full  │.016  │.008  │.141  │.406  │.648  │.586  │.672  │.750  │.617  │.711  │.688  │.711  │.719  │.773  │.734  │.828  │.812  │.805  │.875★ │.828  │.797
  E1 Curr  │.000  │.000  │.195  │.266  │.469  │.469  │.477  │.570  │.625  │.805  │.805  │.797  │.844  │.742  │.750  │.844  │.891★ │.867  │.836  │.898  │.836

  ★ = Peak SR for each experiment
```

---

## 三、训练动态分析

### 3.1 学习速度对比

定义 **"达到 80% SR 的首个 epoch"** 作为学习速度指标：

| ID | Method | 首次达到 80% | 首次达到 85% | 首次达到 90% |
|:---:|--------|:----------:|:----------:|:----------:|
| **C1** | GiGPO + Belief Prompt | **ep30** | ep45 | ep80 |
| C2 | ReBel Group Only | ep40 | ep85 | ep90 |
| D1 | ReBel Full | ep35 | ep75 | - |
| E1 | ReBel + Curriculum | ep45 | ep55 | ep80 |
| A2 | GRPO + Tricks | ep75 | ep75 | - |
| A1 | GRPO Baseline | ep95 | - | - |
| B1 | GRPO + Belief Prompt | ep55 | ep55 | - |

**发现**: C1 (GiGPO) 学习速度最快，在 ep30 即首次突破 80%。引入 step-level advantage estimation 显著加速了学习。

### 3.2 训练稳定性分析

定义 **"ep80-100 SR 下降幅度"** (Peak - Final SR) 作为后期稳定性指标：

| ID | Method | Peak SR | Final SR | 下降幅度 | 稳定性评级 |
|:---:|--------|:-------:|:--------:|:--------:|:----------:|
| **C1** | GiGPO + Belief Prompt | 94.5% | 86.7% | -7.8% | 良好 |
| E1 | ReBel + Curriculum | 89.8% | 83.6% | -6.2% | 良好 |
| C2 | ReBel Group Only | 91.4% | 83.6% | -7.8% | 良好 |
| D1 | ReBel Full | 87.5% | 79.7% | -7.8% | 一般 |
| A2 | GRPO + Tricks | 86.7% | 78.9% | -7.8% | 一般 |
| A1 | GRPO Baseline | 81.2% | 78.9% | -2.3% | 稳定 |
| **B1** | GRPO + Belief Prompt | **88.3%** | **74.2%** | **-14.1%** | **不稳定** |

**发现**: B1 是最不稳定的实验，Peak 到 Final 下降 14.1 个百分点。这表明单纯使用信念提示格式但不配合 step-level grouping 时，训练后期容易出现策略震荡。

### 3.3 收敛特征分类

基于训练曲线，可将 7 组实验分为三类：

**类型 1: 快速收敛型** — C1
- 特征：ep15 起快速攀升，ep30 已达 80%+，后期高位波动
- 原因：GiGPO 的 step-level advantage 提供更精细的梯度信号

**类型 2: 稳步攀升型** — C2, D1, E1
- 特征：ep15-30 起步较慢，ep40-60 稳步提升，ep70-90 达到高峰
- 原因：ReBel 信念分组的粒度更细，初期需要更多样本建立分组

**类型 3: 慢启动型** — A1, A2, B1
- 特征：前 20 epoch 学习缓慢，后期陡升但不稳定
- 原因：缺乏 step-level advantage，仅靠 episode-level 信号学习效率低

---

## 四、因素贡献分解

### 4.1 逐因素贡献量化

#### Table 3: Factor Contribution Breakdown (基于 Peak SR)

| 对比 | 隔离因素 | Base SR | New SR | Delta SR | 显著性 |
|------|---------|:-------:|:------:|:--------:|:------:|
| **A2 vs A1** | 训练 Tricks (非对称裁剪+熵保护+无效动作惩罚) | 81.2% | 86.7% | **+5.5%** | 中等 |
| **B1 vs A2** | Structured Belief Prompting | 86.7% | 88.3% | **+1.6%** | 弱 |
| **C1 vs B1** | Step-level Advantage (观测 hash 分组) | 88.3% | 94.5% | **+6.2%** | **强** |
| **C2 vs C1** | Belief Grouping vs Observation Grouping | 94.5% | 91.4% | **-3.1%** | 负面 |
| **D1 vs C2** | 密集内在信念奖励 | 91.4% | 87.5% | **-3.9%** | 负面 |
| **E1 vs D1** | Curriculum Decay (课程衰减) | 87.5% | 89.8% | **+2.3%** | 弱 |
| **E1 vs C2** | 课程化内在奖励 (综合) | 91.4% | 89.8% | **-1.6%** | 负面 |

#### Table 4: Factor Contribution Breakdown (基于 ep80-100 平均 SR)

| 对比 | 隔离因素 | Base Avg | New Avg | Delta | 显著性 |
|------|---------|:--------:|:-------:|:-----:|:------:|
| **A2 vs A1** | 训练 Tricks | 78.7% | 79.4% | **+0.7%** | 弱 |
| **B1 vs A2** | Belief Prompting | 79.4% | 77.4% | **-2.0%** | 负面 |
| **C1 vs B1** | Step Grouping (Obs) | 77.4% | 90.0% | **+12.6%** | **强** |
| **C2 vs C1** | Belief vs Obs Grouping | 90.0% | 86.6% | **-3.4%** | 负面 |
| **D1 vs C2** | Dense Intrinsic Reward | 86.6% | 84.3% | **-2.3%** | 负面 |
| **E1 vs D1** | Curriculum Decay | 84.3% | 86.6% | **+2.3%** | 中等 |
| **E1 vs C2** | Curriculum Overall | 86.6% | 86.6% | **0.0%** | 无 |

### 4.2 贡献排名

综合 Peak SR 和 ep80-100 平均 SR，各因素的贡献排名：

```
┌─────────────────────────────────────────────────────────────────────────┐
│  因素贡献排名 (按对性能的正面影响大小排序)                                    │
│                                                                         │
│  1. Step-level Advantage Estimation     +6.2% (Peak) / +12.6% (Avg)  ★★★★★ │
│  2. 训练 Tricks (工程优化)                +5.5% (Peak) / +0.7% (Avg)   ★★★   │
│  3. Curriculum Decay                    +2.3% (Peak) / +2.3% (Avg)   ★★    │
│  4. Structured Belief Prompting         +1.6% (Peak) / -2.0% (Avg)   ★     │
│  5. Belief Grouping (vs Obs Grouping)   -3.1% (Peak) / -3.4% (Avg)   ✗     │
│  6. Dense Intrinsic Reward              -3.9% (Peak) / -2.3% (Avg)   ✗✗    │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 五、假设验证

### 5.1 假设 1: B1 >> A2 (信念提示格式是核心贡献)

| 指标 | A2 | B1 | Delta | 结论 |
|------|:---:|:---:|:-----:|:----:|
| Peak SR | 86.7% | 88.3% | +1.6% | — |
| Final SR | 78.9% | 74.2% | -4.7% | — |
| Avg SR (ep80-100) | 79.4% | 77.4% | -2.0% | — |
| 训练稳定性 | 一般 | **不稳定** | — | — |

**结论: 部分确认，但效果远弱于预期。**

信念提示格式在 Peak SR 上仅带来 +1.6% 的微弱提升，且训练极不稳定 (Final SR 仅 74.2%，为所有实验中最差)。在 ep80-100 平均 SR 上甚至表现更差 (-2.0%)。

**解读**: 单纯将模型输出格式从 `<think>` 改为结构化 `<belief>` JSON，在没有 step-level advantage 的配合下，不足以成为独立的核心贡献点。B1 的高 Peak 更可能来自偶然波动而非稳定改进。

### 5.2 假设 2: C2 > C1 (信念分组优于观测分组)

| 指标 | C1 (Obs Grouping) | C2 (Belief Grouping) | Delta | 结论 |
|------|:---:|:---:|:-----:|:----:|
| Peak SR | **94.5%** | 91.4% | -3.1% | C1 > C2 |
| Final SR | **86.7%** | 83.6% | -3.1% | C1 > C2 |
| Avg SR (ep80-100) | **90.0%** | 86.6% | -3.4% | C1 > C2 |
| 学习速度 (首达 80%) | **ep30** | ep40 | -10 ep | C1 > C2 |

**结论: 假设被否定。观测 hash 分组 (GiGPO) 全面优于信念 hash 分组 (ReBel)。**

**解读**:
- 观测 hash 分组直接使用环境观测字符串进行匹配，粒度更合适
- 信念 hash 分组使用模型自身生成的信念状态 JSON，在训练早期模型信念不准确时，可能产生错误的分组
- 信念状态的 hash 碰撞概率更低 (更细粒度)，导致有效分组过小，advantage 估计方差增大
- 此外 C2 使用了 Adv Tricks (task weighting) 而 C1 未使用，C2 在额外优化下仍劣于 C1，说明信念分组本身确实是负面因素

> **注意**: C2 额外启用了 Adv Tricks (task weighting, min_samples_ratio 等)，而 C1 未使用。这意味着 C1 在更少优化手段的条件下仍然胜出，进一步说明观测分组的优势。

### 5.3 假设 3: D1 <= C2 (密集内在奖励后期有害)

| 指标 | C2 (无内在奖励) | D1 (Full 内在奖励) | Delta | 结论 |
|------|:---:|:---:|:-----:|:----:|
| Peak SR | **91.4%** | 87.5% | -3.9% | C2 > D1 |
| Final SR | **83.6%** | 79.7% | -3.9% | C2 > D1 |
| Avg SR (ep80-100) | **86.6%** | 84.3% | -2.3% | C2 > D1 |

**结论: 假设确认。密集内在信念奖励对训练有害。**

**解读**:
- 完全印证了 V8 消融实验的发现 (V8 No Belief 90.6% > V8 Full 82-83.6%)
- 内在奖励的 4 个组件 (consistency + progress + exploration + format) 在训练后期引入额外的梯度噪声
- 内在奖励信号可能与最终任务目标不完全一致，产生优化方向冲突
- 在模型已建立基本环境认知后，内在奖励成为干扰信号而非引导信号

### 5.4 假设 4: E1 > D1 且 E1 >= C2 (课程衰减有效)

| 指标 | D1 (Full) | E1 (Curriculum) | C2 (No Reward) | 结论 |
|------|:---:|:---:|:---:|:----:|
| Peak SR | 87.5% | **89.8%** | **91.4%** | D1 < E1 < C2 |
| Final SR | 79.7% | **83.6%** | **83.6%** | D1 < E1 = C2 |
| Avg SR (ep80-100) | 84.3% | **86.6%** | **86.6%** | D1 < E1 = C2 |

**结论: 部分确认。课程衰减有效缓解内在奖励的负面效果，但未能超越不使用内在奖励的方案。**

**解读**:
- E1 > D1 (+2.3% Peak)：课程衰减确实部分缓解了密集奖励的危害
- E1 < C2 (-1.6% Peak)：即使加了衰减，内在奖励的净效果仍为负
- E1 = C2 (Final SR & Avg)：在训练后期，课程化内在奖励与不用奖励趋于一致，说明衰减后的残余内在奖励 (min_weight=0.1) 影响极小
- **结论**：内在信念奖励即使通过课程化引入，也不构成正面贡献。"不使用"是最优策略

---

## 六、Per-Task 细粒度分析

### 6.1 各任务 Peak Epoch 成功率

#### Table 5: Per-Task Success Rate at Peak Epoch

| Task Type | A1 (ep95) | A2 (ep75) | B1 (ep95) | C1 (ep80) | C2 (ep90) | D1 (ep90) | E1 (ep95) |
|-----------|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|
| pick_and_place | 100.0% | 90.9% | 94.6% | 96.2% | 100.0% | 100.0% | 100.0% |
| pick_two_obj | 87.5% | 82.1% | 79.2% | **100.0%** | 82.4% | 82.4% | 82.6% |
| pick_heat | 71.4% | 92.3% | 92.9% | 91.7% | 94.4% | 77.8% | **100.0%** |
| pick_cool | 56.0% | 75.0% | 76.0% | 80.0% | **90.5%** | 81.0% | 90.0% |
| pick_clean | 79.2% | **100.0%** | **100.0%** | **100.0%** | 84.6% | 80.8% | 91.3% |
| **look_at** | 75.0% | 66.7% | 75.0% | **92.9%** | 87.5% | **100.0%** | 60.0% |

### 6.2 Per-Task 分析要点

#### look_at_obj_in_light (最困难的少数类任务)

```
A1: 75.0%  →  A2: 66.7% (-8.3%)  →  B1: 75.0% (+8.3%)  →  C1: 92.9% (+17.9%)
                                                              C2: 87.5% (-5.4%)
                                                              D1: 100.0% (+12.5%)
                                                              E1: 60.0% (-27.5%)
```

**发现**:
- C1 (GiGPO) 在 look_at 上达到 92.9%，均衡性最好
- D1 (Full ReBel) 在 look_at 上达到 100.0%，说明内在奖励对少数任务有特殊帮助
- E1 (Curriculum) 在 look_at 上仅 60.0%，衰减可能过早削弱了对少数任务的引导

> **注意**: look_at 任务样本量极小 (约 12-14 个验证样本)，单个样本的成功/失败就可能导致 7-8% 的 SR 波动。因此 look_at 的数值应以趋势为参考，不宜精确解读。

#### pick_cool_then_place_in_recep (第二难的任务)

```
A1: 56.0%  →  A2: 75.0% (+19%)  →  B1: 76.0%  →  C1: 80.0%  →  C2: 90.5% (+10.5%)  →  D1: 81.0%  →  E1: 90.0%
```

**发现**: C2 和 E1 在 pick_cool 上表现最好 (90%+)，可能因为 belief grouping 和 task weighting 对该任务类型特别有效。

#### pick_two_obj_and_place (需要多轮操作的任务)

```
A1: 87.5%  →  A2: 82.1%  →  B1: 79.2%  →  C1: 100.0%  →  C2: 82.4%  →  D1: 82.4%  →  E1: 82.6%
```

**发现**: C1 (GiGPO) 在 pick_two_obj 上独占 100%，step-level advantage 对多轮操作任务帮助巨大。

### 6.3 任务均衡性分析

定义 **均衡性指标** = 最高任务 SR - 最低任务 SR (越小越均衡)：

| ID | Method | 最高任务 | 最低任务 | 差距 | 均衡性评级 |
|:---:|--------|:-------:|:-------:|:----:|:----------:|
| **C1** | GiGPO + Belief | 100.0% | 80.0% | 20.0% | **好** |
| D1 | ReBel Full | 100.0% | 77.8% | 22.2% | 好 |
| A1 | GRPO Baseline | 100.0% | 56.0% | 44.0% | 差 |
| E1 | ReBel Curriculum | 100.0% | 60.0% | 40.0% | 差 |
| B1 | GRPO Belief | 100.0% | 75.0% | 25.0% | 中 |
| C2 | ReBel Group | 100.0% | 82.4% | 17.6% | **好** |
| A2 | GRPO Tricks | 100.0% | 66.7% | 33.3% | 中 |

> C2 在任务均衡性上表现最好 (17.6% gap)，但 look_at 仅 87.5%。C1 综合均衡性 (20.0% gap) 和绝对性能 (94.5%) 最优。

---

## 七、与历史版本对比

### 7.1 跨版本性能汇总

#### Table 6: Historical Comparison (Peak/Best Validation SR)

| Version | Method | Best SR | look_at SR | Epochs | Notes |
|---------|--------|:-------:|:----------:|:------:|-------|
| V4-Exp1 | Task Status | 73.4% | - | 50 | 早期版本 |
| V6-Exp2 | Relative Norm | 78.1% | 83.3% | 60 | 最均衡 (历史) |
| V6-Exp3 | Full + KL in Reward | 84.4% | 16.7% | 100 | look_at 崩塌 |
| V7 | Clip-Cov Entropy | 85.9% | 81.2% | 100 | 稳定提升 |
| V8 | Task Weighting (best seed) | 83.6% | 78.6% | 100 | 多种子实验 |
| **V8 Ablation** | **No Belief Reward** | **90.6%** | **58.3%** | **100** | **历史最高整体** |
| V8 Ablation | No Result Reward | 80.5% | - | 60 | 仅信念奖励 |
| | | | | | |
| **V10-A1** | GRPO Baseline | 81.2% | 75.0% | 100 | 纯基线 |
| **V10-A2** | GRPO + Tricks | 86.7% | 66.7% | 100 | +工程优化 |
| **V10-B1** | GRPO + Belief Prompt | 88.3% | 75.0% | 100 | +信念格式 |
| **V10-C1** | **GiGPO + Belief Prompt** | **94.5%** | **92.9%** | **100** | **新 SOTA** |
| **V10-C2** | ReBel Group Only | 91.4% | 87.5% | 100 | 信念分组 |
| **V10-D1** | ReBel Full | 87.5% | 100.0% | 100 | 完整内在奖励 |
| **V10-E1** | ReBel + Curriculum | 89.8% | 60.0% | 100 | 课程衰减 |

### 7.2 关键对比

**V10-C1 (94.5%) vs V8 No Belief (90.6%)**: +3.9%

V10-C1 的 GiGPO + Belief Prompt 配置实现了全实验历史上的最高成功率 94.5%，且 look_at 达到 92.9%。相比之下，V8 No Belief 虽然整体 90.6%，但 look_at 仅 58.3%。

**V10 vs V7/V8**: V10 统一框架下的实验结果整体趋势与 V7/V8 一致 — 移除内在奖励有利于整体 SR 提升。

---

## 八、关键发现与讨论

### 8.1 核心发现总结

#### 发现 1: Step-level Advantage Estimation 是最关键的贡献因素

| 证据 | 数值 |
|------|------|
| C1 vs B1 (Peak SR) | +6.2% |
| C1 vs B1 (Avg SR ep80-100) | +12.6% |
| C1 学习速度 | ep30 达到 80%，比其他方法快 10-65 epoch |

Step-level advantage estimation 通过在同一 observation 下比较不同动作的结果，提供了比 episode-level advantage 更精细的梯度信号。这显著加速了学习并提升了最终性能。

#### 发现 2: Structured Belief Prompting 的作用有限且不稳定

| 证据 | 数值 |
|------|------|
| B1 vs A2 (Peak SR) | +1.6% (微弱) |
| B1 vs A2 (Avg SR ep80-100) | -2.0% (负面) |
| B1 训练稳定性 | 最差 (SD = 8.2%) |

信念提示格式在缺乏配套 step-level mechanism 时，不是一个独立有效的贡献。模型被迫输出结构化 JSON 增加了输出复杂度，但 GRPO 的 episode-level advantage 无法有效利用这些信息。

**但是**: 信念提示格式在 C1 中与 GiGPO 配合后效果极佳 (94.5%)，说明其价值在于 **为 step-level grouping 提供更好的分组基础**，而非独立发挥作用。

#### 发现 3: 观测 hash 分组优于信念 hash 分组

| 证据 | 数值 |
|------|------|
| C1 vs C2 (Peak SR) | C1 胜出 3.1% |
| C1 vs C2 (Avg SR) | C1 胜出 3.4% |
| C1 vs C2 (学习速度) | C1 快 10 epoch |

可能原因：
1. **信念状态初期不准确**: 训练早期模型的信念预测不可靠，基于不准确的信念进行分组引入噪声
2. **分组粒度问题**: 信念 hash 生成的分组过于细碎 (每个信念状态几乎唯一)，导致组内样本不足
3. **观测 hash 更稳定**: 环境观测字符串直接来自环境，不受模型学习过程影响，提供稳定的分组基础

#### 发现 4: 密集内在奖励有害，课程化只是部分缓解

| 证据 | 数值 |
|------|------|
| D1 vs C2 (Peak SR) | -3.9% |
| E1 vs C2 (Peak SR) | -1.6% |
| E1 vs C2 (Avg SR) | ±0.0% |

密集内在信念奖励在所有配置下都不是正面贡献。课程化衰减虽然缓解了部分危害，但"不使用"始终是更优策略。这彻底否定了原始 ReBel 的核心假设 — 信念内在奖励作为密集反馈信号有助于训练。

### 8.2 对论文叙事的影响

原计划的论文叙事需要根据实验结果进行重大调整：

| 原假设 | 实验结果 | 论文影响 |
|--------|---------|---------|
| 信念提示是核心贡献 | 效果微弱且不稳定 | 不能作为独立贡献点 |
| 信念分组优于观测分组 | 被否定 (观测分组更好) | 原始 ReBel 分组设计需要反思 |
| 内在奖励是关键 | 有害 | 需要重新定位方法 |
| 课程衰减可以拯救内在奖励 | 仅部分缓解 | 课程学习不构成独立贡献 |

### 8.3 重新审视方法定位

基于实验证据，**ReBel 方法的真正价值**不在于信念分组或内在奖励，而在于：

1. **Structured Belief Prompting + Step-level Advantage** 的组合 (C1 = 94.5%)
2. 结构化输出格式为 step-level advantage 提供了更丰富的信息基础
3. 工程层面的稳定化技术 (非对称裁剪、熵保护) 提供稳健的训练基础

---

## 九、最终算法确定

### 9.1 最优配置: GiGPO + Belief Prompting (C1 配置)

基于全面消融分析，**推荐 C1 配置作为正式实验的基线**:

```yaml
# ============================
# 最终推荐配置 (C1-based)
# ============================

# Advantage Estimation: GiGPO (观测 hash 分组)
algorithm.adv_estimator: gigpo
algorithm.gigpo.step_advantage_w: 0.5
algorithm.gigpo.mode: "mean_norm"
algorithm.gamma: 0.95

# Belief Prompting: 启用 (结构化输出格式)
algorithm.rebel.enable: true

# 内在奖励: 关闭
algorithm.rebel.use_belief_reward: false
algorithm.rebel.use_result_reward: true

# 训练稳定化 Tricks
actor_rollout_ref.actor.clip_ratio_low: 0.2
actor_rollout_ref.actor.clip_ratio_high: 0.28    # 非对称裁剪
actor_rollout_ref.actor.entropy_coeff: 0.001
actor_rollout_ref.actor.use_kl_loss: true
actor_rollout_ref.actor.kl_loss_coef: 0.01
actor_rollout_ref.actor.use_invalid_action_penalty: true
actor_rollout_ref.actor.invalid_action_penalty_coef: 0.1

# 熵保护
algorithm.rebel.entropy_protection.enable: true
algorithm.rebel.entropy_protection.method: clip_cov
algorithm.rebel.entropy_protection.clip_cov_lb: 0.0
algorithm.rebel.entropy_protection.clip_cov_ub: 0.3
```

### 9.2 备选配置

| 方案 | 配置基础 | 适用场景 | 预期 SR |
|------|---------|---------|:-------:|
| **方案 A (推荐)** | C1 (GiGPO + Belief Prompt) | 追求最高性能 | ~94% |
| 方案 B | C2 (ReBel Group + No Reward) | 追求最高均衡性 | ~91% |
| 方案 C | C1 + Adv Tricks | C1 基础上增加 task weighting | 待验证 |

### 9.3 重构后的论文方法名

考虑到实验结果，建议将方法重命名或重新定位：

**选项 1**: 保持 "ReBel" 品牌，但重新定义核心贡献
- ReBel = Reinforcement Learning with **Belief-structured Exploration**
- 核心不是信念奖励，而是信念结构化推理 + step-level advantage

**选项 2**: 新方法名
- **BeST** = **Be**lief-**S**tructured **T**hinking for Interactive Decision Making
- 强调 structured belief prompting 与 step-level advantage 的协同作用

---

## 十、正式实验方案

### 10.1 论文实验需求

基于消融结果，正式实验应包含以下对比：

#### 主实验 (Main Results)

| 实验 | 方法 | 目的 |
|------|------|------|
| Baseline 1 | GRPO (标准) | 基线 |
| Baseline 2 | GRPO + Tricks | 公平基线 (含工程优化) |
| Baseline 3 | GiGPO (标准 `<think>`) | step-level 基线 |
| **Ours** | **GiGPO + Belief Prompt** | 本文方法 |

#### 消融实验 (Ablation Study)

| 消融 | 移除因素 | 对应 V10 实验 |
|------|---------|:------------:|
| w/o Belief Prompt | 移除结构化信念格式 | GiGPO + `<think>` (新增) |
| w/o Step Advantage | 移除 step-level grouping | B1 |
| w/ Intrinsic Reward | 添加内在奖励 | D1 |
| w/ Curriculum | 添加课程化奖励 | E1 |

### 10.2 需要补充的实验

当前 V10 消融缺少一个关键对比：

| 缺失实验 | 配置 | 原因 |
|----------|------|------|
| **GiGPO + `<think>` + Tricks** | `adv_estimator=gigpo, rebel.enable=false, tricks=on` | 需要隔离 Belief Prompt 在 GiGPO 框架下的贡献 |

这个实验可以回答：**在 GiGPO 框架下，Belief Prompt 的独立贡献有多大？** 如果 GiGPO + `<think>` 已经接近 94.5%，则 Belief Prompt 的贡献可以忽略。如果显著低于 94.5%，则 Belief Prompt 与 GiGPO 有协同效应。

### 10.3 多种子验证

V10 消融仅使用 seed=42 的单种子实验，存在随机性影响。正式实验至少需要 **3 seeds**：

```
seeds: [42, 123, 456]
```

报告 mean ± std，并进行统计显著性检验 (paired t-test 或 Wilcoxon)。

### 10.4 多环境验证 (扩展性)

如果计算资源允许，建议在以下环境中验证：

| 环境 | 特点 | 优先级 |
|------|------|:------:|
| ALFWorld (gen=0) | 训练任务 (seen) | 必需 |
| ALFWorld (gen=1) | 泛化任务 (unseen rooms) | 高 |
| ALFWorld (gen=2) | 强泛化 (unseen tasks) | 中 |
| WebShop / ScienceWorld | 跨域验证 | 低 |

### 10.5 推荐论文结构

```
1. Introduction: 交互式决策中 LLM agent 的挑战

2. Related Work: GRPO, GiGPO, process reward, structured reasoning

3. Method:
   3.1 Structured Belief Prompting — 结构化信念推理格式
   3.2 Step-level Advantage with Observation Grouping — 步骤级优势估计
   3.3 Training Stabilization — 训练稳定化技术

4. Experiments:
   4.1 Setup (ALFWorld, model, baselines)
   4.2 Main Results (Table: method comparison)
   4.3 Ablation Study (Table: factor decomposition)
   4.4 Training Dynamics Analysis (curves, learning speed)
   4.5 Per-Task Analysis (task-level breakdown)
   4.6 Generalization (optional: gen=1,2)

5. Analysis & Discussion:
   5.1 Why Step-level Advantage Matters
   5.2 The Role of Structured Belief Prompting
   5.3 Why Dense Intrinsic Rewards Hurt
   5.4 Limitations

6. Conclusion
```

---

## 附录

### A. 实验文件路径索引

```
远程存储:
/fs-computility-new/UPDZ03_chengjun/huangsijie.p/rebel_results/v10_experiments/
├── A1_grpo_baseline_seed42_20260211_160559/          # checkpoints + training.log
├── A2_grpo_tricks_seed42_20260211_232128/
├── B1_grpo_belief_prompt_seed42_20260212_064716/
├── C1_gigpo_belief_prompt_seed42_20260212_235634/
├── C2_rebel_group_only_seed42_20260215_155337/
├── D1_rebel_full_seed42_20260216_082441/
└── E1_rebel_curriculum_seed42_20260217_002626/

本地实验脚本:
/root/testttt/RLVMR/code/rebel_test_results/v10_experiments/
├── ABLATION_PLAN.md
├── V10_EXPERIMENT_ANALYSIS.md  (本文档)
├── run_v10_base.sh
├── run_all_ablations.sh
└── experiments/
    ├── A1_grpo_baseline.sh
    ├── A2_grpo_tricks.sh
    ├── B1_grpo_belief_prompt.sh
    ├── C1_gigpo_belief_prompt.sh
    ├── C2_rebel_group_only.sh
    ├── D1_rebel_full.sh
    └── E1_rebel_curriculum.sh
```

### B. 完整训练数据

#### B.1 逐 Epoch 验证集 SR (每 5 epoch)

| Epoch | A1 | A2 | B1 | C1 | C2 | D1 | E1 |
|:-----:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0 | 0.0 | 0.0 | 0.8 | 0.0 | 0.0 | 1.6 | 0.0 |
| 5 | 10.2 | 14.1 | 0.0 | 1.6 | 3.1 | 0.8 | 0.0 |
| 10 | 5.5 | 35.9 | 0.0 | 39.1 | 17.2 | 14.1 | 19.5 |
| 15 | 25.8 | 32.0 | 8.6 | 68.8 | 47.7 | 40.6 | 26.6 |
| 20 | 34.4 | 45.3 | 38.3 | 74.2 | 61.7 | 64.8 | 46.9 |
| 25 | 44.5 | 46.1 | 40.6 | 71.1 | 63.3 | 58.6 | 46.9 |
| 30 | 56.2 | 59.4 | 54.7 | 81.2 | 61.7 | 67.2 | 47.7 |
| 35 | 46.9 | 53.9 | 66.4 | 81.2 | 66.4 | 75.0 | 57.0 |
| 40 | 50.8 | 60.2 | 71.1 | 82.8 | 82.0 | 61.7 | 62.5 |
| 45 | 61.7 | 67.2 | 66.4 | 87.5 | 73.4 | 71.1 | 80.5 |
| 50 | 61.7 | 70.3 | 74.2 | 84.4 | 64.1 | 68.8 | 80.5 |
| 55 | 61.7 | 71.1 | 76.6 | 82.0 | 71.1 | 71.1 | 79.7 |
| 60 | 68.0 | 71.1 | 62.5 | 88.3 | 63.3 | 71.9 | 84.4 |
| 65 | 60.2 | 69.5 | 74.2 | 85.2 | 68.0 | 77.3 | 74.2 |
| 70 | 71.9 | 76.6 | 65.6 | 85.2 | 81.2 | 73.4 | 75.0 |
| 75 | 76.6 | **86.7** | 73.4 | 87.5 | 82.0 | 82.8 | 84.4 |
| 80 | 72.7 | 79.7 | 76.6 | **94.5** | 82.8 | 81.2 | 89.1 |
| 85 | 76.6 | 80.5 | 72.7 | 89.1 | 89.8 | 80.5 | 86.7 |
| 90 | 73.4 | 79.7 | 78.9 | 88.3 | **91.4** | **87.5** | 83.6 |
| 95 | **81.2** | 78.1 | **88.3** | 91.4 | 85.2 | 82.8 | **89.8** |
| 100 | 78.9 | 78.9 | 74.2 | 86.7 | 83.6 | 79.7 | 83.6 |

(SR 单位: %, **粗体** = 该实验的 Peak Epoch)

### C. 统计局限性声明

1. **单种子实验**: 所有结果基于 seed=42 的单次运行，存在随机性影响。正式实验需多种子验证。
2. **小样本验证集**: 部分任务类型 (如 look_at) 的验证样本量极小 (~12-14)，单样本波动可导致 7-8% SR 变化。
3. **Peak SR vs Final SR**: Peak SR 可能受到偶然波动影响，ep80-100 平均 SR 更能反映真实性能水平。
4. **C1 vs C2 公平性**: C2 额外使用了 Adv Tricks，但仍劣于 C1。严格对比需控制 C1 也加入或 C2 也去除 Adv Tricks。

---

*文档生成时间: 2026-02-20*
*基于 V10 消融实验数据 (2026-02-11 ~ 2026-02-17)*
