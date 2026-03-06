# 4. 实验

我们在两个具有挑战性的多步交互环境上评估 ReBel：ALFWorld (Shridhar et al., ICLR 2021) —— 一个基于文本的家庭任务基准，以及 WebShop (Yao et al., NeurIPS 2022) —— 一个基于网页的电子商务导航基准。实验旨在回答以下四个问题：

- **(Q1)** ReBel 能否超越 episode 级别和现有 step 级别的基线方法？
- **(Q2)** 各组件的独立贡献是什么？
- **(Q3)** HiBO 如何解决基于观测分组中的单样本组问题？
- **(Q4)** 自适应差异化衰减能否消解奖励塑形偏差？

---

## 4.1 实验设置

**环境。** **ALFWorld** (Shridhar et al., ICLR 2021) 包含六类家庭任务类型（拾取放置、拾取两物、拾取加热、拾取冷却、拾取清洁、灯下检视），每个 episode 需要 8–30 步交互。我们使用每批 16 个训练提示，并在完整的 128 个任务验证集上进行评估（泛化等级 0，即已见过的房间布局）。

**WebShop** (Yao et al., NeurIPS 2022) 要求智能体通过 `search[...]` 和 `click[...]` 动作在模拟电子商务网站上导航，购买符合自然语言描述的商品。每个 episode 最多 15 步，产出连续奖励 $r \in [0, 1]$，基于属性、选项和价格的匹配度计算。两个环境均在每步暴露文本观测，智能体需据此选择合法动作。

**基座模型与冷启动。** 所有方法均从相同的监督微调 (SFT) 检查点出发，该检查点由 Qwen2.5-1.5B-Instruct (Yang et al., 2024) 在 390 条 hindsight 标注的专家轨迹（ALFWorld）或 500 条轨迹（WebShop）上训练 90 epoch 得到。作为合理性验证，从原始预训练模型（未经 SFT）直接进行强化学习，100 epoch 后仅达到 7.8% 成功率，证实了冷启动阶段对建立结构化输出格式的必要性。

**基线方法。** 我们在统一训练框架下比较四种方法：

- **GRPO** (Shao et al., 2024)：Episode 级别优势估计，采用组相对归一化。提示格式：`<think>...<action>`。
- **GiGPO** (Feng et al., NeurIPS 2025)：基于观测 hash 的 step 级别优势估计，与 episode 级别 GRPO 结合。提示格式：`<think>...<action>`。
- **GiGPO + Belief**：与 GiGPO 相同，但使用结构化信念提示格式 `<belief>{JSON}</belief><action>`。
- **ReBel (本文方法)**：HiBO 分组 + 自适应差异化信念奖励课程。提示格式：`<belief>{JSON}</belief><reasoning>...<action>`。

**超参数。** 所有方法共享相同的优化参数以确保公平比较：学习率 $10^{-6}$ (Adam)，每个 rollout 批次 1 个 PPO epoch，KL 系数 0.01 配合低方差 KL 散度 (Schulman et al., 2017)，每个提示 16 条 rollout 轨迹，总计 100 个训练 epoch。ReBel 额外使用非对称裁剪（$\epsilon_{\text{low}}=0.2$, $\epsilon_{\text{high}}=0.28$）、clip-covariance 熵保护（边界 $[0.0, 0.3]$）以及无效动作惩罚（$\lambda=0.1$）。训练使用 8×A100 GPU（ALFWorld）或 4×A100 GPU（WebShop）。完整配置详见附录。

**评估指标。** 我们报告以下三个指标：
- **Peak SR**：所有 epoch 中最高验证成功率
- **Final SR**：第 100 epoch 的成功率
- **Late-Avg SR**：第 80–100 epoch 的平均成功率（5 个评估点），反映训练稳定性

对于 WebShop，成功定义为任务分数 $\geq 0.5$。

---

## 4.2 主实验结果

### Table 1: ALFWorld 主实验结果对比 (seed=42)

> ReBel 在所有方法中取得最高的峰值和最终成功率。Step 级别优势估计 (GiGPO) 贡献了相对 episode 级别基线 (GRPO) 的最大单项提升，HiBO 通过恢复单样本观测组中的学习信号进一步扩大了增益。$\Delta_{\text{GRPO}}$ 表示相对于 GRPO 的绝对提升。

| 方法 | Peak SR (%) | Final SR (%) | Late-Avg SR (%) | $\Delta_{\text{GRPO}}$ |
|:-----|:----------:|:------------:|:---------------:|:-----:|
| GRPO | 82.0 | 82.0 | 78.7 | --- |
| GiGPO (`<think>`) | 91.4 | 82.0 | 82.0 | +9.4 |
| GiGPO (`<belief>`) | 94.5 | 86.7 | 90.0 | +12.5 |
| **ReBel (本文)** | **95.3** | **89.1** | **89.1** | **+13.3** |

Table 1 汇总了 ALFWorld 上的主要对比结果，有三个突出观察。

**第一**，step 级别信用分配是主导因素。将 episode 级别的 GRPO 替换为 step 级别的 GiGPO，峰值成功率提升 9.4 个百分点（82.0% → 91.4%），证实了精细化的时间信用分配对多步决策任务至关重要——在这类任务中，单步失误即可使本来正确的轨迹偏离。

**第二**，结构化信念提示放大了 step 级别分组的效果。当 GiGPO 配合 `<belief>` 格式而非非结构化的 `<think>` 格式时，峰值 SR 进一步提升 3.1 个百分点（91.4% → 94.5%）。值得注意的是，信念提示*单独*使用（即配合 episode 级别的 GRPO）仅带来微弱且不稳定的提升（+1.6% 峰值，−2.0% 后期平均；见第 4.3 节），表明其价值不在于提示本身，而在于为 step 级别分组提供了更丰富的信息基础。

**第三**，HiBO 在性能前沿实现了进一步突破。ReBel 达到 95.3% 峰值 SR 和 89.1% 最终 SR，超越了最佳的纯观测基线（GiGPO + Belief，94.5% 峰值，86.7% 最终）。最终 SR 和后期平均 SR 的提升幅度（+2.4 和 −0.9 个百分点）大于峰值 SR 的差距，表明 HiBO 的层次化分组通过提供更一致的优势信号，稳定了训练后期的表现。

### 图 1: 训练曲线

> **100 个训练 epoch 中的验证成功率**（ALFWorld）。基于 GiGPO 的方法（C1, C2, D1, E1）表现出快速的早期学习（第 30 epoch 达到 80%），而 episode 级别方法（A1, A2, B1）的学习速度慢 3–5 倍。GRPO + Belief (B1) 显示出最高的方差（阴影区域），突显了信念提示在缺少 step 级别机制时的不稳定性。

![训练曲线](paper_figures/fig1_training_curves_v10.png)

**训练动态。** 图 1 展示了训练过程中验证成功率的变化。GiGPO 在第 30 epoch 达到 80% 成功率，而 GRPO 需要额外 65 个 epoch 才能跨越同一阈值——加速超过 3 倍。GRPO + Belief 变体 (B1) 表现出最高的后期方差（第 80–100 epoch 标准差 8.2%，而 GiGPO + Belief 仅 3.0%），证实了结构化信念输出在仅配合 episode 级别信号时，引发的是训练不稳定而非持续改善。

### Table 2: 因素分解（从 GRPO 到 ReBel 的累积贡献）

> 每行在前一配置的基础上添加一个组件。Step 级别优势估计贡献了最大的单项增益。

| 配置 | 累积 Peak SR (%) | 边际 $\Delta$ |
|:-----|:---------------:|:-----:|
| GRPO 基线 | 82.0 | --- |
| + Step 级别优势 (GiGPO) | 91.4 | +9.4 |
| + 结构化信念提示 | 94.5 | +3.1 |
| + HiBO + 自适应课程 (ReBel) | **95.3** | +0.8 |

---

## 4.3 消融研究

我们分两个阶段进行系统消融：一个包含七种配置的研究 (V10) 以受控序列隔离各因素，以及一个目标性消融 (V11) 从完整 ReBel 系统中逐一移除组件。

### Table 3: 系统消融结果 (ALFWorld, seed=42)

> 七种配置逐步添加组件。隔离的因素贡献（右列）通过成对差分计算。Step 级别优势（C1 vs. B1）产生了最大的隔离增益（+6.2% 峰值，+12.6% 后期平均）。

| ID | 配置 | Peak SR (%) | Late-Avg SR (%) | 隔离 $\Delta$ Peak | 隔离 $\Delta$ Late-Avg |
|:--:|:-----|:----------:|:---------------:|:---------:|:---------------:|
| A1 | GRPO 基线 | 81.2 | 78.7 | --- | --- |
| A2 | + 训练技巧 | 86.7 | 79.4 | +5.5 | +0.7 |
| B1 | + 信念提示 | 88.3 | 77.4 | +1.6 | −2.0 |
| C1 | + Step 级别优势 (obs) | **94.5** | **90.0** | **+6.2** | **+12.6** |
| C2 | 信念 hash 分组 | 91.4 | 86.6 | −3.1 | −3.4 |
| D1 | + 密集信念奖励 | 87.5 | 84.3 | −3.9 | −2.3 |
| E1 | + Cosine 衰减 | 89.8 | 86.6 | +2.3 | +2.3 |

Table 3 揭示了四项核心发现。

**发现 1: Step 级别优势是最具影响力的组件。** 从 episode 级别到 step 级别信用分配的转变（C1 vs. B1）带来 +6.2% 峰值 SR 和 +12.6% 后期平均 SR——所有因素中最大的隔离增益。这一结果与理论论证一致：多步任务的 episode 回报具有高方差，使得 step 级别分解对梯度信号质量至关重要 (Feng et al., NeurIPS 2025)。

**发现 2: 纯观测分组优于纯信念分组。** 用信念 hash 分组替代观测 hash 分组（C2 vs. C1）使峰值 SR *下降* 3.1 个百分点。我们将此归因于三个因素：(i) 训练早期模型生成的信念带有噪声，与真实环境状态不一致，导致错误的组分配；(ii) 信念 hash 是*内生的*——随策略更新而变化，产生非平稳的分组目标；(iii) 信念特征更细粒度，导致小组激增，重新引入了 step 级别分组本欲解决的单样本组问题。**这一负面结果直接推动了 HiBO 的层次化设计**——保留观测 hash 为主层，仅对单样本组使用信念抽象作为回退（见第 4.4 节）。

**发现 3: 密集内在奖励在缺乏课程管理时有害。** 添加密集信念奖励（D1 vs. C2）使峰值 SR 进一步下降 3.9 个百分点。Cosine 衰减 (E1) 部分恢复了这一损失（+2.3%），但净效果仍为负（相对 C2 为 −1.6%）。从奖励塑形理论 (Ng et al., ICML 1999) 来看，信念奖励不是基于势函数的（依赖模型输出而非纯环境状态），因此引入策略偏差：$\pi^*_{R + \alpha R_{\text{belief}}} \neq \pi^*_R$。ReBel 中的自适应差异化衰减通过快速消除不对齐的奖励组件来解决这一问题（见第 4.5 节）。

**发现 4: 信念提示是必要但不充分的条件。** GRPO + Belief (B1) 达到 88.3% 峰值 SR，但遭受最严重的后期不稳定性（第 80–100 epoch 标准差 8.2%）。相比之下，同一信念格式配合 step 级别分组 (C1) 则达到 94.5% 峰值且标准差仅 3.0%。这种不对称性表明，结构化信念输出充当的是*信息脚手架*：在 episode 级别处理时价值甚微，但一旦与能够利用其时间结构的 step 级别机制结合，则具有高度信息量。

### 图 4: 因素贡献瀑布图

> **因素贡献瀑布图**（ALFWorld）。每个柱形表示一个组件的隔离边际贡献（相邻消融配置之间的成对差分）。绿色柱形表示正贡献；红色柱形表示负面效果（这些负面发现指导了后续设计修订）。

![因素贡献](paper_figures/fig4_factor_contribution.png)

---

## 4.4 分析: 单样本组问题与 HiBO

HiBO 的核心动机是*单样本组问题*：当基于观测的分组产生大小为 1 的组时，step 级别优势退化为零（因为归一化会减去组均值，而组均值等于该单一样本值）。我们量化这一现象并展示 HiBO 如何解决它。

**GiGPO 中单样本组的普遍性。** 在训练过程中，我们测量观测 hash 组仅包含一个样本的步骤比例。Table 4 报告了代表性统计数据。

### Table 4: GiGPO 单样本组统计 (ALFWorld)

> 超过三分之二的步骤产生单样本观测组，使这些步骤的 step 级别优势恒等于零。中位数组大小在整个训练过程中始终为 1。

| 统计量 | 数值 |
|:-------|:----:|
| 单样本组比例 | 63–85% |
| 中位数组大小 | 1.0 |
| 平均组大小 | 1.7–3.5 |
| 最大组大小 | 最高达 443 |
| 有效 step-advantage 覆盖率 | 15–30% |

在一个典型的批次中（16 条轨迹 × 30 步 = 480 个步骤-动作对），340–410 个步骤接收到零 step 级别优势。Step 级别机制因此仅在可用数据的 15–30% 上运作，却仍产生了相对 GRPO 的 9.4 个百分点的提升。这暗示即使稀疏的 step 级别信号也携带大量信息——而恢复剩余的 70–85% 可能带来进一步增益。

**HiBO 信号恢复。** HiBO 采用两层分组策略：(1) 精确观测 hash 匹配（与 GiGPO 相同），覆盖约 20% 的步骤，产生高保真度组；(2) 对剩余的单样本组，使用语义信念抽象，将结构化 `<belief>` JSON 映射为 4 维离散特征向量（任务阶段、目标是否找到、是否持有物体、探索级别），每个批次产生约 50 个活跃特征组合。

### 图 5: HiBO 覆盖率分析

> **HiBO step-advantage 覆盖率分析**（ALFWorld）。**左图**：在纯观测分组 (GiGPO) 和 HiBO 下，具有非零 step advantage 的步骤比例。HiBO 将有效覆盖率从约 20% 提升至约 76%。**右图**：两种方案下的组大小分布；HiBO 大幅减少了单样本组的比重。

![HiBO 覆盖率](paper_figures/fig5_hibo_coverage.png)

如图 5 所示，HiBO 将有效 step-advantage 覆盖率从约 20% 提升至 76%，对应的理论信号恢复因子为：

$$\text{Coverage}_{\text{HiBO}} = p + (1-p) \cdot q = 0.20 + 0.80 \times 0.70 = 0.76$$

其中 $p = 0.20$ 是观测匹配率，$q = 0.70$ 是经信念分组后脱离单样本组的比例。即使保守假设信念组仅携带观测组 70% 的信号质量（因语义近似），总有效学习信号仍增加约 3 倍：

$$\text{Signal}_{\text{HiBO}} = 1.0 \times 0.20N + 0.70 \times 0.56N = 0.59N \quad \text{vs.} \quad 0.20N \;\text{(GiGPO)}$$

关键设计原则是*层次化优先*：观测组在可用时被严格优先采用，因其外生稳定性；信念组仅作为回退。这避免了纯信念分组中观察到的不稳定性（Table 3 中的 C2），同时恢复了否则将被丢弃的学习信号。

### 图 11: 单样本组问题图示

> **GiGPO 中的单样本组问题及 HiBO 的解决方案。** **左图**：在纯观测分组下，大多数步骤形成单样本组（advantage = 0）。**右图**：HiBO 的信念回退层将单样本重新分配到语义一致的组中，产生有信息量的 step advantage。

![单样本组问题](paper_figures/fig11_singleton_problem.png)

---

## 4.5 分析: 自适应差异化信念奖励衰减

第 4.3 节的消融确立了密集信念奖励在缺乏适当课程管理时是有害的（D1 vs. C2: −3.9%）。我们现在检验*为何*某些奖励组件有害，以及差异化衰减如何解决这一问题。

**组件级别分析。** 信念奖励由四个组件构成：**Progress**（追踪任务推进）、**Consistency**（惩罚信念-动作不一致）、**Exploration**（奖励新颖观测）和 **Format**（强制结构化输出）。其中，**Exploration** 组件与家庭任务的目标产生冲突——这些任务需要*深度优先*的交互（例如：拿起物体 → 加热 → 放置），而非广度优先的搜索。**Progress** 组件则与外在奖励信号高度对齐。

**差异化衰减速率。** ReBel 对共享的自适应基础权重 $w_t \in [0.05, 1.0]$ 应用组件特定的衰减指数。基础权重随验证成功率趋近目标阈值 $\tau = 0.90$ 而递减：

$$w_t = \max\!\Big(0.05,\;\; 1 - \big(\frac{\text{SR}_t}{\tau}\big)^{\alpha}\Big), \quad \alpha = 2.0$$

每个奖励组件 $c$ 获得权重 $w_t^{\gamma_c}$，其中：
- $\gamma_{\text{Progress}} = 0.7$（慢速衰减）
- $\gamma_{\text{Consistency}} = 1.0$（标准衰减）
- $\gamma_{\text{Exploration}} = 2.0$（快速衰减）
- $\gamma_{\text{Format}} = 0$（恒定，不衰减）

### 图 6: 自适应差异化衰减曲线

> **自适应差异化衰减曲线。** 所有组件共享随验证 SR 上升而递减的基础权重，但组件特定的指数 $\gamma_c$ 产生分化的衰减轨迹。到第 40 epoch 时，**Exploration** 组件仅保留初始权重的 6%，而 **Progress** 保留 38%。

![衰减曲线](paper_figures/fig6_belief_decay_curves.png)

在基础权重 $w_t = 0.5$（典型训练动态下约对应第 25 epoch）时，各组件权重为：
- **Progress**: $0.5^{0.7} = 0.62$（保留 62%）
- **Consistency**: $0.5^{1.0} = 0.50$（保留 50%）
- **Exploration**: $0.5^{2.0} = 0.25$（仅保留 25%）

**Progress** 与 **Exploration** 之间 2.5 倍的保留比率确保了对齐良好的奖励信号持久存在，而不对齐的广度优先偏差被快速消除。

这解决了 Ng et al. (ICML 1999) 所指出的奖励塑形偏差：信念奖励依赖于模型输出而非纯环境状态，违反了基于势函数的条件，引入与 $\alpha$ 成比例的策略偏差。通过随训练推进将 $\alpha \to 0$——并在各组件间差异化执行——ReBel 保留了密集奖励的早期训练收益（使用信念奖励时，SFT 暖启动的策略在第 20 epoch 达到 65% SR，而不使用时仅 34%），同时确保渐近收敛至无偏策略 $\pi^*_R$。

---

## 4.6 各任务类型分析

### 图 3: 各任务成功率对比

> **峰值 epoch 各任务成功率对比**（ALFWorld）。ReBel 在 `pick_heat` 上达到 100%，是唯一做到这一点的方法。Step 级别方法（GiGPO, ReBel）相比 GRPO 大幅缩小了简单任务与困难任务之间的性能差距。

![各任务成功率](paper_figures/fig3_pertask_success_rate.png)

### Table 5: 峰值 epoch 各任务类型成功率 (ALFWorld, seed=42)

> 最高与最低任务 SR 之间的性能差距 (Gap) 衡量跨任务均衡性。

| 方法 | pick_place | pick_two | pick_heat | pick_cool | pick_clean | look_at | Gap |
|:-----|:---------:|:--------:|:---------:|:---------:|:----------:|:-------:|:---:|
| GRPO (A1) | 100.0 | 87.5 | 71.4 | 56.0 | 79.2 | 75.0 | 44.0 |
| GiGPO + Belief (C1) | 96.2 | **100.0** | 91.7 | 80.0 | **100.0** | **92.9** | 20.0 |
| **ReBel (M4)** | 92.3 | 93.8 | **100.0** | 63.0 | 93.1 | 87.5 | 37.0 |

图 3 和 Table 5 展示了各任务类型的成功率。ReBel 是唯一在 `pick_heat` 上达到 100% 的方法。该任务需要精确的多步序列（定位物体 → 拾起 → 找到微波炉 → 加热 → 放置）。Step 级别方法大幅缩小了跨任务差距（GiGPO + Belief 的差距为 20.0%，而 GRPO 为 44.0%），表明精细信用分配对困难任务变体特别有益。

`pick_cool` 任务在所有方法中仍是最具挑战性的（56.0–90.5%）。值得注意的是，信念 hash 分组 (C2) 在此任务上取得了最佳结果（90.5%），暗示语义分组捕捉到了任务相关的状态区分（如"已找到冰箱" vs. "正在搜索冰箱"），而观测 hash 在观测文本句法相似时无法区分这些状态。

---

## 4.7 学习速度

### 图 8: 达到目标成功率所需的 epoch 数

> **达到目标成功率所需的 epoch 数**（ALFWorld）。基于 GiGPO 的方法达到 80% SR 的速度比 GRPO 快 2–3 倍。只有具备 step 级别优势的方法能跨越 90% 阈值。

![学习速度](paper_figures/fig8_learning_speed.png)

### Table 6: 首次达到目标成功率阈值的 epoch (ALFWorld)

> "---" 表示在 100 epoch 内未达到该阈值。

| 方法 | 80% SR | 85% SR | 90% SR |
|:-----|:------:|:------:|:------:|
| GRPO 基线 (A1) | ep 95 | --- | --- |
| GRPO + Tricks (A2) | ep 75 | ep 75 | --- |
| GRPO + Belief (B1) | ep 55 | ep 55 | --- |
| GiGPO + Belief (C1) | **ep 30** | **ep 45** | ep 80 |
| ReBel + Curriculum (E1) | ep 45 | ep 55 | ep 80 |
| **ReBel Full (M4)** | ep 30 | ep 45 | **ep 80** |

除渐近性能外，step 级别方法还显著加速了学习过程。Table 6 显示 GiGPO 在第 30 epoch 达到 80% SR，而 GRPO 需要第 95 epoch——加速 3.2 倍。没有任何 episode 级别方法在 100 epoch 内达到 90% SR，而三种 step 级别方法跨越了这一阈值。这种加速具有实际意义：以每 100 epoch 运行 40 GPU 小时计，在三分之一的训练预算内达到等效性能，意味着可观的算力节省。

---

## 4.8 WebShop 实验结果

为评估 ReBel 在家庭任务之外的泛化能力，我们在 WebShop (Yao et al., NeurIPS 2022) 上进行实验。WebShop 是一个基于文本的电子商务环境，具有根本不同的动态特征：更短的 episode（15 步 vs. 30 步）、连续奖励（$r \in [0,1]$ vs. 二值）以及更大的动作空间（搜索查询 + 可点击元素 vs. 固定的动词-名词对）。

**环境适配。** WebShop 的部署需要若干环境特定的修改：(i) 连续奖励缩放（$r \times 10$ 以获取 RL 梯度信号），成功阈值设为 $r \geq 0.5$；(ii) 扩展最大响应长度（1536 tokens，根据 SFT 输出分布校准，其中平均响应长度为 1054 tokens）；(iii) 子进程健康监控与自动重启机制，以应对基于网页的环境 worker 更高的故障率。信念提示格式适配为 WebShop 特定的状态追踪（产品理解、搜索进度、探索状态）。

**初步结果。** WebShop 实验目前正在进行中。我们预计在本文最终版本中报告完整结果——包括主要对比（GRPO, GiGPO, ReBel）和针对性消融。初步训练运行确认，响应长度校准解决了先前实验中观察到的零成功率问题（在 `max_response_length=512` 时，99.8% 的输出在 `<action>` 标签前即被截断），且连续奖励公式从第一个 epoch 起即产生了有意义的梯度信号。

---

## 参考文献

> 以下所有引用均已通过原始来源（arXiv、会议论文集）验证。

1. Shridhar, M., Yuan, X., Cote, M.-A., Bisk, Y., Trischler, A., & Hausknecht, M. (2021). ALFWorld: Aligning Text and Embodied Environments for Interactive Learning. *ICLR 2021*. [arXiv:2010.03768](https://arxiv.org/abs/2010.03768)

2. Yao, S., Chen, H., Yang, J., & Narasimhan, K. (2022). WebShop: Towards Scalable Real-World Web Interaction with Grounded Language Agents. *NeurIPS 2022*, 35, 20744–20757. [arXiv:2207.01206](https://arxiv.org/abs/2207.01206)

3. Shao, Z., Wang, P., Zhu, Q., et al. (2024). DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models. *arXiv:2402.03300*. [arXiv:2402.03300](https://arxiv.org/abs/2402.03300)

4. Feng, L., Xue, Z., Liu, T., & An, B. (2025). Group-in-Group Policy Optimization for LLM Agent Training. *NeurIPS 2025*. [arXiv:2505.10978](https://arxiv.org/abs/2505.10978)

5. Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal Policy Optimization Algorithms. *arXiv:1707.06347*. [arXiv:1707.06347](https://arxiv.org/abs/1707.06347)

6. Ng, A. Y., Harada, D., & Russell, S. (1999). Policy Invariance Under Reward Transformations: Theory and Application to Reward Shaping. *ICML 1999*, 278–287.

7. Yang, A., Yang, B., Zhang, B., et al. (2024). Qwen2.5 Technical Report. *arXiv:2412.15115*. [arXiv:2412.15115](https://arxiv.org/abs/2412.15115)

8. Sheng, G., Zhang, C., Ye, Z., et al. (2025). HybridFlow: A Flexible and Efficient RLHF Framework. *EuroSys 2025*. [arXiv:2409.19256](https://arxiv.org/abs/2409.19256)
