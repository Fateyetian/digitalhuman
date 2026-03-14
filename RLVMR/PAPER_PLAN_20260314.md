# 论文撰写总体规划

**创建日期**: 2026-03-14
**目标会议**: NeurIPS 2026
**论文主题**: RLVMR / ReBel — Belief-Enhanced Policy Optimization for Interactive LLM Agents
**主环境**: ALFWorld（主要结果）
**副环境**: WebShop（泛化性验证）

---

## 执行阶段

### ✅ 阶段零：规划确认（已完成）

### 🔄 阶段一：资料索引与数据清点
**目标**: 建立 `PAPER_INDEX.md`，精确索引所有可用资源

- [ ] 扫描所有实验结果目录（v8~v11, ablations, webshop, alfworld）
- [ ] 提取所有已完成实验的关键数据（SR、训练曲线、消融结果）
- [ ] 列出所有已有文档/报告（V11算法设计、ReBel规划、超参研究等）
- [ ] 整理 baseline 数据（GiGPO、GRPO、RLOO 等对比数据）
- [ ] 整理 WebShop 相关实验数据

### 🔄 阶段二：实验缺口分析
**目标**: 对照论文三大创新点，诊断哪些实验已有数据、哪些缺失

三大创新点 × 所需实验：

| 创新点 | 必要实验 | 目标 |
|--------|---------|------|
| ① 密集信念奖励加速收敛 | vs GRPO/GiGPO 对比 + 训练曲线 | ALFWorld L0/L1/L2 |
| ② O(1)上下文效率 | Token消耗对比（ReBel vs 全历史方法） | ALFWorld + WebShop |
| ③ 信念漂移缓解 | Valid Action Rate、Repeat Rate、信念准确率 | ALFWorld |

消融实验需求：
- A1: 无观测奖励（obs only）
- A2: 无信念奖励（belief only）
- A3: 无信念 prompt
- A4: 固定 decay（非自适应）
- A5: 均匀 decay

对比 baseline 需求：
- M1: GRPO baseline
- M2: GRPO + tricks
- M3: GiGPO + think
- M4: GiGPO + belief（无ReBel奖励）
- M5: ReBel full

### 🔄 阶段三：NeurIPS 论文撰写

按 NeurIPS 标准结构：
```
1. Abstract
2. Introduction（问题 + 3点贡献）
3. Related Work（RL for agents / LLM agents / dense rewards / belief tracking）
4. Method（ReBel算法：三重信念增强）
5. Experiments
   5.1 实验设置
   5.2 主结果（ALFWorld L0/L1/L2 vs baselines）
   5.3 泛化性（WebShop）
   5.4 消融研究
   5.5 分析（训练曲线、Token效率、信念漂移）
6. Conclusion
```

### 🔄 阶段四：补实验建议

- 列出**硬门槛实验**（缺了论文无法发表）
- 列出**加分实验**（提升接受率）
- 按优先级排序，估算工作量

---

## 关键约束

- **NeurIPS 2026 预计截稿**: 2026年5月下旬（Abstract due ~5/15, Full paper ~5/22）
- **主体环境**: ALFWorld — L0 seen / L1 novel combo / L2 novel task
- **泛化验证**: WebShop — 体现方法通用性
- **模型规模**: Qwen2.5-{1.5B, 3B, 7B}-Instruct（至少两个规模）

---

## 输出文件清单

| 文件 | 内容 | 状态 |
|------|------|------|
| `PAPER_PLAN_20260314.md` | 本规划文件 | ✅ |
| `PAPER_INDEX.md` | 所有资源索引 | 待创建 |
| `PAPER_GAP_ANALYSIS.md` | 实验缺口分析 | 待创建 |
| `paper_draft/main.tex` | 论文正文（LaTeX） | 待创建 |
| `paper_draft/sections/` | 各章节文件 | 待创建 |
