# PAPER_INDEX — ReBel/RLVMR 论文资源总索引

**更新日期**: 2026-03-14
**目标会议**: NeurIPS 2026
**论文主题**: ReBel: Belief-Enhanced Policy Optimization for Interactive LLM Agents

---

## 一、核心文档索引

| 文件 | 内容 | 重要性 |
|------|------|--------|
| `code/rebel_test_results/v11_final/ReBel_Experiment_Report.md` | **主实验报告（最全）**，V10/V11全部结果、图表索引、附录 | ⭐⭐⭐⭐⭐ |
| `code/rebel_test_results/v11_final/V11_FINAL_ALGORITHM_AND_EXPERIMENT_PLAN.md` | 算法设计、HiBO伪代码、GiGPO缺陷分析 | ⭐⭐⭐⭐⭐ |
| `code/rebel_test_results/v11_final/experiments_chapter_cn.md` | 论文实验章节中文草稿（可直接参考翻译） | ⭐⭐⭐⭐ |
| `code/rebel_test_results/v11_final/experiments_chapter.tex` | 论文实验章节 LaTeX 草稿 | ⭐⭐⭐⭐ |
| `code/rebel_test_results/v10_experiments/V10_EXPERIMENT_ANALYSIS.md` | V10 7组消融完整分析 | ⭐⭐⭐⭐ |
| `code/rebel_test_results/v11_final/EXPERIMENT_PROGRESS.md` | 实验进度追踪 | ⭐⭐⭐ |
| `code/ReBel论文实验完整规划.md` | 论文整体实验规划（含 baselines/metrics 设计） | ⭐⭐⭐ |
| `code/REBEL_EXPERIMENT_REPORT_PAPER.md` | 超参数研究（step_advantage_w）含 Valid Action Rate | ⭐⭐⭐ |
| `docs/reward_design.md` | WebShop 奖励设计 | ⭐⭐ |
| `docs/alfworld_prompt_guide.md` | ALFWorld Prompt 设计 | ⭐⭐ |
| `docs/webshop_prompt_guide.md` | WebShop Prompt 设计 | ⭐⭐ |
| `README.md` | 项目主页（含引用信息） | ⭐⭐ |

---

## 二、实验数据索引（ALFWorld 主环境）

### 2.1 V11 主实验（2026-02，seed=42，100 epochs）

**配置**: Qwen2.5-1.5B-Instruct SFT→RL，128个验证任务，8×A100
**远程路径**: `/fs-computility-new/UPDZ03_chengjun/huangsijie.p/rebel_results/v11_final/`

| 脚本编号 | 方法 | Peak SR | Final SR | valid_act_ratio(mean) | 状态 |
|---------|------|:-------:|:--------:|:---------------------:|------|
| M1 | GRPO Baseline | **82.0%** | 82.0% | 0.999 | ✅ 100ep |
| M3 | GiGPO (`<think>`) | **91.4%** | 82.0% | 0.996 | ✅ 100ep |
| *M4* | *GiGPO (`<belief>`)* | *94.5%* | *86.7%* | — | ⚠ 借用V10-C1 |
| **M5** | **ReBel Full** | **95.3%** | **89.1%** | 0.921 | ✅ 100ep |
| M5b | ReBel (Base, 无SFT) | 7.8% | — | — | ✅ 对照 |
| A1 | w/o HiBO (obs only) | **94.5%** | 91.4% | 0.911 | ✅ 100ep |
| A3 | w/o Belief Reward | **92.2%** | 91.4% | 0.884 | ✅ 100ep |
| A4 | w/o Adaptive Decay | **92.2%** | 92.2% | 0.889 | ⚠ 90ep(崩溃) |

**valid_action_ratio 逐步演变（step1→step100）**:
- M1 GRPO: 0.968 → 1.000（全程稳定高）
- M5 ReBel: 0.542 → 0.957（早期低，训练后期提升）
- A1 w/o HiBO: 0.537 → 0.918（类似ReBel）

**Per-Task Peak SR（128任务验证集）**:

| Task | M1 GRPO | M3 GiGPO | M5 ReBel | A1 w/oHiBO | A3 w/oReward | A4 w/oDecay |
|------|:-------:|:--------:|:--------:|:----------:|:------------:|:-----------:|
| pick_and_place | 91.7% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| pick_two_obj | 94.4% | 95.2% | **100.0%** | **100.0%** | **100.0%** | 92.0% |
| pick_heat | 91.7% | **100.0%** | **100.0%** | **100.0%** | **100.0%** | **100.0%** |
| pick_cool | 70.8% | 90.0% | **93.1%** | 91.7% | 87.0% | **95.8%** |
| pick_clean | 96.6% | **100.0%** | **100.0%** | **100.0%** | **100.0%** | **100.0%** |
| look_at | 87.5% | 78.9% | 87.5% | 87.5% | 87.5% | 87.5% |

**数据目录**: `code/rebel_test_results/v11_final/experiments/`
- `M5_rebel_full_seed42_20260220_161334/`
- `M5_rebel_full_seed42_20260221_084129/`
- `M1_grpo_baseline_seed42_20260222_152316/`
- `M3_gigpo_think_seed42_20260222_224131/`
- `A1_ablation_obs_only_seed42_20260223_093208/`

---

### 2.2 V10 系统消融实验（2026-02-11~17，seed=42，100 epochs）

**配置**: 同 V11，验证集 128任务

| ID | 方法 | Peak SR | Late-Avg(ep80-100) | Peak Epoch | 状态 |
|----|------|:-------:|:------------------:|:----------:|------|
| A1 | GRPO Baseline | 81.2% | 78.7% | ep95 | ✅ |
| A2 | GRPO + Training Tricks | 86.7% | 79.4% | ep75 | ✅ |
| B1 | GRPO + Belief Prompt | 88.3% | 77.4% | ep95 | ✅ |
| C1 | GiGPO + Belief Prompt (obs hash) | **94.5%** | **90.0%** | ep80 | ✅ |
| C2 | ReBel Group Only (belief hash) | 91.4% | 86.6% | ep90 | ✅ |
| D1 | ReBel Full (dense reward) | 87.5% | 84.3% | ep90 | ✅ |
| E1 | ReBel + Curriculum Decay | 89.8% | 86.6% | ep95 | ✅ |

**关键因素贡献量化（V10）**:

| 对比 | 隔离因素 | Peak SR 增益 | Late-Avg 增益 |
|----|----------|:-----------:|:-------------:|
| A2 vs A1 | Training Tricks | +5.5% | +0.7% |
| B1 vs A2 | Belief Prompting | +1.6% | −2.0% |
| **C1 vs B1** | **Step-Level Advantage** | **+6.2%** | **+12.6%** |
| C2 vs C1 | Belief Hash vs Obs Hash | −3.1% | −3.4% |
| D1 vs C2 | Dense Intrinsic Reward | −3.9% | −2.3% |
| E1 vs D1 | Curriculum Decay | +2.3% | +2.3% |

**Per-Task SR（Peak Epoch，%）**:

| Task | A1 | A2 | B1 | C1 | C2 | D1 | E1 |
|------|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| pick_and_place | 100.0 | 90.9 | 94.6 | 96.2 | 100.0 | 100.0 | 100.0 |
| pick_two_obj | 87.5 | 82.1 | 79.2 | **100.0** | 82.4 | 82.4 | 82.6 |
| pick_heat | 71.4 | 92.3 | 92.9 | 91.7 | 94.4 | 77.8 | **100.0** |
| pick_cool | 56.0 | 75.0 | 76.0 | 80.0 | **90.5** | 81.0 | 90.0 |
| pick_clean | 79.2 | **100.0** | **100.0** | **100.0** | 84.6 | 80.8 | 91.3 |
| look_at | 75.0 | 66.7 | 75.0 | **92.9** | 87.5 | **100.0** | 60.0 |

**数据目录**: `code/rebel_test_results/v10_experiments/`

---

### 2.3 超参数研究（V3时期，step_advantage_w，2025-12-31）

**来源**: `code/REBEL_EXPERIMENT_REPORT_PAPER.md`（含 Valid Action Rate）

| 实验 | step_adv_w | Train SR | Val SR | Valid Action % | Avg Steps |
|------|:----------:|:--------:|:------:|:--------------:|:---------:|
| Baseline | 1.0 | 58.2% | 43.8% | 94.1% | 19.8 |
| step_adv_0.5 | **0.5** | 52.7% | **62.5%** | 92.7% | 21.8 |
| step_adv_2.0 | 2.0 | 61.3% | 43.8% | 92.5% | 19.6 |
| step_adv_0.5_val128 | 0.5 | 88.7% | **70.3%** | 97.7% | 13.0 |

---

### 2.4 实际评测（eval）结果（2026-01-25，128任务）

**模型**: V8 Ablation No Belief (step100 checkpoint)
**来源**: `code/eval_results/ablation_no_belief_reward_step100/eval_20260125_042528/results.json`

| 指标 | 数值 |
|------|------|
| Success Rate | **83.59%** (107/128) |
| Avg Episode Length | 13.97 steps |
| Avg Episode Reward | 9.24 |

**Per-Task（eval）**:
- pick_heat: 100.0% (18/18)
- pick_and_place: 96.2% (25/26)
- pick_two_obj: 86.7% (26/30)
- pick_cool: 78.3% (18/23)
- pick_clean: 81.0% (17/21)
- **look_at: 30.0% (3/10)** ← 存在分布偏差

---

### 2.5 历史版本演进（Peak SR）

| 版本 | 核心改进 | Peak SR | look_at SR | 日期 |
|------|---------|:-------:|:----------:|------|
| Base Model (无SFT) | — | 0% | — | 2025-12 |
| V4-Exp1 | Task Status 分组 | 73.4% | — | 2026-01 |
| V6-Exp2 | 相对阈值归一化 | 78.1% | 83.3% | 2026-01 |
| V6-Exp3 | Full + KL in Reward | 84.4% | 16.7% | 2026-01 |
| V7 | Clip-Cov 熵保护 | 85.9% | 81.2% | 2026-01 |
| V8 Ablation No Belief | 移除信念奖励 | 90.6% | 58.3% | 2026-01 |
| V10-C1 | GiGPO + Belief Prompt | 94.5% | 92.9% | 2026-02 |
| **V11-ReBel Full** | **HiBO + Adaptive Curriculum** | **95.3%** | **87.5%** | **2026-02** |

---

## 三、实验数据索引（WebShop 副环境）

### 3.1 SFT 数据集
- `code/data/webshop_rebel_sft/train.parquet` — 500条，pass_rate=99.8%，2538步
- `code/data/webshop_rebel_sft_v2/train.parquet` — 500条，pass_rate=100%，2541步

### 3.2 测试数据
- `code/data/webshop_rebel_hindsight_test/rebel_hindsight.jsonl` — 10条
- `code/data/webshop_expert_traj/webshop_train.json`

### 3.3 RL 实验脚本（待运行）
- `code/rebel_test_results/v11_final/experiments/webshop/M1_webshop_grpo.sh`
- `code/rebel_test_results/v11_final/experiments/webshop/M3_webshop_gigpo.sh`
- `code/rebel_test_results/v11_final/experiments/webshop/M5_webshop_rebel_full.sh`

### 3.4 WebShop RL 实验结果（M5 ReBel Full，3 seeds）

**远程路径**: `/fs-computility-new/UPDZ03_chengjun/huangsijie.p/rebel_results/v11_final_webshop/`

| 实验 | seed | Peak SR | Final SR | n_evals | valid_act_ratio |
|------|------|:-------:|:--------:|:-------:|:--------------:|
| M5 ReBel Full | 42 | **75.4%** | 72.8% | 21 (100ep) | 0.994 |
| M5 ReBel Full | 123 | 72.4% | **72.4%** | 21 (100ep) | 0.992 |
| M5 ReBel Full | 456 | 74.2% | 67.8% | 21 (100ep) | — |

**仅有 M5（ReBel Full）结果，缺少 M1（GRPO）和 M3（GiGPO）baselines。**

---

## 四、论文图表索引

**目录**: `code/rebel_test_results/v11_final/paper_figures/`（11张图，PDF+PNG双格式）

| 编号 | 文件名 | 内容 | 用于章节 |
|------|--------|------|---------|
| fig1 | `fig1_training_curves_v10.{pdf,png}` | V10 7组消融训练曲线 | §Experiments |
| fig2 | `fig2_main_results_comparison.{pdf,png}` | 主实验方法对比 | §Experiments |
| fig3 | `fig3_pertask_success_rate.{pdf,png}` | Per-Task SR | §Analysis |
| fig4 | `fig4_factor_contribution.{pdf,png}` | 因素贡献柱状图 | §Analysis |
| fig5 | `fig5_hibo_coverage.{pdf,png}` | HiBO 覆盖率分析 | §Method |
| fig6 | `fig6_belief_decay_curves.{pdf,png}` | 自适应衰减曲线 | §Method |
| fig7 | `fig7_ablation_study.{pdf,png}` | 消融实验图 | §Experiments |
| fig8 | `fig8_learning_speed.{pdf,png}` | 学习速度对比 | §Analysis |
| fig9 | `fig9_algorithm_overview.{pdf,png}` | 算法框架图 | §Method |
| fig10 | `fig10_version_evolution.{pdf,png}` | 版本演进曲线 | §Intro/§Analysis |
| fig11 | `fig11_singleton_problem.{pdf,png}` | GiGPO单样本组问题图 | §Motivation |

---

## 五、核心代码索引

| 文件 | 内容 |
|------|------|
| `code/rebel/hibo_grouping.py` | HiBO 分组核心（ALFWorld+WebShop双适配） |
| `code/rebel/core_rebel.py` | ReBel 核心算法（信念哈希、4组件内在奖励、双层优势） |
| `code/agent_system/environments/env_package/alfworld/belief_tracker.py` | ALFWorld 信念跟踪+奖励计算 |
| `code/agent_system/environments/env_package/webshop/belief_tracker.py` | WebShop 信念跟踪+奖励计算 |
| `code/verl/trainer/ppo/ray_trainer.py` | 训练主循环（HiBO advantage 分支） |
| `code/agent_system/multi_turn_rollout/rollout_loop.py` | Rollout 循环（belief_abstract 字段） |

---

## 六、引用与相关工作

**本项目引用信息**（来自 README.md）:
```
@article{zhang2025rlvmr,
  title={RLVMR: Reinforcement Learning with Verifiable Meta-Reasoning Rewards for Robust Long-Horizon Agents},
  author={Zhang, Zijing and Chen, Ziyang and Li, Mingxiao and Tu, Zhaopeng and Li, Xiaolong},
  journal={arXiv preprint arXiv:2507.22844},
  year={2025}
}
```

**需要引用的关键 baselines**:
- GiGPO (我们的 prior work)
- GRPO (Shao et al., 2024)
- PPO (Schulman et al., 2017)
- RLOO (Ahmadian et al., 2024)
- ReAct (Yao et al., 2023)
- Reflexion (Shinn et al., 2024)
- RAGEN (Wang et al., 2025)
- ALFWorld benchmark
- WebShop benchmark
- Qwen2.5 (base model)
- veRL (training infrastructure)
