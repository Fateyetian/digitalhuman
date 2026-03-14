# ALFWorld Prompt 说明

> 2026-03-07 | ReBel V11 (`ReBel_v2_improvements`)

---

## 文件位置

```
code/agent_system/environments/
├── prompts/
│   ├── rebel_prompts.py     # ReBel 标注 Prompt + ReBel SFT Prompt（当前主流）
│   └── cold_start.py        # 早期 SFT Prompt（MC 四模式，已被 rebel_prompts 取代）
│
└── env_package/alfworld/
    └── alfworld_rebel_prompt.py   # RL 训练阶段实际调用的模板
```

---

## 各阶段 Prompt 对照

| 阶段 | 模板变量 | 文件 | 调用入口 |
|------|---------|------|---------|
| 标注（ReBel） | `ALFWORLD_REBEL_TAGGING_TEMPLATE` | `rebel_prompts.py` | `generate_rebel_hindsight.py` |
| 标注（MC） | `ALFWORLD_TAGGING_TEMPLATE` | `cold_start.py` | `alfworld_prepare.py` |
| 标注（BDRS） | `ALFWORLD_TAGGING_TEMPLATE_BDRS` | `cold_start.py` | `alfworld_prepare.py` |
| SFT 第 1 步 | `ALFWORLD_REBEL_TEMPLATE_NO_HIS_CS` | `rebel_prompts.py` | `regenerate_coldstart_clean.py` |
| SFT 后续步 | `ALFWORLD_REBEL_TEMPLATE_CS` | `rebel_prompts.py` | `regenerate_coldstart_clean.py` |
| RL 第 1 步 | `ALFWORLD_TEMPLATE_NO_HIS_REBEL` | `alfworld_rebel_prompt.py` | `env_manager.py:394` |
| RL 后续步 | `ALFWORLD_TEMPLATE_REBEL` | `alfworld_rebel_prompt.py` | `env_manager.py:395` |
| RL Planning | `ALFWORLD_PLANNING_PROMPT_REBEL` | `alfworld_rebel_prompt.py` | `env_manager.py:668` |

---

## Q: 为什么标注 Prompt 没有 Action Selection？

标注是 **Hindsight Annotation（事后标注）**，动作已知，不需要选择：

```
输入 = observation + ground_truth_action（已知）+ prev_belief
输出 = belief_update + reasoning（倒推思维过程）
```

Prompt 中明确写明：
```
The next action to be taken is: `{ground_truth_action}`
YOUR OBJECTIVE: Generate the Belief Update and Reasoning that LEADS TO this action.
```

三阶段 I/O 对比：

| | 标注 | SFT | RL |
|-|------|-----|----|
| 动作来源 | 外部给定（专家） | 模型自由生成 | 从候选列表选 |
| 难度 | 无需决策 | 最高（开放生成） | 中等 |
| 输出 | belief + reasoning | belief + reasoning + **action** | belief + reasoning + **action** |

---

## Q: SFT Prompt 的作用是什么？怎么使用？

**作用：** 在专家轨迹标注数据上监督学习，让模型掌握：
1. 输出格式：`<belief>JSON</belief>` + `<reasoning>` + `<action>`
2. Belief 状态维护和更新
3. **开放式动作生成**（故意不提供 admissible_actions）

**关键设计：SFT 阶段不提供候选动作列表**，强迫模型凭空生成合法动作字符串（如 `go to shelf 1`）。进入 RL 阶段才提供列表，降低探索难度。

**使用流程：**

```
专家轨迹
    ↓
[标注] Teacher LLM + TAGGING_TEMPLATE
    → 每步生成 belief_update + reasoning
    ↓
[数据构造] regenerate_coldstart_clean.py
    第 1 步: ALFWORLD_REBEL_TEMPLATE_NO_HIS_CS.format(current_observation=...)
    后续步: ALFWORLD_REBEL_TEMPLATE_CS.format(
              current_observation, action_history,
              current_belief_state, planning
            )
    + 拼接 Teacher 标注的 belief/reasoning + 专家 action
    → 构成 (prompt, response) 训练样本
    ↓
[SFT 训练] verl SFT trainer
    → 模型学会格式 + belief 追踪 + 开放动作生成
    ↓
[RL 训练] 用 SFT checkpoint 初始化
    → 换用 ALFWORLD_TEMPLATE_REBEL（含 admissible_actions）
    → GRPO / GiGPO + HiBO 继续优化
```

**`ALFWORLD_REBEL_TEMPLATE_CS` 占位符（后续步）：**

```
{task_description}      任务目标
{step_count}            已执行步数
{history_length}        历史窗口长度
{action_history}        最近 N 步的 (obs, action) 对
{current_step}          当前步编号
{current_observation}   当前环境观测
{current_belief_state}  上一步输出的 belief JSON
{planning}              teacher planner 生成的整体计划
```

---

## RL 模板变体（`prompt_template_type`）

在 `ppo_trainer.yaml` 的 `env.alfworld` 下配置：

```yaml
env:
  alfworld:
    prompt_template_type: "default"  # default | belief_conditioned | explicit_task_type
```

| 值 | 特点 |
|----|------|
| `default` | 标准 ReBel 模板 |
| `explicit_task_type` | prompt 中明确标注任务类型（heat/cool/clean 等） |
| `belief_conditioned` | 注入当前 belief state，belief 条件化决策 |
