# WebShop Prompt 说明

> 2026-03-07 | ReBel V11 (`ReBel_v2_improvements`)

---

## 文件位置

```
code/
├── agent_system/environments/
│   ├── prompts/
│   │   └── webshop_rebel_prompts.py       # 全部三阶段 Prompt 模板
│   └── env_manager.py
│       └── WebshopEnvironmentManager      # RL 调用入口（build_text_obs, ~L1427）
│
└── scripts/
    └── generate_webshop_rebel_hindsight.py  # 标注 + Cold-Start 数据生成脚本
```

---

## 各阶段 Prompt 对照

| 阶段 | 模板变量 | 文件 | 调用入口 |
|------|---------|------|---------|
| 标注（Hindsight） | `WEBSHOP_REBEL_TAGGING_TEMPLATE` | `webshop_rebel_prompts.py` | `generate_webshop_rebel_hindsight.py` → `construct_annotation_prompt()` |
| SFT 第 1 步 | `WEBSHOP_REBEL_TEMPLATE_NO_HIS_CS` | `webshop_rebel_prompts.py` | `generate_webshop_rebel_hindsight.py` → `convert_to_coldstart_format()` |
| SFT 后续步 | `WEBSHOP_REBEL_TEMPLATE_CS` | `webshop_rebel_prompts.py` | `generate_webshop_rebel_hindsight.py` → `convert_to_coldstart_format()` |
| RL 第 1 步 | `WEBSHOP_REBEL_TEMPLATE_NO_HIS_RL` | `webshop_rebel_prompts.py` | `env_manager.py` → `build_text_obs()` L1452 |
| RL 后续步 | `WEBSHOP_REBEL_TEMPLATE_RL` | `webshop_rebel_prompts.py` | `env_manager.py` → `build_text_obs()` L1480 |

---

## 信念状态（Belief State）Schema

WebShop 使用 **4 组件**结构，与 ALFWorld 的 world_model/task_progress/exploration_map 三组件完全不同：

```json
{
  "product_understanding": {
    "target_attributes": {"color": "navy", "size": "small"},
    "current_product_match": "none / partial / exact",
    "price_constraint": "under $30 / any"
  },
  "attribute_verification": {
    "verified": [
      {"attribute": "color", "value": "navy", "source": "option_button", "snippet": "navy [clicked]"}
    ],
    "unverified": ["material", "care_instructions"],
    "inferred_only": [
      {"attribute": "slip_resistance", "inference": "title mentions rubber sole", "required_action": "click Features tab"}
    ]
  },
  "search_progress": {
    "search_status": "not_started / searching / product_found / options_selecting / ready_to_buy",
    "evidence": "what was observed and what it means",
    "updated_subgoal": "immediate next goal"
  },
  "exploration_state": {
    "queries_tried": ["navy tennis skirt", "navy skirt small"],
    "products_viewed": ["B091CXXX"],
    "options_selected": ["navy", "small"],
    "tabs_clicked": ["Description", "Features"]
  }
}
```

### 三级验证机制（WebShop 核心）

`attribute_verification` 区分三种确认程度，这是 WebShop 与 ALFWorld 最大的结构差异：

| 级别 | 含义 | 升级条件 |
|------|------|---------|
| `unverified` | 完全未检查 | 看到商品页面，可提升为 inferred_only 或 verified |
| `inferred_only` | 从标题/关键词推断，**未点击 tab 直接确认** | 点击对应 tab 并读到明确信息后升级为 verified |
| `verified` | 通过点击 option button 或 tab 内容**直接观测到** | 终态 |

**核心规则：`current_product_match = "exact"` 必须满足 ALL 属性在 `verified` 且 `inferred_only` 为空。**

### 5 态状态机（search_status）

```
not_started → searching → product_found → options_selecting → ready_to_buy
```

- **禁止跳级**（如直接从 `searching` 跳 `ready_to_buy`）
- `ready_to_buy` 要求：`inferred_only` 为空 + 所有目标属性均在 `verified`
- 违反条件时必须停留在 `options_selecting`

---

## Q: 为什么标注 Prompt 没有 Action Selection？

标注是 **Hindsight Annotation（事后标注）**，专家动作已知，只需反向推理 belief：

```
输入 = current_observation + ground_truth_action（已知）+ prev_belief + 历史轨迹
输出 = belief_update + reasoning（倒推出导向该动作的思维过程）
```

Prompt 中明确写明：
```
The next action to be taken is: `{ground_truth_action}`
YOUR OBJECTIVE: Generate the Belief Update and Reasoning that justifies this action.
```

三阶段 I/O 对比：

| | 标注 | SFT | RL |
|-|------|-----|----|
| 动作来源 | 外部给定（专家轨迹） | 模型自由生成 | 从候选列表选 |
| admissible_actions | 仅供参考，不做选择 | **无**（故意隐藏） | **有**（辅助探索） |
| 输出 | belief_update + reasoning | belief + reasoning + **action** | belief + reasoning + **action** |
| belief 类型 | **增量更新**（本步变化） | **增量更新** | **增量更新** |

---

## Q: SFT Prompt 的作用是什么？怎么使用？

**作用：** 在专家轨迹标注数据上监督学习，让模型掌握：
1. 输出格式：`<belief>JSON</belief>` + `<reasoning>` + `<action>`
2. 三级验证机制（verified/inferred_only/unverified）的推理方式
3. 5 态状态机的正确转换逻辑
4. **开放式动作生成**（SFT 阶段故意不提供 admissible_actions）

**关键设计：SFT 阶段不提供候选动作列表**，强迫模型学会生成合法格式的动作字符串（如 `search[navy tennis skirt]`、`click[Buy Now]`）。进入 RL 阶段才提供列表，降低探索难度。

**完整数据流程：**

```
专家轨迹（steps 格式 或 AgentTraj-L 对话格式）
    ↓
[标注] Teacher LLM + WEBSHOP_REBEL_TAGGING_TEMPLATE
    输入：observation + ground_truth_action + prev_belief + 历史轨迹({traj})
    输出：每步的 belief_update + reasoning（增量）
    调用：construct_annotation_prompt() → call_teacher_llm()
    安全网：inferred_only 非空时自动降级 ready_to_buy → options_selecting
    fallback：LLM 失败时继承当前 belief（不重置）
    ↓
[数据构造] convert_to_coldstart_format()
    第 1 步：WEBSHOP_REBEL_TEMPLATE_NO_HIS_CS.format(
               task_description=task,
               current_observation=obs  ← 从 human_turn 正则提取
             )
    后续步：human_turn 过滤 "Available Actions:" 行后直接用作 prompt
            （保留：Task + Observation + Current Belief State）
    + 拼接 Teacher 标注的 belief_update/reasoning + 专家 action
    → 构成 {question, answer} 训练样本，存为 parquet
    ↓
[SFT 训练] verl SFT trainer
    → 模型学会格式 + 三级验证推理 + 状态机转换 + 开放动作生成
    ↓
[RL 训练] 用 SFT checkpoint 初始化
    → 换用 WEBSHOP_REBEL_TEMPLATE_RL（含 admissible_actions）
    → GRPO/GiGPO + HiBO 继续优化
```

---

## Q: 标注时 `{traj}` 传什么？为何与 ALFWorld 不同？

**ALFWorld**：传空字符串 `""`，因为 ALFWorld 观测文字简洁，单步 obs 已包含足够信息。

**WebShop**：传**实际历史轨迹**（obs snippet → action 对）：
```python
# generate_webshop_rebel_hindsight.py → construct_annotation_prompt()
traj_parts = []
for prev_obs, prev_action in zip(obs_history, action_history):
    obs_snippet = prev_obs[:200].replace('\n', ' ')
    traj_parts.append(f"[Obs: {obs_snippet}... → Action: {prev_action}]")
traj_str = "\n".join(traj_parts) if traj_parts else "(no prior steps)"
```

WebShop 页面内容密集（商品列表/详情页 HTML），Teacher LLM 需要历史轨迹上下文才能正确理解 `exploration_state` 的累积情况。

---

## 各模板占位符汇总

### 标注 `WEBSHOP_REBEL_TAGGING_TEMPLATE`

```
{task_description}       购物目标描述
{current_observation}    当前页面观测（最长约 500 字符，超出截断）
{prev_belief_json}       上一步的完整全局 belief JSON
{admissible_actions}     当前可用动作列表（参考，非选择）
{ground_truth_action}    专家动作（已知，需正向化为自主决策）
{traj}                   历史 (obs_snippet → action) 对字符串
```

### SFT 第 1 步 `WEBSHOP_REBEL_TEMPLATE_NO_HIS_CS`

```
{task_description}       购物目标描述
{current_observation}    当前环境观测（完整，无截断）
```

### SFT 后续步 `WEBSHOP_REBEL_TEMPLATE_CS`

```
{task_description}       购物目标描述
{step_count}             已执行步数
{history_length}         历史窗口长度
{action_history}         最近 N 步的 (obs, action) 对
{current_step}           当前步编号
{current_observation}    当前环境观测
{current_belief_state}   上一步输出的**完整全局** belief JSON（env_manager 累积）
{planning}               当前子目标（来自 cumulative_beliefs[i].search_progress.updated_subgoal）
```

### RL 第 1 步 `WEBSHOP_REBEL_TEMPLATE_NO_HIS_RL`

```
{task_description}       购物目标描述
{current_observation}    当前环境观测
{admissible_actions}     格式化的可用动作列表（出现两次：观测后 + Step 3）
```

初始 belief state 在模板中**硬编码为全空/not_started**（无占位符）。

### RL 后续步 `WEBSHOP_REBEL_TEMPLATE_RL`

```
{task_description}       购物目标描述
{step_count}             已执行步数（env_manager 传入，来自 len(buffers[i])）
{history_length}         有效历史长度
{action_history}         最近 N 步 [Observation N: '...', Action N: '...']
{current_step}           当前步编号
{current_observation}    当前环境观测（完整，超 13000 字符时整体降级为 NO_HIS）
{current_belief_state}   json.dumps(cumulative_beliefs[i], indent=2)（全局累积）
{admissible_actions}     格式化的可用动作列表（出现两次：观测后 + Step 3）
```

---

## 增量 belief vs 全局 belief 的区分

这是理解数据流的关键：

| 场景 | belief 类型 | 说明 |
|------|------------|------|
| Teacher LLM 输出（标注） | **增量** | 只包含本步新增/变化的字段 |
| SFT 训练数据 gpt 回合（answer） | **增量** | 直接用标注输出，Output Format label 注明 |
| SFT 训练数据 human 回合（question） | **全局** | 注入的是 `input_belief_json`（上一步结束时的全局状态） |
| RL 模板 `{current_belief_state}` | **全局** | `json.dumps(cumulative_beliefs[i], indent=2)` |
| `merge_belief_update()` | 增量→全局 | 标注脚本用；三向合并（verified 升级后自动从 inferred_only/unverified 删除） |
| `_update_cumulative_belief()` | 增量→全局 | env_manager 用；逻辑与 merge_belief_update 完全一致 |

---

## 自动追踪机制（标注脚本）

标注脚本在 Teacher LLM 输出**之后**对 `exploration_state` 做自动补全，防止 Teacher LLM 遗漏：

```python
# generate_webshop_rebel_hindsight.py → generate_webshop_rebel_dataset()
search_match → exploration_state.queries_tried      # search[...] 动作
click[B\d+]  → exploration_state.products_viewed    # ASIN 格式：B091CXXX
click[description/features/reviews] → exploration_state.tabs_clicked
其他 click（非导航类）→ exploration_state.options_selected
```

NAV_TARGETS（不追踪）= `{'buy now', 'back to search', '< prev', 'next >', 'description', 'features', 'reviews', 'search'}`

---

## 关键保护机制

### 1. Safety Net（标注时）

Teacher LLM 输出通过验证后，检查：

```python
# call_teacher_llm() 内
if inferred_only 非空 and search_status == "ready_to_buy":
    → 强制降级：search_status = "options_selecting"
    → current_product_match: "exact" → "partial"
```

防止训练数据出现逻辑矛盾（`inferred_only` 非空时不允许 `ready_to_buy`）。

### 2. required_fields 验证（标注时）

Teacher LLM 必须输出以下全部字段，否则触发重试（最多 3 次）：

```python
required_fields = [
    "product_understanding",
    "attribute_verification",   # ← WebShop 核心字段，必须验证
    "search_progress",
    "exploration_state",
    "reasoning"
]
```

### 3. Fallback 策略（标注时）

LLM 3 次重试均失败时，`create_fallback_annotation()` **继承当前全局 belief**（而非重置为空），保持数据一致性。

### 4. 超长 Obs 保护（RL 时）

```python
# env_manager.py → build_text_obs()
if len(obs) > 13000:
    obs = _TEMPLATE_NO_HIS.format(task_description, current_observation, admissible_actions)
```

WebShop 页面内容（商品列表/详情）远长于 ALFWorld 文字观测，此保护防止 prompt 超过模型上下文限制。

### 5. 失败观测检测（标注时）

```python
# _is_failure_observation(obs)
_WEBSHOP_FAILURE_PATTERNS = [
    "sorry, nothing was found", "no products were found",
    "no results found", "no matching", "could not find",
    "0 results", "no items found"
]
```

检测到搜索无结果时，在 Teacher LLM 的观测前注入 NOTE 提示，防止错误升级 `search_status`（等价于 ALFWorld 的 `is_nothing_happens`）。

---

## 数据格式：标注数据 → SFT 训练格式

标注完成后，每条轨迹对应一个 `rebel_trajectory` dict：

```python
{
  "conversations": [
    {"from": "human", "loss": False, "value": "系统初始化消息"},
    {"from": "gpt",   "loss": False, "value": "OK. I will track..."},
    # 每步一对：
    {"from": "human", "loss": False, "value": "Task: ...\nObservation:\n...\nCurrent Belief State:\n{全局JSON}\nAvailable Actions: ..."},
    {"from": "gpt",   "loss": True,  "value": "<belief>{增量JSON}</belief>\n\n<reasoning>...</reasoning>\n\n<action>...</action>"},
    ...
  ],
  "task": "购物任务描述",
  "item_id": "xxx_rebel_hindsight",
  "num_steps": N,
  "annotation_success_rate": 0.85
}
```

`convert_to_coldstart_format()` 将其转为 SFT 训练 parquet：

```python
# 过滤条件：from='gpt' AND loss=True AND '<belief>' in value
# Step 1：question = WEBSHOP_REBEL_TEMPLATE_NO_HIS_CS.format(task, obs)
# Step 2+：question = human_turn 删除 "Available Actions:" 行后的内容
# answer = gpt turn 原文（<belief>增量JSON</belief>...<action>...</action>）
[{"question": "...", "answer": "..."}, ...]
```

---

## SFT 与 RL 格式一致性检查

最关键的一致性要求：**模型 OUTPUT 格式在 SFT 和 RL 阶段完全相同。**

| 格式要素 | SFT 训练数据（gpt 回合） | RL 模板 Output Format hint |
|---------|----------------------|--------------------------|
| belief 标签 | `<belief>\n{JSON}\n</belief>` | `<belief>\n{...}\n</belief>` |
| reasoning 标签 | `<reasoning>\n...\n</reasoning>` | `<reasoning>\n...\n</reasoning>` |
| action 标签 | `<action>\n...\n</action>` | `<action>\n...\n</action>` |
| 标签间空行 | `</belief>\n\n<reasoning>` | `</belief>\n\n<reasoning>` |
| belief JSON 缩进 | `json.dumps(indent=2)` | 示例用 `{...}` 占位，格式一致 |

**完全对齐** ✅

---

## 与 ALFWorld 的关键差异对比

| 维度 | ALFWorld | WebShop | 原因 |
|------|----------|---------|------|
| belief 组件数 | 3（world_model/task_progress/exploration_map） | 4（+attribute_verification） | 购物需区分"知道"和"验证" |
| 状态机 | 无显式状态机 | 5 态 search_status | 购物流程有明确阶段 |
| `{traj}` 内容 | 空字符串 `""` | 实际历史 obs+action 对 | WebShop 页面内容密集 |
| belief 格式化 | 三段分离（world_state/task_state/explore_state） | 单一 JSON dump | WebShop 4 组件结构已清晰 |
| max_tokens（标注） | 800 | 2500 | WebShop 输出复杂度更高 |
| Safety net | 无 | ✅ inferred_only 非空→降级 | 防止错误 Buy Now |
| Fallback 策略 | 重置为空状态 | ✅ 继承当前 belief | 保持数据一致性 |
| 超长 obs 保护 | 无 | ✅ >13000 字符降级 NO_HIS | WebShop 页面更长 |
| Planning 字段 | 独立 `{planning}` 占位符（整体计划） | `{planning}` = `updated_subgoal` | WebShop 规划粒度更细 |
| RL Planning Prompt | ✅ 有（生成整体购物计划） | ✅ 有（`get_planning_prompts`） | 均支持预规划 |
