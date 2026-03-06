# ReBel RLVMR 改进日志

---

## 2026-03-06

### 根因分析：WebShop RL 训练 90 个 epoch 成功率始终为 0%

通过系统性排查，确认失败根因是三个实现层面的 Bug 叠加，**不是奖励设计本身的问题**。

#### Bug #1（最关键）— RL Prompt 缺少格式指令

**问题**：`WEBSHOP_REBEL_TEMPLATE_NO_HIS_RL` 和 `WEBSHOP_REBEL_TEMPLATE_RL` 两个 RL 训练模板
只包含 `Task / Observation / Current Belief State / Available Actions`，完全没有告诉模型要以
`<belief><reasoning><action>` 格式输出。

**SFT 模板** (`WEBSHOP_REBEL_TEMPLATE_NO_HIS_CS`) 包含完整的格式说明，但 RL 用的是不同模板。

**结果**：
- 模型输出随机文本，`projection.py` 解析失败 → `action = "None"`
- 每步只有格式罚分 `delta × r_format = 0.1 × (−0.05) = −0.005`
- WebShop 环境不执行任何操作，始终停留在搜索页面
- 90 个 epoch，reward 无方差（全部 −0.005/step），零学习信号

**证据**：
```
# 2025-12-20 早期测试轨迹（run_20251220_051526）
action: "None" × 120/120
r_format: -0.05 × 120/120
reward: -0.005 × 120/120  ← 完全一致，无任何方差
belief_parsed: False × 120/120
```

**修复**：在 `code/agent_system/environments/prompts/webshop_rebel_prompts.py` 中，
向两个 RL 模板末尾添加：
```
Respond strictly in this format:
<belief>
{"product_understanding": {...}, "attribute_verification": {...}, "search_progress": {...}, "exploration_state": {...}}
</belief>
<reasoning>
Your step-by-step reasoning here.
</reasoning>
<action>
Your chosen action here (must be one of the Available Actions above)
</action>
```

**文件**：`code/agent_system/environments/prompts/webshop_rebel_prompts.py`

---

#### Bug #2（新增）— RL Prompt 缺少 Action History 占位符

**问题**：`WEBSHOP_REBEL_TEMPLATE_RL`（第 2 步起使用）没有 `{action_history}` 占位符。
`build_text_obs()` 已正确计算 `action_history`（最近 N 步的观测+动作历史），
并通过 `.format(action_history=..., current_step=...)` 传入——
但 Python `.format()` 对多余 kwargs **静默忽略**，历史信息完全丢失。

**结果**：
- 模型每步只看到 `Current Observation`，不知道之前做了什么
- 模型每步重复同一个搜索动作（首个 prompt 也无 history，所以步步一样）
- 观测始终是搜索结果页（因为动作始终是 search）

**对比**：SFT 模板 `WEBSHOP_REBEL_TEMPLATE_CS` 有完整的 `{action_history}` 和 `{current_step}`。

**修复**：在 `WEBSHOP_REBEL_TEMPLATE_RL` 中添加：
```
History (last {history_length} steps):
{action_history}

Current Observation (Step {current_step}):
{current_observation}
```
同时将 `build_text_obs()` 的 `history_length` 默认值从 `2` 改为 `5`。

**文件**：
- `code/agent_system/environments/prompts/webshop_rebel_prompts.py`
- `code/agent_system/environments/env_manager.py`

---

#### Bug #3（历史遗留）— Response Length 截断

**问题**：早期训练使用 `max_response_length=512`，而 SFT 模型平均输出 ~1054 tokens（p90=1436）。
512 token 截断后 `<action>` 标签几乎永远不会出现。

**修复**：`run_v11_webshop_base.sh` 中已将默认值改为 1536：
```bash
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-1536}
```
代码注释中明确写出："Previously 512 which truncated 99.8% of outputs before `<action>` tag."

**文件**：`code/rebel_test_results/v11_final/run_v11_webshop_base.sh`

---

#### Bug #3（次要）— WebShop 奖励极度稀疏

**问题**：WebShop 环境奖励结构：
- 所有非购买步骤：`reward = 0.0`
- `click[Buy Now]` 完成后：`reward = match_score × 10.0`

完成一次购买需要 5–6 步（search → click product → select options × 1-3 → Buy Now），
信用分配困难，但这是 RL 常规挑战，**不是根本原因**。

SFT 数据分析：
- 499 条 SFT 轨迹，100% 包含 `click[Buy Now]`
- 平均在第 5.05 步执行购买（最长 6 步）

REBEL intrinsic rewards 量级（每步最大 ~0.091）远小于购买奖励（~10.0），
信噪比低，但一旦 Bug #1/#2 修复后应能正常学习。

---

### Trajectory-Tracer 可视化系统

**问题**：训练生成的轨迹未保存到磁盘（`_save_trajectories_if_enabled` 中 `if is_train: return` 跳过）。

**修复**：
- `rollout_loop.py`：重写 `_save_trajectories_if_enabled`，去掉 `is_train` 拦截，
  捕获 `action_text`、`obs_text`、逐步奖励，保存为 `rollout_{mode}_{timestamp}.jsonl`
- `run_v11_webshop_base.sh`：`SAVE_TRAJECTORIES` 默认改为 `true`，
  轨迹目录命名为 `trajectories/{TIMESTAMP}_{EXP_ID}_{EXP_NAME}_seed{SEED}`
- `Trajectory-Tracer/backend/main.py`：自动递归扫描 `trajectories/` 子目录
- `Trajectory-Tracer/backend/trajectory_adapters.py`：支持 `webshop_rl_jsonl` 格式解析
- 前端新增「任务分组」和「训练曲线」两个 Tab

---

## 2025-03-02

### WebShop RL 训练环境对齐修复

#### 修复 1 — format_obs 不匹配

**问题**：`WebshopEnvironmentManager.format_obs()` 对观测做了额外处理（strip task prefix、加单引号），
与 SFT 训练格式不一致。

**修复**：改为直接返回原始观测：
```python
def format_obs(self, text_obs):
    return list(text_obs)
```

**文件**：`code/agent_system/environments/env_manager.py`

---

#### 修复 2 — System Prompt 不匹配

**问题**：RL rollout 阶段添加了 system prompt，但 SFT 训练时没有使用 system prompt。
模型在推理时看到不同的对话结构。

**修复**：WebShop RL rollout 使用纯 user role（无 system prompt），与 SFT 保持一致。

**文件**：`code/agent_system/multi_turn_rollout/rollout_loop.py`

---

#### 修复 3 — Repetition Loop

**问题**：模型在产品页面反复生成 `<belief>\n{\n<belief>` 类的无限循环，填满 max_tokens。

**修复**：vLLM rollout 添加 `repetition_penalty=1.2`。

**文件**：`code/rebel_test_results/v11_final/run_v11_webshop_base.sh`

---

#### 修复 4 — RL Prompt Template 过于冗长

**问题**：早期 RL 模板有 2928 chars（700+ tokens），远比 SFT 格式复杂，
导致模型在 SFT 分布之外推理。

**修复**：简化为与 SFT 格式对齐的紧凑模板：
`Task: / Observation: / Current Belief State: / Available Actions:`

**文件**：`code/agent_system/environments/prompts/webshop_rebel_prompts.py`

---

#### SFT 训练中断问题

**问题**：SFT 在 global_step_45（0.28 epoch）时因磁盘满中断。

**处理**：
- 清理 `/tmp/ray`（25G）及中间 checkpoint（27G）
- SFT 重新配置为 20 epochs
- 最新 checkpoint：`global_step_180`（~1.13 epochs，仍不足）

**待完成**：SFT 需继续训练至 10–20 epochs 后再启动 RL。

---

## ALFWorld 实验结果（已完成）

- **最优方法**：ReBel（HiBO + 自适应差分衰减）
- **峰值成功率**：95.3%
- **配置**：GRPO + GiGPO + Belief Prompting + HiBO，100 epochs，8×A100
- **结果文档**：`code/rebel_test_results/v11_final/experiments_chapter_cn.md`
