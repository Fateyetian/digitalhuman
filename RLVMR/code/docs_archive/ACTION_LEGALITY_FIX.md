# 【紧急】动作合法性问题修复方案

## 🚨 核心问题确认

**实测数据（eval_20251112_160532）**:
- 合法动作: 54.4%
- 非法动作: **45.6%**
- 成功率: 0%

**根本原因**:
模型没有学会**严格从当前step的admissible actions列表中精确选择动作**，而是：
1. 幻想不存在的动作 (如 `explore drawer 2`)
2. 过拟合训练样本中的动作模式
3. 混淆BDRS标签和action内容
4. 无视环境状态约束

---

## 🎯 立即修复方案

### 方案1: 强化Prompt约束（最高优先级）

#### 1.1 修改BDRS Runtime Prompt

**文件**: `agent_system/environments/prompts/alfworld.py:24-60`

**关键修改**:
```python
ALFWORLD_TEMPLATE_NO_HIS_BDRS = """
You are an expert agent operating in the ALFRED Embodied Environment.
Your current observation is: {current_observation}
Your admissible actions are: [{admissible_actions}].

⚠️ CRITICAL RULE: Your <action> MUST be EXACTLY one string from the admissible actions list above.
   - Copy the EXACT string, character-by-character
   - Do NOT modify, do NOT create new actions
   - Do NOT include BDRS tags in the action

You maintain three internal belief modules:
- M_t (World Model): objects, locations, states, interaction history
- P_t (Task Progress): goals, subgoals, completion status
- E_t (Exploration Map): visited/unvisited areas

Output Format:
<MODE>Brief reasoning mentioning M_t/P_t/E_t (≤20 words)</MODE>
<action>EXACT string from admissible actions list</action>

✓ CORRECT Examples:
Input admissible actions: ['go to fridge 1', 'open fridge 1', 'take apple 1']
Output:
<EXECUTE>M_t shows apple in fridge, P_t needs it.</EXECUTE>
<action>take apple 1</action>

✗ WRONG Examples (DO NOT DO THIS):
<EXECUTE>...</EXECUTE>
<action>take apple</action>           ← Missing " 1"
<action>execute take apple 1</action> ← Added BDRS tag
<action>get apple 1</action>          ← Not in list
<action>take apple 1 from fridge 1</action> ← Not exact match

🎯 Mode Selection:
<PLAN>: Breaking down task into subgoals (steps 1-2)
<EXECUTE>: Direct goal-advancing actions when you have the object
<EXPLORE>: Information-gathering when location unknown
<VERIFY>: After "Nothing happens" or unexpected results

Remember: ALWAYS copy the EXACT action string from [{admissible_actions}].
"""

ALFWORLD_TEMPLATE_BDRS = """
You are an expert agent operating in the ALFRED Embodied Environment. Your task is to: {task_description}
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observaitons and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}
Your admissible actions are: [{admissible_actions}].

⚠️ CRITICAL: Your <action> MUST be EXACTLY copied from the admissible actions list above.
   NO modifications, NO invented actions, NO BDRS tags in action.

You maintain three internal belief modules:
- M_t (World Model): objects, locations, states, interaction history
- P_t (Task Progress): goals, subgoals, completion status
- E_t (Exploration Map): visited/unvisited areas

Output Format:
<MODE>Brief reasoning (≤20 words) referencing M_t/P_t/E_t</MODE>
<action>EXACT string copied from [{admissible_actions}]</action>

Mode Selection:
<PLAN>: Task decomposition, replanning after repeated failures
<EXECUTE>: Taking state-changing actions when you have required objects
<EXPLORE>: Checking unopened containers, visiting new locations
<VERIFY>: After "Nothing happens" or belief contradiction

⚠️ Common Mistakes to AVOID:
1. Creating new actions not in the list
2. Modifying action strings (adding/removing words)
3. Including BDRS mode tags inside <action>
4. Using actions from previous steps that aren't currently available
"""
```

---

### 方案2: 重新生成冷启动数据（必须）

#### 2.1 增强标注Prompt

**文件**: `scripts/alfworld/alfworld_prepare.py`

在annotation_prompt中强调：

```python
ANNOTATION_SYSTEM_PROMPT = """
你是一个专业的强化学习数据标注员。你的任务是为ALFWorld环境的成功轨迹标注BDRS格式的推理过程。

🚨 最关键的规则：
<action>标签中的动作必须精确匹配当前step的admissible actions列表中的某一个字符串。
- 逐字符复制，不能有任何修改
- 不能创造新动作
- 不能在action中包含BDRS模式标签

标注要求：
1. 每一步都要选择合适的BDRS模式：<PLAN>/<EXECUTE>/<EXPLORE>/<VERIFY>
2. 推理要引用M_t/P_t/E_t，不超过25词
3. <action>必须从admissible actions中精确复制

示例：

当前observation: You are at fridge 1. The fridge is closed.
Admissible actions: ['open fridge 1', 'examine fridge 1', 'go to countertop 1']

✓ 正确标注：
<EXPLORE>M_t lacks fridge contents, E_t shows fridge unopened, check it.</EXPLORE>
<action>open fridge 1</action>

✗ 错误标注：
<EXPLORE>...</EXPLORE>
<action>open the fridge</action>  ← 不是精确匹配
<action>EXPLORE open fridge 1</action>  ← 混入了模式标签
<action>open fridge 1 to find apple</action>  ← 添加了额外内容

请严格遵守这些规则。
"""
```

#### 2.2 增加Few-shot示例

在每个标注请求中添加3-5个高质量示例，展示如何**精确复制**admissible actions。

#### 2.3 重新生成数据

```bash
cd /root/digitalhuman/RLVMR/code

# 删除旧数据
mv data/alfworld_cold-start.json data/alfworld_cold-start_old.json

# 用新prompt重新生成
python3 scripts/alfworld/alfworld_prepare.py \
    --api_key $OPENAI_API_KEY \
    --model gpt-4o \
    --num_trajs 600 \
    --output_path data/alfworld_cold-start_v2.json \
    --resume

# 质量检查
python3 << 'EOF'
import json

with open('data/alfworld_cold-start_v2.json', 'r') as f:
    data = json.load(f)

# 检查每个action是否在对应的admissible中
illegal_count = 0
total_count = 0

for sample in data:
    for step in sample.get('data', []):
        response = step.get('response', '')
        prompt = step.get('prompt', '')

        if '<action>' in response and 'admissible actions' in prompt.lower():
            # 提取生成的action
            action = response.split('<action>')[1].split('</action>')[0].strip()

            # 检查是否在admissible中
            total_count += 1
            if action not in prompt:
                illegal_count += 1
                print(f"非法动作: {action}")
                if illegal_count >= 10:
                    break

print(f"\n检查结果: {illegal_count}/{total_count} 非法 ({illegal_count/total_count*100:.1f}%)")
if illegal_count/total_count < 0.05:
    print("✓ 数据质量合格")
else:
    print("✗ 需要重新生成")
EOF
```

---

### 方案3: 训练时强化约束

#### 3.1 数据增强

在训练时，对每个样本：
- 随机打乱admissible actions的顺序（防止位置偏好）
- 确保模型看到各种长度的action列表
- 添加负样本：错误的action + 纠正

#### 3.2 Loss加权

```python
# 在训练脚本中
# 对于<action>标签内的token，增加loss权重
for token in action_tokens:
    if token in admissible_action_tokens:
        loss_weight = 2.0  # 正确选择，强化
    else:
        loss_weight = 5.0  # 错误选择，重惩罚
```

---

### 方案4: Constrained Decoding（可选，高级）

如果重新训练后仍有问题，可以在推理时强制约束：

```python
# 在run_alfworld_rollout.py中
def get_action_with_constraint(prompt, admissible_actions, model):
    """
    使用constrained decoding确保生成的action在admissible中
    """
    response = model.generate(prompt)

    # 提取action
    action = extract_action(response)

    # 如果不在列表中，找最相似的
    if action not in admissible_actions:
        # 使用编辑距离找最接近的合法action
        from difflib import get_close_matches
        matches = get_close_matches(action, admissible_actions, n=1, cutoff=0.6)
        if matches:
            action = matches[0]
            print(f"⚠️ 修正: {action_original} -> {action}")
        else:
            # 回退到第一个合法action
            action = admissible_actions[0]
            print(f"⚠️ 回退到默认action: {action}")

    return action
```

---

## 📋 执行计划

### 立即执行（今天）

1. **修改Prompt** (30分钟)
   ```bash
   # 备份
   cp agent_system/environments/prompts/alfworld.py \
      agent_system/environments/prompts/alfworld.py.backup

   # 应用新prompt（见方案1）
   # 编辑文件...
   ```

2. **小规模测试** (30分钟)
   ```bash
   # 测试2个环境
   python3 examples/bdrs_trainer/rollout/run_alfworld_rollout.py \
       --total_envs 2 \
       --max_steps 10 \
       --dump_path test_prompt_fix.jsonl

   # 检查合法率
   python3 << 'EOF'
   import json
   # ... 检查代码 ...
   EOF
   ```

### 短期（本周）

3. **重新生成冷启动数据** (2-3天)
   - 修改annotation prompt
   - 生成600个样本
   - 质量验证（合法率>95%）

4. **重新训练** (1天)
   ```bash
   bash examples/bdrs_trainer/train_alfworld_sft.sh \
       --data_path data/alfworld_cold-start_v2.json \
       --num_train_epochs 5
   ```

### 中期（下周）

5. **全面评测**
   - 64环境评测
   - 预期：合法率>90%, 成功率>20%

6. **迭代优化**
   - 根据失败case继续优化prompt
   - 必要时添加constrained decoding

---

## 🎯 预期效果

| 阶段 | 改进 | 动作合法率 | 成功率 |
|------|------|-----------|--------|
| 当前 | - | 54.4% | 0% |
| Prompt优化 | 强化约束 | 70-80% | 0-5% |
| 重新训练 | 新数据 | 85-95% | 15-25% |
| Constrained Decoding | 推理约束 | 98%+ | 20-30% |

---

## ✅ 关键成功因素

1. **Prompt中必须强调**："EXACTLY copy from list"
2. **训练数据必须100%合法** - 一个非法样本都不能有
3. **质量检查** - 每次生成后验证合法率
4. **持续监控** - 每次评测都检查合法率

---

**生成时间**: 2025-11-12
**问题定位**: 用户准确诊断
**根本原因**: 模型未学会严格约束选择
**修复优先级**: P0 - 阻塞性问题
