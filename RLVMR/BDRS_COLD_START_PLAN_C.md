# BDRS冷启动数据优化方案C - 完整实施指南

## 文档概述

本文档详细说明BDRS冷启动数据的完整优化方案（方案C），包括逐步标注、信念状态显式化、模式平衡采样、失败-恢复样本生成和数据增强。

**目标**：构造高质量冷启动数据，使冷启动模型成功率达到30-40%，并加速后续RL训练收敛。

---

## 一、方案C核心策略

### 1.1 五大改进策略

| 策略 | 优先级 | 预期效果 | 实施复杂度 |
|------|--------|----------|-----------|
| **策略1：逐步标注** | ⭐⭐⭐⭐⭐ | 避免hindsight bias，提升标注质量30% | 高 |
| **策略2：信念状态显式化** | ⭐⭐⭐⭐⭐ | 让模型学会构造和使用信念，提升20% | 中 |
| **策略3：模式平衡采样** | ⭐⭐⭐⭐ | 解决模式分布不平衡，提升15% | 低 |
| **策略4：失败-恢复样本** | ⭐⭐⭐⭐ | 增强错误恢复能力，提升10% | 中 |
| **策略5：数据增强** | ⭐⭐⭐ | 提升多样性和鲁棒性，提升5-10% | 低 |

**累计预期提升**：50-80%成功率提升（从基线10-15%提升到30-40%）

---

## 二、策略1：逐步标注（Step-by-Step Annotation）

### 2.1 核心思想

**问题**：当前方法让GPT-4o看到完整轨迹后标注，存在"事后偏见"
**解决**：逐步提供信息，模拟真实决策过程

### 2.2 实现方法

```python
def annotate_step_by_step(traj, model="gpt-4o"):
    """
    逐步标注，每次只看到当前观察和历史，不知道未来
    """
    annotated = []
    belief_state = {
        "world_model": {},
        "task_progress": {
            "goal": traj["task"],
            "completed_subgoals": [],
            "current_subgoal": "start",
            "pending_subgoals": []
        },
        "exploration_map": {
            "visited_locations": [],
            "observed_objects": [],
            "unexplored_containers": []
        }
    }

    for i, step in enumerate(traj["traj"]):
        # 构造上下文（只包含历史，不包含未来）
        context = {
            "task": traj["task"],
            "history": traj["traj"][:i] if i > 0 else [],
            "current_observation": step["observation"],
            "current_belief": belief_state.copy(),
            "next_action": step["action"]  # 给标注器看将要执行的动作
        }

        # 标注提示
        annotation_prompt = f"""
You are annotating reasoning for a BDRS agent. Given ONLY the information below (you DON'T know future steps):

**Task**: {context['task']}

**Current Belief State**:
- M_t (World Model): {json.dumps(context['current_belief']['world_model'], indent=2)}
- P_t (Task Progress): {json.dumps(context['current_belief']['task_progress'], indent=2)}
- E_t (Exploration Map): {json.dumps(context['current_belief']['exploration_map'], indent=2)}

**History** (Previous {i} steps):
{format_history(context['history'])}

**Current Observation**: {context['current_observation']}

**Action to Take**: {context['next_action']}

---

Based on the current belief state and observation, choose the MOST appropriate reasoning mode and explain:

1. **Which mode**: <PLAN> / <EXECUTE> / <EXPLORE> / <VERIFY>
2. **Why this mode**: Explain based on M_t, P_t, and E_t
3. **Updated belief**: What changes in the belief state after this action?

Output in JSON format:
{{
    "mode": "PLAN/EXECUTE/EXPLORE/VERIFY",
    "reasoning": "<MODE>Detailed explanation based on beliefs...</MODE>",
    "belief_update": {{
        "world_model_changes": {{}},
        "task_progress_changes": {{}},
        "exploration_changes": {{}}
    }}
}}
"""

        # 调用LLM标注
        try:
            response = llm_json(annotation_prompt, model, temperature=0.2)

            # 验证格式
            assert "mode" in response and "reasoning" in response
            assert response["mode"] in ["PLAN", "EXECUTE", "EXPLORE", "VERIFY"]

            # 更新信念状态（用于下一步标注）
            belief_state = update_belief_state(
                belief_state,
                step["observation"],
                step["action"],
                response.get("belief_update", {})
            )

            annotated.append({
                "step": i + 1,
                "observation": step["observation"],
                "action": step["action"],
                "belief_before": context["current_belief"],
                "belief_after": belief_state.copy(),
                "mode": response["mode"],
                "reasoning": response["reasoning"]
            })

        except Exception as e:
            print(f"Error annotating step {i}: {e}")
            # 降级：使用简单的启发式规则
            fallback_annotation = heuristic_annotate(step, i, belief_state)
            annotated.append(fallback_annotation)

    return annotated


def update_belief_state(belief, observation, action, llm_update):
    """
    根据观察和动作更新信念状态
    结合LLM的更新建议和规则
    """
    new_belief = belief.copy()

    # 1. 更新世界模型（从观察中提取实体和状态）
    entities = extract_entities_from_obs(observation)
    for entity, state in entities.items():
        new_belief["world_model"][entity] = state

    # 2. 更新任务进度
    if action.startswith("take ") and "You pick up" in observation:
        item = action.split("take ")[1].split(" from")[0]
        new_belief["task_progress"]["completed_subgoals"].append(f"obtained_{item}")

    # 3. 更新探索地图
    if action.startswith("go to "):
        location = action.split("go to ")[1]
        if location not in new_belief["exploration_map"]["visited_locations"]:
            new_belief["exploration_map"]["visited_locations"].append(location)

    if action.startswith("open "):
        container = action.split("open ")[1]
        if container in new_belief["exploration_map"]["unexplored_containers"]:
            new_belief["exploration_map"]["unexplored_containers"].remove(container)

    # 4. 应用LLM建议的更新
    if llm_update:
        if "world_model_changes" in llm_update:
            new_belief["world_model"].update(llm_update["world_model_changes"])
        if "task_progress_changes" in llm_update:
            new_belief["task_progress"].update(llm_update["task_progress_changes"])

    return new_belief


def heuristic_annotate(step, step_idx, belief):
    """
    当LLM标注失败时的降级方案：使用启发式规则
    """
    action = step["action"]
    obs = step["observation"]

    # 规则1：第一步通常是PLAN
    if step_idx == 0:
        mode = "PLAN"
        reasoning = "<PLAN>Analyze the task and formulate initial subgoals.</PLAN>"

    # 规则2：open/go to 通常是EXPLORE
    elif action.startswith(("open ", "go to ")):
        mode = "EXPLORE"
        reasoning = f"<EXPLORE>Exploring to gather information. Action: {action}</EXPLORE>"

    # 规则3：take/put/use 通常是EXECUTE
    elif action.startswith(("take ", "put ", "use ", "heat ", "cool ", "clean ")):
        mode = "EXECUTE"
        reasoning = f"<EXECUTE>Executing known action to advance task. Action: {action}</EXECUTE>"

    # 规则4：Nothing happens 后通常是VERIFY
    elif "Nothing happens" in obs or "No known action" in obs:
        mode = "VERIFY"
        reasoning = "<VERIFY>Previous action failed, need to verify assumptions.</VERIFY>"

    else:
        mode = "EXECUTE"
        reasoning = f"<EXECUTE>Continue task execution. Action: {action}</EXECUTE>"

    return {
        "step": step_idx + 1,
        "observation": obs,
        "action": action,
        "belief_before": belief.copy(),
        "belief_after": belief.copy(),
        "mode": mode,
        "reasoning": reasoning
    }
```

### 2.3 关键改进点

✅ **避免事后偏见**：每一步标注时不知道未来
✅ **信念状态演化**：逐步更新，符合真实决策过程
✅ **降级机制**：API失败时使用启发式规则保证鲁棒性

---

## 三、策略2：信念状态显式化

### 3.1 核心思想

**问题**：当前推理只有模式标签，没有显式的信念状态
**解决**：在每一步的推理中包含结构化的信念状态

### 3.2 输出格式

```python
# 标准输出格式
{
    "step": 3,
    "observation": "You are in the kitchen. You see a microwave 1, a countertop 1 with an apple 1.",
    "belief_before": {
        "world_model": {
            "apple 1": "on countertop 1",
            "microwave 1": "location: kitchen, state: unknown"
        },
        "task_progress": {
            "goal": "heat apple and put in fridge",
            "completed_subgoals": ["located_apple"],
            "current_subgoal": "heat apple",
            "pending_subgoals": ["place_in_fridge"]
        },
        "exploration_map": {
            "visited_locations": ["kitchen"],
            "unexplored_containers": ["microwave 1", "fridge 1"]
        }
    },
    "reasoning": "<EXPLORE>M_t shows microwave 1's state is unknown. E_t indicates it's unexplored. Need to open it to check if it can be used for heating.</EXPLORE>",
    "action": "open microwave 1",
    "belief_after": {
        "world_model": {
            "apple 1": "on countertop 1",
            "microwave 1": "location: kitchen, state: open, empty"
        },
        "task_progress": {
            "goal": "heat apple and put in fridge",
            "completed_subgoals": ["located_apple", "microwave_ready"],
            "current_subgoal": "heat apple",
            "pending_subgoals": ["place_in_fridge"]
        },
        "exploration_map": {
            "visited_locations": ["kitchen"],
            "unexplored_containers": ["fridge 1"]
        }
    }
}
```

### 3.3 转换为SFT数据格式

```python
def convert_to_sft_format(annotated_step, task, history):
    """
    将标注好的步骤转换为SFT训练格式
    """
    step_idx = annotated_step["step"] - 1

    # 构造prompt
    if step_idx == 0:
        # 无历史版本
        prompt = ALFWORLD_TEMPLATE_NO_HIS_BDRS_CS.format(
            current_observation=annotated_step["observation"]
        )
    else:
        # 有历史版本
        action_history = format_history(history)
        planning = extract_latest_planning(history)

        prompt = ALFWORLD_TEMPLATE_BDRS_CS.format(
            task_description=task,
            step_count=step_idx,
            history_length=min(3, step_idx),
            action_history=action_history,
            current_step=step_idx + 1,
            current_observation=annotated_step["observation"],
            planning=planning
        )

    # 构造response（包含信念状态）
    # 注意：为了与prompt匹配，这里只保留推理和动作，信念状态隐式学习
    response = f"{annotated_step['reasoning']}\n<action>{annotated_step['action']}</action>"

    return {
        "step": step_idx + 1,
        "prompt": prompt,
        "response": response,
        "metadata": {
            "belief_before": annotated_step["belief_before"],
            "belief_after": annotated_step["belief_after"],
            "mode": annotated_step["mode"]
        }
    }
```

---

## 四、策略3：模式平衡采样

### 4.1 目标分布

```python
TARGET_MODE_DISTRIBUTION = {
    "PLAN": 0.15,      # 15% - 任务开始、重规划
    "EXECUTE": 0.50,   # 50% - 正常执行
    "EXPLORE": 0.25,   # 25% - 主动探索
    "VERIFY": 0.10     # 10% - 验证假设
}
```

### 4.2 实现方法

```python
def balanced_sampling(all_annotated_steps, target_dist, total_samples=None):
    """
    按照目标分布对不同模式的样本进行平衡采样

    Args:
        all_annotated_steps: 所有标注好的步骤列表
        target_dist: 目标分布字典
        total_samples: 目标总样本数（None表示保持原有总数）

    Returns:
        平衡后的样本列表
    """
    # 按模式分组
    mode_samples = {mode: [] for mode in target_dist.keys()}

    for step in all_annotated_steps:
        mode = step["mode"]
        if mode in mode_samples:
            mode_samples[mode].append(step)

    # 统计当前分布
    total = len(all_annotated_steps)
    current_dist = {mode: len(samples)/total for mode, samples in mode_samples.items()}

    print("Current mode distribution:")
    for mode, ratio in current_dist.items():
        print(f"  {mode}: {len(mode_samples[mode])} ({ratio*100:.1f}%)")

    # 计算目标样本数
    if total_samples is None:
        total_samples = total

    target_counts = {mode: int(total_samples * ratio) for mode, ratio in target_dist.items()}

    print(f"\nTarget mode distribution (total={total_samples}):")
    for mode, count in target_counts.items():
        print(f"  {mode}: {count} ({count/total_samples*100:.1f}%)")

    # 平衡采样
    balanced = []

    for mode, target_count in target_counts.items():
        available = mode_samples[mode]
        current_count = len(available)

        if current_count == 0:
            print(f"Warning: No samples for mode {mode}, skipping...")
            continue

        if current_count >= target_count:
            # 欠采样：随机选择
            sampled = random.sample(available, target_count)
        else:
            # 过采样：重复采样
            n_repeats = target_count // current_count
            n_extra = target_count % current_count

            sampled = available * n_repeats
            if n_extra > 0:
                sampled += random.sample(available, n_extra)

        balanced.extend(sampled)
        print(f"  {mode}: sampled {len(sampled)} from {current_count} available")

    # 打乱顺序
    random.shuffle(balanced)

    return balanced
```

### 4.3 使用示例

```python
# 在数据生成pipeline中使用
all_steps = []
for traj in annotated_trajectories:
    for step in traj["steps"]:
        all_steps.append(step)

# 平衡采样
balanced_steps = balanced_sampling(
    all_steps,
    TARGET_MODE_DISTRIBUTION,
    total_samples=5000  # 扩展到5000个样本
)

# 转换为SFT格式
sft_data = []
for step in balanced_steps:
    sft_sample = convert_to_sft_format(step, ...)
    sft_data.append(sft_sample)
```

---

## 五、策略4：失败-恢复样本生成

### 5.1 核心思想

**问题**：只用成功轨迹，模型不会处理失败场景
**解决**：主动构造"卡住→诊断→恢复"的样本

### 5.2 三种失败场景

#### 场景1：找不到目标物体（需要EXPLORE）

```python
def generate_exploration_failure_samples(success_traj):
    """
    模拟：agent以为物体在A位置，实际在B位置
    """
    samples = []

    # 找到轨迹中第一次成功找到目标物体的步骤
    for i, step in enumerate(success_traj):
        if "You see a" in step["observation"] and "take " in success_traj[i+1]["action"]:
            target_object = success_traj[i+1]["action"].split("take ")[1].split(" from")[0]
            wrong_location = generate_alternative_location(step["observation"])

            # 构造失败样本
            failure_sample = {
                "observation": f"You are at {wrong_location}. You don't see {target_object} here.",
                "belief_before": {
                    "world_model": {target_object: f"expected at {wrong_location}"},
                    "task_progress": {"current_subgoal": f"find {target_object}"}
                },
                "reasoning": f"<EXPLORE>M_t prediction was wrong. {target_object} not at {wrong_location}. E_t shows unexplored areas. Need to search elsewhere.</EXPLORE>",
                "action": f"go to {get_next_search_location()}",
                "mode": "EXPLORE"
            }
            samples.append(failure_sample)
            break

    return samples
```

#### 场景2：容器状态错误（需要VERIFY）

```python
def generate_verification_failure_samples(success_traj):
    """
    模拟：agent以为容器是open的，实际是closed
    """
    samples = []

    for i, step in enumerate(success_traj):
        if "open " in step["action"]:
            container = step["action"].split("open ")[1]

            # 构造失败样本
            failure_sample = {
                "observation": f"You try to put something in {container}, but it's closed.",
                "belief_before": {
                    "world_model": {container: "state: open"},
                    "task_progress": {"current_subgoal": f"use {container}"}
                },
                "reasoning": f"<VERIFY>M_t assumed {container} was open, but observation shows it's closed. Need to verify and correct belief.</VERIFY>",
                "action": f"open {container}",
                "mode": "VERIFY"
            }
            samples.append(failure_sample)

    return samples
```

#### 场景3：重复无效动作（需要PLAN重新规划）

```python
def generate_replanning_failure_samples(success_traj):
    """
    模拟：连续3次相同动作失败，需要重新规划
    """
    samples = []

    # 找到可能卡住的步骤
    for i in range(len(success_traj) - 3):
        actions = [success_traj[j]["action"] for j in range(i, i+3)]

        if len(set(actions)) == 1:  # 连续3次相同动作
            failure_sample = {
                "observation": "Nothing happens. Same action repeated 3 times.",
                "belief_before": {
                    "task_progress": {
                        "current_subgoal": "stuck",
                        "failed_attempts": 3
                    }
                },
                "reasoning": "<PLAN>Current approach is not working. P_t shows no progress after 3 attempts. Need to replan and try alternative approach.</PLAN>",
                "action": generate_alternative_action(actions[0]),
                "mode": "PLAN"
            }
            samples.append(failure_sample)
            break

    return samples
```

### 5.3 整合到数据生成

```python
def generate_all_samples(success_trajectories):
    """
    生成所有样本：成功样本 + 失败-恢复样本
    """
    all_samples = []

    for traj in success_trajectories:
        # 1. 标注成功轨迹（逐步标注）
        success_samples = annotate_step_by_step(traj)
        all_samples.extend(success_samples)

        # 2. 生成失败-恢复样本
        exploration_failures = generate_exploration_failure_samples(traj)
        verification_failures = generate_verification_failure_samples(traj)
        replanning_failures = generate_replanning_failure_samples(traj)

        all_samples.extend(exploration_failures)
        all_samples.extend(verification_failures)
        all_samples.extend(replanning_failures)

    return all_samples
```

---

## 六、策略5：数据增强

### 6.1 同义改写

```python
def augment_reasoning_paraphrase(original_reasoning, model="gpt-4o"):
    """
    改写推理，保持语义但增加多样性
    """
    prompt = f"""
Rewrite the following reasoning in 2 different ways while keeping the exact same meaning and mode tag:

Original: {original_reasoning}

Requirements:
1. Keep the mode tag (<PLAN>/<EXECUTE>/<EXPLORE>/<VERIFY>) unchanged
2. Keep the core meaning identical
3. Use different wording and sentence structures
4. Each version should be 1-2 sentences

Output in JSON:
{{
    "version1": "<MODE>rewritten reasoning 1...</MODE>",
    "version2": "<MODE>rewritten reasoning 2...</MODE>"
}}
"""

    try:
        response = llm_json(prompt, model, temperature=0.7)
        return [original_reasoning, response["version1"], response["version2"]]
    except:
        return [original_reasoning]
```

### 6.2 信念状态扰动

```python
def augment_belief_perturbation(sample):
    """
    轻微修改信念状态的表述，增加鲁棒性
    """
    augmented = []

    # 原样本
    augmented.append(sample)

    # 扰动1：改变实体描述方式
    perturbed1 = sample.copy()
    perturbed1["belief_before"] = perturb_entity_descriptions(sample["belief_before"])
    augmented.append(perturbed1)

    # 扰动2：改变容器状态描述
    perturbed2 = sample.copy()
    perturbed2["belief_before"] = perturb_state_descriptions(sample["belief_before"])
    augmented.append(perturbed2)

    return augmented


def perturb_entity_descriptions(belief):
    """
    修改实体描述方式
    例如：
    - "apple 1: on countertop 1" -> "apple 1: countertop 1"
    - "microwave 1: closed" -> "microwave 1: shut"
    """
    synonyms = {
        "closed": ["shut", "not open"],
        "open": ["opened", "not closed"],
        "on": ["at", "located at"],
        "in": ["inside", "within"]
    }

    perturbed = belief.copy()
    # ... 实现扰动逻辑
    return perturbed
```

### 6.3 批量增强

```python
def batch_augmentation(samples, augment_ratio=0.3):
    """
    对30%的样本进行数据增强
    """
    augmented_samples = []

    for sample in samples:
        augmented_samples.append(sample)

        # 随机选择30%的样本进行增强
        if random.random() < augment_ratio:
            # 同义改写
            paraphrases = augment_reasoning_paraphrase(sample["reasoning"])
            for paraphrase in paraphrases[1:]:  # 跳过原始版本
                aug_sample = sample.copy()
                aug_sample["reasoning"] = paraphrase
                augmented_samples.append(aug_sample)

            # 信念状态扰动
            perturbed = augment_belief_perturbation(sample)
            augmented_samples.extend(perturbed[1:])  # 跳过原始版本

    return augmented_samples
```

---

## 七、完整数据生成Pipeline

### 7.1 主流程

```python
def generate_cold_start_data_plan_c(
    expert_trajectories,
    num_trajectories=500,
    output_path="data/alfworld_cold-start_plan_c.json",
    model="gpt-4o"
):
    """
    方案C完整数据生成pipeline
    """
    print("=" * 60)
    print("BDRS Cold Start Data Generation - Plan C")
    print("=" * 60)

    # Step 1: 选择轨迹（扩展到500条）
    print(f"\n[Step 1/6] Selecting {num_trajectories} trajectories...")
    selected_trajs = select_diverse_trajectories(expert_trajectories, num_trajectories)
    print(f"Selected {len(selected_trajs)} trajectories covering all task types")

    # Step 2: 逐步标注（策略1 + 策略2）
    print("\n[Step 2/6] Step-by-step annotation with explicit belief states...")
    annotated_trajs = []
    for i, traj in enumerate(selected_trajs):
        print(f"  Annotating trajectory {i+1}/{len(selected_trajs)}...", end="\r")
        try:
            annotated = annotate_step_by_step(traj, model)
            annotated_trajs.append({
                "task": traj["task"],
                "steps": annotated
            })
        except Exception as e:
            print(f"\n  Error in trajectory {i+1}: {e}")
            continue

    print(f"\nAnnotated {len(annotated_trajs)} trajectories successfully")

    # Step 3: 生成失败-恢复样本（策略4）
    print("\n[Step 3/6] Generating failure-recovery samples...")
    all_steps = []
    for traj in annotated_trajs:
        all_steps.extend(traj["steps"])

        # 添加失败样本
        exploration_failures = generate_exploration_failure_samples(traj)
        verification_failures = generate_verification_failure_samples(traj)
        replanning_failures = generate_replanning_failure_samples(traj)

        all_steps.extend(exploration_failures)
        all_steps.extend(verification_failures)
        all_steps.extend(replanning_failures)

    print(f"Total steps (including failures): {len(all_steps)}")

    # Step 4: 模式平衡采样（策略3）
    print("\n[Step 4/6] Balanced mode sampling...")
    balanced_steps = balanced_sampling(
        all_steps,
        TARGET_MODE_DISTRIBUTION,
        total_samples=8000  # 目标8000个样本
    )

    # Step 5: 数据增强（策略5）
    print("\n[Step 5/6] Data augmentation...")
    augmented_steps = batch_augmentation(balanced_steps, augment_ratio=0.3)
    print(f"Augmented to {len(augmented_steps)} samples")

    # Step 6: 转换为SFT格式
    print("\n[Step 6/6] Converting to SFT format...")
    sft_data = []

    # 按轨迹重新组织
    traj_dict = {}
    for step in augmented_steps:
        traj_id = step.get("trajectory_id", 0)
        if traj_id not in traj_dict:
            traj_dict[traj_id] = []
        traj_dict[traj_id].append(step)

    for traj_id, steps in traj_dict.items():
        steps.sort(key=lambda x: x["step"])

        step_data = []
        latest_planning = "No plan yet."

        for i, step in enumerate(steps):
            if i == 0:
                prompt = ALFWORLD_TEMPLATE_NO_HIS_BDRS_CS.format(
                    current_observation=step["observation"]
                )
            else:
                action_history = format_history(steps[:i])
                prompt = ALFWORLD_TEMPLATE_BDRS_CS.format(
                    task_description=step.get("task", ""),
                    step_count=i,
                    history_length=min(3, i),
                    action_history=action_history,
                    current_step=i + 1,
                    current_observation=step["observation"],
                    planning=latest_planning
                )

            # 更新planning
            if "<PLAN>" in step["reasoning"]:
                latest_planning = extract_planning_content(step["reasoning"])

            response = f"{step['reasoning']}\n<action>{step['action']}</action>"

            step_data.append({
                "step": i + 1,
                "prompt": prompt,
                "response": response,
                "metadata": {
                    "mode": step["mode"],
                    "belief_before": step.get("belief_before", {}),
                    "belief_after": step.get("belief_after", {})
                }
            })

        sft_data.append({
            "task": steps[0].get("task", "unknown"),
            "done": "True",
            "data": step_data
        })

    # 保存
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sft_data, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print("Data generation completed!")
    print(f"{'='*60}")
    print(f"Total trajectories: {len(sft_data)}")
    print(f"Total steps: {len(augmented_steps)}")
    print(f"Output saved to: {output_path}")

    # 数据质量检查
    print(f"\n{'='*60}")
    print("Data Quality Report:")
    print(f"{'='*60}")
    quality_report(sft_data)

    return sft_data
```

### 7.2 辅助函数

```python
def select_diverse_trajectories(all_trajs, num_select):
    """
    选择多样化的轨迹，确保覆盖所有任务类型
    """
    task_types = {}
    for traj in all_trajs:
        task_type = extract_task_type(traj["task"])
        if task_type not in task_types:
            task_types[task_type] = []
        task_types[task_type].append(traj)

    # 每个任务类型均匀采样
    selected = []
    per_type = num_select // len(task_types)

    for task_type, trajs in task_types.items():
        n = min(per_type, len(trajs))
        selected.extend(random.sample(trajs, n))

    # 如果还不够，随机补充
    if len(selected) < num_select:
        remaining = [t for t in all_trajs if t not in selected]
        selected.extend(random.sample(remaining, num_select - len(selected)))

    random.shuffle(selected)
    return selected[:num_select]


def format_history(history_steps):
    """
    格式化历史步骤
    """
    formatted = []
    for step in history_steps[-3:]:  # 只保留最近3步
        formatted.append(
            f"[Observation: {step['observation']}, Action: {step['action']}]"
        )
    return "\n".join(formatted)


def extract_planning_content(reasoning):
    """
    从推理中提取planning内容
    """
    match = re.search(r'<PLAN>(.*?)</PLAN>', reasoning, re.DOTALL)
    if match:
        return match.group(1).strip()
    return "No explicit plan."


def quality_report(sft_data):
    """
    数据质量检查报告
    """
    total_steps = sum(len(item["data"]) for item in sft_data)

    # 1. 模式分布
    modes = []
    for item in sft_data:
        for step in item["data"]:
            if "<PLAN>" in step["response"]:
                modes.append("PLAN")
            elif "<EXECUTE>" in step["response"]:
                modes.append("EXECUTE")
            elif "<EXPLORE>" in step["response"]:
                modes.append("EXPLORE")
            elif "<VERIFY>" in step["response"]:
                modes.append("VERIFY")

    mode_counts = Counter(modes)
    print("\nMode Distribution:")
    for mode, count in mode_counts.most_common():
        print(f"  {mode}: {count} ({count/len(modes)*100:.1f}%)")

    # 2. 轨迹长度
    lengths = [len(item["data"]) for item in sft_data]
    print(f"\nTrajectory Length:")
    print(f"  Mean: {np.mean(lengths):.1f}")
    print(f"  Std: {np.std(lengths):.1f}")
    print(f"  Min: {min(lengths)}, Max: {max(lengths)}")

    # 3. 任务覆盖
    tasks = set(item["task"] for item in sft_data)
    print(f"\nTask Coverage: {len(tasks)} unique tasks")

    # 4. 格式检查
    format_errors = 0
    for item in sft_data:
        for step in item["data"]:
            response = step["response"]
            if not ("<" in response and ">" in response and "<action>" in response):
                format_errors += 1

    print(f"\nFormat Errors: {format_errors} / {total_steps} ({format_errors/total_steps*100:.2f}%)")

    return {
        "total_trajectories": len(sft_data),
        "total_steps": total_steps,
        "mode_distribution": mode_counts,
        "avg_length": np.mean(lengths),
        "format_error_rate": format_errors / total_steps
    }
```

---

## 八、实施步骤

### 8.1 环境准备

```bash
# 1. 确保OpenAI API Key已设置
export OPENAI_API_KEY="sk-..."

# 2. 确保有足够的API额度
# 预估成本：500条轨迹 × 平均15步 × 2次API调用 = 15,000次调用
# GPT-4o成本：约 $75-150（取决于输入/输出长度）

# 3. 准备专家轨迹数据
cd code
python -c "
from datasets import load_from_disk
dataset = load_from_disk('data/alfworld_expert_traj')
print(f'Available trajectories: {len(dataset)}')
"
```

### 8.2 生成数据

```bash
# 运行改进后的数据生成脚本
cd code
python scripts/alfworld_prepare_plan_c.py \
    --num_trajectories 500 \
    --output_path data/alfworld_cold-start_plan_c.json \
    --model gpt-4o \
    --augment_ratio 0.3

# 预计运行时间：4-6小时（取决于API速度）
```

### 8.3 数据验证

```bash
# 运行数据质量检查
python scripts/validate_cold_start_data.py \
    --data_path data/alfworld_cold-start_plan_c.json

# 检查项：
# ✓ 模式分布是否平衡
# ✓ 轨迹长度是否合理
# ✓ 格式是否正确
# ✓ 信念状态是否完整
```

### 8.4 转换为Parquet格式

```bash
# 转换为veRL训练所需的格式
python -m examples.data_preprocess.cold_start_data \
    --local_dir=$HOME/data/alfworld \
    --data_source=data/alfworld_cold-start_plan_c.json

# 输出：
# $HOME/data/alfworld/train.parquet
# $HOME/data/alfworld/val.parquet (从train中划分10%)
```

### 8.5 训练冷启动模型

```bash
# 使用生成的数据训练冷启动模型
# 见 BDRS_EXPERIMENT_PLAN.md 第2.4节

cd code

# 8卡训练（推荐）
torchrun --standalone --nnodes=1 --nproc_per_node=8 \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$HOME/data/alfworld/train.parquet \
    data.val_files=$HOME/data/alfworld/val.parquet \
    data.prompt_key=extra_info \
    data.response_key=extra_info \
    data.max_length=3000 \
    +data.prompt_dict_keys=['question'] \
    +data.response_dict_keys=['answer'] \
    optim.lr=1e-5 \
    data.micro_batch_size_per_gpu=16 \
    model.partial_pretrain=Qwen/Qwen2.5-7B-Instruct \
    trainer.default_hdfs_dir=null \
    trainer.project_name=BDRS \
    trainer.experiment_name=qwen7b_cold_start_plan_c \
    trainer.total_epochs=5 \
    trainer.default_local_dir=./checkpoints/cold_start/alfworld/qwen7b_plan_c \
    trainer.logger=['console','wandb'] \
    ulysses_sequence_parallel_size=4 \
    use_remove_padding=true

# 训练完成后检查点保存在：
# ./checkpoints/cold_start/alfworld/qwen7b_plan_c/default/epoch_5
```

### 8.6 评估冷启动模型

```bash
# 在验证集上测试冷启动模型
bash examples/bdrs_trainer/eval_alfworld.sh \
    actor_rollout_ref.model.path=./checkpoints/cold_start/alfworld/qwen7b_plan_c/default/epoch_5 \
    env.alfworld.generalization_level=0 \
    data.val_batch_size=64

# 预期结果：
# 方案C冷启动模型成功率：30-40%
# 对比基线（无优化）：10-15%
# 提升：2-3x
```

---

## 九、预期效果与对比

### 9.1 数据质量对比

| 指标 | 基线方法 | 方案A | 方案B | 方案C |
|------|---------|-------|-------|-------|
| **轨迹数量** | 300 | 300 | 300 | 500 |
| **总样本数** | ~4,500 | ~4,500 | ~6,000 | ~10,000 |
| **PLAN比例** | 8% | 10% | 12% | 15% |
| **EXECUTE比例** | 68% | 65% | 55% | 50% |
| **EXPLORE比例** | 20% | 22% | 28% | 25% |
| **VERIFY比例** | 4% | 3% | 5% | 10% |
| **包含失败样本** | ✗ | ✗ | ✗ | ✓ |
| **信念状态显式** | ✗ | ✗ | ✓ | ✓ |
| **数据增强** | ✗ | ✗ | ✗ | ✓ |

### 9.2 冷启动模型性能对比

| 方法 | 成功率 | 平均步数 | 有效动作率 | 训练时间 |
|------|--------|----------|-----------|----------|
| **无冷启动** | 2-5% | 30+ | 75% | - |
| **基线冷启动** | 10-15% | 25 | 85% | 1小时 |
| **方案A** | 15-20% | 23 | 88% | 1小时 |
| **方案B** | 20-30% | 21 | 90% | 1.5小时 |
| **方案C** | 30-40% | 18 | 92% | 2小时 |

### 9.3 RL训练加速效果

| 方法 | 达到80%成功率所需Epoch | 最终成功率 | 总训练时间 |
|------|----------------------|-----------|-----------|
| **无冷启动** | 100+ | 75% | 10天 |
| **基线冷启动** | 70 | 82% | 7天 |
| **方案C冷启动** | **45** | **88%** | **5天** |

**提升总结**：
- 冷启动性能提升：2-3x
- RL收敛速度提升：35%
- 最终性能提升：6个百分点
- 总开发时间节省：40%

---

## 十、常见问题与解决

### Q1：API成本太高怎么办？

**方案1**：使用更便宜的模型
```python
# 改用GPT-4o-mini（成本降低10x）
MODEL = "gpt-4o-mini"

# 预估成本：$7-15（vs $75-150）
```

**方案2**：减少轨迹数量
```python
# 先用200条测试
num_trajectories = 200

# 如果效果好，再扩展到500条
```

**方案3**：分阶段实施
```python
# 第一阶段：只实施策略1+2+3（核心策略）
# 第二阶段：如果需要，再加策略4+5
```

### Q2：标注速度太慢怎么办？

**方案1**：并行处理
```python
from multiprocessing import Pool

def parallel_annotation(trajectories, num_workers=8):
    with Pool(num_workers) as pool:
        results = pool.map(annotate_step_by_step, trajectories)
    return results
```

**方案2**：使用缓存
```python
# 缓存LLM响应，避免重复调用
@lru_cache(maxsize=10000)
def llm_cached(prompt, model):
    return llm(prompt, model)
```

### Q3：模式分布仍然不平衡怎么办？

**方案1**：调整目标分布
```python
# 如果EXPLORE样本不足，提高目标比例
TARGET_MODE_DISTRIBUTION = {
    "PLAN": 0.12,
    "EXECUTE": 0.48,
    "EXPLORE": 0.30,  # 提高
    "VERIFY": 0.10
}
```

**方案2**：主动生成少数模式样本
```python
# 专门生成VERIFY样本
def generate_more_verify_samples(trajectories):
    verify_samples = []
    for traj in trajectories:
        # 找到所有可能需要验证的场景
        for step in traj:
            if is_potential_verification_point(step):
                verify_sample = construct_verify_sample(step)
                verify_samples.append(verify_sample)
    return verify_samples
```

### Q4：冷启动模型效果仍不理想怎么办？

**诊断步骤**：
```bash
# 1. 检查数据质量
python scripts/validate_cold_start_data.py --data_path data/alfworld_cold-start_plan_c.json

# 2. 检查模式分布
# 3. 抽查10条样本，人工评估标注质量
# 4. 检查训练loss曲线
# 5. 在验证集上采样模型输出，检查格式正确性
```

**可能原因与解决**：
1. **数据质量问题** → 重新标注
2. **训练不充分** → 增加epoch到10
3. **学习率不当** → 尝试1e-6或2e-5
4. **模型容量不足** → 使用Qwen2.5-7B而非1.5B

---

## 十一、检查清单

### 数据生成前
- [ ] OpenAI API Key已设置且有充足额度
- [ ] 专家轨迹数据已准备（data/alfworld_expert_traj）
- [ ] 代码环境已安装所有依赖
- [ ] 磁盘空间充足（>10GB）
- [ ] 已阅读并理解所有5个策略

### 数据生成中
- [ ] 标注进度正常（每条轨迹<5分钟）
- [ ] API调用无频繁错误
- [ ] 中间结果可以保存（防止中断）
- [ ] 定期检查输出格式

### 数据生成后
- [ ] 总样本数达到8000+
- [ ] 模式分布接近目标（PLAN 15%, EXECUTE 50%, EXPLORE 25%, VERIFY 10%）
- [ ] 格式错误率<5%
- [ ] 随机抽查10条样本，人工确认质量
- [ ] 数据已转换为parquet格式
- [ ] 数据已保存并备份

### 冷启动训练后
- [ ] 训练loss正常下降
- [ ] 无过拟合迹象
- [ ] 验证集成功率达到30-40%
- [ ] 模型输出格式正确率>95%
- [ ] 检查点已保存

---

## 十二、总结

方案C通过五大策略全面优化冷启动数据：

1. **逐步标注** - 避免事后偏见，提升标注真实性
2. **信念显式化** - 让模型学会构造和使用信念
3. **模式平衡** - 解决分布不平衡，覆盖所有推理模式
4. **失败-恢复** - 增强错误处理和恢复能力
5. **数据增强** - 提升多样性和鲁棒性

**预期收益**：
- 冷启动模型成功率：30-40%（基线10-15%）
- RL训练加速：35%（45 vs 70 epochs）
- 最终性能提升：6个百分点（88% vs 82%）

**实施成本**：
- 开发时间：2-3天
- API成本：$75-150（GPT-4o）或$7-15（GPT-4o-mini）
- 计算资源：8×A100训练2小时

**下一步**：立即运行 `scripts/alfworld_prepare_plan_c.py` 开始生成数据！
