# RLVMR (Reinforcement Learning with Verbal Meta-cognitive Rewards) 详细拆解

> 本文档基于代码实现进行分析，所有结论都有明确的代码依据

## 目录
1. [概述](#1-概述)
2. [数据处理流程](#2-数据处理流程)
3. [数据格式详解](#3-数据格式详解)
4. [RLVMR算法核心](#4-rlvmr算法核心)
5. [训练流程](#5-训练流程)
6. [Reward计算机制](#6-reward计算机制)
7. [关键配置参数](#7-关键配置参数)

---

## 1. 概述

RLVMR是一个结合了**Episode-level奖励**和**Step-level元认知奖励**的强化学习算法，专为具身智能Agent设计。

### 1.1 核心特点
- **双重优势函数**：Episode Advantage + Step Advantage
- **Meta-cognitive Tags**：通过语言标签(`<planning>`, `<explore>`, `<reflection>`, `<monitor>`)引导Agent推理
- **分组归一化**：不同tag类型分别进行优势值归一化

**代码依据**：
- `code/rlvmr/core_rlvmr.py:7-40` - RLVMR核心算法
- `code/verl/trainer/ppo/ray_trainer.py:76` - AdvantageEstimator.RLVMR定义

---

## 2. 数据处理流程

### 2.1 完整数据处理Pipeline

```
原始专家轨迹 (HuggingFace)
    ↓
加载到本地 (data/alfworld_expert_traj)
    ↓
GPT-4o标注推理过程 (alfworld_prepare.py)
    ↓
生成Cold-start数据 (data/alfworld_cold-start.json)
    ↓
转换为训练格式 (cold_start_data.py)
    ↓
Parquet训练数据 (~/data/alfworld/train.parquet)
    ↓
RLVMR训练 (main_ppo.py)
```

### 2.2 各阶段详细说明

#### 阶段1: 原始数据加载
**文件**：`code/scripts/alfworld_prepare.py:17`
```python
alfworld_dataset = load_from_disk("data/alfworld_expert_traj")
```

**原始数据来源**：
- 数据集：`AgentGym/AgentTraj-L`
- 包含14,485个样本（多环境：alfworld, babyai, sciworld等）
- 存储位置：`code/data/alfworld_expert_traj/`

**代码依据**：`code/data/alfworld_expert_traj/dataset_info.json:8-12`

#### 阶段2: 轨迹标注
**文件**：`code/scripts/alfworld_prepare.py:100-108`
```python
for traj in trajs[:NUM_TRAJS]:
    prompt = ALFWORLD_TAGGING_TEMPLATE.format(traj=json.dumps(traj["traj"]))
    res = llm_json(prompt, MODEL)  # MODEL = "gpt-4o"
    valid, res = merge(res, traj["traj"])
```

**标注配置**：
- 模型：GPT-4o (`code/scripts/alfworld_prepare.py:14`)
- 轨迹数量：300条 (`code/scripts/alfworld_prepare.py:12`)
- 保存路径：`data/alfworld_cold-start.json` (`code/scripts/alfworld_prepare.py:15`)

#### 阶段3: SFT数据生成
**文件**：`code/scripts/alfworld_prepare.py:116-159`

为每个轨迹的每一步生成prompt-response对：
```python
for i, item in enumerate(res):
    if i == 0:
        prompt = ALFWORLD_TEMPLATE_NO_HIS_CS.format(
            current_observation=item["obs"]
        )
    else:
        prompt = ALFWORLD_TEMPLATE_CS.format(
            task_description=traj["task"],
            action_history=action_history,
            current_observation=item["obs"],
            planning=latest_planning
        )
    response = f"{item['reason']}\n<action>{item['action']}</action>\n"
```

#### 阶段4: 转换为Parquet格式
**文件**：`code/examples/data_preprocess/cold_start_data.py:36-84`
```python
with open(data_source, 'r') as f:
    dataset = json.load(f)
success_data = [item.get('data', item.get('traj')) for item in dataset]
train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
```

---

## 3. 数据格式详解

### 3.1 原始专家轨迹格式

**文件**：`code/data/alfworld_expert_traj/dataset_info.json:55-75`

**格式定义**：
```json
{
    "conversations": [
        {
            "from": "human" | "gpt",
            "value": "对话内容",
            "loss": true | false
        }
    ],
    "item_id": "唯一标识符"
}
```

**真实示例**（直接来自 `code/data/alfworld_expert_traj/data-00000-of-00001.arrow`，AgentGym/AgentTraj-L数据集，共2224条ALFWorld轨迹）：

```json
{
  "conversations": [
    {
      "from": "human",
      "loss": null,
      "value": "Interact with a household to solve a task. Imagine you are an intelligent agent in a household environment and your target is to perform actions to complete the task goal. At the beginning of your interactions, you will be given the detailed description of the current environment and your goal to accomplish. For each of your turn, you will be given a list of actions which you can choose one to perform in this turn. You should first think about the current condition and plan for your future actions, and then output your action in this turn. Your output must strictly follow this format:\"Thought: your thoughts.\nAction: your next action\n\"."
    },
    {
      "from": "gpt",
      "loss": false,
      "value": "OK. I'll follow your instructions and try my best to solve the task."
    },
    {
      "from": "human",
      "loss": null,
      "value": "You are in the middle of a room. Looking quickly around you, you see a bed 1, a desk 1, a drawer 3, a drawer 2, a drawer 1, a garbagecan 1, a laundryhamper 1, a shelf 2, a shelf 1, and a sidetable 1.\nYour task is to: put some alarmclock on desk.\nAVAILABLE ACTIONS: go to bed 1,go to desk 1,go to drawer 1,go to drawer 2,go to drawer 3,go to garbagecan 1,go to laundryhamper 1,go to shelf 1,go to shelf 2,go to sidetable 1,inventory,look"
    },
    {
      "from": "gpt",
      "loss": true,
      "value": "Thought: I need to find an alarmclock and put it on the desk. Alarmclocks usually located on bedside tables or in the drawers. Therefore, I will check the drawers first.\nAction: go to drawer 1 "
    },
    {
      "from": "human",
      "loss": null,
      "value": "The drawer 1 is closed."
    },
    {
      "from": "gpt",
      "loss": true,
      "value": "Action: open drawer 1 "
    },
    {
      "from": "human",
      "loss": null,
      "value": "You open the drawer 1. The drawer 1 is open. In it, you see a remotecontrol 1."
    },
    {
      "from": "gpt",
      "loss": true,
      "value": "Action: go to drawer 2 "
    },
    // ... (省略中间8轮对话：检查drawer 2、drawer 3、shelf 1、shelf 2)
    {
      "from": "human",
      "loss": null,
      "value": "On the sidetable 1, you see a alarmclock 2, a alarmclock 1, a creditcard 1, a desklamp 1, a keychain 1, and a pen 1."
    },
    {
      "from": "gpt",
      "loss": true,
      "value": "Action: take alarmclock 1 from sidetable 1 "
    },
    {
      "from": "human",
      "loss": null,
      "value": "You pick up the alarmclock 1 from the sidetable 1."
    },
    {
      "from": "gpt",
      "loss": true,
      "value": "Action: go to desk 1 "
    },
    {
      "from": "human",
      "loss": null,
      "value": "On the desk 1, you see a book 2, a bowl 1, a cd 3, a cd 2, a cd 1, a cellphone 1, a keychain 2, and a pencil 2."
    },
    {
      "from": "gpt",
      "loss": true,
      "value": "Action: put alarmclock 1 in/on desk 1 "
    }
  ],
  "item_id": "pick_and_place_simple-AlarmClock-None-Desk-307_trial_T20190907_072329_828593"
}
```

**轨迹统计**：
- **对话轮数**: 28轮 (14轮环境观察 + 14轮智能体动作)
- **有效动作**: 13步 (loss=true的gpt回复)
- **任务类型**: pick_and_place_simple
- **完整数据**: 已保存至 `code/alfworld_example.json`

**解析逻辑**（`code/scripts/alfworld_prepare.py:20-37`）：
```python
conversations = traj["conversations"][2:]  # 跳过前两个系统对话
for i in range(0, len(conversations), 2):
    obs = conversations[i]['value'].split("AVAILABLE ACTIONS:")[0]  # ⚠️ 删除AVAILABLE ACTIONS
    llm_action = conversations[i + 1]['value']
    action = llm_action.split("Action:")[-1].strip()
```

#### 3.1.1 AVAILABLE ACTIONS的处理对比

**关键问题**：为什么冷启动训练要删除AVAILABLE ACTIONS？

**原始轨迹（保留AVAILABLE ACTIONS）**：
```
You are in the middle of a room. Looking quickly around you, you see a bed 1, a desk 1, a drawer 3, a drawer 2, a drawer 1, a garbagecan 1, a laundryhamper 1, a shelf 2, a shelf 1, and a sidetable 1.
Your task is to: put some alarmclock on desk.
AVAILABLE ACTIONS: go to bed 1,go to desk 1,go to drawer 1,go to drawer 2,go to drawer 3,go to garbagecan 1,go to laundryhamper 1,go to shelf 1,go to shelf 2,go to sidetable 1,inventory,look
```

**处理后（删除AVAILABLE ACTIONS）**：
```
You are in the middle of a room. Looking quickly around you, you see a bed 1, a desk 1, a drawer 3, a drawer 2, a drawer 1, a garbagecan 1, a laundryhamper 1, a shelf 2, a shelf 1, and a sidetable 1.
Your task is to: put some alarmclock on desk.
```

**设计意图**（渐进式课程学习）：

| 阶段 | 使用的Prompt | AVAILABLE ACTIONS | 难度 | 目的 |
|------|-------------|-------------------|------|------|
| **冷启动SFT** | #7, #8 (cold_start.py) | ❌ **删除** | 🔴 困难 | 强迫模型学会meta-cognitive推理 |
| **冷启动评测** | #5, #6 (alfworld.py) | ✅ **保留** | 🟢 简单 | 测试模型是否准备好进入RL |
| **RLVMR训练+评测** | #5, #6 (alfworld.py) | ✅ **保留** | 🟢 简单 | 在真实交互条件下强化学习 |

**代码证据**：

1. **冷启动数据准备**（`alfworld_prepare.py:222`）：
```python
obs = conversations[i]['value'].split("AVAILABLE ACTIONS:")[0]  # 删除！
```

2. **RLVMR训练时的Prompt**（`alfworld.py:85-127`）：
```python
ALFWORLD_TEMPLATE_NO_HIS_MC = """
Your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].  # 保留！
```

3. **环境管理器注入动作列表**（`env_manager.py:100-110`）：
```python
obs = _ALFWORLD_TEMPLATE_NO_HIS.format(
    current_observation=text_obs[i],
    admissible_actions=reformatted_admissible_actions  # RL阶段重新添加！
)
```

**实验对比价值**：
- 保留AVAILABLE ACTIONS：模型可以直接从列表中选择，降低探索难度
- 删除AVAILABLE ACTIONS：模型必须通过推理生成动作，增强泛化能力

建议通过消融实验对比两种策略的效果差异。

### 3.2 提取后的中间格式

**处理**：`code/scripts/alfworld_prepare.py:20-37`

基于上述真实示例（item_id: pick_and_place_simple-AlarmClock-None-Desk-307），经过解析后得到：
```python
{
    "task": "put some alarmclock on desk",
    "traj": [
        {
            "observation": "You are in the middle of a room. Looking quickly around you, you see a bed 1, a desk 1, a drawer 3, a drawer 2, a drawer 1, a garbagecan 1, a laundryhamper 1, a shelf 2, a shelf 1, and a sidetable 1.",
            "action": "go to drawer 1"
        },
        {
            "observation": "The drawer 1 is closed.",
            "action": "open drawer 1"
        },
        {
            "observation": "You open the drawer 1. The drawer 1 is open. In it, you see a remotecontrol 1.",
            "action": "go to drawer 2"
        },
        // ... (省略中间步骤：检查drawer 2、drawer 3、shelf 1、shelf 2)
        {
            "observation": "On the sidetable 1, you see a alarmclock 2, a alarmclock 1, a creditcard 1, a desklamp 1, a keychain 1, and a pen 1.",
            "action": "take alarmclock 1 from sidetable 1"
        },
        {
            "observation": "You pick up the alarmclock 1 from the sidetable 1.",
            "action": "go to desk 1"
        },
        {
            "observation": "On the desk 1, you see a book 2, a bowl 1, a cd 3, a cd 2, a cd 1, a cellphone 1, a keychain 2, and a pencil 2.",
            "action": "put alarmclock 1 in/on desk 1"
        }
    ]
}
```

**注**: 完整轨迹包含13个action步骤，此处仅展示首尾关键步骤。

### 3.3 GPT-4o标注后的格式

**处理**：`code/scripts/alfworld_prepare.py:100-108` (使用GPT-4o标注推理过程)

基于3.2的数据，经过GPT-4o标注后添加reasoning（示例性标注）：
```json
{
    "task": "put some alarmclock on desk",
    "traj": [
        {
            "obs": "You are in the middle of a room. Looking quickly around you, you see a bed 1, a desk 1, a drawer 3, a drawer 2, a drawer 1, a garbagecan 1, a laundryhamper 1, a shelf 2, a shelf 1, and a sidetable 1.",
            "reason": "<planning>Step 1: Locate an alarm clock (check drawers and sidetable). Step 2: Pick up the alarm clock. Step 3: Navigate to the desk. Step 4: Place the alarm clock on the desk.</planning>",
            "action": "go to drawer 1"
        },
        {
            "obs": "The drawer 1 is closed.",
            "reason": "<monitor>Currently executing step 1 of the plan. The drawer is closed, so I need to open it to check inside.</monitor>",
            "action": "open drawer 1"
        },
        {
            "obs": "You open the drawer 1. The drawer 1 is open. In it, you see a remotecontrol 1.",
            "reason": "<explore>No alarm clock in drawer 1. Need to check other locations: drawer 2, drawer 3, or sidetable might contain the alarm clock.</explore>",
            "action": "go to drawer 2"
        },
        // ... (省略中间标注步骤)
        {
            "obs": "On the sidetable 1, you see a alarmclock 2, a alarmclock 1, a creditcard 1, a desklamp 1, a keychain 1, and a pen 1.",
            "reason": "<monitor>Found alarm clocks on sidetable 1. Now executing step 2: pick up an alarm clock to complete the task.</monitor>",
            "action": "take alarmclock 1 from sidetable 1"
        },
        {
            "obs": "You pick up the alarmclock 1 from the sidetable 1.",
            "reason": "<monitor>Successfully picked up alarm clock 1. Now executing step 3: navigate to the desk.</monitor>",
            "action": "go to desk 1"
        },
        {
            "obs": "On the desk 1, you see a book 2, a bowl 1, a cd 3, a cd 2, a cd 1, a cellphone 1, a keychain 2, and a pencil 2.",
            "reason": "<monitor>Reached desk 1. Now executing step 4: place the alarm clock on the desk to complete the task.</monitor>",
            "action": "put alarmclock 1 in/on desk 1"
        }
    ]
}
```

**注意**：上述reasoning是示例性的，实际由GPT-4o根据`ALFWORLD_TAGGING_TEMPLATE`生成。原始数据没有标注，只有300条轨迹被GPT-4o标注用于cold-start训练。

### 3.4 Cold-start SFT数据格式

**处理**：`code/scripts/alfworld_prepare.py:116-159` (为每一步生成prompt-response对)

基于3.3的标注数据，生成训练用的step-level数据：
```json
{
    "task": "put some alarmclock on desk",
    "done": "True",
    "data": [
        {
            "step": 1,
            "obs": "You are in the middle of a room. Looking quickly around you, you see a bed 1, a desk 1, a drawer 3, a drawer 2, a drawer 1, a garbagecan 1, a laundryhamper 1, a shelf 2, a shelf 1, and a sidetable 1.",
            "prompt": "You are an expert agent operating in the ALFRED Embodied Environment.\nYour current observation is: You are in the middle of a room. Looking quickly around you, you see a bed 1, a desk 1, a drawer 3, a drawer 2, a drawer 1, a garbagecan 1, a laundryhamper 1, a shelf 2, a shelf 1, and a sidetable 1.\n\nNow it's your turn to take an action, following these steps:\n\n1. First, reason using ONLY ONE tag pair:\n   <planning>...</planning>\n   <explore>...</explore>\n   <reflection>...</reflection>\n   <monitor>...</monitor>\n\n2. After your reasoning, select and present an action within <action></action> tags.",
            "response": "<planning>Step 1: Locate an alarm clock (check drawers and sidetable). Step 2: Pick up the alarm clock. Step 3: Navigate to the desk. Step 4: Place the alarm clock on the desk.</planning>\n<action>go to drawer 1</action>\n"
        },
        {
            "step": 2,
            "obs": "The drawer 1 is closed.",
            "prompt": "You are an expert agent operating in the ALFRED Embodied Environment. Your task is to: put some alarmclock on desk\nPrior to this step, you have already taken 1 step(s). Below are the most recent 1 observations and actions: [Observation 1: 'You are in the middle of a room...', Action 1: 'go to drawer 1']\n- recent reasoning process: \n[Observation 1: You are in the middle of a room..., output: '<planning>Step 1: Locate an alarm clock...</planning> <action>go to drawer 1</action>']\nYou are now at step 2 and your current observation is: The drawer 1 is closed.\n\nYour previous overall plan is: Step 1: Locate an alarm clock (check drawers and sidetable). Step 2: Pick up the alarm clock. Step 3: Navigate to the desk. Step 4: Place the alarm clock on the desk.\n\nNow it's your turn to take an action...",
            "response": "<monitor>Currently executing step 1 of the plan. The drawer is closed, so I need to open it to check inside.</monitor>\n<action>open drawer 1</action>\n"
        }
        // ... 后续11个步骤省略 (完整轨迹共13步)
    ]
}
```

**注**: 此为cold-start SFT训练数据格式，保存为JSON文件（如 `alfworld_cs_300_tagged.json`）。

### 3.5 训练Parquet数据格式

**处理**：`code/examples/data_preprocess/cold_start_data.py:36-84`

将3.4的数据展平为单个sample，转换为Parquet格式：
```python
{
    "data_source": "cold_start",
    "prompt": [{
        "role": "user",
        "content": "You are an expert agent operating in the ALFRED Embodied Environment.\nYour current observation is: You are in the middle of a room. Looking quickly around you, you see a bed 1, a desk 1, a drawer 3, a drawer 2, a drawer 1, a garbagecan 1, a laundryhamper 1, a shelf 2, a shelf 1, and a sidetable 1."
    }],
    "ability": "agent",
    "reward_model": {
        "style": "instruction_following",
        "ground_truth": "You are an expert agent operating..."
    },
    "extra_info": {
        "split": "train",
        "index": 0,
        "answer": "<planning>Step 1: Locate an alarm clock (check drawers and sidetable). Step 2: Pick up the alarm clock. Step 3: Navigate to the desk. Step 4: Place the alarm clock on the desk.</planning>\n<action>go to drawer 1</action>\n",
        "question": "You are an expert agent operating..."
    }
}
```

**说明**：
- 每个step都是一个独立的训练样本
- `prompt`：用户输入（包含observation和历史）
- `extra_info.answer`：模型应该生成的响应（reasoning + action）
- 真实数据示例：item_id `pick_and_place_simple-AlarmClock-None-Desk-307` 会生成13个训练样本

### 3.6 训练时DataProto格式

**文件**：`code/agent_system/multi_turn_rollout/rollout_loop.py:252-264`

在RLVMR训练的rollout过程中，收集的实时数据：
```python
{
    # Tensor数据 (batch形式)
    "input_ids": torch.Tensor([[101, 2017, 2024, ...], ...]),  # shape: (batch_size, seq_len)
    "attention_mask": torch.Tensor([[1, 1, 1, ...], ...]),     # shape: (batch_size, seq_len)
    "responses": torch.Tensor([[2175, 2000, ...], ...]),       # shape: (batch_size, response_len)
    "old_log_probs": torch.Tensor([[-2.3, -1.5, ...], ...]),   # shape: (batch_size, response_len)
    "rlvmr_step_reward": torch.Tensor([1.0, 0.0, ...]),        # shape: (batch_size,) RLVMR独有

    # Non-tensor数据 (numpy/list形式)
    "episode_rewards": np.array([1.0, 0.0, 1.0, ...]),         # shape: (batch_size,)
    "episode_lengths": np.array([13, 8, 11, ...]),              # shape: (batch_size,) 示例：13步完成任务
    "rlvmr_tag_type": ["<planning>", "<monitor>", ...],        # RLVMR独有
    "is_action_valid": np.array([True, True, False, ...]),
    "uid": ["uuid-1", "uuid-1", "uuid-2", ...],                # 用于GRPO分组
    "traj_uid": ["traj-1", "traj-2", "traj-3", ...],           # 用于RLVMR轨迹标识
    "full_output": ["<planning>Step 1: Locate an alarm clock...</planning>\n<action>go to drawer 1</action>", ...]
}
```

**说明**：
- 这是训练循环中的实时数据，不同于预处理的SFT数据
- `rlvmr_step_reward`：根据meta-cognitive tag计算的step-level奖励
- `rlvmr_tag_type`：从`full_output`中提取的tag类型，用于分组归一化
- 示例来自真实轨迹：`pick_and_place_simple-AlarmClock-None-Desk-307` (13个action步骤)

**代码依据**：`code/agent_system/multi_turn_rollout/rollout_loop.py:228-248`, `code/rlvmr/core_rlvmr.py:94-99`

---

## 4. RLVMR算法核心

### 4.1 双重优势函数设计

**文件**：`code/rlvmr/core_rlvmr.py:7-40`

```python
def compute_rlvmr_outcome_advantage(
    token_level_rewards: torch.Tensor,    # Episode奖励
    rlvmr_rewards: torch.Tensor,          # Step奖励
    eos_mask: torch.Tensor,
    index: np.array,                      # 用于episode分组
    step_advantage_w: float = 1.0,        # Step优势权重
    mode: str = "mean_norm",              # 归一化模式
    tag_types: np.array = None,           # Meta-cognitive标签
    traj_id: np.array = None              # 轨迹ID
):
    # Episode-level优势函数
    episode_advantages = episode_norm_reward(
        token_level_rewards, eos_mask, index, epsilon, remove_std
    )

    # Step-level优势函数（按tag分组归一化）
    step_advantages = step_norm_reward_by_tag(
        step_rewards=rlvmr_rewards,
        eos_mask=eos_mask,
        index=index,
        tag_types=tag_types,
        epsilon=epsilon,
        remove_std=remove_std,
        traj_id=traj_id
    )

    # 加权组合
    scores = episode_advantages + step_advantage_w * step_advantages
    return scores, scores, {...}
```

### 4.2 Step Reward计算

**文件**：`code/rlvmr/core_rlvmr.py:75-146`

```python
def process_trajectory_rlvmr_rewards(trajectory_list, config, episode_rewards):
    for traj_idx, trajectory in enumerate(trajectory_list):
        is_success = episode_rewards[traj_idx] > 0

        for step_idx, step_data in enumerate(trajectory):
            full_output = step_data["full_output"]
            is_valid = step_data["is_action_valid"]

            # 默认无奖励
            step_data['rlvmr_tag_type'] = "none"
            step_data['rlvmr_step_reward'] = torch.tensor(0.0)

            if not is_valid:
                continue

            # 1. Planning奖励
            if '<planning>' in full_output:
                step_data["rlvmr_tag_type"] = "<planning>"
                if is_success:
                    step_data['rlvmr_step_reward'] = torch.tensor(
                        config.algorithm.rlvmr.planning_reward
                    )

            # 2. Exploration奖励（检查是否为重复动作）
            elif '<explore>' in full_output:
                step_data["rlvmr_tag_type"] = "<explore>"
                is_repeated = check_repeated_action(trajectory, step_idx)
                if not is_repeated:
                    step_data['rlvmr_step_reward'] = torch.tensor(
                        config.algorithm.rlvmr.exploration_reward
                    )

            # 3. Reflection奖励（检查是否在失败后改变动作）
            elif '<reflection>' in full_output:
                step_data["rlvmr_tag_type"] = "<reflection>"
                if prev_invalid and action_changed:
                    step_data['rlvmr_step_reward'] = torch.tensor(
                        config.algorithm.rlvmr.reflection_reward
                    )
```

**奖励逻辑总结**：
| Tag | 条件 | 奖励值 |
|-----|------|--------|
| `<planning>` | 成功轨迹中出现 | `planning_reward` (默认1.0) |
| `<planning>` | 失败轨迹中出现 | 0.0 |
| `<explore>` | 非重复动作 | `exploration_reward` (默认1.0) |
| `<explore>` | 重复动作 | 0.0 |
| `<reflection>` | 前一步失败且改变动作 | `reflection_reward` (默认1.0) |
| `<reflection>` | 其他情况 | 0.0 |
| `<monitor>` | 任何情况 | 0.0（不影响奖励） |

### 4.3 按Tag分组归一化

**文件**：`code/rlvmr/core_rlvmr.py:155-196`

```python
def step_norm_reward_by_tag(
    step_rewards: torch.Tensor,
    eos_mask: torch.Tensor,
    index: np.array,           # prompt ID
    tag_types: np.array,       # Meta-cognitive标签
    remove_std: bool = True
):
    # 构建分组key：(prompt_id, tag_type)
    group_keys = []
    for i in range(len(index)):
        group_key = (index[i], tag_types[i])
        group_keys.append(group_key)

    # 按组计算均值和标准差
    id2score = defaultdict(list)
    for i in range(bsz):
        group_key = group_keys[i]
        id2score[group_key].append(scores[i])

    for group_key in id2score:
        if len(id2score[group_key]) > 1:
            id2mean[group_key] = torch.mean(torch.tensor(id2score[group_key]))
            id2std[group_key] = torch.std(torch.tensor(id2score[group_key]))
        else:
            id2mean[group_key] = torch.tensor(0.0)
            id2std[group_key] = torch.tensor(1.0)

    # 归一化
    for i in range(bsz):
        if remove_std:
            scores[i] = scores[i] - id2mean[group_key]
        else:
            scores[i] = (scores[i] - id2mean[group_key]) / (id2std[group_key] + epsilon)
```

**关键点**：
- 同一个prompt下，不同tag类型分别归一化
- 例如：(prompt_1, "<planning>")和(prompt_1, "<explore>")是两个独立的分组
- 这样可以让不同类型的meta-cognitive行为有独立的奖励分布

---

## 5. 训练流程

### 5.1 完整训练流程

**主入口**：`code/verl/trainer/main_ppo.py:52-183`

```
1. 初始化Ray集群
    ↓
2. 加载模型和Tokenizer
    ↓
3. 创建环境 (make_envs)
    ↓
4. 初始化Workers (ActorRollout, Critic, Ref)
    ↓
5. 创建TrajectoryCollector
    ↓
6. 创建RewardManager
    ↓
7. 启动RayPPOTrainer
```

### 5.2 单个Epoch训练流程

**文件**：`code/verl/trainer/ppo/ray_trainer.py` (RayPPOTrainer.fit方法)

```
1. 加载训练Batch
    ↓
2. Multi-turn Rollout (收集轨迹)
    ├─ 环境reset
    ├─ Loop: 最多max_steps步
    │   ├─ preprocess_batch (构建prompt)
    │   ├─ actor.generate (生成response)
    │   ├─ env.step (环境交互)
    │   ├─ 记录reward, done, info
    │   └─ 更新observation
    └─ 如果启用RLVMR：process_trajectory_rlvmr_rewards
    ↓
3. gather_rollout_data (整理为DataProto)
    ↓
4. 计算Reward (EpisodeRewardManager)
    ↓
5. 计算Advantage (compute_advantage)
    ├─ Episode advantage (按uid分组归一化)
    └─ Step advantage (按uid+tag分组归一化)
    ↓
6. PPO更新
    ├─ Actor更新 (PPO clip)
    └─ Critic更新 (Value loss)
    ↓
7. 记录Metrics
```

### 5.3 Multi-turn Rollout详解

**文件**：`code/agent_system/multi_turn_rollout/rollout_loop.py:266-391`

```python
def vanilla_multi_turn_loop(self, gen_batch, actor_rollout_wg, envs):
    # 环境初始化
    obs, infos = envs.reset()

    # 轨迹收集
    total_batch_list = [[] for _ in range(batch_size)]
    episode_rewards = np.zeros(batch_size)
    episode_lengths = np.zeros(batch_size)

    for _step in range(max_steps):
        # 1. 预处理observation为模型输入
        batch = self.preprocess_batch(gen_batch, obs)

        # 2. 生成action
        batch_output = actor_rollout_wg.generate_sequences(batch_input)

        # 3. 解码action文本
        text_actions = tokenizer.batch_decode(batch_output['responses'])

        # 4. 环境step
        next_obs, rewards, dones, infos = envs.step(text_actions)

        # 5. 记录数据
        batch.non_tensor_batch['full_output'] = text_actions
        batch.non_tensor_batch['is_action_valid'] = [info['is_action_valid'] for info in infos]
        batch.non_tensor_batch['rewards'] = rewards
        batch.non_tensor_batch['active_masks'] = ~dones

        total_batch_list[i].append(batch_list[i])

        # 6. 更新状态
        episode_rewards += rewards * active_masks
        episode_lengths[active_masks] += 1
        is_done = is_done | dones
        obs = next_obs

        if is_done.all():
            break

    return total_batch_list, episode_rewards, episode_lengths, success, traj_uid
```

### 5.4 RLVMR Reward处理流程

**文件**：`code/agent_system/multi_turn_rollout/rollout_loop.py:497-515`

```python
def multi_turn_loop(self, gen_batch, actor_rollout_wg, envs, is_train=True):
    # 1. 执行rollout
    total_batch_list, episode_rewards, episode_lengths, success, traj_uid = \
        self.vanilla_multi_turn_loop(gen_batch, actor_rollout_wg, envs)

    # 2. 如果启用RLVMR，计算step rewards
    if self.config.algorithm.rlvmr.enable:
        total_batch_list = core_rlvmr.process_trajectory_rlvmr_rewards(
            trajectory_list=total_batch_list,
            config=self.config,
            episode_rewards=episode_rewards
        )

    # 3. 整理为DataProto
    gen_batch_output = self.gather_rollout_data(
        total_batch_list, episode_rewards, episode_lengths, success, traj_uid
    )

    # 4. 添加RLVMR元信息
    if self.config.algorithm.rlvmr.enable:
        gen_batch_output.meta_info["rlvmr_step_advantage_w"] = (
            self.config.algorithm.rlvmr.step_advantage_w
        )
        gen_batch_output.meta_info["rlvmr_mode"] = self.config.algorithm.rlvmr.mode

    return gen_batch_output
```

---

## 6. Reward计算机制

### 6.1 Episode Reward Manager

**文件**：`code/agent_system/reward_manager/episode.py:51-109`

```python
class EpisodeRewardManager:
    def __call__(self, data: DataProto):
        reward_tensor = torch.zeros_like(data.batch['responses'])

        for i in range(len(data)):
            # 获取episode奖励
            episode_rewards = data_item.non_tensor_batch['episode_rewards']
            episode_lengths = data_item.non_tensor_batch['episode_lengths']

            # 是否归一化
            if self.normalize_by_length:
                score = episode_rewards / episode_lengths
            else:
                score = episode_rewards

            # 只在最后一个token位置赋值
            reward_tensor[i, valid_response_length - 1] = torch.tensor(score)

        return reward_tensor
```

**关键点**：
- Episode reward只在response的**最后一个token**位置非零
- 其他位置都是0（sparse reward）
- 可选择是否按长度归一化

### 6.2 Invalid Action Penalty

**文件**：`code/verl/trainer/ppo/ray_trainer.py:141-165`

```python
def apply_invalid_action_penalty(data: DataProto, invalid_action_penalty_coef):
    reward_tensor = data.batch['token_level_scores']

    for i in range(len(data)):
        action_valids = data_item.non_tensor_batch['is_action_valid']
        action_invalids = 1 - action_valids

        # 在最后一个token位置减去惩罚
        reward_tensor[i, valid_response_length - 1] -= (
            invalid_action_penalty_coef * action_invalids
        )

        # 如果有step_rewards也要减
        if 'step_rewards' in data.batch:
            step_rewards[i] -= invalid_action_penalty_coef * action_invalids

    return data, metrics
```

**配置依据**：`code/examples/rlvmr_trainer/run_alfworld.sh:50-51`
```bash
actor_rollout_ref.actor.use_invalid_action_penalty=True \
actor_rollout_ref.actor.invalid_action_penalty_coef=0.1
```

### 6.3 完整Token-level Rewards构成

对于RLVMR，每个token位置的reward为：
```
token_level_rewards[i, t] =
    episode_reward (仅在t=最后一个token时非零)
    - invalid_action_penalty (仅在t=最后一个token时，且action无效)
    - kl_penalty[t] (如果启用use_kl_in_reward)
```

**代码依据**：
- Episode reward: `code/agent_system/reward_manager/episode.py:98`
- Invalid penalty: `code/verl/trainer/ppo/ray_trainer.py:158`
- KL penalty: `code/verl/trainer/ppo/ray_trainer.py:183`

---

## 7. 关键配置参数

### 7.1 RLVMR专属配置

**文件**：`code/verl/trainer/config/ppo_trainer.yaml:188-194`

```yaml
algorithm:
  adv_estimator: rlvmr  # 使用RLVMR优势估计器
  rlvmr:
    enable: False
    planning_reward: 1.0       # Planning标签奖励
    exploration_reward: 1.0    # Exploration标签奖励
    reflection_reward: 1.0     # Reflection标签奖励
    step_advantage_w: 0.5      # Step优势权重
    mode: "mean_std_norm"      # 归一化模式："mean_norm" 或 "mean_std_norm"
```

**实际训练配置**：`code/examples/rlvmr_trainer/run_alfworld.sh:15-18`
```bash
algorithm.adv_estimator=rlvmr \
algorithm.rlvmr.enable=True \
algorithm.use_kl_in_reward=False \
algorithm.rlvmr.step_advantage_w=1.0
```

### 7.2 数据配置

**文件**：`code/examples/rlvmr_trainer/run_alfworld.sh:5-12, 19-27`

```bash
# 数据集大小
train_data_size=16
val_data_size=128
group_size=8

# 数据路径和处理
data.train_files=$HOME/data/verl-agent/text/train.parquet
data.val_files=$HOME/data/verl-agent/text/test.parquet
data.train_batch_size=$train_data_size
data.val_batch_size=$val_data_size
data.max_prompt_length=6000
data.max_response_length=1024
data.filter_overlong_prompts=True
data.truncation='error'
data.return_raw_chat=True  # 保留原始chat格式
```

### 7.3 环境配置

**文件**：`code/examples/rlvmr_trainer/run_alfworld.sh:52-57`

```bash
env.env_name=alfworld/AlfredTWEnv
env.seed=0
env.max_steps=30              # 最大步数
env.rollout.n=$group_size     # 环境分组数（GRPO/GiGPO/RLVMR都需要>1）
env.alfworld.generalization_level=0
env.alfworld.meta_think=True  # 启用Meta-cognitive thinking
```

**关键点**：`env.rollout.n`必须>1才能启用分组优势估计

### 7.4 Actor训练配置

**文件**：`code/examples/rlvmr_trainer/run_alfworld.sh:28-51`

```bash
# 模型配置
actor_rollout_ref.model.path=COLD_START_MODEL_PATH  # Cold-start SFT模型
actor_rollout_ref.model.use_remove_padding=True
actor_rollout_ref.model.enable_gradient_checkpointing=True

# 优化器
actor_rollout_ref.actor.optim.lr=1e-6

# PPO超参数
actor_rollout_ref.actor.ppo_mini_batch_size=256
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16
actor_rollout_ref.actor.clip_ratio=0.2

# KL散度loss（用于稳定训练）
actor_rollout_ref.actor.use_kl_loss=True
actor_rollout_ref.actor.kl_loss_coef=0.01
actor_rollout_ref.actor.kl_loss_type=low_var_kl

# Invalid action惩罚
actor_rollout_ref.actor.use_invalid_action_penalty=True
actor_rollout_ref.actor.invalid_action_penalty_coef=0.1
```

### 7.5 Rollout配置

**文件**：`code/examples/rlvmr_trainer/run_alfworld.sh:39-48`

```bash
actor_rollout_ref.rollout.name=vllm  # 使用vLLM推理引擎
actor_rollout_ref.rollout.tensor_model_parallel_size=1
actor_rollout_ref.rollout.gpu_memory_utilization=0.5
actor_rollout_ref.rollout.enable_chunked_prefill=False
actor_rollout_ref.rollout.enforce_eager=False
actor_rollout_ref.rollout.free_cache_engine=False

# 验证时的采样参数
actor_rollout_ref.rollout.val_kwargs.temperature=0.4
actor_rollout_ref.rollout.val_kwargs.do_sample=True
```

### 7.6 训练器配置

**文件**：`code/examples/rlvmr_trainer/run_alfworld.sh:58-68`

```bash
trainer.critic_warmup=0
trainer.logger=['console','wandb']
trainer.project_name='RLVMR'
trainer.experiment_name='rlvmr_qwen2.5-1.5b_L0'
trainer.n_gpus_per_node=8
trainer.nnodes=1
trainer.resume_mode=auto
trainer.save_freq=5
trainer.test_freq=5
trainer.total_epochs=100
trainer.val_before_train=True
```

---

## 8. Prompt模板

### 8.0 Prompt版本总览

RLVMR系统共包含 **12个Prompt模板**，分布在两个文件中：
- `alfworld.py`: 6个模板（用于实际RL训练交互）
- `cold_start.py`: 6个模板（用于数据准备和cold-start训练）

#### 完整Prompt版本对照表

| # | Prompt名称 | 文件位置 | 历史 | 推理标签 | admissible_actions | 使用阶段 | 用途 |
|---|-----------|---------|------|---------|-------------------|---------|------|
| **ALFWorld - 实际RL训练** (`alfworld.py`) |
| 1 | `ALFWORLD_TEMPLATE_NO_HIS` | alfworld.py:2-10 | ❌ 无 | `<think>` | ✅ 有 | RL训练 | 基础版本（首步，带思考） |
| 2 | `ALFWORLD_TEMPLATE` | alfworld.py:12-21 | ✅ 有 | `<think>` | ✅ 有 | RL训练 | 基础版本（后续步，带思考） |
| 3 | `ALFWORLD_TEMPLATE_NO_HIS_NOTHINK` | alfworld.py:23-30 | ❌ 无 | ❌ 无 | ✅ 有 | RL训练 | 基础版本（首步，仅动作） |
| 4 | `ALFWORLD_TEMPLATE_NOTHINK` | alfworld.py:32-40 | ✅ 有 | ❌ 无 | ✅ 有 | RL训练 | 基础版本（后续步，仅动作） |
| 5 | `ALFWORLD_TEMPLATE_NO_HIS_MC` | alfworld.py:85-127 | ❌ 无 | **Meta-Cognitive** | ✅ 有 | **RLVMR训练** | **首步（含4种推理标签）** |
| 6 | `ALFWORLD_TEMPLATE_MC` | alfworld.py:129-174 | ✅ 有 | **Meta-Cognitive** | ✅ 有 | **RLVMR训练** | **后续步（含4种推理标签）** |
| **ALFWorld - Cold-start训练** (`cold_start.py`) |
| 7 | `ALFWORLD_TEMPLATE_NO_HIS_CS` | cold_start.py:2-41 | ❌ 无 | **Meta-Cognitive** | ❌ **无** | **Cold-start SFT** | **首步（强推理训练）** |
| 8 | `ALFWORLD_TEMPLATE_CS` | cold_start.py:43-85 | ✅ 有 | **Meta-Cognitive** | ❌ **无** | **Cold-start SFT** | **后续步（强推理训练）** |
| 9 | `ALFWORLD_TAGGING_TEMPLATE` | cold_start.py:87-123 | N/A | **Meta-Cognitive** | N/A | **数据标注** | **GPT-4o标注推理过程** |
| **SciWorld - Cold-start训练** (`cold_start.py`) |
| 10 | `SCIWORLD_TEMPLATE_NO_HIS_CS` | cold_start.py:126-184 | ❌ 无 | **Meta-Cognitive** | ❌ 无 | Cold-start SFT | SciWorld首步 |
| 11 | `SCIWORLD_TEMPLATE_CS` | cold_start.py:186-247 | ✅ 有 | **Meta-Cognitive** | ❌ 无 | Cold-start SFT | SciWorld后续步 |
| 12 | `SCIWORLD_TAGGING_TEMPLATE` | cold_start.py:249-288 | N/A | **Meta-Cognitive** | N/A | 数据标注 | SciWorld GPT-4o标注 |

#### 命名规则解析

**后缀含义**：
- `_NO_HIS`: 无历史记录（第一步）
- `_NOTHINK`: 无推理标签（只输出action）
- `_MC`: Meta-Cognitive（包含4种meta-cognitive标签）
- `_CS`: Cold Start（用于cold-start训练）
- `_TAGGING`: 用于GPT-4o标注

**推理标签类型**：
1. **`<think>` 标签**（基础版本 #1-4）：
   - 自由格式的step-by-step推理
   - 不区分推理类型

2. **Meta-Cognitive标签**（RLVMR版本 #5-12）：
   - `<planning>`: 规划整体任务
   - `<explore>`: 探索未知信息
   - `<reflection>`: 反思错误原因
   - `<monitor>`: 跟踪当前进度

#### 关键差异对比

##### 1️⃣ **admissible_actions差异**（最重要！）

```python
# Cold-start训练 (#7, #8) - 无动作列表
"Your current observation is: {current_observation}"

# RLVMR训练 (#5, #6) - 有动作列表
"Your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}]."
```

**原因**：见8.3节详细分析

##### 2️⃣ **推理标签差异**

```python
# 基础版本 (#1, #2) - <think>标签
"This reasoning process MUST be enclosed within <think> </think> tags."

# Meta-Cognitive版本 (#5-12) - 4种专用标签
"1. First, reason using ONLY ONE tag pair:
   <planning>...</planning>
   <explore>...</explore>
   <reflection>...</reflection>
   <monitor>...</monitor>"
```

##### 3️⃣ **历史记录差异**

```python
# NO_HIS版本 (#1, #3, #5, #7, #10) - 无历史
"Your current observation is: {current_observation}"

# 带历史版本 (#2, #4, #6, #8, #11) - 有历史
"Prior to this step, you have already taken {step_count} step(s).
Below are the most recent {history_length} observations and actions: {action_history}"
```

#### 实际使用场景映射

**代码位置**：`code/agent_system/environments/env_manager.py:86-94`

```python
if self.meta_think:  # RLVMR训练
    _ALFWORLD_TEMPLATE_NO_HIS = ALFWORLD_TEMPLATE_NO_HIS_MC  # #5
    _ALFWORLD_TEMPLATE = ALFWORLD_TEMPLATE_MC                # #6
elif self.config.env.alfworld.action_only:  # 仅动作模式
    _ALFWORLD_TEMPLATE_NO_HIS = ALFWORLD_TEMPLATE_NO_HIS_NOTHINK  # #3
    _ALFWORLD_TEMPLATE = ALFWORLD_TEMPLATE_NOTHINK                # #4
else:  # 基础训练
    _ALFWORLD_TEMPLATE_NO_HIS = ALFWORLD_TEMPLATE_NO_HIS  # #1
    _ALFWORLD_TEMPLATE = ALFWORLD_TEMPLATE                # #2
```

**训练流程使用**：
```
数据标注阶段：#9 ALFWORLD_TAGGING_TEMPLATE (GPT-4o)
       ↓
Cold-start SFT：#7, #8 (无admissible_actions)
       ↓
RLVMR训练：   #5, #6 (有admissible_actions)
```

#### 三个阶段的Prompt使用详情

##### 🔷 阶段1：数据标注（GPT-4o）

**使用的Prompt**：`#9 ALFWORLD_TAGGING_TEMPLATE`

**代码位置**：`code/scripts/alfworld_prepare.py:101`
```python
prompt = ALFWORLD_TAGGING_TEMPLATE.format(traj=json.dumps(traj["traj"], ensure_ascii=False))
res = llm_json(prompt, MODEL)  # MODEL = "gpt-4o"
```

**输入**：
- 原始专家轨迹（observation-action对）

**输出**：
- 为每个action步添加reasoning（包含4种meta-cognitive标签）
- JSON格式：`[{"reason": "<planning>...</planning>", "action": "go to shelf 1"}, ...]`

**作用**：
- 为300条轨迹生成meta-cognitive推理过程
- 用于cold-start SFT训练

---

##### 🔷 阶段2：Cold-start SFT训练

**使用的Prompt**：
- `#7 ALFWORLD_TEMPLATE_NO_HIS_CS`（第一步，无历史）
- `#8 ALFWORLD_TEMPLATE_CS`（后续步，有历史）

**代码位置**：`code/scripts/alfworld_prepare.py:116-159`
```python
if i == 0:
    prompt = ALFWORLD_TEMPLATE_NO_HIS_CS.format(current_observation=item["obs"])
else:
    prompt = ALFWORLD_TEMPLATE_CS.format(
        task_description=traj["task"],
        action_history=action_history,
        current_observation=item["obs"],
        planning=latest_planning
    )
```

**关键特征**：
- ❌ **无** `admissible_actions` 字段
- ✅ **有** Meta-Cognitive标签要求
- 强制模型从纯观察中推理动作

**训练数据**：
- 300条GPT-4o标注的轨迹
- 转换为step-level的prompt-response对
- 保存为parquet格式

---

##### 🔷 阶段3：RLVMR训练 + 评测

**使用的Prompt**（训练和评测**完全相同**）：
- `#5 ALFWORLD_TEMPLATE_NO_HIS_MC`（第一步，无历史）
- `#6 ALFWORLD_TEMPLATE_MC`（后续步，有历史）

**代码位置**：

1. **环境初始化**（`code/agent_system/environments/env_manager.py:541-546`）：
```python
if config.env.alfworld.meta_think:  # True in run_alfworld.sh:57
    projection_f = partial(alfworld_projection_rlvmr)
else:
    projection_f = partial(alfworld_projection)
envs = AlfWorldEnvironmentManager(_envs, projection_f, config.env.env_name, config)
val_envs = AlfWorldEnvironmentManager(_val_envs, projection_f, config.env.env_name, config)
```

**重要**：`envs`（训练）和`val_envs`（评测）使用**完全相同的config和projection_f**！

2. **Prompt选择**（`code/agent_system/environments/env_manager.py:86-94`）：
```python
if self.meta_think:  # True for RLVMR
    _ALFWORLD_TEMPLATE_NO_HIS = ALFWORLD_TEMPLATE_NO_HIS_MC  # #5
    _ALFWORLD_TEMPLATE = ALFWORLD_TEMPLATE_MC                # #6
elif self.config.env.alfworld.action_only:
    _ALFWORLD_TEMPLATE_NO_HIS = ALFWORLD_TEMPLATE_NO_HIS_NOTHINK  # #3
    _ALFWORLD_TEMPLATE = ALFWORLD_TEMPLATE_NOTHINK                # #4
else:
    _ALFWORLD_TEMPLATE_NO_HIS = ALFWORLD_TEMPLATE_NO_HIS  # #1
    _ALFWORLD_TEMPLATE = ALFWORLD_TEMPLATE                # #2
```

3. **配置**（`code/examples/rlvmr_trainer/run_alfworld.sh:57`）：
```bash
env.alfworld.meta_think=True
```

**关键特征**：
- ✅ **有** `admissible_actions` 字段
- ✅ **有** Meta-Cognitive标签要求
- 降低探索难度，模型从候选动作列表中选择

**训练流程**：
```python
# code/verl/trainer/ppo/ray_trainer.py:610-616
test_output_gen_batch = self.traj_collector.multi_turn_loop(
    gen_batch=test_gen_batch,
    actor_rollout_wg=self.actor_rollout_wg,
    envs=self.val_envs,  # 使用与训练相同的环境和Prompt
    is_train=False,      # 只影响是否使用dynamic sampling，不影响Prompt
)
```

**评测阶段**：
- 使用`val_envs`，与`envs`共享同一个config
- **Prompt完全相同**：`ALFWORLD_TEMPLATE_NO_HIS_MC` + `ALFWORLD_TEMPLATE_MC`
- `is_train=False`只控制是否使用dynamic sampling（DAPO），不改变Prompt

---

#### 三阶段对比总结表

| 阶段 | Prompt编号 | Prompt名称 | admissible_actions | Meta-Cognitive | 执行者 | 数据量 |
|------|----------|-----------|-------------------|---------------|-------|--------|
| **数据标注** | #9 | `ALFWORLD_TAGGING_TEMPLATE` | N/A | ✅ 输出 | GPT-4o | 300轨迹 |
| **Cold-start SFT** | #7, #8 | `ALFWORLD_TEMPLATE_*_CS` | ❌ **无** | ✅ 要求 | 训练模型 | 300轨迹×步数 |
| **RLVMR训练** | #5, #6 | `ALFWORLD_TEMPLATE_*_MC` | ✅ **有** | ✅ 要求 | 训练模型 | 动态交互 |
| **RLVMR评测** | #5, #6 | `ALFWORLD_TEMPLATE_*_MC` | ✅ **有** | ✅ 要求 | 评测模型 | 固定测试集 |

**核心发现**：
1. ✅ **训练和评测使用相同Prompt**（都是 #5, #6）
2. ⚠️ **Cold-start和RLVMR的Prompt不匹配**（见8.3节详细分析）
3. 🎯 **渐进式训练策略**：从困难（无动作列表）到简单（有动作列表）

---

### 8.1 标注Prompt（ALFWORLD_TAGGING_TEMPLATE）

**文件**：`code/agent_system/environments/prompts/cold_start.py:87-123`

```
You are an expert agent operating in the ALFRED Embodied Environment.
I will provide you with a successful trajectory. You need to supplement the reasoning process.

**Reason using ONLY ONE tag pair:**

<planning>
Decompose a complex overall task into clear subgoals, listing each milestone as a separate point.
All points must be listed explicitly: Step 1: xxx; Step 2: xxx; Step 3: xxx
</planning>

<explore>
When searching for unknown objects, use current observations to think outside the box and
generate as many possible hypotheses, locations, items, or actions as possible.
</explore>

<reflection>
Reflect on task progress and reasons for failures when the task has stalled, incorrect
actions taken, or current situation misjudged.
</reflection>

<monitor>
Track current progress and history. Recall the current subgoal based on the plan,
then consider the next action required.
</monitor>

Output JSON format with same length as trajectory:
[{"reason": "<tag>...</tag>", "action": "go to shelf 1"}]

trajectory: {traj}
```

### 8.2 训练Prompt（Cold-start阶段）

#### 第一步（无历史）
**文件**：`code/agent_system/environments/prompts/cold_start.py:2-41`

```
You are an expert agent operating in the ALFRED Embodied Environment.
Your current observation is: {current_observation}

Now it's your turn to take an action, following these steps:

1. First, reason using ONLY ONE tag pair:
   <planning>...</planning>
   <explore>...</explore>
   <reflection>...</reflection>
   <monitor>...</monitor>

2. After your reasoning, select and present an action within <action></action> tags.
```

#### 后续步骤（有历史）
**文件**：`code/agent_system/environments/prompts/cold_start.py:43-85`

```
You are an expert agent operating in the ALFRED Embodied Environment.
Your task is to: {task_description}

Prior to this step, you have already taken {step_count} step(s).
Below are the most recent {history_length} observations and actions: {action_history}

You are now at step {current_step} and your current observation is: {current_observation}

Your previous overall plan is: {planning}.

Now it's your turn to take an action, following these steps:

1. First, reason using ONLY ONE tag pair:
   <planning>...</planning>
   <explore>...</explore>
   <reflection>...</reflection>
   <monitor>...</monitor>

2. After your reasoning, select and present an action within <action></action> tags.
```

### 8.3 实际RLVMR训练Prompt（与Cold-start的关键差异）

#### ⚠️ Prompt不匹配问题

**重要发现**：Cold-start SFT训练和实际RLVMR训练使用的Prompt**存在显著差异**！

#### Cold-start SFT阶段
**文件**：`code/agent_system/environments/prompts/cold_start.py:2-41`

**Prompt内容**：
```
You are an expert agent operating in the ALFRED Embodied Environment.
Your current observation is: {current_observation}

Now it's your turn to take an action, following these steps:
...
```

✅ **没有** `admissible_actions` 字段
- 模型必须从纯观察中推理出合理的动作
- 训练难度高，强迫模型学习强推理能力

#### RLVMR训练阶段
**文件**：`code/agent_system/environments/prompts/alfworld.py:85-127`

**配置**：`code/examples/rlvmr_trainer/run_alfworld.sh:57`
```bash
env.alfworld.meta_think=True
```

**Prompt选择逻辑**：`code/agent_system/environments/env_manager.py:86-94`
```python
if self.meta_think:  # True for RLVMR training
    _ALFWORLD_TEMPLATE_NO_HIS = ALFWORLD_TEMPLATE_NO_HIS_MC
```

**实际Prompt内容**：
```
You are an expert agent operating in the ALFRED Embodied Environment.
Your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action, following these steps:
...
```

❌ **包含** `admissible_actions` 字段
- 环境提供当前可执行的动作列表
- 降低探索难度，限制动作空间

#### 差异对比表

| 阶段 | Prompt模板 | admissible_actions | 动作空间 | 难度 |
|------|-----------|-------------------|---------|------|
| Cold-start SFT | `ALFWORLD_TEMPLATE_NO_HIS_CS` | ❌ 无 | 开放式文本 | 困难 |
| RLVMR训练 | `ALFWORLD_TEMPLATE_NO_HIS_MC` | ✅ 有 | 受限列表 | 简单 |

**示例对比**：

```python
# Cold-start阶段观察
"You are in the middle of a room. Looking quickly around you, you see a bed 1, a desk 1..."

# RLVMR训练阶段观察
"You are in the middle of a room. Looking quickly around you, you see a bed 1, a desk 1...
Your admissible actions of the current situation are: [
 'go to bed 1',
 'go to desk 1',
 'go to drawer 1',
 ...
]."
```

#### 设计意图分析

**推测的设计理由**：

1. **渐进式训练策略**
   ```
   Cold-start (困难) → 删除admissible_actions → 强制学习强推理能力
                  ↓
   RLVMR训练 (简单) → 添加admissible_actions → 专注meta-cognitive质量
   ```

2. **避免RL探索困难**
   - 无 `admissible_actions`：动作空间=任意文本（无限大）
   - 有 `admissible_actions`：动作空间=10-20个候选动作（可控）
   - RL训练更容易收敛

3. **对齐环境交互格式**
   - 原始专家轨迹包含 `AVAILABLE ACTIONS`
   - ALFWorld环境实际返回 `admissible_commands`
   - RL训练时的prompt与环境格式一致

#### 潜在影响

**优点** ✅：
- Cold-start阶段训练出强大的推理能力
- RL阶段降低探索难度，专注meta-cognitive reasoning质量
- 与实际环境交互格式保持一致

**风险** ⚠️：
- **Distribution Shift**：训练和微调时的输入分布不同
- **能力退化**：模型可能过度依赖 `admissible_actions`，丧失开放式推理能力
- **泛化性能**：在没有动作列表的新环境中可能表现下降

#### 代码依据

**Cold-start数据准备**：
```python
# code/scripts/alfworld_prepare.py:20-37
obs = conversations[i]['value'].split("AVAILABLE ACTIONS:")[0]  # 删除动作列表
```

**RLVMR训练时Prompt构建**：
```python
# code/agent_system/environments/env_manager.py:98-104
reformatted_admissible_actions = "\n ".join(f"'{s}'" for s in admissible_actions[i])
obs = _ALFWORLD_TEMPLATE_NO_HIS.format(
    current_observation=text_obs[i],
    admissible_actions=reformatted_admissible_actions  # 添加动作列表
)
```

---

## 9. 代码架构总结

### 9.1 核心模块

```
code/
├── rlvmr/                          # RLVMR核心算法
│   ├── core_rlvmr.py              # 优势函数计算、Step reward处理
│   └── __init__.py
├── verl/
│   └── trainer/
│       ├── main_ppo.py            # 训练主入口
│       └── ppo/
│           ├── ray_trainer.py     # PPO训练器（包含compute_advantage）
│           └── core_algos.py      # 基础RL算法（GAE, GRPO等）
├── agent_system/
│   ├── multi_turn_rollout/
│   │   └── rollout_loop.py        # 多轮交互rollout逻辑
│   ├── reward_manager/
│   │   └── episode.py             # Episode奖励管理
│   └── environments/
│       ├── env_manager.py         # 环境管理器
│       └── prompts/
│           ├── alfworld.py        # ALFWorld推理Prompt
│           └── cold_start.py      # Cold-start标注Prompt
├── scripts/
│   ├── alfworld_prepare.py        # 数据标注脚本
│   └── sciworld_prepare.py
└── examples/
    ├── data_preprocess/
    │   └── cold_start_data.py     # 数据格式转换
    ├── rlvmr_trainer/
    │   ├── run_alfworld.sh        # RLVMR训练脚本
    │   └── run_sciworld.sh
    └── sft/
        └── cold_start/            # Cold-start SFT训练
```

### 9.2 关键函数调用链

#### 训练主流程
```
main_ppo.py:main()
  └─ run_ppo()
      └─ RayPPOTrainer.fit()
          ├─ traj_collector.multi_turn_loop()  # Rollout
          │   ├─ vanilla_multi_turn_loop()
          │   │   ├─ preprocess_batch()
          │   │   ├─ actor.generate()
          │   │   └─ env.step()
          │   ├─ core_rlvmr.process_trajectory_rlvmr_rewards()  # 计算RLVMR rewards
          │   └─ gather_rollout_data()
          ├─ reward_fn(data)  # Episode reward
          ├─ compute_advantage(data)  # RLVMR advantage
          │   └─ core_rlvmr.compute_rlvmr_outcome_advantage()
          │       ├─ episode_norm_reward()
          │       └─ step_norm_reward_by_tag()
          └─ actor.update_policy() & critic.fit()
```

#### 数据处理流程
```
alfworld_prepare.py
  ├─ load_from_disk("data/alfworld_expert_traj")
  ├─ llm_json(ALFWORLD_TAGGING_TEMPLATE)  # GPT-4o标注
  ├─ merge(res, traj)                     # 验证并合并
  └─ json.dump("data/alfworld_cold-start.json")

cold_start_data.py
  ├─ json.load("data/alfworld_cold-start.json")
  ├─ Dataset.from_list(train_dataset)
  └─ to_parquet("train.parquet")
```

---

## 10. 与其他算法对比

### 10.1 优势估计方法对比

| 算法 | Episode Reward | Step Reward | 分组归一化 | 代码文件 |
|------|---------------|-------------|-----------|---------|
| **GAE** | ✓ (value baseline) | ✗ | ✗ | `verl/trainer/ppo/core_algos.py` |
| **GRPO** | ✓ (按uid分组) | ✗ | ✓ (按uid) | `verl/trainer/ppo/core_algos.py` |
| **GiGPO** | ✓ (按uid分组) | ✓ (基于anchor obs) | ✓ (episode+step双重) | `gigpo/core_gigpo.py` |
| **RLVMR** | ✓ (按uid分组) | ✓ (基于meta-cognitive tags) | ✓ (按uid+tag) | `rlvmr/core_rlvmr.py` |

**代码依据**：`code/verl/trainer/ppo/ray_trainer.py:204-283`

### 10.2 RLVMR vs GiGPO

**共同点**：
- 都使用双重优势函数（Episode + Step）
- 都需要`env.rollout.n > 1`进行分组
- 都进行分组归一化

**关键区别**：

| 维度 | GiGPO | RLVMR |
|------|-------|-------|
| **Step信号来源** | 环境observation变化（anchor obs） | Meta-cognitive语言标签 |
| **Step reward计算** | 比较anchor observation | 解析`<planning>`, `<explore>`, `<reflection>` |
| **分组方式** | (uid, anchor_obs) | (uid, tag_type) |
| **适用场景** | 环境observation显式变化 | 语言推理过程显式化 |
| **数据需求** | 需要环境提供anchor obs | 需要LLM生成meta-cognitive标签 |

**代码依据**：
- GiGPO: `code/gigpo/core_gigpo.py`
- RLVMR: `code/rlvmr/core_rlvmr.py`

---

## 11. 常见问题

### Q1: 为什么需要Cold-start SFT？
**A**:
- RLVMR要求Agent输出meta-cognitive标签（`<planning>`, `<explore>`等）
- 预训练模型不会自然生成这些标签
- Cold-start SFT让模型学会这种格式化输出
- **代码依据**：`code/examples/rlvmr_trainer/run_alfworld.sh:28` - 使用cold-start模型作为初始化

### Q2: step_advantage_w如何选择？
**A**:
- 默认值：`0.5`（配置文件）或`1.0`（run_alfworld.sh）
- 较大值：强调meta-cognitive过程，鼓励规划和反思
- 较小值：更关注最终结果
- **代码依据**：
  - 默认: `code/verl/trainer/config/ppo_trainer.yaml:193`
  - 实际: `code/examples/rlvmr_trainer/run_alfworld.sh:18`

### Q3: 如何验证RLVMR是否正常工作？
**A**: 检查以下指标
1. `rlvmr_tag_type`分布：应该有多种tag类型
2. `rlvmr_step_reward`非零率：成功轨迹中应有非零step reward
3. `episode_advantages` vs `step_advantages`：两者应该都有合理的值
4. **代码依据**：`code/rlvmr/core_rlvmr.py:37-40` - 返回detailed advantages

### Q4: mode参数的影响？
**A**:
- `"mean_norm"`: 只减均值，保留标准差信息（`remove_std=True`）
- `"mean_std_norm"`: 减均值并除以标准差，标准化（`remove_std=False`）
- 推荐使用`"mean_std_norm"`以获得更稳定的梯度
- **代码依据**：`code/rlvmr/core_rlvmr.py:14-23`

---

## 12. 参考文献

本文档基于以下代码文件分析：

### 核心算法
- `code/rlvmr/core_rlvmr.py` - RLVMR算法实现
- `code/verl/trainer/ppo/ray_trainer.py` - PPO训练器
- `code/verl/trainer/ppo/core_algos.py` - 基础RL算法

### 数据处理
- `code/scripts/alfworld_prepare.py` - 数据标注
- `code/examples/data_preprocess/cold_start_data.py` - 格式转换
- `code/agent_system/environments/prompts/cold_start.py` - Prompt模板

### 训练流程
- `code/verl/trainer/main_ppo.py` - 训练入口
- `code/agent_system/multi_turn_rollout/rollout_loop.py` - Rollout逻辑
- `code/agent_system/reward_manager/episode.py` - Reward计算

### 配置文件
- `code/verl/trainer/config/ppo_trainer.yaml` - 默认配置
- `code/examples/rlvmr_trainer/run_alfworld.sh` - 训练脚本

---

**文档版本**: v1.0
**最后更新**: 基于代码库最新版本
**所有结论均有明确代码依据，格式为 `文件路径:行号`**
