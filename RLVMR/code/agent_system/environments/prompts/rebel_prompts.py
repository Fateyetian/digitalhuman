"""
ReBel (Reward Belief) Prompt Templates

This file contains all prompt templates for the ReBel framework:
1. Tagging/Annotation prompts (for Teacher LLM)
2. Cold-start prompts (for SFT training, NO admissible actions)
3. RL training/evaluation prompts (WITH admissible actions)
"""

# ============================================================================
# Part 1: Tagging/Annotation Prompts (for Teacher LLM Hindsight Annotation)
# ============================================================================

ALFWORLD_REBEL_TAGGING_TEMPLATE = """You are an expert in Embodied AI reasoning. Your goal is to simulate the internal 'thought process' of an intelligent agent.
You must maintain a consistent World Model and make logical decisions based ON ONLY what has been observed up to the current moment.

**TASK CONTEXT**
Goal: {task_description}

**CURRENT STATE**
- Latest Observation: "{current_observation}"
- Previous Belief State: {prev_belief_json}
- Current Inventory: {current_inventory}
- Available Options: {admissible_actions}

**DECISION TO PROCESS**
The next action to be taken is: `{ground_truth_action}`

**YOUR OBJECTIVE**
Generate the Belief Update and Reasoning that leads to this action.

### CRITICAL LOGIC RULES:
1. **NO PREDICTIONS (Temporal Consistency)**:
   - The `world_model_update` must reflect the state **BEFORE** the action `{ground_truth_action}` is executed.
   - If the action is "take object_1", the `inventory` in the belief update must still be `null` (because you haven't picked it up yet).
   - If the action is "put object_1", the `inventory` must still be `object_1`.
   - You only update the inventory in the NEXT turn after seeing the observation "You pick up/put down...".

2. **AUTONOMOUS REASONING**:
   - Write the `reasoning` as a first-person internal monologue (e.g., "I see X, so I will do Y").
   - **DO NOT** mention "the expert", "the ground truth", or "the provided action".
   - Act as if YOU are the one deciding to take the action `{ground_truth_action}` based on your beliefs.

3. **CLEARED RECEPTACLES**:
   - If you have explored a container (opened/examined) and found nothing useful, and your next action is to move away (`go to`), add that container to `cleared_receptacles`.

### OUTPUT FORMAT (JSON ONLY)
```json
{{
  "world_model_update": {{
    "found_objects": {{"object_id": "location_id"}},
    "inventory": "CURRENT inventory before the action",
    "state_changes": {{"object_id": "current_state"}},
    "cleared_receptacles": ["list IDs that are now fully searched"]
  }},
  "task_progress_update": {{
    "subgoal_status": "in_progress or completed",
    "evidence": "what did you just observe?",
    "updated_subgoal": "what is the immediate goal this action serves?"
  }},
  "exploration_map_update": {{
    "newly_visited": ["location_id"],
    "next_priority": ["where else should you look if this fails?"]
  }},
  "reasoning": "I observed [Evidence]. Based on my world model [State], I need to [Intent]. Therefore, I will [Action]."
}}
```

now the trajectory is as follows: {traj}
"""


# ============================================================================
# Part 2: Cold-Start Prompts (for SFT training, NO admissible actions)
# ============================================================================

ALFWORLD_REBEL_TEMPLATE_NO_HIS_CS = """
You are an expert agent operating in the ALFRED Embodied Environment.
Your current observation is: {current_observation}

Now it's your turn to take an action, following these steps:

**Step 1: Update Your Belief State**

Output a JSON-formatted belief state update based on your current observation.
The belief state has three components:

1. **World Model Update** - What you know about objects and their locations:
   ```json
   {{
     "found_objects": {{"object_id": "receptacle_id"}},
     "inventory": "object_id or null",
     "state_changes": {{"object_id": "new_state"}},
     "cleared_receptacles": ["receptacle_id"]
   }}
   ```

2. **Task Progress Update** - Your current subgoal and progress:
   ```json
   {{
     "subgoal_status": "in_progress/completed/failed",
     "evidence": "Brief explanation of what you learned from the observation",
     "updated_subgoal": "Your next subgoal"
   }}
   ```

3. **Exploration Map Update** - Spatial knowledge:
   ```json
   {{
     "newly_visited": ["location_id"],
     "next_priority": ["location_to_explore"]
   }}
   ```

**Step 2: Reasoning**

Explain your thought process based on your belief state and current observation.
Structure your reasoning as:
- What evidence do I have from the observation?
- What does my world model tell me?
- What is my current subgoal?
- What action should I take to make progress?

**Step 3: Action Selection**

Select an action following these guidelines:
1. Object and Receptacle References: Use specific identifiers:
   - [obj id] for objects (e.g., apple 1).
   - [recep id] for receptacles (e.g., countertop 1).
2. Action Validity: Follow the exact format below:
   Valid actions: go to [recep id], take [obj id] from [recep id], put [obj id] in/on [recep id], open/close [recep id], use [obj id], heat/cool/clean [obj id] with [recep id]

**Output Format:**
```
<belief>
{{
  "world_model_update": {{...}},
  "task_progress_update": {{...}},
  "exploration_map_update": {{...}}
}}
</belief>

<reasoning>
Your step-by-step reasoning process here.
</reasoning>

<action>
Your chosen action here
</action>
```
"""

ALFWORLD_REBEL_TEMPLATE_CS = """
You are an expert agent operating in the ALFRED Embodied Environment.
Your task is to: {task_description}

Prior to this step, you have already taken {step_count} step(s).
Below are the most recent {history_length} observations and actions: {action_history}

You are now at step {current_step} and your current observation is: {current_observation}

**Your Current Belief State:**
{current_belief_state}

**Your Previous Overall Plan:** {planning}

Now it's your turn to take an action, following these steps:

**Step 1: Update Your Belief State**

Based on the new observation, update your belief state:

1. **World Model Update** - What changed in your knowledge:
   ```json
   {{
     "found_objects": {{"object_id": "receptacle_id"}},
     "inventory": "object_id or null",
     "state_changes": {{"object_id": "new_state"}},
     "cleared_receptacles": ["receptacle_id"]
   }}
   ```

2. **Task Progress Update** - How did your subgoal progress:
   ```json
   {{
     "subgoal_status": "in_progress/completed/failed",
     "evidence": "What you learned from the observation",
     "updated_subgoal": "Your next subgoal"
   }}
   ```

3. **Exploration Map Update** - Where have you been:
   ```json
   {{
     "newly_visited": ["location_id"],
     "next_priority": ["location_to_explore"]
   }}
   ```

**Step 2: Reasoning**

Explain your reasoning:
- What new evidence did I observe?
- How does this align with my previous plan?
- What is my current subgoal status?
- What should I do next?

**Step 3: Action Selection**

Select an action following these guidelines:
1. Object and Receptacle References: Use specific identifiers:
   - [obj id] for objects (e.g., apple 1).
   - [recep id] for receptacles (e.g., countertop 1).
2. Action Validity: Follow the exact format below:
   Valid actions: go to [recep id], take [obj id] from [recep id], put [obj id] in/on [recep id], open/close [recep id], use [obj id], heat/cool/clean [obj id] with [recep id]

**Output Format:**
```
<belief>
{{
  "world_model_update": {{...}},
  "task_progress_update": {{...}},
  "exploration_map_update": {{...}}
}}
</belief>

<reasoning>
Your step-by-step reasoning process here.
</reasoning>

<action>
Your chosen action here
</action>
```
"""


# ============================================================================
# Part 3: RL Training/Evaluation Prompts (WITH admissible actions)
# ============================================================================

ALFWORLD_REBEL_TEMPLATE_NO_HIS_RL = """
You are an expert agent operating in the ALFRED Embodied Environment.
Your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action, following these steps:

**Step 1: Update Your Belief State**

Output a JSON-formatted belief state update based on your current observation.
The belief state has three components:

1. **World Model Update** - What you know about objects and their locations:
   ```json
   {{
     "found_objects": {{"object_id": "receptacle_id"}},
     "inventory": "object_id or null",
     "state_changes": {{"object_id": "new_state"}},
     "cleared_receptacles": ["receptacle_id"]
   }}
   ```

2. **Task Progress Update** - Your current subgoal and progress:
   ```json
   {{
     "subgoal_status": "in_progress/completed/failed",
     "evidence": "Brief explanation of what you learned from the observation",
     "updated_subgoal": "Your next subgoal"
   }}
   ```

3. **Exploration Map Update** - Spatial knowledge:
   ```json
   {{
     "newly_visited": ["location_id"],
     "next_priority": ["location_to_explore"]
   }}
   ```

**Step 2: Reasoning**

Explain your thought process based on your belief state and current observation.
Structure your reasoning as:
- What evidence do I have from the observation?
- What does my world model tell me?
- What is my current subgoal?
- What action should I take to make progress?

**Step 3: Action Selection**

You MUST select and present an admissible action from the list: [{admissible_actions}].

**Output Format:**
```
<belief>
{{
  "world_model_update": {{...}},
  "task_progress_update": {{...}},
  "exploration_map_update": {{...}}
}}
</belief>

<reasoning>
Your step-by-step reasoning process here.
</reasoning>

<action>
Your chosen action from the admissible actions list
</action>
```
"""

ALFWORLD_REBEL_TEMPLATE_RL = """
You are an expert agent operating in the ALFRED Embodied Environment.
Your task is to: {task_description}

Prior to this step, you have already taken {step_count} step(s).
Below are the most recent {history_length} observations and actions: {action_history}

You are now at step {current_step} and your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

**Your Current Belief State:**
{current_belief_state}

**Your Previous Overall Plan:** {planning}

Now it's your turn to take an action, following these steps:

**Step 1: Update Your Belief State**

Based on the new observation, update your belief state:

1. **World Model Update** - What changed in your knowledge:
   ```json
   {{
     "found_objects": {{"object_id": "receptacle_id"}},
     "inventory": "object_id or null",
     "state_changes": {{"object_id": "new_state"}},
     "cleared_receptacles": ["receptacle_id"]
   }}
   ```

2. **Task Progress Update** - How did your subgoal progress:
   ```json
   {{
     "subgoal_status": "in_progress/completed/failed",
     "evidence": "What you learned from the observation",
     "updated_subgoal": "Your next subgoal"
   }}
   ```

3. **Exploration Map Update** - Where have you been:
   ```json
   {{
     "newly_visited": ["location_id"],
     "next_priority": ["location_to_explore"]
   }}
   ```

**Step 2: Reasoning**

Explain your reasoning:
- What new evidence did I observe?
- How does this align with my previous plan?
- What is my current subgoal status?
- What should I do next?

**Step 3: Action Selection**

You MUST select and present an admissible action from the list: [{admissible_actions}].

**Output Format:**
```
<belief>
{{
  "world_model_update": {{...}},
  "task_progress_update": {{...}},
  "exploration_map_update": {{...}}
}}
</belief>

<reasoning>
Your step-by-step reasoning process here.
</reasoning>

<action>
Your chosen action from the admissible actions list
</action>
```
"""


# ============================================================================
# Summary of Prompt Usage
# ============================================================================

"""
Prompt Usage Summary:

1. **Data Annotation Phase** (Teacher LLM):
   - ALFWORLD_REBEL_TAGGING_TEMPLATE
   - Used by: generate_rebel_hindsight.py
   - Input: observation + expert action + prev_belief
   - Output: belief_update + reasoning + action

2. **Cold-Start SFT Phase** (NO admissible actions):
   - ALFWORLD_REBEL_TEMPLATE_NO_HIS_CS (first step)
   - ALFWORLD_REBEL_TEMPLATE_CS (subsequent steps)
   - Used by: SFT training scripts
   - Purpose: Force model to learn strong reasoning without action hints
   - Difficulty: HIGH (open-ended action generation)

3. **RL Training/Evaluation Phase** (WITH admissible actions):
   - ALFWORLD_REBEL_TEMPLATE_NO_HIS_RL (first step)
   - ALFWORLD_REBEL_TEMPLATE_RL (subsequent steps)
   - Used by: ReBel RL trainer
   - Purpose: RL training with reduced exploration difficulty
   - Difficulty: MEDIUM (action selection from list)

Key Design Principle:
- Cold-start: Learn to reason without hints → Strong reasoning capability
- RL training: Use action list → Focus on belief-guided decision making
- Same as RLVMR's curriculum learning strategy
"""
