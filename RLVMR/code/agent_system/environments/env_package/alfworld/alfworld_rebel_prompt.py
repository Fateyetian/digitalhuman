"""
ReBel (Reward Belief) Framework - Prompt Templates for ALFWorld

This module defines the prompt templates that instruct the model to maintain
explicit belief states during task execution.
"""

# Template with full history (with belief state)
ALFWORLD_TEMPLATE_REBEL = """You are an expert embodied agent operating in the ALFRED environment. Your goal is to efficiently complete the following task:
{task_description}

You are currently at step {current_step} (Total steps taken: {step_count}).
In this turn, you must perform a two-stage reasoning process: "Belief Update" followed by "Action Decision".

═══════════════════════════════════════════════════════════════════════════════
PART 1: CURRENT CONTEXT
═══════════════════════════════════════════════════════════════════════════════

【Current Observation】
{observation}

【Current Belief State】
1. World Model (Knowledge of objects & locations):
{world_state}

2. Task Progress (Goals & Subgoals):
{task_state}

3. Exploration Map (Visited & Unvisited areas):
{explore_map_state}

【Admissible Actions】
{admissible_actions}

【Recent History】
{history}

═══════════════════════════════════════════════════════════════════════════════
PART 2: INSTRUCTIONS
═══════════════════════════════════════════════════════════════════════════════

Step 1: Incremental Belief Update
Based on the Current Observation, update your internal belief state. Do not repeat old information; only output changes or confirmations based on visual evidence.
- World Model: Update found objects, state changes (heated/cleaned), and CRUCIALLY, list "cleared_receptacles" (containers you checked this turn that were empty or didn't contain the target).
- Task Progress: Verify if the current subgoal is complete. You MUST provide "evidence" from the observation (e.g., "I see the microwave is open").
- Exploration Map: Mark newly visited areas and propose the "next_priority" for exploration.

Step 2: Action Selection
Choose the optimal next action from the Admissible Actions list.
- The action must logically follow from your updated belief and current subgoal.
- Format: Strictly use valid API syntax (e.g., "OpenObject|microwave|1", "PickupObject|apple|1").

═══════════════════════════════════════════════════════════════════════════════
PART 3: OUTPUT FORMAT
═══════════════════════════════════════════════════════════════════════════════

You must strictly follow this format. Output the JSON block for belief updates, followed by your reasoning, and finally the action tag.

<belief>
{{
  "world_model_update": {{
    "found_objects": {{"[obj_id]": "[recep_id]"}},
    "state_changes": {{"[obj_id]": "[new_state]"}},
    "cleared_receptacles": ["(List receptacles confirmed empty/irrelevant this turn)"]
  }},
  "task_progress_update": {{
    "subgoal_status": "(e.g., 'completed' or 'in_progress')",
    "evidence": "(Cite specific visual elements from current observation supporting this status)",
    "updated_subgoal": "(Next subgoal if current is completed, else null)"
  }},
  "exploration_map_update": {{
    "newly_visited": ["(List receptacle/location IDs)"],
    "next_priority": ["(List IDs of where to look next based on logic)"]
  }}
}}
</belief>

<reasoning>
(Briefly explain: Based on the evidence that [X], I have updated my belief. To achieve [Subgoal], I need to [Strategy], so I will take action...)
</reasoning>

<action>
[Insert Exact Action String Here]
</action>

**CRITICAL RULES**:
1. Update ALL three belief components in EVERY response
2. Be accurate - your beliefs will be checked against the actual environment
3. Only report objects you've actually observed this turn
4. Always provide evidence for your task progress updates
5. Choose actions ONLY from the Admissible Actions list
6. Do NOT include Chinese characters in your response
7. Use proper JSON formatting in the belief block

Your response:"""


# Template without history (first step with belief state)
ALFWORLD_TEMPLATE_NO_HIS_REBEL = """You are an expert embodied agent operating in the ALFRED environment. Your goal is to efficiently complete the following task:
{task_description}

You are at the initial step (step 0).
In this turn, you must perform THREE stages: "Task Planning" → "Belief Construction" → "Action Decision".

═══════════════════════════════════════════════════════════════════════════════
PART 1: CURRENT CONTEXT
═══════════════════════════════════════════════════════════════════════════════

【Current Observation】
{observation}

【Admissible Actions】
{admissible_actions}

═══════════════════════════════════════════════════════════════════════════════
PART 2: INSTRUCTIONS
═══════════════════════════════════════════════════════════════════════════════

Step 1: Task Planning (CRITICAL for first step!)
Analyze the task and create a step-by-step plan:
- Break down the main goal into subgoals
- Identify required objects and their likely locations
- Plan the sequence of actions needed

Step 2: Initial Belief Construction
Based on the Current Observation, construct your initial belief state:
- World Model: List all observed objects, their locations, and states
- Task Progress: Set the plan and first subgoal
- Exploration Map: Mark visible areas and identify where to explore next

Step 3: Action Selection
Choose the optimal first action from the Admissible Actions list.

═══════════════════════════════════════════════════════════════════════════════
PART 3: OUTPUT FORMAT
═══════════════════════════════════════════════════════════════════════════════

You must strictly follow this format:

<belief>
{{
  "world_model_update": {{
    "found_objects": {{"[obj_id]": "[recep_id]"}},
    "state_changes": {{"[obj_id]": "[new_state]"}},
    "inventory": [],
    "cleared_receptacles": []
  }},
  "task_progress_update": {{
    "main_goal": "(The overall task goal)",
    "plan": [
      "(Step 1: e.g., Find the apple)",
      "(Step 2: e.g., Pick up the apple)",
      "(Step 3: e.g., Go to the microwave)",
      "(Step 4: e.g., Heat the apple)",
      "(Step 5: e.g., Put the apple on the counter)"
    ],
    "current_subgoal": "(First subgoal from the plan)",
    "subgoal_status": "in_progress",
    "evidence": "(What you observe that relates to the goal)"
  }},
  "exploration_map_update": {{
    "newly_visited": ["(List visible receptacles/locations)"],
    "next_priority": ["(Where to look based on the plan)"]
  }}
}}
</belief>

<reasoning>
(Explain: My task is [main_goal]. My plan is [plan summary]. First I need to [current_subgoal]. Based on what I see, I will...)
</reasoning>

<action>
[Insert Exact Action String Here]
</action>

**CRITICAL RULES**:
1. You MUST create a complete plan with 3-6 steps in the first response
2. The plan should logically lead to completing the main goal
3. Set current_subgoal to the first step of your plan
4. Be accurate - your beliefs will be checked against the actual environment
5. Choose actions ONLY from the Admissible Actions list
6. Do NOT include Chinese characters in your response
7. Use proper JSON formatting in the belief block

Your response:"""


# ═══════════════════════════════════════════════════════════════════════════════
# NEW: Standalone Planning Prompt - Runs BEFORE main interaction
# This prompt generates a task plan that fills the goal section of belief state
# ═══════════════════════════════════════════════════════════════════════════════

ALFWORLD_PLANNING_PROMPT_REBEL = """You are an expert task planner for the ALFRED embodied AI environment.
Your job is to analyze the task and create a detailed execution plan BEFORE the agent starts acting.

═══════════════════════════════════════════════════════════════════════════════
TASK TO PLAN
═══════════════════════════════════════════════════════════════════════════════

【Task Description】
{task_description}

【Initial Observation】
{observation}

【Available Actions】
{admissible_actions}

═══════════════════════════════════════════════════════════════════════════════
PLANNING INSTRUCTIONS
═══════════════════════════════════════════════════════════════════════════════

Analyze this task carefully and create a comprehensive plan:

1. **Goal Analysis**: What is the main objective? What objects are needed?
2. **Object Identification**: What target objects need to be found/manipulated?
3. **Location Prediction**: Where are these objects likely located based on common sense?
4. **Action Sequence**: What is the logical sequence of actions to complete the task?
5. **Contingency**: What alternatives exist if objects aren't in expected locations?

═══════════════════════════════════════════════════════════════════════════════
OUTPUT FORMAT (JSON ONLY)
═══════════════════════════════════════════════════════════════════════════════

Output ONLY a JSON object with the following structure:

{{
  "main_goal": "(The overall task objective in one sentence)",
  "target_objects": [
    {{"object": "(object name)", "required_state": "(e.g., heated, cooled, cleaned, or null)"}}
  ],
  "target_receptacle": "(Final destination for the object, if applicable)",
  "likely_locations": [
    "(Most likely location 1)",
    "(Most likely location 2)",
    "(Most likely location 3)"
  ],
  "plan_steps": [
    {{"step": 1, "subgoal": "(e.g., Find the apple)", "strategy": "(e.g., Check fridge, then countertop)"}},
    {{"step": 2, "subgoal": "(e.g., Pick up the apple)", "strategy": "(e.g., Navigate to location and pickup)"}},
    {{"step": 3, "subgoal": "(e.g., Heat the apple)", "strategy": "(e.g., Put in microwave, turn on, take out)"}},
    {{"step": 4, "subgoal": "(e.g., Place on countertop)", "strategy": "(e.g., Go to counter and put down)"}}
  ],
  "success_criteria": "(How to know the task is complete)"
}}

**RULES**:
1. Output ONLY the JSON object, no other text
2. Keep plan_steps between 3-6 steps
3. Be specific about objects and locations
4. Use proper JSON formatting

Your plan:"""


# ═══════════════════════════════════════════════════════════════════════════════
# Template with pre-computed plan (uses output from ALFWORLD_PLANNING_PROMPT_REBEL)
# ═══════════════════════════════════════════════════════════════════════════════

ALFWORLD_TEMPLATE_WITH_PLAN_REBEL = """You are an expert embodied agent operating in the ALFRED environment. Your goal is to efficiently complete the following task:
{task_description}

You are at the initial step (step 0).

═══════════════════════════════════════════════════════════════════════════════
PART 1: YOUR PRE-COMPUTED PLAN
═══════════════════════════════════════════════════════════════════════════════

【Task Plan】(Generated before interaction started)
- Main Goal: {plan_main_goal}
- Target Objects: {plan_target_objects}
- Target Receptacle: {plan_target_receptacle}
- Plan Steps:
{plan_steps_formatted}
- Success Criteria: {plan_success_criteria}

═══════════════════════════════════════════════════════════════════════════════
PART 2: CURRENT CONTEXT
═══════════════════════════════════════════════════════════════════════════════

【Current Observation】
{observation}

【Admissible Actions】
{admissible_actions}

═══════════════════════════════════════════════════════════════════════════════
PART 3: INSTRUCTIONS
═══════════════════════════════════════════════════════════════════════════════

Based on your pre-computed plan and current observation:
1. Initialize your belief state with the plan information
2. Observe the environment and note what you see
3. Select the first action to execute your plan

═══════════════════════════════════════════════════════════════════════════════
PART 4: OUTPUT FORMAT
═══════════════════════════════════════════════════════════════════════════════

<belief>
{{
  "world_model_update": {{
    "found_objects": {{}},
    "state_changes": {{}},
    "inventory": [],
    "cleared_receptacles": []
  }},
  "task_progress_update": {{
    "main_goal": "{plan_main_goal}",
    "plan": {plan_steps_json},
    "current_subgoal": "(First step from the plan)",
    "subgoal_status": "in_progress",
    "evidence": "(What you currently observe)"
  }},
  "exploration_map_update": {{
    "newly_visited": ["(Visible locations from observation)"],
    "next_priority": {plan_likely_locations}
  }}
}}
</belief>

<reasoning>
(Explain: Following my plan, the first subgoal is [X]. Based on what I see [Y], I will take action [Z]...)
</reasoning>

<action>
[Insert Exact Action String Here]
</action>

**CRITICAL RULES**:
1. Use the pre-computed plan to guide your actions
2. Update belief based on actual observations
3. Choose actions ONLY from the Admissible Actions list
4. Use proper JSON formatting

Your response:"""


# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT V3: Explicit Task Type Label
# Adding explicit task type to help model distinguish between different task strategies
# ═══════════════════════════════════════════════════════════════════════════════

ALFWORLD_TEMPLATE_EXPLICIT_TASK_TYPE = """You are an expert embodied agent operating in the ALFRED environment.

═══════════════════════════════════════════════════════════════════════════════
TASK INFORMATION
═══════════════════════════════════════════════════════════════════════════════

【Task Type】{task_type}
【Task Description】{task_description}

You are at step {current_step} (Total: {step_count}).

═══════════════════════════════════════════════════════════════════════════════
CURRENT CONTEXT
═══════════════════════════════════════════════════════════════════════════════

【Current Observation】
{observation}

【Current Belief State】
1. World Model: {world_state}
2. Task Progress: {task_state}
3. Exploration Map: {explore_map_state}

【Admissible Actions】
{admissible_actions}

【Recent History】
{history}

═══════════════════════════════════════════════════════════════════════════════
INSTRUCTIONS
═══════════════════════════════════════════════════════════════════════════════

Based on the **Task Type** ({task_type}), follow the appropriate strategy:
1. Update your belief state based on current observation
2. Select action that aligns with your task type's requirements

═══════════════════════════════════════════════════════════════════════════════
OUTPUT FORMAT
═══════════════════════════════════════════════════════════════════════════════

<belief>
{{
  "world_model_update": {{
    "found_objects": {{}},
    "state_changes": {{}},
    "cleared_receptacles": []
  }},
  "task_progress_update": {{
    "subgoal_status": "(in_progress/completed)",
    "evidence": "(observation evidence)",
    "updated_subgoal": "(next subgoal if needed)"
  }},
  "exploration_map_update": {{
    "newly_visited": [],
    "next_priority": []
  }}
}}
</belief>

<reasoning>
(Based on task type [{task_type}], I need to... Current subgoal is... So I will...)
</reasoning>

<action>
[Action from Admissible Actions]
</action>

Your response:"""


ALFWORLD_TEMPLATE_NO_HIS_EXPLICIT_TASK_TYPE = """You are an expert embodied agent operating in the ALFRED environment.

═══════════════════════════════════════════════════════════════════════════════
TASK INFORMATION
═══════════════════════════════════════════════════════════════════════════════

【Task Type】{task_type}
【Task Description】{task_description}

You are at the initial step (step 0).

═══════════════════════════════════════════════════════════════════════════════
CURRENT CONTEXT
═══════════════════════════════════════════════════════════════════════════════

【Current Observation】
{observation}

【Admissible Actions】
{admissible_actions}

═══════════════════════════════════════════════════════════════════════════════
INSTRUCTIONS
═══════════════════════════════════════════════════════════════════════════════

For task type **{task_type}**, create your initial plan and belief:

1. **Task Planning**: Create step-by-step plan following the task type strategy
2. **Belief Construction**: Initialize your belief state
3. **Action Selection**: Choose first action

═══════════════════════════════════════════════════════════════════════════════
OUTPUT FORMAT
═══════════════════════════════════════════════════════════════════════════════

<belief>
{{
  "world_model_update": {{
    "found_objects": {{}},
    "state_changes": {{}},
    "inventory": [],
    "cleared_receptacles": []
  }},
  "task_progress_update": {{
    "main_goal": "(task goal)",
    "task_type": "{task_type}",
    "plan": ["(step 1)", "(step 2)", "..."],
    "current_subgoal": "(first subgoal)",
    "subgoal_status": "in_progress",
    "evidence": "(initial observation)"
  }},
  "exploration_map_update": {{
    "newly_visited": [],
    "next_priority": []
  }}
}}
</belief>

<reasoning>
(Task type is [{task_type}], so my strategy is... First I need to...)
</reasoning>

<action>
[Action from Admissible Actions]
</action>

Your response:"""


# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT V3: Belief-Conditioned Action (Stronger subgoal emphasis)
# Make the current subgoal more prominent in action decision
# ═══════════════════════════════════════════════════════════════════════════════

ALFWORLD_TEMPLATE_BELIEF_CONDITIONED = """You are an expert embodied agent operating in the ALFRED environment.

═══════════════════════════════════════════════════════════════════════════════
TASK & CURRENT SUBGOAL
═══════════════════════════════════════════════════════════════════════════════

【Main Task】{task_description}

【⭐ CURRENT SUBGOAL ⭐】
{current_subgoal}

↑ YOUR ACTION MUST DIRECTLY SERVE THIS SUBGOAL ↑

You are at step {current_step} (Total: {step_count}).

═══════════════════════════════════════════════════════════════════════════════
CURRENT CONTEXT
═══════════════════════════════════════════════════════════════════════════════

【Current Observation】
{observation}

【Full Belief State】
- World Model: {world_state}
- Task Progress: {task_state}
- Exploration: {explore_map_state}

【Admissible Actions】
{admissible_actions}

【Recent History】
{history}

═══════════════════════════════════════════════════════════════════════════════
ACTION DECISION PROCESS
═══════════════════════════════════════════════════════════════════════════════

**Step 1**: Check if current subgoal "{current_subgoal}" is completed
**Step 2**: If not completed, what action directly advances this subgoal?
**Step 3**: If completed, what is the next subgoal and first action for it?

═══════════════════════════════════════════════════════════════════════════════
OUTPUT FORMAT
═══════════════════════════════════════════════════════════════════════════════

<belief>
{{
  "world_model_update": {{
    "found_objects": {{}},
    "state_changes": {{}},
    "cleared_receptacles": []
  }},
  "task_progress_update": {{
    "current_subgoal": "{current_subgoal}",
    "subgoal_status": "(in_progress/completed)",
    "evidence": "(what you see that relates to the subgoal)",
    "next_subgoal": "(if current completed, else null)"
  }},
  "exploration_map_update": {{
    "newly_visited": [],
    "next_priority": []
  }}
}}
</belief>

<reasoning>
Current subgoal is: {current_subgoal}
Observation shows: [what I see]
Subgoal status: [completed/in_progress] because [evidence]
To advance this subgoal, I should: [strategy]
Therefore, my action is: [action]
</reasoning>

<action>
[Action that directly serves the current subgoal]
</action>

Your response:"""


ALFWORLD_TEMPLATE_NO_HIS_BELIEF_CONDITIONED = """You are an expert embodied agent operating in the ALFRED environment.

═══════════════════════════════════════════════════════════════════════════════
TASK INFORMATION
═══════════════════════════════════════════════════════════════════════════════

【Main Task】{task_description}

You are at the initial step (step 0). You need to create a plan first.

═══════════════════════════════════════════════════════════════════════════════
CURRENT CONTEXT
═══════════════════════════════════════════════════════════════════════════════

【Current Observation】
{observation}

【Admissible Actions】
{admissible_actions}

═══════════════════════════════════════════════════════════════════════════════
INITIAL PLANNING
═══════════════════════════════════════════════════════════════════════════════

Create a clear plan with specific subgoals. Each subgoal should be:
- Concrete and achievable
- Directly leading to the main goal
- Easy to verify completion

═══════════════════════════════════════════════════════════════════════════════
OUTPUT FORMAT
═══════════════════════════════════════════════════════════════════════════════

<belief>
{{
  "world_model_update": {{
    "found_objects": {{}},
    "state_changes": {{}},
    "inventory": [],
    "cleared_receptacles": []
  }},
  "task_progress_update": {{
    "main_goal": "(the task goal)",
    "plan": [
      "(Subgoal 1: specific action like 'Find the apple')",
      "(Subgoal 2: next step)",
      "(Subgoal 3: ...)",
      "..."
    ],
    "current_subgoal": "(Subgoal 1 - your immediate focus)",
    "subgoal_status": "in_progress",
    "evidence": "(what you observe)"
  }},
  "exploration_map_update": {{
    "newly_visited": [],
    "next_priority": ["(where to look based on subgoal 1)"]
  }}
}}
</belief>

<reasoning>
My main goal is: [goal]
I've created this plan: [plan summary]
My FIRST subgoal is: [subgoal 1]
Based on observation, I will: [first action]
</reasoning>

<action>
[First action to achieve subgoal 1]
</action>

Your response:"""


# ═══════════════════════════════════════════════════════════════════════════════
# Template selector based on configuration
# ═══════════════════════════════════════════════════════════════════════════════

PROMPT_TEMPLATES = {
    "default": {
        "with_history": ALFWORLD_TEMPLATE_REBEL,
        "no_history": ALFWORLD_TEMPLATE_NO_HIS_REBEL,
        "planning": ALFWORLD_PLANNING_PROMPT_REBEL,
        "with_plan": ALFWORLD_TEMPLATE_WITH_PLAN_REBEL,
    },
    "explicit_task_type": {
        "with_history": ALFWORLD_TEMPLATE_EXPLICIT_TASK_TYPE,
        "no_history": ALFWORLD_TEMPLATE_NO_HIS_EXPLICIT_TASK_TYPE,
        "planning": ALFWORLD_PLANNING_PROMPT_REBEL,
        "with_plan": ALFWORLD_TEMPLATE_WITH_PLAN_REBEL,
    },
    "belief_conditioned": {
        "with_history": ALFWORLD_TEMPLATE_BELIEF_CONDITIONED,
        "no_history": ALFWORLD_TEMPLATE_NO_HIS_BELIEF_CONDITIONED,
        "planning": ALFWORLD_PLANNING_PROMPT_REBEL,
        "with_plan": ALFWORLD_TEMPLATE_WITH_PLAN_REBEL,
    },
}


def get_prompt_template(template_type: str = "default", has_history: bool = True, has_plan: bool = False):
    """
    Get the appropriate prompt template based on configuration.

    Args:
        template_type: "default", "explicit_task_type", or "belief_conditioned"
        has_history: Whether this is a step with history
        has_plan: Whether a pre-computed plan is available

    Returns:
        The appropriate prompt template string
    """
    templates = PROMPT_TEMPLATES.get(template_type, PROMPT_TEMPLATES["default"])

    if has_plan and not has_history:
        return templates.get("with_plan", templates["no_history"])
    elif has_history:
        return templates["with_history"]
    else:
        return templates["no_history"]
