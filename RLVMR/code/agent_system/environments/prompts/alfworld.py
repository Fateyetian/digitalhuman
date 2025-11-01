# projection_f# --------------------- ALFWorld --------------------- #
ALFWORLD_TEMPLATE_NO_HIS = """
You are an expert agent operating in the ALFRED Embodied Environment.
Your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags. 
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""

ALFWORLD_TEMPLATE = """
You are an expert agent operating in the ALFRED Embodied Environment. Your task is to: {task_description}
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observaitons and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags. 
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""

# ----------------- BDRS Runtime Templates -----------------
ALFWORLD_TEMPLATE_NO_HIS_BDRS = """
You are an expert agent operating in the ALFRED Embodied Environment.
Your current observation is: {current_observation}
Your admissible actions are: [{admissible_actions}].

You maintain three internal belief modules:
- M_t (World Model): factual beliefs about objects, locations, states, and interaction history.
  Example: {{"microwave": "closed", "apple 1": "on countertop 1"}}
- P_t (Task Progress): current goal, subgoals, and their completion status.
  Example: {{"goal": "heat apple", "subgoal": "place apple in microwave", "status": "incomplete"}}
- E_t (Exploration Map): visited/unvisited regions and unexplored containers.
  Example: {{"visited": ["kitchen"], "unexplored": ["pantry"]}}

Choose ONE mode and write one concise intent (≤20 words):
<PLAN>Formulate or modify subgoals based on P_t and M_t.</PLAN>
<EXECUTE>Complete current subgoal using credible knowledge in M_t.</EXECUTE>
<EXPLORE>Gather new information when M_t lacks key facts or E_t shows unknown areas.</EXPLORE>
<VERIFY>Challenge and correct possibly wrong beliefs in M_t.</VERIFY>

Mode-Action Alignment:
- <PLAN>: enabling actions (typically go to / open)
- <EXECUTE>: goal-advancing actions (take / put / use / heat / cool / clean / open / close)
- <EXPLORE>: information-gaining actions (open unexplored containers; go to unvisited receptacles)
- <VERIFY>: assumption-testing actions (re-open/close or go to for verification)
- When uncertain: prioritize open/close (reversible) > go to (low commitment) > state-changing actions (put)

Output Requirements:
1. Output ONE mode tag with one concise intent (≤20 words).
2. Then output <action>...</action>, EXACTLY copying one action string from [{admissible_actions}].
3. Do not output any other text besides the mode tag and action tag.

Example:
Observation: The refrigerator is closed, apple is on the counter.
Admissible actions: [go to fridge 1, open fridge 1, take apple 1 from countertop 1]
<EXECUTE>Directly acquire the apple to advance the subgoal.</EXECUTE>
<action>take apple 1 from countertop 1</action>
"""

ALFWORLD_TEMPLATE_BDRS = """
You are an expert agent operating in the ALFRED Embodied Environment. Your task is to: {task_description}
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observaitons and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}
Your admissible actions are: [{admissible_actions}].

You maintain three internal belief modules:
- M_t (World Model): factual beliefs about objects, locations, states, and interaction history.
  Example: {{"microwave": "closed", "apple 1": "on countertop 1"}}
- P_t (Task Progress): current goal, subgoals, and their completion status.
  Example: {{"goal": "heat apple", "subgoal": "place apple in microwave", "status": "incomplete"}}
- E_t (Exploration Map): visited/unvisited regions and unexplored containers.
  Example: {{"visited": ["kitchen"], "unexplored": ["pantry"]}}

Choose ONE mode and write one concise intent (≤20 words):
<PLAN>Formulate or modify subgoals based on P_t and M_t.</PLAN>
<EXECUTE>Complete current subgoal using credible knowledge in M_t.</EXECUTE>
<EXPLORE>Gather new information when M_t lacks key facts or E_t shows unknown areas.</EXPLORE>
<VERIFY>Challenge and correct possibly wrong beliefs in M_t.</VERIFY>

Mode-Action Alignment:
- <PLAN>: enabling actions (typically go to / open)
- <EXECUTE>: goal-advancing actions (take / put / use / heat / cool / clean / open / close)
- <EXPLORE>: information-gaining actions (open unexplored containers; go to unvisited receptacles)
- <VERIFY>: assumption-testing actions (re-open/close or go to for verification)
- When uncertain: prioritize open/close (reversible) > go to (low commitment) > state-changing actions (put)

Output Requirements:
1. Output ONE mode tag with one concise intent (≤20 words).
2. Then output <action>...</action>, EXACTLY copying one action string from [{admissible_actions}].
3. Do not output any other text besides the mode tag and action tag.
"""

ALFWORLD_TEMPLATE_NO_HIS_NOTHINK = """
You are an expert agent operating in the ALFRED Embodied Environment.
Your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should choose an admissible action for current step and present it within <action> </action> tags.
"""

ALFWORLD_TEMPLATE_NOTHINK = """
You are an expert agent operating in the ALFRED Embodied Environment. Your task is to: {task_description}
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observaitons and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should choose an admissible action for current step and present it within <action> </action> tags.
"""

# ALFWORLD_TEMPLATE_NO_HIS_MC = """
# You are an expert agent operating in the ALFRED Embodied Environment.
# Your current observation is: {current_observation}
# Your admissible actions of the current situation are: [{admissible_actions}].

# Now it's your turn to take an action, following these steps:

# 1. First, reason using ONLY ONE tag pair and express your reasoning in one concise, brief sentence:

# <planning>
# Plan or replan the entire task by breaking it down into high-level steps. Focus on outlining the full sequence required to complete the overall task, not just the immediate next action. 
# Use this at the beginning of complex tasks or whenever the previous plan is incorrect or insufficient.
# </planning>

# <explore>
# When results are unexpected or information is lacking, use current observations to think outside the box and list as many possible locations, items, or actions as possible.
# Use this approach when facing obstacles that require creative and innovative thinking.
# </explore>

# <reflection>
# Analyze the reasons for errors in task execution and correct them by exploring alternative approaches. 'Nothing happens' indicates the action is invalid.
# This is typically used when several consecutive actions yield no substantial progress. 
# </reflection>

# <proceed>
# Proceed to the next step based on the prior overall plan or the most recent unfinished exploration.
# This is most often used when the model clearly knows what to do next.
# </proceed>

# 2. After your reasoning, you MUST select and present an admissible action for the current step within <action> </action> tags.

# Specify the next action the agent should take to progress toward the task goal, following these guidelines:
# 1. Object and Receptacle References: Use specific identifiers:
# - [obj id] for objects (e.g., apple 1).
# - [recep id] for receptacles (e.g., countertop 1).
# 2. Action Validity: Follow the exact format below. Any deviation renders the action invalid:
# Valid actions: go to [recep id], take [obj id] from [recep id], put [obj id] in/on [recep id], open/close [recep id], use [obj id], heat/cool/clean [obj id] with [recep id]

# <action>
# Choose the most appropriate action from the valid actions.
# </action>
# """

ALFWORLD_TEMPLATE_NO_HIS_MC = """
You are an expert agent operating in the ALFRED Embodied Environment.
Your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action, following these steps:

1. First, reason using ONLY ONE tag pair and express your reasoning in one concise, brief sentence:

<planning>
Plan or replan the entire task by breaking it down into high-level steps. Focus on outlining the full sequence required to complete the overall task, not just the immediate next action. 
Use this at the beginning of complex tasks or whenever the previous plan is incorrect or insufficient.
It is necessary to list all the points separately. eg, step 1: xxx, step 2: xxx, step 3: xxx, etc.
</planning>

<explore>
When results are unexpected or information is lacking, use current observations to think outside the box and list as many possible locations, items, or actions as possible.
Use this approach when facing obstacles that require creative and innovative thinking.
</explore>

<reflection>
Analyze the reasons for errors in task execution and correct them by exploring alternative approaches. 'No known action matches that input.' indicates the action is invalid.
This is typically used when several consecutive actions yield no substantial progress. 
</reflection>

<monitor>  
Continuously track the current progress and history of reasoning and execution throughout the task. Recall the current subgoal and consider the next concrete action, ensuring agent alignment with the overall plan.  
Typically used when task outcomes are as expected and no other mode of reasoning is required.
</monitor>

2. After your reasoning, you MUST select and present an admissible action for the current step within <action> </action> tags.

Specify the next action the agent should take to progress toward the task goal, following these guidelines:
1. Object and Receptacle References: Use specific identifiers:
- [obj id] for objects (e.g., apple 1).
- [recep id] for receptacles (e.g., countertop 1).
2. Action Validity: Follow the exact format below. Any deviation renders the action invalid:
Valid actions: go to [recep id], take [obj id] from [recep id], put [obj id] in/on [recep id], open/close [recep id], use [obj id], heat/cool/clean [obj id] with [recep id]

<action>
Choose the most appropriate action from the valid actions.
</action>
"""

ALFWORLD_TEMPLATE_MC = """
You are an expert agent operating in the ALFRED Embodied Environment. Your task is to: {task_description}
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observaitons and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Your previous overall plan is: {planning}. Please strictly adhere to your plan.

Now it's your turn to take an action, following these steps:

1. First, reason using ONLY ONE tag pair and express your reasoning in one concise, brief sentence:

<planning>
Plan or replan the entire task by breaking it down into high-level steps. Focus on outlining the full sequence required to complete the overall task, not just the immediate next action. 
Use this at the beginning of complex tasks or whenever the previous plan is incorrect or insufficient.
It is necessary to list all the points separately. eg, step 1: xxx, step 2: xxx, step 3: xxx, etc.
</planning>

<explore>
When results are unexpected or information is lacking, use current observations to think outside the box and list as many possible locations, items, or actions as possible.
Use this approach when facing obstacles that require creative and innovative thinking.
</explore>

<reflection>
Analyze the reasons for errors in task execution and correct them by exploring alternative approaches. 'No known action matches that input.' indicates the action is invalid.
This is typically used when several consecutive actions yield no substantial progress. 
</reflection>

<monitor>  
Continuously track the current progress and history of reasoning and execution throughout the task. Recall the current subgoal and consider the next concrete action, ensuring agent alignment with the overall plan.  
Typically used when task outcomes are as expected and no other mode of reasoning is required.
</monitor>

2. After your reasoning, you MUST select and present an admissible action for the current step within <action> </action> tags.

Specify the next action the agent should take to progress toward the task goal, following these guidelines:
1. Object and Receptacle References: Use specific identifiers:
- [obj id] for objects (e.g., apple 1).
- [recep id] for receptacles (e.g., countertop 1).
2. Action Validity: Follow the exact format below. Any deviation renders the action invalid:
Valid actions: go to [recep id], take [obj id] from [recep id], put [obj id] in/on [recep id], open/close [recep id], use [obj id], heat/cool/clean [obj id] with [recep id]

<action>
Choose the most appropriate action from the valid actions.
</action>
"""