"""
WebShop ReBel (Reward Belief) Prompt Templates

This file contains all prompt templates for the ReBel framework applied to WebShop:
1. Tagging/Annotation prompts (for Teacher LLM hindsight annotation)
2. Cold-start prompts (for SFT training, NO admissible actions)
3. RL training/evaluation prompts (WITH admissible actions)
"""

# ============================================================================
# Part 1: Tagging/Annotation Prompts (for Teacher LLM Hindsight Annotation)
# ============================================================================

WEBSHOP_REBEL_TAGGING_TEMPLATE = """You are an expert in e-commerce shopping reasoning. Your goal is to simulate the internal 'thought process' of an intelligent shopping agent.
You must maintain a consistent understanding of the shopping task and make logical decisions based ON ONLY what has been observed up to the current moment.

**TASK CONTEXT**
Shopping Goal: {task_description}

**CURRENT STATE**
- Latest Observation: "{current_observation}"
- Previous Belief State: {prev_belief_json}
- Available Actions: {admissible_actions}

**DECISION TO PROCESS**
The next action to be taken is: `{ground_truth_action}`

**YOUR OBJECTIVE**
Generate the Belief Update and Reasoning that justifies this action.

### CRITICAL LOGIC RULES:

1. **NO PREDICTIONS (Temporal Consistency)**:
   - The belief update must reflect the state **BEFORE** the action `{ground_truth_action}` is executed.
   - If the action is `search[query]`, `search_status` should still reflect the state before the search is issued.
   - If the action is `click[Buy Now]`, `search_status` may only be `ready_to_buy` if all conditions below are already met; otherwise it must remain `options_selecting`.

2. **AUTONOMOUS REASONING**:
   - Write the `reasoning` as a first-person internal monologue (e.g., "I see X, so I will do Y").
   - **DO NOT** mention "the expert", "the ground truth", or "the provided action".
   - Act as if YOU are the one deciding to take `{ground_truth_action}` based on your beliefs.

3. **OBSERVATION-GROUNDED VERIFICATION**:
   - `verified`: attribute explicitly visible in the current observation — an option button was clicked, or a tab's content was directly read and the attribute is stated there.
   - `inferred_only`: inferred from product title or visible text, but NOT confirmed by clicking a tab. Must stay `inferred_only` until directly observed.
   - `unverified`: attribute has not been checked at all.
   - Do NOT move an attribute from `inferred_only` to `verified` just because `{ground_truth_action}` is `click[Buy Now]`.
   - `current_product_match` = `"exact"` ONLY when ALL target attributes are in `verified` AND `inferred_only` is EMPTY.

4. **STATE TRANSITIONS AND READY-TO-BUY**:
   - `search_status` must follow: `not_started → searching → product_found → options_selecting → ready_to_buy`. Never skip stages.
   - `ready_to_buy` requires ALL target attributes in `verified` AND `inferred_only` is EMPTY. If unverified attributes remain, stay at `options_selecting` and explicitly acknowledge the gap in `reasoning`.

5. **CUMULATIVE EXPLORATION TRACKING**:
   - `exploration_state` must reflect ALL actions taken so far (see trajectory below), not just the current step.
   - Copy from Previous Belief State and ADD any new items introduced by the current observation.

### OUTPUT FORMAT (JSON ONLY)
```json
{{
  "product_understanding": {{
    "target_attributes": {{"attr": "value"}},
    "current_product_match": "partial/exact/none",
    "price_constraint": "under $X / any"
  }},
  "attribute_verification": {{
    "verified": [
      {{"attribute": "color", "value": "navy", "source": "option_button", "snippet": "navy [clicked]"}}
    ],
    "unverified": ["material", "care_instructions"],
    "inferred_only": [
      {{"attribute": "slip_resistance", "inference": "title mentions rubber sole but tab not clicked", "required_action": "click Description or Features tab"}}
    ]
  }},
  "search_progress": {{
    "search_status": "not_started/searching/product_found/options_selecting/ready_to_buy",
    "evidence": "what did you just observe? which attributes are verified vs unverified?",
    "updated_subgoal": "what is the immediate goal this action serves?"
  }},
  "exploration_state": {{
    "queries_tried": ["query1", "query2"],
    "products_viewed": ["product_asin"],
    "options_selected": ["option_value"],
    "tabs_clicked": ["Description", "Features"]
  }},
  "reasoning": "I observed [Evidence from the observation]. Verified attributes: [list with source]. Inferred but unconfirmed: [list]. Unverified: [list]. My current search_status is [X] and my subgoal is [Y]. Therefore, I will [{ground_truth_action}] because [justification]."
}}
```

Actions taken so far (chronological):
{traj}
"""


# ============================================================================
# Part 2: Cold-Start Prompts (for SFT training, NO admissible actions)
# ============================================================================

WEBSHOP_REBEL_TEMPLATE_NO_HIS_CS = """
You are an expert autonomous agent operating in the WebShop e-commerce environment.
Your task is to: {task_description}
Your current observation is: {current_observation}

Now it's your turn to take an action, following these steps:

**Step 1: Update Your Belief State**

Output a JSON-formatted belief state update based on your current observation.
The belief state has four components:

1. **Product Understanding** - What you know about the target product:
   ```json
   {{
     "target_attributes": {{"attr": "value"}},
     "current_product_match": "partial/exact/none",
     "price_constraint": "under $X / any"
   }}
   ```

2. **Attribute Verification** - Track what is confirmed vs. assumed:
   ```json
   {{
     "verified": [{{"attribute": "color", "value": "navy", "source": "option_button", "snippet": "navy [clicked]"}}],
     "unverified": ["material"],
     "inferred_only": [{{"attribute": "...", "inference": "...", "required_action": "click Description tab"}}]
   }}
   ```
   Note: `"exact"` match requires ALL attributes in `verified`. If `inferred_only` is non-empty, match must be `"partial"`.

3. **Search Progress** - Your current shopping progress:
   ```json
   {{
     "search_status": "not_started/searching/product_found/options_selecting/ready_to_buy",
     "evidence": "Brief explanation of what you learned from the observation",
     "updated_subgoal": "Your next subgoal"
   }}
   ```
   State transition: `not_started→searching→product_found→options_selecting→ready_to_buy`. Never skip stages. `ready_to_buy` requires `inferred_only` to be EMPTY.

4. **Exploration State** - What you've tried so far:
   ```json
   {{
     "queries_tried": ["query1"],
     "products_viewed": ["product1"],
     "options_selected": ["option1"],
     "tabs_clicked": ["Description"]
   }}
   ```

**Step 2: Reasoning**

Explain your thought process based on your belief state and current observation.
Structure your reasoning as:
- What evidence do I have from the observation?
- How well does the current product match my goal?
- What is my current subgoal?
- What action should I take to make progress?

**Step 3: Action Selection**

Select an action. Valid action formats are:
- search[your search query] - to search for products
- click[element] - to click on a page element (product, option, Buy Now, Back to Search, etc.)

**Output Format** (belief is an incremental update for this step, not the full accumulated state):
```
<belief>
{{
  "product_understanding": {{...}},
  "attribute_verification": {{...}},
  "search_progress": {{...}},
  "exploration_state": {{...}}
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

WEBSHOP_REBEL_TEMPLATE_CS = """
You are an expert autonomous agent operating in the WebShop e-commerce environment.
Your task is to: {task_description}

Prior to this step, you have already taken {step_count} step(s).
Below are the most recent {history_length} observations and actions: {action_history}

You are now at step {current_step} and your current observation is: {current_observation}

**Your Current Belief State:**
{current_belief_state}

**Your Current Subgoal:** {planning}

Now it's your turn to take an action, following these steps:

**Step 1: Update Your Belief State**

Based on the new observation, update your belief state:

1. **Product Understanding** - What changed in your knowledge:
   ```json
   {{
     "target_attributes": {{"attr": "value"}},
     "current_product_match": "partial/exact/none",
     "price_constraint": "under $X / any"
   }}
   ```

2. **Attribute Verification** - Track what is confirmed vs. assumed:
   ```json
   {{
     "verified": [{{"attribute": "color", "value": "navy", "source": "option_button", "snippet": "navy [clicked]"}}],
     "unverified": ["material"],
     "inferred_only": [{{"attribute": "...", "inference": "...", "required_action": "click Description tab"}}]
   }}
   ```
   Note: `"exact"` match requires ALL attributes in `verified`. If `inferred_only` is non-empty, match must be `"partial"`.

3. **Search Progress** - How did your subgoal progress:
   ```json
   {{
     "search_status": "not_started/searching/product_found/options_selecting/ready_to_buy",
     "evidence": "What you learned from the observation",
     "updated_subgoal": "Your next subgoal"
   }}
   ```
   State transition: `not_started→searching→product_found→options_selecting→ready_to_buy`. Never skip stages. `ready_to_buy` requires `inferred_only` to be EMPTY.

4. **Exploration State** - What you've tried:
   ```json
   {{
     "queries_tried": ["query1"],
     "products_viewed": ["product1"],
     "options_selected": ["option1"],
     "tabs_clicked": ["Description"]
   }}
   ```

**Step 2: Reasoning**

Explain your reasoning:
- What new evidence did I observe?
- How does this align with my previous plan?
- What is my current subgoal status?
- What should I do next?

**Step 3: Action Selection**

Select an action. Valid action formats are:
- search[your search query] - to search for products
- click[element] - to click on a page element (product, option, Buy Now, Back to Search, etc.)

**Output Format** (belief is an incremental update for this step, not the full accumulated state):
```
<belief>
{{
  "product_understanding": {{...}},
  "attribute_verification": {{...}},
  "search_progress": {{...}},
  "exploration_state": {{...}}
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
# Simplified to match SFT training data format for better format alignment.
# ============================================================================

WEBSHOP_REBEL_TEMPLATE_NO_HIS_RL = """You are an expert autonomous agent operating in the WebShop e-commerce environment.
Your task is to: {task_description}

This is your first step. Your current observation is:
{current_observation}

Your available actions are: [{admissible_actions}].

**IMPORTANT: Keep your belief JSON concise. Each field value should be brief (under 20 words). Do not repeat the observation text verbatim.**

**Your Current Belief State:**
{{
  "product_understanding": {{
    "target_attributes": {{}},
    "current_product_match": "none",
    "price_constraint": "any"
  }},
  "attribute_verification": {{
    "verified": [],
    "unverified": [],
    "inferred_only": []
  }},
  "search_progress": {{
    "search_status": "not_started",
    "evidence": "",
    "updated_subgoal": "Start searching for the target product"
  }},
  "exploration_state": {{
    "queries_tried": [],
    "products_viewed": [],
    "options_selected": [],
    "tabs_clicked": []
  }}
}}

Now it's your turn to take an action, following these steps:

**Step 1: Update Your Belief State**

Based on the new observation, update your belief state:

1. **Product Understanding** - What you know about the target product:
   ```json
   {{
     "target_attributes": {{"attr": "value"}},
     "current_product_match": "partial/exact/none",
     "price_constraint": "under $X / any"
   }}
   ```

2. **Attribute Verification** - Track what is confirmed vs. assumed:
   ```json
   {{
     "verified": [{{"attribute": "color", "value": "navy", "source": "option_button", "snippet": "navy [clicked]"}}],
     "unverified": ["material"],
     "inferred_only": [{{"attribute": "...", "inference": "...", "required_action": "click Description tab"}}]
   }}
   ```

3. **Search Progress** - How did your shopping progress:
   ```json
   {{
     "search_status": "not_started/searching/product_found/options_selecting/ready_to_buy",
     "evidence": "What you learned from the observation",
     "updated_subgoal": "Your next subgoal"
   }}
   ```

4. **Exploration State** - What you have tried so far:
   ```json
   {{
     "queries_tried": ["query1"],
     "products_viewed": ["product_asin"],
     "options_selected": ["option_value"],
     "tabs_clicked": ["Description", "Features"]
   }}
   ```

**Step 2: Reasoning**

In 2-3 sentences, explain: what did you observe, and why are you taking the next action?

**Step 3: Action Selection**

You MUST select and present an admissible action from the list: [{admissible_actions}].

**Output Format:**

<belief>
{{
  "product_understanding": {{...}},
  "attribute_verification": {{...}},
  "search_progress": {{...}},
  "exploration_state": {{...}}
}}
</belief>

<reasoning>
2-3 sentences only.
</reasoning>

<action>
Your chosen action from the admissible actions list
</action>"""

WEBSHOP_REBEL_TEMPLATE_RL = """You are an expert autonomous agent operating in the WebShop e-commerce environment.
Your task is to: {task_description}

Prior to this step, you have already taken {step_count} step(s).
Below are your most recent {history_length} actions (observations are summarized in your Belief State):
{action_history}

You are now at step {current_step} and your current observation is:
{current_observation}

Your available actions are: [{admissible_actions}].

**Your Current Belief State:**
{current_belief_state}

Now it's your turn to take an action, following these steps:

**Step 1: Update Your Belief State**

Based on the new observation, update your belief state:

1. **Product Understanding** - What you know about the target product:
   ```json
   {{
     "target_attributes": {{"attr": "value"}},
     "current_product_match": "partial/exact/none",
     "price_constraint": "under $X / any"
   }}
   ```

2. **Attribute Verification** - Track what is confirmed vs. assumed:
   ```json
   {{
     "verified": [{{"attribute": "color", "value": "navy", "source": "option_button", "snippet": "navy [clicked]"}}],
     "unverified": ["material"],
     "inferred_only": [{{"attribute": "...", "inference": "...", "required_action": "click Description tab"}}]
   }}
   ```

3. **Search Progress** - How did your shopping progress:
   ```json
   {{
     "search_status": "not_started/searching/product_found/options_selecting/ready_to_buy",
     "evidence": "What you learned from the observation",
     "updated_subgoal": "Your next subgoal"
   }}
   ```

4. **Exploration State** - What you have tried so far:
   ```json
   {{
     "queries_tried": ["query1"],
     "products_viewed": ["product_asin"],
     "options_selected": ["option_value"],
     "tabs_clicked": ["Description", "Features"]
   }}
   ```

**Step 2: Reasoning**

In 2-3 sentences, explain: what did you observe, and why are you taking the next action?

**Step 3: Action Selection**

You MUST select and present an admissible action from the list: [{admissible_actions}].

**Output Format:**

<belief>
{{
  "product_understanding": {{...}},
  "attribute_verification": {{...}},
  "search_progress": {{...}},
  "exploration_state": {{...}}
}}
</belief>

<reasoning>
2-3 sentences only.
</reasoning>

<action>
Your chosen action from the admissible actions list
</action>"""


# ============================================================================
# Summary of Prompt Usage
# ============================================================================

"""
Prompt Usage Summary:

1. **Data Annotation Phase** (Teacher LLM):
   - WEBSHOP_REBEL_TAGGING_TEMPLATE
   - Used by: generate_webshop_rebel_hindsight.py
   - Input: observation + expert action + prev_belief
   - Output: belief_update + reasoning + action

2. **Cold-Start SFT Phase** (NO admissible actions):
   - WEBSHOP_REBEL_TEMPLATE_NO_HIS_CS (first step)
   - WEBSHOP_REBEL_TEMPLATE_CS (subsequent steps)
   - Used by: SFT training scripts
   - Purpose: Force model to learn strong reasoning without action hints

3. **RL Training/Evaluation Phase** (WITH admissible actions):
   - WEBSHOP_REBEL_TEMPLATE_NO_HIS_RL (first step)
   - WEBSHOP_REBEL_TEMPLATE_RL (subsequent steps)
   - Used by: ReBel RL trainer (WebshopEnvironmentManager)
   - Purpose: RL training with reduced exploration difficulty
"""
