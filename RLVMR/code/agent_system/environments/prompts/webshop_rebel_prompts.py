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
- Actions Taken So Far (chronological):
{traj}

**DECISION TO PROCESS**
The next action to be taken is: `{ground_truth_action}`

**YOUR OBJECTIVE**
Generate the Belief Update and Reasoning that leads to this action.

### THREE COGNITIVE LAYERS (You MUST apply these in order):

**L1 — Observation (What is literally on screen?)**
List every piece of information EXPLICITLY visible in the current observation: product title text, price, clickable option buttons, tab labels. ONLY include what you can copy-paste from the observation.

**L2 — Inference (What can be logically deduced?)**
From the visible information, what can you infer? E.g., if the title says "leather loafers", you can infer the material is likely leather — but this is an INFERENCE, not a verification. Inferences belong in `inferred_only`, NOT in `verified`.

**L3 — Verification (What has been confirmed by clicking a tab?)**
An attribute is VERIFIED only if the agent has clicked a tab (Description/Features/Reviews) and the tab content explicitly confirms it. Seeing a filter option or title keyword is NOT verification. The tab must have been clicked AND the content must have been observed.

### CRITICAL LOGIC RULES:
1. **NO PREDICTIONS (Temporal Consistency)**:
   - The belief update must reflect the state **BEFORE** the action `{ground_truth_action}` is executed.
   - If the action is "search[query]", the `search_status` should still be the previous state.
   - If the action is "click[Buy Now]", the `search_status` should be "ready_to_buy" (decided but not yet purchased).

2. **AUTONOMOUS REASONING**:
   - Write the `reasoning` as a first-person internal monologue (e.g., "I see X, so I will do Y").
   - **DO NOT** mention "the expert", "the ground truth", or "the provided action".
   - Act as if YOU are the one deciding to take the action `{ground_truth_action}` based on your beliefs.

3. **PRODUCT MATCHING (Observation-Grounded Only)**:
   - Only mark an attribute as matched if it is **EXPLICITLY visible** in the current observation text.
   - WebShop product pages typically show: product title, price, clickable color/size options, and tab labels (Description, Features, Reviews). The **content** of Description/Features/Reviews is NOT visible unless the agent has clicked on that tab.
   - Attributes like **material** (e.g. polyester spandex), **care** (e.g. machine wash), and **feature** (e.g. moisture wicking, slip resistance, rubber outsole) are usually hidden behind the Description or Features tab. Do NOT assume these attributes are satisfied just because the product title seems related — the agent must have actually clicked the tab and seen the content.
   - `"current_product_match"` rules:
     - `"exact"`: ALL target attributes are in the `verified` list (L3 confirmed).
     - `"partial"`: Some attributes are verified, but `unverified` or `inferred_only` lists are non-empty.
     - `"none"`: No meaningful attributes confirmed yet.
   - In `evidence`, explicitly state which attributes are verified vs. unverified. Example: "Color 'navy' and size 'x-large' are confirmed from the options. Material and care instructions have NOT been verified — would require clicking Description/Features tab."
   - Track which options have been selected and which remain.

4. **STATE TRANSITION MATRIX (HARD RULES)**:
   The `search_status` field MUST follow legal transitions. Illegal jumps are FORBIDDEN:
   - `not_started` → `searching` (only after a search action)
   - `searching` → `product_found` (only after clicking a product from results)
   - `product_found` → `options_selecting` (only after viewing product page)
   - `options_selecting` → `ready_to_buy` (ONLY if Ready-to-Buy Checklist passes)
   - `options_selecting` → `options_selecting` (selecting more options / clicking tabs)
   - ILLEGAL: `searching` → `ready_to_buy` (skips product_found and options_selecting)
   - ILLEGAL: `product_found` → `ready_to_buy` (skips options_selecting)

5. **READY-TO-BUY CHECKLIST (ALL must be true to set ready_to_buy)**:
   a. All target attributes must appear in `attribute_verification.verified` — OR be explicitly flagged as `unverifiable` (attribute genuinely cannot be checked in WebShop UI).
   b. `attribute_verification.inferred_only` MUST be EMPTY (no unconfirmed inferences).
   c. If any attribute remains in `unverified` or `inferred_only`, you MUST stay at `options_selecting`.
   d. If all of the above pass, you may set `ready_to_buy`.

6. **ANTI-HINDSIGHT LEAKAGE (HARD BLOCK)**:
   - Do NOT let the ground truth action bias your belief assessment.
   - If the action is `click[Buy Now]` but unverified attributes remain:
     - `current_product_match` MUST be "partial" (not "exact")
     - `search_status` MUST remain "options_selecting" (not "ready_to_buy")
     - `reasoning` MUST explicitly acknowledge: "I am buying without verifying [list attributes] — this is a gap in my verification."
   - NEVER promote an inference (L2) to a verification (L3) just because Buy Now is clicked.

7. **CUMULATIVE EXPLORATION TRACKING**:
   - `exploration_state` MUST reflect ALL actions taken so far (see "Actions Taken So Far" above), not just the current step.
   - If a previous action was `click[7.5]`, then `options_selected` MUST include "7.5".
   - If a previous action was `click[boulder]`, then `options_selected` MUST include "boulder".
   - If a previous action was `click[Description]`, then `tabs_clicked` MUST include "Description".
   - If a previous action was `click[B07L6DV555]`, then `products_viewed` MUST include that ASIN.
   - Copy these from the Previous Belief State and ADD any new items from the current action.
   - When the observation is identical to the previous step, use the action history to determine what changed.

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
      {{"attribute": "slip_resistance", "inference": "title mentions loafers but no explicit confirmation", "required_action": "click Description/Features tab"}}
    ]
  }},
  "search_progress": {{
    "search_status": "not_started/searching/product_found/options_selecting/ready_to_buy",
    "evidence": "what did you just observe? Which attributes are verified vs unverified?",
    "updated_subgoal": "what is the immediate goal this action serves?"
  }},
  "exploration_state": {{
    "queries_tried": ["query1", "query2"],
    "products_viewed": ["product1"],
    "options_selected": ["option1"],
    "tabs_clicked": ["Description", "Features"]
  }},
  "reasoning": "I observed [Evidence from L1]. I can infer [L2 inferences]. Verified attributes: [L3 list]. Unverified: [list]. Therefore, I will [Action] because [justification]."
}}
```

now the trajectory is as follows: {traj}
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
The belief state has three components:

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

**Output Format:**
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

**Your Previous Overall Plan:** {planning}

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

**Output Format:**
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

WEBSHOP_REBEL_TEMPLATE_NO_HIS_RL = """Task: {task_description}

Observation:
{current_observation}

Current Belief State:
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

Available Actions:
{admissible_actions}

Respond strictly in this format:
<belief>
{{"product_understanding": {{...}}, "attribute_verification": {{...}}, "search_progress": {{...}}, "exploration_state": {{...}}}}
</belief>
<reasoning>
Your step-by-step reasoning here.
</reasoning>
<action>
Your chosen action here (must be one of the Available Actions above)
</action>"""

WEBSHOP_REBEL_TEMPLATE_RL = """Task: {task_description}

History (last {history_length} steps):
{action_history}

Current Observation (Step {current_step}):
{current_observation}

Current Belief State:
{current_belief_state}

Available Actions:
{admissible_actions}

Respond strictly in this format:
<belief>
{{"product_understanding": {{...}}, "attribute_verification": {{...}}, "search_progress": {{...}}, "exploration_state": {{...}}}}
</belief>
<reasoning>
Your step-by-step reasoning here.
</reasoning>
<action>
Your chosen action here (must be one of the Available Actions above)
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
