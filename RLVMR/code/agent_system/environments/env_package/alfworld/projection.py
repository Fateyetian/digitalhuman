"""
ALFWorld Projection Functions

This module contains projection functions that validate and process model outputs.
Supports two modes:
1. Basic mode: <think>...</think> + <action>...</action>
2. ReBel mode: <belief>...</belief> + <action>...</action>

Enhanced with intelligent action matching to handle:
- Ambiguous prepositions (in/on, on/in)
- Minor formatting differences
- Fuzzy matching to admissible actions
"""

import re
from typing import List, Tuple, Optional


def match_action_to_admissible(action: str, admissible_actions: List[str]) -> Tuple[str, bool]:
    """
    Intelligent action matching to handle ambiguous prepositions and minor variations

    Handles common issues:
    - "put X in/on Y" or "put X on/in Y" (ambiguous prepositions from SFT data)
    - Extra whitespace
    - Case sensitivity

    Args:
        action: The extracted action from model output
        admissible_actions: List of valid actions from the environment

    Returns:
        (matched_action, was_corrected): The best matching action and whether it was corrected
    """
    if not admissible_actions:
        return action, False

    # Strategy 1: Direct match (fastest path)
    if action in admissible_actions:
        return action, False

    # Strategy 2: Handle in/on ambiguity (most common issue)
    if 'in/on' in action or 'on/in' in action:
        # Try replacing with 'in'
        action_with_in = action.replace('in/on', 'in').replace('on/in', 'in')
        if action_with_in in admissible_actions:
            return action_with_in, True

        # Try replacing with 'on'
        action_with_on = action.replace('in/on', 'on').replace('on/in', 'on')
        if action_with_on in admissible_actions:
            return action_with_on, True

    # Strategy 3: Normalize whitespace and try again
    normalized_action = ' '.join(action.split())
    if normalized_action in admissible_actions:
        return normalized_action, True

    # Strategy 4: Fuzzy match - find closest admissible action
    # Look for actions with similar prefix (same action type)
    action_prefix = action.split()[0] if action.split() else ""
    if action_prefix:
        candidates = [a for a in admissible_actions if a.startswith(action_prefix)]

        # If only one candidate with same prefix, use it (e.g., only one "put" action available)
        if len(candidates) == 1:
            return candidates[0], True

        # Try to find exact substring match
        for candidate in candidates:
            if action in candidate or candidate in action:
                return candidate, True

    # Strategy 5: No match found, return original action
    # The environment will reject it, but at least we tried
    return action, False


def alfworld_projection(actions: List[str], action_pools: List[List[str]]) -> Tuple[List[str], List[int], List[None], List[bool]]:
    """
    Basic projection function for standard <think>/<action> format

    Args:
        actions: List of model outputs
        action_pools: List of admissible actions for each environment

    Returns:
        actions_out: Extracted actions
        valids: Binary validity flags (1=valid, 0=invalid)
        plannings: Placeholder (always None for basic mode)
        action_available: Whether extracted action is in admissible set
    """
    actions_out = []
    valids = []
    action_available = [False] * len(actions)

    for i in range(len(actions)):
        original_str = actions[i]
        actions_lower = actions[i].lower()
        valid = 1
        act_str = ""

        # Check for Chinese characters
        if re.search(r'[\u4e00-\u9fff]', original_str):
            valid = 0

        # Check for exactly ONE <think>...</think>
        think_matches = re.findall(r"<think>([\s\S]*?)</think>", original_str, re.IGNORECASE)
        if len(think_matches) != 1:
            valid = 0
        else:
            think_content = think_matches[0].strip()
            if not think_content:  # Empty think tag
                valid = 0

        # Check for exactly ONE <action>...</action>
        action_matches = re.findall(r"<action>([\s\S]*?)</action>", actions_lower, re.IGNORECASE)
        if len(action_matches) != 1:
            valid = 0
        else:
            act_str = action_matches[0].strip()

            # Intelligent action matching
            matched_action, was_corrected = match_action_to_admissible(act_str, action_pools[i])
            act_str = matched_action

            # Check if (possibly corrected) action is in admissible actions
            if act_str in action_pools[i]:
                action_available[i] = True

        # If invalid, use fallback
        if not act_str:
            act_str = actions_lower[-30:]  # Fallback to last 30 chars

        actions_out.append(act_str)
        valids.append(valid)

    return actions_out, valids, [None] * len(actions), action_available


def alfworld_projection_rebel(actions: List[str], action_pools: List[List[str]]) -> Tuple[List[str], List[int], List[str], List[bool]]:
    """
    ReBel projection function for <belief>/<reasoning>/<action> format

    Validates:
    1. Exactly one <belief>...</belief> block containing valid JSON with required keys
    2. Optional <reasoning>...</reasoning> block
    3. Exactly one <action>...</action> block
    4. <belief> appears before <action>
    5. No Chinese characters

    Expected belief format:
    {
      "world_model_update": {...},
      "task_progress_update": {...},
      "exploration_map_update": {...}
    }

    Args:
        actions: List of model outputs
        action_pools: List of admissible actions for each environment

    Returns:
        actions_out: Extracted actions
        valids: Binary validity flags (1=valid, 0=invalid)
        beliefs: Extracted belief text (for reward calculation)
        action_available: Whether extracted action is in admissible set
    """
    import json

    actions_out = []
    valids = []
    beliefs = []
    action_available = [False] * len(actions)

    for i, output in enumerate(actions):
        valid = 1
        act_str = ""
        belief_text = ""

        # Check for Chinese characters
        if re.search(r'[\u4e00-\u9fff]', output):
            valid = 0

        # Check for exactly ONE <belief>...</belief>
        belief_matches = re.findall(r"<belief>(.*?)</belief>", output, re.DOTALL | re.IGNORECASE)
        if len(belief_matches) != 1:
            valid = 0
        else:
            belief_text = belief_matches[0].strip()

            # Validate belief structure: try to parse as JSON
            try:
                belief_json = belief_text.replace("'", '"')
                belief_data = json.loads(belief_json)

                # Check for required keys in new format
                has_world_model = 'world_model_update' in belief_data
                has_task_progress = 'task_progress_update' in belief_data
                has_exploration = 'exploration_map_update' in belief_data

                # Also support old format: M_t, P_t, E_t
                has_old_format = (
                    re.search(r'M_t:', belief_text, re.IGNORECASE) and
                    re.search(r'P_t:', belief_text, re.IGNORECASE) and
                    re.search(r'E_t:', belief_text, re.IGNORECASE)
                )

                # Valid if either new format or old format is present
                if not (has_world_model or has_task_progress or has_exploration or has_old_format):
                    valid = 0

            except json.JSONDecodeError:
                # Try old format parsing
                if not (re.search(r'M_t:', belief_text, re.IGNORECASE) and
                        re.search(r'P_t:', belief_text, re.IGNORECASE) and
                        re.search(r'E_t:', belief_text, re.IGNORECASE)):
                    valid = 0

            # Check if belief is empty
            if not belief_text or len(belief_text) < 10:
                valid = 0

        # Check for exactly ONE <action>...</action>
        action_matches = re.findall(r"<action>([\s\S]*?)</action>", output, re.IGNORECASE)
        if len(action_matches) != 1:
            valid = 0
        else:
            act_str = action_matches[0].strip().lower()

            # Intelligent action matching (handles in/on ambiguity and other issues)
            matched_action, was_corrected = match_action_to_admissible(act_str, action_pools[i])
            act_str = matched_action

            # Check if (possibly corrected) action is in admissible actions
            if act_str in action_pools[i]:
                action_available[i] = True

        # Check that <belief> appears before <action>
        belief_pos = output.lower().find("<belief>")
        action_pos = output.lower().find("<action>")
        if belief_pos == -1 or action_pos == -1 or belief_pos > action_pos:
            valid = 0

        # Optional: check for <reasoning> (encouraged but not required)
        reasoning_matches = re.findall(r"<reasoning>(.*?)</reasoning>", output, re.DOTALL | re.IGNORECASE)
        # Having reasoning is good, but not having it doesn't make output invalid
        # Just log it for potential future use

        # If invalid, use fallback
        if not act_str:
            act_str = output.lower()[-30:]  # Fallback to last 30 chars

        actions_out.append(act_str)
        valids.append(valid)
        beliefs.append(belief_text if valid else "")

    return actions_out, valids, beliefs, action_available
