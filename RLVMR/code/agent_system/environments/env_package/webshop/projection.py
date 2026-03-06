from typing import List, Tuple
import re
import json


def webshop_projection(actions: List[str]):
    """
    A function to process the actions.
    actions: the list of actions to be processed, it is a list of strings.
    Expected format:
        <think>some reasoning...</think><action>up/down/left/right/still</action>
    """

    valids = [0] * len(actions)

    for i in range(len(actions)):
        original_str = actions[i]  # keep the original string
        actions[i] = actions[i].lower()

        # Attempt to extract the substring within <action>...</action>
        start_tag = "<action>"
        end_tag = "</action>"
        start_idx = actions[i].find(start_tag)
        end_idx = actions[i].find(end_tag)
        try:
            if start_idx == -1 or end_idx == -1:
                # If we can't find a valid <action>...</action> block, mark as invalid
                actions[i] = actions[i][-20:]  # 0 is invalid action for Sokoban
                continue

            # Extract just the content between the tags
            extracted_action = actions[i][start_idx + len(start_tag):end_idx].strip().lower()
            
            actions[i] = extracted_action
            valids[i] = 1

        except:
            # randomly choose an action from the action list if illegal
            actions[i] = actions[i][-20:]

        # check <think>...</think>
        think_start_idx = original_str.find("<think>")
        think_end_idx = original_str.find("</think>")
        if think_start_idx == -1 or think_end_idx == -1:
            valids[i] = 0

        # check if contains any Chinese characters
        if re.search(r'[\u4e00-\u9fff]', original_str):
            valids[i] = 0

    return actions, valids


def webshop_projection_rebel(actions: List[str], action_infos: List[dict]) -> Tuple[List[str], List[int], List[str], List[bool]]:
    """
    ReBel projection function for <belief>/<reasoning>/<action> format in WebShop.

    Validates:
    1. Exactly one <belief>...</belief> block containing valid JSON with required keys
    2. Optional <reasoning>...</reasoning> block
    3. Exactly one <action>...</action> block
    4. <belief> appears before <action>
    5. No Chinese characters
    6. WebShop-specific: action is search[...] or click[...] format

    Soft-validation philosophy (training-friendly):
    - A well-formed action without a valid belief is still "format-valid"
      so the environment can execute it. The belief reward calculator will
      simply return 0 intrinsic reward for the missing belief.
    - "ready_to_buy + inferred_only" is an inconsistency flag, NOT a hard
      rejection. The action is still executed; the reward calculator applies
      a transition penalty instead.

    Args:
        actions: List of model outputs
        action_infos: List of available_actions dicts for each environment
            Each dict has 'has_search_bar' (bool) and 'clickables' (list)

    Returns:
        actions_out: Extracted actions
        valids: Binary validity flags (1=valid, 0=invalid)
        beliefs: Extracted belief text (for reward calculation)
        action_available: Whether extracted action is in admissible set
    """
    actions_out = []
    valids = []
    beliefs = []
    action_available = [False] * len(actions)

    for i, output in enumerate(actions):
        valid = 1
        act_str = ""
        belief_text = ""
        belief_ok = True  # track belief quality separately

        # Check for Chinese characters
        if re.search(r'[\u4e00-\u9fff]', output):
            valid = 0

        # ---- Parse belief block ----
        belief_matches = re.findall(r"<belief>(.*?)</belief>", output, re.DOTALL | re.IGNORECASE)
        if len(belief_matches) != 1:
            belief_ok = False
        else:
            belief_text = belief_matches[0].strip()

            # Validate belief structure: try to parse as JSON
            try:
                belief_json = belief_text.replace("'", '"')
                belief_data = json.loads(belief_json)

                has_product = 'product_understanding' in belief_data
                has_progress = 'search_progress' in belief_data
                has_exploration = 'exploration_state' in belief_data

                if not (has_product or has_progress or has_exploration):
                    belief_ok = False

                # NOTE: "ready_to_buy with inferred_only" is now a SOFT
                # inconsistency – the reward calculator already applies a
                # transition_penalty for this case (env_manager.py).
                # We no longer hard-reject the entire output here.

            except json.JSONDecodeError:
                belief_ok = False

            # Check if belief is empty
            if not belief_text or len(belief_text) < 10:
                belief_ok = False

        # ---- Parse action block ----
        action_matches = re.findall(r"<action>([\s\S]*?)</action>", output, re.IGNORECASE)
        if len(action_matches) != 1:
            valid = 0
        else:
            act_str = action_matches[0].strip().lower()

            # Validate WebShop action format: search[...] or click[...]
            is_valid_format = bool(
                re.match(r'^search\[.+\]$', act_str) or
                re.match(r'^click\[.+\]$', act_str)
            )
            if not is_valid_format:
                valid = 0

            # Check if action is in admissible set
            if i < len(action_infos) and action_infos[i]:
                avail = action_infos[i]
                clickables = avail.get('clickables', [])
                has_search = avail.get('has_search_bar', False)

                search_match = re.match(r'^search\[(.+)\]$', act_str)
                click_match = re.match(r'^click\[(.+)\]$', act_str)

                if search_match and has_search:
                    action_available[i] = True
                elif click_match:
                    clicked_val = click_match.group(1).strip()
                    # Check if the clicked value is in clickables (case-insensitive)
                    clickables_lower = [c.lower().strip() for c in clickables]
                    if clicked_val.lower().strip() in clickables_lower:
                        action_available[i] = True

        # Check that <belief> appears before <action> (only if both exist)
        belief_pos = output.lower().find("<belief>")
        action_pos = output.lower().find("<action>")
        if belief_pos != -1 and action_pos != -1 and belief_pos > action_pos:
            belief_ok = False  # wrong order → belief is invalid but action still ok
        if action_pos == -1:
            valid = 0

        # If belief is bad but action is well-formed, still mark valid=1
        # so the action executes. The belief reward will simply be 0.
        if not belief_ok:
            belief_text = ""

        # If invalid, use fallback
        if not act_str:
            act_str = output.lower()[-30:]

        actions_out.append(act_str)
        valids.append(valid)
        beliefs.append(belief_text)

    return actions_out, valids, beliefs, action_available