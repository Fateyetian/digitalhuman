#!/usr/bin/env python3
"""
Generate WebShop ReBel Golden Trajectories using Hindsight Annotation

This implements the "Hindsight" annotation method for WebShop where a Teacher LLM
reverse-engineers the belief state based on:
1. Current Observation
2. Previous Belief State (for consistency)
3. Ground Truth Expert Action (as a hint)

Mirrors: scripts_archive/generate_rebel_hindsight.py (ALFWorld version)
"""

import os
import sys
import json
import re
import argparse
import random
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm
from openai import OpenAI
from datasets import Dataset

sys.path.insert(0, '/root/testttt/RLVMR/code')

from agent_system.environments.prompts.webshop_rebel_prompts import (
    WEBSHOP_REBEL_TAGGING_TEMPLATE,
    WEBSHOP_REBEL_TEMPLATE_NO_HIS_CS,
    WEBSHOP_REBEL_TEMPLATE_CS
)


# ============================================================================
# Part 1: State Management
# ============================================================================

def initialize_belief_state() -> Dict[str, Any]:
    """Initialize empty WebShop belief state"""
    return {
        "product_understanding": {
            "target_attributes": {},
            "current_product_match": "none",
            "price_constraint": "any"
        },
        "attribute_verification": {
            "verified": [],
            "unverified": [],
            "inferred_only": []
        },
        "search_progress": {
            "search_status": "not_started",
            "evidence": "",
            "updated_subgoal": "Start searching for the target product"
        },
        "exploration_state": {
            "queries_tried": [],
            "products_viewed": [],
            "options_selected": [],
            "tabs_clicked": []
        }
    }


def merge_belief_update(
    global_belief: Dict[str, Any],
    belief_update: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Merge a belief update into the global belief state.
    Accumulates exploration state lists and updates other fields.
    Handles attribute_verification: promotes verified, deduplicates, removes from unverified/inferred_only.
    """
    merged = json.loads(json.dumps(global_belief))  # Deep copy

    # Merge product understanding
    if "product_understanding" in belief_update:
        pu = belief_update["product_understanding"]
        if isinstance(pu, dict):
            if "target_attributes" in pu and isinstance(pu["target_attributes"], dict):
                merged["product_understanding"]["target_attributes"].update(pu["target_attributes"])
            if "current_product_match" in pu:
                merged["product_understanding"]["current_product_match"] = pu["current_product_match"]
            if "price_constraint" in pu:
                merged["product_understanding"]["price_constraint"] = pu["price_constraint"]

    # Merge attribute_verification
    if "attribute_verification" in belief_update:
        av = belief_update["attribute_verification"]
        if isinstance(av, dict):
            # Ensure merged has attribute_verification
            if "attribute_verification" not in merged:
                merged["attribute_verification"] = {"verified": [], "unverified": [], "inferred_only": []}

            mav = merged["attribute_verification"]

            # Merge verified: deduplicate by attribute name
            new_verified = av.get("verified", [])
            if isinstance(new_verified, list):
                existing_attrs = {v.get("attribute", "") for v in mav["verified"] if isinstance(v, dict)}
                for item in new_verified:
                    if isinstance(item, dict):
                        attr_name = item.get("attribute", "")
                        if attr_name and attr_name not in existing_attrs:
                            mav["verified"].append(item)
                            existing_attrs.add(attr_name)

            # Build set of verified attribute names for removal from other lists
            verified_attrs = {v.get("attribute", "") for v in mav["verified"] if isinstance(v, dict)}

            # Merge unverified: add new, remove any that are now verified
            new_unverified = av.get("unverified", [])
            if isinstance(new_unverified, list):
                for item in new_unverified:
                    if isinstance(item, str) and item not in mav["unverified"]:
                        mav["unverified"].append(item)
            mav["unverified"] = [u for u in mav["unverified"] if u not in verified_attrs]

            # Merge inferred_only: add new, remove any that are now verified
            new_inferred = av.get("inferred_only", [])
            if isinstance(new_inferred, list):
                existing_inferred_attrs = {
                    io.get("attribute", "") for io in mav["inferred_only"] if isinstance(io, dict)
                }
                for item in new_inferred:
                    if isinstance(item, dict):
                        attr_name = item.get("attribute", "")
                        if attr_name and attr_name not in existing_inferred_attrs:
                            mav["inferred_only"].append(item)
                            existing_inferred_attrs.add(attr_name)
            mav["inferred_only"] = [
                io for io in mav["inferred_only"]
                if isinstance(io, dict) and io.get("attribute", "") not in verified_attrs
            ]

    # Merge search progress
    if "search_progress" in belief_update:
        sp = belief_update["search_progress"]
        if isinstance(sp, dict):
            if "search_status" in sp:
                merged["search_progress"]["search_status"] = sp["search_status"]
            if "evidence" in sp:
                merged["search_progress"]["evidence"] = sp["evidence"]
            if "updated_subgoal" in sp:
                merged["search_progress"]["updated_subgoal"] = sp["updated_subgoal"]

    # Merge exploration state (accumulate lists)
    if "exploration_state" in belief_update:
        es = belief_update["exploration_state"]
        if isinstance(es, dict):
            for key in ["queries_tried", "products_viewed", "options_selected", "tabs_clicked"]:
                if key in es and isinstance(es[key], list):
                    if key not in merged["exploration_state"]:
                        merged["exploration_state"][key] = []
                    for item in es[key]:
                        if item and item not in merged["exploration_state"][key]:
                            merged["exploration_state"][key].append(item)

    return merged


# ============================================================================
# Part 2: Teacher LLM Interaction
# ============================================================================

# WebShop-specific failure observation patterns (equivalent to ALFWorld "Nothing happens")
_WEBSHOP_FAILURE_PATTERNS = [
    "sorry, nothing was found",
    "no products were found",
    "no results found",
    "no matching",
    "could not find",
    "0 results",
    "no items found",
]


def _is_failure_observation(obs: str) -> bool:
    """Detect WebShop failure observations: search returned no results, invalid action, etc."""
    obs_lower = obs.lower()
    return any(pat in obs_lower for pat in _WEBSHOP_FAILURE_PATTERNS)


def construct_annotation_prompt(
    task_desc: str,
    current_obs: str,
    prev_belief: Dict[str, Any],
    ground_truth_action: str,
    admissible_actions: List[str] = None,
    action_history: List[str] = None,
    obs_history: List[str] = None,
    is_obs_failure: bool = False
) -> Tuple[str, str]:
    """Construct the hindsight annotation prompt for Teacher LLM."""
    prev_belief_json = json.dumps(prev_belief, indent=2, ensure_ascii=False)
    admissible_str = ", ".join(admissible_actions[:20]) if admissible_actions else "N/A"
    obs_display = current_obs if len(current_obs) < 800 else current_obs[:800] + "..."

    # Prepend failure note so Teacher LLM correctly interprets the observation context
    if is_obs_failure:
        obs_display = "[NOTE: This observation indicates a search failure — no matching products were returned. The search_status must remain 'searching'.]\n" + obs_display

    # Build trajectory summary: (obs snippet → action) pairs for richer context
    action_history_str = ""
    if action_history:
        if obs_history and len(obs_history) == len(action_history):
            lines = []
            for i, (o, a) in enumerate(zip(obs_history, action_history)):
                obs_snippet = o[:120] + "..." if len(o) > 120 else o
                lines.append(f"  Step {i+1}: obs='{obs_snippet}' → action='{a}'")
            action_history_str = "\n".join(lines)
        else:
            action_history_str = "\n".join(
                f"  Step {i+1}: {a}" for i, a in enumerate(action_history)
            )

    # Brief system prompt — detailed instructions are already in the template body
    system_prompt = "You are an expert e-commerce shopping agent annotator. Output valid JSON only — no extra commentary outside the JSON block."

    user_prompt = WEBSHOP_REBEL_TAGGING_TEMPLATE.format(
        task_description=task_desc,
        current_observation=obs_display,
        prev_belief_json=prev_belief_json,
        admissible_actions=admissible_str,
        ground_truth_action=ground_truth_action,
        traj=action_history_str
    )

    return system_prompt, user_prompt


def call_teacher_llm(
    system_prompt: str,
    user_prompt: str,
    client: OpenAI,
    model_name: str,
    temperature: float = 0.3,
    max_retries: int = 3
) -> Optional[Dict[str, Any]]:
    """Call Teacher LLM and parse the response."""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                max_tokens=2500
            )

            content = response.choices[0].message.content.strip()

            # Extract JSON from markdown code block if present
            if "```json" in content:
                json_match = re.search(r'```json\s*(\{.*\})\s*```', content, re.DOTALL)
                if json_match:
                    content = json_match.group(1)
            elif "```" in content:
                json_match = re.search(r'```\s*(\{.*\})\s*```', content, re.DOTALL)
                if json_match:
                    content = json_match.group(1)

            # Also try to find the outermost JSON object if no code block
            if not content.startswith('{'):
                brace_match = re.search(r'(\{.*\})', content, re.DOTALL)
                if brace_match:
                    content = brace_match.group(1)

            parsed = json.loads(content)

            required_fields = ["product_understanding", "attribute_verification",
                             "search_progress", "exploration_state", "reasoning"]
            if all(field in parsed for field in required_fields):
                # Safety net: if ready_to_buy but inferred_only is non-empty, downgrade
                av = parsed.get("attribute_verification", {})
                if isinstance(av, dict):
                    inferred_only = av.get("inferred_only", [])
                    if isinstance(inferred_only, list) and len(inferred_only) > 0:
                        sp = parsed.get("search_progress", {})
                        if isinstance(sp, dict) and sp.get("search_status") == "ready_to_buy":
                            sp["search_status"] = "options_selecting"
                            parsed["search_progress"] = sp
                            # Also enforce partial match
                            pu = parsed.get("product_understanding", {})
                            if isinstance(pu, dict) and pu.get("current_product_match") == "exact":
                                pu["current_product_match"] = "partial"
                                parsed["product_understanding"] = pu
                return parsed
            else:
                print(f"   Missing required fields (attempt {attempt + 1}/{max_retries})")
                continue

        except json.JSONDecodeError as e:
            print(f"   JSON parse error (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                print(f"   Raw response: {content[:300]}")
        except Exception as e:
            print(f"   LLM call failed (attempt {attempt + 1}/{max_retries}): {e}")

    return None


def create_fallback_annotation(
    obs: str,
    action: str,
    task: str,
    current_belief: Dict[str, Any] = None
) -> Dict[str, Any]:
    """Create a fallback annotation if LLM fails. Inherits from current belief if available."""
    if current_belief and isinstance(current_belief, dict):
        # Carry forward existing belief state instead of resetting
        import copy
        result = copy.deepcopy(current_belief)
        # Ensure all required keys exist
        if "attribute_verification" not in result:
            result["attribute_verification"] = {"verified": [], "unverified": [], "inferred_only": []}
        if "exploration_state" not in result:
            result["exploration_state"] = {"queries_tried": [], "products_viewed": [], "options_selected": [], "tabs_clicked": []}
        elif "tabs_clicked" not in result["exploration_state"]:
            result["exploration_state"]["tabs_clicked"] = []
        result["reasoning"] = f"Executing action: {action}"
        return result

    return {
        "product_understanding": {
            "target_attributes": {},
            "current_product_match": "none",
            "price_constraint": "any"
        },
        "attribute_verification": {
            "verified": [],
            "unverified": [],
            "inferred_only": []
        },
        "search_progress": {
            "search_status": "searching",
            "evidence": obs[:100] + "...",
            "updated_subgoal": task
        },
        "exploration_state": {
            "queries_tried": [],
            "products_viewed": [],
            "options_selected": [],
            "tabs_clicked": []
        },
        "reasoning": f"Executing action: {action}"
    }


# ============================================================================
# Part 3: Trajectory Parsing
# ============================================================================

def _extract_action_from_react(text: str) -> str:
    """Extract the pure action (search[...] or click[...]) from a ReAct-format response."""
    # Match "Action:\n search[...]" or "Action: click[...]"
    action_match = re.search(r'Action:\s*\n?\s*((?:search|click)\[.+?\])', text, re.IGNORECASE)
    if action_match:
        return action_match.group(1).strip()
    # Fallback: find any search[...] or click[...] in the text
    fallback = re.search(r'((?:search|click)\[.+?\])', text, re.IGNORECASE)
    if fallback:
        return fallback.group(1).strip()
    return text.strip()


def _extract_task_from_conversations(conversations: List[Dict]) -> str:
    """Extract the shopping task instruction from conversation observations."""
    for turn in conversations:
        if turn.get('from') != 'human':
            continue
        value = turn.get('value', '')
        # AgentTraj-L format: "Instruction: [SEP] <task> [SEP]"
        instr_match = re.search(r'Instruction:\s*\[SEP\]\s*(.+?)\s*\[SEP\]', value)
        if instr_match:
            return instr_match.group(1).strip()
        # Alternative: "Instruction: <task>\n"
        instr_match2 = re.search(r'Instruction:\s*(.+?)(?:\n|$)', value)
        if instr_match2:
            return instr_match2.group(1).strip()
    return 'Complete the shopping task'


def parse_webshop_expert_trajectory(
    trajectory: Dict[str, Any]
) -> Tuple[str, List[Tuple[str, str, List[str]]]]:
    """
    Parse a WebShop expert trajectory into task description and
    (observation, action, available_actions) tuples.

    Supports formats:
    - List of dicts with 'observation', 'action', 'available_actions' keys
    - Conversation format with 'from' and 'value' keys (AgentTraj-L)
    """
    pairs = []

    if 'steps' in trajectory:
        # Structured format
        task = trajectory.get('task', trajectory.get('goal', 'Complete the shopping task'))
        for step in trajectory['steps']:
            obs = step.get('observation', '')
            action = step.get('action', '')
            avail = step.get('available_actions', [])
            if obs and action:
                pairs.append((obs, action, avail))
    elif 'conversations' in trajectory:
        # AgentTraj-L conversation format
        conversations = trajectory['conversations']
        task = trajectory.get('task', trajectory.get('goal',
            _extract_task_from_conversations(conversations)))

        current_obs = None
        for turn in conversations:
            if turn.get('from') == 'human':
                current_obs = turn.get('value', '')
            elif turn.get('from') == 'gpt' and turn.get('loss') is True and current_obs:
                # Extract pure action from ReAct "Thought: ... Action: ..." format
                raw_text = turn.get('value', '').strip()
                action = _extract_action_from_react(raw_text)
                if action:
                    pairs.append((current_obs, action, []))
                current_obs = None
    else:
        task = trajectory.get('task', trajectory.get('goal', 'Complete the shopping task'))

    return task, pairs


# ============================================================================
# Part 4: Main Hindsight Annotation Algorithm
# ============================================================================

def generate_webshop_rebel_dataset(
    expert_trajectory: Dict[str, Any],
    client: OpenAI,
    model_name: str
) -> Optional[Dict[str, Any]]:
    """
    Generate WebShop ReBel dataset using hindsight annotation.

    For each (obs, action) pair:
    1. Call Teacher LLM with full prev_belief + obs + ground_truth_action
    2. Parse belief update and reasoning
    3. Merge update into global belief
    4. Format as training sample
    """
    task_description, pairs = parse_webshop_expert_trajectory(expert_trajectory)

    if not pairs:
        print("   No valid (obs, action) tuples found")
        return None

    global_belief = initialize_belief_state()
    rebel_conversations = []
    action_history = []  # Track all actions taken so far
    obs_history = []     # Track observations corresponding to each action

    rebel_conversations.append({
        'from': 'human',
        'loss': False,
        'value': 'You are a shopping agent in WebShop. Use ReBel format with belief state, reasoning, and action.'
    })
    rebel_conversations.append({
        'from': 'gpt',
        'loss': False,
        'value': "OK. I will track my belief state and reasoning to shop efficiently."
    })

    # Navigation/non-option click targets
    NAV_TARGETS = {'buy now', 'back to search', '< prev', 'next >', 'description',
                   'features', 'reviews', 'search'}
    TAB_TARGETS = {'description', 'features', 'reviews'}

    success_count = 0

    for step, (obs, action, avail_actions) in enumerate(pairs):
        input_belief_json = json.dumps(global_belief, indent=2, ensure_ascii=False)

        # Detect failure observations (no results, invalid action, etc.)
        is_obs_failure = _is_failure_observation(obs)
        if is_obs_failure:
            print(f"   Step {step}: Failure observation detected — search returned no results")

        system_prompt, user_prompt = construct_annotation_prompt(
            task_desc=task_description,
            current_obs=obs,
            prev_belief=global_belief,
            ground_truth_action=action,
            admissible_actions=avail_actions,
            action_history=action_history,
            obs_history=obs_history,
            is_obs_failure=is_obs_failure
        )

        llm_output = call_teacher_llm(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            client=client,
            model_name=model_name
        )

        if llm_output is None:
            print(f"   Step {step}: LLM annotation failed, using fallback")
            llm_output = create_fallback_annotation(obs, action, task_description, current_belief=global_belief)
        else:
            success_count += 1

        belief_update = {
            "product_understanding": llm_output.get("product_understanding", {}),
            "attribute_verification": llm_output.get("attribute_verification", {}),
            "search_progress": llm_output.get("search_progress", {}),
            "exploration_state": llm_output.get("exploration_state", {})
        }
        reasoning = llm_output.get("reasoning", "")

        # Auto-track search queries from action
        search_match = re.match(r'search\[(.+)\]', action.lower())
        if search_match:
            query = search_match.group(1).strip()
            if "exploration_state" not in belief_update:
                belief_update["exploration_state"] = {}
            if "queries_tried" not in belief_update["exploration_state"]:
                belief_update["exploration_state"]["queries_tried"] = []
            if query not in belief_update["exploration_state"]["queries_tried"]:
                belief_update["exploration_state"]["queries_tried"].append(query)

        # Auto-track click actions (options, tabs, products)
        click_match = re.match(r'click\[(.+)\]', action, re.IGNORECASE)
        if click_match:
            clicked = click_match.group(1).strip()
            clicked_lower = clicked.lower()
            if "exploration_state" not in belief_update:
                belief_update["exploration_state"] = {}

            if clicked_lower in TAB_TARGETS:
                # Tab click: track in tabs_clicked
                if "tabs_clicked" not in belief_update["exploration_state"]:
                    belief_update["exploration_state"]["tabs_clicked"] = []
                if clicked not in belief_update["exploration_state"]["tabs_clicked"]:
                    belief_update["exploration_state"]["tabs_clicked"].append(clicked)
            elif re.match(r'^b\d+', clicked_lower):
                # Product ASIN click: track in products_viewed
                if "products_viewed" not in belief_update["exploration_state"]:
                    belief_update["exploration_state"]["products_viewed"] = []
                if clicked not in belief_update["exploration_state"]["products_viewed"]:
                    belief_update["exploration_state"]["products_viewed"].append(clicked)
            elif clicked_lower not in NAV_TARGETS:
                # Option selection (size, color, etc.): track in options_selected
                if "options_selected" not in belief_update["exploration_state"]:
                    belief_update["exploration_state"]["options_selected"] = []
                if clicked not in belief_update["exploration_state"]["options_selected"]:
                    belief_update["exploration_state"]["options_selected"].append(clicked)

        global_belief = merge_belief_update(global_belief, belief_update)

        # Record obs and action in history for subsequent steps
        obs_history.append(obs)
        action_history.append(action)

        admissible_str = ", ".join(avail_actions) if avail_actions else "N/A"
        human_turn_value = f"""Task: {task_description}

Observation:
{obs}

Current Belief State:
{input_belief_json}

Available Actions: {admissible_str}"""

        rebel_conversations.append({
            'from': 'human',
            'loss': False,
            'value': human_turn_value
        })

        belief_update_json = json.dumps(belief_update, indent=2, ensure_ascii=False)
        rebel_output = f"""<belief>
{belief_update_json}
</belief>

<reasoning>
{reasoning}
</reasoning>

<action>
{action}
</action>"""

        rebel_conversations.append({
            'from': 'gpt',
            'loss': True,
            'value': rebel_output
        })

    item_id = expert_trajectory.get('item_id', expert_trajectory.get('id', 'unknown'))

    result = {
        'conversations': rebel_conversations,
        'item_id': f"{item_id}_rebel_hindsight",
        'num_steps': len(pairs),
        'annotation_success_rate': success_count / len(pairs) if pairs else 0,
        'task': task_description
    }

    return result


# ============================================================================
# Part 5: Cold-Start Format Conversion
# ============================================================================

def convert_to_coldstart_format(rebel_trajectory: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Convert ReBel trajectory to cold-start SFT format (parquet-compatible).

    Returns list of {question, answer} dicts for each step.
    Cold-start format DOES NOT include admissible actions.

    Mirrors ALFWorld's convert_to_coldstart_format strategy:
    - Step 1:    Use WEBSHOP_REBEL_TEMPLATE_NO_HIS_CS (full instructional template,
                 no prior belief, no admissible actions) — teaches format from scratch.
    - Steps 2+:  Strip "Available Actions:" from the annotated human turn
                 (compact format: task + obs + current_belief, no admissible actions).
    """
    conversations = rebel_trajectory.get('conversations', [])
    task = rebel_trajectory.get('task', 'Unknown task')
    coldstart_pairs = []

    step_num = 0
    for i in range(len(conversations)):
        turn = conversations[i]
        # Only process model output turns that compute loss (skip system messages and human turns)
        if not (turn['from'] == 'gpt' and turn.get('loss') is True and '<belief>' in turn.get('value', '')):
            continue

        step_num += 1
        # Get the preceding human turn
        human_turn = conversations[i - 1] if i > 0 else None
        if not human_turn or human_turn['from'] != 'human':
            continue

        human_text = human_turn['value']

        if step_num == 1:
            # First step: use the full instructional template.
            # Extract the raw observation from the human turn.
            obs_match = re.search(
                r'Observation:\n(.*?)(?:\n\nCurrent Belief State:|\Z)',
                human_text, re.DOTALL
            )
            if obs_match:
                obs = obs_match.group(1).strip()
            else:
                print(f"   [convert_to_coldstart] WARNING: Could not extract obs from step 1 human turn; using full text as fallback")
                obs = human_text
            prompt = WEBSHOP_REBEL_TEMPLATE_NO_HIS_CS.format(
                task_description=task,
                current_observation=obs
            )
        else:
            # Subsequent steps: strip "Available Actions:" from annotated human turn.
            # Keeps: task description + observation + current belief state.
            lines = human_text.split('\n')
            filtered_lines = [line for line in lines if 'Available Actions:' not in line]
            prompt = '\n'.join(filtered_lines).strip()

        coldstart_pairs.append({
            'question': prompt,
            'answer': turn['value']
        })

    return coldstart_pairs


# ============================================================================
# Part 6: Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate WebShop ReBel Golden Trajectories using Hindsight Annotation"
    )
    parser.add_argument('--expert_data', type=str,
                       default='data/webshop_expert_traj.json',
                       help='Path to expert trajectory data (JSON or JSONL)')
    parser.add_argument('--output_dir', type=str,
                       default='data/webshop_rebel_hindsight',
                       help='Output directory for ReBel dataset')
    parser.add_argument('--num_samples', type=int, default=5,
                       help='Number of trajectories to process')
    parser.add_argument('--teacher_model_url', type=str,
                       default='http://127.0.0.1:8000/v1',
                       help='Teacher LLM API endpoint (OpenAI-compatible)')
    parser.add_argument('--teacher_model_name', type=str,
                       default='gpt-4',
                       help='Teacher model name')
    parser.add_argument('--api_key', type=str,
                       default=None,
                       help='API key for Teacher LLM (overrides OPENAI_API_KEY env var)')
    parser.add_argument('--temperature', type=float, default=0.3,
                       help='Temperature for Teacher LLM')
    parser.add_argument('--random_sample', action='store_true',
                       help='Randomly sample trajectories')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--save_interval', type=int, default=10,
                       help='Save progress every N trajectories')
    parser.add_argument('--generate_coldstart', action='store_true',
                       help='Also generate cold-start parquet files')

    args = parser.parse_args()

    print("=" * 80)
    print("WebShop ReBel Hindsight Annotation System")
    print("=" * 80)
    print(f"Expert data: {args.expert_data}")
    print(f"Teacher model: {args.teacher_model_name}")
    print(f"Processing: {args.num_samples} samples")
    print("=" * 80)

    # Step 1: Load expert data
    print("\n[1/4] Loading expert trajectories...")
    if args.expert_data.endswith('.json') or args.expert_data.endswith('.jsonl'):
        with open(args.expert_data, 'r') as f:
            content = f.read().strip()
            if content.startswith('['):
                expert_data = json.loads(content)
            else:
                expert_data = [json.loads(line) for line in content.splitlines() if line.strip()]
    else:
        from datasets import load_from_disk
        expert_ds = load_from_disk(args.expert_data)
        expert_data = list(expert_ds)

    if args.random_sample:
        random.seed(args.seed)
        random.shuffle(expert_data)
        print(f"Randomly shuffled data (seed={args.seed})")

    expert_data = expert_data[:args.num_samples]
    print(f"Loaded {len(expert_data)} expert trajectories")

    # Step 2: Connect to Teacher LLM
    print("\n[2/4] Connecting to Teacher LLM...")
    try:
        api_key = args.api_key or os.getenv("OPENAI_API_KEY", "EMPTY")
        client = OpenAI(api_key=api_key, base_url=args.teacher_model_url)
        # Quick validation: send a tiny request instead of models.list()
        test_resp = client.chat.completions.create(
            model=args.teacher_model_name,
            messages=[{"role": "user", "content": "Say OK"}],
            max_tokens=5
        )
        print(f"Connected to {args.teacher_model_url} (model: {args.teacher_model_name})")
    except Exception as e:
        print(f"Failed to connect to Teacher LLM: {e}")
        print("   Please check your API key and endpoint")
        return

    # Step 3: Generate ReBel trajectories
    print("\n[3/4] Generating ReBel trajectories with hindsight annotation...")
    os.makedirs(args.output_dir, exist_ok=True)

    rebel_trajectories = []
    all_coldstart_pairs = []
    total_success_steps = 0
    total_steps = 0

    for idx, expert_traj in enumerate(tqdm(expert_data, desc="Annotating", ncols=100)):
        current_num = idx + 1
        print(f"\n[{current_num}/{len(expert_data)}] Processing trajectory...")

        try:
            rebel_traj = generate_webshop_rebel_dataset(
                expert_trajectory=expert_traj,
                client=client,
                model_name=args.teacher_model_name
            )

            if rebel_traj:
                success_rate = rebel_traj['annotation_success_rate']
                num_steps = rebel_traj['num_steps']
                total_steps += num_steps
                total_success_steps += int(num_steps * success_rate)
                print(f"   Generated {num_steps} steps | Success: {success_rate:.1%}")
                rebel_trajectories.append(rebel_traj)

                if args.generate_coldstart:
                    cs_pairs = convert_to_coldstart_format(rebel_traj)
                    all_coldstart_pairs.extend(cs_pairs)
            else:
                print(f"   Failed to generate trajectory")

        except Exception as e:
            print(f"   Error: {e}")
            import traceback
            traceback.print_exc()

        # Real-time save
        if current_num % args.save_interval == 0 or current_num == len(expert_data):
            print(f"\nSaving progress ({current_num}/{len(expert_data)})...")
            jsonl_path = os.path.join(args.output_dir, 'rebel_hindsight.jsonl')
            with open(jsonl_path, 'w') as f:
                for traj in rebel_trajectories:
                    f.write(json.dumps(traj, ensure_ascii=False) + '\n')

            if args.generate_coldstart and all_coldstart_pairs:
                import pyarrow as pa
                import pyarrow.parquet as pq
                table = pa.table({
                    'extra_info': [json.dumps({'question': p['question'], 'answer': p['answer']})
                                  for p in all_coldstart_pairs]
                })
                pq.write_table(table, os.path.join(args.output_dir, 'train.parquet'))
                print(f"   Saved {len(all_coldstart_pairs)} cold-start pairs")

    # Step 4: Final save and statistics
    print(f"\n[4/4] Finalizing results...")

    if rebel_trajectories:
        jsonl_path = os.path.join(args.output_dir, 'rebel_hindsight.jsonl')
        print(f"Saved {len(rebel_trajectories)} trajectories to {args.output_dir}")

        avg_steps = sum(t['num_steps'] for t in rebel_trajectories) / len(rebel_trajectories)
        avg_success = sum(t['annotation_success_rate'] for t in rebel_trajectories) / len(rebel_trajectories)

        print(f"\nStatistics:")
        print(f"   Total trajectories: {len(rebel_trajectories)}")
        print(f"   Avg steps per trajectory: {avg_steps:.1f}")
        print(f"   Avg annotation success rate: {avg_success:.1%}")

    print("\n" + "=" * 80)
    print("Hindsight annotation complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
