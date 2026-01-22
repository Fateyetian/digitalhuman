#!/usr/bin/env python3
"""
Generate ReBel Golden Trajectories using Hindsight Annotation

This implements the "Hindsight" annotation method where a Teacher LLM
reverse-engineers the belief state based on:
1. Current Observation
2. Previous Belief State (for consistency)
3. Ground Truth Expert Action (as a hint)

The key innovation: We ask "what belief would justify this action?"
instead of "what action should we take?"
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
from datasets import load_from_disk, Dataset

sys.path.insert(0, '/root/testttt/RLVMR/code')

# Import ReBel prompts
from agent_system.environments.prompts.rebel_prompts import (
    ALFWORLD_REBEL_TAGGING_TEMPLATE,
    ALFWORLD_REBEL_TEMPLATE_NO_HIS_CS,
    ALFWORLD_REBEL_TEMPLATE_CS
)


# ============================================================================
# Part 1: Observation Parsing and Object Extraction
# ============================================================================

def parse_object_list_from_observation(obs: str) -> Dict[str, str]:
    """
    Parse object listings from ALFWorld observations.

    Example input: "On the sidetable 1, you see a alarmclock 2, a alarmclock 1, and a cd 1."
    Returns: {"alarmclock 2": "sidetable 1", "alarmclock 1": "sidetable 1", "cd 1": "sidetable 1"}
    """
    objects = {}

    # Pattern 1: "On the X, you see a Y, a Z, and a W"
    match = re.search(r'(?:On|In) the (\w+\s+\d+), you see (.+?)\.', obs)
    if match:
        location = match.group(1)
        items_str = match.group(2)

        # Split by comma and "and"
        items = re.split(r',\s*(?:and\s+)?|and\s+', items_str)
        for item in items:
            # Extract "a/an object_name number"
            obj_match = re.search(r'a(?:n)?\s+([\w\s]+\d+)', item.strip())
            if obj_match:
                obj_name = obj_match.group(1).strip()
                objects[obj_name] = location

    # Pattern 2: "The X is open/closed. In it, you see ..."
    match = re.search(r'In it, you see (.+?)\.', obs)
    if match:
        items_str = match.group(1)
        # Try to find the container from previous sentence
        container_match = re.search(r'The (\w+\s+\d+) is', obs)
        if container_match:
            location = container_match.group(1)
            items = re.split(r',\s*(?:and\s+)?|and\s+', items_str)
            for item in items:
                obj_match = re.search(r'a(?:n)?\s+([\w\s]+\d+)', item.strip())
                if obj_match:
                    obj_name = obj_match.group(1).strip()
                    objects[obj_name] = location

    return objects


def extract_current_location(obs: str) -> Optional[str]:
    """Extract current location from observation"""
    # Pattern: "You arrive at loc X. ..."
    match = re.search(r'You arrive at (?:loc )?(\w+\s+\d+)', obs)
    if match:
        return match.group(1)
    return None


def extract_action_components(action: str) -> Tuple[str, Optional[str]]:
    """
    Extract action type and target from action string.

    Returns: (action_type, target)
    Example: "go to shelf 1" -> ("go to", "shelf 1")
    """
    action = action.strip()

    # Common action patterns
    patterns = [
        (r'^go to (.+)$', 'go to'),
        (r'^open (.+)$', 'open'),
        (r'^close (.+)$', 'close'),
        (r'^take (.+) from (.+)$', 'take'),
        (r'^put (.+) in/on (.+)$', 'put'),
        (r'^examine (.+)$', 'examine'),
        (r'^use (.+)$', 'use'),
    ]

    for pattern, action_type in patterns:
        match = re.match(pattern, action)
        if match:
            target = match.group(1)
            return (action_type, target)

    return (action, None)


# ============================================================================
# Part 2: State Management and Merging
# ============================================================================

def initialize_belief_state() -> Dict[str, Any]:
    """Initialize empty belief state"""
    return {
        "world_model": {
            "found_objects": {},
            "inventory": None,  # Track what agent is holding
            "state_changes": {},
            "cleared_receptacles": []
        },
        "task_state": {
            "status": "in_progress",
            "current_subgoal": "Start task",
            "evidence": ""
        },
        "exploration_map": {
            "visited_locations": [],
            "priority_targets": []
        }
    }


def merge_belief_update(
    global_belief: Dict[str, Any],
    belief_update: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Merge a belief update into the global belief state.

    This implements cumulative state tracking with validation:
    - Add new found objects
    - Update inventory (ALFWorld only allows one item at a time)
    - Update state changes
    - Append to cleared receptacles (no duplicates, with validation)
    - Track visited locations
    """
    merged = json.loads(json.dumps(global_belief))  # Deep copy

    # Merge world model
    if "world_model_update" in belief_update:
        update = belief_update["world_model_update"]

        # Add found objects
        if "found_objects" in update:
            merged["world_model"]["found_objects"].update(update["found_objects"])

        # Update inventory (ALFWorld constraint: only one item at a time)
        if "inventory" in update:
            val = update["inventory"]
            # Handle string representations of null
            if isinstance(val, str) and val.lower().strip() in ["null", "none", ""]:
                merged["world_model"]["inventory"] = None
            elif val is None:
                merged["world_model"]["inventory"] = None
            else:
                # Store as lowercase for consistency
                merged["world_model"]["inventory"] = str(val).lower().strip() if val else None

        # Update state changes
        if "state_changes" in update:
            merged["world_model"]["state_changes"].update(update["state_changes"])

        # Add cleared receptacles (unique, with validation and normalization)
        if "cleared_receptacles" in update:
            for receptacle in update["cleared_receptacles"]:
                if receptacle:
                    # Normalize: lowercase and strip whitespace for consistent comparison
                    normalized = str(receptacle).lower().strip()
                    # Validation: check if receptacle ID looks valid (has a number)
                    # This prevents nonsense entries from polluting the data
                    if re.search(r'\d+', normalized):
                        # Check against normalized list
                        normalized_existing = [r.lower().strip() for r in merged["world_model"]["cleared_receptacles"]]
                        if normalized not in normalized_existing:
                            merged["world_model"]["cleared_receptacles"].append(normalized)

    # Update task state
    if "task_progress_update" in belief_update:
        update = belief_update["task_progress_update"]

        if "subgoal_status" in update:
            merged["task_state"]["status"] = update["subgoal_status"]

        if "updated_subgoal" in update and update["updated_subgoal"]:
            merged["task_state"]["current_subgoal"] = update["updated_subgoal"]

        if "evidence" in update:
            merged["task_state"]["evidence"] = update["evidence"]

    # Update exploration map
    if "exploration_map_update" in belief_update:
        update = belief_update["exploration_map_update"]

        if "newly_visited" in update:
            for loc in update["newly_visited"]:
                if loc and loc not in merged["exploration_map"]["visited_locations"]:
                    merged["exploration_map"]["visited_locations"].append(loc)

        if "next_priority" in update:
            merged["exploration_map"]["priority_targets"] = update["next_priority"]

    return merged


# ============================================================================
# Part 3: Teacher LLM Annotation Prompts
# ============================================================================

def construct_annotation_prompt(
    task_desc: str,
    current_obs: str,
    prev_belief: Dict[str, Any],
    ground_truth_action: str,
    admissible_actions: List[str] = None,
    is_nothing_happens: bool = False
) -> str:
    """
    Construct the hindsight annotation prompt for Teacher LLM.

    This is the KEY innovation: we show the expert action and ask
    the model to reverse-engineer what belief would justify it.
    """

    # Format previous belief for context
    prev_belief_json = json.dumps(prev_belief, indent=2, ensure_ascii=False)

    # Format admissible actions
    admissible_str = ""
    if admissible_actions:
        admissible_str = "\n" + ", ".join(admissible_actions[:20])  # Limit to 20 for brevity

    # Truncate observation if too long
    obs_display = current_obs if len(current_obs) < 800 else current_obs[:800] + "..."

    # Extract current inventory for context
    current_inventory = prev_belief.get("world_model", {}).get("inventory")
    inventory_str = current_inventory if current_inventory else 'Empty (not holding anything)'

    # Use the template from rebel_prompts.py
    system_prompt = """You are an expert in Embodied AI reasoning. Your goal is to simulate the internal 'thought process' of an intelligent agent.
You must maintain a consistent World Model and make logical decisions based ON ONLY what has been observed up to the current moment."""

    user_prompt = ALFWORLD_REBEL_TAGGING_TEMPLATE.format(
        task_description=task_desc,
        current_observation=obs_display,
        prev_belief_json=prev_belief_json,
        current_inventory=inventory_str,
        admissible_actions=admissible_str if admissible_str else "N/A",
        ground_truth_action=ground_truth_action,
        traj=""  # Not used in hindsight mode
    )

    return system_prompt, user_prompt


# ============================================================================
# Part 4: LLM Interaction and Response Parsing
# ============================================================================

def call_teacher_llm(
    system_prompt: str,
    user_prompt: str,
    client: OpenAI,
    model_name: str,
    temperature: float = 0.3,
    max_retries: int = 3
) -> Optional[Dict[str, Any]]:
    """
    Call Teacher LLM and parse the response.

    Returns parsed JSON with belief update and reasoning, or None if failed.
    """
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                max_tokens=800
            )

            content = response.choices[0].message.content.strip()

            # Extract JSON from markdown code block if present
            if "```json" in content:
                json_match = re.search(r'```json\s*(\{.*?\})\s*```', content, re.DOTALL)
                if json_match:
                    content = json_match.group(1)
            elif "```" in content:
                json_match = re.search(r'```\s*(\{.*?\})\s*```', content, re.DOTALL)
                if json_match:
                    content = json_match.group(1)

            # Parse JSON
            parsed = json.loads(content)

            # Validate required fields
            required_fields = ["world_model_update", "task_progress_update",
                             "exploration_map_update", "reasoning"]
            if all(field in parsed for field in required_fields):
                return parsed
            else:
                print(f"   ⚠️  Missing required fields (attempt {attempt + 1}/{max_retries})")
                continue

        except json.JSONDecodeError as e:
            print(f"   ⚠️  JSON parse error (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                print(f"   Raw response: {content[:200]}")
        except Exception as e:
            print(f"   ⚠️  LLM call failed (attempt {attempt + 1}/{max_retries}): {e}")

    return None


def create_fallback_annotation(
    obs: str,
    action: str,
    task: str
) -> Dict[str, Any]:
    """Create a minimal fallback annotation if LLM fails"""
    return {
        "world_model_update": {
            "found_objects": {},
            "state_changes": {},
            "cleared_receptacles": []
        },
        "task_progress_update": {
            "subgoal_status": "in_progress",
            "evidence": obs[:100] + "...",
            "updated_subgoal": task
        },
        "exploration_map_update": {
            "newly_visited": [],
            "next_priority": []
        },
        "reasoning": f"Executing action: {action}"
    }


# ============================================================================
# Part 5: Trajectory Parsing and Conversion
# ============================================================================

def parse_expert_trajectory_to_pairs(
    conversations: List[Dict[str, Any]]
) -> List[Tuple[str, str, List[str]]]:
    """
    Parse expert conversations into (Observation, Action, Admissible_Actions) tuples.

    Expert data format:
    - Human: Observation (may include "AVAILABLE ACTIONS: ...")
    - GPT: Thought + Action (or just Action)

    Returns: List of (obs, action, admissible_actions) tuples
    """
    pairs = []
    current_obs = None
    current_admissible = []

    for i, turn in enumerate(conversations):
        if turn['from'] == 'human':
            full_obs = turn['value']

            # Extract admissible actions if present
            admissible_actions = []
            clean_obs = full_obs

            if 'AVAILABLE ACTIONS:' in full_obs:
                parts = full_obs.split('AVAILABLE ACTIONS:')
                clean_obs = parts[0].strip()
                if len(parts) > 1:
                    # Parse comma-separated actions
                    actions_str = parts[1].strip()
                    admissible_actions = [a.strip() for a in actions_str.split(',')]

            current_obs = clean_obs
            current_admissible = admissible_actions

        elif turn['from'] == 'gpt' and current_obs:
            # Extract action from GPT response
            content = turn['value'].strip()

            # Skip acknowledgment
            if "OK. I'll follow" in content:
                continue

            # Extract action
            action = None
            if 'Action:' in content:
                parts = content.split('Action:')
                action = parts[-1].strip()
            elif content and not content.startswith('Thought:'):
                action = content

            if action:
                pairs.append((current_obs, action, current_admissible))
                current_obs = None  # Reset for next pair
                current_admissible = []

    return pairs


def extract_task_from_trajectory(conversations: List[Dict[str, Any]]) -> str:
    """Extract task description from trajectory"""
    for turn in conversations:
        if turn['from'] == 'human' and 'Your task is to:' in turn['value']:
            parts = turn['value'].split('Your task is to:')
            if len(parts) == 2:
                task = parts[1].split('\n')[0].strip()
                return task
    return "Complete the task"


# ============================================================================
# Part 6: Cold-Start Format Conversion
# ============================================================================

def convert_to_coldstart_format(rebel_trajectory: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert ReBel trajectory to cold-start format for SFT training.

    CRITICAL: Cold-start format **DOES NOT** include admissible actions.
    This forces the model to learn strong reasoning without action hints.

    This follows the same strategy as RLVMR:
    - Cold-start SFT: NO admissible actions (hard, forces reasoning)
    - RL training: WITH admissible actions (easier, focuses on belief-guided decisions)
    """
    conversations = rebel_trajectory.get('conversations', [])
    task = rebel_trajectory.get('task', 'Unknown task')

    coldstart_data = {
        "task": task,
        "done": "True",
        "data": []
    }

    step_num = 0

    for i in range(len(conversations)):
        turn = conversations[i]

        # Skip system messages (first 2 turns)
        if turn.get('loss') == False and i < 2:
            continue

        # Find human-gpt pairs
        if turn['from'] == 'human' and 'Observation:' in turn['value']:
            # Get next GPT response
            if i + 1 < len(conversations) and conversations[i + 1]['from'] == 'gpt':
                step_num += 1

                # Extract observation from human turn
                human_value = turn['value']
                obs_start = human_value.index('Observation:') + len('Observation:')
                obs_end = human_value.index('\n\nCurrent Belief State:') if '\n\nCurrent Belief State:' in human_value else len(human_value)
                obs = human_value[obs_start:obs_end].strip()

                # Extract belief state (before "Available Actions")
                belief_start = human_value.index('Current Belief State:') + len('Current Belief State:')
                if 'Available Actions:' in human_value:
                    belief_end = human_value.index('\n\nAvailable Actions:')
                else:
                    belief_end = len(human_value)
                belief_state_str = human_value[belief_start:belief_end].strip()

                # IMPORTANT: Reconstruct prompt WITHOUT admissible actions
                # This is the key difference from RL training
                if step_num == 1:
                    # First step: use NO_HIS_CS template (no history, no admissible actions)
                    prompt = ALFWORLD_REBEL_TEMPLATE_NO_HIS_CS.format(
                        current_observation=obs
                    )
                else:
                    # Subsequent steps: extract task, history, planning from original prompt
                    # Then use CS template (no admissible actions)
                    # Extract necessary info from human_value
                    task_match = re.search(r'Your task is to: (.+)', human_value)
                    task_desc = task_match.group(1) if task_match else task

                    # CRITICAL: Remove "Available Actions" line for cold-start
                    # Must use strip() because the line might have leading whitespace
                    prompt_lines = human_value.split('\n')
                    filtered_lines = [line for line in prompt_lines if 'Available Actions:' not in line]
                    prompt = '\n'.join(filtered_lines)

                response = conversations[i + 1]['value']  # Complete gpt turn as response

                # Add to coldstart data
                coldstart_data["data"].append({
                    "step": step_num,
                    "obs": obs,
                    "prompt": prompt,  # NO admissible actions!
                    "response": response
                })

    return coldstart_data


# ============================================================================
# Part 7: Main Hindsight Annotation Algorithm
# ============================================================================

def generate_rebel_dataset_with_hindsight(
    expert_trajectory: Dict[str, Any],
    client: OpenAI,
    model_name: str
) -> Dict[str, Any]:
    """
    Main algorithm: Generate ReBel dataset using hindsight annotation.

    This implements the improved algorithm with:
    1. Parse expert trajectory into (obs, action, admissible_actions) tuples
    2. For each tuple:
       a. Call Teacher LLM with FULL prev_belief + obs + admissible + ground_truth_action
       b. Parse belief update (incremental) and reasoning
       c. Apply automatic inventory tracking (if LLM missed it)
       d. Apply automatic cleared receptacles detection (pattern: open X -> go to Y)
       e. Merge update into global belief
       f. Format as training sample with FULL belief + admissible in input, INCREMENTAL update in output
    3. Return complete ReBel trajectory

    Key improvements:
    - INPUT: Full belief state (for model context)
    - OUTPUT: Incremental update (for focused learning)
    - Automatic inventory tracking (fallback)
    - Automatic cleared receptacles detection
    - Admissible actions included in training data
    """
    # Step 1: Extract task and parse trajectory
    conversations_orig = expert_trajectory['conversations']
    task_description = extract_task_from_trajectory(conversations_orig)
    pairs = parse_expert_trajectory_to_pairs(conversations_orig)

    if not pairs:
        print("   ⚠️  No valid (obs, action, admissible) tuples found")
        return None

    # Step 2: Initialize state
    global_belief = initialize_belief_state()
    rebel_conversations = []

    # Add system message
    rebel_conversations.append({
        'from': 'human',
        'loss': False,  # System prompts don't compute loss
        'value': 'Interact with a household to solve a task. Use ReBel format with belief state, reasoning, and action.'
    })

    rebel_conversations.append({
        'from': 'gpt',
        'loss': False,  # System acknowledgment doesn't compute loss
        'value': "OK. I will track my belief state and reasoning to solve the task efficiently."
    })

    # Step 3: Process each (observation, action, admissible_actions) tuple
    success_count = 0
    prev_action = None

    for step, (obs, action, admissible_actions) in enumerate(pairs):
        # IMPORTANT: Get current belief BEFORE merging this step's update
        # This is what the model will see as input during training
        input_belief_json = json.dumps(global_belief, indent=2, ensure_ascii=False)

        # Check for failure
        is_nothing_happens = "Nothing happens" in obs

        # Construct annotation prompt (with FULL global belief)
        system_prompt, user_prompt = construct_annotation_prompt(
            task_desc=task_description,
            current_obs=obs,
            prev_belief=global_belief,  # FULL STATE for context
            ground_truth_action=action,
            admissible_actions=admissible_actions,
            is_nothing_happens=is_nothing_happens
        )

        # Call Teacher LLM
        llm_output = call_teacher_llm(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            client=client,
            model_name=model_name
        )

        if llm_output is None:
            print(f"   ⚠️  Step {step}: LLM annotation failed, using fallback")
            llm_output = create_fallback_annotation(obs, action, task_description)
        else:
            success_count += 1

        # Extract components (INCREMENTAL update)
        belief_update = {
            "world_model_update": llm_output.get("world_model_update", {}),
            "task_progress_update": llm_output.get("task_progress_update", {}),
            "exploration_map_update": llm_output.get("exploration_map_update", {})
        }
        reasoning = llm_output.get("reasoning", "")

        # AUTOMATIC INVENTORY TRACKING (fallback if LLM missed it)
        if "world_model_update" in belief_update:
            wm_update = belief_update["world_model_update"]

            # Auto-detect "take" action
            if "take" in action.lower() and "from" in action.lower():
                # Extract object being taken
                match = re.search(r'take\s+([\w\s]+\d+)\s+from', action.lower())
                if match and "inventory" not in wm_update:
                    obj = match.group(1).strip()
                    wm_update["inventory"] = obj
                    print(f"   🔧 Auto-added inventory: {obj}")

            # Auto-detect "put" action
            elif ("put" in action.lower() or "place" in action.lower()) and ("inventory" not in wm_update):
                wm_update["inventory"] = None
                print(f"   🔧 Auto-cleared inventory")

            # Auto-detect cleared receptacles (advanced pattern matching)
            # If prev action was "open X" and current action is "go to Y", mark X as cleared
            if prev_action and "open" in prev_action.lower() and "go to" in action.lower():
                prev_match = re.search(r'open\s+([\w\s]+\d+)', prev_action.lower())
                if prev_match:
                    receptacle = prev_match.group(1).strip()
                    if "cleared_receptacles" not in wm_update:
                        wm_update["cleared_receptacles"] = []
                    if receptacle not in wm_update["cleared_receptacles"]:
                        wm_update["cleared_receptacles"].append(receptacle)
                        print(f"   🔧 Auto-cleared receptacle: {receptacle}")

        # Merge into global state
        global_belief = merge_belief_update(global_belief, belief_update)

        # Format TRAINING DATA
        # IMPORTANT: For hindsight annotation, we include admissible actions
        # because this helps the Teacher LLM generate better belief states.
        # However, when converting to cold-start format, we will REMOVE them.
        # This follows RLVMR's curriculum learning strategy:
        #   - Cold-start SFT: NO admissible actions (forces strong reasoning)
        #   - RL training: WITH admissible actions (focuses on belief-guided decisions)

        admissible_str = ", ".join(admissible_actions) if admissible_actions else "N/A"

        human_turn_value = f"""Task: {task_description}

Observation:
{obs}

Current Belief State:
{input_belief_json}

Available Actions: {admissible_str}"""

        rebel_conversations.append({
            'from': 'human',
            'loss': False,  # User input doesn't compute loss
            'value': human_turn_value
        })

        # OUTPUT: Incremental belief update + reasoning + action
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
            'loss': True,  # Model output computes loss for training
            'value': rebel_output
        })

        # Track previous action for pattern detection
        prev_action = action

    # Step 4: Create final dataset sample
    item_id = expert_trajectory.get('item_id', 'unknown')

    result = {
        'conversations': rebel_conversations,
        'item_id': f"{item_id}_rebel_hindsight",
        'num_steps': len(pairs),
        'annotation_success_rate': success_count / len(pairs) if pairs else 0,
        'task': task_description
    }

    return result


# ============================================================================
# Part 7: Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate ReBel Golden Trajectories using Hindsight Annotation"
    )
    parser.add_argument('--expert_data', type=str,
                       default='data/alfworld_expert_traj.json',
                       help='Path to expert trajectory data (JSON or dataset dir)')
    parser.add_argument('--output_dir', type=str,
                       default='data/alfworld_rebel_hindsight',
                       help='Output directory for ReBel dataset')
    parser.add_argument('--num_samples', type=int, default=5,
                       help='Number of trajectories to process')
    parser.add_argument('--teacher_model_url', type=str,
                       default='http://127.0.0.1:8000/v1',
                       help='Teacher LLM API endpoint (OpenAI-compatible)')
    parser.add_argument('--teacher_model_name', type=str,
                       default='gpt-4',
                       help='Teacher model name (e.g., gpt-4, qwen, etc.)')
    parser.add_argument('--temperature', type=float, default=0.3,
                       help='Temperature for Teacher LLM')
    parser.add_argument('--random_sample', action='store_true',
                       help='Randomly sample trajectories instead of sequential')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for sampling')
    parser.add_argument('--save_interval', type=int, default=10,
                       help='Save progress every N trajectories (default: 10)')
    parser.add_argument('--generate_coldstart', action='store_true',
                       help='Also generate cold-start format file')

    args = parser.parse_args()

    print("=" * 80)
    print("ReBel Hindsight Annotation System")
    print("=" * 80)
    print(f"📚 Expert data: {args.expert_data}")
    print(f"🤖 Teacher model: {args.teacher_model_name}")
    print(f"📊 Processing: {args.num_samples} samples")
    print("=" * 80)

    # Step 1: Load expert data
    print("\n[1/4] Loading expert trajectories...")
    if args.expert_data.endswith('.json') or args.expert_data.endswith('.jsonl'):
        with open(args.expert_data, 'r') as f:
            content = f.read().strip()
            # Support both JSON array and JSONL formats
            if content.startswith('['):
                # Standard JSON array format
                expert_data = json.loads(content)
            else:
                # JSONL format (one JSON object per line)
                expert_data = [json.loads(line) for line in content.splitlines() if line.strip()]
    else:
        # Assume Arrow/Parquet dataset format
        expert_ds = load_from_disk(args.expert_data)
        expert_data = list(expert_ds)

    # Random sampling if requested
    if args.random_sample:
        random.seed(args.seed)
        random.shuffle(expert_data)
        print(f"🔀 Randomly shuffled data (seed={args.seed})")

    expert_data = expert_data[:args.num_samples]
    print(f"✅ Loaded {len(expert_data)} expert trajectories")
    if args.random_sample:
        print(f"   (Randomly sampled from full dataset)")

    # Step 2: Connect to Teacher LLM
    print("\n[2/4] Connecting to Teacher LLM...")
    try:
        # Use API key from environment variable or "EMPTY" for local vLLM
        api_key = os.getenv("OPENAI_API_KEY", "EMPTY")
        client = OpenAI(api_key=api_key, base_url=args.teacher_model_url)
        client.models.list()
        print(f"✅ Connected to {args.teacher_model_url}")
    except Exception as e:
        print(f"❌ Failed to connect to Teacher LLM: {e}")
        print("   Please ensure the model server is running")
        return

    # Step 3: Generate ReBel trajectories with real-time saving
    print("\n[3/4] Generating ReBel trajectories with hindsight annotation...")
    print(f"💾 Real-time save interval: every {args.save_interval} trajectories")
    os.makedirs(args.output_dir, exist_ok=True)

    rebel_trajectories = []
    coldstart_trajectories = []

    # Progress tracking
    total_success_steps = 0
    total_steps = 0

    for idx, expert_traj in enumerate(tqdm(expert_data, desc="Annotating",
                                            ncols=100, unit="traj")):
        current_num = idx + 1
        print(f"\n📝 [{current_num}/{len(expert_data)}] Processing trajectory...")

        try:
            rebel_traj = generate_rebel_dataset_with_hindsight(
                expert_trajectory=expert_traj,
                client=client,
                model_name=args.teacher_model_name
            )

            if rebel_traj:
                success_rate = rebel_traj['annotation_success_rate']
                num_steps = rebel_traj['num_steps']
                total_steps += num_steps
                total_success_steps += int(num_steps * success_rate)

                print(f"   ✅ Generated {num_steps} steps | Success: {success_rate:.1%} | "
                      f"Overall: {total_success_steps}/{total_steps} ({total_success_steps/total_steps:.1%})")
                rebel_trajectories.append(rebel_traj)

                # Convert to cold-start format if requested
                if args.generate_coldstart:
                    coldstart_traj = convert_to_coldstart_format(rebel_traj)
                    coldstart_trajectories.append(coldstart_traj)
            else:
                print(f"   ⚠️  Failed to generate trajectory")

        except Exception as e:
            print(f"   ❌ Error: {e}")
            import traceback
            traceback.print_exc()

        # Real-time save at intervals
        if current_num % args.save_interval == 0 or current_num == len(expert_data):
            print(f"\n💾 Saving progress ({current_num}/{len(expert_data)} completed)...")

            # Save ReBel format
            jsonl_path = os.path.join(args.output_dir, 'rebel_hindsight.jsonl')
            with open(jsonl_path, 'w') as f:
                for traj in rebel_trajectories:
                    f.write(json.dumps(traj, ensure_ascii=False) + '\n')

            # Save cold-start format if generated
            if args.generate_coldstart and coldstart_trajectories:
                coldstart_path = os.path.join(args.output_dir, 'rebel_coldstart.json')
                with open(coldstart_path, 'w') as f:
                    json.dump(coldstart_trajectories, f, indent=2, ensure_ascii=False)

            print(f"   ✅ Saved {len(rebel_trajectories)} trajectories to {args.output_dir}")

    # Step 4: Final save and statistics
    print(f"\n[4/4] Finalizing results...")

    if rebel_trajectories:
        # Save as dataset (Arrow/Parquet format)
        ds = Dataset.from_list(rebel_trajectories)
        ds.save_to_disk(args.output_dir)

        # Save as JSONL (already saved in real-time, this is final)
        jsonl_path = os.path.join(args.output_dir, 'rebel_hindsight.jsonl')

        # Save cold-start format (already saved in real-time if enabled)
        if args.generate_coldstart:
            coldstart_path = os.path.join(args.output_dir, 'rebel_coldstart.json')

        print(f"✅ Saved {len(rebel_trajectories)} trajectories")
        print(f"   📁 Dataset (Arrow): {args.output_dir}")
        print(f"   📄 JSONL format: {jsonl_path}")
        if args.generate_coldstart:
            print(f"   🚀 Cold-start format: {coldstart_path}")

        # Statistics
        avg_steps = sum(t['num_steps'] for t in rebel_trajectories) / len(rebel_trajectories)
        avg_success = sum(t['annotation_success_rate'] for t in rebel_trajectories) / len(rebel_trajectories)

        print(f"\n📊 Statistics:")
        print(f"   Total trajectories: {len(rebel_trajectories)}")
        print(f"   Avg steps per trajectory: {avg_steps:.1f}")
        print(f"   Avg annotation success rate: {avg_success:.1%}")
        print(f"   Total annotation steps: {total_steps}")
        print(f"   Successful annotations: {total_success_steps} / {total_steps}")

        # Show example
        if rebel_trajectories and rebel_trajectories[0]['conversations']:
            print("\n" + "=" * 80)
            print("Example ReBel Turn:")
            print("=" * 80)
            for turn in rebel_trajectories[0]['conversations']:
                if '<belief>' in turn.get('value', ''):
                    print(turn['value'][:700] + "..." if len(turn['value']) > 700 else turn['value'])
                    break

    print("\n" + "=" * 80)
    print("✅ Hindsight annotation complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
