#!/usr/bin/env python3
"""
Generate ReBel Cold-Start Data from Expert Trajectories

This script enhances existing expert trajectories with golden belief states
by simulating the environment and tracking ground truth state at each step.
"""

import os
import sys
import json
import argparse
from datasets import load_from_disk, Dataset
from typing import List, Dict, Any
from tqdm import tqdm

sys.path.insert(0, '/root/testttt/RLVMR/code')
from agent_system.environments.env_package.alfworld import (
    GroundTruthTracker, BeliefStateParser
)


def parse_expert_conversation(conversations: List[Dict]) -> List[Dict]:
    """
    Parse expert conversation into structured trajectory

    Returns:
        List of steps, each containing:
        - observation: environment observation
        - thought: expert's thought (optional)
        - action: expert's action
    """
    steps = []
    current_step = {}

    for i, turn in enumerate(conversations):
        role = turn['from']
        content = turn['value'].strip()

        if role == 'human':
            # Skip system prompt
            if i == 0 and 'Imagine you are an intelligent agent' in content:
                continue

            # Extract observation and available actions
            if 'AVAILABLE ACTIONS:' in content or 'Your task is to:' in content:
                # This is an observation
                parts = content.split('AVAILABLE ACTIONS:')
                if len(parts) == 2:
                    obs = parts[0].strip()
                    actions = parts[1].strip()
                    current_step = {
                        'observation': obs,
                        'available_actions': actions,
                        'thought': None,
                        'action': None
                    }
                else:
                    # Just observation without available actions
                    current_step = {
                        'observation': content,
                        'available_actions': None,
                        'thought': None,
                        'action': None
                    }

        elif role == 'gpt':
            # Extract thought and action
            if 'Thought:' in content and 'Action:' in content:
                parts = content.split('Action:')
                thought = parts[0].replace('Thought:', '').strip()
                action = parts[1].strip()

                if current_step:
                    current_step['thought'] = thought
                    current_step['action'] = action
                    steps.append(current_step.copy())
                    current_step = {}
            elif 'Action:' in content:
                # Only action
                action = content.replace('Action:', '').strip()
                if current_step:
                    current_step['action'] = action
                    steps.append(current_step.copy())
                    current_step = {}
            else:
                # Probably just acknowledgment, skip
                pass

    return steps


def extract_task_goal(observation: str) -> str:
    """Extract task goal from observation"""
    if 'Your task is to:' in observation:
        parts = observation.split('Your task is to:')
        if len(parts) == 2:
            task = parts[1].split('.')[0].strip()
            return task
    return None


def generate_golden_belief_state(
    tracker: GroundTruthTracker,
    observation: str,
    action: str,
    step: int
) -> Dict[str, Any]:
    """
    Generate golden belief state based on ground truth environment state

    Returns:
        Belief state in ReBel format
    """
    # Update tracker with observation and action
    tracker.update_from_observation(observation, action)
    gt_state = tracker.get_ground_truth_state()

    # Build world model update
    world_model_update = {
        'found_objects': {},
        'state_changes': {},
        'cleared_receptacles': []
    }

    # Extract found objects from ground truth
    for obj, loc in gt_state['object_locations'].items():
        world_model_update['found_objects'][obj] = loc

    # Extract state changes
    for obj, state in gt_state['object_states'].items():
        world_model_update['state_changes'][obj] = state

    # Extract cleared receptacles
    world_model_update['cleared_receptacles'] = list(gt_state['cleared_receptacles'])

    # Build task progress update
    task_progress_update = {
        'subgoal_status': 'in_progress',
        'evidence': f'Current observation at step {step}',
        'updated_subgoal': None  # To be inferred from actions
    }

    # Build exploration map update
    exploration_map_update = {
        'newly_visited': list(gt_state['visited'])[-3:] if gt_state['visited'] else [],  # Last 3 visited
        'next_priority': []  # To be inferred from strategy
    }

    return {
        'world_model_update': world_model_update,
        'task_progress_update': task_progress_update,
        'exploration_map_update': exploration_map_update
    }


def format_rebel_output(
    thought: str,
    action: str,
    belief_state: Dict[str, Any]
) -> str:
    """
    Format output in ReBel format:
    <belief>...</belief>
    <reasoning>...</reasoning>
    <action>...</action>
    """
    # Format belief state as JSON
    belief_json = json.dumps(belief_state, indent=2, ensure_ascii=False)

    # Build output
    output = f"""<belief>
{belief_json}
</belief>

<reasoning>
{thought if thought else 'Proceeding with the next logical action based on current state.'}
</reasoning>

<action>
{action}
</action>"""

    return output


def augment_trajectory_with_beliefs(
    conversations: List[Dict],
    item_id: str
) -> Dict[str, Any]:
    """
    Augment a single trajectory with golden belief states

    Returns:
        Augmented trajectory in ReBel format (conversation style)
    """
    # Parse expert trajectory
    steps = parse_expert_conversation(conversations)

    if not steps:
        return None

    # Initialize ground truth tracker
    tracker = GroundTruthTracker()

    # Extract task goal from first observation
    task_goal = extract_task_goal(steps[0]['observation'])
    if task_goal:
        tracker.task_goal = task_goal

    # Build augmented conversations
    augmented_convs = []

    # Add system prompt
    augmented_convs.append({
        'from': 'human',
        'loss': None,
        'value': conversations[0]['value']  # Use original system prompt
    })

    # Add initial acknowledgment
    augmented_convs.append({
        'from': 'gpt',
        'loss': False,
        'value': "OK. I'll follow your instructions and try my best to solve the task."
    })

    # Process each step
    for step_idx, step in enumerate(steps):
        obs = step['observation']
        thought = step['thought']
        action = step['action']

        if not action:
            continue

        # Generate golden belief state
        belief_state = generate_golden_belief_state(tracker, obs, action, step_idx)

        # Format as ReBel output
        rebel_output = format_rebel_output(thought, action, belief_state)

        # Add to conversations
        augmented_convs.append({
            'from': 'human',
            'loss': None,
            'value': obs
        })

        augmented_convs.append({
            'from': 'gpt',
            'loss': True,  # This is the part we want to train on
            'value': rebel_output
        })

    return {
        'conversations': augmented_convs,
        'item_id': f"{item_id}_rebel"
    }


def main():
    parser = argparse.ArgumentParser(description='Generate ReBel cold-start data from expert trajectories')
    parser.add_argument('--input_dir', type=str, default='data/alfworld_expert_traj',
                        help='Input directory containing expert trajectories')
    parser.add_argument('--output_dir', type=str, default='data/alfworld_rebel_cold_start',
                        help='Output directory for ReBel cold-start data')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of trajectories to process (for testing)')

    args = parser.parse_args()

    print("="*60)
    print("ReBel Cold-Start Data Generation")
    print("="*60)
    print(f"Input:  {args.input_dir}")
    print(f"Output: {args.output_dir}")
    print("="*60)

    # Load expert trajectories
    print("\nLoading expert trajectories...")
    dataset = load_from_disk(args.input_dir)
    print(f"Loaded {len(dataset)} trajectories")

    if args.limit:
        dataset = dataset.select(range(min(args.limit, len(dataset))))
        print(f"Limited to {len(dataset)} trajectories for testing")

    # Process each trajectory
    print("\nProcessing trajectories...")
    augmented_data = []
    failed_count = 0

    for idx, example in enumerate(tqdm(dataset)):
        try:
            augmented = augment_trajectory_with_beliefs(
                example['conversations'],
                example['item_id']
            )

            if augmented:
                augmented_data.append(augmented)
            else:
                failed_count += 1
        except Exception as e:
            print(f"\nError processing trajectory {idx}: {e}")
            failed_count += 1
            continue

    print(f"\nSuccessfully processed: {len(augmented_data)}")
    print(f"Failed: {failed_count}")

    # Save augmented dataset
    print(f"\nSaving to {args.output_dir}...")
    os.makedirs(args.output_dir, exist_ok=True)

    # Create HuggingFace dataset
    augmented_dataset = Dataset.from_list(augmented_data)
    augmented_dataset.save_to_disk(args.output_dir)

    # Also save as JSONL for easy inspection
    jsonl_path = os.path.join(args.output_dir, 'rebel_cold_start.jsonl')
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for item in augmented_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"✅ Saved {len(augmented_data)} augmented trajectories")
    print(f"   - HuggingFace Dataset: {args.output_dir}")
    print(f"   - JSONL: {jsonl_path}")

    # Show example
    if augmented_data:
        print("\n" + "="*60)
        print("Example Augmented Trajectory (first GPT turn):")
        print("="*60)
        example_gpt_turn = None
        for turn in augmented_data[0]['conversations']:
            if turn['from'] == 'gpt' and '<belief>' in turn['value']:
                example_gpt_turn = turn['value']
                break

        if example_gpt_turn:
            print(example_gpt_turn[:1000] + "..." if len(example_gpt_turn) > 1000 else example_gpt_turn)

    print("\n" + "="*60)
    print("Done!")
    print("="*60)


if __name__ == '__main__':
    main()
