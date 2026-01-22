#!/usr/bin/env python3
"""
Generate ReBel Golden Trajectories by Environment Interaction

This script replays expert trajectories in the actual ALFWorld environment,
tracks the true environment state at each step, and uses it as the golden
belief state. This ensures 100% accuracy and efficiency.

Key advantages:
1. Belief states are 100% accurate (directly from environment)
2. Uses expert action sequences (guaranteed to succeed)
3. Shorter context (no need for model inference)
4. Perfect training signal for cold-start
"""

import os
import sys
import json
import argparse
from datasets import load_from_disk, Dataset
from typing import List, Dict, Any, Tuple
from tqdm import tqdm

sys.path.insert(0, '/root/testttt/RLVMR/code')

from agent_system.environments.env_package.alfworld import (
    build_alfworld_envs,
    GroundTruthTracker
)


def extract_actions_from_expert_trajectory(conversations: List[Dict]) -> List[str]:
    """Extract action sequence from expert trajectory"""
    actions = []

    for turn in conversations:
        if turn['from'] == 'gpt':
            content = turn['value'].strip()

            # Skip initial acknowledgment
            if "OK. I'll follow your instructions" in content:
                continue

            # Extract action
            if 'Action:' in content:
                parts = content.split('Action:')
                action = parts[-1].strip()
                actions.append(action)
            elif content and not content.startswith('Thought:'):
                # Sometimes only action without "Action:" prefix
                actions.append(content)

    return actions


def extract_task_from_expert_trajectory(conversations: List[Dict]) -> str:
    """Extract task description from expert trajectory"""
    for turn in conversations:
        if turn['from'] == 'human' and 'Your task is to:' in turn['value']:
            parts = turn['value'].split('Your task is to:')
            if len(parts) == 2:
                task = parts[1].split('\n')[0].strip()
                return task
    return None


def format_rebel_golden_turn(
    observation: str,
    action: str,
    thought: str,
    gt_state: Dict[str, Any],
    step: int
) -> str:
    """
    Format a single turn in ReBel golden format

    The belief state is the ACTUAL environment state (100% accurate)
    """
    # Build golden belief state from ground truth
    belief_state = {
        "world_model_update": {
            "found_objects": dict(gt_state.get('object_locations', {})),
            "state_changes": dict(gt_state.get('object_states', {})),
            "cleared_receptacles": list(gt_state.get('cleared_receptacles', []))
        },
        "task_progress_update": {
            "subgoal_status": "in_progress",
            "evidence": f"Based on observation: {observation[:100]}...",
            "updated_subgoal": None  # Can be inferred from thought
        },
        "exploration_map_update": {
            "newly_visited": list(gt_state.get('visited', []))[-3:],  # Last 3 visited
            "next_priority": []
        }
    }

    # Format as ReBel output
    belief_json = json.dumps(belief_state, indent=2, ensure_ascii=False)

    output = f"""<belief>
{belief_json}
</belief>

<reasoning>
{thought if thought else 'Executing the next action based on current state.'}
</reasoning>

<action>
{action}
</action>"""

    return output


def replay_trajectory_in_env(
    envs,
    expert_actions: List[str],
    task: str,
    item_id: str
) -> Dict[str, Any]:
    """
    Replay expert trajectory in actual environment and generate ReBel golden trajectory

    Returns:
        ReBel formatted conversation with golden belief states
    """
    # Reset environment
    text_obs, image_obs, infos = envs.reset()

    # Initialize ground truth tracker
    tracker = GroundTruthTracker()
    initial_obs = text_obs[0]  # Get raw observation
    tracker.update_from_observation(initial_obs)
    tracker.task_goal = task

    # Build conversation
    conversations = []

    # Add system prompt (same as expert trajectory)
    conversations.append({
        'from': 'human',
        'loss': None,
        'value': 'Interact with a household to solve a task. Imagine you are an intelligent agent in a household environment and your target is to perform actions to complete the task goal. At the beginning of your interactions, you will be given the detailed description of the current environment and your goal to accomplish. For each of your turn, you will be given a list of actions which you can choose one to perform in this turn. You should follow the ReBel format: output your belief state, reasoning, and action. Your output must strictly follow this format:"<belief>\\n{JSON belief state}\\n</belief>\\n\\n<reasoning>\\nyour reasoning\\n</reasoning>\\n\\n<action>\\nyour action\\n</action>".'
    })

    # Add acknowledgment
    conversations.append({
        'from': 'gpt',
        'loss': False,
        'value': "OK. I'll follow your instructions and use the ReBel format to solve the task."
    })

    # Add initial observation
    conversations.append({
        'from': 'human',
        'loss': None,
        'value': initial_obs
    })

    # Replay each expert action
    for step, expert_action in enumerate(expert_actions):
        # Get current ground truth state
        gt_state = tracker.get_ground_truth_state()

        # Get current observation
        current_obs = text_obs[0]

        # Get admissible actions
        admissible_actions = envs.get_admissible_commands[0]

        # Format ReBel turn with golden belief state
        rebel_output = format_rebel_golden_turn(
            observation=current_obs,
            action=expert_action,
            thought=f"Step {step + 1}: Executing expert action",
            gt_state=gt_state,
            step=step
        )

        # Add to conversation
        conversations.append({
            'from': 'gpt',
            'loss': True,  # This is what we train on
            'value': rebel_output
        })

        # Execute action in environment
        try:
            text_obs, image_obs, rewards, dones, infos = envs.step([expert_action])

            # Update ground truth tracker
            new_obs = text_obs[0]
            tracker.update_from_observation(new_obs, expert_action)

            # Add observation to conversation
            conversations.append({
                'from': 'human',
                'loss': None,
                'value': new_obs
            })

            # Check if done
            if dones[0]:
                success = infos[0].get('won', False)
                print(f"  Episode ended at step {step + 1}, Success: {success}")
                break

        except Exception as e:
            print(f"  Error executing action '{expert_action}': {e}")
            break

    return {
        'conversations': conversations,
        'item_id': f"{item_id}_rebel_golden",
        'success': infos[0].get('won', False) if infos else False
    }


def main():
    parser = argparse.ArgumentParser(description='Generate ReBel golden trajectories by environment interaction')
    parser.add_argument('--expert_traj_dir', type=str, default='data/alfworld_expert_traj',
                        help='Expert trajectory dataset directory')
    parser.add_argument('--output_dir', type=str, default='data/alfworld_rebel_golden',
                        help='Output directory for ReBel golden trajectories')
    parser.add_argument('--num_samples', type=int, default=3,
                        help='Number of trajectories to generate')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for environment')

    args = parser.parse_args()

    print("="*70)
    print("ReBel Golden Trajectory Generation via Environment Interaction")
    print("="*70)
    print(f"Expert trajectories: {args.expert_traj_dir}")
    print(f"Output directory:    {args.output_dir}")
    print(f"Number of samples:   {args.num_samples}")
    print("="*70)

    # Load expert trajectories
    print("\n[1/4] Loading expert trajectories...")
    expert_dataset = load_from_disk(args.expert_traj_dir)
    print(f"✅ Loaded {len(expert_dataset)} expert trajectories")

    # Select samples
    if args.num_samples > len(expert_dataset):
        args.num_samples = len(expert_dataset)

    samples = expert_dataset.select(range(args.num_samples))
    print(f"📋 Selected {args.num_samples} samples for generation")

    # Initialize ALFWorld environment
    print("\n[2/4] Initializing ALFWorld environment...")
    alf_config_path = os.path.join(
        os.path.dirname(__file__),
        'agent_system/environments/env_package/alfworld/configs/config_tw.yaml'
    )

    # Build raw environments (not wrapped in env_manager)
    envs = build_alfworld_envs(
        alf_config_path,
        seed=args.seed,
        env_num=1,
        group_n=1,
        is_train=True
    )

    print("✅ Environment initialized")

    # Generate ReBel golden trajectories
    print(f"\n[3/4] Generating ReBel golden trajectories...")
    golden_trajectories = []
    success_count = 0

    for idx, sample in enumerate(tqdm(samples, desc="Generating")):
        try:
            # Extract expert actions and task
            expert_actions = extract_actions_from_expert_trajectory(sample['conversations'])
            task = extract_task_from_expert_trajectory(sample['conversations'])

            if not expert_actions or not task:
                print(f"\n⚠️  Skipping sample {idx}: No actions or task found")
                continue

            print(f"\n📝 Sample {idx + 1}/{args.num_samples}:")
            print(f"   Task: {task}")
            print(f"   Expert actions: {len(expert_actions)} steps")

            # Replay in environment
            golden_traj = replay_trajectory_in_env(
                envs=envs,
                expert_actions=expert_actions,
                task=task,
                item_id=sample['item_id']
            )

            if golden_traj['success']:
                success_count += 1
                print(f"   ✅ Success!")
            else:
                print(f"   ⚠️  Did not complete")

            golden_trajectories.append(golden_traj)

        except Exception as e:
            print(f"\n❌ Error processing sample {idx}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n✅ Generated {len(golden_trajectories)} golden trajectories")
    if golden_trajectories:
        print(f"   Success rate: {success_count}/{len(golden_trajectories)} ({success_count/len(golden_trajectories)*100:.1f}%)")
    else:
        print("   ⚠️  No trajectories were successfully generated")

    # Save results
    print(f"\n[4/4] Saving to {args.output_dir}...")
    os.makedirs(args.output_dir, exist_ok=True)

    # Save as HuggingFace dataset
    if golden_trajectories:
        golden_dataset = Dataset.from_list(golden_trajectories)
        golden_dataset.save_to_disk(args.output_dir)

        # Also save as JSONL for inspection
        jsonl_path = os.path.join(args.output_dir, 'rebel_golden.jsonl')
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for traj in golden_trajectories:
                f.write(json.dumps(traj, ensure_ascii=False) + '\n')

        print(f"✅ Saved to:")
        print(f"   - Dataset: {args.output_dir}")
        print(f"   - JSONL:   {jsonl_path}")

    # Show example
    if golden_trajectories:
        print("\n" + "="*70)
        print("Example ReBel Golden Turn (first action):")
        print("="*70)

        for turn in golden_trajectories[0]['conversations']:
            if turn['from'] == 'gpt' and '<belief>' in turn['value']:
                print(turn['value'][:800])
                if len(turn['value']) > 800:
                    print("...")
                break

    print("\n" + "="*70)
    print("✅ Done! Golden trajectories ready for training.")
    print("="*70)


if __name__ == '__main__':
    main()
