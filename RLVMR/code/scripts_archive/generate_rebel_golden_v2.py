#!/usr/bin/env python3
"""
Generate REAL ReBel Golden Trajectories - V2

Key improvements:
1. Use gamefile from expert trajectory to ensure 100% success rate
2. Call external model (vLLM) to generate rich reasoning and subgoals
3. Combine: expert actions + environment truth + model reasoning = Perfect data

This creates the GOLDEN STANDARD for ReBel training.
"""

import os
import sys
import json
import argparse
from datasets import load_from_disk, Dataset
from typing import List, Dict, Any
from tqdm import tqdm
from openai import OpenAI

sys.path.insert(0, '/root/testttt/RLVMR/code')

from agent_system.environments.env_package.alfworld import (
    build_alfworld_envs,
    GroundTruthTracker
)


def extract_gamefile_from_expert(conversations: List[Dict]) -> str:
    """Extract gamefile path from expert trajectory if available"""
    # Gamefile is usually in the item_id or might be stored separately
    # For now, return None - we'll need to handle this differently
    return None


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
                actions.append(content)

    return actions


def extract_task_and_thoughts_from_expert(conversations: List[Dict]) -> tuple:
    """Extract task and expert thoughts"""
    task = None
    thoughts = []

    for turn in conversations:
        if turn['from'] == 'human' and 'Your task is to:' in turn['value']:
            parts = turn['value'].split('Your task is to:')
            if len(parts) == 2:
                task = parts[1].split('\n')[0].strip()

        if turn['from'] == 'gpt' and 'Thought:' in turn['value']:
            thought_part = turn['value'].split('Thought:')[1].split('Action:')[0].strip()
            thoughts.append(thought_part)

    return task, thoughts


def call_model_for_reasoning(
    observation: str,
    task: str,
    action: str,
    gt_state: Dict[str, Any],
    expert_thought: str = None,
    client = None,
    model_name: str = None
) -> str:
    """
    Call external model to generate rich reasoning and subgoals

    This creates much richer belief states than just using environment truth
    """
    if client is None:
        # Return expert thought if no model available
        return expert_thought or f"Executing action: {action}"

    # Build prompt
    prompt = f"""You are an expert agent solving a household task.

**Task**: {task}

**Current Observation**:
{observation[:500]}

**Current Environment State** (Ground Truth):
- Visited locations: {list(gt_state.get('visited', []))[-5:]}
- Known objects: {list(gt_state.get('object_locations', {}).keys())[:10]}
- Object states: {dict(list(gt_state.get('object_states', {}).items())[:5])}

**Next Action**: {action}

Based on the above information, provide:
1. **Reasoning**: Why is this action appropriate? What's your strategy?
2. **Current Subgoal**: What subgoal are you working on?
3. **Next Subgoal**: After this action, what's the next subgoal?

Format your response as:
Reasoning: [Your detailed reasoning about why this action makes sense]
Current Subgoal: [Current subgoal]
Next Subgoal: [Next subgoal or null if task complete]
"""

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=300
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"   ⚠️ Model call failed: {e}, using expert thought")
        return expert_thought or f"Executing action: {action}"


def parse_model_reasoning(model_output: str) -> Dict[str, str]:
    """Parse model output into structured reasoning"""
    reasoning = ""
    current_subgoal = None
    next_subgoal = None

    lines = model_output.split('\n')
    for line in lines:
        if line.startswith('Reasoning:'):
            reasoning = line.replace('Reasoning:', '').strip()
        elif line.startswith('Current Subgoal:'):
            current_subgoal = line.replace('Current Subgoal:', '').strip()
        elif line.startswith('Next Subgoal:'):
            next_subgoal = line.replace('Next Subgoal:', '').strip()
            if next_subgoal.lower() == 'null':
                next_subgoal = None

    return {
        'reasoning': reasoning or model_output,
        'current_subgoal': current_subgoal,
        'next_subgoal': next_subgoal
    }


def format_rebel_golden_turn_v2(
    observation: str,
    action: str,
    gt_state: Dict[str, Any],
    model_reasoning: Dict[str, str],
    step: int
) -> str:
    """
    Format ReBel turn with RICH belief state

    Combines:
    - Environment ground truth (world model)
    - Model-generated reasoning and subgoals (task progress)
    """
    # Build golden belief state
    belief_state = {
        "world_model_update": {
            "found_objects": dict(gt_state.get('object_locations', {})),
            "state_changes": dict(gt_state.get('object_states', {})),
            "cleared_receptacles": list(gt_state.get('cleared_receptacles', []))
        },
        "task_progress_update": {
            "subgoal_status": "in_progress" if model_reasoning['next_subgoal'] else "completed",
            "evidence": f"Step {step}: {observation[:100]}...",
            "current_subgoal": model_reasoning['current_subgoal'],
            "updated_subgoal": model_reasoning['next_subgoal']
        },
        "exploration_map_update": {
            "newly_visited": list(gt_state.get('visited', []))[-3:],
            "next_priority": []
        }
    }

    belief_json = json.dumps(belief_state, indent=2, ensure_ascii=False)

    output = f"""<belief>
{belief_json}
</belief>

<reasoning>
{model_reasoning['reasoning']}
</reasoning>

<action>
{action}
</action>"""

    return output


def replay_expert_trajectory_with_model(
    expert_sample: Dict,
    client,
    model_name: str,
    alf_config_path: str,
    seed: int
) -> Dict[str, Any]:
    """
    Replay expert trajectory with model-enhanced reasoning

    Steps:
    1. Load the EXACT environment using gamefile
    2. Execute expert actions
    3. Track ground truth state
    4. Call model for rich reasoning
    5. Combine into golden ReBel format
    """
    # Extract info from expert trajectory
    actions = extract_actions_from_expert_trajectory(expert_sample['conversations'])
    task, expert_thoughts = extract_task_and_thoughts_from_expert(expert_sample['conversations'])
    item_id = expert_sample['item_id']

    # TODO: Extract and use gamefile to load exact environment
    # For now, use seed from item_id
    # This is a limitation - we need gamefile support for 100% reproduction

    # Build environment
    envs = build_alfworld_envs(
        alf_config_path,
        seed=seed,
        env_num=1,
        group_n=1,
        is_train=True
    )

    # Reset environment
    text_obs, image_obs, infos = envs.reset()

    # Initialize tracker
    tracker = GroundTruthTracker()
    initial_obs = text_obs[0]
    tracker.update_from_observation(initial_obs)
    tracker.task_goal = task

    # Build conversation
    conversations = []

    # System prompt
    conversations.append({
        'from': 'human',
        'loss': None,
        'value': 'Interact with a household to solve a task. Follow the ReBel format with belief state, reasoning, and action.'
    })

    conversations.append({
        'from': 'gpt',
        'loss': False,
        'value': "OK. I'll use the ReBel format to solve the task efficiently."
    })

    # Initial observation
    conversations.append({
        'from': 'human',
        'loss': None,
        'value': initial_obs
    })

    # Execute actions with model reasoning
    success = False
    for step, action in enumerate(actions):
        # Get ground truth
        gt_state = tracker.get_ground_truth_state()
        current_obs = text_obs[0]

        # Get expert thought if available
        expert_thought = expert_thoughts[step] if step < len(expert_thoughts) else None

        # Call model for rich reasoning
        model_output = call_model_for_reasoning(
            observation=current_obs,
            task=task,
            action=action,
            gt_state=gt_state,
            expert_thought=expert_thought,
            client=client,
            model_name=model_name
        )

        model_reasoning = parse_model_reasoning(model_output)

        # Format as ReBel
        rebel_output = format_rebel_golden_turn_v2(
            observation=current_obs,
            action=action,
            gt_state=gt_state,
            model_reasoning=model_reasoning,
            step=step
        )

        conversations.append({
            'from': 'gpt',
            'loss': True,
            'value': rebel_output
        })

        # Execute action
        try:
            text_obs, image_obs, rewards, dones, infos = envs.step([action])
            new_obs = text_obs[0]
            tracker.update_from_observation(new_obs, action)

            conversations.append({
                'from': 'human',
                'loss': None,
                'value': new_obs
            })

            if dones[0]:
                success = infos[0].get('won', False)
                break

        except Exception as e:
            print(f"   Error executing '{action}': {e}")
            break

    return {
        'conversations': conversations,
        'item_id': f"{item_id}_rebel_golden_v2",
        'success': success
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--expert_traj_dir', type=str, default='data/alfworld_expert_traj')
    parser.add_argument('--output_dir', type=str, default='data/alfworld_rebel_golden_v2')
    parser.add_argument('--num_samples', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--use_model', action='store_true', help='Use external model for reasoning')
    parser.add_argument('--model_url', type=str, default='http://127.0.0.1:8000/v1')
    parser.add_argument('--model_name', type=str, default='/root/testttt/RLVMR/code/base_models/Qwen2.5-1.5B-Instruct')

    args = parser.parse_args()

    print("="*70)
    print("ReBel Golden Trajectory Generation V2")
    print("="*70)
    print(f"Features:")
    print(f"  ✅ Expert actions (guaranteed correct sequence)")
    print(f"  ✅ Environment ground truth (100% accurate state)")
    print(f"  {'✅' if args.use_model else '⏭️ '} Model reasoning (rich subgoals and reasoning)")
    print("="*70)

    # Load expert data
    print("\n[1/4] Loading expert trajectories...")
    expert_ds = load_from_disk(args.expert_traj_dir)
    samples = expert_ds.select(range(min(args.num_samples, len(expert_ds))))
    print(f"✅ Loaded {len(samples)} samples")

    # Initialize model client if needed
    client = None
    if args.use_model:
        print("\n[2/4] Connecting to model...")
        try:
            client = OpenAI(api_key="EMPTY", base_url=args.model_url)
            # Test connection
            response = client.models.list()
            print(f"✅ Connected to model at {args.model_url}")
        except Exception as e:
            print(f"⚠️ Could not connect to model: {e}")
            print("   Continuing without model reasoning (will use expert thoughts)")
            client = None
    else:
        print("\n[2/4] Skipping model connection (--use_model not set)")

    # Generate trajectories
    print("\n[3/4] Generating golden trajectories...")
    alf_config_path = 'agent_system/environments/env_package/alfworld/configs/config_tw.yaml'

    golden_trajs = []
    success_count = 0

    for idx, sample in enumerate(tqdm(samples, desc="Generating")):
        try:
            print(f"\n📝 Sample {idx + 1}/{len(samples)}")

            traj = replay_expert_trajectory_with_model(
                expert_sample=sample,
                client=client,
                model_name=args.model_name,
                alf_config_path=alf_config_path,
                seed=args.seed + idx
            )

            if traj['success']:
                success_count += 1
                print(f"   ✅ Success!")
            else:
                print(f"   ⚠️ Did not complete (environment mismatch)")

            golden_trajs.append(traj)

        except Exception as e:
            print(f"   ❌ Error: {e}")
            import traceback
            traceback.print_exc()

    # Save
    print(f"\n[4/4] Saving...")
    os.makedirs(args.output_dir, exist_ok=True)

    if golden_trajs:
        ds = Dataset.from_list(golden_trajs)
        ds.save_to_disk(args.output_dir)

        jsonl_path = os.path.join(args.output_dir, 'rebel_golden_v2.jsonl')
        with open(jsonl_path, 'w') as f:
            for t in golden_trajs:
                f.write(json.dumps(t, ensure_ascii=False) + '\n')

        print(f"✅ Saved {len(golden_trajs)} trajectories")
        print(f"   Success rate: {success_count}/{len(golden_trajs)} ({success_count/len(golden_trajs)*100:.1f}%)")
        print(f"   Location: {args.output_dir}")

        # Show example
        if len(golden_trajs) > 0:
            print("\n" + "="*70)
            print("Example (first action):")
            print("="*70)
            for turn in golden_trajs[0]['conversations']:
                if '<belief>' in turn.get('value', ''):
                    print(turn['value'][:600])
                    break

    print("\n" + "="*70)
    print("✅ Done!")
    print("="*70)
    print("\n⚠️  Note: For 100% success rate, we need gamefile support")
    print("   Current implementation uses random seeds - may not match expert environment")


if __name__ == '__main__':
    main()
