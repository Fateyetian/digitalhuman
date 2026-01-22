#!/usr/bin/env python3
"""
ReBel Component Test
Quick validation of ReBel components without full training
"""

import sys
import json
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

# Add project to path
sys.path.insert(0, '/root/testttt/RLVMR/code')

def test_belief_canonicalization():
    """Test belief state canonicalization"""
    print("\n" + "="*60)
    print("TEST 1: Belief Canonicalization")
    print("="*60)

    from rebel.core_rebel import canonicalize_belief

    # Test case 1: Same subgoal should have same hash
    belief1 = {
        'task_progress_update': {
            'updated_subgoal': 'find tomato',
            'subgoal_status': 'in_progress'
        }
    }

    belief2 = {
        'task_progress_update': {
            'updated_subgoal': 'find tomato',
            'subgoal_status': 'in_progress'
        }
    }

    hash1 = canonicalize_belief(belief1, granularity='subgoal')
    hash2 = canonicalize_belief(belief2, granularity='subgoal')

    print(f"Belief 1 hash: {hash1}")
    print(f"Belief 2 hash: {hash2}")
    print(f"✓ Same subgoal produces same hash: {hash1 == hash2}")

    # Test case 2: Different subgoal should have different hash
    belief3 = {
        'task_progress_update': {
            'updated_subgoal': 'put tomato in fridge',
            'subgoal_status': 'in_progress'
        }
    }

    hash3 = canonicalize_belief(belief3, granularity='subgoal')
    print(f"Belief 3 hash: {hash3}")
    print(f"✓ Different subgoal produces different hash: {hash1 != hash3}")

    return hash1 == hash2 and hash1 != hash3


def test_belief_grouping():
    """Test belief-based grouping"""
    print("\n" + "="*60)
    print("TEST 2: Belief-Based Grouping")
    print("="*60)

    from rebel.core_rebel import build_belief_group

    # Create sample belief states
    belief_states = np.array([
        {'task_progress_update': {'updated_subgoal': 'find apple', 'subgoal_status': 'in_progress'}},
        {'task_progress_update': {'updated_subgoal': 'find apple', 'subgoal_status': 'in_progress'}},
        {'task_progress_update': {'updated_subgoal': 'find apple', 'subgoal_status': 'in_progress'}},
        {'task_progress_update': {'updated_subgoal': 'put apple in fridge', 'subgoal_status': 'in_progress'}},
        {'task_progress_update': {'updated_subgoal': 'put apple in fridge', 'subgoal_status': 'in_progress'}},
        {'task_progress_update': {'updated_subgoal': 'clean potato', 'subgoal_status': 'in_progress'}},
        {'task_progress_update': {'updated_subgoal': 'clean potato', 'subgoal_status': 'in_progress'}},
        {'task_progress_update': {'updated_subgoal': 'clean potato', 'subgoal_status': 'in_progress'}},
        {'task_progress_update': {'updated_subgoal': 'clean potato', 'subgoal_status': 'in_progress'}},
    ], dtype=object)

    # All from same prompt (index 0)
    index = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])

    # Build groups
    belief_group_uids, group_stats = build_belief_group(
        belief_states=belief_states,
        index=index,
        granularity='subgoal',
        summarize=True
    )

    print(f"\n✓ Number of groups: {group_stats['num_groups']}")
    print(f"✓ Mean group size: {group_stats['mean_group_size']:.2f}")
    print(f"✓ Group sizes: {group_stats['group_sizes']}")

    # Expected: 3 groups (find apple, put apple in fridge, clean potato)
    expected_num_groups = 3
    success = group_stats['num_groups'] == expected_num_groups

    if success:
        print(f"\n✅ SUCCESS: Correctly formed {expected_num_groups} groups")
    else:
        print(f"\n❌ FAILED: Expected {expected_num_groups} groups, got {group_stats['num_groups']}")

    return success


def test_advantage_computation():
    """Test ReBel advantage computation"""
    print("\n" + "="*60)
    print("TEST 3: ReBel Advantage Computation")
    print("="*60)

    from rebel.core_rebel import compute_rebel_advantage

    # Create dummy data
    batch_size = 8
    response_length = 10

    token_level_rewards = torch.randn(batch_size, response_length) * 0.1
    rebel_intrinsic_rewards = torch.randn(batch_size) * 0.2
    eos_mask = torch.ones(batch_size, response_length)

    # Create belief states with 2 groups
    belief_states = np.array([
        {'task_progress_update': {'updated_subgoal': 'find apple', 'subgoal_status': 'in_progress'}},
        {'task_progress_update': {'updated_subgoal': 'find apple', 'subgoal_status': 'in_progress'}},
        {'task_progress_update': {'updated_subgoal': 'find apple', 'subgoal_status': 'in_progress'}},
        {'task_progress_update': {'updated_subgoal': 'find apple', 'subgoal_status': 'in_progress'}},
        {'task_progress_update': {'updated_subgoal': 'clean potato', 'subgoal_status': 'in_progress'}},
        {'task_progress_update': {'updated_subgoal': 'clean potato', 'subgoal_status': 'in_progress'}},
        {'task_progress_update': {'updated_subgoal': 'clean potato', 'subgoal_status': 'in_progress'}},
        {'task_progress_update': {'updated_subgoal': 'clean potato', 'subgoal_status': 'in_progress'}},
    ], dtype=object)

    index = np.array([0, 0, 0, 0, 0, 0, 0, 0])

    try:
        advantages, returns, adv_details = compute_rebel_advantage(
            token_level_rewards=token_level_rewards,
            rebel_intrinsic_rewards=rebel_intrinsic_rewards,
            eos_mask=eos_mask,
            belief_states=belief_states,
            index=index,
            epsilon=1e-6,
            step_advantage_w=1.0,
            mode='mean_norm',
            belief_granularity='subgoal',
            summarize=True
        )

        print(f"\n✓ Advantages shape: {advantages.shape}")
        print(f"✓ Returns shape: {returns.shape}")
        print(f"✓ Num groups: {adv_details['belief_group_stats']['num_groups']}")
        print(f"✓ Mean group size: {adv_details['belief_group_stats']['mean_group_size']:.2f}")

        # Check shapes
        shape_correct = advantages.shape == (batch_size, response_length)

        # Check that advantages are not all zero
        not_all_zero = torch.abs(advantages).sum() > 0

        success = shape_correct and not_all_zero

        if success:
            print(f"\n✅ SUCCESS: Advantage computation works correctly")
        else:
            print(f"\n❌ FAILED: Issue with advantage computation")

        return success

    except Exception as e:
        print(f"\n❌ FAILED: Exception during advantage computation")
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_belief_parser():
    """Test belief state parser"""
    print("\n" + "="*60)
    print("TEST 4: Belief State Parser")
    print("="*60)

    from agent_system.environments.env_package.alfworld.belief_tracker import BeliefStateParser

    parser = BeliefStateParser()

    # Test case 1: Valid JSON belief state
    valid_output = '''**Belief State Update:**
```json
{
  "world_model_update": {
    "found_objects": {"tomato_1": "countertop_1"},
    "state_changes": {},
    "cleared_receptacles": []
  },
  "task_progress_update": {
    "subgoal_status": "in_progress",
    "evidence": "Found tomato on countertop",
    "updated_subgoal": "Pick up tomato"
  },
  "exploration_map_update": {
    "newly_visited": ["kitchen"],
    "next_priority": ["fridge"]
  }
}
```

**Action:** take tomato_1 from countertop_1
'''

    belief_state, action, is_valid = parser.parse(valid_output)

    print(f"✓ Belief state parsed: {belief_state is not None}")
    print(f"✓ Action extracted: {action}")
    print(f"✓ Format valid: {is_valid}")

    if belief_state:
        print(f"✓ Subgoal: {belief_state.get('task_progress_update', {}).get('updated_subgoal', 'N/A')}")

    success = is_valid and belief_state is not None

    if success:
        print(f"\n✅ SUCCESS: Belief parser works correctly")
    else:
        print(f"\n❌ FAILED: Belief parser failed")

    return success


def test_reward_calculator():
    """Test ReBel reward calculator"""
    print("\n" + "="*60)
    print("TEST 5: ReBel Reward Calculator")
    print("="*60)

    from agent_system.environments.env_package.alfworld.belief_tracker import RebelRewardCalculator

    calc = RebelRewardCalculator()

    # Test belief state
    belief_state = {
        'world_model_update': {
            'found_objects': {'tomato_1': 'countertop_1'},
            'state_changes': {},
            'cleared_receptacles': []
        },
        'task_progress_update': {
            'subgoal_status': 'in_progress',
            'evidence': 'Found tomato',
            'updated_subgoal': 'Pick up tomato'
        },
        'exploration_map_update': {
            'newly_visited': ['kitchen'],
            'next_priority': ['fridge']
        }
    }

    prev_belief = None

    # Mock ground truth
    class MockGroundTruth:
        def is_correct_object_location(self, obj_id, loc):
            return True
        def check_state_change(self, obj_id, state):
            return True

    gt_tracker = MockGroundTruth()

    # Calculate rewards
    consistency = calc.calculate_consistency_reward(belief_state, gt_tracker)
    progress = calc.calculate_progress_reward(belief_state, prev_belief)
    exploration = calc.calculate_exploration_reward(belief_state, prev_belief)
    format_reward = calc.calculate_format_reward(is_format_valid=True, is_action_available=True)

    total = calc.calculate_total_reward(
        consistency_reward=consistency,
        progress_reward=progress,
        exploration_reward=exploration,
        format_reward=format_reward
    )

    print(f"✓ Consistency reward: {consistency:.4f}")
    print(f"✓ Progress reward: {progress:.4f}")
    print(f"✓ Exploration reward: {exploration:.4f}")
    print(f"✓ Format reward: {format_reward:.4f}")
    print(f"✓ Total intrinsic reward: {total:.4f}")

    # Check that rewards are computed
    success = total != 0

    if success:
        print(f"\n✅ SUCCESS: Reward calculator works correctly")
    else:
        print(f"\n❌ FAILED: Reward calculator issue")

    return success


def save_test_results(results, output_dir):
    """Save test results to file"""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save JSON
    with open(output_dir / "component_test_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    # Save markdown report
    with open(output_dir / "COMPONENT_TEST_REPORT.md", 'w') as f:
        f.write("# ReBel Component Test Report\n\n")
        f.write(f"**Test Date:** {results['timestamp']}\n\n")
        f.write(f"**Overall Status:** {'✅ PASSED' if results['all_passed'] else '❌ FAILED'}\n\n")

        f.write("## Test Results\n\n")
        f.write("| Test | Status |\n")
        f.write("|------|--------|\n")
        for test_name, passed in results['tests'].items():
            status = "✅ PASSED" if passed else "❌ FAILED"
            f.write(f"| {test_name} | {status} |\n")

        f.write("\n## Summary\n\n")
        passed_count = sum(results['tests'].values())
        total_count = len(results['tests'])
        f.write(f"- **Passed:** {passed_count}/{total_count}\n")
        f.write(f"- **Failed:** {total_count - passed_count}/{total_count}\n")


def main():
    print("\n" + "="*70)
    print("ReBel Component Test Suite")
    print("Quick validation without full training")
    print("="*70)

    # Run tests
    results = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'tests': {},
        'all_passed': True
    }

    tests = [
        ("Belief Canonicalization", test_belief_canonicalization),
        ("Belief Grouping", test_belief_grouping),
        ("Advantage Computation", test_advantage_computation),
        ("Belief Parser", test_belief_parser),
        ("Reward Calculator", test_reward_calculator),
    ]

    for test_name, test_func in tests:
        try:
            passed = test_func()
            results['tests'][test_name] = passed
            if not passed:
                results['all_passed'] = False
        except Exception as e:
            print(f"\n❌ EXCEPTION in {test_name}")
            print(f"Error: {str(e)}")
            import traceback
            traceback.print_exc()
            results['tests'][test_name] = False
            results['all_passed'] = False

    # Print final summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)

    for test_name, passed in results['tests'].items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{status}: {test_name}")

    print("\n" + "="*70)
    if results['all_passed']:
        print("✅ ALL TESTS PASSED - ReBel components are working correctly")
    else:
        print("❌ SOME TESTS FAILED - Please check the errors above")
    print("="*70 + "\n")

    # Save results
    output_dir = Path("rebel_test_results") / datetime.now().strftime("%Y%m%d_%H%M%S")
    save_test_results(results, output_dir)

    print(f"Results saved to: {output_dir}/")
    print(f"  - component_test_results.json")
    print(f"  - COMPONENT_TEST_REPORT.md")

    return 0 if results['all_passed'] else 1


if __name__ == "__main__":
    sys.exit(main())
