#!/usr/bin/env python3
"""
ReBel V6 Detailed Evaluation Script
详细评测脚本，用于诊断模型性能问题

功能：
1. 记录评测集任务分布（按task_type）
2. 记录每个step的详细奖励计算（r_consistency, r_progress, r_exploration, r_format）
3. 记录失败轨迹的完整信息（观察、动作、belief状态、GT状态）
4. 输出详细的诊断报告
"""

import os
import sys
import time
import json
import logging
import argparse
from datetime import datetime
from types import SimpleNamespace
from typing import List, Dict, Any, Optional
from collections import defaultdict
import numpy as np

# Add project to path
sys.path.insert(0, '/root/testttt/RLVMR/code')

from openai import OpenAI


# ALFWorld task types
ALFWORLD_TASK_TYPES = [
    "pick_and_place",
    "pick_two_obj_and_place",
    "look_at_obj_in_light",
    "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "pick_clean_then_place_in_recep",
]


def extract_task_type_from_gamefile(gamefile: str) -> str:
    """Extract task type from gamefile path"""
    if not gamefile:
        return "unknown"
    for task in ALFWORLD_TASK_TYPES:
        if task in gamefile:
            return task
    return "unknown"


def build_env_rebel(env_num=4, seed=1, history_length=2, is_train=False):
    """Create ALFWorld environment with ReBel configuration"""
    from agent_system.environments.env_package.alfworld.projection import alfworld_projection_rebel
    from agent_system.environments.env_package.alfworld import build_alfworld_envs
    from agent_system.environments.env_manager import AlfWorldEnvironmentManager

    alf_config_path = os.path.join(
        '/root/testttt/RLVMR/code',
        'agent_system/environments/env_package/alfworld/configs/config_tw.yaml'
    )

    envs = build_alfworld_envs(
        alf_config_path,
        seed=seed,
        env_num=env_num,
        group_n=1,
        is_train=is_train  # Use test set for evaluation
    )

    # ReBel configuration
    cfg = SimpleNamespace(
        env=SimpleNamespace(
            env_name="alfworld/AlfredTWEnv",
            history_length=history_length,
            alfworld=SimpleNamespace(
                use_rebel=True,
                meta_think=True,
                prompt_template_type="explicit_task_type"
            )
        ),
        algorithm=SimpleNamespace(
            rebel=SimpleNamespace(
                enable=True,
                alpha=0.3,
                beta=0.5,
                gamma=0.2,
                delta=0.1
            )
        )
    )

    env_manager = AlfWorldEnvironmentManager(
        envs,
        alfworld_projection_rebel,
        "alfworld/AlfredTWEnv",
        cfg
    )

    return env_manager


class LocalModelAgent:
    """Agent using local vLLM service"""

    def __init__(self, base_url="http://127.0.0.1:8000/v1",
                 model_name="/root/testttt/RLVMR/code/base_models/Qwen2.5-1.5B-Instruct",
                 temperature=0.0):  # Use temperature=0 for deterministic evaluation
        self.model_name = model_name
        self.temperature = temperature
        self.client = OpenAI(
            api_key="EMPTY",
            base_url=base_url,
        )
        print(f"Agent initialized: {model_name}")

    def get_action(self, obs):
        """Get single action"""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": obs}],
                temperature=self.temperature,
                max_tokens=512,
                n=1,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"API call failed: {e}")
            return "None"

    def get_actions_batch(self, prompts: List[str]) -> List[str]:
        """Batch get actions"""
        return [self.get_action(p) for p in prompts]


def serialize_for_json(obj):
    """Convert objects to JSON-serializable format"""
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [serialize_for_json(item) for item in obj]
    if isinstance(obj, dict):
        return {str(k): serialize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, set):
        return list(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    return str(obj)


def run_detailed_evaluation(
    env_num=128,
    max_steps=30,
    seed=42,
    base_url="http://127.0.0.1:8000/v1",
    model_name="/root/testttt/RLVMR/code/base_models/Qwen2.5-1.5B-Instruct",
    temperature=0.0,
    output_dir="detailed_eval_results"
):
    """Run detailed evaluation with comprehensive logging"""

    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(output_dir, f"eval_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    # Setup logging
    log_file = os.path.join(run_dir, "evaluation.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler()
        ],
        force=True
    )

    logging.info("=" * 70)
    logging.info("ReBel V6 Detailed Evaluation")
    logging.info("=" * 70)
    logging.info(f"Environment count: {env_num}")
    logging.info(f"Max steps: {max_steps}")
    logging.info(f"Random seed: {seed}")
    logging.info(f"Model: {model_name}")
    logging.info(f"Temperature: {temperature}")
    logging.info(f"Output directory: {run_dir}")
    logging.info("=" * 70)

    # Create environment and agent
    env_manager = build_env_rebel(env_num=env_num, seed=seed, is_train=False)
    agent = LocalModelAgent(base_url=base_url, model_name=model_name, temperature=temperature)

    # Reset environment
    obs, infos = env_manager.reset()
    env_dones = [False] * env_num

    # Extract task types from gamefiles
    task_types = []
    gamefiles = []
    for i in range(env_num):
        gamefile = infos[i].get('extra.gamefile', '') or env_manager.gamefile[i] if hasattr(env_manager, 'gamefile') else ''
        gamefiles.append(gamefile)
        task_type = extract_task_type_from_gamefile(gamefile)
        task_types.append(task_type)

    # Log task distribution
    task_distribution = defaultdict(int)
    for t in task_types:
        task_distribution[t] += 1

    logging.info("\n" + "=" * 70)
    logging.info("TASK DISTRIBUTION (Evaluation Set)")
    logging.info("=" * 70)
    for task, count in sorted(task_distribution.items()):
        pct = count / env_num * 100
        logging.info(f"  {task}: {count} ({pct:.1f}%)")
    logging.info("")

    # Detailed trajectory storage
    trajectories = {i: {
        'task_type': task_types[i],
        'gamefile': gamefiles[i],
        'initial_obs': obs['anchor'][i] if isinstance(obs, dict) else obs[i],
        'steps': [],
        'success': False,
        'total_reward': 0.0,
        'total_steps': 0
    } for i in range(env_num)}

    # Statistics
    episode_lengths = np.zeros(env_num, dtype=int)
    episode_rewards = np.zeros(env_num, dtype=float)
    success_flags = np.zeros(env_num, dtype=bool)

    # Per-step reward tracking
    step_rewards = {i: {
        'r_consistency': [],
        'r_progress': [],
        'r_exploration': [],
        'r_format': [],
        'r_intrinsic_total': [],
        'task_reward': []
    } for i in range(env_num)}

    # Main loop
    logging.info("\nStarting evaluation...")
    start_time = time.time()

    for step in range(max_steps):
        # Prepare prompts for active environments
        prompts = []
        idx_map = []
        for i in range(env_num):
            if not env_dones[i]:
                prompts.append(obs["text"][i])
                idx_map.append(i)

        if not prompts:
            logging.info("All environments done!")
            break

        # Get actions
        batch_actions = agent.get_actions_batch(prompts)
        actions = ["None"] * env_num
        for k, i in enumerate(idx_map):
            actions[i] = batch_actions[k]

        # Store raw model outputs
        raw_outputs = actions.copy()

        # Environment step
        prev_obs = obs.copy()
        obs, rewards, dones, infos = env_manager.step(actions.copy())
        env_dones = [a or b for a, b in zip(env_dones, dones)]

        # Record data for each environment
        for i in range(env_num):
            if i in idx_map:
                # Basic metrics
                episode_lengths[i] += 1
                episode_rewards[i] += rewards[i]

                # Extract ReBel rewards
                rebel_rewards = infos[i].get("rebel_rewards", {})
                r_consistency = rebel_rewards.get("r_consistency", 0.0)
                r_progress = rebel_rewards.get("r_progress", 0.0)
                r_exploration = rebel_rewards.get("r_exploration", 0.0)
                r_format = rebel_rewards.get("r_format", 0.0)
                r_intrinsic = rebel_rewards.get("r_intrinsic_total", 0.0)

                # Task reward (10.0 if won)
                task_reward = 10.0 if infos[i].get('won', False) else 0.0

                # Store step rewards
                step_rewards[i]['r_consistency'].append(r_consistency)
                step_rewards[i]['r_progress'].append(r_progress)
                step_rewards[i]['r_exploration'].append(r_exploration)
                step_rewards[i]['r_format'].append(r_format)
                step_rewards[i]['r_intrinsic_total'].append(r_intrinsic)
                step_rewards[i]['task_reward'].append(task_reward)

                # Get belief and ground truth states
                belief_state = infos[i].get("belief_state")
                gt_state = infos[i].get("ground_truth_state", {})

                # Record step in trajectory
                step_data = {
                    "step": step,
                    "prompt": prev_obs["text"][i][:500] + "..." if len(prev_obs["text"][i]) > 500 else prev_obs["text"][i],
                    "model_output": raw_outputs[i],
                    "extracted_action": infos[i].get("action", actions[i]) if isinstance(infos[i], dict) else actions[i],
                    "observation_after": obs["anchor"][i] if isinstance(obs, dict) and "anchor" in obs else "",
                    "reward": float(rewards[i]),
                    "done": bool(dones[i]),
                    "rebel_rewards": {
                        "r_consistency": float(r_consistency),
                        "r_progress": float(r_progress),
                        "r_exploration": float(r_exploration),
                        "r_format": float(r_format),
                        "r_intrinsic_total": float(r_intrinsic)
                    },
                    "belief_state_pred": serialize_for_json(belief_state),
                    "ground_truth_state": serialize_for_json(gt_state),
                    "is_action_valid": int(infos[i].get('is_action_valid', 0)),
                    "action_available": bool(infos[i].get('action_available', False))
                }
                trajectories[i]['steps'].append(step_data)
                trajectories[i]['total_reward'] += float(rewards[i])

                # Check success
                if dones[i]:
                    success_flags[i] = bool(infos[i].get("won", False))
                    trajectories[i]['success'] = success_flags[i]
                    trajectories[i]['total_steps'] = episode_lengths[i]

        # Progress log
        done_count = sum(env_dones)
        success_count = sum(success_flags)
        logging.info(
            f"Step {step:2d} | Done: {done_count}/{env_num} | "
            f"Success: {success_count}/{env_num} ({success_count / max(1, done_count) * 100:.1f}%)"
        )

    elapsed = time.time() - start_time

    # ========== Results Analysis ==========
    logging.info("\n" + "=" * 70)
    logging.info("EVALUATION RESULTS ANALYSIS")
    logging.info("=" * 70)

    # Basic metrics
    success_rate = success_flags.mean()
    avg_length = episode_lengths[episode_lengths > 0].mean() if (episode_lengths > 0).any() else 0
    avg_reward = episode_rewards.mean()

    logging.info(f"\n[Basic Metrics]")
    logging.info(f"  Success Rate: {success_rate:.2%} ({success_flags.sum()}/{env_num})")
    logging.info(f"  Average Steps: {avg_length:.1f}")
    logging.info(f"  Average Reward: {avg_reward:.2f}")
    logging.info(f"  Elapsed Time: {elapsed:.1f}s")

    # Per-task-type analysis
    logging.info(f"\n[Per-Task-Type Success Rate]")
    task_success = defaultdict(list)
    task_rewards = defaultdict(list)
    for i in range(env_num):
        task_success[task_types[i]].append(success_flags[i])
        task_rewards[task_types[i]].append(episode_rewards[i])

    for task in sorted(task_success.keys()):
        success_list = task_success[task]
        reward_list = task_rewards[task]
        sr = np.mean(success_list)
        ar = np.mean(reward_list)
        logging.info(f"  {task}: {sr:.2%} ({sum(success_list)}/{len(success_list)}) | Avg Reward: {ar:.2f}")

    # Average step rewards analysis
    logging.info(f"\n[Average Step Rewards Across All Environments]")
    all_r_consistency = []
    all_r_progress = []
    all_r_exploration = []
    all_r_format = []
    all_r_intrinsic = []

    for i in range(env_num):
        all_r_consistency.extend(step_rewards[i]['r_consistency'])
        all_r_progress.extend(step_rewards[i]['r_progress'])
        all_r_exploration.extend(step_rewards[i]['r_exploration'])
        all_r_format.extend(step_rewards[i]['r_format'])
        all_r_intrinsic.extend(step_rewards[i]['r_intrinsic_total'])

    if all_r_consistency:
        logging.info(f"  r_consistency: mean={np.mean(all_r_consistency):.6f}, std={np.std(all_r_consistency):.6f}")
        logging.info(f"  r_progress:    mean={np.mean(all_r_progress):.6f}, std={np.std(all_r_progress):.6f}")
        logging.info(f"  r_exploration: mean={np.mean(all_r_exploration):.6f}, std={np.std(all_r_exploration):.6f}")
        logging.info(f"  r_format:      mean={np.mean(all_r_format):.6f}, std={np.std(all_r_format):.6f}")
        logging.info(f"  r_intrinsic:   mean={np.mean(all_r_intrinsic):.6f}, std={np.std(all_r_intrinsic):.6f}")

    # Failed trajectories analysis
    failed_envs = [i for i in range(env_num) if not success_flags[i]]
    logging.info(f"\n[Failed Trajectories Analysis]")
    logging.info(f"  Total failed: {len(failed_envs)}/{env_num}")

    # Categorize failures by task type
    failed_by_task = defaultdict(list)
    for i in failed_envs:
        failed_by_task[task_types[i]].append(i)

    for task in sorted(failed_by_task.keys()):
        count = len(failed_by_task[task])
        total = len(task_success[task])
        logging.info(f"  {task}: {count}/{total} failed")

    # Save detailed results
    results = {
        "timestamp": timestamp,
        "config": {
            "env_num": env_num,
            "max_steps": max_steps,
            "seed": seed,
            "model": model_name,
            "temperature": temperature
        },
        "task_distribution": dict(task_distribution),
        "basic_metrics": {
            "success_rate": float(success_rate),
            "avg_episode_length": float(avg_length),
            "avg_episode_reward": float(avg_reward),
            "elapsed_time": float(elapsed)
        },
        "per_task_metrics": {
            task: {
                "success_rate": float(np.mean(task_success[task])),
                "count": len(task_success[task]),
                "successes": int(sum(task_success[task])),
                "avg_reward": float(np.mean(task_rewards[task]))
            }
            for task in task_success.keys()
        },
        "avg_step_rewards": {
            "r_consistency": float(np.mean(all_r_consistency)) if all_r_consistency else 0.0,
            "r_progress": float(np.mean(all_r_progress)) if all_r_progress else 0.0,
            "r_exploration": float(np.mean(all_r_exploration)) if all_r_exploration else 0.0,
            "r_format": float(np.mean(all_r_format)) if all_r_format else 0.0,
            "r_intrinsic": float(np.mean(all_r_intrinsic)) if all_r_intrinsic else 0.0
        },
        "per_env_summary": {
            i: {
                "task_type": task_types[i],
                "success": bool(success_flags[i]),
                "steps": int(episode_lengths[i]),
                "total_reward": float(episode_rewards[i])
            }
            for i in range(env_num)
        }
    }

    results_file = os.path.join(run_dir, "results.json")
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Save all trajectories
    all_traj_file = os.path.join(run_dir, "all_trajectories.jsonl")
    with open(all_traj_file, "w", encoding="utf-8") as f:
        for i in range(env_num):
            f.write(json.dumps(serialize_for_json(trajectories[i]), ensure_ascii=False) + "\n")

    # Save failed trajectories separately with full detail
    failed_traj_file = os.path.join(run_dir, "failed_trajectories.jsonl")
    with open(failed_traj_file, "w", encoding="utf-8") as f:
        for i in failed_envs:
            traj = trajectories[i]
            traj['env_id'] = i
            # Add step rewards for analysis
            traj['step_rewards'] = step_rewards[i]
            f.write(json.dumps(serialize_for_json(traj), ensure_ascii=False) + "\n")

    # Save step rewards for detailed analysis
    step_rewards_file = os.path.join(run_dir, "step_rewards.json")
    with open(step_rewards_file, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in step_rewards.items()}, f, indent=2)

    logging.info(f"\n[Output Files]")
    logging.info(f"  Results:            {results_file}")
    logging.info(f"  All Trajectories:   {all_traj_file}")
    logging.info(f"  Failed Trajectories: {failed_traj_file}")
    logging.info(f"  Step Rewards:       {step_rewards_file}")
    logging.info(f"  Log:                {log_file}")
    logging.info("=" * 70 + "\n")

    # Print sample failed trajectory for quick analysis
    if failed_envs:
        sample_env = failed_envs[0]
        logging.info("\n" + "=" * 70)
        logging.info(f"SAMPLE FAILED TRAJECTORY (env_id={sample_env}, task_type={task_types[sample_env]})")
        logging.info("=" * 70)

        traj = trajectories[sample_env]
        logging.info(f"Initial Observation: {traj['initial_obs'][:200]}...")
        logging.info(f"Total Steps: {len(traj['steps'])}")
        logging.info("")

        for step_data in traj['steps'][:5]:  # Show first 5 steps
            logging.info(f"--- Step {step_data['step']} ---")
            logging.info(f"Model Output: {step_data['model_output'][:300]}...")
            logging.info(f"Observation After: {step_data['observation_after'][:200]}...")
            logging.info(f"Rewards: {step_data['rebel_rewards']}")
            logging.info(f"Action Valid: {step_data['is_action_valid']}, Available: {step_data['action_available']}")
            logging.info("")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReBel V6 Detailed Evaluation")
    parser.add_argument("--env_num", type=int, default=128, help="Number of environments")
    parser.add_argument("--max_steps", type=int, default=30, help="Max steps per episode")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--base_url", default="http://127.0.0.1:8000/v1", help="vLLM service URL")
    parser.add_argument("--model", default="/root/testttt/RLVMR/code/base_models/Qwen2.5-1.5B-Instruct",
                       help="Model name/path")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--output_dir", default="detailed_eval_results", help="Output directory")

    args = parser.parse_args()

    run_detailed_evaluation(
        env_num=args.env_num,
        max_steps=args.max_steps,
        seed=args.seed,
        base_url=args.base_url,
        model_name=args.model,
        temperature=args.temperature,
        output_dir=args.output_dir,
    )
