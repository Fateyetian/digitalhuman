#!/usr/bin/env python3
"""
Simple standalone evaluation script - No Ray required
Directly uses ALFWorld environments with vLLM inference
Works with batch/vectorized environments

Uses the same prompt format as SFT training data (from rebel_prompts.py)
Saves trajectories to JSONL file for analysis
"""

import os
import sys

# Force vLLM to use V0 engine to avoid CUDA multiprocessing issues
os.environ["VLLM_USE_V1"] = "0"

import argparse
import time
import json
import numpy as np
from typing import Dict, List, Any
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

try:
    import swanlab
    SWANLAB_AVAILABLE = True
except ImportError:
    SWANLAB_AVAILABLE = False
    print("Warning: SwanLab not installed. Install with: pip install swanlab")

# Import ALFWorld env directly
from agent_system.environments.env_package.alfworld import build_alfworld_envs

# Import ReBel prompt templates (same as used in SFT training)
from agent_system.environments.prompts.rebel_prompts import (
    ALFWORLD_REBEL_TEMPLATE_NO_HIS_RL,
    ALFWORLD_REBEL_TEMPLATE_RL
)


def parse_args():
    parser = argparse.ArgumentParser(description="Simple ALFWorld evaluation")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--num_tasks", type=int, default=10,
                        help="Number of tasks to evaluate")
    parser.add_argument("--max_steps", type=int, default=30,
                        help="Maximum steps per episode")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (0.0 = greedy)")
    parser.add_argument("--max_tokens", type=int, default=512,
                        help="Maximum tokens to generate")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.6,
                        help="GPU memory utilization for vLLM")
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                        help="Tensor parallel size")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed")
    parser.add_argument("--project_name", type=str, default="RLVMR-Eval",
                        help="SwanLab project name")
    parser.add_argument("--experiment_name", type=str, default=None,
                        help="SwanLab experiment name")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed episode info")
    parser.add_argument("--generalization_level", type=int, default=0,
                        help="ALFWorld generalization level (0, 1, or 2)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save trajectories (default: ./results/eval_TIMESTAMP)")
    return parser.parse_args()


def build_prompt(task_desc: str, history: List[Dict[str, str]], current_obs: str,
                  step_count: int = 0, planning: str = "None",
                  admissible_actions: List[str] = None,
                  current_belief_state: str = None) -> str:
    """
    Build prompt for the model - ReBel format with admissible actions.

    ReBel方法不传递原始历史(observation/action)，而是通过current_belief_state传递累积状态。
    与coldstart训练数据格式一致：
    - action_history = "None"
    - step_count = 0
    - history_length = 2
    """
    # Format admissible actions
    if admissible_actions:
        actions_str = ", ".join(admissible_actions)
    else:
        actions_str = "look, inventory"

    if step_count == 0 or not history:
        # First step - use no history template
        prompt = ALFWORLD_REBEL_TEMPLATE_NO_HIS_RL.format(
            current_observation=current_obs,
            admissible_actions=actions_str
        )
    else:
        # Subsequent steps - matching coldstart format exactly
        # NO raw history, only belief state
        if current_belief_state is None:
            current_belief_state = "Update based on new observation."

        prompt = ALFWORLD_REBEL_TEMPLATE_RL.format(
            task_description=task_desc,
            step_count=0,  # Always 0 like coldstart
            history_length=2,  # Always 2 like coldstart
            action_history="None",  # Always None like coldstart
            current_step=1,  # Always 1 like coldstart
            current_observation=current_obs,
            admissible_actions=actions_str,
            current_belief_state=current_belief_state,
            planning=planning if planning else "None"
        )

    return prompt


def extract_belief_and_reasoning(text: str) -> tuple:
    """Extract belief state and reasoning from model output for next step."""
    import re

    belief_state = None
    reasoning = None

    # Extract belief
    belief_match = re.search(r'<belief>\s*(.*?)\s*</belief>', text, re.DOTALL | re.IGNORECASE)
    if belief_match:
        belief_state = belief_match.group(1).strip()

    # Extract reasoning (use as planning for next step)
    reasoning_match = re.search(r'<reasoning>\s*(.*?)\s*</reasoning>', text, re.DOTALL | re.IGNORECASE)
    if reasoning_match:
        reasoning = reasoning_match.group(1).strip()

    return belief_state, reasoning


def extract_action(text: str) -> str:
    """Extract action from model output (ReBel format)"""
    import re
    text = text.strip()

    # First try to extract from <action>...</action> tags (ReBel format)
    action_match = re.search(r'<action>\s*(.*?)\s*</action>', text, re.DOTALL | re.IGNORECASE)
    if action_match:
        action = action_match.group(1).strip()
        # Clean up the action - take first line if multiple lines
        action = action.split('\n')[0].strip()
        return action

    # Fallback: Look for "Action:" prefix
    if "Action:" in text:
        action = text.split("Action:")[-1].strip()
        action = action.split('\n')[0].strip()
        return action

    # Last resort: Take first non-empty line that looks like an action
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    for line in lines:
        # Skip lines that are clearly not actions
        if line.startswith('<') or line.startswith('{') or line.startswith('```'):
            continue
        if any(keyword in line.lower() for keyword in ['go to', 'take', 'put', 'open', 'close', 'use', 'heat', 'cool', 'clean']):
            return line.strip('."\'')

    # If nothing found, return first line
    return lines[0] if lines else "look"


class SimpleEvaluator:
    """Simple evaluator for batch environments"""

    def __init__(self, args):
        self.args = args
        self.num_envs = args.num_tasks

        # Initialize SwanLab
        if SWANLAB_AVAILABLE:
            exp_name = args.experiment_name or f"eval_{Path(args.model_path).name}"
            self.run = swanlab.init(
                project=args.project_name,
                experiment_name=exp_name,
                config={
                    "model_path": args.model_path,
                    "num_tasks": args.num_tasks,
                    "max_steps": args.max_steps,
                    "temperature": args.temperature,
                    "seed": args.seed,
                    "generalization_level": args.generalization_level,
                }
            )
            print(f"[SwanLab] Initialized: {exp_name}")
        else:
            self.run = None

        # Load model
        print(f"Loading model from {args.model_path}...")
        self.llm = LLM(
            model=args.model_path,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        self.sampling_params = SamplingParams(
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            top_p=1.0 if args.temperature == 0.0 else 0.95,
        )
        print("Model loaded!")

        # Create environments
        print(f"Creating {args.num_tasks} ALFWorld environments...")
        alf_config_path = os.path.join(
            os.path.dirname(__file__),
            '../agent_system/environments/env_package/alfworld/configs/config_tw.yaml'
        )
        self.envs = build_alfworld_envs(
            alf_config_path=alf_config_path,
            seed=args.seed,
            env_num=args.num_tasks,
            group_n=1,
            is_train=False
        )
        print("Environments created!")

        # Setup output directory for trajectories
        if args.output_dir:
            self.output_dir = Path(args.output_dir)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = Path(f"./results/eval_{timestamp}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.trajectory_file = self.output_dir / "trajectory.jsonl"
        print(f"Trajectories will be saved to: {self.trajectory_file}")

    def evaluate(self) -> Dict[str, Any]:
        """Run evaluation on all tasks in parallel"""
        print(f"\nStarting evaluation on {self.num_envs} tasks...")
        print("=" * 80)

        start_time = time.time()

        # Open trajectory file for writing
        traj_file = open(self.trajectory_file, 'w')

        # Reset all environments
        text_obs_list, image_obs_list, info_list = self.envs.reset()

        # Get initial admissible actions from info
        admissible_actions_list = []
        for info in info_list:
            if info and 'admissible_commands' in info:
                admissible_actions_list.append(info['admissible_commands'])
            else:
                admissible_actions_list.append(['look', 'inventory'])

        # Initialize tracking for each environment
        histories = [[] for _ in range(self.num_envs)]
        task_descs = []
        for obs in text_obs_list:
            # Extract task from "Your task is to: ..." line
            if "Your task is to:" in obs:
                task_desc = obs.split("Your task is to:")[-1].split("\n")[0].strip()
            else:
                task_desc = "Complete the task"
            task_descs.append(task_desc)

        episode_data = [{
            "env_id": i,
            "task": task_descs[i],
            "steps": 0,
            "success": False,
            "reward": 0.0,
            "actions": [],
            "observations": [text_obs_list[i]],
        } for i in range(self.num_envs)]

        dones = [False] * self.num_envs
        total_rewards = [0.0] * self.num_envs
        plannings = ["None"] * self.num_envs  # Track planning/reasoning for each env
        belief_states = [None] * self.num_envs  # Track belief state for each env

        # Run episodes
        for step in range(self.args.max_steps):
            # Build prompts for all active environments
            prompts = []
            active_indices = []
            for i in range(self.num_envs):
                if not dones[i]:
                    prompt = build_prompt(
                        task_desc=task_descs[i],
                        history=histories[i],
                        current_obs=text_obs_list[i],
                        step_count=step,
                        planning=plannings[i],
                        admissible_actions=admissible_actions_list[i],
                        current_belief_state=belief_states[i]
                    )
                    prompts.append(prompt)
                    active_indices.append(i)

            if not prompts:
                break  # All environments done

            # Generate actions for all active environments
            outputs = self.llm.generate(prompts, self.sampling_params)

            # Extract actions, belief states, and reasoning
            actions = ["look"] * self.num_envs  # Default action for done envs
            generated_texts = [""] * self.num_envs
            for j, idx in enumerate(active_indices):
                generated_text = outputs[j].outputs[0].text
                generated_texts[idx] = generated_text
                action = extract_action(generated_text)
                actions[idx] = action
                episode_data[idx]["actions"].append(action)

                # Extract belief and reasoning for next step
                belief, reasoning = extract_belief_and_reasoning(generated_text)
                if belief:
                    belief_states[idx] = belief
                if reasoning:
                    plannings[idx] = reasoning

            # Step all environments
            prev_obs_list = text_obs_list.copy()
            text_obs_list, image_obs_list, rewards, step_dones, infos = self.envs.step(actions)

            # Update admissible actions from new infos
            for i in range(self.num_envs):
                if infos[i] and 'admissible_commands' in infos[i]:
                    admissible_actions_list[i] = infos[i]['admissible_commands']

            # Save trajectory for each active environment
            for j, idx in enumerate(active_indices):
                traj_record = {
                    "env_id": idx,
                    "step": step,
                    "task": task_descs[idx],
                    "prompt": prompts[j][:2000],  # Truncate for readability
                    "model_output": generated_texts[idx],
                    "action_extracted": actions[idx],
                    "observation_before": prev_obs_list[idx][:500],
                    "observation_after": text_obs_list[idx][:500],
                    "reward": rewards[idx],
                    "done": step_dones[idx],
                    "won": infos[idx].get("won", False) if infos[idx] else False
                }
                traj_file.write(json.dumps(traj_record, ensure_ascii=False) + "\n")

            # Update tracking
            for i in range(self.num_envs):
                if not dones[i]:
                    histories[i].append({"obs": text_obs_list[i], "action": actions[i]})
                    episode_data[i]["observations"].append(text_obs_list[i])
                    total_rewards[i] += rewards[i]
                    episode_data[i]["steps"] = step + 1

                    if step_dones[i]:
                        dones[i] = True
                        success = total_rewards[i] > 0 or infos[i].get("won", False)
                        episode_data[i]["success"] = success
                        episode_data[i]["reward"] = total_rewards[i]

            # Print progress
            num_done = sum(dones)
            num_success = sum(1 for ep in episode_data if ep["success"])
            if self.args.verbose or (step + 1) % 5 == 0:
                print(f"Step {step + 1}: {num_done}/{self.num_envs} done, {num_success} success")

            if all(dones):
                break

        # Mark remaining as done
        for i in range(self.num_envs):
            if not dones[i]:
                episode_data[i]["reward"] = total_rewards[i]
                # Check if last info indicates success
                episode_data[i]["success"] = total_rewards[i] > 0

        elapsed_time = time.time() - start_time

        # Compute final metrics
        success_count = sum(1 for ep in episode_data if ep["success"])
        steps_list = [ep["steps"] for ep in episode_data]
        rewards_list = [ep["reward"] for ep in episode_data]

        final_metrics = {
            "success_rate": success_count / self.num_envs * 100,
            "avg_steps": np.mean(steps_list),
            "std_steps": np.std(steps_list),
            "avg_reward": np.mean(rewards_list),
            "total_tasks": self.num_envs,
            "successful_tasks": success_count,
            "failed_tasks": self.num_envs - success_count,
            "elapsed_time": elapsed_time,
        }

        # Close trajectory file
        traj_file.close()
        print(f"\nTrajectories saved to: {self.trajectory_file}")

        # Save summary
        summary_file = self.output_dir / "summary.json"
        with open(summary_file, 'w') as f:
            json.dump({
                "metrics": final_metrics,
                "config": {
                    "model_path": self.args.model_path,
                    "num_tasks": self.args.num_tasks,
                    "max_steps": self.args.max_steps,
                    "temperature": self.args.temperature,
                    "seed": self.args.seed,
                }
            }, f, indent=2)

        # Print summary
        print("\n" + "=" * 80)
        print("EVALUATION SUMMARY")
        print("=" * 80)
        print(f"Success Rate:     {final_metrics['success_rate']:.2f}%")
        print(f"Successful Tasks: {final_metrics['successful_tasks']}/{final_metrics['total_tasks']}")
        print(f"Average Steps:    {final_metrics['avg_steps']:.2f} ± {final_metrics['std_steps']:.2f}")
        print(f"Average Reward:   {final_metrics['avg_reward']:.2f}")
        print(f"Elapsed Time:     {elapsed_time:.1f}s ({elapsed_time/self.num_envs:.1f}s per task)")
        print(f"Trajectory File:  {self.trajectory_file}")
        print("=" * 80)

        # Log final metrics to SwanLab
        if self.run is not None:
            swanlab.log({
                "eval/success_rate": final_metrics["success_rate"],
                "eval/avg_steps": final_metrics["avg_steps"],
                "eval/std_steps": final_metrics["std_steps"],
                "eval/avg_reward": final_metrics["avg_reward"],
                "eval/successful_tasks": final_metrics["successful_tasks"],
                "eval/failed_tasks": final_metrics["failed_tasks"],
                "eval/elapsed_time": elapsed_time,
            })
            print(f"\n[SwanLab] Results logged!")
            print(f"[SwanLab] View at: https://swanlab.cn")

        return {
            "metrics": final_metrics,
            "results": episode_data,
        }

    def cleanup(self):
        """Cleanup resources"""
        if self.run is not None:
            swanlab.finish()
        print("\nEvaluation completed!")


def main():
    args = parse_args()

    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Run evaluation
    evaluator = SimpleEvaluator(args)
    try:
        results = evaluator.evaluate()
    finally:
        evaluator.cleanup()


if __name__ == "__main__":
    main()
