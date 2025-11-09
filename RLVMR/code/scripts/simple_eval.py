#!/usr/bin/env python3
"""
Simple standalone evaluation script - No Ray required
Directly uses ALFWorld environments with vLLM inference
"""

import os
import sys
import argparse
import time
import numpy as np
from typing import Dict, List, Any
from pathlib import Path

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
    return parser.parse_args()


def build_prompt(task_desc: str, history: List[Dict[str, str]], current_obs: str) -> str:
    """Build prompt for the model"""

    # Initial system message
    prompt = """You are an autonomous intelligent agent tasked with navigating a home.
You will be given a household task. Your goal is to complete the task.

At each step, you will receive:
- Task description
- Observation of current environment state

You should provide:
- Action: A single command to execute

Format your response as follows:
Action: <your action here>

Available actions include:
- go to <receptacle>
- take <object> from <receptacle>
- put <object> in/on <receptacle>
- open <receptacle>
- close <receptacle>
- toggle <object>
- clean <object> with <receptacle>
- heat <object> with <receptacle>
- cool <object> with <receptacle>
- use <object>
- look
- inventory
- examine <object>

"""

    # Add task
    prompt += f"\nTask: {task_desc}\n\n"

    # Add history
    if history:
        prompt += "Previous interactions:\n"
        for turn in history[-5:]:  # Last 5 turns to keep context manageable
            prompt += f"Observation: {turn['obs']}\n"
            prompt += f"Action: {turn['action']}\n"
        prompt += "\n"

    # Current observation
    prompt += f"Current Observation: {current_obs}\n\n"
    prompt += "What is your action?"

    return prompt


def extract_action(text: str) -> str:
    """Extract action from model output"""
    text = text.strip()

    # Look for "Action:" prefix
    if "Action:" in text:
        action = text.split("Action:")[-1].strip()
    else:
        # Take first non-empty line
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        action = lines[0] if lines else text

    # Remove any trailing punctuation or quotes
    action = action.strip('."\'')

    return action


class SimpleEvaluator:
    """Simple evaluator without Ray"""

    def __init__(self, args):
        self.args = args

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
        self.envs = build_alfworld_envs(
            n_envs=args.num_tasks,
            seed=args.seed,
            generalization_level=args.generalization_level
        )
        print("Environments created!")

    def evaluate_single_task(self, env_id: int) -> Dict[str, Any]:
        """Evaluate a single task"""

        # Reset environment
        text_obs, image_obs, info = self.envs.reset(env_id)

        # Extract task description from first observation
        task_desc = text_obs.split("\n")[0] if "\n" in text_obs else "Complete the task"

        history = []
        episode_data = {
            "env_id": env_id,
            "task": task_desc,
            "steps": 0,
            "success": False,
            "reward": 0.0,
            "actions": [],
            "observations": [text_obs],
        }

        done = False
        step = 0
        total_reward = 0

        while not done and step < self.args.max_steps:
            # Build prompt
            prompt = build_prompt(task_desc, history, text_obs)

            # Generate action
            outputs = self.llm.generate([prompt], self.sampling_params)
            generated_text = outputs[0].outputs[0].text
            action = extract_action(generated_text)

            episode_data["actions"].append(action)

            # Execute action
            text_obs, image_obs, reward, done, info = self.envs.step([action], [env_id])
            text_obs = text_obs[0]  # Unpack from list
            reward = reward[0]
            done = done[0]
            info = info[0]

            # Update history
            history.append({"obs": text_obs, "action": action})
            episode_data["observations"].append(text_obs)

            total_reward += reward
            step += 1

            if self.args.verbose:
                print(f"  Step {step}: {action[:60]}...")
                if reward > 0:
                    print(f"    Reward: +{reward}")

        # Check success
        success = done and (total_reward > 0 or info.get("won", False))

        episode_data["steps"] = step
        episode_data["success"] = success
        episode_data["reward"] = total_reward

        return episode_data

    def evaluate(self) -> Dict[str, Any]:
        """Run evaluation on all tasks"""
        print(f"\nStarting evaluation on {self.args.num_tasks} tasks...")
        print("=" * 80)

        results = []
        success_count = 0
        total_steps = []
        total_rewards = []

        start_time = time.time()

        for env_id in range(self.args.num_tasks):
            print(f"\n[Task {env_id + 1}/{self.args.num_tasks}]")

            try:
                episode_data = self.evaluate_single_task(env_id)
                results.append(episode_data)

                # Update statistics
                if episode_data["success"]:
                    success_count += 1
                total_steps.append(episode_data["steps"])
                total_rewards.append(episode_data["reward"])

                # Print progress
                current_success_rate = success_count / (env_id + 1) * 100
                print(f"  Success: {episode_data['success']} | "
                      f"Steps: {episode_data['steps']} | "
                      f"Reward: {episode_data['reward']:.1f}")
                print(f"  Running Success Rate: {current_success_rate:.1f}%")

                # Log to SwanLab (per task)
                if self.run is not None:
                    swanlab.log({
                        "task/success": int(episode_data["success"]),
                        "task/steps": episode_data["steps"],
                        "task/reward": episode_data["reward"],
                        "task/running_success_rate": current_success_rate,
                    }, step=env_id)

            except Exception as e:
                print(f"  Error in task {env_id}: {e}")
                import traceback
                traceback.print_exc()

        elapsed_time = time.time() - start_time

        # Compute final metrics
        final_metrics = {
            "success_rate": success_count / self.args.num_tasks * 100,
            "avg_steps": np.mean(total_steps),
            "std_steps": np.std(total_steps),
            "avg_reward": np.mean(total_rewards),
            "total_tasks": self.args.num_tasks,
            "successful_tasks": success_count,
            "failed_tasks": self.args.num_tasks - success_count,
            "elapsed_time": elapsed_time,
        }

        # Print summary
        print("\n" + "=" * 80)
        print("EVALUATION SUMMARY")
        print("=" * 80)
        print(f"Success Rate:     {final_metrics['success_rate']:.2f}%")
        print(f"Successful Tasks: {final_metrics['successful_tasks']}/{final_metrics['total_tasks']}")
        print(f"Average Steps:    {final_metrics['avg_steps']:.2f} ± {final_metrics['std_steps']:.2f}")
        print(f"Average Reward:   {final_metrics['avg_reward']:.2f}")
        print(f"Elapsed Time:     {elapsed_time:.1f}s ({elapsed_time/self.args.num_tasks:.1f}s per task)")
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
            "results": results,
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
