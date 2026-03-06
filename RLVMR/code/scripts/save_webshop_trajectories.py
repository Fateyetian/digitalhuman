#!/usr/bin/env python3
"""
WebShop RL轨迹保存脚本
用于保存RL训练过程中的轨迹数据，以便使用Trajectory-Tracer进行分析

用法:
    python save_webshop_trajectories.py --checkpoint <checkpoint_path> --output <output_dir> --num_samples <num>
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

# 添加项目路径
sys.path.insert(0, '/root/testttt/RLVMR/code')

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from agent_system.environments.env_package.webshop.envs import build_webshop_envs
from agent_system.environments.prompts.webshop_rebel_prompts import WEBSHOP_REBEL_TEMPLATE_NO_HIS_RL


def load_model_and_tokenizer(checkpoint_path: str, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"):
    """加载模型和tokenizer"""
    print(f"加载模型从: {checkpoint_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()

    return model, tokenizer


def parse_model_response(response: str) -> Dict[str, Any]:
    """解析模型输出，提取belief, reasoning, action"""
    import re

    result = {
        'belief': '',
        'reasoning': '',
        'action': ''
    }

    # 提取belief
    belief_match = re.search(r'<belief>(.*?)</belief>', response, re.DOTALL)
    if belief_match:
        result['belief'] = belief_match.group(1).strip()

    # 提取reasoning
    reasoning_match = re.search(r'<reasoning>(.*?)</reasoning>', response, re.DOTALL)
    if reasoning_match:
        result['reasoning'] = reasoning_match.group(1).strip()

    # 提取action
    action_match = re.search(r'<action>(.*?)</action>', response, re.DOTALL)
    if action_match:
        result['action'] = action_match.group(1).strip()

    return result


def format_prompt(task: str, obs: str, belief_state: str = None, available_actions: str = "N/A") -> str:
    """格式化输入提示"""
    if belief_state:
        return WEBSHOP_REBEL_TEMPLATE_NO_HIS_RL.format(
            task_description=task,
            observation=obs,
            belief_state=belief_state,
            available_actions=available_actions
        )
    else:
        # 初始状态，没有belief state
        return f"Task: {task}\n\nObservation:\n{obs}\n\nCurrent Belief State:\n{{}}\n\nAvailable Actions: {available_actions}"


def run_rollout(
    model,
    tokenizer,
    envs,
    tasks: List[str],
    max_steps: int = 15,
) -> List[Dict[str, Any]]:
    """运行rollout并收集轨迹"""
    trajectories = []

    for task_idx, task in enumerate(tasks):
        print(f"处理任务 {task_idx+1}/{len(tasks)}: {task[:50]}...")

        # 重置环境
        obs, infos = envs.reset()
        obs = obs[0]  # 单个环境
        info = infos[0]

        # 初始化belief state
        belief_state = {
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

        trajectory = {
            "task": task,
            "done": "False",
            "data": []
        }

        total_reward = 0
        step_rewards = []

        for step in range(max_steps):
            # 格式化prompt
            available_actions = info.get('available_actions', {})
            actions_text = "N/A"
            if available_actions:
                clickables = available_actions.get('clickables', [])
                if clickables:
                    actions_text = ", ".join(clickables[:10])

            prompt = format_prompt(task, obs, json.dumps(belief_state), actions_text)

            # 生成response
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
            inputs = {k: v.cuda() for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )

            response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

            # 解析response
            parsed = parse_model_response(response)

            # 执行action
            action = parsed['action']
            if action:
                # 提取action类型和参数
                import re
                action_match = re.match(r'(\w+)\[(.*?)\]', action)
                if action_match:
                    action_type = action_match.group(1)
                    action_arg = action_match.group(2)
                    obs, rewards, dones, infos = envs.step([action])
                    obs = obs[0]
                    reward = rewards[0]
                    done = dones[0]
                    info = infos[0]
                else:
                    # 尝试直接执行
                    obs, rewards, dones, infos = envs.step([action])
                    obs = obs[0]
                    reward = rewards[0]
                    done = dones[0]
                    info = infos[0]
            else:
                # 没有有效action，终止
                done = True
                reward = 0

            total_reward += reward
            step_rewards.append(reward)

            # 更新belief state
            if 'belief' in parsed and parsed['belief']:
                try:
                    belief_state = json.loads(parsed['belief'])
                except:
                    pass

            # 保存步骤数据
            trajectory["data"].append({
                "step": step + 1,
                "obs": obs[:500] if len(obs) > 500 else obs,  # 截断过长的obs
                "prompt": prompt[:500] if len(prompt) > 500 else prompt,
                "response": response[:1000] if len(response) > 1000 else response,
                "reward": reward,
                "parsed": parsed
            })

            if done:
                trajectory["done"] = "True"
                break

        trajectory["total_reward"] = total_reward
        trajectory["step_rewards"] = step_rewards
        trajectories.append(trajectory)

    return trajectories


def convert_to_trajectory_tracer_format(trajectories: List[Dict], output_path: str):
    """转换为Trajectory-Tracer格式"""
    # 读取已有的alfworld_expert_traj格式参考
    output_data = []

    for traj in trajectories:
        # 转换格式
        converted = {
            "task": traj["task"],
            "done": traj["done"],
            "data": []
        }

        for step_data in traj["data"]:
            converted["data"].append({
                "step": step_data["step"],
                "obs": step_data["obs"],
                "prompt": step_data["prompt"],
                "response": f"<belief>\n{step_data['parsed'].get('belief', '')}\n</belief>\n<reasoning>\n{step_data['parsed'].get('reasoning', '')}\n</reasoning>\n<action>\n{step_data['parsed'].get('action', '')}\n</action>"
            })

        output_data.append(converted)

    # 保存
    with open(output_path, 'w') as f:
        for item in output_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"已保存 {len(output_data)} 条轨迹到: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='保存WebShop RL轨迹')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型checkpoint路径')
    parser.add_argument('--output', type=str, default='/root/testttt/RLVMR/Trajectory-Tracer/webshop_rl_trajectories.jsonl', help='输出路径')
    parser.add_argument('--num_samples', type=int, default=10, help='采样数量')
    parser.add_argument('--max_steps', type=int, default=15, help='最大步数')

    args = parser.parse_args()

    # 加载模型
    model, tokenizer = load_model_and_tokenizer(args.checkpoint)

    # 构建环境
    print("构建WebShop环境...")
    envs = build_webshop_envs(
        seed=42,
        env_num=1,
        group_n=1,
        is_train=False
    )

    # 加载任务
    # 从WebShop数据集中采样任务
    import random
    random.seed(42)

    # 使用简单的任务列表
    tasks = [
        "Find me twin size, easy assemble bedframes with wood frame, storage space for living room, and price lower than 240.00 dollars",
        "Find me a coffee table under $100 with glass top and metal legs",
        "Find me a bookshelf with 5 shelves, made of bamboo, price under $80",
    ]

    # 如果需要更多任务，可以从WebShop数据集加载
    # 这里使用简单的测试任务

    # 运行rollout
    print(f"运行 {args.num_samples} 条轨迹...")
    trajectories = run_rollout(
        model=model,
        tokenizer=tokenizer,
        envs=envs,
        tasks=tasks * ((args.num_samples // len(tasks)) + 1),
        max_steps=args.max_steps
    )

    trajectories = trajectories[:args.num_samples]

    # 统计
    success_count = sum(1 for t in trajectories if t["done"] == "True")
    avg_reward = np.mean([t["total_reward"] for t in trajectories])

    print(f"\n统计:")
    print(f"  总轨迹数: {len(trajectories)}")
    print(f"  成功数: {success_count}")
    print(f"  平均reward: {avg_reward:.4f}")

    # 转换为Trajectory-Tracer格式并保存
    convert_to_trajectory_tracer_format(trajectories, args.output)

    # 同时保存详细版本
    detailed_path = args.output.replace('.jsonl', '_detailed.json')
    with open(detailed_path, 'w') as f:
        json.dump(trajectories, f, indent=2, ensure_ascii=False)
    print(f"详细轨迹已保存到: {detailed_path}")


if __name__ == "__main__":
    main()