#!/usr/bin/env python3
"""
ReBel Rollout Evaluation Script
基于ReBel的Prompt与ALFWorld环境进行实际交互并记录分析
"""

import os
import sys
import time
import json
import logging
import argparse
from datetime import datetime
from types import SimpleNamespace
from typing import List
import numpy as np

# Add project to path
sys.path.insert(0, '/root/testttt/RLVMR/code')

from agent_system.environments.env_manager import *
from openai import OpenAI


def build_env_rebel(env_num=4, seed=1, history_length=2):
    """创建带ReBel配置的ALFWorld环境"""
    from agent_system.environments.env_package.alfworld import alfworld_projection_rebel
    from agent_system.environments.env_package.alfworld import build_alfworld_envs

    alf_config_path = os.path.join(
        os.path.dirname(__file__),
        'agent_system/environments/env_package/alfworld/configs/config_tw.yaml'
    )

    envs = build_alfworld_envs(
        alf_config_path,
        seed=seed,
        env_num=env_num,
        group_n=1,
        is_train=True
    )

    # ReBel配置
    cfg = SimpleNamespace(
        env=SimpleNamespace(
            env_name="alfworld/AlfredTWEnv",
            history_length=history_length,
            alfworld=SimpleNamespace(
                use_rebel=True,  # 启用ReBel
                meta_think=True
            )
        ),
        algorithm=SimpleNamespace(
            rebel=SimpleNamespace(
                enable=True,
                alpha=0.3,  # 一致性奖励权重
                beta=0.5,   # 进度奖励权重
                gamma=0.2,  # 探索奖励权重
                delta=0.1   # 格式奖励权重
            )
        )
    )

    env_manager = AlfWorldEnvironmentManager(
        envs,
        alfworld_projection_rebel,  # 使用ReBel projection
        "alfworld/AlfredTWEnv",
        cfg
    )

    print("✅ ReBel环境创建成功")
    print(f"   - 环境数量: {env_num}")
    print(f"   - ReBel启用: {cfg.algorithm.rebel.enable}")
    print(f"   - 奖励权重: α={cfg.algorithm.rebel.alpha}, β={cfg.algorithm.rebel.beta}, γ={cfg.algorithm.rebel.gamma}, δ={cfg.algorithm.rebel.delta}")

    return env_manager


class LocalModelAgent:
    """使用本地vLLM服务的Agent"""

    def __init__(self, base_url="http://127.0.0.1:8000/v1", model_name="/root/testttt/RLVMR/code/base_models/Qwen2.5-1.5B-Instruct", temperature=0.4):
        self.model_name = model_name
        self.temperature = temperature
        self.client = OpenAI(
            api_key="EMPTY",
            base_url=base_url,
        )
        print(f"✅ Agent初始化成功")
        print(f"   - 模型: {model_name}")
        print(f"   - URL: {base_url}")
        print(f"   - Temperature: {temperature}")

    def get_action(self, obs):
        """获取单个action"""
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
            print(f"❌ API调用失败: {e}")
            return "None"

    def get_actions_batch(self, prompts: List[str]) -> List[str]:
        """批量获取actions"""
        return [self.get_action(p) for p in prompts]


def run_rebel_rollout(
    env_num=4,
    max_steps=30,
    seed=1,
    base_url="http://127.0.0.1:8000/v1",
    model_name="/root/testttt/RLVMR/code/base_models/Qwen2.5-1.5B-Instruct",
    temperature=0.4,
    output_dir="rebel_rollout_results"
):
    """运行ReBel rollout评测"""

    # 创建输出目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(output_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    # 设置日志
    log_file = os.path.join(run_dir, "rollout.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler()
        ],
    )

    logging.info("="*60)
    logging.info("ReBel Rollout Evaluation")
    logging.info("="*60)
    logging.info(f"环境数量: {env_num}")
    logging.info(f"最大步数: {max_steps}")
    logging.info(f"随机种子: {seed}")
    logging.info(f"模型: {model_name}")
    logging.info(f"输出目录: {run_dir}")
    logging.info("="*60)

    # 创建环境和Agent
    env_manager = build_env_rebel(env_num=env_num, seed=seed)
    agent = LocalModelAgent(base_url=base_url, model_name=model_name, temperature=temperature)

    # 重置环境
    obs, infos = env_manager.reset()
    env_dones = [False] * env_num

    # 轨迹文件
    trajectory_file = os.path.join(run_dir, "trajectories.jsonl")
    traj_fp = open(trajectory_file, "w", encoding="utf-8")

    # 统计数据
    episode_lengths = np.zeros(env_num, dtype=int)
    episode_rewards = np.zeros(env_num, dtype=float)
    success_flags = np.zeros(env_num, dtype=bool)

    # ReBel指标
    rebel_consistency_sum = np.zeros(env_num, dtype=float)
    rebel_progress_sum = np.zeros(env_num, dtype=float)
    rebel_exploration_sum = np.zeros(env_num, dtype=float)
    rebel_format_sum = np.zeros(env_num, dtype=float)
    belief_parsed_count = np.zeros(env_num, dtype=int)
    belief_total_count = np.zeros(env_num, dtype=int)

    # 动作质量追踪（新增）
    action_history = [[] for _ in range(env_num)]
    repeat_action_count = np.zeros(env_num, dtype=int)
    invalid_action_count = np.zeros(env_num, dtype=int)
    total_action_count = np.zeros(env_num, dtype=int)

    # 主循环
    logging.info("\n开始Rollout...")
    start_time = time.time()

    for step in range(max_steps):
        # 准备prompts
        prompts = []
        idx_map = []
        for i in range(env_num):
            if not env_dones[i]:
                prompts.append(obs["text"][i])
                idx_map.append(i)

        if not prompts:
            logging.info("所有环境已完成！")
            break

        # 获取actions
        batch_actions = agent.get_actions_batch(prompts)
        actions = ["None"] * env_num
        for k, i in enumerate(idx_map):
            actions[i] = batch_actions[k]

        # 环境步进
        raw_actions = actions.copy()
        obs, rewards, dones, infos = env_manager.step(actions.copy())
        env_dones = [a or b for a, b in zip(env_dones, dones)]

        # 记录数据
        for i in range(env_num):
            if i in idx_map:  # 只记录活跃的环境
                # 基础指标
                episode_lengths[i] += 1
                episode_rewards[i] += rewards[i]

                # ReBel奖励分解
                rebel_rewards = infos[i].get("rebel_rewards", {})
                rebel_consistency_sum[i] += rebel_rewards.get("r_consistency", 0.0)
                rebel_progress_sum[i] += rebel_rewards.get("r_progress", 0.0)
                rebel_exploration_sum[i] += rebel_rewards.get("r_exploration", 0.0)
                rebel_format_sum[i] += rebel_rewards.get("r_format", 0.0)

                # 信念状态追踪
                belief_total_count[i] += 1
                belief_state = infos[i].get("belief_state")
                if belief_state is not None:
                    belief_parsed_count[i] += 1

                # 动作质量追踪（新增）
                total_action_count[i] += 1

                # 检测重复动作
                current_action = actions[i]
                if len(action_history[i]) > 0 and current_action == action_history[i][-1]:
                    repeat_action_count[i] += 1
                action_history[i].append(current_action)

                # 检测无效动作（通过观察中的错误信息判断）
                current_obs = obs["text"][i] if isinstance(obs, dict) else str(obs)
                is_invalid = (
                    "Nothing happens" in current_obs or
                    "not valid" in current_obs.lower() or
                    "cannot" in current_obs.lower() or
                    "You arrive at loc" not in current_obs  # 动作执行失败的标志
                )
                if is_invalid:
                    invalid_action_count[i] += 1

                # NEW: Convert ground truth state (convert sets to lists for JSON serialization)
                gt_state = infos[i].get("ground_truth_state")
                if gt_state:
                    gt_state_serializable = {
                        'visited': list(gt_state.get('visited', [])),
                        'object_locations': dict(gt_state.get('object_locations', {})),
                        'object_states': dict(gt_state.get('object_states', {})),
                        'cleared_receptacles': list(gt_state.get('cleared_receptacles', [])),
                        'current_inventory': gt_state.get('current_inventory'),
                        'interactions': list(gt_state.get('interactions', [])),
                        'task_goal': gt_state.get('task_goal')
                    }
                else:
                    gt_state_serializable = None

                # 写入轨迹
                row = {
                    "step": step,
                    "env_id": i,
                    "observation": text_obs[i],  # NEW: Original environment observation
                    "action": raw_actions[i],
                    "reward": float(rewards[i]),
                    "done": bool(dones[i]),
                    "rebel_rewards": {
                        "r_consistency": rebel_rewards.get("r_consistency", 0.0),
                        "r_progress": rebel_rewards.get("r_progress", 0.0),
                        "r_exploration": rebel_rewards.get("r_exploration", 0.0),
                        "r_format": rebel_rewards.get("r_format", 0.0),
                        "r_intrinsic_total": rebel_rewards.get("r_intrinsic_total", 0.0),
                    },
                    "belief_state_pred": belief_state,  # NEW: Model's predicted belief state
                    "belief_state_gt": gt_state_serializable,  # NEW: Ground truth state (serializable)
                    "belief_parsed": belief_state is not None,
                }
                traj_fp.write(json.dumps(row, ensure_ascii=False) + "\n")

                # 检查成功
                if dones[i]:
                    success_flags[i] = bool(infos[i].get("won", False))

        # 进度日志
        done_count = sum(env_dones)
        success_count = sum(success_flags)
        logging.info(
            f"Step {step:2d} | Done: {done_count}/{env_num} | "
            f"Success: {success_count}/{env_num} ({success_count/max(1,done_count)*100:.1f}%)"
        )

    traj_fp.close()
    elapsed = time.time() - start_time

    # ========== 结果分析 ==========
    logging.info("\n" + "="*60)
    logging.info("ReBel Rollout 结果分析")
    logging.info("="*60)

    # 基础指标
    success_rate = success_flags.mean()
    avg_length = episode_lengths[episode_lengths > 0].mean() if (episode_lengths > 0).any() else 0
    avg_reward = episode_rewards.mean()

    logging.info(f"\n📊 基础指标:")
    logging.info(f"   ✅ 成功率: {success_rate:.2%} ({success_flags.sum()}/{env_num})")
    logging.info(f"   📏 平均步数: {avg_length:.1f}")
    logging.info(f"   🏆 平均奖励: {avg_reward:.2f}")
    logging.info(f"   ⏱️  总用时: {elapsed:.1f}s")

    # ReBel指标
    logging.info(f"\n🧠 ReBel框架指标:")

    # 计算平均每步奖励
    avg_r_consistency = (rebel_consistency_sum / np.maximum(episode_lengths, 1)).mean()
    avg_r_progress = (rebel_progress_sum / np.maximum(episode_lengths, 1)).mean()
    avg_r_exploration = (rebel_exploration_sum / np.maximum(episode_lengths, 1)).mean()
    avg_r_format = (rebel_format_sum / np.maximum(episode_lengths, 1)).mean()

    logging.info(f"   📈 平均内在奖励 (每步):")
    logging.info(f"      - r_consistency: {avg_r_consistency:.4f}")
    logging.info(f"      - r_progress:    {avg_r_progress:.4f}")
    logging.info(f"      - r_exploration: {avg_r_exploration:.4f}")
    logging.info(f"      - r_format:      {avg_r_format:.4f}")
    logging.info(f"      - r_total:       {avg_r_consistency + avg_r_progress + avg_r_exploration + avg_r_format:.4f}")

    # 信念状态质量
    avg_belief_parse = (belief_parsed_count / np.maximum(belief_total_count, 1)).mean()
    logging.info(f"\n   🔍 信念状态质量:")
    logging.info(f"      - 解析成功率: {avg_belief_parse:.2%}")

    # 动作质量指标（新增）
    avg_repeat_rate = (repeat_action_count / np.maximum(total_action_count, 1)).mean()
    avg_invalid_rate = (invalid_action_count / np.maximum(total_action_count, 1)).mean()
    logging.info(f"\n   ⚙️  动作质量:")
    logging.info(f"      - 重复动作率: {avg_repeat_rate:.2%}")
    logging.info(f"      - 无效动作率: {avg_invalid_rate:.2%}")

    # 保存结果
    results = {
        "timestamp": timestamp,
        "config": {
            "env_num": env_num,
            "max_steps": max_steps,
            "seed": seed,
            "model": model_name,
        },
        "basic_metrics": {
            "success_rate": float(success_rate),
            "avg_episode_length": float(avg_length),
            "avg_episode_reward": float(avg_reward),
            "elapsed_time": float(elapsed),
        },
        "rebel_metrics": {
            "avg_r_consistency": float(avg_r_consistency),
            "avg_r_progress": float(avg_r_progress),
            "avg_r_exploration": float(avg_r_exploration),
            "avg_r_format": float(avg_r_format),
            "avg_belief_parse_rate": float(avg_belief_parse),
        },
        "action_quality_metrics": {  # 新增动作质量部分
            "avg_repeat_action_rate": float(avg_repeat_rate),
            "avg_invalid_action_rate": float(avg_invalid_rate),
        },
        "per_env": {
            "success": success_flags.tolist(),
            "lengths": episode_lengths.tolist(),
            "rewards": episode_rewards.tolist(),
        }
    }

    results_file = os.path.join(run_dir, "results.json")
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logging.info(f"\n📁 输出文件:")
    logging.info(f"   - 轨迹数据: {trajectory_file}")
    logging.info(f"   - 结果统计: {results_file}")
    logging.info(f"   - 运行日志: {log_file}")
    logging.info("="*60 + "\n")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReBel Rollout Evaluation")
    parser.add_argument("--env_num", type=int, default=4, help="环境数量")
    parser.add_argument("--max_steps", type=int, default=30, help="最大步数")
    parser.add_argument("--seed", type=int, default=1, help="随机种子")
    parser.add_argument("--base_url", default="http://127.0.0.1:8000/v1", help="vLLM服务地址")
    parser.add_argument("--model", default="/root/testttt/RLVMR/code/base_models/Qwen2.5-1.5B-Instruct", help="模型名称")
    parser.add_argument("--temperature", type=float, default=0.4, help="采样温度")
    parser.add_argument("--output_dir", default="rebel_rollout_results", help="输出目录")

    args = parser.parse_args()

    run_rebel_rollout(
        env_num=args.env_num,
        max_steps=args.max_steps,
        seed=args.seed,
        base_url=args.base_url,
        model_name=args.model,
        temperature=args.temperature,
        output_dir=args.output_dir,
    )
