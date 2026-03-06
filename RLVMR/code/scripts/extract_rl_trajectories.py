#!/usr/bin/env python3
"""
从RL训练日志中提取轨迹数据，转换为Trajectory-Tracer格式
支持按任务分组，并标注每步reward和总reward
"""
import re
import json
import argparse
import os
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Any


def parse_log_response_blocks(log_path: str) -> List[Dict]:
    """从训练日志中提取所有response/score对"""
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    # 去除ANSI颜色码
    ansi_escape = re.compile(r'\x1B\[[0-9;]*m')
    content = ansi_escape.sub('', content)
    # 去除(TaskRunner pid=xxx)前缀
    content = re.sub(r'\(TaskRunner pid=\d+\)\s*', '', content)

    # 提取[response]...[score]块
    # 日志格式: [response] <belief>...<action>click[X]</action> 然后 [score] 0.xxx
    response_pattern = re.compile(
        r'\[response\]\s*(.*?)\n\[score\]\s*([\-\d.]+)',
        re.DOTALL
    )
    blocks = []
    for m in response_pattern.finditer(content):
        response_text = m.group(1).strip()
        score = float(m.group(2))
        blocks.append({'response': response_text, 'score': score})

    return blocks


def parse_step_metrics(log_path: str) -> List[Dict]:
    """从训练日志提取每个step的宏观指标"""
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    ansi_escape = re.compile(r'\x1B\[[0-9;]*m')
    content = ansi_escape.sub('', content)
    content = re.sub(r'\(TaskRunner pid=\d+\)\s*', '', content)

    step_pattern = re.compile(
        r'step:(\d+)\s+-\s+.*?episode/reward/mean:([\-\d.]+).*?'
        r'valid_action_ratio:([\d.]+).*?'
        r'response_length/mean:([\d.]+).*?'
        r'actor/entropy_loss:([\d.]+).*?'
        r'episode/success_rate:([\d.]+)',
        re.DOTALL
    )
    metrics = []
    for m in step_pattern.finditer(content):
        metrics.append({
            'step': int(m.group(1)),
            'reward_mean': float(m.group(2)),
            'valid_action_ratio': float(m.group(3)),
            'response_length_mean': float(m.group(4)),
            'entropy_loss': float(m.group(5)),
            'success_rate': float(m.group(6)),
        })
    return metrics


def parse_action(response: str) -> str:
    """从response中提取action"""
    action_match = re.search(r'<action>\s*(.*?)\s*</action>', response, re.DOTALL)
    if action_match:
        return action_match.group(1).strip()
    return ''


def parse_reasoning(response: str) -> str:
    """从response中提取reasoning"""
    m = re.search(r'<reasoning>\s*(.*?)\s*</reasoning>', response, re.DOTALL)
    if m:
        return m.group(1).strip()
    return ''


def parse_belief(response: str) -> str:
    """从response中提取belief"""
    m = re.search(r'<belief>\s*(.*?)\s*</belief>', response, re.DOTALL)
    if m:
        return m.group(1).strip()
    return ''


def extract_task_from_prompt(prompt_text: str) -> str:
    """从prompt中提取任务描述"""
    if 'Task:' in prompt_text:
        start = prompt_text.find('Task:') + 5
        end = prompt_text.find('\n', start)
        if end > start:
            return prompt_text[start:end].strip()
    return ''


def build_trajectories_from_log(log_path: str, max_trajs: int = 200) -> List[Dict]:
    """
    从训练日志提取轨迹
    返回Trajectory-Tracer格式的列表
    """
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    ansi_escape = re.compile(r'\x1B\[[0-9;]*m')
    content = ansi_escape.sub('', content)
    content = re.sub(r'\(TaskRunner pid=\d+\)\s*', '', content)

    # 提取task + obs + response + score 四元组
    # 格式: Task: ... Observation: ... [response] ... [score] ...
    # 每个step包含一个(task, obs, response, score)

    # 先找所有的[response]...[score]块，同时找其前面的prompt
    full_pattern = re.compile(
        r'Task:\s*(.*?)\n.*?Observation:\s*(.*?)\n.*?\[response\]\s*(.*?)\n\[score\]\s*([\-\d.]+)',
        re.DOTALL
    )

    raw_steps = []
    for m in full_pattern.finditer(content):
        task = m.group(1).strip()
        obs = m.group(2).strip()[:300]  # 截断
        response = m.group(3).strip()
        score = float(m.group(4))
        raw_steps.append({
            'task': task,
            'obs': obs,
            'response': response,
            'score': score,
            'action': parse_action(response),
            'reasoning': parse_reasoning(response),
            'belief': parse_belief(response),
        })

    if not raw_steps:
        print("未能从日志中提取到完整的四元组（task+obs+response+score）")
        print("尝试仅提取response+score对...")
        blocks = parse_log_response_blocks(log_path)
        print(f"提取到 {len(blocks)} 个response/score对")

        # 构造简化格式
        trajectories = []
        for i, block in enumerate(blocks[:max_trajs]):
            traj = {
                "task": f"WebShop Task #{i+1}",
                "done": "True" if block['score'] > 0 else "False",
                "total_reward": block['score'],
                "group_id": f"group_{i//8}",  # 每8条一组（rollout_n=8）
                "traj_index": i % 8,
                "data": [{
                    "step": 1,
                    "obs": "",
                    "response": block['response'],
                    "reward": block['score'],
                    "action": parse_action(block['response']),
                    "reasoning": parse_reasoning(block['response']),
                    "belief": parse_belief(block['response']),
                }]
            }
            trajectories.append(traj)
        return trajectories

    # 按任务分组，同一任务的不同rollout放在一起
    task_groups = defaultdict(list)
    for i, step in enumerate(raw_steps):
        task_key = step['task'][:80]  # 用任务前80字符作为key
        task_groups[task_key].append({'step_idx': i, **step})

    trajectories = []
    traj_id = 0
    for task_key, steps in list(task_groups.items())[:max_trajs]:
        # 每个task可能有多条rollout（rollout_n=8）
        # 将同一任务的所有steps视为多条轨迹
        rollout_n = 8  # 默认

        for rollout_idx, step in enumerate(steps):
            traj = {
                "id": f"rl_traj_{traj_id:05d}",
                "task": step['task'],
                "done": "True" if step['score'] > 0 else "False",
                "total_reward": step['score'],
                "group_id": task_key[:50],  # 同一任务的轨迹共享group_id
                "traj_index": rollout_idx,
                "data": [{
                    "step": 1,
                    "obs": step['obs'],
                    "response": step['response'],
                    "reward": step['score'],
                    "action": step['action'],
                    "reasoning": step['reasoning'],
                    "belief": step['belief'],
                }]
            }
            trajectories.append(traj)
            traj_id += 1

    return trajectories


def convert_to_webshop_rl_jsonl(trajectories: List[Dict], output_path: str):
    """
    转换为Trajectory-Tracer的WebShop RL格式JSONL
    格式包含group_id以支持同任务分组可视化
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for traj in trajectories:
            f.write(json.dumps(traj, ensure_ascii=False) + '\n')

    print(f"已保存 {len(trajectories)} 条轨迹到: {output_path}")


def print_analysis(trajectories: List[Dict], step_metrics: List[Dict]):
    """打印分析报告"""
    print("\n" + "="*60)
    print("RL训练轨迹深度分析报告")
    print("="*60)

    # 基本统计
    rewards = [t['total_reward'] for t in trajectories]
    success = [t for t in trajectories if t['done'] == 'True']
    print(f"\n【轨迹统计】")
    print(f"  总轨迹数: {len(trajectories)}")
    print(f"  成功轨迹: {len(success)} ({100*len(success)/len(trajectories):.1f}%)")
    print(f"  Reward均值: {sum(rewards)/len(rewards):.4f}")
    print(f"  Reward范围: [{min(rewards):.4f}, {max(rewards):.4f}]")

    # action分布
    all_actions = [step['action'] for t in trajectories for step in t['data'] if step.get('action')]
    action_types = defaultdict(int)
    for a in all_actions:
        if a.startswith('search['):
            action_types['search'] += 1
        elif a.startswith('click['):
            clicked = a[6:-1].lower()
            if 'buy now' in clicked:
                action_types['click[Buy Now]'] += 1
            elif 'back' in clicked:
                action_types['click[Back]'] += 1
            else:
                action_types['click[product]'] += 1
        else:
            action_types['other'] += 1

    print(f"\n【Action分布】（共 {len(all_actions)} 个动作）")
    for action_type, count in sorted(action_types.items(), key=lambda x: -x[1]):
        pct = 100 * count / len(all_actions) if all_actions else 0
        print(f"  {action_type}: {count} ({pct:.1f}%)")

    # 核心问题定位
    buy_now_count = action_types.get('click[Buy Now]', 0)
    print(f"\n【核心问题定位】")
    if buy_now_count == 0:
        print("  ❌ 致命问题：模型从未执行 click[Buy Now]！")
        print("     → reward永远为负，因为WebShop只在购买后给正reward")
        print("     → 这是reward为0的根本原因")

    search_pct = 100 * action_types.get('search', 0) / len(all_actions) if all_actions else 0
    if search_pct > 50:
        print(f"  ⚠️  模型过度搜索: {search_pct:.1f}%的动作是search")
        print(f"     → 可能存在搜索循环，模型不知道如何从搜索结果进入购买流程")

    # step_metrics分析
    if step_metrics:
        print(f"\n【训练曲线分析】（共{len(step_metrics)}步）")

        # 找reward下降点
        for i in range(1, len(step_metrics)):
            if step_metrics[i]['valid_action_ratio'] < 0.1 and step_metrics[i-1]['valid_action_ratio'] > 0.5:
                print(f"  ⚠️  step {step_metrics[i]['step']}: valid_action_ratio 从 "
                      f"{step_metrics[i-1]['valid_action_ratio']:.2f} 骤降至 "
                      f"{step_metrics[i]['valid_action_ratio']:.2f}")
                print(f"     同时 response_length 从 {step_metrics[i-1]['response_length_mean']:.0f} "
                      f"升至 {step_metrics[i]['response_length_mean']:.0f}")
                print(f"     → 模型发生退化：输出超长重复文本，无有效action")

        early_rewards = [m['reward_mean'] for m in step_metrics[:10]]
        late_rewards = [m['reward_mean'] for m in step_metrics[-10:]]
        print(f"  前10步reward均值: {sum(early_rewards)/len(early_rewards):.4f}")
        print(f"  后10步reward均值: {sum(late_rewards)/len(late_rewards):.4f}")
        print(f"  趋势: {'⬆️ 上升' if sum(late_rewards) > sum(early_rewards) else '⬇️ 下降或持平'}")

    print(f"\n【建议修复方向】")
    print("  1. 检查模型是否能正确识别并输出 click[Buy Now] 动作格式")
    print("  2. 检查WebShop环境的reward函数——是否只有buy now才有正reward？")
    print("  3. 检查prompt template中available_actions是否正确传递了Buy Now选项")
    print("  4. 考虑增加中间步骤的正向reward shaping（点击商品页+0.05等）")
    print("  5. 检查repetition_penalty=1.2是否导致了action格式被破坏")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', type=str, required=True, help='训练日志路径')
    parser.add_argument('--output', type=str,
                        default='/root/testttt/RLVMR/Trajectory-Tracer/webshop_rl_traj/rl_trajectories.jsonl',
                        help='输出JSONL路径')
    parser.add_argument('--max_trajs', type=int, default=500, help='最大轨迹数量')
    args = parser.parse_args()

    print(f"从日志提取轨迹: {args.log}")
    trajectories = build_trajectories_from_log(args.log, args.max_trajs)
    step_metrics = parse_step_metrics(args.log)

    print_analysis(trajectories, step_metrics)

    # 保存
    convert_to_webshop_rl_jsonl(trajectories, args.output)

    # 同时保存step_metrics供前端分析
    metrics_path = args.output.replace('.jsonl', '_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(step_metrics, f, indent=2)
    print(f"训练指标已保存到: {metrics_path}")


if __name__ == '__main__':
    main()
