#!/usr/bin/env python3
"""
WebShop RL训练轨迹分析与保存工具
用于分析RL训练中的问题，保存轨迹数据以便使用Trajectory-Tracer可视化
"""

import json
import os
import re
import numpy as np
from collections import defaultdict
from typing import List, Dict, Any
import argparse


def parse_training_log(log_path: str) -> Dict[str, Any]:
    """从训练日志中提取关键指标"""
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    # 提取episode reward
    reward_pattern = r'episode/reward/mean:([\-\d.]+)'
    rewards = re.findall(reward_pattern, content)

    # 提取success rate
    success_pattern = r'episode/success_rate:([\d.]+)'
    success_rates = re.findall(success_pattern, content)

    # 提取episode length
    length_pattern = r'episode/length/mean:([\d.]+)'
    lengths = re.findall(length_pattern, content)

    # 提取critic score
    critic_pattern = r'critic/score/mean:([\-\d.]+)'
    critic_scores = re.findall(critic_pattern, content)

    # 提取advantage
    adv_pattern = r'critic/advantages/mean:([\-\d.]+)'
    advantages = re.findall(adv_pattern, content)

    # 提取entropy
    entropy_pattern = r'actor/entropy_loss:([\d.]+)'
    entropies = re.findall(entropy_pattern, content)

    # 提取step信息
    step_pattern = r'step:(\d+)'
    steps = re.findall(step_pattern, content)

    return {
        'rewards': [float(x) for x in rewards],
        'success_rates': [float(x) for x in success_rates],
        'lengths': [float(x) for x in lengths],
        'critic_scores': [float(x) for x in critic_scores],
        'advantages': [float(x) for x in advantages],
        'entropies': [float(x) for x in entropies],
        'steps': [int(x) for x in steps]
    }


def analyze_reward_trend(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """分析reward趋势"""
    rewards = metrics['rewards']

    if not rewards:
        return {'error': 'No reward data found'}

    # 分成前中后三段分析
    n = len(rewards)
    if n < 10:
        return {
            'total_steps': n,
            'reward_mean': np.mean(rewards),
            'reward_std': np.std(rewards),
            'reward_trend': 'insufficient_data'
        }

    early = rewards[:n//3]
    middle = rewards[n//3:2*n//3]
    late = rewards[2*n//3:]

    # 计算趋势
    early_mean = np.mean(early)
    middle_mean = np.mean(middle)
    late_mean = np.mean(late)

    # 计算是否提升
    improvement = late_mean - early_mean

    return {
        'total_steps': n,
        'reward_mean': np.mean(rewards),
        'reward_std': np.std(rewards),
        'reward_min': np.min(rewards),
        'reward_max': np.max(rewards),
        'early_mean': early_mean,
        'middle_mean': middle_mean,
        'late_mean': late_mean,
        'improvement': improvement,
        'reward_trend': 'improving' if improvement > 0.01 else 'declining' if improvement < -0.01 else 'stable',
        'entropy_mean': np.mean(metrics['entropies']) if metrics['entropies'] else 0,
        'advantage_mean': np.mean(metrics['advantages']) if metrics['advantages'] else 0,
    }


def check_error_messages(log_path: str) -> List[Dict[str, str]]:
    """检查日志中的错误信息"""
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    errors = []

    # 检查各种错误模式
    error_patterns = [
        (r'Error executing job.*?overrides.*?\n(.*?)(?=\n\n|\Z)', 'job_error'),
        (r'OSError.*?Disk quota exceeded', 'disk_quota'),
        (r'Error code: (\d+) - (.*?)(?:\n|$)', 'api_error'),
        (r'CUDA.*?error', 'cuda_error'),
        (r'RuntimeError', 'runtime_error'),
        (r'Exception.*?Traceback', 'exception'),
    ]

    for pattern, error_type in error_patterns:
        matches = re.findall(pattern, content, re.DOTALL)
        for match in matches:
            errors.append({
                'type': error_type,
                'message': match[:500] if isinstance(match, str) else match[0][:500]
            })

    return errors


def analyze_sft_data(data_path: str, num_samples: int = 100) -> Dict[str, Any]:
    """分析SFT训练数据，查看成功和失败案例的差异"""
    trajectories = []

    with open(data_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= num_samples:
                break
            try:
                data = json.loads(line)
                trajectories.append(data)
            except:
                continue

    # 统计成功/失败
    success_count = sum(1 for t in trajectories if t.get('annotation_success_rate', 0) > 0.5)
    fail_count = len(trajectories) - success_count

    # 分析成功案例的特征
    success_tasks = [t['task'] for t in trajectories if t.get('annotation_success_rate', 0) > 0.5]
    fail_tasks = [t['task'] for t in trajectories if t.get('annotation_success_rate', 0) <= 0.5]

    # 分析步数分布
    step_counts = [t.get('num_steps', 0) for t in trajectories]

    return {
        'total_samples': len(trajectories),
        'success_count': success_count,
        'fail_count': fail_count,
        'success_rate': success_count / len(trajectories) if trajectories else 0,
        'avg_steps': np.mean(step_counts),
        'step_distribution': {
            'min': min(step_counts) if step_counts else 0,
            'max': max(step_counts) if step_counts else 0,
            'mean': np.mean(step_counts) if step_counts else 0
        }
    }


def generate_analysis_report(log_path: str, output_path: str = None):
    """生成完整的分析报告"""
    print(f"分析训练日志: {log_path}")

    # 解析日志
    metrics = parse_training_log(log_path)
    trend = analyze_reward_trend(metrics)
    errors = check_error_messages(log_path)

    # 打印报告
    print("\n" + "="*60)
    print("RL训练问题分析报告")
    print("="*60)

    print("\n【1. 训练状态】")
    if errors:
        # 去重错误
        unique_errors = []
        seen = set()
        for err in errors:
            key = (err['type'], err['message'][:50])
            if key not in seen:
                seen.add(key)
                unique_errors.append(err)
        print("发现错误:")
        for err in unique_errors[:5]:
            print(f"  - {err['type']}: {err['message'][:100]}...")
    else:
        print("  未发现明显错误")

    print("\n【2. Reward趋势分析】")
    print(f"  总训练步数: {trend.get('total_steps', 'N/A')}")
    print(f"  Reward均值: {trend.get('reward_mean', 'N/A'):.4f}")
    print(f"  Reward标准差: {trend.get('reward_std', 'N/A'):.4f}")
    print(f"  Reward范围: [{trend.get('reward_min', 'N/A'):.4f}, {trend.get('reward_max', 'N/A'):.4f}]")
    print(f"  早期均值: {trend.get('early_mean', 'N/A'):.4f}")
    print(f"  中期均值: {trend.get('middle_mean', 'N/A'):.4f}")
    print(f"  后期均值: {trend.get('late_mean', 'N/A'):.4f}")
    print(f"  趋势判断: {trend.get('reward_trend', 'N/A')}")
    print(f"  改进幅度: {trend.get('improvement', 'N/A'):.4f}")

    print("\n【3. 训练稳定性】")
    print(f"  Entropy均值: {trend.get('entropy_mean', 'N/A'):.4f}")
    print(f"  Advantage均值: {trend.get('advantage_mean', 'N/A'):.4f}")

    print("\n【4. 根因分析】")
    if trend.get('reward_trend') == 'stable' and trend.get('improvement', 0) < 0.01:
        print("  问题: Reward没有明显提升")
        print("  可能原因:")
        print("    1. 模型没有学到有效策略")
        print("    2. Reward设计不合理")
        print("    3. 训练数据不足")
        print("    4. 学习率过低")
    if unique_errors if 'unique_errors' in locals() else errors:
        print("  异常终止原因:")
        errs_to_show = unique_errors if 'unique_errors' in locals() else errors[:3]
        for err in errs_to_show:
            print(f"    - {err['type']}: {err['message'][:80]}...")

    # 保存报告
    report = {
        'log_path': log_path,
        'metrics': metrics,
        'trend': trend,
        'errors': errors[:10]  # 只保存前10个错误
    }

    if output_path:
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n报告已保存到: {output_path}")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='分析WebShop RL训练问题')
    parser.add_argument('--log', type=str, help='训练日志路径')
    parser.add_argument('--output', type=str, help='输出报告路径')

    args = parser.parse_args()

    if args.log:
        generate_analysis_report(args.log, args.output)
    else:
        # 默认分析最新的实验
        log_dir = "/fs-computility-new/UPDZ03_chengjun/huangsijie.p/rebel_results/v11_final_webshop"
        logs = [f for f in os.listdir(log_dir) if f.startswith('M5_') and f.endswith('/')]
        if logs:
            latest = sorted(logs)[-1]
            log_path = os.path.join(log_dir, latest, 'training.log')
            if os.path.exists(log_path):
                generate_analysis_report(log_path, args.output)