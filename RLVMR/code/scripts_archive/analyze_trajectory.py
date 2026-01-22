#!/usr/bin/env python3
"""
BDRS轨迹分析工具
用于深度分析评测trajectory，定位失败原因
"""

import json
import sys
from collections import defaultdict, Counter

def analyze_trajectory(traj_file):
    """分析trajectory文件"""

    # 加载数据
    episodes_data = defaultdict(list)
    with open(traj_file, 'r') as f:
        for line in f:
            d = json.loads(line)
            episodes_data[d['env_id']].append(d)

    # 选择一个失败案例详细分析
    env_id = 0
    env_data = episodes_data[env_id]

    print("=" * 80)
    print(f"【失败案例深度分析 - Env {env_id}】")
    print("=" * 80)

    # 提取任务描述
    task_prompt = env_data[0]['prompt']
    task_line = task_prompt.split('Your task is to: ')[1].split('\n')[0] if 'Your task is to: ' in task_prompt else '未知'
    print(f"任务: {task_line}")
    print(f"总步数: {len(env_data)}")
    print(f"是否成功: {env_data[-1].get('won', False)}")

    # 分析动作序列
    print(f"\n【前10步详细分析】")
    for i, step in enumerate(env_data[:10]):
        action = step['action']

        # 提取模式
        mode = 'UNKNOWN'
        for m in ['PLAN', 'EXECUTE', 'EXPLORE', 'VERIFY']:
            if f'<{m}>' in action:
                mode = m
                break

        # 提取action
        if '<action>' in action and '</action>' in action:
            action_line = action.split('<action>')[1].split('</action>')[0].strip()
        else:
            action_line = 'N/A'

        # 提取推理
        if '>' in action and '</' in action:
            reasoning = action.split('>')[1].split('</')[0][:100]
        else:
            reasoning = ''

        print(f"\n  Step {i}: [{mode:8s}] {action_line}")
        if reasoning:
            print(f"           推理: {reasoning}...")
        print(f"           有效: {step.get('is_action_valid', False)}")

    print(f"\n【最后5步】")
    for i in range(max(0, len(env_data)-5), len(env_data)):
        step = env_data[i]
        action = step['action']

        mode = 'UNKNOWN'
        for m in ['PLAN', 'EXECUTE', 'EXPLORE', 'VERIFY']:
            if f'<{m}>' in action:
                mode = m
                break

        if '<action>' in action and '</action>' in action:
            action_line = action.split('<action>')[1].split('</action>')[0].strip()
        else:
            action_line = 'N/A'

        print(f"\n  Step {i}: [{mode:8s}] {action_line}")

    # 统计问题
    print(f"\n" + "=" * 80)
    print("【问题诊断】")
    print("=" * 80)

    # 1. 检查重复动作
    action_sequence = []
    for step in env_data:
        action = step['action']
        if '<action>' in action and '</action>' in action:
            action_line = action.split('<action>')[1].split('</action>')[0].strip()
            action_sequence.append(action_line)

    action_counts = Counter(action_sequence)
    repeated_actions = [(action, count) for action, count in action_counts.most_common(5) if count > 3]

    if repeated_actions:
        print("\n1. 发现重复动作:")
        for action, count in repeated_actions:
            print(f"   '{action}' 重复 {count} 次")
    else:
        print("\n1. 无明显重复动作")

    # 2. 模式分布
    mode_sequence = []
    for step in env_data:
        action = step['action']
        for m in ['PLAN', 'EXECUTE', 'EXPLORE', 'VERIFY']:
            if f'<{m}>' in action:
                mode_sequence.append(m)
                break

    mode_dist = Counter(mode_sequence)
    print(f"\n2. 模式分布:")
    for mode in ['PLAN', 'EXECUTE', 'EXPLORE', 'VERIFY']:
        count = mode_dist.get(mode, 0)
        pct = count / len(mode_sequence) * 100 if mode_sequence else 0
        print(f"   {mode:8s}: {count:3d} ({pct:5.1f}%)")

    # 3. 模式切换
    print(f"\n3. 模式切换序列（前20步）:")
    print(f"   {' -> '.join(mode_sequence[:20])}")

    # 4. 检查是否卡在某个位置
    location_sequence = []
    for i, step in enumerate(env_data):
        if i < len(env_data) - 1:
            next_prompt = env_data[i+1]['prompt']
            if 'You are in the middle' in next_prompt or 'You arrive at' in next_prompt:
                # 简单提取位置信息
                if 'arrive at' in next_prompt:
                    loc = next_prompt.split('arrive at ')[1].split('.')[0] if 'arrive at ' in next_prompt else 'unknown'
                    location_sequence.append(loc)

    if len(location_sequence) > 5:
        loc_counts = Counter(location_sequence)
        stuck_locs = [(loc, count) for loc, count in loc_counts.most_common(3) if count > 5]
        if stuck_locs:
            print(f"\n4. 可能卡在某些位置:")
            for loc, count in stuck_locs:
                print(f"   {loc}: 访问 {count} 次")

    return episodes_data


def compare_success_vs_failure(traj_file):
    """对比成功和失败案例"""

    episodes_data = defaultdict(list)
    with open(traj_file, 'r') as f:
        for line in f:
            d = json.loads(line)
            episodes_data[d['env_id']].append(d)

    # 找到成功和失败的案例
    success_envs = []
    failure_envs = []

    for env_id, data in episodes_data.items():
        if data[-1].get('won', False):
            success_envs.append((env_id, data))
        else:
            failure_envs.append((env_id, data))

    print("\n" + "=" * 80)
    print("【成功 vs 失败对比】")
    print("=" * 80)
    print(f"成功案例数: {len(success_envs)}")
    print(f"失败案例数: {len(failure_envs)}")

    if success_envs:
        print(f"\n成功案例特征:")
        avg_steps = sum(len(data) for _, data in success_envs) / len(success_envs)
        print(f"  平均步数: {avg_steps:.1f}")

        # 模式分布
        all_modes = []
        for _, data in success_envs:
            for step in data:
                action = step['action']
                for m in ['PLAN', 'EXECUTE', 'EXPLORE', 'VERIFY']:
                    if f'<{m}>' in action:
                        all_modes.append(m)
                        break
        mode_dist = Counter(all_modes)
        print(f"  模式分布:")
        for mode in ['PLAN', 'EXECUTE', 'EXPLORE', 'VERIFY']:
            count = mode_dist.get(mode, 0)
            pct = count / len(all_modes) * 100 if all_modes else 0
            print(f"    {mode:8s}: {pct:5.1f}%")

    if failure_envs:
        print(f"\n失败案例特征:")
        avg_steps = sum(len(data) for _, data in failure_envs) / len(failure_envs)
        print(f"  平均步数: {avg_steps:.1f}")

        # 模式分布
        all_modes = []
        for _, data in failure_envs:
            for step in data:
                action = step['action']
                for m in ['PLAN', 'EXECUTE', 'EXPLORE', 'VERIFY']:
                    if f'<{m}>' in action:
                        all_modes.append(m)
                        break
        mode_dist = Counter(all_modes)
        print(f"  模式分布:")
        for mode in ['PLAN', 'EXECUTE', 'EXPLORE', 'VERIFY']:
            count = mode_dist.get(mode, 0)
            pct = count / len(all_modes) * 100 if all_modes else 0
            print(f"    {mode:8s}: {pct:5.1f}%")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("用法: python3 analyze_trajectory.py <trajectory_file>")
        print("示例: python3 analyze_trajectory.py results/eval_20251112_160532/trajectory.jsonl")
        sys.exit(1)

    traj_file = sys.argv[1]

    try:
        episodes_data = analyze_trajectory(traj_file)
        compare_success_vs_failure(traj_file)

        print("\n" + "=" * 80)
        print("分析完成")
        print("=" * 80)

    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
