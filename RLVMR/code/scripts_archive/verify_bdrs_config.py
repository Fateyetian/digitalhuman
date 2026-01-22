#!/usr/bin/env python3
"""
BDRS配置验证脚本
用于检查评测配置是否正确使用BDRS格式
"""

import sys
import json
from pathlib import Path

def check_training_data():
    """检查训练数据格式"""
    print("=" * 80)
    print("1. 检查训练数据格式")
    print("=" * 80)

    data_file = Path(__file__).parent / "data" / "alfworld_cold-start.json"
    if not data_file.exists():
        print("❌ 训练数据文件不存在")
        return False

    with open(data_file, 'r') as f:
        data = json.load(f)

    bdrs_count = 0
    for sample in data:
        if 'data' in sample:
            for step in sample['data']:
                response = step.get('response', '')
                if any(tag in response for tag in ['<PLAN>', '<EXECUTE>', '<EXPLORE>', '<VERIFY>']):
                    bdrs_count += 1
                    break

    bdrs_ratio = bdrs_count / len(data) if data else 0
    print(f"训练样本总数: {len(data)}")
    print(f"使用BDRS格式: {bdrs_count} ({bdrs_ratio:.1%})")

    if bdrs_ratio > 0.9:
        print("✅ 训练数据格式正确")
        return True
    else:
        print("❌ 训练数据格式不一致")
        return False

def check_eval_trajectory(traj_file=None):
    """检查评测trajectory中的prompt格式"""
    print("\n" + "=" * 80)
    print("2. 检查评测prompt格式")
    print("=" * 80)

    if traj_file is None:
        # 查找最新的trajectory文件
        results_dir = Path(__file__).parent / "results"
        if not results_dir.exists():
            print("⚠️  还未运行评测，无法检查")
            return None

        traj_files = list(results_dir.glob("*/trajectory.jsonl"))
        if not traj_files:
            print("⚠️  未找到trajectory文件，无法检查")
            return None

        traj_file = max(traj_files, key=lambda p: p.stat().st_mtime)
        print(f"使用最新的trajectory: {traj_file.parent.name}")

    with open(traj_file, 'r') as f:
        first_line = f.readline()
        if not first_line:
            print("❌ Trajectory文件为空")
            return False

        first_step = json.loads(first_line)

    prompt = first_step.get('prompt', '')
    action = first_step.get('action', '')
    is_valid = first_step.get('is_action_valid', False)

    has_bdrs_in_prompt = any(tag in prompt for tag in ['<PLAN>', '<EXECUTE>', '<EXPLORE>', '<VERIFY>'])
    has_belief_modules = 'M_t' in prompt or 'belief module' in prompt.lower()
    has_think_prompt = '<think>' in prompt

    has_bdrs_in_action = any(tag in action for tag in ['<PLAN>', '<EXECUTE>', '<EXPLORE>', '<VERIFY>'])
    has_think_in_action = '<think>' in action

    print(f"\nPrompt格式:")
    print(f"  - 包含BDRS标签: {'✓' if has_bdrs_in_prompt else '✗'}")
    print(f"  - 包含belief modules: {'✓' if has_belief_modules else '✗'}")
    print(f"  - 要求<think>标签: {'✓' if has_think_prompt else '✗'}")

    print(f"\n模型输出:")
    print(f"  - 使用BDRS标签: {'✓' if has_bdrs_in_action else '✗'}")
    print(f"  - 使用<think>标签: {'✓' if has_think_in_action else '✗'}")
    print(f"  - 动作是否有效: {'✓' if is_valid else '✗'}")

    # 判断配置是否正确
    if has_bdrs_in_prompt and has_belief_modules:
        if has_bdrs_in_action and is_valid:
            print("\n✅ 配置正确，模型正常工作")
            return True
        elif has_bdrs_in_action and not is_valid:
            print("\n⚠️  Prompt正确但动作被拒绝，可能是动作格式问题")
            return False
        else:
            print("\n❌ Prompt正确但模型未生成BDRS格式，需要检查模型")
            return False
    elif has_think_prompt:
        print("\n❌ 使用了<think>格式prompt，与训练数据不匹配")
        print("    👉 需要在rollout脚本中设置use_bdrs=True")
        return False
    else:
        print("\n❌ Prompt格式未知")
        return False

def check_projection_code():
    """检查projection函数配置"""
    print("\n" + "=" * 80)
    print("3. 检查projection函数")
    print("=" * 80)

    proj_file = Path(__file__).parent / "agent_system" / "environments" / "env_package" / "alfworld" / "projection.py"

    if not proj_file.exists():
        print("❌ projection.py文件不存在")
        return False

    with open(proj_file, 'r') as f:
        content = f.read()

    has_bdrs_function = 'def alfworld_projection_bdrs' in content
    checks_bdrs_tags = 'bdrs_tags =' in content or 'BDRS tag' in content

    print(f"alfworld_projection_bdrs函数存在: {'✓' if has_bdrs_function else '✗'}")
    print(f"检查BDRS标签: {'✓' if checks_bdrs_tags else '✗'}")

    if has_bdrs_function and checks_bdrs_tags:
        print("✅ Projection函数配置正确")
        return True
    else:
        print("❌ Projection函数配置异常")
        return False

def main():
    print("BDRS配置验证工具")
    print("=" * 80)
    print()

    results = {
        'training_data': check_training_data(),
        'projection': check_projection_code(),
        'eval_traj': check_eval_trajectory()
    }

    print("\n" + "=" * 80)
    print("总结")
    print("=" * 80)

    if results['training_data'] and results['projection']:
        if results['eval_traj'] is None:
            print("⚠️  训练和代码配置正确，但还未运行评测")
            print("   运行评测后再次执行此脚本检查")
        elif results['eval_traj']:
            print("✅ 所有配置正确！")
        else:
            print("❌ 评测配置错误，需要修复:")
            print("   1. 检查rollout脚本中的use_bdrs参数")
            print("   2. 确保env_manager正确选择BDRS template")
            print("   3. 重新运行评测")
    else:
        print("❌ 存在配置问题，请根据上述检查结果修复")

    print("\n建议:")
    print("  1. 修复配置后重新运行评测")
    print("  2. 检查最新的trajectory.jsonl验证修复效果")
    print("  3. 预期成功率应从0%提升到15-25%")
    print()

if __name__ == '__main__':
    main()
