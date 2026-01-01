#!/usr/bin/env python3
"""
重新生成符合规范的 Cold-start 数据

关键修复:
1. 第一步: 使用 ALFWORLD_REBEL_TEMPLATE_NO_HIS_CS（不包含Available Actions）
2. 后续步: 从hindsight prompt中彻底删除"Available Actions:"行
3. 验证生成的数据确实不含Available Actions
"""

import sys
import json
import re
from typing import Dict, Any, List
from datasets import load_from_disk

sys.path.insert(0, '/root/testttt/RLVMR/code')

# Import ReBel prompts
from agent_system.environments.prompts.rebel_prompts import (
    ALFWORLD_REBEL_TEMPLATE_NO_HIS_CS,
    ALFWORLD_REBEL_TEMPLATE_CS
)


def extract_observation(human_message: str) -> str:
    """从 human 消息中提取 observation"""
    if 'Observation:' in human_message:
        obs_match = re.search(
            r'Observation:\n(.*?)(?:\n\nCurrent Belief State:|\nCurrent Belief State:|$)',
            human_message,
            re.DOTALL
        )
        if obs_match:
            return obs_match.group(1).strip()

    # Fallback
    task_match = re.search(r'Task:.*?\n\n(.*)', human_message, re.DOTALL)
    if task_match:
        return task_match.group(1).strip()

    return human_message.strip()


def extract_task_description(human_message: str) -> str:
    """提取任务描述"""
    match = re.search(r'Your task is to: (.+?)(?:\n|$)', human_message)
    if match:
        return match.group(1).strip()
    return ""


def extract_belief_state(human_message: str) -> str:
    """提取Belief State（JSON字符串）"""
    if 'Current Belief State:' not in human_message:
        return ""

    belief_start = human_message.index('Current Belief State:') + len('Current Belief State:')

    # 找到belief state的结束位置（在Available Actions之前）
    if '\n\nAvailable Actions:' in human_message:
        belief_end = human_message.index('\n\nAvailable Actions:')
    else:
        belief_end = len(human_message)

    belief_str = human_message[belief_start:belief_end].strip()
    return belief_str


def extract_action_history(human_message: str) -> str:
    """提取历史动作和观测"""
    # 匹配 "Below are the most recent X observations and actions: ..."
    match = re.search(
        r'Below are the most recent \d+ observations? and (?:the corresponding )?actions?(?: you took)?:\s*(.*?)(?:\n\nYou are now at step|\nYou are now at step)',
        human_message,
        re.DOTALL
    )
    if match:
        return match.group(1).strip()
    return ""


def extract_step_count(human_message: str) -> int:
    """提取当前步数"""
    match = re.search(r'you have already taken (\d+) step\(s\)', human_message)
    if match:
        return int(match.group(1))
    return 0


def extract_current_step(human_message: str) -> int:
    """提取当前步数"""
    match = re.search(r'You are now at step (\d+)', human_message)
    if match:
        return int(match.group(1))
    return 1


def extract_planning(human_message: str) -> str:
    """提取之前的规划"""
    match = re.search(r'Your [Pp]revious [Oo]verall [Pp]lan(?:\s+is)?:\s*(.+?)(?:\n\n|\n(?=[A-Z]))', human_message, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def rebuild_coldstart_prompt_step1(obs: str) -> str:
    """
    重建第一步的Cold-start Prompt（使用正确的模板，无Available Actions）

    使用: ALFWORLD_REBEL_TEMPLATE_NO_HIS_CS
    """
    prompt = ALFWORLD_REBEL_TEMPLATE_NO_HIS_CS.format(
        current_observation=obs
    )
    return prompt


def rebuild_coldstart_prompt_subsequent(
    task: str,
    step_count: int,
    history_length: int,
    action_history: str,
    current_step: int,
    current_obs: str,
    current_belief_state: str,
    planning: str
) -> str:
    """
    重建后续步骤的Cold-start Prompt（使用正确的模板，无Available Actions）

    使用: ALFWORLD_REBEL_TEMPLATE_CS
    """
    prompt = ALFWORLD_REBEL_TEMPLATE_CS.format(
        task_description=task,
        step_count=step_count,
        history_length=history_length,
        action_history=action_history,
        current_step=current_step,
        current_observation=current_obs,
        current_belief_state=current_belief_state,
        planning=planning
    )
    return prompt


def hindsight_to_coldstart_clean(hindsight_sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    将单个 hindsight 格式样本转换为 clean coldstart 格式

    关键改进:
    1. 第一步使用 ALFWORLD_REBEL_TEMPLATE_NO_HIS_CS 重建
    2. 后续步使用 ALFWORLD_REBEL_TEMPLATE_CS 重建
    3. 完全不包含 Available Actions
    """
    conversations = hindsight_sample['conversations']
    task = hindsight_sample['task']

    # 判断任务是否成功
    success_rate = hindsight_sample.get('annotation_success_rate', 0.0)
    done = "True" if success_rate >= 0.8 else "False"

    data = []
    step_num = 0

    # 从第3轮开始（索引2），每2轮对应一个步骤
    for i in range(2, len(conversations), 2):
        if i + 1 >= len(conversations):
            break

        human_msg = conversations[i]
        gpt_msg = conversations[i + 1]

        # 确保是正确的对话对
        if human_msg['from'] != 'human' or gpt_msg['from'] != 'gpt':
            continue

        step_num += 1
        human_value = human_msg['value']

        # 提取必要信息
        obs = extract_observation(human_value)
        response = gpt_msg['value']

        # 重建prompt（根据步数使用不同策略）
        if step_num == 1:
            # 第一步: 使用 NO_HIS_CS 模板（完全重建，无任何Available Actions）
            prompt = rebuild_coldstart_prompt_step1(obs)
        else:
            # 后续步: 使用 CS 模板重建（完全重建，无任何Available Actions）
            task_desc = extract_task_description(human_value)
            if not task_desc:
                task_desc = task

            step_count = extract_step_count(human_value)
            current_step = extract_current_step(human_value)
            action_history = extract_action_history(human_value)
            belief_state = extract_belief_state(human_value)
            planning = extract_planning(human_value)

            # 计算history_length
            if action_history:
                # 简单估计：每个"Observation X:"算一个历史记录
                history_length = action_history.count('Observation')
                if history_length == 0:
                    history_length = 2  # 默认值
            else:
                history_length = 2

            prompt = rebuild_coldstart_prompt_subsequent(
                task=task_desc,
                step_count=step_count,
                history_length=history_length,
                action_history=action_history if action_history else "None",
                current_step=current_step,
                current_obs=obs,
                current_belief_state=belief_state,
                planning=planning if planning else "None"
            )

        step_data = {
            "step": step_num,
            "obs": obs,
            "prompt": prompt,  # Clean prompt, NO Available Actions!
            "response": response
        }

        data.append(step_data)

    coldstart_sample = {
        "task": task,
        "done": done,
        "data": data
    }

    return coldstart_sample


def verify_no_available_actions(coldstart_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    验证cold-start数据中确实没有Available Actions

    Returns:
        验证报告字典
    """
    total_samples = len(coldstart_data)
    total_steps = 0
    issues = []

    for idx, sample in enumerate(coldstart_data):
        for step_data in sample['data']:
            total_steps += 1
            prompt = step_data['prompt']

            # 检查是否包含 "Available Actions"
            if 'Available Actions' in prompt:
                issues.append({
                    'sample_idx': idx,
                    'task': sample['task'],
                    'step': step_data['step'],
                    'issue': 'Contains "Available Actions" in prompt'
                })

    report = {
        'total_samples': total_samples,
        'total_steps': total_steps,
        'issues_found': len(issues),
        'is_clean': len(issues) == 0,
        'issues': issues[:10]  # 只显示前10个问题
    }

    return report


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='重新生成clean的cold-start数据')
    parser.add_argument(
        '--input_dir',
        type=str,
        default='/root/testttt/RLVMR/code/data/alfworld_rebel_250_new',
        help='Hindsight数据目录（Arrow格式）'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        default='/root/testttt/RLVMR/code/data/alfworld_rebel_250_new/rebel_coldstart_clean.json',
        help='输出的clean cold-start文件'
    )
    parser.add_argument(
        '--success_threshold',
        type=float,
        default=1.0,
        help='成功率阈值（只转换成功率>=该值的样本）'
    )

    args = parser.parse_args()

    print("=" * 80)
    print("重新生成符合规范的 Cold-start 数据")
    print("=" * 80)

    # 1. 加载hindsight数据
    print(f"\n[1/4] 加载hindsight数据...")
    print(f"  输入目录: {args.input_dir}")

    try:
        dataset = load_from_disk(args.input_dir)
        print(f"  ✅ 加载成功，总样本数: {len(dataset)}")
    except Exception as e:
        print(f"  ❌ 加载失败: {e}")
        return

    # 2. 筛选成功样本并转换
    print(f"\n[2/4] 转换数据...")
    print(f"  成功率阈值: >= {args.success_threshold}")

    coldstart_data = []
    for sample in dataset:
        if sample['annotation_success_rate'] >= args.success_threshold:
            try:
                cs_sample = hindsight_to_coldstart_clean(sample)
                coldstart_data.append(cs_sample)
            except Exception as e:
                print(f"  ⚠️ 跳过样本 {sample.get('item_id', 'unknown')}: {e}")
                continue

    print(f"  ✅ 成功转换 {len(coldstart_data)} 个样本")

    # 3. 验证数据
    print(f"\n[3/4] 验证数据...")
    report = verify_no_available_actions(coldstart_data)

    print(f"  总样本数: {report['total_samples']}")
    print(f"  总步骤数: {report['total_steps']}")
    print(f"  问题数量: {report['issues_found']}")

    if report['is_clean']:
        print(f"  ✅ 验证通过！所有prompts都不包含Available Actions")
    else:
        print(f"  ❌ 验证失败！发现 {report['issues_found']} 个包含Available Actions的prompt")
        print(f"\n  前10个问题:")
        for issue in report['issues'][:10]:
            print(f"    - 样本{issue['sample_idx']}, 步骤{issue['step']}: {issue['issue']}")

        # 询问是否继续
        response = input("\n  是否仍要保存数据? (y/n): ")
        if response.lower() != 'y':
            print("  取消保存")
            return

    # 4. 保存数据
    print(f"\n[4/4] 保存数据...")
    print(f"  输出文件: {args.output_file}")

    try:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(coldstart_data, f, indent=2, ensure_ascii=False)
        print(f"  ✅ 保存成功！")
    except Exception as e:
        print(f"  ❌ 保存失败: {e}")
        return

    # 5. 生成验证报告
    report_file = args.output_file.replace('.json', '_validation_report.json')
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"  验证报告已保存: {report_file}")

    print("\n" + "=" * 80)
    print("✅ 完成！")
    print("=" * 80)
    print(f"\n生成的clean cold-start数据: {args.output_file}")
    print(f"验证报告: {report_file}")

    if report['is_clean']:
        print(f"\n🎉 数据质量验证通过！可以用于SFT训练。")
    else:
        print(f"\n⚠️ 数据仍有问题，请检查验证报告。")


if __name__ == '__main__':
    main()
