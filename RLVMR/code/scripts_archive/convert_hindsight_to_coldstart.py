#!/usr/bin/env python3
"""
将 ReBel Hindsight 格式转换为 Coldstart 格式
"""

import json
import re
from typing import List, Dict, Any
from datasets import load_from_disk


def extract_observation(human_message: str) -> str:
    """从 human 消息中提取 observation"""
    # 尝试提取 "Observation:" 和下一个section之间的内容
    if 'Observation:' in human_message:
        # 匹配 Observation: 后面到 "Current Belief State:" 或消息结尾的内容
        obs_match = re.search(
            r'Observation:\n(.*?)(?:\n\nCurrent Belief State:|\nCurrent Belief State:|$)',
            human_message,
            re.DOTALL
        )
        if obs_match:
            return obs_match.group(1).strip()

    # 如果没有找到，返回整个消息去掉 "Task:" 部分
    task_match = re.search(r'Task:.*?\n\n(.*)', human_message, re.DOTALL)
    if task_match:
        return task_match.group(1).strip()

    return human_message.strip()


def hindsight_to_coldstart(hindsight_sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    将单个 hindsight 格式样本转换为 coldstart 格式

    Hindsight 格式:
    {
        "conversations": [...],
        "item_id": "...",
        "num_steps": 24,
        "annotation_success_rate": 1.0,
        "task": "..."
    }

    Coldstart 格式:
    {
        "task": "...",
        "done": "True",
        "data": [
            {
                "step": 1,
                "obs": "...",
                "prompt": "...",
                "response": "..."
            },
            ...
        ]
    }
    """
    conversations = hindsight_sample['conversations']
    task = hindsight_sample['task']
    num_steps = hindsight_sample['num_steps']

    # 判断任务是否成功（成功率 >= 0.8 视为成功）
    success_rate = hindsight_sample.get('annotation_success_rate', 0.0)
    done = "True" if success_rate >= 0.8 else "False"

    data = []
    step_num = 0

    # conversations 格式:
    # [0] human: system prompt
    # [1] gpt: OK
    # [2] human: 第1步的 prompt
    # [3] gpt: 第1步的 response
    # [4] human: 第2步的 prompt
    # [5] gpt: 第2步的 response
    # ...

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

        # 提取 observation
        obs = extract_observation(human_msg['value'])

        # prompt 是完整的 human 消息
        prompt = human_msg['value']

        # response 是完整的 gpt 消息
        response = gpt_msg['value']

        step_data = {
            "step": step_num,
            "obs": obs,
            "prompt": prompt,
            "response": response
        }

        data.append(step_data)

    coldstart_sample = {
        "task": task,
        "done": done,
        "data": data
    }

    return coldstart_sample


def convert_dataset_hindsight_to_coldstart(
    input_path: str,
    output_path: str,
    success_threshold: float = 1.0
) -> List[Dict[str, Any]]:
    """
    将整个 hindsight 格式数据集转换为 coldstart 格式

    Args:
        input_path: 输入数据集路径（Arrow 格式）
        output_path: 输出文件路径（JSON 格式）
        success_threshold: 成功率阈值，只转换成功率 >= 该值的样本
    """
    print("=" * 70)
    print(f"转换 Hindsight 格式 -> Coldstart 格式")
    print("=" * 70)

    # 加载数据集
    print(f"\n加载数据集: {input_path}")
    dataset = load_from_disk(input_path)
    print(f"  总样本数: {len(dataset)}")

    # 筛选成功的样本
    successful_samples = []
    for sample in dataset:
        if sample['annotation_success_rate'] >= success_threshold:
            successful_samples.append(sample)

    print(f"  成功率 >= {success_threshold} 的样本: {len(successful_samples)} 条")

    # 转换格式
    print(f"\n开始转换格式...")
    coldstart_data = []

    for i, sample in enumerate(successful_samples):
        coldstart_sample = hindsight_to_coldstart(sample)
        coldstart_data.append(coldstart_sample)

        if (i + 1) % 50 == 0:
            print(f"  已转换: {i + 1}/{len(successful_samples)}")

    print(f"  ✓ 转换完成: {len(coldstart_data)} 条")

    # 保存为 JSON 格式
    print(f"\n保存到: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(coldstart_data, f, ensure_ascii=False, indent=2)

    print(f"  ✓ 已保存")

    # 统计信息
    print(f"\n转换后数据集统计:")
    done_count = sum(1 for s in coldstart_data if s['done'] == 'True')
    print(f"  总样本数: {len(coldstart_data)}")
    print(f"  done=True: {done_count} 条")
    print(f"  done=False: {len(coldstart_data) - done_count} 条")

    # 步数统计
    all_steps = [len(s['data']) for s in coldstart_data]
    print(f"\n  步数统计:")
    print(f"    平均步数: {sum(all_steps)/len(all_steps):.1f}")
    print(f"    最小步数: {min(all_steps)}")
    print(f"    最大步数: {max(all_steps)}")

    print("\n" + "=" * 70)
    print("✓ 转换完成！")
    print("=" * 70)

    return coldstart_data


if __name__ == "__main__":
    # 转换 alfworld_rebel_full_improved 中的成功样本
    input_path = "/root/testttt/RLVMR/code/data/alfworld_rebel_full_improved"
    output_path = "/root/testttt/RLVMR/code/data/alfworld_rebel_full_improved_coldstart.json"

    coldstart_data = convert_dataset_hindsight_to_coldstart(
        input_path=input_path,
        output_path=output_path,
        success_threshold=1.0  # 只转换100%成功的样本
    )

    print(f"\n输出文件: {output_path}")
    print(f"样本数: {len(coldstart_data)}")
