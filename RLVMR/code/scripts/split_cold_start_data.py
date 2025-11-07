#!/usr/bin/env python3
"""
将冷启动数据分割为训练集和测试集
用于评测冷启动模型的泛化能力
"""

import json
import argparse
import random
from pathlib import Path


def split_cold_start_data(input_file, output_dir, train_ratio=0.8, seed=42):
    """
    将冷启动数据分割为训练集和测试集

    Args:
        input_file: 输入的JSON文件路径
        output_dir: 输出目录
        train_ratio: 训练集比例（默认0.8 = 80%）
        seed: 随机种子
    """
    # 读取数据
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    total = len(data)
    print(f"总数据量: {total} 条")

    # 设置随机种子
    random.seed(seed)

    # 打乱数据
    shuffled_data = data.copy()
    random.shuffle(shuffled_data)

    # 分割
    train_size = int(total * train_ratio)
    train_data = shuffled_data[:train_size]
    test_data = shuffled_data[train_size:]

    print(f"训练集: {len(train_data)} 条 ({len(train_data)/total*100:.1f}%)")
    print(f"测试集: {len(test_data)} 条 ({len(test_data)/total*100:.1f}%)")

    # 保存
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_file = output_dir / "alfworld_cold-start_train.json"
    test_file = output_dir / "alfworld_cold-start_test.json"

    with open(train_file, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    print(f"✓ 训练集保存到: {train_file}")

    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)
    print(f"✓ 测试集保存到: {test_file}")

    # 统计任务类型分布
    print("\n任务类型分布:")
    train_tasks = {}
    test_tasks = {}

    for item in train_data:
        task_type = item['task'].split()[0]  # 取第一个词作为任务类型
        train_tasks[task_type] = train_tasks.get(task_type, 0) + 1

    for item in test_data:
        task_type = item['task'].split()[0]
        test_tasks[task_type] = test_tasks.get(task_type, 0) + 1

    print("训练集:")
    for task_type, count in sorted(train_tasks.items()):
        print(f"  {task_type}: {count}")

    print("测试集:")
    for task_type, count in sorted(test_tasks.items()):
        print(f"  {task_type}: {count}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='分割冷启动数据为训练集和测试集')
    parser.add_argument('--input', '-i',
                        default='../data/alfworld_cold-start.json',
                        help='输入JSON文件路径')
    parser.add_argument('--output', '-o',
                        default='../data',
                        help='输出目录')
    parser.add_argument('--train_ratio', '-r',
                        type=float,
                        default=0.8,
                        help='训练集比例 (0-1, 默认0.8)')
    parser.add_argument('--seed', '-s',
                        type=int,
                        default=42,
                        help='随机种子')

    args = parser.parse_args()

    split_cold_start_data(
        input_file=args.input,
        output_dir=args.output,
        train_ratio=args.train_ratio,
        seed=args.seed
    )
