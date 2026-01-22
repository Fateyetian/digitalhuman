#!/usr/bin/env python3
"""
Arrow格式数据查看工具

使用方法:
    python view_arrow_data.py                           # 查看alfworld_expert_traj
    python view_arrow_data.py --index 5                 # 查看第5条样本
    python view_arrow_data.py --export sample.json      # 导出为JSON
"""

import argparse
from datasets import load_from_disk
import json


def view_dataset_info(dataset_path):
    """查看数据集基本信息"""
    ds = load_from_disk(dataset_path)

    print("="*70)
    print("数据集信息")
    print("="*70)
    print(f"路径: {dataset_path}")
    print(f"样本数: {len(ds)}")
    print(f"字段: {ds.column_names}")
    print(f"特征: {ds.features}")

    # 统计
    if 'conversations' in ds.column_names:
        lengths = [len(sample['conversations']) for sample in ds]
        print(f"\n对话统计:")
        print(f"  最短: {min(lengths)} 轮")
        print(f"  最长: {max(lengths)} 轮")
        print(f"  平均: {sum(lengths)/len(lengths):.1f} 轮")


def view_sample(dataset_path, index=0, num_turns=10):
    """查看特定样本"""
    ds = load_from_disk(dataset_path)

    if index >= len(ds):
        print(f"错误: 索引 {index} 超出范围 (0-{len(ds)-1})")
        return

    sample = ds[index]

    print("="*70)
    print(f"样本 #{index}")
    print("="*70)

    if 'item_id' in sample:
        print(f"Item ID: {sample['item_id']}")

    if 'conversations' in sample:
        convs = sample['conversations']
        print(f"总轮数: {len(convs)}")
        print(f"\n显示前 {min(num_turns, len(convs))} 轮:\n")

        for i, turn in enumerate(convs[:num_turns]):
            role = "🤖 Agent" if turn['from'] == 'gpt' else "🏠 环境"
            print(f"\n【{role} - 第{i+1}轮】")

            content = turn['value']
            # 限制显示长度
            if len(content) > 500:
                print(content[:500] + "...")
            else:
                print(content)


def export_to_json(dataset_path, output_path, num_samples=None):
    """导出为JSON格式"""
    ds = load_from_disk(dataset_path)

    if num_samples:
        ds = ds.select(range(min(num_samples, len(ds))))

    data = [sample for sample in ds]

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"✅ 已导出 {len(data)} 条样本到 {output_path}")


def main():
    parser = argparse.ArgumentParser(description='查看Arrow格式数据集')
    parser.add_argument('--path', type=str, default='data/alfworld_expert_traj',
                        help='数据集路径')
    parser.add_argument('--index', type=int, default=None,
                        help='查看特定索引的样本')
    parser.add_argument('--turns', type=int, default=10,
                        help='显示的对话轮数')
    parser.add_argument('--export', type=str, default=None,
                        help='导出为JSON文件')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='导出样本数量')

    args = parser.parse_args()

    if args.export:
        export_to_json(args.path, args.export, args.num_samples)
    elif args.index is not None:
        view_sample(args.path, args.index, args.turns)
    else:
        view_dataset_info(args.path)


if __name__ == '__main__':
    main()
