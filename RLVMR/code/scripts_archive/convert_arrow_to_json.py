#!/usr/bin/env python3
"""
Arrow格式转JSON工具

将HuggingFace Datasets的Arrow格式转换为标准JSON格式

使用方法:
    python convert_arrow_to_json.py --input data/alfworld_expert_traj --output data/alfworld_expert_traj.json
    python convert_arrow_to_json.py --input data/alfworld_rebel_golden --output data/rebel_golden.json
    python convert_arrow_to_json.py --input data/alfworld_expert_traj --output data/expert.jsonl --format jsonl
"""

import argparse
from datasets import load_from_disk
import json
import os


def convert_to_json(input_path, output_path, format='json', pretty=True):
    """
    将Arrow格式数据集转换为JSON格式

    Args:
        input_path: Arrow数据集目录路径
        output_path: 输出JSON文件路径
        format: 输出格式 ('json' 或 'jsonl')
        pretty: 是否使用格式化输出（仅对json格式有效）
    """
    print(f"正在加载数据集: {input_path}")

    try:
        ds = load_from_disk(input_path)
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return False

    print(f"✅ 加载成功")
    print(f"   - 样本数: {len(ds)}")
    print(f"   - 字段: {ds.column_names}")

    # 转换为列表
    print(f"\n正在转换为{format.upper()}格式...")
    data = [sample for sample in ds]

    # 保存
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    if format == 'jsonl':
        # JSONL格式：每行一个JSON对象
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    else:
        # JSON格式：整个数组
        indent = 2 if pretty else None
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=indent)

    # 检查文件大小
    file_size = os.path.getsize(output_path)
    size_mb = file_size / (1024 * 1024)

    print(f"✅ 转换完成")
    print(f"   - 输出文件: {output_path}")
    print(f"   - 文件大小: {file_size:,} 字节 ({size_mb:.2f} MB)")
    print(f"   - 格式: {format.upper()}")

    return True


def main():
    parser = argparse.ArgumentParser(description='Arrow格式转JSON工具')
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Arrow数据集目录路径')
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='输出JSON文件路径')
    parser.add_argument('--format', '-f', type=str, default='json',
                        choices=['json', 'jsonl'],
                        help='输出格式: json(数组) 或 jsonl(每行一个JSON)')
    parser.add_argument('--compact', action='store_true',
                        help='紧凑格式（不使用缩进，减小文件大小）')

    args = parser.parse_args()

    print("="*70)
    print("Arrow格式转JSON工具")
    print("="*70)

    success = convert_to_json(
        input_path=args.input,
        output_path=args.output,
        format=args.format,
        pretty=not args.compact
    )

    if success:
        print("\n" + "="*70)
        print("✅ 转换成功！")
        print("="*70)

        # 显示预览
        print("\n文件预览（前5行）:")
        print("-"*70)
        with open(args.output, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= 5:
                    break
                # 限制每行显示长度
                display_line = line[:100] + "..." if len(line) > 100 else line
                print(display_line.rstrip())
        print("-"*70)
    else:
        print("\n❌ 转换失败")
        exit(1)


if __name__ == '__main__':
    main()
