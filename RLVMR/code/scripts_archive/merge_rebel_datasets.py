#!/usr/bin/env python3
"""
合并ReBel数据集脚本
从 alfworld_rebel_full_improved 中提取成功轨迹，与 alfworld_rebel_250_new 合并
"""

import json
from datasets import load_from_disk, Dataset
from pathlib import Path
import shutil

def load_and_filter_successful_trajectories(dataset_path, success_threshold=1.0):
    """加载数据集并筛选成功轨迹"""
    print(f"\n加载数据集: {dataset_path}")
    dataset = load_from_disk(dataset_path)

    print(f"  总样本数: {len(dataset)}")

    # 筛选成功率达标的轨迹
    successful_samples = []
    for sample in dataset:
        if sample['annotation_success_rate'] >= success_threshold:
            successful_samples.append(sample)

    print(f"  成功率 >= {success_threshold} 的样本: {len(successful_samples)} 条")
    return successful_samples

def merge_datasets(dataset1_path, dataset2_path, output_path, success_threshold=1.0):
    """合并两个数据集"""
    print("=" * 70)
    print("开始合并ReBel数据集")
    print("=" * 70)

    # 加载第一个数据集（alfworld_rebel_250_new）- 全部使用
    print(f"\n【步骤1】加载第一个数据集: {dataset1_path}")
    dataset1 = load_from_disk(dataset1_path)
    samples1 = [sample for sample in dataset1]
    print(f"  加载样本数: {len(samples1)}")

    # 统计成功率
    success_rates1 = [s['annotation_success_rate'] for s in samples1]
    print(f"  平均成功率: {sum(success_rates1)/len(success_rates1):.2%}")
    print(f"  成功率 = 1.0: {sum(1 for r in success_rates1 if r == 1.0)} 条")

    # 加载第二个数据集并筛选成功轨迹
    print(f"\n【步骤2】加载并筛选第二个数据集: {dataset2_path}")
    samples2 = load_and_filter_successful_trajectories(dataset2_path, success_threshold)

    # 合并
    print(f"\n【步骤3】合并数据集")
    all_samples = samples1 + samples2
    print(f"  合并后总样本数: {len(all_samples)}")

    # 统计合并后的信息
    print(f"\n【步骤4】统计合并后数据集信息")
    all_success_rates = [s['annotation_success_rate'] for s in all_samples]
    print(f"  平均成功率: {sum(all_success_rates)/len(all_success_rates):.2%}")
    print(f"  成功率 = 1.0: {sum(1 for r in all_success_rates if r == 1.0)} 条")
    print(f"  成功率 >= 0.8: {sum(1 for r in all_success_rates if r >= 0.8)} 条")

    # 统计任务类型分布
    task_types = {}
    for sample in all_samples:
        task = sample['task']
        task_type = task.split()[0] if task else 'unknown'
        task_types[task_type] = task_types.get(task_type, 0) + 1

    print(f"\n  任务类型分布:")
    for task_type, count in sorted(task_types.items(), key=lambda x: x[1], reverse=True):
        print(f"    {task_type}: {count} 条 ({count/len(all_samples)*100:.1f}%)")

    # 统计步数
    num_steps = [s['num_steps'] for s in all_samples]
    print(f"\n  步数统计:")
    print(f"    平均步数: {sum(num_steps)/len(num_steps):.1f}")
    print(f"    最小步数: {min(num_steps)}")
    print(f"    最大步数: {max(num_steps)}")

    # 保存合并后的数据集
    print(f"\n【步骤5】保存合并后的数据集到: {output_path}")
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 转换为Dataset格式并保存为Arrow
    merged_dataset = Dataset.from_list(all_samples)
    merged_dataset.save_to_disk(output_path)
    print(f"  ✓ Arrow格式已保存")

    # 保存为JSONL格式
    jsonl_path = output_dir / "rebel_hindsight.jsonl"
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for sample in all_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    print(f"  ✓ JSONL格式已保存: {jsonl_path}")

    # 保存为JSON格式（用于某些训练脚本）
    json_path = output_dir / "rebel_coldstart.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_samples, f, ensure_ascii=False, indent=2)
    print(f"  ✓ JSON格式已保存: {json_path}")

    # 生成数据集统计报告
    report_path = output_dir / "dataset_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("ReBel合并数据集报告\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"【数据来源】\n")
        f.write(f"  数据集1: {dataset1_path}\n")
        f.write(f"    - 样本数: {len(samples1)}\n")
        f.write(f"    - 平均成功率: {sum(success_rates1)/len(success_rates1):.2%}\n\n")

        f.write(f"  数据集2: {dataset2_path}\n")
        f.write(f"    - 原始样本数: {len(load_from_disk(dataset2_path))}\n")
        f.write(f"    - 筛选后样本数: {len(samples2)}\n")
        f.write(f"    - 筛选阈值: 成功率 >= {success_threshold}\n\n")

        f.write(f"【合并后数据集】\n")
        f.write(f"  总样本数: {len(all_samples)}\n")
        f.write(f"  平均成功率: {sum(all_success_rates)/len(all_success_rates):.2%}\n")
        f.write(f"  成功率 = 1.0: {sum(1 for r in all_success_rates if r == 1.0)} 条\n")
        f.write(f"  成功率 >= 0.8: {sum(1 for r in all_success_rates if r >= 0.8)} 条\n\n")

        f.write(f"【任务类型分布】\n")
        for task_type, count in sorted(task_types.items(), key=lambda x: x[1], reverse=True):
            f.write(f"  {task_type}: {count} 条 ({count/len(all_samples)*100:.1f}%)\n")

        f.write(f"\n【步数统计】\n")
        f.write(f"  平均步数: {sum(num_steps)/len(num_steps):.1f}\n")
        f.write(f"  最小步数: {min(num_steps)}\n")
        f.write(f"  最大步数: {max(num_steps)}\n")

        f.write(f"\n【文件列表】\n")
        f.write(f"  - data-00000-of-00001.arrow (Arrow格式)\n")
        f.write(f"  - rebel_hindsight.jsonl (JSONL格式)\n")
        f.write(f"  - rebel_coldstart.json (JSON格式)\n")
        f.write(f"  - dataset_info.json (元信息)\n")
        f.write(f"  - dataset_report.txt (本报告)\n")

    print(f"  ✓ 统计报告已保存: {report_path}")

    print("\n" + "=" * 70)
    print("✓ 数据集合并完成！")
    print("=" * 70)
    print(f"\n输出目录: {output_path}")
    print(f"总样本数: {len(all_samples)}")
    print(f"平均成功率: {sum(all_success_rates)/len(all_success_rates):.2%}")

    return merged_dataset

if __name__ == "__main__":
    # 配置路径
    dataset1_path = "/root/testttt/RLVMR/code/data/alfworld_rebel_250_new"
    dataset2_path = "/root/testttt/RLVMR/code/data/alfworld_rebel_full_improved"
    output_path = "/root/testttt/RLVMR/code/data/alfworld_rebel_merged_final"

    # 成功率阈值（1.0表示只保留100%成功的轨迹）
    success_threshold = 1.0

    # 执行合并
    merged_dataset = merge_datasets(
        dataset1_path=dataset1_path,
        dataset2_path=dataset2_path,
        output_path=output_path,
        success_threshold=success_threshold
    )
