#!/usr/bin/env python3
"""
合并 Coldstart 格式的 ReBel 数据集
"""

import json
from pathlib import Path


def merge_coldstart_datasets(file1: str, file2: str, output_file: str):
    """
    合并两个 coldstart 格式的数据集

    Args:
        file1: 第一个数据集文件路径
        file2: 第二个数据集文件路径
        output_file: 输出文件路径
    """
    print("=" * 70)
    print("合并 Coldstart 格式数据集")
    print("=" * 70)

    # 加载第一个数据集
    print(f"\n【步骤1】加载第一个数据集: {file1}")
    with open(file1, 'r', encoding='utf-8') as f:
        data1 = json.load(f)
    print(f"  样本数: {len(data1)}")

    # 统计信息
    done_count1 = sum(1 for s in data1 if s['done'] == 'True')
    steps1 = [len(s['data']) for s in data1]
    print(f"  done=True: {done_count1} 条")
    print(f"  平均步数: {sum(steps1)/len(steps1):.1f}")

    # 加载第二个数据集
    print(f"\n【步骤2】加载第二个数据集: {file2}")
    with open(file2, 'r', encoding='utf-8') as f:
        data2 = json.load(f)
    print(f"  样本数: {len(data2)}")

    # 统计信息
    done_count2 = sum(1 for s in data2 if s['done'] == 'True')
    steps2 = [len(s['data']) for s in data2]
    print(f"  done=True: {done_count2} 条")
    print(f"  平均步数: {sum(steps2)/len(steps2):.1f}")

    # 合并
    print(f"\n【步骤3】合并数据集")
    merged_data = data1 + data2
    print(f"  合并后总样本数: {len(merged_data)}")

    # 统计合并后信息
    print(f"\n【步骤4】合并后数据集统计")
    done_count = sum(1 for s in merged_data if s['done'] == 'True')
    all_steps = [len(s['data']) for s in merged_data]

    print(f"  总样本数: {len(merged_data)}")
    print(f"  done=True: {done_count} 条 ({done_count/len(merged_data)*100:.1f}%)")
    print(f"  done=False: {len(merged_data) - done_count} 条")

    print(f"\n  步数统计:")
    print(f"    平均步数: {sum(all_steps)/len(all_steps):.1f}")
    print(f"    最小步数: {min(all_steps)}")
    print(f"    最大步数: {max(all_steps)}")

    # 统计任务类型分布
    task_types = {}
    for sample in merged_data:
        task = sample['task']
        task_type = task.split()[0] if task else 'unknown'
        task_types[task_type] = task_types.get(task_type, 0) + 1

    print(f"\n  任务类型分布:")
    for task_type, count in sorted(task_types.items(), key=lambda x: x[1], reverse=True):
        print(f"    {task_type}: {count} 条 ({count/len(merged_data)*100:.1f}%)")

    # 保存合并后的数据集
    print(f"\n【步骤5】保存合并后的数据集")
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=2)

    print(f"  ✓ 已保存到: {output_file}")

    # 文件大小
    file_size = output_path.stat().st_size / (1024 * 1024)
    print(f"  ✓ 文件大小: {file_size:.1f} MB")

    # 生成统计报告
    report_file = output_path.parent / "coldstart_merge_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("Coldstart 格式数据集合并报告\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"【数据来源】\n")
        f.write(f"  数据集1: {file1}\n")
        f.write(f"    - 样本数: {len(data1)}\n")
        f.write(f"    - done=True: {done_count1} 条\n")
        f.write(f"    - 平均步数: {sum(steps1)/len(steps1):.1f}\n\n")

        f.write(f"  数据集2: {file2}\n")
        f.write(f"    - 样本数: {len(data2)}\n")
        f.write(f"    - done=True: {done_count2} 条\n")
        f.write(f"    - 平均步数: {sum(steps2)/len(steps2):.1f}\n\n")

        f.write(f"【合并后数据集】\n")
        f.write(f"  总样本数: {len(merged_data)}\n")
        f.write(f"  done=True: {done_count} 条 ({done_count/len(merged_data)*100:.1f}%)\n")
        f.write(f"  done=False: {len(merged_data) - done_count} 条\n\n")

        f.write(f"【步数统计】\n")
        f.write(f"  平均步数: {sum(all_steps)/len(all_steps):.1f}\n")
        f.write(f"  最小步数: {min(all_steps)}\n")
        f.write(f"  最大步数: {max(all_steps)}\n\n")

        f.write(f"【任务类型分布】\n")
        for task_type, count in sorted(task_types.items(), key=lambda x: x[1], reverse=True):
            f.write(f"  {task_type}: {count} 条 ({count/len(merged_data)*100:.1f}%)\n")

    print(f"  ✓ 统计报告已保存: {report_file}")

    print("\n" + "=" * 70)
    print("✓ 数据集合并完成！")
    print("=" * 70)
    print(f"\n输出文件: {output_file}")
    print(f"总样本数: {len(merged_data)}")
    print(f"成功率: {done_count/len(merged_data)*100:.1f}%")

    return merged_data


if __name__ == "__main__":
    # 配置文件路径
    file1 = "/root/testttt/RLVMR/code/data/alfworld_rebel_250_new/rebel_coldstart.json"
    file2 = "/root/testttt/RLVMR/code/data/alfworld_rebel_full_improved_coldstart.json"
    output_file = "/root/testttt/RLVMR/code/data/alfworld_rebel_merged_final/rebel_coldstart.json"

    # 执行合并
    merged_data = merge_coldstart_datasets(file1, file2, output_file)
