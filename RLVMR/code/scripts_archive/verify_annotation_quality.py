#!/usr/bin/env python3
"""
ReBel标注质量检验脚本

检查生成的ReBel数据集质量，包括：
1. 标注成功率
2. Belief完整性
3. Inventory追踪准确性
4. Cleared receptacles逻辑
5. Reasoning质量
6. 数据格式正确性
"""

import os
import sys
import json
import re
import argparse
from collections import defaultdict, Counter
from typing import Dict, List, Any
from datasets import load_from_disk

# ============================================================================
# 质量检查函数
# ============================================================================

def check_belief_completeness(belief_update: Dict) -> Dict[str, bool]:
    """检查belief更新的完整性"""
    checks = {
        'has_world_model': 'world_model_update' in belief_update,
        'has_task_progress': 'task_progress_update' in belief_update,
        'has_exploration_map': 'exploration_map_update' in belief_update,
    }

    if checks['has_world_model']:
        wm = belief_update['world_model_update']
        checks['has_found_objects'] = 'found_objects' in wm
        checks['has_inventory'] = 'inventory' in wm
        checks['has_state_changes'] = 'state_changes' in wm
        checks['has_cleared_receptacles'] = 'cleared_receptacles' in wm

    if checks['has_task_progress']:
        tp = belief_update['task_progress_update']
        checks['has_subgoal_status'] = 'subgoal_status' in tp
        checks['has_evidence'] = 'evidence' in tp
        checks['has_updated_subgoal'] = 'updated_subgoal' in tp

    return checks


def check_inventory_logic(conversations: List[Dict]) -> Dict[str, Any]:
    """检查inventory追踪逻辑"""
    inventory_updates = []
    actions = []
    errors = []

    for turn in conversations:
        if turn['from'] == 'gpt' and '<belief>' in turn['value']:
            # 提取belief
            belief_match = re.search(r'<belief>\s*(\{.*?\})\s*</belief>', turn['value'], re.DOTALL)
            if belief_match:
                try:
                    belief = json.loads(belief_match.group(1))
                    if 'world_model_update' in belief and 'inventory' in belief['world_model_update']:
                        inventory = belief['world_model_update']['inventory']
                        inventory_updates.append(inventory)
                except:
                    pass

            # 提取action
            action_match = re.search(r'<action>\s*(.+?)\s*</action>', turn['value'], re.DOTALL)
            if action_match:
                action = action_match.group(1).strip()
                actions.append(action)

    # 检查逻辑
    for i, action in enumerate(actions):
        if i >= len(inventory_updates):
            continue

        inventory = inventory_updates[i]

        # 检查take动作
        if 'take' in action.lower() and 'from' in action.lower():
            obj_match = re.search(r'take\s+([\w\s]+\d+)\s+from', action.lower())
            if obj_match:
                expected_obj = obj_match.group(1).strip()
                if inventory is None or expected_obj not in str(inventory).lower():
                    errors.append({
                        'step': i,
                        'action': action,
                        'expected_inventory': expected_obj,
                        'actual_inventory': inventory,
                        'error': 'Inventory not updated after take action'
                    })

        # 检查put动作
        if 'put' in action.lower() or 'place' in action.lower():
            if inventory is not None:
                errors.append({
                    'step': i,
                    'action': action,
                    'inventory': inventory,
                    'error': 'Inventory should be null after put action'
                })

    return {
        'total_inventory_updates': len(inventory_updates),
        'take_actions': sum(1 for a in actions if 'take' in a.lower()),
        'put_actions': sum(1 for a in actions if 'put' in a.lower()),
        'errors': errors,
        'error_rate': len(errors) / max(len(actions), 1)
    }


def check_cleared_receptacles(conversations: List[Dict]) -> Dict[str, Any]:
    """检查cleared receptacles逻辑"""
    prev_action = None
    cleared_updates = []
    potential_patterns = []

    for turn in conversations:
        if turn['from'] == 'gpt' and '<belief>' in turn['value']:
            # 提取action
            action_match = re.search(r'<action>\s*(.+?)\s*</action>', turn['value'], re.DOTALL)
            current_action = action_match.group(1).strip() if action_match else None

            # 提取cleared receptacles
            belief_match = re.search(r'<belief>\s*(\{.*?\})\s*</belief>', turn['value'], re.DOTALL)
            if belief_match:
                try:
                    belief = json.loads(belief_match.group(1))
                    if 'world_model_update' in belief:
                        cleared = belief['world_model_update'].get('cleared_receptacles', [])
                        if cleared:
                            cleared_updates.append({
                                'cleared': cleared,
                                'prev_action': prev_action,
                                'current_action': current_action
                            })
                except:
                    pass

            # 检测模式: open X -> go to Y
            if prev_action and 'open' in prev_action.lower() and current_action and 'go to' in current_action.lower():
                potential_patterns.append({
                    'prev': prev_action,
                    'current': current_action
                })

            prev_action = current_action

    return {
        'total_cleared_updates': len(cleared_updates),
        'potential_patterns': len(potential_patterns),
        'patterns_captured': len([u for u in cleared_updates if u['prev_action'] and 'open' in u['prev_action'].lower()]),
        'coverage_rate': len([u for u in cleared_updates if u['prev_action'] and 'open' in u['prev_action'].lower()]) / max(len(potential_patterns), 1)
    }


def check_reasoning_quality(conversations: List[Dict]) -> Dict[str, Any]:
    """检查reasoning质量"""
    reasonings = []

    for turn in conversations:
        if turn['from'] == 'gpt' and '<reasoning>' in turn['value']:
            reasoning_match = re.search(r'<reasoning>\s*(.+?)\s*</reasoning>', turn['value'], re.DOTALL)
            if reasoning_match:
                reasoning = reasoning_match.group(1).strip()
                reasonings.append({
                    'text': reasoning,
                    'length': len(reasoning),
                    'has_evidence': 'evidence' in reasoning.lower() or 'observe' in reasoning.lower() or 'see' in reasoning.lower(),
                    'has_goal': 'goal' in reasoning.lower() or 'task' in reasoning.lower() or 'objective' in reasoning.lower(),
                    'has_strategy': 'strategy' in reasoning.lower() or 'plan' in reasoning.lower() or 'because' in reasoning.lower() or 'since' in reasoning.lower(),
                })

    if not reasonings:
        return {'total': 0}

    return {
        'total': len(reasonings),
        'avg_length': sum(r['length'] for r in reasonings) / len(reasonings),
        'has_evidence_rate': sum(r['has_evidence'] for r in reasonings) / len(reasonings),
        'has_goal_rate': sum(r['has_goal'] for r in reasonings) / len(reasonings),
        'has_strategy_rate': sum(r['has_strategy'] for r in reasonings) / len(reasonings),
        'too_short': sum(1 for r in reasonings if r['length'] < 50),
        'too_long': sum(1 for r in reasonings if r['length'] > 500),
    }


def check_format_validity(sample: Dict) -> List[str]:
    """检查数据格式有效性"""
    errors = []

    if 'conversations' not in sample:
        errors.append("Missing 'conversations' field")
        return errors

    conversations = sample['conversations']

    # 检查交替模式
    expected_role = 'human'
    for i, turn in enumerate(conversations):
        if i == 0 or i == 1:  # Skip system messages
            continue

        if turn['from'] != expected_role:
            errors.append(f"Step {i}: Expected '{expected_role}', got '{turn['from']}'")

        expected_role = 'gpt' if expected_role == 'human' else 'human'

    # 检查GPT输出格式
    for i, turn in enumerate(conversations):
        if turn['from'] == 'gpt' and i > 1:  # Skip system acknowledgment
            value = turn['value']
            if '<belief>' not in value:
                errors.append(f"Step {i}: Missing <belief> tag")
            if '<reasoning>' not in value:
                errors.append(f"Step {i}: Missing <reasoning> tag")
            if '<action>' not in value:
                errors.append(f"Step {i}: Missing <action> tag")

            # 检查belief是否为有效JSON
            belief_match = re.search(r'<belief>\s*(\{.*?\})\s*</belief>', value, re.DOTALL)
            if belief_match:
                try:
                    json.loads(belief_match.group(1))
                except json.JSONDecodeError:
                    errors.append(f"Step {i}: Invalid JSON in <belief>")

    # 检查Human输入格式
    for i, turn in enumerate(conversations):
        if turn['from'] == 'human' and i > 0:  # Skip system prompt
            value = turn['value']
            if 'Task:' not in value:
                errors.append(f"Step {i}: Missing 'Task:' in human turn")
            if 'Observation:' not in value:
                errors.append(f"Step {i}: Missing 'Observation:' in human turn")
            if 'Current Belief State:' not in value:
                errors.append(f"Step {i}: Missing 'Current Belief State:' in human turn")
            if 'Available Actions:' not in value:
                errors.append(f"Step {i}: Missing 'Available Actions:' in human turn")

    return errors


# ============================================================================
# 主检验函数
# ============================================================================

def verify_dataset_quality(dataset_path: str, output_report: str = None):
    """验证整个数据集的质量"""

    print("=" * 80)
    print("ReBel标注质量检验")
    print("=" * 80)
    print(f"数据集路径: {dataset_path}")
    print()

    # 加载数据
    print("加载数据...")
    if dataset_path.endswith('.jsonl') or dataset_path.endswith('.json'):
        with open(dataset_path, 'r') as f:
            content = f.read().strip()
            if content.startswith('['):
                data = json.loads(content)
            else:
                data = [json.loads(line) for line in content.splitlines() if line.strip()]
    else:
        dataset = load_from_disk(dataset_path)
        data = list(dataset)

    print(f"✅ 加载了 {len(data)} 个样本")
    print()

    # 统计信息
    total_samples = len(data)
    total_steps = sum(len(s.get('conversations', [])) // 2 - 1 for s in data)  # -1 for system message

    # 检查各项指标
    print("=" * 80)
    print("1. 基础统计")
    print("=" * 80)
    print(f"样本数量: {total_samples}")
    print(f"总步数: {total_steps}")
    print(f"平均步数/样本: {total_steps / max(total_samples, 1):.1f}")

    # 标注成功率
    annotation_success_rates = [s.get('annotation_success_rate', 0) for s in data if 'annotation_success_rate' in s]
    if annotation_success_rates:
        avg_success_rate = sum(annotation_success_rates) / len(annotation_success_rates)
        print(f"平均标注成功率: {avg_success_rate:.1%}")

        # 分级
        if avg_success_rate >= 0.9:
            grade = "⭐⭐⭐⭐⭐ 优秀"
        elif avg_success_rate >= 0.7:
            grade = "⭐⭐⭐⭐ 良好"
        elif avg_success_rate >= 0.5:
            grade = "⭐⭐⭐ 可用"
        else:
            grade = "⭐⭐ 需改进"
        print(f"质量评级: {grade}")
    print()

    # Belief完整性检查
    print("=" * 80)
    print("2. Belief完整性检查")
    print("=" * 80)

    completeness_stats = defaultdict(int)
    total_beliefs = 0

    for sample in data:
        for turn in sample.get('conversations', []):
            if turn['from'] == 'gpt' and '<belief>' in turn['value']:
                belief_match = re.search(r'<belief>\s*(\{.*?\})\s*</belief>', turn['value'], re.DOTALL)
                if belief_match:
                    try:
                        belief = json.loads(belief_match.group(1))
                        checks = check_belief_completeness(belief)
                        for key, value in checks.items():
                            if value:
                                completeness_stats[key] += 1
                        total_beliefs += 1
                    except:
                        pass

    if total_beliefs > 0:
        print(f"检查的Belief数量: {total_beliefs}")
        for key, count in sorted(completeness_stats.items()):
            rate = count / total_beliefs
            status = "✅" if rate >= 0.9 else "⚠️" if rate >= 0.7 else "❌"
            print(f"  {status} {key}: {rate:.1%} ({count}/{total_beliefs})")
    print()

    # Inventory追踪检查
    print("=" * 80)
    print("3. Inventory追踪准确性")
    print("=" * 80)

    inventory_stats = {
        'total_inventory_updates': 0,
        'take_actions': 0,
        'put_actions': 0,
        'errors': [],
        'error_rate': 0
    }

    for sample in data:
        result = check_inventory_logic(sample.get('conversations', []))
        inventory_stats['total_inventory_updates'] += result['total_inventory_updates']
        inventory_stats['take_actions'] += result['take_actions']
        inventory_stats['put_actions'] += result['put_actions']
        inventory_stats['errors'].extend(result['errors'])

    if inventory_stats['total_inventory_updates'] > 0:
        error_rate = len(inventory_stats['errors']) / inventory_stats['total_inventory_updates']
        inventory_stats['error_rate'] = error_rate

        print(f"Inventory更新次数: {inventory_stats['total_inventory_updates']}")
        print(f"Take动作数: {inventory_stats['take_actions']}")
        print(f"Put动作数: {inventory_stats['put_actions']}")
        print(f"逻辑错误数: {len(inventory_stats['errors'])}")
        print(f"错误率: {error_rate:.1%}")

        status = "✅ 优秀" if error_rate < 0.05 else "⚠️ 良好" if error_rate < 0.15 else "❌ 需改进"
        print(f"评估: {status}")

        if len(inventory_stats['errors']) > 0 and len(inventory_stats['errors']) <= 5:
            print("\n示例错误:")
            for err in inventory_stats['errors'][:3]:
                print(f"  - Step {err['step']}: {err['error']}")
                print(f"    Action: {err['action']}")
    else:
        print("未检测到inventory更新（可能不涉及take/put任务）")
    print()

    # Cleared receptacles检查
    print("=" * 80)
    print("4. Cleared Receptacles逻辑")
    print("=" * 80)

    cleared_stats = {
        'total_cleared_updates': 0,
        'potential_patterns': 0,
        'patterns_captured': 0,
        'coverage_rate': 0
    }

    for sample in data:
        result = check_cleared_receptacles(sample.get('conversations', []))
        cleared_stats['total_cleared_updates'] += result['total_cleared_updates']
        cleared_stats['potential_patterns'] += result['potential_patterns']
        cleared_stats['patterns_captured'] += result['patterns_captured']

    if cleared_stats['potential_patterns'] > 0:
        cleared_stats['coverage_rate'] = cleared_stats['patterns_captured'] / cleared_stats['potential_patterns']

        print(f"Cleared更新次数: {cleared_stats['total_cleared_updates']}")
        print(f"检测到的'open->go'模式: {cleared_stats['potential_patterns']}")
        print(f"正确捕获的模式: {cleared_stats['patterns_captured']}")
        print(f"覆盖率: {cleared_stats['coverage_rate']:.1%}")

        status = "✅ 优秀" if cleared_stats['coverage_rate'] >= 0.8 else "⚠️ 良好" if cleared_stats['coverage_rate'] >= 0.5 else "❌ 需改进"
        print(f"评估: {status}")
    else:
        print("未检测到cleared receptacles更新（可能不涉及搜索任务）")
    print()

    # Reasoning质量检查
    print("=" * 80)
    print("5. Reasoning质量")
    print("=" * 80)

    all_reasoning_stats = []
    for sample in data:
        result = check_reasoning_quality(sample.get('conversations', []))
        if result.get('total', 0) > 0:
            all_reasoning_stats.append(result)

    if all_reasoning_stats:
        avg_length = sum(r['avg_length'] for r in all_reasoning_stats) / len(all_reasoning_stats)
        avg_evidence = sum(r['has_evidence_rate'] for r in all_reasoning_stats) / len(all_reasoning_stats)
        avg_goal = sum(r['has_goal_rate'] for r in all_reasoning_stats) / len(all_reasoning_stats)
        avg_strategy = sum(r['has_strategy_rate'] for r in all_reasoning_stats) / len(all_reasoning_stats)
        total_too_short = sum(r['too_short'] for r in all_reasoning_stats)
        total_too_long = sum(r['too_long'] for r in all_reasoning_stats)

        print(f"平均Reasoning长度: {avg_length:.0f} 字符")
        print(f"包含证据/观测: {avg_evidence:.1%}")
        print(f"提及任务目标: {avg_goal:.1%}")
        print(f"说明策略/原因: {avg_strategy:.1%}")
        print(f"过短(<50字符): {total_too_short} 个")
        print(f"过长(>500字符): {total_too_long} 个")

        quality_score = (avg_evidence + avg_goal + avg_strategy) / 3
        if quality_score >= 0.8:
            status = "✅ 优秀"
        elif quality_score >= 0.6:
            status = "⚠️ 良好"
        else:
            status = "❌ 需改进"
        print(f"综合评估: {status} (得分: {quality_score:.1%})")
    print()

    # 格式有效性检查
    print("=" * 80)
    print("6. 数据格式有效性")
    print("=" * 80)

    format_errors = []
    for i, sample in enumerate(data):
        errors = check_format_validity(sample)
        if errors:
            format_errors.append({'sample_id': i, 'errors': errors})

    print(f"检查样本数: {len(data)}")
    print(f"格式错误样本数: {len(format_errors)}")
    print(f"格式正确率: {(1 - len(format_errors) / max(len(data), 1)):.1%}")

    if format_errors:
        print("\n示例格式错误 (前3个):")
        for err_info in format_errors[:3]:
            print(f"  样本 {err_info['sample_id']}:")
            for err in err_info['errors'][:2]:
                print(f"    - {err}")
    else:
        print("✅ 所有样本格式正确")
    print()

    # 综合评估
    print("=" * 80)
    print("7. 综合评估")
    print("=" * 80)

    scores = []
    weights = []

    # 标注成功率 (权重30%)
    if annotation_success_rates:
        scores.append(avg_success_rate)
        weights.append(0.3)

    # Belief完整性 (权重20%)
    if total_beliefs > 0:
        completeness_score = sum(completeness_stats.values()) / (len(completeness_stats) * total_beliefs)
        scores.append(completeness_score)
        weights.append(0.2)

    # Inventory准确性 (权重15%)
    if inventory_stats['total_inventory_updates'] > 0:
        scores.append(1 - inventory_stats['error_rate'])
        weights.append(0.15)

    # Cleared逻辑 (权重15%)
    if cleared_stats['potential_patterns'] > 0:
        scores.append(cleared_stats['coverage_rate'])
        weights.append(0.15)

    # Reasoning质量 (权重10%)
    if all_reasoning_stats:
        scores.append(quality_score)
        weights.append(0.1)

    # 格式正确性 (权重10%)
    scores.append(1 - len(format_errors) / max(len(data), 1))
    weights.append(0.1)

    # 归一化权重
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]

    # 计算加权平均
    final_score = sum(s * w for s, w in zip(scores, weights))

    print(f"最终质量得分: {final_score:.1%}")

    if final_score >= 0.85:
        recommendation = "✅ 优秀 - 可直接用于训练"
    elif final_score >= 0.70:
        recommendation = "⚠️ 良好 - 建议检查低分项后使用"
    elif final_score >= 0.55:
        recommendation = "⚠️ 可用 - 建议使用更强的Teacher模型重新标注"
    else:
        recommendation = "❌ 需改进 - 必须使用更强的Teacher模型"

    print(f"推荐: {recommendation}")
    print()

    # 保存报告
    if output_report:
        with open(output_report, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("ReBel标注质量检验报告\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"数据集: {dataset_path}\n")
            f.write(f"生成时间: {os.path.getmtime(dataset_path) if os.path.exists(dataset_path) else 'Unknown'}\n\n")

            f.write("1. 基础统计\n")
            f.write(f"  - 样本数: {total_samples}\n")
            f.write(f"  - 总步数: {total_steps}\n")
            if annotation_success_rates:
                f.write(f"  - 标注成功率: {avg_success_rate:.1%}\n\n")

            f.write("2. 各项指标\n")
            f.write(f"  - Belief完整性: {completeness_score:.1%}\n" if total_beliefs > 0 else "")
            f.write(f"  - Inventory准确性: {1 - inventory_stats['error_rate']:.1%}\n" if inventory_stats['total_inventory_updates'] > 0 else "")
            f.write(f"  - Cleared逻辑覆盖: {cleared_stats['coverage_rate']:.1%}\n" if cleared_stats['potential_patterns'] > 0 else "")
            f.write(f"  - Reasoning质量: {quality_score:.1%}\n" if all_reasoning_stats else "")
            f.write(f"  - 格式正确率: {(1 - len(format_errors) / max(len(data), 1)):.1%}\n\n")

            f.write(f"3. 综合评估\n")
            f.write(f"  - 最终得分: {final_score:.1%}\n")
            f.write(f"  - 推荐: {recommendation}\n\n")

            if format_errors:
                f.write("4. 详细错误\n")
                for err_info in format_errors:
                    f.write(f"  样本 {err_info['sample_id']}:\n")
                    for err in err_info['errors']:
                        f.write(f"    - {err}\n")

        print(f"✅ 报告已保存到: {output_report}")

    return final_score


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="验证ReBel标注数据质量")
    parser.add_argument('--annotated_data', type=str, required=True,
                       help='标注数据路径 (JSONL文件或数据集目录)')
    parser.add_argument('--output_report', type=str, default=None,
                       help='输出报告路径 (可选)')

    args = parser.parse_args()

    score = verify_dataset_quality(args.annotated_data, args.output_report)

    # 返回值用于Shell脚本判断
    if score >= 0.70:
        sys.exit(0)  # 成功
    else:
        sys.exit(1)  # 需要改进


if __name__ == '__main__':
    main()
