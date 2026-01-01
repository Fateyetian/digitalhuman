#!/usr/bin/env python3
"""
测试智能动作匹配功能
验证 match_action_to_admissible 函数是否正确处理 in/on 模糊介词
"""

import sys
sys.path.insert(0, '/root/testttt/RLVMR/code')

from agent_system.environments.env_package.alfworld.projection import match_action_to_admissible


def test_action_matching():
    """测试各种动作匹配场景"""

    print("=" * 70)
    print("动作智能匹配功能测试")
    print("=" * 70)

    # 测试场景1: in/on 模糊介词
    print("\n[测试1] 处理 in/on 模糊介词")
    print("-" * 70)

    admissible = [
        "go to table 1",
        "put apple 1 in drawer 1",
        "put book 1 on desk 1",
        "take key 1 from shelf 1",
        "open cabinet 1"
    ]

    test_cases = [
        ("put apple 1 in/on drawer 1", "put apple 1 in drawer 1"),
        ("put apple 1 on/in drawer 1", "put apple 1 in drawer 1"),
        ("put book 1 in/on desk 1", "put book 1 on desk 1"),
        ("put book 1 on/in desk 1", "put book 1 on desk 1"),
    ]

    for input_action, expected in test_cases:
        matched, corrected = match_action_to_admissible(input_action, admissible)
        status = "✅" if matched == expected else "❌"
        print(f"{status} 输入: {input_action:35s} -> 输出: {matched:35s} (修正: {corrected})")
        if matched != expected:
            print(f"   期望: {expected}")

    # 测试场景2: 直接匹配（无需修正）
    print("\n[测试2] 直接匹配（无需修正）")
    print("-" * 70)

    direct_cases = [
        "go to table 1",
        "take key 1 from shelf 1",
        "open cabinet 1"
    ]

    for action in direct_cases:
        matched, corrected = match_action_to_admissible(action, admissible)
        status = "✅" if not corrected and matched == action else "❌"
        print(f"{status} 输入: {action:35s} -> 修正: {corrected}")

    # 测试场景3: 无匹配动作（应返回原动作）
    print("\n[测试3] 无法匹配的动作")
    print("-" * 70)

    no_match_cases = [
        "put cellphone 1 in/on sidetable 1",  # sidetable不在admissible中
        "jump over table 1",  # 完全不相关的动作
    ]

    for action in no_match_cases:
        matched, corrected = match_action_to_admissible(action, admissible)
        print(f"⚠️  输入: {action:35s} -> 输出: {matched:35s} (修正: {corrected})")

    # 测试场景4: 多种介词组合
    print("\n[测试4] 复杂场景 - 多个可能的匹配")
    print("-" * 70)

    complex_admissible = [
        "put apple 1 in fridge 1",
        "put apple 1 on table 1",
        "put book 1 in drawer 1",
        "put book 1 on shelf 1",
    ]

    complex_cases = [
        ("put apple 1 in/on fridge 1", "put apple 1 in fridge 1"),  # fridge应该用in
        ("put apple 1 in/on table 1", "put apple 1 on table 1"),    # table应该用on
        ("put book 1 in/on drawer 1", "put book 1 in drawer 1"),    # drawer应该用in
        ("put book 1 in/on shelf 1", "put book 1 on shelf 1"),      # shelf应该用on
    ]

    for input_action, expected in complex_cases:
        matched, corrected = match_action_to_admissible(input_action, complex_admissible)
        status = "✅" if matched == expected else "❌"
        print(f"{status} 输入: {input_action:35s} -> 输出: {matched:35s}")
        if matched != expected:
            print(f"   期望: {expected}")

    # 测试场景5: 空白和格式问题
    print("\n[测试5] 处理额外空白")
    print("-" * 70)

    whitespace_admissible = ["go to table 1", "open drawer 1"]
    whitespace_cases = [
        ("go  to  table  1", "go to table 1"),  # 多余空格
        ("  go to table 1  ", "go to table 1"),  # 前后空格
    ]

    for input_action, expected in whitespace_cases:
        matched, corrected = match_action_to_admissible(input_action, whitespace_admissible)
        status = "✅" if matched == expected else "❌"
        print(f"{status} 输入: '{input_action}' -> 输出: '{matched}' (修正: {corrected})")

    print("\n" + "=" * 70)
    print("✅ 测试完成")
    print("=" * 70)


def test_with_real_coldstart_data():
    """使用真实的coldstart数据测试"""
    import json

    print("\n\n" + "=" * 70)
    print("真实数据测试")
    print("=" * 70)

    # 读取coldstart数据
    with open('/root/testttt/RLVMR/code/data/alfworld_rebel_merged_final/rebel_coldstart_clean.json') as f:
        data = json.load(f)

    # 提取前5个put动作
    put_actions = []
    for sample in data[:10]:
        for step_data in sample.get('data', []):
            response = step_data.get('response', '')
            if '<action>' in response and 'put' in response.lower():
                import re
                match = re.search(r'<action>(.*?)</action>', response, re.DOTALL | re.IGNORECASE)
                if match:
                    action = match.group(1).strip()
                    if action.startswith('put'):
                        put_actions.append(action)
                        if len(put_actions) >= 5:
                            break
        if len(put_actions) >= 5:
            break

    print(f"\n提取了 {len(put_actions)} 个真实的put动作")
    print("\n模拟修正过程:")
    print("-" * 70)

    # 模拟环境的admissible_actions
    for action in put_actions:
        # 创建两种可能的admissible actions
        if 'in/on' in action:
            action_with_in = action.replace('in/on', 'in')
            action_with_on = action.replace('in/on', 'on')

            # 模拟: 只有其中一个在admissible中
            # 根据容器类型选择正确的
            if any(cont in action.lower() for cont in ['drawer', 'cabinet', 'fridge', 'microwave']):
                admissible = [action_with_in]
                expected = action_with_in
            else:
                admissible = [action_with_on]
                expected = action_with_on

            matched, corrected = match_action_to_admissible(action, admissible)
            status = "✅" if matched == expected else "❌"
            print(f"{status} 原动作: {action}")
            print(f"   修正为: {matched}")
            print(f"   是否修正: {corrected}")
            print()


if __name__ == '__main__':
    test_action_matching()
    test_with_real_coldstart_data()
