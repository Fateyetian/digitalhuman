#!/usr/bin/env python3
"""
ReBel实验自动化报告生成脚本
从Wandb和本地文件收集各阶段指标，生成完整实验报告
"""

import os
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='生成ReBel实验报告')
    parser.add_argument('--sft_eval_dir', type=str, help='SFT评测结果目录 (rebel_rollout_results/run_*)')
    parser.add_argument('--rl_eval_dir', type=str, help='RL评测结果目录 (rebel_rollout_results/run_*)')
    parser.add_argument('--output', type=str, default='EXPERIMENT_REPORT.md', help='输出报告文件名')
    parser.add_argument('--auto_find', action='store_true', help='自动查找最近的2个评测结果')
    return parser.parse_args()


def find_latest_eval_dirs(base_dir='rebel_rollout_results', n=2):
    """自动查找最近的n个评测目录"""
    base_path = Path(base_dir)
    if not base_path.exists():
        return []

    run_dirs = sorted(base_path.glob('run_*'), key=lambda p: p.stat().st_mtime, reverse=True)
    return [str(d) for d in run_dirs[:n]]


def load_eval_results(eval_dir: str) -> Optional[Dict]:
    """加载评测结果"""
    if not eval_dir:
        return None

    results_path = Path(eval_dir) / 'results.json'
    trajectories_path = Path(eval_dir) / 'trajectories.jsonl'

    # 优先读取results.json
    if results_path.exists():
        with open(results_path) as f:
            return json.load(f)

    # 如果没有results.json，从trajectories.jsonl计算
    if trajectories_path.exists():
        print(f"⚠️  未找到 {results_path}，从trajectories.jsonl计算指标...")
        return compute_metrics_from_trajectories(trajectories_path)

    return None


def compute_metrics_from_trajectories(traj_path: Path) -> Dict:
    """从trajectories.jsonl计算指标"""
    episodes = {}  # env_id -> episode data

    with open(traj_path) as f:
        for line in f:
            row = json.loads(line)
            env_id = row.get('env_id', 0)

            if env_id not in episodes:
                episodes[env_id] = {
                    'steps': 0,
                    'reward': 0,
                    'won': False,
                    'r_consistency_sum': 0,
                    'r_progress_sum': 0,
                    'r_exploration_sum': 0,
                    'r_format_sum': 0,
                    'belief_parsed': 0,
                    'belief_total': 0
                }

            ep = episodes[env_id]
            ep['steps'] += 1
            ep['reward'] += row.get('reward', 0)
            ep['won'] = row.get('won', False)

            # ReBel指标
            ep['r_consistency_sum'] += row.get('r_consistency', 0)
            ep['r_progress_sum'] += row.get('r_progress', 0)
            ep['r_exploration_sum'] += row.get('r_exploration', 0)
            ep['r_format_sum'] += row.get('r_format', 0) if 'r_format' in row else 0

            # Belief解析率
            ep['belief_total'] += 1
            if row.get('belief_parsed', False):
                ep['belief_parsed'] += 1

    # 计算汇总指标
    ep_list = list(episodes.values())
    total_eps = len(ep_list)

    if total_eps == 0:
        return {'error': 'No episodes found'}

    success_rate = sum(1 for ep in ep_list if ep['won']) / total_eps
    avg_length = np.mean([ep['steps'] for ep in ep_list])
    avg_reward = np.mean([ep['reward'] for ep in ep_list])

    # ReBel指标（per step平均）
    all_r_consistency = [ep['r_consistency_sum'] / max(ep['steps'], 1) for ep in ep_list]
    all_r_progress = [ep['r_progress_sum'] / max(ep['steps'], 1) for ep in ep_list]
    all_r_exploration = [ep['r_exploration_sum'] / max(ep['steps'], 1) for ep in ep_list]
    all_r_format = [ep['r_format_sum'] / max(ep['steps'], 1) for ep in ep_list]
    all_belief_parse = [ep['belief_parsed'] / max(ep['belief_total'], 1) for ep in ep_list]

    return {
        'basic_metrics': {
            'success_rate': success_rate,
            'avg_episode_length': avg_length,
            'avg_episode_reward': avg_reward
        },
        'rebel_metrics': {
            'avg_r_consistency': np.mean(all_r_consistency),
            'avg_r_progress': np.mean(all_r_progress),
            'avg_r_exploration': np.mean(all_r_exploration),
            'avg_r_format': np.mean(all_r_format),
            'avg_belief_parse_rate': np.mean(all_belief_parse)
        },
        'action_quality_metrics': {  # 新增
            'avg_repeat_action_rate': np.mean([ep.get('repeat_rate', 0) for ep in ep_list]) if any('repeat_rate' in ep for ep in ep_list) else 0,
            'avg_invalid_action_rate': 1.0 - (sum(all_valid_actions) / max(sum(all_total_actions), 1)) if hasattr(ep_list[0], 'get') else 0,
        },
        'computed_from': 'trajectories.jsonl',
        'total_episodes': total_eps
    }


def format_percentage(value: float) -> str:
    """格式化百分比"""
    return f"{value * 100:.1f}%"


def format_number(value: float, decimals: int = 2) -> str:
    """格式化数字"""
    return f"{value:.{decimals}f}"


def generate_report(sft_results: Optional[Dict], rl_results: Optional[Dict], output_path: str):
    """生成实验报告"""

    report = []
    report.append("# ReBel实验结果报告")
    report.append("")
    report.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    report.append("---")
    report.append("")

    # ==================== SFT评测结果 ====================
    report.append("## 1. Cold-start SFT 评测结果")
    report.append("")

    if sft_results and 'basic_metrics' in sft_results:
        bm = sft_results['basic_metrics']
        rm = sft_results.get('rebel_metrics', {})

        report.append("### 1.1 基础指标")
        report.append("")
        report.append("| 指标 | 数值 |")
        report.append("|------|------|")
        report.append(f"| 成功率 (Success Rate) | {format_percentage(bm['success_rate'])} |")
        report.append(f"| 平均步数 (Avg Steps) | {format_number(bm['avg_episode_length'], 1)} |")
        report.append(f"| 平均奖励 (Avg Reward) | {format_number(bm['avg_episode_reward'])} |")
        report.append("")

        report.append("### 1.2 ReBel密集奖励指标")
        report.append("")
        report.append("| 奖励类型 | 数值 | 说明 |")
        report.append("|---------|------|------|")
        report.append(f"| r_consistency | {format_number(rm.get('avg_r_consistency', 0), 4)} | Belief与环境状态的一致性 |")
        report.append(f"| r_progress | {format_number(rm.get('avg_r_progress', 0), 4)} | 任务进度理解 |")
        report.append(f"| r_exploration | {format_number(rm.get('avg_r_exploration', 0), 4)} | 探索效率 |")
        report.append(f"| r_format | {format_number(rm.get('avg_r_format', 0), 4)} | 输出格式正确性 |")
        report.append("")

        report.append("### 1.3 Belief State质量")
        report.append("")
        report.append("| 指标 | 数值 |")
        report.append("|------|------|")
        report.append(f"| Belief解析成功率 | {format_percentage(rm.get('avg_belief_parse_rate', 0))} |")
        report.append("")

        report.append("### 1.4 动作质量指标")
        report.append("")
        report.append("| 指标 | 数值 |")
        report.append("|------|------|")
        aqm = sft_results.get('action_quality_metrics', {})
        report.append(f"| 重复动作率 | {format_percentage(aqm.get('avg_repeat_action_rate', 0))} |")
        report.append(f"| 无效动作率 | {format_percentage(aqm.get('avg_invalid_action_rate', 0))} |")
        report.append("")

        # 评估与分析
        report.append("### 1.5 SFT阶段评估")
        report.append("")
        sr = bm['success_rate']
        bp = rm.get('avg_belief_parse_rate', 0)
        rc = rm.get('avg_r_consistency', 0)

        if sr >= 0.1:
            report.append(f"✅ **成功率达到 {format_percentage(sr)}**，Cold-start SFT有效")
        elif sr >= 0.05:
            report.append(f"⚠️  **成功率为 {format_percentage(sr)}**，需要改进训练数据或增加训练轮数")
        else:
            report.append(f"❌ **成功率仅 {format_percentage(sr)}**，模型几乎未学习到任务，请检查数据质量")

        report.append("")

        if bp >= 0.8:
            report.append(f"✅ **Belief解析率 {format_percentage(bp)}**，模型已学会ReBel格式")
        elif bp >= 0.5:
            report.append(f"⚠️  **Belief解析率 {format_percentage(bp)}**，格式学习不充分")
        else:
            report.append(f"❌ **Belief解析率 {format_percentage(bp)}**，模型未掌握ReBel格式，需重新训练")

        report.append("")

        if rc > 0.05:
            report.append(f"✅ **一致性奖励 {format_number(rc, 4)}**，Belief与环境有一定对齐")
        else:
            report.append(f"❌ **一致性奖励极低 ({format_number(rc, 4)})**，Belief质量差")

        report.append("")
    else:
        report.append("⚠️  未提供SFT评测结果")
        report.append("")

    report.append("---")
    report.append("")

    # ==================== RL评测结果 ====================
    report.append("## 2. RL训练后评测结果")
    report.append("")

    if rl_results and 'basic_metrics' in rl_results:
        bm = rl_results['basic_metrics']
        rm = rl_results.get('rebel_metrics', {})

        report.append("### 2.1 基础指标")
        report.append("")
        report.append("| 指标 | 数值 |")
        report.append("|------|------|")
        report.append(f"| 成功率 (Success Rate) | {format_percentage(bm['success_rate'])} |")
        report.append(f"| 平均步数 (Avg Steps) | {format_number(bm['avg_episode_length'], 1)} |")
        report.append(f"| 平均奖励 (Avg Reward) | {format_number(bm['avg_episode_reward'])} |")
        report.append("")

        report.append("### 2.2 ReBel密集奖励指标")
        report.append("")
        report.append("| 奖励类型 | 数值 | 说明 |")
        report.append("|---------|------|------|")
        report.append(f"| r_consistency | {format_number(rm.get('avg_r_consistency', 0), 4)} | Belief与环境状态的一致性 |")
        report.append(f"| r_progress | {format_number(rm.get('avg_r_progress', 0), 4)} | 任务进度理解 |")
        report.append(f"| r_exploration | {format_number(rm.get('avg_r_exploration', 0), 4)} | 探索效率 |")
        report.append(f"| r_format | {format_number(rm.get('avg_r_format', 0), 4)} | 输出格式正确性 |")
        report.append("")

        report.append("### 2.3 Belief State质量")
        report.append("")
        report.append("| 指标 | 数值 |")
        report.append("|------|------|")
        report.append(f"| Belief解析成功率 | {format_percentage(rm.get('avg_belief_parse_rate', 0))} |")
        report.append("")

        report.append("### 2.4 动作质量指标")
        report.append("")
        report.append("| 指标 | 数值 |")
        report.append("|------|------|")
        aqm = rl_results.get('action_quality_metrics', {})
        report.append(f"| 重复动作率 | {format_percentage(aqm.get('avg_repeat_action_rate', 0))} |")
        report.append(f"| 无效动作率 | {format_percentage(aqm.get('avg_invalid_action_rate', 0))} |")
        report.append("")

        # RL阶段评估
        report.append("### 2.5 RL阶段评估")
        report.append("")
        sr = bm['success_rate']

        if sr >= 0.3:
            report.append(f"🎉 **成功率达到 {format_percentage(sr)}**，超过预期目标(30%)！")
        elif sr >= 0.2:
            report.append(f"✅ **成功率为 {format_percentage(sr)}**，RL训练有效")
        elif sr >= 0.1:
            report.append(f"⚠️  **成功率为 {format_percentage(sr)}**，RL有改进但不明显，建议增加训练轮数")
        else:
            report.append(f"❌ **成功率仅 {format_percentage(sr)}**，RL训练可能有问题")

        report.append("")
    else:
        report.append("⚠️  未提供RL评测结果")
        report.append("")

    report.append("---")
    report.append("")

    # ==================== 对比分析 ====================
    if sft_results and rl_results and 'basic_metrics' in sft_results and 'basic_metrics' in rl_results:
        report.append("## 3. SFT vs RL 对比分析")
        report.append("")

        sft_bm = sft_results['basic_metrics']
        sft_rm = sft_results.get('rebel_metrics', {})
        rl_bm = rl_results['basic_metrics']
        rl_rm = rl_results.get('rebel_metrics', {})

        report.append("### 3.1 关键指标对比")
        report.append("")
        report.append("| 指标 | SFT | RL | 提升 |")
        report.append("|------|-----|----|----|")

        # 成功率对比
        sft_sr = sft_bm['success_rate']
        rl_sr = rl_bm['success_rate']
        sr_improvement = (rl_sr - sft_sr) / max(sft_sr, 0.001) * 100
        report.append(f"| 成功率 | {format_percentage(sft_sr)} | {format_percentage(rl_sr)} | +{format_number(sr_improvement, 1)}% |")

        # 平均步数（越少越好）
        sft_as = sft_bm['avg_episode_length']
        rl_as = rl_bm['avg_episode_length']
        report.append(f"| 平均步数 | {format_number(sft_as, 1)} | {format_number(rl_as, 1)} | {format_number(rl_as - sft_as, 1)} |")

        # Belief解析率
        sft_bp = sft_rm.get('avg_belief_parse_rate', 0)
        rl_bp = rl_rm.get('avg_belief_parse_rate', 0)
        report.append(f"| Belief解析率 | {format_percentage(sft_bp)} | {format_percentage(rl_bp)} | {'+' if rl_bp > sft_bp else ''}{format_number((rl_bp - sft_bp) * 100, 1)}% |")

        # 一致性奖励
        sft_rc = sft_rm.get('avg_r_consistency', 0)
        rl_rc = rl_rm.get('avg_r_consistency', 0)
        report.append(f"| r_consistency | {format_number(sft_rc, 4)} | {format_number(rl_rc, 4)} | {'+' if rl_rc > sft_rc else ''}{format_number(rl_rc - sft_rc, 4)} |")

        report.append("")

        report.append("### 3.2 提升分析")
        report.append("")

        if rl_sr > sft_sr * 1.2:
            report.append(f"✅ **RL训练显著提升性能**，成功率提高 {format_number(sr_improvement, 1)}%")
        elif rl_sr > sft_sr:
            report.append(f"⚠️  **RL训练有轻微提升**，成功率提高 {format_number(sr_improvement, 1)}%，可能需要更长训练时间")
        else:
            report.append(f"❌ **RL训练未能提升性能**，可能是奖励设计或超参数问题")

        report.append("")

        report.append("---")
        report.append("")

    # ==================== 数据来源 ====================
    report.append("## 4. 数据来源")
    report.append("")

    if sft_results:
        if 'computed_from' in sft_results:
            report.append(f"- **SFT评测**: 从 `{sft_results.get('computed_from')}` 计算 ({sft_results.get('total_episodes', 'N/A')} episodes)")
        else:
            report.append(f"- **SFT评测**: results.json")

    if rl_results:
        if 'computed_from' in rl_results:
            report.append(f"- **RL评测**: 从 `{rl_results.get('computed_from')}` 计算 ({rl_results.get('total_episodes', 'N/A')} episodes)")
        else:
            report.append(f"- **RL评测**: results.json")

    report.append("")
    report.append("---")
    report.append("")

    # ==================== 结论与建议 ====================
    report.append("## 5. 结论与建议")
    report.append("")

    if rl_results and 'basic_metrics' in rl_results:
        rl_sr = rl_results['basic_metrics']['success_rate']
        rl_bp = rl_results.get('rebel_metrics', {}).get('avg_belief_parse_rate', 0)

        if rl_sr >= 0.3 and rl_bp >= 0.8:
            report.append("### 🎉 实验成功！")
            report.append("")
            report.append("ReBel方法达到预期效果：")
            report.append(f"- ✅ 成功率 {format_percentage(rl_sr)} ≥ 30%")
            report.append(f"- ✅ Belief解析率 {format_percentage(rl_bp)} ≥ 80%")
            report.append("")
            report.append("**下一步建议**：")
            report.append("1. 尝试更大的模型（7B）")
            report.append("2. 在其他环境（SciWorld）上验证")
            report.append("3. 进行消融实验，分析各组件贡献")
        elif rl_sr >= 0.2:
            report.append("### ✅ 实验部分成功")
            report.append("")
            report.append(f"成功率 {format_percentage(rl_sr)}，接近目标但仍有提升空间")
            report.append("")
            report.append("**建议**：")
            report.append("1. 增加RL训练轮数（100 → 200 epochs）")
            report.append("2. 调整reward权重，增强belief reward的影响")
            report.append("3. 增加cold-start数据量（390 → 500+）")
        else:
            report.append("### ⚠️  实验需要改进")
            report.append("")
            report.append(f"成功率 {format_percentage(rl_sr)} < 20%，未达到预期")
            report.append("")
            report.append("**排查方向**：")
            report.append("1. 检查cold-start数据质量（是否含Available Actions？）")
            report.append("2. 验证SFT训练是否充分（loss是否收敛？）")
            report.append("3. 检查RL训练配置（reward权重、学习率）")
            report.append("4. 查看训练曲线，是否存在过拟合或欠拟合")
    else:
        report.append("⚠️  无法生成结论，缺少评测数据")

    report.append("")
    report.append("---")
    report.append("")
    report.append(f"*报告生成于 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")

    # 写入文件
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))

    print(f"\n✅ 实验报告已生成: {output_path}")
    print(f"   总行数: {len(report)}")


def main():
    args = parse_args()

    print("=" * 60)
    print("ReBel实验报告生成器")
    print("=" * 60)

    # 自动查找或使用指定目录
    if args.auto_find:
        print("\n📂 自动查找最近的评测结果...")
        dirs = find_latest_eval_dirs()

        if len(dirs) >= 2:
            args.sft_eval_dir = dirs[1]  # 第二新的（假设是SFT评测）
            args.rl_eval_dir = dirs[0]   # 最新的（假设是RL评测）
            print(f"   找到 {len(dirs)} 个评测目录")
            print(f"   SFT评测: {args.sft_eval_dir}")
            print(f"   RL评测:  {args.rl_eval_dir}")
        elif len(dirs) == 1:
            print(f"   ⚠️  仅找到1个评测目录: {dirs[0]}")
            print(f"   将作为RL评测结果")
            args.rl_eval_dir = dirs[0]
        else:
            print("   ❌ 未找到任何评测结果目录")
            return

    print("\n📊 加载评测结果...")
    sft_results = load_eval_results(args.sft_eval_dir)
    rl_results = load_eval_results(args.rl_eval_dir)

    if sft_results:
        print(f"   ✅ SFT评测结果已加载")
    else:
        print(f"   ⚠️  未加载SFT评测结果")

    if rl_results:
        print(f"   ✅ RL评测结果已加载")
    else:
        print(f"   ⚠️  未加载RL评测结果")

    if not sft_results and not rl_results:
        print("\n❌ 没有任何评测结果，无法生成报告")
        return

    print("\n📝 生成实验报告...")
    generate_report(sft_results, rl_results, args.output)

    print(f"\n查看报告: cat {args.output}")
    print("=" * 60)


if __name__ == '__main__':
    main()
