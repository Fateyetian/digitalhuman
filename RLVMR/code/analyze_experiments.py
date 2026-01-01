#!/usr/bin/env python3
"""
ReBel 超参数实验分析脚本
生成论文格式的图表和报告
"""

import os
import re
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 无界面环境
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['figure.dpi'] = 150

# 实验配置
EXPERIMENTS = {
    'baseline': {
        'path': '/fs-computility-new/UPDZ03_chengjun/huangsijie.p/rebel_results/hyperparam_search/rebel_search_20251227_114948/baseline',
        'label': 'Baseline (w=1.0)',
        'color': '#1f77b4',
        'step_adv_w': 1.0,
    },
    'step_adv_0.5': {
        'path': '/fs-computility-new/UPDZ03_chengjun/huangsijie.p/rebel_results/hyperparam_search/rebel_search_20251228_013943/step_adv_0.5',
        'label': 'step_adv_w=0.5',
        'color': '#2ca02c',
        'step_adv_w': 0.5,
    },
    'step_adv_2.0': {
        'path': '/fs-computility-new/UPDZ03_chengjun/huangsijie.p/rebel_results/hyperparam_search/rebel_search_20251229_084222/step_adv_2.0',
        'label': 'step_adv_w=2.0',
        'color': '#ff7f0e',
        'step_adv_w': 2.0,
    },
    'step_adv_0.5_val128': {
        'path': '/fs-computility-new/UPDZ03_chengjun/huangsijie.p/rebel_results/hyperparam_search/rebel_search_20251230_041227/step_adv_0.5_val128_ep100',
        'label': 'step_adv_w=0.5 (128 val, 100 epochs)',
        'color': '#d62728',
        'step_adv_w': 0.5,
    },
}

# 任务类型映射
TASK_TYPES = {
    'pick_and_place': 'Pick & Place',
    'pick_heat_then_place_in_recep': 'Pick Heat Place',
    'pick_clean_then_place_in_recep': 'Pick Clean Place',
    'pick_two_obj_and_place': 'Pick Two Objects',
    'pick_cool_then_place_in_recep': 'Pick Cool Place',
    'look_at_obj_in_light': 'Look at Object',
}

OUTPUT_DIR = '/root/testttt/RLVMR/code/experiment_charts'


def parse_training_log(log_path):
    """解析训练日志，提取各类指标"""
    metrics = {
        'steps': [],
        'train_success_rate': [],
        'val_success_rate': [],
        'train_reward': [],
        'kl_divergence': [],
        'valid_action_ratio': [],
        'episode_length': [],
        # 任务类型成功率
        'train_task_rates': {k: [] for k in TASK_TYPES.keys()},
        'val_task_rates': {k: [] for k in TASK_TYPES.keys()},
    }

    if not os.path.exists(log_path):
        print(f"Warning: Log file not found: {log_path}")
        return metrics

    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    # 移除ANSI颜色代码
    content = re.sub(r'\x1b\[[0-9;]*m', '', content)

    # 解析每个step的指标
    # 格式: step:N - metric1:value1 - metric2:value2 ...
    step_pattern = r'step:(\d+)\s+-\s+(.+?)(?=step:\d+|$)'

    for match in re.finditer(step_pattern, content, re.DOTALL):
        step = int(match.group(1))
        metrics_str = match.group(2)

        # 解析各个指标
        metric_pattern = r'(\w+(?:/\w+)*):([+-]?\d*\.?\d+)'
        step_metrics = {}
        for m in re.finditer(metric_pattern, metrics_str):
            key, value = m.group(1), float(m.group(2))
            step_metrics[key] = value

        if not step_metrics:
            continue

        metrics['steps'].append(step)

        # 训练成功率和奖励
        metrics['train_success_rate'].append(step_metrics.get('episode/success_rate', 0) * 100)
        metrics['train_reward'].append(step_metrics.get('episode/reward/mean', 0))
        metrics['kl_divergence'].append(step_metrics.get('actor/ppo_kl', 0))
        metrics['valid_action_ratio'].append(step_metrics.get('valid_action_ratio', 0) * 100)
        metrics['episode_length'].append(step_metrics.get('episode/length/mean', 30))

        # 验证成功率 (只在某些step有)
        val_rate = step_metrics.get('val/success_rate', None)
        if val_rate is not None:
            metrics['val_success_rate'].append((step, val_rate * 100))

        # 训练任务类型成功率
        for task_key in TASK_TYPES.keys():
            rate = step_metrics.get(f'episode/{task_key}_success_rate', 0)
            metrics['train_task_rates'][task_key].append(rate * 100)

        # 验证任务类型成功率
        for task_key in TASK_TYPES.keys():
            rate = step_metrics.get(f'val/{task_key}_success_rate', None)
            if rate is not None:
                if not metrics['val_task_rates'][task_key] or metrics['val_task_rates'][task_key][-1][0] != step:
                    metrics['val_task_rates'][task_key].append((step, rate * 100))

    return metrics


def load_all_experiments():
    """加载所有实验数据"""
    all_data = {}
    for exp_name, exp_config in EXPERIMENTS.items():
        log_path = os.path.join(exp_config['path'], 'training.log')
        print(f"Loading {exp_name}...")
        data = parse_training_log(log_path)
        data['config'] = exp_config
        all_data[exp_name] = data
        print(f"  - Steps: {len(data['steps'])}, Final train SR: {data['train_success_rate'][-1] if data['train_success_rate'] else 'N/A'}%")
    return all_data


def plot_success_rate_curves(all_data, output_dir):
    """绘制训练/验证成功率曲线"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 训练成功率
    ax1 = axes[0]
    for exp_name, data in all_data.items():
        if data['steps']:
            ax1.plot(data['steps'], data['train_success_rate'],
                    label=data['config']['label'],
                    color=data['config']['color'],
                    linewidth=2, alpha=0.8)
    ax1.set_xlabel('Training Step (Epoch)')
    ax1.set_ylabel('Success Rate (%)')
    ax1.set_title('(a) Training Success Rate')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 100)

    # 验证成功率
    ax2 = axes[1]
    for exp_name, data in all_data.items():
        if data['val_success_rate']:
            steps, rates = zip(*data['val_success_rate'])
            ax2.plot(steps, rates,
                    label=data['config']['label'],
                    color=data['config']['color'],
                    linewidth=2, marker='o', markersize=4, alpha=0.8)
    ax2.set_xlabel('Training Step (Epoch)')
    ax2.set_ylabel('Success Rate (%)')
    ax2.set_title('(b) Validation Success Rate')
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 100)

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'success_rate_curves.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")
    return save_path


def plot_task_type_comparison(all_data, output_dir):
    """绘制各任务类型成功率柱状图"""
    # 获取最后一个验证epoch的任务成功率
    task_names = list(TASK_TYPES.values())
    task_keys = list(TASK_TYPES.keys())

    # 准备数据
    exp_names = list(all_data.keys())
    n_tasks = len(task_keys)
    n_exps = len(exp_names)

    # 获取每个实验最后一次验证的各任务成功率
    task_rates = {exp: [] for exp in exp_names}
    for exp_name, data in all_data.items():
        for task_key in task_keys:
            val_rates = data['val_task_rates'].get(task_key, [])
            if val_rates:
                # 取最后一次验证的值
                task_rates[exp_name].append(val_rates[-1][1])
            else:
                # 如果没有验证数据，使用最后的训练数据
                train_rates = data['train_task_rates'].get(task_key, [])
                if train_rates:
                    task_rates[exp_name].append(train_rates[-1])
                else:
                    task_rates[exp_name].append(0)

    # 绘制分组柱状图
    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(n_tasks)
    width = 0.18

    for i, exp_name in enumerate(exp_names):
        offset = (i - (n_exps - 1) / 2) * width
        bars = ax.bar(x + offset, task_rates[exp_name], width,
                     label=all_data[exp_name]['config']['label'],
                     color=all_data[exp_name]['config']['color'],
                     alpha=0.85)

        # 在柱子上方添加数值
        for bar, rate in zip(bars, task_rates[exp_name]):
            if rate > 5:  # 只显示大于5%的值
                ax.annotate(f'{rate:.0f}',
                           xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                           xytext=(0, 2),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('Task Type')
    ax.set_ylabel('Success Rate (%)')
    ax.set_title('Task-Specific Success Rates (Final Validation)')
    ax.set_xticks(x)
    ax.set_xticklabels(task_names, rotation=15, ha='right')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 110)

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'task_type_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")
    return save_path


def plot_reward_curves(all_data, output_dir):
    """绘制奖励曲线"""
    fig, ax = plt.subplots(figsize=(10, 5))

    for exp_name, data in all_data.items():
        if data['steps'] and data['train_reward']:
            ax.plot(data['steps'], data['train_reward'],
                   label=data['config']['label'],
                   color=data['config']['color'],
                   linewidth=2, alpha=0.8)

    ax.set_xlabel('Training Step (Epoch)')
    ax.set_ylabel('Average Episode Reward')
    ax.set_title('Training Reward Curves')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'reward_curves.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")
    return save_path


def plot_kl_divergence(all_data, output_dir):
    """绘制KL散度曲线"""
    fig, ax = plt.subplots(figsize=(10, 5))

    for exp_name, data in all_data.items():
        if data['steps'] and data['kl_divergence']:
            ax.plot(data['steps'], data['kl_divergence'],
                   label=data['config']['label'],
                   color=data['config']['color'],
                   linewidth=2, alpha=0.8)

    ax.set_xlabel('Training Step (Epoch)')
    ax.set_ylabel('KL Divergence')
    ax.set_title('Policy KL Divergence During Training')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'kl_divergence.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")
    return save_path


def plot_valid_action_ratio(all_data, output_dir):
    """绘制有效动作比例曲线"""
    fig, ax = plt.subplots(figsize=(10, 5))

    for exp_name, data in all_data.items():
        if data['steps'] and data['valid_action_ratio']:
            ax.plot(data['steps'], data['valid_action_ratio'],
                   label=data['config']['label'],
                   color=data['config']['color'],
                   linewidth=2, alpha=0.8)

    ax.set_xlabel('Training Step (Epoch)')
    ax.set_ylabel('Valid Action Ratio (%)')
    ax.set_title('Valid Action Ratio During Training')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(50, 100)

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'valid_action_ratio.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")
    return save_path


def plot_episode_length(all_data, output_dir):
    """绘制episode长度曲线"""
    fig, ax = plt.subplots(figsize=(10, 5))

    for exp_name, data in all_data.items():
        if data['steps'] and data['episode_length']:
            ax.plot(data['steps'], data['episode_length'],
                   label=data['config']['label'],
                   color=data['config']['color'],
                   linewidth=2, alpha=0.8)

    ax.set_xlabel('Training Step (Epoch)')
    ax.set_ylabel('Average Episode Length (Steps)')
    ax.set_title('Average Episode Length During Training')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 35)

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'episode_length.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")
    return save_path


def plot_final_comparison(all_data, output_dir):
    """绘制最终结果对比图"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    exp_names = list(all_data.keys())
    labels = [all_data[e]['config']['label'].split(' (')[0] for e in exp_names]
    colors = [all_data[e]['config']['color'] for e in exp_names]

    # 最终验证成功率
    ax1 = axes[0]
    final_val_rates = []
    for exp_name in exp_names:
        val_rates = all_data[exp_name]['val_success_rate']
        if val_rates:
            final_val_rates.append(val_rates[-1][1])
        else:
            final_val_rates.append(0)

    bars1 = ax1.bar(range(len(exp_names)), final_val_rates, color=colors, alpha=0.85)
    ax1.set_xlabel('Experiment')
    ax1.set_ylabel('Success Rate (%)')
    ax1.set_title('(a) Final Validation Success Rate')
    ax1.set_xticks(range(len(exp_names)))
    ax1.set_xticklabels(labels, rotation=20, ha='right')
    ax1.set_ylim(0, 80)
    for bar, rate in zip(bars1, final_val_rates):
        ax1.annotate(f'{rate:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

    # 最终训练成功率
    ax2 = axes[1]
    final_train_rates = [all_data[e]['train_success_rate'][-1] if all_data[e]['train_success_rate'] else 0
                         for e in exp_names]
    bars2 = ax2.bar(range(len(exp_names)), final_train_rates, color=colors, alpha=0.85)
    ax2.set_xlabel('Experiment')
    ax2.set_ylabel('Success Rate (%)')
    ax2.set_title('(b) Final Training Success Rate')
    ax2.set_xticks(range(len(exp_names)))
    ax2.set_xticklabels(labels, rotation=20, ha='right')
    ax2.set_ylim(0, 100)
    for bar, rate in zip(bars2, final_train_rates):
        ax2.annotate(f'{rate:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

    # 最终平均奖励
    ax3 = axes[2]
    final_rewards = [all_data[e]['train_reward'][-1] if all_data[e]['train_reward'] else 0
                     for e in exp_names]
    bars3 = ax3.bar(range(len(exp_names)), final_rewards, color=colors, alpha=0.85)
    ax3.set_xlabel('Experiment')
    ax3.set_ylabel('Average Reward')
    ax3.set_title('(c) Final Average Reward')
    ax3.set_xticks(range(len(exp_names)))
    ax3.set_xticklabels(labels, rotation=20, ha='right')
    for bar, reward in zip(bars3, final_rewards):
        ax3.annotate(f'{reward:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'final_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")
    return save_path


def generate_summary_stats(all_data):
    """生成统计摘要"""
    stats = {}
    for exp_name, data in all_data.items():
        exp_stats = {
            'total_steps': len(data['steps']),
            'final_train_sr': data['train_success_rate'][-1] if data['train_success_rate'] else 0,
            'final_val_sr': data['val_success_rate'][-1][1] if data['val_success_rate'] else 0,
            'final_reward': data['train_reward'][-1] if data['train_reward'] else 0,
            'final_valid_ratio': data['valid_action_ratio'][-1] if data['valid_action_ratio'] else 0,
            'final_episode_length': data['episode_length'][-1] if data['episode_length'] else 30,
            'final_kl': data['kl_divergence'][-1] if data['kl_divergence'] else 0,
        }

        # 最大验证成功率
        if data['val_success_rate']:
            best_val = max(data['val_success_rate'], key=lambda x: x[1])
            exp_stats['best_val_sr'] = best_val[1]
            exp_stats['best_val_step'] = best_val[0]

        stats[exp_name] = exp_stats

    return stats


def main():
    """主函数"""
    print("=" * 60)
    print("ReBel Hyperparameter Search - Experiment Analysis")
    print("=" * 60)

    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 加载数据
    print("\n[1/7] Loading experiment data...")
    all_data = load_all_experiments()

    # 生成统计摘要
    print("\n[2/7] Computing statistics...")
    stats = generate_summary_stats(all_data)

    # 打印统计表格
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)
    print(f"{'Experiment':<25} {'Steps':<8} {'Train SR':<12} {'Val SR':<12} {'Reward':<10} {'Valid%':<10}")
    print("-" * 80)
    for exp_name, s in stats.items():
        print(f"{exp_name:<25} {s['total_steps']:<8} {s['final_train_sr']:.1f}%{'':<6} {s['final_val_sr']:.1f}%{'':<6} {s['final_reward']:.2f}{'':<5} {s['final_valid_ratio']:.1f}%")
    print("=" * 80)

    # 保存统计数据
    stats_path = os.path.join(OUTPUT_DIR, 'experiment_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\nStats saved to: {stats_path}")

    # 生成图表
    print("\n[3/7] Generating success rate curves...")
    plot_success_rate_curves(all_data, OUTPUT_DIR)

    print("[4/7] Generating task type comparison...")
    plot_task_type_comparison(all_data, OUTPUT_DIR)

    print("[5/7] Generating reward curves...")
    plot_reward_curves(all_data, OUTPUT_DIR)

    print("[6/7] Generating KL divergence plot...")
    plot_kl_divergence(all_data, OUTPUT_DIR)

    print("[7/7] Generating additional plots...")
    plot_valid_action_ratio(all_data, OUTPUT_DIR)
    plot_episode_length(all_data, OUTPUT_DIR)
    plot_final_comparison(all_data, OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print(f"Charts saved to: {OUTPUT_DIR}")
    print("=" * 60)

    # 返回生成的文件列表
    return [
        os.path.join(OUTPUT_DIR, f)
        for f in os.listdir(OUTPUT_DIR)
        if f.endswith('.png')
    ]


if __name__ == '__main__':
    main()
