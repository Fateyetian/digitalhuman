"""
ReBel (Reward Belief) Framework - Core Algorithm

严格按照 ReBel_RL_Algorithm_Guide.md 实现

核心创新:
1. Belief-based Grouping: 按(uid, belief_hash)分组，粒度适中(20-100组)
2. 4组件内在奖励: Consistency + Progress + Exploration (Format是独立惩罚)
3. 双层优势: A_total = A_episode + λ × A_step

公式:
- A_episode: 按uid分组归一化 (与GiGPO相同)
- A_step: 按(uid, belief_hash)分组归一化 (ReBel核心)
"""

import numpy as np
import torch
import json
import hashlib
import re
from collections import defaultdict
from typing import Dict, Any, List, Tuple, Optional


# ============================================================================ #
# ===================== Belief Canonicalization ============================== #
# ============================================================================ #

def canonicalize_belief(belief_state: Dict[str, Any], granularity: str = 'subgoal') -> str:
    """
    将belief state转换为可哈希的规范表示

    Args:
        belief_state: 解析后的belief state字典
        granularity: 粒度级别
            - 'subgoal': 只用subgoal + status (推荐，20-50组)
            - 'medium': + found_objects统计 (50-150组)
            - 'fine': 完整belief state (100-500组)

    Returns:
        belief_hash: 16位MD5哈希字符串
    """
    if belief_state is None or not isinstance(belief_state, dict):
        return "null_belief_0000"

    try:
        if granularity == 'subgoal':
            # 最稳定: 只用subgoal + status
            task_progress = belief_state.get('task_progress_update', {}) or {}
            canonical = {
                'subgoal': str(task_progress.get('updated_subgoal', '')).lower().strip(),
                'status': str(task_progress.get('subgoal_status', '')).lower().strip()
            }

        elif granularity == 'medium':
            # 中等粒度: + found_objects统计
            task_progress = belief_state.get('task_progress_update', {}) or {}
            world_model = belief_state.get('world_model_update', {}) or {}
            found_objects = world_model.get('found_objects', {})
            if not isinstance(found_objects, dict):
                found_objects = {}

            # 提取物体类型
            object_types = []
            for obj in found_objects.keys():
                if obj and isinstance(obj, str):
                    obj_type = obj.split()[0] if obj else ''
                    object_types.append(obj_type)

            canonical = {
                'subgoal': str(task_progress.get('updated_subgoal', '')).lower().strip(),
                'status': str(task_progress.get('subgoal_status', '')).lower().strip(),
                'num_found_objects': len(found_objects),
                'found_object_types': sorted(object_types)
            }

        elif granularity == 'fine':
            # 细粒度: 完整state
            canonical = belief_state

        else:
            # 默认使用subgoal
            task_progress = belief_state.get('task_progress_update', {}) or {}
            canonical = {
                'subgoal': str(task_progress.get('updated_subgoal', '')).lower().strip(),
                'status': str(task_progress.get('subgoal_status', '')).lower().strip()
            }

        # 生成哈希
        canonical_str = json.dumps(canonical, sort_keys=True, ensure_ascii=False)
        belief_hash = hashlib.md5(canonical_str.encode()).hexdigest()[:16]
        return belief_hash

    except Exception as e:
        return "error_belief_000"


# ============================================================================ #
# ====================== Belief Group Building =============================== #
# ============================================================================ #

def build_belief_group(
    belief_states: np.ndarray,
    index: np.ndarray,
    granularity: str = 'subgoal',
    summarize: bool = False
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    按belief相似性将steps分组

    分组策略: G = {j : belief_hash_j = belief_hash_i, uid_j = uid_i}

    Args:
        belief_states: shape (batch_size,), 每个元素是belief字典
        index: shape (batch_size,), 每个step的prompt uid
        granularity: canonicalization粒度
        summarize: 是否打印分组统计

    Returns:
        belief_group_uids: shape (batch_size,), 每个step的group uid
        group_stats: 分组统计信息
    """
    belief_group_uids = np.empty(len(belief_states), dtype=object)
    unique_indices = np.unique(index)
    group_sizes = []
    all_group_uids = set()

    for uid in unique_indices:
        # 1. 获取该uid的所有steps
        step_indices = np.where(index == uid)[0]
        beliefs = belief_states[step_indices]

        # 2. 按belief hash聚类
        clusters = defaultdict(list)
        for i, belief in enumerate(beliefs):
            belief_hash = canonicalize_belief(belief, granularity)
            clusters[belief_hash].append(step_indices[i])

        # 3. 分配group uid (格式: belief_{uid}_{hash})
        for belief_hash, original_indices in clusters.items():
            group_uid = f"belief_{uid}_{belief_hash}"
            all_group_uids.add(group_uid)
            group_sizes.append(len(original_indices))

            for idx in original_indices:
                belief_group_uids[idx] = group_uid

    # 计算统计
    group_stats = {
        'num_groups': len(all_group_uids),
        'group_sizes': group_sizes,
        'mean_group_size': np.mean(group_sizes) if group_sizes else 0,
        'median_group_size': np.median(group_sizes) if group_sizes else 0,
        'min_group_size': np.min(group_sizes) if group_sizes else 0,
        'max_group_size': np.max(group_sizes) if group_sizes else 0,
        'single_sample_groups': sum(1 for s in group_sizes if s == 1),
    }

    if summarize:
        print(f"""
ReBel Belief-Based Grouping:
├─ Num Groups: {group_stats['num_groups']}
├─ Mean Group Size: {group_stats['mean_group_size']:.2f}
├─ Median Group Size: {group_stats['median_group_size']:.1f}
├─ Min/Max Group Size: {group_stats['min_group_size']}/{group_stats['max_group_size']}
└─ Single-Sample Groups: {group_stats['single_sample_groups']}
""")

    return belief_group_uids, group_stats


# ============================================================================ #
# ====================== Episode Norm Reward ================================= #
# ============================================================================ #

def episode_norm_reward(
    token_level_rewards: torch.Tensor,
    eos_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    remove_std: bool = True
) -> torch.Tensor:
    """
    Episode级别优势 (与GiGPO相同)

    按uid分组，计算组内归一化:
    A_episode[i] = R_episode[i] - mean(R_episode[uid])

    Args:
        token_level_rewards: (batch, seq_len)
        eos_mask: (batch, seq_len)
        index: (batch,) prompt uids
        epsilon: 数值稳定性
        remove_std: True=mean_norm, False=mean_std_norm

    Returns:
        episode_advantages: (batch, seq_len)
    """
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)  # Episode reward

    id2list = defaultdict(list)
    id2stats = {}

    with torch.no_grad():
        # 收集每个uid的rewards
        for i in range(scores.shape[0]):
            id2list[index[i]].append(scores[i])

        # 计算每个uid的mean/std
        for uid, rewards in id2list.items():
            t = torch.stack(rewards) if len(rewards) > 1 else torch.tensor(rewards)
            if len(rewards) == 1:
                mean, std = torch.tensor(0.0), torch.tensor(1.0)
            else:
                mean, std = torch.mean(t), torch.std(t)
            id2stats[uid] = (mean, std)

        # 归一化
        for i in range(scores.shape[0]):
            mean, std = id2stats[index[i]]
            if remove_std:
                scores[i] = scores[i] - mean
            else:
                scores[i] = (scores[i] - mean) / (std + epsilon)

        # 广播到所有token
        episode_advantages = scores.unsqueeze(-1).expand(-1, response_length) * eos_mask

    return episode_advantages


# ============================================================================ #
# ====================== Step Norm Reward by Belief ========================== #
# ============================================================================ #

def step_norm_reward_by_belief(
    step_rewards: torch.Tensor,
    eos_mask: torch.Tensor,
    belief_group_uids: np.ndarray,
    epsilon: float = 1e-6,
    remove_std: bool = True
) -> torch.Tensor:
    """
    Step级别优势 - 在belief group内归一化

    按(uid, belief_hash)分组，计算组内归一化:
    A_step[i] = (R_intrinsic[i] - mean(R[group])) / (std + ε)

    Args:
        step_rewards: (batch,) intrinsic rewards
        eos_mask: (batch, seq_len)
        belief_group_uids: (batch,) group ids
        epsilon: 数值稳定性
        remove_std: True=mean_norm, False=mean_std_norm

    Returns:
        step_advantages: (batch, seq_len)
    """
    response_length = eos_mask.shape[-1]
    scores = step_rewards.clone()

    group2list = defaultdict(list)
    group2stats = {}

    with torch.no_grad():
        # 1. 收集每个group的rewards
        for i in range(len(scores)):
            group_uid = belief_group_uids[i]
            group2list[group_uid].append((i, scores[i]))

        # 2. 计算每个group的mean/std
        for group_uid, items in group2list.items():
            rewards = [item[1] for item in items]
            t = torch.stack(rewards) if len(rewards) > 1 else torch.tensor(rewards)

            if len(rewards) == 1:
                # 单样本组: 不减均值，保持原值
                mean, std = torch.tensor(0.0), torch.tensor(1.0)
            else:
                mean, std = torch.mean(t), torch.std(t)

            group2stats[group_uid] = (mean, std)

        # 3. 归一化
        for i in range(len(scores)):
            group_uid = belief_group_uids[i]
            mean, std = group2stats[group_uid]
            if remove_std:
                scores[i] = scores[i] - mean
            else:
                scores[i] = (scores[i] - mean) / (std + epsilon)

        # 4. 广播到所有token
        step_advantages = scores.unsqueeze(-1).expand(-1, response_length) * eos_mask

    return step_advantages


# ============================================================================ #
# ====================== Main: compute_rebel_advantage ======================= #
# ============================================================================ #

def compute_rebel_advantage(
    token_level_rewards: torch.Tensor,
    rebel_intrinsic_rewards: torch.Tensor,
    eos_mask: torch.Tensor,
    belief_states: np.ndarray,
    index: np.ndarray,
    epsilon: float = 1e-6,
    step_advantage_w: float = 1.0,
    mode: str = "mean_norm",
    belief_granularity: str = 'subgoal',
    summarize: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """
    ReBel优势计算主函数

    公式: A_total = A_episode + λ × A_step

    其中:
    - A_episode: 按uid分组归一化
    - A_step: 按(uid, belief_hash)分组归一化
    - λ: step_advantage_w

    Args:
        token_level_rewards: (batch, seq_len) episode rewards
        rebel_intrinsic_rewards: (batch,) intrinsic rewards
        eos_mask: (batch, seq_len)
        belief_states: (batch,) belief dicts
        index: (batch,) prompt uids
        epsilon: 数值稳定性
        step_advantage_w: λ权重
        mode: 'mean_norm' or 'mean_std_norm'
        belief_granularity: 'subgoal', 'medium', 'fine'
        summarize: 是否打印统计

    Returns:
        advantages: (batch, seq_len) 总优势
        returns: (batch, seq_len) 同上
        adv_details: 详细信息
    """
    remove_std = (mode == "mean_norm")

    # 1. Episode优势 (与GiGPO相同)
    episode_advantages = episode_norm_reward(
        token_level_rewards, eos_mask, index, epsilon, remove_std
    )

    # 2. 构建belief groups (ReBel核心)
    belief_group_uids, group_stats = build_belief_group(
        belief_states, index, belief_granularity, summarize
    )

    # 3. Step优势 (按belief group归一化)
    step_advantages = step_norm_reward_by_belief(
        rebel_intrinsic_rewards, eos_mask, belief_group_uids, epsilon, remove_std
    )

    # 4. 组合
    total_advantages = episode_advantages + step_advantage_w * step_advantages

    # 5. 统计信息
    adv_details = {
        'episode_advantages': episode_advantages,
        'step_advantages': step_advantages,
        'belief_group_stats': group_stats,
        'episode_adv_mean': episode_advantages.mean().item(),
        'episode_adv_std': episode_advantages.std().item(),
        'step_adv_mean': step_advantages.mean().item(),
        'step_adv_std': step_advantages.std().item(),
        'total_adv_mean': total_advantages.mean().item(),
        'total_adv_std': total_advantages.std().item(),
    }

    if summarize:
        print(f"""
ReBel Advantage Statistics:
├─ Episode Adv Mean: {adv_details['episode_adv_mean']:.4f}
├─ Episode Adv Std: {adv_details['episode_adv_std']:.4f}
├─ Step Adv Mean: {adv_details['step_adv_mean']:.4f}
├─ Step Adv Std: {adv_details['step_adv_std']:.4f}
├─ Total Adv Mean: {adv_details['total_adv_mean']:.4f}
└─ Total Adv Std: {adv_details['total_adv_std']:.4f}
""")

    return total_advantages, total_advantages, adv_details


# ============================================================================ #
# ====================== Intrinsic Reward Components ========================= #
# ============================================================================ #

def consistency_reward(
    belief: Dict[str, Any],
    ground_truth: Optional[Dict[str, Any]] = None
) -> float:
    """
    评估belief与真实环境的一致性

    Args:
        belief: 模型输出的belief
        ground_truth: 真实环境状态

    Returns:
        reward: [-0.5, 1.0]
    """
    if belief is None:
        return 0.0

    reward = 0.0
    world_model = belief.get('world_model_update', {}) or {}

    if ground_truth is None:
        # 无ground truth时，给予结构化信念的部分奖励
        found_objects = world_model.get('found_objects', {})
        if isinstance(found_objects, dict) and found_objects:
            reward += 0.1 * min(len(found_objects), 5)  # Max 0.5
        return np.clip(reward, 0, 0.5)

    # 有ground truth: 验证正确性
    gt_objects = ground_truth.get('object_locations', {}) or {}
    found_objects = world_model.get('found_objects', {})

    if isinstance(found_objects, dict):
        for obj_id, believed_loc in found_objects.items():
            if not obj_id or not believed_loc:
                continue
            # 检查是否真的在那里
            obj_lower = str(obj_id).lower()
            believed_lower = str(believed_loc).lower()

            for gt_obj, gt_loc in gt_objects.items():
                if obj_lower in gt_obj.lower() or gt_obj.lower() in obj_lower:
                    if believed_lower in gt_loc.lower() or gt_loc.lower() in believed_lower:
                        reward += 0.2  # 正确belief
                    else:
                        reward -= 0.1  # 错误belief
                    break

    return np.clip(reward, -0.5, 1.0)


def progress_reward(
    belief: Dict[str, Any],
    prev_belief: Optional[Dict[str, Any]] = None
) -> float:
    """
    评估任务进度

    Args:
        belief: 当前belief
        prev_belief: 上一步belief

    Returns:
        reward: [0, 1.0]
    """
    if belief is None:
        return 0.0

    reward = 0.0
    task_progress = belief.get('task_progress_update', {}) or {}
    prev_task_progress = (prev_belief.get('task_progress_update', {}) or {}) if prev_belief else {}

    # 1. 子目标完成
    curr_status = str(task_progress.get('subgoal_status', '')).lower()
    prev_status = str(prev_task_progress.get('subgoal_status', '')).lower()

    if 'complete' in curr_status and 'complete' not in prev_status:
        reward += 0.5  # 新完成子目标

    # 2. 有意义的证据
    evidence = str(task_progress.get('evidence', ''))
    if evidence and len(evidence) > 10:
        reward += 0.1

    # 3. 子目标更新
    curr_subgoal = str(task_progress.get('updated_subgoal', '')).lower().strip()
    prev_subgoal = str(prev_task_progress.get('updated_subgoal', '')).lower().strip()

    if curr_subgoal and curr_subgoal != prev_subgoal:
        reward += 0.1  # 子目标推进

    return np.clip(reward, 0, 1.0)


def exploration_reward(
    belief: Dict[str, Any],
    prev_belief: Optional[Dict[str, Any]] = None
) -> float:
    """
    评估探索效率

    Args:
        belief: 当前belief
        prev_belief: 上一步belief

    Returns:
        reward: [-0.1, 0.5]
    """
    if belief is None:
        return 0.0

    reward = 0.0
    exploration = belief.get('exploration_map_update', {}) or {}
    prev_exploration = (prev_belief.get('exploration_map_update', {}) or {}) if prev_belief else {}

    # 1. 新访问的位置
    newly_visited = exploration.get('newly_visited', [])
    if isinstance(newly_visited, str):
        newly_visited = [newly_visited]
    if not isinstance(newly_visited, list):
        newly_visited = []

    prev_visited = prev_exploration.get('newly_visited', [])
    if isinstance(prev_visited, str):
        prev_visited = [prev_visited]
    if not isinstance(prev_visited, list):
        prev_visited = []

    new_locations = set(newly_visited) - set(prev_visited)
    reward += 0.1 * min(len(new_locations), 3)  # Max 0.3

    # 2. 避免重复探索
    world_model = belief.get('world_model_update', {}) or {}
    cleared = world_model.get('cleared_receptacles', [])
    if not isinstance(cleared, list):
        cleared = []

    for loc in newly_visited:
        if loc in cleared:
            reward -= 0.02  # 重复访问惩罚

    return np.clip(reward, -0.1, 0.5)


def format_reward(output: str, is_action_available: bool = True) -> float:
    """
    评估输出格式 - RLVMR风格惩罚机制

    检查规则:
    1. 无中文字符
    2. belief/action标签各1个
    3. 标签内容非空
    4. 标签顺序: belief → action
    5. belief是有效JSON

    Returns:
        reward: 格式正确返回小奖励，格式错误返回-1.0
    """
    # 1. 检查中文字符
    if re.search(r'[\u4e00-\u9fff]', output):
        return -1.0

    # 2. 检查action标签数量
    action_matches = re.findall(r"<action>([\s\S]*?)</action>", output, re.IGNORECASE)
    if len(action_matches) != 1:
        return -1.0

    # 3. 检查belief标签数量
    belief_matches = re.findall(r"<belief>([\s\S]*?)</belief>", output, re.IGNORECASE)
    if len(belief_matches) != 1:
        return -1.0

    # 4. 检查belief内容非空
    belief_content = belief_matches[0].strip()
    if not belief_content:
        return -1.0

    # 5. 检查标签顺序
    belief_pos = output.lower().find("<belief>")
    action_pos = output.lower().find("<action>")
    if belief_pos > action_pos:
        return -1.0

    # 6. 检查belief是否为有效JSON
    try:
        belief_json = json.loads(belief_content)
        valid_keys = ["world_model_update", "task_progress_update", "exploration_map_update"]
        if not any(key in belief_json for key in valid_keys):
            return -0.5  # 缺少必需字段
    except json.JSONDecodeError:
        return -1.0

    # 格式验证通过
    reward = 0.1
    if not is_action_available:
        reward -= 0.2

    return reward


def compute_intrinsic_reward(
    belief: Dict[str, Any],
    prev_belief: Optional[Dict[str, Any]] = None,
    ground_truth: Optional[Dict[str, Any]] = None,
    output: str = "",
    is_format_valid: bool = True,
    is_action_available: bool = True,
    weights: Optional[Dict[str, float]] = None
) -> Tuple[float, Dict[str, float]]:
    """
    计算ReBel内在奖励

    公式: R_intrinsic = α*R_consistency + β*R_progress + γ*R_exploration

    注意: Format不是内在奖励的一部分，是独立的格式惩罚

    Args:
        belief: 当前belief
        prev_belief: 上一步belief
        ground_truth: 真实环境状态
        output: 模型输出
        is_format_valid: 格式是否有效
        is_action_available: action是否可用
        weights: 权重配置

    Returns:
        total_reward: 总内在奖励
        component_rewards: 各组件奖励
    """
    weights = weights or {
        'consistency': 0.3,
        'progress': 0.5,
        'exploration': 0.2,
    }

    # 先检查格式
    r_format = format_reward(output, is_action_available)

    component_rewards = {
        'format': r_format,
        'is_format_valid': r_format > -0.5,
    }

    # 格式无效时，跳过内在奖励计算
    if r_format <= -0.5:
        component_rewards['consistency'] = 0.0
        component_rewards['progress'] = 0.0
        component_rewards['exploration'] = 0.0
        component_rewards['intrinsic'] = 0.0
        return r_format, component_rewards  # 返回格式惩罚

    # 计算各组件
    r_consistency = consistency_reward(belief, ground_truth)
    r_progress = progress_reward(belief, prev_belief)
    r_exploration = exploration_reward(belief, prev_belief)

    component_rewards['consistency'] = r_consistency
    component_rewards['progress'] = r_progress
    component_rewards['exploration'] = r_exploration

    # 加权组合
    intrinsic = (
        weights['consistency'] * r_consistency +
        weights['progress'] * r_progress +
        weights['exploration'] * r_exploration
    )

    component_rewards['intrinsic'] = intrinsic

    # 总奖励 = 内在奖励 + 格式奖励
    total_reward = intrinsic + r_format

    return total_reward, component_rewards


# ============================================================================ #
# ====================== Utility Functions =================================== #
# ============================================================================ #

def print_rebel_summary(adv_details: Dict[str, Any]):
    """打印ReBel统计摘要"""
    group_stats = adv_details.get('belief_group_stats', {})

    print(f"""
{'='*50}
ReBel Statistics Summary
{'='*50}
Belief-Based Grouping:
├─ Num Groups: {group_stats.get('num_groups', 'N/A')}
├─ Mean Group Size: {group_stats.get('mean_group_size', 0):.2f}
├─ Median Group Size: {group_stats.get('median_group_size', 0):.1f}
├─ Min/Max Size: {group_stats.get('min_group_size', 0)}/{group_stats.get('max_group_size', 0)}
└─ Single-Sample Groups: {group_stats.get('single_sample_groups', 0)}

Advantages:
├─ Episode Adv: mean={adv_details.get('episode_adv_mean', 0):.4f}, std={adv_details.get('episode_adv_std', 0):.4f}
├─ Step Adv: mean={adv_details.get('step_adv_mean', 0):.4f}, std={adv_details.get('step_adv_std', 0):.4f}
└─ Total Adv: mean={adv_details.get('total_adv_mean', 0):.4f}, std={adv_details.get('total_adv_std', 0):.4f}
{'='*50}
""")


def health_check(adv_details: Dict[str, Any]) -> Dict[str, str]:
    """健康检查"""
    warnings = {}
    group_stats = adv_details.get('belief_group_stats', {})

    num_groups = group_stats.get('num_groups', 0)
    if num_groups < 20:
        warnings['num_groups'] = f"组数太少({num_groups})，考虑使用更细的粒度"
    elif num_groups > 100:
        warnings['num_groups'] = f"组数太多({num_groups})，考虑使用更粗的粒度"

    mean_size = group_stats.get('mean_group_size', 0)
    if mean_size < 10:
        warnings['group_size'] = f"平均组大小太小({mean_size:.1f})，归一化效果可能不佳"
    elif mean_size > 50:
        warnings['group_size'] = f"平均组大小太大({mean_size:.1f})，考虑使用更细的粒度"

    single_ratio = group_stats.get('single_sample_groups', 0) / max(num_groups, 1)
    if single_ratio > 0.5:
        warnings['single_samples'] = f"单样本组占比太高({single_ratio:.1%})"

    return warnings
