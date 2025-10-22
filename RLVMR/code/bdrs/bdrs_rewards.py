from __future__ import annotations
from typing import Dict, Any
import numpy as np
import torch


class BDRSRewardCalculator:
    def __init__(self, world_w: float = 1.0, progress_w: float = 1.0, explore_w: float = 1.0):
        self.world_w = world_w
        self.progress_w = progress_w
        self.explore_w = explore_w

    def step_reward(self, belief: Dict[str, Any], info: Dict[str, Any]) -> Dict[str, float]:
        """
        基于 infos 中的 belief 快照计算一步的内在奖励分量：
        - world_consistency：最近一次一致性检查是否通过（不通过则负向，反之正向微量）
        - task_progress：子目标从 pending -> completed 的推进信号
        - exploration_efficiency：首次访问新房间/新对象
        返回 dict，并附带总和值 'total'
        """
        # 安全取值
        world = belief.get('world_model', {})
        task = belief.get('task_progress', {})
        explore = belief.get('exploration_map', {})

        # world consistency
        notes = belief.get('notes', []) or []
        has_mismatch = any('mismatch' in n or 'missing' in n for n in notes)
        world_consistency = 0.1 if not has_mismatch else -0.1

        # task progress
        subgoals = task.get('subgoals', []) or []
        completed = sum(1 for g in subgoals if g.get('status') == 'completed')
        cur_idx = task.get('current_subgoal_idx', 0)
        # 简化：更多 completed -> 更高奖励（相对上一时刻，真实应做差分，这里先使用当前态）
        task_progress = float(completed > 0) * 0.2

        # exploration efficiency
        visited_rooms = set(explore.get('visited_rooms', []) or [])
        visited_objects = set(explore.get('visited_objects', []) or [])
        # 新增访问的“幅度”无法仅从快照得知，这里以集合大小作为近似密度奖励
        exploration_efficiency = 0.05 * (len(visited_rooms) > 0) + 0.02 * (len(visited_objects) > 2)

        total = (
            self.world_w * world_consistency +
            self.progress_w * task_progress +
            self.explore_w * exploration_efficiency
        )

        return {
            'world_consistency': float(world_consistency),
            'task_progress': float(task_progress),
            'exploration_efficiency': float(exploration_efficiency),
            'total': float(total),
        }


