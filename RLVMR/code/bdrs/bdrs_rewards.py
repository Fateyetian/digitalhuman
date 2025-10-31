from __future__ import annotations
from typing import Dict, Any, Optional
import numpy as np
import torch


class BDRSRewardCalculator:
    def __init__(
        self,
        world_w: float = 1.0,
        progress_w: float = 1.0,
        explore_w: float = 1.0,
        # 细粒度系数
        reward_correct_belief: float = 0.2,
        reward_new_conflict: float = -0.1,
        reward_subgoal_complete: float = 0.5,
        reward_new_entity: float = 0.05,
        reward_new_location: float = 0.1,
        penalty_revisit: float = -0.02,
    ):
        self.world_w = world_w
        self.progress_w = progress_w
        self.explore_w = explore_w

        self.reward_correct_belief = reward_correct_belief
        self.reward_new_conflict = reward_new_conflict
        self.reward_subgoal_complete = reward_subgoal_complete
        self.reward_new_entity = reward_new_entity
        self.reward_new_location = reward_new_location
        self.penalty_revisit = penalty_revisit

    def step_reward(
        self,
        prev_belief: Optional[Dict[str, Any]],
        curr_belief: Dict[str, Any],
        info: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        基于 belief 状态的差分计算一步的内在奖励分量：
        - world_consistency: 信念修正数 - 新冲突数
        - task_progress: 新完成的子目标数
        - exploration_efficiency: 新访问房间数 + 新发现对象数 - 重复访问惩罚

        Args:
            prev_belief: 上一步的 belief 快照 (第一步时为None)
            curr_belief: 当前步的 belief 快照
            info: environment info (可选的额外信息)

        Returns:
            包含各分量和总和的字典
        """
        # === 1. 世界一致性奖励 (R_consistency) ===
        world_consistency = self._compute_world_consistency(prev_belief, curr_belief)

        # === 2. 任务进展奖励 (R_progress) ===
        task_progress = self._compute_task_progress(prev_belief, curr_belief)

        # === 3. 探索效率奖励 (R_explore) ===
        exploration_efficiency = self._compute_exploration_efficiency(prev_belief, curr_belief)

        # 加权求和
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

    def _compute_world_consistency(
        self,
        prev_belief: Optional[Dict[str, Any]],
        curr_belief: Dict[str, Any]
    ) -> float:
        """
        计算世界一致性奖励：
        R_consistency = reward_correct_belief * N_corrections - reward_new_conflict * N_conflicts
        """
        if prev_belief is None:
            # 第一步没有之前的belief，返回0
            return 0.0

        prev_world = prev_belief.get('world_model', {})
        curr_world = curr_belief.get('world_model', {})

        # 计算新产生的冲突数
        prev_conflicts = len(prev_world.get('belief_conflicts', []))
        curr_conflicts = len(curr_world.get('belief_conflicts', []))
        n_new_conflicts = max(0, curr_conflicts - prev_conflicts)

        # 计算新的修正数
        prev_corrections = len(prev_world.get('belief_corrections', []))
        curr_corrections = len(curr_world.get('belief_corrections', []))
        n_new_corrections = max(0, curr_corrections - prev_corrections)

        reward = (self.reward_correct_belief * n_new_corrections +
                  self.reward_new_conflict * n_new_conflicts)

        return reward

    def _compute_task_progress(
        self,
        prev_belief: Optional[Dict[str, Any]],
        curr_belief: Dict[str, Any]
    ) -> float:
        """
        计算任务进展奖励：
        R_progress = reward_subgoal_complete * ΔN_completed_subgoals
        """
        if prev_belief is None:
            # 第一步
            return 0.0

        prev_task = prev_belief.get('task_progress', {})
        curr_task = curr_belief.get('task_progress', {})

        # 计算新完成的子目标数
        prev_completed = prev_task.get('completed_count', 0)
        curr_completed = curr_task.get('completed_count', 0)
        n_new_completed = max(0, curr_completed - prev_completed)

        reward = self.reward_subgoal_complete * n_new_completed

        return reward

    def _compute_exploration_efficiency(
        self,
        prev_belief: Optional[Dict[str, Any]],
        curr_belief: Dict[str, Any]
    ) -> float:
        """
        计算探索效率奖励：
        R_explore = reward_new_location * ΔN_locations + reward_new_entity * ΔN_entities
                    - penalty_revisit * I_revisit
        """
        curr_explore = curr_belief.get('exploration_map', {})

        # 新发现的房间和对象（已在belief state中计算差分）
        new_rooms = len(curr_explore.get('new_rooms_this_step', set()))
        new_objects = len(curr_explore.get('new_objects_this_step', set()))

        # 检查是否重复访问（访问计数>1）
        room_counts = curr_explore.get('room_visit_counts', {})
        current_room = curr_belief.get('world_model', {}).get('current_room')
        is_revisit = (current_room and room_counts.get(current_room, 0) > 1)

        reward = (self.reward_new_location * new_rooms +
                  self.reward_new_entity * new_objects)

        if is_revisit:
            reward += self.penalty_revisit

        return reward


