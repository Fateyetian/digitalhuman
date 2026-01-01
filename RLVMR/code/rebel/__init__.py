"""
ReBel (Reward Belief) Framework

严格按照 ReBel_RL_Algorithm_Guide.md 实现的RL算法

核心创新:
1. Belief-based Grouping: 按(uid, belief_hash)分组，粒度适中(20-100组)
2. 4组件内在奖励: Consistency + Progress + Exploration (Format是独立惩罚)
3. 双层优势: A_total = A_episode + λ × A_step
"""

from .core_rebel import (
    # Main advantage computation
    compute_rebel_advantage,
    episode_norm_reward,
    step_norm_reward_by_belief,

    # Belief grouping
    build_belief_group,
    canonicalize_belief,

    # Intrinsic reward components
    consistency_reward,
    progress_reward,
    exploration_reward,
    format_reward,
    compute_intrinsic_reward,

    # Utilities
    print_rebel_summary,
    health_check,
)

__all__ = [
    # Main API
    'compute_rebel_advantage',
    'episode_norm_reward',
    'step_norm_reward_by_belief',

    # Belief grouping
    'build_belief_group',
    'canonicalize_belief',

    # Intrinsic rewards
    'consistency_reward',
    'progress_reward',
    'exploration_reward',
    'format_reward',
    'compute_intrinsic_reward',

    # Utilities
    'print_rebel_summary',
    'health_check',
]
