"""
ReBel (Reinforcement Learning with Belief-State Enhancement)

Core innovations:
1. HiBO (Hierarchical Belief-Observation Grouping): obs hash primary + belief abstract fallback
2. Competence-Adaptive Belief Reward Curriculum: SR-based decay + differential component decay
3. Structured Belief Prompting: cognitive scaffolding + HiBO signal + reward basis
4. Dual-layer advantage: A_total = A_episode + λ × A_step
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

from .hibo_grouping import (
    # V11: HiBO grouping
    semantic_belief_abstract,
    classify_stage,
    bucket_exploration,
    build_hibo_groups,
    hibo_step_norm_reward,
    compute_hibo_outcome_advantage,
)

__all__ = [
    # Main API
    'compute_rebel_advantage',
    'episode_norm_reward',
    'step_norm_reward_by_belief',

    # Belief grouping
    'build_belief_group',
    'canonicalize_belief',

    # HiBO (V11)
    'semantic_belief_abstract',
    'classify_stage',
    'bucket_exploration',
    'build_hibo_groups',
    'hibo_step_norm_reward',
    'compute_hibo_outcome_advantage',

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
