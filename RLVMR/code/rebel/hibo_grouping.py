"""
HiBO: Hierarchical Belief-Observation Grouping for Step-Level Advantage Estimation

Core innovation of ReBel V11:
- Layer 1: Precise observation hash grouping (same as GiGPO, high quality but sparse)
- Layer 2: Semantic belief abstraction grouping (lower quality but dense, for singleton rescue)

This module recovers 70-85% of step-level learning signal that GiGPO wastes
due to observation hash singletons.
"""

import numpy as np
import torch
import hashlib
import json
import uuid
from collections import defaultdict, Counter
from typing import Dict, Any, List, Tuple, Optional


# ============================================================================ #
# =================== Semantic Belief Abstraction ============================ #
# ============================================================================ #

def classify_stage(subgoal: str) -> str:
    """
    Classify subgoal text into a discrete stage type (~10 categories).
    Consistent with _get_stage_type_v6 from core_rebel.py.
    """
    subgoal = subgoal.lower().strip()

    if any(x in subgoal for x in ['complete', 'done', 'finished', 'success']):
        return 'complete'
    if any(x in subgoal for x in ['find', 'look for', 'search', 'locate']):
        return 'find'
    if any(x in subgoal for x in ['go to', 'goto', 'navigate', 'move to']):
        return 'navigate'
    if any(x in subgoal for x in ['pick up', 'pick', 'take', 'grab']):
        return 'pickup'
    if any(x in subgoal for x in ['put', 'place', 'drop']):
        return 'place'
    if any(x in subgoal for x in ['heat', 'cook', 'warm', 'microwave']):
        return 'heat'
    if any(x in subgoal for x in ['cool', 'chill', 'fridge', 'refrigerat']):
        return 'cool'
    if any(x in subgoal for x in ['clean', 'wash', 'rinse', 'sink']):
        return 'clean'
    if any(x in subgoal for x in ['turn on', 'use', 'toggle', 'examine']):
        return 'use'
    if any(x in subgoal for x in ['open', 'close']):
        return 'interact'
    return 'other'


def classify_webshop_stage(subgoal: str) -> str:
    """
    Classify WebShop subgoal text into a discrete stage type (~6 categories).
    """
    subgoal = subgoal.lower().strip()

    if any(x in subgoal for x in ['done', 'bought', 'purchased', 'complete', 'finished']):
        return 'done'
    if any(x in subgoal for x in ['buy', 'purchase', 'confirm', 'checkout']):
        return 'ready_to_buy'
    if any(x in subgoal for x in ['select', 'option', 'choose', 'pick', 'configure', 'size', 'color']):
        return 'selecting_options'
    if any(x in subgoal for x in ['view', 'look at', 'check', 'examine', 'detail', 'product page']):
        return 'viewing_product'
    if any(x in subgoal for x in ['browse', 'result', 'compare', 'scroll', 'next', 'prev']):
        return 'browsing_results'
    if any(x in subgoal for x in ['search', 'find', 'query', 'look for']):
        return 'searching'
    return 'searching'  # Default to searching


def webshop_semantic_belief_abstract(belief_json: Optional[Dict[str, Any]]) -> str:
    """
    Extract coarse-grained semantic features from WebShop belief JSON.

    Features:
    - stage: ~6 categories (searching, browsing_results, viewing_product, selecting_options, ready_to_buy, done)
    - target_match: 3 levels (none, partial, exact)
    - options_selected: bool (2 values)
    - query_diversity: 2 levels (low=0-2, high=3+)
    - verification_completeness: 3 levels (none, partial, complete)

    Theoretical combinations: 6 x 3 x 2 x 2 x 3 = 216 (expected active: ~50-80)

    Args:
        belief_json: Parsed belief state dictionary

    Returns:
        16-char MD5 hash of the semantic features
    """
    if belief_json is None or not isinstance(belief_json, dict):
        return "null_belief_hibo"

    try:
        search_progress = belief_json.get('search_progress', {}) or {}
        product_understanding = belief_json.get('product_understanding', {}) or {}
        exploration_state = belief_json.get('exploration_state', {}) or {}
        attribute_verification = belief_json.get('attribute_verification', {}) or {}

        # Dimension 1: Current stage (~6 categories)
        subgoal = str(search_progress.get('updated_subgoal', '')).lower().strip()
        search_status = str(search_progress.get('search_status', '')).lower().strip()
        # Use search_status directly if it matches known stages
        if search_status in ('not_started', 'searching', 'product_found', 'options_selecting', 'ready_to_buy'):
            stage_map = {
                'not_started': 'searching',
                'searching': 'searching',
                'product_found': 'viewing_product',
                'options_selecting': 'selecting_options',
                'ready_to_buy': 'ready_to_buy',
            }
            stage = stage_map.get(search_status, 'searching')
        else:
            stage = classify_webshop_stage(subgoal)

        # Dimension 2: Target product match (3 categories)
        match_level = str(product_understanding.get('current_product_match', 'none')).lower().strip()
        if match_level not in ('none', 'partial', 'exact'):
            match_level = 'none'

        # Dimension 3: Options selected? (2 categories)
        options_selected_list = exploration_state.get('options_selected', [])
        if not isinstance(options_selected_list, list):
            options_selected_list = []
        has_options = len(options_selected_list) > 0

        # Dimension 4: Query diversity (2 categories)
        queries_tried = exploration_state.get('queries_tried', [])
        if not isinstance(queries_tried, list):
            queries_tried = []
        query_diversity = 'high' if len(queries_tried) >= 3 else 'low'

        # Dimension 5: Verification completeness (3 categories)
        verified = attribute_verification.get('verified', [])
        unverified = attribute_verification.get('unverified', [])
        inferred_only = attribute_verification.get('inferred_only', [])
        if not isinstance(verified, list):
            verified = []
        if not isinstance(unverified, list):
            unverified = []
        if not isinstance(inferred_only, list):
            inferred_only = []

        total_tracked = len(verified) + len(unverified) + len(inferred_only)
        if total_tracked == 0:
            verification_level = 'none'
        elif len(unverified) == 0 and len(inferred_only) == 0:
            verification_level = 'complete'
        else:
            verification_level = 'partial'

        features = {
            'stage': stage,
            'match': match_level,
            'has_options': has_options,
            'query_div': query_diversity,
            'verify': verification_level,
        }

        canonical_str = json.dumps(features, sort_keys=True)
        return hashlib.md5(canonical_str.encode()).hexdigest()[:16]

    except Exception:
        return "error_belief_hbo"


def bucket_exploration(cleared_count: int) -> str:
    """
    Bucket exploration progress into 3 discrete levels.

    Args:
        cleared_count: Number of cleared/visited receptacles

    Returns:
        'low' (0-2), 'mid' (3-5), or 'high' (6+)
    """
    if cleared_count <= 2:
        return 'low'
    elif cleared_count <= 5:
        return 'mid'
    else:
        return 'high'


def semantic_belief_abstract(belief_json: Optional[Dict[str, Any]]) -> str:
    """
    Extract coarse-grained semantic features from structured belief JSON.

    Design principles:
    1. Use categorical features (not continuous/text) for reliable hashing
    2. Capture key state dimensions relevant to decision-making
    3. Target ~50-100 unique combinations -> group size ~5-15

    Features:
    - stage: ~10 categories (find, navigate, pickup, place, heat, cool, clean, use, complete, other)
    - target_found: bool (2 values)
    - holding: bool (2 values)
    - explore_level: 3 levels (low, mid, high)

    Theoretical combinations: 10 × 2 × 2 × 3 = 120
    Expected active: ~40-60

    Args:
        belief_json: Parsed belief state dictionary

    Returns:
        16-char MD5 hash of the semantic features
    """
    if belief_json is None or not isinstance(belief_json, dict):
        return "null_belief_hibo"

    try:
        task_progress = belief_json.get('task_progress_update', {}) or {}
        world_model = belief_json.get('world_model_update', {}) or {}
        exploration = belief_json.get('exploration_map_update', {}) or {}

        # Dimension 1: Current stage (~10 categories)
        subgoal = str(task_progress.get('updated_subgoal', '')).lower().strip()
        stage = classify_stage(subgoal)

        # Dimension 2: Target object found? (2 categories)
        found_objects = world_model.get('found_objects', {})
        if not isinstance(found_objects, dict):
            found_objects = {}
        target_found = len(found_objects) > 0

        # Dimension 3: Holding an object? (2 categories)
        inventory = world_model.get('inventory', [])
        if isinstance(inventory, str):
            inventory = [inventory] if inventory else []
        elif not isinstance(inventory, list):
            inventory = []
        holding = len(inventory) > 0

        # Dimension 4: Exploration progress (3 categories)
        cleared = exploration.get('cleared_receptacles', [])
        if not isinstance(cleared, list):
            cleared = []
        explore_level = bucket_exploration(len(cleared))

        features = {
            'stage': stage,
            'target_found': target_found,
            'holding': holding,
            'explore_level': explore_level,
        }

        canonical_str = json.dumps(features, sort_keys=True)
        return hashlib.md5(canonical_str.encode()).hexdigest()[:16]

    except Exception:
        return "error_belief_hbo"


# ============================================================================ #
# =================== HiBO Group Building ==================================== #
# ============================================================================ #

def _to_hashable(x):
    """Convert value to hashable type (for observation hashing)."""
    if isinstance(x, (int, float, str, bool)):
        return x
    elif isinstance(x, (np.integer, np.floating)):
        return x.item()
    elif isinstance(x, np.ndarray):
        return tuple(x.flatten())
    elif isinstance(x, (list, tuple)):
        return tuple(_to_hashable(e) for e in x)
    elif isinstance(x, dict):
        return tuple(sorted((k, _to_hashable(v)) for k, v in x.items()))
    else:
        return str(x)


def build_hibo_groups(
    anchor_obs: np.ndarray,
    belief_abstracts: np.ndarray,
    index: np.ndarray,
    min_obs_group_size: int = 2,
    summarize: bool = False
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Build hierarchical groups: obs hash primary, belief abstract fallback.

    For each (uid, step):
    - If obs hash group has >= min_obs_group_size samples: use obs group (high quality)
    - Otherwise: fall back to belief abstract group (dense coverage)

    Args:
        anchor_obs: (batch_size,) Raw observation strings for obs hashing
        belief_abstracts: (batch_size,) Belief abstract hashes from semantic_belief_abstract()
        index: (batch_size,) Prompt UIDs
        min_obs_group_size: Minimum obs group size to use obs grouping (default 2)
        summarize: Whether to print group statistics

    Returns:
        final_group_uids: (batch_size,) UUID strings for final group assignment
        stats: Dictionary of HiBO grouping statistics
    """
    batch_size = len(anchor_obs)
    final_group_uids = np.empty(batch_size, dtype=object)

    # Track statistics
    obs_group_used = 0
    belief_group_used = 0
    obs_group_sizes = []
    belief_group_sizes = []
    all_group_sizes = []

    unique_uids = np.unique(index)

    for uid in unique_uids:
        uid_mask = (index == uid)
        uid_indices = np.where(uid_mask)[0]

        # --- Layer 1: Obs hash grouping ---
        obs_clusters = defaultdict(list)
        for i in uid_indices:
            obs_key = _to_hashable(anchor_obs[i])
            obs_clusters[obs_key].append(i)

        # --- Layer 2: Belief abstract grouping ---
        belief_clusters = defaultdict(list)
        for i in uid_indices:
            belief_key = belief_abstracts[i]
            belief_clusters[belief_key].append(i)

        # --- Assign groups hierarchically ---
        # First pass: identify which obs clusters are large enough
        obs_assigned = set()
        for obs_key, members in obs_clusters.items():
            if len(members) >= min_obs_group_size:
                # Use obs group
                group_uid = str(uuid.uuid4())
                for idx in members:
                    final_group_uids[idx] = group_uid
                    obs_assigned.add(idx)
                obs_group_used += len(members)
                obs_group_sizes.append(len(members))
                all_group_sizes.append(len(members))

        # Second pass: singleton obs samples fall back to belief groups
        # Build belief groups only for unassigned samples
        unassigned_belief_clusters = defaultdict(list)
        for i in uid_indices:
            if i not in obs_assigned:
                belief_key = belief_abstracts[i]
                unassigned_belief_clusters[belief_key].append(i)

        for belief_key, members in unassigned_belief_clusters.items():
            group_uid = str(uuid.uuid4())
            for idx in members:
                final_group_uids[idx] = group_uid
            belief_group_used += len(members)
            belief_group_sizes.append(len(members))
            all_group_sizes.append(len(members))

    # Validate all assigned
    if None in final_group_uids or np.any(final_group_uids == None):
        missing = np.where(final_group_uids == None)[0]
        raise ValueError(f"HiBO: Failed to assign groups at indices: {missing}")

    # Compute statistics
    total = obs_group_used + belief_group_used
    stats = {
        'num_groups': len(set(final_group_uids)),
        'mean_group_size': float(np.mean(all_group_sizes)) if all_group_sizes else 0.0,
        'median_group_size': float(np.median(all_group_sizes)) if all_group_sizes else 0.0,
        'min_group_size': int(np.min(all_group_sizes)) if all_group_sizes else 0,
        'max_group_size': int(np.max(all_group_sizes)) if all_group_sizes else 0,
        'std_group_size': float(np.std(all_group_sizes)) if all_group_sizes else 0.0,
        'obs_group_ratio': obs_group_used / total if total > 0 else 0.0,
        'belief_fallback_ratio': belief_group_used / total if total > 0 else 0.0,
        'obs_group_count': len(obs_group_sizes),
        'belief_group_count': len(belief_group_sizes),
        'obs_mean_size': float(np.mean(obs_group_sizes)) if obs_group_sizes else 0.0,
        'belief_mean_size': float(np.mean(belief_group_sizes)) if belief_group_sizes else 0.0,
        'single_sample_ratio': sum(1 for s in all_group_sizes if s == 1) / len(all_group_sizes) if all_group_sizes else 0.0,
    }

    if summarize:
        print("=" * 60)
        print("HiBO Grouping Statistics")
        print("=" * 60)
        print(f"  Total samples:       {total}")
        print(f"  Obs groups used:     {obs_group_used} ({stats['obs_group_ratio']:.1%})")
        print(f"  Belief fallback:     {belief_group_used} ({stats['belief_fallback_ratio']:.1%})")
        print(f"  Num groups:          {stats['num_groups']}")
        print(f"  Mean group size:     {stats['mean_group_size']:.1f}")
        print(f"  Median group size:   {stats['median_group_size']:.1f}")
        print(f"  Single-sample ratio: {stats['single_sample_ratio']:.1%}")
        print(f"  Obs mean size:       {stats['obs_mean_size']:.1f}")
        print(f"  Belief mean size:    {stats['belief_mean_size']:.1f}")
        print("=" * 60)
    else:
        print(f"[HiBO] obs={stats['obs_group_ratio']:.1%} belief={stats['belief_fallback_ratio']:.1%} "
              f"groups={stats['num_groups']} mean_size={stats['mean_group_size']:.1f} "
              f"singleton={stats['single_sample_ratio']:.1%}")

    return final_group_uids, stats


# ============================================================================ #
# =================== HiBO Step Advantage Computation ======================== #
# ============================================================================ #

def hibo_step_norm_reward(
    step_rewards: torch.Tensor,
    eos_mask: torch.Tensor,
    hibo_group_uids: np.ndarray,
    epsilon: float = 1e-6,
    remove_std: bool = True,
) -> torch.Tensor:
    """
    Compute step-level advantage using HiBO groups.

    Within each HiBO group, normalize step returns to compute advantage.
    Singleton groups still get A_step = 0, but HiBO drastically reduces
    the number of singletons compared to pure obs grouping.

    Args:
        step_rewards: (batch_size,) Step-level discounted returns
        eos_mask: (batch_size, response_length) Response mask
        hibo_group_uids: (batch_size,) HiBO group UIDs
        epsilon: Numerical stability
        remove_std: If True, use mean-only normalization (recommended)

    Returns:
        step_advantages: (batch_size, response_length)
    """
    response_length = eos_mask.shape[-1]
    scores = step_rewards.clone()

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[hibo_group_uids[i]].append(scores[i])

        for idx in id2score:
            if len(id2score[idx]) == 1:
                # Singleton: mean = self → advantage = 0
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"HiBO: empty group {idx}")

        for i in range(bsz):
            if remove_std:
                scores[i] = scores[i] - id2mean[hibo_group_uids[i]]
            else:
                scores[i] = (scores[i] - id2mean[hibo_group_uids[i]]) / (id2std[hibo_group_uids[i]] + epsilon)

        step_advantages = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask

    return step_advantages


def compute_hibo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    step_rewards: torch.Tensor,
    eos_mask: torch.Tensor,
    anchor_obs: np.ndarray,
    belief_abstracts: np.ndarray,
    index: np.ndarray,
    epsilon: float = 1e-6,
    step_advantage_w: float = 0.5,
    mode: str = "mean_norm",
    min_obs_group_size: int = 2,
    summarize: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """
    Compute HiBO advantage: A_total = A_episode + λ × A_step(HiBO)

    This is the main entry point for HiBO advantage computation.

    Args:
        token_level_rewards: (bs, response_length) Token-level rewards for episode advantage
        step_rewards: (bs,) Step-level discounted returns for step advantage
        eos_mask: (bs, response_length) Response mask
        anchor_obs: (bs,) Raw observation strings for obs hash grouping
        belief_abstracts: (bs,) Belief abstract hashes for belief fallback grouping
        index: (bs,) Prompt UIDs for episode grouping
        epsilon: Numerical stability
        step_advantage_w: Weight for step advantage (lambda)
        mode: Normalization mode ('mean_norm' or 'mean_std_norm')
        min_obs_group_size: Minimum obs group size before falling back to belief
        summarize: Whether to print detailed statistics

    Returns:
        advantages: (bs, response_length) Combined advantages
        returns: (bs, response_length) Same as advantages (for compatibility)
        details: Dictionary of HiBO statistics
    """
    if mode == "mean_std_norm":
        remove_std = False
    elif mode == "mean_norm":
        remove_std = True
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # --- Episode-level advantage (identical to GiGPO) ---
    from gigpo.core_gigpo import episode_norm_reward
    episode_advantages = episode_norm_reward(
        token_level_rewards, eos_mask, index, epsilon, remove_std
    )

    # --- Step-level advantage with HiBO grouping ---
    hibo_group_uids, hibo_stats = build_hibo_groups(
        anchor_obs=anchor_obs,
        belief_abstracts=belief_abstracts,
        index=index,
        min_obs_group_size=min_obs_group_size,
        summarize=summarize,
    )

    step_advantages = hibo_step_norm_reward(
        step_rewards, eos_mask, hibo_group_uids, epsilon, remove_std
    )

    # --- Combine ---
    scores = episode_advantages + step_advantage_w * step_advantages

    details = {
        'episode_advantages': episode_advantages,
        'step_advantages': step_advantages,
        'hibo_stats': hibo_stats,
    }

    return scores, scores, details
