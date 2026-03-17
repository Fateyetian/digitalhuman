"""
GraPO: Graph-based Policy Optimization via Path-Level Causal Credit Assignment

Core innovation: instead of comparing actions at the SAME state (GiGPO/HiBO),
GraPO compares PATH SEGMENTS between the same fork ancestor (b) and merge
successor (m), providing a causally grounded credit signal even for steps
that form singleton groups under exact-observation matching.

Three-layer grouping strategy:
  Layer 1 — Exact observation hash anchors (GiGPO-identical, high fidelity)
  Layer 2 — Belief-abstract soft anchors (HiBO-identical, lower fidelity but denser)
  Layer 3 — Path propagation for remaining singletons:
             For each singleton step v in trajectory t:
               b = nearest anchor step BEFORE v in t  (fork ancestor)
               m = nearest anchor step AFTER  v in t  (merge successor)
               path_group = (uid, b_stable_key, m_stable_key)
             → All steps sharing the same (b, m) bracket across trajectories
               are grouped for comparative advantage computation.

Return metric per environment:
  ALFWorld  — discounted return G[v] (already in step_rewards)
  WebShop   — raw cumulative return from step v to trajectory end
              (richer signal because WebShop reward is continuous in [0,1])
"""

import hashlib
import json
import uuid
from collections import defaultdict
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

# Re-use the semantic abstractors already defined in hibo_grouping
from rebel.hibo_grouping import (
    semantic_belief_abstract,
    webshop_semantic_belief_abstract,
    _to_hashable,
)


# ============================================================================ #
#               Stable anchor-key helpers                                      #
# ============================================================================ #

def _obs_stable_key(uid: str, obs: Any) -> str:
    """
    Stable, cross-trajectory string key for an observation anchor.
    Two steps belong to the same Layer-1 anchor iff they share the same uid
    AND their observation hashes match (GiGPO semantics).
    """
    obs_hash = hashlib.md5(str(_to_hashable(obs)).encode()).hexdigest()[:16]
    return f"obs|{uid}|{obs_hash}"


def _belief_stable_key(uid: str, belief_abstract_hash: str) -> str:
    """
    Stable, cross-trajectory string key for a belief-abstract anchor.
    Two steps belong to the same Layer-2 anchor iff they share the same uid
    AND their belief-abstract hashes match (HiBO-style soft matching).
    """
    return f"bel|{uid}|{belief_abstract_hash}"


def _path_stable_key(uid: str, b_key: str, m_key: str) -> str:
    """
    Stable key for a path group defined by a (fork, merge) anchor bracket.
    """
    return f"path|{uid}|{b_key}|{m_key}"


# ============================================================================ #
#               Cumulative reward computation (WebShop)                        #
# ============================================================================ #

def compute_cumulative_returns(
    raw_rewards: np.ndarray,
    traj_uids: np.ndarray,
) -> np.ndarray:
    """
    Compute raw (undiscounted) cumulative return from step t to trajectory end.

    For WebShop: reward is continuous in [0, 1] and only present at the terminal
    step.  Cumulative return from step t is therefore simply the total episode
    reward (constant within trajectory) — it equals raw_rewards summed from t
    to T-1, which equals sum(raw_rewards) for that trajectory because all
    intermediate rewards are 0.

    Concretely: CumR[t] = sum_{k=t}^{T-1} r_k  (no discount factor)

    Returns:
        cum_returns: (batch_size,) float32 array
    """
    cum_returns = np.zeros(len(raw_rewards), dtype=np.float32)

    for traj_uid in np.unique(traj_uids):
        idx = np.where(traj_uids == traj_uid)[0]   # ordered batch positions
        traj_rewards = raw_rewards[idx].astype(np.float32)

        # Backward cumulative sum
        running = 0.0
        traj_cum = np.zeros(len(idx), dtype=np.float32)
        for t in reversed(range(len(idx))):
            running += traj_rewards[t]
            traj_cum[t] = running

        for pos, batch_i in enumerate(idx):
            cum_returns[batch_i] = traj_cum[pos]

    return cum_returns


# ============================================================================ #
#               Main three-layer grouping                                      #
# ============================================================================ #

def build_grapo_groups(
    anchor_obs: np.ndarray,
    belief_abstracts: np.ndarray,
    traj_uids: np.ndarray,
    uid_array: np.ndarray,
    step_returns: np.ndarray,
    min_anchor_group_size: int = 2,
    summarize: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Build three-layer GraPO groups and compute per-step comparison returns.

    Args:
        anchor_obs:           (bs,) Pre-action observation strings.
        belief_abstracts:     (bs,) Pre-computed belief-abstract hashes
                              (from semantic_belief_abstract or
                               webshop_semantic_belief_abstract).
        traj_uids:            (bs,) Trajectory UIDs (unique per rollout).
        uid_array:            (bs,) Prompt UIDs (shared across rollouts of
                              the same prompt).
        step_returns:         (bs,) Per-step comparison return.
                              • ALFWorld  → discounted G[t] from ray_trainer
                              • WebShop  → cumulative CumR[t] from
                                compute_cumulative_returns()
        min_anchor_group_size: Minimum group size to qualify as an anchor
                              (default 2 — same as GiGPO/HiBO).
        summarize:            Print detailed statistics.

    Returns:
        group_uids:      (bs,) UUID strings for group assignment.
                         Steps in the same group are compared for step-level
                         advantage normalisation.
        comparison_returns: (bs,) Return values to normalise within groups.
                         Same as step_returns for Layer-1/2 groups.
                         For Layer-3 path groups: same — G[t] or CumR[t] is
                         the right comparison metric because we control for
                         trajectory context via the (b, m) bracket.
        layer_assigned:  (bs,) int8 array: 1=obs, 2=belief, 3=path, 0=singleton
        stats:           Dict of grouping diagnostics.
    """
    bs = len(uid_array)

    # ------------------------------------------------------------------ #
    # Phase 1: Identify Layer-1 (obs) and Layer-2 (belief) anchors        #
    # For each step, compute a STABLE cross-trajectory anchor key or None. #
    # ------------------------------------------------------------------ #
    anchor_stable_keys = np.array([None] * bs, dtype=object)

    for uid in np.unique(uid_array):
        uid_idx = np.where(uid_array == uid)[0]

        # --- Layer 1: exact observation hash ---
        obs_groups: Dict[str, list] = defaultdict(list)
        for i in uid_idx:
            key = _obs_stable_key(uid, anchor_obs[i])
            obs_groups[key].append(i)

        obs_anchored: set = set()
        for obs_key, members in obs_groups.items():
            if len(members) >= min_anchor_group_size:
                for i in members:
                    anchor_stable_keys[i] = obs_key
                    obs_anchored.add(i)

        # --- Layer 2: belief-abstract soft anchor (for un-anchored steps) ---
        belief_groups: Dict[str, list] = defaultdict(list)
        for i in uid_idx:
            if i not in obs_anchored:
                key = _belief_stable_key(uid, belief_abstracts[i])
                belief_groups[key].append(i)

        for bel_key, members in belief_groups.items():
            if len(members) >= min_anchor_group_size:
                for i in members:
                    anchor_stable_keys[i] = bel_key

    # ------------------------------------------------------------------ #
    # Phase 2: Build ordered step lists per trajectory                    #
    # np.where preserves the insertion order, which matches the temporal  #
    # order because traj_collector appends steps sequentially.            #
    # ------------------------------------------------------------------ #
    traj_ordered: Dict[str, np.ndarray] = {}
    for traj_uid in np.unique(traj_uids):
        traj_ordered[traj_uid] = np.where(traj_uids == traj_uid)[0]

    # ------------------------------------------------------------------ #
    # Phase 3: Path propagation for remaining singletons (Layer 3)        #
    # For each un-anchored step v in trajectory t:                        #
    #   b = stable key of the NEAREST anchor step BEFORE v in t           #
    #   m = stable key of the NEAREST anchor step AFTER  v in t           #
    # path_key = (uid, b, m) — shared across trajectories                 #
    # ------------------------------------------------------------------ #
    path_keys = np.array([None] * bs, dtype=object)

    for traj_uid, traj_idx in traj_ordered.items():
        T = len(traj_idx)
        uid = uid_array[traj_idx[0]]
        # Anchor key at each temporal position in this trajectory
        anchor_at_pos = [anchor_stable_keys[traj_idx[t]] for t in range(T)]

        for t in range(T):
            batch_i = traj_idx[t]
            if anchor_stable_keys[batch_i] is not None:
                continue  # already anchored in Layer 1 or 2

            # Search backward for b
            b_key = None
            for bt in range(t - 1, -1, -1):
                if anchor_at_pos[bt] is not None:
                    b_key = anchor_at_pos[bt]
                    break

            # Search forward for m
            m_key = None
            for mt in range(t + 1, T):
                if anchor_at_pos[mt] is not None:
                    m_key = anchor_at_pos[mt]
                    break

            if b_key is not None and m_key is not None:
                path_keys[batch_i] = _path_stable_key(uid, b_key, m_key)
            # else: true singleton — b or m (or both) not found in this traj

    # ------------------------------------------------------------------ #
    # Phase 4: Map stable keys → UUID group_uids                          #
    # We use UUIDs to stay compatible with the existing normalisation      #
    # infrastructure (step_norm_reward uses string-keyed id2score dicts). #
    # ------------------------------------------------------------------ #
    stable_key_to_uuid: Dict[str, str] = {}

    def _get_or_create_uuid(key: str) -> str:
        if key not in stable_key_to_uuid:
            stable_key_to_uuid[key] = str(uuid.uuid4())
        return stable_key_to_uuid[key]

    group_uids = np.empty(bs, dtype=object)
    layer_assigned = np.zeros(bs, dtype=np.int8)

    # Layer 1 / Layer 2
    for i in range(bs):
        if anchor_stable_keys[i] is not None:
            group_uids[i] = _get_or_create_uuid(anchor_stable_keys[i])
            layer_assigned[i] = 1 if anchor_stable_keys[i].startswith("obs|") else 2

    # Layer 3 — path groups
    for i in range(bs):
        if anchor_stable_keys[i] is None and path_keys[i] is not None:
            group_uids[i] = _get_or_create_uuid(path_keys[i])
            layer_assigned[i] = 3

    # Layer 0 — true singletons (no bracket found)
    for i in range(bs):
        if group_uids[i] is None:
            group_uids[i] = str(uuid.uuid4())   # unique → A_step = 0
            layer_assigned[i] = 0

    # ------------------------------------------------------------------ #
    # Statistics                                                           #
    # ------------------------------------------------------------------ #
    uid_to_group: Dict[str, list] = defaultdict(list)
    for i in range(bs):
        uid_to_group[group_uids[i]].append(i)

    group_sizes = [len(v) for v in uid_to_group.values()]
    n_layer1 = int(np.sum(layer_assigned == 1))
    n_layer2 = int(np.sum(layer_assigned == 2))
    n_layer3 = int(np.sum(layer_assigned == 3))
    n_singleton = int(np.sum(layer_assigned == 0))
    total = bs

    stats = {
        "num_groups":            len(group_sizes),
        "mean_group_size":       float(np.mean(group_sizes)) if group_sizes else 0.0,
        "median_group_size":     float(np.median(group_sizes)) if group_sizes else 0.0,
        "min_group_size":        int(np.min(group_sizes)) if group_sizes else 0,
        "max_group_size":        int(np.max(group_sizes)) if group_sizes else 0,
        "std_group_size":        float(np.std(group_sizes)) if group_sizes else 0.0,
        # Layer breakdown (fraction of batch)
        "layer1_obs_ratio":      n_layer1 / total,
        "layer2_belief_ratio":   n_layer2 / total,
        "layer3_path_ratio":     n_layer3 / total,
        "layer0_singleton_ratio": n_singleton / total,
        # Coverage = fraction with non-zero step advantage
        "coverage":              (total - n_singleton) / total,
        # Counts
        "n_layer1": n_layer1,
        "n_layer2": n_layer2,
        "n_layer3": n_layer3,
        "n_singleton": n_singleton,
    }

    if summarize:
        print("=" * 65)
        print("GraPO Grouping Statistics")
        print("=" * 65)
        print(f"  Batch size:          {total}")
        print(f"  Layer-1 (obs):       {n_layer1:>5}  ({stats['layer1_obs_ratio']:.1%})")
        print(f"  Layer-2 (belief):    {n_layer2:>5}  ({stats['layer2_belief_ratio']:.1%})")
        print(f"  Layer-3 (path):      {n_layer3:>5}  ({stats['layer3_path_ratio']:.1%})")
        print(f"  Layer-0 (singleton): {n_singleton:>5}  ({stats['layer0_singleton_ratio']:.1%})")
        print(f"  Coverage:            {stats['coverage']:.1%}")
        print(f"  Num groups:          {stats['num_groups']}")
        print(f"  Mean group size:     {stats['mean_group_size']:.2f}")
        print("=" * 65)
    else:
        print(
            f"[GraPO] obs={stats['layer1_obs_ratio']:.1%} "
            f"belief={stats['layer2_belief_ratio']:.1%} "
            f"path={stats['layer3_path_ratio']:.1%} "
            f"singleton={stats['layer0_singleton_ratio']:.1%} "
            f"coverage={stats['coverage']:.1%} "
            f"groups={stats['num_groups']}"
        )

    return group_uids, step_returns.copy(), layer_assigned, stats


# ============================================================================ #
#               Step-level advantage normalisation                             #
# ============================================================================ #

def grapo_step_norm_reward(
    comparison_returns: torch.Tensor,
    eos_mask: torch.Tensor,
    group_uids: np.ndarray,
    epsilon: float = 1e-6,
    remove_std: bool = True,
) -> torch.Tensor:
    """
    Normalise comparison_returns within each GraPO group to produce
    step-level advantages.

    Identical semantics to hibo_step_norm_reward / step_norm_reward:
      • Group singletons (size == 1) get advantage = 0 because
        mean(group) == the single value → score - mean = 0.
      • This naturally handles Layer-0 true singletons.

    Args:
        comparison_returns: (bs,) Returns to normalise.
        eos_mask:           (bs, response_length).
        group_uids:         (bs,) GraPO group assignments.
        epsilon:            Numerical stability.
        remove_std:         True → mean-only; False → mean-std normalisation.

    Returns:
        step_advantages: (bs, response_length)
    """
    response_length = eos_mask.shape[-1]
    scores = comparison_returns.clone()

    id2score: Dict[str, list] = defaultdict(list)
    id2mean: Dict[str, torch.Tensor] = {}
    id2std:  Dict[str, torch.Tensor] = {}

    with torch.no_grad():
        bsz = scores.shape[0]

        # Accumulate scores per group
        for i in range(bsz):
            id2score[group_uids[i]].append(scores[i])

        # Compute group statistics
        for gid, group_scores in id2score.items():
            t = torch.stack(group_scores)
            id2mean[gid] = t.mean()
            id2std[gid]  = t.std() if len(group_scores) > 1 else torch.tensor(1.0)

        # Normalise
        for i in range(bsz):
            gid = group_uids[i]
            if remove_std:
                scores[i] = scores[i] - id2mean[gid]
            else:
                scores[i] = (scores[i] - id2mean[gid]) / (id2std[gid] + epsilon)

        step_advantages = scores.unsqueeze(-1).expand(-1, response_length) * eos_mask

    return step_advantages


# ============================================================================ #
#               Public entry-point                                             #
# ============================================================================ #

def compute_grapo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    step_rewards: torch.Tensor,
    eos_mask: torch.Tensor,
    anchor_obs: np.ndarray,
    belief_abstracts: np.ndarray,
    traj_uids: np.ndarray,
    uid_array: np.ndarray,
    raw_rewards: Optional[np.ndarray] = None,
    epsilon: float = 1e-6,
    step_advantage_w: float = 0.5,
    mode: str = "mean_norm",
    min_anchor_group_size: int = 2,
    env_type: str = "alfworld",
    gamma: float = 1.0,
    summarize: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """
    Compute GraPO advantage: A_total = A_episode + λ × A_step(GraPO)

    This is the main entry point called from ray_trainer.py.

    Args:
        token_level_rewards: (bs, response_length) for episode advantage.
        step_rewards:        (bs,) Discounted returns G[t] (pre-computed by
                             compute_step_discounted_returns in trainer).
        eos_mask:            (bs, response_length).
        anchor_obs:          (bs,) Pre-action observation strings.
        belief_abstracts:    (bs,) Belief-abstract hashes.
        traj_uids:           (bs,) Trajectory UIDs.
        uid_array:           (bs,) Prompt UIDs.
        raw_rewards:         (bs,) Raw step rewards from environment.
                             Required when env_type == 'webshop'.
        epsilon:             Numerical stability.
        step_advantage_w:    λ — weight for step-level advantage term.
        mode:                'mean_norm' or 'mean_std_norm'.
        min_anchor_group_size: Minimum group size for Layer 1/2 anchors.
        env_type:            'alfworld' → use discounted returns (step_rewards).
                             'webshop'  → use raw cumulative returns.
        gamma:               Discount factor (used only for env_type='alfworld').
        summarize:           Print detailed group statistics.

    Returns:
        advantages:  (bs, response_length) Combined advantages.
        returns:     (bs, response_length) Same tensor (for API compatibility).
        details:     Dict containing episode_advantages, step_advantages,
                     layer_assigned, and grapo_stats.
    """
    remove_std = (mode == "mean_norm")
    if mode not in ("mean_norm", "mean_std_norm"):
        raise ValueError(f"GraPO: unknown mode '{mode}'. Use 'mean_norm' or 'mean_std_norm'.")

    if env_type == "webshop":
        if raw_rewards is None:
            raise ValueError(
                "GraPO: env_type='webshop' requires raw_rewards "
                "(non_tensor_batch['rewards'])."
            )
        # Raw cumulative return: richer signal for continuous WebShop rewards
        comparison_returns_np = compute_cumulative_returns(raw_rewards, traj_uids)
        comparison_returns = torch.tensor(
            comparison_returns_np, dtype=torch.float32,
            device=step_rewards.device
        )
    elif env_type == "alfworld":
        # Discounted return already computed by trainer
        comparison_returns = step_rewards
    else:
        raise ValueError(
            f"GraPO: unknown env_type '{env_type}'. Use 'alfworld' or 'webshop'."
        )

    # --- Episode-level advantage (identical to GiGPO / HiBO) ---
    from gigpo.core_gigpo import episode_norm_reward
    episode_advantages = episode_norm_reward(
        token_level_rewards, eos_mask, uid_array, epsilon, remove_std
    )

    # --- GraPO three-layer grouping ---
    comparison_returns_np = comparison_returns.cpu().numpy()
    group_uids, comp_returns_out, layer_assigned, grapo_stats = build_grapo_groups(
        anchor_obs=anchor_obs,
        belief_abstracts=belief_abstracts,
        traj_uids=traj_uids,
        uid_array=uid_array,
        step_returns=comparison_returns_np,
        min_anchor_group_size=min_anchor_group_size,
        summarize=summarize,
    )
    comparison_returns_tensor = torch.tensor(
        comp_returns_out, dtype=torch.float32, device=step_rewards.device
    )

    # --- Step-level advantage via GraPO group normalisation ---
    step_advantages = grapo_step_norm_reward(
        comparison_returns=comparison_returns_tensor,
        eos_mask=eos_mask,
        group_uids=group_uids,
        epsilon=epsilon,
        remove_std=remove_std,
    )

    # --- Combine episode + step ---
    scores = episode_advantages + step_advantage_w * step_advantages

    details = {
        "episode_advantages": episode_advantages,
        "step_advantages":    step_advantages,
        "layer_assigned":     layer_assigned,
        "grapo_stats":        grapo_stats,
    }

    return scores, scores, details
