import numpy as np
import torch


def compute_bdrs_outcome_advantage(
    token_level_rewards: torch.Tensor,
    bdrs_rewards: torch.Tensor,
    eos_mask: torch.Tensor,
    index: np.array,
    epsilon: float = 1e-6,
    step_advantage_w: float = 1.0,
    mode: str = "mean_std_norm",
):
    """
    类似 RLVMR，将 episode 级别 token 奖励与 step 级别 BDRS 奖励融合。
    分组归一化策略与 RLVMR 对齐，便于替换。
    """
    if mode == "mean_std_norm":
        remove_std = False
    elif mode == "mean_norm":
        remove_std = True
    else:
        raise ValueError(f"Unknown mode: {mode}")

    episode_advantages = _episode_norm_reward(token_level_rewards, eos_mask, index, epsilon, remove_std)
    step_advantages = _step_group_norm_reward(bdrs_rewards, eos_mask, index, epsilon, remove_std)
    scores = episode_advantages + step_advantage_w * step_advantages
    return scores, scores, {
        "episode_advantages": episode_advantages,
        "step_advantages": step_advantages,
    }


def _episode_norm_reward(token_level_rewards, eos_mask, index, epsilon, remove_std):
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)
    id2vals = {}
    id2list = {}
    with torch.no_grad():
        for i in range(scores.shape[0]):
            id2list.setdefault(index[i], []).append(scores[i])
        for k, vals in id2list.items():
            t = torch.tensor(vals)
            mean = torch.mean(t)
            std = torch.std(t)
            id2vals[k] = (mean, std)
        for i in range(scores.shape[0]):
            mean, std = id2vals[index[i]]
            if remove_std:
                scores[i] = scores[i] - mean
            else:
                scores[i] = (scores[i] - mean) / (std + epsilon)
        ep_adv = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask
    return ep_adv


def _step_group_norm_reward(step_rewards, eos_mask, index, epsilon, remove_std):
    response_length = eos_mask.shape[-1]
    scores = step_rewards.clone()
    id2vals = {}
    id2list = {}
    with torch.no_grad():
        for i in range(scores.shape[0]):
            id2list.setdefault(index[i], []).append(scores[i])
        for k, vals in id2list.items():
            t = torch.tensor(vals)
            mean = torch.mean(t)
            std = torch.std(t)
            id2vals[k] = (mean, std)
        for i in range(scores.shape[0]):
            mean, std = id2vals[index[i]]
            if remove_std:
                scores[i] = scores[i] - mean
            else:
                scores[i] = (scores[i] - mean) / (std + epsilon)
        step_adv = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask
    return step_adv


