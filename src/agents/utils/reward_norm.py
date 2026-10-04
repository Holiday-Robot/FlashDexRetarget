"""Per-head return-based reward normalization (one upstream RewardNormalizer per reward head)."""

from __future__ import annotations

import torch
from rsl_rl_flashsac.utils import RewardNormalizer


class MultiHeadRewardNormalizer:
    """Rewards (B, H): head h is scaled by its own return statistics; episode ends are shared."""

    def __init__(self, num_heads: int, gamma: float, G_max: float, device: torch.device | str) -> None:
        self.num_heads = num_heads
        self.device = device
        self.heads = [RewardNormalizer(gamma=gamma, G_max=G_max, device=device) for _ in range(num_heads)]

    def update_reward_stats(self, reward: torch.Tensor, terminated: torch.Tensor, truncated: torch.Tensor) -> None:
        reward = reward.reshape(reward.shape[0], self.num_heads)
        for h, head in enumerate(self.heads):
            head.update_reward_stats(reward=reward[:, h], terminated=terminated, truncated=truncated)

    def normalize_rewards(self, rewards: torch.Tensor) -> torch.Tensor:
        rewards_bh = rewards.reshape(rewards.shape[0], self.num_heads)
        cols = [head.normalize_rewards(rewards_bh[:, h]) for h, head in enumerate(self.heads)]
        return torch.stack(cols, dim=-1).reshape(rewards.shape)

    def state_dict(self) -> dict[str, torch.Tensor]:
        """flash_rl layout: G_r (B, H), G_r_max / G_rms_mean / G_rms_var (H,), one shared count."""
        states = [head.state_dict() for head in self.heads]
        return {
            "G_r": torch.stack([s["G_r"] for s in states], dim=-1),
            "G_r_max": torch.cat([s["G_r_max"] for s in states]),
            "G_rms_mean": torch.cat([s["G_rms_mean"] for s in states]),
            "G_rms_var": torch.cat([s["G_rms_var"] for s in states]),
            "G_rms_count": states[0]["G_rms_count"],
        }

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        g_r = state["G_r"].reshape(-1, self.num_heads)
        for h, head in enumerate(self.heads):
            head.load_state_dict(
                {
                    "G_r": g_r[:, h].clone(),
                    "G_r_max": state["G_r_max"].reshape(self.num_heads)[h : h + 1].clone(),
                    "G_rms_mean": state["G_rms_mean"].reshape(self.num_heads)[h : h + 1].clone(),
                    "G_rms_var": state["G_rms_var"].reshape(self.num_heads)[h : h + 1].clone(),
                    "G_rms_count": state["G_rms_count"].clone(),
                }
            )
