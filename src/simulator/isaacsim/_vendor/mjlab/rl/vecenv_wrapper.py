# [isaac-vendor] mjlab/rl/vecenv_wrapper.py minus the mjlab.utils.spaces action clipping:
# TensorDict obs for the rsl_rl PPO runner, and the 4-tuple step() the off-policy runner uses.

from __future__ import annotations

from typing import Any

import torch
from tensordict import TensorDict


class RslRlVecEnvWrapper:
    def __init__(self, env: Any) -> None:
        self.env = env
        self.num_envs = self.unwrapped.num_envs
        self.device = torch.device(self.unwrapped.device)
        self.max_episode_length = self.unwrapped.max_episode_length
        self.num_actions = self.unwrapped.action_manager.total_action_dim
        # Reset at the start since rsl_rl does not call reset (same as upstream).
        self.env.reset()

    @property
    def cfg(self) -> Any:
        return self.unwrapped.cfg

    @property
    def render_mode(self) -> str | None:
        return getattr(self.env, "render_mode", None)

    @property
    def observation_space(self) -> Any:
        return self.env.observation_space

    @property
    def action_space(self) -> Any:
        return self.env.action_space

    @classmethod
    def class_name(cls) -> str:
        return cls.__name__

    @property
    def unwrapped(self) -> Any:
        return self.env.unwrapped

    @property
    def episode_length_buf(self) -> torch.Tensor:
        return self.unwrapped.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor) -> None:
        self.unwrapped.episode_length_buf = value

    def seed(self, seed: int = -1) -> int:
        return self.unwrapped.seed(seed)

    def get_observations(self) -> TensorDict:
        obs_dict = self.unwrapped.observation_manager.compute()
        return TensorDict(obs_dict, batch_size=[self.num_envs])

    def reset(self) -> tuple[TensorDict, dict]:
        obs_dict, extras = self.env.reset()
        return TensorDict(obs_dict, batch_size=[self.num_envs]), extras

    def step(self, actions: torch.Tensor):
        obs_dict, rew, terminated, truncated, extras = self.env.step(actions)
        dones = (terminated | truncated).to(dtype=torch.long)
        if not getattr(self.cfg, "is_finite_horizon", False):
            extras["time_outs"] = truncated
        return TensorDict(obs_dict, batch_size=[self.num_envs]), rew, dones, extras

    def close(self) -> None:
        self.env.close()
