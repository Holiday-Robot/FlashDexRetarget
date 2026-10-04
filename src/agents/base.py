"""The agent interface: everything the runner and evaluate need from a learning algorithm."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from omegaconf import DictConfig

    from envs.vec_env import FlatObsVecEnv


class Agent(ABC):
    """Acts on the flat env obs, learns from the transitions it observes, and checkpoints itself.
    Actions leave in the env's action bounds."""

    @classmethod
    @abstractmethod
    def build(
        cls, cfg: DictConfig, env: FlatObsVecEnv, env_info: dict[str, Any], command_name: str
    ) -> Agent:
        """Construct from ``runner.agent``; may adjust ``cfg`` in place (the runner logs it after)."""

    @abstractmethod
    def act(self, obs: Any, deterministic: bool = False) -> np.ndarray:
        """Exploration actions, or the deterministic policy for eval."""

    @abstractmethod
    def observe(self, transition: Mapping[str, Any]) -> None:
        """Store one env step (observation, action, reward, terminated, truncated, next_observation)."""

    @abstractmethod
    def ready(self) -> bool:
        """Whether the agent has enough data to act from its policy and update."""

    @abstractmethod
    def update(self) -> dict[str, float]:
        """One learning step; returns its metrics."""

    @abstractmethod
    def save(self, path: str) -> None: ...

    @abstractmethod
    def load(self, path: str) -> None: ...

    def save_buffer(self, path: str) -> None:
        """Persist the experience a resume needs (off-policy replay); nothing by default."""

    def load_buffer(self, path: str) -> None:
        """Restore what save_buffer wrote; nothing by default."""
