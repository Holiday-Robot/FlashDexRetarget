"""Learning agents: ``runner.agent.class_name`` picks the implementation (e.g. agents.flashsac.FlashSACAgent)."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from .base import Agent

if TYPE_CHECKING:
    from omegaconf import DictConfig

    from envs.vec_env import FlatObsVecEnv

__all__ = ["Agent", "build_agent"]


def build_agent(cfg: DictConfig, env: FlatObsVecEnv, env_info: dict[str, Any], command_name: str) -> Agent:
    """Import ``cfg.class_name`` ("module.Class") and build it from the rest of ``cfg``."""
    module_name, _, class_name = str(cfg.class_name).rpartition(".")
    cls = getattr(importlib.import_module(module_name), class_name)
    if not (isinstance(cls, type) and issubclass(cls, Agent)):
        raise TypeError(f"runner.agent.class_name={cfg.class_name!r} is not an agents.Agent")
    return cls.build(cfg, env, env_info, command_name)
