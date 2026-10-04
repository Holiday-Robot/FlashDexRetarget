from __future__ import annotations

from mjlab.rl import RslRlVecEnvWrapper
from omegaconf import DictConfig

from .off_policy_runner import OffPolicyRunner


def create_runner(
    runner_cfg: DictConfig,
    eval_cfg: DictConfig,
    env: RslRlVecEnvWrapper,
    eval_env: RslRlVecEnvWrapper | None = None,
    headless: bool = True,
) -> OffPolicyRunner:
    """Build the runner from ``cfg.runner`` + ``cfg.eval``; ``cfg.runner.agent`` picks the algorithm."""
    if eval_env is None:
        raise ValueError("the runner's eval needs the dedicated eval env from create_envs")
    return OffPolicyRunner(env, runner_cfg, eval_cfg, eval_env, headless=headless)
