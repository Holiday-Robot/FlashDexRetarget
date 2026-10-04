# [isaac-vendor] mjlab.rl: the VecEnv wrapper plus the PPO runner (lazy: rsl_rl is optional
# in flashSAC-only isaac envs).

from __future__ import annotations

from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper as RslRlVecEnvWrapper


def __getattr__(name: str):
    if name == "MjlabOnPolicyRunner":
        from mjlab.rl.runner import MjlabOnPolicyRunner

        return MjlabOnPolicyRunner
    raise AttributeError(name)
