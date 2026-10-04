from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from .residual import ResidualAction, ResidualActionCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


@dataclass
class LpfCfg:
    """EMA low-pass on the policy output (DeXtreme, Handa et al., arXiv:2210.13702);
    per-group cutoff, alpha derived from the control step so it holds at any rate."""

    enable: bool = False
    # Hz per group; null = passthrough.
    cutoff_hz: dict[str, float | None] = field(
        default_factory=lambda: {"wrist_trans": 5.0, "wrist_rot": 5.0, "finger": None}
    )


def ema_alpha(cutoff_hz: float | None, dt: float) -> float:
    """First-order low-pass: alpha = 2*pi*fc*dt / (1 + 2*pi*fc*dt); None -> 1.0."""
    if cutoff_hz is None:
        return 1.0
    omega_dt = 2.0 * math.pi * cutoff_hz * dt
    return omega_dt / (1.0 + omega_dt)


class LpfResidualAction(ResidualAction):
    """ResidualAction with the clipped action EMA-filtered once per policy step."""

    def __init__(self, cfg: ResidualActionCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        lpf_cfg = LpfCfg(**cfg.lpf)
        self._lpf_alpha = self._per_joint(
            {
                group: ema_alpha(cutoff_hz, env.step_dt)
                for group, cutoff_hz in lpf_cfg.cutoff_hz.items()
            }
        )
        self._lpf_state = torch.zeros(
            self.num_envs, self._lpf_alpha.shape[0], device=self.device
        )
        self._lpf_pending = torch.ones(
            self.num_envs, dtype=torch.bool, device=self.device
        )

    def process_actions(self, actions: torch.Tensor) -> None:
        super().process_actions(actions)
        # First step after reset passes through (state seeded, no stale pull).
        pending = self._lpf_pending.unsqueeze(1)
        self._lpf_state = torch.where(pending, self._processed_actions, self._lpf_state)
        self._lpf_pending.fill_(False)
        self._lpf_state = (
            self._lpf_alpha * self._processed_actions
            + (1.0 - self._lpf_alpha) * self._lpf_state
        )
        self._processed_actions = self._lpf_state

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        super().reset(env_ids)
        self._lpf_pending[env_ids] = True
