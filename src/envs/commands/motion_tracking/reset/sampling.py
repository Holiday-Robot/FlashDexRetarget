"""MotionTrackingCommand mixin: per-env clip / frame state and the uniform start frame."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from ..motion_tracking_cfg import MotionTrackingCommandCfg


class SamplingMixin:
    def _init_sampling(self, cfg: MotionTrackingCommandCfg) -> None:
        self.motion_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.motion_ids = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )


    def _uniform_sampling(self, env_ids: torch.Tensor) -> None:
        tr = self.motion_ids[env_ids]
        T_m = self.motion_lib._motion_num_frames[tr].float()
        self.motion_steps[env_ids] = (
            torch.rand(len(env_ids), device=self.device) * 0.99 * T_m
        ).long()
