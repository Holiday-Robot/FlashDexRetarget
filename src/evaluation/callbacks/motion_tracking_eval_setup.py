from __future__ import annotations

from typing import Any

import torch


class MotionTrackingEvalSetup:
    """Assigns one batch of trajectories to the eval envs: env ``i`` tracks motion
    ``batch_idx * k + i // num_per_motion``; ragged-batch overflow envs are masked off."""

    def __init__(
        self, command_name: str, num_per_motion: int, start_frame: int = 0
    ) -> None:
        self.command_name = str(command_name)
        self.num_per_motion = int(num_per_motion)
        self.start_frame = int(start_frame)
        if self.start_frame < 0:
            raise ValueError(
                f"start_frame must be >= 0, got {self.start_frame}"
            )
        self.rollout_steps = 1
        self.batch_idx = 0
        self.k: int | None = None
        self.num_motions: int | None = None
        self.active_mask: torch.Tensor | None = None

    def set_batch(self, batch_idx: int, k: int, num_motions: int) -> None:
        """Select which block of trajectories this eval pass scores."""
        self.batch_idx = int(batch_idx)
        self.k = int(k)
        self.num_motions = int(num_motions)

    def on_eval_setup(self, env: Any) -> None:
        cmd = env.command_manager.get_term(self.command_name)
        n = env.num_envs
        num_motions = (
            self.num_motions
            if self.num_motions is not None
            else int(cmd.motion_lib.num_trajectories)
        )
        k = self.k if self.k is not None else num_motions
        offset = self.batch_idx * k
        raw_id = (
            offset
            + torch.arange(n, device=env.device, dtype=torch.long)
            // self.num_per_motion
        )
        self.active_mask = raw_id < num_motions
        cmd.motion_ids[:] = torch.clamp(raw_id, max=num_motions - 1)
        # Per-batch rollout length: the longest active trajectory in THIS batch, so
        # short batches don't over-run to the global max length.
        active_ids = cmd.motion_ids[self.active_mask]
        max_frames = int(cmd.motion_lib._motion_num_frames[active_ids].max().item())
        if self.start_frame >= max_frames:
            raise ValueError(
                f"start_frame {self.start_frame} >= longest trajectory "
                f"length {max_frames} in batch {self.batch_idx}"
            )
        self.rollout_steps = max_frames - 1 - self.start_frame

    def on_start(self, env: Any) -> None:
        pass

    def on_step(self, env: Any) -> None:
        pass

    def get_metrics(self) -> dict[str, float]:
        return {}

    def on_end(self) -> None:
        pass
