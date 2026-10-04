"""Per-step eval rollout capture for the success archive (success_export.py); one flat
device->host copy per step, split back into fields at export time."""

from __future__ import annotations

from typing import Any

import torch


class RolloutRecord:
    """``[joint_pos | joint_vel | actions | root_pos | root_quat | obj_pos | obj_quat]`` per
    eval step; positions are env-local (env_origins subtracted)."""

    def __init__(
        self,
        command_name: str = "motion",
        enabled: bool = True,
    ) -> None:
        self.command_name = str(command_name)
        self.enabled = bool(enabled)
        self.rollout_steps = 1
        self._frames: list[torch.Tensor] = []
        self._roll_spec: list[tuple[str, int, tuple[int, ...]]] = []

    # ── lifecycle ────────────────────────────────────────────────────────────
    def on_eval_setup(self, env: Any) -> None:
        del env

    def on_start(self, env: Any) -> None:
        if not self.enabled:
            return
        self.cmd = env.command_manager.get_term(self.command_name)
        self.num_envs = int(env.num_envs)
        self._env_origins = env.scene.env_origins.detach().clone()
        self.num_sides = int(self.cmd.sim_obj_trans_w.shape[1])
        # Per-env clip length, as the other eval callbacks trim to.
        self.valid_steps_per_env = self.cmd.motion_lib._motion_num_frames[
            self.cmd.motion_ids
        ].to(torch.long)
        self.t = 0
        self._frames = []
        robot = self.cmd.robot
        n_act = int(env.action_manager.action.shape[-1])
        tail = () if self.num_sides == 1 else (self.num_sides,)
        self._roll_spec = [
            ("joint_pos", int(robot.data.joint_pos.shape[-1]), ()),
            ("joint_vel", int(robot.data.joint_vel.shape[-1]), ()),
            ("actions", n_act, ()),
            ("root_pos", 3, ()),
            ("root_quat", 4, ()),
            ("obj_pos", 3 * self.num_sides, tail + (3,)),
            ("obj_quat", 4 * self.num_sides, tail + (4,)),
        ]

    def on_step(self, env: Any) -> None:
        if not self.enabled:
            return
        self.t += 1
        cmd = self.cmd
        robot = cmd.robot
        origins = self._env_origins
        n = self.num_envs
        flat = torch.cat(
            [
                robot.data.joint_pos,
                robot.data.joint_vel,
                env.action_manager.action,
                robot.data.root_link_pos_w - origins,
                robot.data.root_link_quat_w,
                (cmd.sim_obj_trans_w - origins[:, None, :]).reshape(n, -1),
                cmd.sim_obj_quat_w.reshape(n, -1),
            ],
            dim=-1,
        )
        self._frames.append(flat.detach().to("cpu", torch.float32))

    # ── cross-batch reduction ────────────────────────────────────────────────
    def collect_state(self, active_mask: torch.Tensor | None = None) -> dict[str, Any]:
        if not self.enabled:
            return {"rollout": None, "valid_steps_per_env": torch.zeros(0)}
        sel = slice(None) if active_mask is None else active_mask.cpu()
        rollout = torch.stack(self._frames, dim=0)[:, sel] if self._frames else None
        return {
            "rollout": rollout,
            "valid_steps_per_env": self.valid_steps_per_env[
                slice(None) if active_mask is None else active_mask
            ].cpu(),
            "motion_ids": self.cmd.motion_ids[
                slice(None) if active_mask is None else active_mask
            ].cpu(),
        }

    def reduce_state(self, states: list[dict[str, Any]]) -> dict[str, float]:
        del states
        return {}

    def get_metrics(self) -> dict[str, float]:
        return {}

    # ── export helper ────────────────────────────────────────────────────────
    def rollout_arrays(
        self, states: list[dict[str, Any]], env_index: int
    ) -> dict[str, Any] | None:
        """Recorded state for one env of the concatenated sweep, trimmed to its clip."""
        offset = 0
        for s in states:
            n = int(s["valid_steps_per_env"].shape[0])
            if env_index < offset + n:
                if s["rollout"] is None:
                    return None
                row = env_index - offset
                roll = s["rollout"]
                n_frames = min(
                    int(s["valid_steps_per_env"][row].item()), int(roll.shape[0])
                )
                if n_frames <= 0:
                    return None
                clip = roll[:n_frames, row]
                out: dict[str, Any] = {}
                at = 0
                for name, width, tail in self._roll_spec:
                    chunk = clip[:, at : at + width]
                    out[name] = (
                        chunk.reshape(n_frames, *tail).numpy() if tail else chunk.numpy()
                    )
                    at += width
                out["motion_id"] = int(s["motion_ids"][row].item())
                return out
            offset += n
        raise IndexError(f"env_index {env_index} out of range for {offset} envs")
