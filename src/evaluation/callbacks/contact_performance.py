from __future__ import annotations

from typing import Any, cast

import torch

from envs.commands.motion_tracking import MotionTrackingCommand


class ContactPerformance:
    """Per-finger contact ref-distance / penetration / force / found / miss-streak."""

    def __init__(self, grace_steps: int, command_name: str) -> None:
        self.grace_steps = grace_steps
        self.command_name = command_name

    def on_start(self, env: Any) -> None:
        self.cmd = cast(
            MotionTrackingCommand, env.command_manager.get_term(self.command_name)
        )
        self.device = env.device
        self.num_envs = env.num_envs
        self.side_prefixes = tuple(s[0] for s in self.cmd._side_list)
        self.num_sides = len(self.side_prefixes)
        self.finger_names = tuple(self.cmd.finger_names)
        self.num_fingers = len(self.finger_names)
        self.max_motion_frames = int(
            self.cmd.motion_lib._motion_num_frames.max().item()
        )
        self.valid_steps_per_env = (
            self.cmd.motion_num_frames - self.cmd.motion_steps
        ).clamp(min=0)
        self.scored_env_mask = self.valid_steps_per_env > self.grace_steps
        if not torch.any(self.scored_env_mask):
            self.scored_env_mask = torch.ones(
                self.num_envs, device=self.device, dtype=torch.bool
            )
        self.rollout_steps = max(1, int(self.valid_steps_per_env.max().item()))
        self.eval_end = max(self.grace_steps + 1, self.rollout_steps)

        self.pen_sensors = {
            side: env.scene[f"{side[0]}_fingertip_penetration"]
            for side in self.cmd._side_list
        }
        self.force_sensors = {
            side: env.scene[f"{side[0]}_fingertip_contact"]
            for side in self.cmd._side_list
        }

        def _z() -> torch.Tensor:
            return torch.zeros(
                self.num_envs, self.num_sides, self.num_fingers, device=self.device
            )

        self.sum_ref_dist = _z()
        self.sum_pen = _z()
        self.sum_force = _z()
        self.count_ref = _z()
        self.count_contact = _z()
        self.miss_counter = _z()
        self.miss_max = _z()
        self.t = 0

    def on_step(self, env: Any) -> None:
        in_window = (
            (self.t < self.valid_steps_per_env)
            & (self.t >= self.grace_steps)
            & (self.t < self.eval_end)
        )
        self.t += 1
        if not torch.any(in_window):
            return

        cmd = self.cmd
        valid_mask = in_window[:, None].to(torch.float32)
        for si, side in enumerate(cmd._side_list):
            pen_sens = self.pen_sensors[side]
            force_sens = self.force_sensors[side]
            flag = cmd.ref_contact_flags[:, si] * valid_mask
            found = (pen_sens.data.found > 0).to(flag.dtype)
            gate = flag * found
            ref_dist = torch.norm(
                cmd.ref_contact_trans_w[:, si] - cmd.robot_tip_trans_w[:, si], dim=-1
            )
            pen = torch.clamp(-pen_sens.data.dist, min=0.0)
            force = torch.norm(force_sens.data.force, dim=-1)
            self.sum_ref_dist[:, si] += ref_dist * flag
            self.sum_pen[:, si] += pen * gate
            self.sum_force[:, si] += force * gate
            self.count_ref[:, si] += flag
            self.count_contact[:, si] += gate
            miss = flag * (1.0 - found)
            self.miss_counter[:, si] = (self.miss_counter[:, si] + 1.0) * miss
            self.miss_max[:, si] = torch.maximum(
                self.miss_max[:, si], self.miss_counter[:, si]
            )

    def collect_state(
        self, active_mask: torch.Tensor | None = None
    ) -> dict[str, Any]:
        """Per-env contact accumulators for the active envs, for cross-batch reduction.
        Raw ``valid_steps_per_env`` lets scored basis + fallback span the full concat."""
        sel = slice(None) if active_mask is None else active_mask
        return {
            "sum_ref_dist": self.sum_ref_dist[sel],
            "sum_pen": self.sum_pen[sel],
            "sum_force": self.sum_force[sel],
            "count_ref": self.count_ref[sel],
            "count_contact": self.count_contact[sel],
            "miss_max": self.miss_max[sel],
            "valid_steps_per_env": self.valid_steps_per_env[sel],
        }

    def reduce_state(self, states: list[dict[str, Any]]) -> dict[str, float]:
        """Per-finger contact metrics over the concatenation of per-batch states.
        Scored basis + fallback are recomputed over the concat (matches a single pass)."""
        sum_ref_dist = torch.cat([s["sum_ref_dist"] for s in states], dim=0)
        sum_pen = torch.cat([s["sum_pen"] for s in states], dim=0)
        sum_force = torch.cat([s["sum_force"] for s in states], dim=0)
        count_ref = torch.cat([s["count_ref"] for s in states], dim=0)
        count_contact = torch.cat([s["count_contact"] for s in states], dim=0)
        miss_max = torch.cat([s["miss_max"] for s in states], dim=0)
        valid_steps_per_env = torch.cat(
            [s["valid_steps_per_env"] for s in states], dim=0
        )
        num_envs = int(valid_steps_per_env.shape[0])

        scored = valid_steps_per_env > self.grace_steps
        if not torch.any(scored):
            scored = torch.ones(num_envs, device=self.device, dtype=torch.bool)
        cnt_ref_safe = count_ref.clamp(min=1.0)
        cnt_contact_safe = count_contact.clamp(min=1.0)
        avg_ref_dist = (sum_ref_dist / cnt_ref_safe)[scored].mean(dim=0)
        avg_found = (count_contact / cnt_ref_safe)[scored].mean(dim=0)
        avg_pen = (sum_pen / cnt_contact_safe)[scored].mean(dim=0)
        avg_force = (sum_force / cnt_contact_safe)[scored].mean(dim=0)
        avg_miss_max = miss_max[scored].mean(dim=0)

        out: dict[str, float] = {}
        for si, p in enumerate(self.side_prefixes):
            for fi, finger in enumerate(self.finger_names):
                out[f"contact_ref_dist_{p}_{finger}"] = float(
                    avg_ref_dist[si, fi].item()
                )
                out[f"contact_found_{p}_{finger}"] = float(avg_found[si, fi].item())
                out[f"contact_pen_{p}_{finger}"] = float(avg_pen[si, fi].item())
                out[f"contact_force_{p}_{finger}"] = float(avg_force[si, fi].item())
                out[f"contact_miss_max_{p}_{finger}"] = float(
                    avg_miss_max[si, fi].item()
                )
        out["contact_num_scored_envs"] = int(scored.sum().item())
        return out

    def get_metrics(self) -> dict[str, float]:
        return self.reduce_state([self.collect_state()])

    def on_end(self) -> None:
        m = self.get_metrics()
        print("-" * 72)
        print("Per-finger contact performance:")
        for si, p in enumerate(self.side_prefixes):
            for fi, finger in enumerate(self.finger_names):
                print(
                    f"  {p}_{finger:<6s}  ref_dist: {m[f'contact_ref_dist_{p}_{finger}']:.6f} m  "
                    f"pen: {m[f'contact_pen_{p}_{finger}']:.6f} m  "
                    f"force: {m[f'contact_force_{p}_{finger}']:.4f} N  "
                    f"found: {m[f'contact_found_{p}_{finger}']:.4f}"
                )
