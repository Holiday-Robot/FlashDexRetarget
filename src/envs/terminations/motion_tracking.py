from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

import torch
from mjlab.utils.lab_api.math import quat_error_magnitude

from ..commands.motion_tracking import MotionTrackingCommand
from ..rewards import side_twin_name
from .base import BaseTerminations

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def _cmd(env: ManagerBasedRlEnv, command_name: str) -> MotionTrackingCommand:
    return cast(MotionTrackingCommand, env.command_manager.get_term(command_name))


def side_fail(env: ManagerBasedRlEnv, name: str, mask: torch.Tensor) -> None:
    """Record which side tripped a termination term ((B, n_sides) bool) for the per-side
    termination penalty; terms that never record charge both sides."""
    store = getattr(env, "_side_fail", None)
    if store is None:
        store = env._side_fail = {}
    store[name] = mask.bool()


class MotionTrackingTerminations(BaseTerminations):
    """Motion-tracking terminations: pulls current refs from MotionTrackingCommand."""

    @staticmethod
    def tracking_fingertip_diverged(
        env: ManagerBasedRlEnv,
        command_name: str,
        threshold: float,
        grace_steps: int,
    ) -> torch.Tensor:
        """Terminate if ANY fingertip error exceeds threshold (any side).
        Skipped during first ``grace_steps`` of each episode."""
        command = _cmd(env, command_name)
        error = torch.norm(command.mano_tip_trans_w - command.robot_tip_trans_w, dim=-1)
        # grace_steps: the first control steps after a reset, while the hand settles from the
        # teleported start pose; the divergence terms stay off until then.
        live = (env.episode_length_buf >= grace_steps).unsqueeze(1)
        side_exceeded = torch.any(error > threshold, dim=-1) & live  # (B, n_sides)
        side_fail(env, "tracking_fingertip_diverged", side_exceeded)
        return torch.any(side_exceeded, dim=-1)

    @staticmethod
    def tracking_obj_trans_diverged(
        env: ManagerBasedRlEnv,
        command_name: str,
        threshold: float,
        grace_steps: int,
    ) -> torch.Tensor:
        """Terminate if object position error exceeds threshold (any side)."""
        command = _cmd(env, command_name)
        error = torch.norm(command.ref_obj_trans_w - command.sim_obj_trans_w, dim=-1)
        live = (env.episode_length_buf >= grace_steps).unsqueeze(1)
        side_exceeded = (error > threshold) & live  # (B, n_sides)
        side_fail(env, "tracking_obj_trans_diverged", side_exceeded)
        return torch.any(side_exceeded, dim=-1)

    @staticmethod
    def tracking_obj_rot_diverged(
        env: ManagerBasedRlEnv,
        command_name: str,
        threshold_deg: float,
        grace_steps: int,
    ) -> torch.Tensor:
        """Terminate if object rotation error exceeds threshold (any side)."""
        command = _cmd(env, command_name)
        error_rad = quat_error_magnitude(command.ref_obj_quat_w, command.sim_obj_quat_w)
        threshold_rad = threshold_deg * math.pi / 180.0
        live = (env.episode_length_buf >= grace_steps).unsqueeze(1)
        side_exceeded = (error_rad > threshold_rad) & live  # (B, n_sides)
        side_fail(env, "tracking_obj_rot_diverged", side_exceeded)
        return torch.any(side_exceeded, dim=-1)

    @staticmethod
    def object_dropped(
        env: ManagerBasedRlEnv,
        command_name: str,
        drop_margin: float,
        floor_z: float | None = None,
    ) -> torch.Tensor:
        command = _cmd(env, command_name)
        sim_z = command.sim_obj_trans_w[..., 2]  # (B, n_sides)
        ref_z = command.ref_obj_trans_w[..., 2]  # (B, n_sides)
        dropped = sim_z < (ref_z - drop_margin)
        if floor_z is not None:
            dropped = dropped | (sim_z < floor_z)
        side_fail(env, "object_dropped", dropped)
        return torch.any(dropped, dim=-1)

    @staticmethod
    def motion_clip_ended(
        env: ManagerBasedRlEnv,
        command_name: str,
    ) -> torch.Tensor:
        command = _cmd(env, command_name)
        return command.motion_steps >= (command.motion_num_frames - 1)

    @staticmethod
    def task_reward_early_reset(
        env: ManagerBasedRlEnv,
        task_term: str | list[str],
        threshold: float,
        interval: int,
        task_term_mode: str = "any",
    ) -> torch.Tensor:
        if threshold <= 0.0:
            return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

        rm = env.reward_manager
        cmd = env.command_manager.get_term("motion")
        
        if getattr(env, "_task_rew_idx", None) is None:
            names = list(rm.active_terms)
            terms: list[str] = []
            for t in [task_term] if isinstance(task_term, str) else list(task_term):
                twins = [side_twin_name(t, s[0]) for s in cmd._side_list]
                if t not in names and all(tw in names for tw in twins):
                    terms += twins
                    task_term_mode = "mean"
                    print(f"[task_reward_early_reset] {t} -> mean of {twins}", flush=True)
                elif t not in names:
                    raise KeyError(
                        f"task_reward_early_reset: reward term {t!r} not found "
                        f"in active reward terms {names}"
                    )
                else:
                    terms.append(t)
            idx = [names.index(t) for t in terms]
            env._task_rew_idx = idx
            env._task_rew_terms = terms
            env._task_rew_mode = task_term_mode
            env._task_rew_weight = torch.tensor(
                [float(rm._term_cfgs[i].weight) or 1.0 for i in idx], device=env.device
            )
            env._cum_task_rew = torch.zeros(
                env.num_envs, len(idx), device=env.device
            )

        idx: list[int] = env._task_rew_idx
        terms = env._task_rew_terms
        task_term_mode = env._task_rew_mode
        cum: torch.Tensor = env._cum_task_rew  # (B, n_terms)
        ep_len = env.episode_length_buf

        fresh = ep_len <= 1
        cum[fresh] = 0.0
        prev_task = rm._step_reward[:, idx] / env._task_rew_weight
        ungated = getattr(cmd, "_pcd_ungated", None)
        if ungated:
            sides = [s for s in cmd._side_list if s in ungated]
            mean_ungated = torch.stack([ungated[s] for s in sides], dim=-1).mean(dim=-1)

            def _ungated(t: str, c: torch.Tensor) -> torch.Tensor:
                if not t.endswith(("obj_pcd_match_error_exp", "obj_keypoint_match_error_exp")):
                    return c
                own = [s for s in sides if t.startswith(f"tracking_{s[0]}_")]  # a per-side twin
                return ungated[own[0]] if own else mean_ungated

            prev_task = torch.stack(
                [_ungated(t, c) for t, c in zip(terms, prev_task.unbind(dim=1))], dim=1
            )
        cum[~fresh] += prev_task[~fresh]

        k = torch.clamp((ep_len - 1) // interval, min=0) * interval
        per_term = cum < (threshold * k.to(cum.dtype)).unsqueeze(1)  # (B, n_terms)
        if task_term_mode == "mean":
            fired = cum.mean(dim=1) < (threshold * k.to(cum.dtype))
            per_term = torch.where(per_term.any(dim=1, keepdim=True), per_term, fired.unsqueeze(1))
            side_fail(env, "task_reward_early_reset", per_term & fired.unsqueeze(1))
            return fired
        side_fail(env, "task_reward_early_reset", per_term)
        return per_term.any(dim=1)
