from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from mjlab.envs.mdp import rewards as _mdp_rewards
from mjlab.managers.scene_entity_config import SceneEntityCfg

from .._common import action_term
from ..commands.motion_tracking import MotionTrackingCommand

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def _cmd(env: ManagerBasedRlEnv, command_name: str) -> MotionTrackingCommand:
    return cast(MotionTrackingCommand, env.command_manager.get_term(command_name))


def _side_idx(command: MotionTrackingCommand, side: str) -> int:
    return command._side_list.index(side)


def _mean_over_sides(env, command_name: str, side: str | None, fn) -> torch.Tensor:
    """Mean of fn over the motion's hand sides (right-only motion -> right only), or one side
    when `side` is given (rewards.split_sides binds it per hand)."""
    sides = [side] if side is not None else _cmd(env, command_name)._side_list
    return torch.stack([fn(s) for s in sides], dim=-1).mean(dim=-1)


class BaseRewards:
    """Proprioceptive regularizers: action and joint-state terms, no reference motion."""

    @staticmethod
    def action_rate(env: ManagerBasedRlEnv) -> torch.Tensor:
        return _mdp_rewards.action_rate_l2(env)

    @staticmethod
    def joint_limits(
        env: ManagerBasedRlEnv,
        asset_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        return _mdp_rewards.joint_pos_limits(env, asset_cfg=asset_cfg)

    # ── Action magnitude (positive values: use NEGATIVE weights) ─────────────

    @staticmethod
    def action_penalty(env: "ManagerBasedRlEnv", side: str | None = None) -> torch.Tensor:
        """Mean squared action over every action dim, or one hand's dims."""
        actions = env.action_manager.action  # (B, action_dim)
        if side is not None:
            actions = actions[:, BaseRewards._side_action_mask(env, side)]
        return torch.mean(actions**2, dim=-1)

    @staticmethod
    def action_norm_sq(env: "ManagerBasedRlEnv") -> torch.Tensor:
        """CHORD action_norm: SUM of squared raw actions (action_penalty is the mean); whole-body,
        not side-split. Use a NEGATIVE weight."""
        return torch.sum(env.action_manager.action ** 2, dim=-1)

    # Action dims are grouped by type ([wrist_trans R,L | wrist_rot R,L | fingers R,L]), so a
    # side is picked by joint-name prefix, not by halving the vector.

    @staticmethod
    def _side_action_mask(env: "ManagerBasedRlEnv", side: str) -> torch.Tensor:
        act_term = action_term(env)
        key = f"_side_action_mask_{side}"
        mask = getattr(act_term, key, None)
        if mask is None:
            names = act_term._entity.joint_names
            prefixes = ("R_", "right_") if side == "right" else ("L_", "left_")
            mask = torch.tensor(
                [names[int(i)].startswith(prefixes) for i in act_term._all_joint_ids],
                dtype=torch.bool, device=env.device,
            )
            if not bool(mask.any()):
                raise ValueError(f"no {side}-side joints among the action joints {list(names)}")
            setattr(act_term, key, mask)
        return mask

    # ── Joint power (|torque * velocity|) ───────────────────────────────────

    @staticmethod
    def _joint_power_exp(
        env: ManagerBasedRlEnv,
        command_name: str,
        group_ids: str,
        scale: float,
        side: str | None,
    ) -> torch.Tensor:
        """exp(-scale * sum(|torque * velocity|)) over one action-term joint group."""
        command = _cmd(env, command_name)
        ids = getattr(action_term(env), group_ids)
        n_per_side = len(ids) // len(command._side_list)

        def one(s: str) -> torch.Tensor:
            si = _side_idx(command, s)
            side_ids = ids[si * n_per_side : (si + 1) * n_per_side]
            torque = command.robot.data.qfrc_actuator[:, side_ids]
            vel = command.robot.data.joint_vel[:, side_ids]
            power = torch.sum(torch.abs(torque * vel), dim=-1)
            return torch.exp(-scale * power)

        return _mean_over_sides(env, command_name, side, one)

    @staticmethod
    def finger_power_penalty(
        env: ManagerBasedRlEnv,
        command_name: str,
        scale: float,
        side: str | None = None,
    ) -> torch.Tensor:
        return BaseRewards._joint_power_exp(
            env, command_name, "_finger_ids", scale, side
        )

    @staticmethod
    def wrist_power_penalty(
        env: ManagerBasedRlEnv,
        command_name: str,
        scale: float,
        side: str | None = None,
    ) -> torch.Tensor:
        return BaseRewards._joint_power_exp(
            env, command_name, "_wrist_ids", scale, side
        )

