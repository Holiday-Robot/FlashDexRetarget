from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.envs.mdp.terminations import time_out as _mjlab_time_out
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


class BaseTerminations:
    """Reference-free termination terms: physics sanity and the NaN guard."""

    # mjlab built-in: flag-only term that marks episode truncation; the
    # termination manager handles the rest via the ``time_out`` cfg field.
    time_out = _mjlab_time_out

    @staticmethod
    def velocity_diverged(
        env: ManagerBasedRlEnv,
        max_lin_vel: float,
        max_ang_vel: float,
        asset_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        """Terminate if root link velocity is unreasonably high."""
        hand: Entity = env.scene[asset_cfg.name]
        lin = torch.norm(hand.data.root_link_lin_vel_w, dim=-1)
        ang = torch.norm(hand.data.root_link_ang_vel_w, dim=-1)
        return (lin > max_lin_vel) | (ang > max_ang_vel)

    @staticmethod
    def joint_vel_sanity(
        env: ManagerBasedRlEnv,
        max_joint_vel: float,
        asset_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        """Terminate if any joint velocity norm exceeds threshold."""
        hand: Entity = env.scene[asset_cfg.name]
        return torch.norm(hand.data.joint_vel, dim=-1) > max_joint_vel

    @staticmethod
    def joint_vel_mean_sanity(
        env: ManagerBasedRlEnv,
        max_joint_vel_mean: float,
        asset_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        """Terminate if mean absolute joint velocity exceeds threshold.
        ManipTrans: ``torch.abs(current_dof_vel).mean(-1) > 200``."""
        hand: Entity = env.scene[asset_cfg.name]
        return hand.data.joint_vel.abs().mean(dim=-1) > max_joint_vel_mean

    @staticmethod
    def obj_lin_vel_sanity(
        env: ManagerBasedRlEnv,
        command_name: str,
        max_obj_lin_vel: float,
    ) -> torch.Tensor:
        """Terminate if any per-side object linear velocity exceeds threshold."""
        command = env.command_manager.get_term(command_name)
        return torch.any(
            torch.norm(command.sim_obj_lin_vel_w, dim=-1) > max_obj_lin_vel, dim=-1
        )

    @staticmethod
    def obj_ang_vel_sanity(
        env: ManagerBasedRlEnv,
        command_name: str,
        max_obj_ang_vel: float,
    ) -> torch.Tensor:
        """Terminate if any per-side object angular velocity exceeds threshold."""
        command = env.command_manager.get_term(command_name)
        return torch.any(
            torch.norm(command.sim_obj_ang_vel_w, dim=-1) > max_obj_ang_vel, dim=-1
        )

    @staticmethod
    def nan_guard(
        env: ManagerBasedRlEnv,
        command_name: str,
        asset_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        """Terminate envs whose sim state contains NaN/Inf — comparisons return False
        on NaN, so the velocity-cap ``*_sanity`` terms silently miss those envs."""
        command = env.command_manager.get_term(command_name)
        hand: Entity = env.scene[asset_cfg.name]
        bad_joint = torch.isnan(hand.data.joint_pos).any(dim=-1) | torch.isnan(
            hand.data.joint_vel
        ).any(dim=-1)
        obj_trans = command.sim_obj_trans_w
        obj_lin_vel = command.sim_obj_lin_vel_w
        bad_obj = torch.isnan(obj_trans).flatten(1).any(dim=-1) | torch.isnan(
            obj_lin_vel
        ).flatten(1).any(dim=-1)
        return bad_joint | bad_obj
