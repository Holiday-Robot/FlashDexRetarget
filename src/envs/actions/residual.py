from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import torch
from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
from mjlab.utils.lab_api.math import unscale_transform
from mjlab.utils.lab_api.string import resolve_matching_names_values

from ..commands.motion_tracking import MotionTrackingCommand

if TYPE_CHECKING:
    from mjlab.entity import Entity
    from mjlab.envs import ManagerBasedRlEnv


def _enabled(block: dict | None) -> bool:
    return bool(block) and bool(block.get("enable", False))


@dataclass(kw_only=True)
class ResidualActionCfg(ActionTermCfg):
    entity_name: str
    command_name: str

    # Groups {wrist_trans, wrist_rot, finger}; the action vector is laid out in this order.
    actuator_names: dict[str, tuple[str, ...]]
    # Per group: target = ref + a * scale, or a in [-1, 1] mapped onto the limits if null.
    residual_scale: dict[str, float | None]

    action_scale: float = 1.0
    action_offset: float = 0.0

    # Reference frame for the targets and the bc reward = motion_steps - ref_lag.
    ref_lag: int = 0

    # {enable, trans_margin, rot_margin}: wrist limits = demo wrist qpos range +- margin.
    ref_wrist_limits: dict | None = None

    # Output filters; when enabled, lpf.py / fabric.py subclasses replace ResidualAction.
    lpf: dict | None = None
    fabric: dict | None = None

    def build(self, env: ManagerBasedRlEnv) -> ResidualAction:
        from .fabric import FabricResidualAction
        from .lpf import LpfResidualAction

        layers = tuple(
            cls
            for cls, block in (
                (LpfResidualAction, self.lpf),
                (FabricResidualAction, self.fabric),
            )
            if _enabled(block)
        )
        if not layers:
            return ResidualAction(self, env)
        if len(layers) == 1:
            return layers[0](self, env)
        return type("LpfFabricResidualAction", layers, {})(self, env)


class ResidualAction(ActionTerm):
    """Per-group PD target: ref + a * scale, or a mapped onto the joint limits."""

    cfg: ResidualActionCfg
    _entity: Entity

    def __init__(self, cfg: ResidualActionCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self._command_name = cfg.command_name

        # ── Joint groups: action layout = [wrist_trans | wrist_rot | finger] ───────
        group_ids: dict[str, torch.Tensor] = {}
        joint_names: list[str] = []
        for group in ("wrist_trans", "wrist_rot", "finger"):
            ids, names = self._entity.find_joints_by_actuator_names(
                cfg.actuator_names[group]
            )
            group_ids[group] = torch.tensor(ids, device=self.device, dtype=torch.long)
            joint_names += names
        self._group_sizes = {group: len(ids) for group, ids in group_ids.items()}
        self._wrist_trans_ids = group_ids["wrist_trans"]
        self._wrist_rot_ids = group_ids["wrist_rot"]
        self._finger_ids = group_ids["finger"]
        self._wrist_ids = torch.cat([self._wrist_trans_ids, self._wrist_rot_ids])
        self._all_joint_ids = torch.cat([self._wrist_ids, self._finger_ids])

        # ── Joint limits (targets are clamped to these) ────────────────────────────
        limits = self._entity.data.soft_joint_pos_limits[0]
        self._lower = limits[self._all_joint_ids, 0]
        self._upper = limits[self._all_joint_ids, 1]
        if _enabled(cfg.ref_wrist_limits):
            self._apply_ref_wrist_limits(env, dict(cfg.ref_wrist_limits))

        # ── Residual vs absolute, per joint ────────────────────────────────────────
        group_scale = {group: cfg.residual_scale[group] for group in self._group_sizes}
        self._is_residual = self._per_joint(
            {group: scale is not None for group, scale in group_scale.items()}
        ).bool()
        self._residual_scale = self._per_joint(
            {group: scale or 0.0 for group, scale in group_scale.items()}
        )

        # ── Reference frame ────────────────────────────────────────────────────────
        self._ref_lag = int(cfg.ref_lag)
        self.ref_joint_pos_used: torch.Tensor | None = None

        # ── Action buffers + clip (mirrors JointAction) ────────────────────────────
        action_dim = len(self._all_joint_ids)
        self._raw_actions = torch.zeros(self.num_envs, action_dim, device=self.device)
        self._processed_actions = torch.zeros(
            self.num_envs, action_dim, device=self.device
        )
        self._clip = None
        if cfg.clip is not None:
            self._clip = torch.tensor(
                [[-float("inf"), float("inf")]], device=self.device
            ).repeat(self.num_envs, action_dim, 1)
            idx_list, _, val_list = resolve_matching_names_values(
                dict(cfg.clip), joint_names
            )
            self._clip[:, idx_list] = torch.tensor(
                val_list, device=self.device, dtype=torch.float32
            )

    def _per_joint(self, values: dict[str, float]) -> torch.Tensor:
        """Per-group values expanded to the action layout. Shape: (n_dofs,)."""
        return torch.cat(
            [
                torch.full((size,), float(values[group]))
                for group, size in self._group_sizes.items()
            ]
        ).to(self.device)

    def _apply_ref_wrist_limits(self, env: ManagerBasedRlEnv, limits_cfg: dict) -> None:
        command = cast(
            MotionTrackingCommand, env.command_manager.get_term(self._command_name)
        )
        demo_wrist_pos = command.motion_lib.robot_joint_pos[:, self._wrist_ids]
        n_trans, n_rot = self._group_sizes["wrist_trans"], self._group_sizes["wrist_rot"]
        margin = torch.cat(
            [
                torch.full((n_trans,), float(limits_cfg.get("trans_margin", 0.2))),
                torch.full((n_rot,), float(limits_cfg.get("rot_margin", 0.5))),
            ]
        ).to(self.device)
        n_wrist = len(self._wrist_ids)
        self._lower[:n_wrist] = demo_wrist_pos.min(dim=0).values - margin
        self._upper[:n_wrist] = demo_wrist_pos.max(dim=0).values + margin

    @property
    def entity_joint_limits(self) -> torch.Tensor:
        """Soft limits (n_entity_joints, 2) with the action joints' clamp limits."""
        lim = self._entity.data.soft_joint_pos_limits[0].clone()
        lim[self._all_joint_ids, 0] = self._lower
        lim[self._all_joint_ids, 1] = self._upper
        return lim

    @property
    def action_dim(self) -> int:
        return len(self._all_joint_ids)

    @property
    def n_dofs(self) -> int:
        return len(self._all_joint_ids)

    @property
    def raw_action(self) -> torch.Tensor:
        return self._raw_actions

    def process_actions(self, actions: torch.Tensor) -> None:
        """Once-per-policy-step scale -> offset -> clip. Mirrors JointAction."""
        self._raw_actions[:] = actions
        self._processed_actions = (
            self._raw_actions * self.cfg.action_scale + self.cfg.action_offset
        )
        if self._clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions,
                min=self._clip[:, :, 0],
                max=self._clip[:, :, 1],
            )

    def _ref_joint_pos(self) -> torch.Tensor:
        command = cast(
            MotionTrackingCommand,
            self._env.command_manager.get_term(self._command_name),
        )
        if self._ref_lag == 0:
            return command.ref_joint_pos
        ml = command.motion_lib
        t = (command.motion_steps - self._ref_lag).clamp(min=0)
        return ml.robot_joint_pos[ml.length_starts[command.motion_ids] + t]

    def _target_parts(self) -> tuple[torch.Tensor, torch.Tensor]:
        """target = ref_action + residual_action: residual joints (ref, action * scale),
        others (0, action mapped onto the limits)."""
        self.ref_joint_pos_used = self._ref_joint_pos()
        action = self._processed_actions
        ref_action = torch.where(
            self._is_residual, self.ref_joint_pos_used[:, self._all_joint_ids], 0.0
        )
        residual_action = torch.where(
            self._is_residual,
            action * self._residual_scale,
            unscale_transform(action, self._lower, self._upper),
        )
        return ref_action, residual_action

    def apply_actions(self) -> None:
        ref_action, residual_action = self._target_parts()
        target = torch.clamp(ref_action + residual_action, self._lower, self._upper)
        self._entity.set_joint_position_target(target, joint_ids=self._all_joint_ids)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        self._raw_actions[env_ids] = 0.0
