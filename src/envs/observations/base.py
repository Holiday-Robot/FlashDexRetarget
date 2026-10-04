from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_apply_inverse

from .._common import action_term
from ..commands.motion_tracking import MotionTrackingCommand

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def _rotate_vec_world_to_wrist(
    vec_w: torch.Tensor, command: MotionTrackingCommand
) -> torch.Tensor:
    """Rotate world-frame 3-vectors into per-side wrist frame."""
    wrist_quat = command.robot_wrist_quat_w
    if vec_w.dim() == wrist_quat.dim():
        return quat_apply_inverse(wrist_quat, vec_w)
    extra = vec_w.dim() - wrist_quat.dim()
    for _ in range(extra):
        wrist_quat = wrist_quat.unsqueeze(2)
    wrist_quat = wrist_quat.expand(*vec_w.shape[:-1], 4)
    return quat_apply_inverse(wrist_quat, vec_w)

def _log_norm_force(force: torch.Tensor) -> torch.Tensor:
    """Log-norm transform: input (..., 3) → output (..., 4) = [unit_dir * log(|f|+1), log(|f|+1)]."""
    norm = force.norm(dim=-1, keepdim=True)
    unit = force / (norm + 1e-6)
    log_mag = torch.log(norm + 1)
    log_xyz = unit * log_mag
    return torch.cat([log_xyz, log_mag], dim=-1)

def _rotate_force_world_to_wrist(
    force: torch.Tensor, command: MotionTrackingCommand, side: str
) -> torch.Tensor:
    """Rotate per-finger world-frame force (B, n_primaries, 3) into wrist frame."""
    si = command._side_list.index(side)
    wrist_quat = command.robot_wrist_quat_w[:, si : si + 1]
    wrist_quat = wrist_quat.expand(force.shape[0], force.shape[1], 4)
    return quat_apply_inverse(wrist_quat, force)


def _sides(env, command_name: str, side: str | None) -> list[str]:
    """[side], or every hand side of the motion command."""
    if side is not None:
        return [side]
    return list(env.command_manager.get_term(command_name)._side_list)


class BaseObs:
    """Embodiment-independent obs terms (proprio, last action, contact sensors)."""

    @staticmethod
    def robot_joint_pos(
        env: ManagerBasedRlEnv,
        asset_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        """Robot finger joint positions. Shape: (B, n_dofs)."""
        robot: Entity = env.scene[asset_cfg.name]
        return robot.data.joint_pos[:, asset_cfg.joint_ids]

    @staticmethod
    def robot_joint_cos_sin(
        env: ManagerBasedRlEnv,
        asset_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        """Cos and sin of finger joint positions. Shape: (B, 2*n_dofs)."""
        robot: Entity = env.scene[asset_cfg.name]
        q = robot.data.joint_pos[:, asset_cfg.joint_ids]
        return torch.cat([torch.cos(q), torch.sin(q)], dim=-1)

    @staticmethod
    def wrist_state_w(
        env: ManagerBasedRlEnv,
        asset_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        """Wrist state with zeroed pos + quat + lin_vel + ang_vel. Shape: (B, 13)."""
        robot: Entity = env.scene[asset_cfg.name]
        return torch.cat(
            [
                torch.zeros_like(robot.data.root_link_pos_w),
                robot.data.root_link_quat_w,
                robot.data.root_link_lin_vel_w,
                robot.data.root_link_ang_vel_w,
            ],
            dim=-1,
        )

    @staticmethod
    def wrist_pose_w(
        env: ManagerBasedRlEnv,
        asset_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        """Wrist (root link) pose: workspace-frame pos (3) + world quat (4). Shape: (B, 7)."""
        robot: Entity = env.scene[asset_cfg.name]
        pos = robot.data.root_link_pos_w - env.scene.env_origins
        return torch.cat([pos, robot.data.root_link_quat_w], dim=-1)

    @staticmethod
    def robot_joint_vel(
        env: ManagerBasedRlEnv,
        asset_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        """Robot joint velocities. Shape: (B, n_dofs)."""
        robot: Entity = env.scene[asset_cfg.name]
        return robot.data.joint_vel[:, asset_cfg.joint_ids]

    @staticmethod
    def dof_target_pos(
        env: ManagerBasedRlEnv,
        asset_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        """PD position-target error: target - current joint pos. Shape: (B, n_dofs)."""
        robot: Entity = env.scene[asset_cfg.name]
        target = robot.data.joint_pos_target[:, asset_cfg.joint_ids]
        current = robot.data.joint_pos[:, asset_cfg.joint_ids]
        return target - current

    @staticmethod
    def last_action(
        env: ManagerBasedRlEnv,
    ) -> torch.Tensor:
        """First n_dofs dims of the raw policy action. Shape: (B, n_dofs)."""
        return env.action_manager.action[:, : action_term(env).n_dofs]

    @staticmethod
    def fabric_joint_vel(
        env: ManagerBasedRlEnv,
    ) -> torch.Tensor:
        """Fabric velocity qd_f. Shape: (B, n_dofs); zeros when the fabric is off."""
        term = action_term(env)
        qd = getattr(term, "fabric_joint_vel", None)
        return torch.zeros_like(term.raw_action) if qd is None else qd

    @staticmethod
    def robot_joint_pos_unscaled(
        env: ManagerBasedRlEnv,
        asset_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        """Joint positions mapped onto [-1, 1] by the joint limits (DexMachina ``unscale``:
        (2q - hi - lo) / (hi - lo + 1e-5)); limits come from the action term if it exports them. Shape: (B, n_dofs)."""
        robot: Entity = env.scene[asset_cfg.name]
        q = robot.data.joint_pos[:, asset_cfg.joint_ids]
        lim = getattr(action_term(env), "entity_joint_limits", None)
        if lim is None:
            lim = robot.data.soft_joint_pos_limits[0]
        lim = lim[asset_cfg.joint_ids]  # (n, 2)
        lo, hi = lim[:, 0], lim[:, 1]
        return (2.0 * q - hi - lo) / (hi - lo + 1e-5)

    # ── Wrist / fingertips / bodies (per side, from the command's robot sites) ──

    @staticmethod
    def robot_wrist_pose_w(
        env: ManagerBasedRlEnv,
        command_name: str,
    ) -> torch.Tensor:
        """Per-side robot palm-site pose: workspace-frame pos (3) + world quat (4).
        Side-looping replacement for the root-body ``wrist_pose_w``. Shape: (B, S*7)."""
        command = cast(
            MotionTrackingCommand, env.command_manager.get_term(command_name)
        )
        pos = command.robot_wrist_trans_w - env.scene.env_origins.unsqueeze(1)
        feat = torch.cat([pos, command.robot_wrist_quat_w], dim=-1)  # (B, S, 7)
        return feat.reshape(feat.shape[0], -1)

    @staticmethod
    def robot_wrist_lin_vel_w(
        env: ManagerBasedRlEnv,
        command_name: str,
    ) -> torch.Tensor:
        """Robot wrist (palm site) linear velocity, world frame (own state; reference is
        ref_mano_wrist_lin_vel_w); translation-invariant, no env-origin handling. Shape: (B, n_sides*3)."""
        command = cast(MotionTrackingCommand, env.command_manager.get_term(command_name))
        v = command.robot_wrist_lin_vel_w
        return v.reshape(v.shape[0], -1)

    @staticmethod
    def robot_wrist_ang_vel_w(
        env: ManagerBasedRlEnv,
        command_name: str,
    ) -> torch.Tensor:
        """Robot wrist (palm site) angular velocity, world frame; pairs with
        ref_mano_wrist_ang_vel_delta_w. Shape: (B, n_sides*3)."""
        command = cast(MotionTrackingCommand, env.command_manager.get_term(command_name))
        w = command.robot_wrist_ang_vel_w
        return w.reshape(w.shape[0], -1)

    @staticmethod
    def robot_wrist_lin_vel_wrist(
        env: ManagerBasedRlEnv,
        command_name: str,
    ) -> torch.Tensor:
        """CHORD ``wrist_velocity_b`` linear part: palm-site lin vel in the palm's own frame. Shape: (B, S*3)."""
        command = cast(MotionTrackingCommand, env.command_manager.get_term(command_name))
        v = _rotate_vec_world_to_wrist(command.robot_wrist_lin_vel_w, command)
        return v.reshape(v.shape[0], -1)

    @staticmethod
    def robot_wrist_ang_vel_wrist(
        env: ManagerBasedRlEnv,
        command_name: str,
    ) -> torch.Tensor:
        """CHORD ``wrist_velocity_b`` angular part: palm-site ang vel in the palm's own frame. Shape: (B, S*3)."""
        command = cast(MotionTrackingCommand, env.command_manager.get_term(command_name))
        w = _rotate_vec_world_to_wrist(command.robot_wrist_ang_vel_w, command)
        return w.reshape(w.shape[0], -1)

    @staticmethod
    def robot_tip_trans_w(
        env: ManagerBasedRlEnv,
        command_name: str,
    ) -> torch.Tensor:
        """Robot fingertip (5/side) positions relative to the wrist, WORLD axes (no wrist
        rotation); counterpart of robot_tip_trans_wrist. Shape: (B, n_sides*5*3)."""
        command = cast(MotionTrackingCommand, env.command_manager.get_term(command_name))
        tips = command.robot_tip_trans_w  # (B, n_sides, 5, 3) world
        wrist = command.robot_wrist_trans_w  # (B, n_sides, 3) world
        rel = tips - wrist.unsqueeze(2)
        return rel.reshape(rel.shape[0], -1)

    @staticmethod
    def robot_tip_trans_wrist(
        env: ManagerBasedRlEnv,
        command_name: str,
    ) -> torch.Tensor:
        command = cast(MotionTrackingCommand, env.command_manager.get_term(command_name))
        tips = command.robot_tip_trans_w  # (B, n_sides, 5, 3) world
        wrist = command.robot_wrist_trans_w  # (B, n_sides, 3) world
        rel = tips - wrist.unsqueeze(2)
        rel = _rotate_vec_world_to_wrist(rel, command)
        return rel.reshape(rel.shape[0], -1)

    @staticmethod
    def robot_tip_lin_vel_wrist(
        env: ManagerBasedRlEnv,
        command_name: str,
    ) -> torch.Tensor:
        """Robot fingertip (5/side) linear velocity in per-side wrist AXES only (wrist's own
        velocity NOT subtracted — not a true rotating-frame velocity). Shape: (B, n_sides*5*3)."""
        command = cast(MotionTrackingCommand, env.command_manager.get_term(command_name))
        v = _rotate_vec_world_to_wrist(command.robot_tip_lin_vel_w, command)
        return v.reshape(v.shape[0], -1)

    @staticmethod
    def _robot_keypoints_w(
        command: MotionTrackingCommand, si: int, side: str
    ) -> torch.Tensor:
        """DexMachina kpt set per side: 5 tip sites + palm + 12 finger bodies (= the 13
        collision links), world frame. Shape: (B, 18, 3)."""
        return torch.cat(
            [
                command.robot_tip_trans_w[:, si],  # (B, 5, 3)
                command.robot_wrist_trans_w[:, si : si + 1],  # (B, 1, 3) palm
                command.robot_all_joints_trans_w(side),  # (B, 12, 3)
            ],
            dim=1,
        )

    @staticmethod
    def robot_keypoint_trans_w(
        env: ManagerBasedRlEnv,
        command_name: str,
    ) -> torch.Tensor:
        """DexMachina ``kpt_pos``: the 18 keypoints per side (_robot_keypoints_w order) in the
        workspace frame (world - env origin), no wrist-relative shift. Shape: (B, S*18*3)."""
        command = cast(MotionTrackingCommand, env.command_manager.get_term(command_name))
        origins = env.scene.env_origins[:, None, :]
        parts = []
        for si, side in enumerate(command._side_list):
            pts = BaseObs._robot_keypoints_w(command, si, side) - origins
            parts.append(pts.reshape(pts.shape[0], -1))
        return torch.cat(parts, dim=-1)

    @staticmethod
    def robot_joint_pos_target(
        env: ManagerBasedRlEnv,
        asset_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        """CHORD ``processed_action``: the absolute PD joint target (tracked ref + residual). Shape: (B, n_dofs)."""
        robot = env.scene[asset_cfg.name]
        return robot.data.joint_pos_target[:, asset_cfg.joint_ids]

    # ── Contact sensors (per link, wrist frame) ──

    @staticmethod
    def robot_body_contact_wrench_wrist(
        env: ManagerBasedRlEnv,
        command_name: str,
    ) -> torch.Tensor:
        """Per-link contact WRENCH (log-norm force(4)+torque(4), no application point), WRIST
        frame, side-looping; the sensor needs ``torque`` in its fields. Shape: (B, S*13*8)."""
        command = cast(
            MotionTrackingCommand, env.command_manager.get_term(command_name)
        )
        parts = []
        for side in command._side_list:
            sensor: ContactSensor = env.scene[f"{side[0]}_alllink_contact"]
            force = sensor.data.force  # (B, 13, 3) world net force
            torque = sensor.data.torque  # (B, 13, 3) world net torque
            force_w = _log_norm_force(
                _rotate_force_world_to_wrist(force, command, side)
            )
            torque_w = _log_norm_force(
                _rotate_force_world_to_wrist(torque, command, side)
            )
            feat = torch.cat([force_w, torque_w], dim=-1)  # (B, 13, 8)
            parts.append(feat.reshape(feat.shape[0], -1))
        return torch.cat(parts, dim=-1)  # (B, S*13*8)

    @staticmethod
    def robot_body_contact_trans_wrist(
        env: ManagerBasedRlEnv,
        command_name: str,
    ) -> torch.Tensor:
        """Per-link deepest contact point relative to the palm, WRIST frame, side-looping;
        ZEROED when found==0 so stale points can't leak. Shape: (B, S*13*3)."""
        command = cast(
            MotionTrackingCommand, env.command_manager.get_term(command_name)
        )
        parts = []
        for si, side in enumerate(command._side_list):
            sensor: ContactSensor = env.scene[f"{side[0]}_alllink_contact_pos"]
            pos = sensor.data.pos  # (B, 13, 3) world contact point
            found = (sensor.data.found > 0).unsqueeze(-1).to(pos.dtype)  # (B, 13, 1)
            wrist = command.robot_wrist_trans_w[:, si].unsqueeze(1)  # (B, 1, 3)
            point_rel = (pos - wrist) * found  # zero out the point when no contact
            point_w = _rotate_force_world_to_wrist(point_rel, command, side)
            parts.append(point_w.reshape(point_w.shape[0], -1))
        return torch.cat(parts, dim=-1)  # (B, S*13*3)

    @staticmethod
    def tip_penetration(
        env: ManagerBasedRlEnv,
        command_name: str,
        side: str | None = None,
        sensor: str = "fingertip_penetration",
        dist_clamp_min: float = -0.02,
        dist_clamp_max: float = 0.05,
    ) -> torch.Tensor:
        """Per-finger signed penetration distance + found flag from ``{r,l}_<sensor>`` of `side` (every
        command side when None). Shape: (B, S*n_primaries*2)."""
        parts = []
        for s in _sides(env, command_name, side):
            data = env.scene[f"{s[0]}_{sensor}"].data
            dist = data.dist.clamp(dist_clamp_min, dist_clamp_max)
            found = (data.found > 0).to(dist.dtype)
            out = torch.stack([dist, found], dim=-1)
            parts.append(out.reshape(out.shape[0], -1))
        return torch.cat(parts, dim=-1)

    @staticmethod
    def contact_force_norm(
        env: ManagerBasedRlEnv,
        command_name: str,
        side: str | None = None,
        sensor: str = "alllink_contact",
        scale: float = 0.01,
    ) -> torch.Tensor:
        """Per-link contact-force norm * scale (DexMachina-style, frame-free) from ``{r,l}_<sensor>`` of
        `side` (every command side when None). Shape: (B, S*n_links)."""
        parts = []
        for s in _sides(env, command_name, side):
            force = env.scene[f"{s[0]}_{sensor}"].data.force  # (B, n_links, 3)
            parts.append((force.norm(dim=-1) * scale).reshape(force.shape[0], -1))
        return torch.cat(parts, dim=-1)

    @staticmethod
    def contact_force_wrist(
        env: ManagerBasedRlEnv,
        command_name: str,
        side: str | None = None,
        sensor: str = "fingertip_contact",
    ) -> torch.Tensor:
        """Log-norm contact force per sensor primary, wrist frame, from ``{r,l}_<sensor>`` of `side`
        (every command side when None). Shape: (B, S*n_primaries*4)."""
        command = cast(MotionTrackingCommand, env.command_manager.get_term(command_name))
        parts = []
        for s in _sides(env, command_name, side):
            force = _rotate_force_world_to_wrist(env.scene[f"{s[0]}_{sensor}"].data.force, command, s)
            log_force = _log_norm_force(force)
            parts.append(log_force.reshape(log_force.shape[0], -1))
        return torch.cat(parts, dim=-1)

    @staticmethod
    def contact_force_history_wrist(
        env: ManagerBasedRlEnv,
        command_name: str,
        history_len: int,
        side: str | None = None,
        sensor: str = "fingertip_contact",
    ) -> torch.Tensor:
        """Rolling history of contact_force_wrist, per side (`side` only, or every command side).
        Shape: (B, S*history_len*n_primaries*4)."""
        command = cast(MotionTrackingCommand, env.command_manager.get_term(command_name))
        parts = []
        for s in _sides(env, command_name, side):
            key = f"_contact_force_history_{s[0]}_{sensor}"
            force = _rotate_force_world_to_wrist(env.scene[f"{s[0]}_{sensor}"].data.force, command, s)
            current = _log_norm_force(force)
            if key not in env.extras:
                init = torch.zeros(env.num_envs, history_len, force.shape[1], 4, device=env.device, dtype=torch.float)
                init[..., 3] = 1.0
                env.extras[key] = init
            buf = env.extras[key]
            reset_mask = env.episode_length_buf == 0
            if reset_mask.any():
                buf[reset_mask] = 0.0
                buf[reset_mask, ..., 3] = 1.0
            buf = torch.cat([buf[:, 1:], current[:, None]], dim=1)
            env.extras[key] = buf
            parts.append(buf.reshape(env.num_envs, -1))
        return torch.cat(parts, dim=-1)

    @staticmethod
    def alllink_contact_dir_wrist(
        env: ManagerBasedRlEnv,
        command_name: str,
        min_force: float = 1e-3,
    ) -> torch.Tensor:
        """CHORD ``contact_position_direction_in_wrist`` direction half: per-link unit contact-force
        direction in the palm frame, ZERO when |f| < min_force. Shape: (B, S*n_links*3)."""
        command = cast(MotionTrackingCommand, env.command_manager.get_term(command_name))
        parts = []
        for side in command._side_list:
            sensor: ContactSensor = env.scene[f"{side[0]}_alllink_contact"]
            force = sensor.data.force  # (B, L, 3) world net force
            norm = force.norm(dim=-1, keepdim=True)
            unit = force / norm.clamp_min(1e-5) * (norm > min_force).to(force.dtype)
            unit = _rotate_force_world_to_wrist(unit, command, side)
            parts.append(unit.reshape(unit.shape[0], -1))
        return torch.cat(parts, dim=-1)
