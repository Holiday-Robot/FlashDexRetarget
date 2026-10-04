from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import torch
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import matrix_from_quat, quat_error_magnitude

from ..observations.motion_tracking import (
    _keypoints_per_env,
    _obj_keypoints,
    _obj_surface_keypoints,
)
from .._common import action_term
from .base import BaseRewards, _cmd, _mean_over_sides, _side_idx
from .chord import ChordRewards

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def _worst_k_mean(d: torch.Tensor, topk: int = 0, topk_frac: float = 1.0) -> torch.Tensor:
    """(B, P) per-point error -> (B,) mean of the worst `topk` points; topk=0 falls back to
    topk_frac (worst int(P * frac), >=1 = plain mean over the points)."""
    P = d.shape[-1]
    k = topk if topk > 0 else (P if topk_frac >= 1.0 else max(1, int(P * topk_frac)))
    return d.mean(dim=-1) if k >= P else d.topk(k, dim=-1).values.mean(dim=-1)


def _rel_pcd_dist(
    sp: torch.Tensor,
    sim_t: torch.Tensor, sim_q: torch.Tensor,
    ref_t: torch.Tensor, ref_q: torch.Tensor,
    si: int, oi: int,
) -> torch.Tensor:
    """Per-point distance between side si's surface points seen from the OTHER object's frame,
    ref vs sim. sp (B,P,3) object-local; poses (B,S,3)/(B,S,4)."""

    def in_other(t, q):
        R = matrix_from_quat(q)  # (B,S,3,3)
        world = t[:, si, None, :] + torch.einsum("bij,bpj->bpi", R[:, si], sp)
        return torch.einsum("bji,bpj->bpi", R[:, oi], world - t[:, oi, None, :])  # R_o^T (x - t_o)

    return (in_other(ref_t, ref_q) - in_other(sim_t, sim_q)).norm(dim=-1)  # (B,P)


def _per_finger(command, value: float | Mapping[str, float], device) -> torch.Tensor:
    """(F,) over command.finger_names from a scalar or a {finger: value} mapping."""
    if isinstance(value, Mapping):
        return torch.tensor([float(value[f]) for f in command.finger_names], device=device)
    return torch.full((len(command.finger_names),), float(value), device=device)


def _finger_weights(command, weights: Mapping[str, float] | None, device) -> torch.Tensor:
    """(F,) finger weights normalized to sum 1; equal weights when None."""
    w = _per_finger(command, 1.0 if weights is None else weights, device)
    return w / w.sum()


class MotionTrackingRewards(BaseRewards, ChordRewards):
    # side=None terms average over the motion's hands, so each stays in [0, 1] whatever S is;
    # a summed recipe is the same value at twice the weight.

    # ── MANO reference (tracking_ref_mano_*): demo MANO hand vs robot ──────────────────────

    @staticmethod
    def tracking_ref_mano_wrist_trans_error_exp(
        env: ManagerBasedRlEnv, command_name: str, scale: float, side: str | None = None
    ) -> torch.Tensor:
        """exp(-scale * ||mano_wrist_pos - robot_wrist_pos||); one side, or side mean."""
        command = _cmd(env, command_name)

        def one(s: str) -> torch.Tensor:
            si = _side_idx(command, s)
            error = torch.norm(
                command.mano_wrist_trans_w[:, si] - command.robot_wrist_trans_w[:, si],
                dim=-1,
            )
            return torch.exp(-scale * error)

        return _mean_over_sides(env, command_name, side, one)

    @staticmethod
    def tracking_ref_mano_wrist_rot_error_exp(
        env: ManagerBasedRlEnv, command_name: str, scale: float, side: str | None = None
    ) -> torch.Tensor:
        """exp(-scale * quat_angle(mano_wrist, robot_wrist)); one side, or side mean."""
        command = _cmd(env, command_name)

        def one(s: str) -> torch.Tensor:
            si = _side_idx(command, s)
            error = quat_error_magnitude(
                command.mano_wrist_quat_w[:, si], command.robot_wrist_quat_w[:, si]
            )
            return torch.exp(-scale * error)

        return _mean_over_sides(env, command_name, side, one)

    @staticmethod
    def tracking_ref_mano_wrist_lin_vel_error_exp(
        env: ManagerBasedRlEnv, command_name: str, scale: float, side: str | None = None
    ) -> torch.Tensor:
        """exp(-scale * mean(|mano_vel - robot_vel|)); one side, or side mean."""
        command = _cmd(env, command_name)

        def one(s: str) -> torch.Tensor:
            si = _side_idx(command, s)
            error = torch.mean(
                torch.abs(
                    command.mano_wrist_lin_vel_w[:, si]
                    - command.robot_wrist_lin_vel_w[:, si]
                ),
                dim=-1,
            )
            return torch.exp(-scale * error)

        return _mean_over_sides(env, command_name, side, one)

    @staticmethod
    def tracking_ref_mano_wrist_ang_vel_error_exp(
        env: ManagerBasedRlEnv, command_name: str, scale: float, side: str | None = None
    ) -> torch.Tensor:
        """exp(-scale * mean(|mano_ang_vel - robot_ang_vel|)); one side, or side mean."""
        command = _cmd(env, command_name)

        def one(s: str) -> torch.Tensor:
            si = _side_idx(command, s)
            error = torch.mean(
                torch.abs(
                    command.mano_wrist_ang_vel_w[:, si]
                    - command.robot_wrist_ang_vel_w[:, si]
                ),
                dim=-1,
            )
            return torch.exp(-scale * error)

        return _mean_over_sides(env, command_name, side, one)

    @staticmethod
    def tracking_ref_mano_tip_trans_error_exp(
        env: ManagerBasedRlEnv,
        command_name: str,
        scale: float | Mapping[str, float],
        finger_weights: Mapping[str, float] | None = None,
        side: str | None = None,
    ) -> torch.Tensor:
        """Finger-weighted mean of exp(-scale * ||mano tip - robot tip||); scale may be per finger."""
        command = _cmd(env, command_name)
        scale_f = _per_finger(command, scale, env.device)
        weight_f = _finger_weights(command, finger_weights, env.device)

        def one(s: str) -> torch.Tensor:
            si = _side_idx(command, s)
            error = torch.norm(
                command.mano_tip_trans_w[:, si] - command.robot_tip_trans_w[:, si], dim=-1
            )  # (B, F)
            return (torch.exp(-scale_f * error) * weight_f).sum(dim=-1)

        return _mean_over_sides(env, command_name, side, one)

    @staticmethod
    def _tips_sdf(env, command_name: str, side: str) -> tuple[torch.Tensor, torch.Tensor]:
        command = _cmd(env, command_name)
        tips = command.robot_tip_trans_w[:, _side_idx(command, side)]  # (B, 5, 3) live
        sdf, grad = command.sdf_query(tips, side)  # (B, 5) signed, (B, 5, 3)
        return sdf, grad / (grad.norm(dim=-1, keepdim=True) + 1e-6)

    @staticmethod
    def tracking_ref_mano_tip_distance_bound_exp(
        env: "ManagerBasedRlEnv",
        command_name: str,
        sigma: float = 0.01,
        side: str | None = None,
    ) -> torch.Tensor:
        """Tips-mean exp(-max(0, sdf - d_ref)/sigma): each policy tip at most as far from the
        surface as the ref tip was. One-sided, so a ref tip that is off the object gates itself out."""
        def _one(s: str) -> torch.Tensor:
            command = _cmd(env, command_name)
            sdf, _ = MotionTrackingRewards._tips_sdf(env, command_name, s)
            d_ref = command.mano_tips_distance[:, _side_idx(command, s)]  # (B, 5)
            excess = torch.clamp(sdf - d_ref, min=0.0)
            return torch.exp(-excess / sigma).mean(dim=-1)

        return _mean_over_sides(env, command_name, side, _one)

    @staticmethod
    def _ref_mano_level_trans_error_exp(
        env: ManagerBasedRlEnv,
        command_name: str,
        side: str,
        level: int,
        scale: float,
    ) -> torch.Tensor:
        """exp(-scale * mean(||mano_joint - robot_body||)) for level 1 or 2, one side."""
        command = _cmd(env, command_name)
        mano_trans = command.mano_level_trans_w(side, level)  # (B, 5, 3)
        robot_trans = command.robot_level_trans_w(side, level)  # (B, 5, 3)
        error = torch.norm(mano_trans - robot_trans, dim=-1)  # (B, 5)
        return torch.exp(-scale * error.mean(dim=-1))

    @staticmethod
    def tracking_ref_mano_level1_trans_error_exp(
        env: ManagerBasedRlEnv, command_name: str, scale: float, side: str | None = None
    ) -> torch.Tensor:
        # level1 joints (proximal); side mean unless a side is given
        return _mean_over_sides(
            env, command_name, side,
            lambda s: MotionTrackingRewards._ref_mano_level_trans_error_exp(
                env, command_name, side=s, level=1, scale=scale
            ),
        )

    @staticmethod
    def tracking_ref_mano_level2_trans_error_exp(
        env: ManagerBasedRlEnv, command_name: str, scale: float, side: str | None = None
    ) -> torch.Tensor:
        # level2 joints (xhand: distal; 3-phalanx hands: intermediate); side mean unless a side is given
        return _mean_over_sides(
            env, command_name, side,
            lambda s: MotionTrackingRewards._ref_mano_level_trans_error_exp(
                env, command_name, side=s, level=2, scale=scale
            ),
        )

    @staticmethod
    def tracking_ref_mano_level3_trans_error_exp(
        env: ManagerBasedRlEnv, command_name: str, scale: float, side: str | None = None
    ) -> torch.Tensor:
        # level3 (body_mapping.level3, 3-phalanx hands); side mean unless a side is given
        return _mean_over_sides(
            env, command_name, side,
            lambda s: MotionTrackingRewards._ref_mano_level_trans_error_exp(
                env, command_name, side=s, level=3, scale=scale
            ),
        )

    @staticmethod
    def tracking_ref_mano_joints_vel_error_exp(
        env: ManagerBasedRlEnv, command_name: str, scale: float, side: str | None = None
    ) -> torch.Tensor:
        """exp(-scale * mean(|mano_vel - robot_vel|)) for all 17 bodies; one side, or side mean."""
        command = _cmd(env, command_name)

        def one(s: str) -> torch.Tensor:
            si = _side_idx(command, s)
            body_delta = command.mano_all_joints_lin_vel_w(
                s
            ) - command.robot_all_joints_lin_vel_w(s)  # (B, 12, 3)
            tip_mano_vel = command.mano_tip_lin_vel_w[:, si]  # (B, 5, 3)
            tip_robot_vel = command.robot_tip_lin_vel_w[:, si]  # (B, 5, 3) actual fingertip sites
            tip_delta = tip_mano_vel - tip_robot_vel  # (B, 5, 3)
            all_delta = torch.cat([body_delta, tip_delta], dim=1)  # (B, 17, 3)
            return torch.exp(-scale * all_delta.abs().mean(dim=-1).mean(dim=-1))

        return _mean_over_sides(env, command_name, side, one)

    # ── Retargeted-robot reference (tracking_ref_robot_*): retargeted demo vs robot ────────

    @staticmethod
    def tracking_ref_robot_wrist_trans_error_exp(
        env: ManagerBasedRlEnv, command_name: str, scale: float, side: str | None = None
    ) -> torch.Tensor:
        """exp(-scale * ||retargeted-robot ref palm - robot palm||); one side, or side mean."""
        command = _cmd(env, command_name)

        def one(s: str) -> torch.Tensor:
            si = _side_idx(command, s)
            error = torch.norm(
                command.robot_ref_wrist_trans_w[:, si] - command.robot_wrist_trans_w[:, si],
                dim=-1,
            )
            return torch.exp(-scale * error)

        return _mean_over_sides(env, command_name, side, one)

    @staticmethod
    def tracking_ref_robot_wrist_rot_error_exp(
        env: ManagerBasedRlEnv, command_name: str, scale: float, side: str | None = None
    ) -> torch.Tensor:
        """exp(-scale * quat_angle(retargeted-robot ref palm, robot palm)); one side, or side mean."""
        command = _cmd(env, command_name)

        def one(s: str) -> torch.Tensor:
            si = _side_idx(command, s)
            error = quat_error_magnitude(
                command.robot_ref_wrist_quat_w[:, si], command.robot_wrist_quat_w[:, si]
            )
            return torch.exp(-scale * error)

        return _mean_over_sides(env, command_name, side, one)

    @staticmethod
    def tracking_ref_robot_tip_trans_error_exp(
        env: ManagerBasedRlEnv,
        command_name: str,
        scale: float | Mapping[str, float],
        finger_weights: Mapping[str, float] | None = None,
        side: str | None = None,
    ) -> torch.Tensor:
        """Finger-weighted mean of exp(-scale * ||retargeted-robot ref tip - robot tip||); scale may be per finger."""
        command = _cmd(env, command_name)
        scale_f = _per_finger(command, scale, env.device)
        weight_f = _finger_weights(command, finger_weights, env.device)

        def one(s: str) -> torch.Tensor:
            si = _side_idx(command, s)
            error = torch.norm(
                command.robot_ref_tip_trans_w[:, si] - command.robot_tip_trans_w[:, si], dim=-1
            )  # (B, F)
            return (torch.exp(-scale_f * error) * weight_f).sum(dim=-1)

        return _mean_over_sides(env, command_name, side, one)

    @staticmethod
    def tracking_ref_robot_keybody_trans_error_exp(
        env: "ManagerBasedRlEnv",
        command_name: str,
        scale: float,
        side: str | None = None,
    ) -> torch.Tensor:
        """Per side the mean over the 18 key bodies (5 tips + palm + 12 bodies) of
        exp(-scale*||body - retargeted-robot ref||), side mean."""
        command = _cmd(env, command_name)

        def one(s: str) -> torch.Tensor:
            si = _side_idx(command, s)
            cur = torch.cat(
                [
                    command.robot_tip_trans_w[:, si],
                    command.robot_wrist_trans_w[:, si : si + 1],
                    command.robot_all_joints_trans_w(s),
                ],
                dim=1,
            )  # (B, 18, 3)
            ref = torch.cat(
                [
                    command.robot_ref_tip_trans_w[:, si],
                    command.robot_ref_wrist_trans_w[:, si : si + 1],
                    command.robot_ref_all_joints_trans_w(s),
                ],
                dim=1,
            )
            return torch.exp(-scale * (cur - ref).norm(dim=-1)).mean(dim=-1)

        return _mean_over_sides(env, command_name, side, one)

    # ── Object tracking: ref object vs sim object ─────────────────────────────────

    @staticmethod
    def tracking_obj_trans_error_exp(
        env: ManagerBasedRlEnv, command_name: str, scale: float, side: str | None = None
    ) -> torch.Tensor:
        """exp(-scale * ||ref_obj_trans - sim_obj_trans||); one side, or side mean."""
        command = _cmd(env, command_name)

        def one(s: str) -> torch.Tensor:
            si = _side_idx(command, s)
            error = torch.norm(
                command.ref_obj_trans_w[:, si] - command.sim_obj_trans_w[:, si], dim=-1
            )
            return torch.exp(-scale * error)

        return _mean_over_sides(env, command_name, side, one)

    @staticmethod
    def tracking_obj_rot_error_exp(
        env: ManagerBasedRlEnv, command_name: str, scale: float, side: str | None = None
    ) -> torch.Tensor:
        """exp(-scale * quat_angle(ref_obj, sim_obj)); one side, or side mean."""
        command = _cmd(env, command_name)

        def one(s: str) -> torch.Tensor:
            si = _side_idx(command, s)
            error = quat_error_magnitude(
                command.ref_obj_quat_w[:, si], command.sim_obj_quat_w[:, si]
            )
            return torch.exp(-scale * error)

        return _mean_over_sides(env, command_name, side, one)

    @staticmethod
    def tracking_obj_lin_vel_error_exp(
        env: ManagerBasedRlEnv, command_name: str, scale: float, side: str | None = None
    ) -> torch.Tensor:
        """exp(-scale * mean(|ref_lin_vel - sim_lin_vel|)); one side, or side mean."""
        command = _cmd(env, command_name)

        def one(s: str) -> torch.Tensor:
            si = _side_idx(command, s)
            error = torch.mean(
                torch.abs(
                    command.ref_obj_lin_vel_w[:, si] - command.sim_obj_lin_vel_w[:, si]
                ),
                dim=-1,
            )
            return torch.exp(-scale * error)

        return _mean_over_sides(env, command_name, side, one)

    @staticmethod
    def tracking_obj_ang_vel_error_exp(
        env: ManagerBasedRlEnv, command_name: str, scale: float, side: str | None = None
    ) -> torch.Tensor:
        """exp(-scale * mean(|ref_ang_vel - sim_ang_vel|)); one side, or side mean."""
        command = _cmd(env, command_name)

        def one(s: str) -> torch.Tensor:
            si = _side_idx(command, s)
            error = torch.mean(
                torch.abs(
                    command.ref_obj_ang_vel_w[:, si] - command.sim_obj_ang_vel_w[:, si]
                ),
                dim=-1,
            )
            return torch.exp(-scale * error)

        return _mean_over_sides(env, command_name, side, one)

    # ── Fingertip contact ─────────────────────────────────────────────────────────

    @staticmethod
    def _contact_match(
        env: ManagerBasedRlEnv,
        command_name: str,
        sensor_name: str,
        side: str,
        finger: str,
        beta: float,
        gamma: float,
        tol: float,
    ) -> torch.Tensor:
        """reward = ref_flag · (exp(-β·approach_dist) + found·exp(-γ·max(-dist − tol, 0)))
        with approach_dist = max(SDF(robot_tip_site), 0); peak 2.0 at clean landing."""
        command = _cmd(env, command_name)
        si = _side_idx(command, side)
        fi = command.finger_names.index(finger)

        robot_pt = command.robot_tip_trans_w[:, si, fi]  # (B, 3)
        sdf, _ = command.sdf_query(robot_pt, side)  # (B,)
        approach_dist = torch.clamp(sdf, min=0.0)

        pen_sensor: ContactSensor = env.scene[sensor_name]
        found = (pen_sensor.data.found[:, fi] > 0).to(approach_dist.dtype)  # (B,) 0/1
        pen_dist = pen_sensor.data.dist[:, fi]  # (B,) signed, <0 = overlap
        excess = torch.clamp(-pen_dist - tol, min=0.0)  # (B,)

        shaping = torch.exp(-beta * approach_dist)  # (B,)
        bonus = found * torch.exp(-gamma * excess)  # (B,)

        flag = command.ref_contact_flags[:, si, fi]  # (B,) 0 or 1
        return flag * (shaping + bonus)

    @staticmethod
    def tracking_contact_match(
        env: ManagerBasedRlEnv,
        command_name: str,
        beta: float,
        gamma: float,
        tol: float,
        finger_weights: Mapping[str, float] | None = None,
        side: str | None = None,
    ) -> torch.Tensor:
        """Finger-weighted mean of the per-finger contact match (equal weights by default)."""
        command = _cmd(env, command_name)
        weight_f = _finger_weights(command, finger_weights, env.device)

        def one(s: str) -> torch.Tensor:
            per_finger = torch.stack(
                [
                    MotionTrackingRewards._contact_match(
                        env,
                        command_name,
                        sensor_name=f"{s[0]}_fingertip_penetration",
                        side=s,
                        finger=f,
                        beta=beta,
                        gamma=gamma,
                        tol=tol,
                    )
                    for f in command.finger_names
                ],
                dim=-1,
            )  # (B, F)
            return (per_finger * weight_f).sum(dim=-1)

        return _mean_over_sides(env, command_name, side, one)

    # ── Behavior cloning ───────────────────────────────────────────────────

    @staticmethod
    def _action_ref_joint_pos(env: "ManagerBasedRlEnv", command) -> torch.Tensor:
        """Reference qpos the last apply_actions targeted (actions.ref_lag aware); falls
        back to the current-counter reference for action terms that do not record it."""
        used = getattr(action_term(env), "ref_joint_pos_used", None)
        return command.ref_joint_pos if used is None else used

    @staticmethod
    def _action_joint_limits(env: "ManagerBasedRlEnv", command) -> torch.Tensor:
        """(n_joints, 2) limits the action term clamps to (ref-derived wrist rows when
        actions.ref_wrist_limits is on); model soft limits otherwise."""
        lim = getattr(action_term(env), "entity_joint_limits", None)
        return command.robot.data.soft_joint_pos_limits[0] if lim is None else lim

    @staticmethod
    def tracking_joint_pos_bc_exp(
        env: "ManagerBasedRlEnv",
        command_name: str,
        scale: float,
        side: str | None = None,
    ) -> torch.Tensor:
        """Behavior cloning over every action joint, or one hand's joints."""
        command = _cmd(env, command_name)
        act_term = action_term(env)
        ids = act_term._all_joint_ids
        if side is not None:
            ids = ids[MotionTrackingRewards._side_action_mask(env, side)]

        target = command.robot.data.joint_pos_target[:, ids]  # curr_targets
        ref = MotionTrackingRewards._action_ref_joint_pos(env, command)[:, ids]  # curr_res_qpos
        limits = MotionTrackingRewards._action_joint_limits(env, command)  # (n_joints, 2)
        dof_range = limits[ids, 1] - limits[ids, 0]  # (n_ids,)

        err = 0.5 * (target - ref).pow(2) / dof_range
        return torch.exp(-scale * err).mean(dim=-1)

    # ── Object pose tracking (multiplicative pos * rot) ────────────────────

    @staticmethod
    def tracking_obj_pose_match(
        env: "ManagerBasedRlEnv",
        command_name: str,
        pos_scale: float,
        rot_scale: float,
        side_reduce: str = "mean",
        side: str | None = None,
    ) -> torch.Tensor:
        """exp(-pos_scale*d)*exp(-rot_scale*angle) per side (one object per side), then mean
        (default) or product over sides; stays in [0, 1] for the early reset."""
        command = _cmd(env, command_name)

        def one(s: str) -> torch.Tensor:
            return MotionTrackingRewards.tracking_obj_trans_error_exp(
                env, command_name, side=s, scale=pos_scale
            ) * MotionTrackingRewards.tracking_obj_rot_error_exp(
                env, command_name, side=s, scale=rot_scale
            )

        if side is not None or side_reduce == "mean":
            return _mean_over_sides(env, command_name, side, one)
        if side_reduce != "prod":
            raise ValueError(f"side_reduce must be 'mean' or 'prod', got {side_reduce!r}")
        return torch.stack([one(s) for s in command._side_list], dim=-1).prod(dim=-1)

    @staticmethod
    def _obj_pcd_match_error_exp(
        env: "ManagerBasedRlEnv",
        command_name: str,
        scale: float,
        side: str,
        n_points: int = 128,
        n_obj_verts: int = 2048,
        topk_frac: float = 1.0,
        topk: int = 0,
        deadband: float = 0.0,
        norm_radius: float = 0.0,
    ) -> torch.Tensor:
        command = _cmd(env, command_name)
        si = _side_idx(command, side)
        sp = _keypoints_per_env(
            command,
            side,
            _obj_surface_keypoints(command, env.device, n_points, n_obj_verts)[side],
        )  # (B, P, 3)
        # norm_radius>0: rescale each object's keypoints to that max radius, so a rotation error
        # costs the same reward whatever the object size (translation stays 1:1)
        if norm_radius > 0.0:
            sp = sp * (norm_radius / sp.norm(dim=-1).amax(dim=1, keepdim=True).clamp_min(1e-6))[..., None]

        dR = matrix_from_quat(command.ref_obj_quat_w[:, si]) - matrix_from_quat(
            command.sim_obj_quat_w[:, si]
        )  # (B, 3, 3)
        dt = command.ref_obj_trans_w[:, si] - command.sim_obj_trans_w[:, si]  # (B, 3)
        disp = dt[:, None, :] + torch.einsum(
            "bij,bpj->bpi", dR, sp
        )  # (B, P, 3) ref−sim per point
        d = disp.norm(dim=-1)  # (B, P) per-point distance
        # Pooling: mean over points (lenient) or mean of the worst-k (strict, like success_rate_1.0).
        error = _worst_k_mean(d, topk, topk_frac)  # (B,)
        # Deadband: errors below it score full reward, so holding a resting object (mm-scale
        # contact jitter) is not penalised relative to leaving it untouched.
        if deadband > 0.0:
            error = torch.clamp(error - deadband, min=0.0)
        rew = torch.exp(-scale * error)
        # Per-side value cached for task_reward_early_reset, which averages the sides back up.
        ungated = getattr(command, "_pcd_ungated", None)
        if ungated is None:
            ungated = command._pcd_ungated = {}
        ungated[side] = rew
        return rew

    @staticmethod
    def _contact_gate(
        env: "ManagerBasedRlEnv", command, side: str, hold_steps: int
    ) -> torch.Tensor:
        si = _side_idx(command, side)
        sensor = env.scene[f"{side[0]}_alllink_contact_pos"]
        found = (sensor.data.found > 0).any(dim=-1)  # (B,) any of 13 links touching
        need = (command.ref_contact_alllink_flags[:, si] > 0).any(dim=-1)  # (B,)
        ep_len = env.episode_length_buf
        last = getattr(command, "_contact_gate_last", None)
        if last is None:
            last = command._contact_gate_last = {}
        prev = last.get(side)
        if prev is None:
            prev = torch.full_like(ep_len, -(10**6))
        # Fresh episode: forget the previous episode's contact; then stamp this step's contact.
        prev = torch.where(ep_len <= 1, torch.full_like(prev, -(10**6)), prev)
        prev = torch.where(found, ep_len, prev)
        last[side] = prev
        # Hysteresis: rigid contacts flicker at 60 Hz, so a touch within `hold_steps` still counts.
        touching = (ep_len - prev) <= hold_steps
        return (~need | touching).to(torch.float32)

    @staticmethod
    def tracking_obj_pcd_match_error_exp(
        env: "ManagerBasedRlEnv",
        command_name: str,
        scale: float,
        n_points: int = 128,
        n_obj_verts: int = 2048,
        topk_frac: float = 1.0,
        topk: int = 0,
        deadband: float = 0.0,
        side: str | None = None,
        norm_radius: float = 0.0,
    ) -> torch.Tensor:
        """Mean over the motion's hand sides (or one side), so the value stays in [0,1]."""
        return _mean_over_sides(
            env, command_name, side,
            lambda s: MotionTrackingRewards._obj_pcd_match_error_exp(
                env, command_name, scale, s, n_points, n_obj_verts, topk_frac, topk,
                deadband, norm_radius,
            ),
        )

    # ── Object keypoint match (fixed-side cube keypoints, docs: obs obj_keypoint_*) ─────

    @staticmethod
    def _obj_keypoint_match_error_exp(
        env: "ManagerBasedRlEnv",
        command_name: str,
        scale: float,
        side: str,
        cube_side: float = 0.2,
        n_points: int = 4,
        topk_frac: float = 1.0,
        topk: int = 0,
        deadband: float = 0.0,
        contact_gate: bool = False,
        contact_gate_hold_steps: int = 3,
    ) -> torch.Tensor:
        command = _cmd(env, command_name)
        si = _side_idx(command, side)
        kp = _keypoints_per_env(command, side, _obj_keypoints(command, env.device, cube_side, n_points)[side])
        dR = matrix_from_quat(command.ref_obj_quat_w[:, si]) - matrix_from_quat(command.sim_obj_quat_w[:, si])
        dt = command.ref_obj_trans_w[:, si] - command.sim_obj_trans_w[:, si]
        d = (dt[:, None, :] + torch.einsum("bij,bpj->bpi", dR, kp)).norm(dim=-1)  # (B, P) ref-sim
        error = _worst_k_mean(d, topk, topk_frac)
        if deadband > 0.0:
            error = torch.clamp(error - deadband, min=0.0)
        rew = torch.exp(-scale * error)
        ungated = getattr(command, "_pcd_ungated", None)  # read by task_reward_early_reset
        if ungated is None:
            ungated = command._pcd_ungated = {}
        ungated[side] = rew
        if not contact_gate:
            return rew
        return rew * MotionTrackingRewards._contact_gate(env, command, side, contact_gate_hold_steps)

    @staticmethod
    def tracking_obj_keypoint_match_error_exp(
        env: "ManagerBasedRlEnv",
        command_name: str,
        scale: float,
        cube_side: float = 0.2,
        n_points: int = 4,
        topk_frac: float = 1.0,
        topk: int = 0,
        deadband: float = 0.0,
        contact_gate: bool | list[str] = False,
        contact_gate_hold_steps: int = 3,
        side: str | None = None,
    ) -> torch.Tensor:
        """exp(-scale * keypoint error) of the object cube keypoints vs the ref, mean over the hand
        sides (or one side); same knobs as tracking_obj_pcd_match_error_exp."""
        command = _cmd(env, command_name)
        gated = command._side_list if contact_gate is True else (contact_gate or [])
        return _mean_over_sides(
            env, command_name, side,
            lambda s: MotionTrackingRewards._obj_keypoint_match_error_exp(
                env, command_name, scale, s, cube_side, n_points, topk_frac, topk,
                deadband, s in gated, contact_gate_hold_steps,
            ),
        )

    # ── Object <-> object relative pcd match (other-object frame) ─────────────

    @staticmethod
    def _obj_rel_pcd_match_error_exp(
        env: "ManagerBasedRlEnv",
        command_name: str,
        scale: float,
        side: str,
        n_points: int = 128,
        n_obj_verts: int = 2048,
        topk_frac: float = 1.0,
        topk: int = 0,
        deadband: float = 0.0,
        gate_dist: float = 0.0,
    ) -> torch.Tensor:
        """pcd match of this side's object against its ref CARRIED ALONG with the live other
        object (= per-point error in the other object's frame). S<2 -> 0; shared object -> const."""
        command = _cmd(env, command_name)
        S = len(command._side_list)
        B = command.sim_obj_trans_w.shape[0]
        if S < 2:
            return torch.zeros(B, device=env.device)
        si = _side_idx(command, side)
        oi = (si + 1) % S
        sp = _keypoints_per_env(
            command, side, _obj_surface_keypoints(command, env.device, n_points, n_obj_verts)[side]
        )  # (B, P, 3)
        d = _rel_pcd_dist(
            sp, command.sim_obj_trans_w, command.sim_obj_quat_w,
            command.ref_obj_trans_w, command.ref_obj_quat_w, si, oi,
        )
        error = _worst_k_mean(d, topk, topk_frac)
        if deadband > 0.0:
            error = torch.clamp(error - deadband, min=0.0)
        rew = torch.exp(-scale * error)
        if gate_dist > 0.0:
            # Only while the demo keeps the two objects within gate_dist (centre distance); a
            # shared object (alias, distance 0) never earns the constant.
            dist = (command.ref_obj_trans_w[:, si] - command.ref_obj_trans_w[:, oi]).norm(dim=-1)
            rew = rew * ((dist > 0.0) & (dist < gate_dist)).to(rew.dtype)
        return rew

    @staticmethod
    def tracking_obj_rel_pcd_match_error_exp(
        env: "ManagerBasedRlEnv",
        command_name: str,
        scale: float,
        n_points: int = 128,
        n_obj_verts: int = 2048,
        topk_frac: float = 1.0,
        topk: int = 0,
        deadband: float = 0.0,
        gate_dist: float = 0.0,
        side: str | None = None,
    ) -> torch.Tensor:
        """Mean over sides (or one side); rewards.split_sides turns it into r_/l_ twins."""
        return _mean_over_sides(
            env, command_name, side,
            lambda s: MotionTrackingRewards._obj_rel_pcd_match_error_exp(
                env, command_name, scale, s, n_points, n_obj_verts, topk_frac, topk, deadband,
                gate_dist,
            ),
        )

    # ── Contact position match (matched contact, all 13 links) ─────────────

    @staticmethod
    def _contact_alllink_match(
        env: "ManagerBasedRlEnv",
        command_name: str,
        scale: float,
        side: str,
        sensor_name: str,
        max_distance: float = 1.0,
    ) -> torch.Tensor:
        command = _cmd(env, command_name)
        si = _side_idx(command, side)

        demo = command.ref_contact_alllink_trans_w[:, si]  # (B, 13, 3) demo pts
        sensor = env.scene[sensor_name]
        policy = sensor.data.pos  # (B, 13, 3) policy contact points (world)
        found = sensor.data.found > 0  # (B, 13) policy actually contacts
        dist = torch.norm(demo - policy, dim=-1)  # (B, 13)

        flag = command.ref_contact_alllink_flags[:, si] > 0  # demo expects contact
        both = flag & found

        eff = torch.where(both, dist, torch.full_like(dist, max_distance))
        return torch.exp(-scale * eff).mean(dim=-1)  # (B,)

    @staticmethod
    def tracking_contact_alllink_match(
        env: "ManagerBasedRlEnv",
        command_name: str,
        scale: float,
        max_distance: float = 1.0,
        side: str | None = None,
    ) -> torch.Tensor:
        """Mean over the motion's hand sides (or one side); the mindist sensor follows the side."""
        return _mean_over_sides(
            env, command_name, side,
            lambda s: MotionTrackingRewards._contact_alllink_match(
                env, command_name, scale, s, f"{s[0]}_alllink_contact_pos", max_distance,
            ),
        )

    @staticmethod
    def _contact_chamfer_match(
        env: "ManagerBasedRlEnv",
        command_name: str,
        scale: float,
        side: str,
        sensor_name: str,
    ) -> torch.Tensor:
        """exp(-beta * chamfer) between the SET of policy contact points (links in contact)
        and the SET of demo contact points (flagged links)."""
        command = _cmd(env, command_name)
        si = _side_idx(command, side)
        demo = command.ref_contact_alllink_trans_w[:, si]  # (B, 13, 3) demo pts at the live obj pose
        flag = command.ref_contact_alllink_flags[:, si] > 0  # (B, 13)
        sensor = env.scene[sensor_name]
        policy = sensor.data.pos  # (B, 13, 3)
        found = sensor.data.found > 0  # (B, 13)
        d = torch.cdist(policy, demo)  # (B, 13 policy, 13 demo)
        big = torch.full_like(d, 100.0)
        # nearest flagged demo point per policy link / nearest touching policy link per demo point
        d_pf = torch.where(flag[:, None, :], d, big).min(dim=-1).values  # (B, 13)
        d_dp = torch.where(found[:, :, None], d, big).min(dim=1).values  # (B, 13)
        n_p, n_d = found.sum(-1), flag.sum(-1)
        chamfer = 0.5 * (
            (d_pf * found).sum(-1) / n_p.clamp(min=1) + (d_dp * flag).sum(-1) / n_d.clamp(min=1)
        )
        one_zero = (n_p == 0) ^ (n_d == 0)
        both_zero = (n_p == 0) & (n_d == 0)
        chamfer = torch.where(one_zero, torch.full_like(chamfer, 100.0), chamfer)
        rew = torch.exp(-scale * chamfer)
        return torch.where(both_zero, torch.zeros_like(rew), rew)  # mask_zero_contact

    @staticmethod
    def tracking_contact_chamfer_match(
        env: "ManagerBasedRlEnv",
        command_name: str,
        scale: float,
        side: str | None = None,
    ) -> torch.Tensor:
        """Mean over the motion's hand sides of the chamfer contact reward, in [0, 1]
        per side (the per-link alllink_match caps at n_flagged/13 instead)."""
        return _mean_over_sides(
            env, command_name, side,
            lambda s: MotionTrackingRewards._contact_chamfer_match(
                env, command_name, scale, s, f"{s[0]}_alllink_contact_pos",
            ),
        )

    # ── Regularizers (contact force) ─────────────────────────────────────────
    # Positive penalties: the reward manager ADDS weight*value -> use NEGATIVE weights.

    @staticmethod
    def force_penalty(
        env: "ManagerBasedRlEnv", threshold: float = 500.0, mean_over_sides: bool = False
    ) -> torch.Tensor:
        # Sum of per-side link means over the alllink sensors present, or (mean_over_sides) one
        # mean over every link of both hands; Scene has no __contains__.
        excess = []
        for name in ("r_alllink_contact", "l_alllink_contact"):
            try:
                sensor = env.scene[name]
            except KeyError:
                continue
            force_norm = torch.norm(sensor.data.force, dim=-1)  # (B, 13) net force norm
            excess.append(torch.clamp(force_norm - threshold, min=0.0))
        assert excess, "no *_alllink_contact sensor in scene"
        if mean_over_sides:
            return torch.cat(excess, dim=-1).mean(dim=-1)  # (B,)
        return sum(e.mean(dim=-1) for e in excess)  # (B,)

    # ── Termination penalty (standard failure-cost ablation) ────────────────
    # `reset_buf & ~time_out`: bad terminations only; use a NEGATIVE weight.

    @staticmethod
    def termination_penalty(
        env: "ManagerBasedRlEnv",
        command_name: str,
        terms: list[str] | None = None,
        side: str | None = None,
    ) -> torch.Tensor:
        if side is not None:
            return MotionTrackingRewards._termination_penalty_side(env, command_name, side)
        if terms is None:
            return env.reset_terminated.float()  # all non-timeout (= bad) terminations
        tm = env.termination_manager
        pen = tm.get_term(terms[0]).float()
        for name in terms[1:]:
            pen = pen + tm.get_term(name).float()
        return pen  # (B,) number of the listed failure terms firing this step

    @staticmethod
    def _termination_penalty_side(env: "ManagerBasedRlEnv", command_name: str, side: str) -> torch.Tensor:
        """Bad termination charged to the side whose object/hand tripped it; terminations with
        no per-side record (terminations.motion_tracking.side_fail) charge both sides."""
        terminated = env.reset_terminated
        fails = getattr(env, "_side_fail", None) or {}
        si = _side_idx(_cmd(env, command_name), side)
        blame = torch.zeros_like(terminated)
        known = torch.zeros_like(terminated)
        for m in fails.values():  # (B, n_sides) bool per recording termination term
            blame |= m[:, si]
            known |= m.any(dim=-1)
        return (terminated & (blame | ~known)).float()
