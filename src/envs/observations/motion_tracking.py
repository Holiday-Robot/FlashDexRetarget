from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import (
    combine_frame_transforms,
    matrix_from_quat,
    quat_apply_inverse,
    quat_from_matrix,
    quat_inv,
    quat_mul,
    subtract_frame_transforms,
)

from ..commands.motion_tracking import MotionTrackingCommand
from ..object_points import (
    farthest_point_sample,
    generate_bps_basis,
    keypoints_per_env,
    keypoints_world,
    load_obj_verts,
    obj_keypoints,
    obj_surface_keypoints,
    surface_pool,
)
from .base import BaseObs, _rotate_vec_world_to_wrist

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

# pre-2026-09-29 private names, still imported by rewards/motion_tracking.py and scratch tools
_farthest_point_sample = farthest_point_sample
_keypoints_per_env = keypoints_per_env
_load_obj_verts = load_obj_verts
_obj_keypoints = obj_keypoints
_obj_surface_keypoints = obj_surface_keypoints
_surface_pool = surface_pool


def _rotate_quat_world_to_wrist(
    quat_w: torch.Tensor, command: MotionTrackingCommand
) -> torch.Tensor:
    """Express a per-side world-frame quaternion in the per-side wrist frame."""
    return quat_mul(quat_inv(command.robot_wrist_quat_w), quat_w)


def _obj_rel_other_obj(
    trans_w: torch.Tensor, quat_w: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Each side's object pose in the OTHER side's object frame: T_other^-1 T_own.
    (B,S,3) trans + (B,S,3,3) rot; S=1 has no other object -> width-0 (B,0,...)."""
    B, S = trans_w.shape[0], trans_w.shape[1]
    if S < 2:
        return trans_w.new_zeros(B, 0, 3), trans_w.new_zeros(B, 0, 3, 3)
    other = torch.roll(torch.arange(S, device=trans_w.device), shifts=-1)  # S=2: [1, 0]
    R = matrix_from_quat(quat_w)  # (B,S,3,3)
    R_o = R[:, other]
    rel_t = torch.einsum("bsji,bsj->bsi", R_o, trans_w - trans_w[:, other])  # R_o^T (p - p_o)
    rel_R = R_o.transpose(-1, -2) @ R
    return rel_t, rel_R


# ── CHORD hand commands (robotic_grounding recompute_hand_keypoints_from_object) ─────────


def _chord_wrist_cmd_w(command: MotionTrackingCommand, object_relative: bool = True):
    """CHORD wrist command: the retargeted-robot palm pose re-anchored to the LIVE object,
    T_obj_sim x T_obj_ref^-1 x T_palm_ref (plain ref pose when off). (B,S,3), (B,S,4) world."""
    t_ref, q_ref = command.robot_ref_wrist_trans_w, command.robot_ref_wrist_quat_w
    if not object_relative:
        return t_ref, q_ref
    B, S = t_ref.shape[:2]
    t_o, q_o = subtract_frame_transforms(
        command.ref_obj_trans_w.reshape(-1, 3), command.ref_obj_quat_w.reshape(-1, 4),
        t_ref.reshape(-1, 3), q_ref.reshape(-1, 4),
    )
    t, q = combine_frame_transforms(
        command.sim_obj_trans_w.reshape(-1, 3), command.sim_obj_quat_w.reshape(-1, 4), t_o, q_o
    )
    return t.view(B, S, 3), q.view(B, S, 4)


def _chord_tips_cmd_w(command: MotionTrackingCommand, object_relative: bool = True):
    """CHORD fingertip commands: retargeted-robot tip positions re-anchored to the LIVE object
    the same way as _chord_wrist_cmd_w. (B,S,5,3) world."""
    tips = command.robot_ref_tip_trans_w
    if not object_relative:
        return tips
    B, S, P = tips.shape[:3]
    obj = lambda x, k: x[:, :, None].expand(B, S, P, k).reshape(-1, k)
    t_o, _ = subtract_frame_transforms(
        obj(command.ref_obj_trans_w, 3), obj(command.ref_obj_quat_w, 4), tips.reshape(-1, 3)
    )
    t, _ = combine_frame_transforms(obj(command.sim_obj_trans_w, 3), obj(command.sim_obj_quat_w, 4), t_o)
    return t.view(B, S, P, 3)


def _future_obj_traj_parts(command, offsets: torch.Tensor) -> dict[str, torch.Tensor]:
    """Future-object window pieces (future ref - current sim, current wrist frame), keyed by their
    obs_params `components` name, each (B, K, S, width)."""
    fut_t = command.future_obj_traj_trans_w(offsets)          # (B,K,S,3)
    R_fut = command.future_obj_traj_rotmat_w(offsets)         # (B,K,S,3,3)
    fut_lv = command.future_obj_traj_lin_vel_w(offsets)
    fut_av = command.future_obj_traj_ang_vel_w(offsets)
    B, K, S = fut_t.shape[0], fut_t.shape[1], fut_t.shape[2]

    sim_t = command.sim_obj_trans_w[:, None]
    sim_lv = command.sim_obj_lin_vel_w[:, None]
    sim_av = command.sim_obj_ang_vel_w[:, None]
    wrist_q = command.robot_wrist_quat_w[:, None].expand(B, K, S, 4)
    R_w = matrix_from_quat(wrist_q)                            # (B,K,S,3,3)
    R_sim = matrix_from_quat(command.sim_obj_quat_w)[:, None].expand(B, K, S, 3, 3)

    d_trans = quat_apply_inverse(wrist_q, fut_t - sim_t)
    d_lv = quat_apply_inverse(wrist_q, fut_lv - sim_lv)
    d_av = quat_apply_inverse(wrist_q, fut_av - sim_av)
    # (fut in wrist) @ (sim in wrist)^-1, same relative rotation the quat path builds
    R_d = R_w.transpose(-1, -2) @ R_fut @ R_sim.transpose(-1, -2) @ R_w
    d_r6 = R_d[..., :2].reshape(B, K, S, 6)
    return {"obj_trans_wrist": d_trans, "obj_rot6d_wrist": d_r6, "obj_lin_vel_wrist": d_lv, "obj_ang_vel_wrist": d_av}


def _future_mano_traj_parts(command, offsets: torch.Tensor) -> dict[str, torch.Tensor]:
    """Future-MANO window pieces (future ref - current robot, current wrist frame), keyed by their
    obs_params `components` name, each (B, K, S, width)."""
    fut_wt = command.future_mano_wrist_trans_w(offsets)        # (B,K,S,3)
    R_fut = command.future_mano_wrist_rot_w(offsets)           # (B,K,S,3,3)
    fut_tip = command.future_mano_tip_trans_w(offsets)         # (B,K,S,5,3)
    B, K, S = fut_wt.shape[0], fut_wt.shape[1], fut_wt.shape[2]

    cur_wt = command.robot_wrist_trans_w[:, None]
    cur_wq = command.robot_wrist_quat_w[:, None].expand(B, K, S, 4)
    cur_tip = command.robot_tip_trans_w[:, None]
    R_cur = matrix_from_quat(cur_wq)

    d_wt = quat_apply_inverse(cur_wq, fut_wt - cur_wt)
    R_d = R_cur.transpose(-1, -2) @ R_fut
    d_r6 = R_d[..., :2].reshape(B, K, S, 6)
    tip_q = cur_wq[:, :, :, None, :].expand(B, K, S, 5, 4)
    d_tip = quat_apply_inverse(tip_q, fut_tip - cur_tip)
    return {"mano_wrist_trans_wrist": d_wt, "mano_wrist_rot6d_wrist": d_r6,
            "mano_tip_trans_wrist": d_tip.reshape(B, K, S, 15)}


def _future_traj_window(command, offsets: torch.Tensor, components: list[str] | str) -> torch.Tensor:
    """(B, K*S*C): the named window pieces at the K frame offsets, per step and side in order; a prefix
    ("obj_" / "mano_") takes that whole group in its fixed order (the split-window terms)."""
    names = [components] if isinstance(components, str) else list(components)
    parts: dict[str, torch.Tensor] = {}
    if any(n.startswith("obj_") for n in names):
        parts.update(_future_obj_traj_parts(command, offsets))
    if any(n.startswith("mano_") for n in names):
        parts.update(_future_mano_traj_parts(command, offsets))
    if isinstance(components, str):
        names = [n for n in parts if n.startswith(components)]
    bad = [n for n in names if n not in parts]
    if bad or not names:
        known = [*_future_obj_traj_parts(command, offsets), *_future_mano_traj_parts(command, offsets)]
        raise ValueError(f"future window components {bad}: pick from {known}.")
    feat = torch.cat([parts[n] for n in names], dim=-1)  # (B,K,S,C)
    return feat.reshape(feat.shape[0], -1)


def _future_body_delta(env, command_name: str, ref: str, components, wrist: bool) -> torch.Tensor:
    """NEXT-frame `ref` ("mano" / "robot") keypoints (mapped bodies + 5 tips) minus the CURRENT robot's,
    per side, for each component ("trans" / "lin_vel") in order; world or current-wrist frame, flat."""
    command = cast(MotionTrackingCommand, env.command_manager.get_term(command_name))
    nxt, noisy = ("next_robot_ref", "_noisy") if ref == "robot" else ("next_mano", "")
    out = []
    for kind in components:
        if kind not in ("trans", "lin_vel"):
            raise ValueError(f"body delta components {list(components)}: pick from trans / lin_vel.")
        parts = []
        for si, side in enumerate(command._side_list):
            body = getattr(command, f"{nxt}_all_joints_{kind}_w{noisy}")(side) - getattr(
                command, f"robot_all_joints_{kind}_w")(side)
            tip = getattr(command, f"{nxt}_tip_{kind}_w{noisy}")[:, si] - getattr(command, f"robot_tip_{kind}_w")[:, si]
            parts.append(torch.cat([body, tip], dim=1))
        delta = torch.stack(parts, dim=1)  # (B, S, P, 3)
        if wrist:
            delta = _rotate_vec_world_to_wrist(delta, command)
        out.append(delta.reshape(delta.shape[0], -1))
    return torch.cat(out, dim=-1)


def _robot_link_pos_w(command, side: str, links: tuple[str, ...]) -> torch.Tensor:
    """(B, L, 3) world positions of named robot points: "palm" and "<finger>_tip" are the command's
    sites, any other name is a robot body without the side prefix."""
    cache = getattr(command, "_robot_link_idx", None)
    if cache is None:
        cache = command._robot_link_idx = {}
    if (side, links) not in cache:
        sites = {"palm": command._palm_site_ids[side],
                 **dict(zip((f"{f}_tip" for f in command.finger_names), command._tip_site_ids[side]))}
        bodies = command.robot.body_names
        bad = [n for n in links if n not in sites and f"{side}_{n}" not in bodies]
        if bad:
            raise ValueError(f"robot links {bad}: use {list(sites)} or a robot body name without '{side}_'.")
        site_ids = [sites[n] for n in links if n in sites]
        body_ids = [bodies.index(f"{side}_{n}") for n in links if n not in sites]
        # each link's row in cat([site rows, body rows])
        rows, ns, nb = [], 0, len(site_ids)
        for n in links:
            rows.append(ns if n in sites else nb)
            ns, nb = (ns + 1, nb) if n in sites else (ns, nb + 1)
        cache[(side, links)] = (site_ids, body_ids, torch.tensor(rows, device=command.device))
    site_ids, body_ids, rows = cache[(side, links)]
    data = command.robot.data
    pos = torch.cat([data.site_pos_w[:, site_ids], data.body_link_pos_w[:, body_ids]], dim=1)
    return pos[:, rows]


def _robot_obj_sdf(env, command_name: str, links) -> tuple[torch.Tensor, torch.Tensor]:
    """Object SDF (B, S, L) and its gradient in the wrist frame (B, S, L, 3) at the named robot points
    (default palm + fingertips); zeros for a side without an SDF grid."""
    command = cast(MotionTrackingCommand, env.command_manager.get_term(command_name))
    links = tuple(str(n) for n in links) if links else ("palm",) + tuple(f"{f}_tip" for f in command.finger_names)
    B, L = command.robot_wrist_quat_w.shape[0], len(links)
    sdfs, grads = [], []
    for si, side in enumerate(command._side_list):
        if side not in command._obj_sdf_grids:
            sdfs.append(torch.zeros(B, L, device=command.device))
            grads.append(torch.zeros(B, L, 3, device=command.device))
            continue
        sdf, grad_world = command.sdf_query(_robot_link_pos_w(command, side, links), side)
        wrist_quat = command.robot_wrist_quat_w[:, si : si + 1].expand(B, L, 4)
        sdfs.append(sdf)
        grads.append(quat_apply_inverse(wrist_quat, grad_world))
    return torch.stack(sdfs, dim=1), torch.stack(grads, dim=1)


class MotionTrackingObs(BaseObs):
    """Motion-tracking obs. Extend by overriding any ``BaseObs`` @staticmethod or adding new ones."""

    # ── MANO reference (absolute) ──────────────────────────────────────────

    @staticmethod
    def ref_mano_wrist_lin_vel_w(
        env: ManagerBasedRlEnv,
        command_name: str,
    ) -> torch.Tensor:
        """Absolute MANO wrist velocity. Shape: (B, n_sides*3)."""
        command = cast(MotionTrackingCommand, env.command_manager.get_term(command_name))
        return command.mano_wrist_lin_vel_w.reshape(
            command.mano_wrist_lin_vel_w.shape[0], -1
        )

    @staticmethod
    def ref_mano_wrist_quat_w(
        env: ManagerBasedRlEnv,
        command_name: str,
    ) -> torch.Tensor:
        """Absolute MANO wrist quaternion. Shape: (B, n_sides*4)."""
        command = cast(MotionTrackingCommand, env.command_manager.get_term(command_name))
        return command.mano_wrist_quat_w.reshape(command.mano_wrist_quat_w.shape[0], -1)

    @staticmethod
    def ref_mano_wrist_ang_vel_w(
        env: ManagerBasedRlEnv,
        command_name: str,
    ) -> torch.Tensor:
        """Absolute MANO wrist angular velocity. Shape: (B, n_sides*3)."""
        command = cast(MotionTrackingCommand, env.command_manager.get_term(command_name))
        return command.mano_wrist_ang_vel_w.reshape(
            command.mano_wrist_ang_vel_w.shape[0], -1
        )

    @staticmethod
    def ref_mano_body_lin_vel_wrist(
        env: ManagerBasedRlEnv,
        command_name: str,
    ) -> torch.Tensor:
        """MANO reference velocities for 16 keypoints (11 non-tip bodies + 5
        fingertips), in wrist frame. Shape: (B, n_sides*16*3)."""
        command = cast(MotionTrackingCommand, env.command_manager.get_term(command_name))
        parts = []
        for side in command._side_list:
            si = command._side_list.index(side)
            body_vel = command.mano_all_joints_lin_vel_w(side)
            tip_vel = command.mano_tip_lin_vel_w[:, si]
            parts.append(torch.cat([body_vel, tip_vel], dim=1))
        result = torch.stack(parts, dim=1)
        result = _rotate_vec_world_to_wrist(result, command)
        return result.reshape(result.shape[0], -1)

    # ── Reference motion target (replaces the bare ref_episode_phase clock) ─────

    @staticmethod
    def ref_robot_joint_pos(
        env: ManagerBasedRlEnv,
        command_name: str,
    ) -> torch.Tensor:
        """Reference (demo) robot joint positions at the current motion step. Shape: (B, n_dofs)."""
        command = cast(MotionTrackingCommand, env.command_manager.get_term(command_name))
        return command.ref_joint_pos_noisy

    @staticmethod
    def ref_robot_joint_vel(
        env: ManagerBasedRlEnv,
        command_name: str,
    ) -> torch.Tensor:
        """Reference (demo) robot joint velocities at the current motion step. Shape: (B, n_dofs)."""
        command = cast(MotionTrackingCommand, env.command_manager.get_term(command_name))
        return command.ref_joint_vel_noisy

    @staticmethod
    def ref_mano_keybody_trans_mano_wrist(
        env: ManagerBasedRlEnv,
        command_name: str,
    ) -> torch.Tensor:
        """Reference MANO key bodies (5 level-1 joints + 5 tips) relative to the MANO wrist,
        in the MANO's OWN wrist frame (pose-invariant target shape). Shape: (B, n_sides*10*3)."""
        command = cast(MotionTrackingCommand, env.command_manager.get_term(command_name))
        parts = []
        for side in command._side_list:
            si = command._side_list.index(side)
            level1 = command.mano_level_trans_w(side, 1)  # (B, 5, 3) world
            tips = command.mano_tip_trans_w[:, si]  # (B, 5, 3) world
            key = torch.cat([level1, tips], dim=1)  # (B, 10, 3)
            wrist_trans = command.mano_wrist_trans_w[:, si]  # (B, 3)
            wrist_quat = command.mano_wrist_quat_w[:, si]  # (B, 4)
            rel = key - wrist_trans[:, None, :]  # (B, 10, 3)
            wrist_quat = wrist_quat[:, None, :].expand(rel.shape[0], rel.shape[1], 4)
            rel = quat_apply_inverse(wrist_quat, rel)  # (B, 10, 3) in wrist frame
            parts.append(rel)
        out = torch.stack(parts, dim=1)  # (B, n_sides, 10, 3)
        return out.reshape(out.shape[0], -1)

    # ── Reference 1-step lookahead (next-frame ref − current robot) ────────

    @staticmethod
    def ref_future_robot_joint_pos_delta(
        env: ManagerBasedRlEnv,
        command_name: str,
    ) -> torch.Tensor:
        command = cast(MotionTrackingCommand, env.command_manager.get_term(command_name))
        delta = command.next_ref_joint_pos_noisy - command.robot.data.joint_pos
        return delta

    @staticmethod
    def ref_future_robot_joint_vel_delta(
        env: ManagerBasedRlEnv,
        command_name: str,
    ) -> torch.Tensor:
        command = cast(MotionTrackingCommand, env.command_manager.get_term(command_name))
        delta = command.next_ref_joint_vel_noisy - command.robot.data.joint_vel
        return delta

    @staticmethod
    def ref_future_mano_body_delta_w(
        env: ManagerBasedRlEnv,
        command_name: str,
        components: list[str],
    ) -> torch.Tensor:
        """NEXT-frame MANO keypoints (mapped bodies + 5 tips) minus the CURRENT robot's, world frame;
        obs_params `components` (trans / lin_vel) concatenated in order. Shape: (B, C*S*P*3)."""
        return _future_body_delta(env, command_name, "mano", components, wrist=False)

    @staticmethod
    def ref_future_mano_body_delta_wrist(
        env: ManagerBasedRlEnv,
        command_name: str,
        components: list[str],
    ) -> torch.Tensor:
        """NEXT-frame MANO keypoints (mapped bodies + 5 tips) minus the CURRENT robot's, CURRENT robot wrist frame;
        obs_params `components` (trans / lin_vel) concatenated in order. Shape: (B, C*S*P*3)."""
        return _future_body_delta(env, command_name, "mano", components, wrist=True)

    @staticmethod
    def ref_future_robot_body_delta_w(
        env: ManagerBasedRlEnv,
        command_name: str,
        components: list[str],
    ) -> torch.Tensor:
        """NEXT-frame retargeted-robot keypoints (mapped bodies + 5 tips) minus the CURRENT robot's, world frame;
        obs_params `components` (trans / lin_vel) concatenated in order. Shape: (B, C*S*P*3)."""
        return _future_body_delta(env, command_name, "robot", components, wrist=False)

    @staticmethod
    def ref_future_robot_body_delta_wrist(
        env: ManagerBasedRlEnv,
        command_name: str,
        components: list[str],
    ) -> torch.Tensor:
        """NEXT-frame retargeted-robot keypoints (mapped bodies + 5 tips) minus the CURRENT robot's, CURRENT robot wrist frame;
        obs_params `components` (trans / lin_vel) concatenated in order. Shape: (B, C*S*P*3)."""
        return _future_body_delta(env, command_name, "robot", components, wrist=True)

    # ── MANO reference (deltas: ref - sim) ─────────────────────────────────

    @staticmethod
    def ref_mano_wrist_trans_delta_w(
        env: ManagerBasedRlEnv,
        command_name: str,
    ) -> torch.Tensor:
        """Delta from robot wrist to MANO wrist target. Shape: (B, n_sides*3)."""
        command = cast(MotionTrackingCommand, env.command_manager.get_term(command_name))
        delta = command.mano_wrist_trans_w - command.robot_wrist_trans_w
        return delta.reshape(delta.shape[0], -1)

    @staticmethod
    def ref_mano_wrist_quat_delta_w(
        env: ManagerBasedRlEnv,
        command_name: str,
    ) -> torch.Tensor:
        """Quaternion rotation delta. Shape: (B, n_sides*4)."""
        command = cast(MotionTrackingCommand, env.command_manager.get_term(command_name))
        mano_quat = command.mano_wrist_quat_w
        robot_quat = command.robot_wrist_quat_w
        delta = quat_mul(mano_quat, quat_inv(robot_quat))
        return delta.reshape(delta.shape[0], -1)

    @staticmethod
    def ref_mano_body_trans_delta_w(
        env: ManagerBasedRlEnv,
        command_name: str,
    ) -> torch.Tensor:
        """Delta from robot to MANO for 16 tracked keypoints (11 non-tip bodies +
        5 fingertips) per side. Shape: (B, n_sides*16*3)."""
        command = cast(MotionTrackingCommand, env.command_manager.get_term(command_name))
        parts = []
        for side in command._side_list:
            si = command._side_list.index(side)
            body_delta = command.mano_all_joints_trans_w(
                side
            ) - command.robot_all_joints_trans_w(side)
            tip_delta = command.mano_tip_trans_w[:, si] - command.robot_tip_trans_w[:, si]
            parts.append(torch.cat([body_delta, tip_delta], dim=1))
        delta = torch.stack(parts, dim=1)
        return delta.reshape(delta.shape[0], -1)

    @staticmethod
    def ref_mano_wrist_lin_vel_delta_w(
        env: ManagerBasedRlEnv,
        command_name: str,
    ) -> torch.Tensor:
        """Wrist velocity delta. Shape: (B, n_sides*3)."""
        command = cast(MotionTrackingCommand, env.command_manager.get_term(command_name))
        delta = command.mano_wrist_lin_vel_w - command.robot_wrist_lin_vel_w
        return delta.reshape(delta.shape[0], -1)

    @staticmethod
    def ref_mano_wrist_ang_vel_delta_w(
        env: ManagerBasedRlEnv,
        command_name: str,
    ) -> torch.Tensor:
        """Wrist angular velocity delta. Shape: (B, n_sides*3)."""
        command = cast(MotionTrackingCommand, env.command_manager.get_term(command_name))
        delta = command.mano_wrist_ang_vel_w - command.robot_wrist_ang_vel_w
        return delta.reshape(delta.shape[0], -1)

    # ── TIP-ONLY deltas (5 tips; the *_body_* terms also carry the non-tip bodies) ──

    @staticmethod
    def ref_mano_tip_trans_delta_w(
        env: ManagerBasedRlEnv,
        command_name: str,
    ) -> torch.Tensor:
        """TIP-ONLY position delta: MANO fingertip - robot fingertip, 5 tips per side,
        world frame. Shape: (B, n_sides*5*3)."""
        command = cast(MotionTrackingCommand, env.command_manager.get_term(command_name))
        parts = []
        for side in command._side_list:
            si = command._side_list.index(side)
            parts.append(command.mano_tip_trans_w[:, si] - command.robot_tip_trans_w[:, si])
        delta = torch.stack(parts, dim=1)
        return delta.reshape(delta.shape[0], -1)

    @staticmethod
    def ref_future_mano_tip_trans_delta_w(
        env: ManagerBasedRlEnv,
        command_name: str,
    ) -> torch.Tensor:
        """TIP-ONLY 1-step lookahead position delta: NEXT-frame MANO fingertip minus
        CURRENT robot fingertip, 5 tips per side, world frame. Shape: (B, n_sides*5*3)."""
        command = cast(MotionTrackingCommand, env.command_manager.get_term(command_name))
        parts = []
        for side in command._side_list:
            si = command._side_list.index(side)
            parts.append(command.next_mano_tip_trans_w[:, si] - command.robot_tip_trans_w[:, si])
        delta = torch.stack(parts, dim=1)
        return delta.reshape(delta.shape[0], -1)

    @staticmethod
    def ref_future_mano_tip_lin_vel_delta_w(
        env: ManagerBasedRlEnv,
        command_name: str,
    ) -> torch.Tensor:
        """TIP-ONLY world-frame variant of ref_future_mano_fingertip_lin_vel_delta_wrist:
        NEXT MANO tip vel minus CURRENT robot tip vel, world frame. Shape: (B, n_sides*5*3)."""
        command = cast(MotionTrackingCommand, env.command_manager.get_term(command_name))
        parts = []
        for side in command._side_list:
            si = command._side_list.index(side)
            parts.append(command.next_mano_tip_lin_vel_w[:, si] - command.robot_tip_lin_vel_w[:, si])
        result = torch.stack(parts, dim=1)
        return result.reshape(result.shape[0], -1)

    # ── Contact / distance helpers ─────────────────────────────────────────

    @staticmethod
    def ref_mano_contact_flags(
        env: ManagerBasedRlEnv,
        command_name: str,
    ) -> torch.Tensor:
        """Reference binary contact flag per finger per side. Shape: (B, n_sides*5)."""
        command = cast(MotionTrackingCommand, env.command_manager.get_term(command_name))
        flags = command.ref_contact_flags
        return flags.reshape(flags.shape[0], -1)

    @staticmethod
    def ref_mano_tips_obj_surface_distance(
        env: ManagerBasedRlEnv,
        command_name: str,
    ) -> torch.Tensor:
        """Precomputed MANO tip-to-object-surface distance. Shape: (B, n_sides*5)."""
        command = cast(MotionTrackingCommand, env.command_manager.get_term(command_name))
        return command.mano_tips_distance.reshape(command.mano_tips_distance.shape[0], -1)

    @staticmethod
    def robot_keypoints_obj_center_distance(
        env: ManagerBasedRlEnv,
        command_name: str,
    ) -> torch.Tensor:
        """Distance from object center to 18 robot bodies per side. Shape: (B, n_sides*18)."""
        command = cast(MotionTrackingCommand, env.command_manager.get_term(command_name))
        parts = []
        for side in command._side_list:
            si = command._side_list.index(side)
            obj_trans = command.sim_obj_trans_w[:, si : si + 1]
            wrist_trans = command.robot_wrist_trans_w[:, si : si + 1]
            body_trans = command.robot_all_joints_trans_w(side)
            tip_trans = command.robot_tip_trans_w[:, si]
            all_trans = torch.cat([wrist_trans, body_trans, tip_trans], dim=1)
            dist = torch.norm(obj_trans - all_trans, dim=-1)
            parts.append(dist)
        return torch.cat(parts, dim=-1)

    @staticmethod
    def robot_tips_obj_surface_distance(
        env: ManagerBasedRlEnv,
        command_name: str,
    ) -> torch.Tensor:
        """LIVE robot-fingertip to object-surface nearest-vertex distance (DexMachina kpt_dist
        port); unlike ref_mano_tips_obj_surface_distance (precomputed demo), follows policy actions. Shape: (B, n_sides*5)."""
        command = cast(MotionTrackingCommand, env.command_manager.get_term(command_name))
        verts = getattr(command, "_tipdist_verts", None)
        if verts is None:
            verts = load_obj_verts(command, env.device)
            command._tipdist_verts = verts
        parts = []
        for side in command._side_list:
            si = command._side_list.index(side)
            tips = command.robot_tip_trans_w[:, si]  # (B, 5, 3) world, LIVE
            obj_t = command.sim_obj_trans_w[:, si]  # (B, 3)
            obj_R = matrix_from_quat(command.sim_obj_quat_w[:, si])  # (B, 3, 3)
            v = keypoints_per_env(command, side, verts[side])  # (B, V, 3)
            vw = obj_t[:, None, :] + torch.einsum(
                "bij,bvj->bvi", obj_R, v
            )  # (B, V, 3) object verts in world at current pose
            d = torch.cdist(tips, vw)  # (B, 5, V)
            parts.append(d.min(dim=-1).values)  # (B, 5)
        return torch.cat(parts, dim=-1)  # (B, n_sides*5)

    # ── Object shape descriptor (BPS) ──────────────────────────────────────

    @staticmethod
    def obj_bps(
        env: ManagerBasedRlEnv,
        command_name: str,
        n_bps_points: int = 128,
        radius: float = 0.2,
        n_obj_verts: int = 2048,
        center: bool = True,
    ) -> torch.Tensor:
        command = cast(MotionTrackingCommand, env.command_manager.get_term(command_name))

        key = (n_bps_points, radius, n_obj_verts, center)
        enc = getattr(command, "_obj_bps_feat", None)
        if enc is None or getattr(command, "_obj_bps_key", None) != key:
            basis = generate_bps_basis(n_bps_points, radius, env.device)  # (P, 3)
            verts = load_obj_verts(command, env.device, n_sample=n_obj_verts)
            enc = {}
            for side in command._side_list:
                v = verts[side]  # (V, 3) — or (S, V, 3) in multi-object mode
                if center:
                    v = v - 0.5 * (v.amax(dim=-2, keepdim=True) + v.amin(dim=-2, keepdim=True))
                # BPS feature: distance from each basis point to nearest surface pt.
                if v.dim() == 2:
                    enc[side] = torch.cdist(basis, v).min(dim=-1).values  # (P,)
                else:
                    enc[side] = torch.stack(
                        [torch.cdist(basis, vs).min(dim=-1).values for vs in v]
                    )  # (S, P)
            command._obj_bps_feat = enc
            command._obj_bps_key = key

        parts = []
        for side in command._side_list:
            e = enc[side]
            if e.dim() == 1:
                parts.append(e[None, :].expand(env.num_envs, -1))
            else:  # multi-object: per-env gather by active slot
                parts.append(e[command.active_obj_slot(side)])
        return torch.cat(parts, dim=-1)  # (B, n_sides * n_bps_points)

    @staticmethod
    def obj_point_cloud_wrist(
        env: ManagerBasedRlEnv,
        command_name: str,
        n_points: int = 128,
        n_obj_verts: int = 2048,
    ) -> torch.Tensor:
        """LIVE object surface point cloud (cached FPS keypoints at the CURRENT pose), per-side
        WRIST frame, flat (B, n_sides*n_points*3); only the pose->wrist transform runs per step."""
        command = cast(MotionTrackingCommand, env.command_manager.get_term(command_name))
        surf = obj_surface_keypoints(command, env.device, n_points, n_obj_verts)

        pts_sides = []
        for side in command._side_list:
            si = command._side_list.index(side)
            sp = keypoints_per_env(command, side, surf[side])  # (B, P, 3)
            obj_t = command.sim_obj_trans_w[:, si]  # (B, 3)
            obj_R = matrix_from_quat(command.sim_obj_quat_w[:, si])  # (B, 3, 3)
            world = obj_t[:, None, :] + torch.einsum(
                "bij,bpj->bpi", obj_R, sp
            )  # (B, P, 3) surface pts at current pose, world
            pts_sides.append(world)
        world = torch.stack(pts_sides, dim=1)  # (B, n_sides, P, 3)
        rel = world - command.robot_wrist_trans_w[:, :, None, :]
        wrist = _rotate_vec_world_to_wrist(rel, command)  # (B, n_sides, P, 3)
        return wrist.reshape(wrist.shape[0], -1)  # (B, n_sides * P * 3)

    @staticmethod
    def ref_future_obj_point_cloud_delta_wrist(
        env: ManagerBasedRlEnv,
        command_name: str,
        n_points: int = 128,
        n_obj_verts: int = 2048,
    ) -> torch.Tensor:
        """Per-point 1-step lookahead (next ref pose - current sim pose) of the shared FPS
        keypoints, per-side WRIST frame, (B, n_sides*n_points*3); per-point deltas also encode rotation."""
        command = cast(MotionTrackingCommand, env.command_manager.get_term(command_name))
        surf = obj_surface_keypoints(command, env.device, n_points, n_obj_verts)

        deltas = []
        for side in command._side_list:
            si = command._side_list.index(side)
            sp = keypoints_per_env(command, side, surf[side])  # (B, P, 3)
            cur_R = matrix_from_quat(command.sim_obj_quat_w[:, si])  # (B, 3, 3)
            nxt_R = matrix_from_quat(command.next_obj_quat_w[:, si])  # (B, 3, 3)
            cur = command.sim_obj_trans_w[:, si][:, None, :] + torch.einsum(
                "bij,bpj->bpi", cur_R, sp
            )  # (B, P, 3) current world
            nxt = command.next_obj_trans_w[:, si][:, None, :] + torch.einsum(
                "bij,bpj->bpi", nxt_R, sp
            )  # (B, P, 3) next-frame ref world
            deltas.append(nxt - cur)  # (B, P, 3) world-frame per-point delta
        d = torch.stack(deltas, dim=1)  # (B, n_sides, P, 3)
        d = _rotate_vec_world_to_wrist(d, command)  # (B, n_sides, P, 3) wrist
        return d.reshape(d.shape[0], -1)  # (B, n_sides * P * 3)

    @staticmethod
    def _future_traj_offsets(
        env: ManagerBasedRlEnv, n_future: int, max_future_steps: int
    ) -> torch.Tensor:
        """(K,) long tensor of frame offsets = round(linspace(1, max, K))."""
        return (
            torch.linspace(1.0, float(max_future_steps), n_future, device=env.device)
            .round()
            .long()
        )

    @staticmethod
    def ref_future_obj_traj_state_wrist(
        env: ManagerBasedRlEnv,
        command_name: str,
        n_future: int = 10,
        max_future_steps: int = 30,
    ) -> torch.Tensor:
        """K-step future object state window, row-major (B, K, n_sides*13) flattened to
        (B, K*S*13): per step, wrist-frame trans(3)+quat(4)+lin_vel(3)+ang_vel(3) deltas (future ref - current sim)."""
        command = cast(MotionTrackingCommand, env.command_manager.get_term(command_name))
        offsets = MotionTrackingObs._future_traj_offsets(env, n_future, max_future_steps)

        fut_t = command.future_obj_traj_trans_w(offsets)         # (B,K,S,3)
        fut_q = quat_from_matrix(command.future_obj_traj_rotmat_w(offsets))  # (B,K,S,4)
        fut_lv = command.future_obj_traj_lin_vel_w(offsets)      # (B,K,S,3)
        fut_av = command.future_obj_traj_ang_vel_w(offsets)      # (B,K,S,3)
        B, K, S = fut_t.shape[0], fut_t.shape[1], fut_t.shape[2]

        # Current sim state + wrist, broadcast over the K axis.
        sim_t = command.sim_obj_trans_w[:, None]                 # (B,1,S,3)
        sim_q = command.sim_obj_quat_w[:, None].expand(B, K, S, 4)
        sim_lv = command.sim_obj_lin_vel_w[:, None]              # (B,1,S,3)
        sim_av = command.sim_obj_ang_vel_w[:, None]              # (B,1,S,3)
        wrist_q = command.robot_wrist_quat_w[:, None].expand(B, K, S, 4)
        wrist_inv = quat_inv(wrist_q)

        d_trans = quat_apply_inverse(wrist_q, fut_t - sim_t)      # (B,K,S,3)
        d_lv = quat_apply_inverse(wrist_q, fut_lv - sim_lv)       # (B,K,S,3)
        d_av = quat_apply_inverse(wrist_q, fut_av - sim_av)       # (B,K,S,3)
        # Relative rotation expressed in the wrist frame: (fut in wrist) ⊗ (sim in wrist)⁻¹
        sim_in_wrist = quat_mul(wrist_inv, sim_q)
        next_in_wrist = quat_mul(wrist_inv, fut_q)
        d_quat = quat_mul(next_in_wrist, quat_inv(sim_in_wrist))  # (B,K,S,4)

        feat = torch.cat([d_trans, d_quat, d_lv, d_av], dim=-1)   # (B,K,S,13)
        return feat.reshape(B, -1)                               # (B, K*S*13)

    @staticmethod
    def ref_future_obj_traj_point_cloud_wrist(
        env: ManagerBasedRlEnv,
        command_name: str,
        n_future: int = 10,
        max_future_steps: int = 30,
        n_points: int = 64,
        n_obj_verts: int = 2048,
    ) -> torch.Tensor:
        """K-step future object point-cloud window, row-major (B, K, S, P, 3) flattened to
        (B, K*S*P*3): shared FPS keypoints at each future ref pose, relative to the CURRENT wrist (wrist frame)."""
        command = cast(MotionTrackingCommand, env.command_manager.get_term(command_name))
        offsets = MotionTrackingObs._future_traj_offsets(env, n_future, max_future_steps)
        surf = obj_surface_keypoints(command, env.device, n_points, n_obj_verts)

        fut_t = command.future_obj_traj_trans_w(offsets)         # (B,K,S,3)
        fut_R = command.future_obj_traj_rotmat_w(offsets)        # (B,K,S,3,3)
        wrist_t = command.robot_wrist_trans_w                    # (B,S,3)
        wrist_q = command.robot_wrist_quat_w                     # (B,S,4)

        pts_sides = []
        for side in command._side_list:
            si = command._side_list.index(side)
            sp = keypoints_per_env(command, side, surf[side])   # (B,P,3)
            R = fut_R[:, :, si]                                  # (B,K,3,3)
            t = fut_t[:, :, si]                                  # (B,K,3)
            world = t[:, :, None, :] + torch.einsum("bkij,bpj->bkpi", R, sp)  # (B,K,P,3)
            rel = world - wrist_t[:, si][:, None, None, :]       # (B,K,P,3)
            wq = wrist_q[:, si][:, None, None, :].expand(
                rel.shape[0], rel.shape[1], rel.shape[2], 4
            )
            pts_sides.append(quat_apply_inverse(wq, rel))        # (B,K,P,3) wrist
        pts = torch.stack(pts_sides, dim=2)                      # (B,K,S,P,3)
        return pts.reshape(pts.shape[0], -1)                    # (B, K*S*P*3)

    # ── MANO-hand / retarget future-trajectory WINDOWS (encoder inputs): ablation siblings of
    #    future_obj_traj_* with the demo HAND as reference; each returns a FLAT (B, K*feat) window. ──

    @staticmethod
    def ref_future_mano_traj_wrist(
        env: ManagerBasedRlEnv,
        command_name: str,
        n_future: int = 10,
        max_future_steps: int = 30,
    ) -> torch.Tensor:
        """K-step future MANO-hand window, row-major (B, K, ·) flattened to (B, K*feat): demo
        body+tip keypoint position AND velocity deltas vs the CURRENT robot, wrist frame."""
        command = cast(MotionTrackingCommand, env.command_manager.get_term(command_name))
        offsets = MotionTrackingObs._future_traj_offsets(env, n_future, max_future_steps)

        fut_tip = command.future_mano_tip_trans_w(offsets)       # (B,K,S,5,3)
        fut_tip_v = command.future_mano_tip_lin_vel_w(offsets)   # (B,K,S,5,3)
        B, K, S = fut_tip.shape[0], fut_tip.shape[1], fut_tip.shape[2]
        wrist_q = command.robot_wrist_quat_w                     # (B,S,4)

        pos_parts, vel_parts = [], []
        for side in command._side_list:
            si = command._side_list.index(side)
            fut_body = command.future_mano_all_joints_trans_w(side, offsets)      # (B,K,M,3)
            fut_body_v = command.future_mano_all_joints_lin_vel_w(side, offsets)  # (B,K,M,3)
            cur_body = command.robot_all_joints_trans_w(side)[:, None]            # (B,1,M,3)
            cur_body_v = command.robot_all_joints_lin_vel_w(side)[:, None]        # (B,1,M,3)
            tip_dp = fut_tip[:, :, si] - command.robot_tip_trans_w[:, si][:, None]      # (B,K,5,3)
            tip_dv = fut_tip_v[:, :, si] - command.robot_tip_lin_vel_w[:, si][:, None]  # (B,K,5,3)
            dp = torch.cat([fut_body - cur_body, tip_dp], dim=2)        # (B,K,M+5,3)
            dv = torch.cat([fut_body_v - cur_body_v, tip_dv], dim=2)    # (B,K,M+5,3)
            wq = wrist_q[:, si][:, None, None, :].expand(B, K, dp.shape[2], 4)
            pos_parts.append(quat_apply_inverse(wq, dp))
            vel_parts.append(quat_apply_inverse(wq, dv))
        pos = torch.stack(pos_parts, dim=2)   # (B,K,S,M+5,3)
        vel = torch.stack(vel_parts, dim=2)   # (B,K,S,M+5,3)
        feat = torch.cat([pos.reshape(B, K, -1), vel.reshape(B, K, -1)], dim=-1)
        return feat.reshape(B, -1)            # (B, K*feat)

    @staticmethod
    def ref_future_mano_wrist_tips_traj_wrist(
        env: ManagerBasedRlEnv,
        command_name: str,
        n_future: int = 10,
        max_future_steps: int = 30,
    ) -> torch.Tensor:
        """K-step future MANO wrist pose + fingertip window, flat (B, K*S*22): per step,
        current-palm-frame deltas (future ref - current robot) of wrist trans(3)+quat(4)+tips(15)."""
        command = cast(MotionTrackingCommand, env.command_manager.get_term(command_name))
        offsets = MotionTrackingObs._future_traj_offsets(env, n_future, max_future_steps)

        fut_wt = command.future_mano_wrist_trans_w(offsets)      # (B,K,S,3)
        fut_wq = command.future_mano_wrist_quat_w(offsets)       # (B,K,S,4)
        fut_tip = command.future_mano_tip_trans_w(offsets)       # (B,K,S,5,3)
        B, K, S = fut_wt.shape[0], fut_wt.shape[1], fut_wt.shape[2]

        cur_wt = command.robot_wrist_trans_w[:, None]            # (B,1,S,3)
        cur_wq = command.robot_wrist_quat_w[:, None].expand(B, K, S, 4)
        cur_tip = command.robot_tip_trans_w[:, None]             # (B,1,S,5,3)

        d_wt = quat_apply_inverse(cur_wq, fut_wt - cur_wt)
        d_wq = quat_mul(quat_inv(cur_wq), fut_wq)
        tip_q = cur_wq[:, :, :, None, :].expand(B, K, S, 5, 4)
        d_tip = quat_apply_inverse(tip_q, fut_tip - cur_tip)

        feat = torch.cat([d_wt, d_wq, d_tip.reshape(B, K, S, 15)], dim=-1)
        return feat.reshape(B, -1)                               # (B, K*S*22)

    #########################################################################################
    ## Future-traj windows: 6D-rotation (delta) variants
    #########################################################################################

    @staticmethod
    def ref_future_traj_wrist(
        env: ManagerBasedRlEnv,
        command_name: str,
        components: list[str],
        n_future: int = 10,
        stride: int = 1,
    ) -> torch.Tensor:
        """Future window at motion frames stride*(1..n_future), flat (B, K*S*C): per step and side, the
        obs_params `components` (future ref - current, current wrist frame, 6D rotations) in order."""
        command = cast(MotionTrackingCommand, env.command_manager.get_term(command_name))
        offsets = torch.arange(1, n_future + 1, device=env.device) * int(stride)
        return _future_traj_window(command, offsets, [str(c) for c in components])

    @staticmethod
    def ref_future_obj_traj_state_rel_wrist(
        env: ManagerBasedRlEnv,
        command_name: str,
        n_future: int = 10,
        max_future_steps: int = 30,
    ) -> torch.Tensor:
        """Same deltas as ref_future_obj_traj_state_wrist but the rotation is a continuous 6D
        representation instead of a quaternion. (B, K*S*15): per step, wrist-frame
        trans(3) + 6D rot(6) + lin_vel(3) + ang_vel(3)."""
        command = cast(MotionTrackingCommand, env.command_manager.get_term(command_name))
        offsets = MotionTrackingObs._future_traj_offsets(env, n_future, max_future_steps)
        return _future_traj_window(command, offsets, "obj_")

    @staticmethod
    def ref_future_obj_keypoint_traj_wrist(
        env: ManagerBasedRlEnv,
        command_name: str,
        n_future: int = 10,
        max_future_steps: int = 30,
        cube_side: float = 0.2,
        n_points: int = 4,
    ) -> torch.Tensor:
        """ref_future_obj_traj_state_rel_wrist with pose+velocity replaced by cube keypoints: per step and side,
        wrist-frame keypoint deltas (P*3) + keypoint velocity deltas (P*3). (B, K*S*P*6)."""
        command = cast(MotionTrackingCommand, env.command_manager.get_term(command_name))
        offsets = MotionTrackingObs._future_traj_offsets(env, n_future, max_future_steps)
        kp = obj_keypoints(command, env.device, cube_side, n_points)
        kp_b = torch.stack([keypoints_per_env(command, s, kp[s]) for s in command._side_list], dim=1)  # (B,S,P,3)
        fut_t = command.future_obj_traj_trans_w(offsets)          # (B,K,S,3)
        R_fut = command.future_obj_traj_rotmat_w(offsets)         # (B,K,S,3,3)
        fut_lv = command.future_obj_traj_lin_vel_w(offsets)
        fut_av = command.future_obj_traj_ang_vel_w(offsets)
        R_sim = matrix_from_quat(command.sim_obj_quat_w)          # (B,S,3,3)
        arm_fut = torch.einsum("bksij,bspj->bkspi", R_fut, kp_b)  # (B,K,S,P,3) R p at the future poses
        arm_sim = torch.einsum("bsij,bspj->bspi", R_sim, kp_b)[:, None]  # (B,1,S,P,3)
        d_kp = (fut_t[:, :, :, None] + arm_fut) - (command.sim_obj_trans_w[:, None, :, None] + arm_sim)
        v_fut = fut_lv[:, :, :, None] + torch.cross(fut_av[:, :, :, None].expand_as(arm_fut), arm_fut, dim=-1)
        v_sim = command.sim_obj_lin_vel_w[:, None, :, None] + torch.cross(
            command.sim_obj_ang_vel_w[:, None, :, None].expand_as(arm_sim), arm_sim, dim=-1)
        R_w = matrix_from_quat(command.robot_wrist_quat_w)        # (B,S,3,3)
        d_kp = torch.einsum("bsji,bkspj->bkspi", R_w, d_kp)       # R_w^T d
        d_v = torch.einsum("bsji,bkspj->bkspi", R_w, v_fut - v_sim)
        B, K, S, P = d_kp.shape[:4]
        feat = torch.cat([d_kp.reshape(B, K, S, P * 3), d_v.reshape(B, K, S, P * 3)], dim=-1)  # (B,K,S,6P)
        return feat.reshape(B, -1)

    @staticmethod
    def ref_future_mano_wrist_tips_traj_rel_wrist(
        env: ManagerBasedRlEnv,
        command_name: str,
        n_future: int = 10,
        max_future_steps: int = 30,
    ) -> torch.Tensor:
        """Same deltas as ref_future_mano_wrist_tips_traj_wrist with 6D rotation.
        (B, K*S*24): wrist trans(3) + 6D rot(6) + tips(15)."""
        command = cast(MotionTrackingCommand, env.command_manager.get_term(command_name))
        offsets = MotionTrackingObs._future_traj_offsets(env, n_future, max_future_steps)
        return _future_traj_window(command, offsets, "mano_")

    @staticmethod
    def ref_future_robot_joint_traj(
        env: ManagerBasedRlEnv,
        command_name: str,
        n_future: int = 10,
        max_future_steps: int = 30,
    ) -> torch.Tensor:
        """K-step future retargeted-robot window, FLAT (B, n_future*2*n_dofs): reference joint
        pos+vel deltas vs the CURRENT robot joint state (joint space, no frame transform)."""
        command = cast(MotionTrackingCommand, env.command_manager.get_term(command_name))
        offsets = MotionTrackingObs._future_traj_offsets(env, n_future, max_future_steps)
        fut_jp = command.future_ref_joint_pos_noisy(offsets)   # (B,K,n_dofs)
        fut_jv = command.future_ref_joint_vel_noisy(offsets)   # (B,K,n_dofs)
        dp = fut_jp - command.robot.data.joint_pos[:, None]   # (B,K,n_dofs)
        dv = fut_jv - command.robot.data.joint_vel[:, None]   # (B,K,n_dofs)
        feat = torch.cat([dp, dv], dim=-1)               # (B,K,2*n_dofs)
        return feat.reshape(feat.shape[0], -1)           # (B, K*2*n_dofs)

    # ── Object state (sim, in wrist frame) ─────────────────────────────────

    @staticmethod
    def obj_trans_wrist(
        env: ManagerBasedRlEnv,
        command_name: str,
    ) -> torch.Tensor:
        """Object position relative to wrist, in wrist frame. Shape: (B, n_sides*3)."""
        command = cast(MotionTrackingCommand, env.command_manager.get_term(command_name))
        delta = command.sim_obj_trans_w - command.robot_wrist_trans_w
        delta = _rotate_vec_world_to_wrist(delta, command)
        return delta.reshape(delta.shape[0], -1)

    @staticmethod
    def obj_quat_wrist(
        env: ManagerBasedRlEnv,
        command_name: str,
    ) -> torch.Tensor:
        """Object quaternion expressed in wrist frame. Shape: (B, n_sides*4)."""
        command = cast(MotionTrackingCommand, env.command_manager.get_term(command_name))
        q = _rotate_quat_world_to_wrist(command.sim_obj_quat_w, command)
        return q.reshape(q.shape[0], -1)

    @staticmethod
    def obj_box_corners_wrist(
        env: ManagerBasedRlEnv,
        command_name: str,
        cube_side: float = 0.2,
    ) -> torch.Tensor:
        """All 8 corners of the fixed-side object cube, wrist frame (= obj_keypoint_trans_wrist with
        n_points=8). Shape: (B, n_sides*24)."""
        return MotionTrackingObs.obj_keypoint_trans_wrist(env, command_name, cube_side, 8)

    @staticmethod
    def obj_keypoint_trans_wrist(
        env: ManagerBasedRlEnv,
        command_name: str,
        cube_side: float = 0.2,
        n_points: int = 4,
    ) -> torch.Tensor:
        """Keypoints of a fixed-side cube on the object centre (object axes; 4 = crossed diagonals) at
        the sim pose, relative to the wrist in wrist frame; replaces trans+quat. (B, n_sides*P*3)."""
        command = cast(MotionTrackingCommand, env.command_manager.get_term(command_name))
        kp = obj_keypoints(command, env.device, cube_side, n_points)
        world = keypoints_world(command, kp, command.sim_obj_trans_w, command.sim_obj_quat_w)
        rel = _rotate_vec_world_to_wrist(world - command.robot_wrist_trans_w[:, :, None, :], command)
        return rel.reshape(rel.shape[0], -1)

    @staticmethod
    def obj_keypoint_lin_vel_wrist(
        env: ManagerBasedRlEnv,
        command_name: str,
        cube_side: float = 0.2,
        n_points: int = 4,
    ) -> torch.Tensor:
        """Sim keypoint velocities v + w x (R p), wrist frame; replaces obj lin+ang vel.
        Shape: (B, n_sides*P*3)."""
        command = cast(MotionTrackingCommand, env.command_manager.get_term(command_name))
        kp = obj_keypoints(command, env.device, cube_side, n_points)
        arm = keypoints_world(command, kp, torch.zeros_like(command.sim_obj_trans_w), command.sim_obj_quat_w)
        w = command.sim_obj_ang_vel_w[:, :, None, :].expand_as(arm)
        v = command.sim_obj_lin_vel_w[:, :, None, :] + torch.cross(w, arm, dim=-1)
        v = _rotate_vec_world_to_wrist(v, command)
        return v.reshape(v.shape[0], -1)

    @staticmethod
    def ref_future_obj_keypoint_delta_wrist(
        env: ManagerBasedRlEnv,
        command_name: str,
        cube_side: float = 0.2,
        n_points: int = 4,
    ) -> torch.Tensor:
        """Next-frame ref keypoints minus current sim keypoints, wrist frame; replaces the future
        trans+quat deltas. Shape: (B, n_sides*P*3)."""
        command = cast(MotionTrackingCommand, env.command_manager.get_term(command_name))
        kp = obj_keypoints(command, env.device, cube_side, n_points)
        cur = keypoints_world(command, kp, command.sim_obj_trans_w, command.sim_obj_quat_w)
        nxt = keypoints_world(command, kp, command.next_obj_trans_w, command.next_obj_quat_w)
        d = _rotate_vec_world_to_wrist(nxt - cur, command)
        return d.reshape(d.shape[0], -1)

    @staticmethod
    def obj_lin_vel_wrist(
        env: ManagerBasedRlEnv,
        command_name: str,
    ) -> torch.Tensor:
        """Object linear velocity in wrist frame. Shape: (B, n_sides*3)."""
        command = cast(MotionTrackingCommand, env.command_manager.get_term(command_name))
        v = _rotate_vec_world_to_wrist(command.sim_obj_lin_vel_w, command)
        return v.reshape(v.shape[0], -1)

    @staticmethod
    def obj_ang_vel_wrist(
        env: ManagerBasedRlEnv,
        command_name: str,
    ) -> torch.Tensor:
        """Object angular velocity in wrist frame. Shape: (B, n_sides*3)."""
        command = cast(MotionTrackingCommand, env.command_manager.get_term(command_name))
        w = _rotate_vec_world_to_wrist(command.sim_obj_ang_vel_w, command)
        return w.reshape(w.shape[0], -1)

    @staticmethod
    def obj_lin_vel_w(
        env: ManagerBasedRlEnv,
        command_name: str,
    ) -> torch.Tensor:
        """Object linear velocity, world frame (DexMachina root_lin_vel); translation-invariant,
        no env-origin handling. Wrist-frame variant: obj_lin_vel_wrist. Shape: (B, n_sides*3)."""
        command = cast(MotionTrackingCommand, env.command_manager.get_term(command_name))
        v = command.sim_obj_lin_vel_w
        return v.reshape(v.shape[0], -1)

    @staticmethod
    def obj_ang_vel_w(
        env: ManagerBasedRlEnv,
        command_name: str,
    ) -> torch.Tensor:
        """Object angular velocity, world frame (DexMachina root_ang_vel); wrist-frame
        variant: obj_ang_vel_wrist. Shape: (B, n_sides*3)."""
        command = cast(MotionTrackingCommand, env.command_manager.get_term(command_name))
        w = command.sim_obj_ang_vel_w
        return w.reshape(w.shape[0], -1)

    # ── Object state (absolute, world frame — DexMachina parts_pos/quat) ────

    @staticmethod
    def obj_trans_w(
        env: ManagerBasedRlEnv,
        command_name: str,
    ) -> torch.Tensor:
        """Object position in per-env workspace frame = world minus env origin (mjlab world
        pos includes grid spacing); DexMachina parts_pos analog. Shape: (B, n_sides*3)."""
        command = cast(MotionTrackingCommand, env.command_manager.get_term(command_name))
        pos = command.sim_obj_trans_w - env.scene.env_origins[:, None, :]
        return pos.reshape(pos.shape[0], -1)

    @staticmethod
    def obj_quat_w(
        env: ManagerBasedRlEnv,
        command_name: str,
    ) -> torch.Tensor:
        """Object quaternion, world frame (DexMachina parts_quat analog; wrist-frame variant:
        obj_quat_wrist). Shape: (B, n_sides*4)."""
        command = cast(MotionTrackingCommand, env.command_manager.get_term(command_name))
        q = command.sim_obj_quat_w
        return q.reshape(q.shape[0], -1)

    # ── Object deltas (ref - sim, in wrist frame) ──────────────────────────

    @staticmethod
    def ref_obj_trans_delta_wrist(
        env: ManagerBasedRlEnv,
        command_name: str,
    ) -> torch.Tensor:
        """Delta from sim to ref object pos, in wrist frame. Shape: (B, n_sides*3)."""
        command = cast(MotionTrackingCommand, env.command_manager.get_term(command_name))
        delta = command.ref_obj_trans_w_noisy - command.sim_obj_trans_w
        delta = _rotate_vec_world_to_wrist(delta, command)
        return delta.reshape(delta.shape[0], -1)

    @staticmethod
    def ref_obj_quat_delta_wrist(
        env: ManagerBasedRlEnv,
        command_name: str,
    ) -> torch.Tensor:
        """Quaternion delta from sim to ref object orientation, in wrist frame. Shape: (B, n_sides*4)."""
        command = cast(MotionTrackingCommand, env.command_manager.get_term(command_name))
        wrist_inv = quat_inv(command.robot_wrist_quat_w)
        sim_in_wrist = quat_mul(wrist_inv, command.sim_obj_quat_w)
        ref_in_wrist = quat_mul(wrist_inv, command.ref_obj_quat_w_noisy)
        delta = quat_mul(ref_in_wrist, quat_inv(sim_in_wrist))
        return delta.reshape(delta.shape[0], -1)

    @staticmethod
    def ref_obj_lin_vel_delta_wrist(
        env: ManagerBasedRlEnv,
        command_name: str,
    ) -> torch.Tensor:
        """Delta of object linear velocity (ref - sim) in wrist frame. Shape: (B, n_sides*3)."""
        command = cast(MotionTrackingCommand, env.command_manager.get_term(command_name))
        delta = command.ref_obj_lin_vel_w_noisy - command.sim_obj_lin_vel_w
        delta = _rotate_vec_world_to_wrist(delta, command)
        return delta.reshape(delta.shape[0], -1)

    @staticmethod
    def ref_obj_ang_vel_delta_wrist(
        env: ManagerBasedRlEnv,
        command_name: str,
    ) -> torch.Tensor:
        """Delta of object angular velocity (ref - sim) in wrist frame. Shape: (B, n_sides*3)."""
        command = cast(MotionTrackingCommand, env.command_manager.get_term(command_name))
        delta = command.ref_obj_ang_vel_w_noisy - command.sim_obj_ang_vel_w
        delta = _rotate_vec_world_to_wrist(delta, command)
        return delta.reshape(delta.shape[0], -1)

    # ── Object auxiliary (next-frame / SDF) ────────────────────────────────

    @staticmethod
    def ref_future_obj_trans_delta_wrist(
        env: ManagerBasedRlEnv,
        command_name: str,
    ) -> torch.Tensor:
        """Delta from sim obj to next-frame ref obj pos, in wrist frame. Shape: (B, n_sides*3)."""
        command = cast(MotionTrackingCommand, env.command_manager.get_term(command_name))
        delta = command.next_obj_trans_w - command.sim_obj_trans_w
        delta = _rotate_vec_world_to_wrist(delta, command)
        return delta.reshape(delta.shape[0], -1)

    @staticmethod
    def ref_future_obj_trans_delta_w(
        env: ManagerBasedRlEnv,
        command_name: str,
    ) -> torch.Tensor:
        """Next-frame ref minus sim object position, WORLD frame (DexMachina state_diff pos
        part); translation-invariant, env origin cancels. Shape: (B, n_sides*3)."""
        command = cast(MotionTrackingCommand, env.command_manager.get_term(command_name))
        delta = command.next_obj_trans_w - command.sim_obj_trans_w
        return delta.reshape(delta.shape[0], -1)

    @staticmethod
    def ref_future_obj_lin_vel_delta_w(
        env: ManagerBasedRlEnv,
        command_name: str,
    ) -> torch.Tensor:
        """Next-frame ref minus sim object linear velocity, world frame; velocity analog of
        ref_future_obj_trans_delta_w. Shape: (B, n_sides*3)."""
        command = cast(MotionTrackingCommand, env.command_manager.get_term(command_name))
        delta = command.next_obj_vel_w - command.sim_obj_lin_vel_w
        return delta.reshape(delta.shape[0], -1)

    @staticmethod
    def ref_future_obj_ang_vel_delta_w(
        env: ManagerBasedRlEnv,
        command_name: str,
    ) -> torch.Tensor:
        """Next-frame ref minus sim object angular velocity, world frame; counterpart of
        ref_future_obj_lin_vel_delta_w. Shape: (B, n_sides*3)."""
        command = cast(MotionTrackingCommand, env.command_manager.get_term(command_name))
        delta = command.next_obj_ang_vel_w - command.sim_obj_ang_vel_w
        return delta.reshape(delta.shape[0], -1)

    @staticmethod
    def ref_future_obj_lin_vel_wrist(
        env: ManagerBasedRlEnv,
        command_name: str,
    ) -> torch.Tensor:
        """Next-frame ref obj linear velocity, in wrist frame. Shape: (B, n_sides*3)."""
        command = cast(MotionTrackingCommand, env.command_manager.get_term(command_name))
        v = _rotate_vec_world_to_wrist(command.next_obj_vel_w, command)
        return v.reshape(v.shape[0], -1)

    @staticmethod
    def ref_future_obj_quat_delta_w(
        env: ManagerBasedRlEnv,
        command_name: str,
    ) -> torch.Tensor:
        """Relative rotation sim -> next-frame ref object orientation, world frame: proper
        quaternion next * sim^-1 (no sign discontinuities, unlike raw subtraction). Shape: (B, n_sides*4)."""
        command = cast(MotionTrackingCommand, env.command_manager.get_term(command_name))
        delta = quat_mul(command.next_obj_quat_w, quat_inv(command.sim_obj_quat_w))
        return delta.reshape(delta.shape[0], -1)

    @staticmethod
    def ref_future_obj_quat_delta_wrist(
        env: ManagerBasedRlEnv,
        command_name: str,
    ) -> torch.Tensor:
        command = cast(MotionTrackingCommand, env.command_manager.get_term(command_name))
        wrist_inv = quat_inv(command.robot_wrist_quat_w)
        sim_in_wrist = quat_mul(wrist_inv, command.sim_obj_quat_w)
        next_in_wrist = quat_mul(wrist_inv, command.next_obj_quat_w)
        delta = quat_mul(next_in_wrist, quat_inv(sim_in_wrist))
        return delta.reshape(delta.shape[0], -1)

    # ── Object <-> object (other-object frame) ─────

    @staticmethod
    def obj_rel_other_obj(
        env: ManagerBasedRlEnv,
        command_name: str,
    ) -> torch.Tensor:
        """Sim object pose in the other side's object frame: trans(3) + 6D rot(6) per side, (B, S*9).
        Frame-free; a shared object (left slot = alias) gives the identity, S=1 gives width 0."""
        command = cast(MotionTrackingCommand, env.command_manager.get_term(command_name))
        rel_t, rel_R = _obj_rel_other_obj(command.sim_obj_trans_w, command.sim_obj_quat_w)
        B, S = rel_t.shape[0], rel_t.shape[1]
        feat = torch.cat([rel_t, rel_R[..., :2].reshape(B, S, 6)], dim=-1)  # (B,S,9)
        return feat.reshape(B, S * 9)

    @staticmethod
    def ref_future_obj_rel_other_obj_delta(
        env: ManagerBasedRlEnv,
        command_name: str,
    ) -> torch.Tensor:
        """Next-frame ref relative pose minus the sim one, in the other-object frame, (B, S*9):
        trans delta(3) + 6D of R_ref_rel @ R_sim_rel^T (6). Shared object -> all zeros."""
        command = cast(MotionTrackingCommand, env.command_manager.get_term(command_name))
        sim_t, sim_R = _obj_rel_other_obj(command.sim_obj_trans_w, command.sim_obj_quat_w)
        ref_t, ref_R = _obj_rel_other_obj(command.next_obj_trans_w, command.next_obj_quat_w)
        B, S = sim_t.shape[0], sim_t.shape[1]
        R_d = ref_R @ sim_R.transpose(-1, -2)
        feat = torch.cat([ref_t - sim_t, R_d[..., :2].reshape(B, S, 6)], dim=-1)  # (B,S,9)
        return feat.reshape(B, S * 9)

    @staticmethod
    def ref_episode_phase(
        env: ManagerBasedRlEnv,
        command_name: str,
    ) -> torch.Tensor:
        """Normalized demo progress in [-1, 1]: 2*step/num_frames - 1 (DexMachina
        normalize_ep_len demo clock). Shape: (B, 1)."""
        command = cast(MotionTrackingCommand, env.command_manager.get_term(command_name))
        phase = (
            2.0 * command.motion_steps.float() / command.motion_num_frames.float() - 1.0
        )
        return phase.reshape(-1, 1)

    @staticmethod
    def robot_links_obj_sdf(
        env: ManagerBasedRlEnv,
        command_name: str,
        links: tuple[str, ...] | None = None,
    ) -> torch.Tensor:
        """Object SDF (signed distance) at the named robot points (_robot_link_pos_w; default palm +
        fingertips). Shape: (B, S*L)."""
        sdf, _ = _robot_obj_sdf(env, command_name, links)
        return sdf.reshape(sdf.shape[0], -1)

    @staticmethod
    def robot_links_obj_normal_wrist(
        env: ManagerBasedRlEnv,
        command_name: str,
        links: tuple[str, ...] | None = None,
    ) -> torch.Tensor:
        """Object SDF gradient (outward surface normal) at the named robot points, wrist frame.
        Shape: (B, S*L*3)."""
        _, grad = _robot_obj_sdf(env, command_name, links)
        return grad.reshape(grad.shape[0], -1)

    @staticmethod
    def robot_palm_tips_obj_sdf_normal_wrist(
        env: ManagerBasedRlEnv,
        command_name: str,
    ) -> torch.Tensor:
        """Per-keypoint SDF + outward normal at palm + 5 fingertips. Shape: (B, n_sides*6*4)."""
        sdf, grad = _robot_obj_sdf(env, command_name, None)
        out = torch.cat([sdf[..., None], grad], dim=-1)
        return out.reshape(out.shape[0], -1)

    # ── DexMachina parity terms (config/envs/obs/dexmachina_real.yaml) ──────

    @staticmethod
    def robot_keypoints_obj_surface_distance(
        env: ManagerBasedRlEnv,
        command_name: str,
        n_sample: int = 300,
    ) -> torch.Tensor:
        """DexMachina ``kpt_dist``: each of the 18 keypoints to the nearest of n_sample surface
        points of the SAME side's object at its live pose. Shape: (B, S*18)."""
        command = cast(MotionTrackingCommand, env.command_manager.get_term(command_name))
        verts = getattr(command, "_kptdist_verts", None)
        if verts is None or getattr(command, "_kptdist_n_sample", None) != n_sample:
            verts = load_obj_verts(command, env.device, n_sample=n_sample)
            command._kptdist_verts = verts
            command._kptdist_n_sample = n_sample
        parts = []
        for si, side in enumerate(command._side_list):
            kpts = MotionTrackingObs._robot_keypoints_w(command, si, side)  # (B, 18, 3)
            obj_t = command.sim_obj_trans_w[:, si]  # (B, 3)
            obj_R = matrix_from_quat(command.sim_obj_quat_w[:, si])  # (B, 3, 3)
            v = keypoints_per_env(command, side, verts[side])  # (B, V, 3)
            vw = obj_t[:, None, :] + torch.einsum("bij,bvj->bvi", obj_R, v)
            parts.append(torch.cdist(kpts, vw).min(dim=-1).values)  # (B, 18)
        return torch.cat(parts, dim=-1)

    @staticmethod
    def ref_obj_trans_delta_w(
        env: ManagerBasedRlEnv,
        command_name: str,
    ) -> torch.Tensor:
        """DexMachina ``state_diff`` pos part: CURRENT-counter ref (= the frame the upcoming action
        is scored on) minus sim object position, world frame. Shape: (B, S*3)."""
        command = cast(MotionTrackingCommand, env.command_manager.get_term(command_name))
        delta = command.ref_obj_trans_w_noisy - command.sim_obj_trans_w
        return delta.reshape(delta.shape[0], -1)

    @staticmethod
    def ref_obj_quat_sub_w(
        env: ManagerBasedRlEnv,
        command_name: str,
    ) -> torch.Tensor:
        """DexMachina ``state_diff`` quat part: RAW current-counter ref minus sim quaternion (sign
        sensitive); ref_obj_quat_delta_wrist is the proper relative rotation. Shape: (B, S*4)."""
        command = cast(MotionTrackingCommand, env.command_manager.get_term(command_name))
        delta = command.ref_obj_quat_w_noisy - command.sim_obj_quat_w
        return delta.reshape(delta.shape[0], -1)

    # ── CHORD parity terms (config/envs/obs/chord.yaml) ──────────

    @staticmethod
    def ref_robot_joint_pos_delta(
        env: ManagerBasedRlEnv,
        command_name: str,
        asset_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        """CHORD command joint block: reference minus current joint position. Shape: (B, n_dofs)."""
        command = cast(MotionTrackingCommand, env.command_manager.get_term(command_name))
        robot = env.scene[asset_cfg.name]
        delta = command.ref_joint_pos_noisy - robot.data.joint_pos
        return delta[:, asset_cfg.joint_ids]

    @staticmethod
    def ref_robot_wrist_pose_delta_wrist(
        env: ManagerBasedRlEnv,
        command_name: str,
        object_relative: bool = True,
    ) -> torch.Tensor:
        """CHORD command wrist block: the wrist command (live-object re-anchored by default) expressed
        in the CURRENT palm frame, pos (3) + quat (4) per side. Shape: (B, S*7)."""
        command = cast(MotionTrackingCommand, env.command_manager.get_term(command_name))
        t_cmd, q_cmd = _chord_wrist_cmd_w(command, object_relative)
        B, S = t_cmd.shape[:2]
        t, q = subtract_frame_transforms(
            command.robot_wrist_trans_w.reshape(-1, 3), command.robot_wrist_quat_w.reshape(-1, 4),
            t_cmd.reshape(-1, 3), q_cmd.reshape(-1, 4),
        )
        return torch.cat([t.view(B, S, 3), q.view(B, S, 4)], dim=-1).reshape(B, -1)

    @staticmethod
    def ref_obj_pose_delta_obj(
        env: ManagerBasedRlEnv,
        command_name: str,
    ) -> torch.Tensor:
        """CHORD command object block: the reference object pose expressed in the CURRENT object
        frame, pos (3) + quat (4) per side. Shape: (B, S*7)."""
        command = cast(MotionTrackingCommand, env.command_manager.get_term(command_name))
        B, S = command.sim_obj_trans_w.shape[:2]
        t, q = subtract_frame_transforms(
            command.sim_obj_trans_w.reshape(-1, 3), command.sim_obj_quat_w.reshape(-1, 4),
            command.ref_obj_trans_w_noisy.reshape(-1, 3), command.ref_obj_quat_w_noisy.reshape(-1, 4),
        )
        return torch.cat([t.view(B, S, 3), q.view(B, S, 4)], dim=-1).reshape(B, -1)
