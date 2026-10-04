"""CHORD (Zhu et al. 2026): contact-wrench guidance and keypoint / joint tracking rewards, with the
wrench-space math ported 1:1 from robotic_grounding ``tasks/v2d/mdp``."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from mjlab.utils.lab_api.math import matrix_from_quat

from ..observations.motion_tracking import _chord_tips_cmd_w, _chord_wrist_cmd_w, _load_obj_verts
from .base import _cmd, _mean_over_sides, _side_idx

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


# ── Wrench-space support function ───────────────────────────────────────────────


def make_wrench_basis(
    n_basis: int, seed: int, device, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """(K, 6) unit directions in wrench space (CHORD ``sample_wrench_space_basis_scaled``,
    rc=1); seeded on CPU so demo and runtime supports share one basis."""
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    basis = torch.randn(int(n_basis), 6, generator=g, dtype=torch.float32)
    basis = basis / basis.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return basis.to(device=device, dtype=dtype)


def cone_phases(n_edges: int, device, dtype: torch.dtype = torch.float32):
    """(cos, sin) of the ``n_edges`` friction-cone edge phase angles, each (E,)."""
    theta = torch.linspace(0.0, 2.0 * math.pi, steps=int(n_edges) + 1, device=device, dtype=dtype)[:-1]
    return torch.cos(theta), torch.sin(theta)


def friction_cone_edges(
    normals: torch.Tensor,
    cos_t: torch.Tensor,
    sin_t: torch.Tensor,
    mu: float,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Polyhedral cone rays ``n + mu*(cos*t1 + sin*t2)`` (unit) with ``n`` appended:
    (..., N, 3) unit inward normals -> (..., N, E+1, 3). Frisvad tangents as in CHORD."""
    nx, ny, nz = normals[..., 0], normals[..., 1], normals[..., 2]
    sign = torch.where(nz >= 0, torch.ones_like(nz), -torch.ones_like(nz))
    den = sign + nz
    den = torch.where(den.abs() < eps, sign * eps, den)
    a = -1.0 / den
    b = nx * ny * a
    t1 = torch.stack((1.0 + sign * nx * nx * a, sign * b, -sign * nx), dim=-1)
    t2 = torch.stack((b, sign + ny * ny * a, -ny), dim=-1)
    t1 = t1 / t1.norm(dim=-1, keepdim=True).clamp_min(eps)
    t2 = t2 / t2.norm(dim=-1, keepdim=True).clamp_min(eps)
    c = cos_t.view(*([1] * (normals.dim() - 1)), -1, 1)  # (..., 1, E, 1)
    s = sin_t.view_as(c)
    n = normals.unsqueeze(-2)  # (..., N, 1, 3)
    edges = n + mu * (c * t1.unsqueeze(-2) + s * t2.unsqueeze(-2))  # (..., N, E, 3)
    edges = edges / edges.norm(dim=-1, keepdim=True).clamp_min(eps)
    return torch.cat([edges, n], dim=-2)  # (..., N, E+1, 3)


def wrench_support(
    points: torch.Tensor,
    normals_in: torch.Tensor,
    active: torch.Tensor,
    basis: torch.Tensor,
    rc: torch.Tensor | float,
    mu: float,
    cos_t: torch.Tensor,
    sin_t: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Support ``max(0, max_w <b, w>)`` of the unit wrench set ``[f; (p x f)/rc]`` over
    every contact x cone edge, per basis direction: (..., N, 3) x2, (..., N) -> (..., K)."""
    norm = normals_in.norm(dim=-1, keepdim=True)
    n = normals_in / norm.clamp_min(eps)
    act = active & (norm.squeeze(-1) > 1e-3)  # CHORD: zero normal == no contact
    f = friction_cone_edges(n, cos_t, sin_t, mu, eps)  # (..., N, E1, 3)
    tau = torch.cross(points.unsqueeze(-2).expand_as(f), f, dim=-1) / rc
    w = torch.cat([f, tau], dim=-1) * act[..., None, None].to(f.dtype)  # (..., N, E1, 6)
    w = w.flatten(-3, -2)  # (..., N*E1, 6)
    return torch.matmul(w, basis.transpose(0, 1)).amax(dim=-2).clamp_min(0.0)


# ── Per-hand support state for the CHORD rewards (cached on the command) ──────


def _chord_state(command, device, mu, n_edges, n_basis, seed) -> dict:
    """Basis, cone phases and per-side object radius (bounding ball), cached on the
    command; rebuilt when the cone/basis params change."""
    key = (round(float(mu), 6), int(n_edges), int(n_basis), int(seed))
    st = getattr(command, "_chord_state", None)
    if st is not None and st["key"] == key:
        return st
    cos_t, sin_t = cone_phases(n_edges, device)
    verts = _load_obj_verts(command, device, n_sample=2048)  # {side: (V,3) | (S,V,3)}
    radius = {side: v.norm(dim=-1).amax(dim=-1) for side, v in verts.items()}  # () | (S,)
    st = {
        "key": key,
        "mu": float(mu),
        "basis": make_wrench_basis(n_basis, seed, device),
        "cos": cos_t,
        "sin": sin_t,
        "radius": radius,
        "demo": {},
        "step": {},
    }
    command._chord_state = st
    return st


def _chord_demo_table(command, side: str, st: dict) -> torch.Tensor:
    """Demo support sigma_h per motion frame, (T, K); pose-independent so precomputed
    once from the object-local contact_alllink_{pos,normal,flags} of the motion pt."""
    if side in st["demo"]:
        return st["demo"][side]
    ml = command.motion_lib
    pos = torch.nan_to_num(ml.contact_alllink_pos[side])  # (T, 13, 3) object-local
    normal_in = -torch.nan_to_num(ml.contact_alllink_normal[side])  # outward -> inward
    flags = ml.contact_alllink_flags[side] > 0  # (T, 13)
    rc = st["radius"][side]
    if rc.dim() == 1:  # multi-object: per-frame radius of the motion's slot
        rc = rc[ml.traj_obj_slot[side]].repeat_interleave(ml._motion_num_frames)
        rc = rc.view(-1, 1, 1, 1)
    chunks = []
    for s0 in range(0, pos.shape[0], 1024):
        s1 = s0 + 1024
        rc_c = rc[s0:s1] if isinstance(rc, torch.Tensor) and rc.dim() == 4 else rc
        chunks.append(
            wrench_support(
                pos[s0:s1], normal_in[s0:s1], flags[s0:s1], st["basis"], rc_c,
                st["mu"], st["cos"], st["sin"],
            )
        )
    st["demo"][side] = torch.cat(chunks, dim=0)  # (T, K)
    return st["demo"][side]


def _chord_step(env, command, side: str, st: dict, active_eps: float) -> dict:
    """Per-step sigma_h / sigma_r (+ active masks) for one hand, computed once per env
    step and shared by the support reward and the two contact penalties."""
    token = (int(env.common_step_counter), int(getattr(env, "_sim_step_counter", -1)))
    cache = st["step"].get(side)
    if cache is not None and cache["token"] == token:
        return cache
    si = _side_idx(command, side)
    sensor = env.scene[f"{side[0]}_alllink_contact_pos"]
    pos_w = sensor.data.pos  # (B, 13, 3) policy contact points (world)
    found = sensor.data.found > 0  # (B, 13)
    normal_w = sensor.data.normal
    if normal_w is None:
        raise RuntimeError(
            f"{side[0]}_alllink_contact_pos needs the 'normal' field for the CHORD wrench reward"
        )
    if getattr(sensor, "normal_is_outward", False):  # Isaac SDF gradient
        normal_w = -normal_w  # mjlab normal is primary->secondary = hand->object already
    obj_t = command.sim_obj_trans_w[:, si]  # (B, 3)
    obj_rt = matrix_from_quat(command.sim_obj_quat_w[:, si]).transpose(-1, -2)
    pos_l = torch.einsum("bij,bnj->bni", obj_rt, pos_w - obj_t[:, None, :])
    normal_l = torch.einsum("bij,bnj->bni", obj_rt, normal_w)
    rc = st["radius"][side]
    if rc.dim() == 1:
        rc = rc[command.active_obj_slot(side)].view(-1, 1, 1, 1)
    sigma_r = wrench_support(
        pos_l, normal_l, found, st["basis"], rc, st["mu"], st["cos"], st["sin"]
    )  # (B, K)
    sigma_h = _chord_demo_table(command, side, st)[command._motion_flat_ids]  # (B, K)
    cache = {
        "token": token,
        "sigma_h": sigma_h,
        "sigma_r": sigma_r,
        "cmd_active": sigma_h > active_eps,
        "cur_active": sigma_r > active_eps,
    }
    st["step"][side] = cache
    return cache


def chord_side(env, command_name, side, mu, n_edges, n_basis, seed, active_eps):
    """Cached per-step CHORD support state (sigma_h / sigma_r + active masks) for one hand."""
    command = _cmd(env, command_name)
    st = _chord_state(command, env.device, mu, n_edges, n_basis, seed)
    return _chord_step(env, command, side, st, active_eps)


# ── Rewards (mixed into MotionTrackingRewards) ───────────────────────────────


class ChordRewards:
    # ── CHORD contact-wrench guidance (replaces contact_alllink_match) ────────
    # Side-mean (or one-side) robotic_grounding contact_wrench_support_reward /
    # unintended_contact_penalty / missed_contact_penalty (single rigid body per side).

    @staticmethod
    def _contact_wrench_support(
        env, command_name, side, tolerance, var, mu, n_edges, n_basis, seed, active_eps
    ) -> torch.Tensor:
        c = chord_side(env, command_name, side, mu, n_edges, n_basis, seed, active_eps)
        sh, sr = c["sigma_h"], c["sigma_r"]
        under = torch.clamp((1.0 - tolerance) * sh - sr, min=0.0)
        over = torch.clamp(sr - (1.0 + tolerance) * sh, min=0.0)
        loss = under.square() + over.square()  # (B, K)
        both = (c["cmd_active"] & c["cur_active"]).to(loss.dtype)
        cmd_num = c["cmd_active"].sum(dim=-1).to(loss.dtype).clamp(min=1e-6)
        return (both * torch.exp(-loss / var)).sum(dim=-1) / cmd_num  # (B,)

    @staticmethod
    def _missed_contact_penalty(
        env, command_name, side, mu, n_edges, n_basis, seed, active_eps
    ) -> torch.Tensor:
        c = chord_side(env, command_name, side, mu, n_edges, n_basis, seed, active_eps)
        n_exp = c["cmd_active"].sum(dim=-1).float()  # (B,) commanded basis dirs
        n_missed = (c["cmd_active"] & ~c["cur_active"]).sum(dim=-1).float()
        return torch.where(n_exp > 0, n_missed / n_exp.clamp(min=1e-6), torch.zeros_like(n_exp))

    @staticmethod
    def _unintended_contact_penalty(
        env, command_name, side, mu, n_edges, n_basis, seed, active_eps
    ) -> torch.Tensor:
        c = chord_side(env, command_name, side, mu, n_edges, n_basis, seed, active_eps)
        cmd_body = c["cmd_active"].any(dim=-1)  # (B,) demo touches this object
        cur_body = c["cur_active"].any(dim=-1)
        support_sq = c["sigma_r"].clamp(min=0.0).square().mean(dim=-1)  # (B,)
        pen = (~cmd_body & cur_body).float() + (~cmd_body).float() * support_sq
        return pen

    @staticmethod
    def tracking_contact_wrench_support(
        env: "ManagerBasedRlEnv", command_name: str, tolerance: float = 0.1,
        var: float = 0.1, mu: float = 0.1, n_edges: int = 8, n_basis: int = 512,
        seed: int = 0, active_eps: float = 1e-3,
        side: str | None = None,
    ) -> torch.Tensor:
        return _mean_over_sides(
            env, command_name, side,
            lambda s: ChordRewards._contact_wrench_support(
                env, command_name, s, tolerance, var, mu, n_edges, n_basis, seed, active_eps
            ),
        )

    @staticmethod
    def tracking_missed_contact_penalty(
        env: "ManagerBasedRlEnv", command_name: str, mu: float = 0.1, n_edges: int = 8,
        n_basis: int = 512, seed: int = 0, active_eps: float = 1e-3,
        side: str | None = None,
    ) -> torch.Tensor:
        return _mean_over_sides(
            env, command_name, side,
            lambda s: ChordRewards._missed_contact_penalty(
                env, command_name, s, mu, n_edges, n_basis, seed, active_eps
            ),
        )

    @staticmethod
    def tracking_unintended_contact_penalty(
        env: "ManagerBasedRlEnv", command_name: str, mu: float = 0.1, n_edges: int = 8,
        n_basis: int = 512, seed: int = 0, active_eps: float = 1e-3,
        side: str | None = None,
    ) -> torch.Tensor:
        return _mean_over_sides(
            env, command_name, side,
            lambda s: ChordRewards._unintended_contact_penalty(
                env, command_name, s, mu, n_edges, n_basis, seed, active_eps
            ),
        )

    # ── CHORD tracking rewards (robotic_grounding v2d rewards.py; rewards/chord.yaml) ────

    @staticmethod
    def _side_finger_joint_ids(env: "ManagerBasedRlEnv", command, side: str) -> torch.Tensor:
        """One side's finger joint ids (robot.joint_names.finger filtered by the side prefix), cached."""
        key = f"_chord_finger_ids_{side}"
        ids = getattr(command, key, None)
        if ids is None:
            names = command.robot.joint_names
            prefix = "right_" if side == "right" else "left_"
            ids = torch.tensor(
                [int(i) for i in command._finger_joint_ids if names[int(i)].startswith(prefix)],
                dtype=torch.long, device=env.device,
            )
            if ids.numel() == 0:
                raise ValueError(f"no {side}-side finger joints among {list(names)}")
            setattr(command, key, ids)
        return ids

    @staticmethod
    def tracking_chord_obj_keypoints_exp(
        env: "ManagerBasedRlEnv", command_name: str, var: float = 0.1, kp_len: float = 1.0,
        side: str | None = None,
    ) -> torch.Tensor:
        """CHORD object_keypoints_tracking_exp: the 6 axis keypoints (+-kp_len along the object axes)
        at sim vs ref pose, mean_k exp(-||d_k||^2 / var); side mean (CHORD: mean over bodies)."""
        command = _cmd(env, command_name)
        eye = torch.eye(3, device=env.device)
        axes = torch.cat([eye, -eye], dim=0) * kp_len  # (6, 3)

        def one(s: str) -> torch.Tensor:
            si = _side_idx(command, s)
            dR = matrix_from_quat(command.ref_obj_quat_w[:, si]) - matrix_from_quat(command.sim_obj_quat_w[:, si])
            dt = command.ref_obj_trans_w[:, si] - command.sim_obj_trans_w[:, si]
            d = dt[:, None, :] + torch.einsum("bij,pj->bpi", dR, axes)  # (B, 6, 3) ref - sim
            return torch.exp(-d.square().sum(dim=-1) / var).mean(dim=-1)

        return _mean_over_sides(env, command_name, side, one)

    @staticmethod
    def tracking_chord_hand_keypoints_exp(
        env: "ManagerBasedRlEnv", command_name: str, var: float = 0.1, threshold: float = 0.0,
        object_relative: bool = True, side: str | None = None,
    ) -> torch.Tensor:
        """CHORD hand_keypoints_tracking_exp: palm + 5 tips vs the retargeted-robot command (re-anchored
        to the live object by default), mean_k exp(-max(||d||^2 - threshold, 0) / var); side mean."""
        command = _cmd(env, command_name)
        t_cmd, _ = _chord_wrist_cmd_w(command, object_relative)  # (B, S, 3)
        cmd = torch.cat([t_cmd[:, :, None], _chord_tips_cmd_w(command, object_relative)], dim=2)
        cur = torch.cat([command.robot_wrist_trans_w[:, :, None], command.robot_tip_trans_w], dim=2)
        err = (cmd - cur).square().sum(dim=-1)  # (B, S, 6)
        rew = torch.exp(-(err - threshold).clamp(min=0.0) / var).mean(dim=-1)  # (B, S)
        return _mean_over_sides(env, command_name, side, lambda s: rew[:, _side_idx(command, s)])

    @staticmethod
    def tracking_chord_joint_pos_exp(
        env: "ManagerBasedRlEnv", command_name: str, var: float = 1.0, threshold: float = 0.0,
        side: str | None = None,
    ) -> torch.Tensor:
        """CHORD hand_joint_pos_tracking_exp: exp(-max(sum_j (q_ref - q)^2 - threshold, 0) / var) over
        one hand's finger joints; side mean."""
        command = _cmd(env, command_name)
        ref, q = command.ref_joint_pos, command.robot.data.joint_pos

        def one(s: str) -> torch.Tensor:
            ids = ChordRewards._side_finger_joint_ids(env, command, s)
            err = (ref[:, ids] - q[:, ids]).square().sum(dim=-1)
            return torch.exp(-(err - threshold).clamp(min=0.0) / var)

        return _mean_over_sides(env, command_name, side, one)
