"""MotionTrackingCommand mixin: episode-constant reference noise (obs + RSI)."""

from __future__ import annotations

import math

import torch
from mjlab.utils.lab_api.math import quat_mul


def _quat_from_rotvec(rotvec: torch.Tensor) -> torch.Tensor:
    ang = rotvec.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return torch.cat(
        [torch.cos(ang / 2), torch.sin(ang / 2) * rotvec / ang], dim=-1
    )


def _euler_zxy_R(e: torch.Tensor) -> torch.Tensor:
    """(B, 3) euler biases -> (B, 3, 3) in the forearm base-chain order."""
    cz, sz = torch.cos(e[:, 0]), torch.sin(e[:, 0])
    cx, sx = torch.cos(e[:, 1]), torch.sin(e[:, 1])
    cy, sy = torch.cos(-e[:, 2]), torch.sin(-e[:, 2])
    z1 = torch.zeros_like(cz)
    o1 = torch.ones_like(cz)
    Rz = torch.stack([cz, -sz, z1, sz, cz, z1, z1, z1, o1], dim=-1).view(-1, 3, 3)
    Rx = torch.stack([o1, z1, z1, z1, cx, -sx, z1, sx, cx], dim=-1).view(-1, 3, 3)
    Ry = torch.stack([cy, z1, sy, z1, o1, z1, -sy, z1, cy], dim=-1).view(-1, 3, 3)
    return Rz @ Rx @ Ry


class RefNoiseMixin:
    def _init_ref_noise(self) -> None:
        B, S = self.num_envs, len(self._side_list)
        self._refn_enabled = bool(getattr(self.cfg, "ref_noise", None) and self.cfg.ref_noise.enable)
        self._refn_wrist_t = torch.zeros(B, 3, device=self.device)
        self._refn_wrist_e = torch.zeros(B, 3, device=self.device)
        self._refn_wrist_v = torch.zeros(B, 3, device=self.device)
        self._refn_wrist_av = torch.zeros(B, 3, device=self.device)
        self._refn_wrist_R = torch.eye(3, device=self.device).expand(B, 3, 3).clone()
        self._refn_obj_t = torch.zeros(B, S, 3, device=self.device)
        self._refn_obj_rotvec = torch.zeros(B, S, 3, device=self.device)
        self._refn_obj_q = torch.zeros(B, S, 4, device=self.device)
        self._refn_obj_q[..., 0] = 1.0
        self._refn_obj_v = torch.zeros(B, S, 3, device=self.device)
        self._refn_obj_av = torch.zeros(B, S, 3, device=self.device)
        # Depenetration scale on the RSI state noise only; obs keep the full noise.
        self._refn_alpha = torch.ones(B, device=self.device)

    def _resample_ref_noise(self, env_ids: torch.Tensor) -> None:
        if not self._refn_enabled:
            return
        rn = self.cfg.ref_noise
        N = len(env_ids)
        S = len(self._side_list)
        if self._eval_mode:
            for t in (self._refn_wrist_t, self._refn_wrist_e, self._refn_wrist_v,
                      self._refn_wrist_av):
                t[env_ids] = 0.0
            for t in (self._refn_obj_t, self._refn_obj_v, self._refn_obj_av):
                t[env_ids] = 0.0
            self._refn_obj_q[env_ids] = 0.0
            self._refn_obj_q[env_ids, :, 0] = 1.0
        else:
            d2r = math.pi / 180.0
            rnd = lambda *s: torch.randn(*s, device=self.device)  # noqa: E731
            self._refn_wrist_t[env_ids] = rnd(N, 3) * rn.wrist_trans
            self._refn_wrist_e[env_ids] = rnd(N, 3) * (rn.wrist_rot_deg * d2r)
            self._refn_wrist_v[env_ids] = rnd(N, 3) * rn.wrist_lin_vel
            self._refn_wrist_av[env_ids] = rnd(N, 3) * (rn.wrist_ang_vel_deg * d2r)
            self._refn_obj_t[env_ids] = rnd(N, S, 3) * rn.obj_trans
            self._refn_obj_v[env_ids] = rnd(N, S, 3) * rn.obj_lin_vel
            self._refn_obj_av[env_ids] = rnd(N, S, 3) * (rn.obj_ang_vel_deg * d2r)
            rotvec = rnd(N, S, 3) * (rn.obj_rot_deg * d2r)
            self._refn_obj_rotvec[env_ids] = rotvec
            self._refn_obj_q[env_ids] = _quat_from_rotvec(rotvec)
        self._refn_wrist_R = _euler_zxy_R(self._refn_wrist_e)
        self._refn_alpha[env_ids] = 1.0
        if (
            not self._eval_mode
            and bool(getattr(self.cfg.ref_noise, "depenetrate", True))
            and self.has_objects
        ):
            self._refn_depenetrate(env_ids)

    def _refn_depenetrate(self, env_ids: torch.Tensor) -> None:
        """Largest alpha whose noised warm-start adds <= depen_slack penetration."""
        slack = float(getattr(self.cfg.ref_noise, "depen_slack", 0.003))
        alphas = (1.0, 0.75, 0.5, 0.25, 0.0)
        chosen = torch.zeros(len(env_ids), device=self.device)
        done = torch.zeros(len(env_ids), dtype=torch.bool, device=self.device)
        for side in self._side_list:
            if side not in self._obj_sdf_grids or side not in self.motion_lib.obj_trans:
                return  # no SDF: keep alpha=1
        for side in self._side_list:
            si = self._side_list.index(side)
            tips = self.robot_ref_tip_trans_w[env_ids][:, si]  # (N, 5, 3)
            lvl1 = self.robot_ref_level_trans_w(side, 1)[env_ids]  # (N, 5, 3)
            wrist = self.mano_wrist_trans_w[env_ids][:, si][:, None, :]  # (N, 1, 3)
            pts = torch.cat([tips, lvl1, wrist], dim=1)  # (N, 11, 3)
            obj_t = self.ref_obj_trans_w[env_ids][:, si]
            obj_q = self.ref_obj_quat_w[env_ids][:, si]
            t_w = self._refn_wrist_t[env_ids]
            e_w = self._refn_wrist_e[env_ids]
            t_o = self._refn_obj_t[env_ids][:, si]
            rv_o = self._refn_obj_rotvec[env_ids][:, si]

            min_sdf = []
            for a in alphas:
                R = _euler_zxy_R(e_w * a)
                p = torch.einsum("bij,bkj->bki", R, pts - wrist) + wrist + a * t_w[:, None, :]
                q_n = quat_mul(_quat_from_rotvec(a * rv_o), obj_q)
                sdf, _ = self.sdf_query(
                    p, side, obj_trans=obj_t + a * t_o, obj_quat=q_n, env_ids=env_ids
                )
                min_sdf.append(sdf.min(dim=1).values)  # (N,)
            allowed = min_sdf[-1] - slack  # clean (alpha=0) minus tolerance
            side_chosen = torch.zeros_like(chosen)
            side_done = torch.zeros_like(done)
            for a, ms in zip(alphas[:-1], min_sdf[:-1]):
                ok = (ms >= allowed) & ~side_done
                side_chosen[ok] = a
                side_done |= ok
            if si == 0:
                chosen, done = side_chosen, side_done
            else:
                chosen = torch.minimum(chosen, side_chosen)
        self._refn_alpha[env_ids] = chosen

    def _refn_joint(self, arr: torch.Tensor, vel: bool = False) -> torch.Tensor:
        if not self._refn_enabled:
            return arr
        out = arr.clone()
        t = self._refn_wrist_v if vel else self._refn_wrist_t
        e = self._refn_wrist_av if vel else self._refn_wrist_e
        if out.dim() == 3:
            t, e = t[:, None, :], e[:, None, :]
        out[..., self._wrist_trans_ids] += t
        out[..., self._wrist_rot_ids] += e
        return out

    def _refn_points(self, pts: torch.Tensor, wrist: torch.Tensor) -> torch.Tensor:
        if not self._refn_enabled:
            return pts
        rel = pts - wrist
        rot = torch.einsum("bij,b...j->b...i", self._refn_wrist_R, rel)
        t = self._refn_wrist_t.view(-1, *([1] * (pts.dim() - 2)), 3)
        return rot + wrist + t

    def _refn_vels(self, vels: torch.Tensor) -> torch.Tensor:
        if not self._refn_enabled:
            return vels
        return vels + self._refn_wrist_v.view(-1, *([1] * (vels.dim() - 2)), 3)

    @property
    def ref_joint_pos_noisy(self) -> torch.Tensor:
        return self._refn_joint(self.ref_joint_pos)

    @property
    def ref_joint_vel_noisy(self) -> torch.Tensor:
        return self._refn_joint(self.ref_joint_vel, vel=True)

    @property
    def next_ref_joint_pos_noisy(self) -> torch.Tensor:
        return self._refn_joint(self.next_ref_joint_pos)

    @property
    def next_ref_joint_vel_noisy(self) -> torch.Tensor:
        return self._refn_joint(self.next_ref_joint_vel, vel=True)

    def future_ref_joint_pos_noisy(self, offsets: torch.Tensor) -> torch.Tensor:
        return self._refn_joint(self.future_ref_joint_pos(offsets))

    def future_ref_joint_vel_noisy(self, offsets: torch.Tensor) -> torch.Tensor:
        return self._refn_joint(self.future_ref_joint_vel(offsets), vel=True)

    @property
    def next_robot_ref_tip_trans_w_noisy(self) -> torch.Tensor:
        w = self.next_mano_wrist_trans_w[:, :, None, :]  # (B, S, 1, 3)
        return self._refn_points(self.next_robot_ref_tip_trans_w, w)

    def next_robot_ref_level_trans_w_noisy(self, side: str, level: int) -> torch.Tensor:
        si = self._side_list.index(side)
        w = self.next_mano_wrist_trans_w[:, si, None, :]  # (B, 1, 3)
        return self._refn_points(self.next_robot_ref_level_trans_w(side, level), w)

    def next_robot_ref_all_joints_trans_w_noisy(self, side: str) -> torch.Tensor:
        si = self._side_list.index(side)
        w = self.next_mano_wrist_trans_w[:, si, None, :]
        return self._refn_points(self.next_robot_ref_all_joints_trans_w(side), w)

    def next_robot_ref_all_joints_lin_vel_w_noisy(self, side: str) -> torch.Tensor:
        return self._refn_vels(self.next_robot_ref_all_joints_lin_vel_w(side))

    @property
    def next_robot_ref_tip_lin_vel_w_noisy(self) -> torch.Tensor:
        return self._refn_vels(self.next_robot_ref_tip_lin_vel_w)

    @property
    def ref_obj_trans_w_noisy(self) -> torch.Tensor:
        if not self._refn_enabled:
            return self.ref_obj_trans_w
        return self.ref_obj_trans_w + self._refn_obj_t

    @property
    def ref_obj_quat_w_noisy(self) -> torch.Tensor:
        if not self._refn_enabled:
            return self.ref_obj_quat_w
        return quat_mul(self._refn_obj_q, self.ref_obj_quat_w)

    @property
    def ref_obj_lin_vel_w_noisy(self) -> torch.Tensor:
        if not self._refn_enabled:
            return self.ref_obj_lin_vel_w
        return self.ref_obj_lin_vel_w + self._refn_obj_v

    @property
    def ref_obj_ang_vel_w_noisy(self) -> torch.Tensor:
        if not self._refn_enabled:
            return self.ref_obj_ang_vel_w
        return self.ref_obj_ang_vel_w + self._refn_obj_av
