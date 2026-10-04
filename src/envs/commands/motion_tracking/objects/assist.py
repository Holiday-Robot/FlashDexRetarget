"""MotionTrackingCommand mixin: xfrc object assist + grasp perturbation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.utils.lab_api.math import (
    axis_angle_from_quat,
    matrix_from_quat,
    quat_conjugate,
    quat_mul,
)

if TYPE_CHECKING:
    from mjlab.entity import Entity


class ObjectAssistMixin:
    def _init_object_assist(self) -> None:
        # Per-env assist gains from the curriculum; None = scalar cfg gains.
        self._xfrc_kp_pos_env: torch.Tensor | None = None
        self._xfrc_kv_pos_env: torch.Tensor | None = None
        self._xfrc_kp_rot_env: torch.Tensor | None = None
        self._xfrc_kv_rot_env: torch.Tensor | None = None
        self._xfrc_force_range: float | None = None
        # Assist curriculum controller (set by xfrc_curriculum_adaptive).
        self._xfrc_curr_ctrl = None
        self._obj_I_body_cache: dict[str, torch.Tensor] = {}

    def apply_substep(self) -> None:
        """Per-physics-substep hook (mjlab_ext._env): re-evaluate the assist PD."""
        if self.cfg.object.pin_objects and self.has_objects:
            self._apply_object_assist()

    def _apply_object_assist(self) -> None:
        """Assist wrench from the current object state; also run per substep."""
        if self.cfg.object.pin_objects and self.has_objects:
            pin_mode = self.cfg.object.pin_mode
            if pin_mode == "xfrc":
                # ── Gains ──────────────────────────────────────────────────────────
                # World-frame xfrc wrench; gravity stays on, the PD carries the weight.
                xfrc = self.cfg.object.xfrc
                kp_pos = xfrc.kp.pos
                kd_pos = xfrc.kd.pos
                kp_rot = xfrc.kp.rot
                kd_rot = xfrc.kd.rot
                # Curriculum gains override the scalars: (B,) or per-axis (B, 3).
                def _as_col(t: torch.Tensor) -> torch.Tensor:
                    return t.reshape(t.shape[0], -1).to(self.device)

                if self._xfrc_kp_pos_env is not None:
                    kp_pos = _as_col(self._xfrc_kp_pos_env)
                if self._xfrc_kv_pos_env is not None:
                    kd_pos = _as_col(self._xfrc_kv_pos_env)
                if self._xfrc_kp_rot_env is not None:
                    kp_rot = _as_col(self._xfrc_kp_rot_env)
                if self._xfrc_kv_rot_env is not None:
                    kd_rot = _as_col(self._xfrc_kv_rot_env)
                force_range = self._xfrc_force_range

                # ── Object vs reference state ──────────────────────────────────────
                sim_trans_all = self.sim_obj_trans_w  # (B, n_sides, 3)
                sim_quat_all = self.sim_obj_quat_w  # (B, n_sides, 4)
                sim_lin_vel_all = self.sim_obj_lin_vel_w  # (B, n_sides, 3)
                sim_ang_vel_all = self.sim_obj_ang_vel_w  # (B, n_sides, 3)
                ref_trans_all = self.ref_obj_trans_w  # (B, n_sides, 3)
                ref_quat_all = self.ref_obj_quat_w  # (B, n_sides, 4)
                # shared object: both sides alias one entity -> write its wrench once
                assisted: set[tuple[str, ...]] = set()
                for side in self._side_list:
                    if side not in self.motion_lib.obj_trans:
                        continue
                    ent_key = tuple(self.obj_entity_names(side))
                    if ent_key in assisted:
                        continue
                    assisted.add(ent_key)
                    si = self._side_list.index(side)
                    obj_names = self.obj_entity_names(side)

                    ref_trans = ref_trans_all[:, si]
                    ref_quat = ref_quat_all[:, si]
                    ref_lin_vel = self.motion_lib.obj_lin_vel[side][
                        self._motion_flat_ids
                    ]
                    ref_ang_vel = self.motion_lib.obj_ang_vel[side][
                        self._motion_flat_ids
                    ]
                    sim_trans = sim_trans_all[:, si]
                    sim_quat = sim_quat_all[:, si]
                    sim_lin_vel = sim_lin_vel_all[:, si]
                    sim_ang_vel = sim_ang_vel_all[:, si]
                    if not xfrc.vel_feedforward:
                        ref_lin_vel = torch.zeros_like(sim_lin_vel)
                        ref_ang_vel = torch.zeros_like(sim_ang_vel)

                    # ── Linear PD ──────────────────────────────────────────────────
                    force = kp_pos * (ref_trans - sim_trans) + kd_pos * (
                        ref_lin_vel - sim_lin_vel
                    )

                    # ── Rotational PD ──────────────────────────────────────────────
                    delta_q = quat_mul(ref_quat, quat_conjugate(sim_quat))
                    axis_angle = axis_angle_from_quat(delta_q)

                    # Both paths map an angular acceleration to torque via I_world.
                    I_body = self._obj_inertia_body(side).to(dtype=sim_quat.dtype)
                    if I_body.dim() == 3:  # multi-object: (S, 3, 3) → (B, 3, 3)
                        I_body = I_body[self.active_obj_slot(side)]
                    R_sim = matrix_from_quat(sim_quat)  # (B, 3, 3)
                    I_world = R_sim @ I_body @ R_sim.transpose(-1, -2)  # (B, 3, 3)

                    omega_rot = float(xfrc.omega_rot)
                    if omega_rot > 0.0:
                        zeta_rot = float(xfrc.zeta_rot)
                        s_rot = self._contact_assist_scale(side)
                        w = omega_rot if s_rot is None else omega_rot * s_rot
                        kp_rot_eff = w * w
                        kd_rot_eff = 2.0 * zeta_rot * w
                        ang_err = ref_ang_vel - sim_ang_vel
                        torque = kp_rot_eff * torch.einsum(
                            "bij,bj->bi", I_world, axis_angle
                        ) + kd_rot_eff * torch.einsum("bij,bj->bi", I_world, ang_err)
                    else:
                        # Scaling an equivalent frequency by s means kp *= s^2, kd *= s.
                        s_rot = self._contact_assist_scale(side)
                        if s_rot is not None:
                            kp_rot = kp_rot * s_rot * s_rot
                            kd_rot = kd_rot * s_rot
                        ang_accel = kp_rot * axis_angle + kd_rot * (ref_ang_vel - sim_ang_vel)
                        if xfrc.rot_inertia_scale:
                            torque = torch.einsum("bij,bj->bi", I_world, ang_accel)
                        else:
                            torque = ang_accel

                    # ── Clamp, implicit damping, perturbation ──────────────────────
                    if force_range is not None:
                        force = force.clamp(-force_range, force_range)
                        torque = torque.clamp(-force_range, force_range)
                    if xfrc.implicit_damping:
                        force, torque = self._implicit_damping_wrench(
                            side, force, torque, kd_pos, kd_rot, I_world
                        )

                    # Perturbation goes after the assist clamp.
                    force, torque = self._maybe_apply_object_perturb(
                        side, force, torque
                    )

                    # ── Write the wrench ───────────────────────────────────────────
                    if len(obj_names) == 1:
                        alias = self._obj_alias_to_right(side)
                        if alias is not None:  # mixed pt: the right write holds the shared object
                            m = (~alias).to(force.dtype).unsqueeze(-1)
                            force, torque = force * m, torque * m
                        obj: Entity = self._env.scene[obj_names[0]]
                        obj.write_external_wrench_to_sim(
                            forces=force.unsqueeze(1),
                            torques=torque.unsqueeze(1),
                        )
                    else:
                        # xfrc persists: zero inactive slots (sleep: active envs only).
                        slot = self.active_obj_slot(side)
                        for s, name in enumerate(obj_names):
                            if self._sleep_mode:
                                ids = torch.where(slot == s)[0]
                                if ids.numel() == 0:
                                    continue
                                self._env.scene[name].write_external_wrench_to_sim(
                                    forces=force[ids].unsqueeze(1),
                                    torques=torque[ids].unsqueeze(1),
                                    env_ids=ids,
                                )
                            else:
                                m = (slot == s).to(force.dtype).unsqueeze(-1)
                                self._env.scene[name].write_external_wrench_to_sim(
                                    forces=(force * m).unsqueeze(1),
                                    torques=(torque * m).unsqueeze(1),
                                )
            elif pin_mode == "none":
                pass
            else:
                raise ValueError(
                    f"unknown pin_mode={pin_mode!r}; expected 'xfrc' or 'none'"
                )

    def _contact_assist_scale(self, side: str):
        """(B, 1) contact_scale while any hand link touches the object; None if off."""
        scale = float(getattr(self.cfg.object.xfrc, "contact_scale", 1.0))
        if scale >= 1.0:
            return None
        sensor = self._env.scene[f"{side[0]}_alllink_contact"]
        force = getattr(sensor.data, "force", None)
        if force is None:
            return None
        touching = (force.reshape(force.shape[0], -1).abs() > 1e-6).any(dim=-1, keepdim=True)
        return torch.where(touching, torch.full_like(touching, scale, dtype=torch.float32),
                           torch.ones_like(touching, dtype=torch.float32)).to(self.device)

    def _implicit_damping_wrench(self, side, force, torque, kd_pos, kd_rot, I_world):
        """Genesis implicit kv: wrench *= M (M + dt kv)^-1."""
        dt = float(self._env.physics_dt)
        B = force.shape[0]

        def _bcast(k):
            k = torch.as_tensor(k, device=force.device, dtype=force.dtype)
            return k.reshape(1 if k.dim() == 0 else k.shape[0], -1).expand(B, 3)

        m = self._obj_mass_env(side).to(force.dtype).unsqueeze(-1)  # (B, 1)
        force = force * m / (m + dt * _bcast(kd_pos))
        if bool(self.cfg.object.xfrc.rot_inertia_scale):
            # kd_rot is an angular-accel gain here (I already in torque): scalar factor.
            torque = torque / (1.0 + dt * _bcast(kd_rot))
        else:
            A = I_world + dt * torch.diag_embed(_bcast(kd_rot))  # (B, 3, 3)
            torque = torch.linalg.solve(A, I_world @ torque.unsqueeze(-1)).squeeze(-1)
        return force, torque

    def _obj_inertia_body(self, side: str) -> torch.Tensor:
        cached = self._obj_I_body_cache.get(side)
        if cached is not None:
            return cached
        mats = []
        for name in self.obj_entity_names(side):
            obj: Entity = self._env.scene[name]
            body_id = int(obj.indexing.body_ids[0])
            I_diag_p = self._env.sim.model.body_inertia[0, body_id].to(
                self.device, dtype=torch.float32
            )  # (3,) principal-frame diagonal
            iquat = self._env.sim.model.body_iquat[0, body_id].to(
                self.device, dtype=torch.float32
            )  # (4,) principal→body
            R_ip = matrix_from_quat(iquat.unsqueeze(0))[0]  # (3, 3)
            mats.append(R_ip @ torch.diag(I_diag_p) @ R_ip.T)
        out = mats[0] if len(mats) == 1 else torch.stack(mats)
        self._obj_I_body_cache[side] = out
        return out

    def _obj_mass_env(self, side: str) -> torch.Tensor:
        cache = getattr(self, "_perturb_mass_cache", None)
        if cache is None:
            cache = {}
            self._perturb_mass_cache = cache
        m = cache.get(side)
        if m is None:
            tables = getattr(self, "_swap_tables", None)
            if tables is not None and "bmass" in tables:
                m = tables["bmass"].to(self.device, dtype=torch.float32)
            else:
                vals = []
                for name in self.obj_entity_names(side):
                    obj: Entity = self._env.scene[name]
                    bid = int(obj.indexing.body_ids[0])
                    # Batched (1, nbody) table on both backends; isaac has no mj_model.
                    vals.append(float(self._env.sim.model.body_mass[0, bid]))
                m = torch.tensor(vals, device=self.device)
            cache[side] = m
        if m.numel() == 1:
            return m.expand(self._env.num_envs)
        return m[self.active_obj_slot(side)]

    def _perturb_gate_flat(self, side: str) -> torch.Tensor:
        """(F_total,) frames with the reference object lifted, eroded by gate_lag_s."""
        cache = getattr(self, "_perturb_gate_cache", None)
        if cache is None:
            cache = {}
            self._perturb_gate_cache = cache
        g = cache.get(side)
        if g is not None:
            return g
        import numpy as np
        from scipy.ndimage import minimum_filter1d

        p = self.cfg.object.perturb
        z = self.motion_lib.obj_trans[side][:, 2].detach().cpu().numpy()
        starts = self.motion_lib.length_starts.detach().cpu().numpy()
        lens = self.motion_lib._motion_num_frames.detach().cpu().numpy()
        lag = max(1, int(round(p.gate_lag_s / self._env.step_dt)))
        mask = np.zeros(z.shape[0], dtype=np.uint8)
        for s0, ln in zip(starts, lens):
            seg = z[s0 : s0 + ln]
            m = (seg - seg[0] > p.height_thresh).astype(np.uint8)
            if lag > 1:
                m = minimum_filter1d(m, size=2 * lag - 1, mode="nearest")
            mask[s0 : s0 + ln] = m
        g = torch.from_numpy(mask.astype(bool)).to(self.device)
        cache[side] = g
        return g

    def _maybe_apply_object_perturb(
        self, side: str, force: torch.Tensor, torque: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        p = self.cfg.object.perturb
        if not p.enabled or self._eval_mode:
            return force, torque
        B = self._env.num_envs
        dev = self.device
        if p.require_assist_decayed:
            kp_env = self._xfrc_kp_pos_env
            if kp_env is not None:
                assist_off = kp_env.reshape(B, -1).amax(dim=1).to(dev) <= 0.0
            else:
                assist_off = torch.full(
                    (B,),
                    float(self.cfg.object.xfrc.kp.pos) <= 0.0,
                    dtype=torch.bool,
                    device=dev,
                )
        else:
            assist_off = torch.ones(B, dtype=torch.bool, device=dev)
        gate = self._perturb_gate_flat(side)[self._motion_flat_ids] & assist_off

        st = getattr(self, "_perturb_state", None)
        if st is None:
            st = {}
            self._perturb_state = st
        s = st.get(side)
        if s is None:
            s = {
                "active": torch.zeros(B, dtype=torch.bool, device=dev),
                "force": torch.zeros(B, 3, device=dev),
                "torque": torch.zeros(B, 3, device=dev),
            }
            st[side] = s

        starting = (torch.rand(B, device=dev) < p.start_prob) & gate
        continuing = (
            s["active"]
            & ~starting
            & (torch.rand(B, device=dev) < p.continue_prob)
            & gate
        )
        if bool(starting.any()):
            f_dir = torch.randn(B, 3, device=dev)
            f_dir = f_dir / (f_dir.norm(dim=1, keepdim=True) + 1e-8)
            f_mag = (
                torch.rand(B, 1, device=dev)
                * p.force_scale
                * self._obj_mass_env(side).reshape(-1, 1)
                * 9.81
            )
            t_dir = torch.randn(B, 3, device=dev)
            t_dir = t_dir / (t_dir.norm(dim=1, keepdim=True) + 1e-8)
            t_mag = torch.rand(B, 1, device=dev) * p.torque_scale
            sc = starting.reshape(-1, 1)
            s["force"] = torch.where(sc, f_dir * f_mag, s["force"])
            s["torque"] = torch.where(sc, t_dir * t_mag, s["torque"])
        active = starting | continuing
        s["active"] = active
        am = active.to(force.dtype).reshape(-1, 1)
        return force + s["force"] * am, torque + s["torque"] * am
