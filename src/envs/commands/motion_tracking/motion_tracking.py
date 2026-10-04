"""MotionTrackingCommand: the CommandTerm hooks (helpers in motion_tracking_base)."""

from __future__ import annotations

import math
import os
from typing import TYPE_CHECKING

import torch
from mjlab.utils.lab_api.math import quat_apply_inverse, quat_mul

from ..common.motion_library import MotionLibrary
from ..common.raw_contact import register_raw_contact_views
from ..common.sensor_multiplex import register_multi_object_sensor_views
from .reference.noise import _quat_from_rotvec
from .motion_tracking_base import MotionTrackingCommandBase
from .motion_tracking_cfg import MotionTrackingCommandCfg

if TYPE_CHECKING:
    from mjlab.entity import Entity
    from mjlab.envs import ManagerBasedRlEnv


class MotionTrackingCommand(MotionTrackingCommandBase):
    # ══ Setup ══════════════════════════════════════════════════════════════════════

    def __init__(self, cfg: MotionTrackingCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)

        # ── Motion library ─────────────────────────────────────────────────────────
        self.robot: Entity = env.scene[cfg.entity_name]
        self.finger_names = cfg.finger_names
        self.motion_lib = MotionLibrary(
            cfg.motion_file,
            self.robot,
            finger_names=cfg.finger_names,
            device=self.device,
        )
        self._side_list = list(self.motion_lib.hand_sides)

        # ── Index tables, SDF grids, per-feature buffers ───────────────────────────
        self._init_site_ids(cfg)
        self._init_joint_ids(cfg)
        self._init_object_sdf(cfg)
        self._init_robot_body_ids(cfg)
        self._init_mano_body_ids(cfg)
        self._init_sampling(cfg)
        self._init_ref_noise()
        self._init_metrics()
        self._init_object_assist()

        # ── Objects: sensor views, sleep mode, swap ────────────────────────────────
        # Sleep mode: parked objects rest asleep; xfrc must touch only active envs.
        import mujoco as _mj

        self._sleep_mode = bool(
            env.sim.mj_model.opt.enableflags & _mj.mjtEnableBit.mjENBL_SLEEP
        ) if hasattr(_mj.mjtEnableBit, "mjENBL_SLEEP") else False
        if self.multi_object or getattr(cfg.object, "swap_spec", None) is not None:
            if len(self._side_list) > 1:
                raise NotImplementedError(
                    "multi-object motion tracking supports a single hand side"
                )
            if cfg.object.raw_sensor_specs:
                # Raw mode: one reader re-exposes the dropped object sensors.
                register_raw_contact_views(
                    env, self, self._side_list[0], cfg.object.raw_sensor_specs
                )
            elif self.multi_object:
                register_multi_object_sensor_views(
                    env, lambda: self.active_obj_slot(self._side_list[0])
                )
        # FDR_OBJ_SWAP: per-world model rows select each env's object.
        self._swap_tables: dict[str, torch.Tensor] | None = None
        if self.has_objects and getattr(cfg.object, "swap_spec", None) is not None:
            self._init_object_swap(cfg)

        # ── Eval flag + lazy caches ────────────────────────────────────────────────
        self._eval_mode = False
        self._ghost_model = None
        # Lazy FK of the retargeted-robot demo (_compute_robot_ref_state).
        self._robot_ref_tip_pos: torch.Tensor | None = None
        self._robot_ref_wrist_pos: torch.Tensor | None = None
        self._robot_ref_wrist_quat: torch.Tensor | None = None
        self._robot_ref_level1_pos: torch.Tensor | None = None
        self._robot_ref_all_pos: torch.Tensor | None = None
        self._robot_ref_all_lin_vel: torch.Tensor | None = None
        self._robot_ref_tip_lin_vel: torch.Tensor | None = None

    # ══ Command ════════════════════════════════════════════════════════════════════

    @property
    def command(self) -> torch.Tensor:
        return torch.cat([self.ref_joint_pos, self.ref_joint_vel], dim=1)

    # ══ Reset: clip draw, start frame, warm start, object seeding ══════════════════

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        # ── Clip + start frame ─────────────────────────────────────────────────────
        _pst = getattr(self, "_perturb_state", None)
        if _pst is not None:
            for _s in _pst.values():
                _s["active"][env_ids] = False

        # Eval keeps its assigned motion_ids.
        if not self._eval_mode and self.motion_lib.num_trajectories > 1:
            self.motion_ids[env_ids] = torch.randint(
                0,
                self.motion_lib.num_trajectories,
                (len(env_ids),),
                device=self.device,
                dtype=torch.long,
            )

        if self.cfg.sampling.mode == "start":
            sf = int(self.cfg.sampling.start_frame)
            if sf < 0:
                raise ValueError(f"sampling.start_frame must be >= 0, got {sf}")
            t_m = self.motion_lib._motion_num_frames[self.motion_ids[env_ids]]
            self.motion_steps[env_ids] = torch.minimum(
                torch.full_like(t_m, sf), t_m - 1
            )
        elif self.cfg.sampling.mode == "uniform":
            self._uniform_sampling(env_ids)
        else:
            raise ValueError(
                f"sampling.mode must be 'uniform' or 'start', got {self.cfg.sampling.mode!r}"
            )

        # ── Robot warm start: ref noise, static offset, init noise ─────────────────
        self._resample_ref_noise(env_ids)

        # Not an EventTermCfg: mjlab events fire before the command reset.
        joint_pos = self.ref_joint_pos[env_ids].clone()
        joint_vel = self.ref_joint_vel[env_ids].clone()

        static_off = self._sample_static_offset(env_ids)  # {side: (N, 3)} or {}
        for side, off in static_off.items():
            joint_pos[:, self._wrist_xy_ids[side]] += off[:, :2]

        soft_limits = self.robot.data.soft_joint_pos_limits[env_ids]
        mult = float(self.cfg.hand.noise_to_initial_level)
        ns = self.cfg.hand.init_noise_scale
        N = len(env_ids)

        def _randn(n_dofs: int) -> torch.Tensor:
            return torch.randn(N, n_dofs, device=self.device)

        wrist_trans_sigma = float(ns.get("wrist_trans", 0.0)) * mult
        wrist_rot_sigma = math.radians(float(ns.get("wrist_rot_deg", 0.0))) * mult
        finger_range_frac = float(ns.get("finger_range_frac", 0.0)) * mult
        wrist_trans_vel_sigma = float(ns.get("wrist_trans_vel", 0.0)) * mult
        wrist_rot_vel_sigma = float(ns.get("wrist_rot_vel", 0.0)) * mult
        finger_vel_sigma = float(ns.get("finger_vel", 0.0)) * mult

        joint_pos[:, self._wrist_trans_ids] += (
            _randn(len(self._wrist_trans_ids)) * wrist_trans_sigma
        )
        joint_pos[:, self._wrist_rot_ids] += (
            _randn(len(self._wrist_rot_ids)) * wrist_rot_sigma
        )
        finger_range = (
            soft_limits[:, self._finger_joint_ids, 1]
            - soft_limits[:, self._finger_joint_ids, 0]
        )
        joint_pos[:, self._finger_joint_ids] += _randn(len(self._finger_joint_ids)) * (
            finger_range * finger_range_frac
        )
        # Ref-noise RSI, scaled by the depenetration alpha.
        if self._refn_enabled:
            a = self._refn_alpha[env_ids][:, None]
            joint_pos[:, self._wrist_trans_ids] += self._refn_wrist_t[env_ids] * a
            joint_pos[:, self._wrist_rot_ids] += self._refn_wrist_e[env_ids] * a
        joint_pos = torch.clip(joint_pos, soft_limits[:, :, 0], soft_limits[:, :, 1])

        joint_vel[:, self._wrist_trans_ids] += (
            _randn(len(self._wrist_trans_ids)) * wrist_trans_vel_sigma
        )
        joint_vel[:, self._wrist_rot_ids] += (
            _randn(len(self._wrist_rot_ids)) * wrist_rot_vel_sigma
        )
        # Finger velocity replaces the reference (sigma=0 keeps it).
        if finger_vel_sigma > 0.0:
            joint_vel[:, self._finger_joint_ids] = (
                _randn(len(self._finger_joint_ids)) * finger_vel_sigma
            )
        if self._refn_enabled:
            a = self._refn_alpha[env_ids][:, None]
            joint_vel[:, self._wrist_trans_ids] += self._refn_wrist_v[env_ids] * a
            joint_vel[:, self._wrist_rot_ids] += self._refn_wrist_av[env_ids] * a

        if self.cfg.hand.zero_init_vel:
            joint_vel = torch.zeros_like(joint_vel)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self.robot.reset(env_ids=env_ids)

        # ── Object seeding ─────────────────────────────────────────────────────────
        # Always seed the object from the reference, else obj.reset uses the spawn pose.
        if self.has_objects:
            for side in self._side_list:
                if side not in self.motion_lib.obj_trans:
                    continue
                si_r = self._side_list.index(side)
                obj_trans = self.ref_obj_trans_w[env_ids][:, si_r]  # (N, 3)
                if side in static_off:
                    obj_trans = obj_trans + static_off[side]
                obj_quat = self.ref_obj_quat_w[env_ids][:, si_r]  # (N, 4)
                obj_lin_vel = self.ref_obj_lin_vel_w[env_ids][:, si_r]
                obj_ang_vel = self.motion_lib.obj_ang_vel[side][
                    self.motion_lib.length_starts[self.motion_ids[env_ids]]
                    + self.motion_steps[env_ids]
                ]
                if self._refn_enabled:
                    a1 = self._refn_alpha[env_ids][:, None]
                    obj_trans = obj_trans + self._refn_obj_t[env_ids][:, si_r] * a1
                    obj_quat = quat_mul(
                        _quat_from_rotvec(
                            self._refn_obj_rotvec[env_ids][:, si_r] * a1
                        ),
                        obj_quat,
                    )
                    obj_lin_vel = obj_lin_vel + self._refn_obj_v[env_ids][:, si_r] * a1
                    obj_ang_vel = (
                        obj_ang_vel + self._refn_obj_av[env_ids][:, si_r] * a1
                    )
                if self.cfg.object.zero_init_vel:
                    obj_lin_vel = torch.zeros_like(obj_lin_vel)
                    obj_ang_vel = torch.zeros_like(obj_ang_vel)
                root_state = torch.cat(
                    [obj_trans, obj_quat, obj_lin_vel, obj_ang_vel], dim=-1
                )
                names = self.obj_entity_names(side)
                if len(names) == 1:
                    if self._swap_tables is not None:
                        # Retarget geoms/inertia to each env's object before seeding.
                        self._write_swap_rows(
                            env_ids, self.active_obj_slot(side)[env_ids]
                        )
                    obj: Entity = self._env.scene[names[0]]
                    alias = self._obj_alias_to_right(side)
                    if alias is not None:  # mixed pair pt: leave the shared envs' ghost parked
                        keep = ~alias[env_ids]
                        if bool(keep.any()):
                            obj.write_root_state_to_sim(
                                root_state[keep], env_ids=env_ids[keep]
                            )
                    else:
                        obj.write_root_state_to_sim(root_state, env_ids=env_ids)
                    obj.reset(env_ids=env_ids)
                else:
                    # Active slot gets the reference pose, every other slot parks.
                    slot = self.active_obj_slot(side)[env_ids]
                    bufs = self._parked_write_buffers(side)
                    if bufs is None:
                        self._seed_objects_loop(side, names, env_ids, root_state, slot)
                    else:
                        q_adr, v_adr, parked_qpos, s_ar, body_ids = bufs
                        data = self._env.scene[names[0]].data.data
                        rows3 = env_ids[:, None, None]
                        ar = torch.arange(env_ids.shape[0], device=self.device)
                        mask_p = (slot[:, None] != s_ar[None, :])[..., None]
                        new_q = parked_qpos[env_ids].clone()
                        new_q[ar, slot] = root_state[:, 0:7]
                        data.qpos[rows3, q_adr[None]] = new_q
                        # Body-frame angular velocity, as in write_root_velocity.
                        ang_b = quat_apply_inverse(
                            root_state[:, 3:7], root_state[:, 10:13]
                        )
                        new_v = root_state.new_zeros(env_ids.shape[0], len(names), 6)
                        new_v[ar, slot] = torch.cat(
                            [root_state[:, 7:10], ang_b], dim=-1
                        )
                        data.qvel[rows3, v_adr[None]] = new_v
                        # xfrc persists: zero the parked slots' wrench.
                        cur_w = data.xfrc_applied[env_ids[:, None, None], body_ids[None]]
                        data.xfrc_applied[env_ids[:, None, None], body_ids[None]] = (
                            torch.where(
                                mask_p[..., None], torch.zeros_like(cur_w), cur_w
                            )
                        )
                    if self._sleep_mode:
                        # keep the awake-DOF cap valid (see _sleep_park_trees)
                        self._sleep_park_trees(side, env_ids, slot)

        # ── Scene bookkeeping: support disks, caches, island Newton ────────────────
        self._write_support_disks(env_ids)

        # Bump so _sim_obj_field drops its cache (mid-step rewrite, same step counter).
        self._obj_state_version = getattr(self, "_obj_state_version", 0) + 1

    # ══ Step ═══════════════════════════════════════════════════════════════════════

    def _update_command(self) -> None:
        self.motion_steps += 1

        if self._eval_mode:
            max_frames = self.motion_lib._motion_num_frames[self.motion_ids]
            self.motion_steps.clamp_(max=max_frames - 1)
        else:
            wrap_ids = torch.where(
                self.motion_steps >= self.motion_lib._motion_num_frames[self.motion_ids]
            )[0]
            if wrap_ids.numel() > 0:
                self._resample_command(wrap_ids)

        # Sleep mode leaves parked objects asleep; re-teleporting would wake them.
        if self.has_objects and self.multi_object and not self._sleep_mode:
            self._repark_inactive_objects()

        # Curricula fire only on resets; step the assist controller per env step.
        if self._xfrc_curr_ctrl is not None:
            self._xfrc_curr_ctrl.on_step(self._env)

        self._apply_object_assist()
