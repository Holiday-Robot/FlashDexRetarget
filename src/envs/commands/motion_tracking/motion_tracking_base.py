"""Mixins + setup, reference-state and reset helpers under MotionTrackingCommand."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from mjlab.managers import CommandTerm

from .monitor.debug_vis import DebugVisMixin
from .monitor.metrics import MetricsMixin
from .objects.assist import ObjectAssistMixin
from .objects.multi_object import MultiObjectMixin
from .reference.hand import HandPropertiesMixin
from .reference.noise import RefNoiseMixin
from .reference.object import ObjectPropertiesMixin
from .reset.sampling import SamplingMixin
from .reset.static_offset import StaticOffsetMixin
from .reset.support_disks import SupportDisksMixin
from .motion_tracking_cfg import MotionTrackingCommandCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


class MotionTrackingCommandBase(
    SamplingMixin,
    MetricsMixin,
    RefNoiseMixin,
    ObjectAssistMixin,
    MultiObjectMixin,
    StaticOffsetMixin,
    SupportDisksMixin,
    DebugVisMixin,
    HandPropertiesMixin,
    ObjectPropertiesMixin,
    CommandTerm,
):
    cfg: MotionTrackingCommandCfg
    _env: ManagerBasedRlEnv

    # ══ Setup helpers ══════════════════════════════════════════════════════════════

    def _init_site_ids(self, cfg: MotionTrackingCommandCfg) -> None:
        self._palm_site_ids: dict[str, int] = {}
        self._tip_site_ids: dict[str, list[int]] = {}
        self._contact_site_ids: dict[str, list[int]] = {}
        for side in self.motion_lib.hand_sides:
            self._palm_site_ids[side] = self.robot.site_names.index(
                cfg.site_names["palm"][side]
            )
            self._tip_site_ids[side], _ = self.robot.find_sites(
                cfg.site_names["tip"][side], preserve_order=True
            )
            self._contact_site_ids[side], _ = self.robot.find_sites(
                cfg.site_names["contact"][side], preserve_order=True
            )

    def _init_joint_ids(self, cfg: MotionTrackingCommandCfg) -> None:
        def _to_ids(names: list[str]) -> torch.Tensor:
            ids, _ = self.robot.find_joints_by_actuator_names(names)
            return torch.tensor(ids, dtype=torch.long, device=self.device)

        self._wrist_trans_ids = _to_ids(cfg.joint_names["wrist_trans"])
        self._wrist_rot_ids = _to_ids(cfg.joint_names["wrist_rot"])
        self._finger_joint_ids = _to_ids(cfg.joint_names["finger"])
        # World-aligned x/y wrist slides per side (static-offset co-shift).
        self._wrist_xy_ids: dict[str, torch.Tensor] = {}
        for side in self.motion_lib.hand_sides:
            pre = "R" if side == "right" else "L"
            self._wrist_xy_ids[side] = _to_ids(
                [f"{pre}_forearm_pos_x_joint", f"{pre}_forearm_pos_y_joint"]
            )

    def _init_robot_body_ids(self, cfg: MotionTrackingCommandCfg) -> None:
        all_body_mano = cfg.body_mapping["all"]
        level1_mapping = cfg.body_mapping["level1"]
        level2_mapping = cfg.body_mapping["level2"]

        self._level1_body_ids: dict[str, list[int]] = {}
        self._level2_body_ids: dict[str, list[int]] = {}
        self._all_body_ids: dict[str, list[int]] = {}
        # Optional level3 for 3-phalanx hands (proximal/intermediate/distal).
        mid_mapping = cfg.body_mapping.get("level3")
        self._levelmid_body_ids: dict[str, list[int]] = {}

        for side in self.motion_lib.hand_sides:
            self._level1_body_ids[side] = [
                self.robot.body_names.index(f"{side}_{level1_mapping[f][0]}")
                for f in self.finger_names
            ]
            self._level2_body_ids[side] = [
                self.robot.body_names.index(f"{side}_{level2_mapping[f][0]}")
                for f in self.finger_names
            ]
            self._all_body_ids[side] = [
                self.robot.body_names.index(f"{side}_{rb}") for rb, _ in all_body_mano
            ]
            if mid_mapping:
                self._levelmid_body_ids[side] = [
                    self.robot.body_names.index(f"{side}_{mid_mapping[f][0]}")
                    for f in self.finger_names
                ]

    def _levels(self) -> tuple:
        return (1, 2, 3) if getattr(self, "_levelmid_body_ids", None) else (1, 2)

    def _init_mano_body_ids(self, cfg: MotionTrackingCommandCfg) -> None:
        all_body_mano = cfg.body_mapping["all"]
        level1_mapping = cfg.body_mapping["level1"]
        level2_mapping = cfg.body_mapping["level2"]

        self._level1_mano_ids: dict[str, list[int]] = {}
        self._level2_mano_ids: dict[str, list[int]] = {}
        self._all_mano_ids: dict[str, list[int]] = {}
        mid_mapping = cfg.body_mapping.get("level3")
        self._levelmid_mano_ids: dict[str, list[int]] = {}

        for side in self.motion_lib.hand_sides:
            joint_names = self.motion_lib.mano_joint_names[side]
            self._level1_mano_ids[side] = [
                joint_names.index(level1_mapping[f][1]) for f in self.finger_names
            ]
            self._level2_mano_ids[side] = [
                joint_names.index(level2_mapping[f][1]) for f in self.finger_names
            ]
            self._all_mano_ids[side] = [
                joint_names.index(mj) for _, mj in all_body_mano
            ]
            if mid_mapping:
                self._levelmid_mano_ids[side] = [
                    joint_names.index(mid_mapping[f][1]) for f in self.finger_names
                ]

    # ══ Reference state: clip indexing, reference joints, robot-reference FK ═══════

    @property
    def motion_num_frames(self) -> torch.Tensor:
        return self.motion_lib._motion_num_frames[self.motion_ids]

    @property
    def motion_completed(self) -> torch.Tensor:
        return self.motion_steps >= self.motion_num_frames

    @property
    def _motion_flat_ids(self) -> torch.Tensor:
        return self.motion_lib.length_starts[self.motion_ids] + self.motion_steps

    @property
    def ref_joint_pos(self) -> torch.Tensor:
        return self.motion_lib.robot_joint_pos[self._motion_flat_ids]

    @property
    def ref_joint_vel(self) -> torch.Tensor:
        return self.motion_lib.robot_joint_vel[self._motion_flat_ids]

    @property
    def next_ref_joint_pos(self) -> torch.Tensor:
        return self.motion_lib.robot_joint_pos[self._next_motion_flat_ids()]

    @property
    def next_ref_joint_vel(self) -> torch.Tensor:
        return self.motion_lib.robot_joint_vel[self._next_motion_flat_ids()]

    def future_ref_joint_pos(self, offsets: torch.Tensor) -> torch.Tensor:
        return self.motion_lib.robot_joint_pos[self._future_motion_flat_ids(offsets)]

    def future_ref_joint_vel(self, offsets: torch.Tensor) -> torch.Tensor:
        return self.motion_lib.robot_joint_vel[self._future_motion_flat_ids(offsets)]

    def _compute_robot_ref_state(self) -> None:
        import mujoco

        # ── Standalone robot model ─────────────────────────────────────────────────
        # Entity.compile() fails once attached to the scene: compile a fresh spec.
        model = self.robot.cfg.spec_fn().compile()
        data = mujoco.MjData(model)

        # ── Name -> qpos / site / body / dof ids ───────────────────────────────────
        # Map demo columns by joint name; extra (e.g. mocap) qpos slots stay at rest.
        qadr_by_name: dict[str, int] = {}
        for j in range(model.njnt):
            nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
            if nm is not None:
                qadr_by_name[nm.split("/")[-1]] = int(model.jnt_qposadr[j])
        col_qadr = [qadr_by_name[jn] for jn in self.robot.joint_names]

        site_by_name: dict[str, int] = {}
        for s in range(model.nsite):
            nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, s)
            if nm is not None:
                site_by_name[nm.split("/")[-1]] = s
        tip_ids = {
            side: [site_by_name[n] for n in self.cfg.site_names["tip"][side]]
            for side in self._side_list
        }
        palm_ids = {
            side: site_by_name[self.cfg.site_names["palm"][side]]
            for side in self._side_list
        }

        # xpos (body frame) to match the sim robot_level_trans_w targets.
        body_by_name: dict[str, int] = {}
        for b in range(model.nbody):
            nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b)
            if nm is not None:
                body_by_name[nm.split("/")[-1]] = b
        level1_map = self.cfg.body_mapping["level1"]
        lvl1_body_ids = {
            side: [
                body_by_name[f"{side}_{level1_map[f][0]}"] for f in self.finger_names
            ]
            for side in self._side_list
        }
        all_map = self.cfg.body_mapping["all"]
        all_body_ids = {
            side: [body_by_name[f"{side}_{rb}"] for rb, _ in all_map]
            for side in self._side_list
        }

        dadr_by_name: dict[str, int] = {}
        for j in range(model.njnt):
            nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
            if nm is not None:
                dadr_by_name[nm.split("/")[-1]] = int(model.jnt_dofadr[j])
        col_dadr = [dadr_by_name[jn] for jn in self.robot.joint_names]

        # ── FK over every demo frame ───────────────────────────────────────────────
        qpos = self.motion_lib.robot_joint_pos.detach().cpu().numpy()  # (T, n_jnt)
        qvel = self.motion_lib.robot_joint_vel.detach().cpu().numpy()  # (T, n_jnt)
        T = qpos.shape[0]
        n_sides = len(self._side_list)
        n_fing = len(self.finger_names)
        n_all = len(all_map)
        rest = data.qpos.copy()
        rest_v = data.qvel.copy()
        tips = np.zeros((T, n_sides, n_fing, 3), dtype=np.float32)
        lvl1 = np.zeros((T, n_sides, n_fing, 3), dtype=np.float32)
        allp = np.zeros((T, n_sides, n_all, 3), dtype=np.float32)
        allv = np.zeros((T, n_sides, n_all, 3), dtype=np.float32)
        tipv = np.zeros((T, n_sides, n_fing, 3), dtype=np.float32)
        wpos = np.zeros((T, n_sides, 3), dtype=np.float32)
        wquat = np.zeros((T, n_sides, 4), dtype=np.float32)
        q = np.zeros(4, dtype=np.float64)
        vel6 = np.zeros(6, dtype=np.float64)  # mj_objectVelocity [ang(3), lin(3)]
        BODY = mujoco.mjtObj.mjOBJ_XBODY  # body frame (not COM) world velocity
        SITE = mujoco.mjtObj.mjOBJ_SITE
        for t in range(T):
            data.qpos[:] = rest
            data.qvel[:] = rest_v
            data.qpos[col_qadr] = qpos[t]
            data.qvel[col_dadr] = qvel[t]
            mujoco.mj_fwdPosition(model, data)
            mujoco.mj_fwdVelocity(model, data)
            for si, side in enumerate(self._side_list):
                tips[t, si] = data.site_xpos[tip_ids[side]]
                lvl1[t, si] = data.xpos[lvl1_body_ids[side]]
                allp[t, si] = data.xpos[all_body_ids[side]]
                wpos[t, si] = data.site_xpos[palm_ids[side]]
                mujoco.mju_mat2Quat(q, data.site_xmat[palm_ids[side]])
                wquat[t, si] = q
                for k, bid in enumerate(all_body_ids[side]):
                    mujoco.mj_objectVelocity(model, data, BODY, bid, vel6, 0)
                    allv[t, si, k] = vel6[3:6]
                for k, sid in enumerate(tip_ids[side]):
                    mujoco.mj_objectVelocity(model, data, SITE, sid, vel6, 0)
                    tipv[t, si, k] = vel6[3:6]

        # ── Cache as tensors ───────────────────────────────────────────────────────
        self._robot_ref_tip_pos = torch.tensor(tips, device=self.device)
        self._robot_ref_level1_pos = torch.tensor(lvl1, device=self.device)
        self._robot_ref_all_pos = torch.tensor(allp, device=self.device)
        self._robot_ref_all_lin_vel = torch.tensor(allv, device=self.device)
        self._robot_ref_tip_lin_vel = torch.tensor(tipv, device=self.device)
        self._robot_ref_wrist_pos = torch.tensor(wpos, device=self.device)
        self._robot_ref_wrist_quat = torch.tensor(wquat, device=self.device)

    # ══ Eval / train mode ══════════════════════════════════════════════════════════

    def set_eval_mode(
        self,
        *,
        sampling_mode: str,
        noise_to_initial_level: float,
        start_frame: int,
    ) -> None:
        self._train_sampling_mode = self.cfg.sampling.mode
        self._train_noise_level = self.cfg.hand.noise_to_initial_level
        self._train_start_frame = self.cfg.sampling.start_frame
        self.cfg.sampling.mode = sampling_mode
        self.cfg.hand.noise_to_initial_level = noise_to_initial_level
        self.cfg.sampling.start_frame = start_frame
        self._eval_mode = True
        self._eval_real = True

    def set_train_mode(self) -> None:
        self.cfg.sampling.mode = self._train_sampling_mode
        self.cfg.hand.noise_to_initial_level = self._train_noise_level
        self.cfg.sampling.start_frame = self._train_start_frame
        self._eval_mode = False
        self._eval_real = False
