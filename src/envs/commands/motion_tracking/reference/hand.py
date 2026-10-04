"""Hand (wrist/tip/joint) property mixin for MotionTrackingCommand: MANO reference
and robot state properties for wrist, fingertip, and body-level joint tracking."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.utils.lab_api.math import matrix_from_quat, quat_from_matrix

if TYPE_CHECKING:
    pass


# Canonical order of the 13 collision-link bodies per side: the demo contact data
# (contact_alllink_*_<side>) and the r_/l_alllink_contact_pos sensors MUST match it.
def _alllink_names(prefix: str, side: str) -> list[str]:
    return [
        f"{prefix}_forearm_rot_y_link",  # palm
        f"{side}_hand_thumb_bend_link",
        f"{side}_hand_thumb_rota_link1",
        f"{side}_hand_thumb_rota_link2",
        f"{side}_hand_index_bend_link",
        f"{side}_hand_index_rota_link1",
        f"{side}_hand_index_rota_link2",
        f"{side}_hand_mid_link1",
        f"{side}_hand_mid_link2",
        f"{side}_hand_ring_link1",
        f"{side}_hand_ring_link2",
        f"{side}_hand_pinky_link1",
        f"{side}_hand_pinky_link2",
    ]


ALLLINK_BODY_NAMES_BY_SIDE = {
    "right": _alllink_names("R", "right"),
    "left": _alllink_names("L", "left"),
}
# Back-compat alias for right-only call sites.
ALLLINK_BODY_NAMES = ALLLINK_BODY_NAMES_BY_SIDE["right"]


class HandPropertiesMixin:
    """MANO reference + robot state properties for hand tracking."""

    # --- MANO wrist ---

    @property
    def mano_wrist_trans_w(self) -> torch.Tensor:
        """MANO wrist positions. Shape: (B, n_sides, 3)."""
        parts = []
        for side in self._side_list:
            pos = self.motion_lib.mano_wrist_trans[side][
                self._motion_flat_ids
            ]  # (B, 3)
            pos = pos + self._env.scene.env_origins
            parts.append(pos)
        return torch.stack(parts, dim=1)

    @property
    def mano_wrist_rot_w(self) -> torch.Tensor:
        """MANO wrist rotation matrices. Shape: (B, n_sides, 3, 3)."""
        parts = []
        for side in self._side_list:
            parts.append(self.motion_lib.mano_wrist_rot[side][self._motion_flat_ids])
        return torch.stack(parts, dim=1)

    @property
    def mano_wrist_quat_w(self) -> torch.Tensor:
        """MANO wrist quaternions (from rotmat). Shape: (B, n_sides, 4)."""
        return quat_from_matrix(self.mano_wrist_rot_w)

    @property
    def mano_wrist_lin_vel_w(self) -> torch.Tensor:
        """MANO wrist linear velocities. Shape: (B, n_sides, 3)."""
        parts = []
        for side in self._side_list:
            parts.append(
                self.motion_lib.mano_wrist_lin_vel[side][self._motion_flat_ids]
            )
        return torch.stack(parts, dim=1)

    @property
    def mano_wrist_ang_vel_w(self) -> torch.Tensor:
        """MANO wrist angular velocities. Shape: (B, n_sides, 3)."""
        parts = []
        for side in self._side_list:
            parts.append(
                self.motion_lib.mano_wrist_ang_vel[side][self._motion_flat_ids]
            )
        return torch.stack(parts, dim=1)

    # --- MANO next-frame wrist (1-step lookahead; for own-wrist canonical refs) ---

    @property
    def next_mano_wrist_trans_w(self) -> torch.Tensor:
        """Next-frame MANO wrist positions. Shape: (B, n_sides, 3)."""
        nfi = self._next_motion_flat_ids()
        parts = []
        for side in self._side_list:
            pos = self.motion_lib.mano_wrist_trans[side][nfi]  # (B, 3)
            pos = pos + self._env.scene.env_origins
            parts.append(pos)
        return torch.stack(parts, dim=1)

    @property
    def next_mano_wrist_rot_w(self) -> torch.Tensor:
        """Next-frame MANO wrist rotation matrices. Shape: (B, n_sides, 3, 3)."""
        nfi = self._next_motion_flat_ids()
        parts = []
        for side in self._side_list:
            parts.append(self.motion_lib.mano_wrist_rot[side][nfi])
        return torch.stack(parts, dim=1)

    @property
    def next_mano_wrist_quat_w(self) -> torch.Tensor:
        """Next-frame MANO wrist quaternions (from rotmat). Shape: (B, n_sides, 4)."""
        return quat_from_matrix(self.next_mano_wrist_rot_w)

    @property
    def next_mano_wrist_lin_vel_w(self) -> torch.Tensor:
        """Next-frame MANO wrist linear velocities. Shape: (B, n_sides, 3)."""
        nfi = self._next_motion_flat_ids()
        parts = []
        for side in self._side_list:
            parts.append(self.motion_lib.mano_wrist_lin_vel[side][nfi])
        return torch.stack(parts, dim=1)

    # --- MANO fingertips ---

    @property
    def mano_tip_trans_w(self) -> torch.Tensor:
        """MANO fingertip positions (5 per side). Shape: (B, n_sides, 5, 3)."""
        parts = []
        for side in self._side_list:
            all_joints = self.motion_lib.mano_joint_pos[side][
                self._motion_flat_ids
            ]  # (B, 20, 3)
            tips = all_joints[:, self.motion_lib.tip_ids[side]]  # (B, 5, 3)
            tips = tips + self._env.scene.env_origins[:, None, :]
            parts.append(tips)
        return torch.stack(parts, dim=1)

    @property
    def mano_tip_lin_vel_w(self) -> torch.Tensor:
        """MANO fingertip velocities (5 per side). Shape: (B, n_sides, 5, 3)."""
        parts = []
        for side in self._side_list:
            all_vel = self.motion_lib.mano_joint_vel[side][
                self._motion_flat_ids
            ]  # (B, 20, 3)
            tips = all_vel[:, self.motion_lib.tip_ids[side]]  # (B, 5, 3)
            parts.append(tips)
        return torch.stack(parts, dim=1)

    # --- MANO body-level joints ---

    def mano_all_joints_trans_w(self, side: str) -> torch.Tensor:
        """MANO positions for all 12 non-tip joints. Shape: (B, 12, 3)."""
        mano_ids = self._all_mano_ids[side]
        all_joints = self.motion_lib.mano_joint_pos[side][
            self._motion_flat_ids
        ]  # (B, 20, 3)
        pts = all_joints[:, mano_ids]  # (B, 12, 3)
        return pts + self._env.scene.env_origins[:, None, :]

    def mano_all_joints_lin_vel_w(self, side: str) -> torch.Tensor:
        """MANO velocities for all 12 non-tip joints. Shape: (B, 12, 3)."""
        mano_ids = self._all_mano_ids[side]
        all_vel = self.motion_lib.mano_joint_vel[side][
            self._motion_flat_ids
        ]  # (B, 20, 3)
        return all_vel[:, mano_ids]  # (B, 12, 3)

    def mano_level_trans_w(self, side: str, level: int) -> torch.Tensor:
        """MANO joint positions for level 1 or 2. Shape: (B, 5, 3)."""
        mano_ids = {1: self._level1_mano_ids, 2: self._level2_mano_ids,
                    3: self._levelmid_mano_ids}[level][side]
        all_joints = self.motion_lib.mano_joint_pos[side][
            self._motion_flat_ids
        ]  # (B, 20, 3)
        pts = all_joints[:, mano_ids]  # (B, 5, 3)
        return pts + self._env.scene.env_origins[:, None, :]

    # --- MANO next-frame keypoints (1-step lookahead) ---

    @property
    def next_mano_tip_trans_w(self) -> torch.Tensor:
        """Next-frame MANO fingertip positions (5 per side). Shape: (B, n_sides, 5, 3)."""
        nfi = self._next_motion_flat_ids()
        parts = []
        for side in self._side_list:
            all_joints = self.motion_lib.mano_joint_pos[side][nfi]  # (B, 20, 3)
            tips = all_joints[:, self.motion_lib.tip_ids[side]]  # (B, 5, 3)
            tips = tips + self._env.scene.env_origins[:, None, :]
            parts.append(tips)
        return torch.stack(parts, dim=1)

    def next_mano_level_trans_w(self, side: str, level: int) -> torch.Tensor:
        """Next-frame MANO joint positions for level 1 or 2. Shape: (B, 5, 3)."""
        nfi = self._next_motion_flat_ids()
        mano_ids = {1: self._level1_mano_ids, 2: self._level2_mano_ids,
                    3: self._levelmid_mano_ids}[level][side]
        all_joints = self.motion_lib.mano_joint_pos[side][nfi]  # (B, 20, 3)
        pts = all_joints[:, mano_ids]  # (B, 5, 3)
        return pts + self._env.scene.env_origins[:, None, :]

    @property
    def next_mano_tip_lin_vel_w(self) -> torch.Tensor:
        """Next-frame MANO fingertip velocities (5 per side). Shape: (B, n_sides, 5, 3)."""
        nfi = self._next_motion_flat_ids()
        parts = []
        for side in self._side_list:
            all_vel = self.motion_lib.mano_joint_vel[side][nfi]  # (B, 20, 3)
            tips = all_vel[:, self.motion_lib.tip_ids[side]]  # (B, 5, 3)
            parts.append(tips)
        return torch.stack(parts, dim=1)

    def next_mano_all_joints_trans_w(self, side: str) -> torch.Tensor:
        """Next-frame MANO positions for all 12 non-tip joints. Shape: (B, 12, 3)."""
        nfi = self._next_motion_flat_ids()
        mano_ids = self._all_mano_ids[side]
        all_joints = self.motion_lib.mano_joint_pos[side][nfi]  # (B, 20, 3)
        pts = all_joints[:, mano_ids]  # (B, 12, 3)
        return pts + self._env.scene.env_origins[:, None, :]

    def next_mano_all_joints_lin_vel_w(self, side: str) -> torch.Tensor:
        """Next-frame MANO velocities for all 12 non-tip joints. Shape: (B, 12, 3)."""
        nfi = self._next_motion_flat_ids()
        mano_ids = self._all_mano_ids[side]
        all_vel = self.motion_lib.mano_joint_vel[side][nfi]  # (B, 20, 3)
        return all_vel[:, mano_ids]  # (B, 12, 3)

    # --- MANO future-window keypoints: K-step versions of next_mano_* above ---
    # offsets = (K,) frame-offset tensor from the obs term; used by future_mano_traj.

    def future_mano_tip_trans_w(self, offsets: torch.Tensor) -> torch.Tensor:
        """K-step MANO fingertip positions. Shape: (B, K, n_sides, 5, 3)."""
        ffi = self._future_motion_flat_ids(offsets)  # (B, K)
        parts = []
        for side in self._side_list:
            all_joints = self.motion_lib.mano_joint_pos[side][ffi]  # (B, K, 20, 3)
            tips = all_joints[:, :, self.motion_lib.tip_ids[side]]  # (B, K, 5, 3)
            tips = tips + self._env.scene.env_origins[:, None, None, :]
            parts.append(tips)
        return torch.stack(parts, dim=2)  # (B, K, n_sides, 5, 3)

    def future_mano_tip_lin_vel_w(self, offsets: torch.Tensor) -> torch.Tensor:
        """K-step MANO fingertip velocities. Shape: (B, K, n_sides, 5, 3)."""
        ffi = self._future_motion_flat_ids(offsets)  # (B, K)
        parts = []
        for side in self._side_list:
            all_vel = self.motion_lib.mano_joint_vel[side][ffi]  # (B, K, 20, 3)
            tips = all_vel[:, :, self.motion_lib.tip_ids[side]]  # (B, K, 5, 3)
            parts.append(tips)
        return torch.stack(parts, dim=2)  # (B, K, n_sides, 5, 3)

    def future_mano_wrist_trans_w(self, offsets: torch.Tensor) -> torch.Tensor:
        """K-step MANO wrist positions. Shape: (B, K, n_sides, 3)."""
        ffi = self._future_motion_flat_ids(offsets)  # (B, K)
        parts = []
        for side in self._side_list:
            pos = self.motion_lib.mano_wrist_trans[side][ffi]  # (B, K, 3)
            parts.append(pos + self._env.scene.env_origins[:, None, :])
        return torch.stack(parts, dim=2)

    def future_mano_wrist_quat_w(self, offsets: torch.Tensor) -> torch.Tensor:
        """K-step MANO wrist quaternions (from rotmat). Shape: (B, K, n_sides, 4)."""
        ffi = self._future_motion_flat_ids(offsets)  # (B, K)
        parts = [
            quat_from_matrix(self.motion_lib.mano_wrist_rot[side][ffi])
            for side in self._side_list
        ]
        return torch.stack(parts, dim=2)

    def future_mano_wrist_rot_w(self, offsets: torch.Tensor) -> torch.Tensor:
        """K-step MANO wrist rotation matrices, no quat round-trip. Shape: (B, K, n_sides, 3, 3)."""
        ffi = self._future_motion_flat_ids(offsets)  # (B, K)
        parts = [self.motion_lib.mano_wrist_rot[side][ffi] for side in self._side_list]
        return torch.stack(parts, dim=2)

    def future_mano_all_joints_trans_w(
        self, side: str, offsets: torch.Tensor
    ) -> torch.Tensor:
        """K-step MANO positions for the non-tip body joints. Shape: (B, K, M, 3)."""
        ffi = self._future_motion_flat_ids(offsets)  # (B, K)
        mano_ids = self._all_mano_ids[side]
        all_joints = self.motion_lib.mano_joint_pos[side][ffi]  # (B, K, 20, 3)
        pts = all_joints[:, :, mano_ids]  # (B, K, M, 3)
        return pts + self._env.scene.env_origins[:, None, None, :]

    def future_mano_all_joints_lin_vel_w(
        self, side: str, offsets: torch.Tensor
    ) -> torch.Tensor:
        """K-step MANO velocities for the non-tip body joints. Shape: (B, K, M, 3)."""
        ffi = self._future_motion_flat_ids(offsets)  # (B, K)
        mano_ids = self._all_mano_ids[side]
        all_vel = self.motion_lib.mano_joint_vel[side][ffi]  # (B, K, 20, 3)
        return all_vel[:, :, mano_ids]  # (B, K, M, 3)

    # --- MANO contact / distance ---

    @property
    def ref_contact_trans_w(self) -> torch.Tensor:
        """Reference contact points on object, in world frame. Shape: (B, n_sides, 5, 3)."""
        parts = []
        sim_obj_quat = self.sim_obj_quat_w  # (B, n_sides, 4)
        sim_obj_trans = self.sim_obj_trans_w  # (B, n_sides, 3)
        for side in self._side_list:
            si = self._side_list.index(side)
            local_pts = self.motion_lib.contact_pos_full[side][
                self._motion_flat_ids
            ]  # (B, 5, 3)
            obj_trans = sim_obj_trans[:, si]  # (B, 3)
            obj_rot = matrix_from_quat(sim_obj_quat[:, si])  # (B, 3, 3)
            world_pts = obj_trans[:, None, :] + torch.einsum(
                "bij,bkj->bki", obj_rot, local_pts
            )
            parts.append(world_pts)
        return torch.stack(parts, dim=1)

    @property
    def ref_contact_flags(self) -> torch.Tensor:
        """Binary contact expected per finger per side. Shape: (B, n_sides, 5)."""
        parts = []
        for side in self._side_list:
            parts.append(self.motion_lib.contact_flags[side][self._motion_flat_ids])
        return torch.stack(parts, dim=1)

    @property
    def mano_tips_distance(self) -> torch.Tensor:
        """Precomputed MANO tip-to-object-surface distance. Shape: (B, n_sides, 5)."""
        parts = []
        for side in self._side_list:
            parts.append(self.motion_lib.tips_distance[side][self._motion_flat_ids])
        return torch.stack(parts, dim=1)

    # --- Robot wrist ---

    @property
    def robot_wrist_trans_w(self) -> torch.Tensor:
        """Robot palm site positions. Shape: (B, n_sides, 3)."""
        parts = []
        for side in self._side_list:
            idx = self._palm_site_ids[side]
            parts.append(self.robot.data.site_pos_w[:, idx])
        return torch.stack(parts, dim=1)

    @property
    def robot_wrist_quat_w(self) -> torch.Tensor:
        """Robot palm site quaternions. Shape: (B, n_sides, 4)."""
        parts = []
        for side in self._side_list:
            idx = self._palm_site_ids[side]
            parts.append(self.robot.data.site_quat_w[:, idx])
        return torch.stack(parts, dim=1)

    @property
    def robot_wrist_rot_w(self) -> torch.Tensor:
        """Robot wrist rotation matrices (from quat). Shape: (B, n_sides, 3, 3)."""
        return matrix_from_quat(self.robot_wrist_quat_w)

    @property
    def robot_wrist_lin_vel_w(self) -> torch.Tensor:
        """Robot palm site linear velocities. Shape: (B, n_sides, 3)."""
        parts = []
        for side in self._side_list:
            idx = self._palm_site_ids[side]
            parts.append(self.robot.data.site_lin_vel_w[:, idx])
        return torch.stack(parts, dim=1)

    @property
    def robot_wrist_ang_vel_w(self) -> torch.Tensor:
        """Robot palm site angular velocities. Shape: (B, n_sides, 3)."""
        parts = []
        for side in self._side_list:
            idx = self._palm_site_ids[side]
            parts.append(self.robot.data.site_ang_vel_w[:, idx])
        return torch.stack(parts, dim=1)

    # --- Robot fingertips / contact ---

    @property
    def robot_tip_trans_w(self) -> torch.Tensor:
        """Robot fingertip site positions (5 per side). Shape: (B, n_sides, 5, 3)."""
        parts = []
        for side in self._side_list:
            ids = self._tip_site_ids[side]
            parts.append(self.robot.data.site_pos_w[:, ids])
        return torch.stack(parts, dim=1)

    @property
    def robot_tip_lin_vel_w(self) -> torch.Tensor:
        """Robot fingertip site linear velocities (5 per side). Shape: (B, n_sides, 5, 3)."""
        parts = []
        for side in self._side_list:
            ids = self._tip_site_ids[side]
            parts.append(self.robot.data.site_lin_vel_w[:, ids])
        return torch.stack(parts, dim=1)

    @property
    def robot_ref_tip_trans_w(self) -> torch.Tensor:
        """Reference retargeted-robot fingertip positions, world frame (lazy FK of demo
        qpos); robot analogue of ``mano_tip_trans_w``. Shape: (B, n_sides, 5, 3)."""
        if self._robot_ref_tip_pos is None:
            self._compute_robot_ref_state()
        tips = self._robot_ref_tip_pos[self._motion_flat_ids]  # (B, n_sides, 5, 3)
        return tips + self._env.scene.env_origins[:, None, None, :]

    @property
    def robot_ref_wrist_trans_w(self) -> torch.Tensor:
        """Reference retargeted-robot palm (wrist) positions, world frame (lazy FK of
        demo qpos); robot analogue of ``mano_wrist_trans_w``. Shape: (B, n_sides, 3)."""
        if self._robot_ref_tip_pos is None:
            self._compute_robot_ref_state()
        pos = self._robot_ref_wrist_pos[self._motion_flat_ids]  # (B, n_sides, 3)
        return pos + self._env.scene.env_origins[:, None, :]

    @property
    def robot_ref_wrist_quat_w(self) -> torch.Tensor:
        """Reference retargeted-robot palm quaternions (wxyz); robot analogue of
        ``mano_wrist_quat_w`` (no env-origin shift needed). Shape: (B, n_sides, 4)."""
        if self._robot_ref_tip_pos is None:
            self._compute_robot_ref_state()
        return self._robot_ref_wrist_quat[self._motion_flat_ids]  # (B, n_sides, 4)

    def robot_ref_level_trans_w(self, side: str, level: int) -> torch.Tensor:
        """Reference retargeted-robot level-1 finger body positions, world frame; only
        level 1 is FK'd (the key-body set). Shape: (B, 5, 3)."""
        if level != 1:
            raise ValueError("robot_ref level bodies are only FK'd for level 1")
        if self._robot_ref_level1_pos is None:
            self._compute_robot_ref_state()
        si = self._side_list.index(side)
        pts = self._robot_ref_level1_pos[self._motion_flat_ids][:, si]  # (B, 5, 3)
        return pts + self._env.scene.env_origins[:, None, :]

    @property
    def next_robot_ref_tip_trans_w(self) -> torch.Tensor:
        """Next-frame reference retargeted-robot fingertip positions, world frame;
        robot analogue of ``next_mano_tip_trans_w``. Shape: (B, n_sides, 5, 3)."""
        if self._robot_ref_tip_pos is None:
            self._compute_robot_ref_state()
        tips = self._robot_ref_tip_pos[self._next_motion_flat_ids()]  # (B, n_sides, 5, 3)
        return tips + self._env.scene.env_origins[:, None, None, :]

    def next_robot_ref_level_trans_w(self, side: str, level: int) -> torch.Tensor:
        """Next-frame reference retargeted-robot level-1 body positions, world frame;
        only level 1 is FK'd. Shape: (B, 5, 3)."""
        if level != 1:
            raise ValueError("robot_ref level bodies are only FK'd for level 1")
        if self._robot_ref_level1_pos is None:
            self._compute_robot_ref_state()
        si = self._side_list.index(side)
        pts = self._robot_ref_level1_pos[self._next_motion_flat_ids()][:, si]  # (B, 5, 3)
        return pts + self._env.scene.env_origins[:, None, :]

    def robot_ref_all_joints_trans_w(self, side: str) -> torch.Tensor:
        """Reference retargeted-robot positions of the 12 non-tip bodies, world frame;
        current-frame twin of next_robot_ref_all_joints_trans_w. Shape: (B, 12, 3)."""
        if self._robot_ref_all_pos is None:
            self._compute_robot_ref_state()
        si = self._side_list.index(side)
        pts = self._robot_ref_all_pos[self._motion_flat_ids][:, si]  # (B, 12, 3)
        return pts + self._env.scene.env_origins[:, None, :]

    def next_robot_ref_all_joints_trans_w(self, side: str) -> torch.Tensor:
        """Next-frame reference retargeted-robot positions of the 12 non-tip bodies,
        world frame; robot analogue of ``next_mano_all_joints_trans_w``. Shape: (B, 12, 3)."""
        if self._robot_ref_all_pos is None:
            self._compute_robot_ref_state()
        si = self._side_list.index(side)
        pts = self._robot_ref_all_pos[self._next_motion_flat_ids()][:, si]  # (B, 12, 3)
        return pts + self._env.scene.env_origins[:, None, :]

    def next_robot_ref_all_joints_lin_vel_w(self, side: str) -> torch.Tensor:
        """Next-frame reference retargeted-robot linear velocities of the 12 non-tip
        bodies, world frame. Shape: (B, 12, 3)."""
        if self._robot_ref_all_lin_vel is None:
            self._compute_robot_ref_state()
        si = self._side_list.index(side)
        return self._robot_ref_all_lin_vel[self._next_motion_flat_ids()][:, si]  # (B, 12, 3)

    @property
    def next_robot_ref_tip_lin_vel_w(self) -> torch.Tensor:
        """Next-frame reference retargeted-robot fingertip linear velocities, world
        frame. Shape: (B, n_sides, 5, 3)."""
        if self._robot_ref_tip_lin_vel is None:
            self._compute_robot_ref_state()
        return self._robot_ref_tip_lin_vel[self._next_motion_flat_ids()]  # (B, n_sides, 5, 3)

    @property
    def robot_contact_trans_w(self) -> torch.Tensor:
        """Robot contact sensor site positions (5 per side). Shape: (B, n_sides, 5, 3)."""
        parts = []
        for side in self._side_list:
            ids = self._contact_site_ids[side]
            parts.append(self.robot.data.site_pos_w[:, ids])
        return torch.stack(parts, dim=1)

    # --- DexMachina per-collision-link contact (13 links) ---

    @property
    def ref_contact_alllink_trans_w(self) -> torch.Tensor:
        """Reference per-link contact targets: object-local demo contact points placed at
        the *current* object pose (DexMachina transform_contact). Shape: (B, n_sides, 13, 3)."""
        parts = []
        sim_obj_quat = self.sim_obj_quat_w  # (B, n_sides, 4)
        sim_obj_trans = self.sim_obj_trans_w  # (B, n_sides, 3)
        for side in self._side_list:
            si = self._side_list.index(side)
            local_pts = self.motion_lib.contact_alllink_pos[side][
                self._motion_flat_ids
            ]  # (B, 13, 3), object-local
            obj_trans = sim_obj_trans[:, si]  # (B, 3)
            obj_rot = matrix_from_quat(sim_obj_quat[:, si])  # (B, 3, 3)
            world_pts = obj_trans[:, None, :] + torch.einsum(
                "bij,bkj->bki", obj_rot, local_pts
            )
            parts.append(world_pts)
        return torch.stack(parts, dim=1)

    @property
    def ref_contact_alllink_normal_w(self) -> torch.Tensor:
        """Reference per-link contact OUTWARD surface normal, world frame (object-local demo
        normal rotated by current object pose); force presses along -this. Shape: (B, n_sides, 13, 3)."""
        parts = []
        sim_obj_quat = self.sim_obj_quat_w  # (B, n_sides, 4)
        for side in self._side_list:
            si = self._side_list.index(side)
            local_n = self.motion_lib.contact_alllink_normal[side][
                self._motion_flat_ids
            ]  # (B, 13, 3), object-local outward normal
            obj_rot = matrix_from_quat(sim_obj_quat[:, si])  # (B, 3, 3)
            world_n = torch.einsum("bij,bkj->bki", obj_rot, local_n)
            parts.append(world_n)
        return torch.stack(parts, dim=1)

    @property
    def ref_contact_alllink_flags(self) -> torch.Tensor:
        """Binary contact expected per collision link per side.
        Shape: (B, n_sides, 13)."""
        parts = []
        for side in self._side_list:
            parts.append(
                self.motion_lib.contact_alllink_flags[side][self._motion_flat_ids]
            )
        return torch.stack(parts, dim=1)

    # The POLICY contact point comes from the r_alllink_contact_pos sensor (data.pos) in
    # the reward, not a body-center lookup — matches DexMachina's contact_pos_link_a.

    # --- Robot body-level joints ---

    def robot_all_joints_trans_w(self, side: str) -> torch.Tensor:
        """Robot positions for all 12 non-tip bodies. Shape: (B, 12, 3)."""
        body_ids = self._all_body_ids[side]
        return self.robot.data.body_link_pos_w[:, body_ids]  # (B, 12, 3)

    def robot_all_joints_lin_vel_w(self, side: str) -> torch.Tensor:
        """Robot velocities for all 12 non-tip bodies. Shape: (B, 12, 3)."""
        body_ids = self._all_body_ids[side]
        return self.robot.data.body_link_lin_vel_w[:, body_ids]  # (B, 12, 3)

    def robot_level_trans_w(self, side: str, level: int) -> torch.Tensor:
        """Robot body positions for level 1 or 2. Shape: (B, 5, 3)."""
        body_ids = {1: self._level1_body_ids, 2: self._level2_body_ids,
                    3: self._levelmid_body_ids}[level][side]
        return self.robot.data.body_link_pos_w[:, body_ids]  # (B, 5, 3)
