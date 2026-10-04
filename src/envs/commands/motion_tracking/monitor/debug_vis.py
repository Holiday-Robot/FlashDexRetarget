"""MotionTrackingCommand mixin: reference ghost, keypoint and wrench overlays."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import numpy as np
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import matrix_from_quat

if TYPE_CHECKING:
    from mjlab.viewer.debug_visualizer import DebugVisualizer


class DebugVisMixin:
    def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
        # ── Reference ghost (hand + object) ────────────────────────────────────────
        show_robot = self.cfg.viz_robot_ghost
        show_object = self.cfg.viz_object_ghost and self.has_objects

        if show_robot or show_object:
            if self._ghost_model is None:
                self._ghost_model = copy.deepcopy(self._env.sim.mj_model)
                collision = (self._ghost_model.geom_contype != 0) | (
                    self._ghost_model.geom_conaffinity != 0
                )
                object_body_ids: list[int] = []
                if self.has_objects:
                    for side in self.cfg.object.entity_names:
                        for entity_name in self.obj_entity_names(side):
                            entity = self._env.scene[entity_name]
                            object_body_ids.extend(
                                entity.indexing.body_ids.cpu().numpy().tolist()
                            )
                object_mask = np.isin(self._ghost_model.geom_bodyid, object_body_ids)
                robot_visual = ~object_mask & ~collision

                self._ghost_model.geom_rgba[:] = np.array([0.2, 0.6, 1.0, 0.5])
                self._ghost_model.geom_rgba[collision, 3] = 0.0
                self._ghost_model.geom_rgba[robot_visual, 3] = (
                    0.5 if show_robot else 0.0
                )
                self._ghost_model.geom_rgba[object_mask, 3] = (
                    0.5 if show_object else 0.0
                )

            free_q = self.robot.indexing.free_joint_q_adr.cpu().numpy()
            joint_q = self.robot.indexing.joint_q_adr.cpu().numpy()
            obj_q_adr = {}
            if self.has_objects:
                obj_q_adr = {
                    side: [
                        self._env.scene[name].indexing.free_joint_q_adr.cpu().numpy()
                        for name in self.obj_entity_names(side)
                    ]
                    for side in self.cfg.object.entity_names
                }

            # mjlab forces mjVIS_TRANSPARENT, which hides geoms behind the ghost.
            import mujoco

            _t_flag = int(mujoco.mjtVisFlag.mjVIS_TRANSPARENT)
            prev_transparent = bool(visualizer._vopt.flags[_t_flag])
            visualizer._vopt.flags[_t_flag] = False
            try:
                for batch in visualizer.get_env_indices(self.num_envs):
                    qpos = np.zeros(self._env.sim.mj_model.nq)
                    if free_q.size > 0:
                        qpos[free_q[0:3]] = (
                            self.mano_wrist_trans_w[batch, 0].cpu().numpy()
                        )
                        qpos[free_q[3:7]] = (
                            self.mano_wrist_quat_w[batch, 0].cpu().numpy()
                        )
                    qpos[joint_q] = self.ref_joint_pos[batch].cpu().numpy()
                    for si, side in enumerate(self._side_list):
                        if side not in obj_q_adr:
                            continue
                        adrs = obj_q_adr[side]
                        slot_b = (
                            int(self.active_obj_slot(side)[batch])
                            if len(adrs) > 1
                            else 0
                        )
                        adr = adrs[slot_b]
                        qpos[adr[0:3]] = self.ref_obj_trans_w[batch, si].cpu().numpy()
                        qpos[adr[3:7]] = self.ref_obj_quat_w[batch, si].cpu().numpy()
                    visualizer.add_ghost_mesh(
                        qpos, model=self._ghost_model, label=f"ref_{batch}"
                    )
            finally:
                visualizer._vopt.flags[_t_flag] = prev_transparent

        # ── MANO keypoints ─────────────────────────────────────────────────────────
        if self.cfg.viz_human_keypoint:
            RED = (1.0, 0.1, 0.1, 1.0)
            GREEN = (0.2, 0.9, 0.3, 1.0)
            WHITE = (1.0, 1.0, 1.0, 1.0)
            finger_chain: dict[str, dict[str, int]] = {
                f: {} for f in self.finger_names
            }
            for i, (_, mano_joint) in enumerate(self.cfg.body_mapping["all"]):
                for f in self.finger_names:
                    if mano_joint.startswith(f"{f}_"):
                        kind = mano_joint[len(f) + 1 :]
                        finger_chain[f].setdefault(kind, i)
                        break

            for batch in visualizer.get_env_indices(self.num_envs):
                for si, side in enumerate(self._side_list):
                    wrist = self.mano_wrist_trans_w[batch, si].cpu().numpy()
                    tips = self.mano_tip_trans_w[batch, si].cpu().numpy()
                    non_tips = self.mano_all_joints_trans_w(side)[batch].cpu().numpy()
                    visualizer.add_sphere(
                        wrist,
                        radius=0.008,
                        color=WHITE,
                        label=f"mano_kp_{side}_wrist_{batch}",
                    )
                    for i, p in enumerate(tips):
                        visualizer.add_sphere(
                            p,
                            radius=0.005,
                            color=RED,
                            label=f"mano_kp_{side}_tip{i}_{batch}",
                        )
                    for i, p in enumerate(non_tips):
                        visualizer.add_sphere(
                            p,
                            radius=0.004,
                            color=GREEN,
                            label=f"mano_kp_{side}_joint{i}_{batch}",
                        )
                    for fi, finger in enumerate(self.finger_names):
                        chain = finger_chain[finger]
                        if "proximal" not in chain or "intermediate" not in chain:
                            continue
                        prox = non_tips[chain["proximal"]]
                        inter = non_tips[chain["intermediate"]]
                        tip = tips[fi]
                        for seg, (a, b) in enumerate(
                            ((wrist, prox), (prox, inter), (inter, tip))
                        ):
                            visualizer.add_cylinder(
                                a,
                                b,
                                radius=0.0025,
                                color=GREEN,
                                label=f"mano_edge_{side}_{finger}_{seg}_{batch}",
                            )

        # ── Contact wrench ─────────────────────────────────────────────────────────
        if self.cfg.viz_contact_wrench:
            YELLOW = (1.0, 0.9, 0.0, 1.0)    # contact point
            GREEN = (0.1, 1.0, 0.2, 1.0)     # net force
            MAGENTA = (1.0, 0.1, 0.9, 1.0)   # net torque
            pos_sensor: ContactSensor = self._env.scene["r_alllink_contact_pos"]
            force_sensor: ContactSensor = self._env.scene["r_alllink_contact"]
            cpos = pos_sensor.data.pos          # (B, 13, 3) world contact point
            found = pos_sensor.data.found       # (B, 13)
            cforce = force_sensor.data.force    # (B, 13, 3) world net force
            ctorque = force_sensor.data.torque  # (B, 13, 3) world net torque (or None)
            for batch in visualizer.get_env_indices(self.num_envs):
                for si in range(len(self._side_list)):
                    wt = self.robot_wrist_trans_w[batch, si].cpu().numpy()
                    R = matrix_from_quat(
                        self.robot_wrist_quat_w[batch, si : si + 1]
                    )[0].cpu().numpy()
                    visualizer.add_frame(
                        wt, R, scale=0.08, axis_radius=0.005,
                        label=f"wrist_frame_{batch}_{si}",
                    )
                for li in range(cpos.shape[1]):
                    if float(found[batch, li]) <= 0:
                        continue
                    p = cpos[batch, li].cpu().numpy()
                    f = cforce[batch, li].cpu().numpy()
                    fmag = float(np.linalg.norm(f))
                    visualizer.add_sphere(
                        p, radius=0.008, color=YELLOW,
                        label=f"contact_pt_{batch}_{li}",
                    )
                    if fmag > 1e-6:
                        L = min(0.2, 0.04 * float(np.log1p(fmag)))  # log-scaled, like the obs
                        visualizer.add_arrow(
                            p, p + f / fmag * L, color=GREEN, width=0.006,
                            label=f"contact_force_{batch}_{li}",
                        )
                    if ctorque is not None:
                        tq = ctorque[batch, li].cpu().numpy()
                        tmag = float(np.linalg.norm(tq))
                        if tmag > 1e-6:
                            Lt = min(0.14, 0.04 * float(np.log1p(tmag)))
                            visualizer.add_arrow(
                                p, p + tq / tmag * Lt, color=MAGENTA, width=0.005,
                                label=f"contact_torque_{batch}_{li}",
                            )
