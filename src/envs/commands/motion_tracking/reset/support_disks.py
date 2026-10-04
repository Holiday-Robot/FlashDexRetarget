"""MotionTrackingCommand mixin: CHORD support disks posed per motion at reset."""

from __future__ import annotations

import torch

from ....scene._support_disk_setup import support_park_position


class SupportDisksMixin:
    def _write_support_disks(self, env_ids: torch.Tensor) -> None:
        sd = self.cfg.object.support_disks
        if not bool(sd.enable) or not sd.entity_names:
            return
        disks = self.motion_lib.support_disks[self.motion_ids[env_ids]]  # (N, K, 5)
        if disks.shape[1] == 0:
            return
        radii = torch.tensor([float(r) for r in sd.radii], device=self.device)
        r = disks[..., 3]
        r_eff = r + float(sd.radius_margin)
        bucket = (radii[None, None, :] < r_eff[..., None] - 1e-6).sum(-1).clamp_(max=len(radii) - 1)
        origins = self._env.scene.env_origins[env_ids]
        quat = torch.zeros((len(env_ids), 4), device=self.device)
        quat[:, 0] = 1.0
        for k, row in enumerate(sd.entity_names):
            if k >= disks.shape[1]:
                break
            active_k = r[:, k] > 0
            centre = disks[:, k, 0:3].clone()
            # Data holds the disk top; the cylinder centre sits height/2 below.
            centre[:, 2] -= float(sd.height) / 2
            for j, name in enumerate(row):
                use = active_k & (bucket[:, k] == j)
                park = torch.tensor(support_park_position(sd, k, j), device=self.device)
                pos = torch.where(use[:, None], origins + centre, origins + park)
                self._env.scene[name].write_mocap_pose_to_sim(
                    torch.cat([pos, quat], dim=-1), env_ids
                )
