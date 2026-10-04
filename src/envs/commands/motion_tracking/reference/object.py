"""Object property mixin for MotionTrackingCommand: reference trajectory, sim state,
next-frame/K-step lookahead, and SDF query for object tracking."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.utils.lab_api.math import quat_from_matrix

from utils.sdf import bake_object_sdf_grid as _bake_object_sdf_grid

if TYPE_CHECKING:
    from mjlab.entity import Entity

    from ..motion_tracking_cfg import MotionTrackingCommandCfg


class ObjectPropertiesMixin:
    """Reference + sim object state properties and SDF query."""

    # Multi-object helpers: cfg.object.entity_names[side] is a str (single mode) or a
    # slot-indexed list; each env reads its ACTIVE slot's entity.

    def obj_entity_names(self, side: str) -> list[str]:
        """Object entity names for ``side`` as a list (len 1 in single mode)."""
        v = self.cfg.object.entity_names[side]
        return [v] if isinstance(v, str) else list(v)

    @property
    def multi_object(self) -> bool:
        if not self.has_objects:
            return False
        return any(
            len(self.obj_entity_names(side)) > 1
            for side in self._side_list
            if side in self.cfg.object.entity_names
        )

    def active_obj_slot(self, side: str) -> torch.Tensor:
        """Active object slot per env for ``side``. Shape: (B,), long."""
        return self.motion_lib.traj_obj_slot[side][self.motion_ids]

    def _obj_shared_env_mask(self) -> torch.Tensor | None:
        """(B,) bool: envs on a shared slot of a mixed pair pt (cfg.object.shared_slots); the
        left object there IS the right entity and its own entity is an inert ghost. Else None."""
        ss = getattr(self.cfg.object, "shared_slots", None)
        if not ss:
            return None
        t = getattr(self, "_shared_slots_t", None)
        if t is None:
            t = torch.as_tensor(list(ss), dtype=torch.bool, device=self.device)
            self._shared_slots_t = t
        return t[self.active_obj_slot("left")]

    def _obj_alias_to_right(self, side: str) -> torch.Tensor | None:
        """(B,) bool: envs where ``side``'s object reads/writes go to the right entity."""
        return None if side == "right" else self._obj_shared_env_mask()

    def _sim_obj_cache_token(self) -> tuple[int, int]:
        """Cache key for ``_sim_obj_field``: invalidates on sim substep and on slot/state
        rewrites (``_obj_state_version``); out-of-band state writes are NOT tracked."""
        return (
            int(getattr(self._env, "_sim_step_counter", 0)),
            int(getattr(self, "_obj_state_version", 0)),
        )

    def _sim_obj_field(self, side: str, attr: str) -> torch.Tensor:
        """Per-env sim state field of the ACTIVE object entity, shape (B, ...); the S-entity
        gather is slot-count-proportional, so the result is cached per step."""
        names = self.obj_entity_names(side)
        if len(names) == 1:
            obj: Entity = self._env.scene[names[0]]
            out = getattr(obj.data, attr)[:, self._obj_mass_body_idx(obj)]
            alias = self._obj_alias_to_right(side)
            if alias is not None:  # mixed pair pt: shared envs read the right entity
                robj: Entity = self._env.scene[self.obj_entity_names("right")[0]]
                rout = getattr(robj.data, attr)[:, self._obj_mass_body_idx(robj)]
                out = torch.where(alias.view(-1, *([1] * (out.dim() - 1))), rout, out)
            return out

        token = self._sim_obj_cache_token()
        cache = getattr(self, "_sim_obj_cache", None)
        if cache is None or self._sim_obj_cache_token_val != token:
            cache = {}
            self._sim_obj_cache = cache
            self._sim_obj_cache_token_val = token
        key = (side, attr)
        hit = cache.get(key)
        if hit is not None:
            return hit

        parts = []
        for name in names:
            obj = self._env.scene[name]
            parts.append(getattr(obj.data, attr)[:, self._obj_mass_body_idx(obj)])
        stacked = torch.stack(parts, dim=1)  # (B, S, ...)
        slot = self.active_obj_slot(side)
        rows = torch.arange(slot.shape[0], device=slot.device)
        gathered = stacked[rows, slot]
        cache[key] = gathered
        return gathered

    # --- Reference object trajectory ---

    @property
    def has_objects(self) -> bool:
        return self.cfg.object.entity_names is not None

    @property
    def ref_obj_trans_w(self) -> torch.Tensor:
        """Reference object positions. Shape: (B, n_sides, 3)."""
        parts = []
        for side in self._side_list:
            if side in self.motion_lib.obj_trans:
                pos = self.motion_lib.obj_trans[side][self._motion_flat_ids]
                pos = pos + self._env.scene.env_origins
                parts.append(pos)
        return torch.stack(parts, dim=1)

    @property
    def ref_obj_rotmat_w(self) -> torch.Tensor:
        """Reference object rotation matrices. Shape: (B, n_sides, 3, 3)."""
        parts = []
        for side in self._side_list:
            if side in self.motion_lib.obj_rot:
                parts.append(self.motion_lib.obj_rot[side][self._motion_flat_ids])
        return torch.stack(parts, dim=1)

    @property
    def ref_obj_quat_w(self) -> torch.Tensor:
        """Reference object quaternions. Shape: (B, n_sides, 4)."""
        return quat_from_matrix(self.ref_obj_rotmat_w)

    @property
    def ref_obj_lin_vel_w(self) -> torch.Tensor:
        """Reference object velocities. Shape: (B, n_sides, 3)."""
        parts = []
        for side in self._side_list:
            if side in self.motion_lib.obj_lin_vel:
                parts.append(self.motion_lib.obj_lin_vel[side][self._motion_flat_ids])
        return torch.stack(parts, dim=1)

    @property
    def ref_obj_ang_vel_w(self) -> torch.Tensor:
        """Reference object angular velocities. Shape: (B, n_sides, 3)."""
        parts = []
        for side in self._side_list:
            if side in self.motion_lib.obj_ang_vel:
                parts.append(self.motion_lib.obj_ang_vel[side][self._motion_flat_ids])
        return torch.stack(parts, dim=1)

    # --- Sim object state ---

    def _obj_mass_body_idx(self, obj) -> int:
        """Index of the body that carries mass/geoms within the entity's body_ids."""
        return len(obj.indexing.body_ids) - 1

    @property
    def sim_obj_trans_w(self) -> torch.Tensor:
        """Sim object positions. Shape: (B, n_sides, 3)."""
        return torch.stack(
            [self._sim_obj_field(s, "body_link_pos_w") for s in self._side_list],
            dim=1,
        )

    @property
    def sim_obj_quat_w(self) -> torch.Tensor:
        """Sim object quaternions. Shape: (B, n_sides, 4)."""
        return torch.stack(
            [self._sim_obj_field(s, "body_link_quat_w") for s in self._side_list],
            dim=1,
        )

    @property
    def sim_obj_lin_vel_w(self) -> torch.Tensor:
        """Sim object velocities. Shape: (B, n_sides, 3)."""
        return torch.stack(
            [self._sim_obj_field(s, "body_link_lin_vel_w") for s in self._side_list],
            dim=1,
        )

    @property
    def sim_obj_ang_vel_w(self) -> torch.Tensor:
        """Sim object angular velocities. Shape: (B, n_sides, 3)."""
        return torch.stack(
            [self._sim_obj_field(s, "body_link_ang_vel_w") for s in self._side_list],
            dim=1,
        )

    # --- Future trajectory (1-step lookahead) ---

    def _next_motion_flat_ids(self) -> torch.Tensor:
        """Flat index for next frame per env, clamped to trajectory end."""
        cap = self.motion_lib._motion_num_frames[self.motion_ids] - 1
        next_t = torch.minimum(self.motion_steps + 1, cap)
        return self.motion_lib.length_starts[self.motion_ids] + next_t

    @property
    def next_obj_trans_w(self) -> torch.Tensor:
        """Next-frame object positions. Shape: (B, n_sides, 3)."""
        nfi = self._next_motion_flat_ids()
        parts = []
        for side in self._side_list:
            if side in self.motion_lib.obj_trans:
                pos = self.motion_lib.obj_trans[side][nfi] + self._env.scene.env_origins
                parts.append(pos)
        return torch.stack(parts, dim=1)

    @property
    def next_obj_quat_w(self) -> torch.Tensor:
        """Next-frame object quaternions. Shape: (B, n_sides, 4)."""
        nfi = self._next_motion_flat_ids()
        parts = []
        for side in self._side_list:
            if side in self.motion_lib.obj_rot:
                rotmat = self.motion_lib.obj_rot[side][nfi]
                parts.append(quat_from_matrix(rotmat))
        return torch.stack(parts, dim=1)

    @property
    def next_obj_vel_w(self) -> torch.Tensor:
        """Next-frame object velocities. Shape: (B, n_sides, 3)."""
        nfi = self._next_motion_flat_ids()
        parts = []
        for side in self._side_list:
            if side in self.motion_lib.obj_lin_vel:
                parts.append(self.motion_lib.obj_lin_vel[side][nfi])
        return torch.stack(parts, dim=1)

    @property
    def next_obj_ang_vel_w(self) -> torch.Tensor:
        """Next-frame object angular velocities. Shape: (B, n_sides, 3)."""
        nfi = self._next_motion_flat_ids()
        parts = []
        for side in self._side_list:
            if side in self.motion_lib.obj_ang_vel:
                parts.append(self.motion_lib.obj_ang_vel[side][nfi])
        return torch.stack(parts, dim=1)

    # Future trajectory (K-step lookahead): offsets = (K,) frame-offset tensor built by the
    # obs term; each property returns (B, K, n_sides, *). Used by future_obj_traj_* terms.

    def _future_motion_flat_ids(self, offsets: torch.Tensor) -> torch.Tensor:
        """Flat motion-library indices for K future frames per env, clamped to the clip end.
        offsets: (K,) long frame offsets ahead of the current step; returns (B, K) long."""
        cap = self.motion_lib._motion_num_frames[self.motion_ids] - 1  # (B,)
        future_t = self.motion_steps[:, None] + offsets[None, :].to(self.motion_steps)  # (B, K)
        future_t = torch.minimum(future_t, cap[:, None])
        starts = self.motion_lib.length_starts[self.motion_ids][:, None]  # (B, 1)
        return starts + future_t  # (B, K)

    def future_obj_traj_trans_w(self, offsets: torch.Tensor) -> torch.Tensor:
        """K-step future object positions. Shape: (B, K, n_sides, 3)."""
        ffi = self._future_motion_flat_ids(offsets)  # (B, K)
        parts = []
        for side in self._side_list:
            if side in self.motion_lib.obj_trans:
                pos = self.motion_lib.obj_trans[side][ffi]  # (B, K, 3)
                pos = pos + self._env.scene.env_origins[:, None, :]
                parts.append(pos)
        return torch.stack(parts, dim=2)  # (B, K, n_sides, 3)

    def future_obj_traj_rotmat_w(self, offsets: torch.Tensor) -> torch.Tensor:
        """K-step future object rotation matrices. Shape: (B, K, n_sides, 3, 3)."""
        ffi = self._future_motion_flat_ids(offsets)  # (B, K)
        parts = []
        for side in self._side_list:
            if side in self.motion_lib.obj_rot:
                parts.append(self.motion_lib.obj_rot[side][ffi])  # (B, K, 3, 3)
        return torch.stack(parts, dim=2)  # (B, K, n_sides, 3, 3)

    def future_obj_traj_lin_vel_w(self, offsets: torch.Tensor) -> torch.Tensor:
        """K-step future object linear velocities. Shape: (B, K, n_sides, 3)."""
        ffi = self._future_motion_flat_ids(offsets)  # (B, K)
        parts = []
        for side in self._side_list:
            if side in self.motion_lib.obj_lin_vel:
                parts.append(self.motion_lib.obj_lin_vel[side][ffi])  # (B, K, 3)
        return torch.stack(parts, dim=2)  # (B, K, n_sides, 3)

    def future_obj_traj_ang_vel_w(self, offsets: torch.Tensor) -> torch.Tensor:
        """K-step future object angular velocities. Shape: (B, K, n_sides, 3)."""
        ffi = self._future_motion_flat_ids(offsets)  # (B, K)
        parts = []
        for side in self._side_list:
            if side in self.motion_lib.obj_ang_vel:
                parts.append(self.motion_lib.obj_ang_vel[side][ffi])  # (B, K, 3)
        return torch.stack(parts, dim=2)  # (B, K, n_sides, 3)

    # --- SDF query ---

    def _init_object_sdf(self, cfg: MotionTrackingCommandCfg) -> None:
        self._obj_sdf_grids: dict[str, torch.Tensor] = {}
        self._obj_sdf_extent: float = float(cfg.object.sdf.grid_extent)
        self._obj_sdf_n: int = int(cfg.object.sdf.grid_n)
        if cfg.object.mesh_paths is not None:
            scales = cfg.object.mesh_scales or {}
            for side, mesh_path in cfg.object.mesh_paths.items():
                if side not in self.motion_lib.hand_sides or mesh_path is None:
                    continue
                if isinstance(mesh_path, str):
                    self._obj_sdf_grids[side] = _bake_object_sdf_grid(
                        mesh_path,
                        float(scales.get(side, 1.0)),
                        self._obj_sdf_extent,
                        self._obj_sdf_n,
                        device=str(self.device),
                    )
                else:
                    # One grid per slot: (S, 4, N, N, N).
                    slot_scales = list(scales.get(side, [1.0] * len(mesh_path)))
                    self._obj_sdf_grids[side] = torch.stack(
                        [
                            _bake_object_sdf_grid(
                                mp,
                                float(slot_scales[s]),
                                self._obj_sdf_extent,
                                self._obj_sdf_n,
                                device=str(self.device),
                            )
                            for s, mp in enumerate(mesh_path)
                        ]
                    )

    def sdf_query(
        self,
        world_pts: torch.Tensor,
        side: str,
        obj_trans: torch.Tensor | None = None,
        obj_quat: torch.Tensor | None = None,
        env_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Query the per-side baked SDF + gradient at world-frame points (B, K, 3) or (B, 3).
        Returns (sdf, grad_world); obj pose/env_ids override the sim pose (all envs)."""
        if side not in self._obj_sdf_grids:
            raise KeyError(
                f"sdf_query: no SDF grid baked for side {side!r}. Set "
                f"`object_mesh_paths[{side!r}]` on the MotionTrackingCommandCfg."
            )
        grid = self._obj_sdf_grids[side]  # (4, N, N, N) or (S, 4, N, N, N)
        si = self._side_list.index(side)
        if obj_trans is None:
            obj_trans = self.sim_obj_trans_w[:, si]  # (B, 3)
        if obj_quat is None:
            obj_quat = self.sim_obj_quat_w[:, si]  # (B, 4)

        squeeze_K = world_pts.dim() == 2
        if squeeze_K:
            world_pts = world_pts.unsqueeze(1)  # (B, 1, 3)

        B, K = world_pts.shape[0], world_pts.shape[1]
        delta = world_pts - obj_trans[:, None, :]  # (B, K, 3)
        quat_b = obj_quat[:, None, :].expand(B, K, 4)
        from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse

        local = quat_apply_inverse(quat_b, delta)  # (B, K, 3) in object frame

        norm = local / self._obj_sdf_extent  # (B, K, 3)
        norm_zyx = norm.flip(-1)  # (B, K, 3) reordered to (z, y, x)
        if grid.dim() == 5:
            # Multi-object: pack envs by active slot into ONE batched grid_sample (batch=S);
            # object-count-flat and bit-identical vs the old per-slot launch loop.
            S = grid.shape[0]
            slot = self.active_obj_slot(side)  # (B,)
            if env_ids is not None:
                slot = slot[env_ids]
            order = torch.argsort(slot)  # env ids grouped by slot
            counts = torch.bincount(slot, minlength=S)
            n_max = int(counts.max().item())
            starts = torch.cumsum(
                torch.cat([counts.new_zeros(1), counts[:-1]]), 0
            )  # (S,) first sorted-position of each slot
            within = torch.arange(B, device=self.device) - starts[slot[order]]
            packed = norm_zyx.new_zeros(S, n_max, K, 3)
            packed[slot[order], within] = norm_zyx[order]
            o = torch.nn.functional.grid_sample(
                grid,
                packed.reshape(S, n_max * K, 1, 1, 3),
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            )  # (S, 4, n_max*K, 1, 1)
            o = o.view(S, 4, n_max, K).permute(0, 2, 3, 1)  # (S, n_max, K, 4)
            out = norm_zyx.new_empty(B, K, 4)
            out[order] = o[slot[order], within]
        else:
            sample_grid = norm_zyx.reshape(1, B * K, 1, 1, 3)
            grid_5d = grid.unsqueeze(0)  # (1, 4, N, N, N)
            out = torch.nn.functional.grid_sample(
                grid_5d,
                sample_grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            )  # (1, 4, B*K, 1, 1)
            out = out.view(4, B, K).permute(1, 2, 0)  # (B, K, 4)
        sdf = out[..., 0]  # (B, K)
        grad_local = out[..., 1:]  # (B, K, 3) in object frame
        grad_local = grad_local / (
            grad_local.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        )
        grad_world = quat_apply(quat_b, grad_local)  # (B, K, 3)

        if squeeze_K:
            sdf = sdf.squeeze(1)
            grad_world = grad_world.squeeze(1)
        return sdf, grad_world
