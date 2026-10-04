"""MotionTrackingCommand mixin: multi-object swap rows, slot parking, sleep."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch

from ....scene._object_setup import object_park_position
from ..motion_tracking_cfg import MotionTrackingCommandCfg

if TYPE_CHECKING:
    from mjlab.entity import Entity


class MultiObjectMixin:
    def _init_object_swap(self, cfg: MotionTrackingCommandCfg) -> None:
        """Harvest swap rows + expand model fields (recaptures CUDA graphs)."""
        import numpy as np
        from omegaconf import OmegaConf

        from ....scene._object_swap import (
            SWAP_MODEL_FIELDS,
            harvest_swap_row_table,
            swap_mesh_name,
        )

        inject_fn = blob_fn = None
        if getattr(self._env.sim, "wp_model", None) is not None:
            from simulator.mujoco.swap_collision import (
                inject_swap_collision,
                mesh_collision_blob,
            )

            inject_fn, blob_fn = inject_swap_collision, mesh_collision_blob

        spec = cfg.object.swap_spec
        if OmegaConf.is_config(spec):
            spec = OmegaConf.to_container(spec, resolve=True)
        body_name = str(spec["body_name"])
        p_max = int(spec["p_max"])
        dirs = list(spec["obj_dirs"])
        tbl, hulls = harvest_swap_row_table(
            dirs,
            list(spec["scales"]),
            float(spec["density"]),
            p_max,
            mesh_blob_fn=blob_fn,
        )
        mjm = self._env.sim.mj_model
        # mjlab namespaces merged-entity names ("object_right/obj_right_...").
        ent = self.obj_entity_names(str(spec["side"]))[0]

        def _named(accessor, name):
            try:
                return accessor(f"{ent}/{name}")
            except KeyError:
                return accessor(name)

        null_id = _named(mjm.mesh, f"{body_name}_null_tet").id
        dataid = np.full((len(dirs), p_max), null_id, np.int32)
        for i in range(len(dirs)):
            for k in range(int(tbl["nhull"][i])):
                dataid[i, k] = _named(
                    mjm.mesh, swap_mesh_name(body_name, i, k)
                ).id
        tbl["dataid"] = dataid

        # Must run before expand_model_fields recaptures the CUDA graphs.
        if inject_fn is not None:
            inject_fn(self._env.sim, dataid, tbl["nhull"], hulls)

        dev = self.device
        self._swap_tables = {
            k: torch.as_tensor(v, device=dev) for k, v in tbl.items()
        }
        self._swap_geom_ids = torch.tensor(
            [
                _named(mjm.geom, f"{body_name}_col_{k}").id
                for k in range(p_max)
            ],
            device=dev,
            dtype=torch.long,
        )
        self._swap_body_id = int(_named(mjm.body, body_name).id)
        self._swap_dof0 = int(_named(mjm.body, body_name).dofadr[0])
        base = float(mjm.body_subtreemass[0]) - float(
            mjm.body_mass[self._swap_body_id]
        )
        self._swap_world_subm = base + self._swap_tables["bmass"]
        self._env.sim.expand_model_fields(tuple(SWAP_MODEL_FIELDS))

    def _write_swap_rows(
        self, env_ids: torch.Tensor, slots: torch.Tensor
    ) -> None:
        """Per-world writes of each env's object rows (CUDA graph safe)."""
        import os as _os

        dbg = _os.environ.get("FDR_SWAP_DEBUG_UNIFORM")
        if dbg:
            # Perf diagnostics: "K" = all envs use object K; "blockN" = clustered.
            if dbg.startswith("block"):
                n = int(self._swap_tables["nhull"].shape[0])
                slots = (env_ids // int(dbg[5:])) % n
            else:
                slots = torch.full_like(slots, int(dbg) if dbg.isdigit() else 0)
        m = self._env.sim.model
        T = self._swap_tables
        assert T is not None
        gid = self._swap_geom_ids
        w = env_ids[:, None]
        m.geom_dataid[w, gid] = T["dataid"][slots]
        m.geom_pos[w, gid] = T["gpos"][slots]
        m.geom_quat[w, gid] = T["gquat"][slots]
        m.geom_size[w, gid] = T["gsize"][slots]
        m.geom_rbound[w, gid] = T["grbound"][slots]
        m.geom_aabb[w, gid] = T["gaabb"][slots]
        b = self._swap_body_id
        m.body_mass[env_ids, b] = T["bmass"][slots]
        m.body_inertia[env_ids, b] = T["binertia"][slots]
        m.body_ipos[env_ids, b] = T["bipos"][slots]
        m.body_iquat[env_ids, b] = T["biquat"][slots]
        m.body_invweight0[env_ids, b] = T["binvw"][slots]
        m.body_subtreemass[env_ids, b] = T["bsubm"][slots]
        m.body_subtreemass[env_ids, 0] = self._swap_world_subm[slots]
        d0 = self._swap_dof0
        m.dof_invweight0[env_ids, d0 : d0 + 6] = T["dinvw"][slots]

    def _slot_tree_ids(self, side: str) -> torch.Tensor:
        cache = getattr(self, "_slot_tree_ids_cache", None)
        if cache is None:
            cache = {}
            self._slot_tree_ids_cache = cache
        t = cache.get(side)
        if t is None:
            mjm = self._env.sim.mj_model
            ids = []
            for name in self.obj_entity_names(side):
                obj = self._env.scene[name]
                bid = int(obj.indexing.body_ids[self._obj_mass_body_idx(obj)])
                ids.append(int(mjm.body_treeid[bid]))
            t = torch.tensor(ids, device=self.device, dtype=torch.long)
            cache[side] = t
        return t

    def _sleep_park_trees(
        self, side: str, env_ids: torch.Tensor, slot: torch.Tensor
    ) -> None:
        """Parked trees asleep, active awake (NVMAX compact solve assumes this)."""
        try:
            ta = self._env.sim.data.tree_awake  # (nworld, ntree)
        except AttributeError:
            return  # MJWarp < 3.10: no sleeping islands
        trees = self._slot_tree_ids(side)
        ta[env_ids[:, None], trees[None, :]] = 0
        ta[env_ids, trees[slot]] = 1

    def _park_root_state(self, slot: int, env_ids: torch.Tensor) -> torch.Tensor:
        n = env_ids.shape[0]
        state = torch.zeros(n, 13, device=self.device)
        park = torch.tensor(
            object_park_position(slot, on_ground=self._sleep_mode),
            device=self.device,
            dtype=state.dtype,
        )
        state[:, 0:3] = self._env.scene.env_origins[env_ids] + park
        state[:, 3] = 1.0  # identity quat (w, x, y, z)
        return state

    def _repark_inactive_objects(self) -> None:
        """Re-park inactive objects (else they free-fall) in one masked write."""
        for side in self._side_list:
            if side not in self.motion_lib.obj_trans:
                continue
            names = self.obj_entity_names(side)
            if len(names) == 1:
                continue
            slot = self.active_obj_slot(side)
            bufs = self._parked_write_buffers(side)
            if bufs is None:
                self._repark_inactive_objects_loop(side, names, slot)
                continue
            q_adr, v_adr, parked_qpos, slots_arange, _body_ids = bufs
            data = self._env.scene[names[0]].data.data  # sim-wide qpos/qvel views
            mask = (slot[:, None] != slots_arange[None, :])[..., None]  # (B, S, 1)
            cur_q = data.qpos[:, q_adr]  # (B, S, 7)
            data.qpos[:, q_adr] = torch.where(mask, parked_qpos, cur_q)
            cur_v = data.qvel[:, v_adr]  # (B, S, 6)
            data.qvel[:, v_adr] = torch.where(mask, torch.zeros_like(cur_v), cur_v)

    def _seed_objects_loop(
        self,
        side: str,
        names: list[str],
        env_ids: torch.Tensor,
        root_state: torch.Tensor,
        slot: torch.Tensor,
    ) -> None:
        for s, name in enumerate(names):
            obj: Entity = self._env.scene[name]
            active = slot == s
            ids_a = env_ids[active]
            if ids_a.numel() > 0:
                obj.write_root_state_to_sim(root_state[active], env_ids=ids_a)
            ids_p = env_ids[~active]
            if ids_p.numel() > 0:
                obj.write_root_state_to_sim(
                    self._park_root_state(s, ids_p), env_ids=ids_p
                )
                zeros = torch.zeros(ids_p.shape[0], 1, 3, device=self.device)
                obj.write_external_wrench_to_sim(
                    forces=zeros, torques=zeros.clone(), env_ids=ids_p
                )
            obj.reset(env_ids=env_ids)

    def _repark_inactive_objects_loop(
        self, side: str, names: list[str], slot: torch.Tensor
    ) -> None:
        for s, name in enumerate(names):
            ids = torch.where(slot != s)[0]
            if ids.numel() == 0:
                continue
            obj: Entity = self._env.scene[name]
            obj.write_root_state_to_sim(self._park_root_state(s, ids), env_ids=ids)

    def _parked_write_buffers(
        self, side: str
    ) -> (
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        | None
    ):
        """(q_adr, v_adr, parked qpos, arange, body_ids); None = legacy loops."""
        cache = getattr(self, "_repark_buffers", None)
        if cache is None:
            cache = {}
            self._repark_buffers = cache
        if side in cache:
            return cache[side]
        names = self.obj_entity_names(side)
        out = None
        if os.environ.get("FDR_LEGACY_SLOT_WRITES"):
            cache[side] = None
            return None
        try:
            q_adr = torch.stack(
                [self._env.scene[n].data.indexing.free_joint_q_adr for n in names]
            ).to(self.device)
            v_adr = torch.stack(
                [self._env.scene[n].data.indexing.free_joint_v_adr for n in names]
            ).to(self.device)
            if q_adr.shape == (len(names), 7) and v_adr.shape == (len(names), 6):
                B = self._env.num_envs
                parked = torch.zeros(B, len(names), 7, device=self.device)
                for s in range(len(names)):
                    p = torch.tensor(
                        object_park_position(s, on_ground=self._sleep_mode),
                        device=self.device,
                        dtype=parked.dtype,
                    )
                    parked[:, s, 0:3] = self._env.scene.env_origins + p
                    parked[:, s, 3] = 1.0  # identity quat (w, x, y, z)
                body_ids = torch.stack(
                    [
                        self._env.scene[n].data.indexing.body_ids.to(self.device)
                        for n in names
                    ]
                )  # (S, nb)
                out = (
                    q_adr,
                    v_adr,
                    parked,
                    torch.arange(len(names), device=self.device),
                    body_ids,
                )
        except (AttributeError, RuntimeError):
            out = None
        cache[side] = out
        return out
