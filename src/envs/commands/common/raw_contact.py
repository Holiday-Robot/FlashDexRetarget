"""Raw-contact-buffer contact sensing for multi-object mode: replaces the per-slot native
contact sensors with ONE torch pass, replicating mujoco_warp's sensor semantics exactly."""

from __future__ import annotations

from typing import TYPE_CHECKING

import mujoco
import numpy as np
import torch
from mjlab.sensor.contact_sensor import ContactData

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


class RawContactReader:
    """One-pass contact extraction for all replicated multi-object sensors. Spec dict keys:
    name, primary_mode (body|site), primary_names (entity-local), primary_entity, fields, reduce."""

    def __init__(self, env: ManagerBasedRlEnv, command, side: str, specs: list[dict]):
        self._env = env
        self._command = command
        self._side = side
        self._specs = [dict(s) for s in specs]
        self._cache: dict[str, ContactData] | None = None

        mj_model = env.sim.mj_model
        device = env.device
        self._cone = int(mj_model.opt.cone)
        if self._cone not in (
            int(mujoco.mjtCone.mjCONE_PYRAMIDAL),
            int(mujoco.mjtCone.mjCONE_ELLIPTIC),
        ):
            raise NotImplementedError(f"unsupported opt.cone={self._cone}")

        nbody, ngeom, nsite = mj_model.nbody, mj_model.ngeom, mj_model.nsite

        def _name(objtype, i):
            return mujoco.mj_id2name(mj_model, objtype, i) or ""

        body_names = [_name(mujoco.mjtObj.mjOBJ_BODY, i) for i in range(nbody)]
        site_names = [_name(mujoco.mjtObj.mjOBJ_SITE, i) for i in range(nsite)]

        # Object slot table: body id → slot idx (−1 elsewhere). Slot-free swap mode has ONE
        # shared entity: its body is unconditionally "the active object" (no slot compare).
        self._single_entity = len(command.obj_entity_names(side)) == 1
        slot_of_body = np.full(nbody, -1, dtype=np.int64)
        for slot, ent_name in enumerate(command.obj_entity_names(side)):
            ent = env.scene[ent_name]
            for bid in ent.indexing.body_ids.cpu().numpy().tolist():
                slot_of_body[int(bid)] = slot
        self._slot_of_body = torch.as_tensor(slot_of_body, device=device)
        self._body_of_geom = torch.as_tensor(
            mj_model.geom_bodyid.astype(np.int64), device=device
        )

        # ── per-spec primary resolution ─────────────────────────────────────
        for spec in self._specs:
            prefix = f"{spec['primary_entity']}/"
            names = [prefix + n for n in spec["primary_names"]]
            if spec["primary_mode"] == "body":
                link_of_body = np.full(nbody, -1, dtype=np.int64)
                for k, nm in enumerate(names):
                    bid = body_names.index(nm)
                    link_of_body[bid] = k
                spec["link_of_body"] = torch.as_tensor(link_of_body, device=device)
            elif spec["primary_mode"] == "site":
                sids = [site_names.index(nm) for nm in names]
                spec["site_ids"] = torch.as_tensor(sids, device=device)
                spec["site_type"] = torch.as_tensor(
                    mj_model.site_type[sids].astype(np.int64), device=device
                )
                spec["site_size"] = torch.as_tensor(
                    mj_model.site_size[sids].astype(np.float32), device=device
                )
            else:
                raise ValueError(f"unsupported primary_mode {spec['primary_mode']!r}")
            spec["n_primary"] = len(names)

    # ── cache management (mirrors native sensor per-step invalidation) ─────

    def invalidate(self) -> None:
        self._cache = None

    def get(self, name: str) -> ContactData:
        if self._cache is None:
            self._cache = self._compute()
        return self._cache[name]

    # ── core ────────────────────────────────────────────────────────────────

    def _decode_forces(self, d, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode contact-frame wrench for every valid contact slot. Returns (f_local, t_local):
        (n, 3) force and torque in the contact frame (torque nonzero only for condim > 3)."""
        dim = d.contact.dim.long()  # (n,)
        adr = d.contact.efc_address[:, 0].long()  # (n,)
        mu = d.contact.friction  # (n, 5)
        wid = d.contact.worldid.long()  # (n,)
        efc_force = d.efc.force  # (nworld, njmax)
        njmax = efc_force.shape[1]
        n = dim.shape[0]

        f_local = torch.zeros(n, 3, device=dim.device)
        t_local = torch.zeros(n, 3, device=dim.device)
        ok = valid & (adr >= 0)

        if self._cone == int(mujoco.mjtCone.mjCONE_ELLIPTIC):
            # Elliptic: efc rows map DIRECTLY to wrench components
            # (normal, t1, t2, torsion, roll1, roll2) — one row per condim.
            addr = d.contact.efc_address.long()  # (n, >=condim)
            ncols = addr.shape[1]
            for i in range(min(6, ncols)):
                mi = ok & (dim > i)
                if not mi.any():
                    continue
                ai = addr[:, i]
                mi = mi & (ai < njmax) & (ai >= 0)
                if not mi.any():
                    continue
                val = efc_force[wid[mi], ai[mi]]
                if i < 3:
                    f_local[mi, i] = val
                else:
                    t_local[mi, i - 3] = val
            return f_local, t_local

        # Pyramidal cone.
        # condim == 1: single normal row.
        m1 = ok & (dim == 1)
        if m1.any():
            a = adr[m1].clamp(max=njmax - 1)
            f_local[m1, 0] = efc_force[wid[m1], a]

        # condim >= 3: pyramid pairs; component i+1 = (dir1 - dir2) * mu[i].
        for cd in (3, 4, 6):
            mc = ok & (dim == cd)
            if not mc.any():
                continue
            a0 = adr[mc]
            w = wid[mc]
            fn = torch.zeros(a0.shape[0], device=dim.device)
            for i in range(cd - 1):
                a1 = a0 + 2 * i
                a2 = a0 + 2 * i + 1
                d1 = torch.where(
                    a1 < njmax, efc_force[w, a1.clamp(max=njmax - 1)], torch.zeros_like(fn)
                )
                d2 = torch.where(
                    a2 < njmax, efc_force[w, a2.clamp(max=njmax - 1)], torch.zeros_like(fn)
                )
                fn = fn + d1 + d2
                comp = (d1 - d2) * mu[mc, i]
                if i < 2:
                    f_local[mc, i + 1] = comp
                else:
                    t_local[mc, i - 2] = comp
            f_local[mc, 0] = fn

        return f_local, t_local

    def _site_inside(self, spec: dict, pos: torch.Tensor, wid: torch.Tensor) -> torch.Tensor:
        """(n, n_sites) bool: contact pos inside each primary site's volume."""
        d = self._env.sim.data
        sids = spec["site_ids"]  # (S,)
        wid_safe = wid.clamp(min=0, max=d.site_xpos.shape[0] - 1)
        # Gather the S primary sites FIRST (small), then per-contact worlds.
        # site_xpos: (nworld, nsite, 3); site_xmat: (nworld, nsite, 3, 3)
        sp = d.site_xpos[:, sids][wid_safe]  # (n, S, 3)
        sm = d.site_xmat[:, sids][wid_safe]  # (n, S, 3, 3) local→world
        rel = pos[:, None, :] - sp  # (n, S, 3)
        # local = R^T @ rel  (xmat maps local→world)
        local = torch.einsum("nsij,nsi->nsj", sm, rel)  # R^T rel via row index i
        size = spec["site_size"]  # (S, 3)
        stype = spec["site_type"]  # (S,)
        x, y, z = local[..., 0], local[..., 1], local[..., 2]
        r2 = local.pow(2).sum(-1)

        inside = torch.zeros(local.shape[:2], dtype=torch.bool, device=local.device)
        SPHERE = int(mujoco.mjtGeom.mjGEOM_SPHERE)
        CAPSULE = int(mujoco.mjtGeom.mjGEOM_CAPSULE)
        ELLIPSOID = int(mujoco.mjtGeom.mjGEOM_ELLIPSOID)
        CYLINDER = int(mujoco.mjtGeom.mjGEOM_CYLINDER)
        BOX = int(mujoco.mjtGeom.mjGEOM_BOX)
        for s in range(sids.shape[0]):
            t = int(stype[s])
            sz = size[s]
            if t == SPHERE:
                inside[:, s] = r2[:, s] < sz[0] ** 2
            elif t == CAPSULE:
                zz = z[:, s].clamp(-sz[1], sz[1])
                inside[:, s] = (x[:, s] ** 2 + y[:, s] ** 2 + (z[:, s] - zz) ** 2) < sz[0] ** 2
            elif t == ELLIPSOID:
                inside[:, s] = (local[:, s] / sz[None, :]).pow(2).sum(-1) < 1.0
            elif t == CYLINDER:
                inside[:, s] = (z[:, s].abs() < sz[1]) & (
                    (x[:, s] ** 2 + y[:, s] ** 2) < sz[0] ** 2
                )
            elif t == BOX:
                inside[:, s] = (
                    (x[:, s].abs() < sz[0])
                    & (y[:, s].abs() < sz[1])
                    & (z[:, s].abs() < sz[2])
                )
            else:
                raise NotImplementedError(f"site type {t} volume test")
        return inside

    def _compute(self) -> dict[str, ContactData]:
        env = self._env
        d = env.sim.data
        B = env.num_envs
        device = env.device

        geom = d.contact.geom.long()  # (n, 2)
        n = geom.shape[0]
        wid = d.contact.worldid.long()  # (n,)
        valid = torch.arange(n, device=device) < d.nacon[0]

        b1 = self._body_of_geom[geom[:, 0]]
        b2 = self._body_of_geom[geom[:, 1]]
        slot1 = self._slot_of_body[b1]
        slot2 = self._slot_of_body[b2]
        if self._single_entity:
            obj1 = valid & (slot1 >= 0)
            obj2 = valid & (slot2 >= 0)
        else:
            act = self._command.active_obj_slot(self._side)  # (B,)
            act_c = act[wid.clamp(max=B - 1)]  # (n,)
            obj1 = valid & (slot1 >= 0) & (slot1 == act_c)
            obj2 = valid & (slot2 >= 0) & (slot2 == act_c)

        f_local, t_local = self._decode_forces(d, valid)
        frame = d.contact.frame  # (n, 3, 3), rows = axes
        F_w = torch.einsum("nij,ni->nj", frame, f_local)  # (n, 3)
        T_w = torch.einsum("nij,ni->nj", frame, t_local)  # (n, 3)
        pos = d.contact.pos  # (n, 3)
        dist = d.contact.dist  # (n,)
        weight = F_w.norm(dim=-1)  # |force|, sign-independent

        out: dict[str, ContactData] = {}
        for spec in self._specs:
            NP = spec["n_primary"]
            if spec["primary_mode"] == "body":
                link1 = spec["link_of_body"][b1]
                link2 = spec["link_of_body"][b2]
                regular = (link1 >= 0) & obj2
                reverse = (link2 >= 0) & obj1
                matched = regular | reverse
                link = torch.where(regular, link1, link2)
                dirn = torch.where(regular, 1.0, -1.0)
                m = matched
                key = (wid * NP + link)[m]
                dir_m = dirn[m]
            else:  # site
                inside = self._site_inside(spec, pos, wid)  # (n, S)
                involved = obj1 | obj2
                matched2d = inside & involved[:, None] & valid[:, None]
                # dir: normal must point toward the object (secondary).
                dirn = torch.where(obj2, 1.0, -1.0)  # (n,)
                m2 = matched2d
                cidx, sidx = m2.nonzero(as_tuple=True)
                key = wid[cidx] * NP + sidx
                m = cidx  # index tensor (may repeat contacts across sites)
                dir_m = dirn[cidx]

            found = torch.zeros(B * NP, device=device)
            found.index_add_(0, key, torch.ones_like(key, dtype=found.dtype))

            data_kwargs: dict[str, torch.Tensor | None] = {
                "found": found.view(B, NP)
            }

            if spec["reduce"] == "netforce":
                Fm = F_w[m] * dir_m[:, None]
                Tm = (T_w[m] + torch.cross(pos[m], F_w[m], dim=-1)) * dir_m[:, None]
                wm = weight[m]
                net_F = torch.zeros(B * NP, 3, device=device)
                net_T = torch.zeros(B * NP, 3, device=device)
                wsum = torch.zeros(B * NP, device=device)
                wpos = torch.zeros(B * NP, 3, device=device)
                net_F.index_add_(0, key, Fm)
                net_T.index_add_(0, key, Tm)
                wsum.index_add_(0, key, wm)
                wpos.index_add_(0, key, pos[m] * wm[:, None])
                net_pos = wpos / wsum.clamp(min=1e-15)[:, None]
                net_T = net_T - torch.cross(net_pos, net_F, dim=-1)
                data_kwargs["force"] = net_F.view(B, NP, 3)
                data_kwargs["torque"] = net_T.view(B, NP, 3)
                data_kwargs["pos"] = net_pos.view(B, NP, 3)
            elif spec["reduce"] == "mindist":
                dm = dist[m]
                min_d = torch.full((B * NP,), torch.inf, device=device)
                min_d.scatter_reduce_(0, key, dm, reduce="amin", include_self=True)
                is_win = dm <= min_d[key]
                win_key = key[is_win]
                out_dist = torch.zeros(B * NP, device=device)
                out_pos = torch.zeros(B * NP, 3, device=device)
                out_dist[win_key] = dm[is_win]
                out_pos[win_key] = pos[m][is_win]
                data_kwargs["dist"] = out_dist.view(B, NP)
                data_kwargs["pos"] = out_pos.view(B, NP, 3)
            else:
                raise ValueError(f"unsupported reduce {spec['reduce']!r}")

            fields = set(spec["fields"])
            out[spec["name"]] = ContactData(
                found=data_kwargs["found"],
                force=data_kwargs.get("force") if "force" in fields else None,
                torque=data_kwargs.get("torque") if "torque" in fields else None,
                dist=data_kwargs.get("dist") if "dist" in fields else None,
                pos=data_kwargs.get("pos") if "pos" in fields else None,
            )
        return out


class RawContactSensorView:
    """Scene-sensor-shaped view over one RawContactReader output."""

    requires_sensor_context: bool = False

    def __init__(self, reader: RawContactReader, name: str) -> None:
        self._reader = reader
        self._name = name

    @property
    def data(self) -> ContactData:
        return self._reader.get(self._name)

    def reset(self, env_ids=None) -> None:
        self._reader.invalidate()

    def update(self, dt: float) -> None:
        self._reader.invalidate()

    def debug_vis(self, visualizer) -> None:
        pass


def register_raw_contact_views(
    env: ManagerBasedRlEnv, command, side: str, specs: list[dict]
) -> None:
    """Build one reader + register a view per replicated sensor name."""
    reader = RawContactReader(env, command, side, specs)
    for spec in specs:
        name = spec["name"]
        if name in env.scene.sensors:
            raise ValueError(f"raw contact view {name!r} collides with a scene sensor")
        env.scene.sensors[name] = RawContactSensorView(reader, name)
