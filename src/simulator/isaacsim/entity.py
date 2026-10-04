"""mjlab-Entity-compatible adapters over Isaac assets, reading PhysX views with
a freshness token. Conventions: wxyz quats, world frame, MJCF ordering."""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

import torch
from mjlab.utils.lab_api.math import quat_apply, quat_mul

from .mjcf_meta import RobotMjcfMeta, site_table

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject


def _xyzw_to_wxyz(q: torch.Tensor) -> torch.Tensor:
    return q[..., [3, 0, 1, 2]]


class _FreshnessClock:
    """Monotonic token shared by all adapters; bump = invalidate caches."""

    def __init__(self) -> None:
        self.token = 0

    def bump(self) -> None:
        self.token += 1


class _Cached:
    """Per-clock-token memo for the raw physx fetches."""

    def __init__(self, clock: _FreshnessClock) -> None:
        self._clock = clock
        self._token = -1
        self._store: dict[str, torch.Tensor] = {}

    def get(self, key: str, fn) -> torch.Tensor:
        if self._token != self._clock.token:
            self._store.clear()
            self._token = self._clock.token
        v = self._store.get(key)
        if v is None:
            v = fn()
            self._store[key] = v
        return v


# ─── Robot (articulation) ────────────────────────────────────────────────────


class ArticulationData:
    """Duck-typed ``EntityData`` view for the robot articulation."""

    def __init__(self, adapter: "ArticulationAdapter") -> None:
        self._a = adapter

    # joint space (entity == MJCF order)
    @property
    def joint_pos(self) -> torch.Tensor:
        return self._a._joint_pos()

    @property
    def joint_vel(self) -> torch.Tensor:
        return self._a._joint_vel()

    @property
    def joint_pos_target(self) -> torch.Tensor:
        return self._a._joint_pos_target

    @property
    def soft_joint_pos_limits(self) -> torch.Tensor:
        return self._a._soft_limits

    @property
    def default_joint_pos(self) -> torch.Tensor:
        return self._a._default_joint_pos

    @property
    def default_joint_vel(self) -> torch.Tensor:
        return self._a._default_joint_vel

    @property
    def qfrc_actuator(self) -> torch.Tensor:
        # applied joint efforts (implicit PD), entity joint order
        tau = self._a._art.data.applied_torque
        return tau[:, self._a._isaac_joint_ids]

    # bodies (entity == MJCF order, world excluded)
    @property
    def body_link_pos_w(self) -> torch.Tensor:
        return self._a._body_pose()[0]

    @property
    def body_link_quat_w(self) -> torch.Tensor:
        return self._a._body_pose()[1]

    @property
    def body_link_lin_vel_w(self) -> torch.Tensor:
        return self._a._body_vel()[0]

    @property
    def body_link_ang_vel_w(self) -> torch.Tensor:
        return self._a._body_vel()[1]

    # root == first MJCF body (fixed-base chain anchor), matching mjlab's
    # ``indexing.root_body_id`` semantics for this entity.
    @property
    def root_link_pos_w(self) -> torch.Tensor:
        return self.body_link_pos_w[:, 0]

    @property
    def root_link_quat_w(self) -> torch.Tensor:
        return self.body_link_quat_w[:, 0]

    @property
    def root_link_lin_vel_w(self) -> torch.Tensor:
        return self.body_link_lin_vel_w[:, 0]

    @property
    def root_link_ang_vel_w(self) -> torch.Tensor:
        return self.body_link_ang_vel_w[:, 0]

    # sites (MJCF site frames rigidly attached to parent bodies)
    @property
    def site_pos_w(self) -> torch.Tensor:
        return self._a._site_kin()[0]

    @property
    def site_quat_w(self) -> torch.Tensor:
        return self._a._site_kin()[1]

    @property
    def site_lin_vel_w(self) -> torch.Tensor:
        return self._a._site_kin()[2]

    @property
    def site_ang_vel_w(self) -> torch.Tensor:
        return self._a._site_kin()[3]


class ArticulationAdapter:
    """mjlab ``Entity`` surface for the xhand articulation."""

    is_fixed_base = True

    def __init__(
        self,
        articulation: "Articulation",
        meta: RobotMjcfMeta,
        clock: _FreshnessClock,
        device: str,
        spec_fn=None,
    ) -> None:
        self._art = articulation
        self._meta = meta
        self._clock = clock
        self._cache = _Cached(clock)
        self.device = device

        self.body_names: list[str] = list(meta.body_names)
        self.joint_names: list[str] = list(meta.joint_names)
        self.num_joints = len(self.joint_names)
        self._body_index = {n: i for i, n in enumerate(self.body_names)}

        # MJCF -> URDF link-name divergences (the URDF twin renames the palms).
        alias = {
            "R_forearm_rot_y_link": "right_hand_link",
            "L_forearm_rot_y_link": "left_hand_link",
        }
        # Only the xhand URDF renames the palms; MJCF-named USDs (sharpa) keep the MJCF name.
        have = set(articulation.body_names)
        urdf_names = [alias[n] if (n in alias and n not in have) else n for n in self.body_names]

        # entity(MJCF) order -> isaac articulation order permutations
        ids, _ = articulation.find_bodies(urdf_names, preserve_order=True)
        self._isaac_body_ids = torch.tensor(ids, dtype=torch.long, device=device)
        jids, _ = articulation.find_joints(self.joint_names, preserve_order=True)
        self._isaac_joint_ids = torch.tensor(jids, dtype=torch.long, device=device)
        # inverse permutation: isaac joint slot -> entity column
        inv = torch.empty_like(self._isaac_joint_ids)
        inv[torch.argsort(self._isaac_joint_ids)] = torch.arange(
            self.num_joints, device=device
        )

        self.site_names, self._site_parent, self._site_lpos, self._site_lquat = (
            site_table(meta, self._body_index, device)
        )

        B = articulation.num_instances
        self._joint_pos_target = torch.zeros(B, self.num_joints, device=device)
        limits = torch.tensor(meta.joint_range, dtype=torch.float32, device=device)
        self._soft_limits = limits.unsqueeze(0).expand(B, -1, -1).contiguous()
        self._default_joint_pos = torch.zeros(B, self.num_joints, device=device)
        self._default_joint_vel = torch.zeros(B, self.num_joints, device=device)

        # link-frame <- com offset (constant, physx link com in link frame)
        view = articulation.root_physx_view
        coms = view.get_coms()  # (B, nl, 7) or (B*nl, 7) depending on backend
        coms = coms.reshape(B, -1, 7)[..., :3].to(device)
        self._com_local = coms  # (B, nl, 3), isaac link order
        self._gravcomp_full: torch.Tensor | None = None  # (B, nl, 3) world +z m*g per link
        self._gravcomp_scale = 0.0

        # A standalone MJCF spec factory so the command's CPU-FK reference
        # path (``_compute_robot_ref_state``) works unchanged.
        if spec_fn is None:
            import mujoco

            xml = meta.xml_path
            spec_fn = lambda: mujoco.MjSpec.from_file(xml)  # noqa: E731
        from types import SimpleNamespace

        self.cfg = SimpleNamespace(spec_fn=spec_fn)
        self.data = ArticulationData(self)

    # ── name/index resolution (mjlab API) ──────────────────────────────────

    def find_sites(self, names, preserve_order: bool = False):
        from mjlab.utils.lab_api.string import resolve_matching_names

        return resolve_matching_names(names, self.site_names, preserve_order)

    def find_joints(self, names, preserve_order: bool = False):
        from mjlab.utils.lab_api.string import resolve_matching_names

        return resolve_matching_names(names, self.joint_names, preserve_order)

    def find_bodies(self, names, preserve_order: bool = False):
        from mjlab.utils.lab_api.string import resolve_matching_names

        return resolve_matching_names(names, self.body_names, preserve_order)

    def find_joints_by_actuator_names(self, names):
        """Actuator names OR joint names -> entity joint ids (input order)."""
        joint_names = []
        for n in names:
            if n in self._meta.actuator_joint:
                joint_names.append(self._meta.actuator_joint[n])
            elif n in self.joint_names:
                joint_names.append(n)
            else:
                raise KeyError(f"unknown actuator/joint name {n!r}")
        return [self.joint_names.index(j) for j in joint_names], joint_names

    # ── raw fetches (freshness-cached) ─────────────────────────────────────

    def _joint_pos(self) -> torch.Tensor:
        def fetch():
            q = self._art.root_physx_view.get_dof_positions().to(self.device)
            return q[:, self._isaac_joint_ids]

        return self._cache.get("joint_pos", fetch)

    def _joint_vel(self) -> torch.Tensor:
        def fetch():
            v = self._art.root_physx_view.get_dof_velocities().to(self.device)
            return v[:, self._isaac_joint_ids]

        return self._cache.get("joint_vel", fetch)

    def _body_pose(self):
        def fetch_pair():
            tf = self._art.root_physx_view.get_link_transforms()
            tf = tf.reshape(self._art.num_instances, -1, 7).to(self.device)
            return tf

        tf = self._cache.get("link_tf", fetch_pair)
        pos = tf[:, self._isaac_body_ids, :3]
        quat = _xyzw_to_wxyz(tf[:, self._isaac_body_ids, 3:7])
        return pos, quat

    def _body_vel(self):
        def fetch_vel():
            v = self._art.root_physx_view.get_link_velocities()
            return v.reshape(self._art.num_instances, -1, 6).to(self.device)

        v = self._cache.get("link_vel", fetch_vel)
        tf = self._cache.get(
            "link_tf",
            lambda: self._art.root_physx_view.get_link_transforms()
            .reshape(self._art.num_instances, -1, 7)
            .to(self.device),
        )
        # physx: linear velocity reported at link COM; shift to link origin.
        quat_all = _xyzw_to_wxyz(tf[..., 3:7])
        r_com = quat_apply(quat_all, self._com_local)  # link->world com offset
        lin_com = v[..., 0:3]
        ang = v[..., 3:6]
        lin_link = lin_com + torch.cross(ang, -r_com, dim=-1)
        lin = lin_link[:, self._isaac_body_ids]
        ang = ang[:, self._isaac_body_ids]
        return lin, ang

    def _site_kin(self):
        def compute():
            pos, quat = self._body_pose()
            lin, ang = self._body_vel()
            bp = pos[:, self._site_parent]  # (B, S, 3)
            bq = quat[:, self._site_parent]
            bl = lin[:, self._site_parent]
            ba = ang[:, self._site_parent]
            lpos = self._site_lpos.unsqueeze(0).expand(bp.shape[0], -1, -1)
            lquat = self._site_lquat.unsqueeze(0).expand(bq.shape[0], -1, -1)
            r = quat_apply(bq, lpos)
            spos = bp + r
            squat = quat_mul(bq, lquat)
            slin = bl + torch.cross(ba, r, dim=-1)
            return spos, squat, slin, ba

        key = "site_kin"
        if self._cache._token != self._clock.token or key not in self._cache._store:
            out = compute()
            # stash as tuple via individual keys
            self._cache.get("_sync", lambda: torch.zeros(1))
            self._cache._store[key] = out  # type: ignore[assignment]
        return self._cache._store[key]

    # ── writes (mjlab API) ─────────────────────────────────────────────────

    def write_joint_state_to_sim(
        self,
        position: torch.Tensor,
        velocity: torch.Tensor,
        joint_ids=None,
        env_ids: torch.Tensor | None = None,
    ) -> None:
        assert joint_ids is None, "full-state writes only"
        ids = self._isaac_joint_ids
        self._art.write_joint_state_to_sim(
            position.to(self.device), velocity.to(self.device),
            joint_ids=ids.tolist(), env_ids=env_ids,
        )
        # keep targets consistent with the teleported pose
        if env_ids is None:
            self._joint_pos_target[:] = position
        else:
            self._joint_pos_target[env_ids] = position
        self._art.set_joint_position_target(
            position.to(self.device), joint_ids=ids.tolist(), env_ids=env_ids
        )
        self._clock.bump()

    def set_joint_position_target(
        self, target: torch.Tensor, joint_ids: torch.Tensor | list | None = None
    ) -> None:
        if joint_ids is None:
            ent_ids = torch.arange(self.num_joints, device=self.device)
        elif isinstance(joint_ids, torch.Tensor):
            ent_ids = joint_ids
        else:
            ent_ids = torch.tensor(joint_ids, dtype=torch.long, device=self.device)
        self._joint_pos_target[:, ent_ids] = target
        isaac_ids = self._isaac_joint_ids[ent_ids]
        self._art.set_joint_position_target(target, joint_ids=isaac_ids.tolist())

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        self._art.reset(env_ids=env_ids)
        # Isaac Lab reset zeroes the external wrench; restore the gravity compensation.
        if self._gravcomp_full is not None:
            self._write_gravcomp()

    def init_gravcomp(self, scale: float) -> None:
        """Per-link world-frame m*g lift at the COM, kept through resets (MJCF gravcomp)."""
        masses = self._art.root_physx_view.get_masses().to(self.device)
        masses = masses.reshape(self._art.num_instances, -1)
        self._gravcomp_full = torch.zeros(*masses.shape, 3, device=self.device)
        self._gravcomp_full[..., 2] = 9.81 * masses
        self._gravcomp_scale = float(scale)
        self._write_gravcomp()

    def _write_gravcomp(self) -> None:
        # Always the full buffer: Isaac Lab derives has_external_wrench from the written slice only.
        f = self._gravcomp_full * self._gravcomp_scale
        self._art.set_external_force_and_torque(f, torch.zeros_like(f), is_global=True)


# ─── Object (single free rigid body) ─────────────────────────────────────────


class _ObjIndexing:
    def __init__(self, body_ids: torch.Tensor) -> None:
        self.body_ids = body_ids


class RigidObjectData:
    def __init__(self, adapter: "RigidObjectAdapter") -> None:
        self._a = adapter
        self.indexing = adapter.indexing

    @property
    def default_root_state(self) -> torch.Tensor:
        return self._a._default_root_state

    @property
    def body_link_pos_w(self) -> torch.Tensor:
        return self._a._pose()[0].unsqueeze(1)

    @property
    def body_link_quat_w(self) -> torch.Tensor:
        return self._a._pose()[1].unsqueeze(1)

    @property
    def body_link_lin_vel_w(self) -> torch.Tensor:
        return self._a._vel()[0].unsqueeze(1)

    @property
    def body_link_ang_vel_w(self) -> torch.Tensor:
        return self._a._vel()[1].unsqueeze(1)

    @property
    def root_link_pos_w(self) -> torch.Tensor:
        return self._a._pose()[0]

    @property
    def root_link_quat_w(self) -> torch.Tensor:
        return self._a._pose()[1]


class RigidObjectAdapter:
    """mjlab ``Entity`` surface for one object slot (single free body)."""

    is_fixed_base = False

    def __init__(
        self,
        obj: "RigidObject",
        global_body_id: int,
        clock: _FreshnessClock,
        device: str,
        default_root_state: torch.Tensor,
    ) -> None:
        self._obj = obj
        self._clock = clock
        self._cache = _Cached(clock)
        self.device = device
        self.indexing = _ObjIndexing(
            torch.tensor([global_body_id], dtype=torch.long, device=device)
        )
        self._default_root_state = default_root_state  # (B, 13)
        view = obj.root_physx_view
        com = view.get_coms().reshape(obj.num_instances, -1)[:, :3].to(device)
        self._com_local = com  # (B, 3)
        self.data = RigidObjectData(self)
        # world-frame external wrench support probe (isaaclab >= is_global)
        sig = inspect.signature(obj.set_external_force_and_torque)
        self._has_is_global = "is_global" in sig.parameters

    def _pose(self):
        def fetch():
            tf = self._obj.root_physx_view.get_transforms().to(self.device)
            return tf

        tf = self._cache.get("tf", fetch)
        return tf[:, :3], _xyzw_to_wxyz(tf[:, 3:7])

    def _vel(self):
        v = self._cache.get(
            "vel", lambda: self._obj.root_physx_view.get_velocities().to(self.device)
        )
        pos, quat = self._pose()
        r_com = quat_apply(quat, self._com_local)
        lin = v[:, 0:3] + torch.cross(v[:, 3:6], -r_com, dim=-1)
        return lin, v[:, 3:6]

    # ── writes ─────────────────────────────────────────────────────────────

    def write_root_state_to_sim(
        self, root_state: torch.Tensor, env_ids: torch.Tensor | None = None
    ) -> None:
        pose = root_state[:, 0:7].to(self.device)
        lin_link = root_state[:, 7:10].to(self.device)
        ang = root_state[:, 10:13].to(self.device)
        self._obj.write_root_link_pose_to_sim(pose, env_ids=env_ids)
        # convert link-origin linear velocity -> com velocity for physx
        quat = pose[:, 3:7]
        if env_ids is None:
            com_local = self._com_local
        else:
            com_local = self._com_local[env_ids]
        r_com = quat_apply(quat, com_local)
        lin_com = lin_link + torch.cross(ang, r_com, dim=-1)
        vel = torch.cat([lin_com, ang], dim=-1)
        self._obj.write_root_com_velocity_to_sim(vel, env_ids=env_ids)
        self._clock.bump()

    def write_external_wrench_to_sim(
        self,
        forces: torch.Tensor,
        torques: torch.Tensor,
        env_ids: torch.Tensor | None = None,
    ) -> None:
        f = forces.reshape(-1, 1, 3)
        t = torques.reshape(-1, 1, 3)
        # Apply at the COM (mujoco xfrc parity); FDR_PIN_AT_COM=0
        # reverts to the default application point for A/B probing.
        import os as _os

        if _os.environ.get("FDR_PIN_AT_COM", "0") == "1":
            com = self._com_local if env_ids is None else self._com_local[env_ids]
            pos = com.reshape(-1, 1, 3)
        else:
            pos = None
        if self._has_is_global:
            self._obj.set_external_force_and_torque(
                f, t, positions=pos, env_ids=env_ids, is_global=True
            )
            return
        # world -> body frame (isaaclab applies wrench in body frame)
        _, quat = self._pose()
        if env_ids is not None:
            quat = quat[env_ids]
        from mjlab.utils.lab_api.math import quat_apply_inverse

        q = quat.unsqueeze(1)
        self._obj.set_external_force_and_torque(
            quat_apply_inverse(q, f), quat_apply_inverse(q, t),
            positions=pos, env_ids=env_ids,
        )

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        self._obj.reset(env_ids=env_ids)
