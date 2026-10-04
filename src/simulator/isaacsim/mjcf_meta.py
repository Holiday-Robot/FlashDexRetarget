"""Robot metadata from the MJCF via CPU mujoco: joint order (= motion .pt
columns), PD gains, site frames. The MJCF stays the source of truth."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np
import torch


@dataclass
class SiteMeta:
    name: str
    body_name: str
    local_pos: np.ndarray  # (3,)
    local_quat: np.ndarray  # (4,) wxyz


@dataclass
class RobotMjcfMeta:
    xml_path: str
    body_names: list[str]  # MJCF order, world excluded
    joint_names: list[str]  # MJCF order == motion .pt column order
    joint_range: np.ndarray  # (n_jnt, 2)
    joint_damping: np.ndarray  # (n_jnt,)
    joint_armature: np.ndarray  # (n_jnt,)
    joint_frictionloss: np.ndarray  # (n_jnt,)
    actuator_names: list[str]  # MJCF order
    actuator_joint: dict[str, str]  # actuator name -> joint name
    actuator_kp: np.ndarray  # (n_act,) position-servo stiffness
    actuator_kv: np.ndarray  # (n_act,) position-servo damping
    actuator_forcerange: np.ndarray  # (n_act, 2)
    sites: dict[str, SiteMeta] = field(default_factory=dict)
    body_bound: dict[str, float] = field(default_factory=dict)

    @property
    def num_joints(self) -> int:
        return len(self.joint_names)

    def kp_kv_for_joint(self, joint_name: str) -> tuple[float, float]:
        for act, jnt in self.actuator_joint.items():
            if jnt == joint_name:
                i = self.actuator_names.index(act)
                return float(self.actuator_kp[i]), float(self.actuator_kv[i])
        raise KeyError(f"no actuator drives joint {joint_name!r}")


def load_robot_mjcf_meta(xml_path: str | Path) -> RobotMjcfMeta:
    xml_path = str(xml_path)
    m = mujoco.MjModel.from_xml_path(xml_path)

    def _name(objtype: mujoco.mjtObj, i: int) -> str:
        n = mujoco.mj_id2name(m, objtype, i)
        assert n is not None
        return n

    body_names = [
        _name(mujoco.mjtObj.mjOBJ_BODY, i) for i in range(1, m.nbody)
    ]  # skip world
    joint_names = [_name(mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(m.njnt)]
    dof = [int(m.jnt_dofadr[j]) for j in range(m.njnt)]

    actuator_names = [_name(mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)]
    actuator_joint = {
        actuator_names[i]: joint_names[int(m.actuator_trnid[i, 0])]
        for i in range(m.nu)
    }
    # MuJoCo position servo: gainprm[0] = kp, biasprm = [0, -kp, -kv].
    actuator_kp = m.actuator_gainprm[:, 0].copy()
    actuator_kv = -m.actuator_biasprm[:, 2].copy()

    sites: dict[str, SiteMeta] = {}
    for s in range(m.nsite):
        sname = _name(mujoco.mjtObj.mjOBJ_SITE, s)
        bid = int(m.site_bodyid[s])
        sites[sname] = SiteMeta(
            name=sname,
            body_name=_name(mujoco.mjtObj.mjOBJ_BODY, bid),
            local_pos=m.site_pos[s].copy(),
            local_quat=m.site_quat[s].copy(),
        )

    # Per-body collision reach: max over collision geoms of |offset| + rbound. Raw
    # MJCF ships collision geoms contype=0, so also match "collision_*" names.
    body_bound: dict[str, float] = {}
    for g in range(m.ngeom):
        gname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        active = m.geom_contype[g] != 0 or m.geom_conaffinity[g] != 0
        if not active and not gname.startswith("collision"):
            continue
        bname = _name(mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g]))
        reach = float(np.linalg.norm(m.geom_pos[g])) + float(m.geom_rbound[g])
        body_bound[bname] = max(body_bound.get(bname, 0.0), reach)

    return RobotMjcfMeta(
        xml_path=xml_path,
        body_names=body_names,
        joint_names=joint_names,
        joint_range=m.jnt_range.copy(),
        joint_damping=np.array([m.dof_damping[d] for d in dof]),
        joint_armature=np.array([m.dof_armature[d] for d in dof]),
        joint_frictionloss=np.array([m.dof_frictionloss[d] for d in dof]),
        actuator_names=actuator_names,
        actuator_joint=actuator_joint,
        actuator_kp=actuator_kp,
        actuator_kv=actuator_kv,
        actuator_forcerange=m.actuator_forcerange.copy(),
        sites=sites,
        body_bound=body_bound,
    )


def site_table(
    meta: RobotMjcfMeta, body_index: dict[str, int], device: str
) -> tuple[list[str], torch.Tensor, torch.Tensor, torch.Tensor]:
    """(site_names, parent_body_idx (S,), local_pos (S,3), local_quat (S,4))."""
    names = list(meta.sites.keys())
    parent = torch.tensor(
        [body_index[meta.sites[n].body_name] for n in names],
        dtype=torch.long,
        device=device,
    )
    pos = torch.tensor(
        np.stack([meta.sites[n].local_pos for n in names]),
        dtype=torch.float32,
        device=device,
    )
    quat = torch.tensor(
        np.stack([meta.sites[n].local_quat for n in names]),
        dtype=torch.float32,
        device=device,
    )
    return names, parent, pos, quat
