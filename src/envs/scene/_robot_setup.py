from __future__ import annotations

from typing import Any

import mujoco
from mjlab.actuator.xml_actuator import XmlActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg


def build_robot(cfg: Any) -> EntityCfg:
    """Compose a robot ``EntityCfg`` from a parsed ``robot`` yaml block."""
    xml_path = str(cfg.xml_path)
    return EntityCfg(
        init_state=_build_init_state(cfg.get("init_state")),
        spec_fn=lambda p=xml_path: mujoco.MjSpec.from_file(p),
        articulation=_build_articulation(cfg.get("articulation")),
        collisions=_build_collisions(cfg.get("collisions")),
    )


def _build_init_state(cfg: Any) -> EntityCfg.InitialStateCfg:
    if cfg is None:
        return EntityCfg.InitialStateCfg()
    kw: dict[str, Any] = {}
    if "pos" in cfg:
        kw["pos"] = tuple(cfg.pos)
    if "rot" in cfg:
        kw["rot"] = tuple(cfg.rot)
    if "lin_vel" in cfg:
        kw["lin_vel"] = tuple(cfg.lin_vel)
    if "ang_vel" in cfg:
        kw["ang_vel"] = tuple(cfg.ang_vel)
    if "joint_pos" in cfg:
        jp = cfg.joint_pos
        kw["joint_pos"] = None if jp is None else dict(jp)
    if "joint_vel" in cfg:
        kw["joint_vel"] = dict(cfg.joint_vel)
    return EntityCfg.InitialStateCfg(**kw)


def _build_articulation(cfg: Any) -> EntityArticulationInfoCfg | None:
    if cfg is None:
        return None
    actuators = tuple(_build_actuator(a) for a in (cfg.get("actuators") or ()))
    return EntityArticulationInfoCfg(
        actuators=actuators,
        soft_joint_pos_limit_factor=float(cfg.get("soft_joint_pos_limit_factor", 1.0)),
    )


def _build_actuator(a: Any) -> XmlActuatorCfg:
    return XmlActuatorCfg(target_names_expr=tuple(a.target_names_expr))


def _build_collisions(items: Any) -> tuple[CollisionCfg, ...]:
    out: list[CollisionCfg] = []
    for c in items or ():
        kw: dict[str, Any] = dict(geom_names_expr=tuple(c.geom_names_expr))
        if "contype" in c:
            kw["contype"] = int(c.contype)
        if "conaffinity" in c:
            kw["conaffinity"] = int(c.conaffinity)
        if "condim" in c:
            kw["condim"] = int(c.condim)
        if "priority" in c:
            kw["priority"] = int(c.priority)
        if c.get("friction") is not None:
            kw["friction"] = tuple(c.friction)
        if "disable_other_geoms" in c:
            kw["disable_other_geoms"] = bool(c.disable_other_geoms)
        out.append(CollisionCfg(**kw))
    return tuple(out)
