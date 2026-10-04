"""CHORD-style support disks as fixed-base (mocap) cylinder entities: count x radii bodies per
env, posed per motion at reset by MotionTrackingCommand._write_support_disks."""

from __future__ import annotations

from typing import Any

import mujoco
from mjlab.entity import EntityCfg


def support_disk_name(k: int, j: int) -> str:
    return f"support_k{k}_r{j}"


def support_park_position(cfg: Any, k: int, j: int) -> tuple[float, float, float]:
    """Parking slot for disk (k, j): one lane per radius bucket, stride along +y (objects park at -y)."""
    px, py, pz = (float(v) for v in cfg.park)
    return (px + 0.6 * j, py + float(cfg.park_stride) * k, pz)


def get_support_disk_spec(
    name: str, radius: float, height: float, friction, rgba, robot_collision: bool = True
) -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    spec.modelname = name
    body = spec.worldbody.add_body(name=name)  # no joint -> mjlab wraps it in a mocap base
    body.add_geom(
        name=name,
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=[float(radius), float(height) / 2, 0.0],
        # hand (contype/conaffinity 1) and object (2/3) contacts; disk-disk/floor are same-weld.
        # robot_collision=false -> 2/2: the object pair still holds, every hand pair is masked out.
        contype=1 if robot_collision else 2,
        conaffinity=3 if robot_collision else 2,
        condim=3,
        friction=[float(v) for v in friction],
        rgba=[float(v) for v in rgba],
    )
    return spec


def build_support_disks(cfg: Any) -> dict[str, EntityCfg]:
    """count x len(radii) fixed-base cylinder entities, or {} when disabled. Writes the
    [K][R] entity-name grid back into commands.object.support_disks.entity_names."""
    sd = cfg.commands.object.support_disks
    if not bool(sd.enable):
        return {}
    from omegaconf import open_dict

    entities: dict[str, EntityCfg] = {}
    names: list[list[str]] = []
    for k in range(int(sd.count)):
        row = []
        for j, r in enumerate(sd.radii):
            name = support_disk_name(k, j)
            entities[name] = EntityCfg(
                spec_fn=lambda n=name, rr=float(r), sd=sd: get_support_disk_spec(
                    n, rr, float(sd.height), list(sd.friction), list(sd.rgba),
                    bool(sd.get("robot_collision", True)),
                ),
                init_state=EntityCfg.InitialStateCfg(
                    pos=support_park_position(sd, k, j), rot=(1.0, 0.0, 0.0, 0.0)
                ),
            )
            row.append(name)
        names.append(row)
    with open_dict(sd):
        sd.entity_names = names
    return entities
