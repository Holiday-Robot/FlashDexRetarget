from __future__ import annotations

import copy
import os
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from mjlab.entity import EntityCfg
from omegaconf import OmegaConf

from ._object_entity import get_object_cfg

# Parking spots for INACTIVE slot objects: in the air outside the workspace,
# strided so they never touch.
_OBJ_PARK_BASE = (0.0, -2.0, 0.5)
_OBJ_PARK_STRIDE = 0.6


def object_pool_dir(motion_cfg: Any) -> Path:
    """A packed motion pt's objects live next to it: <motion dir>/objects/<id>."""
    return Path(str(motion_cfg.motion_file)).resolve().parent


def object_park_position(slot: int, on_ground: bool = False) -> tuple[float, float, float]:
    """World-frame parking position for object ``slot``. ``on_ground=True``
    (sleep mode) parks LOW so it settles and sleeps; else re-teleported per step."""
    # Demo toggle: DON'T send inactive objects away — pack them in a tight grid at
    # the workspace so the clutter that parking prevents is visible (any count).
    if os.environ.get("FDR_NO_PARK"):
        per = 6
        gx, gy, gz = slot % per, (slot // per) % per, slot // (per * per)
        return (0.12 * (gx - 2.5), 0.12 * (gy - 2.5), 0.35 + 0.12 * gz)
    return (
        _OBJ_PARK_BASE[0],
        _OBJ_PARK_BASE[1] - _OBJ_PARK_STRIDE * slot,
        0.2 if on_ground else _OBJ_PARK_BASE[2],
    )


def build_object(cfg: Any) -> EntityCfg:
    """Compose an object ``EntityCfg`` from a cfg block (obj_dir, body_name,
    density, mesh_scale); mirrors :func:`_robot_setup.build_robot`."""
    return get_object_cfg(
        obj_dir=str(cfg.obj_dir),
        name=str(cfg.body_name),
        density=float(cfg.density),
        mesh_scale=float(getattr(cfg, "mesh_scale", 1.0)),
    )


def discover_objects_from_motion(cfg: Any) -> dict[str, EntityCfg]:
    motion_cfg = cfg.commands
    motion_file = str(motion_cfg.motion_file)
    if not motion_file.endswith(".pt"):
        return {}

    packed = torch.load(motion_file, weights_only=False)
    if not any(packed.get(f"{s}_object_mesh_dir{p}") for s in ("right", "left") for p in ("", "s")):
        return {}

    pool_dir = object_pool_dir(motion_cfg)
    density = float(motion_cfg.object.density)

    # Multi-object packed pt (merge_motion_multi.py): one entity PER SLOT — or,
    # with FDR_OBJ_SWAP=1, ONE swapped entity (see scene/_object_swap.py).
    multi_sides = tuple(
        s for s in ("right", "left") if packed.get(f"{s}_object_mesh_dirs")
    )
    if multi_sides:
        if os.environ.get("FDR_OBJ_SWAP"):
            return _discover_swap(motion_cfg, packed, pool_dir, density, multi_sides)
        return _discover_multi(motion_cfg, packed, pool_dir, density, multi_sides)

    sides = tuple(
        s for s in ("right", "left") if packed.get(f"{s}_object_mesh_dir") is not None
    )

    entities: dict[str, EntityCfg] = {}
    entity_names: dict[str, str] = {}
    mesh_paths: dict[str, str] = {}
    mesh_scales: dict[str, float] = {}

    # Shared object (both slots -> same mesh dir): one physical entity, the
    # left side aliases it (DexMachina-style single-object bimanual).
    shared = (
        len(sides) == 2
        and packed.get("right_object_mesh_dir") == packed.get("left_object_mesh_dir")
    )
    for side in sides:
        mesh_rel = packed.get(f"{side}_object_mesh_dir")
        if mesh_rel is None:
            continue
        if shared and side == "left":
            entity_names["left"] = "object_right"
            mesh_paths["left"] = mesh_paths["right"]
            mesh_scales["left"] = mesh_scales["right"]
            continue
        obj_dir = str(pool_dir / mesh_rel)
        scale = float(packed.get(f"{side}_object_mesh_scale", 1.0))
        entity_name = f"object_{side}"
        body_name = f"obj_{side}"

        entities[entity_name] = build_object(
            SimpleNamespace(
                obj_dir=obj_dir,
                body_name=body_name,
                density=density,
                mesh_scale=scale,
            )
        )
        entity_names[side] = entity_name
        # Motion command's SDF baker expects an explicit mesh file path
        # (legacy train_single_object_teacher.py appends "visual.obj").
        mesh_paths[side] = str(Path(obj_dir) / "visual.obj")
        mesh_scales[side] = scale

    if entity_names:
        motion_cfg.object.entity_names = entity_names
        motion_cfg.object.mesh_paths = mesh_paths
        motion_cfg.object.mesh_scales = mesh_scales

    return entities


def _discover_multi(
    motion_cfg: Any,
    packed: dict,
    pool_dir: Path,
    density: float,
    sides: tuple[str, ...],
) -> dict[str, EntityCfg]:
    """One entity per slot, body names UNIQUE per slot (mesh VFS keys derive
    from them); entity_names/mesh_* become lists."""
    entities: dict[str, EntityCfg] = {}
    entity_names: dict[str, list[str]] = {}
    mesh_paths: dict[str, list[str]] = {}
    mesh_scales: dict[str, list[float]] = {}

    for side in sides:
        dirs = list(packed[f"{side}_object_mesh_dirs"])
        scales = list(
            packed.get(f"{side}_object_mesh_scales") or [1.0] * len(dirs)
        )
        names: list[str] = []
        paths: list[str] = []
        for slot, mesh_rel in enumerate(dirs):
            obj_dir = str(pool_dir / mesh_rel)
            entity_name = f"object_{side}_{slot}"
            body_name = f"obj_{side}_{slot}"
            entity_cfg = build_object(
                SimpleNamespace(
                    obj_dir=obj_dir,
                    body_name=body_name,
                    density=density,
                    mesh_scale=float(scales[slot]),
                )
            )
            # Park in init_state AND the spec: qpos0 (from the SPEC body pos)
            # sizes mjwarp's nconmax, so stacked defaults overflow pre-reset.
            park = object_park_position(slot)
            entity_cfg.init_state.pos = park
            entity_cfg.spec_fn = _parked_spec_fn(
                entity_cfg.spec_fn, body_name, park
            )
            entities[entity_name] = entity_cfg
            names.append(entity_name)
            paths.append(str(Path(obj_dir) / "visual.obj"))
        entity_names[side] = names
        mesh_paths[side] = paths
        mesh_scales[side] = [float(s) for s in scales]

    motion_cfg.object.entity_names = entity_names
    motion_cfg.object.mesh_paths = mesh_paths
    motion_cfg.object.mesh_scales = mesh_scales
    return entities


def _discover_swap(
    motion_cfg: Any,
    packed: dict,
    pool_dir: Path,
    density: float,
    sides: tuple[str, ...],
) -> dict[str, EntityCfg]:
    """Slot-free multi-object: ONE entity per side, objects swapped per world.
    entity_names stays str, mesh_* stay lists."""
    from ._object_swap import _hull_files, get_swap_object_cfg, maybe_decimated_dirs

    if len(sides) > 1:
        raise NotImplementedError("swap-mode motion tracking supports one side")
    side = sides[0]
    dirs = [str(pool_dir / rel) for rel in packed[f"{side}_object_mesh_dirs"]]
    scales = [
        float(s)
        for s in (
            packed.get(f"{side}_object_mesh_scales") or [1.0] * len(dirs)
        )
    ]
    # Decimated mirror dirs feed collision hulls only; mesh_paths below MUST
    # keep the ORIGINAL dirs (SDF cache is keyed on them).
    col_dirs = maybe_decimated_dirs(dirs)
    p_max = max(len(_hull_files(d)) for d in col_dirs)
    entity_name = f"object_{side}"
    body_name = f"obj_{side}"

    entities = {
        entity_name: get_swap_object_cfg(col_dirs, scales, body_name, density, p_max)
    }
    motion_cfg.object.entity_names = {side: entity_name}
    motion_cfg.object.mesh_paths = {
        side: [str(Path(d) / "visual.obj") for d in dirs]
    }
    motion_cfg.object.mesh_scales = {side: scales}
    from omegaconf import open_dict

    with open_dict(motion_cfg.object):
        motion_cfg.object.swap_spec = {
            "side": side,
            "body_name": body_name,
            "obj_dirs": col_dirs,
            "scales": scales,
            "density": float(density),
            "p_max": int(p_max),
        }
    return entities


def _parked_spec_fn(
    spec_fn: Any, body_name: str, pos: tuple[float, float, float]
) -> Any:
    """Wrap an object ``spec_fn``: SPEC pos = parking spot (defines qpos0), and
    conaffinity 3 -> 1 so objects never collide with EACH OTHER (docs)."""

    def wrapped(fn=spec_fn, name=body_name, p=pos):
        spec = fn()
        for body in spec.bodies:
            if body.name == name:
                body.pos = p
        for geom in spec.geoms:
            if geom.contype:
                geom.conaffinity = 1
        return spec

    return wrapped


def expand_multi_object_sensors(
    items: Any, entities: dict[str, EntityCfg], motion_cfg: Any = None
) -> Any:
    """Rewire object-bound contact sensors: "raw" (default) drops them for the
    raw-contact reader, "native" clones per slot."""
    if not items:
        return items
    mode = "raw"
    if motion_cfg is not None:
        mode = str(motion_cfg.object.get("contact_sensor_mode", "raw"))

    # Swap mode: the entity EXISTS, but native contact sensors still dominate a
    # 1024-env forward even for one entity (~4x sim.forward) — drop them too.
    swap_entity = None
    if motion_cfg is not None and motion_cfg.object.get("swap_spec") is not None:
        if mode != "raw":
            raise NotImplementedError("swap mode requires contact_sensor_mode=raw")
        swap_entity = motion_cfg.object.swap_spec["side"]
        swap_entity = f"object_{swap_entity}"

    out = []
    raw_specs: list[dict] = []
    changed = False
    for s in items:
        secondary = s.get("secondary") if hasattr(s, "get") else None
        ent = secondary.get("entity") if secondary is not None else None
        slot_names: list[str] = []
        if ent is not None and ent not in entities:
            slot_names = sorted(
                (n for n in entities if re.fullmatch(re.escape(ent) + r"_\d+", n)),
                key=lambda n: int(n.rsplit("_", 1)[1]),
            )
        if not slot_names and not (swap_entity is not None and ent == swap_entity):
            out.append(s)
            continue
        changed = True
        if mode == "raw":
            pattern = s.primary.pattern
            names = [pattern] if isinstance(pattern, str) else list(pattern)
            raw_specs.append(
                {
                    "name": str(s.name),
                    "primary_mode": str(s.primary.mode),
                    "primary_names": names,
                    "primary_entity": str(s.primary.entity),
                    "fields": [str(f) for f in s.get("fields", ("found", "force"))],
                    "reduce": str(s.get("reduce", "maxforce")),
                }
            )
            continue  # drop the native sensor cfg
        for slot, entity_name in enumerate(slot_names):
            c = OmegaConf.create(
                OmegaConf.to_container(s, resolve=True)
                if OmegaConf.is_config(s)
                else copy.deepcopy(dict(s))
            )
            c.name = f"{s.name}__s{slot}"
            c.secondary.entity = entity_name
            # Body names are per-slot too (obj_right → obj_right_<slot>); the
            # yaml pattern targets the single-object body name.
            if isinstance(c.secondary.get("pattern"), str):
                c.secondary.pattern = f"{c.secondary.pattern}_{slot}"
            out.append(c)

    if raw_specs and motion_cfg is not None:
        from omegaconf import open_dict

        with open_dict(motion_cfg.object):
            motion_cfg.object.raw_sensor_specs = raw_specs
    return out if changed else items
