from __future__ import annotations

from typing import Any

from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.spec_config import LightCfg, MaterialCfg, TextureCfg


def build_terrain(t: Any) -> TerrainEntityCfg:
    """Build ``TerrainEntityCfg`` from a parsed ``terrain`` yaml block."""
    return TerrainEntityCfg(
        terrain_type=t.terrain_type,
        terrain_generator=t.get("terrain_generator"),
        env_spacing=float(t.env_spacing),
        max_init_terrain_level=t.get("max_init_terrain_level"),
        lights=_build_lights(t.get("lights")),
        textures=_build_textures(t.get("textures")),
        materials=_build_materials(t.get("materials")),
    )


def _build_lights(items: Any) -> tuple[LightCfg, ...]:
    return tuple(LightCfg(**dict(item)) for item in (items or ()))


def _build_textures(items: Any) -> tuple[TextureCfg, ...]:
    out: list[TextureCfg] = []
    for item in items or ():
        d = dict(item)
        if "rgb1" in d:
            d["rgb1"] = tuple(d["rgb1"])
        if "rgb2" in d:
            d["rgb2"] = tuple(d["rgb2"])
        if "markrgb" in d:
            d["markrgb"] = tuple(d["markrgb"])
        out.append(TextureCfg(**d))
    return tuple(out)


def _build_materials(items: Any) -> tuple[MaterialCfg, ...]:
    out: list[MaterialCfg] = []
    for item in items or ():
        d = dict(item)
        if "rgba" in d:
            d["rgba"] = tuple(d["rgba"])
        if "texrepeat" in d:
            d["texrepeat"] = tuple(d["texrepeat"])
        if d.get("geom_names_expr") is not None:
            d["geom_names_expr"] = tuple(d["geom_names_expr"])
        out.append(MaterialCfg(**d))
    return tuple(out)
