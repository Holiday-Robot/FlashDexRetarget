"""Generic ``_target_``-driven command builder: yaml ``commands`` gives name + CommandTermCfg
path + kwargs; yaml-only fields are silently dropped (still readable off the raw cfg)."""

from __future__ import annotations

import dataclasses
import types
import typing
from typing import Any

from hydra.utils import get_class
from mjlab.managers.command_manager import CommandTermCfg
from omegaconf import DictConfig, OmegaConf


def build_commands(cfg: Any) -> dict[str, CommandTermCfg]:
    data = OmegaConf.to_container(cfg, resolve=True) if isinstance(cfg, DictConfig) else dict(cfg)
    name = data.pop("name")
    target = data.pop("_target_")
    cls = get_class(target)
    return {name: _instantiate_dataclass(cls, data)}


def _instantiate_dataclass(cls: type, data: Any) -> Any:
    if not dataclasses.is_dataclass(cls) or not isinstance(data, dict):
        return data

    hints = typing.get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for f in dataclasses.fields(cls):
        if f.name not in data:
            continue  # let dataclass default apply
        value = data[f.name]
        type_hint = _unwrap_optional(hints.get(f.name, f.type))
        if dataclasses.is_dataclass(type_hint) and isinstance(value, dict):
            kwargs[f.name] = _instantiate_dataclass(type_hint, value)
        elif typing.get_origin(type_hint) is tuple and isinstance(value, list):
            kwargs[f.name] = tuple(value)
        else:
            kwargs[f.name] = value
    return cls(**kwargs)


def _unwrap_optional(t: Any) -> Any:
    """``Optional[X]`` / ``X | None`` → ``X``; otherwise pass through."""
    origin = typing.get_origin(t)
    if origin is typing.Union or origin is types.UnionType:
        non_none = [a for a in typing.get_args(t) if a is not type(None)]
        if len(non_none) == 1:
            return non_none[0]
    return t
