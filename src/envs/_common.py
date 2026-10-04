from __future__ import annotations

import functools
import inspect
from typing import Any

from mjlab.managers.scene_entity_config import SceneEntityCfg
from omegaconf import DictConfig, OmegaConf


def materialize(value: Any) -> Any:
    """Convert yaml-loaded values into runtime objects: a dict/DictConfig with a
    ``"name"`` key becomes a :class:`SceneEntityCfg`; anything else passes through."""
    if isinstance(value, DictConfig):
        value = OmegaConf.to_container(value, resolve=True)
    if isinstance(value, dict) and "name" in value:
        return SceneEntityCfg(**value)
    return value


def auto_inject(
    fn: Any,
    params: dict[str, Any],
    command_name: str | None = None,
    entity_name: str | None = None,
) -> None:
    """Fill missing ``command_name`` / ``asset_cfg`` params when ``fn``'s signature
    accepts them; an explicit yaml ``asset_cfg`` always wins over the injected one."""
    sig = inspect.signature(fn)
    if command_name is not None and "command_name" in sig.parameters:
        params.setdefault("command_name", command_name)
    if entity_name is not None and "asset_cfg" in sig.parameters:
        if "asset_cfg" not in params:
            params["asset_cfg"] = SceneEntityCfg(entity_name, joint_names=(".*",))


def shape_checked(fn: Any, term_name: str, manager_name: str) -> Any:
    """Wrap a manager term so a wrong-shaped return fails loudly at the term: anything
    but ``(num_envs,)`` broadcasts silently and corrupts every other term's signal."""

    @functools.wraps(fn)
    def wrapped(env: Any, **kwargs: Any) -> Any:
        value = fn(env, **kwargs)
        if value.shape != (env.num_envs,):
            raise ValueError(
                f"{manager_name} term '{term_name}' returned shape "
                f"{tuple(value.shape)}, expected ({env.num_envs},)."
            )
        return value

    return wrapped



def action_term(env: Any) -> Any:
    """The env's action term, whatever its configured name (every preset registers exactly one)."""
    terms = env.action_manager._terms
    if len(terms) != 1:
        raise ValueError(f"expected one action term, got {list(terms)}")
    return next(iter(terms.values()))


# Keep old name for any callers that haven't been updated yet.
def inject_command_name(fn: Any, params: dict[str, Any], command_name: str) -> None:
    auto_inject(fn, params, command_name=command_name)
