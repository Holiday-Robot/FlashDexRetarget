from __future__ import annotations

from typing import Any

from hydra.utils import get_class
from mjlab.managers.action_manager import ActionTermCfg
from omegaconf import DictConfig, OmegaConf


def build_actions(cfg: Any) -> dict[str, ActionTermCfg]:
    """Build the action term dict from a parsed ``actions`` cfg block."""
    data = (
        OmegaConf.to_container(cfg, resolve=True)
        if isinstance(cfg, DictConfig)
        else dict(cfg)
    )
    assert isinstance(data, dict)
    target = data.pop("_target_")
    name = data.pop("name")
    cls = get_class(target)
    return {name: cls(**data)}
