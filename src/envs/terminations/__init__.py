"""Termination factory: ``terminations._target_`` picks the class, every other key
maps to a ``@staticmethod`` term; ``command_name`` / ``asset_cfg`` auto-injected."""

from __future__ import annotations

from typing import Any

from hydra.utils import get_class
from mjlab.managers.termination_manager import TerminationTermCfg

from .._common import auto_inject, materialize, shape_checked


def build_terminations(
    cfg: Any,
    command_name: str = "motion",
    entity_name: str = "hand",
) -> dict[str, TerminationTermCfg]:
    """Build termination terms from a parsed ``terminations`` block."""
    cls = get_class(cfg._target_)
    out: dict[str, TerminationTermCfg] = {}
    for name, term_cfg in cfg.items():
        if name == "_target_":
            continue
        if not term_cfg.get("enabled", True):
            continue
        fn = getattr(cls, name)
        raw_params = dict(term_cfg.get("params", {}) or {})
        params: dict[str, Any] = {k: materialize(v) for k, v in raw_params.items()}
        auto_inject(fn, params, command_name=command_name, entity_name=entity_name)
        out[name] = TerminationTermCfg(
            func=shape_checked(fn, name, "TerminationManager"),
            params=params,
            time_out=bool(term_cfg.get("time_out", False)),
        )
    return out
