from __future__ import annotations

from typing import Any

from mjlab.managers.curriculum_manager import CurriculumTermCfg

from .._common import auto_inject, materialize
from . import base as _t


def build_curriculum(
    cfg: Any,
    command_name: str = "motion",
    entity_name: str = "hand",
) -> dict[str, CurriculumTermCfg]:
    """Build curriculum terms from a parsed ``curriculum`` block."""
    out: dict[str, CurriculumTermCfg] = {}
    for name, term_cfg in cfg.items():
        fn = getattr(_t, name)
        raw_params = dict(term_cfg.get("params", {}) or {})
        params: dict[str, Any] = {k: materialize(v) for k, v in raw_params.items()}
        auto_inject(fn, params, command_name=command_name, entity_name=entity_name)
        out[name] = CurriculumTermCfg(func=fn, params=params)
    return out
