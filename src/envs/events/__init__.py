from __future__ import annotations

from typing import Any

from mjlab.managers.event_manager import EventTermCfg

from .._common import materialize
from . import base as _t


def build_events(cfg: Any) -> dict[str, EventTermCfg]:
    """Build event terms from a parsed ``events`` block."""
    out: dict[str, EventTermCfg] = {}
    for name, term_cfg in cfg.items():
        fn = getattr(_t, name)
        raw_params = dict(term_cfg.get("params", {}) or {})
        params: dict[str, Any] = {k: materialize(v) for k, v in raw_params.items()}

        kwargs: dict[str, Any] = dict(func=fn, params=params, mode=term_cfg.mode)
        if term_cfg.get("interval_range_s") is not None:
            kwargs["interval_range_s"] = tuple(term_cfg.interval_range_s)
        out[name] = EventTermCfg(**kwargs)
    return out
