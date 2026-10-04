from __future__ import annotations

import inspect
from typing import Any

from hydra.utils import get_class
from mjlab.managers.reward_manager import RewardTermCfg

from .._common import auto_inject, materialize, shape_checked


# rewards.split_sides: every term whose function takes `side=None` becomes one term per hand
# (tracking_X -> tracking_{r,l}_X, others -> X_{r,l}) at half the weight; termination_penalty
# keeps the full weight since it is charged to the side at fault.
_FULL_WEIGHT_TERMS = {"termination_penalty"}
_SIDES = (("r", "right"), ("l", "left"))


def side_twin_name(name: str, tok: str) -> str:
    """Per-hand term name under split_sides: tracking_X -> tracking_{tok}_X, else X_{tok}."""
    if name.startswith("tracking_"):
        return f"tracking_{tok}_{name[len('tracking_'):]}"
    return f"{name}_{tok}"


def split_sides(cls: Any, name: str, weight: float) -> list[tuple[str, str, float, dict[str, Any]]] | None:
    """Per-hand (term name, method name, weight, extra params) for one yaml term, or None if
    its function has no optional `side` (e.g. force_penalty) and the term stays whole."""
    side = inspect.signature(getattr(cls, name)).parameters.get("side")
    if side is None or side.default is not None:
        return None
    w = weight if name in _FULL_WEIGHT_TERMS else weight / len(_SIDES)
    return [(side_twin_name(name, tok), name, w, {"side": s}) for tok, s in _SIDES]


def build_rewards(
    cfg: Any,
    command_name: str = "motion",
    entity_name: str = "hand",
) -> dict[str, RewardTermCfg]:
    """Build reward terms from a parsed ``rewards`` config block."""
    cls = get_class(cfg._target_)
    command_name = cfg.get("command_name", command_name)
    entity_name = cfg.get("entity_name", entity_name)
    weights = cfg.reward_weights
    scales = cfg.get("reward_scales", {}) or {}
    reward_params = cfg.get("reward_params", {}) or {}

    split = bool(cfg.get("split_sides", False))

    out: dict[str, RewardTermCfg] = {}
    for name, weight in weights.items():
        params: dict[str, Any] = {}

        # A term may draw extra named params from reward_params AND its exp
        # sharpness from reward_scales (e.g. contact: scale + max_distance).
        if name in reward_params:
            for k, v in dict(reward_params[name]).items():
                params[k] = materialize(v)
        if name in scales:
            params["scale"] = scales[name]

        twins = split_sides(cls, name, weight) if split else None
        for tname, method, tweight, extra in twins or [(name, name, weight, {})]:
            fn = getattr(cls, method)
            tparams = dict(params, **extra)
            if twins:  # a twin may take fewer params than the whole-body term (action_penalty_r)
                sig = inspect.signature(fn).parameters
                tparams = {k: v for k, v in tparams.items() if k in sig}
            auto_inject(fn, tparams, command_name=command_name, entity_name=entity_name)
            out[tname] = RewardTermCfg(
                func=shape_checked(fn, tname, "RewardManager"),
                weight=float(tweight),
                params=tparams,
            )
    return out
