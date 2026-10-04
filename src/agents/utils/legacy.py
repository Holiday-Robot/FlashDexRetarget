"""flash_rl FlashSACAgent checkpoint directories -> MultiHeadFlashSAC checkpoint dicts."""

from __future__ import annotations

import os
from typing import Any

import torch
import torch.nn as nn

from ..algorithms.flashsac import MultiHeadFlashSAC
from .reward_norm import MultiHeadRewardNormalizer

LEGACY_FILES = ("actor.pt", "critic.pt", "target_critic.pt", "temperature.pt", "agent_state.pt")


def is_legacy_checkpoint(path: str) -> bool:
    return all(os.path.exists(os.path.join(path, name)) for name in LEGACY_FILES)


def _strip_compile_prefix(state: dict[str, Any]) -> dict[str, Any]:
    """Drop torch.compile's ``_orig_mod.`` wrapper prefix from flash_rl's saved module keys."""
    return {key.removeprefix("_orig_mod."): value for key, value in state.items()}


def actor_state_from_legacy(net_state: dict[str, Any], actor: nn.Module) -> dict[str, Any]:
    """flash_rl actor weights (FlashSACActor / FlashSACSideActors keys match ours) over our actor's
    state, which supplies the affine action buffers flash_rl did not persist."""
    net_state = _strip_compile_prefix(net_state)
    state = actor.state_dict()
    unknown = sorted(set(net_state) - set(state))
    if unknown:
        raise KeyError(f"flash_rl actor keys without a counterpart (side bridge / shared blocks?): {unknown}")
    return {**state, **net_state}


def critic_state_from_legacy(critic_state: dict[str, Any], target_state: dict[str, Any]) -> dict[str, Any]:
    """flash_rl online / target FlashSACDoubleCritic weights -> MultiHeadCritic state (critic., critic_target.)."""
    return {
        **{f"critic.{key}": value for key, value in _strip_compile_prefix(critic_state).items()},
        **{f"critic_target.{key}": value for key, value in _strip_compile_prefix(target_state).items()},
    }


def _param_names(module: nn.Module) -> list[str]:
    return [name for name, param in module.named_parameters() if param.requires_grad]


def _remap_optimizer(opt_state: dict[str, Any], legacy_state: dict[str, Any], names: list[str]) -> dict[str, Any]:
    """Reindex a single-group optimizer state from flash_rl's parameter order to ours, by name."""
    name_set = set(names)
    legacy_names = [key for key in _strip_compile_prefix(legacy_state) if key in name_set]
    if sorted(legacy_names) != sorted(names):
        raise KeyError(f"optimizer parameters differ: {sorted(name_set.symmetric_difference(legacy_names))}")
    legacy_index = {name: i for i, name in enumerate(legacy_names)}
    state = {
        i: opt_state["state"][legacy_index[name]]
        for i, name in enumerate(names)
        if legacy_index[name] in opt_state["state"]
    }
    (group,) = opt_state["param_groups"]
    return {"state": state, "param_groups": [{**group, "params": list(range(len(names)))}]}


def load_legacy_checkpoint(path: str, alg: MultiHeadFlashSAC) -> dict[str, Any]:
    """Read flash_rl's per-network files in ``path`` into the dict MultiHeadFlashSAC.load() takes."""

    def read(name: str) -> dict[str, Any]:
        return torch.load(os.path.join(path, name), map_location=alg.device)

    actor, critic, target, temperature = (read(name) for name in LEGACY_FILES[:4])
    agent_state = read("agent_state.pt")
    loaded: dict[str, Any] = {
        "actor_state_dict": actor_state_from_legacy(actor["network_state_dict"], alg.actor),
        "critic_state_dict": critic_state_from_legacy(critic["network_state_dict"], target["network_state_dict"]),
        "temperature_state_dict": temperature["network_state_dict"],
        "actor_optimizer_state_dict": _remap_optimizer(
            actor["optimizer_state_dict"], actor["network_state_dict"], _param_names(alg.actor)
        ),
        "critic_optimizer_state_dict": _remap_optimizer(
            critic["optimizer_state_dict"], critic["network_state_dict"], _param_names(alg.critic.critic)
        ),
        "temperature_optimizer_state_dict": temperature["optimizer_state_dict"],
        "actor_scheduler_state_dict": actor["scheduler_state_dict"],
        "critic_scheduler_state_dict": critic["scheduler_state_dict"],
        "temperature_scheduler_state_dict": temperature["scheduler_state_dict"],
        "grad_scaler_state_dict": agent_state["grad_scaler_state_dict"],
        "update_step": agent_state["update_step"],
        "reward_normalizer_state_dict": None,
    }
    normalizer_path = os.path.join(path, "reward_normalizer.pt")
    if os.path.exists(normalizer_path):
        state = read("reward_normalizer.pt")
        if not isinstance(alg.reward_normalizer, MultiHeadRewardNormalizer):
            # flash_rl kept (B, 1) / (1,) columns for a single head; upstream keeps (B,) / (1,)
            state = {key: value.reshape(-1) if key != "G_rms_count" else value for key, value in state.items()}
        loaded["reward_normalizer_state_dict"] = state
    return loaded
