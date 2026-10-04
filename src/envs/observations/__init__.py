from __future__ import annotations

from typing import Any

from hydra.utils import get_class
from mjlab.managers.observation_manager import (
    ObservationGroupCfg,
    ObservationTermCfg,
)
from mjlab.utils.noise import UniformNoiseCfg
from omegaconf import DictConfig, OmegaConf

from .._common import auto_inject, materialize


def _build_group(
    cls: type,
    term_names: list[str],
    obs_params: dict[str, Any],
    obs_scales: dict[str, float],
    noise_scales: dict[str, float],
    obs_clip: float | dict[str, float] | None,
    command_name: str,
    entity_name: str,
    enable_corruption: bool,
    nan_policy: str,
) -> ObservationGroupCfg:
    terms: dict[str, ObservationTermCfg] = {}
    for name in term_names:
        fn = getattr(cls, name)
        params: dict[str, Any] = {}

        if name in obs_params:
            for k, v in dict(obs_params[name]).items():
                params[k] = materialize(v)

        auto_inject(fn, params, command_name=command_name, entity_name=entity_name)

        obs_scale_val = float(obs_scales.get(name, 1.0))
        obs_scale = obs_scale_val if obs_scale_val != 1.0 else None

        obs_noise_amp = float(noise_scales.get(name, 0.0))
        obs_noise = (
            UniformNoiseCfg(n_min=-obs_noise_amp, n_max=obs_noise_amp)
            if obs_noise_amp > 0.0
            else None
        )

        # obs_clip is a POST-scale amplitude (DexMachina obs_clip); mjlab clips BEFORE scaling,
        # so divide by the term's scale. A float applies to every term, a mapping per term.
        clip_amp = obs_clip.get(name, 0.0) if isinstance(obs_clip, dict) else obs_clip
        clip_amp = float(clip_amp or 0.0)
        clip_pre = clip_amp / (abs(obs_scale_val) or 1.0)
        obs_clip_range = (-clip_pre, clip_pre) if clip_amp > 0.0 else None

        terms[name] = ObservationTermCfg(
            func=fn,
            params=params,
            scale=obs_scale,
            noise=obs_noise,
            clip=obs_clip_range,
        )

    return ObservationGroupCfg(
        terms=terms,
        concatenate_terms=True,
        enable_corruption=enable_corruption,
        nan_policy=nan_policy,
    )


def build_observations(
    cfg: Any,
    command_name: str = "motion",
    entity_name: str = "hand",
) -> dict[str, ObservationGroupCfg]:
    """Build actor + critic observation groups from a parsed ``obs`` block."""
    cls = get_class(cfg._target_)
    command_name = cfg.get("command_name", command_name)
    entity_name = cfg.get("entity_name", entity_name)
    obs_dict = cfg.obs_dict
    obs_params = cfg.get("obs_params", {}) or {}
    obs_scales = cfg.get("obs_scales", {}) or {}
    noise_scales = cfg.get("noise_scales", {}) or {}
    obs_clip = cfg.get("obs_clip", None)
    if isinstance(obs_clip, DictConfig):
        obs_clip = dict(OmegaConf.to_container(obs_clip, resolve=True))
    nan_policy = cfg.get("nan_policy", "disabled")

    common = dict(
        cls=cls,
        obs_params=obs_params,
        obs_scales=obs_scales,
        noise_scales=noise_scales,
        obs_clip=obs_clip,
        command_name=command_name,
        entity_name=entity_name,
        nan_policy=nan_policy,
    )

    groups = {
        "actor": _build_group(
            term_names=list(obs_dict.actor_obs),
            enable_corruption=bool(cfg.get("actor_enable_corruption", False)),
            **common,
        ),
        "critic": _build_group(
            term_names=list(obs_dict.critic_obs),
            enable_corruption=bool(cfg.get("critic_enable_corruption", False)),
            **common,
        ),
    }

    # Encoded obs groups: each is a single un-concatenated term kept as its OWN group so a
    # custom model can route it to an encoder (runner obs_groups); no corruption (reference info).
    encoded_groups = cfg.get("encoded_groups", {}) or {}
    for group_name, term_name in dict(encoded_groups).items():
        if group_name in groups:
            raise ValueError(
                f"encoded group {group_name!r} collides with an existing obs group."
            )
        groups[group_name] = _build_group(
            term_names=[str(term_name)],
            enable_corruption=False,
            **common,
        )

    # Drop groups with no active terms (e.g. shared critic_obs: []) — they would register a
    # zero-dim group; the guard lives here so vanilla mjlab stays untouched.
    for group_name in [g for g, cfg_ in groups.items() if not cfg_.terms]:
        print(f"group: {group_name} has no active terms, skipping...")
        del groups[group_name]

    return groups
