"""FlashSAC as an agents.Agent (runner.agent.class_name: agents.flashsac.FlashSACAgent): its config,
the builder of MultiHeadFlashSAC (rsl_rl_flashsac) and the env-facing adapter."""

from __future__ import annotations

import json
import math
import os
from collections.abc import MutableMapping
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict

from .algorithms.flashsac import MultiHeadFlashSAC
from .base import Agent
from .modules.actor_critic import ActorTower, MultiHeadCritic, SideFactoredActor
from .storage.replay_buffer import ReplayBuffer, ReplayObsLayout
from .utils.legacy import is_legacy_checkpoint, load_legacy_checkpoint

CHECKPOINT_FILE = "flashsac.pt"
REPLAY_BUFFER_FILE = "replay_buffer.pt"


# ─── config ────────────────────────────────────────────────────────────────────
@dataclass
class FlashSACConfig:
    """runner.agent of config/runner/flashsac.yaml (flash_rl's field names)."""

    seed: int
    normalize_reward: bool
    normalized_G_max: float

    asymmetric_observation: bool
    device_type: str

    buffer_max_length: int
    buffer_min_length: int
    buffer_device_type: str
    sample_batch_size: int

    learning_rate_init: float
    learning_rate_peak: float
    learning_rate_end: float
    learning_rate_warmup_rate: float
    learning_rate_warmup_step: int
    learning_rate_decay_rate: float
    learning_rate_decay_step: int

    actor_num_blocks: int
    actor_hidden_dim: int
    actor_bc_alpha: float
    actor_noise_zeta_mu: float
    actor_noise_zeta_max: int
    actor_update_period: int

    critic_num_blocks: int
    critic_hidden_dim: int
    critic_num_bins: int
    critic_min_v: float
    critic_max_v: float
    critic_target_update_tau: float

    temp_initial_value: float
    temp_target_sigma: float
    temp_target_entropy: float | None

    gamma: float
    n_step: int

    use_compile: bool
    compile_mode: str
    use_amp: bool

    load_optimizer: bool
    load_reward_normalizer: bool

    buffer_obs_dtype: str | None = None
    buffer_optimize_memory_usage: bool = True
    # Store only non-reconstructible obs columns plus the int32 (motion id, step) aux group.
    buffer_compact_obs: bool = False
    buffer_action_dtype: str | None = None

    # {enable, aggregation, n_future, n_sides, latent_dim, hidden_dim, num_layers}
    traj_encoder: dict[str, Any] | None = None
    # {enable, overrides, side_actors, side_shared_blocks, side_bridge}
    reward_heads: dict[str, Any] | None = None

    def algorithm_kwargs(self) -> dict[str, Any]:
        """Keyword arguments of rsl_rl_flashsac's FlashSAC (one gradient update per update() call)."""
        return {
            "replay_buffer_size": self.buffer_max_length,
            "num_learning_epochs": 1,
            "num_mini_batches": 1,
            "mini_batch_size": self.sample_batch_size,
            "learning_rate_init": self.learning_rate_init,
            "learning_rate_peak": self.learning_rate_peak,
            "learning_rate_end": self.learning_rate_end,
            "learning_rate_warmup_steps": self.learning_rate_warmup_step,
            "learning_rate_decay_steps": self.learning_rate_decay_step,
            "actor_bc_alpha": self.actor_bc_alpha,
            "actor_noise_zeta_mu": self.actor_noise_zeta_mu,
            "actor_noise_zeta_max": self.actor_noise_zeta_max,
            "actor_update_period": self.actor_update_period,
            "critic_target_update_tau": self.critic_target_update_tau,
            "temp_initial_value": self.temp_initial_value,
            "temp_target_sigma": self.temp_target_sigma,
            "gamma": self.gamma,
            "n_steps": self.n_step,
            "normalize_reward": self.normalize_reward,
            "normalized_G_max": self.normalized_G_max,
            "use_compile": self.use_compile,
            "compile_mode": self.compile_mode,
            "use_amp": self.use_amp,
        }


# ─── builder ───────────────────────────────────────────────────────────────────
def _dtype(name: str | None) -> torch.dtype | None:
    return getattr(torch, name) if name else None


def build_flashsac(
    cfg: FlashSACConfig,
    layout: ReplayObsLayout,
    num_actions: int,
    *,
    action_bias: torch.Tensor,
    action_scale: torch.Tensor,
    target_entropy: list[float],
    device: str,
    buffer_device: str,
    traj_cfg: dict[str, Any] | None = None,
    num_heads: int = 1,
    side_action_ids: list[list[int]] | None = None,
) -> MultiHeadFlashSAC:
    """Actor (one tower per side if side_action_ids), 2 * num_heads critic, compact-aware buffer."""
    if not cfg.buffer_optimize_memory_usage:
        raise ValueError("only the memory-efficient replay buffer is ported (buffer_optimize_memory_usage=true).")
    template = layout.network_template()
    groups = layout.network_groups

    actor: ActorTower | SideFactoredActor
    tower_kwargs = {"num_blocks": cfg.actor_num_blocks, "hidden_dim": cfg.actor_hidden_dim, "traj_cfg": traj_cfg}
    if side_action_ids:
        actor = SideFactoredActor(template, groups, "actor", side_action_ids, **tower_kwargs)
    else:
        actor = ActorTower(template, groups, "actor", num_actions, **tower_kwargs)
    actor.set_action_scaling(action_bias.cpu(), action_scale.cpu())

    critic = MultiHeadCritic(
        template,
        groups,
        "critic",
        num_actions=num_actions,
        num_blocks=cfg.critic_num_blocks,
        hidden_dim=cfg.critic_hidden_dim,
        num_bins=cfg.critic_num_bins,
        min_v=cfg.critic_min_v,
        max_v=cfg.critic_max_v,
        num_heads=num_heads,
        traj_cfg=traj_cfg,
    )

    replay_buffer = ReplayBuffer(
        obs=layout.store_template(),
        num_actions=num_actions,
        n_step=cfg.n_step,
        gamma=cfg.gamma,
        max_length=cfg.buffer_max_length,
        min_length=cfg.buffer_min_length,
        sample_batch_size=cfg.sample_batch_size,
        device=buffer_device,
        obs_storage_dtype=_dtype(cfg.buffer_obs_dtype),
        store_groups=layout.store_groups,
        aux_groups=layout.aux_groups,
        reward_dim=num_heads if num_heads > 1 else 0,
        action_storage_dtype=_dtype(cfg.buffer_action_dtype),
    )

    return MultiHeadFlashSAC(
        actor,
        critic,
        replay_buffer,
        obs_layout=layout,
        target_entropy=target_entropy,
        num_heads=num_heads,
        device=device,
        **cfg.algorithm_kwargs(),
    )


# ─── agent ─────────────────────────────────────────────────────────────────────
def resolve_device_type(device_type: str) -> str:
    """cuda[:i] as given (bare cuda -> cuda:$LOCAL_RANK); anything else -> cpu."""
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        raise NotImplementedError("multi-GPU (torchrun) is not wired for the rsl_rl_flashsac agent")
    if not device_type.startswith("cuda"):
        return "cpu"
    return device_type if ":" in device_type else f"cuda:{int(os.environ.get('LOCAL_RANK', '0'))}"


class FlashSACAgent(Agent):
    """Env-facing FlashSAC agent: actions leave in the env's bounds and are stored in [-1, 1];
    observations arrive flat ([actor | critic] or shared + encoded windows) as numpy or tensors."""

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        env_info: dict[str, Any],
        cfg: FlashSACConfig,
    ) -> None:
        if not isinstance(action_space, gym.spaces.Box):
            raise TypeError(f"FlashSAC expects a Box action space, got {type(action_space).__name__}.")
        self._cfg = cfg
        self._device = torch.device(resolve_device_type(cfg.device_type))
        full_dim = int(observation_space.shape[-1])  # type: ignore[index]
        action_dim = int(action_space.shape[-1])
        actor_dim = int(env_info["actor_observation_size"][-1]) if cfg.asymmetric_observation else None

        self._encoded_layout = [[str(n), int(d)] for n, d in (env_info.get("encoded_obs_dims") or [])]
        self._obs_terms = [list(t) for t in (env_info.get("obs_terms") or [])]
        head_names = self._reward_head_names(cfg, env_info)
        side_action_ids = self._side_action_ids(cfg, env_info, head_names)

        low = torch.as_tensor(np.asarray(action_space.low), dtype=torch.float32).reshape(-1, action_dim)[0]
        high = torch.as_tensor(np.asarray(action_space.high), dtype=torch.float32).reshape(-1, action_dim)[0]
        if not torch.isfinite(low).all() or not torch.isfinite(high).all():
            raise ValueError("FlashSAC requires finite action bounds for tanh-squashed policies.")
        if (high < low).any():
            raise ValueError("FlashSAC received an action space with high < low.")
        self._action_bias = (0.5 * (high + low)).to(self._device)
        self._action_scale = (0.5 * (high - low)).to(self._device)

        compactor = None
        if cfg.buffer_compact_obs:
            compactor = env_info.get("obs_compactor")
            if compactor is None:
                raise ValueError("buffer_compact_obs=true but env_info has no obs_compactor (runner must build it).")
        self._layout = ReplayObsLayout(full_dim, actor_dim, compactor)

        self._alg = build_flashsac(
            cfg,
            self._layout,
            action_dim,
            action_bias=self._action_bias,
            action_scale=self._action_scale,
            target_entropy=self._target_entropy(cfg, action_dim, side_action_ids),
            device=str(self._device),
            buffer_device=resolve_device_type(cfg.buffer_device_type),
            traj_cfg=self._traj_cfg(cfg, env_info),
            num_heads=max(1, len(head_names)),
            side_action_ids=side_action_ids or None,
        )

    @classmethod
    def build(
        cls, cfg: DictConfig, env: Any, env_info: dict[str, Any], command_name: str
    ) -> FlashSACAgent:
        if env.share_obs and bool(cfg.asymmetric_observation):
            # One stored obs copy has no [actor | critic] split, so the critic reads the actor group.
            print("[flashsac] share_actor_critic_obs=true -> forcing agent.asymmetric_observation=false")
            cfg.asymmetric_observation = False
        fields = OmegaConf.to_container(cfg, resolve=True)
        fields.pop("class_name", None)
        agent_cfg = FlashSACConfig(**fields)
        if agent_cfg.buffer_compact_obs:
            from .storage.obs_compact import build_obs_compactor  # needs mjlab

            compactor = build_obs_compactor(env, command_name, resolve_device_type(agent_cfg.device_type))
            env.set_aux_provider(compactor.aux_from_command)
            env_info = {**env_info, "obs_compactor": compactor}
        return cls(env.single_observation_space, env.single_action_space, env_info, agent_cfg)

    # ─── env_info / config resolution ──────────────────────────────────────────

    @staticmethod
    def _traj_cfg(cfg: FlashSACConfig, env_info: dict[str, Any]) -> dict[str, Any] | None:
        """Traj-encoder spec; window dims come from the env's encoded obs groups at the flat obs tail."""
        te = dict(cfg.traj_encoder or {})
        if not te.pop("enable", False):
            return None
        encoded_dims = list(env_info.get("encoded_obs_dims") or [])
        if not encoded_dims:
            raise ValueError("traj_encoder.enable=true but the env has no encoded obs groups.")
        if cfg.asymmetric_observation:
            raise ValueError("traj_encoder requires asymmetric_observation=false (shared obs).")
        return {**te, "window_dims": [int(d) for _, d in encoded_dims]}

    @staticmethod
    def _reward_head_names(cfg: FlashSACConfig, env_info: dict[str, Any]) -> list[str]:
        heads = dict(cfg.reward_heads or {})
        if not heads.get("enable", False):
            return []
        names = list(env_info.get("reward_heads") or [])
        if not names:
            raise ValueError("reward_heads.enable=true but the env produced no reward heads.")
        if int(heads.get("side_shared_blocks", 0) or 0) or (heads.get("side_bridge") or {}).get("enable", False):
            raise NotImplementedError("reward_heads.side_shared_blocks / side_bridge are not ported.")
        return names

    @staticmethod
    def _side_action_ids(cfg: FlashSACConfig, env_info: dict[str, Any], head_names: list[str]) -> list[list[int]]:
        """Action dims per reward head (env joint-name sides) when side actors are on, else []."""
        if not (cfg.reward_heads or {}).get("side_actors", False):
            return []
        if len(head_names) < 2:
            raise ValueError("reward_heads.side_actors=true needs reward_heads.enable=true with >=2 heads.")
        action_sides = dict(env_info.get("action_sides") or {})
        missing = [n for n in head_names if not action_sides.get(n)]
        if missing:
            raise ValueError(f"reward_heads.side_actors=true but the env has no action dims for {missing}.")
        return [[int(i) for i in action_sides[n]] for n in head_names]

    @staticmethod
    def _target_entropy(cfg: FlashSACConfig, action_dim: int, side_action_ids: list[list[int]]) -> list[float]:
        """Per-actor target entropy of a Gaussian with std temp_target_sigma per (normalized) action dim;
        an explicit total is split across the side actors by their dims."""
        dims = [len(ids) for ids in side_action_ids] or [action_dim]
        if cfg.temp_target_entropy is not None:
            return [float(cfg.temp_target_entropy) * n / action_dim for n in dims]
        return [0.5 * n * math.log(2 * math.pi * math.e * cfg.temp_target_sigma**2) for n in dims]

    # ─── Agent API ─────────────────────────────────────────────────────────────

    def _tensor(self, value: Any, dtype: torch.dtype | None = torch.float32) -> torch.Tensor:
        return torch.as_tensor(value, dtype=dtype).to(self._device, non_blocking=True)

    def _stored_obs(self, transition: MutableMapping[str, Any], key: str) -> TensorDict:
        aux = None
        if self._layout.compactor is not None:
            if f"{key}_aux" not in transition:
                raise KeyError(f"buffer_compact_obs=true needs transition[{key + '_aux'!r}] (runner must supply it).")
            aux = self._tensor(transition[f"{key}_aux"], dtype=None)
        return self._layout.to_store(self._tensor(transition[key]), aux)

    def act(self, obs: Any, deterministic: bool = False) -> np.ndarray:
        """Exploration actions (temporally repeated noise), or deterministic tanh(mean)."""
        obs = self._layout.actor_input(self._tensor(obs))
        actions = self._alg.act_inference(obs) if deterministic else self._alg.act(obs)
        return (self._action_bias + self._action_scale * actions).cpu().numpy()

    def observe(self, transition: MutableMapping[str, Any]) -> None:
        self._alg.add_transition(
            obs=self._stored_obs(transition, "observation"),
            actions=(self._tensor(transition["action"]) - self._action_bias) / self._action_scale,
            rewards=self._tensor(transition["reward"]),
            terminated=self._tensor(transition["terminated"], dtype=None),
            truncated=self._tensor(transition["truncated"], dtype=None),
            next_obs=self._stored_obs(transition, "next_observation"),
        )

    def ready(self) -> bool:
        return self._alg.can_start_training()

    def update(self) -> dict[str, float]:
        return self._alg.update()

    # ─── checkpoints ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        torch.save(self._alg.save(), os.path.join(path, CHECKPOINT_FILE))
        with open(os.path.join(path, "obs_layout.json"), "w") as f:
            json.dump({"encoded_obs_dims": self._encoded_layout, "obs_terms": self._obs_terms}, f)
        print(f"\033[32m[FlashSAC]\033[0m Successfully saved checkpoint {self._alg.update_step} at {path}.")

    def load(self, path: str) -> None:
        """Load our checkpoint, or a flash_rl one (per-network .pt files) through utils/legacy.py."""
        self._check_obs_layout(path)
        ckpt = os.path.join(path, CHECKPOINT_FILE)
        if os.path.exists(ckpt):
            loaded = torch.load(ckpt, map_location=self._device)
        elif is_legacy_checkpoint(path):
            loaded = load_legacy_checkpoint(path, self._alg)
        else:
            raise FileNotFoundError(f"no FlashSAC checkpoint in {path}")
        load_cfg = {
            "actor": True,
            "critic": True,
            "optimizer": self._cfg.load_optimizer,
            "reward_normalizer": self._cfg.load_reward_normalizer,
        }
        self._alg.load(loaded, load_cfg=load_cfg, strict=True)
        print(f"\033[32m[FlashSAC]\033[0m Successfully loaded checkpoint from {path}.")

    def save_buffer(self, path: str) -> None:
        self._alg.replay_buffer.save(os.path.join(path, REPLAY_BUFFER_FILE))
        print(f"\033[32m[FlashSAC]\033[0m Successfully saved replay buffer at {path}.")

    def load_buffer(self, path: str) -> None:
        self._alg.replay_buffer.load(os.path.join(path, REPLAY_BUFFER_FILE))
        print(f"\033[32m[FlashSAC]\033[0m Successfully loaded replay buffer from {path}.")

    def _check_obs_layout(self, path: str) -> None:
        """Refuse a checkpoint whose obs layout differs from this env's: equal-width changes (merged
        future_traj window, split SDF terms) would otherwise load silently with permuted inputs."""
        f = os.path.join(path, "obs_layout.json")
        saved = json.load(open(f)) if os.path.exists(f) else {}
        enc = saved.get("encoded_obs_dims")
        if enc is None and not any(n == "future_traj" for n, _ in self._encoded_layout):
            return  # checkpoints without obs_layout.json predate future_traj (split windows)
        if enc != self._encoded_layout:
            raise ValueError(
                f"checkpoint {path} has encoded obs {enc or 'of the split-window layout'}, env has "
                f"{self._encoded_layout}; split-window runs need envs/obs@obs=flashdexretarget_split."
            )
        # a renamed term is fine; a different [group, width] sequence means the inputs moved
        terms = saved.get("obs_terms")
        if terms is not None and self._obs_terms:
            widths = [[g, w] for g, _, w in terms]
            if widths != [[g, w] for g, _, w in self._obs_terms]:
                raise ValueError(
                    f"checkpoint {path} obs layout differs from the env's:\n  ckpt {terms}\n  env  {self._obs_terms}"
                )
