"""Gymnasium VectorEnv adapting an mjlab ManagerBasedRlEnv to agents.Agent: flat [actor | critic]
obs, auto_reset single-pass stepping, final_obs truncation bootstrap, share_obs dedup."""

from __future__ import annotations

import re
from typing import Any, Union

import gymnasium as gym
import numpy as np
import numpy.typing as npt
import torch
from gymnasium.vector import VectorEnv
from gymnasium.vector.utils import batch_space

NDArray = npt.NDArray[Any]
F32NDArray = npt.NDArray[np.float32]


class FlatObsVecEnv(VectorEnv):
    """VectorEnv over a mjlab ManagerBasedRlEnv with [actor | critic] flat obs."""

    def __init__(
        self,
        env: Any,
        to_numpy: bool = True,
        share_obs: bool = False,
        reward_heads: dict[str, Any] | None = None,
    ) -> None:
        # mjlab owns resets (single all-env obs/command pass per step); the
        # pre-reset terminal obs arrives via extras["final_obs"] when needed.
        env.cfg.auto_reset = True
        env.cfg.capture_final_obs = True

        self._env = env
        self._device = str(env.device)
        self._to_numpy = to_numpy
        self._share_obs = share_obs
        self.num_envs = int(env.num_envs)

        obs_space = env.single_observation_space
        obs_groups = list(obs_space.spaces.keys())
        if "actor" not in obs_groups or "critic" not in obs_groups:
            raise ValueError(f"need both 'actor' and 'critic' obs groups, got {obs_groups}.")
        self._actor_obs_key = "actor"
        self._critic_obs_key = "critic"
        self._actor_obs_dim = int(obs_space.spaces[self._actor_obs_key].shape[0])
        self._critic_obs_dim = int(obs_space.spaces[self._critic_obs_key].shape[0])
        if share_obs and self._actor_obs_dim != self._critic_obs_dim:
            raise ValueError(
                f"share_obs=True needs identical actor/critic obs groups, got dims "
                f"{self._actor_obs_dim} vs {self._critic_obs_dim}."
            )
        # Extra (encoded) obs groups ride along appended to the flat obs tail, sorted for
        # a deterministic layout shared with the agent via env_info["encoded_obs_dims"].
        self._encoded_obs_keys = sorted(k for k in obs_groups if k not in ("actor", "critic"))
        self._encoded_obs_dims = [int(obs_space.spaces[k].shape[0]) for k in self._encoded_obs_keys]
        if self._encoded_obs_keys and not share_obs:
            raise ValueError(
                f"encoded obs groups {self._encoded_obs_keys} require share_obs=True "
                f"(flat layout is [shared | windows])."
            )
        flat_dim = self._actor_obs_dim if share_obs else self._actor_obs_dim + self._critic_obs_dim
        flat_dim += sum(self._encoded_obs_dims)

        action_dim = int(env.single_action_space.shape[0])

        self.single_observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(flat_dim,), dtype=np.float32)
        self.observation_space = batch_space(self.single_observation_space, self.num_envs)
        self.single_action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32)
        self.action_space = batch_space(self.single_action_space, self.num_envs)

        # consumed by the agent setup
        self.obs_size = (flat_dim,)
        self.action_size = (action_dim,)

        self._ep_returns = np.zeros(self.num_envs, dtype=np.float32)
        self._ep_lengths = np.zeros(self.num_envs, dtype=np.int32)
        # Side-factored reward heads: reward becomes (B, H) with one column per hand side,
        # each the dt-scaled sum of that side's weighted terms.
        self._head_names: list[str] = []
        self._head_matrix: torch.Tensor | None = None
        self._head_checks_left = 0
        self._action_sides: dict[str, list[int]] = {}
        if reward_heads and bool(reward_heads.get("enable", False)):
            self._setup_reward_heads(dict(reward_heads.get("overrides") or {}))
            self._ep_head_returns = np.zeros((self.num_envs, len(self._head_names)), dtype=np.float32)
        # Optional obs-aux provider (obs compaction): called right after each obs compute.
        self._aux_fn: Any = None
        self._last_aux: NDArray | None = None

    @property
    def reward_head_names(self) -> list[str]:
        return list(self._head_names)

    @property
    def share_obs(self) -> bool:
        return self._share_obs

    def current_aux(self) -> NDArray | None:
        """Obs aux of the current state (None without a provider)."""
        return self._aux_now()

    def _setup_reward_heads(self, overrides: dict[str, str]) -> None:
        """Map every active reward term to a hand side by its `_r_`/`_l_` name token (or an
        explicit override: right|left|both); untagged terms are split evenly across heads."""
        cmd = self._env.command_manager.get_term("motion")
        sides = list(cmd._side_list)
        if len(sides) < 2:
            raise ValueError(f"reward_heads needs a bimanual command, got sides {sides}.")
        rm = self._env.reward_manager
        names = list(rm.active_terms)
        rows, table = [], []
        for name in names:
            tag = overrides.get(name)
            if tag is None:
                m = re.search(r"(^|_)([rl])(_|$)", name)
                tag = {"r": "right", "l": "left"}.get(m.group(2)) if m else "both"
            if tag == "both":
                row = [1.0 / len(sides)] * len(sides)
            elif tag in sides:
                row = [1.0 if s_ == tag else 0.0 for s_ in sides]
            else:
                raise ValueError(f"reward_heads: term {name!r} tagged {tag!r}, expected right|left|both.")
            rows.append(row)
            table.append(f"  {name:<44s} -> {tag}")
        self._head_names = sides
        self._head_matrix = torch.tensor(rows, dtype=torch.float32, device=self._device)
        self._head_checks_left = 20
        print(f"[reward_heads] {len(names)} terms -> heads {sides}:\n" + "\n".join(table), flush=True)
        self._action_sides = self._action_side_indices(sides)

    def _action_side_indices(self, sides: list[str]) -> dict[str, list[int]]:
        """Action-vector indices per side (joint-name prefix R_/right_ vs L_/left_); the action
        term groups dims by type ([wrist_trans R,L | wrist_rot R,L | fingers R,L]), not by side."""
        from envs._common import action_term

        act_term = action_term(self._env)
        names = act_term._entity.joint_names
        joint_ids = [int(j) for j in act_term._all_joint_ids.tolist()]
        prefixes = {"right": ("R_", "right_"), "left": ("L_", "left_")}
        out = {s_: [i for i, j in enumerate(joint_ids) if names[j].startswith(prefixes[s_])] for s_ in sides}
        covered = sorted(i for v in out.values() for i in v)
        if covered != list(range(len(joint_ids))):
            raise ValueError(f"action dims not partitioned by side: {out} over {len(joint_ids)} dims")
        print("[reward_heads] action sides: " + ", ".join(f"{k}={v}" for k, v in out.items()), flush=True)
        return out

    def _head_rewards(self, rewards: torch.Tensor) -> torch.Tensor:
        """(B,) total -> (B, H) per-side rewards from the manager's per-term step rewards."""
        rm = self._env.reward_manager
        scale = self._env.step_dt if getattr(rm, "_scale_by_dt", True) else 1.0
        heads = (rm._step_reward @ self._head_matrix) * scale  # (B, H)
        if self._head_checks_left > 0:
            self._head_checks_left -= 1
            gap = (heads.sum(dim=-1) - rewards).abs().max().item()
            if gap > 1e-4 * max(1.0, rewards.abs().max().item()):
                print(f"[reward_heads] WARNING head sum != total reward (max gap {gap:.3e})", flush=True)
        return heads

    def set_aux_provider(self, fn: Any) -> None:
        """Register the obs-aux provider; snapshots the current aux so the next step's
        final_obs_aux is valid even when registered after reset()."""
        self._aux_fn = fn
        self._last_aux = self._aux_now()

    def _aux_now(self) -> NDArray | None:
        return None if self._aux_fn is None else self._aux_fn().cpu().numpy()

    @classmethod
    def from_env(
        cls,
        env: Any,
        to_numpy: bool = True,
        share_obs: bool = False,
        reward_heads: dict[str, Any] | None = None,
    ) -> "FlatObsVecEnv":
        return cls(env, to_numpy=to_numpy, share_obs=share_obs, reward_heads=reward_heads)

    @property
    def unwrapped(self) -> Any:
        """Raw mjlab env (for eval callbacks / command_manager)."""
        return self._env

    def _flatten_obs(self, obs_dict: dict[str, Any]) -> F32NDArray:
        if self._share_obs:
            parts = [obs_dict[self._actor_obs_key]]
        else:
            parts = [obs_dict[self._actor_obs_key], obs_dict[self._critic_obs_key]]
        parts += [obs_dict[k] for k in self._encoded_obs_keys]
        flat = parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)
        return flat.cpu().numpy().astype(np.float32)

    def _obs_terms(self) -> list[list]:
        """[group, term, width] of every flat obs column block, in layout order."""
        om = self.unwrapped.observation_manager
        groups = [self._actor_obs_key] + ([] if self._share_obs else [self._critic_obs_key]) + self._encoded_obs_keys
        return [[g, n, int(np.prod(d))] for g in groups for n, d in zip(om.active_terms[g], om.group_obs_term_dim[g])]

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[F32NDArray, dict[str, Any]]:
        obs_dict, _ = self._env.reset()
        if self._share_obs and not torch.equal(
            obs_dict[self._actor_obs_key], obs_dict[self._critic_obs_key]
        ):
            raise ValueError(
                "share_obs=True but actor and critic obs differ numerically — the obs "
                "config must use identical term lists with zero actor noise to dedup."
            )
        self._ep_returns[:] = 0.0
        self._ep_lengths[:] = 0
        env_info: dict[str, Any] = {
            "actor_observation_size": (self._actor_obs_dim,),
            "encoded_obs_dims": list(zip(self._encoded_obs_keys, self._encoded_obs_dims)),
            "obs_terms": self._obs_terms(),
            "reward_heads": list(self._head_names),
            "action_sides": {k: list(v) for k, v in self._action_sides.items()},
        }
        if self._head_matrix is not None:
            self._ep_head_returns[:] = 0.0
        self._last_aux = self._aux_now()
        if self._last_aux is not None:
            env_info["obs_aux"] = self._last_aux
        return self._flatten_obs(obs_dict), env_info

    def step(
        self,
        actions: Union[F32NDArray, torch.Tensor],
    ) -> tuple[F32NDArray, F32NDArray, NDArray, NDArray, dict[str, Any]]:
        if isinstance(actions, np.ndarray):
            actions_t = torch.from_numpy(actions).float().to(self._device)
        else:
            actions_t = actions.to(self._device)

        obs_dict, rewards, terminateds, truncateds, extras = self._env.step(actions_t)

        rewards_np = rewards.cpu().numpy().astype(np.float32)
        self._ep_returns += rewards_np
        self._ep_lengths += 1
        if self._head_matrix is not None:
            head_np = self._head_rewards(rewards).cpu().numpy().astype(np.float32)  # (B, H)
            self._ep_head_returns += head_np
            rewards_np = head_np

        # auto_reset=True: done envs already reset inside step(), obs_dict is the
        # post-reset buffer -> this IS next_obs, computed in mjlab's single pass.
        next_obs = self._flatten_obs(obs_dict)

        # Pre-reset terminal obs (present only when some env timed out); fallback next_obs
        # is never consumed: only truncated rows read final_obs, terminated envs don't bootstrap.
        final_obs_dict = extras.get("final_obs")
        terminal_obs = self._flatten_obs(final_obs_dict) if final_obs_dict is not None else next_obs

        dones = terminateds | truncateds
        done_ids = dones.nonzero(as_tuple=False).squeeze(-1)

        infos: dict[str, Any] = {"final_obs": terminal_obs}
        if self._aux_fn is not None:
            # final_obs is snapshotted BEFORE the command update -> it carries the previous aux
            infos["final_obs_aux"] = self._last_aux
            self._last_aux = self._aux_now()
            infos["obs_aux"] = self._last_aux

        # episode return/length for done envs + mjlab's per-term log means
        done_ids_np = done_ids.cpu().numpy()
        raw_log = extras.get("log") or {}
        episode_info: dict[str, Any] = {
            k: float(v.mean().item()) if isinstance(v, torch.Tensor) else v for k, v in raw_log.items()
        }
        if len(done_ids_np) > 0:
            episode_info["episode_rewards"] = float(self._ep_returns[done_ids_np].mean())
            episode_info["episode_length"] = float(self._ep_lengths[done_ids_np].mean())
            self._ep_returns[done_ids_np] = 0.0
            self._ep_lengths[done_ids_np] = 0
            if self._head_matrix is not None:
                for h, name in enumerate(self._head_names):
                    episode_info[f"episode_rewards_{name}"] = float(self._ep_head_returns[done_ids_np, h].mean())
                self._ep_head_returns[done_ids_np] = 0.0
        if episode_info:
            infos["episode_info"] = episode_info

        return (
            next_obs,
            rewards_np,
            terminateds.cpu().numpy(),
            truncateds.cpu().numpy(),
            infos,
        )

    def close(self, **kwargs: Any) -> None:
        if hasattr(self, "_env"):
            self._env.close()
