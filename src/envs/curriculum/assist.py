from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

import numpy as np
import torch

from ..rewards import side_twin_name

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def _with_side_twins(names: tuple[str, ...]) -> tuple[str, ...]:
    """names + their per-hand split_sides twins, order-preserving and without repeats."""
    twins = tuple(side_twin_name(n, s) for n in names for s in ("r", "l"))
    return tuple(dict.fromkeys(names + twins))


class XfrcAssistCurriculum:
    """Object-assist curriculum: decays the xfrc pin gains once the reward groups and the
    clip-completion rate clear their gates."""

    def __init__(
        self,
        *,
        command,
        num_envs: int,
        max_episode_length: int,
        num_steps_per_env: int,
        kp_init: float,
        kv_init: float,
        force_range: float,
        wait_epochs: int,
        deque_len: int,
        reward_terms: dict,
        rew_thresholds: dict,
        upper_ratios: dict,
        lower_ratios: dict,
        dialback_completion_thres: float,
        dialback_min_epochs: int,
        dialback_ratios: dict,
        decay_cooldown_epochs: int,
        kp_zero_thres: float,
        kp_lower_zero_thres: float,
        completion_thres: float,
        kp_lower_init: float | None = None,
        kv_lower_init: float | None = None,
        epochs_since_decay_init: int = 0,
        rot_stiffness_mult: float = 1.0,
        kp_floor: float = 0.0,
        kv_floor: float = 0.0,
        per_dof_gains: bool = False,
        zero_epoch: int = -1,
        completion_term: str | None = None,
        seed: int = 42,
        device: str = "cuda:0",
    ) -> None:
        self._command = command
        self.num_envs = int(num_envs)
        self.max_episode_length = int(max_episode_length)
        self.num_steps_per_env = int(num_steps_per_env)
        self.device = device

        self.init_gains = {"kp": float(kp_init), "kv": float(kv_init), "fr": float(force_range)}
        self.curr_gains = self.init_gains.copy()
        self.curr_gains_lower = self.init_gains.copy()
        if kp_lower_init is not None:
            self.curr_gains_lower["kp"] = float(kp_lower_init)
        if kv_lower_init is not None:
            self.curr_gains_lower["kv"] = float(kv_lower_init)
        self.gain_history: list[dict] = []
        self.decay_terms = ["kp", "kv", "fr"]

        self.wait_epochs = int(wait_epochs)
        self.deque_len = int(deque_len)
        self.rew_thresholds = dict(rew_thresholds)
        self.upper_ratios = dict(upper_ratios)
        self.lower_ratios = dict(lower_ratios)
        self.dialback_completion_thres = float(dialback_completion_thres)
        self.dialback_min_epochs = int(dialback_min_epochs)
        self.dialback_ratios = dict(dialback_ratios)
        self.decay_cooldown_epochs = int(decay_cooldown_epochs)
        self.kp_zero_thres = float(kp_zero_thres)
        self.kp_lower_zero_thres = float(kp_lower_zero_thres)
        self.completion_thres = float(completion_thres)
        self.gain_floors = {"kp": float(kp_floor), "kv": float(kv_floor), "fr": 0.0}
        self.rot_stiffness_mult = float(rot_stiffness_mult)
        self.per_dof_gains = bool(per_dof_gains)
        self.zero_epoch = int(zero_epoch)
        self.completion_term = completion_term
        # Only the terms active in a run count toward their group.
        self.reward_terms = {k: _with_side_twins(tuple(v)) for k, v in reward_terms.items()}
        if not self.reward_terms:
            raise ValueError("xfrc_curriculum_adaptive: reward_terms is empty")

        self.rew_deques = {k: deque(maxlen=self.deque_len) for k in self.reward_terms}
        # completion_rates gates the decay; ep_lens is logged only.
        self.ep_lens: deque = deque(maxlen=self.deque_len)
        self.completion_rates: deque = deque(maxlen=self.deque_len)
        self.num_epoch_since_last_decay = int(epochs_since_decay_init)

        np.random.seed(int(seed))
        torch.manual_seed(int(seed))

        self._epoch_rew_sum = {k: 0.0 for k in self.reward_terms}
        self._epoch_steps = 0
        self._epoch_achieved: list[float] = []
        self._epoch_timeout = 0
        self._epoch_terminated = 0
        self._prev_ep_len: torch.Tensor | None = None
        self._last_epoch = -1

        # group -> [(term index, weight)], resolved once the reward manager exists.
        self._term_idx: dict[str, list[tuple[int, float]]] | None = None

        self._last_decayed = False

        # Seed the per-env gains so the pin uses them from step 0.
        self._reset_object_gains()

    # ── reward-term resolution ───────────────────────────────────────────

    def _resolve_term_idx(self, env: "ManagerBasedRlEnv") -> None:
        rm = env.reward_manager
        names = list(rm.active_terms)

        def _idx_w(name: str) -> tuple[int, float]:
            i = names.index(name)
            return i, float(rm._term_cfgs[i].weight)

        self._term_idx = {
            k: [_idx_w(n) for n in terms if n in names] for k, terms in self.reward_terms.items()
        }

    def _weighted_rewards(self, env: "ManagerBasedRlEnv") -> dict[str, float]:
        rm = env.reward_manager
        step_r = rm._step_reward  # (B, n_terms) = fn * weight
        out: dict[str, float] = {}
        for key, members in self._term_idx.items():  # type: ignore[union-attr]
            total: torch.Tensor | None = None
            for idx, weight in members:
                if weight == 0.0:
                    continue
                col = step_r[:, idx]
                total = col if total is None else total + col
            out[key] = float(total.mean().item()) if total is not None else 0.0
        return out

    # ── per-step driver ──────────────────────────────────────────────────

    def on_step(self, env: "ManagerBasedRlEnv") -> None:
        """Called every env step by the command; updates the gains at each epoch boundary."""
        if self._term_idx is None:
            self._resolve_term_idx(env)

        self._accumulate_step(env)

        epoch_num = int(env.common_step_counter) // self.num_steps_per_env
        if epoch_num == self._last_epoch:
            return
        self._last_epoch = epoch_num
        if self._epoch_steps == 0:
            return

        reward = {
            k: self._epoch_rew_sum[k] / self._epoch_steps for k in self.reward_terms
        }

        # An epoch without resets counts as fully completed, so it never blocks the decay.
        achieved_length = (
            float(np.mean(self._epoch_achieved)) if self._epoch_achieved else 0.0
        )
        total_resets = self._epoch_timeout + self._epoch_terminated
        completion_rate = (
            self._epoch_timeout / total_resets if total_resets > 0 else 1.0
        )

        self._update_progress(reward, achieved_length, completion_rate)
        self._set_curriculum(env, epoch_num)

        self._epoch_rew_sum = {k: 0.0 for k in self.reward_terms}
        self._epoch_steps = 0
        self._epoch_achieved = []
        self._epoch_timeout = 0
        self._epoch_terminated = 0

    def _accumulate_step(self, env: "ManagerBasedRlEnv") -> None:
        for k, v in self._weighted_rewards(env).items():
            self._epoch_rew_sum[k] += v
        self._epoch_steps += 1

        # Completed = time-out or completion_term; any other termination failed.
        tm = getattr(env, "termination_manager", None)
        if tm is not None:
            done_ok = tm.time_outs
            if self.completion_term is not None:
                done_ok = done_ok | tm.get_term(self.completion_term)
            self._epoch_timeout += int(done_ok.sum().item())
            self._epoch_terminated += int((tm.terminated & ~done_ok).sum().item())

        # An env whose length dropped just reset: its previous length is one episode.
        ep_len = env.episode_length_buf
        if self._prev_ep_len is not None:
            reset_mask = ep_len < self._prev_ep_len
            if torch.any(reset_mask):
                done = self._prev_ep_len[reset_mask].float()
                self._epoch_achieved.extend(done.tolist())
        self._prev_ep_len = ep_len.clone()

    # ── decay / dial-back ────────────────────────────────────────────────

    def _update_progress(
        self, rewards: dict, achieved_length: float, completion_rate: float
    ) -> None:
        for k, val in rewards.items():
            self.rew_deques[k].append(val)
        self.ep_lens.append(achieved_length)
        self.completion_rates.append(completion_rate)

    def _determine_decay(self, epoch_num: int) -> bool:
        reduce_gains = True
        if epoch_num < self.wait_epochs:
            return False
        for key in self.rew_deques.keys():
            # A group with no active term or no threshold is not a gate.
            if not (self._term_idx or {}).get(key) or key not in self.rew_thresholds:
                continue
            if len(self.rew_deques[key]) < self.deque_len:
                return False
            rew_mean = float(np.mean(self.rew_deques[key]))
            rew_thres = self.rew_thresholds[key]
            if rew_mean < rew_thres:
                reduce_gains = False
                break
        if self.num_epoch_since_last_decay < self.decay_cooldown_epochs:
            reduce_gains = False
        if len(self.completion_rates) < self.deque_len:
            reduce_gains = False
        else:
            completion = float(np.mean(self.completion_rates))
            if completion < self.completion_thres:
                reduce_gains = False
        return reduce_gains

    def _determine_dialback(self, epoch_num: int) -> bool:
        dialed = False
        if len(self.completion_rates) < self.deque_len:
            return dialed
        completion = float(np.mean(self.completion_rates))
        if completion < self.dialback_completion_thres:
            if self.num_epoch_since_last_decay > self.dialback_min_epochs:
                upper_gains = self.gain_history[-1]["gains"]
                lower_gains = self.gain_history[-1]["lower"]
                for k in self.decay_terms:
                    if k in upper_gains:
                        ratio = self.dialback_ratios.get(k, 1.0)
                        upper_gains[k] *= ratio
                        lower_gains[k] *= ratio
                self.curr_gains = upper_gains
                self.curr_gains_lower = lower_gains
                dialed = True
        return dialed

    def _set_auto_uniform_decay(self) -> None:
        """upper *= upper_ratio; lower = upper * lower_ratio, both clamped at the floors."""
        for k in self.decay_terms:
            self.curr_gains[k] = max(
                self.curr_gains[k] * self.upper_ratios.get(k, 1.0),
                self.gain_floors.get(k, 0.0),
            )
        for k in self.decay_terms:
            self.curr_gains_lower[k] = max(
                self.curr_gains[k] * self.lower_ratios.get(k, 1.0),
                self.gain_floors.get(k, 0.0),
            )

        # Full-release thresholds only apply when no floor is set.
        if self.gain_floors["kp"] <= 0.0:
            if self.curr_gains["kp"] < self.kp_zero_thres:
                for k in self.decay_terms:
                    self.curr_gains[k] = 0.0
                    self.curr_gains_lower[k] = 0.0
            if self.curr_gains_lower.get("kp", 1.0) < self.kp_lower_zero_thres:
                for k in self.decay_terms:
                    self.curr_gains_lower[k] = 0.0

    def _set_curriculum(self, env: "ManagerBasedRlEnv", epoch_num: int) -> None:
        self.num_epoch_since_last_decay += 1
        self._last_decayed = False

        learning_stabilized = self._determine_decay(epoch_num)
        zero_gains = all(self.curr_gains[k] <= 0.0 for k in self.decay_terms)
        if self.zero_epoch >= 0 and epoch_num > self.zero_epoch and not zero_gains:
            for k in self.decay_terms:
                self.curr_gains[k] = 0.0
                self.curr_gains_lower[k] = 0.0
            self._last_decayed = True
            self._reset_object_gains()
            return
        if zero_gains or not learning_stabilized:
            return

        dialed_back = self._determine_dialback(epoch_num)
        if dialed_back:
            self._last_decayed = True
        else:
            self.gain_history.append(
                dict(gains=self.curr_gains.copy(), lower=self.curr_gains_lower.copy())
            )
            self._set_auto_uniform_decay()
            self._last_decayed = True

        if self._last_decayed:
            self._reset_object_gains()
            for key in self.rew_deques.keys():
                self.rew_deques[key].clear()
            self.ep_lens.clear()
            self.completion_rates.clear()
            self.num_epoch_since_last_decay = 0

    def _reset_object_gains(self) -> None:
        """Per-env assist gains drawn uniformly from [lower, upper]."""
        cmd = self._command
        B = self.num_envs

        shape = (B, 3) if self.per_dof_gains else (B,)

        def _sample(key: str) -> torch.Tensor:
            lower = self.curr_gains_lower[key]
            upper = self.curr_gains[key]
            if upper <= 0.0:
                return torch.zeros(shape, device=self.device)
            return torch.rand(shape, device=self.device) * (upper - lower) + lower

        kp = _sample("kp")
        kv = _sample("kv")
        if self.per_dof_gains:  # independent draw for the rotation axes
            kp_r, kv_r = _sample("kp"), _sample("kv")
        else:
            kp_r, kv_r = kp, kv
        cmd._xfrc_kp_pos_env = kp
        cmd._xfrc_kv_pos_env = kv
        # Rotation: kp·mult and kv·√mult keep the damping ratio of the translation PD.
        rot_mult = self.rot_stiffness_mult
        cmd._xfrc_kp_rot_env = kp_r * rot_mult
        cmd._xfrc_kv_rot_env = kv_r * (rot_mult ** 0.5)
        cmd._xfrc_force_range = float(self.curr_gains["fr"]) if self.curr_gains["fr"] > 0.0 else None

    # ── logging ──────────────────────────────────────────────────────────

    def log_dict(self) -> dict[str, torch.Tensor]:
        def _m(seq) -> float:
            return float(np.mean(seq)) if len(seq) else 0.0

        return {
            "xfrc_kp_upper": torch.tensor(self.curr_gains["kp"]),
            "xfrc_kp_lower": torch.tensor(self.curr_gains_lower["kp"]),
            "xfrc_kv_upper": torch.tensor(self.curr_gains["kv"]),
            "xfrc_kv_lower": torch.tensor(self.curr_gains_lower["kv"]),
            "xfrc_fr": torch.tensor(self.curr_gains["fr"]),
            "xfrc_epoch": torch.tensor(float(self._last_epoch)),
            "xfrc_decayed": torch.tensor(1.0 if self._last_decayed else 0.0),
            "xfrc_epochs_since_decay": torch.tensor(float(self.num_epoch_since_last_decay)),
            **{f"xfrc_mean_{k}_rew": torch.tensor(_m(d)) for k, d in self.rew_deques.items()},
            "xfrc_mean_ep_len": torch.tensor(_m(self.ep_lens)),
            "xfrc_completion_rate": torch.tensor(_m(self.completion_rates)),
            "xfrc_completion_deque_len": torch.tensor(float(len(self.completion_rates))),
            "xfrc_task_deque_len": torch.tensor(float(len(self.rew_deques.get("task", ())))),
        }
