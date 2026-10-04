"""rsl_rl_flashsac's FlashSAC extended with the recipe's reward heads, side actors and compact replay obs."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from typing import Any

import torch
import torch.optim as optim
from rsl_rl_flashsac.algorithms import FlashSAC
from tensordict import TensorDict

from ..storage.replay_buffer import ReplayBuffer, ReplayObsLayout
from ..modules.actor_critic import ActorTower, MultiHeadCritic, SideTemperature, SideFactoredActor
from ..utils.reward_norm import MultiHeadRewardNormalizer


class MultiHeadFlashSAC(FlashSAC):
    """FlashSAC with H reward heads (2H critic members, per-head normalizer) and optional side actors
    (one tower + temperature per head); H=1 without side actors runs upstream's update unchanged."""

    actor: ActorTower | SideFactoredActor  # type: ignore[assignment]
    critic: MultiHeadCritic

    def __init__(
        self,
        actor: ActorTower | SideFactoredActor,
        critic: MultiHeadCritic,
        replay_buffer: ReplayBuffer,
        *,
        obs_layout: ReplayObsLayout,
        target_entropy: list[float],
        num_heads: int = 1,
        temp_initial_value: float = 0.01,
        **kwargs: Any,
    ) -> None:
        """target_entropy has one entry per side actor (a single entry without side actors)."""
        super().__init__(
            actor,  # type: ignore[arg-type]
            critic,
            replay_buffer,
            temp_initial_value=temp_initial_value,
            temp_target_entropy=float(sum(target_entropy)),
            **kwargs,
        )
        self.obs_layout = obs_layout
        self.num_heads = num_heads
        self.side_masks: torch.Tensor | None = actor.side_masks if isinstance(actor, SideFactoredActor) else None
        num_sides = 1 if self.side_masks is None else self.side_masks.shape[0]
        if len(target_entropy) != num_sides:
            raise ValueError(f"target_entropy needs {num_sides} entries, got {len(target_entropy)}.")
        if self.side_masks is not None and num_sides != num_heads:
            raise ValueError(f"side actors need one reward head per side ({num_sides} sides, {num_heads} heads).")
        self.side_target_entropy = torch.tensor(target_entropy, dtype=torch.float32, device=self.device)

        if isinstance(actor, SideFactoredActor):
            self._init_side_temperature(temp_initial_value, num_sides)
            self._actor_sample_sides = (
                torch.compile(actor.sample_action_logp_sides, mode=self.compile_mode)
                if self.use_compile
                else actor.sample_action_logp_sides
            )
        if num_heads > 1 and self.reward_normalizer is not None:
            self.reward_normalizer = MultiHeadRewardNormalizer(  # type: ignore[assignment]
                num_heads, self.gamma, self.reward_normalizer.G_max, self.device
            )

    def _init_side_temperature(self, initial_value: float, num_sides: int) -> None:
        """Replace the scalar temperature by one per side, keeping upstream's optimizer and LR schedule."""
        lr_lambda = self.temperature_scheduler.lr_lambdas[0]
        base_lr = self.temperature_scheduler.base_lrs[0]
        fused = self.temperature_optimizer.defaults["fused"]
        self.temperature = SideTemperature(initial_value, num_sides).to(self.device)
        self.temperature_parameters = list(self.temperature.parameters())
        self.temperature_optimizer = optim.Adam(self.temperature_parameters, lr=base_lr, fused=fused)
        self.temperature_scheduler = optim.lr_scheduler.LambdaLR(self.temperature_optimizer, lr_lambda=lr_lambda)

    @property
    def alpha(self) -> float:
        """Mean temperature over the side actors (for logging)."""
        with torch.no_grad():
            return self.temperature().mean().item()

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        """Deterministic tanh(mean) action; the exploration-noise state is left untouched."""
        with torch.no_grad():
            self._mark_cudagraph_step()
            mean, _ = self._actor_mean_std(self.actor.flatten_obs(obs), training=False)
            return torch.tanh(mean)

    def add_transition(
        self,
        obs: TensorDict,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        next_obs: TensorDict,
    ) -> None:
        """Store one vector-env step; a step both terminated and truncated stays terminated (no bootstrap)."""
        self.replay_buffer.add(
            {
                "observation": obs,
                "action": actions,
                "reward": rewards,
                "terminated": terminated,
                "truncated": truncated,
                "next_observation": next_obs,
            }
        )
        if self.reward_normalizer is not None:
            self.reward_normalizer.update_reward_stats(reward=rewards, terminated=terminated, truncated=truncated)

    def _sample_batch(self) -> dict:
        batch = super()._sample_batch()
        if self.obs_layout.compactor is not None:
            # one rebuild over obs + next obs: the expand is launch-bound, not row-bound
            n = batch["observation"].batch_size[0]
            stored = torch.cat([batch["observation"], batch["next_observation"]]).to(self.device)
            full = self.obs_layout.to_network(stored)
            batch["observation"], batch["next_observation"] = full[:n], full[n:]
        return batch

    def update(self) -> dict[str, float]:  # type: ignore[override]
        """Run the configured updates; returns flash_rl's metric names (actor/*, critic/*, temperature/*)."""
        if not self.replay_buffer.can_sample():
            return {}
        if self.num_heads == 1 and self.side_masks is None:
            return self._update_single_head()
        return self._update_heads()

    def _update_single_head(self) -> dict[str, float]:
        """Upstream's update loop and losses; metrics renamed (temperature/value is the pre-update value)."""
        num_updates = self.num_learning_epochs * self.num_mini_batches
        actor_updates = any((self.update_step + i) % self.actor_update_period == 0 for i in range(num_updates))
        temperature = self.alpha if actor_updates else None
        losses = super().update()
        info = {"critic/loss": losses["critic"]}
        if temperature is not None:
            info["actor/loss"] = losses["actor"]
            info["actor/entropy"] = losses["entropy"]
            info["temperature/loss"] = losses["temperature"]
            info["temperature/value"] = temperature
        return info

    def _update_heads(self) -> dict[str, float]:
        sums: dict[str, float] = defaultdict(float)
        counts: dict[str, int] = defaultdict(int)
        gamma_n = self.gamma**self.n_steps

        for _ in range(self.num_learning_epochs * self.num_mini_batches):
            self._mark_cudagraph_step()
            batch = self._sample_batch()

            obs_batch = batch["observation"].to(self.device, non_blocking=True)
            next_obs_batch = batch["next_observation"].to(self.device, non_blocking=True)
            actions_batch = batch["action"].to(self.device, non_blocking=True)
            rewards = batch["reward"].to(self.device, non_blocking=True)
            terminated = batch["terminated"].to(self.device, non_blocking=True)

            actor_obs = self.actor.flatten_obs(obs_batch, training=True)
            actor_next_obs = self.actor.flatten_obs(next_obs_batch, training=True)
            critic_obs = self.critic.flatten_obs(obs_batch)
            critic_next_obs = self.critic.flatten_obs(next_obs_batch)

            if self.reward_normalizer is not None:
                rewards = self.reward_normalizer.normalize_rewards(rewards)

            info: dict[str, torch.Tensor] = {}
            if self.update_step % self.actor_update_period == 0:
                actor_info, entropy_sides = self._update_actor_heads(
                    actor_obs, actor_next_obs, critic_obs, actions_batch
                )
                info.update(actor_info)
                info.update(self._update_temperature_sides(entropy_sides))
            info.update(
                self._update_critic_heads(
                    critic_obs, critic_next_obs, actor_next_obs, actions_batch, rewards, terminated, gamma_n
                )
            )

            with torch.no_grad():
                self._ema_update()
            self.update_step += 1

            for key, value in info.items():
                sums[key] += float(value)
                counts[key] += 1
        return {key: sums[key] / counts[key] for key in sums}

    def _gate_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """(B, A) -> (2H, B, A) with member twin*H + h differentiable only in side h's action dims."""
        if self.side_masks is None:
            return actions
        detached = actions.detach()
        gated = torch.stack([torch.where(mask, actions, detached) for mask in self.side_masks], dim=0)
        return gated.repeat(2, 1, 1)

    def _update_actor_heads(
        self,
        actor_obs: torch.Tensor,
        actor_next_obs: torch.Tensor,
        critic_obs: torch.Tensor,
        actions_batch: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """SAC actor loss against every head's min-twin Q; side actors: head h trains tower h only.
        Returns the logged metrics and the per-side entropies (S,)."""
        num_heads = self.num_heads
        with torch.autocast(device_type=self._device_type, dtype=torch.float16, enabled=self.use_amp):
            # BatchNorm sees obs + next obs as in the critic update; only the first half enters the loss
            actor_obs_all = torch.cat([actor_obs, actor_next_obs], dim=0)
            if self.side_masks is not None:
                actions_all, log_prob_sides_all = self._actor_sample_sides(actor_obs_all, training=True)
                log_prob_sides = torch.chunk(log_prob_sides_all, 2, dim=1)[0]
            else:
                actions_all, log_probs_all = self._actor_sample(actor_obs_all, training=True)
                log_probs = torch.chunk(log_probs_all, 2, dim=0)[0]
            actions = torch.chunk(actions_all, 2, dim=0)[0]

            # Disable critic gradients to prevent CUDA graph overwriting
            self.critic.critic.requires_grad_(False)
            qs, _ = self._critic_eval(critic_obs, self._gate_actions(actions), training=False)
            q_heads = qs.reshape(2, num_heads, -1)
            q_head_min = torch.minimum(q_heads[0], q_heads[1])  # (H, B)
            self.critic.critic.requires_grad_(True)

            temp_value = self.temperature().detach()
            q = q_head_min.mean(dim=0)
            if self.side_masks is not None:
                # disjoint towers: summing the per-side losses equals stepping every side alone
                side_losses = (log_prob_sides * temp_value[:, None] - q_head_min).mean(dim=1)
                actor_loss = side_losses.sum()
                entropy_sides = -log_prob_sides.mean(dim=1)
            else:
                # the head mean keeps the single-head Q scale (each head target carries the entropy bonus)
                actor_loss = (log_probs * temp_value - q).mean()
                entropy_sides = (-log_probs.mean()).reshape(1)

            if self.actor_bc_alpha > 0:
                # https://arxiv.org/abs/2306.02451
                q_abs = torch.abs(q).mean().detach()
                bc_loss = ((actions - actions_batch) ** 2).mean()
                actor_loss = actor_loss + self.actor_bc_alpha * q_abs * bc_loss
            entropy = entropy_sides.sum()
            mean_action = actions.mean()

        self._step(actor_loss, self.actor_optimizer, self.actor_scheduler, self.actor_parameters, self._actor_normalize)

        info = {
            "actor/loss": actor_loss.detach(),
            "actor/entropy": entropy.detach(),
            "actor/mean_action": mean_action.detach(),
        }
        for h in range(num_heads):
            info[f"actor/q_head{h}"] = q_head_min[h].mean().detach()
        if self.side_masks is not None:
            for h in range(num_heads):
                info[f"actor/loss_side{h}"] = side_losses[h].detach()
                info[f"actor/entropy_side{h}"] = entropy_sides[h].detach()
        return info, entropy_sides.detach()

    def _update_temperature_sides(self, entropy_sides: torch.Tensor) -> dict[str, torch.Tensor]:
        """Drive each side's temperature toward that side's target entropy (one side: upstream's loss)."""
        temperature_value = self.temperature().clone()
        temperature_loss = (temperature_value * (entropy_sides.detach() - self.side_target_entropy)).sum()
        self._step(
            temperature_loss,
            self.temperature_optimizer,
            self.temperature_scheduler,
            self.temperature_parameters,
            normalize=None,
            scaled=False,
        )
        info = {"temperature/value": temperature_value.mean().detach(), "temperature/loss": temperature_loss.detach()}
        if temperature_value.numel() > 1:
            for h in range(temperature_value.numel()):
                info[f"temperature/value_side{h}"] = temperature_value[h].detach()
        return info

    def _update_critic_heads(
        self,
        critic_obs: torch.Tensor,
        critic_next_obs: torch.Tensor,
        actor_next_obs: torch.Tensor,
        actions_batch: torch.Tensor,
        rewards: torch.Tensor,
        terminated: torch.Tensor,
        gamma_n: float,
    ) -> dict[str, torch.Tensor]:
        """Categorical TD per head (rows head-major, shared done); with side actors head h bootstraps
        with its own side's entropy bonus temp_h * logpi_h only."""
        num_heads = self.num_heads
        batch_size = rewards.shape[0]
        with torch.autocast(device_type=self._device_type, dtype=torch.float16, enabled=self.use_amp):
            with torch.no_grad():
                if self.side_masks is not None:
                    next_actions, next_log_prob_sides = self._actor_sample_sides(actor_next_obs, training=False)
                    # Clone to prevent CUDA graph overwriting
                    next_actions = next_actions.clone()
                    entropy_heads = self.temperature()[:, None] * next_log_prob_sides.clone()
                else:
                    next_actions, next_log_probs = self._actor_sample(actor_next_obs, training=False)
                    next_actions = next_actions.clone()
                    entropy_heads = (self.temperature() * next_log_probs.clone()).unsqueeze(0)
                    entropy_heads = entropy_heads.expand(num_heads, batch_size)

                # Joint forward over (obs, action) and (next_obs, next_action) for the BatchNorm statistics
                obs_all = torch.cat([critic_obs, critic_next_obs], dim=0)
                act_all = torch.cat([actions_batch, next_actions], dim=0)
                qs_all, q_infos_all = self._critic_eval_target(obs_all, act_all, training=True)
                # (2H, B) member rows -> (2, H * B): every head is its own TD row
                next_qs = qs_all.chunk(2, dim=1)[1].reshape(2, num_heads * batch_size)
                next_q_log_probs = q_infos_all["log_prob"].chunk(2, dim=1)[1]
                next_q_log_probs = next_q_log_probs.reshape(2, num_heads * batch_size, self.critic.num_bins)
                next_q_log_probs = self._select_min_q(next_qs, next_q_log_probs)

                target_probs = self._compute_td_target(
                    next_q_log_probs,
                    rewards.reshape(batch_size, num_heads).transpose(0, 1).reshape(-1),
                    terminated.repeat(num_heads),
                    entropy_heads.reshape(-1),
                    gamma_n,
                    self.critic.num_bins,
                    self.critic.min_v,
                    self.critic.max_v,
                ).reshape(num_heads, batch_size, self.critic.num_bins)
                max_entropy_bonus = entropy_heads.max()

            _pred_qs_all, pred_q_infos = self._critic_eval(obs_all, act_all, training=True)
            pred_log_probs = torch.chunk(pred_q_infos["log_prob"], 2, dim=1)[0]
            pred_log_probs = pred_log_probs.reshape(2, num_heads, batch_size, self.critic.num_bins)

            ce_loss = -(target_probs.unsqueeze(0) * pred_log_probs).sum(dim=-1)  # (2, H, B)
            critic_loss = ce_loss.mean()

        self._step(
            critic_loss, self.critic_optimizer, self.critic_scheduler, self.critic_parameters, self._critic_normalize
        )

        info = {"critic/loss": critic_loss.detach(), "critic/max_entropy_bonus": max_entropy_bonus}
        for h in range(num_heads):
            info[f"critic/loss_head{h}"] = ce_loss[:, h].mean().detach()
        return info

    def _step(
        self,
        loss: torch.Tensor,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler.LRScheduler,
        params: list[torch.nn.Parameter],
        normalize: Callable[[], None] | None,
        scaled: bool = True,
    ) -> None:
        """Upstream's optimizer step: (AMP-scaled) backward, rank average, step, LR schedule, weight norm."""
        optimizer.zero_grad(set_to_none=True)
        if scaled and self.use_amp:
            self.grad_scaler.scale(loss).backward()
            # Average scaled grads before unscale so an AMP overflow on any rank is seen by every rank
            if self.is_multi_gpu:
                self.reduce_parameters(params)
            self.grad_scaler.unscale_(optimizer)
            self.grad_scaler.step(optimizer)
            self.grad_scaler.update()
        else:
            loss.backward()
            if self.is_multi_gpu:
                self.reduce_parameters(params)
            optimizer.step()
        scheduler.step()
        if normalize is not None:
            with torch.no_grad():
                normalize()

    def load(self, loaded_dict: dict, load_cfg: dict | None = None, strict: bool = True) -> bool:
        """Upstream load with the reward normalizer behind its own ``reward_normalizer`` flag."""
        if load_cfg is None:
            load_cfg = {"actor": True, "critic": True, "optimizer": True, "reward_normalizer": True, "iteration": True}
        restore_iteration = super().load({**loaded_dict, "reward_normalizer_state_dict": None}, load_cfg, strict)
        if load_cfg.get("reward_normalizer") and self.reward_normalizer is not None:
            state = loaded_dict.get("reward_normalizer_state_dict")
            if state is None:
                raise ValueError("load_reward_normalizer=true but the checkpoint has no reward normalizer state.")
            self.reward_normalizer.load_state_dict(state)
        return restore_iteration
