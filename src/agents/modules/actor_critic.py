"""FlashSAC networks of the dexterous-manipulation recipe, built on rsl_rl_flashsac's models."""

from __future__ import annotations

import copy
import math
from typing import Any

import torch
import torch.nn as nn
from rsl_rl_flashsac.models import FlashSACActor, FlashSACCritic, FlashSACTemperature
from rsl_rl_flashsac.models.flash_sac_model import FlashSACDoubleCritic
from rsl_rl_flashsac.modules.flash_sac_layers import FlashSACEmbedder, NormalTanhPolicy
from tensordict import TensorDict

from .traj_encoder import FutureTrajEncoder


class Linear(nn.Module):
    """Bias-free nn.Linear under ``w``, like UnitLinear (NormalTanhPolicy reads ``mean_w.w.weight`` and
    checkpoints use that key) but never re-normalized to unit rows."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.w = nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w(x)


class PolicyHead(NormalTanhPolicy):
    """Tanh-Gaussian head with plain linear layers: mean weight ~ N(0, 1e-3), std weight 0."""

    def __init__(self, hidden_dim: int, action_dim: int, log_std_min: float = -5.0, log_std_max: float = 0.0) -> None:
        super().__init__(hidden_dim, action_dim, log_std_min=log_std_min, log_std_max=log_std_max)
        self.mean_w = Linear(hidden_dim, action_dim)
        self.std_w = Linear(hidden_dim, action_dim)
        nn.init.normal_(self.mean_w.w.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.std_w.w.weight)


class ActorTower(FlashSACActor):
    """FlashSACActor with the trajectory encoder in front of the embedder and a PolicyHead."""

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        num_blocks: int = 2,
        hidden_dim: int = 128,
        log_std_min: float = -5.0,
        log_std_max: float = 0.0,
        traj_cfg: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            obs,
            obs_groups,
            obs_set,
            output_dim,
            num_blocks=num_blocks,
            hidden_dim=hidden_dim,
            log_std_min=log_std_min,
            log_std_max=log_std_max,
        )
        self.traj_encoder = FutureTrajEncoder(traj_cfg) if traj_cfg else None
        if self.traj_encoder is not None:
            input_dim = self.obs_dim + self.traj_encoder.latent_dim - self.traj_encoder.window_width
            self.embedder = FlashSACEmbedder(input_dim=input_dim, hidden_dim=hidden_dim)
        self.predictor = PolicyHead(hidden_dim, output_dim, log_std_min=log_std_min, log_std_max=log_std_max)

    def features(self, observations: torch.Tensor, training: bool) -> torch.Tensor:
        """Traj encoder + embedder + residual blocks + post norm -> policy-head input."""
        x = observations if self.traj_encoder is None else self.traj_encoder(observations)
        x = self.embedder(x, training)
        for block in self.encoder:
            x = block(x, training)
        return self.post_norm(x)

    def get_mean_and_std(self, observations: torch.Tensor, training: bool) -> tuple[torch.Tensor, torch.Tensor]:
        return self.predictor.get_mean_and_std(self.features(observations, training), training)

    def sample_action_logp(self, observations: torch.Tensor, training: bool) -> tuple[torch.Tensor, torch.Tensor]:
        actions, info = self.predictor(self.features(observations, training), training)
        return actions, info["log_prob"]

    def as_jit(self) -> nn.Module:
        self._check_exportable()
        return super().as_jit()

    def as_onnx(self, verbose: bool = False) -> nn.Module:
        self._check_exportable()
        return super().as_onnx(verbose)

    def _check_exportable(self) -> None:
        if self.traj_encoder is not None:
            raise NotImplementedError("rsl_rl_flashsac's actor export does not include the trajectory encoder.")


class SideFactoredActor(nn.Module):
    """One ActorTower per hand side, each reading the full observation and emitting its side's action
    dims (returned in env order); per-side log-probs (S, B) let critic head h train tower h only."""

    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        side_action_ids: list[list[int]],
        **tower_kwargs: Any,
    ) -> None:
        super().__init__()
        ids = [torch.as_tensor(i, dtype=torch.long).reshape(-1) for i in side_action_ids]
        order = torch.cat(ids)
        self.output_dim = int(order.numel())
        if sorted(order.tolist()) != list(range(self.output_dim)):
            raise ValueError(f"side action ids must partition range({self.output_dim}), got {order.tolist()}")
        self.side_action_ids = [i.tolist() for i in ids]
        # side-major concat -> env order: full[:, d] = cat[:, argsort(order)[d]]
        self.register_buffer("scatter_idx", torch.argsort(order), persistent=False)
        masks = torch.zeros(len(ids), self.output_dim, dtype=torch.bool)
        for s, i in enumerate(ids):
            masks[s, i] = True
        self.register_buffer("side_masks", masks, persistent=False)
        self.sides = nn.ModuleList(
            ActorTower(obs, obs_groups, obs_set, int(i.numel()), **tower_kwargs) for i in ids
        )
        self.obs_groups = self.sides[0].obs_groups
        self.last_action_std = torch.zeros(self.output_dim)

    def flatten_obs(self, obs: TensorDict, training: bool = False) -> torch.Tensor:
        return self.sides[0].flatten_obs(obs, training)

    def set_action_scaling(self, action_bias: torch.Tensor, action_scale: torch.Tensor) -> None:
        for side, ids in zip(self.sides, self.side_action_ids):
            side.set_action_scaling(action_bias[ids], action_scale[ids])

    def _scatter(self, parts: list[torch.Tensor]) -> torch.Tensor:
        return torch.cat(parts, dim=-1)[:, self.scatter_idx]

    def get_mean_and_std(self, observations: torch.Tensor, training: bool) -> tuple[torch.Tensor, torch.Tensor]:
        outs = [side.get_mean_and_std(observations, training) for side in self.sides]
        return self._scatter([mean for mean, _ in outs]), self._scatter([std for _, std in outs])

    def sample_action_logp_sides(
        self, observations: torch.Tensor, training: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample every tower (side order): actions (B, A) in env order and log-probs (S, B)."""
        outs = [side.sample_action_logp(observations, training) for side in self.sides]
        return self._scatter([a for a, _ in outs]), torch.stack([lp for _, lp in outs], dim=0)

    def sample_action_logp(self, observations: torch.Tensor, training: bool) -> tuple[torch.Tensor, torch.Tensor]:
        actions, log_prob_sides = self.sample_action_logp_sides(observations, training)
        # the sides are independent Gaussians, so the joint log-prob is the sum
        return actions, log_prob_sides.sum(dim=0)


class MultiHeadDoubleCritic(FlashSACDoubleCritic):
    """2 * num_heads categorical critics (member twin * num_heads + head) behind the traj encoder;
    actions are (B, A), or (num_qs, B, A) for one gated action view per member."""

    def __init__(
        self,
        num_blocks: int,
        input_dim: int,
        hidden_dim: int,
        num_bins: int,
        min_v: float,
        max_v: float,
        num_heads: int = 1,
        traj_cfg: dict[str, Any] | None = None,
    ) -> None:
        traj_encoder = FutureTrajEncoder(traj_cfg) if traj_cfg else None
        if traj_encoder is not None:
            input_dim += traj_encoder.latent_dim - traj_encoder.window_width
        super().__init__(num_blocks, input_dim, hidden_dim, num_bins, min_v, max_v, num_qs=2 * num_heads)
        self.traj_encoder = traj_encoder
        self.num_heads = num_heads

    def forward(
        self, observations: torch.Tensor, actions: torch.Tensor, training: bool
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.traj_encoder is not None:
            observations = self.traj_encoder(observations)
        if actions.dim() == 2:
            return super().forward(observations, actions, training)
        x = torch.cat((observations.unsqueeze(0).expand(self.num_qs, -1, -1), actions), dim=-1)
        x = self.embedder(x, training)
        for block in self.encoder:
            x = block(x, training)
        return self.predictor(self.post_norm(x), training)


class MultiHeadCritic(FlashSACCritic):
    """FlashSACCritic whose online and EMA target networks are MultiHeadDoubleCritic."""

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int = 1,
        num_actions: int = 0,
        num_blocks: int = 2,
        hidden_dim: int = 256,
        num_bins: int = 101,
        min_v: float = -5.0,
        max_v: float = 5.0,
        num_heads: int = 1,
        traj_cfg: dict[str, Any] | None = None,
    ) -> None:
        # Upstream hard-codes FlashSACDoubleCritic: build a minimal one, then swap in ours.
        super().__init__(
            obs,
            obs_groups,
            obs_set,
            output_dim,
            num_actions,
            num_blocks=0,
            hidden_dim=1,
            num_bins=num_bins,
            min_v=min_v,
            max_v=max_v,
            num_qs=1,
        )
        self.num_heads = num_heads
        self.critic = MultiHeadDoubleCritic(
            num_blocks,
            self.obs_dim + num_actions,
            hidden_dim,
            num_bins,
            min_v,
            max_v,
            num_heads=num_heads,
            traj_cfg=traj_cfg,
        )
        self.critic_target = copy.deepcopy(self.critic)
        for param in self.critic_target.parameters():
            param.requires_grad = False


class SideTemperature(FlashSACTemperature):
    """One learnable log-temperature per side actor; forward() -> (num,)."""

    def __init__(self, initial_value: float = 0.01, num: int = 1) -> None:
        super().__init__(initial_value)
        self.log_temp = nn.Parameter(torch.full((num,), math.log(initial_value), dtype=torch.float32))
