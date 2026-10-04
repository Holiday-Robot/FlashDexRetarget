"""Temporal Conv1D encoder over the future-trajectory windows at the flat obs tail."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class FutureTrajEncoder(nn.Module):
    """Temporal Conv1D over the flat obs tail [g1|g2|...] (each group row-major (K, feat)):
    forward returns [flat_obs | latent], replacing the raw window with its encoding."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        self.window_dims = [int(d) for d in cfg["window_dims"]]
        self.n_future = int(cfg["n_future"])
        for d in self.window_dims:
            if d % self.n_future != 0:
                raise ValueError(f"window dim {d} not divisible by n_future={self.n_future}.")
        self.per_step = [d // self.n_future for d in self.window_dims]
        self.window_width = sum(self.window_dims)
        # n_sides>1: one tower per hand; each group is (K, S, ch) row-major, so a side's
        # per-step slice is ch = per_step/S and the latents are concatenated side-major.
        self.n_sides = int(cfg.get("n_sides", 1))
        self.sides: nn.ModuleList | None = None
        if self.n_sides > 1:
            for ps in self.per_step:
                if ps % self.n_sides != 0:
                    raise ValueError(f"per-step width {ps} not divisible by n_sides={self.n_sides}.")
            sub = {**cfg, "n_sides": 1, "window_dims": [d // self.n_sides for d in self.window_dims]}
            self.sides = nn.ModuleList(FutureTrajEncoder(sub) for _ in range(self.n_sides))
            self.latent_dim = self.n_sides * self.sides[0].latent_dim
            return
        self.latent_dim = int(cfg.get("latent_dim", 128))
        h_dim = int(cfg.get("hidden_dim", 128))
        num_layers = int(cfg.get("num_layers", 2))
        self.conv_in = nn.Conv1d(sum(self.per_step), h_dim, kernel_size=1)
        self.agg = str(cfg.get("aggregation", "maxpool"))
        # "conv1d" drops the same-length k=3 trunk: per-step projection -> strided convs -> flatten.
        n_blocks = 0 if self.agg == "conv1d" else num_layers
        self.blocks = nn.ModuleList(nn.Conv1d(h_dim, h_dim, kernel_size=3, padding=1) for _ in range(n_blocks))
        self.act = nn.SiLU()
        # maxpool: max over K after the trunk; conv_flatten / conv1d: strided convs + flatten.
        if self.agg == "maxpool":
            self.shrink = None
            self.head = nn.Linear(h_dim, self.latent_dim)
        elif self.agg in ("conv_flatten", "conv1d"):
            # k=4 leaves l1=1 for K<=6 and the second conv underflows; use k=2 there.
            if self.n_future <= 6:
                k1, s1 = (2, 2)
            else:
                k1, s1 = (4, 2) if self.n_future <= 12 else (6, 2)
            c1 = max(h_dim // 2, 8)
            l1 = (self.n_future - k1) // s1 + 1
            # l1==1 (K=3) leaves nothing for a k=2 conv; a 1-tap conv still projects channels.
            k2, s2 = (1, 1) if l1 <= 1 else ((2, 1) if l1 <= 5 else (4, 2))
            c2 = max(c1 // 2, 8)
            l2 = (l1 - k2) // s2 + 1
            if l2 < 1:
                raise ValueError(f"conv_flatten cannot shrink n_future={self.n_future}.")
            self.shrink = nn.ModuleList(
                [
                    nn.Conv1d(h_dim, c1, kernel_size=k1, stride=s1),
                    nn.Conv1d(c1, c2, kernel_size=k2, stride=s2),
                ]
            )
            self.head = nn.Linear(c2 * l2, self.latent_dim)
        else:
            raise ValueError(f"unknown traj_encoder.aggregation={self.agg!r}")

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        flat = observations[:, : -self.window_width]
        win = observations[:, -self.window_width :]
        return torch.cat([flat, self.encode(win)], dim=-1)

    def encode(self, win: torch.Tensor) -> torch.Tensor:
        """(B, window_width) raw window -> (B, latent_dim)."""
        B = win.shape[0]
        n_sides = max(self.n_sides, 1)
        parts: list[list[torch.Tensor]] = [[] for _ in range(n_sides)]
        off = 0
        for d, ps in zip(self.window_dims, self.per_step):
            g = win[:, off : off + d].reshape(B, self.n_future, n_sides, ps // n_sides)
            for s in range(n_sides):
                parts[s].append(g[:, :, s])
            off += d
        if self.sides is not None:
            return torch.cat(
                [enc._encode_steps(torch.cat(p, dim=-1)) for enc, p in zip(self.sides, parts)], dim=-1
            )
        return self._encode_steps(torch.cat(parts[0], dim=-1))

    def _encode_steps(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)  # (B, feat, K)
        y = self.act(self.conv_in(x))
        for block in self.blocks:
            y = self.act(block(y))
        if self.shrink is None:
            return self.head(y.max(dim=-1).values)
        for conv in self.shrink:
            y = self.act(conv(y))
        return self.head(y.flatten(1))
