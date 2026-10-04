"""MotionTrackingCommand mixin: reset co-shift of a still reference object + wrist."""

from __future__ import annotations

import math

import torch


class StaticOffsetMixin:
    def _sample_static_offset(self, env_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        cfg = self.cfg.object.static_offset
        # Isaac's per-env resample flips _eval_mode transiently; use _eval_real.
        if not cfg.enable or getattr(self, "_eval_real", False) or not self.has_objects:
            return {}
        N = len(env_ids)
        mids = self.motion_ids[env_ids]
        t0 = self.motion_steps[env_ids]
        last = self.motion_lib._motion_num_frames[mids] - 1
        t1 = torch.minimum(t0 + int(cfg.static_window_steps), last)
        starts = self.motion_lib.length_starts[mids]
        wrist_w = self.mano_wrist_trans_w[env_ids]  # (N, S, 3)
        out = {}
        for si, side in enumerate(self._side_list):
            if side not in self.motion_lib.obj_trans or side not in tuple(cfg.sides):
                continue
            tr = self.motion_lib.obj_trans[side]
            moved = (tr[starts + t1] - tr[starts + t0]).norm(dim=-1)
            eligible = moved < (float(cfg.static_eps_cm) / 100.0)
            pick = eligible & (torch.rand(N, device=self.device) < float(cfg.prob))
            alias = self._obj_alias_to_right(side)
            if alias is not None:  # shared-object envs: no co-shift
                pick = pick & ~alias[env_ids]
            if not bool(pick.any()):
                continue
            ang = torch.rand(N, device=self.device) * (2 * math.pi)
            dirn = torch.stack([torch.cos(ang), torch.sin(ang), torch.zeros_like(ang)], -1)
            if cfg.away_from_other_hand and wrist_w.shape[1] > 1:
                other = wrist_w[:, 1 - si] - wrist_w[:, si]
                flip = (dirn * other).sum(-1) > 0
                dirn[flip] = -dirn[flip]
            mag = (float(cfg.min_cm) + torch.rand(N, device=self.device)
                   * (float(cfg.max_cm) - float(cfg.min_cm))) / 100.0
            out[side] = dirn * (mag * pick.float())[:, None]
        return out
