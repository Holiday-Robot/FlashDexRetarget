"""Isaac motion-tracking command: neutralizes mjwarp sensor registration,
binds the Isaac sensor adapters, restricts sampling in per-env-object mode."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from envs.commands.motion_tracking import motion_tracking as _mt
from envs.commands.motion_tracking.motion_tracking import MotionTrackingCommand
from envs.commands.motion_tracking.motion_tracking_cfg import MotionTrackingCommandCfg


class IsaacMotionTrackingCommand(MotionTrackingCommand):
    def __init__(self, cfg: MotionTrackingCommandCfg, env) -> None:
        orig_raw = _mt.register_raw_contact_views
        orig_mux = _mt.register_multi_object_sensor_views
        _mt.register_raw_contact_views = lambda *a, **k: None
        _mt.register_multi_object_sensor_views = lambda *a, **k: None
        try:
            super().__init__(cfg, env)
        finally:
            _mt.register_raw_contact_views = orig_raw
            _mt.register_multi_object_sensor_views = orig_mux

        bind = getattr(env.scene, "_bind_sensor_callbacks", None)
        if bind is not None:
            side = self._side_list[0]
            bind(lambda: self.active_obj_slot(side), lambda: self)

        # Per-env-object mode: resampling must stay within the env's fixed
        # object; build a padded (S, maxM) per-slot motion table once.
        self._slot_motion_table = None
        self._slot_motion_count = None
        if getattr(env.scene, "assigned_obj_slot", None) is not None:
            self._build_slot_tables()

    def _build_slot_tables(self) -> None:
        """(S, maxM) per-slot draw table: row s lists the motions of object slot s."""
        traj_slot = self.motion_lib.traj_obj_slot[self._side_list[0]]  # (M,)
        S = int(traj_slot.max().item()) + 1
        rows = [torch.where(traj_slot == s)[0] for s in range(S)]
        table = torch.zeros(
            S, max(1, max(int(r.numel()) for r in rows)),
            dtype=torch.long, device=self.device,
        )
        counts = torch.zeros(S, dtype=torch.long, device=self.device)
        for s, r in enumerate(rows):
            table[s, : r.numel()] = r
            counts[s] = r.numel()
        self._slot_motion_table = table
        self._slot_motion_count = counts

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        assigned = getattr(self._env.scene, "assigned_obj_slot", None)
        if (
            assigned is None
            or self._eval_mode
            or self.motion_lib.num_trajectories <= 1
        ):
            return super()._resample_command(env_ids)

        # Pin the draw to the env's own object, then run the base resample
        # (_eval_mode=True only gates the base's motion_ids draw).
        slot = assigned[env_ids]
        counts = self._slot_motion_count[slot].clamp(min=1)
        r = torch.rand(len(env_ids), device=self.device)
        idx = (r * counts.float()).long().clamp(max=int(counts.max().item()) - 1)
        idx = torch.minimum(idx, counts - 1)
        self.motion_ids[env_ids] = self._slot_motion_table[slot, idx]

        prev = self._eval_mode
        self._eval_mode = True
        try:
            super()._resample_command(env_ids)
        finally:
            self._eval_mode = prev

    def _obj_inertia_body(self, side: str) -> torch.Tensor:
        """Per-slot inertia stack (S, 3, 3) for the xfrc pin; the single model
        row can't describe per-env heterogeneous objects."""
        stack = getattr(self._env.scene, "obj_inertia_stack", None)
        if stack is not None and getattr(
            self._env.scene, "assigned_obj_slot", None
        ) is not None:
            if not isinstance(stack, dict):
                return stack
            if side not in stack:
                # shared-object pt: only one side was spawned, the other aliases its entity
                names = self.cfg.object.entity_names
                for s2 in stack:
                    if names.get(s2) == names.get(side):
                        return stack[s2]
            return stack[side]
        return super()._obj_inertia_body(side)


@dataclass(kw_only=True)
class IsaacMotionTrackingCommandCfg(MotionTrackingCommandCfg):
    def build(self, env) -> IsaacMotionTrackingCommand:
        return IsaacMotionTrackingCommand(self, env)
