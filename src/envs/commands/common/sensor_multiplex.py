"""Slot-gathering view over the per-slot contact-sensor fan-out (multi-object mode): registered
under the ORIGINAL name, gathers each env's ACTIVE-slot row."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import torch
from mjlab.sensor.contact_sensor import ContactData

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_GATHER_FIELDS = (
    "found",
    "force",
    "torque",
    "dist",
    "pos",
    "normal",
    "force_history",
    "torque_history",
    "dist_history",
)


class MultiObjectContactSensorView:
    """Duck-typed read-only Sensor exposing slot-gathered ``ContactData``; injected AFTER
    scene.initialize, so reset/update are no-ops (per-slot sensors own their lifecycle)."""

    requires_sensor_context: bool = False

    def __init__(
        self,
        slot_sensors: list,
        get_slot: Callable[[], torch.Tensor],
    ) -> None:
        self._slot_sensors = slot_sensors
        self._get_slot = get_slot

    @property
    def data(self) -> ContactData:
        slot = self._get_slot()  # (B,) long, active object slot per env
        rows = torch.arange(slot.shape[0], device=slot.device)
        datas = [s.data for s in self._slot_sensors]
        gathered: dict[str, torch.Tensor | None] = {}
        for f in _GATHER_FIELDS:
            vals = [getattr(d, f, None) for d in datas]
            if vals[0] is None:
                gathered[f] = None
                continue
            stacked = torch.stack(vals, dim=1)  # (B, S, ...)
            gathered[f] = stacked[rows, slot]
        return ContactData(**gathered)

    # ── no-op Sensor lifecycle (scene.reset / scene.update iterate sensors,
    # and the interactive viewer calls debug_vis on every scene sensor) ──

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        pass

    def update(self, dt: float) -> None:
        pass

    def debug_vis(self, visualizer) -> None:
        pass


def register_multi_object_sensor_views(
    env: ManagerBasedRlEnv,
    get_slot: Callable[[], torch.Tensor],
) -> None:
    """Group ``<base>__s<slot>`` sensors and register a view per base name."""
    groups: dict[str, dict[int, object]] = {}
    for name, sensor in env.scene.sensors.items():
        base, sep, idx = name.rpartition("__s")
        if not sep or not idx.isdigit():
            continue
        groups.setdefault(base, {})[int(idx)] = sensor

    for base, by_slot in groups.items():
        if base in env.scene.sensors:
            raise ValueError(
                f"multi-object sensor view {base!r} collides with an existing sensor"
            )
        slots = sorted(by_slot)
        if slots != list(range(len(slots))):
            raise ValueError(
                f"multi-object sensor group {base!r} has gaps: slots={slots}"
            )
        env.scene.sensors[base] = MultiObjectContactSensorView(
            [by_slot[i] for i in slots], get_slot
        )
