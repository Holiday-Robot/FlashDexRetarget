from __future__ import annotations

from typing import Any

from mjlab.entity import EntityCfg
from mjlab.sensor import SensorCfg
from mjlab.sensor.contact_sensor import ContactMatch, ContactSensorCfg


def build_sensors(items: Any, entities: dict[str, EntityCfg]) -> tuple[SensorCfg, ...]:
    """Build sensor cfgs, dropping any that reference a missing entity."""
    out: list[SensorCfg] = []
    for s in items or ():
        needed: set[str] = set()
        for side in ("primary", "secondary"):
            side_cfg = s.get(side)
            if side_cfg is None:
                continue
            ent = side_cfg.get("entity")
            if ent is not None:
                needed.add(ent)
        missing = needed - set(entities)
        if missing:
            continue  # silently skip; hand-only / partial-side mode
        out.append(_build_sensor(s))
    return tuple(out)


def _build_sensor(s: Any) -> SensorCfg:
    stype = s.type
    if stype == "contact":
        secondary = s.get("secondary")
        return ContactSensorCfg(
            name=s.name,
            primary=_build_contact_match(s.primary),
            secondary=_build_contact_match(secondary)
            if secondary is not None
            else None,
            fields=tuple(s.get("fields", ("found", "force"))),
            reduce=s.get("reduce", "maxforce"),
            num_slots=int(s.get("num_slots", 1)),
            secondary_policy=s.get("secondary_policy", "first"),
            track_air_time=bool(s.get("track_air_time", False)),
            global_frame=bool(s.get("global_frame", False)),
            history_length=int(s.get("history_length", 0)),
            debug=bool(s.get("debug", False)),
        )
    raise ValueError(f"Unknown sensor type: {stype!r}")


def _build_contact_match(m: Any) -> ContactMatch:
    pattern = m.pattern
    if not isinstance(pattern, str):
        pattern = tuple(pattern)
    return ContactMatch(
        mode=m.mode,
        pattern=pattern,
        entity=m.get("entity"),
        exclude=tuple(m.get("exclude", ()) or ()),
    )
