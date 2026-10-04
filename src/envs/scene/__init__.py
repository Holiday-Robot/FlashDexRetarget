from __future__ import annotations

from typing import Any

from mjlab.entity import EntityCfg
from mjlab.scene import SceneCfg

from ._object_setup import discover_objects_from_motion, expand_multi_object_sensors
from ._robot_setup import build_robot
from ._sensor_setup import build_sensors
from ._support_disk_setup import build_support_disks
from ._terrain_setup import build_terrain


def build_scene(cfg: Any) -> SceneCfg:
    robot_entity_cfg = build_robot(cfg.robot)
    scene_entity_key = cfg.robot.entity_name
    entities: dict[str, EntityCfg] = {scene_entity_key: robot_entity_cfg}
    entities.update(discover_objects_from_motion(cfg))
    entities.update(build_support_disks(cfg))

    s = cfg.scene
    sensor_items = expand_multi_object_sensors(s.get("sensors"), entities, cfg.commands)
    return SceneCfg(
        num_envs=int(s.num_envs),
        env_spacing=float(s.env_spacing),
        terrain=build_terrain(s.terrain),
        entities=entities,
        sensors=build_sensors(sensor_items, entities),
    )
