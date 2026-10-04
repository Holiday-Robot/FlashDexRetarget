from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import mujoco
# [isaac-vendor] trimmed: import mujoco_warp as mjwarp
import numpy as np
import torch

from mjlab.entity import Entity, EntityCfg
# [isaac-vendor] trimmed: from mjlab.sensor import BuiltinSensor, RayCastSensor, Sensor, SensorCfg
from mjlab.sensor import SensorCfg  # [isaac-vendor] cfg-only re-import
# [isaac-vendor] trimmed: from mjlab.sensor.camera_sensor import CameraSensor
# [isaac-vendor] trimmed: from mjlab.sensor.sensor_context import SensorContext
from mjlab.terrains.terrain_entity import TerrainEntity, TerrainEntityCfg
# [isaac-vendor] trimmed: from mjlab.utils.spec import export_spec, non_default_option_fields

_SCENE_XML = Path(__file__).parent / "scene.xml"


@dataclass(kw_only=True)
class SceneCfg:
  """Configuration for a simulation scene."""

  num_envs: int = 1
  """Number of parallel environments."""

  env_spacing: float = 2.0
  """Spacing between environment origins in meters."""

  terrain: TerrainEntityCfg | None = None
  """Terrain configuration. If ``None``, no terrain is added."""

  entities: dict[str, EntityCfg] = field(default_factory=dict)
  """Mapping of entity names to their configurations."""

  sensors: tuple[SensorCfg, ...] = field(default_factory=tuple)
  """Sensor configurations to attach to the scene."""

  extent: float | None = None
  """Override for ``mjModel.stat.extent``. If ``None``, MuJoCo computes
  it automatically."""

  spec_fn: Callable[[mujoco.MjSpec], None] | None = None
  """Optional callback to modify the ``MjSpec`` after entities and sensors
  have been added but before compilation."""


class Scene:
  """[isaac-vendor] Stub of mjlab.scene.Scene (mujoco_warp-backed).

  The Isaac backend provides its own scene adapter with env_origins,
  __getitem__, sensors, etc.
  """
