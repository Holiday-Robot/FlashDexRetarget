from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import mujoco
# [isaac-vendor] trimmed: import mujoco_warp as mjwarp
import numpy as np
import torch

from mjlab import actuator
# [isaac-vendor] trimmed: from mjlab.actuator import BuiltinActuatorGroup
from mjlab.actuator.actuator import TransmissionType
# [isaac-vendor] trimmed: from mjlab.actuator.xml_actuator import XmlActuator
# [isaac-vendor] trimmed: from mjlab.entity.data import EntityData
from mjlab.utils import spec_config as spec_cfg
# [isaac-vendor] trimmed: from mjlab.utils.lab_api.string import resolve_matching_names
# [isaac-vendor] trimmed: from mjlab.utils.mujoco import dof_width, qpos_width
# [isaac-vendor] trimmed: from mjlab.utils.spec import auto_wrap_fixed_base_mocap
# [isaac-vendor] trimmed: from mjlab.utils.string import resolve_expr
# [isaac-vendor] trimmed: from mjlab.utils.xml import fix_spec_xml, strip_buffer_textures


@dataclass(frozen=False)
class EntityIndexing:
  """Maps entity elements to global indices and addresses in the simulation."""

  # Elements.
  bodies: tuple[mujoco.MjsBody, ...]
  joints: tuple[mujoco.MjsJoint, ...]
  geoms: tuple[mujoco.MjsGeom, ...]
  sites: tuple[mujoco.MjsSite, ...]
  tendons: tuple[mujoco.MjsTendon, ...]
  cameras: tuple[mujoco.MjsCamera, ...]
  lights: tuple[mujoco.MjsLight, ...]
  materials: tuple[mujoco.MjsMaterial, ...]
  pairs: tuple[mujoco.MjsPair, ...]
  actuators: tuple[mujoco.MjsActuator, ...] | None

  # Indices.
  body_ids: torch.Tensor
  geom_ids: torch.Tensor
  site_ids: torch.Tensor
  tendon_ids: torch.Tensor
  cam_ids: torch.Tensor
  light_ids: torch.Tensor
  mat_ids: torch.Tensor
  pair_ids: torch.Tensor
  ctrl_ids: torch.Tensor
  joint_ids: torch.Tensor
  mocap_id: int | None

  # Addresses.
  joint_q_adr: torch.Tensor
  joint_v_adr: torch.Tensor
  free_joint_q_adr: torch.Tensor
  free_joint_v_adr: torch.Tensor

  @property
  def root_body_id(self) -> int:
    return self.bodies[0].id


@dataclass
class EntityCfg:
  @dataclass
  class InitialStateCfg:
    # Root position and orientation.
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    # Root linear and angular velocity (only for floating base entities).
    lin_vel: tuple[float, float, float] = (0.0, 0.0, 0.0)
    ang_vel: tuple[float, float, float] = (0.0, 0.0, 0.0)
    # Articulation (only for articulated entities).
    # Set to None to use the model's existing keyframe (errors if none exists).
    joint_pos: dict[str, float] | None = field(default_factory=lambda: {".*": 0.0})
    joint_vel: dict[str, float] = field(default_factory=lambda: {".*": 0.0})

  init_state: InitialStateCfg = field(default_factory=InitialStateCfg)
  spec_fn: Callable[[], mujoco.MjSpec] = field(
    default_factory=lambda: (lambda: mujoco.MjSpec())
  )
  articulation: EntityArticulationInfoCfg | None = None
  sort_actuators: bool = False
  """When True, reorder actuators so that ``model.ctrl`` follows joint/tendon/site
  definition order rather than the order actuators appear in the config. XML actuators
  are excluded from sorting and always retain their declaration order.
  """

  # Editors.
  lights: tuple[spec_cfg.LightCfg, ...] = field(default_factory=tuple)
  cameras: tuple[spec_cfg.CameraCfg, ...] = field(default_factory=tuple)
  textures: tuple[spec_cfg.TextureCfg, ...] = field(default_factory=tuple)
  materials: tuple[spec_cfg.MaterialCfg, ...] = field(default_factory=tuple)
  collisions: tuple[spec_cfg.CollisionCfg, ...] = field(default_factory=tuple)

  def build(self) -> Entity:
    """Build entity instance from this config.

    Override in subclasses to return custom Entity types.
    """
    return Entity(self)


@dataclass
class EntityArticulationInfoCfg:
  actuators: tuple[actuator.ActuatorCfg, ...] = field(default_factory=tuple)
  soft_joint_pos_limit_factor: float = 1.0


class Entity:
  """[isaac-vendor] Stub of mjlab.entity.Entity.

  The real Entity is mujoco_warp-backed. In the Isaac backend, entity access
  goes through IsaacEntityAdapter (src/envs_isaac), which may subclass this
  stub so isinstance/type annotations keep working. EntityCfg.build() is
  unsupported here.
  """

  def __init__(self, cfg: "EntityCfg | None" = None):
    self.cfg = cfg
