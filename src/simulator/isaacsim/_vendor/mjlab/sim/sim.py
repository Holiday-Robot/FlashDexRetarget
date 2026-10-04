# [isaac-vendor] Trimmed mjlab.sim.sim: cfg dataclasses + option maps only.
# The mujoco_warp Simulation runtime is not vendored (Isaac backend supplies
# its own sim adapter).
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import mujoco

from mjlab.utils.nan_guard import NanGuardCfg

_JACOBIAN_MAP = {
  "auto": mujoco.mjtJacobian.mjJAC_AUTO,
  "dense": mujoco.mjtJacobian.mjJAC_DENSE,
  "sparse": mujoco.mjtJacobian.mjJAC_SPARSE,
}
_CONE_MAP = {
  "elliptic": mujoco.mjtCone.mjCONE_ELLIPTIC,
  "pyramidal": mujoco.mjtCone.mjCONE_PYRAMIDAL,
}
_INTEGRATOR_MAP = {
  "euler": mujoco.mjtIntegrator.mjINT_EULER,
  "implicitfast": mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
}
_SOLVER_MAP = {
  "newton": mujoco.mjtSolver.mjSOL_NEWTON,
  "cg": mujoco.mjtSolver.mjSOL_CG,
  "pgs": mujoco.mjtSolver.mjSOL_PGS,
}

# Maps short flag names to MuJoCo enum values.
# Names match the XML <flag> attribute names (e.g. <flag contact="disable"/>).
_DISABLE_FLAG_MAP: dict[str, int] = {
  name.removeprefix("mjDSBL_").lower(): getattr(mujoco.mjtDisableBit, name).value
  for name in dir(mujoco.mjtDisableBit)
  if name.startswith("mjDSBL_")
}
_ENABLE_FLAG_MAP: dict[str, int] = {
  name.removeprefix("mjENBL_").lower(): getattr(mujoco.mjtEnableBit, name).value
  for name in dir(mujoco.mjtEnableBit)
  if name.startswith("mjENBL_")
}


@dataclass
class MujocoCfg:
  """Configuration for MuJoCo simulation parameters."""

  # Integrator settings.
  timestep: float = 0.002
  integrator: Literal["euler", "implicitfast"] = "implicitfast"

  # Friction settings.
  impratio: float = 1.0
  cone: Literal["pyramidal", "elliptic"] = "pyramidal"

  # Solver settings.
  jacobian: Literal["auto", "dense", "sparse"] = "auto"
  solver: Literal["newton", "cg", "pgs"] = "newton"
  iterations: int = 100
  tolerance: float = 1e-8
  ls_iterations: int = 50
  ls_tolerance: float = 0.01
  ccd_iterations: int = 50

  # Other.
  gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
  # Global MuJoCo option flags. Names match the XML <flag> attributes
  # (e.g. "contact", "gravity", "sensor"). See mjtDisableBit / mjtEnableBit.
  disableflags: tuple[str, ...] = ()
  """Disable flags to set (e.g. ``("contact",)`` to disable contacts)."""
  enableflags: tuple[str, ...] = ()
  """Enable flags to set (e.g. ``("energy",)`` to enable energy computation)."""

  def apply(self, model: mujoco.MjModel) -> None:
    """Apply configuration settings to a compiled MjModel."""
    model.opt.jacobian = _JACOBIAN_MAP[self.jacobian]
    model.opt.cone = _CONE_MAP[self.cone]
    model.opt.integrator = _INTEGRATOR_MAP[self.integrator]
    model.opt.solver = _SOLVER_MAP[self.solver]
    model.opt.timestep = self.timestep
    model.opt.impratio = self.impratio
    model.opt.gravity[:] = self.gravity
    model.opt.iterations = self.iterations
    model.opt.tolerance = self.tolerance
    model.opt.ls_iterations = self.ls_iterations
    model.opt.ls_tolerance = self.ls_tolerance
    model.opt.ccd_iterations = self.ccd_iterations
    for flag in self.disableflags:
      if flag not in _DISABLE_FLAG_MAP:
        raise ValueError(
          f"Unknown disable flag {flag!r}. Valid flags: {sorted(_DISABLE_FLAG_MAP)}"
        )
      model.opt.disableflags |= _DISABLE_FLAG_MAP[flag]
    for flag in self.enableflags:
      if flag not in _ENABLE_FLAG_MAP:
        raise ValueError(
          f"Unknown enable flag {flag!r}. Valid flags: {sorted(_ENABLE_FLAG_MAP)}"
        )
      model.opt.enableflags |= _ENABLE_FLAG_MAP[flag]


@dataclass(kw_only=True)
class SimulationCfg:
  nconmax: int | None = None
  """Number of contacts to allocate per world.

  Contacts exist in large heterogenous arrays: one world may have more than nconmax
  contacts. If None, a heuristic value is used."""
  njmax: int | None = None
  """Number of constraints to allocate per world.

  Constraint arrays are batched by world: no world may have more than njmax
  constraints. If None, a heuristic value is used."""
  ls_parallel: bool = True  # Boosts perf quite noticeably.
  contact_sensor_maxmatch: int = 64
  mujoco: MujocoCfg = field(default_factory=MujocoCfg)
  nan_guard: NanGuardCfg = field(default_factory=NanGuardCfg)


class Simulation:
  """[isaac-vendor] Stub of mjlab.sim.sim.Simulation (mujoco_warp-backed)."""
