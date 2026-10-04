"""mjlab-Simulation-compatible adapter over Isaac's SimulationContext; mjwarp-
only surfaces no-op or raise so terms take their documented fallbacks."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from isaaclab.sim import SimulationContext


class _ModelTables:
    """Synthetic per-body model fields (world row 0 + one row per object),
    indexed [0, body_id] like mjwarp's batched model."""

    def __init__(self, device: str) -> None:
        self._rows_mass: list[float] = [0.0]
        self._rows_inertia: list[np.ndarray] = [np.zeros(3)]
        self._rows_iquat: list[np.ndarray] = [np.array([1.0, 0.0, 0.0, 0.0])]
        self._device = device
        self._built = False
        self.opt = SimpleNamespace(gravity=np.array([0.0, 0.0, -9.81]))

    def register_body(self, mass: float, inertia_diag, iquat) -> int:
        assert not self._built, "model tables frozen after first read"
        self._rows_mass.append(float(mass))
        self._rows_inertia.append(np.asarray(inertia_diag, dtype=np.float64))
        self._rows_iquat.append(np.asarray(iquat, dtype=np.float64))
        return len(self._rows_mass) - 1

    def _build(self) -> None:
        if self._built:
            return
        self.body_mass = torch.tensor(
            [self._rows_mass], dtype=torch.float32, device=self._device
        )
        self.body_inertia = torch.tensor(
            np.stack(self._rows_inertia)[None], dtype=torch.float32, device=self._device
        )
        self.body_iquat = torch.tensor(
            np.stack(self._rows_iquat)[None], dtype=torch.float32, device=self._device
        )
        self._built = True

    def __getattr__(self, name: str):
        if name in ("body_mass", "body_inertia", "body_iquat"):
            self._build()
            return self.__dict__[name]
        raise AttributeError(name)


def hull_mass_props(obj_dir, scale: float, density: float):
    """(mass, com(3,), I(3,3)) from the COACD hulls, additively composed like
    MuJoCo's per-geom body inertia (mjwarp parity; visual-mesh fallback)."""
    import glob
    import os

    import trimesh

    hulls = sorted(glob.glob(os.path.join(str(obj_dir), "convex", "*.obj")))
    meshes = []
    for h in hulls:
        try:
            m = trimesh.load(h, force="mesh")
            if scale != 1.0:
                m.apply_scale(float(scale))
            if m.volume > 1e-12:
                meshes.append(m)
        except Exception:
            pass
    if not meshes:
        vis = trimesh.load(os.path.join(str(obj_dir), "visual.obj"), force="mesh")
        if scale != 1.0:
            vis.apply_scale(float(scale))
        m = vis if vis.is_watertight else vis.convex_hull
        meshes = [m]
    masses = np.array([m.volume * density for m in meshes])
    total = float(masses.sum())
    coms = np.stack([m.center_mass for m in meshes])
    com = (coms * masses[:, None]).sum(0) / total
    I = np.zeros((3, 3))
    for m, mm, c in zip(meshes, masses, coms):
        I_own = np.asarray(m.moment_inertia) * density  # about own com
        r = c - com
        I += I_own + mm * ((r @ r) * np.eye(3) - np.outer(r, r))
    mass = float(np.clip(total, 0.05, 0.5))
    I = I * (mass / max(total, 1e-9))
    return mass, com, I


def principal_inertia(mesh, density: float) -> tuple[float, np.ndarray, np.ndarray]:
    """(mass, principal moments, principal->body quat wxyz); same density/clamp
    rule as the USD authoring, convex-hull fallback if not watertight."""
    m = mesh if mesh.is_watertight else mesh.convex_hull
    vol_mass = float(m.volume) * density
    mass = float(np.clip(vol_mass, 0.05, 0.5))
    # trimesh moment_inertia is density-1 based: scale by EFFECTIVE density.
    inertia = np.asarray(m.moment_inertia) * (mass / max(float(m.volume), 1e-9))
    evals, evecs = np.linalg.eigh(inertia)
    if np.linalg.det(evecs) < 0:
        evecs[:, 0] = -evecs[:, 0]
    # rotation matrix (principal->body) -> quat wxyz
    import trimesh.transformations as tt

    T = np.eye(4)
    T[:3, :3] = evecs
    q = tt.quaternion_from_matrix(T)  # wxyz
    return mass, np.maximum(evals, 1e-7), q


class IsaacSimAdapter:
    """Duck-typed ``env.sim`` for the reused mjlab-style env loop."""

    def __init__(
        self, sim_ctx: "SimulationContext", device: str, render: bool = False
    ) -> None:
        self._sim = sim_ctx
        self.device = device
        self.render = render
        self.model = _ModelTables(device)
        # Advertise MuJoCo's sleep flag: parked slots settle on the ground and
        # auto-sleep (no per-step repark). FDR_SLEEP_PARK=0 disables.
        import os

        enableflags = 0
        if os.environ.get("FDR_SLEEP_PARK", "1") != "0":
            import mujoco as _mj

            if hasattr(_mj.mjtEnableBit, "mjENBL_SLEEP"):
                enableflags = int(_mj.mjtEnableBit.mjENBL_SLEEP)
        self.mj_model = SimpleNamespace(opt=SimpleNamespace(enableflags=enableflags))
        # bare namespace: attribute misses steer terms onto fallback paths
        # (tree_awake / qpos direct writes are mjwarp-only).
        self.data = SimpleNamespace()

    def step(self) -> None:
        self._sim.step(render=self.render)

    def forward(self) -> None:
        """mjlab sim.forward() parity: PhysX applies no FK on dof writes, so run
        update_articulations_kinematic to refresh link poses after resets."""
        view = getattr(self._sim, "physics_sim_view", None)
        upd = getattr(view, "update_articulations_kinematic", None)
        if upd is not None:
            upd()
        clock = getattr(self, "_clock", None)
        if clock is not None:
            clock.bump()

    def sense(self) -> None:
        pass

    def reset(self, env_ids=None) -> None:
        pass

    def expand_model_fields(self, fields) -> None:
        if fields:
            raise NotImplementedError(
                f"domain-randomization model fields unsupported on isaac: {fields}"
            )
