from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from .residual import ResidualAction, ResidualActionCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


@dataclass
class FabricCfg:
    """Geometric-fabric smoothing layer knobs (Van Wyk et al., arXiv:2405.02250)."""

    enable: bool = False
    # Attractor stiffness omega^2 pulling q_f toward the policy target; scalar or
    # per-group dict. 900 -> omega=30 rad/s (~4.8 Hz), the paper's "under 5 Hz".
    stiffness: float | dict[str, float] = 900.0
    damping_ratio: float | dict[str, float] = 1.0
    # Filter only the residual of residual groups; their reference feeds forward lag-free.
    residual_only: bool = False
    # Per-group caps; wrist_trans is in meter units, wrist_rot/finger in radians.
    max_accel: dict[str, float] = field(
        default_factory=lambda: {
            "wrist_trans": 50.0,
            "wrist_rot": 300.0,
            "finger": 500.0,
        }
    )
    # Jerk bound via paper eq.(10): effective accel cap = min(a_max, jerk*dt/2).
    max_jerk: dict[str, float] = field(
        default_factory=lambda: {
            "wrist_trans": 5.0e4,
            "wrist_rot": 3.0e5,
            "finger": 5.0e5,
        }
    )
    max_vel: dict[str, float] = field(
        default_factory=lambda: {"wrist_trans": 2.0, "wrist_rot": 12.0, "finger": 20.0}
    )
    # Velocity-gated inverse-square joint-limit barrier metric; 0 disables. Off by
    # default: targets are pre-clamped and full-close grasps must reach the limits.
    barrier_gain: float = 0.0
    barrier_accel: float = 2.0
    barrier_damping: float = 2.0


class JointFabric:
    """Second-order fabric state (q_f, qd_f) integrated toward the policy target;
    the smoothed q_f replaces the raw target as the PD setpoint (arXiv:2405.02250)."""

    def __init__(
        self,
        cfg: FabricCfg,
        dt: float,
        lower: torch.Tensor,
        upper: torch.Tensor,
        group_sizes: dict[str, int],
        num_envs: int,
        device: torch.device | str,
    ):
        self._dt = float(dt)
        n = int(sum(group_sizes.values()))
        assert lower.shape == (n,) and upper.shape == (n,)

        def per_joint(groups: dict[str, float]) -> torch.Tensor:
            vals: list[float] = []
            for name, size in group_sizes.items():
                vals += [float(groups[name])] * size
            return torch.tensor(vals, device=device)

        def per_joint_or_scalar(v: float | dict[str, float]) -> torch.Tensor:
            if isinstance(v, dict):
                return per_joint(v)
            return torch.full((n,), float(v), device=device)

        # Deviation from the paper: caps are elementwise per joint, not a single
        # direction-preserving scale, so mixed m/rad units never cross-couple.
        self._acc_cap = torch.minimum(
            per_joint(cfg.max_accel), per_joint(cfg.max_jerk) * (self._dt / 2.0)
        )
        self._vel_cap = per_joint(cfg.max_vel)
        self._kp = per_joint_or_scalar(cfg.stiffness)
        self._kd = 2.0 * per_joint_or_scalar(cfg.damping_ratio) * self._kp.sqrt()

        self._lower = lower
        self._upper = upper
        rng = upper - lower
        self._finite = torch.isfinite(rng)
        self._rng = torch.where(self._finite, rng, torch.ones_like(rng))
        self._barrier_gain = float(cfg.barrier_gain)
        self._barrier_accel = float(cfg.barrier_accel)
        self._barrier_damping = float(cfg.barrier_damping)

        self._q = torch.zeros(num_envs, n, device=device)
        self._qd = torch.zeros_like(self._q)
        self._pending = torch.ones(num_envs, dtype=torch.bool, device=device)

    @property
    def joint_pos(self) -> torch.Tensor:
        return self._q

    @property
    def joint_vel(self) -> torch.Tensor:
        return self._qd

    def mark_reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        """Defer re-seeding to the next step(): reset states are written after
        ActionManager.reset (command_manager.reset runs later in _reset_idx)."""
        self._pending[env_ids] = True

    def step(
        self, target: torch.Tensor, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> torch.Tensor:
        """Advance one physics substep and return the smoothed PD target q_f."""
        # Unconditional where instead of a .any() gate to avoid a GPU->CPU sync.
        pending = self._pending.unsqueeze(1)
        self._q = torch.where(
            pending, joint_pos.clamp(self._lower, self._upper), self._q
        )
        self._qd = torch.where(pending, joint_vel, self._qd)
        self._pending.fill_(False)

        acc = self._kp * (target - self._q) - self._kd * self._qd
        if self._barrier_gain > 0.0:
            acc = self._apply_barrier(acc)
        acc = torch.clamp(acc, -self._acc_cap, self._acc_cap)
        qd = torch.clamp(self._qd + self._dt * acc, -self._vel_cap, self._vel_cap)
        q = self._q + self._dt * qd

        # Hard limit clamp; zero any velocity component still pointing into the wall.
        below = q < self._lower
        above = q > self._upper
        self._q = torch.clamp(q, self._lower, self._upper)
        qd = torch.where(below, qd.clamp_min(0.0), qd)
        self._qd = torch.where(above, qd.clamp_max(0.0), qd)
        return self._q

    def _apply_barrier(self, acc: torch.Tensor) -> torch.Tensor:
        """RMP-style combine: barrier mass grows near a limit the joint moves toward,
        overriding the attractor with a bounded push-back (paper Sec. III-A.2)."""
        d_lo = ((self._q - self._lower) / self._rng).clamp(min=1e-3)
        d_hi = ((self._upper - self._q) / self._rng).clamp(min=1e-3)
        gate_lo = ((self._qd < 0.0) & self._finite).to(acc.dtype)
        gate_hi = ((self._qd > 0.0) & self._finite).to(acc.dtype)
        m_lo = self._barrier_gain * gate_lo / d_lo.square()
        m_hi = self._barrier_gain * gate_hi / d_hi.square()
        g = self._barrier_accel * self._rng
        f = m_lo * (g - self._barrier_damping * self._qd) + m_hi * (
            -g - self._barrier_damping * self._qd
        )
        return (acc + f) / (1.0 + m_lo + m_hi)


class FabricResidualAction(ResidualAction):
    """ResidualAction whose PD target is routed through JointFabric every physics substep."""

    def __init__(self, cfg: ResidualActionCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        fabric_cfg = FabricCfg(**cfg.fabric)
        self._residual_only = bool(fabric_cfg.residual_only)
        lower, upper = self._lower, self._upper
        if self._residual_only:
            # Residual joints live in residual space, bounded by their scales.
            lower = torch.where(self._is_residual, -self._residual_scale, lower)
            upper = torch.where(self._is_residual, self._residual_scale, upper)
        self._fabric = JointFabric(
            fabric_cfg,
            dt=env.physics_dt,
            lower=lower,
            upper=upper,
            group_sizes=self._group_sizes,
            num_envs=self.num_envs,
            device=self.device,
        )

    def apply_actions(self) -> None:
        ref_action, residual_action = self._target_parts()
        joint_pos = self._entity.data.joint_pos[:, self._all_joint_ids]
        joint_vel = self._entity.data.joint_vel[:, self._all_joint_ids]
        if self._residual_only:
            filtered = self._fabric.step(
                residual_action,
                joint_pos - ref_action,
                torch.where(self._is_residual, 0.0, joint_vel),
            )
            target = torch.clamp(ref_action + filtered, self._lower, self._upper)
        else:
            target = self._fabric.step(
                torch.clamp(ref_action + residual_action, self._lower, self._upper),
                joint_pos,
                joint_vel,
            )
        self._entity.set_joint_position_target(target, joint_ids=self._all_joint_ids)

    @property
    def fabric_joint_vel(self) -> torch.Tensor:
        """Fabric velocity qd_f. Shape: (B, n_dofs)."""
        return self._fabric.joint_vel

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        super().reset(env_ids)
        self._fabric.mark_reset(env_ids)
