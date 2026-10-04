"""mjlab-ContactSensor views over per-link Isaac sensors: net force + SDF
gating (PhysX GPU can't filter COACD pairs)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import torch

if TYPE_CHECKING:
    from isaaclab.sensors import ContactSensor as IsaacContactSensor

# Fingertip rows within the 13-link alllink order (thumb/index: rota_link2,
# middle/ring/pinky: *_link2) — see ALLLINK_BODY_NAMES in motion_tracking/reference/hand.py.
TIP_ROWS = (3, 6, 8, 10, 12)

# A link touches the object only if its origin is within this SDF distance;
# fallback value — per-link margins (geom reach + skin) come from the MJCF.
SDF_CONTACT_MARGIN = 0.04
FORCE_EPS = 1e-6


class LinkContactBank:
    """Per-step cache: link net forces + SDF-derived contact geometry."""

    def __init__(
        self,
        sensors: list["IsaacContactSensor"],
        clock,
        device: str,
    ) -> None:
        self._sensors = sensors
        self._clock = clock
        self._token = -1
        self._store: dict[str, torch.Tensor] = {}
        self.device = device
        # (13,) per-link contact margin; set by the scene builder from the MJCF.
        self.margins: torch.Tensor | None = None
        # bound after command-term construction (scene._bind_sensor_callbacks)
        self.command_fn: Callable | None = None
        self.link_pos_fn: Callable | None = None
        self.side = "right"
        # alllink rows of the five fingertip links; the scene builder sets them from the MJCF contact sites
        self.tip_rows: tuple[int, ...] = TIP_ROWS

    def _fresh(self) -> dict[str, torch.Tensor]:
        if self._token != self._clock.token:
            self._store.clear()
            self._token = self._clock.token
        return self._store

    def force(self) -> torch.Tensor:
        """(B, 13, 3) net contact force per alllink body (world frame)."""
        s = self._fresh()
        if "force" not in s:
            # NOREPORT ablation drops the PhysX sensors: keep the obs shape valid.
            s["force"] = (
                torch.stack([x.data.net_forces_w[:, 0] for x in self._sensors], dim=1)
                .to(self.device)
                if self._sensors
                else torch.zeros_like(self.link_pos_fn())
            )
        return s["force"]

    def _link_sdf(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(sdf (B,13), grad (B,13,3), link_pos (B,13,3)) vs active object."""
        s = self._fresh()
        if "sdf" not in s:
            assert self.command_fn is not None and self.link_pos_fn is not None
            cmd = self.command_fn()
            p = self.link_pos_fn()  # (B, 13, 3) world link origins
            sdf, grad = cmd.sdf_query(p, self.side)
            s["sdf"], s["grad"], s["lpos"] = sdf, grad, p
        return s["sdf"], s["grad"], s["lpos"]

    def found(self) -> torch.Tensor:
        """(B, 13) 1.0 where the link is in contact with the ACTIVE object."""
        s = self._fresh()
        if "found" not in s:
            sdf, _, _ = self._link_sdf()
            f = self.force().norm(dim=-1)
            m = self.margins if self.margins is not None else SDF_CONTACT_MARGIN
            s["found"] = ((f > FORCE_EPS) & (sdf < m)).to(torch.float32)
        return s["found"]

    def contact_pos(self) -> torch.Tensor:
        """(B, 13, 3) link origin projected onto the object surface."""
        s = self._fresh()
        if "cpos" not in s:
            sdf, grad, lpos = self._link_sdf()
            s["cpos"] = lpos - sdf.unsqueeze(-1) * grad
        return s["cpos"]

    def normal(self) -> torch.Tensor:
        """(B, 13, 3) unit OUTWARD object-surface normal (SDF gradient) at contact_pos."""
        _, grad, _ = self._link_sdf()
        return grad


class _SensorDataView:
    """Duck-typed mjlab ``ContactSensor.data`` namespace."""

    def __init__(self, owner: "ContactAdapterBase") -> None:
        self._o = owner

    @property
    def force(self) -> torch.Tensor:
        return self._o.force()

    @property
    def torque(self) -> torch.Tensor:
        return self._o.torque()

    @property
    def pos(self) -> torch.Tensor:
        return self._o.pos()

    @property
    def found(self) -> torch.Tensor:
        return self._o.found()

    @property
    def dist(self) -> torch.Tensor:
        return self._o.dist()

    @property
    def normal(self) -> torch.Tensor:
        return self._o.normal()


class ContactAdapterBase:
    def __init__(self, bank: LinkContactBank):
        self._bank = bank
        self.data = _SensorDataView(self)

    def force(self) -> torch.Tensor:
        raise NotImplementedError

    def torque(self) -> torch.Tensor:
        raise NotImplementedError

    def pos(self) -> torch.Tensor:
        raise NotImplementedError

    def found(self) -> torch.Tensor:
        raise NotImplementedError

    def dist(self) -> torch.Tensor:
        raise NotImplementedError

    def normal(self) -> torch.Tensor:
        raise NotImplementedError

    def reset(self, env_ids=None) -> None:  # mjlab Scene.reset parity
        pass


class AlllinkContactAdapter(ContactAdapterBase):
    """``r_alllink_contact``: found-masked net force + torque approximated
    about the link origin from the SDF-projected contact point."""

    def force(self) -> torch.Tensor:
        return self._bank.force() * self._bank.found().unsqueeze(-1)

    def found(self) -> torch.Tensor:
        return self._bank.found()

    def pos(self) -> torch.Tensor:
        return self._bank.contact_pos()

    def torque(self) -> torch.Tensor:
        f = self.force()
        _, _, lpos = self._bank._link_sdf()
        lever = self._bank.contact_pos() - lpos
        return torch.cross(lever, f, dim=-1)


class AlllinkContactPosAdapter(ContactAdapterBase):
    """``r_alllink_contact_pos``: contact point + found (+ outward normal) per link."""

    # mjlab's contact normal is primary->secondary (into the object); ours is the SDF
    # gradient, so consumers flip it (envs.rewards.chord._chord_step).
    normal_is_outward = True

    def found(self) -> torch.Tensor:
        return self._bank.found()

    def pos(self) -> torch.Tensor:
        return self._bank.contact_pos()

    def normal(self) -> torch.Tensor:
        return self._bank.normal()


class FingertipContactAdapter(ContactAdapterBase):
    """``r_fingertip_contact``: tip-link rows of the alllink bank."""

    def force(self) -> torch.Tensor:
        f = self._bank.force() * self._bank.found().unsqueeze(-1)
        return f[:, list(self._bank.tip_rows)]

    def found(self) -> torch.Tensor:
        return self._bank.found()[:, list(self._bank.tip_rows)]


class FingertipPenetrationAdapter(ContactAdapterBase):
    """``r_fingertip_penetration``: signed SDF distance at the contact sites
    (negative inside), close enough to the mjwarp mindist consumers."""

    def __init__(self, bank: LinkContactBank, side: str = "right"):
        super().__init__(bank)
        self._side = side

    def dist(self) -> torch.Tensor:
        cmd = self._bank.command_fn()
        pts = cmd.robot_contact_trans_w[:, cmd._side_list.index(self._side)]
        sdf, _ = cmd.sdf_query(pts, self._side)  # (B, 5)
        return sdf

    def found(self) -> torch.Tensor:
        force_found = self._bank.found()[:, list(self._bank.tip_rows)] > 0
        touch = self.dist() < 0.0
        return (force_found | touch).to(torch.float32)
