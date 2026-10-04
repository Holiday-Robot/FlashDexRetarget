from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from mjlab.entity import Entity


class MotionLibrary:
    """Loads motion files into flat ``(total_frames, ...)`` per-frame tensors, indexed via
    ``length_starts[traj_idx] + frame_idx`` (same pattern as ASAP's MotionLibBase)."""

    def __init__(
        self,
        motion_file: str,
        robot: Entity,
        *,
        finger_names: tuple[str, ...],
        device: str,
    ) -> None:
        self.device = device
        self.finger_names = finger_names
        self.num_robot_joints = robot.num_joints

        motion_dicts, motion_files = load_motions(motion_file)
        # Keep the per-trajectory identifiers: eval-side exports name clips by them.
        self.motion_file = str(motion_file)
        self.motion_files = motion_files
        self.hand_sides = _detect_sides(motion_dicts[0])

        self.num_trajectories = len(motion_dicts)
        self._motion_num_frames = torch.tensor(
            [int(d["joint_pos"].shape[0]) for d in motion_dicts],
            dtype=torch.long,
            device=device,
        )
        lengths_shifted = self._motion_num_frames.roll(1)
        lengths_shifted[0] = 0
        self.length_starts = lengths_shifted.cumsum(0)

        self._parse_robot_data(motion_dicts)
        self._parse_mano_data(motion_dicts, motion_files)
        self._parse_contact_data(motion_dicts, motion_files)
        self._parse_object_traj(motion_dicts, motion_files)
        self._parse_support_disks(motion_dicts)

    def _parse_support_disks(self, motion_dicts: list[dict]) -> None:
        """(M, K, 5) support disks + (M,) counts; all-zero when the pt carries none."""
        have = [d for d in motion_dicts if "support_disks" in d]
        k = int(have[0]["support_disks"].shape[0]) if have else 0
        rows = [
            d["support_disks"] if "support_disks" in d else np.zeros((k, 5), np.float32)
            for d in motion_dicts
        ]
        self.support_disks = torch.tensor(
            np.stack(rows, axis=0) if k else np.zeros((len(motion_dicts), 0, 5), np.float32),
            dtype=torch.float32,
            device=self.device,
        )
        self.support_count = (self.support_disks[:, :, 3] > 0).sum(1)

    # ── Robot (sim) ──────────────────────────────────────────────────────

    def _parse_robot_data(self, motion_dicts: list[dict]) -> None:
        """Joint positions and velocities for warm-start reset."""
        n = self.num_robot_joints
        joint_offset = n if self.hand_sides == ("left",) else 0
        s = slice(joint_offset, joint_offset + n)
        self.robot_joint_pos = _concat(
            [d["joint_pos"][:, s] for d in motion_dicts], self.device
        )
        self.robot_joint_vel = _concat(
            [d["joint_vel"][:, s] for d in motion_dicts], self.device
        )

    # ── MANO reference ──────────────────────────────────────────────────

    def _parse_mano_data(
        self,
        motion_dicts: list[dict],
        motion_files: list[str],
    ) -> None:
        """Per-side MANO wrist, joints, and fingertip indices."""
        self.mano_wrist_trans: dict[str, torch.Tensor] = {}
        self.mano_wrist_rot: dict[str, torch.Tensor] = {}
        self.mano_wrist_lin_vel: dict[str, torch.Tensor] = {}
        self.mano_wrist_ang_vel: dict[str, torch.Tensor] = {}
        self.mano_joint_pos: dict[str, torch.Tensor] = {}
        self.mano_joint_vel: dict[str, torch.Tensor] = {}
        self.mano_joint_names: dict[str, list[str]] = {}
        self.tip_ids: dict[str, torch.Tensor] = {}

        for side in self.hand_sides:
            prefix = f"mano_{side}_"
            concat = lambda key: _concat(
                [d[prefix + key] for d in motion_dicts], self.device
            )
            self.mano_wrist_trans[side] = concat("wrist_pos")
            self.mano_wrist_rot[side] = concat("wrist_rot")
            self.mano_wrist_lin_vel[side] = concat("wrist_vel")
            self.mano_wrist_ang_vel[side] = concat("wrist_angvel")
            self.mano_joint_pos[side] = concat("joints")
            self.mano_joint_vel[side] = concat("joints_vel")

            names_lists = [list(d[prefix + "joint_names"]) for d in motion_dicts]
            base_names = names_lists[0]
            for i, names in enumerate(names_lists[1:], start=1):
                if names != base_names:
                    raise ValueError(
                        f"joint_names mismatch for side {side!r}: traj 0 vs traj {i} "
                        f"({motion_files[0]} vs {motion_files[i]})"
                    )
            self.mano_joint_names[side] = base_names
            self.tip_ids[side] = torch.tensor(
                [base_names.index(f"{f}_tip") for f in self.finger_names],
                dtype=torch.long,
                device=self.device,
            )

    # ── Contact ─────────────────────────────────────────────────────────

    def _parse_contact_data(
        self,
        motion_dicts: list[dict],
        motion_files: list[str],
    ) -> None:
        """Per-side precomputed tip-to-surface distance and contact targets."""
        self.tips_distance: dict[str, torch.Tensor] = {}
        self.contact_pos_full: dict[str, torch.Tensor] = {}
        self.contact_flags: dict[str, torch.Tensor] = {}
        # DexMachina-style per-collision-link contact targets (13 links).
        self.contact_alllink_pos: dict[str, torch.Tensor] = {}
        self.contact_alllink_flags: dict[str, torch.Tensor] = {}
        # Reference contact OUTWARD surface normal per link (object-local), for the
        # vector-contact reward. Optional key (only in motion_dexmachina_vec.pt).
        self.contact_alllink_normal: dict[str, torch.Tensor] = {}

        for side in self.hand_sides:
            args = (motion_dicts, motion_files, self.device)
            _load_optional(f"tips_distance_{side}", self.tips_distance, side, *args)
            _load_optional(
                f"contact_contact_pos_full_{side}", self.contact_pos_full, side, *args
            )
            _load_optional(f"contact_contact_{side}", self.contact_flags, side, *args)
            _load_optional(
                f"contact_alllink_pos_full_{side}", self.contact_alllink_pos, side, *args
            )
            _load_optional(
                f"contact_alllink_{side}", self.contact_alllink_flags, side, *args
            )
            _load_optional(
                f"contact_alllink_normal_full_{side}",
                self.contact_alllink_normal,
                side,
                *args,
            )

    # ── Object trajectory ───────────────────────────────────────────────

    def _parse_object_traj(
        self,
        motion_dicts: list[dict],
        motion_files: list[str],
    ) -> None:
        """Per-side object position, rotation, velocity, angular velocity."""
        self.obj_trans: dict[str, torch.Tensor] = {}
        self.obj_rot: dict[str, torch.Tensor] = {}
        self.obj_lin_vel: dict[str, torch.Tensor] = {}
        self.obj_ang_vel: dict[str, torch.Tensor] = {}
        # Per-trajectory object slot (multi-object packed pt). All zeros for
        # single-object files, so traj_obj_slot[side][motion_ids] is always valid.
        self.traj_obj_slot: dict[str, torch.Tensor] = {
            side: torch.tensor(
                [int(d.get(f"obj_slot_{side}", 0)) for d in motion_dicts],
                dtype=torch.long,
                device=self.device,
            )
            for side in self.hand_sides
        }

        for side in self.hand_sides:
            prefix = f"obj_{side}_"
            if not any(prefix + "pos" in d for d in motion_dicts):
                continue
            _check_field_consistency(prefix + "pos", motion_dicts, motion_files)
            for attr, suffix in [
                (self.obj_trans, "pos"),
                (self.obj_rot, "rotmat"),
                (self.obj_lin_vel, "vel"),
                (self.obj_ang_vel, "angvel"),
            ]:
                attr[side] = _concat(
                    [d[prefix + suffix] for d in motion_dicts],
                    self.device,
                )


# ── Module-level functions ─────────────────────────────────────────────


def _detect_sides(d: dict) -> tuple[str, ...]:
    """Auto-detect which sides are present from key names."""
    sides = tuple(s for s in ("right", "left") if f"mano_{s}_wrist_pos" in d)
    if not sides:
        raise ValueError("No mano_right_wrist_pos or mano_left_wrist_pos found")
    return sides


def load_motions(
    motion_file: str,
) -> tuple[list[dict[str, np.ndarray]], list[str]]:
    """Load motion data from a packed .pt or a single .npz file. Returns (motion_dicts,
    motion_files): per-trajectory dict-of-arrays + string identifiers for error messages."""
    if str(motion_file).endswith(".pt"):
        return _load_packed_pt(str(motion_file))

    motion_dicts = [dict(np.load(motion_file, allow_pickle=True))]
    return motion_dicts, [str(motion_file)]


def _load_packed_pt(
    path: str,
) -> tuple[list[dict[str, np.ndarray]], list[str]]:
    """Unpack a .pt file into per-trajectory dicts mimicking np.load() output."""
    packed = torch.load(path, weights_only=False)
    motion_num_frames = packed["motion_num_frames"].tolist()
    length_starts = packed["length_starts"].tolist()
    motion_filenames = packed.get("motion_filename", [])
    num_total = len(motion_num_frames)

    total_frames = int(sum(motion_num_frames))
    flat_keys = [
        k
        for k, v in packed.items()
        if torch.is_tensor(v) and v.ndim >= 1 and v.size(0) == total_frames
    ]

    motion_dicts: list[dict[str, np.ndarray]] = []
    motion_files: list[str] = []
    for m in range(num_total):
        start = length_starts[m]
        n_frames = motion_num_frames[m]
        d: dict[str, np.ndarray] = {}
        for k in flat_keys:
            d[k] = packed[k][start : start + n_frames].cpu().numpy()
        for jn_key in ("mano_right_joint_names", "mano_left_joint_names"):
            if jn_key in packed:
                d[jn_key] = np.array(packed[jn_key])
        # Multi-object pt: per-trajectory object slot index (0-dim per traj).
        for side in ("right", "left"):
            slot_key = f"motion_object_slot_{side}"
            if slot_key in packed:
                d[f"obj_slot_{side}"] = np.int64(int(packed[slot_key][m]))
        # CHORD-grounded pt: per-trajectory support disks [x, y, z_top, r, h] (K, 5), zero-padded.
        if "support_disks" in packed:
            d["support_disks"] = packed["support_disks"][m].cpu().numpy().astype(np.float32)
        motion_dicts.append(d)
        ident = motion_filenames[m] if m < len(motion_filenames) else f"motion[{m}]"
        motion_files.append(f"{path}#{ident}")

    return motion_dicts, motion_files


def _concat(
    arrays: list[np.ndarray],
    device: str,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Concatenate per-trajectory arrays into a single flat tensor."""
    return torch.tensor(np.concatenate(arrays, axis=0), dtype=dtype, device=device)


def _load_optional(
    key: str,
    target_dict: dict[str, torch.Tensor],
    side: str,
    motion_dicts: list[dict],
    motion_files: list[str],
    device: str,
) -> None:
    """Load an optional field (all-or-none across trajectories)."""
    if not any(key in d for d in motion_dicts):
        return
    _check_field_consistency(key, motion_dicts, motion_files)
    target_dict[side] = _concat([d[key] for d in motion_dicts], device)


def _check_field_consistency(
    key: str,
    motion_dicts: list[dict],
    motion_files: list[str],
) -> None:
    """Raise if a field exists in some trajectories but not all."""
    present = [key in d for d in motion_dicts]
    if not all(present):
        missing = [motion_files[i] for i, p in enumerate(present) if not p]
        raise ValueError(
            f"Field {key!r} missing in {len(missing)}/{len(present)} trajectories: {missing}"
        )
