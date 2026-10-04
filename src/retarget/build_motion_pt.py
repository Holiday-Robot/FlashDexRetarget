from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from retarget.human_demo import clips as C


def velocity(x: np.ndarray, dt: float, sigma: float) -> np.ndarray:
    """Finite difference, Gaussian-smoothed over time (sigma in frames)."""
    v = np.gradient(x, axis=0) / dt
    return gaussian_filter1d(v, sigma=sigma, axis=0, mode="nearest").astype(np.float32)


def angular_velocity(rot: np.ndarray, dt: float, sigma: float) -> np.ndarray:
    """World-frame angular velocity of a (T, 3, 3) rotation sequence."""
    w = Rotation.from_matrix(rot[1:] @ np.swapaxes(rot[:-1], -2, -1)).as_rotvec() / dt
    w = np.concatenate([w, w[-1:]], axis=0)
    return gaussian_filter1d(w, sigma=sigma, axis=0, mode="nearest").astype(np.float32)


def tip_contacts(tips: np.ndarray, pose: np.ndarray, verts: np.ndarray, tree: cKDTree,
                 threshold: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fingertips (T, 5, 3) vs the object: distance to the nearest mesh vertex (T, 5), contact flags
    (distance < threshold, dilated by one frame), and that vertex in the object frame (T, 5, 3)."""
    rot, pos = pose[:, :3, :3], pose[:, :3, 3]
    local = np.einsum("tji,tkj->tki", rot, tips - pos[:, None])  # R^T (tip - p)
    dist, idx = tree.query(local.reshape(-1, 3))
    dist = dist.reshape(local.shape[:2]).astype(np.float32)
    near = dist < threshold
    flags = near.copy()
    flags[1:] |= near[:-1]
    flags[:-1] |= near[1:]
    return dist, flags.astype(np.float32), verts[idx].reshape(local.shape).astype(np.float32)


def load_clip(path: Path, meshes: C.ObjectMeshes, fps: float, cfg) -> tuple[dict, dict]:
    clip = C.load_clip(path)
    dt, sigma = 1.0 / fps, cfg.vel_sigma
    qpos = clip["qpos"].astype(np.float32)
    T = len(qpos)
    if T < 2:
        raise ValueError(f"{path.name}: needs at least 2 frames")
    sides = C.sides(clip)
    if not sides:
        raise ValueError(f"{path.name}: no mano_right_* / mano_left_* arrays")
    d = {"joint_pos": qpos, "joint_vel": velocity(qpos, dt, sigma)}
    meta: dict = {"sides": sides, "support": clip.get("support_disks")}
    tip_cols = [C.MANO_JOINT_NAMES.index(f"{f}_tip") for f in C.FINGERS]
    for s in sides:
        joints = C.joints(clip, s).astype(np.float32)
        wrist_pos = clip[f"mano_{s}_wrist_pos"].astype(np.float32)
        wrist_rot = clip[f"mano_{s}_wrist_rot"].astype(np.float32)
        pose = C.obj_pose(clip, s)
        for key, arr in (("mano_joints", joints), ("wrist_pos", wrist_pos), ("wrist_rot", wrist_rot), ("obj_pose", pose)):
            if len(arr) != T:
                raise ValueError(f"{path.name}: {s} {key} has {len(arr)} frames, qpos {T}")
        obj_id, scale = C.obj_id(clip, s), C.obj_scale(clip, s)
        mass = float(clip[f"obj_{s}_mass"]) if f"obj_{s}_mass" in clip else 0.0
        d[f"mano_{s}_wrist_pos"] = wrist_pos
        d[f"mano_{s}_wrist_rot"] = wrist_rot
        d[f"mano_{s}_wrist_vel"] = velocity(wrist_pos, dt, sigma)
        d[f"mano_{s}_wrist_angvel"] = angular_velocity(wrist_rot.astype(np.float64), dt, sigma)
        d[f"mano_{s}_joints"] = joints
        d[f"mano_{s}_joints_vel"] = velocity(joints, dt, sigma)
        d[f"obj_{s}_pos"] = pose[:, :3, 3].astype(np.float32)
        d[f"obj_{s}_rotmat"] = pose[:, :3, :3].astype(np.float32)
        d[f"obj_{s}_vel"] = velocity(pose[:, :3, 3], dt, sigma)
        d[f"obj_{s}_angvel"] = angular_velocity(pose[:, :3, :3], dt, sigma)
        mesh = meshes.get(obj_id, scale)
        dist, flags, local = tip_contacts(joints[:, tip_cols].astype(np.float64), pose, mesh["verts"], mesh["tree"],
                                          cfg.contact_threshold)
        d[f"tips_distance_{s}"] = dist
        d[f"contact_contact_{s}"] = flags
        d[f"contact_contact_pos_full_{s}"] = local
        meta[s] = (obj_id, scale, mass)
    return d, meta


def build(clips: Path, dataset: Path, fps: float, cfg, names: set[str] | None = None) -> None:
    """Pack <clips>/*.npz (only those in names, if given) into <dataset>/motion.pt, next to its objects/
    (cfg: pack)."""
    files = sorted(p for p in Path(clips).glob("*.npz") if names is None or p.stem in names)
    if not files:
        raise FileNotFoundError(f"no .npz clips in {clips}")
    out = Path(dataset).resolve() / "motion.pt"
    meshes = C.ObjectMeshes(dataset)
    rows, metas = [], []
    for f in files:
        d, m = load_clip(f, meshes, fps, cfg)
        rows.append(d)
        metas.append(m)
        print(f"[build] {f.stem}: {len(d['joint_pos'])} frames, objects {[m[s] for s in m['sides']]}", flush=True)
    sides = metas[0]["sides"]
    bad = [f.name for f, m in zip(files, metas) if m["sides"] != sides]
    if bad:
        raise ValueError(f"every clip needs the same hand sides {sides}; differs: {bad[:5]}")

    packed: dict = {k: torch.from_numpy(np.concatenate([r[k] for r in rows])) for k in rows[0]}
    frames = torch.tensor([len(r["joint_pos"]) for r in rows], dtype=torch.long)
    packed["motion_num_frames"] = frames
    packed["length_starts"] = torch.cumsum(frames, 0) - frames
    packed["motion_filename"] = [f.stem for f in files]
    for s in sides:
        packed[f"mano_{s}_joint_names"] = list(C.MANO_JOINT_NAMES)

    # One object slot per distinct (right, left) object pair; a single-hand file has one per object.
    slots: dict[tuple, int] = {}
    slot_of = [slots.setdefault(tuple(m[s] for s in sides), len(slots)) for m in metas]
    for i, s in enumerate(sides):
        packed[f"{s}_object_mesh_dirs"] = [f"objects/{key[i][0]}" for key in slots]
        packed[f"{s}_object_mesh_scales"] = [key[i][1] for key in slots]
        if any(key[i][2] > 0 for key in slots):
            packed[f"{s}_object_masses"] = [key[i][2] for key in slots]
        packed[f"motion_object_slot_{s}"] = torch.tensor(slot_of, dtype=torch.long)

    disks = [m["support"] for m in metas]
    if any(x is not None for x in disks):
        k = max(len(x) for x in disks if x is not None)
        table = np.zeros((len(rows), k, 5), np.float32)
        for i, x in enumerate(disks):
            if x is not None and len(x):
                table[i, : len(x)] = x
        packed["support_disks"] = torch.from_numpy(table)

    torch.save(packed, out)
    print(f"[build] {len(rows)} clips, {int(frames.sum())} frames, {len(slots)} object slots -> {out}", flush=True)


def object_meshes(packed: dict, dataset: Path) -> list[tuple[Path, float]]:
    """Unique (visual.obj, scale) pairs a motion.pt references, in slot order."""
    seen: set[tuple[str, float]] = set()
    out: list[tuple[Path, float]] = []
    for side in ("right", "left"):
        dirs = list(packed.get(f"{side}_object_mesh_dirs") or [])
        scales = list(packed.get(f"{side}_object_mesh_scales") or [1.0] * len(dirs))
        if not dirs and packed.get(f"{side}_object_mesh_dir"):
            dirs, scales = [packed[f"{side}_object_mesh_dir"]], [packed.get(f"{side}_object_mesh_scale", 1.0)]
        for rel, scale in zip(dirs, scales):
            key = (str(rel), float(scale))
            if key not in seen:
                seen.add(key)
                out.append((dataset / str(rel) / "visual.obj", float(scale)))
    return out
