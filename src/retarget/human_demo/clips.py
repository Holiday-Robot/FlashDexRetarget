from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

FINGERS = ("thumb", "index", "middle", "ring", "pinky")
MANO_JOINT_NAMES = tuple(
    sorted(f"{f}_{p}" for f in FINGERS for p in ("proximal", "intermediate", "distal", "tip"))
)
SIDES = ("right", "left")
TIP_VERTEX = {"thumb": 766, "index": 353, "middle": 467, "ring": 576, "pinky": 695}  # MANO fingertip vertices


def mano_reference(joints: np.ndarray, verts: np.ndarray, global_rot: np.ndarray,
                   side: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MANO layer output (joints (T, 16, 3), verts (T, 778, 3), global_rot (T, 3, 3)) -> clip hand arrays."""
    first_joint = {"index": 1, "middle": 4, "pinky": 7, "ring": 10, "thumb": 13}
    wrist = joints[:, 0]
    named = {}
    for finger, j in first_joint.items():
        for k, part in enumerate(("proximal", "intermediate", "distal")):
            named[f"{finger}_{part}"] = joints[:, j + k]
        named[f"{finger}_tip"] = verts[:, TIP_VERTEX[finger]]
    joints20 = np.stack([named[n] for n in MANO_JOINT_NAMES], axis=1)
    rebase = np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=np.float64)
    if side == "left":
        rebase = rebase @ np.diag([-1.0, -1.0, 1.0])
    rot = np.asarray(global_rot, dtype=np.float64) @ rebase
    return wrist.astype(np.float32), rot.astype(np.float32), joints20.astype(np.float32)


def load_clip(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as z:
        return {k: z[k] for k in z.files}


def save_clip(path: str | Path, clip: dict[str, np.ndarray]) -> None:
    """Write via a temp file so an interrupted run never leaves a truncated clip."""
    path = Path(path)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, **clip)
    os.replace(tmp, path)


def list_clips(clip_dir: str | Path) -> list[Path]:
    return sorted(p for p in Path(clip_dir).glob("*.npz") if not p.name.endswith(".tmp.npz"))


def sides(clip: dict) -> tuple[str, ...]:
    return tuple(s for s in SIDES if f"mano_{s}_wrist_pos" in clip)


def num_frames(clip: dict) -> int:
    return len(clip[f"mano_{sides(clip)[0]}_wrist_pos"])


def joints(clip: dict, side: str) -> np.ndarray:
    """(T, 20, 3) MANO joints in MANO_JOINT_NAMES order."""
    names = MANO_JOINT_NAMES
    if f"mano_{side}_joint_names" in clip:
        names = [str(n) for n in clip[f"mano_{side}_joint_names"]]
    j = clip[f"mano_{side}_joints"]
    return j[:, [names.index(n) for n in MANO_JOINT_NAMES]].astype(np.float64)


def tips(clip: dict, side: str) -> np.ndarray:
    """(T, 5, 3) MANO fingertips, thumb .. pinky."""
    cols = [MANO_JOINT_NAMES.index(f"{f}_tip") for f in FINGERS]
    return joints(clip, side)[:, cols]


def obj_id(clip: dict, side: str) -> str:
    return str(clip[f"obj_{side}_id"])


def obj_scale(clip: dict, side: str) -> float:
    return float(clip[f"obj_{side}_scale"]) if f"obj_{side}_scale" in clip else 1.0


def obj_pose(clip: dict, side: str) -> np.ndarray:
    return clip[f"obj_{side}_pose"].astype(np.float64)


def shared_object(clip: dict) -> bool:
    """Both hands on one object: same id and the same pose track."""
    if sides(clip) != SIDES or obj_id(clip, "right") != obj_id(clip, "left"):
        return False
    return bool(np.allclose(obj_pose(clip, "right"), obj_pose(clip, "left"), atol=1e-6))


def objects(clip: dict) -> list[str]:
    """The clip's distinct object bodies, as hand sides (a shared object counts once)."""
    return ["right"] if shared_object(clip) else list(sides(clip))


class ObjectMeshes:
    """visual.obj (scaled) per (object, scale): vertices, KD-tree, convex hull and centre of mass."""

    def __init__(self, pool: str | Path) -> None:
        self.pool = Path(pool)
        self._cache: dict[tuple[str, float], dict] = {}

    def get(self, oid: str, scale: float = 1.0) -> dict:
        key = (oid, float(scale))
        if key not in self._cache:
            path = self.pool / "objects" / oid / "visual.obj"
            mesh = trimesh.load(path, force="mesh", process=False, skip_materials=True)
            mesh.apply_scale(scale)
            verts = np.asarray(mesh.vertices, dtype=np.float64)
            com = mesh.center_mass if mesh.is_watertight else verts.mean(0)
            self._cache[key] = {
                "verts": verts,
                "tree": cKDTree(verts),
                "hull": np.asarray(mesh.convex_hull.vertices, dtype=np.float64),
                "com": np.asarray(com, dtype=np.float64),
            }
        return self._cache[key]


def clip_fps(paths) -> float:
    """The frame rate the clips share; every clip stores its own fps."""
    rates = set()
    for p in paths:
        with np.load(p, allow_pickle=True) as z:
            if "fps" not in z.files:
                raise ValueError(f"{p.name} has no fps")
            rates.add(float(z["fps"]))
    if len(rates) != 1:
        raise ValueError(f"the clips need one fps, got {sorted(rates)}")
    return rates.pop()


def object_ids(paths) -> list[str]:
    ids = set()
    for p in paths:
        with np.load(p, allow_pickle=True) as z:
            ids.update(str(z[f"obj_{s}_id"]) for s in SIDES if f"obj_{s}_id" in z.files)
    return sorted(ids)
