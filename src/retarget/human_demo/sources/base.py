from __future__ import annotations

import functools
import importlib
import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import trimesh
from scipy.interpolate import interp1d
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation, Slerp

from retarget.human_demo import clips as C

YUP_TO_ZUP = np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=np.float64)


@dataclass
class Hand:
    """MANO parameters in the source world: wrist places joint 0, else transl is the MANO translation."""

    global_orient: np.ndarray
    hand_pose: np.ndarray
    betas: np.ndarray
    wrist: np.ndarray | None = None
    transl: np.ndarray | None = None


@dataclass
class Demo:
    """One clip as released: hands, the object each hand holds, object poses (object -> source world).
    context: every object's poses over the whole recording: the height (source.table_z) and, with
    source.desk_over_recording, the desk are fitted on them."""

    name: str
    fps: float
    hands: dict[str, Hand]
    objects: dict[str, str]
    poses: dict[str, np.ndarray]
    to_world: np.ndarray = field(default_factory=lambda: np.eye(3))
    scales: dict[str, float] = field(default_factory=dict)
    scene: dict[str, np.ndarray] = field(default_factory=dict)
    context: dict[str, np.ndarray] = field(default_factory=dict)


class Source:
    """A released dataset: units of work (sequences or clips), their demos, and object meshes."""

    mano: dict = {}

    def __init__(self, raw: Path, cfg, mano_dir: str | None = None) -> None:
        self.raw, self.cfg, self.mano_dir = Path(raw), cfg, mano_dir
        self._meshes: dict[str, trimesh.Trimesh] = {}
        self._trees: dict[tuple[str, float], tuple] = {}

    def units(self) -> list[str]:
        raise NotImplementedError

    def demos(self, unit: str) -> list[Demo | str]:
        """The unit's demos; a str is a skipped demo: '<name>: <reason>'."""
        raise NotImplementedError

    def load_mesh(self, obj_id: str) -> trimesh.Trimesh:
        """Metres, in the object frame the poses refer to."""
        raise NotImplementedError

    def mesh(self, obj_id: str) -> trimesh.Trimesh:
        if obj_id not in self._meshes:
            self._meshes[obj_id] = self.load_mesh(obj_id)
        return self._meshes[obj_id]

    def near(self, obj_id: str, pts: np.ndarray, bound: float, scale: float = 1.0) -> np.ndarray:
        """Distance from each (N, 3) object-frame point to the nearest vertex of the scaled mesh where below bound,
        else inf; points outside the vertices' box padded by bound skip the KD-tree."""
        if (obj_id, scale) not in self._trees:
            v = np.asarray(self.mesh(obj_id).vertices) * scale
            self._trees[obj_id, scale] = cKDTree(v), v.min(0), v.max(0)
        tree, lo, hi = self._trees[obj_id, scale]
        d = np.full(len(pts), np.inf)
        box = ((pts >= lo - bound) & (pts <= hi + bound)).all(1)
        d[box] = tree.query(pts[box], distance_upper_bound=bound)[0]
        return d


def assign(wrists: dict[str, np.ndarray], centres: dict[str, np.ndarray]) -> dict[str, str]:
    """Hand side -> object with the least total mean wrist-centre distance; a single object goes to every hand."""
    if len(centres) == 1:
        return {s: next(iter(centres)) for s in wrists}
    dist = {(s, o): float(np.linalg.norm(w - c, axis=1).mean()) for s, w in wrists.items() for o, c in centres.items()}
    pick = min(itertools.permutations(centres, len(wrists)), key=lambda p: sum(dist[k] for k in zip(wrists, p)))
    return dict(zip(wrists, pick))


@functools.cache
def _mano(mano_dir: str, side: str, kwargs: tuple):
    import smplx

    path = Path(mano_dir) / f"MANO_{side.upper()}.pkl"
    return smplx.create(str(path), model_type="mano", is_rhand=side == "right", **dict(kwargs)).eval()


def _forward(model, hand: Hand) -> tuple[np.ndarray, np.ndarray]:
    T = len(hand.global_orient)
    betas = hand.betas if hand.betas.ndim == 2 else np.broadcast_to(hand.betas, (T, hand.betas.shape[-1]))
    args = {"global_orient": hand.global_orient, "hand_pose": hand.hand_pose, "betas": betas}
    if hand.transl is not None:
        args["transl"] = hand.transl
    with torch.no_grad():
        out = model(**{k: torch.from_numpy(np.array(v, dtype=np.float32)) for k, v in args.items()})
    joints, verts = out.joints.numpy().astype(np.float64), out.vertices.numpy().astype(np.float64)
    if hand.wrist is not None:
        shift = hand.wrist[:, None] - joints[:, :1]
        joints, verts = joints + shift, verts + shift
    return joints, verts


def _max_rot_step_deg(rot: np.ndarray) -> float:
    if len(rot) < 2:
        return 0.0
    step = Rotation.from_matrix(rot[1:] @ np.swapaxes(rot[:-1], 1, 2)).magnitude()
    return float(np.degrees(step.max()))


def _floor(poses: dict[str, np.ndarray], source: Source, scales: dict[str, float], fps: float) -> float:
    """Lowest point of the objects over the frames they stand still (< 0.1 m/s): the surface they rest on."""
    lows = []
    for oid, pose in poses.items():
        mesh = source.mesh(oid)
        s = scales.get(oid, 1.0)
        centre = pose[:, :3, :3] @ (mesh.vertices.mean(0) * s) + pose[:, :3, 3]
        speed = np.linalg.norm(np.diff(centre, axis=0), axis=1) * fps
        still = np.concatenate([speed < 0.1, speed[-1:] < 0.1])
        if still.any():
            hull = np.asarray(mesh.convex_hull.vertices) * s
            lows.append(float((np.einsum("tij,vj->tvi", pose[still, :3, :3], hull) + pose[still, None, :3, 3])[..., 2].min()))
    return min(lows) if lows else 0.0


def _resample(x: np.ndarray, src: float, dst: float, rot: bool = False) -> np.ndarray:
    if src == dst:
        return x
    t_src = np.arange(len(x)) / src
    t_dst = np.arange(int((len(x) - 1) * dst / src + 1e-6) + 1) / dst
    if rot:
        return Slerp(t_src, Rotation.from_matrix(x))(t_dst).as_matrix()
    return interp1d(t_src, x, axis=0)(t_dst)


def _rotate(pose: np.ndarray, R: np.ndarray) -> np.ndarray:
    p = np.array(pose, dtype=np.float64)
    p[:, :3, :3] = R @ p[:, :3, :3]
    p[:, :3, 3] = p[:, :3, 3] @ R.T
    return p


def _to_clip(pose: np.ndarray, z0: float, src: float, dst: float) -> np.ndarray:
    """World poses lowered by z0 and resampled to the clip rate, float32."""
    out = np.tile(np.eye(4), (len(_resample(pose[:, 0, 3], src, dst)), 1, 1))
    out[:, :3, :3] = _resample(pose[:, :3, :3], src, dst, rot=True)
    out[:, :3, 3] = _resample(pose[:, :3, 3], src, dst)
    out[:, 2, 3] -= z0
    return out.astype(np.float32)


def _write(source: Source, demo: Demo, cfg, out_dir: Path, shared: dict) -> dict:
    """shared: what a unit's demos with the same context share (the floor and the context poses)."""
    row = {"clip": demo.name}
    sides = ("right", "left") if cfg.sides == "both" else (cfg.sides,)
    missing = [s for s in sides if s not in demo.hands or s not in demo.objects]
    if missing:
        return {**row, "skip": f"no hand with an object ({'/'.join(missing)})"}
    lengths = {len(demo.hands[s].global_orient) for s in sides} | {len(demo.poses[demo.objects[s]]) for s in sides}
    if len(lengths) > 1 or min(lengths) < 2:
        return {**row, "skip": f"frame counts differ ({sorted(lengths)})"}
    flips = [o for o in {demo.objects[s] for s in sides} if _max_rot_step_deg(demo.poses[o][:, :3, :3]) > cfg.max_obj_rot_step_deg]
    if flips:
        return {**row, "skip": f"object flips ({flips[0]})"}

    R = demo.to_world
    world = {oid: _rotate(pose, R) for oid, pose in {**demo.scene, **demo.poses}.items()}
    table_z = source.cfg.get("table_z")
    if table_z is None:
        z0 = 0.0
    elif demo.context:  # over the whole recording, so all its clips share one height
        if ("floor", id(demo.context)) not in shared:
            context = {oid: _rotate(p, R) for oid, p in demo.context.items()}
            shared["floor", id(demo.context)] = _floor(context, source, demo.scales, demo.fps)
        z0 = shared["floor", id(demo.context)] - table_z
    else:
        z0 = _floor(world, source, demo.scales, demo.fps) - table_z
    clip: dict = {"fps": np.float32(cfg.fps)}
    mano_dir = str(cfg.mano_dir)
    for side in sides:
        hand = demo.hands[side]
        joints, verts = _forward(_mano(mano_dir, side, tuple(sorted(source.mano.items()))), hand)
        joints, verts = joints @ R.T, verts @ R.T
        joints[..., 2] -= z0
        verts[..., 2] -= z0
        rot = R @ Rotation.from_rotvec(hand.global_orient).as_matrix()
        wrist_pos, wrist_rot, mano_joints = C.mano_reference(joints, verts, rot, side)
        clip[f"mano_{side}_wrist_pos"] = _resample(wrist_pos, demo.fps, cfg.fps).astype(np.float32)
        clip[f"mano_{side}_wrist_rot"] = _resample(wrist_rot.astype(np.float64), demo.fps, cfg.fps, rot=True).astype(np.float32)
        clip[f"mano_{side}_joints"] = _resample(mano_joints, demo.fps, cfg.fps).astype(np.float32)
        oid = demo.objects[side]
        clip[f"obj_{side}_pose"] = _to_clip(world[oid], z0, demo.fps, cfg.fps)
        clip[f"obj_{side}_id"] = np.array(oid)
        if demo.scales.get(oid, 1.0) != 1.0:
            clip[f"obj_{side}_scale"] = np.float32(demo.scales[oid])
    if demo.context and source.cfg.get("desk_over_recording"):
        ids = sorted(demo.context)
        clip["desk_context_ids"] = np.array(ids)
        clip["desk_context_scales"] = np.array([demo.scales.get(o, 1.0) for o in ids], dtype=np.float32)
        key = ("desk", id(demo.context), z0)
        if key not in shared:
            shared[key] = np.stack([_to_clip(_rotate(demo.context[o], R), z0, demo.fps, cfg.fps) for o in ids])
        clip["desk_context_poses"] = shared[key]
    C.save_clip(out_dir / f"{demo.name}.npz", clip)
    return {**row, "frames": len(clip[f"mano_{sides[0]}_wrist_pos"]),
            "objects": " ".join(sorted({demo.objects[s] for s in sides})), "context": " ".join(sorted(demo.context))}


def _unit(source: Source, unit: str, cfg, out_dir: Path) -> list[dict]:
    rows, shared = [], {}
    for demo in source.demos(unit):
        if isinstance(demo, str):
            name, reason = demo.split(": ", 1)
            rows.append({"clip": name, "skip": reason})
        else:
            rows.append(_write(source, demo, cfg, out_dir, shared))
    return rows


def import_demos(cfg, mano_dir: Path, workers: int) -> None:
    """<raw> (a released dataset, cfg.source) -> <dataset>/human_demo/<clip>.npz + <dataset>/objects/<id>/visual.obj."""
    for f in ("MANO_RIGHT.pkl", "MANO_LEFT.pkl"):
        if not (mano_dir / f).exists():
            raise FileNotFoundError(f"{mano_dir / f} missing (MANO models: mano.is.tue.mpg.de)")
    cfg.mano_dir = str(mano_dir)
    module = importlib.import_module(f"retarget.human_demo.sources.{cfg.source.name}")
    source = module.Source(Path(cfg.raw), cfg.source, str(mano_dir))
    dataset = Path(cfg.dataset)
    out_dir = dataset / "human_demo"
    out_dir.mkdir(parents=True, exist_ok=True)
    units = source.units()
    print(f"[import] {cfg.source.name}: {len(units)} units in {cfg.raw}", flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=max(1, min(workers, len(units)))) as ex:
        futs = [ex.submit(_unit, source, u, cfg, out_dir) for u in units]
        for i, fut in enumerate(as_completed(futs), 1):
            rows += fut.result()
            if i % 50 == 0 or i == len(futs):
                print(f"[import] {i}/{len(futs)} units", flush=True)
    for oid in sorted({o for r in rows for k in ("objects", "context") if r.get(k) for o in r[k].split()}):
        path = dataset / "objects" / oid / "visual.obj"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            source.mesh(oid).export(path)
    kept = [r for r in rows if "skip" not in r]
    reasons: dict[str, int] = {}
    for r in rows:
        if "skip" in r:
            key = r["skip"].split(" (")[0]
            reasons[key] = reasons.get(key, 0) + 1
    print(f"[import] {len(kept)} clips -> {out_dir}, skipped {len(rows) - len(kept)} {reasons}", flush=True)
