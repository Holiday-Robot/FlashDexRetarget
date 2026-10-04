from __future__ import annotations

import numpy as np

from . import clips as C


def _smooth(x: np.ndarray, k: int) -> np.ndarray:
    """k-frame moving average, the smoothing the retargeting applies to the wrist."""
    if len(x) < k:
        return x
    c = np.cumsum(np.pad(x, ((1, 0), (0, 0))), axis=0)
    return (c[k:] - c[:-k]) / k


def _supported(hull: np.ndarray, com: np.ndarray, band: float, slack: float) -> bool:
    """COM over the box of the hull points within band of the lowest one (+ slack)."""
    low = hull[np.abs(hull[:, 2] - hull[:, 2].min()) < band]
    lo, hi = low[:, :2].min(0) - slack, low[:, :2].max(0) + slack
    return bool(np.all(com[:2] >= lo) and np.all(com[:2] <= hi))


def _bottom(hull: np.ndarray, pose: np.ndarray) -> np.ndarray:
    """(T,) height of the object's lowest hull point."""
    return np.einsum("tij,vj->tvi", pose[:, :3, :3], hull)[..., 2].min(1) + pose[:, 2, 3]


def _touched(clip: dict, side: str, meshes: C.ObjectMeshes, contact: float) -> np.ndarray:
    """(T,) a fingertip of either hand, or the other object, within contact of this object's mesh."""
    mesh = meshes.get(C.obj_id(clip, side), C.obj_scale(clip, side))
    pose = C.obj_pose(clip, side)
    rot, pos = pose[:, :3, :3], pose[:, :3, 3]
    out = np.zeros(len(pose), dtype=bool)
    for s in C.sides(clip):
        local = np.einsum("tji,tkj->tki", rot, C.tips(clip, s) - pos[:, None])
        dist, _ = mesh["tree"].query(local.reshape(-1, 3), distance_upper_bound=contact)
        out |= (dist.reshape(local.shape[:2]) < contact).any(1)
    hull = np.einsum("tij,vj->tvi", rot, mesh["hull"]) + pos[:, None]
    for o in C.objects(clip):  # an object stacked on or leaning against the other one
        if C.obj_id(clip, o) == C.obj_id(clip, side):
            continue
        other, po = meshes.get(C.obj_id(clip, o), C.obj_scale(clip, o)), C.obj_pose(clip, o)
        local = np.einsum("tji,tvj->tvi", po[:, :3, :3], hull - po[:, None, :3, 3])
        dist, _ = other["tree"].query(local.reshape(-1, 3), distance_upper_bound=contact)
        out |= (dist.reshape(local.shape[:2]) < contact).any(1)
    return out


def features(clip: dict, meshes: C.ObjectMeshes, cfg, fps: float) -> dict:
    """Per-clip quantities the reject rules test; cfg: the filters config. Heights are of an object's lowest
    hull point above the lowest disk top; the floor and the disks under its centre of mass support it."""
    n = cfg.edge_frames
    disks = [d for d in np.asarray(clip.get("support_disks", np.zeros((0, 5)))).reshape(-1, 5) if d[3] > 0]
    z_desk = min(d[2] for d in disks) if disks else 0.0
    T = C.num_frames(clip)
    out: dict = {"frames": T, "seconds": T / fps, "disks": len(disks)}
    acc = [0.0]
    for s in C.sides(clip):
        w = _smooth(clip[f"mano_{s}_wrist_pos"].astype(np.float64), cfg.wrist_acc_spike.smooth)
        speed = np.linalg.norm(np.diff(w, axis=0), axis=1) * fps
        if len(speed) > 1:
            acc.append(float(np.abs(np.diff(speed)).max() * fps))
    out["wrist_acc_max"] = max(acc)
    offdisk, z_first, z_last, first_sup = 0, np.inf, np.inf, 1.0
    for s in C.objects(clip):
        mesh = meshes.get(C.obj_id(clip, s), C.obj_scale(clip, s))
        pose = C.obj_pose(clip, s)
        p, com = pose[:, :3, 3], np.einsum("tij,j->ti", pose[:, :3, :3], mesh["com"]) + pose[:, :3, 3]
        bottom = _bottom(mesh["hull"], pose)
        oz, on_floor = bottom - z_desk, bottom < cfg.contact
        supported = on_floor.copy()
        for x, y, _, r, _ in disks:
            supported |= np.hypot(com[:, 0] - x, com[:, 1] - y) < r
        free = ~_touched(clip, s, meshes, cfg.contact)
        low = free & (oz < cfg.obj_offdisk_untouched.max_height)  # higher up, an untouched object is held anyway
        offdisk = max(offdisk, int((low & ~supported).sum()))
        sunk = free & ~on_floor
        if sunk[:n].any():
            z_first = min(z_first, float(oz[:n][sunk[:n]].mean()))
        if sunk[-n:].any():
            z_last = min(z_last, float(oz[-n:][sunk[-n:]].mean()))
        sup = [_supported(mesh["hull"] @ pose[t, :3, :3].T + p[t], com[t],
                          cfg.first_unsupported.base_band, cfg.first_unsupported.slack)
               for t in range(min(n, T)) if low[t]]
        first_sup = min(first_sup, float(np.mean(sup)) if sup else 1.0)
    out.update(obj_offdisk_untouched_frames=offdisk, obj_z_first=float(z_first),
               obj_z_last=float(z_last), first_supported=first_sup)
    return out


def reject_reasons(f: dict, cfg) -> list[str]:
    failed = {
        "obj_offdisk_untouched": f["obj_offdisk_untouched_frames"] >= cfg.obj_offdisk_untouched.frames,
        "wrist_acc_spike": f["wrist_acc_max"] > cfg.wrist_acc_spike.max_acc,
        "obj_below_desk": min(f["obj_z_first"], f["obj_z_last"]) < cfg.obj_below_desk.min_z,
        "too_short": f["seconds"] < cfg.too_short.min_sec,
        "first_unsupported": f["first_supported"] < 1.0,
    }
    return [rule for rule, bad in failed.items() if bad]
