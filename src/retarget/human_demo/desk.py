"""Ported from CHORD's support-surface reconstruction (NVIDIA, Apache-2.0,
robotic_grounding/retarget/support_recon.py)."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from . import clips as C


def still_frames(pos: np.ndarray, rot: np.ndarray, cfg, fps: float) -> np.ndarray:
    """Frames in runs of >= cfg.min_still_sec with small linear and angular speed."""
    n = len(pos)
    if n < 2:
        return np.arange(n)
    dpos = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    q = Rotation.from_matrix(rot).as_quat()
    dot = np.clip(np.abs(np.sum(q[1:] * q[:-1], axis=1)), 0.0, 1.0)
    dang = 2.0 * np.arccos(dot)
    step = (dpos < cfg.still_speed / fps) & (dang < cfg.still_ang_speed / fps)
    still = np.concatenate([step[:1], step])
    out = np.zeros(n, dtype=bool)
    start = -1
    for i in range(n + 1):
        if i < n and still[i]:
            start = i if start < 0 else start
        else:
            if start >= 0 and i - start >= round(cfg.min_still_sec * fps):
                out[start:i] = True
            start = -1
    return np.nonzero(out)[0]


def segments(frames: np.ndarray) -> list[np.ndarray]:
    if len(frames) == 0:
        return []
    return np.split(frames, np.where(np.diff(frames) > 1)[0] + 1)


def footprint_disk(verts: np.ndarray) -> tuple[float, float, float, float]:
    """(cx, cy, z_min, r): centre of the xy box, lowest point, half its larger side."""
    lo, hi = verts.min(0), verts.max(0)
    return ((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, float(lo[2]),
            float(max(hi[0] - lo[0], hi[1] - lo[1]) / 2))


def _enclose(a, b):
    (x1, y1, r1), (x2, y2, r2) = a, b
    d = float(np.hypot(x2 - x1, y2 - y1))
    if d + r2 <= r1:
        return x1, y1, r1
    if d + r1 <= r2:
        return x2, y2, r2
    r = (d + r1 + r2) / 2
    t = (r - r1) / d if d > 0 else 0.5
    return x1 + (x2 - x1) * t, y1 + (y2 - y1) * t, r


def merge_disks(disks: list, z_tol: float | None = None) -> list:
    """Merge xy-overlapping disks (and, with z_tol, only those at about the same height)."""
    out = list(disks)
    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(out):
            j = i + 1
            while j < len(out):
                x1, y1, z1, r1 = out[i]
                x2, y2, z2, r2 = out[j]
                if np.hypot(x2 - x1, y2 - y1) < r1 + r2 and (z_tol is None or abs(z1 - z2) <= z_tol):
                    x, y, r = _enclose((x1, y1, r1), (x2, y2, r2))
                    out[i] = (x, y, min(z1, z2), r)
                    out.pop(j)
                    changed = True
                else:
                    j += 1
            i += 1
    return out


def _drop_phantoms(by_body: dict[str, list], tracks: dict[str, np.ndarray], cfg, solid: set[str]) -> dict[str, list]:
    """A disk under one object that another, simulated (solid) object occupies most of the time is that object;
    under an object the clip leaves out, the disk stands in for it."""
    out = {}
    for name, disks in by_body.items():
        kept = []
        for x, y, z, r in disks:
            phantom = False
            for other, p in tracks.items():
                if other == name or other not in solid:
                    continue
                inside = (np.hypot(p[:, 0] - x, p[:, 1] - y) < r) & (np.abs(p[:, 2] - z) < cfg.phantom_z)
                if inside.mean() >= cfg.phantom_in_disk_frac:
                    phantom = True
                    break
            if not phantom:
                kept.append((x, y, z, r))
        if kept:
            out[name] = kept
    return out


def prune(disks: list, objects: list[tuple[np.ndarray, np.ndarray, np.ndarray]], rest_band,
          contact_z: float | None = None) -> list:
    """Keep the surfaces the clip's objects rest on: one passes over the disk within rest_band (with contact_z:
    stands on it, centre of mass over it and lowest point within contact_z of its top), and none stands inside it
    (centre of mass over it, lowest point below the band: e.g. a disk fitted elsewhere in the recording on top of
    an object of this clip). objects: (origin, centre of mass, lowest point) tracks. Never leave a clip without
    its lowest disk."""
    lo, hi = rest_band
    out = []
    for d in disks:
        x, y, z_top, r, _ = d
        over = lambda p: np.hypot(p[:, 0] - x, p[:, 1] - y) <= r  # noqa: E731
        if any((over(com) & (bottom < z_top + lo)).any() for _, com, bottom in objects):
            continue
        if contact_z is None:
            used = any((over(p) & (p[:, 2] > z_top + lo) & (p[:, 2] < z_top + hi)).any() for p, _, _ in objects)
        else:
            used = any((over(com) & (np.abs(bottom - z_top) < contact_z)).any() for _, com, bottom in objects)
        if used:
            out.append(d)
    if not out and disks:
        out = [min(disks, key=lambda d: d[2])]
    return out


def _bodies(clip: dict) -> dict[str, tuple[str, float, np.ndarray]]:
    """Name -> (object id, scale, (T, 4, 4) poses) the desk is fitted on: every object over the whole recording
    when the clip carries them (desk_context_*), else the clip's own objects."""
    if "desk_context_ids" in clip:
        return {str(o): (str(o), float(s), p) for o, s, p in
                zip(clip["desk_context_ids"], clip["desk_context_scales"], clip["desk_context_poses"])}
    return {s: (C.obj_id(clip, s), C.obj_scale(clip, s), C.obj_pose(clip, s)) for s in C.objects(clip)}


def support_disks(clip: dict, meshes: C.ObjectMeshes, cfg, fps: float) -> np.ndarray:
    """(K, 5) support disks [x, y, z_top, radius, height] of one clip (K = 0: floor only); cfg: the
    desk config."""
    by_body, tracks = {}, {}
    for name, (oid, scale, pose) in _bodies(clip).items():
        verts = meshes.get(oid, scale)["verts"]
        tracks[name] = pose[:, :3, 3]
        disks = []
        for seg in segments(still_frames(pose[:, :3, 3], pose[:, :3, :3], cfg, fps)):
            t = int(seg[0])
            disks.append(footprint_disk(verts @ pose[t, :3, :3].T + pose[t, :3, 3]))
        disks = merge_disks(disks)  # unlike CHORD, a table at floor height keeps its disk
        if disks:
            by_body[name] = disks
    context = "desk_context_ids" in clip  # bodies are every object of the recording, most of them left out
    solid = {C.obj_id(clip, s) for s in C.objects(clip)} if context else set(tracks)
    by_body = _drop_phantoms(by_body, tracks, cfg, solid)
    flat = [d for ds in by_body.values() for d in ds]
    if len(flat) > 1:
        flat = merge_disks(flat, z_tol=cfg.consolidate_z)
    disks = [[x, y, z, r, cfg.disk_height] for x, y, z, r in flat]
    own = []
    for s in C.objects(clip):
        pose = C.obj_pose(clip, s)
        mesh = meshes.get(C.obj_id(clip, s), C.obj_scale(clip, s))
        com = np.einsum("tij,j->ti", pose[:, :3, :3], mesh["com"]) + pose[:, :3, 3]
        bottom = np.einsum("tij,vj->tvi", pose[:, :3, :3], mesh["hull"])[..., 2].min(1) + pose[:, 2, 3]
        own.append((pose[:, :3, 3], com, bottom))
    disks = prune(disks, own, cfg.rest_band, cfg.contact_z if context else None)
    return np.asarray(disks, dtype=np.float32).reshape(-1, 5)
