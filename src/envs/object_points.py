"""Object surface points and keypoints (object-local, per side or per slot), shared by the obs terms,
the rewards and the replay-buffer obs compactor."""

from __future__ import annotations

import os

import torch
from mjlab.utils.lab_api.math import matrix_from_quat


# Candidate pool for the object-surface obs terms. "verts" restores the pre-2026-08-31 mesh-vertex
# pool, whose coverage followed the tessellation.
_OBJ_POOL = os.environ.get("FDR_OBJ_PCD_POOL", "surface")


def surface_pool(mesh, n: int):
    """Deterministic area-uniform points ON the faces: even spacing, topped up area-weighted."""
    import numpy as np
    import trimesh

    pts, _ = trimesh.sample.sample_surface_even(mesh, n, seed=0)
    if len(pts) < n:
        extra, _ = trimesh.sample.sample_surface(mesh, n - len(pts), seed=1)
        pts = np.vstack([pts, extra])
    return np.asarray(pts[:n], dtype=np.float32)


def load_obj_verts(command, device, n_sample: int = 400):
    """Object VISUAL-MESH surface points (object-local, mesh-scaled) per side: ``{side: (V, 3)}``;
    multi-object stacks slots to (S, n_sample, 3) (resolve via keypoints_per_env)."""
    import numpy as np
    import trimesh

    def _one(path, scale):
        m = trimesh.load(path, force="mesh", process=False, skip_materials=True)
        pts = (np.asarray(m.vertices, dtype=np.float32) if _OBJ_POOL == "verts"
               else surface_pool(m, n_sample))
        return pts * float(scale)

    out = {}
    mesh_paths = command.cfg.object.mesh_paths or {}
    scales = command.cfg.object.mesh_scales or {}
    for side in command._side_list:
        mp = mesh_paths[side]
        if isinstance(mp, str):
            v = _one(mp, scales.get(side, 1.0))
            if len(v) > n_sample:
                v = v[np.linspace(0, len(v) - 1, n_sample).astype(int)]
            out[side] = torch.as_tensor(v, device=device)  # (V, 3)
        else:
            slot_scales = list(scales.get(side, [1.0] * len(mp)))
            slot_verts = []
            for s, path in enumerate(mp):
                v = _one(path, slot_scales[s])
                idx = np.linspace(0, len(v) - 1, n_sample).astype(int)
                slot_verts.append(torch.as_tensor(v[idx], device=device))
            out[side] = torch.stack(slot_verts)  # (S, n_sample, 3)
    return out


def keypoints_per_env(command, side, pts: torch.Tensor) -> torch.Tensor:
    """Object-local points per env, ``(B, P, 3)``: single-object (P, 3) is expanded zero-copy
    over the batch; multi-object (S, P, 3) is gathered by each env's active slot."""
    if pts.dim() == 2:
        return pts[None].expand(command.num_envs, -1, -1)
    return pts[command.active_obj_slot(side)]


def generate_bps_basis(n_points: int, radius: float, device) -> torch.Tensor:
    res = 4
    while True:
        lin = torch.linspace(-1.0, 1.0, res)
        gx, gy, gz = torch.meshgrid(lin, lin, lin, indexing="ij")
        pts = torch.stack([gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)], dim=-1)
        inside = pts[pts.norm(dim=-1) <= 1.0]
        if inside.shape[0] >= n_points:
            break
        res += 1
    order = torch.argsort(inside.norm(dim=-1), stable=True)
    inside = inside[order]
    idx = torch.linspace(0, inside.shape[0] - 1, n_points).round().long()
    return (inside[idx] * float(radius)).to(device)


def farthest_point_sample(pts: torch.Tensor, n: int) -> torch.Tensor:
    """Deterministic greedy farthest-point sampling over ``pts`` (V, 3): no RNG, the same
    ``n`` evenly-spread points are picked every call (stable obs). Returns (n,) index LongTensor."""
    V = pts.shape[0]
    if V <= n:
        # fewer verts than requested: use all, padding by repeating the last.
        pad = torch.full((n - V,), V - 1, dtype=torch.long, device=pts.device)
        return torch.cat([torch.arange(V, device=pts.device), pad])
    first = int(torch.argmax((pts - pts.mean(dim=0)).norm(dim=-1)))
    idx = torch.empty(n, dtype=torch.long, device=pts.device)
    idx[0] = first
    dist = (pts - pts[first]).norm(dim=-1)  # (V,) min dist to chosen set
    for i in range(1, n):
        nxt = int(torch.argmax(dist))
        idx[i] = nxt
        dist = torch.minimum(dist, (pts - pts[nxt]).norm(dim=-1))
    return idx


def obj_surface_keypoints(command, device, n_points: int, n_obj_verts: int):
    """FPS object surface keypoints (object-local) per side, computed once and cached on the
    command; the SAME points back all point-cloud terms. Returns ``{side: (n_points, 3)}``."""
    key = (n_points, n_obj_verts)
    surf = getattr(command, "_obj_pcd_surf", None)
    if surf is None or getattr(command, "_obj_pcd_surf_key", None) != key:
        verts = load_obj_verts(command, device, n_sample=n_obj_verts)
        surf = {}
        for side in command._side_list:
            v = verts[side]  # (V, 3) — or (S, V, 3) in multi-object mode
            if v.dim() == 2:
                idx = farthest_point_sample(v, n_points)  # even coverage
                surf[side] = v[idx]  # (n_points, 3) evenly-spread surface pts
            else:
                surf[side] = torch.stack(
                    [vs[farthest_point_sample(vs, n_points)] for vs in v]
                )  # (S, n_points, 3)
        command._obj_pcd_surf = surf
        command._obj_pcd_surf_key = key
    return surf


_BOX_SIGNS = [(x, y, z) for x in (-1.0, 1.0) for y in (-1.0, 1.0) for z in (-1.0, 1.0)]


def _keypoint_pattern(n_points: int):
    """Unit-cube keypoints: 8 corners, 4 = top diagonal + crossed bottom diagonal (a tetrahedron),
    6 = face centres, 1 = centre."""
    if n_points == 8:
        return list(_BOX_SIGNS)
    if n_points == 4:
        return [c for c in _BOX_SIGNS if c[0] * c[1] * c[2] > 0]
    if n_points == 6:
        return [tuple(s * float(i == k) for i in range(3)) for k in range(3) for s in (-1.0, 1.0)]
    if n_points == 1:
        return [(0.0, 0.0, 0.0)]
    raise ValueError(f"n_points must be 1, 4, 6 or 8, got {n_points}")


def obj_keypoints(command, device, cube_side: float, n_points: int):
    """Object-local keypoints (P, 3) of a fixed-side cube centred on the visual-mesh bbox, per side
    (multi-object: (S, P, 3)); computed once per (cube_side, n_points) and cached on the command."""
    key = (float(cube_side), int(n_points))
    cache = getattr(command, "_obj_keypoints_cache", None)
    if cache is None:
        cache = command._obj_keypoints_cache = {}
    if key in cache:
        return cache[key]
    import numpy as np
    import trimesh

    pattern = np.asarray(_keypoint_pattern(int(n_points)), dtype=np.float32)

    def _one(path, scale):
        m = trimesh.load(path, force="mesh", process=False, skip_materials=True)
        lo, hi = np.asarray(m.bounds, dtype=np.float32) * float(scale)
        return (lo + hi) / 2 + 0.5 * float(cube_side) * pattern

    kp = {}
    mesh_paths = command.cfg.object.mesh_paths or {}
    scales = command.cfg.object.mesh_scales or {}
    for side in command._side_list:
        mp = mesh_paths[side]
        if isinstance(mp, str):
            kp[side] = torch.as_tensor(_one(mp, scales.get(side, 1.0)), device=device)
        else:
            sc = list(scales.get(side, [1.0] * len(mp)))
            kp[side] = torch.stack([torch.as_tensor(_one(p, s), device=device) for p, s in zip(mp, sc)])
    cache[key] = kp
    return kp


def keypoints_world(command, kp, trans_w, quat_w):
    """Object-local keypoints {side: (P,3)|(S,P,3)} placed at the given poses: (B, n_sides, P, 3)."""
    out = []
    for si, side in enumerate(command._side_list):
        p = keypoints_per_env(command, side, kp[side])  # (B, P, 3)
        R = matrix_from_quat(quat_w[:, si])  # (B, 3, 3)
        out.append(trans_w[:, si][:, None, :] + torch.einsum("bij,bpj->bpi", R, p))
    return torch.stack(out, dim=1)
